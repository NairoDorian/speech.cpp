// engine/models/canary_qwen/causal_lm.cpp - the Qwen3 decoder block math and
// KV cache the canary_qwen arch drove, ported from
// src/runtime/causal_lm/causal_lm.cpp with the same op sequence (pre-LN
// RMSNorm, GQA, per-head Q/K RMSNorm, NeoX RoPE, SwiGLU over the packed
// gate+up, F32 accumulation for F16 weights). Only what canary_qwen reaches:
// no per-layer FFN scale, no multi-position step (spec decode), no chunked
// prefill.

#include "engine/models/canary_qwen/graphs.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>

namespace engine::models::canary_qwen::causal_lm {

namespace {

ggml_tensor *rms_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight, float eps) {
  return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

// causal_lm.cpp: mul_mat_f32acc - F32 accumulation for F16 weights (CUDA
// cuBLAS COMPUTE_16F overflows on the residual outliers); a no-op elsewhere.
ggml_tensor *mul_mat_f32acc(ggml_context *ctx, ggml_tensor *w, ggml_tensor *x) {
  ggml_tensor *y = ggml_mul_mat(ctx, w, x);
  if (w->type == GGML_TYPE_F16) {
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
  }
  return y;
}

ggml_tensor *rope(ggml_context *ctx, ggml_tensor *t, ggml_tensor *pos, const BlockParams &p) {
  return ggml_rope_ext(ctx, t, pos, nullptr, p.head_dim, GGML_ROPE_TYPE_NEOX, p.max_position,
                       p.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

ggml_tensor *ffn(ggml_context *ctx, ggml_tensor *x, const DecBlockWeights &w, float rms_eps) {
  ggml_tensor *ff_norm = rms_norm(ctx, x, w.norm_ffn_w, rms_eps);
  ggml_tensor *gate_up = mul_mat_f32acc(ctx, w.ffn_gate_up_w, ff_norm);
  ggml_tensor *ff = ggml_swiglu(ctx, gate_up);
  ff = mul_mat_f32acc(ctx, w.ffn_down_w, ff);
  return ggml_add(ctx, x, ff);
}

// The manual GQA attention of block_prefill / block_step (repeat_interleave
// emulated by reshape -> repeat -> collapse).
ggml_tensor *manual_gqa(ggml_context *ctx, ggml_tensor *Q_att, ggml_tensor *K_att,
                        ggml_tensor *V_att, ggml_tensor *mask, int64_t head_dim, int64_t n_kv,
                        int64_t n_heads, int64_t n_kv_heads, float scale_attn, int64_t q_dim,
                        int64_t T_out) {
  const int64_t n_groups = n_heads / n_kv_heads;
  ggml_tensor *K_att_c = ggml_cont(ctx, K_att);
  ggml_tensor *V_att_c = ggml_cont(ctx, V_att);
  ggml_tensor *K_4d = ggml_reshape_4d(ctx, K_att_c, head_dim, n_kv, 1, n_kv_heads);
  ggml_tensor *V_4d = ggml_reshape_4d(ctx, V_att_c, head_dim, n_kv, 1, n_kv_heads);
  ggml_tensor *K_rep_template =
      ggml_new_tensor_4d(ctx, K_att->type, head_dim, n_kv, n_groups, n_kv_heads);
  ggml_tensor *V_rep_template =
      ggml_new_tensor_4d(ctx, V_att->type, head_dim, n_kv, n_groups, n_kv_heads);
  ggml_tensor *K_rep = ggml_repeat(ctx, K_4d, K_rep_template);
  ggml_tensor *V_rep = ggml_repeat(ctx, V_4d, V_rep_template);
  ggml_tensor *K_full = ggml_reshape_3d(ctx, K_rep, head_dim, n_kv, n_heads);
  ggml_tensor *V_full = ggml_reshape_3d(ctx, V_rep, head_dim, n_kv, n_heads);

  ggml_tensor *kq = ggml_mul_mat(ctx, K_full, Q_att);
  ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, mask, scale_attn, 0.0f);
  ggml_tensor *V_t = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 0, 2, 3));
  ggml_tensor *o = ggml_mul_mat(ctx, V_t, kq_soft);
  o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
  return ggml_reshape_2d(ctx, o, q_dim, T_out);
}

bool alloc_cache(KvCache &cache, ggml_backend_t backend, int64_t elems, ggml_type kv_type,
                 const char *k_name, const char *v_name) {
  if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
    std::fprintf(stderr, "canary_qwen kv_init: unsupported kv_type=%d\n", static_cast<int>(kv_type));
    return false;
  }
  ggml_init_params params{};
  params.mem_size = 2 * ggml_tensor_overhead() + 256;
  params.mem_buffer = nullptr;
  params.no_alloc = true;
  cache.ctx = ggml_init(params);
  if (cache.ctx == nullptr) {
    return false;
  }
  cache.self_k = ggml_new_tensor_1d(cache.ctx, kv_type, elems);
  cache.self_v = ggml_new_tensor_1d(cache.ctx, kv_type, elems);
  ggml_set_name(cache.self_k, k_name);
  ggml_set_name(cache.self_v, v_name);
  cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
  if (cache.buffer == nullptr) {
    ggml_free(cache.ctx);
    cache.ctx = nullptr;
    cache.self_k = nullptr;
    cache.self_v = nullptr;
    return false;
  }
  ggml_backend_buffer_clear(cache.buffer, 0);
  return true;
}

} // namespace

void KvCache::free() {
  if (buffer != nullptr) {
    ggml_backend_buffer_free(buffer);
    buffer = nullptr;
  }
  if (ctx != nullptr) {
    ggml_free(ctx);
    ctx = nullptr;
  }
  self_k = nullptr;
  self_v = nullptr;
  n = 0;
  head = 0;
}

bool kv_init(KvCache &cache, ggml_backend_t backend, int n_ctx, int n_kv_heads, int head_dim,
             int n_layer, ggml_type kv_type) {
  cache.free();
  const int64_t elems = static_cast<int64_t>(n_kv_heads) * head_dim * n_ctx * n_layer;
  if (!alloc_cache(cache, backend, elems, kv_type, "kv_self_k", "kv_self_v")) {
    return false;
  }
  cache.n_ctx = n_ctx;
  cache.n = 0;
  cache.head = 0;
  cache.n_batch = 1;
  return true;
}

bool kv_init_batched(KvCache &cache, ggml_backend_t backend, int n_ctx, int n_kv_heads,
                     int head_dim, int n_layer, int n_batch, ggml_type kv_type) {
  if (n_batch <= 1) {
    return kv_init(cache, backend, n_ctx, n_kv_heads, head_dim, n_layer, kv_type);
  }
  cache.free();
  const int64_t elems = static_cast<int64_t>(n_kv_heads) * head_dim * n_ctx * n_batch * n_layer;
  if (!alloc_cache(cache, backend, elems, kv_type, "kv_self_k_batched", "kv_self_v_batched")) {
    return false;
  }
  cache.n_ctx = n_ctx;
  cache.n = 0;
  cache.head = 0;
  cache.n_batch = n_batch;
  return true;
}

// causal_lm.cpp: block_prefill (kv_batch_slot 0, kv_n_batch 1, no ffn_scale).
ggml_tensor *block_prefill(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                           const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                           int layer_idx, int T_seq, ggml_tensor *mask, ggml_tensor *positions,
                           bool use_flash, bool slice_last_before_ffn) {
  const int64_t n_heads = params.n_heads;
  const int64_t n_kv_heads = params.n_kv_heads;
  const int64_t head_dim = params.head_dim;
  const int64_t q_dim = n_heads * head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  const int n_ctx = kv.n_ctx;
  const float rms_eps = params.rms_eps;
  const float scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const size_t k_elem = ggml_element_size(kv.self_k);
  const size_t v_elem = ggml_element_size(kv.self_v);

  ggml_tensor *x_norm = rms_norm(ctx, x, w.norm_attn_w, rms_eps);

  ggml_tensor *Q = mul_mat_f32acc(ctx, w.attn_q_w, x_norm);
  ggml_tensor *K = mul_mat_f32acc(ctx, w.attn_k_w, x_norm);
  ggml_tensor *V = mul_mat_f32acc(ctx, w.attn_v_w, x_norm);

  Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, T_seq, 1);
  K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, T_seq, 1);
  V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, T_seq, 1);

  if (w.attn_q_norm != nullptr) {
    Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), w.attn_q_norm);
  }
  if (w.attn_k_norm != nullptr) {
    K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), w.attn_k_norm);
  }

  Q = rope(ctx, Q, positions, params);
  K = rope(ctx, K, positions, params);

  const size_t slab_off = static_cast<size_t>(layer_idx) * n_ctx * kv_dim;
  {
    const size_t n_elem = static_cast<size_t>(T_seq) * kv_dim;
    ggml_tensor *k_dst = ggml_view_1d(ctx, kv.self_k, n_elem, k_elem * slab_off);
    ggml_tensor *v_dst = ggml_view_1d(ctx, kv.self_v, n_elem, v_elem * slab_off);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
  }

  ggml_tensor *K_att = ggml_view_3d(ctx, kv.self_k, head_dim, T_seq, n_kv_heads, k_elem * kv_dim,
                                    k_elem * head_dim, k_elem * slab_off);
  ggml_tensor *V_att = ggml_view_3d(ctx, kv.self_v, head_dim, T_seq, n_kv_heads, v_elem * kv_dim,
                                    v_elem * head_dim, v_elem * slab_off);

  ggml_tensor *Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

  ggml_tensor *o;
  if (use_flash) {
    o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask, scale_attn, 0.0f, 0.0f);
    o = ggml_reshape_2d(ctx, o, q_dim, T_seq);
  } else {
    o = manual_gqa(ctx, Q_att, K_att, V_att, mask, head_dim, T_seq, n_heads, n_kv_heads,
                   scale_attn, q_dim, T_seq);
  }

  o = mul_mat_f32acc(ctx, w.attn_o_w, o);
  x = ggml_add(ctx, x, o);

  if (slice_last_before_ffn) {
    const int64_t hidden = x->ne[0];
    const size_t elem = ggml_element_size(x);
    x = ggml_view_2d(ctx, x, hidden, 1, elem * hidden, elem * hidden * static_cast<size_t>(T_seq - 1));
    x = ggml_cont(ctx, x);
  }

  return ffn(ctx, x, w, rms_eps);
}

// causal_lm.cpp: block_step.
ggml_tensor *block_step(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                        const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                        int layer_idx, int max_n_kv, ggml_tensor *mask, ggml_tensor *position,
                        ggml_tensor *kv_idx, bool use_flash) {
  const int64_t n_heads = params.n_heads;
  const int64_t n_kv_heads = params.n_kv_heads;
  const int64_t head_dim = params.head_dim;
  const int64_t q_dim = n_heads * head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  const int n_ctx = kv.n_ctx;
  const float rms_eps = params.rms_eps;
  const float scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const size_t k_elem = ggml_element_size(kv.self_k);
  const size_t v_elem = ggml_element_size(kv.self_v);

  ggml_tensor *x_norm = rms_norm(ctx, x, w.norm_attn_w, rms_eps);

  ggml_tensor *Q = mul_mat_f32acc(ctx, w.attn_q_w, x_norm);
  ggml_tensor *K = mul_mat_f32acc(ctx, w.attn_k_w, x_norm);
  ggml_tensor *V = mul_mat_f32acc(ctx, w.attn_v_w, x_norm);

  Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, 1, 1);
  K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, 1, 1);
  V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, 1, 1);

  if (w.attn_q_norm != nullptr) {
    Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), w.attn_q_norm);
  }
  if (w.attn_k_norm != nullptr) {
    K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), w.attn_k_norm);
  }

  Q = rope(ctx, Q, position, params);
  K = rope(ctx, K, position, params);

  {
    const size_t layer_off_k = k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim;
    const size_t layer_off_v = v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim;
    ggml_tensor *k_layer = ggml_view_2d(ctx, kv.self_k, kv_dim, n_ctx, k_elem * kv_dim, layer_off_k);
    ggml_tensor *v_layer = ggml_view_2d(ctx, kv.self_v, kv_dim, n_ctx, v_elem * kv_dim, layer_off_v);
    ggml_tensor *K_row = ggml_reshape_2d(ctx, K, kv_dim, 1);
    ggml_tensor *V_row = ggml_reshape_2d(ctx, V, kv_dim, 1);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_layer, K_row, kv_idx));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_layer, V_row, kv_idx));
  }

  const size_t layer_off_bytes = k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim;
  ggml_tensor *K_att = ggml_view_3d(ctx, kv.self_k, head_dim, max_n_kv, n_kv_heads, k_elem * kv_dim,
                                    k_elem * head_dim, layer_off_bytes);
  ggml_tensor *V_att = ggml_view_3d(ctx, kv.self_v, head_dim, max_n_kv, n_kv_heads, v_elem * kv_dim,
                                    v_elem * head_dim,
                                    v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim);

  ggml_tensor *Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

  ggml_tensor *o;
  if (use_flash) {
    o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask, scale_attn, 0.0f, 0.0f);
    o = ggml_reshape_2d(ctx, o, q_dim, 1);
  } else {
    o = manual_gqa(ctx, Q_att, K_att, V_att, mask, head_dim, max_n_kv, n_heads, n_kv_heads,
                   scale_attn, q_dim, 1);
  }

  o = mul_mat_f32acc(ctx, w.attn_o_w, o);
  x = ggml_add(ctx, x, o);
  return ffn(ctx, x, w, rms_eps);
}

// causal_lm.cpp: block_step_batched (flash only, as in the arch).
ggml_tensor *block_step_batched(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                                const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                                int layer_idx, int max_n_kv, int n_batch, ggml_tensor *mask,
                                ggml_tensor *position, ggml_tensor *kv_idx, bool use_flash) {
  if (!use_flash) {
    std::fprintf(stderr, "canary_qwen block_step_batched: non-flash path unsupported\n");
    return nullptr;
  }
  const int64_t n_heads = params.n_heads;
  const int64_t n_kv_heads = params.n_kv_heads;
  const int64_t head_dim = params.head_dim;
  const int64_t q_dim = n_heads * head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  const int64_t n_ctx = kv.n_ctx;
  const int64_t B = n_batch;
  const float rms_eps = params.rms_eps;
  const float scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const size_t k_elem = ggml_element_size(kv.self_k);
  const size_t v_elem = ggml_element_size(kv.self_v);

  ggml_tensor *x_norm = rms_norm(ctx, x, w.norm_attn_w, rms_eps);

  ggml_tensor *Q = mul_mat_f32acc(ctx, w.attn_q_w, x_norm);
  ggml_tensor *K = mul_mat_f32acc(ctx, w.attn_k_w, x_norm);
  ggml_tensor *V = mul_mat_f32acc(ctx, w.attn_v_w, x_norm);

  Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, B);
  K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, B);
  V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, B);

  if (w.attn_q_norm != nullptr) {
    Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), w.attn_q_norm);
  }
  if (w.attn_k_norm != nullptr) {
    K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), w.attn_k_norm);
  }

  Q = rope(ctx, Q, position, params);
  K = rope(ctx, K, position, params);

  {
    const size_t layer_off_k = k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim * B;
    const size_t layer_off_v = v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim * B;
    ggml_tensor *k_layer = ggml_view_3d(ctx, kv.self_k, kv_dim, n_ctx, B, k_elem * kv_dim,
                                        k_elem * kv_dim * n_ctx, layer_off_k);
    ggml_tensor *v_layer = ggml_view_3d(ctx, kv.self_v, kv_dim, n_ctx, B, v_elem * kv_dim,
                                        v_elem * kv_dim * n_ctx, layer_off_v);
    ggml_tensor *K_row = ggml_reshape_3d(ctx, K, kv_dim, 1, B);
    ggml_tensor *V_row = ggml_reshape_3d(ctx, V, kv_dim, 1, B);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_layer, K_row, kv_idx));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_layer, V_row, kv_idx));
  }

  const size_t layer_off_bytes_k = k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim * B;
  const size_t layer_off_bytes_v = v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim * B;
  ggml_tensor *K_att = ggml_view_4d(ctx, kv.self_k, head_dim, max_n_kv, n_kv_heads, B,
                                    k_elem * kv_dim, k_elem * head_dim, k_elem * kv_dim * n_ctx,
                                    layer_off_bytes_k);
  ggml_tensor *V_att = ggml_view_4d(ctx, kv.self_v, head_dim, max_n_kv, n_kv_heads, B,
                                    v_elem * kv_dim, v_elem * head_dim, v_elem * kv_dim * n_ctx,
                                    layer_off_bytes_v);

  ggml_tensor *Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 3, 1));

  ggml_tensor *o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask, scale_attn, 0.0f, 0.0f);
  o = ggml_reshape_2d(ctx, o, q_dim, B);

  o = mul_mat_f32acc(ctx, w.attn_o_w, o);
  x = ggml_add(ctx, x, o);
  return ffn(ctx, x, w, rms_eps);
}

// causal_lm.cpp: block_prefill_batched (flash only, as in the arch).
ggml_tensor *block_prefill_batched(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                                   const DecBlockWeights &w, const BlockParams &params,
                                   KvCache &kv, int layer_idx, int T_seq, int n_batch,
                                   ggml_tensor *mask, ggml_tensor *positions, ggml_tensor *kv_idx,
                                   bool use_flash) {
  if (!use_flash) {
    std::fprintf(stderr, "canary_qwen block_prefill_batched: non-flash unsupported\n");
    return nullptr;
  }
  const int64_t n_heads = params.n_heads;
  const int64_t n_kv_heads = params.n_kv_heads;
  const int64_t head_dim = params.head_dim;
  const int64_t q_dim = n_heads * head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  const int64_t n_ctx = kv.n_ctx;
  const int64_t T = T_seq;
  const int64_t B = n_batch;
  const float rms_eps = params.rms_eps;
  const float scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const size_t k_elem = ggml_element_size(kv.self_k);
  const size_t v_elem = ggml_element_size(kv.self_v);

  ggml_tensor *x_norm = rms_norm(ctx, x, w.norm_attn_w, rms_eps);

  ggml_tensor *Q = mul_mat_f32acc(ctx, w.attn_q_w, x_norm);
  ggml_tensor *K = mul_mat_f32acc(ctx, w.attn_k_w, x_norm);
  ggml_tensor *V = mul_mat_f32acc(ctx, w.attn_v_w, x_norm);

  Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, T, B);
  K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, T, B);
  V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, T, B);

  if (w.attn_q_norm != nullptr) {
    Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), w.attn_q_norm);
  }
  if (w.attn_k_norm != nullptr) {
    K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), w.attn_k_norm);
  }

  Q = rope(ctx, Q, positions, params);
  K = rope(ctx, K, positions, params);

  {
    const size_t layer_off_k = k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim * B;
    const size_t layer_off_v = v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim * B;
    ggml_tensor *k_layer = ggml_view_3d(ctx, kv.self_k, kv_dim, n_ctx, B, k_elem * kv_dim,
                                        k_elem * kv_dim * n_ctx, layer_off_k);
    ggml_tensor *v_layer = ggml_view_3d(ctx, kv.self_v, kv_dim, n_ctx, B, v_elem * kv_dim,
                                        v_elem * kv_dim * n_ctx, layer_off_v);
    ggml_tensor *K_rows = ggml_reshape_3d(ctx, K, kv_dim, T, B);
    ggml_tensor *V_rows = ggml_reshape_3d(ctx, V, kv_dim, T, B);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_layer, K_rows, kv_idx));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_layer, V_rows, kv_idx));
  }

  const size_t layer_off_bytes_k = k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim * B;
  const size_t layer_off_bytes_v = v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim * B;
  ggml_tensor *K_att = ggml_view_4d(ctx, kv.self_k, head_dim, T, n_kv_heads, B, k_elem * kv_dim,
                                    k_elem * head_dim, k_elem * kv_dim * n_ctx, layer_off_bytes_k);
  ggml_tensor *V_att = ggml_view_4d(ctx, kv.self_v, head_dim, T, n_kv_heads, B, v_elem * kv_dim,
                                    v_elem * head_dim, v_elem * kv_dim * n_ctx, layer_off_bytes_v);

  ggml_tensor *Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

  ggml_tensor *o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask, scale_attn, 0.0f, 0.0f);
  o = ggml_reshape_3d(ctx, o, q_dim, T, B);

  o = mul_mat_f32acc(ctx, w.attn_o_w, o);
  x = ggml_add(ctx, x, o);
  return ffn(ctx, x, w, rms_eps);
}

} // namespace engine::models::canary_qwen::causal_lm
