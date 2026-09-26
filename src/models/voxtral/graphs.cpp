// engine/models/voxtral/graphs.cpp - ggml graph builders of the native engine
// Voxtral package (see graphs.h for the numerics contract).
//
// Sources, op for op:
//   encoder   : src/runtime/arch/voxtral/encoder.cpp (build_encoder_graph,
//               mha_encoder, ffn, build_block, add_conv1d_bias) over
//               src/runtime/conformer/conformer.cpp (layer_norm, conv_1d_f32)
//   decoder   : src/runtime/arch/voxtral/decoder.cpp (build_prefill_graph,
//               build_prefill_chunk_graph, build_step_graph)
//   causal_lm : src/runtime/causal_lm/causal_lm.cpp (kv_init, block_prefill,
//               block_step, block_step_n, mul_mat_f32acc)
// Dump naming / debug::mark_tensor_for_dump and the batched builders are not
// carried over.

#include "engine/models/voxtral/graphs.h"

#include "engine/models/voxtral/assets.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace engine::models::voxtral {

namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("voxtral graph: " + message);
}

// conformer.cpp: kLayerNormEps.
constexpr float kLayerNormEps = 1e-5f;

// conformer.cpp: layer_norm. y = gamma * (x - mean) / sqrt(var + eps) + beta
// along ne[0].
ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *gamma, ggml_tensor *beta) {
  ggml_tensor *y = ggml_norm(ctx, x, kLayerNormEps);
  y = ggml_mul(ctx, y, gamma);
  if (beta != nullptr) {
    y = ggml_add(ctx, y, beta);
  }
  return y;
}

// conformer.cpp: conv_1d_f32 (single-shot N == 1 path). im2col keeps the
// kernel's own type (vendored ggml_conv_1d forces F16); F32 accumulation is
// requested for F16 kernels (CUDA COMPUTE_16F saturation, a no-op on CPU).
ggml_tensor *conv_1d_f32(ggml_context *ctx, ggml_tensor *kernel, ggml_tensor *data, int stride,
                         int padding, int dilation) {
  ggml_tensor *im2col = ggml_im2col(ctx, kernel, data, stride, /*s1=*/0, padding, /*p1=*/0, dilation,
                                    /*d1=*/0, /*is_2D=*/false, /*dst_type=*/kernel->type);
  // im2col ne = [IC*K, OW, N]; this package only builds N == 1.
  if (im2col->ne[2] != 1) {
    fail("conv_1d_f32: batched input is not supported");
  }
  ggml_tensor *kernel_2d =
      ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);  // [IC*K, OC]
  ggml_tensor *result = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1]),
                                     kernel_2d);  // [OW, OC]
  if (kernel->type == GGML_TYPE_F16) {
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
  }
  return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], 1);
}

// encoder.cpp: add_conv1d_bias. bias [C] -> [1, C, 1, 1], broadcast over T.
ggml_tensor *add_conv1d_bias(ggml_context *ctx, ggml_tensor *conv_out, ggml_tensor *bias_1d) {
  if (bias_1d == nullptr) {
    return conv_out;
  }
  const int64_t channels = bias_1d->ne[0];
  return ggml_add(ctx, conv_out, ggml_reshape_4d(ctx, bias_1d, 1, channels, 1, 1));
}

// encoder.cpp: mha_encoder. Whisper MHSA, x [d_model, T]; q/v/out bias, k
// none; the reference's q * head_dim**-0.5 is folded into the softmax scale.
ggml_tensor *mha_encoder(ggml_context *ctx, ggml_tensor *x, const EncBlockWeights &b, int n_heads,
                         int d_model, bool use_flash) {
  const int head_dim = d_model / n_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t T = x->ne[1];

  ggml_tensor *q = ggml_mul_mat(ctx, b.attn_q_w, x);
  if (b.attn_q_b != nullptr) {
    q = ggml_add(ctx, q, b.attn_q_b);
  }
  ggml_tensor *k = ggml_mul_mat(ctx, b.attn_k_w, x);  // no bias
  ggml_tensor *v = ggml_mul_mat(ctx, b.attn_v_w, x);
  if (b.attn_v_b != nullptr) {
    v = ggml_add(ctx, v, b.attn_v_b);
  }

  q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, n_heads, T), 0, 2, 1, 3);
  k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, n_heads, T), 0, 2, 1, 3);
  v = ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, n_heads, T), 0, 2, 1, 3);

  ggml_tensor *o = nullptr;
  if (use_flash) {
    ggml_tensor *q_c = ggml_cont(ctx, q);
    ggml_tensor *k_c = ggml_cont(ctx, k);
    ggml_tensor *v_c = ggml_cont(ctx, v);
    o = ggml_flash_attn_ext(ctx, q_c, k_c, v_c, nullptr, scale, 0.0f, 0.0f);
    o = ggml_reshape_2d(ctx, o, d_model, T);
  } else {
    ggml_tensor *kq = ggml_mul_mat(ctx, ggml_cont(ctx, k), ggml_cont(ctx, q));
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
    o = ggml_reshape_2d(ctx, o, d_model, T);
  }

  o = ggml_mul_mat(ctx, b.attn_out_w, o);
  if (b.attn_out_b != nullptr) {
    o = ggml_add(ctx, o, b.attn_out_b);
  }
  return o;
}

// encoder.cpp: ffn. fc2(gelu_erf(fc1(x))).
ggml_tensor *encoder_ffn(ggml_context *ctx, ggml_tensor *x, const EncBlockWeights &b) {
  ggml_tensor *h = ggml_mul_mat(ctx, b.fc1_w, x);
  if (b.fc1_b != nullptr) {
    h = ggml_add(ctx, h, b.fc1_b);
  }
  h = ggml_gelu_erf(ctx, h);
  ggml_tensor *o = ggml_mul_mat(ctx, b.fc2_w, h);
  if (b.fc2_b != nullptr) {
    o = ggml_add(ctx, o, b.fc2_b);
  }
  return o;
}

// encoder.cpp: build_block. Pre-LN attention + pre-LN FFN, full residuals.
ggml_tensor *encoder_block(ggml_context *ctx, ggml_tensor *x, const EncBlockWeights &b, int n_heads,
                           int d_model, bool use_flash) {
  {
    ggml_tensor *y = layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    y = mha_encoder(ctx, y, b, n_heads, d_model, use_flash);
    x = ggml_add(ctx, x, y);
  }
  {
    ggml_tensor *y = layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
    y = encoder_ffn(ctx, y, b);
    x = ggml_add(ctx, x, y);
  }
  return x;
}

causal_lm::BlockParams block_params(const VoxtralHParams &hp) {
  causal_lm::BlockParams p;
  p.n_heads = hp.dec_n_heads;
  p.n_kv_heads = hp.dec_n_kv_heads;
  p.head_dim = hp.dec_head_dim;
  p.max_position = hp.dec_max_position_embeddings;
  p.rms_eps = hp.dec_rms_norm_eps;
  p.rope_theta = hp.dec_rope_theta;
  return p;
}

// causal_lm.cpp: rms_norm / mul_mat_f32acc.
ggml_tensor *rms_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight, float eps) {
  return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

ggml_tensor *mul_mat_f32acc(ggml_context *ctx, ggml_tensor *w, ggml_tensor *x) {
  ggml_tensor *y = ggml_mul_mat(ctx, w, x);
  if (w->type == GGML_TYPE_F16) {
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
  }
  return y;
}

// causal_lm.cpp: NeoX RoPE with the arch's exact ext arguments.
ggml_tensor *rope(ggml_context *ctx, ggml_tensor *t, ggml_tensor *positions,
                  const causal_lm::BlockParams &p) {
  return ggml_rope_ext(ctx, t, positions, /*c=*/nullptr, p.head_dim, GGML_ROPE_TYPE_NEOX,
                       p.max_position, p.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

// causal_lm.cpp: attention over K_att / V_att views [D, n_kv, Hkv] with the
// flash path or the manual GQA path (repeat_interleave via reshape+repeat).
ggml_tensor *attend(ggml_context *ctx, ggml_tensor *Q_att, ggml_tensor *K_att, ggml_tensor *V_att,
                    ggml_tensor *mask, const causal_lm::BlockParams &p, int64_t n_kv, int64_t T_q,
                    bool use_flash) {
  const int64_t n_heads = p.n_heads;
  const int64_t n_kv_heads = p.n_kv_heads;
  const int64_t n_groups = n_heads / n_kv_heads;
  const int64_t head_dim = p.head_dim;
  const int64_t q_dim = n_heads * head_dim;
  const float scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

  ggml_tensor *o = nullptr;
  if (use_flash) {
    // flash_attn_ext handles GQA natively (K/V carry n_kv_heads on axis 2).
    o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask, scale_attn, /*max_bias=*/0.0f,
                            /*logit_softcap=*/0.0f);
    o = ggml_reshape_2d(ctx, o, q_dim, T_q);
  } else {
    ggml_tensor *K_att_c = ggml_cont(ctx, K_att);
    ggml_tensor *V_att_c = ggml_cont(ctx, V_att);
    ggml_tensor *K_4d = ggml_reshape_4d(ctx, K_att_c, head_dim, n_kv, 1, n_kv_heads);
    ggml_tensor *V_4d = ggml_reshape_4d(ctx, V_att_c, head_dim, n_kv, 1, n_kv_heads);
    ggml_tensor *K_rep_template = ggml_new_tensor_4d(ctx, K_att->type, head_dim, n_kv, n_groups, n_kv_heads);
    ggml_tensor *V_rep_template = ggml_new_tensor_4d(ctx, V_att->type, head_dim, n_kv, n_groups, n_kv_heads);
    ggml_tensor *K_rep = ggml_repeat(ctx, K_4d, K_rep_template);
    ggml_tensor *V_rep = ggml_repeat(ctx, V_4d, V_rep_template);
    ggml_tensor *K_full = ggml_reshape_3d(ctx, K_rep, head_dim, n_kv, n_heads);
    ggml_tensor *V_full = ggml_reshape_3d(ctx, V_rep, head_dim, n_kv, n_heads);

    ggml_tensor *kq = ggml_mul_mat(ctx, K_full, Q_att);
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, mask, scale_attn, /*max_bias=*/0.0f);
    ggml_tensor *V_t = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, V_t, kq_soft);
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
    o = ggml_reshape_2d(ctx, o, q_dim, T_q);
  }
  return o;
}

// causal_lm.cpp: the post-attention half shared by every block form -
// o-proj + residual, optional last-position slice, RMSNorm, packed SwiGLU,
// down-proj + residual.
ggml_tensor *finish_block(ggml_context *ctx, ggml_tensor *x, ggml_tensor *o, const DecBlockWeights &w,
                          const causal_lm::BlockParams &p, int T_seq, bool slice_last_before_ffn) {
  o = mul_mat_f32acc(ctx, w.attn_o_w, o);
  x = ggml_add(ctx, x, o);
  if (slice_last_before_ffn) {
    const int64_t hidden = x->ne[0];
    const size_t elem = ggml_element_size(x);
    x = ggml_view_2d(ctx, x, hidden, 1, elem * hidden, elem * hidden * static_cast<size_t>(T_seq - 1));
    x = ggml_cont(ctx, x);
  }
  ggml_tensor *ff_norm = rms_norm(ctx, x, w.norm_ffn_w, p.rms_eps);
  ggml_tensor *gate_up = mul_mat_f32acc(ctx, w.ffn_gate_up_w, ff_norm);
  ggml_tensor *ff = ggml_swiglu(ctx, gate_up);
  ff = mul_mat_f32acc(ctx, w.ffn_down_w, ff);
  return ggml_add(ctx, x, ff);
}

// Final RMSNorm (decoder.cpp); the UNTIED lm_head matmul follows at each call site.
ggml_tensor *final_norm(ggml_context *ctx, ggml_tensor *x, const VoxtralWeights &w, const VoxtralHParams &hp) {
  x = ggml_mul(ctx, ggml_rms_norm(ctx, x, hp.dec_rms_norm_eps), w.output_norm_w);
  return x;
}

}  // namespace

// ---------------------------------------------------------------------------
// causal_lm
// ---------------------------------------------------------------------------

namespace causal_lm {

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
  n_ctx = 0;
}

bool kv_init(KvCache &cache, ggml_backend_t backend, int n_ctx, int n_kv_heads, int head_dim,
             int n_layer, ggml_type kv_type) {
  cache.free();
  if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
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
  const int64_t elems = static_cast<int64_t>(n_kv_heads) * head_dim * n_ctx * n_layer;
  cache.self_k = ggml_new_tensor_1d(cache.ctx, kv_type, elems);
  cache.self_v = ggml_new_tensor_1d(cache.ctx, kv_type, elems);
  ggml_set_name(cache.self_k, "kv_self_k");
  ggml_set_name(cache.self_v, "kv_self_v");
  cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
  if (cache.buffer == nullptr) {
    cache.free();
    return false;
  }
  ggml_backend_buffer_clear(cache.buffer, 0);
  cache.n_ctx = n_ctx;
  return true;
}

ggml_tensor *block_prefill(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                           const DecBlockWeights &w, const BlockParams &p, KvCache &kv,
                           int layer_idx, int T_seq, ggml_tensor *mask, ggml_tensor *positions,
                           bool use_flash, bool slice_last_before_ffn) {
  const int64_t n_heads = p.n_heads;
  const int64_t n_kv_heads = p.n_kv_heads;
  const int64_t head_dim = p.head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  const size_t n_ctx = static_cast<size_t>(kv.n_ctx);
  const size_t k_elem = ggml_element_size(kv.self_k);
  const size_t v_elem = ggml_element_size(kv.self_v);

  ggml_tensor *x_norm = rms_norm(ctx, x, w.norm_attn_w, p.rms_eps);
  // Q/K/V kept as three mul_mats (packing regresses on Metal in the arch).
  ggml_tensor *Q = mul_mat_f32acc(ctx, w.attn_q_w, x_norm);
  ggml_tensor *K = mul_mat_f32acc(ctx, w.attn_k_w, x_norm);
  ggml_tensor *V = mul_mat_f32acc(ctx, w.attn_v_w, x_norm);
  Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, T_seq, 1);
  K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, T_seq, 1);
  V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, T_seq, 1);
  // Llama: no per-head Q/K norm.
  Q = rope(ctx, Q, positions, p);
  K = rope(ctx, K, positions, p);

  // KV write: [D, Hkv, T, 1] is the cache's own layout, so a 1-D cpy into the
  // layer slab (kv_batch_slot 0, kv_n_batch 1 in the arch) handles it.
  const size_t slab_off = static_cast<size_t>(layer_idx) * n_ctx * static_cast<size_t>(kv_dim);
  {
    const size_t n_elem = static_cast<size_t>(T_seq) * static_cast<size_t>(kv_dim);
    ggml_tensor *k_dst = ggml_view_1d(ctx, kv.self_k, static_cast<int64_t>(n_elem), k_elem * slab_off);
    ggml_tensor *v_dst = ggml_view_1d(ctx, kv.self_v, static_cast<int64_t>(n_elem), v_elem * slab_off);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
  }
  // Read back as strided [D, T, Hkv] views.
  ggml_tensor *K_att = ggml_view_3d(ctx, kv.self_k, head_dim, T_seq, n_kv_heads, k_elem * kv_dim,
                                    k_elem * head_dim, k_elem * slab_off);
  ggml_tensor *V_att = ggml_view_3d(ctx, kv.self_v, head_dim, T_seq, n_kv_heads, v_elem * kv_dim,
                                    v_elem * head_dim, v_elem * slab_off);
  ggml_tensor *Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

  ggml_tensor *o = attend(ctx, Q_att, K_att, V_att, mask, p, T_seq, T_seq, use_flash);
  return finish_block(ctx, x, o, w, p, T_seq, slice_last_before_ffn);
}

ggml_tensor *block_step(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                        const DecBlockWeights &w, const BlockParams &p, KvCache &kv,
                        int layer_idx, int max_n_kv, ggml_tensor *mask, ggml_tensor *position,
                        ggml_tensor *kv_idx, bool use_flash) {
  const int64_t n_heads = p.n_heads;
  const int64_t n_kv_heads = p.n_kv_heads;
  const int64_t head_dim = p.head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  const int64_t n_ctx = kv.n_ctx;
  const size_t k_elem = ggml_element_size(kv.self_k);
  const size_t v_elem = ggml_element_size(kv.self_v);

  ggml_tensor *x_norm = rms_norm(ctx, x, w.norm_attn_w, p.rms_eps);
  ggml_tensor *Q = mul_mat_f32acc(ctx, w.attn_q_w, x_norm);
  ggml_tensor *K = mul_mat_f32acc(ctx, w.attn_k_w, x_norm);
  ggml_tensor *V = mul_mat_f32acc(ctx, w.attn_v_w, x_norm);
  Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, 1, 1);
  K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, 1, 1);
  V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, 1, 1);
  Q = rope(ctx, Q, position, p);
  K = rope(ctx, K, position, p);

  // KV write via set_rows at a dynamic index: static topology, only kv_idx's
  // value changes per step.
  const size_t layer_off_k = k_elem * static_cast<size_t>(layer_idx) * static_cast<size_t>(n_ctx) *
                             static_cast<size_t>(kv_dim);
  const size_t layer_off_v = v_elem * static_cast<size_t>(layer_idx) * static_cast<size_t>(n_ctx) *
                             static_cast<size_t>(kv_dim);
  {
    ggml_tensor *k_layer = ggml_view_2d(ctx, kv.self_k, kv_dim, n_ctx, k_elem * kv_dim, layer_off_k);
    ggml_tensor *v_layer = ggml_view_2d(ctx, kv.self_v, kv_dim, n_ctx, v_elem * kv_dim, layer_off_v);
    ggml_tensor *K_row = ggml_reshape_2d(ctx, K, kv_dim, 1);
    ggml_tensor *V_row = ggml_reshape_2d(ctx, V, kv_dim, 1);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_layer, K_row, kv_idx));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_layer, V_row, kv_idx));
  }
  // Attention reads the FULL [0, max_n_kv) window; the mask hides empty slots.
  ggml_tensor *K_att = ggml_view_3d(ctx, kv.self_k, head_dim, max_n_kv, n_kv_heads, k_elem * kv_dim,
                                    k_elem * head_dim, layer_off_k);
  ggml_tensor *V_att = ggml_view_3d(ctx, kv.self_v, head_dim, max_n_kv, n_kv_heads, v_elem * kv_dim,
                                    v_elem * head_dim, layer_off_v);
  ggml_tensor *Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

  ggml_tensor *o = attend(ctx, Q_att, K_att, V_att, mask, p, max_n_kv, 1, use_flash);
  return finish_block(ctx, x, o, w, p, 1, /*slice_last_before_ffn=*/false);
}

ggml_tensor *block_step_n(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                          const DecBlockWeights &w, const BlockParams &p, KvCache &kv,
                          int layer_idx, int T_seq, int max_n_kv, ggml_tensor *mask,
                          ggml_tensor *positions, ggml_tensor *kv_idx, bool use_flash) {
  const int64_t n_heads = p.n_heads;
  const int64_t n_kv_heads = p.n_kv_heads;
  const int64_t head_dim = p.head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  const int64_t n_ctx = kv.n_ctx;
  const size_t k_elem = ggml_element_size(kv.self_k);
  const size_t v_elem = ggml_element_size(kv.self_v);

  ggml_tensor *x_norm = rms_norm(ctx, x, w.norm_attn_w, p.rms_eps);
  ggml_tensor *Q = mul_mat_f32acc(ctx, w.attn_q_w, x_norm);
  ggml_tensor *K = mul_mat_f32acc(ctx, w.attn_k_w, x_norm);
  ggml_tensor *V = mul_mat_f32acc(ctx, w.attn_v_w, x_norm);
  Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, T_seq, 1);
  K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, T_seq, 1);
  V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, T_seq, 1);
  Q = rope(ctx, Q, positions, p);
  K = rope(ctx, K, positions, p);

  const size_t layer_off_k = k_elem * static_cast<size_t>(layer_idx) * static_cast<size_t>(n_ctx) *
                             static_cast<size_t>(kv_dim);
  const size_t layer_off_v = v_elem * static_cast<size_t>(layer_idx) * static_cast<size_t>(n_ctx) *
                             static_cast<size_t>(kv_dim);
  {
    ggml_tensor *k_layer = ggml_view_2d(ctx, kv.self_k, kv_dim, n_ctx, k_elem * kv_dim, layer_off_k);
    ggml_tensor *v_layer = ggml_view_2d(ctx, kv.self_v, kv_dim, n_ctx, v_elem * kv_dim, layer_off_v);
    ggml_tensor *K_rows = ggml_reshape_2d(ctx, K, kv_dim, T_seq);
    ggml_tensor *V_rows = ggml_reshape_2d(ctx, V, kv_dim, T_seq);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_layer, K_rows, kv_idx));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_layer, V_rows, kv_idx));
  }
  ggml_tensor *K_att = ggml_view_3d(ctx, kv.self_k, head_dim, max_n_kv, n_kv_heads, k_elem * kv_dim,
                                    k_elem * head_dim, layer_off_k);
  ggml_tensor *V_att = ggml_view_3d(ctx, kv.self_v, head_dim, max_n_kv, n_kv_heads, v_elem * kv_dim,
                                    v_elem * head_dim, layer_off_v);
  ggml_tensor *Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

  ggml_tensor *o = attend(ctx, Q_att, K_att, V_att, mask, p, max_n_kv, T_seq, use_flash);
  return finish_block(ctx, x, o, w, p, T_seq, /*slice_last_before_ffn=*/false);
}

}  // namespace causal_lm

// ---------------------------------------------------------------------------
// Encoder + projector (arch encoder.cpp: build_encoder_graph)
// ---------------------------------------------------------------------------

EncoderBuild build_encoder_graph(ggml_context *ctx, const VoxtralWeights &w,
                                 const VoxtralHParams &hp, int n_mel_frames, bool use_flash) {
  if (ctx == nullptr || n_mel_frames <= 0 || n_mel_frames % 2 != 0) {
    fail("encoder: invalid n_mel_frames=" + std::to_string(n_mel_frames) + " (must be positive even)");
  }
  const int d_model = hp.enc_d_model;
  const int n_mels = hp.enc_num_mel_bins;
  const int n_heads = hp.enc_n_heads;
  const int T_enc = n_mel_frames / 2;
  if (T_enc != hp.enc_max_source_positions) {
    fail("encoder: T_enc=" + std::to_string(T_enc) + " != max_source_positions=" +
         std::to_string(hp.enc_max_source_positions) + " (mel must be padded to " +
         std::to_string(2 * hp.enc_max_source_positions) + " frames)");
  }

  EncoderBuild eb;
  eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_mels, n_mel_frames);
  ggml_set_name(eb.mel_in, "enc.mel.in");
  ggml_set_input(eb.mel_in);

  // Conv stem: [n_mels, T] -> [T, n_mels] for conv_1d, two GELU convs
  // (k3 s1 p1, then k3 s2 p1), back to [d_model, T_enc].
  ggml_tensor *x = ggml_cont(ctx, ggml_transpose(ctx, eb.mel_in));  // [T, n_mels]
  x = conv_1d_f32(ctx, w.conv0_w, x, /*s=*/1, /*p=*/1, /*d=*/1);
  x = add_conv1d_bias(ctx, x, w.conv0_b);
  x = ggml_gelu_erf(ctx, x);  // [T, d_model]
  x = conv_1d_f32(ctx, w.conv1_w, x, /*s=*/2, /*p=*/1, /*d=*/1);
  x = add_conv1d_bias(ctx, x, w.conv1_b);
  x = ggml_gelu_erf(ctx, x);
  x = ggml_cont(ctx, ggml_transpose(ctx, x));  // [d_model, T_enc]

  // + fixed sinusoidal positional embedding (a weight, not an input).
  x = ggml_add(ctx, x, w.pos_emb_w);

  for (const auto &block : w.enc_blocks) {
    x = encoder_block(ctx, x, block, n_heads, d_model, use_flash);
  }
  x = layer_norm(ctx, x, w.ln_post_w, w.ln_post_b);  // enc.out

  // Projector: group 4 consecutive frames ([d_model, 1500] -> [5120, 375];
  // ggml's d-innermost order is the reference's C-order [1500, 1280], so a
  // plain reshape yields token i = concat(frames 4i..4i+3)), then
  // Linear -> GELU -> Linear (no biases).
  const int n_audio = hp.audio_tokens_per_chunk();
  ggml_tensor *xc = ggml_cont(ctx, x);
  ggml_tensor *grouped = ggml_reshape_2d(ctx, xc, hp.proj_in, n_audio);
  ggml_tensor *p = ggml_mul_mat(ctx, w.proj_linear_1_w, grouped);
  p = ggml_gelu_erf(ctx, p);
  p = ggml_mul_mat(ctx, w.proj_linear_2_w, p);  // [hidden, n_audio]
  ggml_set_name(p, "proj.out");

  eb.out = p;
  ggml_set_output(eb.out);
  eb.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
  ggml_build_forward_expand(eb.graph, eb.out);
  return eb;
}

// ---------------------------------------------------------------------------
// Decoder (arch decoder.cpp)
// ---------------------------------------------------------------------------

PrefillBuild build_prefill_graph(ggml_context *ctx, const VoxtralWeights &w,
                                 const VoxtralHParams &hp, causal_lm::KvCache &kv, int T_prompt,
                                 int T_enc, int prefix_len, int suffix_len, bool use_flash) {
  if (ctx == nullptr || T_prompt <= 0 || T_enc <= 0) {
    fail("prefill: invalid arg (T_prompt=" + std::to_string(T_prompt) + ", T_enc=" + std::to_string(T_enc) + ")");
  }
  if (prefix_len < 0 || suffix_len < 0 || prefix_len + T_enc + suffix_len != T_prompt) {
    fail("prefill: prefix+T_enc+suffix != T_prompt");
  }
  if (!kv.allocated()) {
    fail("prefill: kv cache not initialized");
  }
  if (T_prompt > kv.n_ctx) {
    fail("prefill: T_prompt=" + std::to_string(T_prompt) + " exceeds kv n_ctx=" + std::to_string(kv.n_ctx));
  }
  const int64_t hidden = hp.dec_hidden;
  const int64_t vocab = hp.dec_vocab_size;
  const int n_layer = hp.dec_n_layers;
  const auto params = block_params(hp);

  PrefillBuild pb;
  pb.input_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt);
  ggml_set_name(pb.input_ids_in, "dec.input_ids");
  ggml_set_input(pb.input_ids_in);
  pb.enc_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, T_enc);
  ggml_set_name(pb.enc_out_in, "dec.enc_out");
  ggml_set_input(pb.enc_out_in);
  pb.positions_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt);
  ggml_set_name(pb.positions_in, "dec.positions");
  ggml_set_input(pb.positions_in);
  pb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T_prompt, T_prompt);
  ggml_set_name(pb.mask_in, "dec.attn_mask");
  ggml_set_input(pb.mask_in);

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, /*size=*/16384, /*grads=*/false);
  pb.graph = gf;

  // Token-embed every prompt position, then splice the projector output over
  // the placeholder run: prefix | audio | suffix (3-way concat).
  ggml_tensor *token_emb_all = ggml_get_rows(ctx, w.token_embd_w, pb.input_ids_in);
  const size_t emb_elem = ggml_element_size(token_emb_all);
  ggml_tensor *x_prefix = nullptr;
  if (prefix_len > 0) {
    x_prefix = ggml_cont(ctx, ggml_view_2d(ctx, token_emb_all, hidden, prefix_len, emb_elem * hidden, 0));
  }
  ggml_tensor *x_suffix = nullptr;
  if (suffix_len > 0) {
    x_suffix = ggml_cont(ctx, ggml_view_2d(ctx, token_emb_all, hidden, suffix_len, emb_elem * hidden,
                                           emb_elem * hidden * static_cast<size_t>(prefix_len + T_enc)));
  }
  ggml_tensor *x = pb.enc_out_in;
  if (x_prefix != nullptr) {
    x = ggml_concat(ctx, x_prefix, x, /*dim=*/1);
  }
  if (x_suffix != nullptr) {
    x = ggml_concat(ctx, x, x_suffix, /*dim=*/1);
  }

  for (int il = 0; il < n_layer; ++il) {
    // slice_last (the arch's non-dump setting) on the last block only.
    x = causal_lm::block_prefill(ctx, gf, x, w.dec_blocks[static_cast<size_t>(il)], params, kv, il, T_prompt,
                                 pb.mask_in, pb.positions_in, use_flash, /*slice_last=*/il == n_layer - 1);
  }

  // Final RMSNorm + UNTIED lm_head on the (already sliced) last position.
  x = final_norm(ctx, x, w, hp);
  ggml_tensor *logits = ggml_mul_mat(ctx, w.output_w, x);
  logits = ggml_reshape_1d(ctx, logits, vocab);
  ggml_set_name(logits, "dec.logits_raw");
  pb.out = logits;
  ggml_set_output(pb.out);
  ggml_build_forward_expand(gf, pb.out);
  return pb;
}

PrefillChunkBuild build_prefill_chunk_graph(ggml_context *ctx, const VoxtralWeights &w,
                                            const VoxtralHParams &hp, causal_lm::KvCache &kv,
                                            int T_chunk, int max_n_kv, int pre_n, int aud_n,
                                            int suf_n, bool use_flash, bool want_logits) {
  if (ctx == nullptr || T_chunk <= 0 || max_n_kv < T_chunk) {
    fail("prefill chunk: invalid chunk (T_chunk=" + std::to_string(T_chunk) + " max_n_kv=" +
         std::to_string(max_n_kv) + ")");
  }
  if (pre_n < 0 || aud_n < 0 || suf_n < 0 || pre_n + aud_n + suf_n != T_chunk) {
    fail("prefill chunk: segments do not sum to T_chunk");
  }
  if (!kv.allocated()) {
    fail("prefill chunk: kv cache not initialized");
  }
  if (max_n_kv > kv.n_ctx) {
    fail("prefill chunk: max_n_kv=" + std::to_string(max_n_kv) + " exceeds kv n_ctx=" + std::to_string(kv.n_ctx));
  }
  const int64_t hidden = hp.dec_hidden;
  const int64_t vocab = hp.dec_vocab_size;
  const int n_layer = hp.dec_n_layers;
  const auto params = block_params(hp);

  PrefillChunkBuild pb;
  // A chunk wholly inside the audio run needs no token embeddings; leave the
  // input out so nothing unreachable is ever an (unallocated) upload target.
  if (pre_n + suf_n > 0) {
    pb.input_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_chunk);
    ggml_set_name(pb.input_ids_in, "dec.chunk.input_ids");
    ggml_set_input(pb.input_ids_in);
  }
  if (aud_n > 0) {
    pb.enc_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, aud_n);
    ggml_set_name(pb.enc_out_in, "dec.chunk.enc_out");
    ggml_set_input(pb.enc_out_in);
  }
  pb.positions_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_chunk);
  ggml_set_name(pb.positions_in, "dec.chunk.positions");
  ggml_set_input(pb.positions_in);
  pb.kv_idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T_chunk);
  ggml_set_name(pb.kv_idx_in, "dec.chunk.kv_idx");
  ggml_set_input(pb.kv_idx_in);
  pb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, max_n_kv, T_chunk);
  ggml_set_name(pb.mask_in, "dec.chunk.attn_mask");
  ggml_set_input(pb.mask_in);

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, /*size=*/16384, /*grads=*/false);
  pb.graph = gf;

  ggml_tensor *token_emb = pb.input_ids_in != nullptr ? ggml_get_rows(ctx, w.token_embd_w, pb.input_ids_in) : nullptr;
  const size_t emb_elem = token_emb != nullptr ? ggml_element_size(token_emb) : 0;
  ggml_tensor *x = nullptr;
  if (pre_n > 0) {
    x = ggml_cont(ctx, ggml_view_2d(ctx, token_emb, hidden, pre_n, emb_elem * hidden, 0));
  }
  if (aud_n > 0) {
    x = (x == nullptr) ? pb.enc_out_in : ggml_concat(ctx, x, pb.enc_out_in, /*dim=*/1);
  }
  if (suf_n > 0) {
    ggml_tensor *x_suffix = ggml_cont(ctx, ggml_view_2d(ctx, token_emb, hidden, suf_n, emb_elem * hidden,
                                                        emb_elem * hidden * static_cast<size_t>(pre_n + aud_n)));
    x = (x == nullptr) ? x_suffix : ggml_concat(ctx, x, x_suffix, /*dim=*/1);
  }

  for (int il = 0; il < n_layer; ++il) {
    x = causal_lm::block_step_n(ctx, gf, x, w.dec_blocks[static_cast<size_t>(il)], params, kv, il, T_chunk,
                                max_n_kv, pb.mask_in, pb.positions_in, pb.kv_idx_in, use_flash);
  }

  if (want_logits) {
    x = final_norm(ctx, x, w, hp);
    ggml_tensor *last_x = ggml_view_2d(ctx, x, hidden, 1, ggml_element_size(x) * hidden,
                                       ggml_element_size(x) * hidden * static_cast<size_t>(T_chunk - 1));
    last_x = ggml_cont(ctx, last_x);
    ggml_tensor *logits = ggml_mul_mat(ctx, w.output_w, last_x);
    logits = ggml_reshape_1d(ctx, logits, vocab);
    ggml_set_name(logits, "dec.logits_raw");
    pb.out = logits;
    ggml_set_output(pb.out);
    ggml_build_forward_expand(gf, pb.out);
  } else {
    ggml_build_forward_expand(gf, x);
  }
  return pb;
}

StepBuild build_step_graph(ggml_context *ctx, const VoxtralWeights &w, const VoxtralHParams &hp,
                           causal_lm::KvCache &kv, int max_n_kv, bool use_flash) {
  if (ctx == nullptr || max_n_kv <= 0) {
    fail("step: invalid max_n_kv=" + std::to_string(max_n_kv));
  }
  if (!kv.allocated()) {
    fail("step: kv cache not initialized");
  }
  if (max_n_kv > kv.n_ctx) {
    fail("step: max_n_kv=" + std::to_string(max_n_kv) + " exceeds kv n_ctx=" + std::to_string(kv.n_ctx));
  }
  const int64_t vocab = hp.dec_vocab_size;
  const int n_layer = hp.dec_n_layers;
  const auto params = block_params(hp);

  StepBuild sb;
  sb.input_id_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
  ggml_set_name(sb.input_id_in, "step.input_id");
  ggml_set_input(sb.input_id_in);
  sb.position_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
  ggml_set_name(sb.position_in, "step.position");
  ggml_set_input(sb.position_in);
  sb.kv_idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
  ggml_set_name(sb.kv_idx_in, "step.kv_idx");
  ggml_set_input(sb.kv_idx_in);
  sb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, max_n_kv, 1);
  ggml_set_name(sb.mask_in, "step.mask");
  ggml_set_input(sb.mask_in);

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
  sb.graph = gf;

  ggml_tensor *x = ggml_get_rows(ctx, w.token_embd_w, sb.input_id_in);
  for (int il = 0; il < n_layer; ++il) {
    x = causal_lm::block_step(ctx, gf, x, w.dec_blocks[static_cast<size_t>(il)], params, kv, il, max_n_kv,
                              sb.mask_in, sb.position_in, sb.kv_idx_in, use_flash);
  }
  x = final_norm(ctx, x, w, hp);
  ggml_tensor *logits = ggml_mul_mat(ctx, w.output_w, x);
  logits = ggml_reshape_1d(ctx, logits, vocab);
  ggml_set_name(logits, "step.logits");
  ggml_set_output(logits);
  sb.logits = logits;

  ggml_tensor *amax = ggml_argmax(ctx, logits);
  ggml_set_name(amax, "step.argmax");
  sb.out = amax;
  ggml_set_output(sb.out);
  ggml_build_forward_expand(gf, sb.out);
  ggml_build_forward_expand(gf, logits);
  return sb;
}

}  // namespace engine::models::voxtral
