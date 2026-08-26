// engine/models/moonshine_streaming/graphs.cpp - ggml graph builders for the
// native engine Moonshine-Streaming ASR package.
//
// Ported from src/runtime/arch/moonshine_streaming/{encoder,decoder}.cpp with
// identical graph topology and numerics. The transcribe-side tensor-dump
// plumbing is intentionally not carried over (engine tracing covers
// observability), and the three conformer helpers the arch borrowed
// (conv_1d_f32 / layer_norm / named) are reimplemented locally so the package
// has no dependency on src/runtime/.

#include "engine/models/moonshine_streaming/graphs_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace engine::models::moonshine_streaming {

namespace {

constexpr float kLayerNormEps = 1e-5f;

ggml_tensor *named(ggml_tensor *t, const char *name) {
  if (t != nullptr) {
    ggml_set_name(t, name);
  }
  return t;
}

ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *gamma,
                        ggml_tensor *beta) {
  ggml_tensor *y = ggml_norm(ctx, x, kLayerNormEps);
  y = ggml_mul(ctx, y, gamma);
  if (beta != nullptr) {
    y = ggml_add(ctx, y, beta);
  }
  return y;
}

// im2col + mul_mat Conv1d on [T, C] layout data (mirrors
// src/runtime/conformer/conformer.cpp conv_1d_f32).
ggml_tensor *conv_1d_f32(ggml_context *ctx, ggml_tensor *kernel,
                         ggml_tensor *data, int stride, int padding,
                         int dilation) {
  ggml_tensor *im2col = ggml_im2col(
      ctx, kernel, data, stride, /*s1=*/0, padding, /*p1=*/0, dilation,
      /*d1=*/0, /*is_2D=*/false, /*dst_type=*/kernel->type);

  const int64_t N = im2col->ne[2];
  ggml_tensor *kernel_2d = ggml_reshape_2d(
      ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);

  const bool kernel_needs_f32_acc = (kernel->type == GGML_TYPE_F16);

  if (N == 1) {
    ggml_tensor *result = ggml_mul_mat(
        ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1]),
        kernel_2d);
    if (kernel_needs_f32_acc) {
      ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    }
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], 1);
    return result;
  }

  ggml_tensor *result = ggml_mul_mat(ctx, kernel_2d, im2col);
  if (kernel_needs_f32_acc) {
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
  }
  result = ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
  return result;
}

ggml_tensor *add_conv1d_bias(ggml_context *ctx, ggml_tensor *conv_out,
                             ggml_tensor *bias_1d) {
  if (bias_1d == nullptr) {
    return conv_out;
  }
  const int64_t channels = bias_1d->ne[0];
  ggml_tensor *bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, channels, 1, 1);
  return ggml_add(ctx, conv_out, bias_4d);
}

ggml_tensor *pad_head_dim(ggml_context *ctx, ggml_tensor *t, int pad) {
  if (pad <= 0) {
    return t;
  }
  return ggml_pad(ctx, t, /*p0=*/pad, /*p1=*/0, /*p2=*/0, /*p3=*/0);
}

ggml_tensor *unpad_head_dim(ggml_context *ctx, ggml_tensor *t, int head_dim,
                            int pad) {
  if (pad <= 0) {
    return t;
  }
  return ggml_view_3d(ctx, t, head_dim, t->ne[1], t->ne[2], t->nb[1], t->nb[2],
                      0);
}

// GPT-J / interleaved partial RoPE (GGML_ROPE_TYPE_NORMAL). Decoder only -
// the streaming encoder has no RoPE.
ggml_tensor *apply_partial_rope(ggml_context *ctx, ggml_tensor *x,
                                ggml_tensor *positions,
                                const MoonshineStreamingHParams &hp,
                                int head_dim_rot) {
  return ggml_rope_ext(ctx, x, positions, /*c=*/nullptr, head_dim_rot,
                       GGML_ROPE_TYPE_NORMAL,
                       /*n_ctx_orig=*/0, hp.rope_theta,
                       /*freq_scale=*/1.0f,
                       /*ext_factor=*/0.0f,
                       /*attn_factor=*/1.0f,
                       /*beta_fast=*/32.0f,
                       /*beta_slow=*/1.0f);
}

// asinh(z) = log(z + sqrt(z^2 + 1)).
// PCM range and exp(log_k) ~ 0.5 keep z in roughly [-0.5, 0.5], so the
// stability recasting PyTorch does for large |z| is not needed.
ggml_tensor *asinh_op(ggml_context *ctx, ggml_tensor *z) {
  ggml_tensor *z2 = ggml_sqr(ctx, z);
  ggml_tensor *z2_plus_one = ggml_scale_bias(ctx, z2, /*s=*/1.0f, /*b=*/1.0f);
  ggml_tensor *s = ggml_sqrt(ctx, z2_plus_one);
  ggml_tensor *z_plus_s = ggml_add(ctx, z, s);
  return ggml_log(ctx, z_plus_s);
}

// SwiGLU MLP: fc1 hidden -> 2*intermediate split into [x_proj, gate].
ggml_tensor *ffn_decoder_swiglu(ggml_context *ctx, ggml_tensor *x,
                                ggml_tensor *fc1_w, ggml_tensor *fc1_b,
                                ggml_tensor *fc2_w, ggml_tensor *fc2_b,
                                int ffn_dim) {
  ggml_tensor *h = ggml_mul_mat(ctx, fc1_w, x);
  if (fc1_b != nullptr) {
    h = ggml_add(ctx, h, fc1_b);
  }
  const int64_t T = h->ne[1];
  const size_t el = ggml_element_size(h);
  const size_t half_bytes = static_cast<size_t>(ffn_dim) * el;

  ggml_tensor *x_proj = ggml_view_2d(ctx, h, ffn_dim, T, h->nb[1], 0);
  ggml_tensor *gate = ggml_view_2d(ctx, h, ffn_dim, T, h->nb[1], half_bytes);

  ggml_tensor *y = ggml_mul(ctx, ggml_silu(ctx, ggml_cont(ctx, gate)),
                            ggml_cont(ctx, x_proj));
  ggml_tensor *o = ggml_mul_mat(ctx, fc2_w, y);
  if (fc2_b != nullptr) {
    o = ggml_add(ctx, o, fc2_b);
  }
  return o;
}

// ---------------------------------------------------------------------------
// Encoder attention / blocks
// ---------------------------------------------------------------------------

// Encoder MHSA without RoPE, with a per-layer sliding-window mask.
//
// x:    [enc_d_model, T]
// mask: [T, T] f32 (uploaded by the caller; cast to F16 inside the graph)
//
// q_w/k_w/v_w have ggml ne [d_model, attn_dim]; out_w has ne
// [attn_dim, d_model]. attn_dim = n_heads * head_dim, which may be smaller
// than d_model (small/medium); for tiny they are equal.
ggml_tensor *mha_encoder_swa(ggml_context *ctx, ggml_tensor *x,
                             ggml_tensor *mask_f32, ggml_tensor *q_w,
                             ggml_tensor *k_w, ggml_tensor *v_w,
                             ggml_tensor *out_w,
                             const MoonshineStreamingHParams &hp, int n_heads,
                             int head_dim, bool use_flash) {
  const int attn_dim = n_heads * head_dim;
  const int head_dim_pad = hp.enc_head_dim_padded();
  const int pad = head_dim_pad - head_dim;
  // HF reference uses the unpadded head_dim for the scale.
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t T = x->ne[1];

  ggml_tensor *Q = ggml_mul_mat(ctx, q_w, x);
  ggml_tensor *K = ggml_mul_mat(ctx, k_w, x);
  ggml_tensor *V = ggml_mul_mat(ctx, v_w, x);

  Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, T, 1);
  K = ggml_reshape_4d(ctx, K, head_dim, n_heads, T, 1);
  V = ggml_reshape_4d(ctx, V, head_dim, n_heads, T, 1);

  auto to_attn_layout = [&](ggml_tensor *t) -> ggml_tensor * {
    t = ggml_permute(ctx, t, 0, 2, 1, 3);
    t = ggml_cont(ctx, t);
    if (pad > 0) {
      t = ggml_pad(ctx, t, pad, 0, 0, 0);
    }
    return t;
  };
  Q = to_attn_layout(Q);
  K = to_attn_layout(K);
  V = to_attn_layout(V);

  // ggml_soft_max_ext / ggml_flash_attn_ext require an F16 mask.
  ggml_tensor *mask_f16 = ggml_cast(ctx, mask_f32, GGML_TYPE_F16);

  ggml_tensor *o;
  if (use_flash) {
    o = ggml_flash_attn_ext(ctx, Q, K, V, mask_f16, scale, 0.0f, 0.0f);
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont(ctx, o);
  } else {
    ggml_tensor *kq = ggml_mul_mat(ctx, K, Q);
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, mask_f16, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
  }

  if (pad > 0) {
    o = ggml_view_3d(ctx, o, head_dim, T, n_heads, o->nb[1], o->nb[2], 0);
    o = ggml_cont(ctx, o);
  }

  o = ggml_permute(ctx, o, 0, 2, 1, 3);
  o = ggml_cont(ctx, o);
  o = ggml_reshape_2d(ctx, o, attn_dim, T);

  o = ggml_mul_mat(ctx, out_w, o);
  return o;
}

// Encoder MLP: fc1 (with bias) -> GELU(erf) -> fc2 (with bias).
ggml_tensor *ffn_encoder(ggml_context *ctx, ggml_tensor *x, ggml_tensor *fc1_w,
                         ggml_tensor *fc1_b, ggml_tensor *fc2_w,
                         ggml_tensor *fc2_b) {
  ggml_tensor *h = ggml_mul_mat(ctx, fc1_w, x);
  if (fc1_b != nullptr) {
    h = ggml_add(ctx, h, fc1_b);
  }
  h = ggml_gelu_erf(ctx, h);
  ggml_tensor *o = ggml_mul_mat(ctx, fc2_w, h);
  if (fc2_b != nullptr) {
    o = ggml_add(ctx, o, fc2_b);
  }
  return o;
}

ggml_tensor *build_encoder_block(ggml_context *ctx, ggml_tensor *x,
                                 ggml_tensor *mask_f32,
                                 const MoonshineStreamingEncBlock &b,
                                 const MoonshineStreamingHParams &hp,
                                 int n_heads, int head_dim, bool use_flash) {
  {
    ggml_tensor *y = layer_norm(ctx, x, b.norm_attn_w, /*beta=*/nullptr);
    y = mha_encoder_swa(ctx, y, mask_f32, b.attn_q_w, b.attn_k_w, b.attn_v_w,
                        b.attn_out_w, hp, n_heads, head_dim, use_flash);
    x = ggml_add(ctx, x, y);
  }
  {
    ggml_tensor *y = layer_norm(ctx, x, b.norm_ffn_w, /*beta=*/nullptr);
    y = ffn_encoder(ctx, y, b.ffn_fc1_w, b.ffn_fc1_b, b.ffn_fc2_w, b.ffn_fc2_b);
    x = ggml_add(ctx, x, y);
  }
  return x;
}

// Output time of one streaming Conv1d (left-pad k-1, stride 2):
//   T_out = floor((T_in + (k-1) - k) / s) + 1 = ceil(T_in / s)
int causal_conv1d_t_out(int T_in, int /*k*/, int s) {
  if (T_in <= 0 || s <= 0) {
    return 0;
  }
  return (T_in + s - 1) / s;
}

// ---------------------------------------------------------------------------
// Decoder attention (KV-cached)
// ---------------------------------------------------------------------------

ggml_tensor *mha_self_cached(ggml_context *ctx, ggml_cgraph *gf,
                             ggml_tensor *x,
                             MoonshineStreamingKvCache &kv_cache,
                             ggml_tensor *pos_ids, ggml_tensor *mask,
                             ggml_tensor *q_w, ggml_tensor *k_w,
                             ggml_tensor *v_w, ggml_tensor *out_w,
                             const MoonshineStreamingHParams &hp, int n_heads,
                             int d_model, int il, int n_past, int n_tokens,
                             int n_kv, bool use_flash) {
  const int head_dim = d_model / n_heads;
  const int head_dim_pad = hp.dec_head_dim_padded();
  const int head_dim_rot = hp.dec_head_dim_rot();
  const int pad = head_dim_pad - head_dim;
  const int n_ctx = kv_cache.n_ctx;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  ggml_tensor *Qcur = ggml_mul_mat(ctx, q_w, x);
  ggml_tensor *Kcur = ggml_mul_mat(ctx, k_w, x);
  ggml_tensor *Vcur = ggml_mul_mat(ctx, v_w, x);

  Qcur = ggml_reshape_4d(ctx, Qcur, head_dim, n_heads, n_tokens, 1);
  Kcur = ggml_reshape_4d(ctx, Kcur, head_dim, n_heads, n_tokens, 1);
  Vcur = ggml_reshape_4d(ctx, Vcur, head_dim, n_heads, n_tokens, 1);

  ggml_tensor *Q_rope = apply_partial_rope(ctx, Qcur, pos_ids, hp, head_dim_rot);
  ggml_tensor *K_rope = apply_partial_rope(ctx, Kcur, pos_ids, hp, head_dim_rot);

  // Q needs [head_dim, n_tokens, n_heads, 1] for attention. K/V are already in
  // cache memory order [head_dim, n_heads, n_tokens, 1] after RoPE (or
  // reshape, for V), so they are written straight to the cache.
  ggml_tensor *Q_unpad = ggml_cont(ctx, ggml_permute(ctx, Q_rope, 0, 2, 1, 3));

  {
    const size_t k_elem = ggml_element_size(kv_cache.self_k);
    const size_t v_elem = ggml_element_size(kv_cache.self_v);

    ggml_tensor *k_dst = ggml_view_1d(
        ctx, kv_cache.self_k, static_cast<int64_t>(n_tokens) * d_model,
        k_elem * static_cast<size_t>(static_cast<int64_t>(il) * n_ctx * d_model +
                                     static_cast<int64_t>(n_past) * d_model));
    ggml_tensor *v_dst = ggml_view_1d(
        ctx, kv_cache.self_v, static_cast<int64_t>(n_tokens) * d_model,
        v_elem * static_cast<size_t>(static_cast<int64_t>(il) * n_ctx * d_model +
                                     static_cast<int64_t>(n_past) * d_model));

    ggml_build_forward_expand(gf, ggml_cpy(ctx, K_rope, k_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v_dst));
  }

  const size_t k_elem = ggml_element_size(kv_cache.self_k);
  ggml_tensor *K = ggml_view_3d(
      ctx, kv_cache.self_k, head_dim, n_kv, n_heads, k_elem * d_model,
      k_elem * head_dim,
      k_elem * static_cast<size_t>(static_cast<int64_t>(il) * n_ctx * d_model));

  const size_t v_elem = ggml_element_size(kv_cache.self_v);
  ggml_tensor *V = ggml_view_3d(
      ctx, kv_cache.self_v, head_dim, n_kv, n_heads, v_elem * d_model,
      v_elem * head_dim,
      v_elem * static_cast<size_t>(static_cast<int64_t>(il) * n_ctx * d_model));

  ggml_tensor *Q = pad_head_dim(ctx, Q_unpad, pad);
  ggml_tensor *K_p = pad_head_dim(ctx, ggml_cont(ctx, K), pad);
  ggml_tensor *V_p = pad_head_dim(ctx, ggml_cont(ctx, V), pad);

  ggml_tensor *o;
  if (use_flash) {
    o = ggml_flash_attn_ext(ctx, Q, K_p, V_p, mask, scale, 0.0f, 0.0f);
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont(ctx, o);
  } else {
    ggml_tensor *kq = ggml_mul_mat(ctx, K_p, Q);
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, V_p, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
  }

  if (pad > 0) {
    o = unpad_head_dim(ctx, o, head_dim, pad);
    o = ggml_cont(ctx, o);
  }

  o = ggml_permute(ctx, o, 0, 2, 1, 3);
  o = ggml_cont(ctx, o);
  o = ggml_reshape_2d(ctx, o, d_model, n_tokens);

  o = ggml_mul_mat(ctx, out_w, o);
  return o;
}

ggml_tensor *mha_cross_cached(ggml_context *ctx, ggml_tensor *x,
                              MoonshineStreamingKvCache &kv_cache,
                              ggml_tensor *q_w, ggml_tensor *out_w,
                              const MoonshineStreamingHParams &hp, int n_heads,
                              int d_model, int il, int T_enc, bool use_flash) {
  const int head_dim = d_model / n_heads;
  const int head_dim_pad = hp.dec_head_dim_padded();
  const int pad = head_dim_pad - head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t n_tokens = x->ne[1];

  ggml_tensor *Qcur = ggml_mul_mat(ctx, q_w, x);
  ggml_tensor *Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
  Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
  Q = ggml_cont(ctx, Q);
  Q = pad_head_dim(ctx, Q, pad);

  const size_t k_elem = ggml_element_size(kv_cache.cross_k);
  ggml_tensor *K = ggml_view_3d(
      ctx, kv_cache.cross_k, head_dim, T_enc, n_heads, k_elem * d_model,
      k_elem * head_dim,
      k_elem * static_cast<size_t>(static_cast<int64_t>(il) * T_enc * d_model));

  const size_t v_elem = ggml_element_size(kv_cache.cross_v);
  ggml_tensor *V = ggml_view_3d(
      ctx, kv_cache.cross_v, head_dim, T_enc, n_heads, v_elem * d_model,
      v_elem * head_dim,
      v_elem * static_cast<size_t>(static_cast<int64_t>(il) * T_enc * d_model));

  ggml_tensor *K_p = pad_head_dim(ctx, ggml_cont(ctx, K), pad);
  ggml_tensor *V_p = pad_head_dim(ctx, ggml_cont(ctx, V), pad);

  ggml_tensor *o;
  if (use_flash) {
    o = ggml_flash_attn_ext(ctx, Q, K_p, V_p, /*mask=*/nullptr, scale, 0.0f,
                            0.0f);
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont(ctx, o);
  } else {
    ggml_tensor *kq = ggml_mul_mat(ctx, K_p, Q);
    ggml_tensor *kq_soft =
        ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, V_p, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
  }

  if (pad > 0) {
    o = unpad_head_dim(ctx, o, head_dim, pad);
    o = ggml_cont(ctx, o);
  }
  o = ggml_permute(ctx, o, 0, 2, 1, 3);
  o = ggml_cont(ctx, o);
  o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
  o = ggml_mul_mat(ctx, out_w, o);
  return o;
}

} // namespace

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

ggml_tensor *find_tensor_by_name(ggml_context *gctx, const char *name) {
  for (ggml_tensor *t = ggml_get_first_tensor(gctx); t != nullptr;
       t = ggml_get_next_tensor(gctx, t)) {
    if (std::strcmp(t->name, name) == 0) {
      return t;
    }
  }
  return nullptr;
}

void MoonshineStreamingKvCache::free() {
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
  cross_k = nullptr;
  cross_v = nullptr;
  n = 0;
  head = 0;
  T_enc = 0;
  cross_populated = false;
}

bool kv_cache_init(MoonshineStreamingKvCache &cache, ggml_backend_t backend,
                   int n_ctx, int T_enc, int d_model, int n_layer,
                   ggml_type kv_type) {
  if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
    return false;
  }
  if (backend == nullptr) {
    return false;
  }

  const size_t ctx_size = 4 * ggml_tensor_overhead() + 256;
  ggml_init_params params{ctx_size, nullptr, /*no_alloc=*/true};
  cache.ctx = ggml_init(params);
  if (cache.ctx == nullptr) {
    return false;
  }

  const int64_t self_elements = static_cast<int64_t>(d_model) * n_layer * n_ctx;
  const int64_t cross_elements =
      static_cast<int64_t>(d_model) * n_layer * T_enc;

  cache.self_k = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
  cache.self_v = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
  cache.cross_k = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
  cache.cross_v = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);

  ggml_set_name(cache.self_k, "kv_self_k");
  ggml_set_name(cache.self_v, "kv_self_v");
  ggml_set_name(cache.cross_k, "kv_cross_k");
  ggml_set_name(cache.cross_v, "kv_cross_v");

  cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
  if (cache.buffer == nullptr) {
    ggml_free(cache.ctx);
    cache.ctx = nullptr;
    return false;
  }
  ggml_backend_buffer_clear(cache.buffer, 0);

  cache.n_ctx = n_ctx;
  cache.T_enc = T_enc;
  cache.n = 0;
  cache.head = 0;
  cache.cross_populated = false;

  return true;
}

int encoder_t_enc(const MoonshineStreamingHParams &hp, int n_samples) {
  if (n_samples <= 0 || hp.enc_frame_len <= 0) {
    return 0;
  }
  const int T_frames = n_samples / hp.enc_frame_len;
  if (T_frames <= 0) {
    return 0;
  }
  // Two strided causal conv layers, stride 2 each.
  const int T1 = causal_conv1d_t_out(T_frames, /*k=*/5, /*s=*/2);
  return causal_conv1d_t_out(T1, /*k=*/5, /*s=*/2);
}

void build_sliding_window_mask(int T_enc, int left_window, int right_window,
                               float *out_mask) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  // ggml mask layout: ne0 = n_kv, ne1 = n_q. Row-major memory has q outer,
  // k inner.
  for (int q = 0; q < T_enc; ++q) {
    for (int k = 0; k < T_enc; ++k) {
      const int dist = q - k;
      const bool left_ok = (dist >= 0) && (dist < left_window);
      const bool right_ok = (dist < 0) && (-dist < right_window);
      out_mask[static_cast<size_t>(q) * T_enc + k] =
          (left_ok || right_ok) ? 0.0f : kNegInf;
    }
  }
}

// ---------------------------------------------------------------------------
// Encoder graph
// ---------------------------------------------------------------------------

EncoderBuild build_encoder_graph(ggml_context *ctx,
                                 const MoonshineStreamingWeights &w,
                                 const MoonshineStreamingHParams &hp,
                                 int n_samples, bool use_flash) {
  EncoderBuild eb{};

  if (ctx == nullptr || n_samples <= 0) {
    return eb;
  }

  const int frame_len = hp.enc_frame_len;
  const int hidden = hp.enc_d_model;
  const int hidden2 = 2 * hidden;
  const int n_heads = hp.enc_n_heads;
  const int T_enc = encoder_t_enc(hp, n_samples);

  if (T_enc <= 0) {
    return eb;
  }
  // The caller must right-pad so the frame reshape is well-defined.
  if (frame_len <= 0 || n_samples % frame_len != 0) {
    return eb;
  }
  const int T_frames = n_samples / frame_len;

  eb.audio_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_samples);
  if (eb.audio_in == nullptr) {
    return eb;
  }
  named(eb.audio_in, "enc.audio.in");
  ggml_set_input(eb.audio_in);

  // Frame reshape: [n_samples] -> [frame_len, T_frames].
  ggml_tensor *x = ggml_reshape_2d(ctx, eb.audio_in, frame_len, T_frames);

  // CMVN: layer-norm-style, no affine.
  x = ggml_norm(ctx, x, hp.cmvn_eps);

  // asinh compression.
  ggml_tensor *exp_logk = ggml_exp(ctx, w.embedder.comp_log_k); // [1]
  ggml_tensor *z = ggml_mul(ctx, x, exp_logk); // broadcast [1] over frames
  x = asinh_op(ctx, z);

  // Linear (no bias) + SiLU.
  x = ggml_mul_mat(ctx, w.embedder.linear_w, x); // [hidden, T_frames]
  x = ggml_silu(ctx, x);

  // conv1: causal, stride 2, hidden -> 2*hidden, +bias, +SiLU.
  {
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T_frames, hidden]
    x = ggml_reshape_3d(ctx, x, T_frames, hidden, 1);
    x = ggml_pad_ext(ctx, x, /*lp0=*/4, /*rp0=*/0, /*lp1=*/0, /*rp1=*/0,
                     /*lp2=*/0, /*rp2=*/0, /*lp3=*/0, /*rp3=*/0);
    x = conv_1d_f32(ctx, w.embedder.conv1_w, x, /*stride=*/2, /*padding=*/0,
                    /*dilation=*/1);
    x = add_conv1d_bias(ctx, x, w.embedder.conv1_b);
    x = ggml_silu(ctx, x);
    const int64_t T1 = x->ne[0];
    x = ggml_reshape_2d(ctx, x, T1, hidden2);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [hidden2, T1]
  }

  // conv2: causal, stride 2, 2*hidden -> hidden, +bias, NO SiLU.
  {
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T1, hidden2]
    const int64_t T1 = x->ne[0];
    x = ggml_reshape_3d(ctx, x, T1, hidden2, 1);
    x = ggml_pad_ext(ctx, x, /*lp0=*/4, /*rp0=*/0, /*lp1=*/0, /*rp1=*/0,
                     /*lp2=*/0, /*rp2=*/0, /*lp3=*/0, /*rp3=*/0);
    x = conv_1d_f32(ctx, w.embedder.conv2_w, x, /*stride=*/2, /*padding=*/0,
                    /*dilation=*/1);
    x = add_conv1d_bias(ctx, x, w.embedder.conv2_b);
    const int64_t T_enc_actual = x->ne[0];
    x = ggml_reshape_2d(ctx, x, T_enc_actual, hidden);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [hidden, T_enc]
  }

  // Per-layer sliding-window masks (graph inputs).
  eb.per_layer_masks.assign(static_cast<size_t>(hp.enc_n_layers), nullptr);
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_enc, T_enc);
    char mname[64];
    std::snprintf(mname, sizeof(mname), "enc.swa_mask.%d", i);
    named(mask, mname);
    ggml_set_input(mask);
    eb.per_layer_masks[static_cast<size_t>(i)] = mask;
  }

  // Transformer blocks.
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    x = build_encoder_block(ctx, x, eb.per_layer_masks[static_cast<size_t>(i)],
                            w.enc_blocks[static_cast<size_t>(i)], hp, n_heads,
                            hp.enc_head_dim, use_flash);
  }

  // Final LN (no bias; the scale already includes the +1.0 unit offset).
  x = layer_norm(ctx, x, w.enc_top.final_norm_w, /*beta=*/nullptr);
  named(x, "enc.final");

  eb.out = x;
  eb.T_enc = T_enc;
  ggml_set_output(eb.out);

  eb.graph = ggml_new_graph_custom(ctx, 8192, false);
  if (eb.graph == nullptr) {
    return eb;
  }
  ggml_build_forward_expand(eb.graph, eb.out);

  return eb;
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

AdapterBuild build_adapter_graph(ggml_context *ctx,
                                 const MoonshineStreamingWeights &w,
                                 const MoonshineStreamingHParams &hp, int T) {
  AdapterBuild ab{};

  if (ctx == nullptr || T <= 0) {
    return ab;
  }

  const int enc_h = hp.enc_d_model;
  const int dec_h = hp.dec_d_model;

  ab.encoder_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc_h, T);
  named(ab.encoder_out_in, "adapter.encoder_in");
  ggml_set_input(ab.encoder_out_in);

  ab.pos_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
  named(ab.pos_ids_in, "adapter.pos_ids");
  ggml_set_input(ab.pos_ids_in);

  // Absolute-frame pos_emb slice: get_rows yields ne = [enc_h, T].
  ggml_tensor *pos_emb =
      ggml_get_rows(ctx, w.adapter.pos_emb_w, ab.pos_ids_in);
  named(pos_emb, "adapter.pos_emb");

  ggml_tensor *y = ggml_add(ctx, ab.encoder_out_in, pos_emb);
  if (hp.adapter_has_proj) {
    // proj_w ne = [enc_h, dec_h]; result ne = [dec_h, T].
    y = ggml_mul_mat(ctx, w.adapter.proj_w, y);
  } else if (enc_h != dec_h) {
    // Width mismatch with no projection is a malformed model.
    return AdapterBuild{};
  }
  named(y, "adapter.out");
  ab.out = y;
  ggml_set_output(ab.out);

  ab.graph = ggml_new_graph_custom(ctx, 4096, false);
  if (ab.graph == nullptr) {
    return ab;
  }
  ggml_build_forward_expand(ab.graph, ab.out);

  return ab;
}

// ---------------------------------------------------------------------------
// Cross-KV projection / commit
// ---------------------------------------------------------------------------

CrossKVProjectionBuild
build_cross_kv_projection_graph(ggml_context *ctx,
                                const MoonshineStreamingWeights &w,
                                const MoonshineStreamingHParams &hp,
                                int n_frames) {
  CrossKVProjectionBuild pb{};

  if (ctx == nullptr || n_frames <= 0) {
    return pb;
  }

  const int d_model = hp.dec_d_model;

  pb.encoder_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_frames);
  named(pb.encoder_out_in, "dec.encoder_in_adapted.proj");
  ggml_set_input(pb.encoder_out_in);

  pb.graph = ggml_new_graph_custom(ctx, 4096, false);
  if (pb.graph == nullptr) {
    return pb;
  }

  const int n_layers = static_cast<int>(w.dec_blocks.size());
  pb.per_layer_k.reserve(static_cast<size_t>(n_layers));
  pb.per_layer_v.reserve(static_cast<size_t>(n_layers));
  for (int il = 0; il < n_layers; ++il) {
    const auto &blk = w.dec_blocks[static_cast<size_t>(il)];

    ggml_tensor *Kcross = ggml_mul_mat(ctx, blk.cross_k_w, pb.encoder_out_in);
    ggml_tensor *Vcross = ggml_mul_mat(ctx, blk.cross_v_w, pb.encoder_out_in);

    char kname[64], vname[64];
    std::snprintf(kname, sizeof(kname), "xkv.proj.k.%d", il);
    std::snprintf(vname, sizeof(vname), "xkv.proj.v.%d", il);
    named(Kcross, kname);
    named(Vcross, vname);
    ggml_set_output(Kcross);
    ggml_set_output(Vcross);

    pb.per_layer_k.push_back(Kcross);
    pb.per_layer_v.push_back(Vcross);

    ggml_build_forward_expand(pb.graph, Kcross);
    ggml_build_forward_expand(pb.graph, Vcross);
  }

  return pb;
}

CrossKVCommitBuild
build_cross_kv_commit_graph(ggml_context *ctx,
                            const MoonshineStreamingHParams &hp,
                            MoonshineStreamingKvCache &kv_cache, int T_enc) {
  CrossKVCommitBuild cb{};

  if (ctx == nullptr || T_enc <= 0 || kv_cache.cross_k == nullptr ||
      kv_cache.cross_v == nullptr) {
    return cb;
  }

  const int d_model = hp.dec_d_model;
  cb.graph = ggml_new_graph_custom(ctx, 4096, false);
  if (cb.graph == nullptr) {
    return cb;
  }

  const int n_layers = static_cast<int>(hp.dec_n_layers);
  cb.per_layer_k_in.reserve(static_cast<size_t>(n_layers));
  cb.per_layer_v_in.reserve(static_cast<size_t>(n_layers));

  const size_t k_elem = ggml_element_size(kv_cache.cross_k);
  const size_t v_elem = ggml_element_size(kv_cache.cross_v);

  for (int il = 0; il < n_layers; ++il) {
    ggml_tensor *K_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_enc);
    ggml_tensor *V_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_enc);
    char kname[64], vname[64];
    std::snprintf(kname, sizeof(kname), "xkv.commit.k_in.%d", il);
    std::snprintf(vname, sizeof(vname), "xkv.commit.v_in.%d", il);
    named(K_in, kname);
    named(V_in, vname);
    ggml_set_input(K_in);
    ggml_set_input(V_in);
    cb.per_layer_k_in.push_back(K_in);
    cb.per_layer_v_in.push_back(V_in);

    ggml_tensor *k_dst = ggml_view_1d(
        ctx, kv_cache.cross_k, static_cast<int64_t>(T_enc) * d_model,
        k_elem *
            static_cast<size_t>(static_cast<int64_t>(il) * T_enc * d_model));
    ggml_tensor *v_dst = ggml_view_1d(
        ctx, kv_cache.cross_v, static_cast<int64_t>(T_enc) * d_model,
        v_elem *
            static_cast<size_t>(static_cast<int64_t>(il) * T_enc * d_model));

    ggml_build_forward_expand(cb.graph, ggml_cpy(ctx, K_in, k_dst));
    ggml_build_forward_expand(cb.graph, ggml_cpy(ctx, V_in, v_dst));
  }

  return cb;
}

// ---------------------------------------------------------------------------
// Decoder graph (KV-cached prompt + step)
// ---------------------------------------------------------------------------

DecoderBuild build_decoder_graph_kv(ggml_context *ctx,
                                    const MoonshineStreamingWeights &w,
                                    const MoonshineStreamingHParams &hp,
                                    MoonshineStreamingKvCache &kv_cache,
                                    int n_tokens, int n_past, int T_enc,
                                    bool skip_log_softmax, bool use_flash) {
  DecoderBuild db{};

  if (ctx == nullptr || n_tokens <= 0 || T_enc <= 0) {
    return db;
  }
  const int n_kv = n_past + n_tokens;
  if (n_kv > kv_cache.n_ctx) {
    return db;
  }

  const int d_model = hp.dec_d_model;
  const int n_heads = hp.dec_n_heads;

  db.token_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  named(db.token_ids_in, "dec.token_ids");
  ggml_set_input(db.token_ids_in);

  db.pos_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  named(db.pos_ids_in, "dec.pos_ids");
  ggml_set_input(db.pos_ids_in);

  ggml_tensor *causal_mask = nullptr;
  const bool need_mask = (n_tokens > 1);
  if (need_mask) {
    db.causal_mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    named(db.causal_mask_in, "dec.causal_mask");
    ggml_set_input(db.causal_mask_in);
    causal_mask = ggml_cast(ctx, db.causal_mask_in, GGML_TYPE_F16);
  }

  ggml_tensor *tok_emb =
      ggml_get_rows(ctx, w.dec_top.token_embd_w, db.token_ids_in);
  named(tok_emb, "dec.token_emb");

  ggml_tensor *x = ggml_scale(ctx, tok_emb, 1.0f);
  named(x, "dec.embed_sum");

  db.graph = ggml_new_graph_custom(ctx, 8192, false);
  if (db.graph == nullptr) {
    return db;
  }

  const int n_blocks = static_cast<int>(w.dec_blocks.size());
  for (int i = 0; i < n_blocks; ++i) {
    const auto &b = w.dec_blocks[static_cast<size_t>(i)];

    {
      ggml_tensor *y = layer_norm(ctx, x, b.norm_self_w, /*beta=*/nullptr);
      y = mha_self_cached(ctx, db.graph, y, kv_cache, db.pos_ids_in,
                          causal_mask, b.self_q_w, b.self_k_w, b.self_v_w,
                          b.self_out_w, hp, n_heads, d_model, i, n_past,
                          n_tokens, n_kv, use_flash);
      x = ggml_add(ctx, x, y);
    }
    {
      ggml_tensor *y = layer_norm(ctx, x, b.norm_cross_w, /*beta=*/nullptr);
      y = mha_cross_cached(ctx, y, kv_cache, b.cross_q_w, b.cross_out_w, hp,
                           n_heads, d_model, i, T_enc, use_flash);
      x = ggml_add(ctx, x, y);
    }
    {
      ggml_tensor *y = layer_norm(ctx, x, b.norm_ffn_w, /*beta=*/nullptr);
      y = ffn_decoder_swiglu(ctx, y, b.ffn_fc1_w, b.ffn_fc1_b, b.ffn_fc2_w,
                             b.ffn_fc2_b, hp.dec_ffn_dim);
      x = ggml_add(ctx, x, y);
    }
  }

  x = layer_norm(ctx, x, w.dec_top.final_norm_w, /*beta=*/nullptr);
  named(x, "dec.out_before_head");

  // Untied lm_head: project against dec.lm_head.weight, NOT the token embed.
  ggml_tensor *logits_raw = ggml_mul_mat(ctx, w.dec_top.lm_head_w, x);
  named(logits_raw, "dec.logits_raw");

  ggml_tensor *logits = logits_raw;
  if (!skip_log_softmax) {
    logits = ggml_log(ctx, ggml_soft_max(ctx, logits_raw));
    named(logits, "dec.logits");
  }

  db.out = logits;
  ggml_set_output(db.out);

  // Backend argmax over the last position when skipping log_softmax - every
  // step pass then downloads a single int32 instead of full vocab logits.
  if (skip_log_softmax) {
    ggml_tensor *last_logits = logits;
    if (n_tokens > 1) {
      const int64_t vocab = logits->ne[0];
      const size_t row_bytes =
          ggml_element_size(logits) * static_cast<size_t>(vocab);
      last_logits = ggml_view_2d(ctx, logits, vocab, /*n=*/1, row_bytes,
                                 row_bytes * static_cast<size_t>(n_tokens - 1));
      last_logits = ggml_cont(ctx, last_logits);
    }
    db.argmax_out = ggml_argmax(ctx, last_logits);
    ggml_set_name(db.argmax_out, "dec.argmax");
    ggml_set_output(db.argmax_out);
    ggml_build_forward_expand(db.graph, db.argmax_out);
  } else {
    ggml_build_forward_expand(db.graph, db.out);
  }

  return db;
}

} // namespace engine::models::moonshine_streaming
