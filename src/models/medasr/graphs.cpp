// engine/models/medasr/graphs.cpp - ggml encoder graph for the native engine
// MedASR package (see graphs_internal.h).
//
// Ported from src/runtime/arch/medasr/encoder.cpp with identical op order.
// The four src/runtime/conformer helpers the arch reached for are ported here
// on exactly the paths MedASR takes (so the package has no dependency on
// src/runtime/):
//   - conv_1d_f32         (subsampling convs; im2col pointwise fallback)
//   - conv_module         (BatchNorm variant, direct 2-D depthwise, explicit
//                          zero-concat for the asymmetric (15, 16) pad, no conv
//                          biases, optional post-GLU valid-frame mask)
//   - fused_batch_norm
//   - detect_direct_pw / resolve_conv_direct (same env overrides)
// The framework's conformer / attention modules are not reused: their LN eps,
// residual scalars and RoPE placement differ from LASR's.

#include "engine/models/medasr/graphs_internal.h"

#include "engine/models/medasr/assets.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace engine::models::medasr {

namespace {

// transcribe::env::flag: set, non-empty, and not starting with '0'.
bool env_flag(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool resolve_conv_direct(const char *direct_env, const char *no_direct_env,
                         bool backend_default) {
  if (env_flag(direct_env)) {
    return true;
  }
  if (env_flag(no_direct_env)) {
    return false;
  }
  return backend_default;
}

// Weight matmul forcing F32 accumulation for F16 weights (the arch's CUDA
// COMPUTE_16F guard: MedASR's scaled residual stream reaches ~2e6 between
// blocks and would saturate an F16 accumulator). No-op for other types and on
// CPU / Metal.
ggml_tensor *mul_mat_f32acc(ggml_context *ctx, ggml_tensor *w, ggml_tensor *x) {
  ggml_tensor *y = ggml_mul_mat(ctx, w, x);
  if (w->type == GGML_TYPE_F16) {
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
  }
  return y;
}

// LASR LayerNorm: affine scale, no bias, eps from the GGUF.
ggml_tensor *lasr_layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale,
                             float eps) {
  x = ggml_norm(ctx, x, eps);
  if (scale != nullptr) {
    x = ggml_mul(ctx, x, scale);
  }
  return x;
}

// y = down(SiLU(up(x))), no biases.
ggml_tensor *lasr_feed_forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *up_w,
                               ggml_tensor *down_w) {
  x = mul_mat_f32acc(ctx, up_w, x);
  x = ggml_silu(ctx, x);
  x = mul_mat_f32acc(ctx, down_w, x);
  return x;
}

// out = w0 * residual + w1 * branch (a 1.0 weight skips its scale op).
ggml_tensor *scaled_residual(ggml_context *ctx, ggml_tensor *residual, ggml_tensor *branch,
                             float w0, float w1) {
  ggml_tensor *a = (w0 == 1.0f) ? residual : ggml_scale(ctx, residual, w0);
  ggml_tensor *b = (w1 == 1.0f) ? branch : ggml_scale(ctx, branch, w1);
  return ggml_add(ctx, a, b);
}

// conformer::conv_1d_f32: Conv1D whose im2col keeps the kernel's own type.
// kernel ne [K, IC, OC], data ne [W, IC, N] -> [OW, OC, N].
ggml_tensor *conv_1d_f32(ggml_context *ctx, ggml_tensor *kernel, ggml_tensor *data,
                         int stride, int padding, int dilation) {
  ggml_tensor *im2col = ggml_im2col(ctx, kernel, data, stride, /*s1=*/0, padding, /*p1=*/0,
                                    dilation, /*d1=*/0, /*is_2D=*/false,
                                    /*dst_type=*/kernel->type);
  const int64_t N = im2col->ne[2];
  ggml_tensor *kernel_2d =
      ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
  const bool kernel_needs_f32_acc = (kernel->type == GGML_TYPE_F16);

  if (N == 1) {
    ggml_tensor *result = ggml_mul_mat(
        ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1]), kernel_2d); // [OW, OC]
    if (kernel_needs_f32_acc) {
      ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    }
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], 1);
  }

  // Batched: mul_mat the 3-D im2col directly (kernel broadcasts over the
  // batch) -> [OC, OW, N], then permute to [OW, OC, N].
  ggml_tensor *result = ggml_mul_mat(ctx, kernel_2d, im2col);
  if (kernel_needs_f32_acc) {
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
  }
  return ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
}

// ggml_conv_2d_dw_direct needs an F32 kernel (hard assert on CUDA).
ggml_tensor *dw_kernel_for_direct(ggml_context *ctx, ggml_tensor *kernel) {
  if (kernel == nullptr || kernel->type == GGML_TYPE_F32) {
    return kernel;
  }
  return ggml_cast(ctx, kernel, GGML_TYPE_F32);
}

// conformer::fused_batch_norm: y = x * scale + bias over [T, C, B].
ggml_tensor *fused_batch_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale_1d,
                              ggml_tensor *bias_1d) {
  const int64_t C = scale_1d->ne[0];
  ggml_tensor *scale_4d = ggml_reshape_4d(ctx, scale_1d, 1, C, 1, 1);
  ggml_tensor *bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, C, 1, 1);
  ggml_tensor *y = ggml_mul(ctx, x, scale_4d);
  return ggml_add(ctx, y, bias_4d);
}

// ---------------------------------------------------------------------------
// Subsampling: dense_0 -> ReLU -> conv_0 (s=2) -> ReLU -> conv_1 (s=2) -> ReLU
// -> dense_1 (no ReLU). Channel-fast for the denses, time-fast for the convs.
// ---------------------------------------------------------------------------
ggml_tensor *build_subsampling(ggml_context *ctx, const MedAsrSubsamplingWeights &ss,
                               ggml_tensor *mel_in, const MedAsrHParams &hp) {
  const int sub_k = hp.enc_sub_kernel;

  // [n_mels, T_mel, 1, B] -> [n_mels, T_mel, B]
  ggml_tensor *x = ggml_reshape_3d(ctx, mel_in, mel_in->ne[0], mel_in->ne[1], mel_in->ne[3]);

  x = mul_mat_f32acc(ctx, ss.dense0_w, x);
  if (ss.dense0_b != nullptr) {
    x = ggml_add(ctx, x, ss.dense0_b);
  }
  x = ggml_relu(ctx, x); // [d, T_mel, B]

  x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [T_mel, d, B]

  x = conv_1d_f32(ctx, ss.conv0_w, x, /*stride=*/sub_k > 0 ? hp.enc_sub_stride : 2,
                  /*padding=*/0, /*dilation=*/1);
  if (ss.conv0_b != nullptr) {
    ggml_tensor *bias_r = ggml_reshape_3d(ctx, ss.conv0_b, 1, x->ne[1], 1);
    x = ggml_add(ctx, x, bias_r);
  }
  x = ggml_relu(ctx, x); // [T1, d, B]

  x = conv_1d_f32(ctx, ss.conv1_w, x, /*stride=*/hp.enc_sub_stride, /*padding=*/0,
                  /*dilation=*/1);
  if (ss.conv1_b != nullptr) {
    ggml_tensor *bias_r = ggml_reshape_3d(ctx, ss.conv1_b, 1, x->ne[1], 1);
    x = ggml_add(ctx, x, bias_r);
  }
  x = ggml_relu(ctx, x); // [T_enc, sub_channels, B]

  x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [sub_channels, T_enc, B]

  // The arch casts a quantized dense_1 weight to F32 in-graph: a Q4_K MMQ
  // kernel on CUDA overflows for this 256-wide input at these activation
  // magnitudes. F32 / F16 weights pass through.
  ggml_tensor *dense1_w = ss.dense1_w;
  if (dense1_w->type != GGML_TYPE_F32 && dense1_w->type != GGML_TYPE_F16) {
    dense1_w = ggml_cast(ctx, dense1_w, GGML_TYPE_F32);
  }
  x = mul_mat_f32acc(ctx, dense1_w, x);
  if (ss.dense1_b != nullptr) {
    x = ggml_add(ctx, x, ss.dense1_b);
  }
  return x; // [d, T_enc, B]
}

// ---------------------------------------------------------------------------
// Conv module: pw1 -> GLU -> [valid-frame mask] -> depthwise (PyTorch
// padding="same": (k-1)/2 left, the rest right; (15, 16) for k = 32) ->
// fused BN -> SiLU -> pw2. Operates on the post-LayerNorm activation.
// ---------------------------------------------------------------------------
ggml_tensor *build_conv_module(ggml_context *ctx, ggml_tensor *x, const MedAsrBlockWeights &b,
                               int conv_kernel, const MedAsrGraphPolicy &policy,
                               ggml_tensor *conv_pad_mask) {
  const int pad_left = (conv_kernel - 1) / 2;
  const int pad_right = conv_kernel - 1 - pad_left;
  const int64_t d_model = x->ne[0];
  const int64_t B = x->ne[2];

  if (policy.direct_pw) {
    ggml_tensor *pw1 = ggml_reshape_2d(ctx, b.conv_pw1_w, d_model, 2 * d_model);
    x = ggml_mul_mat(ctx, pw1, x); // [2d, T, B]
    if (b.conv_pw1_w->type == GGML_TYPE_F16) {
      ggml_mul_mat_set_prec(x, GGML_PREC_F32);
    }
    // GLU over ne[0]: first half * sigmoid(second half), per utterance.
    const int64_t T = x->ne[1];
    const int64_t half = x->ne[0] / 2;
    ggml_tensor *gate = ggml_view_3d(ctx, x, half, T, B, x->nb[1], x->nb[2], /*offset=*/0);
    ggml_tensor *value =
        ggml_view_3d(ctx, x, half, T, B, x->nb[1], x->nb[2], half * ggml_element_size(x));
    x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value)); // [d, T, B]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [T, d, B]
  } else {
    // im2col pointwise in [T, C] layout (the arch's Vulkan default).
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    x = conv_1d_f32(ctx, b.conv_pw1_w, x, /*s=*/1, /*p=*/0, /*d=*/1); // [T, 2d, B]
    const int64_t half = x->ne[1] / 2;
    // The arch viewed this as [T, half, 1, 1] - single-utterance only. The
    // batch-aware view below is the same memory for B == 1 and keeps every
    // utterance for B > 1.
    ggml_tensor *gate =
        ggml_view_3d(ctx, x, x->ne[0], half, x->ne[2], x->nb[1], x->nb[2], /*offset=*/0);
    ggml_tensor *value =
        ggml_view_3d(ctx, x, x->ne[0], half, x->ne[2], x->nb[1], x->nb[2], x->nb[1] * half);
    x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
    x = ggml_cont(ctx, x); // [T, d, B]
  }

  if (conv_pad_mask != nullptr) {
    x = ggml_mul(ctx, x, conv_pad_mask); // [T, 1, B, 1] broadcast over d
  }

  const bool symmetric_pad = (pad_left == pad_right);
  if (!symmetric_pad) {
    if (pad_left > 0) {
      ggml_tensor *pad_l = ggml_new_tensor_4d(ctx, x->type, pad_left, x->ne[1], x->ne[2], x->ne[3]);
      pad_l = ggml_fill(ctx, pad_l, 0.0f);
      x = ggml_concat(ctx, pad_l, x, /*dim=*/0);
    }
    if (pad_right > 0) {
      ggml_tensor *pad_r = ggml_new_tensor_4d(ctx, x->type, pad_right, x->ne[1], x->ne[2], x->ne[3]);
      pad_r = ggml_fill(ctx, pad_r, 0.0f);
      x = ggml_concat(ctx, x, pad_r, /*dim=*/0);
    }
  }
  const int padding_op = symmetric_pad ? pad_left : 0;

  // Direct depthwise (the arch pins direct_dw_in_block = true: the im2col
  // depthwise is not batch-stable for LASR's asymmetric pad).
  {
    ggml_tensor *knl =
        ggml_reshape_4d(ctx, dw_kernel_for_direct(ctx, b.conv_dw_w), conv_kernel, 1, 1, d_model);
    ggml_tensor *data = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], B);
    x = ggml_conv_2d_dw_direct(ctx, knl, data, /*s0=*/1, /*s1=*/1, /*p0=*/padding_op, /*p1=*/0,
                               /*d0=*/1, /*d1=*/1);
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[2], x->ne[3]); // [T, d, B]
  }

  x = fused_batch_norm(ctx, x, b.conv_bn_scale, b.conv_bn_bias);
  x = ggml_silu(ctx, x);

  if (policy.direct_pw) {
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [d, T, B]
    ggml_tensor *pw2 = ggml_reshape_2d(ctx, b.conv_pw2_w, d_model, d_model);
    x = ggml_mul_mat(ctx, pw2, x);
    if (b.conv_pw2_w->type == GGML_TYPE_F16) {
      ggml_mul_mat_set_prec(x, GGML_PREC_F32);
    }
  } else {
    x = conv_1d_f32(ctx, b.conv_pw2_w, x, /*s=*/1, /*p=*/0, /*d=*/1);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
  }
  return x; // [d, T, B]
}

// ---------------------------------------------------------------------------
// RoPE self-attention (LasrEncoderAttention): no-bias Q/K/V, NEOX RoPE on Q/K
// AFTER projection, SDPA scaled by 1/sqrt(head_dim), no-bias output.
// ---------------------------------------------------------------------------
ggml_tensor *build_rope_attn(ggml_context *ctx, ggml_tensor *x, ggml_tensor *positions,
                             const MedAsrBlockWeights &b, int d_model, int n_head,
                             float rope_theta, int rope_max_pos, bool use_flash,
                             ggml_tensor *attn_pad_mask) {
  const int head_dim = d_model / n_head;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t T = x->ne[1];
  const int64_t Bb = x->ne[2];

  // A content-dependent key-padding mask forces the manual path.
  const bool flash = use_flash && (attn_pad_mask == nullptr);

  ggml_tensor *q = mul_mat_f32acc(ctx, b.attn_q_w, x);
  ggml_tensor *k = mul_mat_f32acc(ctx, b.attn_k_w, x);
  ggml_tensor *v = mul_mat_f32acc(ctx, b.attn_v_w, x);

  q = ggml_reshape_4d(ctx, q, head_dim, n_head, T, Bb);
  k = ggml_reshape_4d(ctx, k, head_dim, n_head, T, Bb);

  q = ggml_rope_ext(ctx, q, positions, /*c=*/nullptr, /*n_dims=*/head_dim, GGML_ROPE_TYPE_NEOX,
                    /*n_ctx_orig=*/rope_max_pos, rope_theta, /*freq_scale=*/1.0f,
                    /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                    /*beta_slow=*/1.0f);
  k = ggml_rope_ext(ctx, k, positions, /*c=*/nullptr, /*n_dims=*/head_dim, GGML_ROPE_TYPE_NEOX,
                    /*n_ctx_orig=*/rope_max_pos, rope_theta, /*freq_scale=*/1.0f,
                    /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                    /*beta_slow=*/1.0f);

  v = ggml_reshape_4d(ctx, v, head_dim, n_head, T, Bb);

  // [head_dim, T, n_head, B]
  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
  k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
  v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

  ggml_tensor *o = nullptr;
  if (flash) {
    o = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, scale, 0.0f, 0.0f);
    // [head_dim, n_head, T, B] -> [head_dim, T, n_head, B]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
  } else {
    ggml_tensor *kq = ggml_mul_mat(ctx, k, q); // [T_k, T_q, n_head, B]
    if (attn_pad_mask != nullptr) {
      kq = ggml_add(ctx, kq, attn_pad_mask);
    }
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
  }
  // [head_dim, T, n_head, B] -> [head_dim, n_head, T, B] -> [d, T, B]
  o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
  o = ggml_reshape_3d(ctx, o, d_model, T, Bb);

  return mul_mat_f32acc(ctx, b.attn_o_w, o);
}

ggml_tensor *build_block(ggml_context *ctx, ggml_tensor *x, ggml_tensor *positions,
                         const MedAsrBlockWeights &b, const MedAsrHParams &hp,
                         const MedAsrGraphPolicy &policy, bool use_flash,
                         ggml_tensor *attn_pad_mask, ggml_tensor *conv_pad_mask) {
  const float ln_eps = hp.enc_layer_norm_eps;

  { // Macaron FF1.
    ggml_tensor *y = lasr_layer_norm(ctx, x, b.norm_ff1_w, ln_eps);
    y = lasr_feed_forward(ctx, y, b.ff1_up_w, b.ff1_down_w);
    x = scaled_residual(ctx, x, y, hp.enc_ff_resid_w0, hp.enc_ff_resid_w1);
  }
  { // Self-attention.
    ggml_tensor *y = lasr_layer_norm(ctx, x, b.norm_attn_w, ln_eps);
    y = build_rope_attn(ctx, y, positions, b, hp.enc_hidden, hp.enc_n_heads, hp.enc_rope_theta,
                        hp.enc_max_pos_emb, use_flash, attn_pad_mask);
    x = ggml_add(ctx, x, y);
  }
  { // Conv module.
    ggml_tensor *y = lasr_layer_norm(ctx, x, b.norm_conv_w, ln_eps);
    y = build_conv_module(ctx, y, b, hp.enc_conv_kernel, policy, conv_pad_mask);
    x = scaled_residual(ctx, x, y, hp.enc_conv_resid_w0, hp.enc_conv_resid_w1);
  }
  { // Macaron FF2.
    ggml_tensor *y = lasr_layer_norm(ctx, x, b.norm_ff2_w, ln_eps);
    y = lasr_feed_forward(ctx, y, b.ff2_up_w, b.ff2_down_w);
    x = scaled_residual(ctx, x, y, hp.enc_ff_resid_w0, hp.enc_ff_resid_w1);
  }
  return lasr_layer_norm(ctx, x, b.norm_post_w, ln_eps);
}

// Encoder frame count after one stride-2 conv (padding 0), 0 when too short.
int64_t conv_out_frames(int64_t t_in, int kernel, int stride) {
  return (t_in < kernel) ? 0 : (t_in - kernel) / stride + 1;
}

} // namespace

MedAsrGraphPolicy resolve_graph_policy(const char *backend_name) {
  MedAsrGraphPolicy policy;
  const bool vulkan = backend_name != nullptr && std::strstr(backend_name, "Vulkan") != nullptr;
  policy.direct_pw =
      resolve_conv_direct("TRANSCRIBE_CONV_DIRECT_PW", "TRANSCRIBE_CONV_NO_DIRECT_PW", !vulkan);
  policy.use_flash = true;
  if (env_flag("TRANSCRIBE_NO_FLASH")) {
    policy.use_flash = false;
  }
  if (env_flag("TRANSCRIBE_FORCE_FLASH")) {
    policy.use_flash = true;
  }
  return policy;
}

size_t encoder_graph_context_bytes() {
  return ggml_tensor_overhead() * static_cast<size_t>(kEncoderGraphNodes) +
         ggml_graph_overhead_custom(static_cast<size_t>(kEncoderGraphNodes), false);
}

MedAsrEncoderBuild build_encoder_graph(ggml_context *ctx, const MedAsrWeights &w,
                                       const MedAsrHParams &hp, int n_mel_frames, int n_batch,
                                       bool batch_var_len, const MedAsrGraphPolicy &policy) {
  MedAsrEncoderBuild eb{};
  if (ctx == nullptr || n_mel_frames <= 0) {
    return eb;
  }
  if (n_batch < 1) {
    n_batch = 1;
  }
  // The stem would produce no frame (the arch asserted inside ggml here).
  const int64_t T_enc = conv_out_frames(
      conv_out_frames(n_mel_frames, hp.enc_sub_kernel, hp.enc_sub_stride), hp.enc_sub_kernel,
      hp.enc_sub_stride);
  if (T_enc <= 0) {
    return eb;
  }
  const bool var_len_masks = batch_var_len && n_batch > 1;
  const int d_model = hp.enc_hidden;
  const int n_layers = hp.enc_n_layers;

  // Frame-major mel: ne [n_mels, T_mel, 1, B], each frame n_mels contiguous
  // floats (numpy [B, T_mel, n_mels]).
  eb.mel_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hp.fe_num_mels, n_mel_frames, 1, n_batch);
  ggml_set_name(eb.mel_in, "mel.in");
  ggml_set_input(eb.mel_in);

  ggml_tensor *x = build_subsampling(ctx, w.subsampling, eb.mel_in, hp);
  ggml_set_name(x, "enc.subsampling.out");
  eb.T_enc = x->ne[1];

  eb.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, eb.T_enc);
  ggml_set_name(eb.positions, "enc.positions");
  ggml_set_input(eb.positions);

  if (var_len_masks) {
    eb.attn_pad_mask_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, eb.T_enc, 1, 1, n_batch);
    eb.conv_pad_mask_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, eb.T_enc, 1, n_batch, 1);
    ggml_set_name(eb.attn_pad_mask_in, "enc.attn_pad_mask");
    ggml_set_name(eb.conv_pad_mask_in, "enc.conv_pad_mask");
    ggml_set_input(eb.attn_pad_mask_in);
    ggml_set_input(eb.conv_pad_mask_in);
  }

  for (int i = 0; i < n_layers; ++i) {
    x = build_block(ctx, x, eb.positions, w.blocks[static_cast<size_t>(i)], hp, policy,
                    policy.use_flash, eb.attn_pad_mask_in, eb.conv_pad_mask_in);
    char nm[32];
    std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
    ggml_set_name(x, nm);
  }

  x = lasr_layer_norm(ctx, x, w.enc_out_norm_w, hp.enc_layer_norm_eps);
  ggml_set_name(x, "enc.out_norm.out");

  // CTC head: Conv1d(d -> vocab, k=1) == mul_mat on the [d, vocab] view.
  // Output ne [vocab, T_enc, B]: per utterance, frame-major logits.
  {
    ggml_tensor *proj_w = ggml_reshape_2d(ctx, w.ctc_proj_w, d_model, hp.ctc_vocab_size);
    x = mul_mat_f32acc(ctx, proj_w, x);
    if (w.ctc_proj_b != nullptr) {
      x = ggml_add(ctx, x, w.ctc_proj_b);
    }
  }
  ggml_set_name(x, "enc.ctc_logits.raw");
  eb.logits = x;
  ggml_set_output(eb.logits);

  eb.graph = ggml_new_graph_custom(ctx, static_cast<size_t>(kEncoderGraphNodes), /*grads=*/false);
  if (eb.graph != nullptr) {
    ggml_build_forward_expand(eb.graph, eb.logits);
  }
  return eb;
}

} // namespace engine::models::medasr
