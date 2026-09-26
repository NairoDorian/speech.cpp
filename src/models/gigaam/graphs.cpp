// engine/models/gigaam/graphs.cpp - Conformer encoder graph builder for the
// native engine GigaAM ASR package.
//
// Ported from src/runtime/arch/gigaam/encoder.cpp with identical graph
// topology and numerics. The shared transcribe::conformer helpers the arch
// borrowed (layer_norm, feed_forward, macaron_ff_residual, conv_1d_f32,
// conv_module, detect_direct_pw) are reimplemented locally - restricted to
// the paths GigaAM actually takes (LayerNorm conv norm, symmetric depthwise
// padding, direct depthwise op) - so the package has no dependency on
// src/runtime/. The transcribe-side tensor-dump plumbing is not carried over,
// and the arch's final [B, T, D] -> [B, D, T] transpose (`enc.out`, a dump
// only; the decoders read `rnnt.encoded`) is not built.

#include "engine/models/gigaam/graphs_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace engine::models::gigaam {

namespace {

// LayerNorm epsilon (NeMo / GigaAM default, not stored in the GGUF).
constexpr float kLayerNormEps = 1e-5f;

// Manual attention tiles the query axis above this many encoder frames.
constexpr int64_t kAttentionQueryChunk = 256;
constexpr int64_t kAttentionChunkMinT = 2048;

// GigaAM's rotary base. The arch hard-codes 5000 (the reference passes
// pos_emb_max_len, 5000 in every published checkpoint, as the base; the
// dumped enc.pos_emb matches base=5000 to ~1e-6 and base=10000 is off by
// ~0.1). Kept as the arch's constant, not re-derived from the KV.
constexpr float kRotaryBase = 5000.0f;

bool env_flag(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// LayerNorm with affine over ne[0] (the d_model axis of [d_model, T, B]).
ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *gamma,
                        ggml_tensor *beta) {
  ggml_tensor *y = ggml_norm(ctx, x, kLayerNormEps);
  y = ggml_mul(ctx, y, gamma);
  if (beta != nullptr) {
    y = ggml_add(ctx, y, beta);
  }
  return y;
}

// y = Linear2(SiLU(Linear1(x))).
ggml_tensor *feed_forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *lin1_w,
                          ggml_tensor *lin1_b, ggml_tensor *lin2_w, ggml_tensor *lin2_b) {
  ggml_tensor *y = ggml_mul_mat(ctx, lin1_w, x);
  if (lin1_b != nullptr) {
    y = ggml_add(ctx, y, lin1_b);
  }
  y = ggml_silu(ctx, y);
  y = ggml_mul_mat(ctx, lin2_w, y);
  if (lin2_b != nullptr) {
    y = ggml_add(ctx, y, lin2_b);
  }
  return y;
}

// x + 0.5 * FF(LN(x)).
ggml_tensor *macaron_ff_residual(ggml_context *ctx, ggml_tensor *x, ggml_tensor *norm_w,
                                 ggml_tensor *norm_b, ggml_tensor *lin1_w,
                                 ggml_tensor *lin1_b, ggml_tensor *lin2_w,
                                 ggml_tensor *lin2_b) {
  ggml_tensor *y = layer_norm(ctx, x, norm_w, norm_b);
  y = feed_forward(ctx, y, lin1_w, lin1_b, lin2_w, lin2_b);
  y = ggml_scale(ctx, y, 0.5f);
  return ggml_add(ctx, x, y);
}

// f32-friendly Conv1D: ggml_conv_1d but im2col keeps the kernel's real type
// (vendored ggml_conv_1d forces an F16 im2col and asserts on F32 kernels).
//   kernel ne [K, IC, OC]; data ne [W, IC, N] -> result ne [OW, OC, N].
ggml_tensor *conv_1d_f32(ggml_context *ctx, ggml_tensor *kernel, ggml_tensor *data, int stride,
                         int padding, int dilation) {
  ggml_tensor *im2col = ggml_im2col(ctx, kernel, data, stride, /*s1=*/0, padding, /*p1=*/0,
                                    dilation, /*d1=*/0, /*is_2D=*/false,
                                    /*dst_type=*/kernel->type);
  const int64_t N = im2col->ne[2];
  ggml_tensor *kernel_2d =
      ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]); // [IC*K, OC]
  // F32 accumulation for F16 kernels (CUDA COMPUTE_16F saturation).
  const bool kernel_needs_f32_acc = (kernel->type == GGML_TYPE_F16);

  if (N == 1) {
    ggml_tensor *result = ggml_mul_mat(
        ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1]), kernel_2d); // [OW, OC]
    if (kernel_needs_f32_acc) {
      ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    }
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], 1);
  }

  // Batched: mul_mat the 3-D im2col directly (the kernel broadcasts over the
  // batch), giving [OC, OW, N], then permute to [OW, OC, N].
  ggml_tensor *result = ggml_mul_mat(ctx, kernel_2d, im2col);
  if (kernel_needs_f32_acc) {
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
  }
  return ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
}

// ggml_conv_2d_dw_direct needs an F32 kernel (hard assert on CUDA); the
// depthwise kernel is tiny, so promote in-graph.
ggml_tensor *dw_kernel_for_direct(ggml_context *ctx, ggml_tensor *kernel) {
  if (kernel == nullptr || kernel->type == GGML_TYPE_F32) {
    return kernel;
  }
  return ggml_cast(ctx, kernel, GGML_TYPE_F32);
}

// Conv module: pw1 -> GLU -> [pad mask] -> depthwise -> LayerNorm -> SiLU ->
// pw2 on the post-LayerNorm activation [d_model, T, B]. GigaAM's paths of the
// arch's conformer::conv_module: symmetric (k-1)/2 depthwise padding, the
// direct depthwise op (ConvPolicy::direct_dw_in_block = true), LayerNorm.
ggml_tensor *conv_module(ggml_context *ctx, ggml_tensor *x, const GigaamBlockWeights &b,
                         int conv_kernel, bool direct_pw, ggml_tensor *conv_pad_mask) {
  const int pad = (conv_kernel - 1) / 2;
  const int64_t d_model = x->ne[0];
  const int64_t B = x->ne[2];

  if (direct_pw) {
    // Pointwise conv 1 as a mul_mat in [d_model, T, B] layout.
    ggml_tensor *pw1 = ggml_reshape_2d(ctx, b.conv_pw1_w, d_model, 2 * d_model);
    x = ggml_mul_mat(ctx, pw1, x); // [2*d_model, T, B]
    if (b.conv_pw1_w->type == GGML_TYPE_F16) {
      ggml_mul_mat_set_prec(x, GGML_PREC_F32);
    }
    if (b.conv_pw1_b != nullptr) {
      x = ggml_add(ctx, x, b.conv_pw1_b);
    }
    // GLU over ne[0], per utterance.
    {
      const int64_t T = x->ne[1];
      const int64_t half = x->ne[0] / 2;
      ggml_tensor *gate = ggml_view_3d(ctx, x, half, T, B, x->nb[1], x->nb[2], /*offset=*/0);
      ggml_tensor *value =
          ggml_view_3d(ctx, x, half, T, B, x->nb[1], x->nb[2], half * ggml_element_size(x));
      x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
    }
    // [d_model, T, B] -> [T, d_model, B] for the depthwise conv.
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
  } else {
    // im2col fallback in [T, d_model] layout (single utterance, as the arch).
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    x = conv_1d_f32(ctx, b.conv_pw1_w, x, /*s=*/1, /*p=*/0, /*d=*/1);
    if (b.conv_pw1_b != nullptr) {
      x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
      ggml_tensor *bias_r = ggml_reshape_2d(ctx, b.conv_pw1_b, 1, 2 * d_model);
      x = ggml_add(ctx, x, bias_r);
    }
    const int64_t half = x->ne[1] / 2;
    ggml_tensor *gate =
        ggml_view_4d(ctx, x, x->ne[0], half, 1, 1, x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor *value =
        ggml_view_4d(ctx, x, x->ne[0], half, 1, 1, x->nb[1], x->nb[2], x->nb[3], x->nb[1] * half);
    x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
    x = ggml_cont(ctx, x);
  }

  // Zero padded frames before the depthwise conv (variable-length batches).
  if (conv_pad_mask != nullptr) {
    x = ggml_mul(ctx, x, conv_pad_mask);
  }

  // Depthwise conv as the fused single op: data [T, 1, d_model, B], kernel
  // [k, 1, 1, d_model].
  {
    ggml_tensor *knl =
        ggml_reshape_4d(ctx, dw_kernel_for_direct(ctx, b.conv_dw_w), conv_kernel, 1, 1, d_model);
    ggml_tensor *data = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], B);
    x = ggml_conv_2d_dw_direct(ctx, knl, data, /*s0=*/1, /*s1=*/1, /*p0=*/pad, /*p1=*/0,
                               /*d0=*/1, /*d1=*/1);
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[2], x->ne[3]); // [T, d_model, B]
  }
  if (b.conv_dw_b != nullptr) {
    ggml_tensor *bias_r = ggml_reshape_2d(ctx, b.conv_dw_b, 1, d_model);
    x = ggml_add(ctx, x, bias_r);
  }

  // Post-depthwise LayerNorm over channels: permute to [d_model, T], normalize,
  // permute back.
  x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
  x = layer_norm(ctx, x, b.conv_ln_w, b.conv_ln_b);
  x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

  x = ggml_silu(ctx, x);

  if (direct_pw) {
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [d_model, T, B]
    ggml_tensor *pw2 = ggml_reshape_2d(ctx, b.conv_pw2_w, d_model, d_model);
    x = ggml_mul_mat(ctx, pw2, x);
    if (b.conv_pw2_w->type == GGML_TYPE_F16) {
      ggml_mul_mat_set_prec(x, GGML_PREC_F32);
    }
    if (b.conv_pw2_b != nullptr) {
      x = ggml_add(ctx, x, b.conv_pw2_b);
    }
  } else {
    x = conv_1d_f32(ctx, b.conv_pw2_w, x, /*s=*/1, /*p=*/0, /*d=*/1);
    if (b.conv_pw2_b != nullptr) {
      x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
      ggml_tensor *bias_r = ggml_reshape_2d(ctx, b.conv_pw2_b, 1, d_model);
      x = ggml_add(ctx, x, bias_r);
    }
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
  }
  return x;
}

// 2-conv1d pre-encode: mel_in ne [T_mel, n_mels, B] -> [d_model, T_enc, B].
// With masks (variable-length batch), each conv's padded time region is
// zeroed after its ReLU (NeMo masked subsampling), so padding cannot leak
// into an utterance's last valid frame through the next stride-2 conv.
ggml_tensor *build_pre_encode(ggml_context *ctx, const GigaamPreEncodeWeights &pe,
                              ggml_tensor *mel_in, int subs_kernel_size, ggml_tensor *mask_s1,
                              ggml_tensor *mask_s2) {
  const int padding = (subs_kernel_size - 1) / 2;

  ggml_tensor *x = conv_1d_f32(ctx, pe.conv0_w, mel_in, /*stride=*/2, padding, /*dilation=*/1);
  x = ggml_add(ctx, x, ggml_reshape_3d(ctx, pe.conv0_b, 1, x->ne[1], 1));
  x = ggml_relu(ctx, x);
  if (mask_s1 != nullptr) {
    x = ggml_mul(ctx, x, mask_s1);
  }

  x = conv_1d_f32(ctx, pe.conv2_w, x, /*stride=*/2, padding, /*dilation=*/1);
  x = ggml_add(ctx, x, ggml_reshape_3d(ctx, pe.conv2_b, 1, x->ne[1], 1));
  x = ggml_relu(ctx, x);
  if (mask_s2 != nullptr) {
    x = ggml_mul(ctx, x, mask_s2);
  }

  // [T_enc, d_model, B] -> [d_model, T_enc, B].
  return ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
}

// Rotary self-attention. GigaAM rotates the PRE-projection x (reshaped to
// [head_dim, n_head, T, B], NEOX split-halves) and applies Wq / Wk to the
// rotated tensor; Wv sees the unrotated x.
ggml_tensor *build_rotary_attn(ggml_context *ctx, ggml_tensor *x, ggml_tensor *positions,
                               const GigaamBlockWeights &b, int d_model, int n_head,
                               bool use_flash, ggml_tensor *attn_pad_mask) {
  const int head_dim = d_model / n_head;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t T = x->ne[1];
  const int64_t Bb = x->ne[2];

  // A per-utterance key-padding mask goes through the manual path.
  const bool flash = use_flash && (attn_pad_mask == nullptr);

  ggml_tensor *x_rot = ggml_reshape_4d(ctx, x, head_dim, n_head, T, Bb);
  x_rot = ggml_rope_ext(ctx, x_rot, positions, /*c=*/nullptr,
                        /*n_dims=*/head_dim, GGML_ROPE_TYPE_NEOX,
                        /*n_ctx_orig=*/0,
                        /*freq_base=*/kRotaryBase,
                        /*freq_scale=*/1.0f,
                        /*ext_factor=*/0.0f,
                        /*attn_factor=*/1.0f,
                        /*beta_fast=*/32.0f,
                        /*beta_slow=*/1.0f);
  x_rot = ggml_cont(ctx, ggml_reshape_3d(ctx, x_rot, d_model, T, Bb));

  ggml_tensor *q = ggml_mul_mat(ctx, b.attn_q_w, x_rot);
  if (b.attn_q_b != nullptr) {
    q = ggml_add(ctx, q, b.attn_q_b);
  }
  ggml_tensor *k = ggml_mul_mat(ctx, b.attn_k_w, x_rot);
  if (b.attn_k_b != nullptr) {
    k = ggml_add(ctx, k, b.attn_k_b);
  }
  ggml_tensor *v = ggml_mul_mat(ctx, b.attn_v_w, x);
  if (b.attn_v_b != nullptr) {
    v = ggml_add(ctx, v, b.attn_v_b);
  }

  // [d_model, T, B] -> [head_dim, T, n_head, B].
  auto to_attn = [&](ggml_tensor *t) -> ggml_tensor * {
    t = ggml_reshape_4d(ctx, t, head_dim, n_head, T, Bb);
    t = ggml_permute(ctx, t, 0, 2, 1, 3);
    return ggml_cont(ctx, t);
  };
  q = to_attn(q);
  k = to_attn(k);
  v = to_attn(v);

  ggml_tensor *o = nullptr;
  if (flash) {
    // Writes contiguous [head_dim, n_head, T, B].
    o = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, scale, 0.0f, 0.0f);
  } else if (T <= kAttentionChunkMinT) {
    ggml_tensor *kq = ggml_mul_mat(ctx, k, q); // [T_k, T_q, n_head, B]
    if (attn_pad_mask != nullptr) {
      kq = ggml_add(ctx, kq, attn_pad_mask);
    }
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);                // [head_dim, T, n_head, B]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)); // [head_dim, n_head, T, B]
  } else {
    // Tile the query axis only: every query still attends all T keys.
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    std::vector<ggml_tensor *> chunks;
    chunks.reserve(static_cast<size_t>((T + kAttentionQueryChunk - 1) / kAttentionQueryChunk));
    for (int64_t q0 = 0; q0 < T; q0 += kAttentionQueryChunk) {
      const int64_t n_query = std::min<int64_t>(kAttentionQueryChunk, T - q0);
      ggml_tensor *q_chunk = ggml_view_4d(ctx, q, head_dim, n_query, n_head, Bb, q->nb[1],
                                          q->nb[2], q->nb[3], q0 * q->nb[1]);
      q_chunk = ggml_cont(ctx, q_chunk);
      ggml_tensor *kq = ggml_mul_mat(ctx, k, q_chunk);
      if (attn_pad_mask != nullptr) {
        kq = ggml_add(ctx, kq, attn_pad_mask);
      }
      ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr, scale, 0.0f);
      ggml_tensor *chunk = ggml_mul_mat(ctx, v_t, kq_soft);
      chunk = ggml_cont(ctx, ggml_permute(ctx, chunk, 0, 2, 1, 3));
      chunk = ggml_reshape_3d(ctx, chunk, d_model, n_query, Bb);
      chunks.push_back(chunk);
    }
    while (chunks.size() > 1) {
      std::vector<ggml_tensor *> next;
      next.reserve((chunks.size() + 1) / 2);
      for (size_t i = 0; i < chunks.size(); i += 2) {
        next.push_back(i + 1 < chunks.size()
                           ? ggml_concat(ctx, chunks[i], chunks[i + 1], /*dim=*/1)
                           : chunks[i]);
      }
      chunks.swap(next);
    }
    o = chunks.front();
  }
  o = ggml_reshape_3d(ctx, o, d_model, T, Bb);

  o = ggml_mul_mat(ctx, b.attn_out_w, o);
  if (b.attn_out_b != nullptr) {
    o = ggml_add(ctx, o, b.attn_out_b);
  }
  return o;
}

// One Conformer block; returns norm_out(x) (the reference block output IS the
// post-LN value, not the residual stream).
ggml_tensor *build_block(ggml_context *ctx, ggml_tensor *x, ggml_tensor *positions,
                         const GigaamBlockWeights &b, const GigaamHParams &hp,
                         const GigaamGraphPolicy &policy, ggml_tensor *attn_pad_mask,
                         ggml_tensor *conv_pad_mask) {
  x = macaron_ff_residual(ctx, x, b.norm_ff1_w, b.norm_ff1_b, b.ff1_lin1_w, b.ff1_lin1_b,
                          b.ff1_lin2_w, b.ff1_lin2_b);
  {
    ggml_tensor *y = layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    y = build_rotary_attn(ctx, y, positions, b, hp.enc_d_model, hp.enc_n_heads,
                          policy.use_flash, attn_pad_mask);
    x = ggml_add(ctx, x, y);
  }
  {
    ggml_tensor *y = layer_norm(ctx, x, b.norm_conv_w, b.norm_conv_b);
    y = conv_module(ctx, y, b, hp.enc_conv_kernel, policy.direct_pw, conv_pad_mask);
    x = ggml_add(ctx, x, y);
  }
  x = macaron_ff_residual(ctx, x, b.norm_ff2_w, b.norm_ff2_b, b.ff2_lin1_w, b.ff2_lin1_b,
                          b.ff2_lin2_w, b.ff2_lin2_b);
  return layer_norm(ctx, x, b.norm_out_w, b.norm_out_b);
}

} // namespace

GigaamGraphPolicy resolve_gigaam_graph_policy(const char *backend_name) {
  GigaamGraphPolicy policy;
  const std::string name = backend_name != nullptr ? backend_name : "";
  policy.use_flash = !name.empty() && name.find("CPU") == std::string::npos;
  // conformer::detect_direct_pw: Vulkan defaults to im2col, everything else
  // to the direct mul_mat; the DIRECT env var wins over NO_DIRECT.
  const bool backend_default = name.find("Vulkan") == std::string::npos;
  if (env_flag("TRANSCRIBE_CONV_DIRECT_PW")) {
    policy.direct_pw = true;
  } else if (env_flag("TRANSCRIBE_CONV_NO_DIRECT_PW")) {
    policy.direct_pw = false;
  } else {
    policy.direct_pw = backend_default;
  }
  return policy;
}

int gigaam_pre_encode_t_out(int in) { return (in - 1) / 2 + 1; }

int gigaam_encoder_frames(int n_mel_frames) {
  return gigaam_pre_encode_t_out(gigaam_pre_encode_t_out(n_mel_frames));
}

size_t gigaam_encoder_context_bytes(int n_mel_frames) {
  // ~80 tensors per block on the manual path, plus ~10 per 256-frame query
  // tile once the chunked path engages; generous upper bound.
  const size_t T_enc = static_cast<size_t>(std::max(1, gigaam_encoder_frames(n_mel_frames)));
  const size_t n_tensors = 4096 + 64 * ((T_enc + 255) / 256) * 32;
  return n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false) +
         1024 * 1024;
}

GigaamEncoderBuild build_gigaam_encoder_graph(ggml_context *ctx,
                                              const GigaamEncoderWeights &weights,
                                              const GigaamHParams &hp, int n_mel_frames,
                                              const GigaamGraphPolicy &policy, int n_batch,
                                              bool batch_var_len) {
  GigaamEncoderBuild eb;
  if (ctx == nullptr || n_mel_frames <= 0 || weights.blocks.empty()) {
    return eb;
  }
  if (n_batch < 1) {
    n_batch = 1;
  }
  const bool var_len_masks = batch_var_len && n_batch > 1;

  eb.mel_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_mel_frames, hp.fe_num_mels, n_batch);
  ggml_set_name(eb.mel_in, "mel.in");
  ggml_set_input(eb.mel_in);

  if (var_len_masks) {
    const int T_s1 = gigaam_pre_encode_t_out(n_mel_frames);
    const int T_s2 = gigaam_pre_encode_t_out(T_s1);
    eb.pre_encode_mask_s1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_s1, 1, n_batch);
    eb.pre_encode_mask_s2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_s2, 1, n_batch);
    ggml_set_name(eb.pre_encode_mask_s1, "enc.pre_encode.mask_s1");
    ggml_set_name(eb.pre_encode_mask_s2, "enc.pre_encode.mask_s2");
    ggml_set_input(eb.pre_encode_mask_s1);
    ggml_set_input(eb.pre_encode_mask_s2);
  }

  ggml_tensor *x = build_pre_encode(ctx, weights.pre_encode, eb.mel_in, hp.enc_subs_kernel_size,
                                    eb.pre_encode_mask_s1, eb.pre_encode_mask_s2);
  ggml_set_name(x, "enc.subsample.out");
  const int64_t T_enc = x->ne[1];
  eb.T_enc = static_cast<int>(T_enc);

  if (var_len_masks) {
    eb.attn_pad_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T_enc, 1, 1, n_batch);
    eb.conv_pad_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T_enc, 1, n_batch, 1);
    ggml_set_name(eb.attn_pad_mask, "enc.attn_pad_mask");
    ggml_set_name(eb.conv_pad_mask, "enc.conv_pad_mask");
    ggml_set_input(eb.attn_pad_mask);
    ggml_set_input(eb.conv_pad_mask);
  }

  eb.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_enc);
  ggml_set_name(eb.positions, "enc.positions");
  ggml_set_input(eb.positions);

  for (size_t i = 0; i < weights.blocks.size(); ++i) {
    x = build_block(ctx, x, eb.positions, weights.blocks[i], hp, policy, eb.attn_pad_mask,
                    eb.conv_pad_mask);
  }

  // ne [d_model, T_enc, B]: the arch's `rnnt.encoded`, which both heads read.
  ggml_set_name(x, "rnnt.encoded");
  ggml_set_output(x);
  eb.encoded = x;

  eb.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
  ggml_build_forward_expand(eb.graph, eb.encoded);
  return eb;
}

} // namespace engine::models::gigaam
