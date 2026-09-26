// engine/models/granite_nar/graphs.cpp - ggml graph builders for the native
// engine Granite Speech NAR package (see graphs.h).
//
// Every builder is a line-for-line port; each helper names its origin:
//   arch encoder.cpp / projector.cpp / decoder.cpp  = src/runtime/arch/granite_nar/
//   shaw_attn.cpp                                   = src/runtime/granite_conformer/
//   conformer.cpp                                   = src/runtime/conformer/
// where "parent" means transcribe.cpp HEAD (585b98f7 Shaw skew path,
// b174a427 bounded BPE-CTC graph), which speech.cpp's arch has not adopted.

#include "engine/models/granite_nar/graphs.h"

#include "engine/models/granite_nar/assets.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace engine::models::granite_nar {

namespace {

// arch encoder.cpp kLayerNormEps (and conformer.h kLayerNormEps): 1e-5.
constexpr float kEncLayerNormEps = 1e-5f;

// arch encoder.cpp / projector.cpp layer_norm (conformer.cpp layer_norm with
// an explicit eps): gamma * norm(x) (+ beta).
ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *gamma, ggml_tensor *beta,
                        float eps) {
  ggml_tensor *y = ggml_norm(ctx, x, eps);
  y = ggml_mul(ctx, y, gamma);
  if (beta != nullptr) {
    y = ggml_add(ctx, y, beta);
  }
  return y;
}

// arch encoder.cpp / projector.cpp linear.
ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b) {
  ggml_tensor *y = ggml_mul_mat(ctx, w, x);
  if (b != nullptr) {
    y = ggml_add(ctx, y, b);
  }
  return y;
}

// conformer.cpp macaron_ff_residual: x + 0.5 * (lin2(SiLU(lin1(LN(x))))).
ggml_tensor *macaron_ff_residual(ggml_context *ctx, ggml_tensor *x, ggml_tensor *norm_w,
                                 ggml_tensor *norm_b, ggml_tensor *lin1_w, ggml_tensor *lin1_b,
                                 ggml_tensor *lin2_w, ggml_tensor *lin2_b) {
  ggml_tensor *y = layer_norm(ctx, x, norm_w, norm_b, kEncLayerNormEps);
  y = ggml_mul_mat(ctx, lin1_w, y);
  if (lin1_b != nullptr) {
    y = ggml_add(ctx, y, lin1_b);
  }
  y = ggml_silu(ctx, y);
  y = ggml_mul_mat(ctx, lin2_w, y);
  if (lin2_b != nullptr) {
    y = ggml_add(ctx, y, lin2_b);
  }
  y = ggml_scale(ctx, y, 0.5f);
  return ggml_add(ctx, x, y);
}

// conformer.cpp rel_shift: Transformer-XL skew of [pos_len, T_q, H, B] so
// column k holds relative offset k.
ggml_tensor *rel_shift(ggml_context *ctx, ggml_tensor *x) {
  const int64_t pos_len = x->ne[0];
  const int64_t T_q = x->ne[1];
  const int64_t H = x->ne[2];
  const int64_t B = x->ne[3];
  ggml_tensor *zero_template = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T_q, H, B);
  ggml_tensor *zeros = ggml_fill(ctx, zero_template, 0.0f);
  ggml_tensor *y = ggml_concat(ctx, zeros, x, /*dim=*/0);
  y = ggml_reshape_4d(ctx, y, T_q, pos_len + 1, H, B);
  y = ggml_view_4d(ctx, y, T_q, pos_len, H, B, y->nb[1], y->nb[2], y->nb[3], /*offset=*/y->nb[1]);
  y = ggml_cont(ctx, y);
  y = ggml_reshape_4d(ctx, y, pos_len, T_q, H, B);
  return y;
}

// conformer.cpp conv_1d_dw_f32, B == 1 im2col branch (the fallback when the
// direct depthwise is switched off by TRANSCRIBE_CONV_NO_DIRECT_DW).
ggml_tensor *conv_1d_dw_f32(ggml_context *ctx, ggml_tensor *kernel, ggml_tensor *data, int stride,
                            int padding, int dilation) {
  ggml_tensor *data_4d = ggml_reshape_4d(ctx, data, data->ne[0], 1, data->ne[1], data->ne[2]);
  ggml_tensor *im2col = ggml_im2col(ctx, kernel, data_4d, stride, /*s1=*/0, padding, /*p1=*/0,
                                    dilation, /*d1=*/0, /*is_2D=*/false, /*dst_type=*/kernel->type);
  ggml_tensor *result = ggml_mul_mat(ctx, im2col, kernel);
  result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);
  return result;
}

// conformer.cpp fused_batch_norm: x * scale + bias, [C] reshaped to
// [1, C, 1, 1] to broadcast over time.
ggml_tensor *fused_batch_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale_1d,
                              ggml_tensor *bias_1d) {
  const int64_t C = scale_1d->ne[0];
  ggml_tensor *scale_4d = ggml_reshape_4d(ctx, scale_1d, 1, C, 1, 1);
  ggml_tensor *bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, C, 1, 1);
  ggml_tensor *y = ggml_mul(ctx, x, scale_4d);
  return ggml_add(ctx, y, bias_4d);
}

// arch encoder.cpp conv_module: LN -> pointwise1 -> GLU -> depthwise ->
// fused BN -> SiLU -> pointwise2 (the caller adds the residual).
ggml_tensor *conv_module(ggml_context *ctx, ggml_tensor *x, const EncBlockWeights &b,
                         int conv_kernel, int inner_dim, bool conv_dw_direct) {
  const int64_t d_model = x->ne[0];
  const int64_t T = x->ne[1];

  x = layer_norm(ctx, x, b.norm_conv_w, b.norm_conv_b, kEncLayerNormEps);
  {
    ggml_tensor *pw1 = ggml_reshape_2d(ctx, b.conv_pointwise1_w, d_model, 2 * inner_dim);
    x = ggml_mul_mat(ctx, pw1, x);
    x = ggml_add(ctx, x, b.conv_pointwise1_b);
  }
  {
    ggml_tensor *gate = ggml_view_2d(ctx, x, inner_dim, T, x->nb[1], 0);
    ggml_tensor *value =
        ggml_view_2d(ctx, x, inner_dim, T, x->nb[1], inner_dim * ggml_element_size(x));
    x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
  }
  x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
  const int padding = (conv_kernel - 1) / 2;
  if (conv_dw_direct) {
    ggml_tensor *knl = b.conv_depthwise_w;
    if (knl->type != GGML_TYPE_F32) {
      knl = ggml_cast(ctx, knl, GGML_TYPE_F32);
    }
    knl = ggml_reshape_4d(ctx, knl, conv_kernel, 1, 1, inner_dim);
    ggml_tensor *d4 = ggml_reshape_4d(ctx, x, x->ne[0], 1, inner_dim, 1);
    x = ggml_conv_2d_dw_direct(ctx, knl, d4, /*s0=*/1, /*s1=*/1, /*p0=*/padding, /*p1=*/0,
                               /*d0=*/1, /*d1=*/1);
    x = ggml_reshape_3d(ctx, x, x->ne[0], inner_dim, 1);
  } else {
    x = conv_1d_dw_f32(ctx, b.conv_depthwise_w, x, /*stride=*/1, /*padding=*/padding,
                       /*dilation=*/1);
  }
  x = fused_batch_norm(ctx, x, b.conv_bn_fused_scale, b.conv_bn_fused_bias);
  x = ggml_silu(ctx, x);
  x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
  {
    ggml_tensor *pw2 = ggml_reshape_2d(ctx, b.conv_pointwise2_w, inner_dim, d_model);
    x = ggml_mul_mat(ctx, pw2, x);
    x = ggml_add(ctx, x, b.conv_pointwise2_b);
  }
  return x;
}

// Parent shaw_attn.cpp shaw_block_attn (585b98f7), single utterance (B = 1;
// the arch never batched). pos_rows != null selects the skew bias, else the
// direct `dists` lookup. Returns [d_model, T_enc].
ggml_tensor *shaw_block_attn(ggml_context *ctx, ggml_tensor *x, ggml_tensor *zero_pad,
                             ggml_tensor *dists, ggml_tensor *pos_rows, ggml_tensor *pad_mask_3d,
                             const EncBlockWeights &w, int n_heads, int head_dim,
                             int context_size, int num_blocks, int T_enc) {
  const int64_t inner_dim = static_cast<int64_t>(n_heads) * head_dim;
  const int64_t T_pad = static_cast<int64_t>(context_size) * num_blocks;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t B = x->ne[2];
  const int64_t num_blocks_eff = static_cast<int64_t>(num_blocks) * B;

  ggml_tensor *h = layer_norm(ctx, x, w.norm_attn_w, w.norm_attn_b, kEncLayerNormEps);
  if (T_pad > T_enc) {
    if (zero_pad == nullptr) {
      return nullptr;
    }
    h = ggml_concat(ctx, h, zero_pad, /*dim=*/1);
  }

  ggml_tensor *q = ggml_mul_mat(ctx, w.attn_q_w, h);
  ggml_tensor *kv = ggml_mul_mat(ctx, w.attn_kv_w, h);
  ggml_tensor *k = ggml_view_3d(ctx, kv, inner_dim, kv->ne[1], kv->ne[2], kv->nb[1], kv->nb[2], 0);
  ggml_tensor *v = ggml_view_3d(ctx, kv, inner_dim, kv->ne[1], kv->ne[2], kv->nb[1], kv->nb[2],
                                inner_dim * ggml_element_size(kv));
  k = ggml_cont(ctx, k);
  v = ggml_cont(ctx, v);

  auto reshape_qkv = [&](ggml_tensor *t) -> ggml_tensor * {
    ggml_tensor *r = ggml_reshape_4d(ctx, t, head_dim, n_heads, context_size, num_blocks_eff);
    return ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
  };
  q = reshape_qkv(q);
  k = reshape_qkv(k);
  v = reshape_qkv(v);

  // [k=ctx, q=ctx, n_heads, num_blocks]
  ggml_tensor *kq = ggml_mul_mat(ctx, k, q);

  ggml_tensor *pos_attn = nullptr;
  if (pos_rows != nullptr) {
    // Skew (parent): bd[d, q] = q . e_sub[d]; rel_shift -> out[k, q] =
    // bd[k - q + ctx - 1, q]; the leading ctx columns are the in-block keys.
    ggml_tensor *e_sub = ggml_get_rows(ctx, w.attn_rel_pos_emb, pos_rows);
    pos_attn = ggml_mul_mat(ctx, e_sub, q);
    pos_attn = rel_shift(ctx, pos_attn);
    pos_attn = ggml_view_4d(ctx, pos_attn, context_size, context_size, n_heads, num_blocks_eff,
                            pos_attn->nb[1], pos_attn->nb[2], pos_attn->nb[3], /*offset=*/0);
  } else {
    // Direct (speech.cpp's arch, pre-585b98f7).
    ggml_tensor *dists_flat =
        ggml_reshape_1d(ctx, dists, static_cast<int64_t>(context_size) * context_size);
    ggml_tensor *rel_lookup = ggml_get_rows(ctx, w.attn_rel_pos_emb, dists_flat);
    rel_lookup = ggml_reshape_3d(ctx, rel_lookup, head_dim, context_size, context_size);
    ggml_tensor *q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    pos_attn = ggml_mul_mat(ctx, rel_lookup, q_perm);
    pos_attn = ggml_cont(ctx, ggml_permute(ctx, pos_attn, 0, 2, 1, 3));
  }

  ggml_tensor *scores = ggml_add(ctx, kq, pos_attn);
  scores = ggml_scale(ctx, scores, scale);
  ggml_tensor *pad_mask_4d =
      ggml_reshape_4d(ctx, pad_mask_3d, context_size, context_size, 1, num_blocks_eff);
  scores = ggml_add(ctx, scores, pad_mask_4d);
  ggml_tensor *attn = ggml_soft_max(ctx, scores);

  ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
  ggml_tensor *out = ggml_mul_mat(ctx, v_t, attn);
  out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
  out = ggml_reshape_3d(ctx, out, inner_dim, T_pad, B);

  // Slice the pad rows off BEFORE out_proj (reference order; keeps pad-row
  // garbage out of the following depthwise conv).
  if (T_pad > T_enc) {
    out = ggml_view_3d(ctx, out, inner_dim, T_enc, B, out->nb[1], out->nb[2], 0);
    out = ggml_cont(ctx, out);
  }
  out = ggml_mul_mat(ctx, w.attn_out_w, out);
  if (w.attn_out_b != nullptr) {
    out = ggml_add(ctx, out, w.attn_out_b);
  }
  return out;
}

// arch projector.cpp cross_attn: plain multi-head attention, inputs
// [hidden, seq, batch], output [hidden, q_seq, batch].
ggml_tensor *cross_attn(ggml_context *ctx, ggml_tensor *q_in, ggml_tensor *k_in,
                        ggml_tensor *v_in, int n_heads, int head_dim) {
  const int64_t q_seq = q_in->ne[1];
  const int64_t k_seq = k_in->ne[1];
  const int64_t batch = q_in->ne[2];
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto split = [&](ggml_tensor *t, int64_t seq) {
    ggml_tensor *r = ggml_reshape_4d(ctx, t, head_dim, n_heads, seq, batch);
    return ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
  };
  ggml_tensor *q = split(q_in, q_seq);
  ggml_tensor *k = split(k_in, k_seq);
  ggml_tensor *v = split(v_in, k_seq);

  ggml_tensor *kq = ggml_mul_mat(ctx, k, q);
  kq = ggml_scale(ctx, kq, scale);
  ggml_tensor *attn = ggml_soft_max(ctx, kq);

  ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
  ggml_tensor *out = ggml_mul_mat(ctx, v_t, attn);
  out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
  return ggml_reshape_3d(ctx, out, static_cast<int64_t>(head_dim) * n_heads, q_seq, batch);
}

// arch decoder.cpp rms_norm.
ggml_tensor *rms_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight, float eps) {
  return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

// arch decoder.cpp block_bidi: one bidirectional Granite-4 block (pre-RMSNorm
// GQA with NeoX RoPE, softmax scaled by attention_multiplier, no mask,
// residual_multiplier on both residuals, SwiGLU MLP).
ggml_tensor *block_bidi(ggml_context *ctx, ggml_tensor *x, const DecBlockWeights &b,
                        const GraniteNarHParams &hp, int T_total, ggml_tensor *positions) {
  const int64_t n_heads = hp.dec_n_heads;
  const int64_t n_kv_heads = hp.dec_n_kv_heads;
  const int64_t n_groups = n_heads / n_kv_heads;
  const int64_t head_dim = hp.dec_head_dim;

  ggml_tensor *x_norm = rms_norm(ctx, x, b.norm_attn_w, hp.dec_rms_norm_eps);
  ggml_tensor *Q = ggml_mul_mat(ctx, b.attn_q_w, x_norm);
  ggml_tensor *K = ggml_mul_mat(ctx, b.attn_k_w, x_norm);
  ggml_tensor *V = ggml_mul_mat(ctx, b.attn_v_w, x_norm);

  Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, T_total, 1);
  K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, T_total, 1);
  V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, T_total, 1);

  Q = ggml_rope_ext(ctx, Q, positions, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX,
                    hp.dec_max_pos_emb, hp.dec_rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
  K = ggml_rope_ext(ctx, K, positions, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX,
                    hp.dec_max_pos_emb, hp.dec_rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

  ggml_tensor *Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
  ggml_tensor *K_att = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
  ggml_tensor *V_att = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

  ggml_tensor *K_full = K_att;
  ggml_tensor *V_full = V_att;
  if (n_groups != 1) {
    ggml_tensor *K_4d = ggml_reshape_4d(ctx, K_att, head_dim, T_total, 1, n_kv_heads);
    ggml_tensor *V_4d = ggml_reshape_4d(ctx, V_att, head_dim, T_total, 1, n_kv_heads);
    ggml_tensor *K_tpl = ggml_new_tensor_4d(ctx, K_att->type, head_dim, T_total, n_groups, n_kv_heads);
    ggml_tensor *V_tpl = ggml_new_tensor_4d(ctx, V_att->type, head_dim, T_total, n_groups, n_kv_heads);
    K_full = ggml_reshape_3d(ctx, ggml_repeat(ctx, K_4d, K_tpl), head_dim, T_total, n_heads);
    V_full = ggml_reshape_3d(ctx, ggml_repeat(ctx, V_4d, V_tpl), head_dim, T_total, n_heads);
  }

  ggml_tensor *kq = ggml_mul_mat(ctx, K_full, Q_att);
  ggml_tensor *kq_soft =
      ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr, hp.dec_attention_multiplier, /*max_bias=*/0.0f);

  ggml_tensor *V_t = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 0, 2, 3));
  ggml_tensor *o = ggml_mul_mat(ctx, V_t, kq_soft);
  o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
  o = ggml_reshape_2d(ctx, o, n_heads * head_dim, T_total);

  o = ggml_mul_mat(ctx, b.attn_o_w, o);
  o = ggml_scale(ctx, o, hp.dec_residual_multiplier);
  x = ggml_add(ctx, x, o);

  ggml_tensor *f_norm = rms_norm(ctx, x, b.norm_ffn_w, hp.dec_rms_norm_eps);
  ggml_tensor *gate = ggml_mul_mat(ctx, b.ffn_gate_w, f_norm);
  ggml_tensor *up = ggml_mul_mat(ctx, b.ffn_up_w, f_norm);
  ggml_tensor *ff = ggml_mul(ctx, ggml_silu(ctx, gate), up);
  ff = ggml_mul_mat(ctx, b.ffn_down_w, ff);
  ff = ggml_scale(ctx, ff, hp.dec_residual_multiplier);
  return ggml_add(ctx, x, ff);
}

} // namespace

bool env_flag(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool detect_conv_dw_direct() {
  // conformer.cpp resolve_conv_direct(DIRECT_DW, NO_DIRECT_DW, true).
  if (env_flag("TRANSCRIBE_CONV_DIRECT_DW")) {
    return true;
  }
  if (env_flag("TRANSCRIBE_CONV_NO_DIRECT_DW")) {
    return false;
  }
  return true;
}

size_t graph_context_bytes() {
  return ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false) +
         1024 * 1024;
}

// ---------------------------------------------------------------------------
// Encoder (arch encoder.cpp build_encoder_graph at the parent's revision)
// ---------------------------------------------------------------------------

EncoderBuild build_encoder_graph(ggml_context *ctx, const GraniteNarWeights &w,
                                 const GraniteNarHParams &hp, int T_enc, ShawBias shaw_bias,
                                 bool conv_dw_direct) {
  EncoderBuild eb{};
  const int ctx_size = hp.enc_context_size;
  eb.n_blocks_local = (T_enc + ctx_size - 1) / ctx_size;
  const int T_pad = eb.n_blocks_local * ctx_size;
  eb.last_block_rem = T_enc - (eb.n_blocks_local - 1) * ctx_size;

  const int64_t d_model = hp.enc_hidden;
  const int64_t inner_dim = static_cast<int64_t>(hp.enc_hidden) * hp.enc_conv_expansion;
  const int n_layers = hp.enc_n_layers;

  eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.enc_input_dim, T_enc);
  ggml_set_name(eb.mel_in, "enc.mel_in");
  ggml_set_input(eb.mel_in);

  if (shaw_bias == ShawBias::Skew) {
    eb.pos_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * static_cast<int64_t>(ctx_size) - 1);
    ggml_set_name(eb.pos_rows, "enc.pos_rows");
    ggml_set_input(eb.pos_rows);
  } else {
    eb.attention_dists =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(ctx_size) * ctx_size);
    ggml_set_name(eb.attention_dists, "enc.attention_dists");
    ggml_set_input(eb.attention_dists);
  }

  eb.last_block_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ctx_size, ctx_size, eb.n_blocks_local);
  ggml_set_name(eb.last_block_mask, "enc.last_block_mask");
  ggml_set_input(eb.last_block_mask);

  if (T_pad > T_enc) {
    eb.zero_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_pad - T_enc);
    ggml_set_name(eb.zero_pad, "enc.zero_pad");
    ggml_set_input(eb.zero_pad);
  }

  ggml_tensor *x = linear(ctx, eb.mel_in, w.enc_top.input_linear_w, w.enc_top.input_linear_b);

  const int bypass_after = n_layers / 2; // 1-indexed boundary (arch: not self_cond_layer)

  // enc_layer_indices -> 0-indexed block outputs (arch encoder.cpp).
  std::vector<int> capture_idx;
  capture_idx.reserve(hp.enc_layer_indices.size());
  for (const int32_t li : hp.enc_layer_indices) {
    if (li > 0) {
      capture_idx.push_back(li - 1);
    } else if (li < 0) {
      capture_idx.push_back(n_layers + li);
    } else {
      capture_idx.push_back(0);
    }
  }
  // Parent (b174a427): the BPE head reads the final layer's state out of
  // cat_out, so the final layer must be one of the captures.
  for (size_t k = 0; k < capture_idx.size(); ++k) {
    if (capture_idx[k] == n_layers - 1) {
      eb.final_capture_offset = static_cast<int64_t>(k) * d_model;
      break;
    }
  }
  if (eb.final_capture_offset < 0) {
    return eb;
  }
  std::vector<ggml_tensor *> captures(capture_idx.size(), nullptr);

  for (int i = 0; i < n_layers; ++i) {
    const auto &b = w.enc_blocks[static_cast<size_t>(i)];

    x = macaron_ff_residual(ctx, x, b.norm_ff1_w, b.norm_ff1_b, b.ff1_up_w, b.ff1_up_b,
                            b.ff1_down_w, b.ff1_down_b);

    ggml_tensor *attn_out = shaw_block_attn(ctx, x, eb.zero_pad, eb.attention_dists, eb.pos_rows,
                                            eb.last_block_mask, b, hp.enc_n_heads, hp.enc_head_dim,
                                            ctx_size, eb.n_blocks_local, T_enc);
    if (attn_out == nullptr) {
      return eb;
    }
    x = ggml_add(ctx, x, attn_out);

    ggml_tensor *conv_out = conv_module(ctx, x, b, hp.enc_conv_kernel_size,
                                        static_cast<int>(inner_dim), conv_dw_direct);
    x = ggml_add(ctx, x, conv_out);

    x = macaron_ff_residual(ctx, x, b.norm_ff2_w, b.norm_ff2_b, b.ff2_up_w, b.ff2_up_b,
                            b.ff2_down_w, b.ff2_down_b);
    x = layer_norm(ctx, x, b.norm_post_w, b.norm_post_b, kEncLayerNormEps);

    // Self-conditioned char-CTC bypass at layer N/2, BEFORE the capture (the
    // reference appends all_hidden_states after the injection).
    if ((i + 1) == bypass_after) {
      ggml_tensor *cl = ggml_mul_mat(ctx, w.enc_top.ctc_proj_w, x);
      cl = ggml_add(ctx, cl, w.enc_top.ctc_proj_b);
      // mid_logits: not an output here (the parent no longer reads it back;
      // the arch marked it only for its dump tap).
      ggml_tensor *cs = ggml_soft_max(ctx, cl);
      ggml_tensor *blank_prob = ggml_view_2d(ctx, cs, 1, T_enc, cs->nb[1], 0);
      blank_prob = ggml_cont(ctx, blank_prob);
      blank_prob = ggml_reshape_1d(ctx, blank_prob, T_enc);
      ggml_set_name(blank_prob, "enc.mid_blank_probs");
      ggml_set_output(blank_prob);
      eb.mid_blank_probs = blank_prob;

      ggml_tensor *bypass = ggml_mul_mat(ctx, w.enc_top.ctc_bypass_w, cs);
      bypass = ggml_add(ctx, bypass, w.enc_top.ctc_bypass_b);
      x = ggml_add(ctx, x, bypass);
    }

    for (size_t k = 0; k < capture_idx.size(); ++k) {
      if (capture_idx[k] == i) {
        captures[k] = x;
      }
    }
  }

  ggml_tensor *cat = nullptr;
  for (ggml_tensor *c : captures) {
    if (c == nullptr) {
      return eb; // a capture index past the last layer
    }
    cat = (cat == nullptr) ? c : ggml_concat(ctx, cat, c, /*dim=*/0);
  }
  ggml_set_name(cat, "enc.cat_out");
  ggml_set_output(cat);
  eb.cat_out = cat;
  if (eb.mid_blank_probs == nullptr) {
    return eb;
  }

  eb.graph = ggml_new_graph_custom(ctx, kGraphNodes, /*grads=*/false);
  ggml_build_forward_expand(eb.graph, eb.cat_out);
  ggml_build_forward_expand(eb.graph, eb.mid_blank_probs);
  return eb;
}

// ---------------------------------------------------------------------------
// Bounded BPE-CTC head (parent encoder.cpp build_bpe_ctc_graph, b174a427)
// ---------------------------------------------------------------------------

BpeCtcBuild build_bpe_ctc_graph(ggml_context *ctx, const GraniteNarWeights &w,
                                const GraniteNarHParams &hp, int n_windows) {
  BpeCtcBuild bb{};
  if (ctx == nullptr || w.enc_top.ctc_bpe_w == nullptr || w.enc_top.ctc_bpe_b == nullptr ||
      hp.enc_hidden <= 0 || hp.enc_bpe_pool_window <= 0 || n_windows <= 0) {
    return bb;
  }
  bb.n_windows = n_windows;
  bb.hidden_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.enc_hidden, n_windows);
  ggml_set_name(bb.hidden_in, "enc.ctc_bpe.hidden_in");
  ggml_set_input(bb.hidden_in);

  ggml_tensor *logits = ggml_mul_mat(ctx, w.enc_top.ctc_bpe_w, bb.hidden_in);
  logits = ggml_add(ctx, logits, w.enc_top.ctc_bpe_b);

  bb.token_ids = ggml_argmax(ctx, logits);
  ggml_set_name(bb.token_ids, "enc.ctc_bpe.token_ids");
  ggml_set_output(bb.token_ids);

  bb.graph = ggml_new_graph_custom(ctx, kGraphNodes, /*grads=*/false);
  ggml_build_forward_expand(bb.graph, bb.token_ids);
  return bb;
}

// ---------------------------------------------------------------------------
// Projector (arch projector.cpp build_projector_graph)
// ---------------------------------------------------------------------------

ProjectorBuild build_projector_graph(ggml_context *ctx, const GraniteNarWeights &w,
                                     const GraniteNarHParams &hp, int T_enc) {
  ProjectorBuild pb{};
  const int block_size = hp.prj_block_size;
  const int downsample = hp.prj_downsample_rate;
  const int n_query = block_size / downsample;
  const int n_heads = hp.prj_n_heads;
  const int prj_hidden = hp.prj_hidden;
  const int enc_layers = hp.prj_num_encoder_layers;
  const int enc_hidden = hp.prj_encoder_dim;
  const int cat_in = enc_layers * enc_hidden;
  const float ln_eps = hp.prj_layernorm_eps;
  if (prj_hidden % n_heads != 0) {
    return pb;
  }
  const int head_dim = prj_hidden / n_heads;

  pb.nblocks = (T_enc + block_size - 1) / block_size;
  const int T_pad = pb.nblocks * block_size;
  const int t_enc_pad = T_pad - T_enc;
  pb.n_audio_tokens = pb.nblocks * n_query;

  pb.enc_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cat_in, T_enc);
  ggml_set_name(pb.enc_in, "proj.enc_in");
  ggml_set_input(pb.enc_in);

  // Per-capture LayerNorm over channel slices, concatenated back.
  ggml_tensor *normed = nullptr;
  const size_t slice_bytes = ggml_element_size(pb.enc_in) * static_cast<size_t>(enc_hidden);
  for (int j = 0; j < enc_layers; ++j) {
    ggml_tensor *slc = ggml_view_2d(ctx, pb.enc_in, enc_hidden, T_enc, pb.enc_in->nb[1],
                                    slice_bytes * static_cast<size_t>(j));
    slc = ggml_cont(ctx, slc);
    slc = layer_norm(ctx, slc, w.proj_top.layer_norms_w[static_cast<size_t>(j)],
                     w.proj_top.layer_norms_b[static_cast<size_t>(j)], ln_eps);
    normed = (normed == nullptr) ? slc : ggml_concat(ctx, normed, slc, /*dim=*/0);
  }

  ggml_tensor *proj_lp =
      linear(ctx, normed, w.proj_top.layer_projector_w, w.proj_top.layer_projector_b);
  proj_lp = ggml_gelu(ctx, proj_lp);

  ggml_tensor *full = proj_lp;
  if (t_enc_pad > 0) {
    pb.enc_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, prj_hidden, t_enc_pad);
    ggml_set_name(pb.enc_pad, "proj.t_pad_zero");
    ggml_set_input(pb.enc_pad);
    full = ggml_concat(ctx, proj_lp, pb.enc_pad, /*dim=*/1);
  }

  ggml_tensor *windowed = ggml_reshape_3d(ctx, full, prj_hidden, block_size, pb.nblocks);

  ggml_tensor *wpos = ggml_cast(ctx, w.proj_top.window_positions, GGML_TYPE_F32);
  ggml_tensor *kv = ggml_add(ctx, windowed, wpos);

  // Mean over each downsample group: [h, ds, n_query, nblocks] -> permute ds
  // to ne[0] -> ggml_mean -> [h, n_query, nblocks].
  ggml_tensor *mp_view = ggml_reshape_4d(ctx, windowed, prj_hidden, downsample, n_query, pb.nblocks);
  ggml_tensor *mp_perm = ggml_cont(ctx, ggml_permute(ctx, mp_view, 1, 0, 2, 3));
  ggml_tensor *mp_mean = ggml_mean(ctx, mp_perm);
  ggml_tensor *mean_pool = ggml_reshape_3d(ctx, mp_mean, prj_hidden, n_query, pb.nblocks);

  ggml_tensor *query = ggml_cast(ctx, w.proj_top.query, GGML_TYPE_F32);
  query = ggml_add(ctx, mean_pool, query);

  for (int i = 0; i < hp.prj_n_layers; ++i) {
    const auto &b = w.proj_blocks[static_cast<size_t>(i)];
    ggml_tensor *q_norm = layer_norm(ctx, query, b.norm_attn_w, b.norm_attn_b, ln_eps);
    ggml_tensor *q_proj = linear(ctx, q_norm, b.cross_attn_q_w, b.cross_attn_q_b);
    ggml_tensor *k_proj = linear(ctx, kv, b.cross_attn_k_w, b.cross_attn_k_b);
    ggml_tensor *v_proj = linear(ctx, kv, b.cross_attn_v_w, b.cross_attn_v_b);
    ggml_tensor *attn = cross_attn(ctx, q_proj, k_proj, v_proj, n_heads, head_dim);
    attn = linear(ctx, attn, b.cross_attn_o_w, b.cross_attn_o_b);
    query = ggml_add(ctx, query, attn);

    ggml_tensor *f_norm = layer_norm(ctx, query, b.norm_ffn_w, b.norm_ffn_b, ln_eps);
    ggml_tensor *mid = linear(ctx, f_norm, b.ffn_fc1_w, b.ffn_fc1_b);
    mid = ggml_silu(ctx, mid);
    ggml_tensor *f_out = linear(ctx, mid, b.ffn_fc2_w, b.ffn_fc2_b);
    query = ggml_add(ctx, query, f_out);
  }

  ggml_tensor *onorm = layer_norm(ctx, query, w.proj_top.out_norm_w, w.proj_top.out_norm_b, ln_eps);
  ggml_tensor *tokens =
      ggml_reshape_2d(ctx, onorm, prj_hidden, static_cast<int64_t>(n_query) * pb.nblocks);
  ggml_tensor *out = ggml_mul_mat(ctx, w.proj_top.out_linear_w, tokens);
  out = ggml_add(ctx, out, w.proj_top.out_linear_b);
  ggml_set_name(out, "proj.out");
  ggml_set_output(out);
  pb.out = out;

  pb.graph = ggml_new_graph_custom(ctx, kGraphNodes, /*grads=*/false);
  ggml_build_forward_expand(pb.graph, pb.out);
  return pb;
}

// ---------------------------------------------------------------------------
// Bidirectional editor (arch decoder.cpp build_forward_graph)
// ---------------------------------------------------------------------------

ForwardBuild build_forward_graph(ggml_context *ctx, const GraniteNarWeights &w,
                                 const GraniteNarHParams &hp, int n_audio_tokens, int n_text) {
  ForwardBuild fb{};
  fb.n_audio_tokens = n_audio_tokens;
  fb.n_text = n_text;
  fb.T_total = n_audio_tokens + n_text;
  if (ctx == nullptr || n_audio_tokens <= 0 || n_text <= 0) {
    return fb;
  }
  const int64_t hidden = hp.dec_hidden;

  fb.audio_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_audio_tokens);
  ggml_set_name(fb.audio_in, "dec.audio_in");
  ggml_set_input(fb.audio_in);
  fb.text_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text);
  ggml_set_name(fb.text_ids_in, "dec.text_ids");
  ggml_set_input(fb.text_ids_in);
  fb.positions_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, fb.T_total);
  ggml_set_name(fb.positions_in, "dec.positions");
  ggml_set_input(fb.positions_in);

  // get_rows returns the table dtype; cast so the concat with f32 audio is legal.
  ggml_tensor *text_emb = ggml_get_rows(ctx, w.dec_token_embd, fb.text_ids_in);
  if (text_emb->type != GGML_TYPE_F32) {
    text_emb = ggml_cast(ctx, text_emb, GGML_TYPE_F32);
  }
  ggml_tensor *flat = ggml_concat(ctx, fb.audio_in, text_emb, /*dim=*/1);
  // embedding_multiplier over the whole sequence (audio rows were pre-divided).
  ggml_tensor *x = ggml_scale(ctx, flat, hp.dec_embedding_multiplier);

  for (int il = 0; il < hp.dec_n_layers; ++il) {
    x = block_bidi(ctx, x, w.dec_blocks[static_cast<size_t>(il)], hp, fb.T_total, fb.positions_in);
  }
  x = rms_norm(ctx, x, w.dec_output_norm, hp.dec_rms_norm_eps);

  ggml_tensor *text_x =
      ggml_view_2d(ctx, x, hidden, n_text, ggml_element_size(x) * hidden,
                   ggml_element_size(x) * hidden * static_cast<size_t>(n_audio_tokens));
  text_x = ggml_cont(ctx, text_x);

  ggml_tensor *logits = ggml_mul_mat(ctx, w.dec_token_embd, text_x);
  if (hp.dec_logits_scaling > 0.0f && hp.dec_logits_scaling != 1.0f) {
    logits = ggml_scale(ctx, logits, 1.0f / hp.dec_logits_scaling);
  }
  ggml_set_name(logits, "dec.text_logits");
  ggml_set_output(logits);
  fb.out = logits;

  fb.graph = ggml_new_graph_custom(ctx, kGraphNodes, /*grads=*/false);
  ggml_build_forward_expand(fb.graph, fb.out);
  return fb;
}

} // namespace engine::models::granite_nar
