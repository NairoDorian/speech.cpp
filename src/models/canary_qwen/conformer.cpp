// engine/models/canary_qwen/conformer.cpp - the FastConformer encoder
// primitives the canary_qwen arch drove, ported from src/runtime/conformer/
// conformer.cpp with the same op sequence. Only the reachable subset is
// here: offline (no streaming caches), full attention (no local window, no
// chunked / key-padding masks), symmetric depthwise padding, BatchNorm, the
// non-causal pre_encode stem without valid-frame masks. Each function names
// its runtime origin; any divergence from it is a bug.

#include "engine/models/canary_qwen/graphs.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace engine::models::canary_qwen::conformer {

namespace {

// conformer.cpp: layer_norm
ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *gamma, ggml_tensor *beta) {
  ggml_tensor *y = ggml_norm(ctx, x, kLayerNormEps);
  y = ggml_mul(ctx, y, gamma);
  if (beta != nullptr) {
    y = ggml_add(ctx, y, beta);
  }
  return y;
}

// conformer.cpp: feed_forward
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

// conformer.cpp: macaron_ff_residual (x + 0.5 * FF(LN(x)))
ggml_tensor *macaron_ff_residual(ggml_context *ctx, ggml_tensor *x, ggml_tensor *norm_w,
                                 ggml_tensor *norm_b, ggml_tensor *lin1_w, ggml_tensor *lin1_b,
                                 ggml_tensor *lin2_w, ggml_tensor *lin2_b) {
  ggml_tensor *y = layer_norm(ctx, x, norm_w, norm_b);
  y = feed_forward(ctx, y, lin1_w, lin1_b, lin2_w, lin2_b);
  y = ggml_scale(ctx, y, 0.5f);
  return ggml_add(ctx, x, y);
}

// conformer.cpp: conv_1d_f32 (im2col with the kernel's own dtype)
ggml_tensor *conv_1d_f32(ggml_context *ctx, ggml_tensor *kernel, ggml_tensor *data, int stride,
                         int padding, int dilation) {
  ggml_tensor *im2col = ggml_im2col(ctx, kernel, data, stride, 0, padding, 0, dilation, 0,
                                    /*is_2D=*/false, kernel->type);
  const int64_t N = im2col->ne[2];
  ggml_tensor *kernel_2d = ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
  const bool kernel_needs_f32_acc = (kernel->type == GGML_TYPE_F16);

  if (N == 1) {
    ggml_tensor *result =
        ggml_mul_mat(ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1]), kernel_2d);
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

// conformer.cpp: conv_2d_dw_f32 (ggml_conv_2d_dw with the kernel's dtype)
ggml_tensor *conv_2d_dw_f32(ggml_context *ctx, ggml_tensor *kernel, ggml_tensor *data, int s0,
                            int s1, int p0, int p1, int d0, int d1) {
  ggml_tensor *new_a =
      ggml_reshape_4d(ctx, kernel, kernel->ne[0], kernel->ne[1], 1, kernel->ne[2] * kernel->ne[3]);
  ggml_tensor *data_4d =
      ggml_reshape_4d(ctx, data, data->ne[0], data->ne[1], 1, data->ne[2] * data->ne[3]);
  ggml_tensor *im2col =
      ggml_im2col(ctx, new_a, data_4d, s0, s1, p0, p1, d0, d1, /*is_2D=*/true, kernel->type);
  ggml_tensor *new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1],
                                       data->ne[2], data->ne[3]);
  new_a = ggml_reshape_4d(ctx, new_a, new_a->ne[0] * new_a->ne[1], new_a->ne[2], new_a->ne[3], 1);
  ggml_tensor *result = ggml_mul_mat(ctx, new_a, new_b);
  result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], data->ne[2], data->ne[3]);
  return result;
}

// conformer.cpp: dw_kernel_for_direct (direct depthwise needs an F32 kernel)
ggml_tensor *dw_kernel_for_direct(ggml_context *ctx, ggml_tensor *kernel) {
  if (kernel == nullptr || kernel->type == GGML_TYPE_F32) {
    return kernel;
  }
  return ggml_cast(ctx, kernel, GGML_TYPE_F32);
}

// conformer.cpp: conv_1d_dw_f32 (single-utterance im2col path)
ggml_tensor *conv_1d_dw_f32(ggml_context *ctx, ggml_tensor *kernel, ggml_tensor *data, int stride,
                            int padding, int dilation) {
  const int64_t B = data->ne[2];
  if (B > 1) {
    const int64_t k = kernel->ne[0];
    const int64_t C = data->ne[1];
    ggml_tensor *knl = kernel;
    if (knl->type != GGML_TYPE_F32) {
      knl = ggml_cast(ctx, knl, GGML_TYPE_F32);
    }
    knl = ggml_reshape_4d(ctx, knl, k, 1, 1, C);
    ggml_tensor *d4 = ggml_reshape_4d(ctx, data, data->ne[0], 1, C, B);
    ggml_tensor *o = ggml_conv_2d_dw_direct(ctx, knl, d4, stride, 1, padding, 0, dilation, 1);
    return ggml_reshape_3d(ctx, o, o->ne[0], C, B);
  }
  ggml_tensor *data_4d = ggml_reshape_4d(ctx, data, data->ne[0], 1, data->ne[1], data->ne[2]);
  ggml_tensor *im2col = ggml_im2col(ctx, kernel, data_4d, stride, 0, padding, 0, dilation, 0,
                                    /*is_2D=*/false, kernel->type);
  ggml_tensor *result = ggml_mul_mat(ctx, im2col, kernel);
  result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);
  return result;
}

// conformer.cpp: fused_batch_norm
ggml_tensor *fused_batch_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale_1d,
                              ggml_tensor *bias_1d) {
  const int64_t C = scale_1d->ne[0];
  ggml_tensor *scale_4d = ggml_reshape_4d(ctx, scale_1d, 1, C, 1, 1);
  ggml_tensor *bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, C, 1, 1);
  ggml_tensor *y = ggml_mul(ctx, x, scale_4d);
  y = ggml_add(ctx, y, bias_4d);
  return y;
}

// conformer.cpp: add_conv_bias / add_conv_bias_inplace
ggml_tensor *add_conv_bias(ggml_context *ctx, ggml_tensor *conv_out, ggml_tensor *bias_1d) {
  if (bias_1d == nullptr) {
    return conv_out;
  }
  ggml_tensor *bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, 1, bias_1d->ne[0], 1);
  return ggml_add(ctx, conv_out, bias_4d);
}

ggml_tensor *add_conv_bias_inplace(ggml_context *ctx, ggml_tensor *conv_out, ggml_tensor *bias_1d) {
  if (bias_1d == nullptr) {
    return conv_out;
  }
  ggml_tensor *bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, 1, bias_1d->ne[0], 1);
  return ggml_add_inplace(ctx, conv_out, bias_4d);
}

// conformer.cpp: conv_module, symmetric-padding / BatchNorm / offline path.
ggml_tensor *conv_module(ggml_context *ctx, ggml_tensor *x, const EncBlockWeights &b,
                         const BlockParams &params) {
  const int conv_kernel = params.conv_kernel;
  const ConvPolicy &policy = params.policy;
  const int pad = (conv_kernel - 1) / 2;
  const int64_t d_model = x->ne[0];
  const int64_t B = x->ne[2];

  if (policy.direct_pw) {
    {
      ggml_tensor *pw1 = ggml_reshape_2d(ctx, b.conv_pw1_w, d_model, 2 * d_model);
      x = ggml_mul_mat(ctx, pw1, x);
      if (b.conv_pw1_w->type == GGML_TYPE_F16) {
        ggml_mul_mat_set_prec(x, GGML_PREC_F32);
      }
      if (b.conv_pw1_b != nullptr) {
        x = ggml_add(ctx, x, b.conv_pw1_b);
      }
    }
    {
      const int64_t T = x->ne[1];
      const int64_t half = x->ne[0] / 2;
      ggml_tensor *gate = ggml_view_3d(ctx, x, half, T, B, x->nb[1], x->nb[2], 0);
      ggml_tensor *value =
          ggml_view_3d(ctx, x, half, T, B, x->nb[1], x->nb[2], half * ggml_element_size(x));
      x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
    }
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
  } else {
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    x = conv_1d_f32(ctx, b.conv_pw1_w, x, 1, 0, 1);
    if (b.conv_pw1_b != nullptr) {
      x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
      ggml_tensor *bias_r = ggml_reshape_2d(ctx, b.conv_pw1_b, 1, 2 * d_model);
      x = ggml_add(ctx, x, bias_r);
    }
    const int64_t half = x->ne[1] / 2;
    ggml_tensor *gate = ggml_view_4d(ctx, x, x->ne[0], half, 1, 1, x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor *value =
        ggml_view_4d(ctx, x, x->ne[0], half, 1, 1, x->nb[1], x->nb[2], x->nb[3], x->nb[1] * half);
    x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
    x = ggml_cont(ctx, x);
  }

  if (policy.direct_dw_in_block) {
    ggml_tensor *knl =
        ggml_reshape_4d(ctx, dw_kernel_for_direct(ctx, b.conv_dw_w), conv_kernel, 1, 1, d_model);
    ggml_tensor *data = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], B);
    x = ggml_conv_2d_dw_direct(ctx, knl, data, 1, 1, pad, 0, 1, 1);
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[2], x->ne[3]);
  } else {
    x = conv_1d_dw_f32(ctx, b.conv_dw_w, x, 1, pad, 1);
  }

  if (b.conv_dw_b != nullptr) {
    ggml_tensor *bias_r = ggml_reshape_2d(ctx, b.conv_dw_b, 1, d_model);
    x = ggml_add(ctx, x, bias_r);
  }

  x = fused_batch_norm(ctx, x, b.conv_bn_fused_scale, b.conv_bn_fused_bias);
  x = ggml_silu(ctx, x);

  if (policy.direct_pw) {
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor *pw2 = ggml_reshape_2d(ctx, b.conv_pw2_w, d_model, d_model);
    x = ggml_mul_mat(ctx, pw2, x);
    if (b.conv_pw2_w->type == GGML_TYPE_F16) {
      ggml_mul_mat_set_prec(x, GGML_PREC_F32);
    }
    if (b.conv_pw2_b != nullptr) {
      x = ggml_add(ctx, x, b.conv_pw2_b);
    }
  } else {
    x = conv_1d_f32(ctx, b.conv_pw2_w, x, 1, 0, 1);
    if (b.conv_pw2_b != nullptr) {
      x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
      ggml_tensor *bias_r = ggml_reshape_2d(ctx, b.conv_pw2_b, 1, d_model);
      x = ggml_add(ctx, x, bias_r);
    }
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
  }
  return x;
}

// conformer.cpp: rel_pos_mhsa, square self-attention without window or masks.
ggml_tensor *rel_pos_mhsa(ggml_context *ctx, ggml_tensor *x, ggml_tensor *pos_emb,
                          const EncBlockWeights &b, const BlockParams &params) {
  const int d_model = params.d_model;
  const int n_head = params.n_head;
  const ggml_type kv_type = params.kv_type;
  const int head_dim = d_model / n_head;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t T_kv = x->ne[1];
  const int64_t T_q = x->ne[1];
  const int64_t pos_len = pos_emb->ne[1];
  const int64_t B = x->ne[2];
  const bool flash = params.use_flash;

  ggml_tensor *q = ggml_mul_mat(ctx, b.attn_q_w, x);
  if (b.attn_q_b != nullptr) {
    q = ggml_add(ctx, q, b.attn_q_b);
  }
  ggml_tensor *k = ggml_mul_mat(ctx, b.attn_k_w, x);
  if (b.attn_k_b != nullptr) {
    k = ggml_add(ctx, k, b.attn_k_b);
  }
  ggml_tensor *v = ggml_mul_mat(ctx, b.attn_v_w, x);
  if (b.attn_v_b != nullptr) {
    v = ggml_add(ctx, v, b.attn_v_b);
  }
  ggml_tensor *p = ggml_mul_mat(ctx, b.attn_pos_w, pos_emb);

  q = ggml_reshape_4d(ctx, q, head_dim, n_head, T_q, B);
  ggml_tensor *q_u = ggml_add(ctx, q, b.attn_pos_u);
  ggml_tensor *q_v = ggml_add(ctx, q, b.attn_pos_v);

  q_u = ggml_permute(ctx, q_u, 0, 2, 1, 3);
  q_v = ggml_cont(ctx, ggml_permute(ctx, q_v, 0, 2, 1, 3));

  k = ggml_reshape_4d(ctx, k, head_dim, n_head, T_kv, B);
  k = ggml_permute(ctx, k, 0, 2, 1, 3);

  v = ggml_reshape_4d(ctx, v, head_dim, n_head, T_kv, B);
  v = ggml_permute(ctx, v, 0, 2, 1, 3);

  p = ggml_reshape_4d(ctx, p, head_dim, n_head, pos_len, 1);
  p = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));

  auto shifted_view = [&](ggml_tensor *scores, int64_t heads) {
    const size_t query_stride =
        T_q == 1 ? static_cast<size_t>(T_kv) * scores->nb[0] : scores->nb[1] - scores->nb[0];
    return ggml_view_4d(ctx, scores, T_kv, T_q, heads, B, query_stride, scores->nb[2],
                        scores->nb[3], (T_q - 1) * scores->nb[0]);
  };

  // Full attention + flash: per-head position scores (split_flash_mask).
  const bool split_flash_mask = flash;
  ggml_tensor *matrix_bd = nullptr;
  if (split_flash_mask) {
    for (int h = 0; h < n_head; ++h) {
      ggml_tensor *p_h = ggml_view_4d(ctx, p, head_dim, pos_len, 1, 1, p->nb[1], p->nb[2], p->nb[3],
                                      static_cast<size_t>(h) * p->nb[2]);
      ggml_tensor *q_v_h = ggml_view_4d(ctx, q_v, head_dim, T_q, 1, B, q_v->nb[1], q_v->nb[2],
                                        q_v->nb[3], static_cast<size_t>(h) * q_v->nb[2]);
      ggml_tensor *head_mask = ggml_mul_mat(ctx, p_h, q_v_h);
      head_mask = shifted_view(head_mask, 1);
      head_mask = ggml_cont(ctx, head_mask);
      head_mask = ggml_scale(ctx, head_mask, scale);
      head_mask = ggml_cast(ctx, head_mask, GGML_TYPE_F16);
      matrix_bd = matrix_bd == nullptr ? head_mask : ggml_concat(ctx, matrix_bd, head_mask, 2);
    }
  } else {
    matrix_bd = ggml_mul_mat(ctx, p, q_v);
    matrix_bd = shifted_view(matrix_bd, n_head);
  }

  ggml_tensor *o;
  if (flash) {
    ggml_type effective_kv = kv_type;
    if (effective_kv == GGML_TYPE_COUNT) {
      effective_kv = (b.attn_k_w->type != GGML_TYPE_F32) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    }
    if (effective_kv != GGML_TYPE_F32) {
      k = ggml_cast(ctx, k, effective_kv);
      v = ggml_cast(ctx, v, effective_kv);
    }
    o = ggml_flash_attn_ext(ctx, q_u, k, v, matrix_bd, scale, 0.0f, 0.0f);
  } else {
    ggml_tensor *kq = ggml_mul_mat(ctx, k, q_u);
    kq = ggml_add(ctx, kq, matrix_bd);
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont(ctx, o);
  }
  o = ggml_reshape_3d(ctx, o, d_model, T_q, B);

  o = ggml_mul_mat(ctx, b.attn_out_w, o);
  if (b.attn_out_b != nullptr) {
    o = ggml_add(ctx, o, b.attn_out_b);
  }
  return o;
}

// conformer.cpp: regular_dw_2d_time_chunked (bounded im2col depthwise).
ggml_tensor *regular_dw_2d_time_chunked(ggml_context *ctx, ggml_tensor *kernel, ggml_tensor *data,
                                        int64_t time_chunk) {
  constexpr int64_t kStride = 2;
  const int64_t extent = kernel->ne[1];
  const int64_t pad = (extent - 1) / 2;
  const int64_t time_out = (data->ne[1] + 2 * pad - extent) / kStride + 1;
  if (time_out <= time_chunk) {
    return conv_2d_dw_f32(ctx, kernel, data, static_cast<int>(kStride), static_cast<int>(kStride),
                          static_cast<int>(pad), static_cast<int>(pad), 1, 1);
  }

  std::vector<ggml_tensor *> chunks;
  chunks.reserve(static_cast<size_t>((time_out + time_chunk - 1) / time_chunk));
  for (int64_t out0 = 0; out0 < time_out; out0 += time_chunk) {
    const int64_t n_out = std::min<int64_t>(time_chunk, time_out - out0);
    const int64_t src0 = out0 * kStride - pad;
    const int64_t src1 = src0 + (n_out - 1) * kStride + extent;
    const int64_t view0 = std::max<int64_t>(0, src0);
    const int64_t view1 = std::min<int64_t>(data->ne[1], src1);
    const int64_t pad_top = view0 - src0;
    const int64_t pad_bot = src1 - view1;
    ggml_tensor *input = ggml_view_4d(ctx, data, data->ne[0], view1 - view0, data->ne[2],
                                      data->ne[3], data->nb[1], data->nb[2], data->nb[3],
                                      view0 * data->nb[1]);
    input = ggml_pad_ext(ctx, input, 0, 0, static_cast<int>(pad_top), static_cast<int>(pad_bot), 0,
                         0, 0, 0);
    chunks.push_back(conv_2d_dw_f32(ctx, kernel, input, static_cast<int>(kStride),
                                    static_cast<int>(kStride), static_cast<int>(pad), 0, 1, 1));
  }

  while (chunks.size() > 1) {
    std::vector<ggml_tensor *> next;
    next.reserve((chunks.size() + 1) / 2);
    for (size_t i = 0; i < chunks.size(); i += 2) {
      next.push_back(i + 1 < chunks.size() ? ggml_concat(ctx, chunks[i], chunks[i + 1], 1)
                                           : chunks[i]);
    }
    chunks.swap(next);
  }
  return chunks.front();
}

} // namespace

bool env_flag(const char *name) {
  // transcribe::env::flag: set, non-empty, and not starting with '0'.
  const char *v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool resolve_conv_direct(const char *direct_env, const char *no_direct_env, bool backend_default) {
  if (env_flag(direct_env)) {
    return true;
  }
  if (env_flag(no_direct_env)) {
    return false;
  }
  return backend_default;
}

bool detect_direct_pw(const char *backend_name) {
  const bool backend_default =
      !(backend_name != nullptr && std::strstr(backend_name, "Vulkan") != nullptr);
  return resolve_conv_direct("TRANSCRIBE_CONV_DIRECT_PW", "TRANSCRIBE_CONV_NO_DIRECT_PW",
                             backend_default);
}

// conformer.cpp: build_conformer_block, offline path.
ggml_tensor *build_conformer_block(ggml_context *ctx, ggml_tensor *x, ggml_tensor *pos_emb,
                                   const EncBlockWeights &b, const BlockParams &params) {
  x = macaron_ff_residual(ctx, x, b.norm_ff1_w, b.norm_ff1_b, b.ff1_lin1_w, b.ff1_lin1_b,
                          b.ff1_lin2_w, b.ff1_lin2_b);
  {
    ggml_tensor *x_norm = layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    ggml_tensor *attn_out = rel_pos_mhsa(ctx, x_norm, pos_emb, b, params);
    x = ggml_add(ctx, x, attn_out);
  }
  {
    ggml_tensor *x_norm = layer_norm(ctx, x, b.norm_conv_w, b.norm_conv_b);
    ggml_tensor *conv_out = conv_module(ctx, x_norm, b, params);
    x = ggml_add(ctx, x, conv_out);
  }
  x = macaron_ff_residual(ctx, x, b.norm_ff2_w, b.norm_ff2_b, b.ff2_lin1_w, b.ff2_lin1_b,
                          b.ff2_lin2_w, b.ff2_lin2_b);
  x = layer_norm(ctx, x, b.norm_out_w, b.norm_out_b);
  return x;
}

// conformer.cpp: build_pre_encode, non-causal, no valid-frame masks.
ggml_tensor *build_pre_encode(ggml_context *ctx, const PreEncodeWeights &pe, ggml_tensor *mel_in,
                              const ConvPolicy &policy) {
  // [T_mel, n_mels] -> [n_mels, T_mel] = [W=F, H=T, IC, N].
  ggml_tensor *x = ggml_permute(ctx, mel_in, 1, 0, 2, 3);
  x = ggml_cont(ctx, x);

  const bool inplace_pe = policy.inplace_pre_encode;
  const int pe_p_op = 1;
  auto add_pre_encode_bias = [ctx, inplace_pe](ggml_tensor *value, ggml_tensor *bias) {
    return inplace_pe ? add_conv_bias_inplace(ctx, value, bias) : add_conv_bias(ctx, value, bias);
  };
  auto pre_encode_relu = [ctx, inplace_pe](ggml_tensor *value) {
    return inplace_pe ? ggml_relu_inplace(ctx, value) : ggml_relu(ctx, value);
  };

  // conv0 (1 -> C, k=3 s=2 p=1)
  if (policy.direct_conv0_in_pre_encode) {
    x = ggml_conv_2d_direct(ctx, pe.conv0_w, x, 2, 2, pe_p_op, pe_p_op, 1, 1);
  } else {
    x = ggml_conv_2d(ctx, pe.conv0_w, x, 2, 2, pe_p_op, pe_p_op, 1, 1);
  }
  x = add_pre_encode_bias(x, pe.conv0_b);
  x = pre_encode_relu(x);

  // conv2 (depthwise, k=3 s=2)
  if (policy.direct_dw_in_pre_encode) {
    x = ggml_conv_2d_dw_direct(ctx, dw_kernel_for_direct(ctx, pe.conv2_w), x, 2, 2, pe_p_op,
                               pe_p_op, 1, 1);
  } else if (policy.pre_encode_dw_time_chunk > 0) {
    x = regular_dw_2d_time_chunked(ctx, pe.conv2_w, x, policy.pre_encode_dw_time_chunk);
  } else {
    x = conv_2d_dw_f32(ctx, pe.conv2_w, x, 2, 2, pe_p_op, pe_p_op, 1, 1);
  }
  x = add_pre_encode_bias(x, pe.conv2_b);

  // conv3 (pointwise)
  x = ggml_conv_2d(ctx, pe.conv3_w, x, 1, 1, 0, 0, 1, 1);
  x = add_pre_encode_bias(x, pe.conv3_b);
  x = pre_encode_relu(x);

  // conv5 (depthwise) -> conv6 (pointwise) -> ReLU
  if (policy.direct_dw_in_pre_encode) {
    x = ggml_conv_2d_dw_direct(ctx, dw_kernel_for_direct(ctx, pe.conv5_w), x, 2, 2, pe_p_op,
                               pe_p_op, 1, 1);
  } else if (policy.pre_encode_dw_time_chunk > 0) {
    x = regular_dw_2d_time_chunked(ctx, pe.conv5_w, x, policy.pre_encode_dw_time_chunk);
  } else {
    x = conv_2d_dw_f32(ctx, pe.conv5_w, x, 2, 2, pe_p_op, pe_p_op, 1, 1);
  }
  x = add_pre_encode_bias(x, pe.conv5_b);

  x = ggml_conv_2d(ctx, pe.conv6_w, x, 1, 1, 0, 0, 1, 1);
  x = add_pre_encode_bias(x, pe.conv6_b);
  x = pre_encode_relu(x);

  // [F', T_enc, C, B] -> [F', C, T_enc, B] -> [F'*C, T_enc, B]
  const int64_t F_prime = x->ne[0];
  const int64_t T_enc = x->ne[1];
  const int64_t C = x->ne[2];
  const int64_t B = x->ne[3];
  const int64_t pre_encode_in = F_prime * C;
  if (pe.out_w != nullptr && pre_encode_in != pe.out_w->ne[0]) {
    std::fprintf(stderr,
                 "canary_qwen encoder: pre_encode_in mismatch: F'*C=%lld but out_w expects %lld\n",
                 static_cast<long long>(pre_encode_in), static_cast<long long>(pe.out_w->ne[0]));
    return nullptr;
  }

  x = ggml_permute(ctx, x, 0, 2, 1, 3);
  x = ggml_cont(ctx, x);
  x = ggml_reshape_3d(ctx, x, pre_encode_in, T_enc, B);

  x = ggml_mul_mat(ctx, pe.out_w, x);
  if (pe.out_b != nullptr) {
    const int64_t d_model = pe.out_b->ne[0];
    ggml_tensor *bias_4d = ggml_reshape_4d(ctx, pe.out_b, d_model, 1, 1, 1);
    x = ggml_add(ctx, x, bias_4d);
  }
  return x;
}

} // namespace engine::models::canary_qwen::conformer
