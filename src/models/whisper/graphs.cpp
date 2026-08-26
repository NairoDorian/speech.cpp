// engine/models/whisper/graphs.cpp - ggml graph builders for the native
// engine Whisper ASR package.
//
// Ported from src/runtime/arch/whisper/{encoder,decoder}.cpp with identical
// graph topology and numerics. The transcribe-side tensor-dump plumbing is not
// carried over (engine tracing covers observability), and the two conformer
// helpers the arch borrowed (conv_1d_f32 / layer_norm) are reimplemented
// locally so the package has no dependency on src/runtime/.
//
// W2a simplifications, both backend-shape choices rather than numerics:
//   - The cross-KV cache is allocated at exactly T_enc rows per layer (the
//     arch pads to GGML_PAD(T_enc, 256) for Metal's FA block size). With no
//     padding there are no zero slots to dilute the softmax, so the cross
//     mask the arch needs is unnecessary here.
//   - Self-attention KV is likewise unpadded (kv_pad = 1), so a mask is built
//     only for the prompt pass (n_tokens > 1), exactly as the arch does when
//     kv_pad == 1.
// Metal support would restore both paddings; nothing else changes.
//
// Whisper attention quirk, carried over verbatim: q / v / out carry a bias,
// k does NOT - self-attention and cross-attention alike.

#include "engine/models/whisper/graphs_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace engine::models::whisper {

namespace {

constexpr float kLayerNormEps = 1e-5f;

ggml_tensor *named(ggml_tensor *t, const char *name) {
  if (t != nullptr) {
    ggml_set_name(t, name);
  }
  return t;
}

// Whisper LayerNorms carry a bias, unlike the moonshine families.
ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *gamma,
                        ggml_tensor *beta) {
  ggml_tensor *y = ggml_norm(ctx, x, kLayerNormEps);
  y = ggml_mul(ctx, y, gamma);
  if (beta != nullptr) {
    y = ggml_add(ctx, y, beta);
  }
  return y;
}

// im2col + mul_mat Conv1d on [T, C] layout data. Whisper's conv kernels are
// F32 and ggml_conv_1d's im2col hard-codes an F16 dst whose CPU kernel asserts
// src0 == F16, so the arch uses this helper instead - carried over for the
// same reason.
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

// Reshape a 1D conv bias [Cout] to [1, Cout, 1, 1] so it broadcasts across T
// against a conv_1d output [T_out, Cout, 1, 1].
ggml_tensor *add_conv1d_bias(ggml_context *ctx, ggml_tensor *conv_out,
                             ggml_tensor *bias_1d) {
  if (bias_1d == nullptr) {
    return conv_out;
  }
  const int64_t channels = bias_1d->ne[0];
  ggml_tensor *bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, channels, 1, 1);
  return ggml_add(ctx, conv_out, bias_4d);
}

// FFN: fc2(GELU_erf(fc1(x))); pre-LN wrapped outside. Both carry bias.
ggml_tensor *ffn(ggml_context *ctx, ggml_tensor *x, ggml_tensor *fc1_w,
                 ggml_tensor *fc1_b, ggml_tensor *fc2_w, ggml_tensor *fc2_b) {
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

// ---------------------------------------------------------------------------
// Encoder attention / blocks
// ---------------------------------------------------------------------------

// Bidirectional MHSA, no relative position, no causal mask.
ggml_tensor *mha_encoder(ggml_context *ctx, ggml_tensor *x, ggml_tensor *q_w,
                         ggml_tensor *q_b, ggml_tensor *k_w, ggml_tensor *v_w,
                         ggml_tensor *v_b, ggml_tensor *out_w,
                         ggml_tensor *out_b, int n_heads, int d_model,
                         bool use_flash) {
  const int head_dim = d_model / n_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t T = x->ne[1];

  ggml_tensor *q = ggml_mul_mat(ctx, q_w, x);
  if (q_b != nullptr) {
    q = ggml_add(ctx, q, q_b);
  }
  ggml_tensor *k = ggml_mul_mat(ctx, k_w, x); // k has NO bias
  ggml_tensor *v = ggml_mul_mat(ctx, v_w, x);
  if (v_b != nullptr) {
    v = ggml_add(ctx, v, v_b);
  }

  q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, n_heads, T), 0, 2, 1,
                   3);
  k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, n_heads, T), 0, 2, 1,
                   3);
  v = ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, n_heads, T), 0, 2, 1,
                   3);

  ggml_tensor *o;
  if (use_flash) {
    ggml_tensor *q_c = ggml_cont(ctx, q);
    ggml_tensor *k_c = ggml_cont(ctx, k);
    ggml_tensor *v_c = ggml_cont(ctx, v);
    o = ggml_flash_attn_ext(ctx, q_c, k_c, v_c, nullptr, scale, 0.0f, 0.0f);
    o = ggml_reshape_2d(ctx, o, d_model, T);
  } else {
    ggml_tensor *kq =
        ggml_mul_mat(ctx, ggml_cont(ctx, k), ggml_cont(ctx, q));
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont(ctx, o);
    o = ggml_reshape_2d(ctx, o, d_model, T);
  }

  o = ggml_mul_mat(ctx, out_w, o);
  if (out_b != nullptr) {
    o = ggml_add(ctx, o, out_b);
  }
  return o;
}

ggml_tensor *build_enc_block(ggml_context *ctx, ggml_tensor *x,
                             const WhisperEncBlock &b, int n_heads, int d_model,
                             bool use_flash) {
  {
    ggml_tensor *y = layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    y = mha_encoder(ctx, y, b.attn_q_w, b.attn_q_b, b.attn_k_w, b.attn_v_w,
                    b.attn_v_b, b.attn_out_w, b.attn_out_b, n_heads, d_model,
                    use_flash);
    x = ggml_add(ctx, x, y);
  }
  {
    ggml_tensor *y = layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
    y = ffn(ctx, y, b.ffn_fc1_w, b.ffn_fc1_b, b.ffn_fc2_w, b.ffn_fc2_b);
    x = ggml_add(ctx, x, y);
  }
  return x;
}

// ---------------------------------------------------------------------------
// Decoder attention (KV-cached)
// ---------------------------------------------------------------------------

ggml_tensor *mha_self_cached(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                             WhisperKvCache &kv_cache, ggml_tensor *mask,
                             ggml_tensor *q_w, ggml_tensor *q_b,
                             ggml_tensor *k_w, ggml_tensor *v_w,
                             ggml_tensor *v_b, ggml_tensor *out_w,
                             ggml_tensor *out_b, int n_heads, int d_model,
                             int il, int n_past, int n_tokens, int n_kv,
                             bool use_flash) {
  const int head_dim = d_model / n_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int n_ctx = kv_cache.n_ctx;

  ggml_tensor *Qcur = ggml_mul_mat(ctx, q_w, x);
  if (q_b != nullptr) {
    Qcur = ggml_add(ctx, Qcur, q_b);
  }
  ggml_tensor *Kcur = ggml_mul_mat(ctx, k_w, x); // NO bias
  ggml_tensor *Vcur = ggml_mul_mat(ctx, v_w, x);
  if (v_b != nullptr) {
    Vcur = ggml_add(ctx, Vcur, v_b);
  }

  ggml_tensor *Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
  Q = ggml_permute(ctx, Q, 0, 2, 1, 3);

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

    ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k_dst));
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

  ggml_tensor *o;
  if (use_flash) {
    o = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
    o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
  } else {
    ggml_tensor *kq = ggml_mul_mat(ctx, K, Q);
    ggml_tensor *kq_soft = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont(ctx, o);
    o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
  }

  o = ggml_mul_mat(ctx, out_w, o);
  if (out_b != nullptr) {
    o = ggml_add(ctx, o, out_b);
  }
  return o;
}

ggml_tensor *mha_cross_cached(ggml_context *ctx, ggml_tensor *x,
                              WhisperKvCache &kv_cache, ggml_tensor *q_w,
                              ggml_tensor *q_b, ggml_tensor *out_w,
                              ggml_tensor *out_b, int n_heads, int d_model,
                              int il, int T_enc, bool use_flash) {
  const int head_dim = d_model / n_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t n_tokens = x->ne[1];

  ggml_tensor *Qcur = ggml_mul_mat(ctx, q_w, x);
  if (q_b != nullptr) {
    Qcur = ggml_add(ctx, Qcur, q_b);
  }
  ggml_tensor *Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
  Q = ggml_permute(ctx, Q, 0, 2, 1, 3);

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

  ggml_tensor *o;
  if (use_flash) {
    // No cross mask: the cache is allocated at exactly T_enc rows, so there
    // are no padded slots to exclude.
    o = ggml_flash_attn_ext(ctx, Q, K, V, /*mask=*/nullptr, scale, 0.0f, 0.0f);
    o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
  } else {
    ggml_tensor *kq = ggml_mul_mat(ctx, K, Q);
    ggml_tensor *kq_soft =
        ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr, scale, 0.0f);
    ggml_tensor *v_t = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
    o = ggml_mul_mat(ctx, v_t, kq_soft);
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont(ctx, o);
    o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
  }

  o = ggml_mul_mat(ctx, out_w, o);
  if (out_b != nullptr) {
    o = ggml_add(ctx, o, out_b);
  }
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

void WhisperKvCache::free() {
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

bool kv_cache_init(WhisperKvCache &cache, ggml_backend_t backend, int n_ctx,
                   int T_enc, int d_model, int n_layer, ggml_type kv_type) {
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

// ---------------------------------------------------------------------------
// Encoder graph
// ---------------------------------------------------------------------------

EncoderBuild build_encoder_graph(ggml_context *ctx, const WhisperWeights &w,
                                 const WhisperHParams &hp, int n_mel_frames,
                                 bool use_flash) {
  EncoderBuild eb{};

  if (ctx == nullptr || n_mel_frames <= 0 || n_mel_frames % 2 != 0) {
    return eb;
  }

  const int d_model = hp.enc_d_model;
  const int n_mels = hp.enc_num_mel_bins;
  const int n_heads = hp.enc_n_heads;
  const int T_enc = n_mel_frames / 2;

  if (T_enc > hp.enc_max_source_positions) {
    return eb;
  }

  eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_mels, n_mel_frames);
  if (eb.mel_in == nullptr) {
    return eb;
  }
  named(eb.mel_in, "enc.mel.in");
  ggml_set_input(eb.mel_in);

  // conv stem: conv_1d wants [T, Cin]; mel arrives [n_mels, T].
  ggml_tensor *x = ggml_cont(ctx, ggml_transpose(ctx, eb.mel_in));

  // conv1: k=3 s=1 p=1 -> [T, d_model]; HF uses exact-erf GELU here.
  x = conv_1d_f32(ctx, w.enc_stem.conv0_w, x, 1, 1, 1);
  x = add_conv1d_bias(ctx, x, w.enc_stem.conv0_b);
  x = ggml_gelu_erf(ctx, x);
  x = ggml_cont(ctx, ggml_transpose(ctx, x)); // -> [d_model, T]

  // conv2: k=3 s=2 p=1 -> [T_enc, d_model].
  x = ggml_cont(ctx, ggml_transpose(ctx, x));
  x = conv_1d_f32(ctx, w.enc_stem.conv1_w, x, 2, 1, 1);
  x = add_conv1d_bias(ctx, x, w.enc_stem.conv1_b);
  x = ggml_gelu_erf(ctx, x);
  x = ggml_cont(ctx, ggml_transpose(ctx, x)); // -> [d_model, T_enc]

  // Learned absolute positional embedding; prefix view when T_enc < max.
  ggml_tensor *pos_emb = w.enc_top.pos_emb_w;
  if (T_enc != hp.enc_max_source_positions) {
    pos_emb = ggml_view_2d(ctx, w.enc_top.pos_emb_w, d_model, T_enc,
                           w.enc_top.pos_emb_w->nb[1], 0);
  }
  named(pos_emb, "enc.pos_emb");
  x = ggml_add(ctx, x, pos_emb);

  const int n_blocks = static_cast<int>(w.enc_blocks.size());
  for (int i = 0; i < n_blocks; ++i) {
    x = build_enc_block(ctx, x, w.enc_blocks[static_cast<size_t>(i)], n_heads,
                        d_model, use_flash);
  }

  x = layer_norm(ctx, x, w.enc_top.final_norm_w, w.enc_top.final_norm_b);
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
// Cross-KV precompute
// ---------------------------------------------------------------------------

CrossKvBuild build_cross_kv_graph(ggml_context *ctx, const WhisperWeights &w,
                                  const WhisperHParams &hp,
                                  WhisperKvCache &kv_cache, int T_enc) {
  CrossKvBuild cb{};

  if (ctx == nullptr || T_enc <= 0) {
    return cb;
  }

  const int d_model = hp.dec_d_model;

  // Unlike the arch (which views a backend-persistent encoder tensor), the
  // engine package re-uploads the encoder output from host: each graph lives
  // in its own compute context and cannot share tensor handles.
  cb.encoder_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_enc);
  named(cb.encoder_out_in, "dec.encoder_out");
  ggml_set_input(cb.encoder_out_in);

  cb.graph = ggml_new_graph_custom(ctx, 4096, false);
  if (cb.graph == nullptr) {
    return cb;
  }

  const int n_layers = static_cast<int>(w.dec_blocks.size());
  for (int il = 0; il < n_layers; ++il) {
    const auto &blk = w.dec_blocks[static_cast<size_t>(il)];

    // Cross k has NO bias; cross v does.
    ggml_tensor *Kcross = ggml_mul_mat(ctx, blk.cross_k_w, cb.encoder_out_in);
    ggml_tensor *Vcross = ggml_mul_mat(ctx, blk.cross_v_w, cb.encoder_out_in);
    if (blk.cross_v_b != nullptr) {
      Vcross = ggml_add(ctx, Vcross, blk.cross_v_b);
    }

    const size_t k_elem = ggml_element_size(kv_cache.cross_k);
    const size_t v_elem = ggml_element_size(kv_cache.cross_v);

    ggml_tensor *k_dst = ggml_view_1d(
        ctx, kv_cache.cross_k, static_cast<int64_t>(T_enc) * d_model,
        k_elem *
            static_cast<size_t>(static_cast<int64_t>(il) * T_enc * d_model));
    ggml_tensor *v_dst = ggml_view_1d(
        ctx, kv_cache.cross_v, static_cast<int64_t>(T_enc) * d_model,
        v_elem *
            static_cast<size_t>(static_cast<int64_t>(il) * T_enc * d_model));

    ggml_build_forward_expand(cb.graph, ggml_cpy(ctx, Kcross, k_dst));
    ggml_build_forward_expand(cb.graph, ggml_cpy(ctx, Vcross, v_dst));
  }

  return cb;
}

// ---------------------------------------------------------------------------
// Decoder graph (KV-cached prompt + step)
// ---------------------------------------------------------------------------

DecoderBuild build_decoder_graph_kv(ggml_context *ctx, const WhisperWeights &w,
                                    const WhisperHParams &hp,
                                    WhisperKvCache &kv_cache, int n_tokens,
                                    int n_past, int T_enc, bool use_flash) {
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

  ggml_tensor *pos_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  named(pos_ids_in, "dec.pos_ids");
  ggml_set_input(pos_ids_in);

  // Self-attention mask only for the prompt pass. With kv_pad == 1 the
  // single-token step attends the full real cache window and needs none.
  ggml_tensor *causal_mask = nullptr;
  if (n_tokens > 1) {
    db.causal_mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    named(db.causal_mask_in, "dec.causal_mask");
    ggml_set_input(db.causal_mask_in);
    causal_mask = ggml_cast(ctx, db.causal_mask_in, GGML_TYPE_F16);
  }

  ggml_tensor *tok_emb =
      ggml_get_rows(ctx, w.dec_top.token_embd_w, db.token_ids_in);
  named(tok_emb, "dec.token_emb");

  ggml_tensor *pos_emb = ggml_get_rows(ctx, w.dec_top.pos_emb_w, pos_ids_in);
  named(pos_emb, "dec.pos_emb");

  // Embed sum - NO post-embed LayerNorm in Whisper.
  ggml_tensor *x = ggml_add(ctx, tok_emb, pos_emb);
  named(x, "dec.embed_sum");

  db.graph = ggml_new_graph_custom(ctx, 8192, false);
  if (db.graph == nullptr) {
    return db;
  }

  const int n_blocks = static_cast<int>(w.dec_blocks.size());
  for (int i = 0; i < n_blocks; ++i) {
    const auto &b = w.dec_blocks[static_cast<size_t>(i)];

    {
      ggml_tensor *y = layer_norm(ctx, x, b.norm_self_w, b.norm_self_b);
      y = mha_self_cached(ctx, db.graph, y, kv_cache, causal_mask, b.self_q_w,
                          b.self_q_b, b.self_k_w, b.self_v_w, b.self_v_b,
                          b.self_out_w, b.self_out_b, n_heads, d_model, i,
                          n_past, n_tokens, n_kv, use_flash);
      x = ggml_add(ctx, x, y);
    }
    {
      ggml_tensor *y = layer_norm(ctx, x, b.norm_cross_w, b.norm_cross_b);
      y = mha_cross_cached(ctx, y, kv_cache, b.cross_q_w, b.cross_q_b,
                           b.cross_out_w, b.cross_out_b, n_heads, d_model, i,
                           T_enc, use_flash);
      x = ggml_add(ctx, x, y);
    }
    {
      ggml_tensor *y = layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
      y = ffn(ctx, y, b.ffn_fc1_w, b.ffn_fc1_b, b.ffn_fc2_w, b.ffn_fc2_b);
      x = ggml_add(ctx, x, y);
    }
  }

  x = layer_norm(ctx, x, w.dec_top.final_norm_w, w.dec_top.final_norm_b);
  named(x, "dec.out_before_head");

  // Logits head: TIED to the token embedding, no bias. Raw logits - the host
  // loop applies Whisper's suppress masks before argmax, so unlike the
  // moonshine families there is no in-graph argmax shortcut.
  ggml_tensor *logits_raw = ggml_mul_mat(ctx, w.dec_top.token_embd_w, x);
  named(logits_raw, "dec.logits_raw");

  db.logits_out = logits_raw;
  ggml_set_output(db.logits_out);
  ggml_build_forward_expand(db.graph, db.logits_out);

  return db;
}

} // namespace engine::models::whisper
