// engine/models/canary_qwen/graphs.cpp - encoder / prefill / step graph
// builders, ported from src/runtime/arch/canary_qwen/{encoder,decoder}.cpp.
// The debug-dump side branches (enc.final permute, named intermediates) are
// dropped: they never fed the output and do not change a value.

#include "engine/models/canary_qwen/graphs.h"

#include "engine/models/canary_qwen/assets.h"

#include <cmath>
#include <cstdio>

namespace engine::models::canary_qwen {

namespace {

causal_lm::BlockParams decoder_block_params(const CanaryQwenHParams &hp) {
  causal_lm::BlockParams p;
  p.n_heads = hp.dec_n_heads;
  p.n_kv_heads = hp.dec_n_kv_heads;
  p.head_dim = hp.dec_head_dim;
  p.max_position = hp.dec_max_position;
  p.rms_eps = hp.dec_rms_norm_eps;
  p.rope_theta = hp.dec_rope_theta;
  return p;
}

} // namespace

// ---------------------------------------------------------------------------
// Encoder (arch encoder.cpp: build_encoder_graph)
// ---------------------------------------------------------------------------

EncoderBuild build_encoder_graph(ggml_context *ctx, const CanaryQwenWeights &w,
                                 const CanaryQwenHParams &hp, int n_mel_frames,
                                 const EncoderOptions &options) {
  conformer::ConvPolicy policy;
  policy.direct_pw = conformer::detect_direct_pw(options.backend_name);
  policy.direct_dw_in_block = conformer::resolve_conv_direct(
      "TRANSCRIBE_CONV_DIRECT_DW", "TRANSCRIBE_CONV_NO_DIRECT_DW", /*backend_default=*/true);
  policy.direct_dw_in_pre_encode = false;
  policy.inplace_pre_encode = true;
  policy.pre_encode_dw_time_chunk = 256;

  EncoderBuild eb;
  if (ctx == nullptr || n_mel_frames <= 0) {
    return eb;
  }
  if (hp.enc_subsampling_factor != 8 || hp.fe_num_mels != 128) {
    return eb;
  }

  // Mel input: ne = [n_frames, n_mels], i.e. mel-major host data.
  eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_mel_frames, hp.fe_num_mels);
  ggml_set_name(eb.mel_in, "mel.in");
  ggml_set_input(eb.mel_in);

  ggml_tensor *x = conformer::build_pre_encode(ctx, w.pre_encode, eb.mel_in, policy);
  if (x == nullptr) {
    return eb;
  }
  eb.T_enc = static_cast<int>(x->ne[1]);

  if (!w.blocks.empty()) {
    const int64_t T_enc = x->ne[1];
    const int64_t pos_len = 2 * T_enc - 1;
    eb.pos_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.enc_d_model, pos_len);
    ggml_set_name(eb.pos_emb_in, "pos_emb.in");
    ggml_set_input(eb.pos_emb_in);

    conformer::BlockParams bparams;
    bparams.d_model = hp.enc_d_model;
    bparams.n_head = hp.enc_n_heads;
    bparams.conv_kernel = hp.enc_conv_kernel;
    bparams.kv_type = GGML_TYPE_COUNT;
    bparams.use_flash = options.use_flash;
    bparams.policy = policy;

    for (const auto &block : w.blocks) {
      x = conformer::build_conformer_block(ctx, x, eb.pos_emb_in, block, bparams);
    }
  }

  // Perception projection: Linear(d_enc, out_dim) + bias.
  ggml_tensor *proj = ggml_mul_mat(ctx, w.proj_w, x);
  if (w.proj_b != nullptr) {
    proj = ggml_add(ctx, proj, w.proj_b);
  }
  eb.out = proj;
  ggml_set_name(eb.out, "perception.proj.out");
  ggml_set_output(eb.out);

  eb.graph = ggml_new_graph_custom(ctx, 16384, false);
  if (eb.graph == nullptr) {
    eb.out = nullptr;
    return eb;
  }
  ggml_build_forward_expand(eb.graph, eb.out);
  return eb;
}

// arch model.cpp: build_relpos_emb_host
void build_relpos_emb_host(std::vector<float> &pos_buf, int d_model, int T_enc) {
  const int pos_len = 2 * T_enc - 1;
  pos_buf.assign(static_cast<size_t>(pos_len) * d_model, 0.0f);
  std::vector<float> div_term(static_cast<size_t>(d_model / 2));
  const float ln_10000 = std::log(10000.0f);
  for (int k = 0; k < d_model / 2; ++k) {
    div_term[static_cast<size_t>(k)] =
        std::exp(static_cast<float>(2 * k) * (-ln_10000 / static_cast<float>(d_model)));
  }
  for (int i = 0; i < pos_len; ++i) {
    const float pos = static_cast<float>((T_enc - 1) - i);
    float *row = pos_buf.data() + static_cast<size_t>(i) * d_model;
    for (int k = 0; k < d_model / 2; ++k) {
      const float div = div_term[static_cast<size_t>(k)];
      row[2 * k] = std::sin(pos * div);
      row[2 * k + 1] = std::cos(pos * div);
    }
  }
}

// ---------------------------------------------------------------------------
// Decoder (arch decoder.cpp)
// ---------------------------------------------------------------------------

PrefillBuild build_prefill_graph(ggml_context *ctx, const CanaryQwenWeights &w,
                                 const CanaryQwenHParams &hp, causal_lm::KvCache &kv,
                                 int T_prompt, int T_audio, int prefix_len, int suffix_len,
                                 bool use_flash, bool slice_last) {
  PrefillBuild pb;
  if (ctx == nullptr || T_prompt <= 0 || T_audio < 0 || prefix_len < 0 || suffix_len < 0 ||
      prefix_len + T_audio + suffix_len != T_prompt || kv.self_k == nullptr ||
      kv.self_v == nullptr || T_prompt > kv.n_ctx) {
    return pb;
  }

  const int64_t hidden = hp.dec_hidden;
  const int64_t vocab = hp.dec_vocab_size;
  const int n_layer = hp.dec_n_layers;
  const float rms_eps = hp.dec_rms_norm_eps;
  const auto block_params = decoder_block_params(hp);

  pb.input_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt);
  ggml_set_name(pb.input_ids_in, "dec.input_ids");
  ggml_set_input(pb.input_ids_in);

  if (T_audio > 0) {
    pb.audio_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, T_audio);
    ggml_set_name(pb.audio_in, "dec.audio_in");
    ggml_set_input(pb.audio_in);
  }

  pb.positions_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt);
  ggml_set_name(pb.positions_in, "dec.positions");
  ggml_set_input(pb.positions_in);

  pb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T_prompt, T_prompt);
  ggml_set_name(pb.mask_in, "dec.attn_mask");
  ggml_set_input(pb.mask_in);

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16384, false);
  if (gf == nullptr) {
    return pb;
  }

  // Token embedding for ALL prompt positions; the locator rows are replaced
  // by the audio rows through the 3-way concat (NeMo SALM
  // replace_placeholders_and_build_targets for B=1).
  ggml_tensor *token_emb_all = ggml_get_rows(ctx, w.token_embd, pb.input_ids_in);

  const size_t emb_elem = ggml_element_size(token_emb_all);
  ggml_tensor *x_prefix = nullptr;
  if (prefix_len > 0) {
    x_prefix = ggml_view_2d(ctx, token_emb_all, hidden, prefix_len, emb_elem * hidden, 0);
    x_prefix = ggml_cont(ctx, x_prefix);
  }
  ggml_tensor *x_suffix = nullptr;
  if (suffix_len > 0) {
    x_suffix = ggml_view_2d(ctx, token_emb_all, hidden, suffix_len, emb_elem * hidden,
                            emb_elem * hidden * static_cast<size_t>(prefix_len + T_audio));
    x_suffix = ggml_cont(ctx, x_suffix);
  }

  ggml_tensor *x;
  if (T_audio > 0) {
    x = pb.audio_in;
    if (x_prefix != nullptr) {
      x = ggml_concat(ctx, x_prefix, x, 1);
    }
    if (x_suffix != nullptr) {
      x = ggml_concat(ctx, x, x_suffix, 1);
    }
  } else {
    x = token_emb_all;
  }

  for (int il = 0; il < n_layer; ++il) {
    x = causal_lm::block_prefill(ctx, gf, x, w.dec_blocks[static_cast<size_t>(il)], block_params,
                                 kv, il, T_prompt, pb.mask_in, pb.positions_in, use_flash,
                                 slice_last && (il == n_layer - 1));
  }

  x = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps), w.output_norm);

  ggml_tensor *last_x;
  if (slice_last) {
    last_x = x;
  } else {
    last_x = ggml_view_2d(ctx, x, hidden, 1, ggml_element_size(x) * hidden,
                          ggml_element_size(x) * hidden * static_cast<size_t>(T_prompt - 1));
    last_x = ggml_cont(ctx, last_x);
  }

  ggml_tensor *logits = ggml_mul_mat(ctx, w.token_embd, last_x);
  logits = ggml_reshape_1d(ctx, logits, vocab);
  ggml_set_name(logits, "dec.logits.prefill");

  pb.out = logits;
  ggml_set_output(pb.out);
  ggml_build_forward_expand(gf, pb.out);
  pb.graph = gf;
  return pb;
}

StepBuild build_step_graph(ggml_context *ctx, const CanaryQwenWeights &w,
                           const CanaryQwenHParams &hp, causal_lm::KvCache &kv, int max_n_kv,
                           bool use_flash) {
  StepBuild sb;
  if (ctx == nullptr || max_n_kv <= 0 || kv.self_k == nullptr || max_n_kv > kv.n_ctx) {
    return sb;
  }

  const int64_t vocab = hp.dec_vocab_size;
  const int n_layer = hp.dec_n_layers;
  const float rms_eps = hp.dec_rms_norm_eps;
  const auto block_params = decoder_block_params(hp);

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

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 8192, false);
  if (gf == nullptr) {
    return sb;
  }

  ggml_tensor *x = ggml_get_rows(ctx, w.token_embd, sb.input_id_in);
  for (int il = 0; il < n_layer; ++il) {
    x = causal_lm::block_step(ctx, gf, x, w.dec_blocks[static_cast<size_t>(il)], block_params, kv,
                              il, max_n_kv, sb.mask_in, sb.position_in, sb.kv_idx_in, use_flash);
  }

  x = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps), w.output_norm);
  ggml_tensor *logits = ggml_mul_mat(ctx, w.token_embd, x);
  logits = ggml_reshape_1d(ctx, logits, vocab);
  ggml_set_name(logits, "step.logits");

  ggml_tensor *amax = ggml_argmax(ctx, logits);
  ggml_set_name(amax, "step.argmax");

  sb.out = amax;
  sb.logits = logits;
  ggml_set_output(sb.out);
  ggml_set_output(sb.logits);
  ggml_build_forward_expand(gf, sb.out);
  ggml_build_forward_expand(gf, logits);
  sb.graph = gf;
  return sb;
}

PrefillBuildBatched build_prefill_graph_batched(ggml_context *ctx, const CanaryQwenWeights &w,
                                                const CanaryQwenHParams &hp,
                                                causal_lm::KvCache &kv, int T_prompt_max,
                                                int T_audio_max, int n_batch, bool use_flash) {
  PrefillBuildBatched pb;
  if (ctx == nullptr || T_prompt_max <= 0 || T_audio_max <= 0 || n_batch <= 0 || !use_flash ||
      kv.self_k == nullptr || kv.n_batch != n_batch || T_prompt_max > kv.n_ctx) {
    return pb;
  }

  const int64_t hidden = hp.dec_hidden;
  const int64_t vocab = hp.dec_vocab_size;
  const int n_layer = hp.dec_n_layers;
  const float rms_eps = hp.dec_rms_norm_eps;
  const int B = n_batch;
  const auto block_params = decoder_block_params(hp);

  pb.input_ids_in = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, T_prompt_max, B);
  ggml_set_input(pb.input_ids_in);
  pb.audio_dense_in =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, static_cast<int64_t>(T_prompt_max) * B);
  ggml_set_input(pb.audio_dense_in);
  pb.keep_mask_in =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, static_cast<int64_t>(T_prompt_max) * B);
  ggml_set_input(pb.keep_mask_in);
  pb.positions_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt_max);
  ggml_set_input(pb.positions_in);
  pb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T_prompt_max, T_prompt_max);
  ggml_set_input(pb.mask_in);
  pb.kv_idx_in = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, T_prompt_max, B);
  ggml_set_input(pb.kv_idx_in);
  pb.last_idx_in = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, B);
  ggml_set_input(pb.last_idx_in);

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16384, false);
  if (gf == nullptr) {
    return pb;
  }

  ggml_tensor *ids_flat =
      ggml_reshape_1d(ctx, pb.input_ids_in, static_cast<int64_t>(T_prompt_max) * B);
  // Audio injection by elementwise blend: x * keep + audio_dense.
  ggml_tensor *x = ggml_get_rows(ctx, w.token_embd, ids_flat);
  x = ggml_add(ctx, ggml_mul(ctx, x, pb.keep_mask_in), pb.audio_dense_in);
  x = ggml_reshape_3d(ctx, x, hidden, T_prompt_max, B);

  for (int il = 0; il < n_layer; ++il) {
    x = causal_lm::block_prefill_batched(ctx, gf, x, w.dec_blocks[static_cast<size_t>(il)],
                                         block_params, kv, il, T_prompt_max, B, pb.mask_in,
                                         pb.positions_in, pb.kv_idx_in, use_flash);
    if (x == nullptr) {
      return pb;
    }
  }

  ggml_tensor *x_last = ggml_get_rows(ctx, x, pb.last_idx_in);
  x_last = ggml_reshape_2d(ctx, x_last, hidden, B);
  x_last = ggml_mul(ctx, ggml_rms_norm(ctx, x_last, rms_eps), w.output_norm);
  ggml_tensor *logits = ggml_mul_mat(ctx, w.token_embd, x_last);
  logits = ggml_reshape_2d(ctx, logits, vocab, B);
  pb.out = ggml_argmax(ctx, logits);
  ggml_set_output(pb.out);
  ggml_build_forward_expand(gf, pb.out);
  ggml_build_forward_expand(gf, logits);
  pb.graph = gf;
  return pb;
}

StepBuildBatched build_step_graph_batched(ggml_context *ctx, const CanaryQwenWeights &w,
                                          const CanaryQwenHParams &hp, causal_lm::KvCache &kv,
                                          int max_n_kv, int n_batch, bool use_flash) {
  StepBuildBatched sb;
  if (ctx == nullptr || max_n_kv <= 0 || n_batch <= 0 || !use_flash || kv.self_k == nullptr ||
      kv.n_batch != n_batch || max_n_kv > kv.n_ctx) {
    return sb;
  }

  const int64_t vocab = hp.dec_vocab_size;
  const int n_layer = hp.dec_n_layers;
  const float rms_eps = hp.dec_rms_norm_eps;
  const int B = n_batch;
  const auto block_params = decoder_block_params(hp);

  sb.input_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B);
  ggml_set_input(sb.input_ids_in);
  sb.position_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B);
  ggml_set_input(sb.position_in);
  sb.kv_idx_in = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 1, B);
  ggml_set_input(sb.kv_idx_in);
  sb.mask_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, max_n_kv, 1, 1, B);
  ggml_set_input(sb.mask_in);

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 8192, false);
  if (gf == nullptr) {
    return sb;
  }

  ggml_tensor *x = ggml_get_rows(ctx, w.token_embd, sb.input_ids_in);
  for (int il = 0; il < n_layer; ++il) {
    x = causal_lm::block_step_batched(ctx, gf, x, w.dec_blocks[static_cast<size_t>(il)],
                                      block_params, kv, il, max_n_kv, B, sb.mask_in,
                                      sb.position_in, sb.kv_idx_in, use_flash);
    if (x == nullptr) {
      return sb;
    }
  }

  x = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps), w.output_norm);
  ggml_tensor *logits = ggml_mul_mat(ctx, w.token_embd, x);
  logits = ggml_reshape_2d(ctx, logits, vocab, B);
  sb.out = ggml_argmax(ctx, logits);
  ggml_set_output(sb.out);
  ggml_build_forward_expand(gf, sb.out);
  ggml_build_forward_expand(gf, logits);
  sb.graph = gf;
  return sb;
}

} // namespace engine::models::canary_qwen
