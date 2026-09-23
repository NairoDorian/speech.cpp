// engine/models/whisper/runtime.cpp - inference for the native engine Whisper
// package: the model that the decoding policy (decoding.h) drives.
//
// Flow per run: PCM -> Phase-9 MelExtractor over the whole input (short-form
// is padded to one 30 s window first, as HF's processor does) -> the policy's
// seek loop, which calls back into encode_window() / decode_prompt() here for
// every 30 s window, fallback tier and generated token. Cancellation and
// progress go through RunControl on every hook call.
//
// Weights come through the shared TensorSource for both distribution formats
// (transcribe.cpp GGUF and whisper.cpp .bin); names are asked for by their
// legacy whisper.cpp spelling and mapped by WhisperAssets::tensor_name().

#include "engine/models/whisper/runtime.h"

#include "engine/models/whisper/graphs_internal.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::whisper {

namespace {

constexpr const char *kTag = "whisper";

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(std::string(kTag) + ": " + message);
}

std::string lname(const char *fmt, int i) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), fmt, i);
  return std::string(buf);
}

std::string shape_string(const std::vector<int64_t> &shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += (i ? ", " : "") + std::to_string(shape[i]);
  }
  return out + "]";
}

core::TensorShape to_tensor_shape(const std::vector<int64_t> &dims) {
  core::TensorShape shape;
  shape.rank = dims.size();
  for (size_t i = 0; i < dims.size(); ++i) {
    shape.dims[i] = dims[i];
  }
  return shape;
}

std::string trim_ascii(std::string s) {
  const auto is_space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  };
  size_t a = 0;
  size_t b = s.size();
  while (a < b && is_space(s[a])) {
    ++a;
  }
  while (b > a && is_space(s[b - 1])) {
    --b;
  }
  return s.substr(a, b - a);
}

} // namespace

void WhisperRuntime::GraphRun::free() {
  if (gallocr != nullptr) {
    ggml_gallocr_free(gallocr);
    gallocr = nullptr;
  }
  if (ctx != nullptr) {
    ggml_free(ctx);
    ctx = nullptr;
  }
  graph = nullptr;
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

core::TensorValue WhisperRuntime::load_weight(const std::string &legacy_name,
                                              const std::vector<int64_t> &shape) {
  const auto &source = *assets_->source;
  const std::string name = assets_->tensor_name(legacy_name);
  if (!source.has_tensor(name)) {
    fail("missing tensor " + name + " in " + assets_->model_path.string());
  }
  const auto metadata = source.require_metadata(name);
  if (metadata.shape == shape) {
    return store_->load_tensor(source, name, storage_type_, shape);
  }
  // A .bin stores conv biases as [C, 1]; any other mismatch is a wrong file.
  const auto elements = [](const std::vector<int64_t> &dims) {
    return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                           std::multiplies<int64_t>());
  };
  if (elements(metadata.shape) != elements(shape)) {
    fail("tensor " + name + " has shape " + shape_string(metadata.shape) +
         ", expected " + shape_string(shape));
  }
  return store_->load_tensor_as_shape(source, name, storage_type_, metadata.shape,
                                      to_tensor_shape(shape));
}

WhisperRuntime::WhisperRuntime(std::shared_ptr<const WhisperAssets> model_assets,
                               core::ExecutionContext &execution_context,
                               assets::TensorStorageType storage_type,
                               std::optional<ggml_type> kv_type)
    : assets_(std::move(model_assets)), execution_context_(execution_context),
      storage_type_(storage_type) {
  if (!assets_ || !assets_->source) {
    fail("runtime requires loaded assets");
  }
  backend_ = execution_context_.backend();
  if (backend_ == nullptr) {
    fail("execution backend is not initialized");
  }
  store_ = std::make_shared<core::BackendWeightStore>(
      backend_, execution_context_.backend_type(), "whisper.weights",
      512ull * 1024ull * 1024ull);

  const auto &hp = assets_->hparams;
  const int64_t d = hp.enc_d_model;
  const int64_t ffn = hp.enc_ffn_dim;

  // ----- encoder -----
  weights_.enc.conv1.weight = load_weight("encoder.conv1.weight", {d, hp.enc_num_mel_bins, 3});
  weights_.enc.conv1.bias = load_weight("encoder.conv1.bias", {d});
  weights_.enc.conv2.weight = load_weight("encoder.conv2.weight", {d, d, 3});
  weights_.enc.conv2.bias = load_weight("encoder.conv2.bias", {d});
  weights_.enc.positional_embedding =
      load_weight("encoder.positional_embedding", {hp.enc_max_source_positions, d});
  weights_.enc.final_norm.weight = load_weight("encoder.ln_post.weight", {d});
  weights_.enc.final_norm.bias = load_weight("encoder.ln_post.bias", {d});

  weights_.enc.layers.resize(static_cast<size_t>(hp.enc_n_layers));
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &layer = weights_.enc.layers[static_cast<size_t>(i)];
    auto w = [&](const char *fmt, std::vector<int64_t> shape) {
      return load_weight(lname(fmt, i), shape);
    };
    layer.attention_norm.weight = w("encoder.blocks.%d.attn_ln.weight", {d});
    layer.attention_norm.bias = w("encoder.blocks.%d.attn_ln.bias", {d});
    layer.attention.query.weight = w("encoder.blocks.%d.attn.query.weight", {d, d});
    layer.attention.query.bias = w("encoder.blocks.%d.attn.query.bias", {d});
    layer.attention.key.weight = w("encoder.blocks.%d.attn.key.weight", {d, d});
    layer.attention.key.bias = std::nullopt; // Whisper: k has NO bias
    layer.attention.value.weight = w("encoder.blocks.%d.attn.value.weight", {d, d});
    layer.attention.value.bias = w("encoder.blocks.%d.attn.value.bias", {d});
    layer.attention.out.weight = w("encoder.blocks.%d.attn.out.weight", {d, d});
    layer.attention.out.bias = w("encoder.blocks.%d.attn.out.bias", {d});
    layer.mlp_norm.weight = w("encoder.blocks.%d.mlp_ln.weight", {d});
    layer.mlp_norm.bias = w("encoder.blocks.%d.mlp_ln.bias", {d});
    layer.mlp.fc1_weight = w("encoder.blocks.%d.mlp.0.weight", {ffn, d});
    layer.mlp.fc1_bias = w("encoder.blocks.%d.mlp.0.bias", {ffn});
    layer.mlp.fc2_weight = w("encoder.blocks.%d.mlp.2.weight", {d, ffn});
    layer.mlp.fc2_bias = w("encoder.blocks.%d.mlp.2.bias", {d});
  }

  // ----- decoder -----
  const int64_t dd = hp.dec_d_model;
  const int64_t dffn = hp.dec_ffn_dim;
  weights_.dec_top.token_embd_w =
      load_weight("decoder.token_embedding.weight", {hp.dec_vocab_size, dd}).tensor;
  weights_.dec_top.pos_emb_w =
      load_weight("decoder.positional_embedding", {hp.dec_max_target_positions, dd}).tensor;
  weights_.dec_top.final_norm_w = load_weight("decoder.ln.weight", {dd}).tensor;
  weights_.dec_top.final_norm_b = load_weight("decoder.ln.bias", {dd}).tensor;

  weights_.dec_blocks.resize(static_cast<size_t>(hp.dec_n_layers));
  for (int i = 0; i < hp.dec_n_layers; ++i) {
    auto &b = weights_.dec_blocks[static_cast<size_t>(i)];
    auto w = [&](const char *fmt, std::vector<int64_t> shape) {
      return load_weight(lname(fmt, i), shape).tensor;
    };
    b.norm_self_w = w("decoder.blocks.%d.attn_ln.weight", {dd});
    b.norm_self_b = w("decoder.blocks.%d.attn_ln.bias", {dd});
    b.self_q_w = w("decoder.blocks.%d.attn.query.weight", {dd, dd});
    b.self_q_b = w("decoder.blocks.%d.attn.query.bias", {dd});
    b.self_k_w = w("decoder.blocks.%d.attn.key.weight", {dd, dd});
    b.self_v_w = w("decoder.blocks.%d.attn.value.weight", {dd, dd});
    b.self_v_b = w("decoder.blocks.%d.attn.value.bias", {dd});
    b.self_out_w = w("decoder.blocks.%d.attn.out.weight", {dd, dd});
    b.self_out_b = w("decoder.blocks.%d.attn.out.bias", {dd});
    b.norm_cross_w = w("decoder.blocks.%d.cross_attn_ln.weight", {dd});
    b.norm_cross_b = w("decoder.blocks.%d.cross_attn_ln.bias", {dd});
    b.cross_q_w = w("decoder.blocks.%d.cross_attn.query.weight", {dd, dd});
    b.cross_q_b = w("decoder.blocks.%d.cross_attn.query.bias", {dd});
    b.cross_k_w = w("decoder.blocks.%d.cross_attn.key.weight", {dd, dd});
    b.cross_v_w = w("decoder.blocks.%d.cross_attn.value.weight", {dd, dd});
    b.cross_v_b = w("decoder.blocks.%d.cross_attn.value.bias", {dd});
    b.cross_out_w = w("decoder.blocks.%d.cross_attn.out.weight", {dd, dd});
    b.cross_out_b = w("decoder.blocks.%d.cross_attn.out.bias", {dd});
    b.norm_ffn_w = w("decoder.blocks.%d.mlp_ln.weight", {dd});
    b.norm_ffn_b = w("decoder.blocks.%d.mlp_ln.bias", {dd});
    b.ffn_fc1_w = w("decoder.blocks.%d.mlp.0.weight", {dffn, dd});
    b.ffn_fc1_b = w("decoder.blocks.%d.mlp.0.bias", {dffn});
    b.ffn_fc2_w = w("decoder.blocks.%d.mlp.2.weight", {dd, dffn});
    b.ffn_fc2_b = w("decoder.blocks.%d.mlp.2.bias", {dd});
  }

  store_->upload();
  assets_->source->release_storage();

  // KV-cache dtype: F32 unless the session asks otherwise. `auto` reproduces
  // the arch / whisper.cpp policy (F16 cache for F16 / quantized decoders).
  // Measured 2026-09-23 against the arch on whisper-tiny Q8_0 over six modes
  // (whisper_engine_arch_parity_test, before B16c): F32 differed from the arch in one letter
  // of case on one clip; F16 in a comma on one clip and the last words of a
  // translation on another. Both are float-level near-ties (the two paths
  // order their attention reductions differently); F32 is the more precise.
  const bool f32_decoder =
      weights_.dec_blocks.empty() || weights_.dec_blocks[0].self_q_w->type == GGML_TYPE_F32;
  kv_type_ = kv_type.has_value() ? *kv_type : GGML_TYPE_F32;
  if (kv_type_ == GGML_TYPE_COUNT) { // "auto"
    kv_type_ = f32_decoder ? GGML_TYPE_F32 : GGML_TYPE_F16;
  }

  // ----- mel frontend (Phase-9 unified extractor) -----
  audio::FrontendSpec spec;
  spec.kind = audio::FrontendKind::MelSpectrogram;
  spec.sample_rate = hp.fe_sample_rate;
  spec.num_mels = hp.fe_num_mels;
  spec.n_fft = hp.fe_n_fft;
  spec.win_length = hp.fe_win_length;
  spec.hop_length = hp.fe_hop_length;
  spec.pre_emphasis = 0.0f;
  spec.f_min = 0.0f;
  spec.f_max = 8000.0f;
  spec.pad_mode = audio::PadMode::Reflect;
  spec.window_type = audio::WindowType::HannPeriodic;
  // Whisper: log10 -> global clamp to (max - 8) -> (x + 4) / 4.
  spec.normalize_mode = audio::NormalizeMode::PerUtterance;
  // The model's own slaney filterbank (and, for a GGUF, its window) verbatim
  // rather than recomputed from f_min / f_max.
  spec.filterbank = assets_->mel_filterbank;
  spec.window = assets_->window;
  mel_.emplace(spec);
}

WhisperRuntime::~WhisperRuntime() {
  encoder_run_.free();
  cross_kv_run_.free();
  step_run_.free();
  kv_cache_.free();
}

int32_t WhisperRuntime::sample_rate() const noexcept {
  return assets_ ? assets_->hparams.fe_sample_rate : 16000;
}

WhisperTokenContract WhisperRuntime::token_contract() const {
  const auto &hp = assets_->hparams;
  WhisperTokenContract ids;
  ids.vocab_size = hp.dec_vocab_size;
  ids.eot = hp.eot_token_id;
  ids.sot = hp.decoder_start_token_id;
  ids.transcribe = hp.is_multilingual ? hp.transcribe_token_id : -1;
  ids.translate = hp.is_multilingual ? hp.translate_token_id : -1;
  ids.prev_sot = hp.prev_sot_token_id;
  ids.no_timestamps = hp.no_timestamps_token_id;
  ids.multilingual = hp.is_multilingual;
  return ids;
}

// ---------------------------------------------------------------------------
// Model hooks
// ---------------------------------------------------------------------------

void WhisperRuntime::encode_window(const std::vector<float> &mel, int total_frames,
                                   int seek, int n_frames) {
  const auto &hp = assets_->hparams;
  const int n_mels = hp.enc_num_mel_bins;
  const int window = hp.fe_nb_max_frames;

  // The encoder always sees a full window. LAYOUT: MelExtractor output is
  // mel-major (m * frames + t) and so is the graph input ([1, n_mels, frames],
  // ggml ne = [frames, n_mels]); a window is a per-mel-row copy, zero-padded
  // after the real frames (HF's F.pad).
  std::vector<float> chunk(static_cast<size_t>(n_mels) * static_cast<size_t>(window), 0.0f);
  for (int m = 0; m < n_mels; ++m) {
    std::memcpy(chunk.data() + static_cast<size_t>(m) * window,
                mel.data() + static_cast<size_t>(m) * total_frames + seek,
                static_cast<size_t>(n_frames) * sizeof(float));
  }

  std::vector<float> enc_host;
  {
    ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, /*no_alloc=*/true};
    encoder_run_.free();
    encoder_run_.ctx = ggml_init(params);
    if (encoder_run_.ctx == nullptr) {
      fail("failed to init encoder compute context");
    }
    EncoderBuild eb = build_encoder_graph(encoder_run_.ctx, weights_, hp, window,
                                          /*use_flash=*/true);
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
      fail("encoder graph build failed");
    }
    encoder_run_.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (encoder_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(encoder_run_.gallocr, eb.graph)) {
      fail("encoder graph allocation failed");
    }
    encoder_run_.graph = eb.graph;
    ggml_backend_tensor_set(eb.mel_in, chunk.data(), 0, chunk.size() * sizeof(float));
    core::set_backend_threads(backend_, std::max(1, execution_context_.config().threads));
    if (core::compute_backend_graph(backend_, eb.graph) != GGML_STATUS_SUCCESS) {
      fail("encoder compute failed");
    }
    ggml_backend_synchronize(backend_);
    T_enc_ = eb.T_enc;
    enc_host.assign(static_cast<size_t>(hp.enc_d_model) * static_cast<size_t>(T_enc_), 0.0f);
    ggml_backend_tensor_get(eb.out, enc_host.data(), 0, enc_host.size() * sizeof(float));
  }

  // KV cache: allocated once per geometry, self/cross state reset per window.
  if (kv_cache_.buffer != nullptr && kv_cache_.T_enc != T_enc_) {
    kv_cache_.free();
  }
  if (kv_cache_.buffer == nullptr) {
    const int n_ctx = hp.dec_max_target_positions > 0 ? hp.dec_max_target_positions : 448;
    if (!engine::asr::kv_cache_init(kv_cache_, backend_, n_ctx, T_enc_, hp.dec_d_model,
                                    hp.dec_n_layers, kv_type_)) {
      fail("KV cache allocation failed");
    }
  }
  kv_cache_.n = 0;
  kv_cache_.head = 0;
  kv_cache_.cross_populated = false;

  // Cross-attention K/V for this window (tier- and token-invariant).
  {
    ggml_init_params params{32ull * 1024ull * 1024ull, nullptr, /*no_alloc=*/true};
    cross_kv_run_.free();
    cross_kv_run_.ctx = ggml_init(params);
    if (cross_kv_run_.ctx == nullptr) {
      fail("failed to init cross_kv compute context");
    }
    CrossKvBuild cb = build_cross_kv_graph(cross_kv_run_.ctx, weights_, hp, kv_cache_, T_enc_);
    if (cb.graph == nullptr || cb.encoder_out_in == nullptr) {
      fail("cross_kv graph build failed");
    }
    cross_kv_run_.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (cross_kv_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(cross_kv_run_.gallocr, cb.graph)) {
      fail("cross_kv graph allocation failed");
    }
    cross_kv_run_.graph = cb.graph;
    ggml_backend_tensor_set(cb.encoder_out_in, enc_host.data(), 0,
                            enc_host.size() * sizeof(float));
    if (core::compute_backend_graph(backend_, cb.graph) != GGML_STATUS_SUCCESS) {
      fail("cross_kv compute failed");
    }
    ggml_backend_synchronize(backend_);
    kv_cache_.cross_populated = true;
  }
}

void WhisperRuntime::decode_prompt(const std::vector<int32_t> &tokens, int n_past,
                                   int sot_row, std::vector<float> *sot_logits,
                                   std::vector<float> &last_logits) {
  const auto &hp = assets_->hparams;
  const int vocab = hp.dec_vocab_size;
  const int n_tokens = static_cast<int>(tokens.size());

  ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, /*no_alloc=*/true};
  step_run_.free();
  step_run_.ctx = ggml_init(params);
  if (step_run_.ctx == nullptr) {
    fail("failed to init decoder compute context");
  }
  DecoderBuild db = build_decoder_graph_kv(step_run_.ctx, weights_, hp, kv_cache_, n_tokens,
                                           n_past, T_enc_, /*use_flash=*/true);
  if (db.logits_out == nullptr || db.graph == nullptr) {
    fail("decoder graph build failed");
  }
  step_run_.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (step_run_.gallocr == nullptr || !ggml_gallocr_alloc_graph(step_run_.gallocr, db.graph)) {
    fail("decoder graph allocation failed");
  }
  step_run_.graph = db.graph;

  ggml_backend_tensor_set(db.token_ids_in, tokens.data(), 0, tokens.size() * sizeof(int32_t));
  std::vector<int32_t> pos_ids(static_cast<size_t>(n_tokens));
  for (int i = 0; i < n_tokens; ++i) {
    pos_ids[static_cast<size_t>(i)] = n_past + i;
  }
  ggml_tensor *pos_in = find_tensor_by_name(step_run_.ctx, "dec.pos_ids");
  if (pos_in == nullptr) {
    fail("decoder graph is missing dec.pos_ids");
  }
  ggml_backend_tensor_set(pos_in, pos_ids.data(), 0, pos_ids.size() * sizeof(int32_t));

  if (db.causal_mask_in != nullptr) {
    const int n_kv = n_past + n_tokens;
    std::vector<float> mask(static_cast<size_t>(n_kv) * static_cast<size_t>(n_tokens), 0.0f);
    for (int q = 0; q < n_tokens; ++q) {
      for (int k = 0; k < n_kv; ++k) {
        if (k > n_past + q) {
          mask[static_cast<size_t>(q) * n_kv + k] = -std::numeric_limits<float>::infinity();
        }
      }
    }
    ggml_backend_tensor_set(db.causal_mask_in, mask.data(), 0, mask.size() * sizeof(float));
  }

  core::set_backend_threads(backend_, std::max(1, execution_context_.config().threads));
  if (core::compute_backend_graph(backend_, db.graph) != GGML_STATUS_SUCCESS) {
    fail("decoder compute failed");
  }
  ggml_backend_synchronize(backend_);

  const size_t row_bytes = static_cast<size_t>(vocab) * sizeof(float);
  last_logits.resize(static_cast<size_t>(vocab));
  ggml_backend_tensor_get(db.logits_out, last_logits.data(),
                          row_bytes * static_cast<size_t>(n_tokens - 1), row_bytes);
  if (sot_logits != nullptr && sot_row >= 0 && sot_row < n_tokens) {
    sot_logits->resize(static_cast<size_t>(vocab));
    ggml_backend_tensor_get(db.logits_out, sot_logits->data(),
                            row_bytes * static_cast<size_t>(sot_row), row_bytes);
  }
  kv_cache_.n = n_past + n_tokens;
  kv_cache_.head = kv_cache_.n;
}

// ---------------------------------------------------------------------------
// transcribe
// ---------------------------------------------------------------------------

WhisperTranscription WhisperRuntime::transcribe(const runtime::AudioBuffer &audio,
                                                const runtime::RunControl &control,
                                                const WhisperDecodeOptions &options,
                                                WhisperTranscription *partial) {
  const auto &hp = assets_->hparams;
  std::vector<float> pcm = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
      audio.samples, audio.sample_rate, audio.channels, hp.fe_sample_rate);
  if (pcm.empty()) {
    fail("empty audio input");
  }

  // HF's processor pads short-form input to one 30 s window before the mel;
  // long-form input is featurized as is and windowed by the seek loop. (This
  // is what the arch did; it keeps the per-utterance normalization identical.)
  const size_t window_samples = static_cast<size_t>(hp.fe_n_samples);
  if (pcm.size() < window_samples) {
    pcm.resize(window_samples, 0.0f);
  }

  control.emit_progress("mel", 0, 1);
  std::vector<float> mel;
  int n_mels = 0;
  int n_frames = 0;
  if (!mel_->compute(pcm.data(), pcm.size(), mel, n_mels, n_frames,
                     std::max(1, execution_context_.config().threads))) {
    fail("mel extraction failed");
  }
  if (n_mels != hp.enc_num_mel_bins || n_frames <= 0) {
    fail("mel produced " + std::to_string(n_mels) + " x " + std::to_string(n_frames) +
         ", expected " + std::to_string(hp.enc_num_mel_bins) + " bins");
  }

  WhisperModelHooks hooks;
  hooks.encode_window = [&](int seek, int frames) { encode_window(mel, n_frames, seek, frames); };
  hooks.prefill = [&](const std::vector<int32_t> &prompt, int sot_row,
                      std::vector<float> &sot_logits, std::vector<float> &last_logits) {
    kv_cache_.n = 0;
    kv_cache_.head = 0;
    decode_prompt(prompt, 0, sot_row, &sot_logits, last_logits);
  };
  hooks.step = [&](int32_t token, int n_past, std::vector<float> &logits) {
    decode_prompt({token}, n_past, -1, nullptr, logits);
  };
  hooks.poll = [&](int done, int total) { control.emit_progress("decode", done, total); };

  const auto finish = [&](const WhisperDecodeResult &decoded) {
    WhisperTranscription out;
    out.text = trim_ascii(assets_->decode(decoded.text_ids));
    out.truncated = decoded.truncated;
    out.traces = decoded.traces;
    out.language_detected = decoded.language_detected;
    if (decoded.language_token >= 0) {
      const auto &ids = assets_->language_token_ids;
      const auto it = std::find(ids.begin(), ids.end(), decoded.language_token);
      if (it != ids.end()) {
        out.language = assets_->language_codes[static_cast<size_t>(it - ids.begin())];
      }
    }
    for (const auto &seg : decoded.segments) {
      std::string text = trim_ascii(assets_->decode(seg.text_ids));
      if (!text.empty()) {
        out.segments.push_back({seg.t0_ms, seg.t1_ms, std::move(text)});
      }
    }
    return out;
  };
  if (partial != nullptr) {
    hooks.on_interrupted = [&](const WhisperDecodeResult &completed) { *partial = finish(completed); };
  }

  WhisperDecodeResult decoded;
  try {
    decoded = run_whisper_decode(n_frames, hp.fe_nb_max_frames, hp.dec_max_target_positions,
                                 token_contract(), hp.suppress_tokens, hp.begin_suppress_tokens,
                                 options, hooks);
  } catch (...) {
    kv_cache_.free();
    throw;
  }
  kv_cache_.free();
  return finish(decoded);
}

} // namespace engine::models::whisper
