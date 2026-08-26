// engine/models/whisper/runtime.cpp - inference orchestration for the native
// engine Whisper package.
//
// Flow mirrors the arch run(): mel (Phase-9 engine MelExtractor, not a private
// implementation) -> encoder -> host readback -> cross-KV precompute ->
// prompt pass -> greedy single-token decode loop. Cancellation polls
// RunControl at every decode step via emit_progress.
//
// Weights come from the legacy whisper.cpp `.bin`: the manifest located at
// parse time gives (offset, nbytes, type, ne) per tensor, and each slot is
// streamed straight into the BackendWeightStore with the file's own ggml type.
// No GGUF is involved - see assets.cpp for why.

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
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
// Weight streaming
// ---------------------------------------------------------------------------

ggml_tensor *WhisperRuntime::load_bin_tensor(const char *legacy_name,
                                             bool squeeze_leading) {
  static thread_local std::unordered_map<std::string, const WhisperBinTensor *>
      unused; // silence unused-warning tooling on some compilers
  (void)unused;

  const WhisperBinTensor *entry = nullptr;
  for (const auto &t : assets_->tensors) {
    if (t.name == legacy_name) {
      entry = &t;
      break;
    }
  }
  if (entry == nullptr) {
    fail(std::string("missing tensor in .bin: ") + legacy_name);
  }

  // The file stores dims in ggml ne order; TensorShape is logical and gets
  // reversed by to_ggml_dims, so hand it the reverse of the file's ne.
  int n_dims = entry->n_dims;
  int64_t ne[4] = {entry->ne[0], entry->ne[1], entry->ne[2], entry->ne[3]};
  if (squeeze_leading && n_dims == 2 && ne[0] == 1) {
    // Legacy whisper stores conv biases as [1, Cout]; the graph wants [Cout].
    ne[0] = ne[1];
    ne[1] = 1;
    n_dims = 1;
  }

  std::vector<int64_t> logical;
  logical.reserve(static_cast<size_t>(n_dims));
  for (int d = n_dims - 1; d >= 0; --d) {
    logical.push_back(ne[d]);
  }
  core::TensorShape shape;
  shape.rank = static_cast<size_t>(n_dims);
  for (size_t i = 0; i < logical.size(); ++i) {
    shape.dims[i] = logical[i];
  }

  std::vector<char> bytes(static_cast<size_t>(entry->nbytes));
  {
    std::ifstream f(assets_->model_path, std::ios::binary);
    if (!f) {
      fail("cannot reopen model for weight streaming: " +
           assets_->model_path.string());
    }
    f.seekg(static_cast<std::streamoff>(entry->offset));
    f.read(bytes.data(), static_cast<std::streamsize>(entry->nbytes));
    if (!f) {
      fail(std::string("short read streaming tensor ") + legacy_name);
    }
  }

  return store_->make_tensor(shape, entry->type, bytes.data(), bytes.size())
      .tensor;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

WhisperRuntime::WhisperRuntime(std::shared_ptr<const WhisperAssets> model_assets,
                               core::ExecutionContext &execution_context,
                               assets::TensorStorageType storage_type)
    : assets_(std::move(model_assets)), execution_context_(execution_context) {
  (void)storage_type; // .bin weights are uploaded in their stored type
  if (!assets_) {
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

  // ----- encoder stem / top -----
  weights_.enc_stem.conv0_w = load_bin_tensor("encoder.conv1.weight", false);
  weights_.enc_stem.conv0_b = load_bin_tensor("encoder.conv1.bias", true);
  weights_.enc_stem.conv1_w = load_bin_tensor("encoder.conv2.weight", false);
  weights_.enc_stem.conv1_b = load_bin_tensor("encoder.conv2.bias", true);

  weights_.enc_top.pos_emb_w =
      load_bin_tensor("encoder.positional_embedding", false);
  weights_.enc_top.final_norm_w =
      load_bin_tensor("encoder.ln_post.weight", false);
  weights_.enc_top.final_norm_b = load_bin_tensor("encoder.ln_post.bias", false);

  weights_.enc_blocks.resize(static_cast<size_t>(hp.enc_n_layers));
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &b = weights_.enc_blocks[static_cast<size_t>(i)];
    b.norm_attn_w =
        load_bin_tensor(lname("encoder.blocks.%d.attn_ln.weight", i).c_str(),
                        false);
    b.norm_attn_b = load_bin_tensor(
        lname("encoder.blocks.%d.attn_ln.bias", i).c_str(), false);
    b.attn_q_w = load_bin_tensor(
        lname("encoder.blocks.%d.attn.query.weight", i).c_str(), false);
    b.attn_q_b = load_bin_tensor(
        lname("encoder.blocks.%d.attn.query.bias", i).c_str(), false);
    // k has NO bias.
    b.attn_k_w = load_bin_tensor(
        lname("encoder.blocks.%d.attn.key.weight", i).c_str(), false);
    b.attn_v_w = load_bin_tensor(
        lname("encoder.blocks.%d.attn.value.weight", i).c_str(), false);
    b.attn_v_b = load_bin_tensor(
        lname("encoder.blocks.%d.attn.value.bias", i).c_str(), false);
    b.attn_out_w = load_bin_tensor(
        lname("encoder.blocks.%d.attn.out.weight", i).c_str(), false);
    b.attn_out_b = load_bin_tensor(
        lname("encoder.blocks.%d.attn.out.bias", i).c_str(), false);

    b.norm_ffn_w = load_bin_tensor(
        lname("encoder.blocks.%d.mlp_ln.weight", i).c_str(), false);
    b.norm_ffn_b = load_bin_tensor(
        lname("encoder.blocks.%d.mlp_ln.bias", i).c_str(), false);
    b.ffn_fc1_w = load_bin_tensor(
        lname("encoder.blocks.%d.mlp.0.weight", i).c_str(), false);
    b.ffn_fc1_b = load_bin_tensor(
        lname("encoder.blocks.%d.mlp.0.bias", i).c_str(), false);
    b.ffn_fc2_w = load_bin_tensor(
        lname("encoder.blocks.%d.mlp.2.weight", i).c_str(), false);
    b.ffn_fc2_b = load_bin_tensor(
        lname("encoder.blocks.%d.mlp.2.bias", i).c_str(), false);
  }

  // ----- decoder top -----
  weights_.dec_top.token_embd_w =
      load_bin_tensor("decoder.token_embedding.weight", false);
  weights_.dec_top.pos_emb_w =
      load_bin_tensor("decoder.positional_embedding", false);
  weights_.dec_top.final_norm_w = load_bin_tensor("decoder.ln.weight", false);
  weights_.dec_top.final_norm_b = load_bin_tensor("decoder.ln.bias", false);

  weights_.dec_blocks.resize(static_cast<size_t>(hp.dec_n_layers));
  for (int i = 0; i < hp.dec_n_layers; ++i) {
    auto &b = weights_.dec_blocks[static_cast<size_t>(i)];
    b.norm_self_w = load_bin_tensor(
        lname("decoder.blocks.%d.attn_ln.weight", i).c_str(), false);
    b.norm_self_b = load_bin_tensor(
        lname("decoder.blocks.%d.attn_ln.bias", i).c_str(), false);
    b.self_q_w = load_bin_tensor(
        lname("decoder.blocks.%d.attn.query.weight", i).c_str(), false);
    b.self_q_b = load_bin_tensor(
        lname("decoder.blocks.%d.attn.query.bias", i).c_str(), false);
    b.self_k_w = load_bin_tensor(
        lname("decoder.blocks.%d.attn.key.weight", i).c_str(), false);
    b.self_v_w = load_bin_tensor(
        lname("decoder.blocks.%d.attn.value.weight", i).c_str(), false);
    b.self_v_b = load_bin_tensor(
        lname("decoder.blocks.%d.attn.value.bias", i).c_str(), false);
    b.self_out_w = load_bin_tensor(
        lname("decoder.blocks.%d.attn.out.weight", i).c_str(), false);
    b.self_out_b = load_bin_tensor(
        lname("decoder.blocks.%d.attn.out.bias", i).c_str(), false);

    b.norm_cross_w = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn_ln.weight", i).c_str(), false);
    b.norm_cross_b = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn_ln.bias", i).c_str(), false);
    b.cross_q_w = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn.query.weight", i).c_str(), false);
    b.cross_q_b = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn.query.bias", i).c_str(), false);
    b.cross_k_w = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn.key.weight", i).c_str(), false);
    b.cross_v_w = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn.value.weight", i).c_str(), false);
    b.cross_v_b = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn.value.bias", i).c_str(), false);
    b.cross_out_w = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn.out.weight", i).c_str(), false);
    b.cross_out_b = load_bin_tensor(
        lname("decoder.blocks.%d.cross_attn.out.bias", i).c_str(), false);

    b.norm_ffn_w = load_bin_tensor(
        lname("decoder.blocks.%d.mlp_ln.weight", i).c_str(), false);
    b.norm_ffn_b = load_bin_tensor(
        lname("decoder.blocks.%d.mlp_ln.bias", i).c_str(), false);
    b.ffn_fc1_w = load_bin_tensor(
        lname("decoder.blocks.%d.mlp.0.weight", i).c_str(), false);
    b.ffn_fc1_b = load_bin_tensor(
        lname("decoder.blocks.%d.mlp.0.bias", i).c_str(), false);
    b.ffn_fc2_w = load_bin_tensor(
        lname("decoder.blocks.%d.mlp.2.weight", i).c_str(), false);
    b.ffn_fc2_b = load_bin_tensor(
        lname("decoder.blocks.%d.mlp.2.bias", i).c_str(), false);
  }

  store_->upload();

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
  // The .bin ships the exact slaney filterbank the model was trained with;
  // use it verbatim rather than recomputing from f_min/f_max.
  spec.filterbank = assets_->mel_filterbank;
  mel_.emplace(spec);

  suppress_tokens_ = hp.suppress_tokens;
  begin_suppress_tokens_ = hp.begin_suppress_tokens;
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

std::string
WhisperRuntime::decode_token_ids(const std::vector<int32_t> &ids) const {
  // Whisper's `.bin` vocabulary stores raw byte strings, so detokenization is
  // byte concatenation. Special ids (>= eot) never reach here.
  std::string out;
  for (const int32_t id : ids) {
    if (id < 0 ||
        static_cast<size_t>(id) >= assets_->vocab_tokens.size()) {
      continue;
    }
    out += assets_->vocab_tokens[static_cast<size_t>(id)];
  }
  // Whisper emits a leading space on the first word.
  if (!out.empty() && out.front() == ' ') {
    out.erase(out.begin());
  }
  return out;
}

// ---------------------------------------------------------------------------
// transcribe
// ---------------------------------------------------------------------------

WhisperTranscription
WhisperRuntime::transcribe(const runtime::AudioBuffer &audio,
                           const runtime::RunControl &control,
                           int32_t max_tokens) {
  const auto &hp = assets_->hparams;

  std::vector<float> pcm =
      engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
          audio.samples, audio.sample_rate, audio.channels, hp.fe_sample_rate);
  if (pcm.empty()) {
    fail("empty audio input");
  }

  // Whisper always sees a 30 s window: pad with silence, or trim (W2a has no
  // long-form seek - that is W2b).
  bool truncated_audio = false;
  const size_t window_samples = static_cast<size_t>(hp.fe_n_samples);
  if (pcm.size() > window_samples) {
    pcm.resize(window_samples);
    truncated_audio = true;
  } else if (pcm.size() < window_samples) {
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
  if (n_mels != hp.enc_num_mel_bins) {
    fail("mel produced " + std::to_string(n_mels) + " bins, expected " +
         std::to_string(hp.enc_num_mel_bins));
  }
  // The encoder needs an even frame count; PerUtterance already drops the
  // trailing frame, giving 3000 for a full window.
  if (n_frames % 2 != 0) {
    n_frames -= 1;
    mel.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
  }
  if (n_frames <= 0) {
    fail("mel produced no frames");
  }

  // LAYOUT: MelExtractor writes MEL-MAJOR (element (m, t) at m*n_frames + t),
  // but the graph's mel_in has ggml ne = [n_mels, n_frames], whose fastest
  // axis is n_mels - i.e. it expects FRAME-MAJOR (t*n_mels + m). Transpose on
  // the host before upload; feeding the extractor's buffer straight through
  // silently produces a transposed spectrogram, which the encoder happily
  // consumes and the decoder turns into confident, unrelated text.
  {
    std::vector<float> mel_frame_major(mel.size());
    for (int m = 0; m < n_mels; ++m) {
      for (int t = 0; t < n_frames; ++t) {
        mel_frame_major[static_cast<size_t>(t) * n_mels + m] =
            mel[static_cast<size_t>(m) * n_frames + t];
      }
    }
    mel.swap(mel_frame_major);
  }

  control.emit_progress("encode", 0, 1);

  // ----- encoder -----
  std::vector<float> enc_host;
  int T_enc = 0;
  {
    ggml_init_params params{64ull * 1024ull * 1024ull, nullptr,
                            /*no_alloc=*/true};
    encoder_run_.free();
    encoder_run_.ctx = ggml_init(params);
    if (encoder_run_.ctx == nullptr) {
      fail("failed to init encoder compute context");
    }
    EncoderBuild eb = build_encoder_graph(encoder_run_.ctx, weights_, hp,
                                          n_frames, /*use_flash=*/true);
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
      fail("encoder graph build failed");
    }
    encoder_run_.gallocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (encoder_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(encoder_run_.gallocr, eb.graph)) {
      fail("encoder graph allocation failed");
    }
    encoder_run_.graph = eb.graph;

    ggml_backend_tensor_set(eb.mel_in, mel.data(), 0,
                            mel.size() * sizeof(float));

    core::set_backend_threads(backend_,
                              std::max(1, execution_context_.config().threads));
    if (core::compute_backend_graph(backend_, eb.graph) !=
        GGML_STATUS_SUCCESS) {
      fail("encoder compute failed");
    }
    ggml_backend_synchronize(backend_);

    T_enc = eb.T_enc;
    enc_host.assign(static_cast<size_t>(hp.enc_d_model) *
                        static_cast<size_t>(T_enc),
                    0.0f);
    ggml_backend_tensor_get(eb.out, enc_host.data(), 0,
                            enc_host.size() * sizeof(float));
  }

  // ----- KV cache -----
  {
    if (kv_cache_.buffer != nullptr && kv_cache_.T_enc != T_enc) {
      kv_cache_.free();
    }
    if (kv_cache_.buffer == nullptr) {
      const int n_ctx = hp.dec_max_target_positions > 0
                            ? hp.dec_max_target_positions
                            : 448;
      if (!kv_cache_init(kv_cache_, backend_, n_ctx, T_enc, hp.dec_d_model,
                         hp.dec_n_layers, GGML_TYPE_F32)) {
        fail("KV cache allocation failed");
      }
    } else {
      kv_cache_.n = 0;
      kv_cache_.head = 0;
      kv_cache_.cross_populated = false;
    }
  }

  // ----- cross-KV precompute -----
  {
    ggml_init_params params{32ull * 1024ull * 1024ull, nullptr,
                            /*no_alloc=*/true};
    cross_kv_run_.free();
    cross_kv_run_.ctx = ggml_init(params);
    if (cross_kv_run_.ctx == nullptr) {
      fail("failed to init cross_kv compute context");
    }
    CrossKvBuild cb =
        build_cross_kv_graph(cross_kv_run_.ctx, weights_, hp, kv_cache_, T_enc);
    if (cb.graph == nullptr || cb.encoder_out_in == nullptr) {
      fail("cross_kv graph build failed");
    }
    cross_kv_run_.gallocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (cross_kv_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(cross_kv_run_.gallocr, cb.graph)) {
      fail("cross_kv graph allocation failed");
    }
    cross_kv_run_.graph = cb.graph;

    ggml_backend_tensor_set(cb.encoder_out_in, enc_host.data(), 0,
                            enc_host.size() * sizeof(float));
    if (core::compute_backend_graph(backend_, cb.graph) !=
        GGML_STATUS_SUCCESS) {
      fail("cross_kv compute failed");
    }
    ggml_backend_synchronize(backend_);
    kv_cache_.cross_populated = true;
  }

  control.emit_progress("encode", 1, 1);

  // ----- decode -----
  const int vocab = hp.dec_vocab_size;
  const int eot = hp.eot_token_id;

  int max_pos = hp.dec_max_target_positions;
  if (max_tokens > 0) {
    max_pos = std::min(max_pos, static_cast<int>(max_tokens));
  }

  // .en prompt is [SOT, notimestamps]; multilingual would add [lang, task]
  // between them (W2b - language detection is not wired in W2a).
  std::vector<int32_t> prompt_ids;
  prompt_ids.push_back(hp.decoder_start_token_id);
  if (hp.is_multilingual) {
    prompt_ids.push_back(hp.first_language_token_id); // <|en|>
    prompt_ids.push_back(hp.transcribe_token_id);
  }
  prompt_ids.push_back(hp.no_timestamps_token_id);

  std::vector<float> logits(static_cast<size_t>(vocab), 0.0f);

  // Runs one decoder pass over n_tokens starting at n_past and leaves the
  // final position's raw logits in `logits`.
  auto run_pass = [&](const std::vector<int32_t> &token_ids, int n_past) {
    const int n_tokens = static_cast<int>(token_ids.size());
    ggml_init_params params{64ull * 1024ull * 1024ull, nullptr,
                            /*no_alloc=*/true};
    step_run_.free();
    step_run_.ctx = ggml_init(params);
    if (step_run_.ctx == nullptr) {
      fail("failed to init decoder compute context");
    }
    DecoderBuild db =
        build_decoder_graph_kv(step_run_.ctx, weights_, hp, kv_cache_, n_tokens,
                               n_past, T_enc, /*use_flash=*/true);
    if (db.logits_out == nullptr || db.graph == nullptr) {
      fail("decoder graph build failed");
    }
    step_run_.gallocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (step_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(step_run_.gallocr, db.graph)) {
      fail("decoder graph allocation failed");
    }
    step_run_.graph = db.graph;

    ggml_backend_tensor_set(db.token_ids_in, token_ids.data(), 0,
                            token_ids.size() * sizeof(int32_t));

    std::vector<int32_t> pos_ids(static_cast<size_t>(n_tokens));
    for (int i = 0; i < n_tokens; ++i) {
      pos_ids[static_cast<size_t>(i)] = n_past + i;
    }
    ggml_tensor *pos_in = find_tensor_by_name(step_run_.ctx, "dec.pos_ids");
    if (pos_in == nullptr) {
      fail("decoder graph is missing dec.pos_ids");
    }
    ggml_backend_tensor_set(pos_in, pos_ids.data(), 0,
                            pos_ids.size() * sizeof(int32_t));

    if (db.causal_mask_in != nullptr) {
      const int n_kv = n_past + n_tokens;
      std::vector<float> mask(static_cast<size_t>(n_kv) *
                                  static_cast<size_t>(n_tokens),
                              0.0f);
      for (int q = 0; q < n_tokens; ++q) {
        for (int k = 0; k < n_kv; ++k) {
          if (k > n_past + q) {
            mask[static_cast<size_t>(q) * n_kv + k] =
                -std::numeric_limits<float>::infinity();
          }
        }
      }
      ggml_backend_tensor_set(db.causal_mask_in, mask.data(), 0,
                              mask.size() * sizeof(float));
    }

    core::set_backend_threads(backend_,
                              std::max(1, execution_context_.config().threads));
    if (core::compute_backend_graph(backend_, db.graph) !=
        GGML_STATUS_SUCCESS) {
      fail("decoder compute failed");
    }
    ggml_backend_synchronize(backend_);

    // Read the LAST position's logits row.
    const size_t row_bytes = static_cast<size_t>(vocab) * sizeof(float);
    ggml_backend_tensor_get(db.logits_out, logits.data(),
                            row_bytes * static_cast<size_t>(n_tokens - 1),
                            row_bytes);

    kv_cache_.n = n_past + n_tokens;
    kv_cache_.head = kv_cache_.n;
  };

  auto apply_suppression = [&](bool first_generated_step) {
    for (const int32_t id : suppress_tokens_) {
      if (id >= 0 && id < vocab) {
        logits[static_cast<size_t>(id)] =
            -std::numeric_limits<float>::infinity();
      }
    }
    if (first_generated_step) {
      for (const int32_t id : begin_suppress_tokens_) {
        if (id >= 0 && id < vocab) {
          logits[static_cast<size_t>(id)] =
              -std::numeric_limits<float>::infinity();
        }
      }
    }
    // W2a decodes without timestamps: everything at or above the first
    // timestamp token is out of play.
    const int timestamp_begin = hp.no_timestamps_token_id + 1;
    for (int id = timestamp_begin; id < vocab; ++id) {
      logits[static_cast<size_t>(id)] =
          -std::numeric_limits<float>::infinity();
    }
  };

  auto argmax_logits = [&]() {
    int best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab; ++i) {
      if (logits[static_cast<size_t>(i)] > best_v) {
        best_v = logits[static_cast<size_t>(i)];
        best = i;
      }
    }
    return best;
  };

  // Prompt pass.
  run_pass(prompt_ids, /*n_past=*/0);
  apply_suppression(/*first_generated_step=*/true);
  int next_token = argmax_logits();

  std::vector<int32_t> generated_ids;
  if (next_token != eot) {
    generated_ids.push_back(next_token);
  }
  int n_past = kv_cache_.n;

  while (next_token != eot && n_past < max_pos) {
    control.emit_progress("decode", n_past, max_pos);
    run_pass({next_token}, n_past);
    apply_suppression(/*first_generated_step=*/false);
    next_token = argmax_logits();
    if (next_token != eot) {
      generated_ids.push_back(next_token);
    }
    n_past = kv_cache_.n;
  }

  WhisperTranscription result;
  result.truncated = truncated_audio || (next_token != eot);
  result.text = decode_token_ids(generated_ids);

  kv_cache_.free();
  return result;
}

} // namespace engine::models::whisper
