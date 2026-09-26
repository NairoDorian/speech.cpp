// engine/models/medasr/runtime.cpp - inference for the native engine MedASR
// package (see runtime.h).
//
// Ported from src/runtime/arch/medasr/model.cpp (run, run_batch_encode,
// decode_one_utterance). Flow per call: PCM -> engine::audio::MelExtractor
// (the arch's LASR MelFrontend configuration, reproduced exactly: pad_mode
// none / center=False, the GGUF's window left-aligned in the 512-point FFT,
// the GGUF's filterbank, natural log with a 1e-5 floor, no normalization) ->
// frame-major transpose -> encoder graph -> host greedy CTC.
//
// Differences from the arch, none numeric:
//   - one backend + ggml_gallocr (the engine convention) instead of a
//     multi-backend scheduler with op offload; on the CPU the same kernels run
//     on the same data;
//   - the folded BatchNorm scale / bias are weights (uploaded once with the
//     rest) instead of graph inputs re-uploaded every call;
//   - audio too short for the subsampling stem is rejected with
//     std::invalid_argument (the arch hit a ggml assertion building the conv);
//   - TRANSCRIBE_MEL_FROM_REF and the tensor-dump hooks are not carried over.

#include "engine/models/medasr/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::medasr {

namespace {

constexpr const char *kTag = "medasr";

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

double ms_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

void MedAsrRuntime::GraphRun::free() {
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

// A weight by its transcribe.cpp name, checked against the row-major shape
// (the reverse of the ggml ne the arch's catalog checks).
ggml_tensor *MedAsrRuntime::load_matrix(const std::string &name,
                                        const std::vector<int64_t> &shape) {
  const auto &source = *assets_->source;
  if (!source.has_tensor(name)) {
    fail("missing tensor " + name + " in " + assets_->model_path.string());
  }
  const auto metadata = source.require_metadata(name);
  if (metadata.shape != shape) {
    fail("tensor " + name + " has shape " + shape_string(metadata.shape) + ", expected " +
         shape_string(shape));
  }
  return store_->load_tensor(source, name, storage_type_, shape).tensor;
}

// LayerNorm scales and biases: F32 in every MedASR GGUF (the arch requires it).
ggml_tensor *MedAsrRuntime::load_vector(const std::string &name, int64_t n) {
  const auto &source = *assets_->source;
  if (!source.has_tensor(name)) {
    fail("missing tensor " + name + " in " + assets_->model_path.string());
  }
  const std::vector<int64_t> shape{n};
  const auto metadata = source.require_metadata(name);
  if (metadata.shape != shape) {
    fail("tensor " + name + " has shape " + shape_string(metadata.shape) + ", expected " +
         shape_string(shape));
  }
  return store_->load_f32_tensor(source, name, shape).tensor;
}

MedAsrRuntime::MedAsrRuntime(std::shared_ptr<const MedAsrAssets> model_assets,
                             core::ExecutionContext &execution_context,
                             assets::TensorStorageType storage_type)
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
      backend_, execution_context_.backend_type(), "medasr.weights", 512ull * 1024ull * 1024ull);

  const auto &hp = assets_->hparams;
  const int64_t d = hp.enc_hidden;
  const int64_t dff = hp.enc_intermediate;
  const int64_t k_sub = hp.enc_sub_kernel;
  const int64_t c_sub = hp.enc_sub_channels;
  const int64_t k_dw = hp.enc_conv_kernel;

  // Subsampling (arch catalog: GET_LIN_W(n_mels, d), GET_CONV_W(k, d, d),
  // GET_CONV_W(k, d, sub_c), GET_LIN_W(sub_c, d), biases F32).
  auto &ss = weights_.subsampling;
  ss.dense0_w = load_matrix("enc.subsampling.dense_0.weight", {d, hp.enc_num_mel_bins});
  ss.dense0_b = load_vector("enc.subsampling.dense_0.bias", d);
  ss.conv0_w = load_matrix("enc.subsampling.conv_0.weight", {d, d, k_sub});
  ss.conv0_b = load_vector("enc.subsampling.conv_0.bias", d);
  ss.conv1_w = load_matrix("enc.subsampling.conv_1.weight", {c_sub, d, k_sub});
  ss.conv1_b = load_vector("enc.subsampling.conv_1.bias", c_sub);
  ss.dense1_w = load_matrix("enc.subsampling.dense_1.weight", {d, c_sub});
  ss.dense1_b = load_vector("enc.subsampling.dense_1.bias", d);

  weights_.blocks.resize(static_cast<size_t>(hp.enc_n_layers));
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &b = weights_.blocks[static_cast<size_t>(i)];
    auto m = [&](const char *fmt, std::vector<int64_t> shape) {
      return load_matrix(lname(fmt, i), shape);
    };
    auto v = [&](const char *fmt) { return load_vector(lname(fmt, i), d); };
    b.norm_ff1_w = v("enc.blocks.%d.norm_ff1.weight");
    b.ff1_up_w = m("enc.blocks.%d.ff1_up.weight", {dff, d});
    b.ff1_down_w = m("enc.blocks.%d.ff1_down.weight", {d, dff});
    b.norm_attn_w = v("enc.blocks.%d.norm_attn.weight");
    b.attn_q_w = m("enc.blocks.%d.attn_q.weight", {d, d});
    b.attn_k_w = m("enc.blocks.%d.attn_k.weight", {d, d});
    b.attn_v_w = m("enc.blocks.%d.attn_v.weight", {d, d});
    b.attn_o_w = m("enc.blocks.%d.attn_o.weight", {d, d});
    b.norm_conv_w = v("enc.blocks.%d.norm_conv.weight");
    b.conv_pw1_w = m("enc.blocks.%d.conv_pointwise1.weight", {2 * d, d, 1});
    b.conv_dw_w = m("enc.blocks.%d.conv_depthwise.weight", {d, 1, k_dw});
    b.conv_pw2_w = m("enc.blocks.%d.conv_pointwise2.weight", {d, d, 1});
    // Folded on the host at asset load (arch fuse_batch_norm); constant, so
    // they are weights here rather than per-call graph inputs.
    b.conv_bn_scale = store_
                          ->make_f32(core::TensorShape::from_dims({d}),
                                     assets_->bn_scale[static_cast<size_t>(i)])
                          .tensor;
    b.conv_bn_bias = store_
                         ->make_f32(core::TensorShape::from_dims({d}),
                                    assets_->bn_bias[static_cast<size_t>(i)])
                         .tensor;
    b.norm_ff2_w = v("enc.blocks.%d.norm_ff2.weight");
    b.ff2_up_w = m("enc.blocks.%d.ff2_up.weight", {dff, d});
    b.ff2_down_w = m("enc.blocks.%d.ff2_down.weight", {d, dff});
    b.norm_post_w = v("enc.blocks.%d.norm_post.weight");
  }

  weights_.enc_out_norm_w = load_vector("enc.out_norm.weight", d);
  weights_.ctc_proj_w = load_matrix("ctc.proj.weight", {hp.ctc_vocab_size, d, 1});
  weights_.ctc_proj_b = load_vector("ctc.proj.bias", hp.ctc_vocab_size);

  store_->upload();
  assets_->source->release_storage();

  policy_ = resolve_graph_policy(ggml_backend_name(backend_));

  // The arch's MelConfig, field for field.
  audio::FrontendSpec spec;
  spec.kind = audio::FrontendKind::MelSpectrogram;
  spec.sample_rate = hp.fe_sample_rate;
  spec.num_mels = hp.fe_num_mels;
  spec.n_fft = hp.fe_n_fft;
  spec.win_length = hp.fe_win_length;
  spec.hop_length = hp.fe_hop_length;
  spec.pre_emphasis = 0.0f;
  spec.f_min = hp.fe_mel_lower_hz;
  spec.f_max = hp.fe_mel_upper_hz;
  spec.pad_mode = audio::PadMode::None; // PyTorch center=False, window left-aligned
  spec.window_type = audio::WindowType::HannSymmetric; // unused: the GGUF window wins
  spec.normalize_mode = audio::NormalizeMode::None;
  spec.log_clamp_min = hp.fe_log_clamp_min; // log(max(power, 1e-5))
  spec.filterbank = assets_->mel_filterbank;
  spec.window = assets_->window;
  mel_.emplace(spec);
}

MedAsrRuntime::~MedAsrRuntime() { encoder_run_.free(); }

int32_t MedAsrRuntime::sample_rate() const noexcept {
  return assets_ ? assets_->hparams.fe_sample_rate : 16000;
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

std::vector<float> MedAsrRuntime::prepare_pcm(const runtime::AudioBuffer &audio) const {
  if (audio.samples.empty()) {
    throw std::invalid_argument("medasr: empty audio input");
  }
  if (audio.sample_rate <= 0 || audio.channels <= 0) {
    throw std::invalid_argument("medasr: audio needs a positive sample rate and channel count");
  }
  return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
      audio.samples, audio.sample_rate, audio.channels, sample_rate());
}

void MedAsrRuntime::validate_length(size_t n_samples) const {
  const auto &hp = assets_->hparams;
  const double rate_khz = static_cast<double>(hp.fe_sample_rate) / 1000.0;
  if (n_samples < static_cast<size_t>(hp.fe_win_length)) {
    // The arch's MelFrontend::compute returned INVALID_ARG here.
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "medasr: audio is %zu samples (%.1f ms); the STFT needs at least one "
                  "%d-sample (%.1f ms) window",
                  n_samples, static_cast<double>(n_samples) / rate_khz, hp.fe_win_length,
                  static_cast<double>(hp.fe_win_length) / rate_khz);
    throw std::invalid_argument(msg);
  }
  const int64_t min_samples = assets_->min_audio_samples();
  if (static_cast<int64_t>(n_samples) < min_samples) {
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "medasr: audio is %zu samples (%.1f ms); at least %lld samples (%.1f ms) "
                  "are needed for one encoder frame after the 4x subsampling stem",
                  n_samples, static_cast<double>(n_samples) / rate_khz,
                  static_cast<long long>(min_samples),
                  static_cast<double>(min_samples) / rate_khz);
    throw std::invalid_argument(msg);
  }
}

std::vector<float> MedAsrRuntime::compute_mel(const std::vector<float> &pcm,
                                              int &n_frames) const {
  const auto &hp = assets_->hparams;
  std::vector<float> mel_major;
  int n_mels = 0;
  n_frames = 0;
  if (!mel_->compute(pcm.data(), pcm.size(), mel_major, n_mels, n_frames,
                     std::max(1, execution_context_.config().threads))) {
    throw std::invalid_argument("medasr: mel extraction rejected the audio (" +
                                std::to_string(pcm.size()) + " samples)");
  }
  if (n_mels != hp.fe_num_mels || n_frames <= 0) {
    fail("mel produced " + std::to_string(n_mels) + " x " + std::to_string(n_frames) +
         ", expected " + std::to_string(hp.fe_num_mels) + " bins");
  }
  // LAYOUT: MelExtractor is mel-major (m * T + t); the encoder's
  // [n_mels, T_mel] input wants each frame's n_mels floats contiguous.
  std::vector<float> frame_major(static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
  for (int m = 0; m < n_mels; ++m) {
    const float *src = mel_major.data() + static_cast<size_t>(m) * n_frames;
    for (int t = 0; t < n_frames; ++t) {
      frame_major[static_cast<size_t>(t) * n_mels + m] = src[t];
    }
  }
  return frame_major;
}

void MedAsrRuntime::warn_if_over_window(int enc_frames, int utterance) const {
  const auto &hp = assets_->hparams;
  if (enc_frames <= hp.enc_max_pos_emb) {
    return;
  }
  // Soft window (docs/input-limits.md): RoPE extrapolates past the trained
  // range, so accuracy may degrade; warn and proceed, never reject.
  const double audio_s = static_cast<double>(enc_frames) *
                         static_cast<double>(assets_->ms_per_encoder_frame()) / 1000.0;
  char msg[384];
  if (utterance >= 0) {
    std::snprintf(msg, sizeof(msg),
                  "utterance %d: audio is %.1f s (%d encoder frames), beyond the %d-frame "
                  "RoPE range this model was trained on; transcription may be degraded past "
                  "this point",
                  utterance, audio_s, enc_frames, hp.enc_max_pos_emb);
  } else {
    std::snprintf(msg, sizeof(msg),
                  "audio is %.1f s (%d encoder frames), beyond the %d-frame RoPE range this "
                  "model was trained on; transcription may be degraded past this point",
                  audio_s, enc_frames, hp.enc_max_pos_emb);
  }
  engine::debug::log_message(engine::debug::LogLevel::Warning, kTag, msg);
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

std::vector<float> MedAsrRuntime::encode(const std::vector<float> &mel, int T_mel, int n_batch,
                                         const std::vector<int> &real_tenc, bool var_len,
                                         int &T_enc, int &vocab) {
  const auto &hp = assets_->hparams;

  // A fresh graph per call: every input below is uploaded after allocation
  // and before the one compute, so no state survives into the next call.
  encoder_run_.free();
  ggml_init_params params{encoder_graph_context_bytes(), nullptr, /*no_alloc=*/true};
  encoder_run_.ctx = ggml_init(params);
  if (encoder_run_.ctx == nullptr) {
    fail("failed to init the encoder compute context");
  }
  MedAsrEncoderBuild eb = build_encoder_graph(encoder_run_.ctx, weights_, hp, T_mel, n_batch,
                                              var_len, policy_);
  if (eb.mel_in == nullptr || eb.logits == nullptr || eb.graph == nullptr) {
    encoder_run_.free();
    throw std::invalid_argument("medasr: " + std::to_string(T_mel) +
                                " mel frames are too few for the subsampling stem");
  }
  encoder_run_.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (encoder_run_.gallocr == nullptr ||
      !ggml_gallocr_alloc_graph(encoder_run_.gallocr, eb.graph)) {
    encoder_run_.free();
    fail("encoder graph allocation failed - out of memory; split long audio into shorter "
         "segments (the model's trained window is " +
         std::to_string(assets_->max_audio_ms()) + " ms)");
  }
  encoder_run_.graph = eb.graph;

  if (mel.size() * sizeof(float) != ggml_nbytes(eb.mel_in)) {
    encoder_run_.free();
    fail("mel input size mismatch");
  }
  ggml_backend_tensor_set(eb.mel_in, mel.data(), 0, mel.size() * sizeof(float));

  const int64_t t_enc = eb.T_enc;
  {
    std::vector<int32_t> pos(static_cast<size_t>(t_enc));
    for (int64_t i = 0; i < t_enc; ++i) {
      pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(eb.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
  }

  if (eb.attn_pad_mask_in != nullptr && eb.conv_pad_mask_in != nullptr) {
    // transcribe::fill_keypad_mask / fill_valid_frame_mask: -1e30 keeps a
    // fully masked row finite on the manual softmax path.
    const size_t n = static_cast<size_t>(n_batch);
    const size_t T = static_cast<size_t>(t_enc);
    std::vector<float> keypad(T * n);
    std::vector<float> valid(T * n);
    for (size_t b = 0; b < n; ++b) {
      const int64_t real = b < real_tenc.size() ? real_tenc[b] : t_enc;
      for (size_t k = 0; k < T; ++k) {
        const bool keep = static_cast<int64_t>(k) < real;
        keypad[b * T + k] = keep ? 0.0f : -1e30f;
        valid[b * T + k] = keep ? 1.0f : 0.0f;
      }
    }
    ggml_backend_tensor_set(eb.attn_pad_mask_in, keypad.data(), 0, keypad.size() * sizeof(float));
    ggml_backend_tensor_set(eb.conv_pad_mask_in, valid.data(), 0, valid.size() * sizeof(float));
  }

  core::set_backend_threads(backend_, std::max(1, execution_context_.config().threads));
  if (core::compute_backend_graph(backend_, eb.graph) != GGML_STATUS_SUCCESS) {
    encoder_run_.free();
    fail("encoder compute failed");
  }
  ggml_backend_synchronize(backend_);

  vocab = static_cast<int>(eb.logits->ne[0]);
  T_enc = static_cast<int>(eb.logits->ne[1]);
  if (vocab != hp.ctc_vocab_size || T_enc <= 0) {
    encoder_run_.free();
    fail("ctc_logits shape mismatch (vocab=" + std::to_string(vocab) +
         " T_enc=" + std::to_string(T_enc) + ")");
  }
  std::vector<float> logits(static_cast<size_t>(vocab) * static_cast<size_t>(T_enc) *
                            static_cast<size_t>(n_batch));
  ggml_backend_tensor_get(eb.logits, logits.data(), 0, logits.size() * sizeof(float));

  // Release the compute buffer after every call, as the arch frees its
  // scheduler, so repeated runs do not pin peak activation memory.
  encoder_run_.free();
  return logits;
}

MedAsrTranscription MedAsrRuntime::decode_utterance(const float *logits, int vocab,
                                                    int n_frames) const {
  const CtcGreedyResult greedy =
      ctc_greedy_decode(logits, vocab, n_frames, assets_->hparams.ctc_blank_id);
  MedAsrTranscription out;
  out.raw_text = assets_->decode(greedy.tokens);
  out.text = out.raw_text;
  if (!out.text.empty() && out.text.front() == ' ') {
    out.text.erase(out.text.begin());
  }
  out.tokens = make_ctc_tokens(greedy);
  return out;
}

// ---------------------------------------------------------------------------
// transcribe / transcribe_batch
// ---------------------------------------------------------------------------

MedAsrTranscription MedAsrRuntime::transcribe(const std::vector<float> &pcm,
                                              const runtime::RunControl &control) {
  control.emit_progress("mel", 0, 3); // the arch polled abort at the top of run()
  validate_length(pcm.size());

  const auto t_mel = std::chrono::steady_clock::now();
  int T_mel = 0;
  const std::vector<float> mel = compute_mel(pcm, T_mel);
  engine::debug::timing_log_scalar("medasr.mel_ms", ms_since(t_mel));

  const int enc_frames = assets_->encoder_frames_for(T_mel);
  warn_if_over_window(enc_frames, -1);

  control.emit_progress("encode", 1, 3);
  const auto t_enc = std::chrono::steady_clock::now();
  int T_enc = 0;
  int vocab = 0;
  const std::vector<float> logits =
      encode(mel, T_mel, /*n_batch=*/1, {enc_frames}, /*var_len=*/false, T_enc, vocab);
  engine::debug::timing_log_scalar("medasr.encode_ms", ms_since(t_enc));

  control.emit_progress("decode", 2, 3);
  MedAsrTranscription out = decode_utterance(logits.data(), vocab, T_enc);
  out.over_trained_window = enc_frames > assets_->hparams.enc_max_pos_emb;
  return out;
}

std::vector<MedAsrTranscription>
MedAsrRuntime::transcribe_batch(const std::vector<std::vector<float>> &pcm,
                                const runtime::RunControl &control) {
  const auto &hp = assets_->hparams;
  const int n = static_cast<int>(pcm.size());
  if (n == 0) {
    return {};
  }
  control.emit_progress("mel", 0, 3);

  const int n_mels = hp.fe_num_mels;
  std::vector<std::vector<float>> mels(static_cast<size_t>(n));
  std::vector<int> nf(static_cast<size_t>(n), 0);
  const auto t_mel = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) {
    validate_length(pcm[static_cast<size_t>(i)].size());
    mels[static_cast<size_t>(i)] = compute_mel(pcm[static_cast<size_t>(i)], nf[static_cast<size_t>(i)]);
  }
  engine::debug::timing_log_scalar("medasr.mel_ms", ms_since(t_mel));

  const int T_max = *std::max_element(nf.begin(), nf.end());
  bool var_len = false;
  for (int i = 0; i < n; ++i) {
    var_len = var_len || nf[static_cast<size_t>(i)] != T_max;
  }

  // Pack [n][T_max][n_mels], each utterance zero-padded along time (the
  // arch's pack_pad_time_major).
  std::vector<float> packed(static_cast<size_t>(n_mels) * static_cast<size_t>(T_max) *
                                static_cast<size_t>(n),
                            0.0f);
  for (int b = 0; b < n; ++b) {
    const auto &src = mels[static_cast<size_t>(b)];
    const size_t slab = static_cast<size_t>(b) * static_cast<size_t>(n_mels) * T_max;
    const size_t count = static_cast<size_t>(nf[static_cast<size_t>(b)]) * n_mels;
    if (count > 0 && count <= src.size()) {
      std::memcpy(packed.data() + slab, src.data(), count * sizeof(float));
    }
  }
  mels.clear();

  std::vector<int> real_tenc(static_cast<size_t>(n), 0);
  for (int b = 0; b < n; ++b) {
    real_tenc[static_cast<size_t>(b)] = assets_->encoder_frames_for(nf[static_cast<size_t>(b)]);
    warn_if_over_window(real_tenc[static_cast<size_t>(b)], b);
  }

  control.emit_progress("encode", 1, 3);
  const auto t_enc = std::chrono::steady_clock::now();
  int T_enc = 0;
  int vocab = 0;
  const std::vector<float> logits = encode(packed, T_max, n, real_tenc, var_len, T_enc, vocab);
  engine::debug::timing_log_scalar("medasr.encode_ms", ms_since(t_enc));

  // Per-utterance host decode over each slab's valid frames.
  std::vector<MedAsrTranscription> results;
  results.reserve(static_cast<size_t>(n));
  const size_t utt_elems = static_cast<size_t>(vocab) * static_cast<size_t>(T_enc);
  for (int b = 0; b < n; ++b) {
    control.emit_progress("decode", b, n);
    MedAsrTranscription out = decode_utterance(
        logits.data() + static_cast<size_t>(b) * utt_elems, vocab,
        std::min(real_tenc[static_cast<size_t>(b)], T_enc));
    out.over_trained_window = real_tenc[static_cast<size_t>(b)] > hp.enc_max_pos_emb;
    results.push_back(std::move(out));
  }
  return results;
}

} // namespace engine::models::medasr
