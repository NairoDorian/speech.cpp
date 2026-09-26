// engine/models/gigaam/runtime.cpp - inference for the native engine GigaAM
// package: PCM -> framework MelExtractor -> Conformer encoder (ggml) -> host
// RNN-T / CTC greedy decode. Port of src/runtime/arch/gigaam/model.cpp
// (run / run_batch_encode) and mel.cpp.
//
// Frontend. The arch's private mel (arch/gigaam/mel.cpp) is replaced by the
// framework's engine::audio::MelExtractor, configured to the same recipe:
// the GGUF's own HTK filterbank and periodic Hann window, power 2, no
// pre-emphasis / dither / normalization, natural log of clamp(x, 1e-9, 1e9).
// Framing: the arch is center=False - frame t covers x[t*hop, t*hop + win).
// MelExtractor's PadMode::None is NOT usable for that at n_fft = 320: its
// non-power-of-two FFT path always places the signal at offset n_fft / 2 in
// its padded buffer (the pad-mode test there only chooses reflect vs zeros),
// so its frame t would start at t*hop - n_fft/2. PadMode::Constant has that
// same centred framing by definition, for every FFT size, so it is used
// here and the arch's frames are read off at a fixed offset:
//     arch frame t == MelExtractor(Constant) frame t + (n_fft / 2) / hop
// (= t + 1 for 320 / 160). Those frames lie wholly inside the signal, so the
// zero padding never reaches them. assets.cpp rejects geometries where this
// identity does not hold (n_fft != win_length, n_fft / 2 not a multiple of
// hop). LAYOUT: MelExtractor output is mel-major (m * frames + t), and so is
// the encoder input (ggml ne = [T_mel, n_mels]: element (t, m) at m*T + t) -
// frames are sliced per mel row, never transposed.
//
// Numerics of the frontend: same filterbank, window, framing and log domain;
// the arithmetic differs in the last bits - MelExtractor accumulates the mel
// projection in f64 (the arch in f32), uses a table-driven f32 radix-2
// twiddle / 5-point DFT leaf (the arch computes twiddles per call and its
// DFT leaf in f64), and takes the log in f64. Log-mel values agree to a few
// f32 ulps, not bit for bit; transcripts are expected to match (the parity
// test checks it). Everything after the mel (encoder graph, host heads,
// detokenization) is op-for-op the arch.

#include "engine/models/gigaam/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::gigaam {

namespace {

constexpr const char *kTag = "gigaam";

// GigaAM is a soft-window family: trained on utterances up to ~25 s, longer
// audio is transcribed whole with a warning (the arch's k_safe_audio_ms;
// published through transcribe_capabilities::max_audio_ms by the arch).
constexpr int kSafeSeconds = 25;
constexpr int64_t kSafeAudioMs = 25000;

// The arch's mel clamp bounds (hard-coded in arch/gigaam/mel.cpp; equal to
// stt.frontend.log_clamp_{min,max} in every published file).
constexpr float kLogClampMin = 1.0e-9f;
constexpr float kLogClampMax = 1.0e9f;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(std::string(kTag) + ": " + message);
}

std::string lname(const char *fmt, int i) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), fmt, i);
  return std::string(buf);
}

// Owns one encoder graph's metadata context and compute buffer. Released
// after every run, as the arch frees its scheduler after each run (GPU
// memory is not held between calls).
struct GraphRun {
  ggml_context *ctx = nullptr;
  ggml_gallocr_t gallocr = nullptr;

  GraphRun() = default;
  GraphRun(const GraphRun &) = delete;
  GraphRun &operator=(const GraphRun &) = delete;
  ~GraphRun() {
    if (gallocr != nullptr) {
      ggml_gallocr_free(gallocr);
    }
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }
};

} // namespace

GigaamRuntime::GigaamRuntime(std::shared_ptr<const GigaamAssets> model_assets,
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
      backend_, execution_context_.backend_type(), "gigaam.weights", 512ull * 1024ull * 1024ull);

  const auto &hp = assets_->hparams;
  const auto &source = *assets_->source;
  const int64_t d = hp.enc_d_model;
  const int64_t dff = hp.enc_d_ff;

  // Matrices / conv kernels follow the storage preference (native = the
  // GGUF's own types, what the arch runs); norms and biases stay F32.
  const auto weight = [&](const std::string &name, std::vector<int64_t> shape) {
    return store_->load_tensor(source, name, storage_type_, shape).tensor;
  };
  const auto f32 = [&](const std::string &name, std::vector<int64_t> shape) {
    return store_->load_f32_tensor(source, name, shape).tensor;
  };

  auto &pe = weights_.pre_encode;
  pe.conv0_w = weight("enc.pre_encode.conv.0.weight", {d, hp.enc_feat_in, hp.enc_subs_kernel_size});
  pe.conv0_b = f32("enc.pre_encode.conv.0.bias", {d});
  pe.conv2_w = weight("enc.pre_encode.conv.2.weight", {d, d, hp.enc_subs_kernel_size});
  pe.conv2_b = f32("enc.pre_encode.conv.2.bias", {d});

  weights_.blocks.resize(static_cast<size_t>(hp.enc_n_layers));
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &b = weights_.blocks[static_cast<size_t>(i)];
    const auto w = [&](const char *fmt, std::vector<int64_t> shape) {
      return weight(lname(fmt, i), std::move(shape));
    };
    const auto v = [&](const char *fmt, std::vector<int64_t> shape) {
      return f32(lname(fmt, i), std::move(shape));
    };
    b.norm_ff1_w = v("enc.blocks.%d.norm_ff1.weight", {d});
    b.norm_ff1_b = v("enc.blocks.%d.norm_ff1.bias", {d});
    b.ff1_lin1_w = w("enc.blocks.%d.ff1.linear1.weight", {dff, d});
    b.ff1_lin1_b = v("enc.blocks.%d.ff1.linear1.bias", {dff});
    b.ff1_lin2_w = w("enc.blocks.%d.ff1.linear2.weight", {d, dff});
    b.ff1_lin2_b = v("enc.blocks.%d.ff1.linear2.bias", {d});

    b.norm_conv_w = v("enc.blocks.%d.norm_conv.weight", {d});
    b.norm_conv_b = v("enc.blocks.%d.norm_conv.bias", {d});
    b.conv_pw1_w = w("enc.blocks.%d.conv.pointwise1.weight", {2 * d, d, 1});
    b.conv_pw1_b = v("enc.blocks.%d.conv.pointwise1.bias", {2 * d});
    b.conv_dw_w = w("enc.blocks.%d.conv.depthwise.weight", {d, 1, hp.enc_conv_kernel});
    b.conv_dw_b = v("enc.blocks.%d.conv.depthwise.bias", {d});
    b.conv_ln_w = v("enc.blocks.%d.conv.ln.weight", {d});
    b.conv_ln_b = v("enc.blocks.%d.conv.ln.bias", {d});
    b.conv_pw2_w = w("enc.blocks.%d.conv.pointwise2.weight", {d, d, 1});
    b.conv_pw2_b = v("enc.blocks.%d.conv.pointwise2.bias", {d});

    b.norm_attn_w = v("enc.blocks.%d.norm_attn.weight", {d});
    b.norm_attn_b = v("enc.blocks.%d.norm_attn.bias", {d});
    b.attn_q_w = w("enc.blocks.%d.attn.linear_q.weight", {d, d});
    b.attn_q_b = v("enc.blocks.%d.attn.linear_q.bias", {d});
    b.attn_k_w = w("enc.blocks.%d.attn.linear_k.weight", {d, d});
    b.attn_k_b = v("enc.blocks.%d.attn.linear_k.bias", {d});
    b.attn_v_w = w("enc.blocks.%d.attn.linear_v.weight", {d, d});
    b.attn_v_b = v("enc.blocks.%d.attn.linear_v.bias", {d});
    b.attn_out_w = w("enc.blocks.%d.attn.linear_out.weight", {d, d});
    b.attn_out_b = v("enc.blocks.%d.attn.linear_out.bias", {d});

    b.norm_ff2_w = v("enc.blocks.%d.norm_ff2.weight", {d});
    b.norm_ff2_b = v("enc.blocks.%d.norm_ff2.bias", {d});
    b.ff2_lin1_w = w("enc.blocks.%d.ff2.linear1.weight", {dff, d});
    b.ff2_lin1_b = v("enc.blocks.%d.ff2.linear1.bias", {dff});
    b.ff2_lin2_w = w("enc.blocks.%d.ff2.linear2.weight", {d, dff});
    b.ff2_lin2_b = v("enc.blocks.%d.ff2.linear2.bias", {d});

    b.norm_out_w = v("enc.blocks.%d.norm_out.weight", {d});
    b.norm_out_b = v("enc.blocks.%d.norm_out.bias", {d});
  }

  store_->upload();
  assets_->source->release_storage();

  policy_ = resolve_gigaam_graph_policy(ggml_backend_name(backend_));

  // ----- frontend: framework MelExtractor, the arch's recipe -----
  audio::FrontendSpec spec;
  spec.kind = audio::FrontendKind::MelSpectrogram;
  spec.sample_rate = hp.fe_sample_rate;
  spec.num_mels = hp.fe_num_mels;
  spec.n_fft = hp.fe_n_fft;
  spec.win_length = hp.fe_win_length;
  spec.hop_length = hp.fe_hop_length;
  spec.pre_emphasis = 0.0f; // the arch never applies stt.frontend.pre_emphasis
  spec.dither = 0.0f;       // nor stt.frontend.dither
  spec.f_min = hp.fe_f_min; // unused: the filterbank is supplied
  spec.f_max = hp.fe_f_max;
  spec.pad_mode = audio::PadMode::Constant; // centred zero-pad framing; see the header
  spec.window_type = audio::WindowType::Custom;
  spec.normalize_mode = audio::NormalizeMode::None;
  spec.log_clamp_min = kLogClampMin; // log(max(x, 1e-9)); the upper clamp is applied after
  spec.filterbank = assets_->mel_filterbank; // row-major [n_mels, n_fft/2+1], HTK, baked
  spec.window = assets_->window;             // [win_length] periodic Hann, baked
  mel_.emplace(spec);
  mel_frame_offset_ = (hp.fe_n_fft / 2) / hp.fe_hop_length;
}

GigaamRuntime::~GigaamRuntime() = default;

int32_t GigaamRuntime::sample_rate() const noexcept {
  return assets_ ? assets_->hparams.fe_sample_rate : 16000;
}

std::vector<float> GigaamRuntime::to_model_pcm(const runtime::AudioBuffer &audio) const {
  const auto &hp = assets_->hparams;
  if (audio.sample_rate <= 0 || audio.channels <= 0) {
    throw std::invalid_argument("gigaam: audio needs a positive sample rate and channel count");
  }
  if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
    throw std::invalid_argument("gigaam: interleaved sample count is not a multiple of the channel count");
  }
  std::vector<float> pcm = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
      audio.samples, audio.sample_rate, audio.channels, hp.fe_sample_rate);
  if (pcm.size() < static_cast<size_t>(hp.fe_win_length)) {
    throw std::invalid_argument(
        "gigaam: audio is " + std::to_string(pcm.size()) + " samples at " +
        std::to_string(hp.fe_sample_rate) + " Hz; at least " + std::to_string(hp.fe_win_length) +
        " (one analysis window) are required");
  }
  return pcm;
}

void GigaamRuntime::warn_if_past_window(size_t n_samples) const {
  const int sr = assets_->hparams.fe_sample_rate;
  if (sr <= 0) {
    return;
  }
  const int64_t audio_ms = static_cast<int64_t>(n_samples) * 1000 / sr;
  if (audio_ms > kSafeAudioMs) {
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "audio is %.1f s, beyond the ~%d s window this model was trained on; "
                  "transcription may be degraded. Split long audio into <=%d s segments.",
                  static_cast<double>(audio_ms) / 1000.0, kSafeSeconds, kSafeSeconds);
    engine::debug::log_message(engine::debug::LogLevel::Warning, kTag, msg);
  }
}

std::vector<float> GigaamRuntime::log_mel(const std::vector<float> &pcm, int &n_frames) const {
  const auto &hp = assets_->hparams;
  if (pcm.size() < static_cast<size_t>(hp.fe_win_length)) {
    throw std::invalid_argument("gigaam: audio shorter than one analysis window");
  }
  const int T = static_cast<int>((pcm.size() - static_cast<size_t>(hp.fe_win_length)) /
                                 static_cast<size_t>(hp.fe_hop_length)) +
                1;

  std::vector<float> full;
  int n_mels = 0;
  int n_full = 0;
  if (!mel_->compute(pcm.data(), pcm.size(), full, n_mels, n_full,
                     std::max(1, execution_context_.config().threads))) {
    fail("mel extraction failed");
  }
  if (n_mels != hp.fe_num_mels || mel_frame_offset_ + T > n_full) {
    fail("mel produced " + std::to_string(n_mels) + " x " + std::to_string(n_full) +
         ", expected " + std::to_string(hp.fe_num_mels) + " x >= " +
         std::to_string(mel_frame_offset_ + T));
  }

  // Slice the arch's frames out of each mel row (mel-major in, mel-major
  // out) and apply the arch's upper clamp: log(clamp(x, 1e-9, 1e9)).
  const float log_max = std::log(kLogClampMax);
  std::vector<float> out(static_cast<size_t>(n_mels) * static_cast<size_t>(T));
  for (int m = 0; m < n_mels; ++m) {
    const float *src = full.data() + static_cast<size_t>(m) * static_cast<size_t>(n_full) +
                       static_cast<size_t>(mel_frame_offset_);
    float *dst = out.data() + static_cast<size_t>(m) * static_cast<size_t>(T);
    for (int t = 0; t < T; ++t) {
      dst[t] = std::min(src[t], log_max);
    }
  }
  n_frames = T;
  return out;
}

std::vector<float> GigaamRuntime::encode(const std::vector<float> &mel, int T_mel,
                                         const std::vector<int> &n_frames, int &T_enc) {
  const auto &hp = assets_->hparams;
  const int n_batch = static_cast<int>(n_frames.size());
  if (n_batch <= 0 || T_mel <= 0 ||
      mel.size() != static_cast<size_t>(n_batch) * hp.fe_num_mels * static_cast<size_t>(T_mel)) {
    fail("encoder input does not match the batch geometry");
  }
  bool var_len = false;
  for (const int nf : n_frames) {
    var_len = var_len || (nf != T_mel);
  }

  GraphRun run;
  ggml_init_params params{gigaam_encoder_context_bytes(T_mel), nullptr, /*no_alloc=*/true};
  run.ctx = ggml_init(params);
  if (run.ctx == nullptr) {
    fail("failed to init the encoder compute context");
  }
  GigaamEncoderBuild eb =
      build_gigaam_encoder_graph(run.ctx, weights_, hp, T_mel, policy_, n_batch, var_len);
  if (eb.mel_in == nullptr || eb.encoded == nullptr || eb.graph == nullptr) {
    fail("encoder graph build failed");
  }
  run.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (run.gallocr == nullptr || !ggml_gallocr_alloc_graph(run.gallocr, eb.graph)) {
    fail("encoder graph allocation failed (out of memory); split long audio into "
         "shorter segments");
  }

  // Every input is uploaded here, after allocation and before the compute:
  // the graph is built per run, so no value has to survive an earlier one.
  ggml_backend_tensor_set(eb.mel_in, mel.data(), 0, mel.size() * sizeof(float));
  T_enc = eb.T_enc;
  {
    std::vector<int32_t> pos(static_cast<size_t>(T_enc));
    for (int i = 0; i < T_enc; ++i) {
      pos[static_cast<size_t>(i)] = i;
    }
    ggml_backend_tensor_set(eb.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
  }
  if (var_len) {
    // The arch's masks: key padding 0 / -1e30 (finite, so fully padded rows
    // stay NaN-free), valid frames 1 / 0, and one masked-subsampling mask per
    // conv-ReLU stage.
    std::vector<int> real_tenc(static_cast<size_t>(n_batch));
    for (int b = 0; b < n_batch; ++b) {
      real_tenc[static_cast<size_t>(b)] =
          std::min(gigaam_encoder_frames(n_frames[static_cast<size_t>(b)]), T_enc);
    }
    {
      std::vector<float> buf(static_cast<size_t>(T_enc) * n_batch);
      for (int b = 0; b < n_batch; ++b) {
        for (int k = 0; k < T_enc; ++k) {
          buf[static_cast<size_t>(b) * T_enc + k] =
              (k < real_tenc[static_cast<size_t>(b)]) ? 0.0f : -1e30f;
        }
      }
      ggml_backend_tensor_set(eb.attn_pad_mask, buf.data(), 0, buf.size() * sizeof(float));
    }
    {
      std::vector<float> buf(static_cast<size_t>(T_enc) * n_batch);
      for (int b = 0; b < n_batch; ++b) {
        for (int t = 0; t < T_enc; ++t) {
          buf[static_cast<size_t>(b) * T_enc + t] =
              (t < real_tenc[static_cast<size_t>(b)]) ? 1.0f : 0.0f;
        }
      }
      ggml_backend_tensor_set(eb.conv_pad_mask, buf.data(), 0, buf.size() * sizeof(float));
    }
    const auto fill_pe_mask = [&](ggml_tensor *mask, int n_down) {
      const int H = static_cast<int>(mask->ne[0]);
      std::vector<float> mb(static_cast<size_t>(H) * n_batch, 0.0f);
      for (int b = 0; b < n_batch; ++b) {
        int v = n_frames[static_cast<size_t>(b)];
        for (int k = 0; k < n_down; ++k) {
          v = gigaam_pre_encode_t_out(v);
        }
        v = std::min(v, H);
        for (int t = 0; t < v; ++t) {
          mb[static_cast<size_t>(b) * H + t] = 1.0f;
        }
      }
      ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
    };
    fill_pe_mask(eb.pre_encode_mask_s1, 1);
    fill_pe_mask(eb.pre_encode_mask_s2, 2);
  }

  core::set_backend_threads(backend_, std::max(1, execution_context_.config().threads));
  if (core::compute_backend_graph(backend_, eb.graph) != GGML_STATUS_SUCCESS) {
    fail("encoder compute failed");
  }
  ggml_backend_synchronize(backend_);

  // Full read of [d_model, T_enc, n_batch], host-sliced per utterance (the
  // arch avoids non-zero-offset reads: unreliable across backends).
  std::vector<float> enc_host(static_cast<size_t>(hp.enc_d_model) * static_cast<size_t>(T_enc) *
                              static_cast<size_t>(n_batch));
  ggml_backend_tensor_get(eb.encoded, enc_host.data(), 0, enc_host.size() * sizeof(float));
  return enc_host;
}

GigaamTranscription GigaamRuntime::decode_utterance(const float *encoded, int T_enc,
                                                    const runtime::RunControl &control) const {
  const auto &hp = assets_->hparams;
  std::vector<int32_t> tokens;
  std::vector<int> frames;
  const GigaamDecodePoll poll = [&control](int done, int total) {
    control.emit_progress("decode", done, total);
  };
  if (hp.head_kind == GigaamHeadKind::Rnnt) {
    decode_rnnt_greedy(assets_->host_decoder, hp, encoded, T_enc, kGigaamMaxSymbolsPerStep, tokens,
                       frames, poll);
  } else {
    decode_ctc_greedy(assets_->host_decoder, hp, encoded, T_enc, tokens, frames, poll);
  }

  GigaamTranscription out;
  out.raw_text = assets_->decode(tokens);
  out.text = out.raw_text;
  // SentencePiece convention: the first piece's leading U+2581 decodes to a
  // space; trim exactly one (the arch's post-decode).
  if (!out.text.empty() && out.text.front() == ' ') {
    out.text.erase(out.text.begin());
  }

  // One encoder frame = subsampling * hop samples (40 ms for every variant).
  const int64_t frame_ms = static_cast<int64_t>(hp.enc_subsampling_factor) * hp.fe_hop_length *
                           1000 / hp.fe_sample_rate;
  out.tokens.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    GigaamToken token;
    token.id = tokens[i];
    token.piece = assets_->piece(tokens[i]);
    token.t0_ms = static_cast<int64_t>(frames[i]) * frame_ms;
    token.t1_ms = token.t0_ms + frame_ms;
    out.tokens.push_back(std::move(token));
  }
  return out;
}

GigaamTranscription GigaamRuntime::transcribe(const runtime::AudioBuffer &audio,
                                              const runtime::RunControl &control) {
  // Abort check at the top of the run, as the arch's run().
  control.emit_progress("mel", 0, 1);
  const std::vector<float> pcm = to_model_pcm(audio);
  warn_if_past_window(pcm.size());

  int n_frames = 0;
  const std::vector<float> mel = log_mel(pcm, n_frames);

  control.emit_progress("encoder", 0, 1);
  int T_enc = 0;
  const std::vector<float> encoded = encode(mel, n_frames, {n_frames}, T_enc);

  control.emit_progress("decode", 0, T_enc);
  return decode_utterance(encoded.data(), T_enc, control);
}

std::vector<GigaamTranscription>
GigaamRuntime::transcribe_batch(const std::vector<const runtime::AudioBuffer *> &audios,
                                const runtime::RunControl &control) {
  const auto &hp = assets_->hparams;
  const int n = static_cast<int>(audios.size());
  if (n == 0) {
    return {};
  }

  // Per-utterance mels (validated first: any unusable input sends the whole
  // call down the caller's per-utterance path, as in the arch).
  std::vector<std::vector<float>> pcms(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (audios[static_cast<size_t>(i)] == nullptr) {
      throw std::invalid_argument("gigaam: batch entry without audio");
    }
    pcms[static_cast<size_t>(i)] = to_model_pcm(*audios[static_cast<size_t>(i)]);
  }
  control.emit_progress("mel", 0, n);
  std::vector<std::vector<float>> mels(static_cast<size_t>(n));
  std::vector<int> nf(static_cast<size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    mels[static_cast<size_t>(i)] = log_mel(pcms[static_cast<size_t>(i)], nf[static_cast<size_t>(i)]);
    control.emit_progress("mel", i + 1, n);
  }
  for (int i = 0; i < n; ++i) {
    warn_if_past_window(pcms[static_cast<size_t>(i)].size());
  }
  pcms.clear();

  // Pack [n][n_mels][T_max], zero-padding each utterance along time.
  int T_max = 0;
  for (const int f : nf) {
    T_max = std::max(T_max, f);
  }
  const int n_mels = hp.fe_num_mels;
  const size_t per = static_cast<size_t>(n_mels) * static_cast<size_t>(T_max);
  std::vector<float> packed(per * static_cast<size_t>(n), 0.0f);
  for (int b = 0; b < n; ++b) {
    const int len = nf[static_cast<size_t>(b)];
    const float *src = mels[static_cast<size_t>(b)].data();
    float *dst = packed.data() + static_cast<size_t>(b) * per;
    for (int m = 0; m < n_mels; ++m) {
      std::copy(src + static_cast<size_t>(m) * len, src + static_cast<size_t>(m) * len + len,
                dst + static_cast<size_t>(m) * T_max);
    }
  }
  mels.clear();

  control.emit_progress("encoder", 0, 1);
  int T_enc = 0;
  const std::vector<float> encoded = encode(packed, T_max, nf, T_enc);

  // Decode each utterance's valid frames off its slice.
  const size_t utt_elems = static_cast<size_t>(hp.enc_d_model) * static_cast<size_t>(T_enc);
  std::vector<GigaamTranscription> results;
  results.reserve(static_cast<size_t>(n));
  for (int b = 0; b < n; ++b) {
    control.emit_progress("decode", b, n);
    const int real = std::min(gigaam_encoder_frames(nf[static_cast<size_t>(b)]), T_enc);
    results.push_back(
        decode_utterance(encoded.data() + static_cast<size_t>(b) * utt_elems, real, control));
  }
  return results;
}

} // namespace engine::models::gigaam
