#pragma once

// engine/models/canary_qwen/mel_frontend.h - the NeMo log-mel frontend the
// canary_qwen arch ran (src/runtime/transcribe-mel.{h,cpp}, MelFrontend),
// ported verbatim.
//
// Why not engine::audio::MelExtractor: its per-feature normalization is
// numerically different from the arch's - it divides by sqrt(var + 1e-5)
// where NeMo (and the arch) divide by (sqrt(var) + 1e-5), and it forms the
// mean with a float reciprocal instead of a double division. On a 32-block
// FastConformer feeding a greedy LLM decode that is enough to change tokens,
// so the package carries the arch's frontend bit for bit (fp64 radix-2 STFT
// for n_fft = 512, the scalar fused filterbank + log off Apple, vDSP + BLAS on
// Apple, per-feature unbiased normalization).
//
// LAYOUT: compute() returns mel-major data (m * n_frames + t), i.e. row-major
// [n_mels, n_frames]. That is exactly the canary_qwen encoder input, a ggml
// tensor with ne = [n_frames, n_mels] (time fastest) - no transpose.

#include <cstddef>
#include <string>
#include <vector>

namespace engine::models::canary_qwen {

// The arch's MelConfig. Strings keep the arch's spelling (they come straight
// from the stt.frontend.* KVs).
struct NemoMelConfig {
  int sample_rate = 16000;
  int num_mels = 128;
  int n_fft = 512;
  int win_length = 400;
  int hop_length = 160;
  float pre_emphasis = 0.97f;  // 0 disables
  float f_min = 0.0f;
  float f_max = 8000.0f;
  std::string pad_mode = "reflect";          // reflect | constant | none
  std::string window_type = "hann_symmetric"; // hann_symmetric | hann_periodic
  std::string normalize = "per_feature";     // per_feature | per_utterance | global | none
  float global_log_mel_max = 1.5f;
  std::vector<float> filterbank;  // optional [num_mels * (n_fft/2+1)]
  float log_clamp_min = 0.0f;
  std::vector<float> window;      // optional [win_length]
  bool nemo_seq_len_ceil = false;
};

class NemoMelFrontend {
public:
  explicit NemoMelFrontend(const NemoMelConfig &cfg);

  // pcm: 16 kHz mono float32. Returns false exactly where the arch returned
  // TRANSCRIBE_ERR_INVALID_ARG (null pcm, too few samples for two frames or
  // for the reflect pad). Thread-safe (const).
  bool compute(const float *pcm, size_t n_samples, std::vector<float> &out_mel,
               int &out_n_mels, int &out_n_frames, int n_threads = 0) const;

  int num_mels() const { return cfg_.num_mels; }
  int n_frames_for(size_t n_samples) const;
  const NemoMelConfig &config() const { return cfg_; }

private:
  NemoMelConfig cfg_;
  int n_freq_ = 0;
  std::vector<double> window_;
  std::vector<float> mel_fb_;
  std::vector<int> fb_begin_;
  std::vector<int> fb_end_;
  std::vector<float> cos_lut_;
  std::vector<float> sin_lut_;
};

} // namespace engine::models::canary_qwen
