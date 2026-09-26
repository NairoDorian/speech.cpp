#pragma once

// engine/models/granite_nar/mel_frontend.h - the log-mel frontend the
// granite_nar arch ran (src/runtime/transcribe-mel.{h,cpp}, MelFrontend),
// ported verbatim. The source is the canary_qwen package's port of the same
// MelFrontend (src/models/canary_qwen/mel_frontend.cpp), copied rather than
// linked so this package builds without canary_qwen in the model set.
//
// Path granite_nar reaches (convert-granite_nar.py writes these KVs): n_fft
// 512 -> fp64 radix-2 STFT, the GGUF's periodic Hann window (400) centred in
// the 512-point frame, reflect pad, the GGUF's librosa-htk filterbank, no
// pre-emphasis, log10(max(x, 1e-10)) and Whisper-style per-utterance
// normalization (clamp to max - 8, (x + 4) / 4, trailing centre frame
// dropped). engine::audio::MelExtractor is not used: this file is the arch's
// arithmetic bit for bit (fp64 FFT, fp64-accumulated sparse filterbank dot,
// fp64 per-utterance max), which a 40-layer bidirectional editor needs.
//
// NOT adopted: the parent's AVX2 filterbank dot (transcribe.cpp 6c767184 /
// 28a3f985, compute_filterbank_dot) - a family-wide mel change outside
// 585b98f7 that speech.cpp's C ABI mel does not carry either.
//
// LAYOUT: compute() returns mel-major data (m * n_frames + t), i.e. row-major
// [n_mels, n_frames]. The granite_nar runtime stacks frame pairs out of it
// (arch encoder.cpp compute_mel_encoder_input).

#include <cstddef>
#include <string>
#include <vector>

namespace engine::models::granite_nar {

// The arch's MelConfig. Strings keep the arch's spelling (they come straight
// from the stt.frontend.* KVs).
struct GraniteMelConfig {
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

class GraniteMelFrontend {
public:
  explicit GraniteMelFrontend(const GraniteMelConfig &cfg);

  // pcm: 16 kHz mono float32. Returns false exactly where the arch returned
  // TRANSCRIBE_ERR_INVALID_ARG (null pcm, too few samples for two frames or
  // for the reflect pad). Thread-safe (const).
  bool compute(const float *pcm, size_t n_samples, std::vector<float> &out_mel,
               int &out_n_mels, int &out_n_frames, int n_threads = 0) const;

  int num_mels() const { return cfg_.num_mels; }
  int n_frames_for(size_t n_samples) const;
  const GraniteMelConfig &config() const { return cfg_; }

private:
  GraniteMelConfig cfg_;
  int n_freq_ = 0;
  std::vector<double> window_;
  std::vector<float> mel_fb_;
  std::vector<int> fb_begin_;
  std::vector<int> fb_end_;
  std::vector<float> cos_lut_;
  std::vector<float> sin_lut_;
};

} // namespace engine::models::granite_nar
