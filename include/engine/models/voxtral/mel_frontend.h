#pragma once

// engine/models/voxtral/mel_frontend.h - the Whisper log-mel frontend the
// voxtral arch ran (src/runtime/transcribe-mel.{h,cpp}, MelFrontend with
// window_type "hann_periodic" / normalize "per_utterance" / n_fft 400), ported
// verbatim for the path this family reaches.
//
// Why not engine::audio::MelExtractor: the arch's features come from a
// specific fp32 mixed-radix FFT (25-point DFT leaves under four radix-2
// levels, a fp32 twiddle LUT), a fp64-accumulated sparse filterbank dot
// product, log10(max(x, 1e-10)), and a per-utterance clamp to max - 8 then
// (x + 4) / 4 over all but the trailing center frame. A 32-block encoder
// feeding a greedy LLM decode turns small feature drift into different
// tokens, so the package carries the arch's arithmetic bit for bit.
//
// Scope: only the non-power-of-two n_fft path (the one Voxtral's n_fft = 400
// takes) and per_utterance normalization are ported; the constructor rejects
// any other configuration (assets.cpp already refuses such a GGUF at load).
//
// LAYOUT: compute() returns mel-major data (m * n_frames + t), i.e. row-major
// [n_mels, n_frames] - what the arch's MelFrontend returned. The encoder input
// is a ggml [n_mels, n_frames] tensor (n_mels fastest), so the runtime
// transposes each 30 s chunk to frame-major before upload, exactly as the arch
// did (see speechcpp mel-layout note in runtime.cpp).

#include <cstddef>
#include <string>
#include <vector>

namespace engine::models::voxtral {

// The arch's MelConfig fields this path reads. Strings keep the arch's
// spelling (they come straight from the stt.frontend.* KVs).
struct WhisperMelConfig {
  int sample_rate = 16000;
  int num_mels = 128;
  int n_fft = 400;
  int win_length = 400;
  int hop_length = 160;
  float pre_emphasis = 0.0f;               // 0 disables
  float f_min = 0.0f;
  float f_max = 8000.0f;
  std::string pad_mode = "reflect";        // reflect | constant
  std::string window_type = "hann_periodic";  // hann_periodic | hann_symmetric
  std::string normalize = "per_utterance";
  std::vector<float> filterbank;  // optional [num_mels * (n_fft/2+1)], mel-major
  std::vector<float> window;      // optional [win_length]
};

class WhisperMelFrontend {
public:
  // Throws std::runtime_error for a configuration outside the ported path.
  explicit WhisperMelFrontend(const WhisperMelConfig &cfg);

  // pcm: mono float32 at cfg.sample_rate. Returns false exactly where the
  // arch returned TRANSCRIBE_ERR_INVALID_ARG (null pcm, fewer samples than
  // two frames or the reflect pad need). Thread-safe (const). `n_threads`
  // only partitions frames across threads; it never changes a value.
  bool compute(const float *pcm, size_t n_samples, std::vector<float> &out_mel, int &out_n_mels,
               int &out_n_frames, int n_threads) const;

  int num_mels() const noexcept { return cfg_.num_mels; }
  // floor(n_samples / hop) + 1 (before the per-utterance path drops one frame).
  int n_frames_for(size_t n_samples) const noexcept;

private:
  WhisperMelConfig cfg_;
  int n_freq_ = 0;
  std::vector<double> window_;  // [n_fft], window zero-padded (centered)
  std::vector<float> mel_fb_;   // [num_mels * n_freq] row-major
  std::vector<int> fb_begin_;   // nonzero support start per mel band
  std::vector<int> fb_end_;     // nonzero support end per mel band
  std::vector<float> cos_lut_;  // [n_fft] fp32 twiddles for the mixed-radix FFT
  std::vector<float> sin_lut_;
};

}  // namespace engine::models::voxtral
