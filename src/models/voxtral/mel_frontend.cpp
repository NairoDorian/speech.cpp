// engine/models/voxtral/mel_frontend.cpp - the arch's Whisper log-mel
// frontend (src/runtime/transcribe-mel.cpp, MelFrontend), ported verbatim for
// the path Voxtral reaches. See mel_frontend.h for scope and rationale.
//
// Pipeline (transcribe-mel.cpp, non-pow2 branch):
//   1. optional pre-emphasis, reflect (or zero) pad by n_fft/2, all fp32;
//   2. per frame (strided across threads): window * frame -> fp32 mixed-radix
//      FFT -> |X|^2 -> sparse filterbank dot product accumulated in fp64 ->
//      log10(max(sum, 1e-10)), written mel-major;
//   3. per-utterance normalization over the first n_frames - 1 frames:
//      clamp to (global max - 8), then (x + 4) / 4; the trailing center-pad
//      frame is dropped (output has n_samples / hop frames).

#include "engine/models/voxtral/mel_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace engine::models::voxtral {

namespace {

constexpr double kSlaneyFsp = 200.0 / 3.0;
constexpr double kSlaneyMinLogHz = 1000.0;

double slaney_hz_to_mel(double hz) {
  const double min_log_mel = kSlaneyMinLogHz / kSlaneyFsp;
  const double logstep = std::log(6.4) / 27.0;
  if (hz < kSlaneyMinLogHz) {
    return hz / kSlaneyFsp;
  }
  return min_log_mel + std::log(hz / kSlaneyMinLogHz) / logstep;
}

double slaney_mel_to_hz(double mel) {
  const double min_log_mel = kSlaneyMinLogHz / kSlaneyFsp;
  const double logstep = std::log(6.4) / 27.0;
  if (mel < min_log_mel) {
    return mel * kSlaneyFsp;
  }
  return kSlaneyMinLogHz * std::exp(logstep * (mel - min_log_mel));
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney")
// (transcribe-mel.cpp build_mel_filterbank_slaney). Only used when the GGUF
// carries no frontend.mel_filterbank.
void build_mel_filterbank_slaney(int sr, int n_fft, int n_mels, double fmin, double fmax,
                                 std::vector<float> &out) {
  const int n_freq = n_fft / 2 + 1;
  out.assign(static_cast<size_t>(n_mels) * n_freq, 0.0f);
  std::vector<double> fft_freqs(static_cast<size_t>(n_freq));
  for (int k = 0; k < n_freq; ++k) {
    fft_freqs[static_cast<size_t>(k)] = static_cast<double>(sr) * k / static_cast<double>(n_fft);
  }
  const double mel_min = slaney_hz_to_mel(fmin);
  const double mel_max = slaney_hz_to_mel(fmax);
  std::vector<double> hz_freqs(static_cast<size_t>(n_mels + 2));
  for (int m = 0; m < n_mels + 2; ++m) {
    const double mel = mel_min + (mel_max - mel_min) * m / static_cast<double>(n_mels + 1);
    hz_freqs[static_cast<size_t>(m)] = slaney_mel_to_hz(mel);
  }
  std::vector<double> fdiff(static_cast<size_t>(n_mels + 1));
  for (int m = 0; m < n_mels + 1; ++m) {
    fdiff[static_cast<size_t>(m)] = hz_freqs[static_cast<size_t>(m + 1)] - hz_freqs[static_cast<size_t>(m)];
  }
  for (int m = 0; m < n_mels; ++m) {
    const double enorm = 2.0 / (hz_freqs[static_cast<size_t>(m + 2)] - hz_freqs[static_cast<size_t>(m)]);
    for (int k = 0; k < n_freq; ++k) {
      const double lower = (fft_freqs[static_cast<size_t>(k)] - hz_freqs[static_cast<size_t>(m)]) /
                           fdiff[static_cast<size_t>(m)];
      const double upper = (hz_freqs[static_cast<size_t>(m + 2)] - fft_freqs[static_cast<size_t>(k)]) /
                           fdiff[static_cast<size_t>(m + 1)];
      const double w = std::max(0.0, std::min(lower, upper));
      out[static_cast<size_t>(m) * n_freq + k] = static_cast<float>(w * enorm);
    }
  }
}

// Hann window of length win_length, zero-padded (centered) to n_fft.
void build_hann_window_padded(int win_length, int n_fft, bool periodic, std::vector<double> &out) {
  out.assign(static_cast<size_t>(n_fft), 0.0);
  const int pad_each = (n_fft - win_length) / 2;
  const double denom = periodic ? static_cast<double>(win_length) : static_cast<double>(win_length - 1);
  for (int k = 0; k < win_length; ++k) {
    out[static_cast<size_t>(pad_each + k)] = 0.5 - 0.5 * std::cos(2.0 * M_PI * k / denom);
  }
}

// Naive O(N^2) DFT leaf for the mixed-radix path (odd N).
void dft_naive_f32(const float *in, int N, const float *cos_lut, const float *sin_lut, int lut_size,
                   float *out) {
  const int stride = lut_size / N;
  for (int k = 0; k < N; ++k) {
    float re = 0.0f;
    float im = 0.0f;
    for (int n = 0; n < N; ++n) {
      const int idx = (k * n * stride) % lut_size;
      re += in[n] * cos_lut[idx];
      im -= in[n] * sin_lut[idx];
    }
    out[2 * k] = re;
    out[2 * k + 1] = im;
  }
}

// Cooley-Tukey mixed-radix FFT (whisper.cpp's algorithm): halve while N is
// even, naive DFT on odd leaves. `in` needs 2N floats of scratch, `out` 8N.
void mixed_radix_fft_f32(float *in, int N, const float *cos_lut, const float *sin_lut, int lut_size,
                         float *out) {
  if (N == 1) {
    out[0] = in[0];
    out[1] = 0.0f;
    return;
  }
  if (N & 1) {
    dft_naive_f32(in, N, cos_lut, sin_lut, lut_size, out);
    return;
  }
  const int half_N = N / 2;
  float *even = in + N;
  for (int i = 0; i < half_N; ++i) {
    even[i] = in[2 * i];
  }
  float *even_fft = out + 2 * N;
  mixed_radix_fft_f32(even, half_N, cos_lut, sin_lut, lut_size, even_fft);

  float *odd = even;
  for (int i = 0; i < half_N; ++i) {
    odd[i] = in[2 * i + 1];
  }
  float *odd_fft = even_fft + N;
  mixed_radix_fft_f32(odd, half_N, cos_lut, sin_lut, lut_size, odd_fft);

  const int step = lut_size / N;
  for (int k = 0; k < half_N; ++k) {
    const int idx = k * step;
    const float w_re = cos_lut[idx];
    const float w_im = -sin_lut[idx];
    const float re_odd = odd_fft[2 * k];
    const float im_odd = odd_fft[2 * k + 1];
    out[2 * k] = even_fft[2 * k] + w_re * re_odd - w_im * im_odd;
    out[2 * k + 1] = even_fft[2 * k + 1] + w_re * im_odd + w_im * re_odd;
    out[2 * (k + half_N)] = even_fft[2 * k] - w_re * re_odd + w_im * im_odd;
    out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - w_re * im_odd - w_im * re_odd;
  }
}

}  // namespace

WhisperMelFrontend::WhisperMelFrontend(const WhisperMelConfig &cfg) : cfg_(cfg) {
  const bool n_fft_is_pow2 = (cfg.n_fft > 0) && ((cfg.n_fft & (cfg.n_fft - 1)) == 0);
  if (cfg.n_fft <= 0 || n_fft_is_pow2) {
    throw std::runtime_error("voxtral mel: only the non-power-of-two n_fft (Whisper 400) path is ported");
  }
  if (cfg.normalize != "per_utterance") {
    throw std::runtime_error("voxtral mel: only per_utterance normalization is ported");
  }
  if (cfg.pad_mode != "reflect" && cfg.pad_mode != "constant") {
    throw std::runtime_error("voxtral mel: pad_mode must be reflect or constant");
  }
  if (cfg.num_mels <= 0 || cfg.hop_length <= 0 || cfg.win_length <= 0 || cfg.win_length > cfg.n_fft) {
    throw std::runtime_error("voxtral mel: invalid frontend geometry");
  }
  n_freq_ = cfg.n_fft / 2 + 1;

  if (!cfg.window.empty()) {
    // Checkpoint window, fp32 -> fp64, centered in n_fft (the centered
    // pad modes; pad_mode "none" is not reachable here).
    window_.assign(static_cast<size_t>(cfg.n_fft), 0.0);
    const int left_pad = (cfg.n_fft - cfg.win_length) / 2;
    for (int i = 0; i < cfg.win_length && i < static_cast<int>(cfg.window.size()); ++i) {
      window_[static_cast<size_t>(left_pad + i)] = static_cast<double>(cfg.window[static_cast<size_t>(i)]);
    }
  } else {
    build_hann_window_padded(cfg.win_length, cfg.n_fft, cfg.window_type == "hann_periodic", window_);
  }

  if (!cfg.filterbank.empty()) {
    if (cfg.filterbank.size() != static_cast<size_t>(cfg.num_mels) * static_cast<size_t>(n_freq_)) {
      throw std::runtime_error("voxtral mel: filterbank size does not match num_mels x (n_fft/2+1)");
    }
    mel_fb_ = cfg.filterbank;
  } else {
    build_mel_filterbank_slaney(cfg.sample_rate, cfg.n_fft, cfg.num_mels, static_cast<double>(cfg.f_min),
                                static_cast<double>(cfg.f_max), mel_fb_);
  }

  // Nonzero support per mel band (trailing zeros trimmed first).
  fb_begin_.assign(static_cast<size_t>(cfg.num_mels), 0);
  fb_end_.assign(static_cast<size_t>(cfg.num_mels), 0);
  for (int m = 0; m < cfg.num_mels; ++m) {
    const float *row = mel_fb_.data() + static_cast<size_t>(m) * n_freq_;
    int lo = 0;
    while (lo < n_freq_ && row[lo] == 0.0f) {
      ++lo;
    }
    int hi = n_freq_;
    while (hi > lo && row[hi - 1] == 0.0f) {
      --hi;
    }
    fb_begin_[static_cast<size_t>(m)] = lo;
    fb_end_[static_cast<size_t>(m)] = hi;
  }

  // fp32 twiddle LUT sized to n_fft (computed in fp64, stored fp32).
  cos_lut_.resize(static_cast<size_t>(cfg.n_fft));
  sin_lut_.resize(static_cast<size_t>(cfg.n_fft));
  for (int i = 0; i < cfg.n_fft; ++i) {
    const double theta = 2.0 * M_PI * i / cfg.n_fft;
    cos_lut_[static_cast<size_t>(i)] = static_cast<float>(std::cos(theta));
    sin_lut_[static_cast<size_t>(i)] = static_cast<float>(std::sin(theta));
  }
}

int WhisperMelFrontend::n_frames_for(size_t n_samples) const noexcept {
  return static_cast<int>(n_samples / static_cast<size_t>(cfg_.hop_length)) + 1;
}

bool WhisperMelFrontend::compute(const float *pcm, size_t n_samples, std::vector<float> &out_mel,
                                 int &out_n_mels, int &out_n_frames, int n_threads) const {
  if (pcm == nullptr) {
    return false;
  }
  const int n_fft = cfg_.n_fft;
  const int hop = cfg_.hop_length;
  const int n_mels = cfg_.num_mels;
  const int n_freq = n_freq_;
  const int pad = n_fft / 2;
  const int n_frames = n_frames_for(n_samples);
  const bool use_reflect = (cfg_.pad_mode != "constant");
  if (n_frames < 2 || (use_reflect && n_samples < static_cast<size_t>(pad + 1))) {
    return false;
  }

  // ---- 1. Pad + pre-emphasis (fp32) ----
  std::vector<float> padded(n_samples + 2 * static_cast<size_t>(pad));
  if (cfg_.pre_emphasis != 0.0f) {
    const float alpha = cfg_.pre_emphasis;
    padded[static_cast<size_t>(pad)] = pcm[0];
    for (size_t i = 1; i < n_samples; ++i) {
      padded[static_cast<size_t>(pad) + i] = pcm[i] - alpha * pcm[i - 1];
    }
  } else {
    std::memcpy(padded.data() + pad, pcm, n_samples * sizeof(float));
  }
  if (use_reflect) {
    for (int i = 0; i < pad; ++i) {
      padded[static_cast<size_t>(i)] = padded[static_cast<size_t>(2 * pad - i)];
    }
    for (int i = 0; i < pad; ++i) {
      padded[static_cast<size_t>(pad) + n_samples + static_cast<size_t>(i)] =
          padded[static_cast<size_t>(pad) + n_samples - 2 - static_cast<size_t>(i)];
    }
  } else {
    std::memset(padded.data(), 0, static_cast<size_t>(pad) * sizeof(float));
    std::memset(padded.data() + pad + n_samples, 0, static_cast<size_t>(pad) * sizeof(float));
  }
  std::vector<float> window_f32(static_cast<size_t>(n_fft));
  for (int i = 0; i < n_fft; ++i) {
    window_f32[static_cast<size_t>(i)] = static_cast<float>(window_[static_cast<size_t>(i)]);
  }

  // ---- 2. STFT + filterbank + log10 -> log_mel[n_mels, n_frames] ----
  std::vector<float> log_mel(static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
  int stft_threads = n_threads;
  if (stft_threads <= 0) {
    stft_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  }
  if (stft_threads > n_frames) {
    stft_threads = std::max(1, n_frames);
  }

  auto worker = [&](int tid) {
    std::vector<float> fft_in(2 * static_cast<size_t>(n_fft), 0.0f);
    std::vector<float> fft_out(8 * static_cast<size_t>(n_fft), 0.0f);
    std::vector<float> power(static_cast<size_t>(n_freq));
    for (int t = tid; t < n_frames; t += stft_threads) {
      const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
      for (int n = 0; n < n_fft; ++n) {
        fft_in[static_cast<size_t>(n)] = padded[start + static_cast<size_t>(n)] * window_f32[static_cast<size_t>(n)];
      }
      mixed_radix_fft_f32(fft_in.data(), n_fft, cos_lut_.data(), sin_lut_.data(),
                          static_cast<int>(cos_lut_.size()), fft_out.data());
      for (int k = 0; k < n_freq; ++k) {
        const float re = fft_out[static_cast<size_t>(2 * k)];
        const float im = fft_out[static_cast<size_t>(2 * k + 1)];
        power[static_cast<size_t>(k)] = re * re + im * im;
      }
      for (int m = 0; m < n_mels; ++m) {
        const float *fb_row = mel_fb_.data() + static_cast<size_t>(m) * n_freq;
        // The band's nonzero span, starting on the same 4-aligned boundary the
        // dense loop used: every group holding a nonzero keeps its exact
        // accumulation order; skipped elements are all 0 (bit-identical).
        const int k_end = fb_end_[static_cast<size_t>(m)];
        double sum = 0.0;
        int k = (fb_begin_[static_cast<size_t>(m)] / 4) * 4;
        for (; k < n_freq - 3 && k < k_end; k += 4) {
          sum += static_cast<double>(fb_row[k]) * static_cast<double>(power[static_cast<size_t>(k)]) +
                 static_cast<double>(fb_row[k + 1]) * static_cast<double>(power[static_cast<size_t>(k + 1)]) +
                 static_cast<double>(fb_row[k + 2]) * static_cast<double>(power[static_cast<size_t>(k + 2)]) +
                 static_cast<double>(fb_row[k + 3]) * static_cast<double>(power[static_cast<size_t>(k + 3)]);
        }
        for (; k < n_freq && k < k_end; ++k) {
          sum += static_cast<double>(fb_row[k]) * static_cast<double>(power[static_cast<size_t>(k)]);
        }
        if (sum < 1.0e-10) {
          sum = 1.0e-10;
        }
        log_mel[static_cast<size_t>(m) * n_frames + t] = static_cast<float>(std::log10(sum));
      }
    }
  };
  if (stft_threads <= 1) {
    worker(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(stft_threads - 1));
    for (int tid = 1; tid < stft_threads; ++tid) {
      pool.emplace_back(worker, tid);
    }
    worker(0);
    for (auto &th : pool) {
      th.join();
    }
  }

  // ---- 3. Per-utterance normalization; drop the trailing center frame ----
  const int n_out = n_frames - 1;
  if (n_out <= 0) {
    return false;
  }
  double global_max = -std::numeric_limits<double>::infinity();
  for (int m = 0; m < n_mels; ++m) {
    const float *src = log_mel.data() + static_cast<size_t>(m) * n_frames;
    for (int t = 0; t < n_out; ++t) {
      const double v = static_cast<double>(src[t]);
      if (v > global_max) {
        global_max = v;
      }
    }
  }
  const double floor_val = global_max - 8.0;
  out_mel.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(n_out));
  for (int m = 0; m < n_mels; ++m) {
    const float *src = log_mel.data() + static_cast<size_t>(m) * n_frames;
    float *dst = out_mel.data() + static_cast<size_t>(m) * n_out;
    for (int t = 0; t < n_out; ++t) {
      double v = static_cast<double>(src[t]);
      if (v < floor_val) {
        v = floor_val;
      }
      dst[t] = static_cast<float>((v + 4.0) / 4.0);
    }
  }
  out_n_mels = n_mels;
  out_n_frames = n_out;
  return true;
}

}  // namespace engine::models::voxtral
