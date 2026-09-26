// engine/models/canary_qwen/mel_frontend.cpp - the arch's NeMo log-mel
// frontend (src/runtime/transcribe-mel.cpp, MelFrontend), ported verbatim so
// the canary_qwen engine package sees the exact features the arch saw. See
// mel_frontend.h for why engine::audio::MelExtractor is not used.
//
// Constants / ordering (NeMo AudioToMelSpectrogramPreprocessor):
//   * pre-emphasis alpha (0.97; 0 disables), reflect pad by n_fft/2
//   * STFT fp64 radix-2 for pow2 n_fft (canary_qwen: 512), fp32 mixed radix
//     otherwise; symmetric Hann zero-padded to n_fft (or the GGUF's window)
//   * power = re^2 + im^2; Slaney filterbank (or the GGUF's); log(x + 2^-24)
//   * per-feature normalize: unbiased variance, (x - mean) / (std + 1e-5)
//
// The build never defined TRANSCRIBE_HAS_BLAS, so off Apple the arch ran the
// scalar fused filterbank + log; on Apple it ran vDSP + cblas_sgemm. Both are
// reproduced (the Apple branch needs Accelerate, which the arch already
// linked into the same library).

#include "engine/models/canary_qwen/mel_frontend.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace engine::models::canary_qwen {

namespace {

constexpr float kLogEps = 5.9604644775390625e-08f; // 2^-24 (NeMo log_zero_guard_value)
constexpr float kNormEps = 1.0e-05f;               // NeMo CONSTANT

constexpr double kSlaneyFsp = 200.0 / 3.0;
constexpr double kSlaneyMinLogHz = 1000.0;

inline double slaney_hz_to_mel(double hz) {
  const double min_log_mel = kSlaneyMinLogHz / kSlaneyFsp;
  const double logstep = std::log(6.4) / 27.0;
  if (hz < kSlaneyMinLogHz) {
    return hz / kSlaneyFsp;
  }
  return min_log_mel + std::log(hz / kSlaneyMinLogHz) / logstep;
}

inline double slaney_mel_to_hz(double mel) {
  const double min_log_mel = kSlaneyMinLogHz / kSlaneyFsp;
  const double logstep = std::log(6.4) / 27.0;
  if (mel < min_log_mel) {
    return mel * kSlaneyFsp;
  }
  return kSlaneyMinLogHz * std::exp(logstep * (mel - min_log_mel));
}

void build_mel_filterbank_slaney(int sr, int n_fft, int n_mels, double fmin, double fmax,
                                 std::vector<float> &out) {
  const int n_freq = n_fft / 2 + 1;
  out.assign(static_cast<size_t>(n_mels) * n_freq, 0.0f);

  std::vector<double> fft_freqs(n_freq);
  for (int k = 0; k < n_freq; ++k) {
    fft_freqs[k] = static_cast<double>(sr) * k / static_cast<double>(n_fft);
  }

  const double mel_min = slaney_hz_to_mel(fmin);
  const double mel_max = slaney_hz_to_mel(fmax);
  std::vector<double> hz_freqs(n_mels + 2);
  for (int m = 0; m < n_mels + 2; ++m) {
    const double mel = mel_min + (mel_max - mel_min) * m / static_cast<double>(n_mels + 1);
    hz_freqs[m] = slaney_mel_to_hz(mel);
  }

  std::vector<double> fdiff(n_mels + 1);
  for (int m = 0; m < n_mels + 1; ++m) {
    fdiff[m] = hz_freqs[m + 1] - hz_freqs[m];
  }

  for (int m = 0; m < n_mels; ++m) {
    const double enorm = 2.0 / (hz_freqs[m + 2] - hz_freqs[m]);
    for (int k = 0; k < n_freq; ++k) {
      const double lower = (fft_freqs[k] - hz_freqs[m]) / fdiff[m];
      const double upper = (hz_freqs[m + 2] - fft_freqs[k]) / fdiff[m + 1];
      const double w = std::max(0.0, std::min(lower, upper));
      out[static_cast<size_t>(m) * n_freq + k] = static_cast<float>(w * enorm);
    }
  }
}

void build_hann_window_symmetric_padded(int win_length, int n_fft, bool periodic,
                                        std::vector<double> &out) {
  out.assign(n_fft, 0.0);
  const int pad_each = (n_fft - win_length) / 2;
  const double denom =
      periodic ? static_cast<double>(win_length) : static_cast<double>(win_length - 1);
  for (int k = 0; k < win_length; ++k) {
    out[pad_each + k] = 0.5 - 0.5 * std::cos(2.0 * M_PI * k / denom);
  }
}

void dft_naive_f32(const float *in, int N, const float *cos_lut, const float *sin_lut,
                   int lut_size, float *out) {
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

void mixed_radix_fft_f32(float *in, int N, const float *cos_lut, const float *sin_lut,
                         int lut_size, float *out) {
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

#ifndef __APPLE__
void fft_radix2(double *data, int n) {
  int j = 0;
  for (int i = 1; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(data[2 * i], data[2 * j]);
      std::swap(data[2 * i + 1], data[2 * j + 1]);
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * M_PI / static_cast<double>(len);
    const double wlen_re = std::cos(ang);
    const double wlen_im = std::sin(ang);
    for (int i = 0; i < n; i += len) {
      double w_re = 1.0;
      double w_im = 0.0;
      for (int k = 0; k < len / 2; ++k) {
        const int a = 2 * (i + k);
        const int b = 2 * (i + k + len / 2);
        const double u_re = data[a];
        const double u_im = data[a + 1];
        const double v_re = data[b] * w_re - data[b + 1] * w_im;
        const double v_im = data[b] * w_im + data[b + 1] * w_re;
        data[a] = u_re + v_re;
        data[a + 1] = u_im + v_im;
        data[b] = u_re - v_re;
        data[b + 1] = u_im - v_im;
        const double tmp_re = w_re * wlen_re - w_im * wlen_im;
        const double tmp_im = w_re * wlen_im + w_im * wlen_re;
        w_re = tmp_re;
        w_im = tmp_im;
      }
    }
  }
}
#endif

// transcribe::default_n_threads() without the affinity query: the thread
// count only splits frames (each frame is computed by one thread), so it
// never changes a value.
int default_threads() {
  int n = static_cast<int>(std::thread::hardware_concurrency());
  return n < 1 ? 1 : n;
}

} // namespace

NemoMelFrontend::NemoMelFrontend(const NemoMelConfig &cfg) : cfg_(cfg) {
  n_freq_ = cfg.n_fft / 2 + 1;

  if (!cfg.window.empty()) {
    window_.resize(cfg.n_fft, 0.0);
    const int total_pad = cfg.n_fft - cfg.win_length;
    const int left_pad = (cfg.pad_mode == "none") ? 0 : (total_pad / 2);
    for (int i = 0; i < cfg.win_length && i < static_cast<int>(cfg.window.size()); ++i) {
      window_[left_pad + i] = static_cast<double>(cfg.window[i]);
    }
  } else {
    const bool periodic = (cfg.window_type == "hann_periodic");
    build_hann_window_symmetric_padded(cfg.win_length, cfg.n_fft, periodic, window_);
  }

  if (!cfg.filterbank.empty()) {
    mel_fb_ = cfg.filterbank;
  } else {
    build_mel_filterbank_slaney(cfg.sample_rate, cfg.n_fft, cfg.num_mels,
                                static_cast<double>(cfg.f_min), static_cast<double>(cfg.f_max),
                                mel_fb_);
  }

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

  const bool n_fft_is_pow2 = ((cfg.n_fft > 0) && ((cfg.n_fft & (cfg.n_fft - 1)) == 0));
  if (!n_fft_is_pow2) {
    cos_lut_.resize(cfg.n_fft);
    sin_lut_.resize(cfg.n_fft);
    for (int i = 0; i < cfg.n_fft; ++i) {
      const double theta = 2.0 * M_PI * i / cfg.n_fft;
      cos_lut_[i] = static_cast<float>(std::cos(theta));
      sin_lut_[i] = static_cast<float>(std::sin(theta));
    }
  }
}

int NemoMelFrontend::n_frames_for(size_t n_samples) const {
  if (cfg_.pad_mode == "none") {
    if (static_cast<int>(n_samples) < cfg_.win_length) {
      return 0;
    }
    return static_cast<int>((n_samples - static_cast<size_t>(cfg_.win_length)) /
                            static_cast<size_t>(cfg_.hop_length)) +
           1;
  }
  if (cfg_.nemo_seq_len_ceil) {
    return static_cast<int>((n_samples + static_cast<size_t>(cfg_.hop_length) - 1) /
                            static_cast<size_t>(cfg_.hop_length));
  }
  return static_cast<int>(n_samples / static_cast<size_t>(cfg_.hop_length)) + 1;
}

bool NemoMelFrontend::compute(const float *pcm, size_t n_samples, std::vector<float> &out_mel,
                              int &out_n_mels, int &out_n_frames, int n_threads) const {
  if (pcm == nullptr) {
    return false;
  }
  const int n_fft = cfg_.n_fft;
  const int hop = cfg_.hop_length;
  const int n_mels = cfg_.num_mels;
  const int n_freq = n_freq_;
  const int win = cfg_.win_length;
  const int pad = n_fft / 2;
  const bool no_pad = (cfg_.pad_mode == "none");

  const int n_frames = n_frames_for(n_samples);

  const bool use_reflect = (!no_pad && cfg_.pad_mode != "constant");
  if (n_frames < (no_pad ? 1 : 2) || (use_reflect && n_samples < static_cast<size_t>(pad + 1)) ||
      (no_pad && static_cast<int>(n_samples) < win)) {
    return false;
  }

  // ---- 1. Pad + pre-emphasis ----
  const bool n_fft_is_pow2 = ((n_fft > 0) && ((n_fft & (n_fft - 1)) == 0));

  std::vector<float> padded_f32;
  std::vector<float> window_f32;
  std::vector<double> padded;

  if (!n_fft_is_pow2) {
    padded_f32.resize(n_samples + 2 * static_cast<size_t>(pad));
    if (cfg_.pre_emphasis != 0.0f) {
      const float alpha = cfg_.pre_emphasis;
      padded_f32[pad] = pcm[0];
      for (size_t i = 1; i < n_samples; ++i) {
        padded_f32[pad + i] = pcm[i] - alpha * pcm[i - 1];
      }
    } else {
      std::memcpy(padded_f32.data() + pad, pcm, n_samples * sizeof(float));
    }
    if (use_reflect) {
      for (int i = 0; i < pad; ++i) {
        padded_f32[i] = padded_f32[2 * pad - i];
      }
      for (int i = 0; i < pad; ++i) {
        padded_f32[static_cast<size_t>(pad) + n_samples + i] =
            padded_f32[static_cast<size_t>(pad) + n_samples - 2 - i];
      }
    } else {
      std::memset(padded_f32.data(), 0, static_cast<size_t>(pad) * sizeof(float));
      std::memset(padded_f32.data() + pad + n_samples, 0, static_cast<size_t>(pad) * sizeof(float));
    }

    window_f32.resize(n_fft);
    for (int i = 0; i < n_fft; ++i) {
      window_f32[i] = static_cast<float>(window_[i]);
    }
  } else {
    std::vector<double> emph(n_samples);
    if (cfg_.pre_emphasis != 0.0f) {
      const double alpha = static_cast<double>(cfg_.pre_emphasis);
      emph[0] = static_cast<double>(pcm[0]);
      for (size_t i = 1; i < n_samples; ++i) {
        emph[i] = static_cast<double>(pcm[i]) - alpha * static_cast<double>(pcm[i - 1]);
      }
    } else {
      for (size_t i = 0; i < n_samples; ++i) {
        emph[i] = static_cast<double>(pcm[i]);
      }
    }

    if (no_pad) {
      padded.resize(n_samples + static_cast<size_t>(n_fft - win), 0.0);
      std::memcpy(padded.data(), emph.data(), n_samples * sizeof(double));
    } else {
      padded.resize(n_samples + 2 * static_cast<size_t>(pad));
      if (use_reflect) {
        for (int i = 0; i < pad; ++i) {
          padded[i] = emph[static_cast<size_t>(pad - i)];
        }
        std::memcpy(padded.data() + pad, emph.data(), n_samples * sizeof(double));
        for (int i = 0; i < pad; ++i) {
          padded[static_cast<size_t>(pad) + n_samples + i] = emph[n_samples - 2 - static_cast<size_t>(i)];
        }
      } else {
        std::memset(padded.data(), 0, static_cast<size_t>(pad) * sizeof(double));
        std::memcpy(padded.data() + pad, emph.data(), n_samples * sizeof(double));
        std::memset(padded.data() + pad + n_samples, 0, static_cast<size_t>(pad) * sizeof(double));
      }
    }
  }

  // ---- 2. STFT + mel filterbank + log -> log_mel[n_mels, n_frames] ----
  std::vector<float> log_mel(static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));

  const bool whisper_mode = (cfg_.normalize == "per_utterance" || cfg_.normalize == "global");

  int stft_threads = n_threads;
  if (stft_threads <= 0) {
    stft_threads = default_threads();
  }
  if (stft_threads > n_frames) {
    stft_threads = std::max(1, n_frames);
  }

  auto run_threaded = [&](auto &&worker) {
    if (stft_threads <= 1) {
      worker(0);
      return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(stft_threads - 1));
    for (int tid = 1; tid < stft_threads; ++tid) {
      pool.emplace_back(worker, tid);
    }
    worker(0);
    for (auto &th : pool) {
      th.join();
    }
  };

  if (!n_fft_is_pow2) {
    auto worker = [&](int tid) {
      std::vector<float> fft_in(2 * static_cast<size_t>(n_fft), 0.0f);
      std::vector<float> fft_out(8 * static_cast<size_t>(n_fft), 0.0f);
      std::vector<float> power_scratch(static_cast<size_t>(n_freq));
      for (int t = tid; t < n_frames; t += stft_threads) {
        const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
        for (int n = 0; n < n_fft; ++n) {
          fft_in[n] = padded_f32[start + n] * window_f32[n];
        }
        mixed_radix_fft_f32(fft_in.data(), n_fft, cos_lut_.data(), sin_lut_.data(),
                            static_cast<int>(cos_lut_.size()), fft_out.data());
        for (int k = 0; k < n_freq; ++k) {
          const float re = fft_out[2 * k];
          const float im = fft_out[2 * k + 1];
          power_scratch[k] = re * re + im * im;
        }
        for (int m = 0; m < n_mels; ++m) {
          const float *fb_row = mel_fb_.data() + static_cast<size_t>(m) * n_freq;
          const int k_end = fb_end_[static_cast<size_t>(m)];
          double sum = 0.0;
          int k = (fb_begin_[static_cast<size_t>(m)] / 4) * 4;
          for (; k < n_freq - 3 && k < k_end; k += 4) {
            sum += static_cast<double>(fb_row[k]) * static_cast<double>(power_scratch[k]) +
                   static_cast<double>(fb_row[k + 1]) * static_cast<double>(power_scratch[k + 1]) +
                   static_cast<double>(fb_row[k + 2]) * static_cast<double>(power_scratch[k + 2]) +
                   static_cast<double>(fb_row[k + 3]) * static_cast<double>(power_scratch[k + 3]);
          }
          for (; k < n_freq && k < k_end; ++k) {
            sum += static_cast<double>(fb_row[k]) * static_cast<double>(power_scratch[k]);
          }
          float result;
          if (whisper_mode) {
            if (sum < 1.0e-10) {
              sum = 1.0e-10;
            }
            result = static_cast<float>(std::log10(sum));
          } else {
            result = static_cast<float>(std::log(sum + static_cast<double>(kLogEps)));
          }
          log_mel[static_cast<size_t>(m) * n_frames + t] = result;
        }
      }
    };
    run_threaded(worker);
  } else {
    std::vector<float> power(static_cast<size_t>(n_frames) * static_cast<size_t>(n_freq));

#ifdef __APPLE__
    int log2n = 0;
    {
      int tmp = n_fft;
      while (tmp > 1) {
        tmp >>= 1;
        ++log2n;
      }
    }
    FFTSetupD fft_setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
    const size_t half_n = static_cast<size_t>(n_fft / 2);

    auto worker = [&](int tid) {
      std::vector<double> fft_real(half_n);
      std::vector<double> fft_imag(half_n);
      DSPDoubleSplitComplex split = {fft_real.data(), fft_imag.data()};

      for (int t = tid; t < n_frames; t += stft_threads) {
        const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
        for (size_t k = 0; k < half_n; ++k) {
          fft_real[k] = padded[start + 2 * k] * window_[2 * k];
          fft_imag[k] = padded[start + 2 * k + 1] * window_[2 * k + 1];
        }
        vDSP_fft_zripD(fft_setup, &split, 1, log2n, FFT_FORWARD);
        float *pwr_row = power.data() + static_cast<size_t>(t) * n_freq;
        pwr_row[0] = static_cast<float>(fft_real[0] * fft_real[0] * 0.25);
        pwr_row[n_fft / 2] = static_cast<float>(fft_imag[0] * fft_imag[0] * 0.25);
        for (size_t k = 1; k < half_n; ++k) {
          pwr_row[k] =
              static_cast<float>((fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k]) * 0.25);
        }
      }
    };
    run_threaded(worker);
    vDSP_destroy_fftsetupD(fft_setup);

    // C = A @ B^T  where A=[n_mels, n_freq], B=[n_frames, n_freq].
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_mels, n_frames, n_freq, 1.0f,
                mel_fb_.data(), n_freq, power.data(), n_freq, 0.0f, log_mel.data(), n_frames);
    {
      const size_t total = static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames);
      if (whisper_mode) {
        for (size_t i = 0; i < total; ++i) {
          double v = static_cast<double>(log_mel[i]);
          if (v < 1.0e-10) {
            v = 1.0e-10;
          }
          log_mel[i] = static_cast<float>(std::log10(v));
        }
      } else if (cfg_.log_clamp_min > 0.0f) {
        const float clamp = cfg_.log_clamp_min;
        for (size_t i = 0; i < total; ++i) {
          float v = log_mel[i];
          if (v < clamp) {
            v = clamp;
          }
          log_mel[i] = std::log(v);
        }
      } else {
        for (size_t i = 0; i < total; ++i) {
          log_mel[i] = std::log(log_mel[i] + kLogEps);
        }
      }
    }
#else
    auto worker = [&](int tid) {
      std::vector<double> frame(2 * static_cast<size_t>(n_fft));
      for (int t = tid; t < n_frames; t += stft_threads) {
        const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
        for (int k = 0; k < n_fft; ++k) {
          frame[2 * static_cast<size_t>(k)] = padded[start + static_cast<size_t>(k)] * window_[k];
          frame[2 * static_cast<size_t>(k) + 1] = 0.0;
        }
        fft_radix2(frame.data(), n_fft);
        float *pwr_row = power.data() + static_cast<size_t>(t) * n_freq;
        for (int k = 0; k < n_freq; ++k) {
          const double re = frame[2 * static_cast<size_t>(k)];
          const double im = frame[2 * static_cast<size_t>(k) + 1];
          pwr_row[k] = static_cast<float>(re * re + im * im);
        }
      }
    };
    run_threaded(worker);

    // Scalar fused matmul + log (the arch's no-BLAS path), same strided split.
    auto matmul_worker = [&](int tid) {
      for (int t = tid; t < n_frames; t += stft_threads) {
        const float *pwr = power.data() + static_cast<size_t>(t) * n_freq;
        for (int m = 0; m < n_mels; ++m) {
          const float *fb_row = mel_fb_.data() + static_cast<size_t>(m) * n_freq;
          const int k_end = fb_end_[static_cast<size_t>(m)];
          double sum = 0.0;
          int k = (fb_begin_[static_cast<size_t>(m)] / 4) * 4;
          for (; k < n_freq - 3 && k < k_end; k += 4) {
            sum += static_cast<double>(fb_row[k]) * static_cast<double>(pwr[k]) +
                   static_cast<double>(fb_row[k + 1]) * static_cast<double>(pwr[k + 1]) +
                   static_cast<double>(fb_row[k + 2]) * static_cast<double>(pwr[k + 2]) +
                   static_cast<double>(fb_row[k + 3]) * static_cast<double>(pwr[k + 3]);
          }
          for (; k < n_freq && k < k_end; ++k) {
            sum += static_cast<double>(fb_row[k]) * static_cast<double>(pwr[k]);
          }
          float result;
          if (whisper_mode) {
            if (sum < 1.0e-10) {
              sum = 1.0e-10;
            }
            result = static_cast<float>(std::log10(sum));
          } else if (cfg_.log_clamp_min > 0.0f) {
            const double clamp = static_cast<double>(cfg_.log_clamp_min);
            if (sum < clamp) {
              sum = clamp;
            }
            result = static_cast<float>(std::log(sum));
          } else {
            result = static_cast<float>(std::log(sum + static_cast<double>(kLogEps)));
          }
          log_mel[static_cast<size_t>(m) * n_frames + t] = result;
        }
      }
    };
    run_threaded(matmul_worker);
#endif
  }

  std::vector<double>().swap(padded);
  std::vector<float>().swap(padded_f32);
  std::vector<float>().swap(window_f32);

  // ---- 3n. No-op normalize; trailing center-pad frame masked to zero ----
  if (cfg_.normalize == "none") {
    out_mel = std::move(log_mel);
    out_n_mels = n_mels;
    out_n_frames = n_frames;
    if (!no_pad) {
      const int valid = static_cast<int>(n_samples / static_cast<size_t>(cfg_.hop_length));
      for (int m = 0; m < n_mels; ++m) {
        float *row = out_mel.data() + static_cast<size_t>(m) * n_frames;
        for (int t = valid; t < n_frames; ++t) {
          row[t] = 0.0f;
        }
      }
    }
    return true;
  }

  // ---- 3a. Whisper-style per-utterance normalization ----
  if (cfg_.normalize == "per_utterance") {
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

  // ---- 3b. Fixed global max (Voxtral Realtime) ----
  if (cfg_.normalize == "global") {
    const int n_out = n_frames - 1;
    if (n_out <= 0) {
      return false;
    }
    const double floor_val = static_cast<double>(cfg_.global_log_mel_max) - 8.0;
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

  // ---- 3. Per-feature normalize, unbiased (canary_qwen) ----
  // pad_mode "constant": the last STFT frame is a padding artifact (stats
  // over the first n/hop frames, last frame zeroed). pad_mode "reflect": all
  // frames are valid, nothing masked.
  const bool mask_last = (cfg_.pad_mode == "constant");
  const int n_norm = mask_last ? (n_frames - 1) : n_frames;

  out_mel.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
  for (int m = 0; m < n_mels; ++m) {
    const float *src = log_mel.data() + static_cast<size_t>(m) * n_frames;
    float *dst = out_mel.data() + static_cast<size_t>(m) * n_frames;

    double sum = 0.0;
    for (int t = 0; t < n_norm; ++t) {
      sum += static_cast<double>(src[t]);
    }
    const double mean = sum / static_cast<double>(n_norm);

    double sumsq = 0.0;
    for (int t = 0; t < n_norm; ++t) {
      const double d = static_cast<double>(src[t]) - mean;
      sumsq += d * d;
    }
    const double var = sumsq / static_cast<double>(n_norm - 1);
    const double stddev = std::sqrt(var);
    const double inv = 1.0 / (stddev + static_cast<double>(kNormEps));

    for (int t = 0; t < n_norm; ++t) {
      dst[t] = static_cast<float>((static_cast<double>(src[t]) - mean) * inv);
    }
    if (mask_last) {
      dst[n_norm] = 0.0f;
    }
  }

  out_n_mels = n_mels;
  out_n_frames = n_frames;
  return true;
}

} // namespace engine::models::canary_qwen
