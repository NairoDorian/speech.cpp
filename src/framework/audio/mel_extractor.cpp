#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES 1
#endif

#include "engine/framework/audio/mel_extractor.h"

#ifdef __APPLE__
#    include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

namespace engine::audio {

namespace {

constexpr float kLogEps  = 5.9604644775390625e-08f;  // 2^-24 (NeMo log_zero_guard_value)
constexpr float kNormEps = 1.0e-05f;                 // NeMo CONSTANT

constexpr double kSlaneyFsp      = 200.0 / 3.0;
constexpr double kSlaneyMinLogHz = 1000.0;

inline double slaney_hz_to_mel(double hz) {
    const double min_log_mel = kSlaneyMinLogHz / kSlaneyFsp;
    const double logstep     = std::log(6.4) / 27.0;
    if (hz < kSlaneyMinLogHz) {
        return hz / kSlaneyFsp;
    }
    return min_log_mel + std::log(hz / kSlaneyMinLogHz) / logstep;
}

inline double slaney_mel_to_hz(double mel) {
    const double min_log_mel = kSlaneyMinLogHz / kSlaneyFsp;
    const double logstep     = std::log(6.4) / 27.0;
    if (mel < min_log_mel) {
        return mel * kSlaneyFsp;
    }
    return kSlaneyMinLogHz * std::exp(logstep * (mel - min_log_mel));
}

inline double htk_hz_to_mel(double hz) {
    return 1127.0 * std::log(1.0 + hz / 700.0);
}

inline double htk_mel_to_hz(double mel) {
    return 700.0 * (std::exp(mel / 1127.0) - 1.0);
}

void build_mel_filterbank_slaney(int sr, int n_fft, int n_mels, double fmin, double fmax, std::vector<float> & out) {
    const int n_freq = n_fft / 2 + 1;
    out.assign(static_cast<size_t>(n_mels) * n_freq, 0.0f);

    std::vector<double> fft_freqs(n_freq);
    for (int k = 0; k < n_freq; ++k) {
        fft_freqs[k] = static_cast<double>(sr) * k / static_cast<double>(n_fft);
    }

    const double        mel_min = slaney_hz_to_mel(fmin);
    const double        mel_max = slaney_hz_to_mel(fmax);
    std::vector<double> hz_freqs(n_mels + 2);
    for (int m = 0; m < n_mels + 2; ++m) {
        const double mel = mel_min + (mel_max - mel_min) * m / static_cast<double>(n_mels + 1);
        hz_freqs[m]      = slaney_mel_to_hz(mel);
    }

    std::vector<double> fdiff(n_mels + 1);
    for (int m = 0; m < n_mels + 1; ++m) {
        fdiff[m] = hz_freqs[m + 1] - hz_freqs[m];
    }

    for (int m = 0; m < n_mels; ++m) {
        const double enorm = 2.0 / (hz_freqs[m + 2] - hz_freqs[m]);
        for (int k = 0; k < n_freq; ++k) {
            const double lower                       = (fft_freqs[k] - hz_freqs[m]) / fdiff[m];
            const double upper                       = (hz_freqs[m + 2] - fft_freqs[k]) / fdiff[m + 1];
            const double w                           = std::max(0.0, std::min(lower, upper));
            out[static_cast<size_t>(m) * n_freq + k] = static_cast<float>(w * enorm);
        }
    }
}

void build_htk_filterbank(int sr, int n_fft, int n_mels, double fmin, double fmax, std::vector<float> & out) {
    const int n_freq = n_fft / 2 + 1;
    if (fmax <= 0.0) {
        fmax = static_cast<double>(sr) * 0.5;
    }
    const double mel_low  = htk_hz_to_mel(fmin);
    const double mel_high = htk_hz_to_mel(fmax);
    const double mel_step = (mel_high - mel_low) / (n_mels + 1);
    const double fft_bin_width = static_cast<double>(sr) / static_cast<double>(n_fft);

    out.assign(static_cast<size_t>(n_mels) * n_freq, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const double left_mel   = mel_low + m * mel_step;
        const double center_mel = mel_low + (m + 1) * mel_step;
        const double right_mel  = mel_low + (m + 2) * mel_step;
        const double left_hz    = htk_mel_to_hz(left_mel);
        const double center_hz  = htk_mel_to_hz(center_mel);
        const double right_hz   = htk_mel_to_hz(right_mel);

        for (int b = 0; b < n_freq; ++b) {
            const double bin_hz = b * fft_bin_width;
            double       w      = 0.0;
            if (bin_hz > left_hz && bin_hz < right_hz) {
                if (bin_hz <= center_hz) {
                    w = (bin_hz - left_hz) / (center_hz - left_hz);
                } else {
                    w = (right_hz - bin_hz) / (right_hz - center_hz);
                }
            }
            if (w < 0.0) {
                w = 0.0;
            }
            out[static_cast<size_t>(m) * n_freq + b] = static_cast<float>(w);
        }
    }
}

void dft_naive_f32(const float * in, int N, const float * cos_lut, const float * sin_lut, int lut_size, float * out) {
    const int stride = lut_size / N;
    for (int k = 0; k < N; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        for (int n = 0; n < N; ++n) {
            const int idx = (k * n * stride) % lut_size;
            re += in[n] * cos_lut[idx];
            im -= in[n] * sin_lut[idx];
        }
        out[2 * k]     = re;
        out[2 * k + 1] = im;
    }
}

void mixed_radix_fft_f32(float * in, int N, const float * cos_lut, const float * sin_lut, int lut_size, float * out) {
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
    float *   even   = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i] = in[2 * i];
    }
    float * even_fft = out + 2 * N;
    mixed_radix_fft_f32(even, half_N, cos_lut, sin_lut, lut_size, even_fft);

    float * odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2 * i + 1];
    }
    float * odd_fft = even_fft + N;
    mixed_radix_fft_f32(odd, half_N, cos_lut, sin_lut, lut_size, odd_fft);

    const int step = lut_size / N;
    for (int k = 0; k < half_N; ++k) {
        const int   idx           = k * step;
        const float w_re          = cos_lut[idx];
        const float w_im          = -sin_lut[idx];
        const float re_odd        = odd_fft[2 * k];
        const float im_odd        = odd_fft[2 * k + 1];
        out[2 * k]                = even_fft[2 * k] + w_re * re_odd - w_im * im_odd;
        out[2 * k + 1]            = even_fft[2 * k + 1] + w_re * im_odd + w_im * re_odd;
        out[2 * (k + half_N)]     = even_fft[2 * k] - w_re * re_odd + w_im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - w_re * im_odd - w_im * re_odd;
    }
}

void fft_radix2(double * data, int n) {
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
        const double ang     = -2.0 * M_PI / static_cast<double>(len);
        const double wlen_re = std::cos(ang);
        const double wlen_im = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;
            for (int k = 0; k < len / 2; ++k) {
                const int    a      = 2 * (i + k);
                const int    b      = 2 * (i + k + len / 2);
                const double u_re   = data[a];
                const double u_im   = data[a + 1];
                const double v_re   = data[b] * w_re - data[b + 1] * w_im;
                const double v_im   = data[b] * w_im + data[b + 1] * w_re;
                data[a]             = u_re + v_re;
                data[a + 1]         = u_im + v_im;
                data[b]             = u_re - v_re;
                data[b + 1]         = u_im - v_im;
                const double tmp_re = w_re * wlen_re - w_im * wlen_im;
                const double tmp_im = w_re * wlen_im + w_im * wlen_re;
                w_re                = tmp_re;
                w_im                = tmp_im;
            }
        }
    }
}

}  // namespace

MelExtractor::MelExtractor(const FrontendSpec & spec) : spec_(spec) {
    if (spec_.kind == FrontendKind::RawPcm) {
        spec_.num_mels = 1;
        return;
    }

    n_freq_ = spec_.n_fft / 2 + 1;
    init_window();
    init_mel_filterbank();
    init_support_spans();
}

void MelExtractor::init_window() {
    if (!spec_.window.empty()) {
        window_.resize(static_cast<size_t>(spec_.n_fft), 0.0);
        const int total_pad = spec_.n_fft - spec_.win_length;
        const int left_pad  = (spec_.pad_mode == PadMode::None) ? 0 : (total_pad / 2);
        for (int i = 0; i < spec_.win_length && i < static_cast<int>(spec_.window.size()); ++i) {
            window_[static_cast<size_t>(left_pad + i)] = static_cast<double>(spec_.window[static_cast<size_t>(i)]);
        }
        return;
    }

    window_.assign(static_cast<size_t>(spec_.n_fft), 0.0);
    const int pad_each = (spec_.pad_mode == PadMode::None) ? 0 : ((spec_.n_fft - spec_.win_length) / 2);

    if (spec_.window_type == WindowType::Hamming) {
        const double denom = (spec_.win_length <= 1) ? 1.0 : static_cast<double>(spec_.win_length - 1);
        for (int k = 0; k < spec_.win_length; ++k) {
            window_[static_cast<size_t>(pad_each + k)] = 0.54 - 0.46 * std::cos(2.0 * M_PI * k / denom);
        }
    } else {
        const bool periodic = (spec_.window_type == WindowType::HannPeriodic);
        const double denom = periodic ? static_cast<double>(spec_.win_length) : static_cast<double>(spec_.win_length - 1);
        for (int k = 0; k < spec_.win_length; ++k) {
            window_[static_cast<size_t>(pad_each + k)] = 0.5 - 0.5 * std::cos(2.0 * M_PI * k / denom);
        }
    }
}

void MelExtractor::init_mel_filterbank() {
    if (!spec_.filterbank.empty()) {
        filterbank_ = spec_.filterbank;
        return;
    }

    if (spec_.kind == FrontendKind::KaldiFbank) {
        build_htk_filterbank(spec_.sample_rate, spec_.n_fft, spec_.num_mels,
                             static_cast<double>(spec_.f_min), static_cast<double>(spec_.f_max), filterbank_);
    } else {
        build_mel_filterbank_slaney(spec_.sample_rate, spec_.n_fft, spec_.num_mels,
                                    static_cast<double>(spec_.f_min), static_cast<double>(spec_.f_max), filterbank_);
    }
}

void MelExtractor::init_support_spans() {
    fb_begin_.assign(static_cast<size_t>(spec_.num_mels), 0);
    fb_end_.assign(static_cast<size_t>(spec_.num_mels), 0);

    for (int m = 0; m < spec_.num_mels; ++m) {
        const float * row = filterbank_.data() + static_cast<size_t>(m) * static_cast<size_t>(n_freq_);
        int lo = 0;
        while (lo < n_freq_ && row[lo] == 0.0f) {
            ++lo;
        }
        int hi = n_freq_;
        while (hi > lo && row[hi - 1] == 0.0f) {
            --hi;
        }
        fb_begin_[static_cast<size_t>(m)] = lo;
        fb_end_[static_cast<size_t>(m)]   = hi;
    }
}

int MelExtractor::n_frames_for(size_t n_samples) const {
    if (spec_.kind == FrontendKind::RawPcm) {
        return static_cast<int>(n_samples);
    }
    if (spec_.pad_mode == PadMode::None) {
        if (static_cast<int>(n_samples) < spec_.win_length) {
            return 0;
        }
        return static_cast<int>((n_samples - static_cast<size_t>(spec_.win_length)) /
                                static_cast<size_t>(spec_.hop_length)) + 1;
    }
    if (spec_.nemo_seq_len_ceil) {
        return static_cast<int>((n_samples + static_cast<size_t>(spec_.hop_length) - 1) /
                                static_cast<size_t>(spec_.hop_length));
    }
    return static_cast<int>(n_samples / static_cast<size_t>(spec_.hop_length)) + 1;
}

bool MelExtractor::compute(const float *        pcm,
                           size_t               n_samples,
                           std::vector<float> & out_mel,
                           int &                out_n_mels,
                           int &                out_n_frames,
                           int                  n_threads) const {
    if (pcm == nullptr) {
        return false;
    }

    if (spec_.kind == FrontendKind::RawPcm) {
        out_n_mels   = 1;
        out_n_frames = static_cast<int>(n_samples);
        out_mel.assign(pcm, pcm + n_samples);
        return true;
    }

    const int  n_fft  = spec_.n_fft;
    const int  hop    = spec_.hop_length;
    const int  n_mels = spec_.num_mels;
    const int  n_freq = n_freq_;
    const int  win    = spec_.win_length;
    const int  pad    = n_fft / 2;
    const bool no_pad = (spec_.pad_mode == PadMode::None);

    const int n_frames = n_frames_for(n_samples);
    const bool use_reflect = (!no_pad && spec_.pad_mode != PadMode::Constant);

    if (n_frames < (no_pad ? 1 : 2) || (use_reflect && n_samples < static_cast<size_t>(pad + 1)) ||
        (no_pad && static_cast<int>(n_samples) < win)) {
        return false;
    }

    const bool n_fft_is_pow2 = ((n_fft > 0) && ((n_fft & (n_fft - 1)) == 0));

    std::vector<float>  padded_f32;
    std::vector<float>  window_f32;
    std::vector<double> padded;

    if (!n_fft_is_pow2) {
        padded_f32.resize(n_samples + 2 * static_cast<size_t>(pad));
        if (spec_.pre_emphasis != 0.0f) {
            const float alpha = spec_.pre_emphasis;
            padded_f32[static_cast<size_t>(pad)] = pcm[0];
            for (size_t i = 1; i < n_samples; ++i) {
                padded_f32[static_cast<size_t>(pad) + i] = pcm[i] - alpha * pcm[i - 1];
            }
        } else {
            std::memcpy(padded_f32.data() + pad, pcm, n_samples * sizeof(float));
        }

        if (use_reflect) {
            for (int i = 0; i < pad; ++i) {
                padded_f32[static_cast<size_t>(i)] = padded_f32[static_cast<size_t>(2 * pad - i)];
            }
            for (int i = 0; i < pad; ++i) {
                padded_f32[static_cast<size_t>(pad) + n_samples + static_cast<size_t>(i)] =
                    padded_f32[static_cast<size_t>(pad) + n_samples - 2 - static_cast<size_t>(i)];
            }
        } else {
            std::memset(padded_f32.data(), 0, static_cast<size_t>(pad) * sizeof(float));
            std::memset(padded_f32.data() + pad + n_samples, 0, static_cast<size_t>(pad) * sizeof(float));
        }

        window_f32.resize(static_cast<size_t>(n_fft));
        for (int i = 0; i < n_fft; ++i) {
            window_f32[static_cast<size_t>(i)] = static_cast<float>(window_[static_cast<size_t>(i)]);
        }
    } else {
        std::vector<double> emph(n_samples);
        if (spec_.pre_emphasis != 0.0f) {
            const double alpha = static_cast<double>(spec_.pre_emphasis);
            emph[0]            = static_cast<double>(pcm[0]);
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
                    padded[static_cast<size_t>(i)] = emph[static_cast<size_t>(pad - i)];
                }
                std::memcpy(padded.data() + pad, emph.data(), n_samples * sizeof(double));
                for (int i = 0; i < pad; ++i) {
                    padded[static_cast<size_t>(pad) + n_samples + static_cast<size_t>(i)] =
                        emph[n_samples - 2 - static_cast<size_t>(i)];
                }
            } else {
                std::memset(padded.data(), 0, static_cast<size_t>(pad) * sizeof(double));
                std::memcpy(padded.data() + pad, emph.data(), n_samples * sizeof(double));
                std::memset(padded.data() + pad + n_samples, 0, static_cast<size_t>(pad) * sizeof(double));
            }
        }
    }

    std::vector<float> log_mel(static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
    const bool whisper_mode = (spec_.normalize_mode == NormalizeMode::PerUtterance ||
                               spec_.normalize_mode == NormalizeMode::Global);

    int stft_threads = n_threads;
    if (stft_threads <= 0) {
        stft_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (stft_threads <= 0) stft_threads = 4;
        if (stft_threads > 8) stft_threads = 8;
    }
    if (stft_threads > n_frames) {
        stft_threads = std::max(1, n_frames);
    }

    auto run_threaded = [&](auto && worker) {
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
        for (auto & th : pool) {
            th.join();
        }
    };

    if (!n_fft_is_pow2) {
        std::vector<float> cos_lut(static_cast<size_t>(n_fft));
        std::vector<float> sin_lut(static_cast<size_t>(n_fft));
        for (int i = 0; i < n_fft; ++i) {
            const double theta = 2.0 * M_PI * i / n_fft;
            cos_lut[static_cast<size_t>(i)] = static_cast<float>(std::cos(theta));
            sin_lut[static_cast<size_t>(i)] = static_cast<float>(std::sin(theta));
        }

        auto worker = [&](int tid) {
            std::vector<float> fft_in(2 * static_cast<size_t>(n_fft), 0.0f);
            std::vector<float> fft_out(8 * static_cast<size_t>(n_fft), 0.0f);
            std::vector<float> power_scratch(static_cast<size_t>(n_freq));
            for (int t = tid; t < n_frames; t += stft_threads) {
                const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
                for (int n = 0; n < n_fft; ++n) {
                    fft_in[static_cast<size_t>(n)] = padded_f32[start + static_cast<size_t>(n)] * window_f32[static_cast<size_t>(n)];
                }
                mixed_radix_fft_f32(fft_in.data(), n_fft, cos_lut.data(), sin_lut.data(), n_fft, fft_out.data());
                for (int k = 0; k < n_freq; ++k) {
                    const float re   = fft_out[static_cast<size_t>(2 * k)];
                    const float im   = fft_out[static_cast<size_t>(2 * k + 1)];
                    power_scratch[static_cast<size_t>(k)] = re * re + im * im;
                }
                for (int m = 0; m < n_mels; ++m) {
                    const float * fb_row = filterbank_.data() + static_cast<size_t>(m) * static_cast<size_t>(n_freq);
                    const int     k_end  = fb_end_[static_cast<size_t>(m)];
                    double        sum    = 0.0;
                    int           k      = (fb_begin_[static_cast<size_t>(m)] / 4) * 4;
                    for (; k < n_freq - 3 && k < k_end; k += 4) {
                        sum += static_cast<double>(fb_row[k]) * static_cast<double>(power_scratch[static_cast<size_t>(k)]) +
                               static_cast<double>(fb_row[k + 1]) * static_cast<double>(power_scratch[static_cast<size_t>(k + 1)]) +
                               static_cast<double>(fb_row[k + 2]) * static_cast<double>(power_scratch[static_cast<size_t>(k + 2)]) +
                               static_cast<double>(fb_row[k + 3]) * static_cast<double>(power_scratch[static_cast<size_t>(k + 3)]);
                    }
                    for (; k < n_freq && k < k_end; ++k) {
                        sum += static_cast<double>(fb_row[k]) * static_cast<double>(power_scratch[static_cast<size_t>(k)]);
                    }
                    float result;
                    if (whisper_mode) {
                        if (sum < 1.0e-10) {
                            sum = 1.0e-10;
                        }
                        result = static_cast<float>(std::log10(sum));
                    } else if (spec_.normalize_mode == NormalizeMode::None && spec_.log_clamp_min > 0.0f) {
                        result = static_cast<float>(std::log(std::max(sum, static_cast<double>(spec_.log_clamp_min))));
                    } else {
                        result = static_cast<float>(std::log(sum + static_cast<double>(kLogEps)));
                    }
                    log_mel[static_cast<size_t>(m) * static_cast<size_t>(n_frames) + static_cast<size_t>(t)] = result;
                }
            }
        };
        run_threaded(worker);
    } else {
        auto worker = [&](int tid) {
            std::vector<double> fft_buf(2 * static_cast<size_t>(n_fft), 0.0);
            std::vector<float>  power_scratch(static_cast<size_t>(n_freq));
            for (int t = tid; t < n_frames; t += stft_threads) {
                const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
                for (int n = 0; n < n_fft; ++n) {
                    fft_buf[static_cast<size_t>(2 * n)]     = padded[start + static_cast<size_t>(n)] * window_[static_cast<size_t>(n)];
                    fft_buf[static_cast<size_t>(2 * n + 1)] = 0.0;
                }
                fft_radix2(fft_buf.data(), n_fft);
                for (int k = 0; k < n_freq; ++k) {
                    const double re = fft_buf[static_cast<size_t>(2 * k)];
                    const double im = fft_buf[static_cast<size_t>(2 * k + 1)];
                    power_scratch[static_cast<size_t>(k)] = static_cast<float>(re * re + im * im);
                }
                for (int m = 0; m < n_mels; ++m) {
                    const float * fb_row = filterbank_.data() + static_cast<size_t>(m) * static_cast<size_t>(n_freq);
                    const int     k_end  = fb_end_[static_cast<size_t>(m)];
                    double        sum    = 0.0;
                    int           k      = (fb_begin_[static_cast<size_t>(m)] / 4) * 4;
                    for (; k < n_freq - 3 && k < k_end; k += 4) {
                        sum += static_cast<double>(fb_row[k]) * static_cast<double>(power_scratch[static_cast<size_t>(k)]) +
                               static_cast<double>(fb_row[k + 1]) * static_cast<double>(power_scratch[static_cast<size_t>(k + 1)]) +
                               static_cast<double>(fb_row[k + 2]) * static_cast<double>(power_scratch[static_cast<size_t>(k + 2)]) +
                               static_cast<double>(fb_row[k + 3]) * static_cast<double>(power_scratch[static_cast<size_t>(k + 3)]);
                    }
                    for (; k < n_freq && k < k_end; ++k) {
                        sum += static_cast<double>(fb_row[k]) * static_cast<double>(power_scratch[static_cast<size_t>(k)]);
                    }
                    float result;
                    if (whisper_mode) {
                        if (sum < 1.0e-10) {
                            sum = 1.0e-10;
                        }
                        result = static_cast<float>(std::log10(sum));
                    } else if (spec_.normalize_mode == NormalizeMode::None && spec_.log_clamp_min > 0.0f) {
                        result = static_cast<float>(std::log(std::max(sum, static_cast<double>(spec_.log_clamp_min))));
                    } else {
                        result = static_cast<float>(std::log(sum + static_cast<double>(kLogEps)));
                    }
                    log_mel[static_cast<size_t>(m) * static_cast<size_t>(n_frames) + static_cast<size_t>(t)] = result;
                }
            }
        };
        run_threaded(worker);
    }

    // ---- 4. Normalization ----
    if (spec_.normalize_mode == NormalizeMode::PerFeature) {
        if (n_frames < 2) {
            return false;
        }
        const float denom_mean = 1.0f / static_cast<float>(n_frames);
        const float denom_var  = 1.0f / static_cast<float>(n_frames - 1);
        for (int m = 0; m < n_mels; ++m) {
            float * row  = log_mel.data() + static_cast<size_t>(m) * static_cast<size_t>(n_frames);
            double  mean = 0.0;
            for (int t = 0; t < n_frames; ++t) {
                mean += static_cast<double>(row[t]);
            }
            mean *= static_cast<double>(denom_mean);

            double var = 0.0;
            for (int t = 0; t < n_frames; ++t) {
                const double diff = static_cast<double>(row[t]) - mean;
                var += diff * diff;
            }
            var *= static_cast<double>(denom_var);

            const double std_inv = 1.0 / std::sqrt(var + static_cast<double>(kNormEps));
            for (int t = 0; t < n_frames; ++t) {
                row[t] = static_cast<float>((static_cast<double>(row[t]) - mean) * std_inv);
            }
        }
        out_mel      = std::move(log_mel);
        out_n_mels   = n_mels;
        out_n_frames = n_frames;
    } else if (spec_.normalize_mode == NormalizeMode::PerUtterance) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (float v : log_mel) {
            if (v > max_val) {
                max_val = v;
            }
        }
        const float clamp_min = max_val - 8.0f;
        for (float & v : log_mel) {
            v = (std::max(v, clamp_min) + 4.0f) / 4.0f;
        }

        const int out_frames = (no_pad || n_frames <= 0) ? n_frames : (n_frames - 1);
        out_mel.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(out_frames));
        for (int m = 0; m < n_mels; ++m) {
            std::memcpy(out_mel.data() + static_cast<size_t>(m) * static_cast<size_t>(out_frames),
                        log_mel.data() + static_cast<size_t>(m) * static_cast<size_t>(n_frames),
                        static_cast<size_t>(out_frames) * sizeof(float));
        }
        out_n_mels   = n_mels;
        out_n_frames = out_frames;
    } else if (spec_.normalize_mode == NormalizeMode::Global) {
        const float clamp_min = spec_.global_log_mel_max - 8.0f;
        for (float & v : log_mel) {
            v = (std::max(v, clamp_min) + 4.0f) / 4.0f;
        }

        const int out_frames = (no_pad || n_frames <= 0) ? n_frames : (n_frames - 1);
        out_mel.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(out_frames));
        for (int m = 0; m < n_mels; ++m) {
            std::memcpy(out_mel.data() + static_cast<size_t>(m) * static_cast<size_t>(out_frames),
                        log_mel.data() + static_cast<size_t>(m) * static_cast<size_t>(n_frames),
                        static_cast<size_t>(out_frames) * sizeof(float));
        }
        out_n_mels   = n_mels;
        out_n_frames = out_frames;
    } else {
        out_mel      = std::move(log_mel);
        out_n_mels   = n_mels;
        out_n_frames = n_frames;
    }

    return true;
}

}  // namespace engine::audio
