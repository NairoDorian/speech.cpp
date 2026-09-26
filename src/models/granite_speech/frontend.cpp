// granite_speech frontend: PCM -> stacked log-mel encoder input.
//
// Ported from the arch's host path, which is what the parity target
// computes:
//   src/runtime/transcribe-mel.cpp  MelFrontend::MelFrontend / n_frames_for /
//                                   compute, the n_fft power-of-two (fp64
//                                   radix-2) branch with the scalar fused
//                                   matmul + log post-pass, "per_utterance"
//                                   normalisation
//   src/runtime/arch/granite/encoder.cpp  compute_mel_encoder_input (odd-frame
//                                   drop + 2-frame stack)
//
// Why not engine::audio::NemoMelFrontend: its FFT is the engine's fp32
// RealFFTPlan while the arch STFTs in fp64, and GraniteSpeechFeatureExtractor's
// whisper-style normalisation (global max - 8 floor over the frames kept after
// dropping the trailing centre frame) is not one of its MelNorm modes. The
// mel frontend is the first place a port drifts ("MelExtractor is mel-major"
// class of bug), so it is carried over verbatim instead.
//
// Known divergence: on Apple the arch uses vDSP's FFT; this port always uses
// the hand-rolled radix-2 (the arch's non-Apple path), and it never takes the
// arch's cblas_sgemm branch (TRANSCRIBE_HAS_BLAS is not defined in speech.cpp
// builds, so the arch runs the same scalar fp64 matmul).

#include "engine/models/granite_speech/model.h"

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

namespace engine::models::granite_speech {
namespace {

// Verbatim port of transcribe-mel.cpp:fft_radix2 (in-place Cooley-Tukey,
// fp64, interleaved re/im, incremental twiddles).
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

// Strided frame split (arch MelFrontend::compute run_threaded). Frames are
// independent, so the thread count never changes the arithmetic.
template <typename Worker>
void run_threaded(int threads, Worker && worker) {
    if (threads <= 1) {
        worker(0);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads - 1));
    for (int tid = 1; tid < threads; ++tid) {
        pool.emplace_back(worker, tid);
    }
    worker(0);
    for (auto & th : pool) {
        th.join();
    }
}

}  // namespace

GraniteSpeechFeatures compute_granite_speech_features(
    const std::vector<float> & pcm, const GraniteSpeechFrontend & fe, int threads) {
    const int n_fft = fe.n_fft;
    const int hop = fe.hop_length;
    const int n_mels = fe.n_mels;
    const int n_freq = n_fft / 2 + 1;
    const int pad = n_fft / 2;
    const size_t n_samples = pcm.size();
    const bool use_reflect = fe.pad_mode == "reflect";

    // MelFrontend::n_frames_for, legacy (non-"none", non-ceil) rule:
    // floor(n / hop) + 1 centred frames.
    const int n_frames = static_cast<int>(n_samples / static_cast<size_t>(hop)) + 1;
    if (n_frames < 2 || (use_reflect && n_samples < static_cast<size_t>(pad + 1))) {
        throw std::invalid_argument("Granite Speech needs at least " + std::to_string(std::max(pad + 1, hop)) +
                                    " samples of audio");
    }

    // ---- 1. Pre-emphasis + centre padding (fp64 power-of-two path) ----
    std::vector<double> emph(n_samples);
    if (fe.pre_emphasis != 0.0f) {
        const double alpha = static_cast<double>(fe.pre_emphasis);
        emph[0] = static_cast<double>(pcm[0]);
        for (size_t i = 1; i < n_samples; ++i) {
            emph[i] = static_cast<double>(pcm[i]) - alpha * static_cast<double>(pcm[i - 1]);
        }
    } else {
        for (size_t i = 0; i < n_samples; ++i) {
            emph[i] = static_cast<double>(pcm[i]);
        }
    }
    std::vector<double> padded(n_samples + 2 * static_cast<size_t>(pad), 0.0);
    std::memcpy(padded.data() + pad, emph.data(), n_samples * sizeof(double));
    if (use_reflect) {
        for (int i = 0; i < pad; ++i) {
            padded[static_cast<size_t>(i)] = emph[static_cast<size_t>(pad - i)];
            padded[static_cast<size_t>(pad) + n_samples + static_cast<size_t>(i)] =
                emph[n_samples - 2 - static_cast<size_t>(i)];
        }
    }  // "constant": the zero fill above.

    int stft_threads = threads;
    if (stft_threads <= 0) {
        stft_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    if (stft_threads > n_frames) {
        stft_threads = std::max(1, n_frames);
    }

    // ---- 2. STFT power spectrum (fp64 FFT, stored fp32) ----
    std::vector<float> power(static_cast<size_t>(n_frames) * static_cast<size_t>(n_freq));
    run_threaded(stft_threads, [&](int tid) {
        std::vector<double> frame(2 * static_cast<size_t>(n_fft));
        for (int t = tid; t < n_frames; t += stft_threads) {
            const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
            for (int k = 0; k < n_fft; ++k) {
                frame[2 * static_cast<size_t>(k)] = padded[start + static_cast<size_t>(k)] * fe.window[static_cast<size_t>(k)];
                frame[2 * static_cast<size_t>(k) + 1] = 0.0;
            }
            fft_radix2(frame.data(), n_fft);
            float * pwr_row = power.data() + static_cast<size_t>(t) * static_cast<size_t>(n_freq);
            for (int k = 0; k < n_freq; ++k) {
                const double re = frame[2 * static_cast<size_t>(k)];
                const double im = frame[2 * static_cast<size_t>(k) + 1];
                pwr_row[k] = static_cast<float>(re * re + im * im);
            }
        }
    });
    std::vector<double>().swap(padded);

    // ---- 3. Mel matmul + log10 (scalar fused path, fp64 accumulation over
    // each band's nonzero span, 4-way unrolled exactly as the arch) ----
    std::vector<float> log_mel(static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
    run_threaded(stft_threads, [&](int tid) {
        for (int t = tid; t < n_frames; t += stft_threads) {
            const float * pwr = power.data() + static_cast<size_t>(t) * static_cast<size_t>(n_freq);
            for (int m = 0; m < n_mels; ++m) {
                const float * fb_row = fe.filterbank.data() + static_cast<size_t>(m) * static_cast<size_t>(n_freq);
                const int k_end = fe.fb_end[static_cast<size_t>(m)];
                double sum = 0.0;
                int k = (fe.fb_begin[static_cast<size_t>(m)] / 4) * 4;
                for (; k < n_freq - 3 && k < k_end; k += 4) {
                    sum += static_cast<double>(fb_row[k]) * static_cast<double>(pwr[k]) +
                           static_cast<double>(fb_row[k + 1]) * static_cast<double>(pwr[k + 1]) +
                           static_cast<double>(fb_row[k + 2]) * static_cast<double>(pwr[k + 2]) +
                           static_cast<double>(fb_row[k + 3]) * static_cast<double>(pwr[k + 3]);
                }
                for (; k < n_freq && k < k_end; ++k) {
                    sum += static_cast<double>(fb_row[k]) * static_cast<double>(pwr[k]);
                }
                if (sum < 1.0e-10) {
                    sum = 1.0e-10;
                }
                log_mel[static_cast<size_t>(m) * static_cast<size_t>(n_frames) + static_cast<size_t>(t)] =
                    static_cast<float>(std::log10(sum));
            }
        }
    });

    // ---- 4. Whisper-style per-utterance normalisation ----
    // Drop the trailing centre-pad frame, floor at (global max - 8), then
    // (x + 4) / 4. Output [n_mels, n_out] row-major.
    const int n_out = n_frames - 1;
    double global_max = -std::numeric_limits<double>::infinity();
    for (int m = 0; m < n_mels; ++m) {
        const float * src = log_mel.data() + static_cast<size_t>(m) * static_cast<size_t>(n_frames);
        for (int t = 0; t < n_out; ++t) {
            global_max = std::max(global_max, static_cast<double>(src[t]));
        }
    }
    const double floor_val = global_max - 8.0;
    std::vector<float> mel(static_cast<size_t>(n_mels) * static_cast<size_t>(n_out));
    for (int m = 0; m < n_mels; ++m) {
        const float * src = log_mel.data() + static_cast<size_t>(m) * static_cast<size_t>(n_frames);
        float * dst = mel.data() + static_cast<size_t>(m) * static_cast<size_t>(n_out);
        for (int t = 0; t < n_out; ++t) {
            double v = static_cast<double>(src[t]);
            if (v < floor_val) {
                v = floor_val;
            }
            dst[t] = static_cast<float>((v + 4.0) / 4.0);
        }
    }

    // ---- 5. Granite: drop the last frame when odd, stack frame pairs ----
    // Row t of the output is [mel(2t, 0..n_mels) || mel(2t+1, 0..n_mels)].
    int kept = n_out;
    if ((kept % 2) == 1) {
        --kept;
    }
    const int t_enc = kept / 2;
    if (t_enc <= 0) {
        throw std::invalid_argument("Granite Speech audio is too short to produce one encoder frame");
    }
    const int input_dim = 2 * n_mels;
    GraniteSpeechFeatures out;
    out.t_enc = t_enc;
    out.values.assign(static_cast<size_t>(t_enc) * static_cast<size_t>(input_dim), 0.0f);
    for (int t = 0; t < t_enc; ++t) {
        float * dst = out.values.data() + static_cast<size_t>(t) * static_cast<size_t>(input_dim);
        for (int m = 0; m < n_mels; ++m) {
            dst[m] = mel[static_cast<size_t>(m) * static_cast<size_t>(n_out) + static_cast<size_t>(2 * t)];
            dst[m + n_mels] = mel[static_cast<size_t>(m) * static_cast<size_t>(n_out) + static_cast<size_t>(2 * t + 1)];
        }
    }
    return out;
}

}  // namespace engine::models::granite_speech
