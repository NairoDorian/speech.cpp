#pragma once

#include "engine/framework/audio/frontend_spec.h"

#include <cstddef>
#include <vector>

namespace engine::audio {

// Pure C++ unified audio frontend and log-mel feature extractor.
// Thread-safe: const after construction; multiple threads may call compute() concurrently.
class MelExtractor {
public:
    explicit MelExtractor(const FrontendSpec & spec);

    // Run the full extraction pipeline according to spec_.
    // pcm: input mono float32 in [-1, 1]
    // n_samples: number of input samples
    // out_mel: output buffer, resized to out_n_mels * out_n_frames in row-major [out_n_mels, out_n_frames]
    // n_threads: parallelism (0 auto-detects up to 8 threads)
    // Returns true on success, false on invalid args or processing failure.
    bool compute(const float *        pcm,
                 size_t               n_samples,
                 std::vector<float> & out_mel,
                 int &                out_n_mels,
                 int &                out_n_frames,
                 int                  n_threads = 0) const;

    // Number of feature dimensions (e.g. spec.num_mels)
    int num_mels() const { return spec_.num_mels; }

    // Computes expected frame count for a given sample count before calling compute().
    int n_frames_for(size_t n_samples) const;

    const FrontendSpec & spec() const { return spec_; }

    // Test and introspection accessors
    const std::vector<float> & filterbank() const { return filterbank_; }
    const std::vector<double> & window() const { return window_; }
    const std::vector<int> & fb_begin() const { return fb_begin_; }
    const std::vector<int> & fb_end() const { return fb_end_; }

private:
    FrontendSpec        spec_;
    std::vector<float>  filterbank_;  // [num_mels * n_freq]
    std::vector<double> window_;      // [n_fft] zero-padded or [win_length]
    std::vector<int>    fb_begin_;    // [num_mels] first non-zero bin index
    std::vector<int>    fb_end_;      // [num_mels] one past last non-zero bin index
    int                 n_freq_ = 0;  // n_fft / 2 + 1

    void init_mel_filterbank();
    void init_window();
    void init_support_spans();
};

}  // namespace engine::audio
