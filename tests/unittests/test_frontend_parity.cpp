// These checks use assert() and the suite builds Release, where NDEBUG
// compiles every one of them out - until 2026-09-23 this test could not
// fail. Keep the checks live regardless of build type.
#undef NDEBUG

#include "engine/framework/audio/frontend_spec.h"
#include "engine/framework/audio/mel_extractor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace engine::audio;

int main() {
    std::cout << "[frontend_parity_test] Starting frontend parity verification..." << std::endl;

    // Generate 1 second of 440Hz sine wave at 16kHz
    const int sample_rate = 16000;
    const size_t n_samples = 16000;
    std::vector<float> pcm(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        pcm[i] = 0.5f * std::sin(2.0f * 3.14159265358979323846f * 440.0f * static_cast<float>(i) / static_cast<float>(sample_rate));
    }

    // 1. Test MelExtractor with PerFeature normalization (Parakeet style)
    {
        FrontendSpec spec;
        spec.sample_rate = sample_rate;
        spec.num_mels = 128;
        spec.n_fft = 512;
        spec.win_length = 400;
        spec.hop_length = 160;
        spec.window_type = WindowType::HannSymmetric;
        spec.normalize_mode = NormalizeMode::PerFeature;

        MelExtractor extractor(spec);
        std::vector<float> mel;
        int out_mels = 0;
        int out_frames = 0;
        bool ok = extractor.compute(pcm.data(), pcm.size(), mel, out_mels, out_frames, 2);
        assert(ok);
        assert(out_mels == 128);
        assert(out_frames == 101);
        assert(mel.size() == static_cast<size_t>(out_mels * out_frames));

        // Check finite and zero mean per mel bin
        for (int m = 0; m < out_mels; ++m) {
            double mean = 0.0;
            for (int t = 0; t < out_frames; ++t) {
                float val = mel[static_cast<size_t>(m * out_frames + t)];
                assert(std::isfinite(val));
                mean += val;
            }
            mean /= out_frames;
            assert(std::abs(mean) < 1e-4);
        }
        std::cout << "  [PASS] Parakeet-style per-feature normalization parity" << std::endl;
    }

    // 2. Test MelExtractor with PerUtterance normalization (Whisper style)
    {
        FrontendSpec spec;
        spec.sample_rate = sample_rate;
        spec.num_mels = 80;
        spec.n_fft = 400;
        spec.win_length = 400;
        spec.hop_length = 160;
        spec.window_type = WindowType::HannPeriodic;
        spec.normalize_mode = NormalizeMode::PerUtterance;

        MelExtractor extractor(spec);
        std::vector<float> mel;
        int out_mels = 0;
        int out_frames = 0;
        bool ok = extractor.compute(pcm.data(), pcm.size(), mel, out_mels, out_frames, 2);
        assert(ok);
        assert(out_mels == 80);
        assert(out_frames == 100); // 101 - 1 dropped trailing frame
        assert(mel.size() == static_cast<size_t>(out_mels * out_frames));

        // Whisper normalization is log10 -> clamp to (max - 8) -> (x + 4) / 4, so
        // the absolute range follows the input's loudness; what it guarantees
        // is a dynamic range of at most 8 / 4 = 2.0 with the maximum at
        // (max_log10 + 4) / 4. The fixed [-0.1, 1.5] window asserted here
        // before 2026-09-23 was never exercised (NDEBUG) and is wrong for this
        // 0.5-amplitude tone.
        float min_v = mel.front();
        float max_v = mel.front();
        for (float val : mel) {
            assert(std::isfinite(val));
            min_v = std::min(min_v, val);
            max_v = std::max(max_v, val);
        }
        std::cout << "  whisper mel range [" << min_v << ", " << max_v << "]" << std::endl;
        assert(max_v - min_v <= 2.0f + 1e-5f);
        assert(max_v > min_v);  // a pure tone is not flat across 80 bins
        std::cout << "  [PASS] Whisper-style per-utterance normalization parity" << std::endl;
    }

    // 3. Test RawPcm passthrough
    {
        FrontendSpec spec;
        spec.kind = FrontendKind::RawPcm;

        MelExtractor extractor(spec);
        std::vector<float> out_pcm;
        int out_mels = 0;
        int out_frames = 0;
        bool ok = extractor.compute(pcm.data(), pcm.size(), out_pcm, out_mels, out_frames, 1);
        assert(ok);
        assert(out_mels == 1);
        assert(out_frames == static_cast<int>(n_samples));
        assert(out_pcm == pcm);
        std::cout << "  [PASS] RawPcm exact bit-level passthrough" << std::endl;
    }

    std::cout << "[frontend_parity_test] All parity checks PASSED." << std::endl;
    return 0;
}
