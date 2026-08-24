#include "engine/framework/audio/frontend_spec.h"
#include "engine/framework/audio/mel_extractor.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace engine::audio;

int main() {
    std::cout << "[frontend_contract_test] Starting frontend contract tests..." << std::endl;

    // 1. Whisper (128 mels, 400 n_fft, 160 hop, hann_periodic, per_utterance)
    {
        FrontendSpec spec;
        spec.sample_rate = 16000;
        spec.num_mels = 128;
        spec.n_fft = 400;
        spec.win_length = 400;
        spec.hop_length = 160;
        spec.window_type = WindowType::HannPeriodic;
        spec.normalize_mode = NormalizeMode::PerUtterance;
        spec.pad_mode = PadMode::Reflect;

        MelExtractor extractor(spec);
        assert(extractor.num_mels() == 128);
        assert(extractor.n_frames_for(16000) == 101); // 16000/160 + 1
        std::cout << "  [PASS] Whisper contract" << std::endl;
    }

    // 2. Parakeet / NeMo (128 mels, 512 n_fft, 400 win, 160 hop, hann_symmetric, per_feature)
    {
        FrontendSpec spec;
        spec.sample_rate = 16000;
        spec.num_mels = 128;
        spec.n_fft = 512;
        spec.win_length = 400;
        spec.hop_length = 160;
        spec.window_type = WindowType::HannSymmetric;
        spec.normalize_mode = NormalizeMode::PerFeature;
        spec.pad_mode = PadMode::Reflect;

        MelExtractor extractor(spec);
        assert(extractor.num_mels() == 128);
        assert(extractor.n_frames_for(16000) == 101);
        std::cout << "  [PASS] Parakeet contract" << std::endl;
    }

    // 3. GigaAM (64 mels, 320 n_fft, 320 win, 160 hop, hann_periodic, none normalize, pad_mode none)
    {
        FrontendSpec spec;
        spec.sample_rate = 16000;
        spec.num_mels = 64;
        spec.n_fft = 320;
        spec.win_length = 320;
        spec.hop_length = 160;
        spec.window_type = WindowType::HannPeriodic;
        spec.normalize_mode = NormalizeMode::None;
        spec.pad_mode = PadMode::None;

        MelExtractor extractor(spec);
        assert(extractor.num_mels() == 64);
        assert(extractor.n_frames_for(16000) == (16000 - 320) / 160 + 1); // 99 frames
        std::cout << "  [PASS] GigaAM contract" << std::endl;
    }

    // 4. Moonshine / RawPcm (1 mel channel, raw pcm passthrough)
    {
        FrontendSpec spec;
        spec.kind = FrontendKind::RawPcm;
        spec.sample_rate = 16000;

        MelExtractor extractor(spec);
        assert(extractor.num_mels() == 1);
        assert(extractor.n_frames_for(16000) == 16000);
        std::cout << "  [PASS] RawPcm / Moonshine contract" << std::endl;
    }

    // 5. SenseVoice / FunASR (80 mels, 400 n_fft, 160 hop, hamming)
    {
        FrontendSpec spec;
        spec.sample_rate = 16000;
        spec.num_mels = 80;
        spec.n_fft = 400;
        spec.win_length = 400;
        spec.hop_length = 160;
        spec.window_type = WindowType::Hamming;
        spec.normalize_mode = NormalizeMode::PerFeature;

        MelExtractor extractor(spec);
        assert(extractor.num_mels() == 80);
        assert(extractor.n_frames_for(16000) == 101);
        std::cout << "  [PASS] SenseVoice / FunASR contract" << std::endl;
    }

    std::cout << "[frontend_contract_test] All contract checks PASSED." << std::endl;
    return 0;
}
