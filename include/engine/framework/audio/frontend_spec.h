#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::audio {

enum class FrontendKind {
    MelSpectrogram,
    KaldiFbank,
    RawPcm,
};

enum class PadMode {
    Reflect,
    Constant,
    None,
};

enum class WindowType {
    HannSymmetric,
    HannPeriodic,
    Hamming,
    Custom,
};

enum class NormalizeMode {
    PerFeature,    // NeMo: per-mel-bin zero-mean / unit-variance (unbiased n-1)
    PerUtterance,  // Whisper: log10, global clamp to (max - 8.0), (x + 4) / 4, drop trailing frame
    Global,        // Voxtral Realtime: log-clamp floor with fixed global_log_mel_max, drop trailing frame
    None,          // Raw log-mel as-is
};

struct FrontendSpec {
    FrontendKind kind = FrontendKind::MelSpectrogram;

    int   sample_rate        = 16000;
    int   num_mels           = 128;
    int   n_fft              = 512;
    int   win_length         = 400;
    int   hop_length         = 160;
    float pre_emphasis       = 0.97f;
    float f_min              = 0.0f;
    float f_max              = 8000.0f;

    PadMode       pad_mode       = PadMode::Reflect;
    WindowType    window_type    = WindowType::HannSymmetric;
    NormalizeMode normalize_mode = NormalizeMode::PerFeature;

    float global_log_mel_max = 1.5f;
    float log_clamp_min      = 0.0f;
    bool  nemo_seq_len_ceil  = false;

    // Kaldi fbank options
    float dither             = 0.0f;
    float energy_floor       = 0.0f;
    bool  use_energy         = false;
    int   lfr_m              = 1;
    int   lfr_n              = 1;

    // Optional pre-baked buffers (e.g. from GGUF)
    std::vector<float> filterbank;  // [num_mels * (n_fft / 2 + 1)]
    std::vector<float> window;      // [win_length]
};

}  // namespace engine::audio
