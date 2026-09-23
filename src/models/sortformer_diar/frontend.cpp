#include "engine/models/sortformer_diar/frontend.h"

#include "engine/framework/audio/nemo_mel_frontend.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace engine::models::sortformer_diar {

// Upstream audio.cpp (#628) moved the Sortformer mel frontend onto the shared
// NemoMelFrontend. speech.cpp's package layouts (SortformerPackageLayout) carry
// a layout-dependent frontend contract, so the shared frontend is configured
// from it rather than from the HF defaults alone:
//   - peak_normalize   -> WaveScale (HF scales to peak; NeMo does not)
//   - normalize        -> MelNorm   (HF per_feature; NeMo GGUF may be "NA")
//   - frame_count      -> ValidFrameRule at run time (HF floor; NeMo ceil)
audio::NemoMelFrontend make_sortformer_frontend(const SortformerAssets & assets) {
    const auto & source = assets.feature_config;
    audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.num_mel_bins;
    // Caution: this is based on NeMo Sortformer preprocessing, which uses
    // torch.stft(..., center=True, pad_mode="constant").
    config.stft = {source.n_fft, source.hop_length, source.win_length,
                   true, audio::STFTPadMode::Constant, audio::STFTFamily::Default};
    config.input_rate = audio::MelInputRate::RequireMatch;
    config.wave_scale = source.peak_normalize
        ? audio::WaveScale::DivideByMaxPlusEps
        : audio::WaveScale::None;
    config.preemphasis = source.preemphasis;
    config.mel_path = audio::MelPath::LogMelSpectrogram;
    config.norm = source.normalize == SortformerFeatureNormalize::PerFeature
        ? audio::MelNorm::PerBinF32
        : audio::MelNorm::None;
    config.frame_multiple = 16;
    config.pad_basis = audio::PadBasis::ValidFrames;
    return audio::NemoMelFrontend(std::move(config));
}

SortformerFeatureBatch compute_sortformer_features(
    const runtime::AudioBuffer & audio,
    const SortformerAssets & assets,
    int64_t threads,
    SortformerRunTimings * timings) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Sortformer diar requires positive sample rate");
    }
    if (audio.sample_rate != assets.feature_config.sample_rate) {
        throw std::runtime_error("Sortformer diar currently requires 16 kHz input audio");
    }
    if (assets.frontend == nullptr) {
        throw std::runtime_error("Sortformer diar assets have no mel frontend");
    }
    const auto & feature_config = assets.feature_config;
    // Caution: use the valid frame count, not the raw centered-STFT frame count,
    // when sizing the padded feature buffer. The HF port tracks floor(n / hop);
    // NeMo's preprocessor reports ceil(n / hop) (the transcribe.cpp frontend's
    // nemo_seq_len_ceil). The centered STFT always yields 1 + floor(n / hop)
    // frames, so either count fits in what was computed.
    const auto valid_rule = feature_config.frame_count == SortformerFrameCount::Ceil
        ? audio::ValidFrameRule::CeilHops
        : audio::ValidFrameRule::FloorHops;
    const auto mel = assets.frontend->extract_audio(
        audio.samples, audio.sample_rate, audio.channels,
        {true, valid_rule},
        static_cast<size_t>(std::max<int64_t>(1, threads)));
    if (timings != nullptr) {
        timings->log_mel_ms += mel.mel_ms;
        timings->feature_normalizer_ms += mel.normalize_ms;
    }
    SortformerFeatureBatch batch;
    batch.frames = mel.frames;
    batch.valid_frames = mel.valid_frames;
    batch.time_major = mel.values;
    // speech.cpp contract: frames at or past valid_frames are zero padding.
    // PerFeature normalization already zeroes them; with normalize=None the
    // shared frontend copies up to raw_frames, which can leave one raw
    // centered-STFT frame past the valid count (NeMo ceil vs 1 + floor).
    const int64_t n_mels = feature_config.num_mel_bins;
    for (int64_t t = batch.valid_frames; t < batch.frames; ++t) {
        std::fill_n(
            batch.time_major.begin() + static_cast<std::ptrdiff_t>(t * n_mels),
            static_cast<size_t>(n_mels),
            0.0f);
    }
    return batch;
}

}  // namespace engine::models::sortformer_diar
