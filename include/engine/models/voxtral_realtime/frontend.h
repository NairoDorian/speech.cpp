#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/models/voxtral_realtime/assets.h"
#include "engine/models/voxtral_realtime/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::voxtral_realtime {

struct VoxtralRealtimeFrontendStreamState {
    std::vector<float> cached_magnitude_frame;
    int64_t cached_freq_bins = 0;
    bool cached_frame_ready = false;
};

class VoxtralRealtimeFrontend {
public:
    explicit VoxtralRealtimeFrontend(std::shared_ptr<const VoxtralRealtimeAssets> assets);

    // num_delay_tokens defaults to the model value when left unset; it sets
    // the right-hand pad of the offline pass and the first streaming chunk's
    // geometry, so every caller has to agree on one value per stream.
    VoxtralRealtimeFeatures extract(
        const runtime::AudioBuffer & audio,
        bool first_chunk,
        int64_t num_delay_tokens = kVoxtralRealtimeDelayUnset) const;
    // A steady chunk carries `steady_tokens` audio tokens of 80 ms each; batching them into one
    // extraction yields 8 * steady_tokens feature frames from a single STFT pass.
    // num_delay_tokens sets the FIRST chunk's length (a steady chunk does not
    // depend on it); it must be the same value the prompt was built with.
    VoxtralRealtimeFeatures extract_stream_chunk(
        const runtime::AudioBuffer & audio,
        bool first_chunk,
        VoxtralRealtimeFrontendStreamState & state,
        int64_t steady_tokens = 1,
        int64_t num_delay_tokens = kVoxtralRealtimeDelayUnset) const;

    int64_t first_stream_chunk_samples(int64_t num_delay_tokens = kVoxtralRealtimeDelayUnset) const;
    int64_t steady_stream_chunk_samples(int64_t steady_tokens = 1) const;
    int64_t first_stream_chunk_advance_samples(int64_t num_delay_tokens = kVoxtralRealtimeDelayUnset) const;
    int64_t steady_stream_chunk_advance_samples(int64_t steady_tokens = 1) const;

    // The value the geometry helpers above would use for `num_delay_tokens`.
    int64_t effective_num_delay_tokens(int64_t num_delay_tokens = kVoxtralRealtimeDelayUnset) const;

private:
    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
    engine::audio::SparseMelFilterbank mel_filterbank_;
};

}  // namespace engine::models::voxtral_realtime
