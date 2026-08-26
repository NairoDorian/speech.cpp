#pragma once

#include "engine/models/voxtral_realtime/assets.h"
#include "engine/models/voxtral_realtime/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::voxtral_realtime {

class VoxtralRealtimeTokenizer {
public:
    explicit VoxtralRealtimeTokenizer(std::shared_ptr<const VoxtralRealtimeAssets> assets);
    ~VoxtralRealtimeTokenizer();

    // num_delay_tokens defaults to the model value when left unset. The
    // prompt's streaming pad and the audio-token count both scale with it, so
    // it has to be the same value the frontend geometry used.
    VoxtralRealtimePrompt build_transcription_prompt(
        int64_t audio_samples,
        bool streaming,
        int64_t num_delay_tokens = kVoxtralRealtimeDelayUnset) const;
    std::string decode(const std::vector<int32_t> & token_ids) const;
    bool is_stream_text_token(int32_t token_id) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace engine::models::voxtral_realtime
