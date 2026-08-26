#pragma once

#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::voxtral_realtime {

// Transcription delay in 12.5 Hz audio tokens (80 ms each): the model emits
// the token for audio position t after observing audio through
// t + num_delay_tokens, trading latency for accuracy. Merged from the
// transcribe.cpp typed stream extension in Phase 10.5, including its
// validation rule: the publisher's validated set is a multiple of 80 ms in
// [80, 1200] (tokens 1..15) OR the standalone 2400 ms (token 30). Other
// values are rejected even though the architecture would run them - they are
// outside what the model card and mistral-common validate.
//
// kVoxtralRealtimeDelayUnset (-1) means "use the model default"
// (VoxtralRealtimeConfig::default_num_delay_tokens, 6 on the published
// checkpoint).
constexpr int64_t kVoxtralRealtimeDelayUnset = -1;
constexpr int64_t kVoxtralRealtimeMaxContiguousDelayTokens = 15;
constexpr int64_t kVoxtralRealtimeLongDelayTokens = 30;

inline bool is_valid_voxtral_realtime_delay(int64_t num_delay_tokens) {
    return (num_delay_tokens >= 1 && num_delay_tokens <= kVoxtralRealtimeMaxContiguousDelayTokens) ||
        num_delay_tokens == kVoxtralRealtimeLongDelayTokens;
}

// Throws on a value outside the publisher's validated set; passes
// kVoxtralRealtimeDelayUnset through untouched for the caller to default.
inline int64_t validate_voxtral_realtime_delay(int64_t num_delay_tokens) {
    if (num_delay_tokens == kVoxtralRealtimeDelayUnset || is_valid_voxtral_realtime_delay(num_delay_tokens)) {
        return num_delay_tokens;
    }
    throw std::runtime_error(
        "VoxTral realtime num_delay_tokens must be 1..15 (80..1200 ms) or 30 (2400 ms); got " +
        std::to_string(num_delay_tokens));
}

struct VoxtralRealtimeFeatures {
    std::vector<float> values;
    int64_t mel_bins = 0;
    int64_t frames = 0;
};

struct VoxtralRealtimePrompt {
    std::vector<int32_t> input_ids;
    int64_t audio_tokens = 0;
    int64_t num_delay_tokens = 0;
};

struct VoxtralRealtimeAudioEmbeddings {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct VoxtralRealtimeAudioEncoderStreamState {
    bool first_chunk = true;
    int64_t seen_encoder_steps = 0;
    int64_t cached_encoder_steps = 0;
};

struct VoxtralRealtimeGenerationOptions {
    int64_t max_new_tokens = 0;
    bool max_new_tokens_set = false;
    bool do_sample = false;
    float temperature = 1.0F;
    float top_p = 1.0F;
    int64_t top_k = 50;
    uint64_t seed = 1234;
    // Cooperative cancellation (Phase 10.5, the arch's
    // TRANSCRIBE_FEATURE_CANCELLATION): polled once per decode step, so
    // request_abort() or a declining progress callback unwinds a long decode
    // instead of running it to completion.
    const runtime::RunControl * run_control = nullptr;
};

struct VoxtralRealtimeGeneratedTokens {
    std::vector<int32_t> token_ids;
};

struct VoxtralRealtimeRequest {
    runtime::AudioBuffer audio;
    VoxtralRealtimeGenerationOptions generation;
    bool streaming = false;
    // kVoxtralRealtimeDelayUnset unless the caller asked for a specific
    // transcription delay; the frontend pad and the prompt must agree on it.
    int64_t num_delay_tokens = kVoxtralRealtimeDelayUnset;
};

}  // namespace engine::models::voxtral_realtime
