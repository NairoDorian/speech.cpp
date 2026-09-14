#pragma once

#include <cstdint>
#include <string>

namespace engine::asr {

// Shared ASR transcription result. Engine ASR packages (Moonshine W1a/W1b,
// Whisper W2a, and future ports) return this from their transcribe() entry
// point instead of each defining a per-family struct. The `truncated` field
// carries the L11 rule: every ASR result must honestly report whether the
// model hit its context or audio limit before producing an EOS.
struct AsrResult {
    std::string text;
    bool truncated = false;
};

// Model-imposed limits that bound a single offline transcribe() call.
// Populated from model hyperparameters at load time; consulted by the
// session layer to decide truncation and to report effective limits back
// to the caller.
struct AsrLimits {
    // Maximum audio window the encoder accepts, in milliseconds. 0 = unlimited.
    int32_t max_audio_ms = 0;
    // Maximum decoder output tokens (context window). 0 = model default.
    int32_t max_output_tokens = 0;
    // Sample rate the model expects (encoder input).
    int32_t sample_rate = 16000;
};

}  // namespace engine::asr
