#pragma once

// Internal helpers shared between the C ABI translation unit and its unit
// tests. NOT part of the public ABI (capi/include/audiocpp.h); do not export.

#include "audiocpp.h"

#include <string>
#include <unordered_map>

struct cJSON;

namespace engine::runtime {
struct AudioBuffer;
enum class VoiceTaskKind;
}

namespace audiocpp::detail {

// Map an AUDIOCPP_TASK_* constant to the engine task kind. Defined in
// audiocpp_capi.cpp at namespace scope (not file-static) so the enum-sync
// unit test can verify every C constant maps to a distinct engine kind —
// a new AUDIOCPP_TASK_* value without a matching case would silently fall
// back to Tts.
engine::runtime::VoiceTaskKind map_task(int task);

// Render a cJSON/JSON number into the string form stored in the options map.
//
// Option consumers later split into int (stoll) and float (stof) parsers.
// If an integral JSON number like 2 were rendered as "2.000000", stoll would
// reject it ("must be an integer") — the original bug this guards against.
//
// Integers within double's exact-integer range (< 2^53) render as plain integer
// strings. Values at or beyond 2^53 cannot be represented exactly by a double,
// so they are emitted via the float serializer (scientific notation); stoll
// then rejects them rather than silently rounding a value the caller never sent.
// This matches the CLI's app/cli/request.cpp json_option_string behavior.
std::string option_number_to_string(double value);

// Collect every string/number/bool member of a parsed JSON object into an
// options map, rendering values exactly like the request-options path:
// numbers via option_number_to_string, booleans as "true"/"false", strings
// passed through. Used by audiocpp_load_model_ex for session options.
void collect_option_fields(const cJSON * root, std::unordered_map<std::string, std::string> & options);

// Parse a session-options JSON string into a string map. NULL/"" yields an
// empty map (defaults). Malformed JSON or a non-object root returns false —
// load-time configuration must never be silently dropped.
bool parse_session_options_json(const char * options_json, std::unordered_map<std::string, std::string> & options);

// Pack an engine audio buffer into an owned audiocpp_audio_t, preserving the
// channel count (the stereo-output fix). File-local in audiocpp_capi.cpp (not
// exported); declared here so unit tests can verify channel propagation.
// n_samples is the total sample count (channels * frames). Caller owns the
// result (free with audiocpp_free_audio). Throws on allocation failure.
audiocpp_audio_t * pack_audio_output(const engine::runtime::AudioBuffer & buf);

}  // namespace audiocpp::detail
