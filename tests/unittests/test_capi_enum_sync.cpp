// Enum-sync unit tests between the C ABI surface and the engine runtime.
//
// Guards two contracts that upstream merges can silently break:
//   1. AUDIOCPP_ARTIFACT_* values mirror engine::runtime::ArtifactKind
//      numerically (documented in audiocpp.h). An upstream ArtifactKind
//      insertion without renumbering the C enum desynchronizes every
//      artifact consumer — this exact drift happened with Midi (=4).
//   2. map_task covers every AUDIOCPP_TASK_* constant with a DISTINCT
//      engine kind. A constant without a case silently falls back to Tts.
//
// Also locks the current task-constant numbering so a header renumber forces
// a conscious test update (and a review of every hardcoded task integer).
//
// This TU compiles audiocpp_capi.cpp directly (same static-link pattern as
// capi_option_number_test / capi_session_options_test).

#include "audiocpp_internal.h"

#include "engine/framework/runtime/session.h"

#include "test_assert.h"

#include <cstdio>
#include <set>

using engine::runtime::ArtifactKind;
using engine::runtime::VoiceTaskKind;

// --- Compile-time contracts -------------------------------------------------

static_assert(AUDIOCPP_ARTIFACT_SPEAKER_EMBEDDING == static_cast<int>(ArtifactKind::SpeakerEmbedding));
static_assert(AUDIOCPP_ARTIFACT_STYLE_EMBEDDING == static_cast<int>(ArtifactKind::StyleEmbedding));
static_assert(AUDIOCPP_ARTIFACT_PROMPT_EMBEDDING == static_cast<int>(ArtifactKind::PromptEmbedding));
static_assert(AUDIOCPP_ARTIFACT_ACOUSTIC_TOKENS == static_cast<int>(ArtifactKind::AcousticTokens));
static_assert(AUDIOCPP_ARTIFACT_MIDI == static_cast<int>(ArtifactKind::Midi));
static_assert(AUDIOCPP_ARTIFACT_ALIGNMENT == static_cast<int>(ArtifactKind::TranscriptAlignment));
static_assert(AUDIOCPP_ARTIFACT_DIARIZATION_STATE == static_cast<int>(ArtifactKind::DiarizationState));
static_assert(AUDIOCPP_ARTIFACT_VAD_STATE == static_cast<int>(ArtifactKind::VadState));
static_assert(AUDIOCPP_ARTIFACT_CUSTOM == static_cast<int>(ArtifactKind::Custom));

// Task constants are deliberately NOT value-mirrored (the C order predates
// the engine order); these locks just pin the current documented values.
static_assert(AUDIOCPP_TASK_VAD == 0);
static_assert(AUDIOCPP_TASK_ASR == 1);
static_assert(AUDIOCPP_TASK_DIAR == 2);
static_assert(AUDIOCPP_TASK_SEP == 3);
static_assert(AUDIOCPP_TASK_GEN == 4);
static_assert(AUDIOCPP_TASK_TTS == 5);
static_assert(AUDIOCPP_TASK_ALIGN == 6);
static_assert(AUDIOCPP_TASK_VC == 7);
static_assert(AUDIOCPP_TASK_CLON == 8);
static_assert(AUDIOCPP_TASK_S2S == 9);
static_assert(AUDIOCPP_TASK_VDES == 10);
static_assert(AUDIOCPP_TASK_SPK == 11);
static_assert(AUDIOCPP_TASK_SVC == 12);
static_assert(AUDIOCPP_TASK_MIDI == 13);

namespace {

struct TaskCase {
    int c_constant;
    const char * label;
    VoiceTaskKind expected;
};

// Every AUDIOCPP_TASK_* constant and the engine kind it must map to.
// When adding a task kind upstream: extend the C enum, map_task, AND this
// table — the injectivity check below then re-establishes full coverage.
const TaskCase kTaskCases[] = {
    {AUDIOCPP_TASK_TTS, "TTS", VoiceTaskKind::Tts},
    {AUDIOCPP_TASK_ASR, "ASR", VoiceTaskKind::Asr},
    {AUDIOCPP_TASK_VAD, "VAD", VoiceTaskKind::Vad},
    {AUDIOCPP_TASK_DIAR, "DIAR", VoiceTaskKind::Diarization},
    {AUDIOCPP_TASK_SEP, "SEP", VoiceTaskKind::SourceSeparation},
    {AUDIOCPP_TASK_GEN, "GEN", VoiceTaskKind::AudioGeneration},
    {AUDIOCPP_TASK_ALIGN, "ALIGN", VoiceTaskKind::Alignment},
    {AUDIOCPP_TASK_VC, "VC", VoiceTaskKind::VoiceConversion},
    {AUDIOCPP_TASK_CLON, "CLON", VoiceTaskKind::VoiceCloning},
    {AUDIOCPP_TASK_S2S, "S2S", VoiceTaskKind::SpeechToSpeech},
    {AUDIOCPP_TASK_VDES, "VDES", VoiceTaskKind::VoiceDesign},
    {AUDIOCPP_TASK_SPK, "SPK", VoiceTaskKind::SpeakerRecognition},
    {AUDIOCPP_TASK_SVC, "SVC", VoiceTaskKind::Svc},
    {AUDIOCPP_TASK_MIDI, "MIDI", VoiceTaskKind::Midi},
};

}  // namespace

int main() {
    // map_task: every constant maps to its expected engine kind...
    std::set<int> distinct;
    for (const auto & c : kTaskCases) {
        const auto mapped = audiocpp::detail::map_task(c.c_constant);
        engine::test::require(
            mapped == c.expected,
            std::string("map_task(") + c.label + ") returned wrong engine kind");
        distinct.insert(static_cast<int>(mapped));
    }
    // ...and the mapping is injective: 14 constants -> 14 distinct kinds means
    // full coverage of the current VoiceTaskKind enumerators. Two constants
    // collapsing onto one kind means a case is missing (default Tts fallback).
    engine::test::require_eq(
        distinct.size(), sizeof(kTaskCases) / sizeof(kTaskCases[0]),
        "map_task must be injective over all AUDIOCPP_TASK_* constants");

    // Unknown task ids degrade to Tts (documented fallback).
    engine::test::require(
        audiocpp::detail::map_task(-1) == VoiceTaskKind::Tts &&
            audiocpp::detail::map_task(9999) == VoiceTaskKind::Tts,
        "map_task unknown ids must fall back to Tts");

    std::printf("capi_enum_sync_test: all cases passed\n");
    return 0;
}
