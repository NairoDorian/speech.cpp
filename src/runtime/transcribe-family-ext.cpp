// transcribe-family-ext.cpp - public extension initializers (and, for
// Whisper, the chunk-trace accessors) for families whose transcribe.cpp arch
// has been retired (Phase 10.5 / 11).
//
// A family's typed extension is part of the public C ABI: the struct, its kind
// constant and its init function are published in include/transcribe/<family>.h
// and callers link against them. The struct and the constant are header-only
// and survive on their own; the init function had a definition inside the arch
// that owned the family. When an arch is deleted the definition has to land
// somewhere that still ships, or the retirement silently breaks the ABI.
//
// That is what this file is. It holds no model code - only the initializers -
// and each entry names the family and the commit that retired its arch. The
// knobs themselves are served by the ArchAdapter, which translates the
// extension into the framework session's request options.

#include "transcribe-abi.h"      // check_struct_size, copy_out_prefix
#include "transcribe-arch.h"     // Arch (session->model->arch)
#include "transcribe-model.h"    // transcribe_model (session->model)
#include "transcribe-session.h"  // transcribe_session::decode_traces
#include "transcribe/moonshine_streaming.h"
#include "transcribe/sortformer.h"
#include "transcribe/voxtral_realtime.h"
#include "transcribe/whisper.h"

#include <cstddef>
#include <cstring>

// voxtral_realtime: arch retired in Phase 10.5 (ledger B12). The adapter
// validates and applies both fields; see adapter_check_stream_ext /
// adapter_apply_stream_ext in transcribe-arch-adapter.cpp.
extern "C" void transcribe_voxtral_realtime_stream_ext_init(struct transcribe_voxtral_realtime_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size               = sizeof(*p);
    p->ext.kind               = TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM;
    p->num_delay_tokens       = -1;
    p->min_decode_interval_ms = -1;
}

// sortformer: arch retired in Phase 10.5 (ledger B15). The engine
// `sortformer_diar` family serves both packages through the adapter, which
// translates the preset into the stream_preset request option; the parakeet
// multitalker bundle keeps the embedded-diarizer core. The init function
// stamps the transcribe_ext header (size + kind) and the preset default.
extern "C" void transcribe_sortformer_stream_ext_init(struct transcribe_sortformer_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM;
    p->preset   = TRANSCRIBE_SORTFORMER_PRESET_DEFAULT;
}

// moonshine_streaming: arch retired in Phase 11b (ledger B16b). The native
// engine `moonshine_streaming` package serves streaming and offline ASR
// through the adapter. The init function stamps the transcribe_ext header
// (size + kind) and default min_decode_interval_ms (-1 -> 240 ms).
extern "C" void transcribe_moonshine_streaming_stream_ext_init(struct transcribe_moonshine_streaming_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size               = sizeof(*p);
    p->ext.kind               = TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM;
    p->min_decode_interval_ms = -1;
}

// whisper: arch retired in Phase 11 Wave W2b (ledger B16c). The native engine
// `whisper` package serves offline ASR for both distribution formats (the
// transcribe.cpp GGUF and the whisper.cpp .bin) through the adapter, which
// validates this extension and forwards its fields as request options (see
// adapter_check_run_ext / adapter_apply_run_ext). The defaults below are the
// Whisper recipe (openai/whisper transcribe(), HF generate_with_fallback); the
// engine applies the same values when no extension is passed.
extern "C" void transcribe_whisper_run_ext_init(struct transcribe_whisper_run_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_WHISPER_RUN;
    // Zero-valued defaults come from the memset: initial_prompt / prompt_tokens
    // NULL, n_prompt_tokens 0, condition_on_prev_tokens false, temperature 0.0,
    // seed 0.
    p->prompt_condition        = TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;
    p->max_prev_context_tokens = 223;  // max_target_positions / 2 - 1
    p->temperature_inc         = 0.2f;
    p->compression_ratio_thold = 2.4f;
    p->logprob_thold           = -1.0f;
    p->no_speech_thold         = 0.6f;
    p->max_initial_timestamp   = 1.0f;
}

extern "C" void transcribe_whisper_chunk_trace_init(struct transcribe_whisper_chunk_trace * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

namespace {

// The library relies on the prefix up to and including n_fallbacks (the last
// field). Matches the central dispatcher's TRANSCRIBE_FIELD_END rule.
constexpr size_t k_min_whisper_chunk_trace_size = offsetof(struct transcribe_whisper_chunk_trace, n_fallbacks) +
                                                  sizeof(((struct transcribe_whisper_chunk_trace *) 0)->n_fallbacks);

// The traces of a Whisper session, or nullptr for any other family: the
// accessors are defined as "0 / zeroed on non-Whisper contexts" (whisper.h),
// and the arch name is the family check (no RTTI needed). The adapter fills
// transcribe_session::decode_traces from TaskResult::decode_telemetry.
const std::vector<transcribe_session::DecodeTraceEntry> * whisper_traces(const struct transcribe_session * session) {
    if (session == nullptr || session->model == nullptr || session->model->arch == nullptr) {
        return nullptr;
    }
    const char * name = session->model->arch->name;
    if (name == nullptr || std::strcmp(name, "whisper") != 0) {
        return nullptr;
    }
    return &session->decode_traces;
}

}  // namespace

extern "C" int transcribe_get_whisper_chunk_count(const struct transcribe_session * session) {
    const auto * traces = whisper_traces(session);
    return traces == nullptr ? 0 : static_cast<int>(traces->size());
}

extern "C" transcribe_status transcribe_get_whisper_chunk_trace(const struct transcribe_session *       session,
                                                                int                                     i,
                                                                struct transcribe_whisper_chunk_trace * out_trace) {
    if (out_trace == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = transcribe::check_struct_size(out_trace->struct_size, k_min_whisper_chunk_trace_size);
        st != TRANSCRIBE_OK) {
        return st;
    }
    const uint64_t                 caller_size = out_trace->struct_size;
    transcribe_whisper_chunk_trace staged{};
    staged.struct_size = caller_size;

    // Out-of-range indices and non-Whisper sessions read back zeroed.
    const auto * traces = whisper_traces(session);
    if (traces != nullptr && i >= 0 && static_cast<size_t>(i) < traces->size()) {
        const transcribe_session::DecodeTraceEntry & t = (*traces)[static_cast<size_t>(i)];
        staged.t0_ms               = t.t0_ms;
        staged.t1_ms               = t.t1_ms;
        staged.temperature_used    = t.temperature_used;
        staged.compression_ratio   = t.compression_ratio;
        staged.avg_logprob         = t.avg_logprob;
        staged.no_speech_prob      = t.no_speech_prob;
        staged.no_speech_triggered = t.no_speech_triggered;
        staged.n_fallbacks         = t.n_fallbacks;
    }
    transcribe::copy_out_prefix(out_trace, &staged, caller_size, sizeof(staged));
    return TRANSCRIBE_OK;
}
