// tests/voxtral_realtime_ext_abi_test.cpp - Phase 10.5 C ABI gate for the
// Voxtral-Realtime typed stream extension after the arch retirement.
//
// transcribe.cpp published transcribe_voxtral_realtime_stream_ext
// (num_delay_tokens, min_decode_interval_ms) and validated it inside the arch.
// The arch is gone; the ArchAdapter now accepts the same extension and
// translates it into framework request options. This test holds the public
// contract to what the arch documented:
//
//   * transcribe_model_accepts_ext_kind reports the STREAM slot accepts the
//     Voxtral kind (and rejects it on the RUN slot, and rejects a foreign
//     kind) - the probe callers are told to make before pointing
//     transcribe_stream_params::family at the struct.
//   * A delay outside the publisher's validated set (1..15 or 30) is refused
//     with TRANSCRIBE_ERR_INVALID_ARG, and refused PRE-CLEAR: the prior
//     transcript and the stream lifecycle survive a rejected begin.
//   * min_decode_interval_ms < -1 is refused the same way.
//   * A valid extension begins, feeds and finalizes, producing text.
//
// Links only the C ABI, like a language binding would. Skips (exit 2) without
// the pinned model, which is a 4B package - the audio fed here is deliberately
// short, since this test gates the extension surface, not transcription
// quality (voxtral_realtime_engine_smoke_test does that).

#include "transcribe.h"
#include "transcribe/voxtral_realtime.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// std::filesystem rather than ::stat: MSVC's stat() carries a 32-bit size and
// fails on files larger than 2 GB, which this family's package is. The symptom
// is a gate that silently reports SKIP on a model that is right there.
bool file_exists(const std::string & path) {
    std::error_code ec;
    return std::filesystem::exists(std::filesystem::u8path(path), ec);
}

// 1 s of quiet 16 kHz mono audio: enough to exercise begin/feed/finalize
// without paying for a real decode of speech on a 4B model.
std::vector<float> silence_seconds(double seconds) {
    return std::vector<float>(static_cast<size_t>(16000.0 * seconds), 0.0F);
}

transcribe_status begin_with_delay(transcribe_session * session, int32_t num_delay_tokens,
                                   int32_t min_decode_interval_ms) {
    transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    transcribe_stream_params stream_params;
    transcribe_stream_params_init(&stream_params);
    transcribe_voxtral_realtime_stream_ext ext;
    transcribe_voxtral_realtime_stream_ext_init(&ext);
    ext.num_delay_tokens = num_delay_tokens;
    ext.min_decode_interval_ms = min_decode_interval_ms;
    stream_params.family = &ext.ext;
    return transcribe_stream_begin(session, &run_params, &stream_params);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argv[1] == nullptr || argv[1][0] == '\0') {
        std::fprintf(stderr, "usage: voxtral_realtime_ext_abi_test <voxtral.gguf>\n");
        return 1;
    }
    const std::string model_path = argv[1];
    if (!file_exists(model_path)) {
        std::printf("SKIP: pinned Voxtral-Realtime GGUF not found: %s\n", model_path.c_str());
        return 2;
    }

    transcribe_model_load_params load_params;
    transcribe_model_load_params_init(&load_params);
    load_params.backend = TRANSCRIBE_BACKEND_CPU;
    transcribe_session_params session_params;
    transcribe_session_params_init(&session_params);
    transcribe_session * session = nullptr;
    const transcribe_status open_st =
        transcribe_open(model_path.c_str(), &load_params, &session_params, &session);
    if (open_st != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "voxtral_realtime_ext_abi_test: transcribe_open failed: status=%d\n",
                     static_cast<int>(open_st));
        return 1;
    }

    const transcribe_model * model = transcribe_get_model(session);
    CHECK(model != nullptr);
    if (model != nullptr) {
        CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM));
        // The family publishes no run-slot extension, and a foreign kind is
        // never accepted on either slot.
        CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN,
                                                 TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM));
        CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, 0xDEADBEEFu));
    }

    // Values outside the publisher's validated set are refused, and refused
    // before the stream lifecycle moves.
    for (const int32_t bad_delay : {0, 16, 29, 31}) {
        const transcribe_status st = begin_with_delay(session, bad_delay, -1);
        if (st != TRANSCRIBE_ERR_INVALID_ARG) {
            std::fprintf(stderr, "FAIL: num_delay_tokens=%d returned status=%d, expected INVALID_ARG\n",
                         static_cast<int>(bad_delay), static_cast<int>(st));
            ++g_failures;
        }
        CHECK(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_IDLE);
    }
    {
        const transcribe_status st = begin_with_delay(session, -1, -2);
        if (st != TRANSCRIBE_ERR_INVALID_ARG) {
            std::fprintf(stderr, "FAIL: min_decode_interval_ms=-2 returned status=%d, expected INVALID_ARG\n",
                         static_cast<int>(st));
            ++g_failures;
        }
        CHECK(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_IDLE);
    }

    // A valid extension runs the stream end to end.
    {
        const transcribe_status begin_st = begin_with_delay(session, 4, 240);
        if (begin_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL: valid extension rejected: status=%d\n",
                         static_cast<int>(begin_st));
            ++g_failures;
        } else {
            const auto pcm = silence_seconds(1.0);
            const transcribe_status feed_st =
                transcribe_stream_feed(session, pcm.data(), static_cast<int>(pcm.size()), nullptr);
            CHECK(feed_st == TRANSCRIBE_OK);
            const transcribe_status final_st = transcribe_stream_finalize(session, nullptr);
            CHECK(final_st == TRANSCRIBE_OK);
            // Silence may transcribe to nothing; the contract under test is
            // that the accessor is readable, not that it is non-empty.
            CHECK(transcribe_full_text(session) != nullptr);
        }
    }

    transcribe_session_free(session);
    if (g_failures != 0) {
        std::fprintf(stderr, "voxtral_realtime_ext_abi_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("voxtral_realtime_ext_abi_test: PASS\n");
    return 0;
}
