// tests/unittests/test_voxtral_realtime_delay.cpp - Phase 10.5 unit test for
// the transcription-delay contract merged from transcribe.cpp's typed stream
// extension (include/transcribe/voxtral_realtime.h).
//
// The publisher's validated set is a multiple of 80 ms in [80, 1200]
// (tokens 1..15) OR the standalone 2400 ms (token 30). Values outside it are
// rejected even though the architecture would run them, which is exactly what
// the C ABI documented; this test pins that rule so the engine cannot quietly
// widen or narrow it. Header-only, no model.

#include "engine/models/voxtral_realtime/types.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using engine::models::voxtral_realtime::is_valid_voxtral_realtime_delay;
using engine::models::voxtral_realtime::kVoxtralRealtimeDelayUnset;
using engine::models::voxtral_realtime::validate_voxtral_realtime_delay;

bool rejects(int64_t value) {
    try {
        (void) validate_voxtral_realtime_delay(value);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

void test_publisher_validated_set() {
    for (int64_t tokens = 1; tokens <= 15; ++tokens) {
        CHECK(is_valid_voxtral_realtime_delay(tokens));
        CHECK(validate_voxtral_realtime_delay(tokens) == tokens);
    }
    // 2400 ms is validated standalone; the gap 16..29 is not.
    CHECK(is_valid_voxtral_realtime_delay(30));
    CHECK(validate_voxtral_realtime_delay(30) == 30);
    for (int64_t tokens = 16; tokens <= 29; ++tokens) {
        CHECK(!is_valid_voxtral_realtime_delay(tokens));
        CHECK(rejects(tokens));
    }
}

void test_out_of_range_is_rejected() {
    CHECK(!is_valid_voxtral_realtime_delay(0));
    CHECK(rejects(0));
    CHECK(rejects(31));
    CHECK(rejects(1000));
    // Negative values other than the unset sentinel are caller bugs.
    CHECK(rejects(-2));
    CHECK(!is_valid_voxtral_realtime_delay(kVoxtralRealtimeDelayUnset));
}

void test_unset_passes_through() {
    // The sentinel is not a delay; it means "use the model default", and
    // validation must hand it back untouched for the caller to resolve.
    CHECK(validate_voxtral_realtime_delay(kVoxtralRealtimeDelayUnset) == kVoxtralRealtimeDelayUnset);
}

void test_message_names_the_rule() {
    // A caller that passes 20 should learn the accepted set from the error,
    // not have to read the source.
    try {
        (void) validate_voxtral_realtime_delay(20);
        CHECK(false);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        CHECK(message.find("1..15") != std::string::npos);
        CHECK(message.find("30") != std::string::npos);
    }
}

}  // namespace

int main() {
    test_publisher_validated_set();
    test_out_of_range_is_rejected();
    test_unset_passes_through();
    test_message_names_the_rule();
    if (g_failures != 0) {
        std::fprintf(stderr, "voxtral_realtime_delay_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("voxtral_realtime_delay_test: ALL PASSED\n");
    return 0;
}
