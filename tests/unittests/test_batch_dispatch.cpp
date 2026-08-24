// test_batch_dispatch.cpp - unit tests for batch decode dispatch and pre-clear validation (Fusion Roadmap §7.5).

#include "audiocpp.h"
#include "transcribe-arch.h"
#include "transcribe-arch-adapter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void test_capi_asr_batch_null_args() {
    audiocpp_error_t err{};

    // Null model
    audiocpp_text_batch_t * batch = audiocpp_asr_batch(nullptr, nullptr, nullptr, 0, 16000, nullptr, &err);
    CHECK(batch == nullptr);
    CHECK(err.code != 0);
    audiocpp_clear_error(&err);

    // Free batch null safety
    audiocpp_free_text_batch(nullptr);
}

void test_adapter_validation_hooks() {
    const transcribe::Arch * arch = transcribe::adapter_find_arch("qwen3_asr");
    CHECK(arch != nullptr);
    if (!arch) return;

    CHECK(arch->run_validate != nullptr);
    CHECK(arch->stream_validate != nullptr);

    // Test run_validate with NULL session -> INVALID_ARG
    transcribe_status st = arch->run_validate(nullptr, nullptr);
    CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);

    // Test stream_validate with NULL session -> INVALID_ARG
    st = arch->stream_validate(nullptr, nullptr, nullptr);
    CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
}

void test_adapter_run_batch_null_args() {
    const transcribe::Arch * arch = transcribe::adapter_find_arch("citrinet_asr");
    CHECK(arch != nullptr);
    if (!arch) return;

    CHECK(arch->run_batch != nullptr);

    // Test run_batch with NULL ctx -> INVALID_ARG
    transcribe_status st = arch->run_batch(nullptr, nullptr, nullptr, 0, nullptr);
    CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
}

}  // namespace

int main() {
    test_capi_asr_batch_null_args();
    test_adapter_validation_hooks();
    test_adapter_run_batch_null_args();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_batch_dispatch: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_batch_dispatch: ALL PASSED\n");
    return 0;
}
