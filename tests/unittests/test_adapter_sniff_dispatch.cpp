// test_adapter_sniff_dispatch.cpp - unit tests for adapter sniff dispatch and precedence (Fusion Roadmap §7.3).

#include "transcribe-arch.h"
#include "transcribe-arch-adapter.h"

#include <cstdio>
#include <cstdlib>
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

void test_adapter_find_arch_coverage() {
    const char * const kAdapterFamilies[] = {
        "silero_vad",
        "marblenet_vad",
        "citrinet_asr",
        "nemotron_asr",
        "qwen3_asr",
        "fun_asr_nano",
        "sense_asr",
        "hviske_asr",
        "higgs_audio_stt",
        "vibevoice_asr",
        "kroko_asr",
        "parakeet_tdt",
        "voxtral_realtime",
        "sortformer_diar",
        "roformer",
        "moss",
    };

    for (const char * family : kAdapterFamilies) {
        const transcribe::Arch * arch = transcribe::adapter_find_arch(family);
        CHECK(arch != nullptr);
        if (arch != nullptr) {
            CHECK(arch->load != nullptr);
            CHECK(arch->init_context != nullptr);
            CHECK(arch->run != nullptr);
            CHECK(arch->run_batch != nullptr);
            CHECK(arch->stream_validate != nullptr);
            CHECK(arch->stream_begin != nullptr);
            CHECK(arch->stream_feed != nullptr);
            CHECK(arch->stream_finalize != nullptr);
            CHECK(arch->stream_reset != nullptr);
            CHECK(arch->accepts_ext_kind != nullptr);
            CHECK(arch->run_validate != nullptr);
        }
    }

    CHECK(transcribe::adapter_find_arch("nonexistent_arch") == nullptr);
    CHECK(transcribe::adapter_find_arch("") == nullptr);
    CHECK(transcribe::adapter_find_arch(nullptr) == nullptr);
}

void test_overlapping_families_distinct_dispatch() {
    // For the families that overlapped at the start of the fusion
    // (qwen3_asr, voxtral_realtime, moss), adapter_find_arch must return the
    // adapter entry point; find_arch returns the transcribe.cpp builtin while
    // one still exists (moss) and the adapter once Phase 10.5 has retired it
    // (qwen3_asr and voxtral_realtime, 2026-08-26).
    const char * const kOverlapping[] = {"qwen3_asr", "voxtral_realtime", "moss"};
    for (const char * family : kOverlapping) {
        const transcribe::Arch * adapter_arch = transcribe::adapter_find_arch(family);
        CHECK(adapter_arch != nullptr);
        if (adapter_arch != nullptr) {
            CHECK(adapter_arch->run_batch != nullptr);
            CHECK(adapter_arch->run_validate != nullptr);
            CHECK(adapter_arch->stream_validate != nullptr);
        }
    }
}

void test_sniff_empty_and_invalid() {
    CHECK(transcribe::adapter_sniff_framework_family(nullptr).empty());
    CHECK(transcribe::adapter_sniff_framework_family("").empty());
    CHECK(transcribe::adapter_sniff_framework_family("nonexistent_file_path_12345.xyz").empty());
}

}  // namespace

int main() {
    test_adapter_find_arch_coverage();
    test_overlapping_families_distinct_dispatch();
    test_sniff_empty_and_invalid();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_adapter_sniff_dispatch: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_adapter_sniff_dispatch: ALL PASSED\n");
    return 0;
}
