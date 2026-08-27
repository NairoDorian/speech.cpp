// tests/sortformer_diar_ext_abi_test.cpp - Phase 10.5 C ABI gate for the
// Sortformer streaming operating-point run extension, served by the
// ArchAdapter over the engine package.
//
// transcribe.cpp published transcribe_sortformer_stream_ext (a preset enum on
// the RUN slot) and validated it inside the arch. The engine package now owns
// the chunked scheduler the presets select, and the adapter translates the
// extension into the `stream_preset` request option. This test holds the
// public contract to what the arch's own unit test held:
//
//   * transcribe_model_accepts_ext_kind: the kind is accepted on RUN only;
//     a foreign kind is rejected on both slots.
//   * transcribe_sortformer_stream_ext_init stamps size / kind / DEFAULT.
//   * Pre-clear rejection: a wrong-kind ext and an out-of-range preset both
//     fail with INVALID_ARG and PRESERVE the previous speaker segments.
//   * Ext preset == env preset: running with ext=VERY_HIGH_LATENCY yields the
//     same rows as TRANSCRIBE_SORTFORMER_STREAM_PRESET=very_high_latency (the
//     validation-harness route), and LOW_LATENCY runs and produces rows.
//
// Links only the C ABI, like a language binding would. Skips (exit 2) without
// the pinned package or the fixtures.

#include "abi_test_wav.h"
#include "transcribe.h"
#include "transcribe/sortformer.h"

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

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    if (value[0] == '\0') {
        ::unsetenv(name);
    } else {
        ::setenv(name, value, 1);
    }
#endif
}

std::vector<transcribe_speaker_segment> read_segments(const transcribe_session * session) {
    std::vector<transcribe_speaker_segment> rows;
    const int n = transcribe_n_speaker_segments(session);
    for (int i = 0; i < n; ++i) {
        transcribe_speaker_segment row;
        transcribe_speaker_segment_init(&row);
        if (transcribe_get_speaker_segment(session, i, &row) == TRANSCRIBE_OK) {
            rows.push_back(row);
        }
    }
    return rows;
}

bool same_segments(const std::vector<transcribe_speaker_segment> & a,
                   const std::vector<transcribe_speaker_segment> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].t0_ms != b[i].t0_ms || a[i].t1_ms != b[i].t1_ms || a[i].speaker_id != b[i].speaker_id) {
            return false;
        }
    }
    return true;
}

// The longest LibriSpeech fixture (14 s): long enough that the presets differ
// in geometry while staying cheap.
std::filesystem::path pick_fixture(const std::filesystem::path & dir) {
    std::filesystem::path best;
    uintmax_t best_size = 0;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".wav" && entry.file_size() > best_size) {
            best = entry.path();
            best_size = entry.file_size();
        }
    }
    return best;
}

transcribe_status run_with_preset(transcribe_session * session, const std::vector<float> & pcm,
                                  transcribe_sortformer_preset preset) {
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    transcribe_sortformer_stream_ext ext;
    transcribe_sortformer_stream_ext_init(&ext);
    ext.preset = preset;
    rp.family = &ext.ext;
    return transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: sortformer_diar_ext_abi_test <sortformer.gguf> <librispeech_fixture_dir>\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::filesystem::path fixture_dir = argv[2];
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::u8path(model_path), ec)) {
        std::printf("SKIP: pinned Sortformer GGUF not found: %s\n", model_path.c_str());
        return 2;
    }
    if (!std::filesystem::is_directory(fixture_dir, ec)) {
        std::printf("SKIP: librispeech fixture dir not found\n");
        return 2;
    }
    const auto wav = pick_fixture(fixture_dir);
    if (wav.empty()) {
        std::printf("SKIP: no .wav fixture found\n");
        return 2;
    }
    int sample_rate = 0;
    const std::vector<float> pcm = abi_test::read_wav_mono_f32(wav.string(), sample_rate);
    if (sample_rate != 16000 || pcm.empty()) {
        std::fprintf(stderr, "FAIL: fixture is not 16 kHz mono\n");
        return 1;
    }

    // The parity leg compares the ext path against the env path, so the
    // environment must start clean of the validation overrides.
    set_env("TRANSCRIBE_SORTFORMER_STREAM_PRESET", "");

    transcribe_model_load_params load_params;
    transcribe_model_load_params_init(&load_params);
    load_params.backend = TRANSCRIBE_BACKEND_CPU;  // deterministic float order for the parity leg
    transcribe_session_params session_params;
    transcribe_session_params_init(&session_params);
    transcribe_session * session = nullptr;
    const transcribe_status open_st =
        transcribe_open(model_path.c_str(), &load_params, &session_params, &session);
    if (open_st != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "sortformer_diar_ext_abi_test: transcribe_open failed: status=%d\n",
                     static_cast<int>(open_st));
        return 1;
    }
    const transcribe_model * model = transcribe_get_model(session);
    CHECK(model != nullptr);

    // 1. Kind + slot probe.
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, 0x4E524857u /* WHRN */));
    CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_DIARIZATION));

    // 2. Init function stamps the header + default.
    transcribe_sortformer_stream_ext ext;
    transcribe_sortformer_stream_ext_init(&ext);
    CHECK(ext.ext.size == sizeof(ext));
    CHECK(ext.ext.kind == TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM);
    CHECK(ext.preset == TRANSCRIBE_SORTFORMER_PRESET_DEFAULT);

    // Baseline: the DEFAULT operating point through the ext.
    {
        const transcribe_status st = run_with_preset(session, pcm, TRANSCRIBE_SORTFORMER_PRESET_DEFAULT);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL: default run returned status=%d\n", static_cast<int>(st));
            ++g_failures;
        }
    }
    const std::vector<transcribe_speaker_segment> default_rows = read_segments(session);
    CHECK(!default_rows.empty());
    std::printf("  default: %zu speaker segment(s)\n", default_rows.size());

    // 3. Pre-clear rejection preserves the previous result.
    {
        transcribe_run_params rp;
        transcribe_run_params_init(&rp);
        transcribe_sortformer_stream_ext bad;
        transcribe_sortformer_stream_ext_init(&bad);
        bad.ext.kind = 0x4E524857u;  // WHRN: wrong family kind
        rp.family = &bad.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(same_segments(read_segments(session), default_rows));

        transcribe_sortformer_stream_ext oor;
        transcribe_sortformer_stream_ext_init(&oor);
        oor.preset = static_cast<transcribe_sortformer_preset>(99);
        rp.family = &oor.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(same_segments(read_segments(session), default_rows));
    }

    // 4a. Ext preset == env preset (same operating point, same rows).
    std::vector<transcribe_speaker_segment> ext_rows;
    {
        const transcribe_status st = run_with_preset(session, pcm, TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL: very_high_latency run returned status=%d\n", static_cast<int>(st));
            ++g_failures;
        }
        ext_rows = read_segments(session);
        CHECK(!ext_rows.empty());
        std::printf("  very_high_latency (ext): %zu speaker segment(s)\n", ext_rows.size());

        set_env("TRANSCRIBE_SORTFORMER_STREAM_PRESET", "very_high_latency");
        const transcribe_status env_st = run_with_preset(session, pcm, TRANSCRIBE_SORTFORMER_PRESET_DEFAULT);
        set_env("TRANSCRIBE_SORTFORMER_STREAM_PRESET", "");
        CHECK(env_st == TRANSCRIBE_OK);
        CHECK(same_segments(read_segments(session), ext_rows));
    }

    // 4b. A different geometry is accepted and produces speaker rows.
    {
        const transcribe_status st = run_with_preset(session, pcm, TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL: low_latency run returned status=%d\n", static_cast<int>(st));
            ++g_failures;
        }
        const auto rows = read_segments(session);
        CHECK(!rows.empty());
        std::printf("  low_latency (ext): %zu speaker segment(s)\n", rows.size());
    }

    transcribe_close(session);

    if (g_failures != 0) {
        std::fprintf(stderr, "sortformer_diar_ext_abi_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("sortformer_diar_ext_abi_test: OK\n");
    return 0;
}
