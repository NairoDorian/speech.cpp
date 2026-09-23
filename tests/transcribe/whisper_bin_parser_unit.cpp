// whisper_bin_parser_unit.cpp - the whisper.cpp `.bin` path, as it ships
// since the Whisper arch retired (Phase 11 W2b, ledger B16c): the magic-byte
// dispatch in transcribe_model_load_file, the engine `whisper` loader's
// hparams / mel-filter gates (src/models/whisper/assets.cpp, load_bin) and the
// statuses the C ABI surfaces.
//
// This test used to exercise src/runtime/transcribe-bin-loader.cpp, a second
// .bin parser the arch owned. With the arch gone that parser had no
// production consumer and was deleted; the checks moved onto the parser that
// actually runs. The status contract:
//
//   missing path                         -> TRANSCRIBE_ERR_FILE_NOT_FOUND
//   ggml magic, not Whisper-shaped       -> TRANSCRIBE_ERR_UNSUPPORTED_ARCH
//     (Silero VAD .bin; also any .bin in a build without `whisper` linked)
//   Whisper-shaped but malformed         -> TRANSCRIBE_ERR_UNSUPPORTED_ARCH
//     (non-canonical mel filters, truncated). The retired arch answered
//     ERR_GGUF here; the adapter surfaces every family load failure as
//     UNSUPPORTED_ARCH (the B13 sensevoice / B14 fun_asr_nano convention
//     pinned by loader_smoke), and Whisper now follows it.
//
// Engine-level checks name the gate that fired (the loader's message), so a
// geometry rejection cannot pass for a truncation or the reverse.
//
// Env-gated (upstream artifacts not in this repo; each sub-test skips when
// its variable is unset or the file is missing):
//   TRANSCRIBE_WHISPER_BIN_SILERO     - a non-Whisper ggml .bin (upstream
//                                       models/for-tests-silero-v6.2.0-ggml.bin)
//   TRANSCRIBE_WHISPER_BIN_TRUNCATED  - a stripped upstream for-tests-*.bin
//                                       (Whisper-shaped hparams, no payload)
//   TRANSCRIBE_WHISPER_BIN_TINY_Q8_0  - a real multilingual tiny .bin
//                                       (ggml-tiny.bin, pinned by
//                                       scripts/fetch_asr_test_model.py)

#include "engine/models/whisper/assets.h"
#include "transcribe.h"

#include <sys/stat.h>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <process.h>
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_skipped  = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

bool file_exists(const char * path) {
    struct stat st{};
    return ::stat(path, &st) == 0;
}

const char * env_or_null(const char * key) {
    const char * v = std::getenv(key);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

// transcribe_model_load_file's status; a model that did load is freed.
transcribe_status c_abi_load(const std::string & path_utf8) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend             = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * m   = nullptr;
    const transcribe_status st = transcribe_model_load_file(path_utf8.c_str(), &mp, &m);
    if (st != TRANSCRIBE_OK) {
        CHECK(m == nullptr);
    }
    if (m != nullptr) {
        transcribe_model_free(m);
    }
    return st;
}

// The engine loader's rejection message, or "" when it loaded.
std::string engine_load_error(const std::filesystem::path & path) {
    try {
        (void) engine::models::whisper::load_whisper_assets(path);
        return "";
    } catch (const std::exception & e) {
        return e.what();
    }
}

bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

void test_silero_rejected() {
    const char * path = env_or_null("TRANSCRIBE_WHISPER_BIN_SILERO");
    if (path == nullptr || !file_exists(path)) {
        ++g_skipped;
        return;
    }
    CHECK(contains(engine_load_error(path), "not Whisper-shaped"));
    CHECK(c_abi_load(path) == TRANSCRIBE_ERR_UNSUPPORTED_ARCH);
}

void test_truncated_rejected() {
    const char * path = env_or_null("TRANSCRIBE_WHISPER_BIN_TRUNCATED");
    if (path == nullptr || !file_exists(path)) {
        ++g_skipped;
        return;
    }
    // for-tests fixtures pass the hparams gate (real geometry) and fail
    // later, on the missing payload.
    const std::string err = engine_load_error(path);
    CHECK(!err.empty());
    CHECK(!contains(err, "not Whisper-shaped"));
    CHECK(c_abi_load(path) == TRANSCRIBE_ERR_UNSUPPORTED_ARCH);
}

void test_tiny_q8_parsed() {
    const char * path = env_or_null("TRANSCRIBE_WHISPER_BIN_TINY_Q8_0");
    if (path == nullptr || !file_exists(path)) {
        ++g_skipped;
        return;
    }
    std::shared_ptr<const engine::models::whisper::WhisperAssets> assets;
    try {
        assets = engine::models::whisper::load_whisper_assets(path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: engine could not load %s: %s\n", path, e.what());
        ++g_failures;
        return;
    }
    const auto & hp = assets->hparams;

    // Tiny multilingual whisper geometry.
    CHECK(assets->layout == engine::models::whisper::WhisperWeightLayout::LegacyBin);
    CHECK(hp.enc_n_layers == 4);
    CHECK(hp.dec_n_layers == 4);
    CHECK(hp.enc_d_model == 384);
    CHECK(hp.dec_d_model == 384);
    CHECK(hp.enc_n_heads == 6);
    CHECK(hp.dec_n_heads == 6);
    CHECK(hp.enc_num_mel_bins == 80);
    CHECK(hp.is_multilingual);
    CHECK(hp.n_languages == 99);
    CHECK(assets->language_codes.size() == 99u);

    // Mel filterbank for whisper: 80 mels x 201 freq bins.
    CHECK(assets->mel_filterbank.size() == 80u * 201u);

    // Tensor manifest: tiny multilingual has 167 tensors (10 input +
    // 15 + 15*4 + 24*4). Allow a small range to permit minor variants.
    const size_t n_tensors = assets->source->tensors().size();
    CHECK(n_tensors >= 160 && n_tensors <= 200);

    // Spot-check canonical legacy names through the layout's name mapping.
    CHECK(assets->source->has_tensor(assets->tensor_name("encoder.conv1.weight")));
    CHECK(assets->source->has_tensor(assets->tensor_name("decoder.token_embedding.weight")));

    // And the C ABI loads it.
    CHECK(c_abi_load(path) == TRANSCRIBE_OK);
}

void test_missing_path() {
    CHECK(c_abi_load("/nonexistent/path/that/does/not/exist.bin") == TRANSCRIBE_ERR_FILE_NOT_FOUND);
}

// Write a minimal header-only .bin with the given hparams + mel
// filter dims to `path`. Vocab and tensors are intentionally absent
// so the parser will fail at the manifest stage if it gets past the
// header checks — but for these tests we only care about the
// hparams + mel-filter-dim gates.
bool write_synthetic_bin(const std::filesystem::path & path,
                         int32_t                       n_vocab,
                         int32_t                       n_audio_layer,
                         int32_t                       n_text_layer,
                         int32_t                       n_mels,
                         int32_t                       n_mel_filters,
                         int32_t                       n_fft_filters) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    auto wi = [&](int32_t v) {
        f.write(reinterpret_cast<const char *>(&v), sizeof(v));
    };
    const uint32_t magic = 0x67676d6cu;
    f.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    // 11 × int32 hparams.
    wi(n_vocab);
    wi(1500);  // n_audio_ctx
    wi(384);   // n_audio_state
    wi(6);     // n_audio_head
    wi(n_audio_layer);
    wi(448);   // n_text_ctx
    wi(384);   // n_text_state
    wi(6);     // n_text_head
    wi(n_text_layer);
    wi(n_mels);
    wi(0);  // ftype
    // mel filter dims, then n_mel_filters * n_fft_filters floats.
    wi(n_mel_filters);
    wi(n_fft_filters);
    const size_t fb_bytes = static_cast<size_t>(n_mel_filters) * static_cast<size_t>(n_fft_filters) * sizeof(float);
    std::vector<char> zeros(fb_bytes, 0);
    f.write(zeros.data(), zeros.size());
    return static_cast<bool>(f);
}

// Create an empty .bin path without weakening mkstemps()'s atomic
// uniqueness guarantee. The Windows branch uses CREATE_NEW for the
// equivalent create-if-absent behavior and native wide paths.
std::filesystem::path make_temp_bin_path() {
#ifdef _WIN32
    std::error_code             ec;
    const std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        return {};
    }

    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
        const std::filesystem::path path =
            dir / ("transcribe_bin_test_" + std::to_string(::_getpid()) + "_" + std::to_string(attempt) + ".bin");
        const HANDLE file =
            ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            ::CloseHandle(file);
            return path;
        }
        const DWORD error = ::GetLastError();
        if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS) {
            return {};
        }
    }
    return {};
#else
    char      tmpl[] = "/tmp/transcribe_bin_test_XXXXXX.bin";
    const int fd     = ::mkstemps(tmpl, 4);
    if (fd < 0) {
        return {};
    }
    ::close(fd);
    return std::filesystem::path(tmpl);
#endif
}

void remove_temp_file(const std::filesystem::path & path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void test_bad_n_fft() {
    const std::filesystem::path path = make_temp_bin_path();
    if (path.empty()) {
        std::fprintf(stderr, "SKIP: could not create tempfile for synthetic bin\n");
        ++g_skipped;
        return;
    }

    // Whisper-shaped hparams but non-canonical n_fft (200 instead of
    // 201). The loader must reject at the mel gate, before the vocab.
    if (!write_synthetic_bin(path, 51865, 4, 4, 80, 80, 200)) {
        std::fprintf(stderr, "SKIP: failed to write synthetic .bin\n");
        ++g_skipped;
        remove_temp_file(path);
        return;
    }
    CHECK(contains(engine_load_error(path), "mel filterbank is 80x200"));
    CHECK(c_abi_load(path.u8string()) == TRANSCRIBE_ERR_UNSUPPORTED_ARCH);
    remove_temp_file(path);
}

void test_not_whisper_shaped() {
    // ggml magic with hparams no Whisper has (n_mels 64): the Silero case,
    // synthesized so it runs without the upstream artifact.
    const std::filesystem::path path = make_temp_bin_path();
    if (path.empty()) {
        ++g_skipped;
        return;
    }
    if (!write_synthetic_bin(path, 1000, 4, 4, 64, 64, 201)) {
        ++g_skipped;
        remove_temp_file(path);
        return;
    }
    CHECK(contains(engine_load_error(path), "not Whisper-shaped"));
    CHECK(c_abi_load(path.u8string()) == TRANSCRIBE_ERR_UNSUPPORTED_ARCH);
    remove_temp_file(path);
}

void test_distil_layer_count_accepted() {
    // Distil-style asymmetric layers: n_text_layer=2 with otherwise
    // whisper-shaped hparams should pass the hparams gate. We don't
    // care that the parser later fails at "no tensors" — the hparams
    // check should not be the gating step.
    const std::filesystem::path path = make_temp_bin_path();
    if (path.empty()) {
        ++g_skipped;
        return;
    }

    if (!write_synthetic_bin(path, 51865, 24, 2, 80, 80, 201)) {
        ++g_skipped;
        remove_temp_file(path);
        return;
    }
    // Header + mel filters parse; we then run out of bytes at the vocab.
    // A vocab-stage error, not the geometry gate's, is the proof that the
    // gate accepts distil layers.
    const std::string err = engine_load_error(path);
    CHECK(contains(err, "vocab"));
    CHECK(!contains(err, "not Whisper-shaped"));
    CHECK(!contains(err, "mel filterbank"));
    remove_temp_file(path);
}

}  // namespace

int main() {
    test_missing_path();
    test_bad_n_fft();
    test_not_whisper_shaped();
    test_distil_layer_count_accepted();
    test_silero_rejected();
    test_truncated_rejected();
    test_tiny_q8_parsed();

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED: %d check(s); %d sub-test(s) skipped\n", g_failures, g_skipped);
        return 1;
    }
    std::fprintf(stderr, "OK (%d sub-test(s) skipped)\n", g_skipped);
    return 0;
}
