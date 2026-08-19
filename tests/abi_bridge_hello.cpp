// tests/abi_bridge_hello.cpp - Phase 0.M hello-world bridge test.
//
// Validates the unified C ABI end-to-end BEFORE any family porting work:
// load a GGUF through transcribe_open(), run 16 kHz mono f32 PCM through
// transcribe_run(), and assert transcribe_full_text() is non-NULL. This
// exercises the ported dispatcher plus one arch family through exactly the
// surface external bindings (ctypes / koffi / Rust / Swift) will use.
//
// Usage:
//   abi_bridge_hello <model.gguf>
// or set the TRANSCRIBE_BRIDGE_TEST_MODEL environment variable. Without a
// model path the test exits 2 (skipped), so it is safe under CTest without
// shipping model assets.

#include "transcribe.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string resolve_model_path(int argc, char ** argv) {
    if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0') {
        return argv[1];
    }
    const char * env = std::getenv("TRANSCRIBE_BRIDGE_TEST_MODEL");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return std::string();
}

// Synthesize `seconds` of a sine tone at 16 kHz. Not speech — the point is
// to exercise the full load -> run -> accessor pipeline, not to produce a
// meaningful transcript.
std::vector<float> make_tone_pcm(double seconds, double frequency) {
    constexpr int kSampleRate = 16000;
    const int     n           = static_cast<int>(kSampleRate * seconds);
    std::vector<float> pcm(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(kSampleRate);
        pcm[static_cast<size_t>(i)] = static_cast<float>(0.25 * std::sin(2.0 * kPi * frequency * t));
    }
    return pcm;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string model_path = resolve_model_path(argc, argv);
        if (model_path.empty()) {
            std::cerr << "abi_bridge_hello: skipped (no model path; pass argv[1] or "
                         "TRANSCRIBE_BRIDGE_TEST_MODEL)\n";
            return 2;
        }

        struct transcribe_model_load_params load_params;
        transcribe_model_load_params_init(&load_params);
        struct transcribe_session_params session_params;
        transcribe_session_params_init(&session_params);
        struct transcribe_run_params run_params;
        transcribe_run_params_init(&run_params);

        // Convenience path: load + session in one call. The returned session
        // owns its model; transcribe_session_free frees both.
        struct transcribe_session * session = nullptr;
        const transcribe_status load_st =
            transcribe_open(model_path.c_str(), &load_params, &session_params, &session);
        if (load_st != TRANSCRIBE_OK || session == nullptr) {
            std::cerr << "abi_bridge_hello: transcribe_open failed: status=" << static_cast<int>(load_st)
                      << " path=" << model_path << "\n";
            return 1;
        }

        // Model introspection through the borrowed handle.
        struct transcribe_capabilities caps;
        transcribe_capabilities_init(&caps);
        const transcribe_status caps_st = transcribe_model_get_capabilities(transcribe_get_model(session), &caps);
        require(caps_st == TRANSCRIBE_OK || caps_st == TRANSCRIBE_ERR_BAD_STRUCT_SIZE,
                "transcribe_model_get_capabilities failed");

        const std::vector<float> pcm = make_tone_pcm(2.0, 440.0);
        const transcribe_status run_st = transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &run_params);
        require(run_st == TRANSCRIBE_OK,
                "transcribe_run failed: status=" + std::to_string(static_cast<int>(run_st)));

        const char * text = transcribe_full_text(session);
        require(text != nullptr, "transcribe_full_text() returned NULL after a successful run");

        std::cout << "abi_bridge_hello: ok (backend="
                  << (transcribe_get_model(session) != nullptr ? "loaded" : "null") << " text_len="
                  << std::strlen(text) << ")\n";

        transcribe_session_free(session);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "abi_bridge_hello: " << ex.what() << "\n";
        return 1;
    }
}