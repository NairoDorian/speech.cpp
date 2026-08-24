// tests/asr_e2e_edits_test.cpp - tight end-to-end ASR word edits gate (Fusion Roadmap §7.6).
//
// asr_e2e_wer_test acts as the structural tripwire (10.0% max WER); this gate enforces
// exact text preservation on the 4 LibriSpeech fixtures: total_edits <= 1.
//
// Usage:
//   asr_e2e_edits_test <model.gguf> <fixture_dir> [max_total_edits]
// or the SPEECHCPP_ASR_E2E_MODEL / SPEECHCPP_ASR_E2E_FIXTURES /
// SPEECHCPP_ASR_E2E_MAX_EDITS environment variables.

#include "abi_test_wav.h"
#include "asr_test_text.h"
#include "transcribe.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int kSampleRate = 16000;
constexpr size_t kDefaultMaxTotalEdits = 1;

std::string env_or_arg(int argc, char ** argv, int index, const char * env_name) {
    if (argc > index && argv[index] != nullptr && argv[index][0] != '\0') {
        return argv[index];
    }
    const char * env = std::getenv(env_name);
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return std::string();
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string model_path = env_or_arg(argc, argv, 1, "SPEECHCPP_ASR_E2E_MODEL");
        if (model_path.empty()) {
            std::cerr << "asr_e2e_edits_test: skipped (no model path; pass argv[1] or "
                         "SPEECHCPP_ASR_E2E_MODEL)\n";
            return 2;
        }
        if (!fs::exists(model_path)) {
            std::cerr << "asr_e2e_edits_test: skipped (model not found: " << model_path
                      << "; fetch with: uv run scripts/fetch_asr_test_model.py)\n";
            return 2;
        }

        const std::string fixture_dir = env_or_arg(argc, argv, 2, "SPEECHCPP_ASR_E2E_FIXTURES");
        require(!fixture_dir.empty(),
                "usage: asr_e2e_edits_test <model.gguf> <fixture_dir> [max_total_edits]");
        require(fs::is_directory(fixture_dir), "fixture directory not found: " + fixture_dir);

        size_t max_edits = kDefaultMaxTotalEdits;
        const std::string bound_arg = env_or_arg(argc, argv, 3, "SPEECHCPP_ASR_E2E_MAX_EDITS");
        if (!bound_arg.empty()) {
            max_edits = static_cast<size_t>(std::stoul(bound_arg));
        }

        const std::vector<asr_test::Fixture> fixtures = asr_test::collect_fixtures(fixture_dir);
        require(!fixtures.empty(), "no .wav/.txt fixture pairs in " + fixture_dir);

        struct transcribe_model_load_params load_params;
        transcribe_model_load_params_init(&load_params);
        struct transcribe_session_params session_params;
        transcribe_session_params_init(&session_params);

        struct transcribe_session * session = nullptr;
        const transcribe_status load_st =
            transcribe_open(model_path.c_str(), &load_params, &session_params, &session);
        require(load_st == TRANSCRIBE_OK && session != nullptr,
                "transcribe_open failed: status=" + std::to_string(static_cast<int>(load_st))
                    + " path=" + model_path);

        std::cout << "asr_e2e_edits_test: model=" << fs::path(model_path).filename().string() << ", "
                  << fixtures.size() << " fixture(s) from " << fixture_dir << "\n";

        size_t total_ref_words = 0;
        size_t total_edits = 0;
        double total_audio_s = 0.0;
        double total_decode_s = 0.0;
        bool any_empty_hyp = false;

        for (const auto & fixture : fixtures) {
            int rate = 0;
            const std::vector<float> pcm = abi_test::read_wav_mono_f32(fixture.wav.string(), rate);
            require(rate == kSampleRate,
                    "fixture must be 16 kHz (got " + std::to_string(rate) + "): " + fixture.wav.string());
            require(!pcm.empty(), "fixture has no samples: " + fixture.wav.string());

            struct transcribe_run_params run_params;
            transcribe_run_params_init(&run_params);

            const auto t0 = std::chrono::steady_clock::now();
            const transcribe_status run_st =
                transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &run_params);
            const auto t1 = std::chrono::steady_clock::now();
            require(run_st == TRANSCRIBE_OK,
                    "transcribe_run failed: status=" + std::to_string(static_cast<int>(run_st)) + " on "
                        + fixture.wav.filename().string());

            const char * raw = transcribe_full_text(session);
            require(raw != nullptr, "transcribe_full_text returned NULL on " + fixture.wav.filename().string());
            const std::string hypothesis = raw;

            const std::vector<std::string> ref_words = asr_test::normalize_words(asr_test::read_text_file(fixture.txt));
            const std::vector<std::string> hyp_words = asr_test::normalize_words(hypothesis);
            require(!ref_words.empty(), "reference transcript is empty: " + fixture.txt.string());
            if (hyp_words.empty()) {
                any_empty_hyp = true;
            }

            const size_t edits = asr_test::word_edit_distance(ref_words, hyp_words);
            total_ref_words += ref_words.size();
            total_edits += edits;

            const double audio_s = static_cast<double>(pcm.size()) / kSampleRate;
            const double decode_s = std::chrono::duration<double>(t1 - t0).count();
            total_audio_s += audio_s;
            total_decode_s += decode_s;

            const double wer = 100.0 * static_cast<double>(edits) / static_cast<double>(ref_words.size());
            std::cout << "  " << fixture.wav.filename().string() << ": " << edits << "/" << ref_words.size()
                      << " word edits (WER " << wer << "%), " << audio_s << " s audio in " << decode_s
                      << " s\n";
            if (edits != 0) {
                std::cout << "    ref: " << asr_test::join_words(ref_words) << "\n"
                          << "    hyp: " << asr_test::join_words(hyp_words) << "\n";
            }
        }

        transcribe_session_free(session);

        const double corpus_wer = 100.0 * static_cast<double>(total_edits) / static_cast<double>(total_ref_words);
        const double rtf = total_audio_s > 0.0 ? total_decode_s / total_audio_s : 0.0;
        std::cout << "asr_e2e_edits_test: total edits " << total_edits << "/" << total_ref_words
                  << " (WER " << corpus_wer << "% over " << fixtures.size() << " utterances), bound <= "
                  << max_edits << " edit(s), RTF " << rtf << "\n";

        require(!any_empty_hyp, "at least one hypothesis was empty — the model produced no text");
        require(total_edits <= max_edits,
                "total edits " + std::to_string(total_edits) + " exceeds the bound of "
                    + std::to_string(max_edits));

        std::cout << "asr_e2e_edits_test: ok\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "asr_e2e_edits_test: " << ex.what() << "\n";
        return 1;
    }
}
