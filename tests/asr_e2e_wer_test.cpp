// tests/asr_e2e_wer_test.cpp - end-to-end ASR transcription gate (plan §6 /
// progress "NEXT 1": the first test that checks TEXT, not plumbing).
//
// abi_bridge_hello proves the C ABI can load a model and return *a* string;
// abi_stream_hello proves the streaming cursor algebra. Neither ever reads
// what the model wrote. This test transcribes the in-tree LibriSpeech
// fixtures through the public C ABI and scores word error rate against their
// reference transcripts, so an engine that quietly starts emitting garbage
// (wrong mel layout, desynchronized decoder, tokenizer drift) fails here and
// nowhere else.
//
// Usage:
//   asr_e2e_wer_test <model.gguf> <fixture_dir> [max_corpus_wer_pct]
// or the SPEECHCPP_ASR_E2E_MODEL / SPEECHCPP_ASR_E2E_FIXTURES /
// SPEECHCPP_ASR_E2E_MAX_WER environment variables.
//
// The fixture directory is scanned for *.wav files with a sibling .txt
// reference (assets/asr_validation/librispeech ships four). The gate is
// CORPUS WER — total word edits over total reference words — not per-file,
// so one hard utterance cannot fail the suite while the aggregate is healthy.
//
// The model is NOT in-tree (models/ is gitignored). CMake registers this test
// pointing at models/moonshine-tiny-Q8_0.gguf; fetch it with
//   uv run scripts/fetch_asr_test_model.py     (any Python 3 works; stdlib only)
// Without the model file the test exits 2 (skipped), the same contract as the
// bridge tests, so CTest stays green on checkouts that never fetched it.
//
// Like the bridge tests this links ONLY the public C ABI — no engine_runtime,
// no ggml — so a WER regression observed here is a regression in exactly what
// a language binding ships.

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

// Default gate. Measured baseline for moonshine-tiny-Q8_0 on the four
// fixtures: 1.45% corpus WER (1 edit / 69 reference words, CPU, 2026-08-20) —
// one word costs 1.45 pp, so the granularity of this corpus is coarse. 10%
// allows 6 edits: far above single-word jitter, far below the 50-100% a
// structural break (wrong mel layout, desynchronized decoder, tokenizer
// drift) produces.
constexpr double kDefaultMaxCorpusWerPct = 10.0;

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

// Text normalization, word-level edit distance, and fixture scanning live in
// tests/asr_test_text.h, shared with asr_stream_text_wer_test so the two
// gates score text identically.

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string model_path = env_or_arg(argc, argv, 1, "SPEECHCPP_ASR_E2E_MODEL");
        if (model_path.empty()) {
            std::cerr << "asr_e2e_wer_test: skipped (no model path; pass argv[1] or "
                         "SPEECHCPP_ASR_E2E_MODEL)\n";
            return 2;
        }
        // CMake registers this test unconditionally, pointing at the canonical
        // (gitignored) models/ path — an absent file is "not fetched", not a
        // failure. A wrong path passed by hand skips identically; the skip
        // message says where the file was expected.
        if (!fs::exists(model_path)) {
            std::cerr << "asr_e2e_wer_test: skipped (model not found: " << model_path
                      << "; fetch with: uv run scripts/fetch_asr_test_model.py)\n";
            return 2;
        }

        const std::string fixture_dir = env_or_arg(argc, argv, 2, "SPEECHCPP_ASR_E2E_FIXTURES");
        require(!fixture_dir.empty(),
                "usage: asr_e2e_wer_test <model.gguf> <fixture_dir> [max_corpus_wer_pct]");
        require(fs::is_directory(fixture_dir), "fixture directory not found: " + fixture_dir);

        double            max_wer_pct = kDefaultMaxCorpusWerPct;
        const std::string bound_arg   = env_or_arg(argc, argv, 3, "SPEECHCPP_ASR_E2E_MAX_WER");
        if (!bound_arg.empty()) {
            max_wer_pct = std::stod(bound_arg);
        }

        const std::vector<asr_test::Fixture> fixtures = asr_test::collect_fixtures(fixture_dir);
        require(!fixtures.empty(), "no .wav/.txt fixture pairs in " + fixture_dir);

        struct transcribe_model_load_params load_params;
        transcribe_model_load_params_init(&load_params);
        struct transcribe_session_params session_params;
        transcribe_session_params_init(&session_params);

        struct transcribe_session * session = nullptr;
        const transcribe_status     load_st =
            transcribe_open(model_path.c_str(), &load_params, &session_params, &session);
        require(load_st == TRANSCRIBE_OK && session != nullptr,
                "transcribe_open failed: status=" + std::to_string(static_cast<int>(load_st))
                    + " path=" + model_path);

        std::cout << "asr_e2e_wer_test: model=" << fs::path(model_path).filename().string() << ", "
                  << fixtures.size() << " fixture(s) from " << fixture_dir << "\n";

        size_t total_ref_words = 0;
        size_t total_edits     = 0;
        double total_audio_s   = 0.0;
        double total_decode_s  = 0.0;
        bool   any_empty_hyp   = false;

        for (const auto & fixture : fixtures) {
            int                      rate = 0;
            const std::vector<float> pcm  = abi_test::read_wav_mono_f32(fixture.wav.string(), rate);
            require(rate == kSampleRate,
                    "fixture must be 16 kHz (got " + std::to_string(rate) + "): " + fixture.wav.string());
            require(!pcm.empty(), "fixture has no samples: " + fixture.wav.string());

            struct transcribe_run_params run_params;
            transcribe_run_params_init(&run_params);

            const auto              t0     = std::chrono::steady_clock::now();
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

            const double audio_s  = static_cast<double>(pcm.size()) / kSampleRate;
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
        const double rtf        = total_audio_s > 0.0 ? total_decode_s / total_audio_s : 0.0;
        std::cout << "asr_e2e_wer_test: corpus WER " << corpus_wer << "% (" << total_edits << "/"
                  << total_ref_words << " over " << fixtures.size() << " utterances), bound " << max_wer_pct
                  << "%, RTF " << rtf << "\n";

        require(!any_empty_hyp, "at least one hypothesis was empty — the model produced no text");
        require(corpus_wer <= max_wer_pct,
                "corpus WER " + std::to_string(corpus_wer) + "% exceeds the bound of "
                    + std::to_string(max_wer_pct) + "%");

        std::cout << "asr_e2e_wer_test: ok\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "asr_e2e_wer_test: " << ex.what() << "\n";
        return 1;
    }
}
