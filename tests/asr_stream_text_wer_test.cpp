// tests/asr_stream_text_wer_test.cpp - end-to-end STREAMING text gate
// (progress "NEXT 1": streaming ASR text validation).
//
// asr_e2e_wer_test proves the offline path produces correct TEXT;
// abi_stream_hello proves the streaming surface's cursor algebra and state
// machine but never reads what the model wrote. This test closes the gap: it
// streams the in-tree LibriSpeech fixtures into a real streaming-family model
// through the public C ABI (begin -> odd-sized feeds -> finalize ->
// transcribe_stream_get_text) and scores corpus WER of the STREAMED text
// against the reference transcripts, alongside the same model's offline text,
// so a streaming path that garbles, drops, or duplicates words fails here and
// nowhere else.
//
// Usage:
//   asr_stream_text_wer_test <model.gguf> <fixture_dir>
//                            [max_corpus_wer_pct] [max_stream_vs_offline_edits]
// or the SPEECHCPP_ASR_STREAM_MODEL / SPEECHCPP_ASR_STREAM_FIXTURES /
// SPEECHCPP_ASR_STREAM_MAX_WER / SPEECHCPP_ASR_STREAM_MAX_DIVERGENCE
// environment variables.
//
// The model is NOT in-tree (models/ is gitignored). CMake registers this test
// pointing at models/moonshine-streaming-tiny-Q8_0.gguf (48 MB, MIT,
// UsefulSensors' streaming moonshine ported and WER-validated by
// transcribe.cpp at 4.52% offline / 4.54% streamed on full test-clean);
// fetch it with
//   uv run scripts/fetch_asr_test_model.py     (any Python 3 works; stdlib only)
// Without the model file the test exits 2 (skipped), the same contract as the
// other gates. A model that reports supports_streaming == false is a FAILURE
// here, not a skip: this gate exists to validate streaming text, and its
// pinned model is a streaming family by construction.
//
// All fixtures run on ONE session, alternating offline transcribe_run and a
// begin/feed/finalize stream cycle per fixture — so the test also proves that
// run/stream mode switching and stream_reset really clear per-utterance text
// state (the reset contract with real text, which abi_stream_hello can only
// check via segment counts).
//
// Like the other gates this links ONLY the public C ABI — no engine_runtime,
// no ggml — so a regression observed here is a regression in exactly what a
// language binding ships.

#include "abi_test_wav.h"
#include "asr_test_text.h"
#include "transcribe.h"

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

// Default corpus-WER gate, same rationale as asr_e2e_wer_test: the four-
// fixture corpus has 69 reference words, so one word costs 1.45 pp. Measured
// baseline for moonshine-streaming-tiny-Q8_0 (CPU, 2026-08-20): streamed
// 4.35% == offline 4.35% (3/69 — FORWARDED->VOTED plus "I AM"->"I'M"),
// divergence 0, streamed RTF 0.55. 10% allows 6 edits — far above
// single-word jitter, far below the 50-100% a structural break produces.
constexpr double kDefaultMaxCorpusWerPct = 10.0;

// Streamed and offline text come from the same weights; on the 4 LibriSpeech
// fixtures moonshine-streaming-tiny produces identical transcripts in both
// modes (divergence 0). Tightened to 0 per Fusion Roadmap §7.6.
constexpr size_t kDefaultMaxStreamVsOfflineEdits = 0;

// Feed sizes in samples: odd, mutually coprime-ish, none aligned to any
// plausible internal chunk size, spanning ~100-400 ms like real capture
// callbacks. The pathological tiny-feed torture (1- and 2-sample feeds)
// belongs to abi_stream_hello; this test is about the text.
const int kFeedSizes[] = { 1601, 4099, 6399, 2911 };

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
        const std::string model_path = env_or_arg(argc, argv, 1, "SPEECHCPP_ASR_STREAM_MODEL");
        if (model_path.empty()) {
            std::cerr << "asr_stream_text_wer_test: skipped (no model path; pass argv[1] or "
                         "SPEECHCPP_ASR_STREAM_MODEL)\n";
            return 2;
        }
        if (!fs::exists(model_path)) {
            std::cerr << "asr_stream_text_wer_test: skipped (model not found: " << model_path
                      << "; fetch with: uv run scripts/fetch_asr_test_model.py)\n";
            return 2;
        }

        const std::string fixture_dir = env_or_arg(argc, argv, 2, "SPEECHCPP_ASR_STREAM_FIXTURES");
        require(!fixture_dir.empty(),
                "usage: asr_stream_text_wer_test <model.gguf> <fixture_dir> "
                "[max_corpus_wer_pct] [max_stream_vs_offline_edits]");
        require(fs::is_directory(fixture_dir), "fixture directory not found: " + fixture_dir);

        double            max_wer_pct = kDefaultMaxCorpusWerPct;
        const std::string wer_arg     = env_or_arg(argc, argv, 3, "SPEECHCPP_ASR_STREAM_MAX_WER");
        if (!wer_arg.empty()) {
            max_wer_pct = std::stod(wer_arg);
        }
        size_t            max_divergence = kDefaultMaxStreamVsOfflineEdits;
        const std::string div_arg = env_or_arg(argc, argv, 4, "SPEECHCPP_ASR_STREAM_MAX_DIVERGENCE");
        if (!div_arg.empty()) {
            max_divergence = static_cast<size_t>(std::stoul(div_arg));
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

        struct transcribe_capabilities caps;
        transcribe_capabilities_init(&caps);
        require(transcribe_model_get_capabilities(transcribe_get_model(session), &caps) == TRANSCRIBE_OK,
                "transcribe_model_get_capabilities failed");
        require(caps.supports_streaming,
                "the pinned streaming model reports supports_streaming == false — the streaming "
                "gate's model must stream; this is a capability regression, not a skip");

        std::cout << "asr_stream_text_wer_test: model=" << fs::path(model_path).filename().string()
                  << ", " << fixtures.size() << " fixture(s) from " << fixture_dir << "\n";

        size_t total_ref_words       = 0;
        size_t total_stream_edits    = 0;
        size_t total_offline_edits   = 0;
        size_t total_divergence      = 0;
        double total_audio_s         = 0.0;
        double total_stream_s        = 0.0;
        bool   any_empty_stream_text = false;

        for (const auto & fixture : fixtures) {
            int                      rate = 0;
            const std::vector<float> pcm  = abi_test::read_wav_mono_f32(fixture.wav.string(), rate);
            require(rate == kSampleRate,
                    "fixture must be 16 kHz (got " + std::to_string(rate) + "): " + fixture.wav.string());
            require(!pcm.empty(), "fixture has no samples: " + fixture.wav.string());

            // Offline pass first: same weights, contiguous audio. This is the
            // in-run reference the streamed text must agree with.
            struct transcribe_run_params run_params;
            transcribe_run_params_init(&run_params);
            const transcribe_status run_st =
                transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &run_params);
            require(run_st == TRANSCRIBE_OK,
                    "offline transcribe_run failed: status=" + std::to_string(static_cast<int>(run_st))
                        + " on " + fixture.wav.filename().string());
            const char * offline_raw = transcribe_full_text(session);
            require(offline_raw != nullptr,
                    "transcribe_full_text returned NULL on " + fixture.wav.filename().string());
            const std::string offline_text = offline_raw;

            // Streaming pass on the same session.
            struct transcribe_run_params stream_run_params;
            transcribe_run_params_init(&stream_run_params);
            struct transcribe_stream_params stream_params;
            transcribe_stream_params_init(&stream_params);

            const auto              t0       = std::chrono::steady_clock::now();
            const transcribe_status begin_st =
                transcribe_stream_begin(session, &stream_run_params, &stream_params);
            require(begin_st == TRANSCRIBE_OK,
                    "transcribe_stream_begin failed: status=" + std::to_string(static_cast<int>(begin_st))
                        + " on " + fixture.wav.filename().string());

            size_t offset     = 0;
            size_t feed_index = 0;
            while (offset < pcm.size()) {
                const size_t want =
                    static_cast<size_t>(kFeedSizes[feed_index % (sizeof(kFeedSizes) / sizeof(kFeedSizes[0]))]);
                const size_t take = std::min(want, pcm.size() - offset);
                ++feed_index;
                const transcribe_status feed_st =
                    transcribe_stream_feed(session, pcm.data() + offset, static_cast<int>(take), nullptr);
                require(feed_st == TRANSCRIBE_OK,
                        "transcribe_stream_feed failed at sample " + std::to_string(offset) + ": status="
                            + std::to_string(static_cast<int>(feed_st)) + " on "
                            + fixture.wav.filename().string());
                offset += take;
            }

            const transcribe_status fin_st = transcribe_stream_finalize(session, nullptr);
            require(fin_st == TRANSCRIBE_OK,
                    "transcribe_stream_finalize failed: status=" + std::to_string(static_cast<int>(fin_st))
                        + " on " + fixture.wav.filename().string());
            const auto t1 = std::chrono::steady_clock::now();

            struct transcribe_stream_text stream_text;
            transcribe_stream_text_init(&stream_text);
            require(transcribe_stream_get_text(session, &stream_text) == TRANSCRIBE_OK,
                    "transcribe_stream_get_text failed on " + fixture.wav.filename().string());
            require(stream_text.full_text != nullptr,
                    "stream full_text is NULL after finalize on " + fixture.wav.filename().string());
            // Documented finalize contract: tentative text is empty after a
            // successful finalize; full_text is the authoritative transcript.
            require(stream_text.tentative_text_bytes == 0,
                    "finalize left " + std::to_string(stream_text.tentative_text_bytes)
                        + " tentative bytes on " + fixture.wav.filename().string());
            const std::string streamed_text = stream_text.full_text;

            transcribe_stream_reset(session);

            const std::vector<std::string> ref_words =
                asr_test::normalize_words(asr_test::read_text_file(fixture.txt));
            const std::vector<std::string> stream_words  = asr_test::normalize_words(streamed_text);
            const std::vector<std::string> offline_words = asr_test::normalize_words(offline_text);
            require(!ref_words.empty(), "reference transcript is empty: " + fixture.txt.string());
            if (stream_words.empty()) {
                any_empty_stream_text = true;
            }

            const size_t stream_edits  = asr_test::word_edit_distance(ref_words, stream_words);
            const size_t offline_edits = asr_test::word_edit_distance(ref_words, offline_words);
            const size_t divergence    = asr_test::word_edit_distance(offline_words, stream_words);
            total_ref_words += ref_words.size();
            total_stream_edits += stream_edits;
            total_offline_edits += offline_edits;
            total_divergence += divergence;

            const double audio_s  = static_cast<double>(pcm.size()) / kSampleRate;
            const double stream_s = std::chrono::duration<double>(t1 - t0).count();
            total_audio_s += audio_s;
            total_stream_s += stream_s;

            std::cout << "  " << fixture.wav.filename().string() << ": streamed " << stream_edits << "/"
                      << ref_words.size() << " word edits (offline " << offline_edits
                      << ", stream-vs-offline " << divergence << "), " << audio_s << " s audio in "
                      << stream_s << " s streamed\n";
            if (stream_edits != 0 || divergence != 0) {
                std::cout << "    ref: " << asr_test::join_words(ref_words) << "\n"
                          << "    hyp: " << asr_test::join_words(stream_words) << "\n"
                          << "    off: " << asr_test::join_words(offline_words) << "\n";
            }
        }

        transcribe_session_free(session);

        const double stream_wer =
            100.0 * static_cast<double>(total_stream_edits) / static_cast<double>(total_ref_words);
        const double offline_wer =
            100.0 * static_cast<double>(total_offline_edits) / static_cast<double>(total_ref_words);
        const double rtf = total_audio_s > 0.0 ? total_stream_s / total_audio_s : 0.0;
        std::cout << "asr_stream_text_wer_test: streamed corpus WER " << stream_wer << "% ("
                  << total_stream_edits << "/" << total_ref_words << "), offline corpus WER "
                  << offline_wer << "% (" << total_offline_edits << "/" << total_ref_words
                  << "), stream-vs-offline divergence " << total_divergence << " word(s), bound "
                  << max_wer_pct << "% / " << max_divergence << " word(s), streamed RTF " << rtf << "\n";

        require(!any_empty_stream_text,
                "at least one streamed hypothesis was empty — the stream produced no text");
        require(stream_wer <= max_wer_pct,
                "streamed corpus WER " + std::to_string(stream_wer) + "% exceeds the bound of "
                    + std::to_string(max_wer_pct) + "%");
        require(offline_wer <= max_wer_pct,
                "offline corpus WER " + std::to_string(offline_wer) + "% exceeds the bound of "
                    + std::to_string(max_wer_pct) + "%");
        require(total_divergence <= max_divergence,
                "streamed text diverges from offline text by " + std::to_string(total_divergence)
                    + " word(s), bound " + std::to_string(max_divergence)
                    + " — same weights, so this is a streaming-path defect");

        std::cout << "asr_stream_text_wer_test: ok\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "asr_stream_text_wer_test: " << ex.what() << "\n";
        return 1;
    }
}
