// test_whisper_decoding.cpp - the graph-agnostic Whisper decoding policy
// (src/models/whisper/decoding.cpp) under a scripted fake decoder: no model,
// no ggml. Covers segment retrieval and seek advance, the timestamp logits
// rules, the temperature-fallback ladder, the no-speech gate, long-form
// windowing, language detection, prompt / previous-context carry, seeded
// sampling determinism and argument validation.

#include "engine/models/whisper/decoding.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace engine::models::whisper;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Toy vocabulary laid out like Whisper's: text, then specials, then 1501
// timestamps <|0.00|> .. <|30.00|>.
constexpr int32_t kEot = 10, kSot = 11, kLangEn = 12, kLangDe = 13, kTranslate = 14, kTranscribe = 15,
                  kPrevSot = 16, kNoSpeech = 17, kNoTimestamps = 18, kTsBegin = 19;
constexpr int32_t kVocab = kTsBegin + 1501;

WhisperTokenContract contract(bool multilingual) {
    WhisperTokenContract ids;
    ids.vocab_size = kVocab;
    ids.eot = kEot;
    ids.sot = kSot;
    ids.transcribe = multilingual ? kTranscribe : -1;
    ids.translate = multilingual ? kTranslate : -1;
    ids.prev_sot = kPrevSot;
    ids.no_timestamps = kNoTimestamps;
    ids.multilingual = multilingual;
    return ids;
}

int32_t ts(int position) { return kTsBegin + position; }

// Emits a fixed token script per window: the chosen token gets `peak`, the
// rest 0. Records every call so tests can assert on what the policy asked.
struct FakeDecoder {
    std::map<int, std::vector<int32_t>> scripts;  // window index -> tokens
    float peak = 30.0f;
    std::vector<float> sot_row;                   // optional custom SOT row
    std::vector<float> detect_row;                // logits for the [SOT]-only prefill
    std::vector<std::pair<int, int>> encoded;     // (seek, n_frames)
    std::vector<std::vector<int32_t>> prompts;    // every prefill prompt
    int window = -1;
    size_t pos = 0;

    void fill(std::vector<float> & logits) const {
        std::fill(logits.begin(), logits.end(), 0.0f);
        const auto it = scripts.find(window);
        const int32_t tok = (it != scripts.end() && pos < it->second.size()) ? it->second[pos] : kEot;
        logits[static_cast<size_t>(tok)] = peak;
    }

    WhisperModelHooks hooks() {
        WhisperModelHooks h;
        h.encode_window = [this](int seek, int n) {
            encoded.emplace_back(seek, n);
            ++window;
        };
        h.prefill = [this](const std::vector<int32_t> & prompt, int sot_row_index, std::vector<float> & sot,
                           std::vector<float> & last) {
            prompts.push_back(prompt);
            if (prompt.size() == 1 && !detect_row.empty()) {
                last = detect_row;
                return;
            }
            pos = 0;
            fill(last);
            if (sot_row_index >= 0) {
                if (sot_row.empty()) {
                    std::fill(sot.begin(), sot.end(), 0.0f);
                } else {
                    sot = sot_row;
                }
            }
        };
        h.step = [this](int32_t, int, std::vector<float> & logits) {
            ++pos;
            fill(logits);
        };
        return h;
    }
};

WhisperDecodeResult run(FakeDecoder & fake, const WhisperTokenContract & ids, const WhisperDecodeOptions & opt,
                        int total_frames = 1500) {
    return run_whisper_decode(total_frames, 3000, 448, ids, {}, {}, opt, fake.hooks());
}

void test_retrieve_segments() {
    const auto ids = contract(false);
    // Two closed segments ending in a lone timestamp: advance the full window.
    {
        const std::vector<int32_t> gen = {ts(0), 1, 2, ts(100), ts(100), 3, ts(250)};
        const auto w = retrieve_segments(gen, ids, 30000, 3000, true);
        CHECK(w.segments.size() == 2);
        if (w.segments.size() == 2) {
            CHECK(w.segments[0].t0_ms == 30000 && w.segments[0].t1_ms == 32000);
            CHECK((w.segments[0].text_ids == std::vector<int32_t>{1, 2}));
            CHECK(w.segments[1].t0_ms == 32000 && w.segments[1].t1_ms == 35000);
            CHECK((w.segments[1].text_ids == std::vector<int32_t>{3}));
        }
        CHECK(w.seek_advance_frames == 3000);
        CHECK(w.history_slices.size() == 2);
    }
    // A "<ts><ts>" ending: the model closed early; re-enter at that timestamp.
    {
        const std::vector<int32_t> gen = {ts(0), 1, ts(100), ts(100)};
        const auto w = retrieve_segments(gen, ids, 0, 3000, true);
        CHECK(w.segments.size() == 1);
        CHECK(w.seek_advance_frames == 200);  // position 100 * input stride 2
    }
    // No pairs (the <|notimestamps|> case): one full-window slice.
    {
        const std::vector<int32_t> gen = {4, 5, 6};
        const auto w = retrieve_segments(gen, ids, 0, 1234, false);
        CHECK(w.segments.empty());
        CHECK(w.history_slices.size() == 1);
        CHECK(w.seek_advance_frames == 1234);
    }
}

void test_timestamp_rules() {
    const auto ids = contract(false);
    std::vector<float> logits(kVocab, 1.0f);
    // First generated token: text masked, timestamps capped at index 50 (1.0 s).
    apply_timestamp_rules(logits.data(), {}, ids, 50);
    CHECK(std::isinf(logits[3]) && logits[3] < 0);
    CHECK(std::isinf(logits[kNoTimestamps]));
    CHECK(logits[ts(50)] == 1.0f);
    CHECK(std::isinf(logits[ts(51)]));

    // After "<text> <ts>" the pair must close: text (below EOT) is masked, EOT
    // stays. Timestamps are made unlikely here so rule 5 (force a timestamp
    // when their summed mass beats every other token) does not also mask EOT -
    // with 1,461 equal-logit timestamps it legitimately would.
    std::fill(logits.begin(), logits.end(), 1.0f);
    for (int32_t id = kTsBegin; id < kVocab; ++id) {
        logits[id] = -20.0f;
    }
    apply_timestamp_rules(logits.data(), {ts(0), 3, ts(40)}, ids, -1);
    CHECK(std::isinf(logits[3]));
    CHECK(logits[kEot] == 1.0f);
    CHECK(std::isinf(logits[ts(39)]));  // monotonic: nothing below the open pair
    CHECK(logits[ts(40)] == -20.0f);    // closing on the same timestamp is allowed

    // Rule 5 on its own: timestamps that dominate force a timestamp.
    std::fill(logits.begin(), logits.end(), 1.0f);
    apply_timestamp_rules(logits.data(), {ts(0), 3, ts(40)}, ids, -1);
    CHECK(std::isinf(logits[kEot]));
}

void test_greedy_short_form() {
    FakeDecoder fake;
    fake.scripts[0] = {1, 2, 3};
    WhisperDecodeOptions opt;
    const auto r = run(fake, contract(false), opt);
    CHECK((r.text_ids == std::vector<int32_t>{1, 2, 3}));
    CHECK(r.traces.size() == 1);
    CHECK(r.traces[0].n_fallbacks == 0 && r.traces[0].temperature_used == 0.0f);
    CHECK(!r.truncated);
    CHECK(r.language_token == -1);
    // English prompt: [SOT, <|notimestamps|>].
    CHECK(!fake.prompts.empty() && (fake.prompts[0] == std::vector<int32_t>{kSot, kNoTimestamps}));
}

void test_fallback_ladder() {
    // 200 identical tokens compress far below the 2.4 bar, at every tier.
    FakeDecoder fake;
    fake.scripts[0] = std::vector<int32_t>(200, 7);
    WhisperDecodeOptions opt;
    opt.seed = 42;
    const auto r = run(fake, contract(false), opt);
    CHECK(r.traces.size() == 1);
    CHECK(r.traces[0].n_fallbacks == 5);  // 0.0 .. 1.0 in steps of 0.2
    CHECK(std::fabs(r.traces[0].temperature_used - 1.0f) < 1e-3f);
    CHECK(r.traces[0].compression_ratio > 2.4f);
    CHECK(r.text_ids.size() == 200);      // the last tier wins when none passes

    // A disabled threshold accepts tier 0.
    FakeDecoder fake2;
    fake2.scripts[0] = std::vector<int32_t>(200, 7);
    opt.compression_ratio_thold = INFINITY;
    const auto r2 = run(fake2, contract(false), opt);
    CHECK(r2.traces[0].n_fallbacks == 0);
}

void test_token_budget_marks_truncated() {
    FakeDecoder fake;
    fake.scripts[0] = std::vector<int32_t>(300, 7);  // never reaches EOT
    WhisperDecodeOptions opt;
    opt.compression_ratio_thold = INFINITY;
    opt.max_new_tokens = 16;
    const auto r = run(fake, contract(false), opt);
    CHECK(r.text_ids.size() == 16);
    CHECK(r.truncated);  // L11: said so, not silently cut
}

// Timestamp mode: the budget cuts the window mid-segment, but after a closed
// "<ts><ts>" pair, so the seek re-enters at that timestamp and the next window
// decodes the rest. Nothing was lost: not truncated.
void test_token_budget_with_reentry_is_not_truncated() {
    FakeDecoder fake;
    fake.scripts[0] = {ts(0), 1, 2, ts(50), ts(50), 3, 3, 3, 3, 3, 3, 3};  // no EOT
    WhisperDecodeOptions opt;
    opt.timestamps = true;
    opt.compression_ratio_thold = INFINITY;
    opt.max_new_tokens = 8;
    const auto r = run(fake, contract(false), opt);
    CHECK(fake.encoded.size() == 2);           // re-entered: a second window
    CHECK(fake.encoded.size() == 2 && fake.encoded[1].first == 100);  // at <|1.00|>
    CHECK(!r.truncated);
}

void test_no_speech_gate() {
    FakeDecoder fake;
    fake.scripts[0] = {1, 2, 3};
    fake.peak = 0.5f;  // flat distribution: avg logprob far below -1
    fake.sot_row.assign(kVocab, 0.0f);
    fake.sot_row[kNoSpeech] = 50.0f;
    WhisperDecodeOptions opt;
    const auto r = run(fake, contract(false), opt);
    CHECK(r.traces.size() == 1);
    CHECK(r.traces[0].no_speech_triggered);
    CHECK(r.traces[0].no_speech_prob > 0.99f);
    CHECK(r.text_ids.empty());
}

void test_long_form_windows() {
    FakeDecoder fake;
    fake.scripts[0] = {1};
    fake.scripts[1] = {2};
    fake.scripts[2] = {3};
    WhisperDecodeOptions opt;
    const auto r = run(fake, contract(false), opt, 7000);
    CHECK((fake.encoded == std::vector<std::pair<int, int>>{{0, 3000}, {3000, 3000}, {6000, 1000}}));
    CHECK(r.traces.size() == 3);
    if (r.traces.size() == 3) {
        CHECK(r.traces[1].t0_ms == 30000 && r.traces[2].t0_ms == 60000 && r.traces[2].t1_ms == 70000);
    }
    CHECK((r.text_ids == std::vector<int32_t>{1, 2, 3}));
}

void test_language_detection_and_task() {
    FakeDecoder fake;
    fake.scripts[0] = {4};
    fake.detect_row.assign(kVocab, 0.0f);
    fake.detect_row[kLangEn] = 1.0f;
    fake.detect_row[kLangDe] = 5.0f;
    WhisperDecodeOptions opt;
    opt.language_candidates = {kLangEn, kLangDe};
    opt.translate = true;
    const auto r = run(fake, contract(true), opt);
    CHECK(r.language_detected && r.language_token == kLangDe);
    // [SOT] detection prefill, then [SOT, <|de|>, <|translate|>, <|notimestamps|>].
    CHECK(fake.prompts.size() >= 2);
    if (fake.prompts.size() >= 2) {
        CHECK((fake.prompts[0] == std::vector<int32_t>{kSot}));
        CHECK((fake.prompts[1] == std::vector<int32_t>{kSot, kLangDe, kTranslate, kNoTimestamps}));
    }

    // A hint skips detection.
    FakeDecoder hinted;
    hinted.scripts[0] = {4};
    opt.language_token = kLangEn;
    const auto r2 = run(hinted, contract(true), opt);
    CHECK(!r2.language_detected && r2.language_token == kLangEn);
    CHECK(!hinted.prompts.empty() && hinted.prompts[0].size() == 4);
}

void test_prompt_and_context_carry() {
    // FIRST_SEGMENT: the prompt primes window 0 only ...
    FakeDecoder fake;
    fake.scripts[0] = {1, 2};
    fake.scripts[1] = {3};
    WhisperDecodeOptions opt;
    opt.prompt_ids = {8, 9};
    run(fake, contract(false), opt, 4000);
    CHECK(fake.prompts.size() == 2);
    if (fake.prompts.size() == 2) {
        CHECK((fake.prompts[0] == std::vector<int32_t>{kPrevSot, 8, 9, kSot, kNoTimestamps}));
        CHECK((fake.prompts[1] == std::vector<int32_t>{kSot, kNoTimestamps}));
    }

    // ... while condition_on_prev_tokens carries prompt + window 0's tokens.
    FakeDecoder carry;
    carry.scripts[0] = {1, 2};
    carry.scripts[1] = {3};
    opt.condition_on_prev_tokens = true;
    run(carry, contract(false), opt, 4000);
    CHECK(carry.prompts.size() == 2);
    if (carry.prompts.size() == 2) {
        CHECK((carry.prompts[1] == std::vector<int32_t>{kPrevSot, 8, 9, 1, 2, kSot, kNoTimestamps}));
    }
}

void test_seeded_sampling_is_reproducible() {
    auto sample_run = [](uint32_t seed) {
        FakeDecoder fake;
        fake.scripts[0] = {1, 2, 3, 4, 5, 6};
        fake.peak = 1.0f;  // soft enough that T = 1 really samples
        WhisperDecodeOptions opt;
        opt.temperature = 1.0f;
        opt.temperature_inc = 0.0f;
        opt.logprob_thold = -INFINITY;
        opt.no_speech_thold = INFINITY;
        opt.compression_ratio_thold = INFINITY;
        opt.max_new_tokens = 24;
        opt.seed = seed;
        return run(fake, contract(false), opt).raw_ids;
    };
    CHECK(sample_run(7) == sample_run(7));
    CHECK(sample_run(7) != sample_run(8));
}

void test_validation() {
    FakeDecoder fake;
    WhisperDecodeOptions opt;
    bool threw = false;

    opt.prompt_all_segments = true;  // HF: all-segments needs condition_on_prev
    try {
        run(fake, contract(false), opt);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);

    opt = {};
    opt.translate = true;  // English-only model
    threw = false;
    try {
        run(fake, contract(false), opt);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);

    opt = {};
    opt.prompt_ids = {kSot};  // specials are not prompt text
    threw = false;
    try {
        run(fake, contract(false), opt);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_retrieve_segments();
    test_timestamp_rules();
    test_greedy_short_form();
    test_fallback_ladder();
    test_token_budget_marks_truncated();
    test_token_budget_with_reentry_is_not_truncated();
    test_no_speech_gate();
    test_long_form_windows();
    test_language_detection_and_task();
    test_prompt_and_context_carry();
    test_seeded_sampling_is_reproducible();
    test_validation();
    if (g_failures != 0) {
        std::fprintf(stderr, "whisper_decoding_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("whisper_decoding_test: ok\n");
    return 0;
}
