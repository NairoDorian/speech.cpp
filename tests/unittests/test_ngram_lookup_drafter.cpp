// tests/unittests/test_ngram_lookup_drafter.cpp - Phase 10.5 unit test for
// the 1-gram-lookup speculative drafter merged from transcribe.cpp's
// arch/qwen3_asr + arch/voxtral_realtime decode loops.
//
// Pure host logic, no model. The cases are hand-computed against the
// reference bookkeeping: a draft is the run that followed the LATEST earlier
// occurrence of the pending token; misses repeat the pending token; a token
// becomes a lookup target only once it is a previous occurrence (pinned at
// commit, after its own lookup); the last committed token stays unpinned
// until the next commit so its lookup can still find an earlier run.

#include "engine/framework/modules/transformers/causal_lm_ops.h"

#include <cstdio>
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

using engine::modules::transformers::NgramLookupDrafter;

std::vector<int32_t> draft_of(const NgramLookupDrafter & drafter, int32_t next_token, int64_t k) {
    std::vector<int32_t> out(static_cast<size_t>(k), -1);
    drafter.draft(next_token, k, out.data());
    return out;
}

void test_lookup_follows_latest_occurrence() {
    // prompt = [5, 6, 7, 6, 8], first generated token = 6 (position 5).
    const NgramLookupDrafter drafter({5, 6, 7, 6, 8}, 6, 16);
    CHECK(drafter.sequence().size() == 6);
    // Latest prompt occurrence of 6 is position 3 -> the run after it is
    // [8, 6(first token), <end>], and the miss past the end repeats 6.
    CHECK((draft_of(drafter, 6, 3) == std::vector<int32_t>{8, 6, 6}));
    // 7 occurred once, at position 2 -> [6, 8, 6].
    CHECK((draft_of(drafter, 7, 3) == std::vector<int32_t>{6, 8, 6}));
    // Never seen -> repeat the pending token for every column.
    CHECK((draft_of(drafter, 9, 2) == std::vector<int32_t>{9, 9}));
}

void test_commit_pins_previous_occurrences_only() {
    NgramLookupDrafter drafter({5, 6, 7, 6, 8}, 6, 16);
    // The verify pass at position 5 (pending token 6) confirmed [8, 9, 10].
    const int32_t committed[] = {8, 9, 10};
    drafter.commit(6, 5, committed, 3);
    CHECK((drafter.sequence() == std::vector<int32_t>{5, 6, 7, 6, 8, 6, 8, 9, 10}));
    // 6 is now pinned at 5: its run is [8, 9, 10].
    CHECK((draft_of(drafter, 6, 3) == std::vector<int32_t>{8, 9, 10}));
    // 8 is pinned at 6 (a previous occurrence): [9, 10, miss->8].
    CHECK((draft_of(drafter, 8, 3) == std::vector<int32_t>{9, 10, 8}));
    // 10 is the next pending token and deliberately NOT pinned yet: no earlier
    // occurrence -> misses.
    CHECK((draft_of(drafter, 10, 2) == std::vector<int32_t>{10, 10}));

    // Next pass at position 8 (pending 10) confirms just [11]: 10 gets pinned
    // at 8, 11 stays unpinned.
    const int32_t committed2[] = {11};
    drafter.commit(10, 8, committed2, 1);
    CHECK(drafter.sequence().size() == 10);
    CHECK((draft_of(drafter, 10, 2) == std::vector<int32_t>{11, 10}));
    CHECK((draft_of(drafter, 11, 1) == std::vector<int32_t>{11}));
}

void test_repeated_token_uses_latest_run() {
    // A token seen twice drafts from its LATEST run, not its first.
    NgramLookupDrafter drafter({1, 2, 3, 1, 4, 5}, 1, 8);
    CHECK((draft_of(drafter, 1, 2) == std::vector<int32_t>{4, 5}));
    const int32_t committed[] = {4, 9};
    drafter.commit(1, 6, committed, 2);
    // 1 is now latest at 6 -> run [4, 9]; 4 latest at 7 -> [9, miss->4].
    CHECK((draft_of(drafter, 1, 2) == std::vector<int32_t>{4, 9}));
    CHECK((draft_of(drafter, 4, 2) == std::vector<int32_t>{9, 4}));
}

void test_zero_drafts_is_a_no_op() {
    const NgramLookupDrafter drafter({1, 2}, 3, 0);
    std::vector<int32_t> out;
    drafter.draft(3, 0, out.data());  // must not touch out
    CHECK(out.empty());
}

}  // namespace

int main() {
    test_lookup_follows_latest_occurrence();
    test_commit_pins_previous_occurrences_only();
    test_repeated_token_uses_latest_run();
    test_zero_drafts_is_a_no_op();
    if (g_failures != 0) {
        std::fprintf(stderr, "ngram_lookup_drafter_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("ngram_lookup_drafter_test: ALL PASSED\n");
    return 0;
}
