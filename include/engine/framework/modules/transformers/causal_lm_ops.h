#pragma once

#include "ggml.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace engine::modules::transformers {

constexpr int kPrefillChunkDefault = 2048;
constexpr int kPrefillChunkMax = 32768;

int pick_kv_cache_context(int needed, int model_max);
void fill_prefill_chunk_mask(ggml_fp16_t * dst, int max_n_kv, int T_chunk, int n_past);
int prefill_chunk_size();

// 1-gram-lookup speculative drafter - the mechanism transcribe.cpp's
// arch/qwen3_asr and arch/voxtral_realtime decode loops ran, merged into the
// engine in Phase 10.5. Tracks every token of the sequence (prompt + generated)
// and the latest position of each token id; a draft for `next_token` is the
// run of tokens that followed its most recent earlier occurrence, padded with
// next_token itself when there is none (a miss only wastes the draft columns,
// whose compute rides along with the mandatory column 0).
//
// Position bookkeeping mirrors the reference exactly: a token's position
// becomes a lookup target only once it is a PREVIOUS occurrence. next_token is
// pinned at commit time (after its own lookup), and the last committed token -
// the next next_token - is deliberately left unpinned so its lookup can find
// an earlier occurrence instead of its own tail position.
class NgramLookupDrafter {
public:
    // prompt_ids occupy positions [0, n); first_token (the first generated
    // token) sits at position n. reserve_extra pre-sizes for the generation
    // budget.
    NgramLookupDrafter(const std::vector<int32_t> & prompt_ids, int32_t first_token, int64_t reserve_extra = 0);

    // Writes k draft tokens to out[0..k). k == 0 writes nothing.
    void draft(int32_t next_token, int64_t k, int32_t * out) const;

    // next_token sat at absolute `position`; committed[0..n_committed) are
    // the tokens the verify pass confirmed, in order (the last one becomes
    // the next next_token).
    void commit(int32_t next_token, int64_t position, const int32_t * committed, int64_t n_committed);

    const std::vector<int32_t> & sequence() const noexcept { return all_ids_; }

private:
    std::vector<int32_t> all_ids_;
    std::unordered_map<int32_t, int64_t> last_pos_by_tok_;
};

} // namespace engine::modules::transformers
