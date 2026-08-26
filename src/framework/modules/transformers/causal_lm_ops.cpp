#include "engine/framework/modules/transformers/causal_lm_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace engine::modules::transformers {

int pick_kv_cache_context(int needed, int model_max) {
    if (model_max <= 0) {
        return 0;
    }

    constexpr int k_min_context = 1024;
    constexpr int k_large_step  = 4096;

    int64_t want = k_min_context;
    if (needed > k_large_step) {
        want = (static_cast<int64_t>(needed) + k_large_step - 1) / k_large_step * k_large_step;
    } else {
        while (want < needed) {
            want *= 2;
        }
    }
    return static_cast<int>(std::min<int64_t>(want, model_max));
}

void fill_prefill_chunk_mask(ggml_fp16_t * dst, int max_n_kv, int T_chunk, int n_past) {
    if (dst == nullptr || max_n_kv <= 0 || T_chunk <= 0 || n_past < 0 || n_past + T_chunk > max_n_kv) {
        return;
    }
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T_chunk; ++q) {
        ggml_fp16_t * row  = dst + static_cast<size_t>(q) * max_n_kv;
        const int     last = n_past + q;
        std::fill(row, row + last + 1, keep);
        std::fill(row + last + 1, row + max_n_kv, drop);
    }
}

int prefill_chunk_size() {
    const char * env = std::getenv("TRANSCRIBE_PREFILL_CHUNK");
    if (env != nullptr && *env != '\0') {
        const int parsed = std::atoi(env);
        if (parsed > 0) {
            return std::min(parsed, kPrefillChunkMax);
        }
    }
    return kPrefillChunkDefault;
}

NgramLookupDrafter::NgramLookupDrafter(const std::vector<int32_t> & prompt_ids, int32_t first_token, int64_t reserve_extra) {
    all_ids_.reserve(prompt_ids.size() + 1 + static_cast<size_t>(reserve_extra > 0 ? reserve_extra : 0));
    all_ids_.insert(all_ids_.end(), prompt_ids.begin(), prompt_ids.end());
    all_ids_.push_back(first_token);
    last_pos_by_tok_.reserve(all_ids_.capacity());
    for (size_t p = 0; p < prompt_ids.size(); ++p) {
        last_pos_by_tok_[prompt_ids[p]] = static_cast<int64_t>(p);
    }
}

void NgramLookupDrafter::draft(int32_t next_token, int64_t k, int32_t * out) const {
    if (k <= 0 || out == nullptr) {
        return;
    }
    const auto it = last_pos_by_tok_.find(next_token);
    const int64_t origin = it != last_pos_by_tok_.end() ? it->second : -1;
    const int64_t size = static_cast<int64_t>(all_ids_.size());
    for (int64_t c = 1; c <= k; ++c) {
        const int64_t src = origin >= 0 ? origin + c : -1;
        out[c - 1] = (src >= 0 && src < size) ? all_ids_[static_cast<size_t>(src)] : next_token;
    }
}

void NgramLookupDrafter::commit(int32_t next_token, int64_t position, const int32_t * committed, int64_t n_committed) {
    last_pos_by_tok_[next_token] = position;
    for (int64_t i = 0; i < n_committed; ++i) {
        all_ids_.push_back(committed[i]);
    }
    for (int64_t j = 0; j + 1 < n_committed; ++j) {
        last_pos_by_tok_[committed[j]] = position + 1 + j;
    }
}

} // namespace engine::modules::transformers
