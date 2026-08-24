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

} // namespace engine::modules::transformers
