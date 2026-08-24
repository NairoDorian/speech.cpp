#pragma once

#include "ggml.h"
#include <cstdint>

namespace engine::modules::transformers {

constexpr int kPrefillChunkDefault = 2048;
constexpr int kPrefillChunkMax = 32768;

int pick_kv_cache_context(int needed, int model_max);
void fill_prefill_chunk_mask(ggml_fp16_t * dst, int max_n_kv, int T_chunk, int n_past);
int prefill_chunk_size();

} // namespace engine::modules::transformers
