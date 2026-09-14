// engine/framework/asr/enc_dec_kv_cache.cpp - shared KV cache for
// encoder-decoder ASR families.

#include "engine/framework/asr/enc_dec_kv_cache.h"

void engine::asr::EncDecKVCache::free() {
  if (buffer != nullptr) {
    ggml_backend_buffer_free(buffer);
    buffer = nullptr;
  }
  if (ctx != nullptr) {
    ggml_free(ctx);
    ctx = nullptr;
  }
  self_k = nullptr;
  self_v = nullptr;
  cross_k = nullptr;
  cross_v = nullptr;
  n = 0;
  head = 0;
  T_enc = 0;
  cross_populated = false;
}

bool engine::asr::kv_cache_init(EncDecKVCache &cache, ggml_backend_t backend,
                                int n_ctx, int T_enc, int d_model, int n_layer,
                                ggml_type kv_type) {
  if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
    return false;
  }
  if (backend == nullptr) {
    return false;
  }

  const size_t ctx_size = 4 * ggml_tensor_overhead() + 256;
  ggml_init_params params{ctx_size, nullptr, /*no_alloc=*/true};
  cache.ctx = ggml_init(params);
  if (cache.ctx == nullptr) {
    return false;
  }

  const int64_t self_elements =
      static_cast<int64_t>(d_model) * n_layer * n_ctx;
  const int64_t cross_elements =
      static_cast<int64_t>(d_model) * n_layer * T_enc;

  cache.self_k = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
  cache.self_v = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
  cache.cross_k = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
  cache.cross_v = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);

  ggml_set_name(cache.self_k, "kv_self_k");
  ggml_set_name(cache.self_v, "kv_self_v");
  ggml_set_name(cache.cross_k, "kv_cross_k");
  ggml_set_name(cache.cross_v, "kv_cross_v");

  cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
  if (cache.buffer == nullptr) {
    ggml_free(cache.ctx);
    cache.ctx = nullptr;
    return false;
  }
  ggml_backend_buffer_clear(cache.buffer, 0);

  cache.n_ctx = n_ctx;
  cache.T_enc = T_enc;
  cache.n = 0;
  cache.head = 0;
  cache.cross_populated = false;

  return true;
}
