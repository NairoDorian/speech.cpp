// engine/framework/asr/enc_dec_kv_cache.h - shared KV cache for
// encoder-decoder ASR families (Moonshine W1a/W1b, Whisper W2a).
//
// Replaces the three identical private structs that were copy-pasted
// across include/engine/models/{moonshine,moonshine_streaming,whisper}/
// graphs_internal.h (B30). The dual-slab layout is:
//   - self K/V  [d_model, n_layer * n_ctx]  (grows one position per decode step)
//   - cross K/V [d_model, n_layer * T_enc]  (precomputed once per audio window)
//
// All three families stored tensors as flattened 1-D buffers and indexed
// them in their graph builders via ggml_view_2d/3d at per-layer offsets.
// This type preserves that exact memory layout so the graph builders
// can migrate with a single `using` alias.
//
// The view helpers below (kv_cache_self_kv_view, kv_cache_cross_kv_view)
// encapsulate the offset arithmetic that was inlined in each family's
// graph builder, cutting ~240 LOC of duplicated indexing code.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>

namespace engine::asr {

struct EncDecKVCache {
  ggml_tensor *self_k = nullptr;
  ggml_tensor *self_v = nullptr;
  ggml_tensor *cross_k = nullptr;
  ggml_tensor *cross_v = nullptr;

  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;

  int n_ctx = 0;   // decoder context length
  int n = 0;       // current decode position (tokens emitted so far)
  int head = 0;    // position within the current chunk (batching)
  int T_enc = 0;   // encoder sequence length

  bool cross_populated = false;

  void free();

  // True after init (buffer allocated, tensors owned by the backend).
  bool initialized() const { return buffer != nullptr; }
};

// Allocate the dual KV slabs on the given backend.
//
// Returns false (not thrown) so callers can degrade gracefully — e.g.
// fall back to no KV cache on OOM, matching the arch run_one contract.
bool kv_cache_init(EncDecKVCache &cache, ggml_backend_t backend, int n_ctx,
                   int T_enc, int d_model, int n_layer, ggml_type kv_type);

// Reset self-attention position without re-allocating (re-uses buffer).
// Does NOT touch cross-attn; that is set by cross_populated / reset_cross().
inline void reset_self(EncDecKVCache &cache) {
  cache.n = 0;
  cache.head = 0;
}

// Reset the cross-attention slab and its populated flag.
inline void reset_cross(EncDecKVCache &cache) {
  cache.cross_populated = false;
}

// ---------------------------------------------------------------------------
// View helpers: create per-layer 3-D views into the flattened 1-D KV slab.
// Mirrors the indexing inlined in each family's graph builder
// (Whisper mha_self_cached / mha_cross_cached).
//
// Flat layout per tensor: [layer0_block, layer1_block, ...] where each
// block is [d_model * span] (span = n_ctx for self, T_enc for cross).
// Within a block: [token0[d_model], token1[d_model], ...] and within
// a token: [head0[d_head], head1[d_head], ...] (ggml row-major: d_head
// is fastest, then n_heads, then span).
//
// ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset) where:
//   ne0 = d_head       (fastest: elements within a head)
//   ne1 = n_span       (positions  )
//   ne2 = n_heads      (heads       )
//   nb1 = stride between positions  in bytes = d_model * elem_size
//   nb2 = stride between heads      in bytes = d_head  * elem_size
//   offset = byte offset of (layer * d_model * span + n_past * d_model)
// ---------------------------------------------------------------------------

// Read-write view of self-attention K/V for one layer: covers n_tokens
// positions starting at n_past.  Used by the write-back (ggml_cpy) in
// mha_self_cached and by the read view in the same function.
inline ggml_tensor *kv_cache_self_view(ggml_context *ctx, ggml_tensor *kv,
                                       int d_model, int n_layer, int layer,
                                       int d_head, int n_heads, int n_past,
                                       int n_tokens, int n_ctx) {
  const size_t es = ggml_element_size(kv);
  const int64_t layer_off = static_cast<int64_t>(layer) * n_ctx * d_model;
  const int64_t pos_off   = static_cast<int64_t>(n_past) * d_model;
  return ggml_view_3d(ctx, kv, d_head, n_tokens, n_heads,
                      es * d_model, es * d_head,
                      es * (layer_off + pos_off));
}

// Read-only view of the full self-attention history for one layer
// (positions 0..n_kv-1).  Used by attention-score passes.
inline ggml_tensor *kv_cache_self_history_view(ggml_context *ctx,
                                               ggml_tensor *kv,
                                               int d_model, int n_layer,
                                               int layer, int d_head,
                                               int n_heads, int n_kv, int n_ctx) {
  const size_t es = ggml_element_size(kv);
  const int64_t layer_off = static_cast<int64_t>(layer) * n_ctx * d_model;
  return ggml_view_3d(ctx, kv, d_head, n_kv, n_heads,
                      es * d_model, es * d_head,
                      es * layer_off);
}

// Full cross-attention view for one layer: [d_head, T_enc, n_heads].
inline ggml_tensor *kv_cache_cross_view(ggml_context *ctx, ggml_tensor *kv,
                                        int d_model, int n_layer, int layer,
                                        int d_head, int n_heads, int T_enc) {
  const size_t es = ggml_element_size(kv);
  const int64_t layer_off = static_cast<int64_t>(layer) * T_enc * d_model;
  return ggml_view_3d(ctx, kv, d_head, T_enc, n_heads,
                      es * d_model, es * d_head,
                      es * layer_off);
}

} // namespace engine::asr
