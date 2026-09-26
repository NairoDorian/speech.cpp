#pragma once

// engine/models/voxtral/graphs.h - weights, KV cache and ggml graph builders
// of the native engine Voxtral (2507) package. INTERNAL to src/models/voxtral/
// (published under include/ like canary_qwen's graphs.h so tests can reach it).
//
// Numerics contract: every builder emits the SAME ggml op sequence the arch
// emitted (src/runtime/arch/voxtral/{encoder,decoder}.cpp over
// src/runtime/conformer/ (layer_norm, conv_1d_f32) and src/runtime/causal_lm/
// (block_prefill, block_step, block_step_n)). Those shared runtime modules are
// ported into this package rather than re-derived from engine::modules, whose
// op ordering differs: on a 32-block encoder feeding a greedy 30-layer LLM,
// op-order drift is enough to flip tokens. Only the paths the arch's single
// utterance run() reaches are ported (no batched decode, no debug dumps).

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

namespace engine::models::voxtral {

struct VoxtralHParams;

// ---------------------------------------------------------------------------
// Weights (borrowed ggml tensors living in the runtime's weight store). Slot
// names follow arch/voxtral/weights.h; ggml ne order is noted per slot.
// ---------------------------------------------------------------------------

// One Whisper encoder block. q/v/out carry bias; k does NOT.
struct EncBlockWeights {
  ggml_tensor *norm_attn_w = nullptr, *norm_attn_b = nullptr;  // [d]
  ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;        // [d, d], [d]
  ggml_tensor *attn_k_w = nullptr;                              // [d, d] (no bias)
  ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
  ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
  ggml_tensor *norm_ffn_w = nullptr, *norm_ffn_b = nullptr;
  ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;  // [d, ffn], [ffn]
  ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;  // [ffn, d], [d]
};

// One Llama / Ministral decoder block (no biases, no per-head Q/K norm).
struct DecBlockWeights {
  ggml_tensor *norm_attn_w = nullptr;    // [hidden] input_layernorm (RMSNorm)
  ggml_tensor *norm_ffn_w = nullptr;     // [hidden] post_attention_layernorm
  ggml_tensor *attn_q_w = nullptr;       // [hidden, n_heads * head_dim]
  ggml_tensor *attn_k_w = nullptr;       // [hidden, n_kv_heads * head_dim]
  ggml_tensor *attn_v_w = nullptr;       // [hidden, n_kv_heads * head_dim]
  ggml_tensor *attn_o_w = nullptr;       // [n_heads * head_dim, hidden]
  ggml_tensor *ffn_gate_up_w = nullptr;  // packed [hidden, 2 * intermediate] (gate bytes, then up)
  ggml_tensor *ffn_down_w = nullptr;     // [intermediate, hidden]
};

struct VoxtralWeights {
  // Encoder stem + top.
  ggml_tensor *conv0_w = nullptr;  // [3, n_mels, d] (F16 in every shipped GGUF)
  ggml_tensor *conv0_b = nullptr;  // [d]
  ggml_tensor *conv1_w = nullptr;  // [3, d, d]
  ggml_tensor *conv1_b = nullptr;  // [d]
  ggml_tensor *pos_emb_w = nullptr;  // [d, max_source_positions] (fixed sinusoidal, F32)
  ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
  std::vector<EncBlockWeights> enc_blocks;

  // Projector (no biases).
  ggml_tensor *proj_linear_1_w = nullptr;  // [proj_in, hidden]
  ggml_tensor *proj_linear_2_w = nullptr;  // [hidden, hidden]

  // Text LM.
  ggml_tensor *token_embd_w = nullptr;   // [hidden, vocab]
  ggml_tensor *output_w = nullptr;       // [hidden, vocab] UNTIED lm_head
  std::vector<DecBlockWeights> dec_blocks;
  ggml_tensor *output_norm_w = nullptr;  // [hidden]
};

// ---------------------------------------------------------------------------
// Causal LM primitives (port of src/runtime/causal_lm, single-utterance subset)
// ---------------------------------------------------------------------------

namespace causal_lm {

struct BlockParams {
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int max_position = 0;  // RoPE n_ctx_orig passed to ggml_rope_ext
  float rms_eps = 0.0f;
  float rope_theta = 0.0f;
};

// Self-attention KV cache. Flat 1-D K and V of n_kv_heads * head_dim * n_ctx *
// n_layer elements; layout (slowest -> fastest) layer, position, head, dim -
// exactly the arch's. Lives in its own backend buffer (never graph-allocated).
struct KvCache {
  ggml_tensor *self_k = nullptr;
  ggml_tensor *self_v = nullptr;
  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  int n_ctx = 0;

  KvCache() = default;
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;
  ~KvCache() { free(); }

  bool allocated() const noexcept { return self_k != nullptr && self_v != nullptr; }
  void free();
};

// Allocates (zero-filled) K/V on `backend`. Returns false on an allocation
// failure; only F16 / F32 caches (the arch's kv_init).
bool kv_init(KvCache &cache, ggml_backend_t backend, int n_ctx, int n_kv_heads, int head_dim,
             int n_layer, ggml_type kv_type);

// Prefill block over [hidden, T_seq], writing K/V rows [0, T_seq) of layer
// `layer_idx` (1-D cpy). mask [T_seq, T_seq] f16, positions [T_seq] i32.
// slice_last_before_ffn: keep only the last position before the FFN (the
// arch's production setting on the last block).
ggml_tensor *block_prefill(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                           const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                           int layer_idx, int T_seq, ggml_tensor *mask, ggml_tensor *positions,
                           bool use_flash, bool slice_last_before_ffn);

// Single-token step: writes the K/V row at kv_idx ([1] i64) with set_rows and
// attends over the full [0, max_n_kv) window under mask [max_n_kv, 1] f16.
ggml_tensor *block_step(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                        const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                        int layer_idx, int max_n_kv, ggml_tensor *mask, ggml_tensor *position,
                        ggml_tensor *kv_idx, bool use_flash);

// Multi-position step (chunked prefill): T_seq rows at kv_idx ([T_seq] i64),
// full [0, max_n_kv) window under mask [max_n_kv, T_seq] f16.
ggml_tensor *block_step_n(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                          const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                          int layer_idx, int T_seq, int max_n_kv, ggml_tensor *mask,
                          ggml_tensor *positions, ggml_tensor *kv_idx, bool use_flash);

}  // namespace causal_lm

// ---------------------------------------------------------------------------
// Graph builders (port of arch/voxtral/{encoder,decoder}.cpp). Builders throw
// std::runtime_error on invalid geometry (the arch logged and returned null).
// ---------------------------------------------------------------------------

struct EncoderBuild {
  ggml_tensor *mel_in = nullptr;  // [n_mels, n_mel_frames] f32 (frame-major data)
  ggml_tensor *out = nullptr;     // [hidden, n_audio_tokens] projector output
  ggml_cgraph *graph = nullptr;
};

// Encoder + projector for ONE 30 s chunk. n_mel_frames must be
// 2 * max_source_positions (3000) so conv2 yields exactly 1500 frames.
EncoderBuild build_encoder_graph(ggml_context *ctx, const VoxtralWeights &w,
                                 const VoxtralHParams &hp, int n_mel_frames, bool use_flash);

struct PrefillBuild {
  ggml_tensor *input_ids_in = nullptr;  // [T_prompt] i32
  ggml_tensor *enc_out_in = nullptr;    // [hidden, T_enc] f32 (projector output)
  ggml_tensor *positions_in = nullptr;  // [T_prompt] i32
  ggml_tensor *mask_in = nullptr;       // [T_prompt, T_prompt] f16 causal
  ggml_tensor *out = nullptr;           // [vocab] f32 last-position logits
  ggml_cgraph *graph = nullptr;
};

// Single-shot prefill, the arch's production form (slice_last = true: debug
// dumps off). Audio is spliced over the placeholder run via a 3-way concat.
PrefillBuild build_prefill_graph(ggml_context *ctx, const VoxtralWeights &w,
                                 const VoxtralHParams &hp, causal_lm::KvCache &kv, int T_prompt,
                                 int T_enc, int prefix_len, int suffix_len, bool use_flash);

struct PrefillChunkBuild {
  ggml_tensor *input_ids_in = nullptr;  // [T_chunk] i32 (null when the chunk is pure audio)
  ggml_tensor *enc_out_in = nullptr;    // [hidden, aud_n] f32 (null when aud_n == 0)
  ggml_tensor *positions_in = nullptr;  // [T_chunk] i32
  ggml_tensor *kv_idx_in = nullptr;     // [T_chunk] i64
  ggml_tensor *mask_in = nullptr;       // [max_n_kv, T_chunk] f16
  ggml_tensor *out = nullptr;           // [vocab] f32 (final chunk only)
  ggml_cgraph *graph = nullptr;
};

// One chunk of a long prompt against the KV earlier chunks wrote (the arch's
// build_prefill_chunk_graph). The chunk holds pre_n token rows, then aud_n
// audio rows, then suf_n token rows.
PrefillChunkBuild build_prefill_chunk_graph(ggml_context *ctx, const VoxtralWeights &w,
                                            const VoxtralHParams &hp, causal_lm::KvCache &kv,
                                            int T_chunk, int max_n_kv, int pre_n, int aud_n,
                                            int suf_n, bool use_flash, bool want_logits);

struct StepBuild {
  ggml_tensor *input_id_in = nullptr;  // [1] i32
  ggml_tensor *position_in = nullptr;  // [1] i32 (= n_past)
  ggml_tensor *kv_idx_in = nullptr;    // [1] i64 (KV write row)
  ggml_tensor *mask_in = nullptr;      // [max_n_kv, 1] f16
  ggml_tensor *out = nullptr;          // [1] i32 argmax token id
  ggml_tensor *logits = nullptr;       // [vocab] f32
  ggml_cgraph *graph = nullptr;
};

// Static-shape single-token step graph, reused across every decode step.
// Every input above is uploaded before EVERY compute (the cached-graph rule).
StepBuild build_step_graph(ggml_context *ctx, const VoxtralWeights &w, const VoxtralHParams &hp,
                           causal_lm::KvCache &kv, int max_n_kv, bool use_flash);

}  // namespace engine::models::voxtral
