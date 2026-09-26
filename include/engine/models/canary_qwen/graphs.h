#pragma once

// engine/models/canary_qwen/graphs.h - weights, KV cache and ggml graph
// builders of the native engine Canary-Qwen package. INTERNAL to
// src/models/canary_qwen/ (published under include/ like whisper's
// graphs_internal.h so the tests can reach it).
//
// Numerics contract: every builder emits the SAME ggml op sequence the arch
// emitted (src/runtime/arch/canary_qwen/{encoder,decoder}.cpp over
// src/runtime/conformer/ and src/runtime/causal_lm/). The shared runtime
// modules are ported into this package (conformer.cpp, causal_lm.cpp) rather
// than re-derived from the framework modules, whose op ordering differs
// (engine::modules conformer / qwen decoder): on a 32-block encoder feeding a
// greedy 28-layer LLM, op-order drift is enough to flip tokens. Only the
// paths canary_qwen reaches are ported (offline, full attention, symmetric
// depthwise padding, BatchNorm, no streaming caches, no masks).

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace engine::models::canary_qwen {

struct CanaryQwenHParams;

// ---------------------------------------------------------------------------
// Weights (borrowed ggml tensors living in the runtime's weight store)
// ---------------------------------------------------------------------------

struct PreEncodeWeights {
  // DwStridingSubsampling (factor 8, 256 channels).
  ggml_tensor *conv0_w = nullptr;  // [3, 3, 1, C]
  ggml_tensor *conv0_b = nullptr;
  ggml_tensor *conv2_w = nullptr;  // depthwise [3, 3, 1, C]
  ggml_tensor *conv2_b = nullptr;
  ggml_tensor *conv3_w = nullptr;  // pointwise [1, 1, C, C]
  ggml_tensor *conv3_b = nullptr;
  ggml_tensor *conv5_w = nullptr;
  ggml_tensor *conv5_b = nullptr;
  ggml_tensor *conv6_w = nullptr;
  ggml_tensor *conv6_b = nullptr;
  ggml_tensor *out_w = nullptr;    // [C * n_mels / 8, d_model]
  ggml_tensor *out_b = nullptr;
};

struct EncBlockWeights {
  ggml_tensor *norm_ff1_w = nullptr, *norm_ff1_b = nullptr;
  ggml_tensor *ff1_lin1_w = nullptr, *ff1_lin1_b = nullptr;
  ggml_tensor *ff1_lin2_w = nullptr, *ff1_lin2_b = nullptr;

  ggml_tensor *norm_attn_w = nullptr, *norm_attn_b = nullptr;
  ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
  ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
  ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
  ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
  ggml_tensor *attn_pos_w = nullptr;  // linear_pos (no bias)
  ggml_tensor *attn_pos_u = nullptr;  // [head_dim, n_heads]
  ggml_tensor *attn_pos_v = nullptr;

  ggml_tensor *norm_conv_w = nullptr, *norm_conv_b = nullptr;
  ggml_tensor *conv_pw1_w = nullptr, *conv_pw1_b = nullptr;  // [1, d, 2d]
  ggml_tensor *conv_dw_w = nullptr, *conv_dw_b = nullptr;    // [k, 1, d]
  ggml_tensor *conv_pw2_w = nullptr, *conv_pw2_b = nullptr;  // [1, d, d]
  // BatchNorm fused at load (scale, bias) from weight/bias/mean/var, eps 1e-5.
  ggml_tensor *conv_bn_fused_scale = nullptr;
  ggml_tensor *conv_bn_fused_bias = nullptr;

  ggml_tensor *norm_ff2_w = nullptr, *norm_ff2_b = nullptr;
  ggml_tensor *ff2_lin1_w = nullptr, *ff2_lin1_b = nullptr;
  ggml_tensor *ff2_lin2_w = nullptr, *ff2_lin2_b = nullptr;

  ggml_tensor *norm_out_w = nullptr, *norm_out_b = nullptr;
};

struct DecBlockWeights {
  ggml_tensor *norm_attn_w = nullptr;
  ggml_tensor *norm_ffn_w = nullptr;
  ggml_tensor *attn_q_w = nullptr;
  ggml_tensor *attn_k_w = nullptr;
  ggml_tensor *attn_v_w = nullptr;
  ggml_tensor *attn_o_w = nullptr;
  ggml_tensor *attn_q_norm = nullptr;  // [head_dim]
  ggml_tensor *attn_k_norm = nullptr;
  ggml_tensor *ffn_gate_up_w = nullptr; // packed [hidden, 2 * intermediate]
  ggml_tensor *ffn_down_w = nullptr;
};

struct CanaryQwenWeights {
  PreEncodeWeights pre_encode;
  std::vector<EncBlockWeights> blocks;
  ggml_tensor *proj_w = nullptr;  // enc.proj.weight [d_enc, out_dim]
  ggml_tensor *proj_b = nullptr;

  ggml_tensor *token_embd = nullptr;  // dec.token_embd.weight (tied lm_head)
  std::vector<DecBlockWeights> dec_blocks;
  ggml_tensor *output_norm = nullptr; // dec.output_norm.weight
};

// ---------------------------------------------------------------------------
// Conformer primitives (port of src/runtime/conformer, offline subset)
// ---------------------------------------------------------------------------

namespace conformer {

constexpr float kLayerNormEps = 1e-5f;

struct ConvPolicy {
  bool direct_pw = true;
  bool direct_conv0_in_pre_encode = false;
  bool direct_dw_in_block = false;
  bool direct_dw_in_pre_encode = false;
  bool inplace_pre_encode = false;
  int pre_encode_dw_time_chunk = 0;
};

struct BlockParams {
  int d_model = 0;
  int n_head = 0;
  int conv_kernel = 0;
  ggml_type kv_type = GGML_TYPE_COUNT;  // COUNT = auto (flash path only)
  bool use_flash = false;
  ConvPolicy policy;
};

// The arch's env overrides, same spelling and semantics (set and not
// starting with '0'): TRANSCRIBE_CONV_{DIRECT,NO_DIRECT}_{PW,DW}.
bool env_flag(const char *name);
bool resolve_conv_direct(const char *direct_env, const char *no_direct_env, bool backend_default);
bool detect_direct_pw(const char *backend_name);

ggml_tensor *build_pre_encode(ggml_context *ctx, const PreEncodeWeights &pe, ggml_tensor *mel_in,
                              const ConvPolicy &policy);

ggml_tensor *build_conformer_block(ggml_context *ctx, ggml_tensor *x, ggml_tensor *pos_emb,
                                   const EncBlockWeights &b, const BlockParams &params);

} // namespace conformer

// ---------------------------------------------------------------------------
// Causal LM primitives (port of src/runtime/causal_lm)
// ---------------------------------------------------------------------------

namespace causal_lm {

struct BlockParams {
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int max_position = 0;
  float rms_eps = 0.0f;
  float rope_theta = 0.0f;
};

// Self-attention KV cache. Flat 1-D K and V; layout (slowest -> fastest)
// layer, [batch,] position, head, dim - exactly the arch's.
struct KvCache {
  ggml_tensor *self_k = nullptr;
  ggml_tensor *self_v = nullptr;
  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  int n_ctx = 0;
  int n = 0;
  int head = 0;
  int n_batch = 1;

  KvCache() = default;
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;
  ~KvCache() { free(); }

  void free();
};

// Returns false on an allocation failure (only F16 / F32 caches).
bool kv_init(KvCache &cache, ggml_backend_t backend, int n_ctx, int n_kv_heads, int head_dim,
             int n_layer, ggml_type kv_type);
bool kv_init_batched(KvCache &cache, ggml_backend_t backend, int n_ctx, int n_kv_heads,
                     int head_dim, int n_layer, int n_batch, ggml_type kv_type);

ggml_tensor *block_prefill(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                           const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                           int layer_idx, int T_seq, ggml_tensor *mask, ggml_tensor *positions,
                           bool use_flash, bool slice_last_before_ffn);

ggml_tensor *block_step(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                        const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                        int layer_idx, int max_n_kv, ggml_tensor *mask, ggml_tensor *position,
                        ggml_tensor *kv_idx, bool use_flash);

ggml_tensor *block_prefill_batched(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                                   const DecBlockWeights &w, const BlockParams &params,
                                   KvCache &kv, int layer_idx, int T_seq, int n_batch,
                                   ggml_tensor *mask, ggml_tensor *positions, ggml_tensor *kv_idx,
                                   bool use_flash);

ggml_tensor *block_step_batched(ggml_context *ctx, ggml_cgraph *gf, ggml_tensor *x,
                                const DecBlockWeights &w, const BlockParams &params, KvCache &kv,
                                int layer_idx, int max_n_kv, int n_batch, ggml_tensor *mask,
                                ggml_tensor *position, ggml_tensor *kv_idx, bool use_flash);

} // namespace causal_lm

// ---------------------------------------------------------------------------
// Graph builders (port of arch/canary_qwen/{encoder,decoder}.cpp)
// ---------------------------------------------------------------------------

struct EncoderOptions {
  bool use_flash = false;       // arch default: off (rel-pos path)
  const char *backend_name = ""; // ggml_backend_name(); Vulkan => im2col pointwise
};

struct EncoderBuild {
  ggml_tensor *mel_in = nullptr;     // [n_frames, n_mels] f32 (mel-major data)
  ggml_tensor *pos_emb_in = nullptr; // [d_model, 2*T_enc - 1] f32
  ggml_tensor *out = nullptr;        // [out_dim, T_enc] perception output
  ggml_cgraph *graph = nullptr;
  int T_enc = 0;
};

EncoderBuild build_encoder_graph(ggml_context *ctx, const CanaryQwenWeights &w,
                                 const CanaryQwenHParams &hp, int n_mel_frames,
                                 const EncoderOptions &options);

// NeMo RelPositionalEncoding rows for positions (T_enc-1) .. -(T_enc-1):
// row-major [2*T_enc-1, d_model], sin/cos interleaved (the arch's
// build_relpos_emb_host).
void build_relpos_emb_host(std::vector<float> &pos_buf, int d_model, int T_enc);

struct PrefillBuild {
  ggml_tensor *input_ids_in = nullptr;  // [T_prompt] i32
  ggml_tensor *audio_in = nullptr;      // [hidden, T_audio] f32
  ggml_tensor *positions_in = nullptr;  // [T_prompt] i32
  ggml_tensor *mask_in = nullptr;       // [T_prompt, T_prompt] f16
  ggml_tensor *out = nullptr;           // [vocab] f32 (last-position logits)
  ggml_cgraph *graph = nullptr;
};

// `slice_last` is the arch's production setting (debug dumps off): the last
// block keeps only the final position before its FFN.
PrefillBuild build_prefill_graph(ggml_context *ctx, const CanaryQwenWeights &w,
                                 const CanaryQwenHParams &hp, causal_lm::KvCache &kv,
                                 int T_prompt, int T_audio, int prefix_len, int suffix_len,
                                 bool use_flash, bool slice_last);

struct StepBuild {
  ggml_tensor *input_id_in = nullptr;  // [1] i32
  ggml_tensor *position_in = nullptr;  // [1] i32
  ggml_tensor *kv_idx_in = nullptr;    // [1] i64
  ggml_tensor *mask_in = nullptr;      // [max_n_kv, 1] f16
  ggml_tensor *out = nullptr;          // [1] i32 argmax
  ggml_tensor *logits = nullptr;       // [vocab] f32
  ggml_cgraph *graph = nullptr;
};

StepBuild build_step_graph(ggml_context *ctx, const CanaryQwenWeights &w,
                           const CanaryQwenHParams &hp, causal_lm::KvCache &kv, int max_n_kv,
                           bool use_flash);

struct PrefillBuildBatched {
  ggml_tensor *input_ids_in = nullptr;   // [T_prompt_max, B] i32
  ggml_tensor *audio_dense_in = nullptr; // [hidden, T_prompt_max*B] f32
  ggml_tensor *keep_mask_in = nullptr;   // [1, T_prompt_max*B] f32
  ggml_tensor *positions_in = nullptr;   // [T_prompt_max] i32
  ggml_tensor *mask_in = nullptr;        // [T_prompt_max, T_prompt_max] f16
  ggml_tensor *kv_idx_in = nullptr;      // [T_prompt_max, B] i64
  ggml_tensor *last_idx_in = nullptr;    // [1, B] i32
  ggml_tensor *out = nullptr;            // [B] i32 argmax
  ggml_cgraph *graph = nullptr;
};

PrefillBuildBatched build_prefill_graph_batched(ggml_context *ctx, const CanaryQwenWeights &w,
                                                const CanaryQwenHParams &hp,
                                                causal_lm::KvCache &kv, int T_prompt_max,
                                                int T_audio_max, int n_batch, bool use_flash);

struct StepBuildBatched {
  ggml_tensor *input_ids_in = nullptr; // [B] i32
  ggml_tensor *position_in = nullptr;  // [B] i32
  ggml_tensor *kv_idx_in = nullptr;    // [1, B] i64
  ggml_tensor *mask_in = nullptr;      // [max_n_kv, 1, 1, B] f16
  ggml_tensor *out = nullptr;          // [B] i32 argmax
  ggml_cgraph *graph = nullptr;
};

StepBuildBatched build_step_graph_batched(ggml_context *ctx, const CanaryQwenWeights &w,
                                          const CanaryQwenHParams &hp, causal_lm::KvCache &kv,
                                          int max_n_kv, int n_batch, bool use_flash);

} // namespace engine::models::canary_qwen
