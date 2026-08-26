// engine/models/moonshine_streaming/graphs_internal.h - native engine
// Moonshine-Streaming ASR graph layer: hparams, weight slots, KV cache, and
// ggml graph builders.
//
// Phase 11 W1b port of src/runtime/arch/moonshine_streaming/ into the engine
// framework. The numerics mirror the arch implementation exactly (same graph
// topology, same sliding-window masks, same partial-RoPE handling, same
// head-dim padding policy); only the surrounding machinery differs (engine
// BackendWeightStore / TokenizerHub / RunControl instead of transcribe_model /
// transcribe::Tokenizer / poll_abort).
//
// Differences from the offline Moonshine package (src/models/moonshine):
//
//   - Frontend. No raw-PCM 3-conv stem. Instead a time-domain stack:
//       CMVN (parameter-free, eps=1e-6) -> asinh(exp(log_k) * x)
//       -> Linear(frame_len -> hidden, no bias) + SiLU
//       -> CausalConv1d(hidden -> 2*hidden, k=5, s=2) + bias + SiLU
//       -> CausalConv1d(2*hidden -> hidden, k=5, s=2) + bias
//   - Encoder LayerNorms use unit_offset=True; the converter PRE-FOLDS the
//     +1.0 into gamma, so C++ uses ordinary LayerNorm(no affine) * scale.
//   - Encoder attention has NO RoPE. Position information comes from
//     per-layer (left, right) sliding-window attention masks, flattened as
//     [L0, R0, L1, R1, ...] in stt.moonshine_streaming.encoder.sliding_windows.
//   - Adapter between encoder and decoder cross-attn:
//       adapter.pos_emb.weight [max_pos, enc_hidden]  (always present)
//       adapter.proj.weight    [enc_hidden, dec_hidden] (only when the
//                                                        widths differ)
//     pos_emb is an ABSOLUTE-frame get_rows, which is what makes per-window
//     slicing equal the one-shot result.
//   - Untied lm_head: dec.lm_head.weight is separate from dec.token_embd.
//   - Decoder LayerNorms are vanilla nn.LayerNorm(bias=False).

#pragma once

// Internal graph-layer header for the native Moonshine-Streaming engine
// package. Not part of any installed public API surface.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;

namespace engine::models::moonshine_streaming {

// ---------------------------------------------------------------------------
// Hyperparameters (read from GGUF metadata keys
// stt.moonshine_streaming.*)
// ---------------------------------------------------------------------------

struct MoonshineStreamingHParams {
  // Encoder.
  int32_t enc_n_layers = 0;
  int32_t enc_d_model = 0;
  int32_t enc_n_heads = 0;
  int32_t enc_n_kv_heads = 0; // streaming-tiny has no GQA: == n_heads
  int32_t enc_head_dim = 0;   // explicit head_dim in config
  int32_t enc_ffn_dim = 0;    // intermediate_size
  std::string enc_activation = "gelu";
  float enc_frame_ms = 0.0f;
  int32_t enc_frame_len = 0; // = round(sample_rate * frame_ms / 1000)
  // Flattened per-layer (left, right) windows: 2 ints per layer.
  std::vector<int32_t> enc_sliding_windows;

  // Decoder.
  int32_t dec_n_layers = 0;
  int32_t dec_d_model = 0;
  int32_t dec_n_heads = 0;
  int32_t dec_n_kv_heads = 0;
  int32_t dec_head_dim = 0;
  int32_t dec_ffn_dim = 0;
  int32_t dec_max_position_embeddings = 0;
  int32_t dec_vocab_size = 0;
  std::string dec_activation = "silu";
  bool dec_tie_word_embeddings = false;

  // Special tokens.
  int32_t bos_token_id = -1;           // 1
  int32_t eos_token_id = -1;           // 2
  int32_t pad_token_id = -1;           // 0 (vs moonshine's 2)
  int32_t decoder_start_token_id = -1; // 1

  // Attention / RoPE (decoder only - the encoder has no RoPE).
  float partial_rotary_factor = 0.8f;
  float rope_theta = 10000.0f;
  bool attention_bias = false;
  int32_t pad_head_dim_multiple = 0;

  // Frontend (raw waveform - framed inside the encoder).
  std::string fe_type = "raw";
  int32_t fe_sample_rate = 16000;

  // CMVN epsilon (fixed at 1e-6 in the reference).
  float cmvn_eps = 1e-6f;

  // Adapter.
  int32_t encoder_hidden_size = 0; // mirrors enc_d_model; convenience read
  bool adapter_has_proj = false;

  // Encoder attention "channel" dimension = n_heads * head_dim. NOT
  // necessarily equal to enc_d_model: small/medium have enc_d_model=620/768
  // while enc_attn_dim=512/640. Q/K/V project enc_d_model -> enc_attn_dim and
  // the output projection takes enc_attn_dim back to enc_d_model. Tiny
  // coincidentally has both equal (320 = 8 * 40).
  int32_t enc_attn_dim() const { return enc_n_heads * enc_head_dim; }

  int32_t padded_head_dim(int32_t head_dim) const {
    const int32_t m = pad_head_dim_multiple;
    if (m <= 0 || head_dim <= 0) {
      return head_dim;
    }
    return ((head_dim + m - 1) / m) * m;
  }

  int32_t enc_head_dim_padded() const { return padded_head_dim(enc_head_dim); }
  int32_t dec_head_dim_padded() const { return padded_head_dim(dec_head_dim); }

  // Decoder partial RoPE rotation width: 32 of 40 head_dim for tiny
  // (head_dim * 0.8 = 32). Mask off the odd bit so the interleaved rotate
  // halves match HF.
  int32_t dec_head_dim_rot() const {
    const int32_t r = static_cast<int32_t>(static_cast<float>(dec_head_dim) *
                                           partial_rotary_factor);
    return r & ~int32_t{1};
  }

  // Per-layer sliding window accessors (flat [L0, R0, L1, R1, ...]).
  int32_t layer_left_window(int32_t layer) const {
    const size_t idx = static_cast<size_t>(layer) * 2;
    return idx < enc_sliding_windows.size() ? enc_sliding_windows[idx] : 0;
  }
  int32_t layer_right_window(int32_t layer) const {
    const size_t idx = static_cast<size_t>(layer) * 2 + 1;
    return idx < enc_sliding_windows.size() ? enc_sliding_windows[idx] : 0;
  }
};

// ---------------------------------------------------------------------------
// Weight slots
// ---------------------------------------------------------------------------

// Encoder embedder: time-domain frontend living inside the encoder graph.
struct MoonshineStreamingEmbedder {
  ggml_tensor *comp_log_k = nullptr; // [1] f32 - learned scalar
  ggml_tensor *linear_w = nullptr;   // [frame_len, hidden] no bias
  ggml_tensor *conv1_w = nullptr;    // [5, hidden, 2*hidden]
  ggml_tensor *conv1_b = nullptr;    // [2*hidden]
  ggml_tensor *conv2_w = nullptr;    // [5, 2*hidden, hidden]
  ggml_tensor *conv2_b = nullptr;    // [hidden]
};

struct MoonshineStreamingEncTop {
  ggml_tensor *final_norm_w = nullptr; // [hidden] (pre-folded gain)
};

// One encoder transformer block. attention_bias=False on q/k/v/o.
// LayerNorm scales are pre-folded (gain = original gamma + 1.0).
struct MoonshineStreamingEncBlock {
  ggml_tensor *norm_attn_w = nullptr;
  ggml_tensor *attn_q_w = nullptr;
  ggml_tensor *attn_k_w = nullptr;
  ggml_tensor *attn_v_w = nullptr;
  ggml_tensor *attn_out_w = nullptr;

  ggml_tensor *norm_ffn_w = nullptr;
  ggml_tensor *ffn_fc1_w = nullptr; // [hidden, ffn_dim]
  ggml_tensor *ffn_fc1_b = nullptr; // [ffn_dim]
  ggml_tensor *ffn_fc2_w = nullptr; // [ffn_dim, hidden]
  ggml_tensor *ffn_fc2_b = nullptr; // [hidden]
};

// Adapter: pos_emb add (+ proj when enc_hidden != dec_hidden).
struct MoonshineStreamingAdapter {
  ggml_tensor *pos_emb_w = nullptr; // [max_pos, enc_hidden] always present
  ggml_tensor *proj_w = nullptr;    // [enc_hidden, dec_hidden] optional
};

// Decoder top: untied embed + final LN + lm_head.
struct MoonshineStreamingDecTop {
  ggml_tensor *token_embd_w = nullptr; // [hidden, vocab]
  ggml_tensor *final_norm_w = nullptr; // [hidden] (no offset trick)
  ggml_tensor *lm_head_w = nullptr;    // [hidden, vocab] separate from embd
};

// One decoder block. q/k/v/o are bias-less; norms are vanilla LN.
struct MoonshineStreamingDecBlock {
  // Self-attention.
  ggml_tensor *norm_self_w = nullptr;
  ggml_tensor *self_q_w = nullptr;
  ggml_tensor *self_k_w = nullptr;
  ggml_tensor *self_v_w = nullptr;
  ggml_tensor *self_out_w = nullptr;

  // Cross-attention (no RoPE).
  ggml_tensor *norm_cross_w = nullptr;
  ggml_tensor *cross_q_w = nullptr;
  ggml_tensor *cross_k_w = nullptr;
  ggml_tensor *cross_v_w = nullptr;
  ggml_tensor *cross_out_w = nullptr;

  // SwiGLU MLP (with biases).
  ggml_tensor *norm_ffn_w = nullptr;
  ggml_tensor *ffn_fc1_w = nullptr; // [hidden, 2*ffn_dim]
  ggml_tensor *ffn_fc1_b = nullptr; // [2*ffn_dim]
  ggml_tensor *ffn_fc2_w = nullptr; // [ffn_dim, hidden]
  ggml_tensor *ffn_fc2_b = nullptr; // [hidden]
};

struct MoonshineStreamingWeights {
  MoonshineStreamingEmbedder embedder;
  MoonshineStreamingEncTop enc_top;
  std::vector<MoonshineStreamingEncBlock> enc_blocks;
  MoonshineStreamingAdapter adapter;
  MoonshineStreamingDecTop dec_top;
  std::vector<MoonshineStreamingDecBlock> dec_blocks;
};

// ---------------------------------------------------------------------------
// KV cache: dual slab layout shared with the arch implementation -
//   self_k/self_v   [d_model, n_ctx] per layer (grows per decode step)
//   cross_k/cross_v [d_model, T_enc] per layer (committed from host)
// ---------------------------------------------------------------------------

struct MoonshineStreamingKvCache {
  ggml_tensor *self_k = nullptr;
  ggml_tensor *self_v = nullptr;
  ggml_tensor *cross_k = nullptr;
  ggml_tensor *cross_v = nullptr;

  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;

  int n_ctx = 0;
  int n = 0;
  int head = 0;
  int T_enc = 0;

  bool cross_populated = false;

  void free();
};

bool kv_cache_init(MoonshineStreamingKvCache &cache, ggml_backend_t backend,
                   int n_ctx, int T_enc, int d_model, int n_layer,
                   ggml_type kv_type);

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

// Compute the encoder output frame count for a given input sample count:
//   T_frames = n_samples / frame_len
//   T1       = ceil(T_frames / 2)   (causal conv stride 2 keeps every frame
//                                    thanks to the left pad of k-1)
//   T_enc    = ceil(T1 / 2)
// Returns 0 if the input is too short.
int encoder_t_enc(const MoonshineStreamingHParams &hp, int n_samples);

// Build a host-side sliding-window mask of shape [T_enc, T_enc] for the
// (L, R) window. Allowed positions are 0.0; blocked are -INF.
//
// For row q (0..T_enc), col k is allowed iff:
//     (q-k >= 0 && q-k < L)   // up to L-1 positions back, including self
//   || (k-q >= 1 && k-q < R)  // up to R-1 positions ahead
//
// The caller-provided buffer must hold at least T_enc*T_enc floats.
void build_sliding_window_mask(int T_enc, int left_window, int right_window,
                               float *out_mask);

struct EncoderBuild {
  // Raw PCM [n_samples] f32; uploaded by the caller after alloc.
  ggml_tensor *audio_in = nullptr;

  // Per-layer sliding-window attention masks, f32 [T_enc, T_enc] (n_kv, n_q).
  std::vector<ggml_tensor *> per_layer_masks;

  // Final encoder hidden state [enc_d_model, T_enc] f32.
  ggml_tensor *out = nullptr;

  ggml_cgraph *graph = nullptr;
  int T_enc = 0;
};

EncoderBuild build_encoder_graph(ggml_context *ctx,
                                 const MoonshineStreamingWeights &w,
                                 const MoonshineStreamingHParams &hp,
                                 int n_samples, bool use_flash);

// ---------------------------------------------------------------------------
// Adapter / cross-KV / decoder
// ---------------------------------------------------------------------------

// x = encoder_out + pos_emb[abs_frame_offset .. abs_frame_offset + T)
// x = proj(x) when adapter_has_proj
//
// Positions are ABSOLUTE frame indices, which is what makes a per-window
// slice concatenate to the one-shot result.
struct AdapterBuild {
  ggml_tensor *encoder_out_in = nullptr; // [enc_hidden, T] f32
  ggml_tensor *pos_ids_in = nullptr;     // [T] i32 (absolute frame indices)
  ggml_tensor *out = nullptr;            // [dec_hidden, T]
  ggml_cgraph *graph = nullptr;
};

AdapterBuild build_adapter_graph(ggml_context *ctx,
                                 const MoonshineStreamingWeights &w,
                                 const MoonshineStreamingHParams &hp, int T);

// Per-layer cross-attn K/V projection over an adapter slice, left readable to
// host. Does NOT touch the persistent KV cache: the caller accumulates K/V
// into host buffers across feeds and pushes them into the cache via
// build_cross_kv_commit_graph. This keeps per-feed work bounded and decouples
// cache allocation (which needs the final T_enc) from per-feed computation.
struct CrossKVProjectionBuild {
  ggml_tensor *encoder_out_in = nullptr; // [dec_d_model, n_frames] f32
  std::vector<ggml_tensor *> per_layer_k; // [dec_d_model, n_frames] f32 each
  std::vector<ggml_tensor *> per_layer_v;
  ggml_cgraph *graph = nullptr;
};

CrossKVProjectionBuild
build_cross_kv_projection_graph(ggml_context *ctx,
                                const MoonshineStreamingWeights &w,
                                const MoonshineStreamingHParams &hp,
                                int n_frames);

// Upload per-layer host K/V buffers into kv_cache.cross_k / cross_v via
// ggml_cpy, letting the backend handle any F32->F16 conversion the cache's
// storage dtype requires.
struct CrossKVCommitBuild {
  std::vector<ggml_tensor *> per_layer_k_in; // [dec_d_model, T_enc] f32 each
  std::vector<ggml_tensor *> per_layer_v_in;
  ggml_cgraph *graph = nullptr;
};

CrossKVCommitBuild
build_cross_kv_commit_graph(ggml_context *ctx,
                            const MoonshineStreamingHParams &hp,
                            MoonshineStreamingKvCache &kv_cache, int T_enc);

struct DecoderBuild {
  ggml_tensor *token_ids_in = nullptr;   // [n_tokens] i32
  ggml_tensor *pos_ids_in = nullptr;     // [n_tokens] i32
  ggml_tensor *causal_mask_in = nullptr; // [n_kv, n_tokens] f32 (n_tokens > 1)

  ggml_tensor *out = nullptr;        // logits (or log-softmax)
  ggml_tensor *argmax_out = nullptr; // [n_tokens] i32 when skip_log_softmax

  ggml_cgraph *graph = nullptr;
};

DecoderBuild build_decoder_graph_kv(ggml_context *ctx,
                                    const MoonshineStreamingWeights &w,
                                    const MoonshineStreamingHParams &hp,
                                    MoonshineStreamingKvCache &kv_cache,
                                    int n_tokens, int n_past, int T_enc,
                                    bool skip_log_softmax, bool use_flash);

// Search a compute context for a tensor by name (used to locate named graph
// inputs such as "dec.causal_mask" before uploading host data).
ggml_tensor *find_tensor_by_name(ggml_context *gctx, const char *name);

} // namespace engine::models::moonshine_streaming
