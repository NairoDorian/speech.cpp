// engine/models/moonshine/graphs.h - native engine Moonshine ASR graph
// layer: hparams, weight slots, KV cache, and ggml graph builders.
//
// Phase 11 W1 port of src/runtime/arch/moonshine/ into the engine framework.
// The numerics mirror the arch implementation exactly (same graph topology,
// same partial-RoPE handling, same head-dim padding policy); only the
// surrounding machinery differs (engine BackendWeightStore / TokenizerHub /
// RunControl instead of transcribe_model / transcribe::Tokenizer /
// poll_abort).
//
// Architecture (see docs/porting/families/moonshine.md):
//   - Encoder: 3-conv stem on raw 16 kHz PCM (k=127 s=64 -> tanh ->
//     GroupNorm(1); k=7 s=3 -> gelu_erf; k=3 s=2 -> gelu_erf), then
//     pre-LN transformer blocks (bidirectional MHSA with partial RoPE 0.9
//     + GELU(erf) MLP), final bias-less LN.
//   - Decoder: token embedding tied to lm_head, pre-LN blocks (causal
//     self-attn with partial RoPE on q/k; cross-attn with no RoPE;
//     SwiGLU MLP), final bias-less LN, tied logits head.
//   - All attention projections and LayerNorms are bias-less.

#pragma once

// Internal graph-layer header for the native Moonshine engine package.
// Not part of any installed public API surface.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;

namespace engine::models::moonshine {

// ---------------------------------------------------------------------------
// Hyperparameters (read from GGUF metadata keys stt.moonshine.*)
// ---------------------------------------------------------------------------

struct MoonshineHParams {
  // Encoder.
  int32_t enc_n_layers = 0;
  int32_t enc_d_model = 0;
  int32_t enc_n_heads = 0;
  int32_t enc_n_kv_heads = 0;
  int32_t enc_ffn_dim = 0;
  std::string enc_activation = "gelu";

  // Decoder.
  int32_t dec_n_layers = 0;
  int32_t dec_d_model = 0;
  int32_t dec_n_heads = 0;
  int32_t dec_n_kv_heads = 0;
  int32_t dec_ffn_dim = 0;
  std::string dec_activation = "silu";
  int32_t dec_vocab_size = 0;
  int32_t dec_max_position_embeddings = 0;
  bool dec_tie_word_embeddings = true;

  // Special tokens.
  int32_t bos_token_id = -1;
  int32_t eos_token_id = -1;
  int32_t pad_token_id = -1;
  int32_t decoder_start_token_id = -1;

  // Attention / RoPE.
  float partial_rotary_factor = 0.9f;
  float rope_theta = 10000.0f;
  bool attention_bias = false;
  int32_t pad_head_dim_multiple = 0;

  // Conv stem (3-layer raw-PCM frontend).
  std::vector<int32_t> conv_channels;
  std::vector<int32_t> conv_kernel_sizes;
  std::vector<int32_t> conv_strides;
  int32_t conv_groupnorm_num_groups = 1;
  float conv_groupnorm_eps = 1e-5f;

  // Frontend (raw waveform - no mel).
  std::string fe_type = "raw";
  int32_t fe_sample_rate = 16000;

  int32_t enc_head_dim() const {
    return enc_n_heads > 0 ? enc_d_model / enc_n_heads : 0;
  }
  int32_t dec_head_dim() const {
    return dec_n_heads > 0 ? dec_d_model / dec_n_heads : 0;
  }

  int32_t padded_head_dim(int32_t head_dim) const {
    const int32_t m = pad_head_dim_multiple;
    if (m <= 0 || head_dim <= 0) {
      return head_dim;
    }
    return ((head_dim + m - 1) / m) * m;
  }

  int32_t enc_head_dim_padded() const {
    return padded_head_dim(enc_head_dim());
  }
  int32_t dec_head_dim_padded() const {
    return padded_head_dim(dec_head_dim());
  }

  // Number of leading head-dim positions rotated by partial RoPE
  // (floor to even so the interleaved rotate halves match HF).
  int32_t rotated_head_dim(int32_t head_dim) const {
    const int32_t r = static_cast<int32_t>(static_cast<float>(head_dim) *
                                           partial_rotary_factor);
    return r & ~int32_t{1};
  }

  int32_t enc_head_dim_rot() const { return rotated_head_dim(enc_head_dim()); }
  int32_t dec_head_dim_rot() const { return rotated_head_dim(dec_head_dim()); }
};

// ---------------------------------------------------------------------------
// Weight slots
// ---------------------------------------------------------------------------

struct MoonshineEncStem {
  ggml_tensor *conv0_w = nullptr; // [k0, 1, d_model]
  ggml_tensor *conv1_w = nullptr; // [k1, d_model, 2*d_model]
  ggml_tensor *conv1_b = nullptr; // [2*d_model]
  ggml_tensor *conv2_w = nullptr; // [k2, 2*d_model, d_model]
  ggml_tensor *conv2_b = nullptr; // [d_model]
  ggml_tensor *gn_w = nullptr;    // [d_model]
  ggml_tensor *gn_b = nullptr;    // [d_model]
};

struct MoonshineEncTop {
  ggml_tensor *final_norm_w = nullptr; // [d_model]
};

struct MoonshineEncBlock {
  ggml_tensor *norm_attn_w = nullptr;
  ggml_tensor *attn_q_w = nullptr;
  ggml_tensor *attn_k_w = nullptr;
  ggml_tensor *attn_v_w = nullptr;
  ggml_tensor *attn_out_w = nullptr;

  ggml_tensor *norm_ffn_w = nullptr;
  ggml_tensor *ffn_fc1_w = nullptr; // [d_model, ffn_dim]
  ggml_tensor *ffn_fc1_b = nullptr; // [ffn_dim]
  ggml_tensor *ffn_fc2_w = nullptr; // [ffn_dim, d_model]
  ggml_tensor *ffn_fc2_b = nullptr; // [d_model]
};

struct MoonshineDecTop {
  ggml_tensor *token_embd_w = nullptr; // [d_model, vocab_size] (tied lm_head)
  ggml_tensor *final_norm_w = nullptr; // [d_model]
};

struct MoonshineDecBlock {
  ggml_tensor *norm_self_w = nullptr;
  ggml_tensor *self_q_w = nullptr;
  ggml_tensor *self_k_w = nullptr;
  ggml_tensor *self_v_w = nullptr;
  ggml_tensor *self_out_w = nullptr;

  ggml_tensor *norm_cross_w = nullptr;
  ggml_tensor *cross_q_w = nullptr;
  ggml_tensor *cross_k_w = nullptr;
  ggml_tensor *cross_v_w = nullptr;
  ggml_tensor *cross_out_w = nullptr;

  ggml_tensor *norm_ffn_w = nullptr;
  ggml_tensor *ffn_fc1_w = nullptr; // [d_model, 2*ffn_dim]
  ggml_tensor *ffn_fc1_b = nullptr; // [2*ffn_dim]
  ggml_tensor *ffn_fc2_w = nullptr; // [ffn_dim, d_model]
  ggml_tensor *ffn_fc2_b = nullptr; // [d_model]
};

struct MoonshineWeights {
  MoonshineEncStem enc_stem;
  MoonshineEncTop enc_top;
  std::vector<MoonshineEncBlock> enc_blocks;
  MoonshineDecTop dec_top;
  std::vector<MoonshineDecBlock> dec_blocks;
};

// ---------------------------------------------------------------------------
// KV cache: dual slab layout shared with the arch implementation -
//   self_k/self_v   [d_model, n_ctx] per layer (grows per decode step)
//   cross_k/cross_v [d_model, T_enc] per layer (precomputed once)
// ---------------------------------------------------------------------------

struct MoonshineKvCache {
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

bool kv_cache_init(MoonshineKvCache &cache, ggml_backend_t backend, int n_ctx,
                   int T_enc, int d_model, int n_layer, ggml_type kv_type);

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

int encoder_t_enc(const MoonshineHParams &hp, int n_samples);

struct EncoderBuild {
  ggml_tensor *audio_in = nullptr;   // [n_samples] f32
  ggml_tensor *pos_ids_in = nullptr; // [T_enc] i32
  ggml_tensor *out = nullptr;        // [d_model, T_enc]
  int T_enc = 0;
  ggml_cgraph *graph = nullptr;
};

EncoderBuild build_encoder_graph(ggml_context *ctx, const MoonshineWeights &w,
                                 const MoonshineHParams &hp, int n_samples,
                                 bool use_flash);

struct DecoderBuild {
  ggml_tensor *token_ids_in = nullptr; // [n_tokens] i32
  ggml_tensor *pos_ids_in = nullptr;   // [n_tokens] i32
  ggml_tensor *encoder_out_in =
      nullptr; // [d_model, T_enc] f32 (cross_kv graph)
  ggml_tensor *causal_mask_in = nullptr; // [n_kv, n_tokens] f32 (n_tokens > 1)

  ggml_tensor *out = nullptr;        // logits (or log-softmax on prompt pass)
  ggml_tensor *argmax_out = nullptr; // [n_tokens] i32 when skip_log_softmax

  ggml_cgraph *graph = nullptr;
};

DecoderBuild build_cross_kv_graph(ggml_context *ctx, const MoonshineWeights &w,
                                  const MoonshineHParams &hp,
                                  MoonshineKvCache &kv_cache, int T_enc);

DecoderBuild build_decoder_graph_kv(ggml_context *ctx,
                                    const MoonshineWeights &w,
                                    const MoonshineHParams &hp,
                                    MoonshineKvCache &kv_cache, int n_tokens,
                                    int n_past, int T_enc,
                                    bool skip_log_softmax, bool use_flash);

// Search a compute context for a tensor by name (used to locate named
// graph inputs such as "dec.causal_mask" before uploading host data).
ggml_tensor *find_tensor_by_name(ggml_context *gctx, const char *name);

} // namespace engine::models::moonshine
