// engine/models/whisper/graphs_internal.h - native engine Whisper ASR graph
// layer: hparams, weight slots, KV cache, and ggml graph builders.
//
// Phase 11 Wave W2a port of src/runtime/arch/whisper/ into the engine
// framework. The numerics mirror the arch implementation exactly; only the
// surrounding machinery differs (engine BackendWeightStore / MelExtractor /
// TokenizerHub / RunControl instead of transcribe_model / MelFrontend /
// transcribe::Tokenizer / poll_abort).
//
// Architecture:
//   - Frontend: 80/128-bin log-mel, Whisper normalization (log10, global clamp
//     to max-8, (x+4)/4), 30 s / 3000-frame window. Delivered by the Phase-9
//     engine::audio::MelExtractor, not a private implementation.
//   - Encoder: 2-layer Conv1d stem (k=3 s=1 p=1 -> GELU(erf); k=3 s=2 p=1 ->
//     GELU(erf)), learned absolute positional embedding, pre-LN transformer
//     blocks, final LayerNorm.
//   - Decoder: token embedding + learned positional embedding (NO post-embed
//     LayerNorm), pre-LN blocks (causal self-attn, cross-attn, GELU FFN),
//     final LayerNorm, logits head TIED to the token embedding (no bias).
//
// Whisper attention quirk, carried over verbatim: q / v / out projections
// carry a bias, k does NOT — self-attention and cross-attention alike.
//
// W2a scope: offline greedy decode over the legacy whisper.cpp `.bin` weights.
// The temperature-fallback ladder, DecodeTelemetry, timestamps, long-form seek
// continuation and the batched decoder are W2b.

#pragma once

// Internal graph-layer header for the native Whisper engine package. Not part
// of any installed public API surface.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;

namespace engine::models::whisper {

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct WhisperHParams {
  // Encoder.
  int32_t enc_n_layers = 0;
  int32_t enc_d_model = 0;
  int32_t enc_n_heads = 0;
  int32_t enc_ffn_dim = 0;
  int32_t enc_num_mel_bins = 0;
  int32_t enc_max_source_positions = 0; // 1500 across variants
  std::string enc_activation = "gelu";

  // Decoder.
  int32_t dec_n_layers = 0;
  int32_t dec_d_model = 0;
  int32_t dec_n_heads = 0;
  int32_t dec_ffn_dim = 0;
  int32_t dec_max_target_positions = 0;
  int32_t dec_vocab_size = 0;
  std::string dec_activation = "gelu";
  bool dec_tie_word_embeddings = true;
  bool dec_scale_embedding = false;

  // Whisper generation contract.
  int32_t decoder_start_token_id = -1; // <|startoftranscript|>
  int32_t eot_token_id = -1;           // <|endoftext|>
  int32_t no_timestamps_token_id = -1; // <|notimestamps|>
  int32_t transcribe_token_id = -1;    // <|transcribe|>
  int32_t translate_token_id = -1;     // <|translate|>
  int32_t prev_sot_token_id = -1;      // <|startofprev|>
  int32_t no_speech_token_id = -1;     // <|nospeech|>
  int32_t first_language_token_id = -1;
  int32_t n_languages = 0;
  bool is_multilingual = false;

  // Suppression lists (may be empty for .en variants).
  std::vector<int32_t> suppress_tokens;       // applied every step
  std::vector<int32_t> begin_suppress_tokens; // first generated step only

  // Frontend.
  std::string fe_type = "log_mel";
  int32_t fe_num_mels = 0;
  int32_t fe_sample_rate = 16000;
  int32_t fe_n_fft = 400;
  int32_t fe_win_length = 400;
  int32_t fe_hop_length = 160;
  int32_t fe_chunk_length = 30;    // seconds
  int32_t fe_n_samples = 480000;   // 30 s at 16 kHz
  int32_t fe_nb_max_frames = 3000; // n_samples / hop_length

  int32_t enc_head_dim() const {
    return enc_n_heads > 0 ? enc_d_model / enc_n_heads : 0;
  }
  int32_t dec_head_dim() const {
    return dec_n_heads > 0 ? dec_d_model / dec_n_heads : 0;
  }
};

// ---------------------------------------------------------------------------
// Weight slots
// ---------------------------------------------------------------------------

// Encoder conv stem: two 1D convolutions, kernel=3, strides {1, 2}.
struct WhisperEncStem {
  ggml_tensor *conv0_w = nullptr; // [d_model, num_mel_bins, 3]
  ggml_tensor *conv0_b = nullptr; // [d_model]
  ggml_tensor *conv1_w = nullptr; // [d_model, d_model, 3]
  ggml_tensor *conv1_b = nullptr; // [d_model]
};

// Learned positional embedding + final LayerNorm (LN carries a bias here,
// unlike the moonshine families).
struct WhisperEncTop {
  ggml_tensor *pos_emb_w = nullptr; // [d_model, max_source_positions]
  ggml_tensor *final_norm_w = nullptr;
  ggml_tensor *final_norm_b = nullptr;
};

struct WhisperEncBlock {
  ggml_tensor *norm_attn_w = nullptr;
  ggml_tensor *norm_attn_b = nullptr;
  ggml_tensor *attn_q_w = nullptr;
  ggml_tensor *attn_q_b = nullptr;
  ggml_tensor *attn_k_w = nullptr; // NO bias
  ggml_tensor *attn_v_w = nullptr;
  ggml_tensor *attn_v_b = nullptr;
  ggml_tensor *attn_out_w = nullptr;
  ggml_tensor *attn_out_b = nullptr;
  ggml_tensor *norm_ffn_w = nullptr;
  ggml_tensor *norm_ffn_b = nullptr;
  ggml_tensor *ffn_fc1_w = nullptr;
  ggml_tensor *ffn_fc1_b = nullptr;
  ggml_tensor *ffn_fc2_w = nullptr;
  ggml_tensor *ffn_fc2_b = nullptr;
};

// Decoder token + position embedding and final LN. token_embd doubles as the
// (tied) lm_head weight.
struct WhisperDecTop {
  ggml_tensor *token_embd_w = nullptr; // [d_model, vocab_size]
  ggml_tensor *pos_emb_w = nullptr;    // [d_model, max_target_positions]
  ggml_tensor *final_norm_w = nullptr;
  ggml_tensor *final_norm_b = nullptr;
};

struct WhisperDecBlock {
  ggml_tensor *norm_self_w = nullptr;
  ggml_tensor *norm_self_b = nullptr;
  ggml_tensor *self_q_w = nullptr;
  ggml_tensor *self_q_b = nullptr;
  ggml_tensor *self_k_w = nullptr; // NO bias
  ggml_tensor *self_v_w = nullptr;
  ggml_tensor *self_v_b = nullptr;
  ggml_tensor *self_out_w = nullptr;
  ggml_tensor *self_out_b = nullptr;

  ggml_tensor *norm_cross_w = nullptr;
  ggml_tensor *norm_cross_b = nullptr;
  ggml_tensor *cross_q_w = nullptr;
  ggml_tensor *cross_q_b = nullptr;
  ggml_tensor *cross_k_w = nullptr; // NO bias
  ggml_tensor *cross_v_w = nullptr;
  ggml_tensor *cross_v_b = nullptr;
  ggml_tensor *cross_out_w = nullptr;
  ggml_tensor *cross_out_b = nullptr;

  ggml_tensor *norm_ffn_w = nullptr;
  ggml_tensor *norm_ffn_b = nullptr;
  ggml_tensor *ffn_fc1_w = nullptr;
  ggml_tensor *ffn_fc1_b = nullptr;
  ggml_tensor *ffn_fc2_w = nullptr;
  ggml_tensor *ffn_fc2_b = nullptr;
};

struct WhisperWeights {
  WhisperEncStem enc_stem;
  WhisperEncTop enc_top;
  std::vector<WhisperEncBlock> enc_blocks;
  WhisperDecTop dec_top;
  std::vector<WhisperDecBlock> dec_blocks;
};

// ---------------------------------------------------------------------------
// KV cache: self over [d_model, n_ctx] per layer (grows per decode step),
// cross over [d_model, T_enc] per layer (precomputed once per window).
// ---------------------------------------------------------------------------

struct WhisperKvCache {
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

bool kv_cache_init(WhisperKvCache &cache, ggml_backend_t backend, int n_ctx,
                   int T_enc, int d_model, int n_layer, ggml_type kv_type);

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

struct EncoderBuild {
  ggml_tensor *mel_in = nullptr; // [n_mels, n_mel_frames] f32
  ggml_tensor *out = nullptr;    // [d_model, T_enc] f32
  int T_enc = 0;
  ggml_cgraph *graph = nullptr;
};

// n_mel_frames must be positive and even (stride-2 conv2 would otherwise
// produce a fractional T_enc).
EncoderBuild build_encoder_graph(ggml_context *ctx, const WhisperWeights &w,
                                 const WhisperHParams &hp, int n_mel_frames,
                                 bool use_flash);

struct CrossKvBuild {
  ggml_tensor *encoder_out_in = nullptr; // [d_model, T_enc] f32
  ggml_cgraph *graph = nullptr;
};

CrossKvBuild build_cross_kv_graph(ggml_context *ctx, const WhisperWeights &w,
                                  const WhisperHParams &hp,
                                  WhisperKvCache &kv_cache, int T_enc);

struct DecoderBuild {
  ggml_tensor *token_ids_in = nullptr;   // [n_tokens] i32
  ggml_tensor *causal_mask_in = nullptr; // [n_kv, n_tokens] f32 (n_tokens > 1)

  // Raw pre-softmax logits [vocab, n_tokens]. Whisper's host loop needs full
  // logits for the suppress masks, so unlike the moonshine families there is
  // no in-graph argmax shortcut.
  ggml_tensor *logits_out = nullptr;

  ggml_cgraph *graph = nullptr;
};

// Handles both the prompt pass (n_tokens > 1, n_past = 0) and the per-step
// pass (n_tokens = 1, n_past = current). Positions are n_past + i.
DecoderBuild build_decoder_graph_kv(ggml_context *ctx, const WhisperWeights &w,
                                    const WhisperHParams &hp,
                                    WhisperKvCache &kv_cache, int n_tokens,
                                    int n_past, int T_enc, bool use_flash);

ggml_tensor *find_tensor_by_name(ggml_context *gctx, const char *name);

} // namespace engine::models::whisper
