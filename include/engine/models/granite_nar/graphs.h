#pragma once

// engine/models/granite_nar/graphs.h - weight slots and ggml graph builders
// for the native engine Granite Speech NAR package.
//
// Port of src/runtime/arch/granite_nar/{encoder,projector,decoder}.cpp plus
// the shared runtime helpers they call (src/runtime/granite_conformer/
// shaw_attn.cpp, src/runtime/conformer/conformer.cpp: macaron_ff_residual,
// rel_shift, conv_1d_dw_f32, fused_batch_norm, resolve_conv_direct), carried
// into this package so it depends on neither src/runtime nor another model
// package. Op order is the arch's, at the parent's CURRENT revision:
//
//   encoder   mel [160, T_enc] -> input_linear -> 16 x conformer block
//             { macaron FF1 -> Shaw block-local MHSA (SKEW bias, 585b98f7)
//               -> GLU conv (direct depthwise) + fused BN + SiLU -> FF2
//               -> post LN }, char-CTC self-conditioning at layer N/2
//             (mid_blank_probs = softmax(ctc)[0]), captures concatenated
//             -> cat_out [n_capt * 1024, T_enc]
//   bpe_ctc   pooled final-layer states [1024, n_windows] -> ctc_bpe linear
//             -> argmax per window (b174a427: bounded chunks, no full
//             [vocab, T_enc] logits)
//   projector per-capture LN -> layer_projector + GELU -> window pad ->
//             (+window_positions) K/V, mean-pool + query Q -> 2 x pre-LN
//             {cross-attn, SiLU MLP} -> out_norm -> out_linear
//   editor    [audio / emb_mul | embed(text slots)] * emb_mul -> 40 x
//             bidirectional Granite-4 block -> RMSNorm -> tied lm_head over
//             the text slots / logits_scaling
//
// Not carried: the transcribe tensor-dump taps (debug only; they never feed
// an output).

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_cgraph;

namespace engine::models::granite_nar {

struct GraniteNarHParams;

struct EncTopWeights {
  ggml_tensor *input_linear_w = nullptr; // ne [input_dim, hidden]
  ggml_tensor *input_linear_b = nullptr;
  ggml_tensor *ctc_proj_w = nullptr;     // ne [hidden, output_dim]
  ggml_tensor *ctc_proj_b = nullptr;
  ggml_tensor *ctc_bypass_w = nullptr;   // ne [output_dim, hidden]
  ggml_tensor *ctc_bypass_b = nullptr;
  ggml_tensor *ctc_bpe_w = nullptr;      // ne [hidden, bpe_output_dim] (null when dim 0)
  ggml_tensor *ctc_bpe_b = nullptr;
};

struct EncBlockWeights {
  ggml_tensor *norm_ff1_w = nullptr, *norm_ff1_b = nullptr;
  ggml_tensor *ff1_up_w = nullptr, *ff1_up_b = nullptr;
  ggml_tensor *ff1_down_w = nullptr, *ff1_down_b = nullptr;
  ggml_tensor *norm_attn_w = nullptr, *norm_attn_b = nullptr;
  ggml_tensor *attn_q_w = nullptr;         // ne [hidden, inner]
  ggml_tensor *attn_kv_w = nullptr;        // ne [hidden, 2 * inner] (K|V)
  ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
  ggml_tensor *attn_rel_pos_emb = nullptr; // ne [head_dim, 2 * max_pos_emb + 1]
  ggml_tensor *norm_conv_w = nullptr, *norm_conv_b = nullptr;
  ggml_tensor *conv_pointwise1_w = nullptr, *conv_pointwise1_b = nullptr; // ne [1, hidden, 2 * inner_c]
  ggml_tensor *conv_depthwise_w = nullptr;                                // ne [k, 1, inner_c]
  ggml_tensor *conv_bn_fused_scale = nullptr, *conv_bn_fused_bias = nullptr; // [inner_c], host-folded
  ggml_tensor *conv_pointwise2_w = nullptr, *conv_pointwise2_b = nullptr; // ne [1, inner_c, hidden]
  ggml_tensor *norm_ff2_w = nullptr, *norm_ff2_b = nullptr;
  ggml_tensor *ff2_up_w = nullptr, *ff2_up_b = nullptr;
  ggml_tensor *ff2_down_w = nullptr, *ff2_down_b = nullptr;
  ggml_tensor *norm_post_w = nullptr, *norm_post_b = nullptr;
};

struct ProjTopWeights {
  std::vector<ggml_tensor *> layer_norms_w; // one per capture, [encoder_dim]
  std::vector<ggml_tensor *> layer_norms_b;
  ggml_tensor *layer_projector_w = nullptr, *layer_projector_b = nullptr;
  ggml_tensor *out_norm_w = nullptr, *out_norm_b = nullptr;
  ggml_tensor *out_linear_w = nullptr, *out_linear_b = nullptr;
  ggml_tensor *query = nullptr;            // ne [hidden, n_query, 1] (F32/BF16/F16)
  ggml_tensor *window_positions = nullptr; // ne [hidden, block_size, 1]
};

struct ProjBlockWeights {
  ggml_tensor *norm_attn_w = nullptr, *norm_attn_b = nullptr;
  ggml_tensor *cross_attn_q_w = nullptr, *cross_attn_q_b = nullptr;
  ggml_tensor *cross_attn_k_w = nullptr, *cross_attn_k_b = nullptr;
  ggml_tensor *cross_attn_v_w = nullptr, *cross_attn_v_b = nullptr;
  ggml_tensor *cross_attn_o_w = nullptr, *cross_attn_o_b = nullptr;
  ggml_tensor *norm_ffn_w = nullptr, *norm_ffn_b = nullptr;
  ggml_tensor *ffn_fc1_w = nullptr, *ffn_fc1_b = nullptr;
  ggml_tensor *ffn_fc2_w = nullptr, *ffn_fc2_b = nullptr;
};

struct DecBlockWeights {
  ggml_tensor *norm_attn_w = nullptr;
  ggml_tensor *norm_ffn_w = nullptr;
  ggml_tensor *attn_q_w = nullptr, *attn_k_w = nullptr, *attn_v_w = nullptr, *attn_o_w = nullptr;
  ggml_tensor *ffn_gate_w = nullptr, *ffn_up_w = nullptr, *ffn_down_w = nullptr;
};

struct GraniteNarWeights {
  EncTopWeights enc_top;
  std::vector<EncBlockWeights> enc_blocks;
  ProjTopWeights proj_top;
  std::vector<ProjBlockWeights> proj_blocks;
  ggml_tensor *dec_token_embd = nullptr; // ne [hidden, vocab]; tied lm_head
  std::vector<DecBlockWeights> dec_blocks;
  ggml_tensor *dec_output_norm = nullptr;
};

// How the Shaw positional bias is formed. Same math either way.
enum class ShawBias {
  // transcribe.cpp 585b98f7 (the parent's current code): get_rows the
  // 2*ctx - 1 distinct offsets, one mul_mat against q, rel_shift into kq's
  // layout. Default.
  Skew,
  // speech.cpp's still-unadopted arch (pre-585b98f7): the full
  // [head_dim, ctx, ctx] lookup and a batched mul_mat. Diagnostic only
  // (session option granite_nar.shaw_bias=direct).
  Direct,
};

// transcribe::env::flag: set, non-empty, first character not '0'.
bool env_flag(const char *name);

// arch encoder.cpp detect_conv_dw_direct: direct depthwise on every backend
// by default; TRANSCRIBE_CONV_DIRECT_DW forces it, TRANSCRIBE_CONV_NO_DIRECT_DW
// falls back to the im2col depthwise.
bool detect_conv_dw_direct();

// Node budget shared by every graph here (the arch's ggml_new_graph_custom
// sizes were 16384 / 4096 / 16384 / 1024; one budget covers them all).
constexpr size_t kGraphNodes = 16384;
// Bytes a no_alloc ggml context needs for one graph's tensor + graph metadata.
size_t graph_context_bytes();

struct EncoderBuild {
  ggml_tensor *mel_in = nullptr;          // [input_dim, T_enc] f32
  ggml_tensor *pos_rows = nullptr;        // [2 * ctx - 1] i32 (Skew)
  ggml_tensor *attention_dists = nullptr; // [ctx * ctx] i32 (Direct)
  ggml_tensor *last_block_mask = nullptr; // [ctx, ctx, n_blocks] f32
  ggml_tensor *zero_pad = nullptr;        // [hidden, T_pad - T_enc] f32, or null
  ggml_tensor *cat_out = nullptr;         // [n_capt * hidden, T_enc] (output)
  ggml_tensor *mid_blank_probs = nullptr; // [T_enc] softmax(mid ctc)[blank] (output)
  ggml_cgraph *graph = nullptr;
  int n_blocks_local = 0;
  int last_block_rem = 0;
  int64_t final_capture_offset = -1;      // channel offset of the last layer in cat_out
};

// Returns a build whose graph is null on a shape error (missing capture, the
// final layer not captured - the parent's requirement for the BPE head).
EncoderBuild build_encoder_graph(ggml_context *ctx, const GraniteNarWeights &w,
                                 const GraniteNarHParams &hp, int T_enc, ShawBias shaw_bias,
                                 bool conv_dw_direct);

struct BpeCtcBuild {
  ggml_tensor *hidden_in = nullptr; // [enc_hidden, n_windows] f32
  ggml_tensor *token_ids = nullptr; // [n_windows] i32 (output)
  ggml_cgraph *graph = nullptr;
  int n_windows = 0;
};

// transcribe.cpp HEAD encoder.cpp build_bpe_ctc_graph.
BpeCtcBuild build_bpe_ctc_graph(ggml_context *ctx, const GraniteNarWeights &w,
                                const GraniteNarHParams &hp, int n_windows);

struct ProjectorBuild {
  ggml_tensor *enc_in = nullptr;  // [n_capt * enc_hidden, T_enc] f32
  ggml_tensor *enc_pad = nullptr; // [prj_hidden, T_pad - T_enc] f32 zeros, or null
  ggml_tensor *out = nullptr;     // [llm_dim, nblocks * n_query] (output)
  ggml_cgraph *graph = nullptr;
  int nblocks = 0;
  int n_audio_tokens = 0;         // nblocks * n_query (before the T_enc / ds clip)
};

// arch projector.cpp build_projector_graph. Null graph on a shape error.
ProjectorBuild build_projector_graph(ggml_context *ctx, const GraniteNarWeights &w,
                                     const GraniteNarHParams &hp, int T_enc);

struct ForwardBuild {
  ggml_tensor *audio_in = nullptr;     // [hidden, n_audio] f32 (already / emb_mul)
  ggml_tensor *text_ids_in = nullptr;  // [n_text] i32
  ggml_tensor *positions_in = nullptr; // [T_total] i32
  ggml_tensor *out = nullptr;          // [vocab, n_text] (output)
  ggml_cgraph *graph = nullptr;
  int n_audio_tokens = 0;
  int n_text = 0;
  int T_total = 0;
};

// arch decoder.cpp build_forward_graph. Null graph when n_audio or n_text <= 0.
ForwardBuild build_forward_graph(ggml_context *ctx, const GraniteNarWeights &w,
                                 const GraniteNarHParams &hp, int n_audio_tokens, int n_text);

} // namespace engine::models::granite_nar
