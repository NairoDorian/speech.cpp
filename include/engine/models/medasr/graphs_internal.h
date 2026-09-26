#pragma once

// engine/models/medasr/graphs_internal.h - weight slots and the ggml encoder
// graph builder for the native engine MedASR (Google LASR-CTC) package.
//
// Port of src/runtime/arch/medasr/encoder.{h,cpp} plus the pieces of the
// shared src/runtime/conformer helpers it calls (conv_1d_f32, conv_module on
// its BatchNorm / direct-depthwise / asymmetric-pad path, fused_batch_norm,
// detect_direct_pw). Graph topology and op order are identical to the arch:
//
//   mel [n_mels, T_mel, 1, B]
//   -> subsampling: dense_0 + ReLU -> Conv1d(k=5, s=2) + ReLU
//                   -> Conv1d(k=5, s=2) + ReLU -> dense_1        [d, T_enc, B]
//   -> N x block { LN -> FF1 (SiLU) -> 1.5 x + 0.5 y
//                  LN -> RoPE MHSA (NEOX, theta from GGUF) -> x + y
//                  LN -> pw1 -> GLU -> [pad mask] -> dw k=32 pad (15,16)
//                        -> fused BN -> SiLU -> pw2 -> 2.0 x + 1.0 y
//                  LN -> FF2 (SiLU) -> 1.5 x + 0.5 y
//                  LN (block output) }
//   -> out_norm -> CTC head Conv1d(k=1) + bias                   [vocab, T_enc, B]
//
// Every LayerNorm is bias-free with eps = stt.medasr.encoder.layer_norm_eps
// (1e-6, not the conformer helper's 1e-5). F16 weight matmuls force F32
// accumulation (the arch's CUDA COMPUTE_16F guard; a no-op on CPU).
//
// What the arch had and this does not: the transcribe tensor-dump plumbing
// (mark_tensor_for_dump, the transposed "enc.ctc_logits" dump side-output) -
// debug observability only, it never feeds the logits.

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_cgraph;

namespace engine::models::medasr {

struct MedAsrHParams;

struct MedAsrSubsamplingWeights {
  ggml_tensor *dense0_w = nullptr; // ne [n_mels, d]
  ggml_tensor *dense0_b = nullptr; // [d]
  ggml_tensor *conv0_w = nullptr;  // ne [k, d, d]
  ggml_tensor *conv0_b = nullptr;  // [d]
  ggml_tensor *conv1_w = nullptr;  // ne [k, d, sub_channels]
  ggml_tensor *conv1_b = nullptr;  // [sub_channels]
  ggml_tensor *dense1_w = nullptr; // ne [sub_channels, d]
  ggml_tensor *dense1_b = nullptr; // [d]
};

// One Conformer block. LASR has no LayerNorm biases and no linear / conv
// biases inside the block.
struct MedAsrBlockWeights {
  ggml_tensor *norm_ff1_w = nullptr;
  ggml_tensor *ff1_up_w = nullptr;   // ne [d, d_ff]
  ggml_tensor *ff1_down_w = nullptr; // ne [d_ff, d]

  ggml_tensor *norm_attn_w = nullptr;
  ggml_tensor *attn_q_w = nullptr; // ne [d, d]
  ggml_tensor *attn_k_w = nullptr;
  ggml_tensor *attn_v_w = nullptr;
  ggml_tensor *attn_o_w = nullptr;

  ggml_tensor *norm_conv_w = nullptr;
  ggml_tensor *conv_pw1_w = nullptr;    // ne [1, d, 2d]
  ggml_tensor *conv_dw_w = nullptr;     // ne [k, 1, d]
  ggml_tensor *conv_bn_scale = nullptr; // [d], folded on the host at load
  ggml_tensor *conv_bn_bias = nullptr;  // [d]
  ggml_tensor *conv_pw2_w = nullptr;    // ne [1, d, d]

  ggml_tensor *norm_ff2_w = nullptr;
  ggml_tensor *ff2_up_w = nullptr;
  ggml_tensor *ff2_down_w = nullptr;

  ggml_tensor *norm_post_w = nullptr;
};

struct MedAsrWeights {
  MedAsrSubsamplingWeights subsampling;
  std::vector<MedAsrBlockWeights> blocks;
  ggml_tensor *enc_out_norm_w = nullptr; // [d]
  ggml_tensor *ctc_proj_w = nullptr;     // ne [1, d, vocab]
  ggml_tensor *ctc_proj_b = nullptr;     // [vocab]
};

// Backend-dependent op choices, resolved exactly as the arch resolves them.
struct MedAsrGraphPolicy {
  // Pointwise convs as a direct mul_mat (true) or im2col (false). The arch's
  // detect_direct_pw: false on Vulkan, true elsewhere; env
  // TRANSCRIBE_CONV_DIRECT_PW forces true, TRANSCRIBE_CONV_NO_DIRECT_PW false.
  bool direct_pw = true;
  // ggml_flash_attn_ext for mask-free attention. Arch default on; env
  // TRANSCRIBE_NO_FLASH turns it off, TRANSCRIBE_FORCE_FLASH back on. A
  // variable-length batch always takes the manual path (key-padding mask).
  bool use_flash = true;
};

MedAsrGraphPolicy resolve_graph_policy(const char *backend_name);

struct MedAsrEncoderBuild {
  ggml_tensor *mel_in = nullptr;    // [n_mels, T_mel, 1, B] f32, frame-major
  ggml_tensor *positions = nullptr; // [T_enc] i32, 0..T_enc-1
  // Variable-length batches only (else null): per-key additive mask
  // [T_enc, 1, 1, B] (0 / -1e30) and post-GLU valid-frame mask
  // [T_enc, 1, B, 1] (1 / 0).
  ggml_tensor *attn_pad_mask_in = nullptr;
  ggml_tensor *conv_pad_mask_in = nullptr;
  ggml_tensor *logits = nullptr; // [vocab, T_enc, B]: frame-major per utterance
  ggml_cgraph *graph = nullptr;
  int64_t T_enc = 0;
};

// Node budget of the encoder graph (the arch's ggml_new_graph_custom size).
constexpr int kEncoderGraphNodes = 8192;

// Bytes a no_alloc ggml context needs to hold the encoder graph's tensor and
// graph metadata.
size_t encoder_graph_context_bytes();

// Builds the whole encoder + CTC head. `n_batch` utterances ride ne[3] of the
// mel input (ne[2] of every activation). `batch_var_len` (only meaningful for
// n_batch > 1) adds the two padding masks and forces the manual attention.
// Returns a build whose `graph` is null when `n_mel_frames` is too short for
// the subsampling stem.
MedAsrEncoderBuild build_encoder_graph(ggml_context *ctx, const MedAsrWeights &w,
                                       const MedAsrHParams &hp, int n_mel_frames,
                                       int n_batch, bool batch_var_len,
                                       const MedAsrGraphPolicy &policy);

} // namespace engine::models::medasr
