// engine/models/gigaam/graphs_internal.h - native engine GigaAM ASR graph
// layer: hparams, encoder weight slots and the Conformer encoder graph builder.
//
// Phase 11 port of src/runtime/arch/gigaam/ into the engine framework. The
// encoder graph mirrors the arch op for op (same ggml ops, same order, same
// constants); only the surrounding machinery differs (BackendWeightStore /
// ggml_gallocr on the session backend instead of transcribe_model /
// ggml_backend_sched).
//
// Architecture (all four published GigaAM-v3 variants):
//   - Frontend: 64-bin HTK log-mel, n_fft = win_length = 320, hop 160,
//     center=False, periodic Hann, power 2, log(clamp(x, 1e-9, 1e9)). The
//     filterbank and window are baked into the GGUF. Delivered by the
//     framework engine::audio::MelExtractor (see runtime.cpp for the exact
//     configuration and the frame alignment).
//   - Encoder: 2 stride-2 Conv1d (k=5, ReLU) pre-encode, then N Conformer
//     blocks: macaron FF1 (x0.5) -> ROTARY self-attention (rotation applied
//     to the PRE-projection activation; Wv sees the unrotated input) -> conv
//     module (pw1 -> GLU -> depthwise -> LayerNorm -> SiLU -> pw2) -> macaron
//     FF2 (x0.5) -> per-block LayerNorm. Every linear / conv carries a bias.
//   - Heads (host side, decoding.h): RNN-T (1-layer LSTM predictor + ReLU
//     joint) or a 1x1-Conv1d CTC head.

#pragma once

// Internal graph-layer header for the native GigaAM engine package. Not part
// of any installed public API surface.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;

namespace engine::models::gigaam {

// ---------------------------------------------------------------------------
// Hyperparameters (the arch's GigaamHParams, field for field; read from the
// same stt.gigaam.* / stt.frontend.* KVs by assets.cpp)
// ---------------------------------------------------------------------------

enum class GigaamHeadKind { Rnnt, Ctc };

struct GigaamHParams {
  GigaamHeadKind head_kind = GigaamHeadKind::Rnnt;

  // Encoder (Conformer).
  int32_t enc_n_layers = 0;
  int32_t enc_d_model = 0;
  int32_t enc_n_heads = 0;
  int32_t enc_d_ff = 0;
  int32_t enc_conv_kernel = 0;
  int32_t enc_subsampling_factor = 0;
  int32_t enc_subs_kernel_size = 0;
  int32_t enc_pos_emb_max_len = 0;
  int32_t enc_feat_in = 0;
  std::string enc_self_attention_model; // "rotary"
  std::string enc_conv_norm_type;       // "layer_norm"

  // RNN-T predictor + joint (Rnnt only).
  int32_t pred_hidden = 0;
  int32_t pred_n_layers = 0;
  int32_t pred_vocab = 0;
  int32_t joint_hidden = 0;
  int32_t joint_n_classes = 0;
  std::string joint_activation;

  // CTC head (Ctc only).
  int32_t head_feat_in = 0;
  int32_t head_n_classes = 0;

  // Frontend.
  std::string fe_type; // "mel"
  int32_t fe_num_mels = 0;
  int32_t fe_sample_rate = 0;
  int32_t fe_n_fft = 0;
  int32_t fe_win_length = 0;
  int32_t fe_hop_length = 0;
  std::string fe_window;    // "hann_periodic"
  std::string fe_normalize; // "none"
  float fe_dither = 0.0f;
  float fe_pre_emphasis = 0.0f;
  float fe_f_min = 0.0f;
  float fe_f_max = 0.0f;
  bool fe_center = false;
  std::string fe_mel_norm; // "htk"
  float fe_log_clamp_min = 1e-9f;
  float fe_log_clamp_max = 1e9f;

  int32_t enc_head_dim() const { return enc_n_heads > 0 ? enc_d_model / enc_n_heads : 0; }
  int32_t n_classes() const {
    return head_kind == GigaamHeadKind::Rnnt ? joint_n_classes : head_n_classes;
  }
  // Blank is the last class for both heads (the converter appends it).
  int32_t blank_id() const { return n_classes() - 1; }
};

// ---------------------------------------------------------------------------
// Encoder weight slots (backend tensors owned by the session's weight store).
// ggml ne order, i.e. the reverse of the TensorSource logical shape.
// ---------------------------------------------------------------------------

struct GigaamPreEncodeWeights {
  ggml_tensor *conv0_w = nullptr; // ne [k, feat_in, d_model]
  ggml_tensor *conv0_b = nullptr; // ne [d_model]
  ggml_tensor *conv2_w = nullptr; // ne [k, d_model, d_model]
  ggml_tensor *conv2_b = nullptr; // ne [d_model]
};

struct GigaamBlockWeights {
  // Macaron FF1.
  ggml_tensor *norm_ff1_w = nullptr;
  ggml_tensor *norm_ff1_b = nullptr;
  ggml_tensor *ff1_lin1_w = nullptr; // ne [d_model, d_ff]
  ggml_tensor *ff1_lin1_b = nullptr;
  ggml_tensor *ff1_lin2_w = nullptr; // ne [d_ff, d_model]
  ggml_tensor *ff1_lin2_b = nullptr;

  // Conv module (LayerNorm post-depthwise).
  ggml_tensor *norm_conv_w = nullptr;
  ggml_tensor *norm_conv_b = nullptr;
  ggml_tensor *conv_pw1_w = nullptr; // ne [1, d_model, 2*d_model]
  ggml_tensor *conv_pw1_b = nullptr;
  ggml_tensor *conv_dw_w = nullptr; // ne [conv_kernel, 1, d_model]
  ggml_tensor *conv_dw_b = nullptr;
  ggml_tensor *conv_ln_w = nullptr;
  ggml_tensor *conv_ln_b = nullptr;
  ggml_tensor *conv_pw2_w = nullptr; // ne [1, d_model, d_model]
  ggml_tensor *conv_pw2_b = nullptr;

  // Rotary self-attention (no linear_pos / pos_bias_*).
  ggml_tensor *norm_attn_w = nullptr;
  ggml_tensor *norm_attn_b = nullptr;
  ggml_tensor *attn_q_w = nullptr;
  ggml_tensor *attn_q_b = nullptr;
  ggml_tensor *attn_k_w = nullptr;
  ggml_tensor *attn_k_b = nullptr;
  ggml_tensor *attn_v_w = nullptr;
  ggml_tensor *attn_v_b = nullptr;
  ggml_tensor *attn_out_w = nullptr;
  ggml_tensor *attn_out_b = nullptr;

  // Macaron FF2.
  ggml_tensor *norm_ff2_w = nullptr;
  ggml_tensor *norm_ff2_b = nullptr;
  ggml_tensor *ff2_lin1_w = nullptr;
  ggml_tensor *ff2_lin1_b = nullptr;
  ggml_tensor *ff2_lin2_w = nullptr;
  ggml_tensor *ff2_lin2_b = nullptr;

  // Final per-block LayerNorm.
  ggml_tensor *norm_out_w = nullptr;
  ggml_tensor *norm_out_b = nullptr;
};

struct GigaamEncoderWeights {
  GigaamPreEncodeWeights pre_encode;
  std::vector<GigaamBlockWeights> blocks;
};

// Backend-dependent op dispatch, resolved exactly as the arch resolves it.
struct GigaamGraphPolicy {
  // Flash attention on every non-CPU backend (the arch: backend name does
  // not contain "CPU"); the manual mul_mat + soft_max path on CPU.
  bool use_flash = false;
  // Pointwise convs as direct mul_mats (conformer::detect_direct_pw): on by
  // default, off on Vulkan; TRANSCRIBE_CONV_DIRECT_PW / _NO_DIRECT_PW
  // override, as they do for the arch.
  bool direct_pw = true;
};

GigaamGraphPolicy resolve_gigaam_graph_policy(const char *backend_name);

// One stride-2 conv with symmetric (k-1)/2 padding: floor((in-1)/2)+1.
int gigaam_pre_encode_t_out(int in);

// Encoder output frames for a mel of `n_mel_frames` (two stride-2 convs).
int gigaam_encoder_frames(int n_mel_frames);

struct GigaamEncoderBuild {
  // Graph inputs. mel_in ne = [T_mel, n_mels, n_batch]: per utterance the
  // data is MEL-MAJOR (element (t, m) at m * T_mel + t), i.e. exactly the
  // layout MelExtractor produces - no transpose is needed.
  ggml_tensor *mel_in = nullptr;
  ggml_tensor *positions = nullptr; // I32 [T_enc], 0..T_enc-1 (rotary)

  // Variable-length batch masks (null unless n_batch > 1 && var_len):
  //   attn_pad_mask       ne [T_enc, 1, 1, B]: 0 real key / -1e30 padded
  //   conv_pad_mask       ne [T_enc, 1, B, 1]: 1 real frame / 0 padded
  //   pre_encode_mask_s1  ne [T_s1, 1, B]    : after conv0 ReLU
  //   pre_encode_mask_s2  ne [T_s2, 1, B]    : after conv2 ReLU
  ggml_tensor *attn_pad_mask = nullptr;
  ggml_tensor *conv_pad_mask = nullptr;
  ggml_tensor *pre_encode_mask_s1 = nullptr;
  ggml_tensor *pre_encode_mask_s2 = nullptr;

  // Output: the arch's `rnnt.encoded`, ne = [d_model, T_enc, n_batch] -
  // per utterance T-major (frame t's d_model vector contiguous), which is
  // what both host decoders consume. Marked as a graph output.
  ggml_tensor *encoded = nullptr;

  ggml_cgraph *graph = nullptr;
  int T_enc = 0;
};

// Build a fresh encoder forward graph (the arch's build_encoder_graph).
// n_batch == 1 is the single-shot graph; n_batch > 1 packs utterances along
// ne[2]; batch_var_len (with n_batch > 1) adds the masks above. Every input
// must be uploaded after allocation, before each compute - the graph is
// built per run, so nothing relies on a value surviving a previous compute.
GigaamEncoderBuild build_gigaam_encoder_graph(ggml_context *ctx,
                                              const GigaamEncoderWeights &weights,
                                              const GigaamHParams &hp, int n_mel_frames,
                                              const GigaamGraphPolicy &policy, int n_batch = 1,
                                              bool batch_var_len = false);

// Metadata bytes for the no_alloc context build_gigaam_encoder_graph needs.
size_t gigaam_encoder_context_bytes(int n_mel_frames);

} // namespace engine::models::gigaam
