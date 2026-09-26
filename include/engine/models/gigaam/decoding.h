// engine/models/gigaam/decoding.h - GigaAM RNN-T / CTC greedy decoders over
// the host-side head weights.
//
// Port of src/runtime/arch/gigaam/decoder.{h,cpp}. The heads are tiny (one
// 320-wide LSTM layer + a single joint linear, or a 1x1-conv CTC head), so the
// arch runs them on the host, off the encoder output, and so does this package.
// Numerics are the arch's non-BLAS paths verbatim (speech.cpp never defines
// TRANSCRIBE_HAS_BLAS, so those are the paths the arch actually runs): f32
// accumulation in the same order, the same blank-span speculation, the same
// predictor cache and max-symbols rule, the same CTC log_softmax + collapse.

#pragma once

#include "engine/models/gigaam/graphs_internal.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace engine::models::gigaam {

// Host mirror of the RNN-T predictor + joint or of the CTC head, dequantized
// to f32 once at load. Row-major numpy layouts ([out, in]).
struct GigaamHostDecoder {
  // RNN-T.
  std::vector<float> pred_embed;               // [pred_vocab, pred_hidden]
  std::vector<std::vector<float>> lstm_Wx;     // per layer [4*H, H], gates (i, f, g, o)
  std::vector<std::vector<float>> lstm_Wh;     // per layer [4*H, H]
  std::vector<std::vector<float>> lstm_b;      // per layer [4*H] (bias_ih + bias_hh)
  std::vector<float> joint_enc_w;              // [joint_hidden, d_model]
  std::vector<float> joint_enc_b;              // [joint_hidden]
  std::vector<float> joint_pred_w;             // [joint_hidden, pred_hidden]
  std::vector<float> joint_pred_b;             // [joint_hidden]
  std::vector<float> joint_out_w;              // [n_classes, joint_hidden]
  std::vector<float> joint_out_b;              // [n_classes]
  // CTC.
  std::vector<float> ctc_w;                    // [n_classes, d_model]
  std::vector<float> ctc_b;                    // [n_classes]
};

// Called at decode-step boundaries with (frames done, total frames). The
// runtime forwards it to RunControl::emit_progress, which throws
// ProgressCanceled on an abort request - that is how a cancel unwinds.
using GigaamDecodePoll = std::function<void(int done, int total)>;

// The arch's max_symbols_per_step for the RNN-T loop.
constexpr int kGigaamMaxSymbolsPerStep = 10;

// RNN-T greedy decode. `encoded` is one utterance's encoder output,
// T-major [T_enc * d_model]. Emits the non-blank tokens and the encoder
// frame each was emitted on.
void decode_rnnt_greedy(const GigaamHostDecoder &host, const GigaamHParams &hp,
                        const float *encoded, int T_enc, int max_symbols_per_step,
                        std::vector<int32_t> &out_tokens, std::vector<int> &out_frames,
                        const GigaamDecodePoll &poll = {});

// CTC greedy decode: per-frame linear head, log_softmax, argmax, collapse
// repeats, drop blanks. Same `encoded` layout. Throws std::runtime_error when
// the host head does not match the hparams.
void decode_ctc_greedy(const GigaamHostDecoder &host, const GigaamHParams &hp,
                       const float *encoded, int T_enc, std::vector<int32_t> &out_tokens,
                       std::vector<int> &out_frames, const GigaamDecodePoll &poll = {});

} // namespace engine::models::gigaam
