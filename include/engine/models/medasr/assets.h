#pragma once

// engine/models/medasr/assets.h - host-side model resources for the native
// engine MedASR (Google LASR-CTC) package.
//
// Port of src/runtime/arch/medasr/weights.{h,cpp} (read_medasr_hparams,
// fuse_batch_norm) and of the frontend / tokenizer half of
// arch/medasr/model.cpp::load(). Reads EXACTLY the GGUF the arch reads - the
// transcribe.cpp layout published under huggingface.co/handy-computer/
// medasr-gguf (`general.architecture = "medasr"`, `stt.medasr.*` /
// `stt.frontend.*` KVs, the baked `frontend.mel_filterbank` /
// `frontend.window` tensors, a SentencePiece-BPE `tokenizer.ggml.*` vocab
// trimmed to the 512-wide CTC head) with the same validation.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/text/tokenizer_hub.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::medasr {

// Hyperparameters, one field per KV the arch reads (weights.h, MedAsrHParams).
struct MedAsrHParams {
  // Encoder (stt.medasr.encoder.*).
  int32_t enc_n_layers = 0;
  int32_t enc_hidden = 0; // d_model
  int32_t enc_n_heads = 0;
  int32_t enc_n_kv_heads = 0; // = n_heads (no GQA)
  int32_t enc_head_dim = 0;
  int32_t enc_intermediate = 0; // d_ff
  std::string enc_hidden_act;   // "silu"
  int32_t enc_conv_kernel = 0;  // 32
  float enc_conv_resid_w0 = 0.0f; // 2.0
  float enc_conv_resid_w1 = 0.0f; // 1.0
  float enc_ff_resid_w0 = 0.0f;   // 1.5
  float enc_ff_resid_w1 = 0.0f;   // 0.5
  float enc_layer_norm_eps = 1e-6f;
  int32_t enc_max_pos_emb = 0;
  float enc_batch_norm_momentum = 0.01f;
  float enc_rope_theta = 10000.0f;
  std::string enc_rope_type; // "default"

  // Subsampling stem.
  int32_t enc_num_mel_bins = 0; // 128
  int32_t enc_sub_kernel = 0;   // 5
  int32_t enc_sub_stride = 0;   // 2
  int32_t enc_sub_channels = 0; // 256
  int32_t enc_sub_n_layers = 0; // 2

  // CTC head (stt.medasr.ctc.*).
  int32_t ctc_vocab_size = 0;
  int32_t ctc_blank_id = 0; // 0 (<epsilon>)

  // Frontend (stt.frontend.*). Read (required) like the arch; the arch then
  // hard-wires the LASR STFT (center=False, the GGUF's window + filterbank,
  // natural log with a floor clamp), and so does this package.
  std::string fe_type;
  int32_t fe_sample_rate = 0;
  int32_t fe_num_mels = 0;
  int32_t fe_n_fft = 0;
  int32_t fe_win_length = 0;
  int32_t fe_hop_length = 0;
  std::string fe_window;
  std::string fe_normalize;
  std::string fe_pad_mode;
  std::string fe_mel_norm;
  float fe_log_clamp_min = 1e-5f;
  float fe_mel_lower_hz = 125.0f;
  float fe_mel_upper_hz = 7500.0f;
};

// How the vocabulary pieces are joined back into text, picked from
// tokenizer.ggml.model as the arch's transcribe::Tokenizer does. The MedASR
// converter always writes "bpe" (SentencePiece BPE: U+2581 -> space, <0xNN>
// byte fallback), which TokenizerHub decodes identically; only "char"
// (verbatim concat) is joined locally.
enum class MedAsrPieceDecode { SentencePiece, RawConcat, ByteLevel };

struct MedAsrAssets {
  MedAsrHParams hparams;
  std::string variant; // stt.variant, else "medasr"
  std::filesystem::path model_path;

  // Weights through the shared TensorSource (transcribe.cpp canonical names:
  // enc.subsampling.*, enc.blocks.<i>.*, enc.out_norm.weight, ctc.proj.*).
  std::shared_ptr<const assets::TensorSource> source;

  // tokenizer.ggml.* read through TokenizerHub (pieces + specials); decode()
  // below applies the arch's join rule for the file's tokenizer model.
  text::TokenizerPtr tokenizer;
  MedAsrPieceDecode piece_decode = MedAsrPieceDecode::SentencePiece;
  std::vector<std::string> pieces; // tokenizer.ggml.tokens, by id

  // general.languages (empty when the GGUF does not say - the arch's
  // "information gap" reading, n_languages = 0).
  std::vector<std::string> language_codes;

  // Frontend buffers, ready for engine::audio::MelExtractor: the filterbank
  // transposed once to row-major [n_mels, n_fft / 2 + 1] (the GGUF stores
  // ggml ne = [n_mels, n_freq], i.e. [n_freq][n_mels] row-major), and the
  // window as shipped [win_length].
  std::vector<float> mel_filterbank;
  std::vector<float> window;

  // Conv-module BatchNorm folded on the host at load (arch fuse_batch_norm):
  // scale = w / sqrt(var + 1e-5), bias = b - mean * scale, per layer [d_model].
  std::vector<std::vector<float>> bn_scale;
  std::vector<std::vector<float>> bn_bias;

  const MedAsrHParams &config() const noexcept { return hparams; }

  // Encoder frames after the two stride-2 subsampling convs (padding 0), the
  // exact formula the arch's run() / run_batch() use; 0 when too short.
  int encoder_frames_for(int n_mel_frames) const noexcept;
  // sub_stride ^ sub_n_layers (4 for the shipped checkpoint).
  int subsample_factor() const noexcept;
  // hop * factor * 1000 / sample_rate (40 ms for the shipped checkpoint).
  int64_t ms_per_encoder_frame() const noexcept;
  // The arch's advisory transcribe_capabilities::max_audio_ms: the
  // RoPE-trained window, enc_max_pos_emb * ms_per_encoder_frame (400 s).
  // Past it MedASR warns and proceeds; it never rejects on length.
  int64_t max_audio_ms() const noexcept;
  // Smallest PCM length (samples) that yields >= 1 encoder frame.
  int64_t min_audio_samples() const noexcept;

  // Vocabulary piece for an id ("" when out of range) - the arch's token-row
  // text (transcribe::Tokenizer::token(id): the raw piece, U+2581 included).
  std::string piece(int32_t id) const;
  // Join ids into text exactly as the arch's Tokenizer::decode does.
  std::string decode(const std::vector<int32_t> &ids) const;
};

// Loads a MedASR GGUF. Throws std::runtime_error on anything the arch's
// load() would have rejected (missing / mistyped KV, invariant violations,
// missing tokenizer or frontend tensors, a malformed BatchNorm).
std::shared_ptr<const MedAsrAssets>
load_medasr_assets(const std::filesystem::path &model_path);

// Cheap sniff for the loader's can_load(): GGUF magic and
// general.architecture == "medasr".
bool looks_like_medasr_gguf(const std::filesystem::path &path);

} // namespace engine::models::medasr
