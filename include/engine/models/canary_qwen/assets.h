#pragma once

// engine/models/canary_qwen/assets.h - host-side resources for the native
// engine Canary-Qwen (NeMo SALM: FastConformer encoder + Qwen3-1.7B LM)
// package.
//
// One distribution format: parent transcribe.cpp's GGUF
// (`general.architecture = "canary_qwen"`, `stt.canary_qwen.*` /
// `stt.frontend.*` KVs, a Qwen2 byte-level BPE tokenizer, the NeMo mel
// filterbank and window as tensors), published as
// huggingface.co/handy-computer/canary-qwen-2.5b-gguf (BF16, F16, Q8_0, Q6_K,
// Q5_K_M, Q4_K_M). Every KV and tensor name read here is one the retired arch
// (src/runtime/arch/canary_qwen/weights.cpp, read_canary_qwen_hparams /
// build_canary_qwen_weights) read; nothing is invented.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/text/tokenizer_hub.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::canary_qwen {

// KV hyperparameters (the arch's CanaryQwenHParams, field for field).
struct CanaryQwenHParams {
  // Encoder (FastConformer; matches canary-1b-flash).
  int32_t enc_n_layers = 0;           // 32
  int32_t enc_d_model = 0;            // 1024
  int32_t enc_n_heads = 0;            // 8 (-> head_dim 128)
  int32_t enc_d_ff = 0;               // 4096
  int32_t enc_conv_kernel = 0;        // 9
  int32_t enc_subsampling_factor = 0; // 8
  int32_t enc_subsampling_chans = 0;  // 256
  int32_t enc_pos_emb_max_len = 0;    // 5000

  // Perception projection (Linear(1024, 2048) + bias).
  int32_t perception_output_dim = 0;  // 2048 (== dec_hidden)
  int32_t audio_locator_id = -1;      // 151669 = "<|audioplaceholder|>"

  // Decoder (Qwen3-1.7B).
  int32_t dec_n_layers = 0;       // 28
  int32_t dec_hidden = 0;         // 2048
  int32_t dec_intermediate = 0;   // 6144
  int32_t dec_n_heads = 0;        // 16
  int32_t dec_n_kv_heads = 0;     // 8
  int32_t dec_head_dim = 0;       // 128
  int32_t dec_max_position = 0;   // 40960
  int32_t dec_vocab_size = 0;     // 151936
  float dec_rms_norm_eps = 0.0f;  // 1e-6
  float dec_rope_theta = 0.0f;    // 1e6
  bool dec_tie_word_embeddings = true;

  // Tokenizer-derived.
  int32_t vocab_size = 0;
  int32_t bos_token_id = -1;
  int32_t eos_token_id = -1;  // 151645 = "<|im_end|>"

  // Frontend (NeMo AudioToMelSpectrogramPreprocessor).
  int32_t fe_sample_rate = 0;  // 16000
  int32_t fe_num_mels = 0;     // 128
  int32_t fe_n_fft = 0;        // 512
  int32_t fe_win_length = 0;   // 400
  int32_t fe_hop_length = 0;   // 160
  float fe_pre_emphasis = 0.0f;
  float fe_f_min = 0.0f;
  float fe_f_max = 0.0f;
  std::string fe_pad_mode;   // "reflect"
  std::string fe_normalize;  // "per_feature"
};

// Resolved chat-template special-token ids (the arch's ChatTokens).
struct ChatTokens {
  int32_t im_start = -1;        // "<|im_start|>"
  int32_t im_end = -1;          // "<|im_end|>"
  int32_t role_user = -1;       // "user"
  int32_t role_assistant = -1;  // "assistant"
};

// Input-length contract (docs/input-limits.md), identical to the arch: audio
// tokens + chat prompt + generation share the Qwen3 context window
// (decoder.max_position_embeddings, optionally lowered by the session n_ctx).
inline constexpr int kGenReserve = 256;      // input-gate reserve and decode-budget floor
inline constexpr int kPromptOverhead = 32;   // advisory template overhead for max_audio_ms

struct CanaryQwenAssets {
  CanaryQwenHParams hparams;
  std::string variant;  // stt.variant, else "canary-qwen-2.5b"
  std::filesystem::path model_path;

  // Weights, through the shared TensorSource (transcribe.cpp tensor names).
  std::shared_ptr<const assets::TensorSource> source;

  // Qwen2 byte-level BPE from tokenizer.ggml.* (TokenizerHub).
  text::TokenizerPtr tokenizer;

  // Capability KVs (general.languages, stt.capability.*); absent = empty/false.
  std::vector<std::string> languages;
  bool supports_translate = false;
  bool supports_language_detect = false;

  // Checkpoint frontend buffers; empty when the GGUF does not carry them (the
  // frontend then computes them, as the arch did).
  std::vector<float> mel_filterbank;  // [num_mels * (n_fft / 2 + 1)]
  std::vector<float> window;          // [win_length]

  // Static chat-template segments, built once at load exactly as the arch:
  //   prefix = [<|im_start|>] + bpe("user\nTranscribe the following: ")
  //   suffix = [<|im_end|>] + bpe("\n") + [<|im_start|>] + bpe("assistant\n")
  // The audio locator id is repeated T_enc times between them at run time.
  ChatTokens chat_tokens;
  std::vector<int32_t> prompt_prefix_ids;
  std::vector<int32_t> prompt_suffix_ids;

  // Advisory input bound (transcribe_capabilities::max_audio_ms in the arch)
  // and the rate the decode budget is predicted from.
  int64_t max_audio_ms = 0;
  double ms_per_audio_token = 0.0;

  const CanaryQwenHParams &config() const noexcept { return hparams; }

  std::vector<int32_t> encode(std::string_view text) const;
  std::string decode(const std::vector<int32_t> &ids) const;
};

// Loads a transcribe.cpp canary_qwen GGUF. Throws std::runtime_error on
// anything that is not a well-formed canary_qwen model (every rejection the
// arch's loader made, with its message).
std::shared_ptr<const CanaryQwenAssets>
load_canary_qwen_assets(const std::filesystem::path &model_path);

// Cheap sniff used by the loader's can_load(): GGUF magic + architecture.
bool looks_like_canary_qwen_gguf(const std::filesystem::path &path);

// Effective decoder context ceiling in tokens: the model's trained maximum,
// optionally lowered - never raised - by the session n_ctx knob.
int canary_qwen_context_ceiling(int n_ctx_knob, const CanaryQwenHParams &hp);

// The arch's canary_qwen_max_audio_ms (advisory; 0 = unknown).
int64_t canary_qwen_max_audio_ms(const CanaryQwenHParams &hp);

} // namespace engine::models::canary_qwen
