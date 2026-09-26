#pragma once

// engine/models/granite_nar/assets.h - host-side resources for the native
// engine Granite Speech NAR (non-autoregressive editor) package.
//
// One distribution format: parent transcribe.cpp's GGUF
// (`general.architecture = "granite_speech_nar"`, `stt.granite_nar.*` /
// `stt.frontend.*` KVs, a granite-4 byte-level BPE tokenizer
// (tokenizer.ggml.model "gpt2", pre "granite"), the librosa-htk filterbank and
// periodic Hann window as `frontend.*` tensors), published as
// huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf (BF16, F16,
// Q8_0, Q6_K, Q5_K_M, Q4_K_M). Every KV and tensor name read here is one the
// arch (src/runtime/arch/granite_nar/weights.cpp read_granite_nar_hparams /
// build_granite_nar_weights, model.cpp load()) read, with the same defaults
// and the same cross-field checks; nothing is invented.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/text/tokenizer_hub.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::granite_nar {

// The arch's GraniteNarHParams, field for field (weights.h).
struct GraniteNarHParams {
  // Encoder (stt.granite_nar.encoder.*).
  int32_t enc_n_layers = 0;         // 16
  int32_t enc_hidden = 0;           // 1024
  int32_t enc_n_heads = 0;          // 8
  int32_t enc_head_dim = 0;         // 128
  int32_t enc_input_dim = 0;        // 160 (2 stacked mel frames)
  int32_t enc_output_dim = 0;       // 348 (char-CTC vocab)
  int32_t enc_bpe_output_dim = 0;   // 100352 (new snapshot) / 100353 (old)
  int32_t enc_bpe_pool_window = 0;  // 4
  // BPE-CTC blank id: 100257 (new snapshot, channels ARE the LLM ids) or 0
  // (old snapshot, channel 0 is a synthetic blank; LLM id = argmax - 1).
  // Optional KV, default 0 (the arch's legacy scheme).
  int32_t enc_bpe_blank_id = 0;
  int32_t enc_self_cond_layer = 0;  // 8 (read, validated as present; the
                                    //  graph uses n_layers / 2, as the arch)
  int32_t enc_feedforward_mult = 0; // 4
  int32_t enc_conv_kernel_size = 0; // 15
  int32_t enc_conv_expansion = 0;   // 2
  int32_t enc_max_pos_emb = 0;      // 512
  int32_t enc_context_size = 0;     // 200
  // 1-indexed capture points for the projector (e.g. [4, 8, 12, -1]);
  // absent -> {-1}.
  std::vector<int32_t> enc_layer_indices;

  // Projector (EncoderProjectorQFormer, stt.granite_nar.projector.*).
  int32_t prj_n_layers = 0;           // 2
  int32_t prj_hidden = 0;             // 2048
  int32_t prj_mlp_ratio = 0;          // 2
  int32_t prj_n_heads = 0;            // 32
  int32_t prj_encoder_dim = 0;        // 1024
  int32_t prj_num_encoder_layers = 0; // 4 (== len(enc_layer_indices))
  int32_t prj_block_size = 0;         // 15
  int32_t prj_downsample_rate = 0;    // 5 (15 / 5 = 3 queries per window)
  int32_t prj_llm_dim = 0;            // 2048
  float prj_layernorm_eps = 0.0f;
  bool prj_attn_bias = true;
  bool prj_mlp_bias = true;
  bool scale_projected_embeddings = true;

  // Text LM (Granite-4, stt.granite_nar.text.*).
  int32_t dec_n_layers = 0;     // 40
  int32_t dec_hidden = 0;       // 2048
  int32_t dec_intermediate = 0; // 4096
  int32_t dec_n_heads = 0;      // 16
  int32_t dec_n_kv_heads = 0;   // 4
  int32_t dec_head_dim = 0;     // 128
  std::string dec_hidden_act;
  float dec_rms_norm_eps = 0.0f;
  float dec_rope_theta = 0.0f;
  int32_t dec_max_pos_emb = 0;  // 4096
  bool dec_tie_word_embeddings = true;
  int32_t dec_vocab_size = 0;   // 100352
  float dec_embedding_multiplier = 0.0f; // 12
  float dec_logits_scaling = 0.0f;       // 8
  float dec_attention_multiplier = 0.0f; // 1/128
  float dec_residual_multiplier = 0.0f;  // 0.22
  // bos / eos are OVERWRITTEN from the tokenizer after loading, exactly as the
  // arch's load() does (m->hparams.dec_eos_id = m->tok.eos_id()); the KVs are
  // still required.
  int32_t dec_bos_id = 0;
  int32_t dec_eos_id = 0;
  int32_t dec_pad_id = 0;

  // Frontend (stt.frontend.*).
  std::string fe_type;
  int32_t fe_sample_rate = 0;
  int32_t fe_num_mels = 0;
  int32_t fe_n_fft = 0;
  int32_t fe_win_length = 0;
  int32_t fe_hop_length = 0;
  std::string fe_window;
  std::string fe_normalize;
  std::string fe_pad_mode; // optional, "reflect"
  std::string fe_mel_norm; // optional, "htk"

  // stt.granite_nar.ctc_chars (index -> char; optional, informational).
  std::vector<std::string> ctc_chars;
};

// Input-length contract (docs/input-limits.md), identical to the arch
// (model.cpp): a hard context cap on audio tokens + editor text.
// Representative text budget reserved when advertising max_audio_ms.
inline constexpr int kTextBudget = 256;
// add_insertion_slots floors the editor text at 8 positions (decoding.h), so
// n_audio + 8 is the smallest T_total any utterance can reach.
inline constexpr int kMinEditorText = 8;

struct GraniteNarAssets {
  GraniteNarHParams hparams;
  std::string variant; // stt.variant, else "granite-speech-nar"
  std::filesystem::path model_path;

  // Weights through the shared TensorSource (transcribe.cpp tensor names:
  // enc.*, prj.*, dec.*).
  std::shared_ptr<const assets::TensorSource> source;

  // Granite-4 byte-level BPE (tokenizer.ggml.*), TokenizerHub.
  text::TokenizerPtr tokenizer;

  // read_capability_kv / read_languages_kv (absent = empty / false).
  std::vector<std::string> languages;
  bool supports_translate = false;
  bool supports_language_detect = false;

  // Frontend buffers baked by the converter; empty when absent (the frontend
  // then builds its own, as the arch's MelFrontend did).
  std::vector<float> mel_filterbank; // [num_mels * (n_fft / 2 + 1)] row-major [mel][freq]
  std::vector<float> window;         // [win_length]

  // Advisory transcribe_capabilities::max_audio_ms (granite_nar_max_audio_ms).
  int64_t max_audio_ms = 0;

  const GraniteNarHParams &config() const noexcept { return hparams; }

  std::vector<int32_t> encode(std::string_view text) const;
  std::string decode(const std::vector<int32_t> &ids) const;
};

// Loads a transcribe.cpp granite_nar GGUF. Throws std::runtime_error on
// anything the arch's load() rejected.
std::shared_ptr<const GraniteNarAssets>
load_granite_nar_assets(const std::filesystem::path &model_path);

// Cheap sniff for the loader's can_load(): GGUF magic and
// general.architecture == "granite_speech_nar".
bool looks_like_granite_nar_gguf(const std::filesystem::path &path);

// arch model.cpp granite_nar_context_ceiling: the model's trained maximum
// (dec_max_pos_emb), optionally lowered - never raised - by the session n_ctx.
int granite_nar_context_ceiling(int n_ctx_knob, const GraniteNarHParams &hp);

// arch model.cpp granite_nar_num_queries: block_size / downsample_rate (3).
int granite_nar_num_queries(const GraniteNarHParams &hp);

// arch model.cpp granite_nar_max_audio_ms (advisory; 0 = unknown).
int64_t granite_nar_max_audio_ms(const GraniteNarHParams &hp);

} // namespace engine::models::granite_nar
