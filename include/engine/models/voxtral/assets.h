#pragma once

// engine/models/voxtral/assets.h - host-side resources for the native engine
// Voxtral (2507, offline) package.
//
// One distribution format: parent transcribe.cpp's GGUF
// (`general.architecture = "voxtral"`, `stt.voxtral.*` / `stt.frontend.*` KVs,
// a Mistral tekken vocabulary stored llama.cpp-style as a GPT-2 byte-level BPE
// under `tokenizer.ggml.*` with `tokenizer.ggml.pre = "tekken"`, and the
// Whisper mel filterbank + window baked in as `frontend.*` tensors), published
// as huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf and
// huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf. No audio.cpp
// layout of this model exists, so this is the only layout the package reads.
//
// Every KV and tensor name read here is one the retired arch read
// (src/runtime/arch/voxtral/weights.cpp: read_voxtral_hparams /
// build_voxtral_weights; model.cpp: load / resolve_specials); nothing is
// invented, and every rejection the arch made is made here with the same
// meaning (std::runtime_error instead of TRANSCRIBE_ERR_GGUF).

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::models::voxtral {

// KV hyperparameters (the arch's VoxtralHParams, field for field).
struct VoxtralHParams {
  // Audio encoder (Whisper-large-v3 encoder).
  int32_t enc_n_layers = 0;
  int32_t enc_d_model = 0;
  int32_t enc_n_heads = 0;
  int32_t enc_head_dim = 0;
  int32_t enc_ffn_dim = 0;
  int32_t enc_num_mel_bins = 0;
  int32_t enc_max_source_positions = 0;  // 1500
  std::string enc_activation;            // "gelu"

  // Projector (multi-modal).
  int32_t proj_downsample = 0;  // 4x frame grouping
  int32_t proj_in = 0;          // enc_d_model * downsample (5120)
  std::string proj_hidden_act;  // "gelu"

  // Text LM (Llama / Ministral).
  int32_t dec_n_layers = 0;
  int32_t dec_hidden = 0;
  int32_t dec_intermediate = 0;
  int32_t dec_n_heads = 0;
  int32_t dec_n_kv_heads = 0;
  int32_t dec_head_dim = 0;
  std::string dec_hidden_act;  // "silu"
  float dec_rms_norm_eps = 0.0f;
  float dec_rope_theta = 0.0f;
  int32_t dec_max_position_embeddings = 0;
  bool dec_tie_word_embeddings = false;  // UNTIED for Voxtral (read, not used)
  int32_t dec_vocab_size = 0;

  // Audio-token injection id ([AUDIO] placeholder, 24).
  int32_t audio_token_id = 0;

  // Tokenizer-derived.
  int32_t bos_token_id = -1;
  int32_t eos_token_id = -1;
  int32_t vocab_size = 0;

  // Frontend (Whisper feature extractor).
  std::string fe_type;
  int32_t fe_num_mels = 0;
  int32_t fe_sample_rate = 0;
  int32_t fe_n_fft = 0;
  int32_t fe_win_length = 0;
  int32_t fe_hop_length = 0;
  std::string fe_window;
  std::string fe_normalize;
  float fe_dither = 0.0f;
  float fe_pre_emphasis = 0.0f;
  float fe_f_min = 0.0f;
  float fe_f_max = 0.0f;
  std::string fe_pad_mode;
  bool fe_center = true;
  std::string fe_mel_norm;
  int32_t fe_chunk_length = 0;   // 30 (seconds)
  int32_t fe_n_samples = 0;      // 480000
  int32_t fe_nb_max_frames = 0;  // 3000

  // Audio tokens produced per 30 s chunk: encoder frames (1500) / downsample.
  int32_t audio_tokens_per_chunk() const noexcept {
    return (enc_max_source_positions > 0 && proj_downsample > 0)
               ? enc_max_source_positions / proj_downsample
               : 0;
  }
  // PCM samples per encoder chunk (arch run(): n_samples, else
  // chunk_length * sample_rate, else 30 s).
  int32_t samples_per_chunk() const noexcept;
  // Mel frames per chunk (arch run(): nb_max_frames, else samples / hop).
  int32_t frames_per_chunk() const noexcept;
};

// mistral-common transcription/instruct control tokens, resolved through the
// loaded vocabulary at load time (the arch's PromptSpecials / resolve_specials).
//
// Transcription template:
//   [BOS] [INST] [BEGIN_AUDIO] [AUDIO]*N [/INST] (lang:<l>)? [TRANSCRIBE]
// Instruct template (translate):
//   [BOS] [INST] [BEGIN_AUDIO] [AUDIO]*N BPE(instruction) [/INST]
struct PromptSpecials {
  int32_t bos = -1;
  int32_t inst = -1;         // [INST]
  int32_t begin_audio = -1;  // [BEGIN_AUDIO]
  int32_t end_inst = -1;     // [/INST]
  int32_t transcribe = -1;   // [TRANSCRIBE]
  int32_t eos = -1;
};

// The arch's transcribe::Tokenizer, restricted to what a "gpt2" GGUF vocab
// reaches (src/runtime/transcribe-tokenizer.cpp):
//   - decode: GPT-2 byte-unicode inverse per code point; a code point that is
//     not a byte glyph passes through as raw UTF-8 (control pieces such as
//     "[INST]" are therefore emitted literally, as the arch did);
//   - encode: pretokenize, then the arch's merge-rank BPE loop (lowest rank
//     first, leftmost on ties) with a per-code-point vocab fallback.
//
// Pretokenizer: the arch has no "tekken" flavor, so for
// tokenizer.ggml.pre = "tekken" it logged a warning and fell back to the
// Qwen2 regex. This class reproduces that exact fallback (NOT the GPT-2 regex
// engine::text::TokenizerHub would pick for an unknown pre), because the two
// split "lang:en" differently (":en" vs ":" + "en") and the language hint
// would then be a different token sequence than the arch fed the decoder.
class VoxtralTokenizer {
public:
  VoxtralTokenizer() = default;
  // `tokens` / `merges` are tokenizer.ggml.tokens / tokenizer.ggml.merges;
  // `pre` is tokenizer.ggml.pre ("" when absent). Throws std::runtime_error on
  // a malformed merge line (the arch's "merge %zu has no space separator").
  VoxtralTokenizer(std::vector<std::string> tokens, const std::vector<std::string> &merges,
                   std::string pre, int32_t bos_id, int32_t eos_id);

  size_t size() const noexcept { return tokens_.size(); }
  int32_t bos_id() const noexcept { return bos_id_; }
  int32_t eos_id() const noexcept { return eos_id_; }
  const std::string &pretokenizer() const noexcept { return pre_; }

  // Piece -> id (first occurrence wins, as the arch's piece_to_id_); -1 when absent.
  int32_t find(std::string_view piece) const;

  // Text -> ids (no BOS/EOS). Throws std::runtime_error where the arch's
  // encode() returned an error (no merges, byte-fallback piece not in vocab).
  std::vector<int32_t> encode(std::string_view text) const;

  // Ids -> UTF-8 text (out-of-range ids are skipped, as the arch did).
  std::string decode(const std::vector<int32_t> &ids) const;

private:
  std::vector<std::string> tokens_;
  std::unordered_map<std::string, int32_t> piece_to_id_;
  std::unordered_map<std::string, int32_t> merge_rank_;  // "left\x1Fright" -> rank
  std::string pre_;
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
};

// Input-length contract (docs/input-limits.md), identical to the arch: audio
// tokens + prompt + generation share the decoder context window
// (decoder.max_position_embeddings, optionally lowered by the session n_ctx).
inline constexpr int kDecodeBudgetMin = 448;  // decode-budget floor (Whisper's per-chunk cap)
inline constexpr int kGenReserve = kDecodeBudgetMin;  // up-front gate reserve
inline constexpr int kPromptOverhead = 16;  // advisory template overhead for max_audio_ms

struct VoxtralAssets {
  VoxtralHParams hparams;
  std::string variant;  // stt.variant, else "voxtral-mini-3b-2507"
  std::filesystem::path model_path;

  // Weights, through the shared TensorSource (transcribe.cpp tensor names:
  // enc.*, proj.*, dec.*).
  std::shared_ptr<const assets::TensorSource> source;

  VoxtralTokenizer tokenizer;
  PromptSpecials specials;

  // Capability KVs (general.languages, stt.capability.*,
  // stt.translation.target_languages); absent = empty / the arch's default.
  std::vector<std::string> languages;
  std::vector<std::string> translate_target_languages;
  bool supports_translate = true;         // arch apply_family_invariants default
  bool supports_language_detect = false;  // transcribe_capabilities default

  // Checkpoint frontend buffers; empty when the GGUF does not carry them (the
  // frontend then computes them, as the arch did).
  std::vector<float> mel_filterbank;  // [num_mels * (n_fft / 2 + 1)], mel-major
  std::vector<float> window;          // [win_length]

  // Advisory input bound (transcribe_capabilities::max_audio_ms in the arch)
  // and the rate it is derived from.
  int64_t max_audio_ms = 0;
  double ms_per_audio_token = 0.0;

  const VoxtralHParams &config() const noexcept { return hparams; }
};

// Loads a transcribe.cpp voxtral GGUF. Throws std::runtime_error on anything
// that is not a well-formed voxtral model (every rejection the arch's load()
// made, with its message).
std::shared_ptr<const VoxtralAssets> load_voxtral_assets(const std::filesystem::path &model_path);

// Cheap sniff used by the loader's can_load(): GGUF magic and
// general.architecture == "voxtral" (what transcribe.cpp's
// scripts/convert-voxtral.py writes via gguf_writer(path, "voxtral")).
bool looks_like_voxtral_gguf(const std::filesystem::path &path);

// Effective decoder context ceiling in tokens: the model's trained maximum,
// optionally lowered - never raised - by the session n_ctx knob
// (arch voxtral_context_ceiling).
int voxtral_context_ceiling(int n_ctx_knob, const VoxtralHParams &hp);

// The arch's voxtral_max_audio_ms (advisory; 0 = unknown).
int64_t voxtral_max_audio_ms(const VoxtralHParams &hp);

}  // namespace engine::models::voxtral
