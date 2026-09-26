// engine/models/canary_qwen/assets.cpp - model resources for the native
// engine Canary-Qwen package, from parent transcribe.cpp's GGUF.
//
// Hyperparameters come from exactly the KVs the arch read
// (src/runtime/arch/canary_qwen/weights.cpp, read_canary_qwen_hparams) with
// the same validation; the tokenizer from tokenizer.ggml.* (Qwen2 byte-level
// BPE, TokenizerHub); the chat-template prompt segments, capability KVs and
// frontend buffers as the arch's load() built them.

#include "engine/models/canary_qwen/assets.h"

#include "engine/framework/assets/gguf_metadata.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::canary_qwen {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747u; // "GGUF" little-endian
constexpr const char *kDefaultVariant = "canary-qwen-2.5b";

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("canary_qwen: " + message);
}

uint32_t read_magic(const std::filesystem::path &path) {
  std::ifstream f(path, std::ios::binary);
  uint32_t magic = 0;
  if (f) {
    f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  }
  return f ? magic : 0u;
}

void read_hparams(const engine::assets::GgufMetadata &meta, CanaryQwenHParams &hp) {
  // Encoder.
  hp.enc_n_layers = meta.require_i32("stt.canary_qwen.encoder.n_layers");
  hp.enc_d_model = meta.require_i32("stt.canary_qwen.encoder.d_model");
  hp.enc_n_heads = meta.require_i32("stt.canary_qwen.encoder.n_heads");
  hp.enc_d_ff = meta.require_i32("stt.canary_qwen.encoder.d_ff");
  hp.enc_conv_kernel = meta.require_i32("stt.canary_qwen.encoder.conv_kernel");
  hp.enc_subsampling_factor = meta.require_i32("stt.canary_qwen.encoder.subsampling_factor");
  hp.enc_subsampling_chans = meta.require_i32("stt.canary_qwen.encoder.subsampling_channels");
  hp.enc_pos_emb_max_len = meta.require_i32("stt.canary_qwen.encoder.pos_emb_max_len");

  // Perception. The arch fell back to the LM hidden size when output_dim was
  // unreadable (older converters); keep that.
  try {
    hp.perception_output_dim = meta.find_i32("stt.canary_qwen.perception.output_dim").value_or(0);
  } catch (const std::exception &) {
    hp.perception_output_dim = 0;
  }
  const auto locator = meta.find_i32("stt.canary_qwen.perception.audio_locator_id");
  if (!locator.has_value()) {
    fail("missing or invalid stt.canary_qwen.perception.audio_locator_id");
  }
  hp.audio_locator_id = *locator;

  // Decoder (Qwen3).
  hp.dec_n_layers = meta.require_i32("stt.canary_qwen.decoder.n_layers");
  hp.dec_hidden = meta.require_i32("stt.canary_qwen.decoder.hidden_size");
  hp.dec_intermediate = meta.require_i32("stt.canary_qwen.decoder.intermediate_size");
  hp.dec_n_heads = meta.require_i32("stt.canary_qwen.decoder.n_heads");
  hp.dec_n_kv_heads = meta.require_i32("stt.canary_qwen.decoder.n_kv_heads");
  hp.dec_head_dim = meta.require_i32("stt.canary_qwen.decoder.head_dim");
  hp.dec_max_position = meta.require_i32("stt.canary_qwen.decoder.max_position_embeddings");
  hp.dec_vocab_size = meta.require_i32("stt.canary_qwen.decoder.vocab_size");
  hp.dec_rms_norm_eps = meta.require_float("stt.canary_qwen.decoder.rms_norm_eps");
  hp.dec_rope_theta = meta.require_float("stt.canary_qwen.decoder.rope_theta");
  hp.dec_tie_word_embeddings =
      meta.find_bool("stt.canary_qwen.decoder.tie_word_embeddings").value_or(true);

  if (hp.perception_output_dim == 0) {
    hp.perception_output_dim = hp.dec_hidden;
  }

  // Frontend.
  hp.fe_sample_rate = meta.require_i32("stt.frontend.sample_rate");
  hp.fe_num_mels = meta.require_i32("stt.frontend.num_mels");
  hp.fe_n_fft = meta.require_i32("stt.frontend.n_fft");
  hp.fe_win_length = meta.require_i32("stt.frontend.win_length");
  hp.fe_hop_length = meta.require_i32("stt.frontend.hop_length");
  hp.fe_pre_emphasis = meta.require_float("stt.frontend.pre_emphasis");
  hp.fe_f_min = meta.require_float("stt.frontend.f_min");
  hp.fe_f_max = meta.require_float("stt.frontend.f_max");
  hp.fe_pad_mode = meta.require_string("stt.frontend.pad_mode");
  hp.fe_normalize = meta.require_string("stt.frontend.normalize");

  // Cross-field validation (the arch's, same order and meaning).
  if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 ||
      hp.enc_d_model % hp.enc_n_heads != 0 || hp.dec_n_layers <= 0 || hp.dec_n_heads <= 0 ||
      hp.dec_n_kv_heads <= 0 || hp.dec_head_dim <= 0 || hp.fe_hop_length <= 0 ||
      hp.fe_sample_rate <= 0 || hp.fe_n_fft <= 0 || hp.fe_win_length <= 0) {
    fail("non-positive geometry in stt.canary_qwen.* / stt.frontend.*");
  }
  if (hp.dec_n_heads % hp.dec_n_kv_heads != 0) {
    fail("dec.n_heads (" + std::to_string(hp.dec_n_heads) + ") must be divisible by dec.n_kv_heads (" +
         std::to_string(hp.dec_n_kv_heads) + ")");
  }
  if (hp.dec_n_heads * hp.dec_head_dim != hp.dec_hidden) {
    fail("dec.n_heads * dec.head_dim != dec.hidden");
  }
  if (hp.perception_output_dim != hp.dec_hidden) {
    fail("perception.output_dim (" + std::to_string(hp.perception_output_dim) +
         ") must equal dec.hidden (" + std::to_string(hp.dec_hidden) + ")");
  }
  if (hp.audio_locator_id < 0) {
    fail("audio_locator_id must be >= 0");
  }
  if (hp.enc_subsampling_factor != 8 || hp.fe_num_mels != 128) {
    fail("only subsampling_factor=8 num_mels=128 supported (got " +
         std::to_string(hp.enc_subsampling_factor) + ", " + std::to_string(hp.fe_num_mels) + ")");
  }
}

// arch model.cpp: resolve_chat_tokens + build_static_prompt_segments.
void build_prompt_segments(CanaryQwenAssets &out) {
  const auto &tok = *out.tokenizer;
  struct PieceSlot {
    const char *piece;
    int32_t *slot;
  };
  const PieceSlot pieces[] = {
      {"<|im_start|>", &out.chat_tokens.im_start},
      {"<|im_end|>", &out.chat_tokens.im_end},
      {"user", &out.chat_tokens.role_user},
      {"assistant", &out.chat_tokens.role_assistant},
  };
  for (const auto &p : pieces) {
    const auto id = tok.find(p.piece);
    if (!id.has_value() || *id < 0) {
      fail(std::string("chat-template piece \"") + p.piece + "\" not in tokenizer");
    }
    *p.slot = *id;
  }

  auto encode = [&](const char *text) {
    std::vector<int32_t> ids = tok.encode(text);
    if (ids.empty()) {
      fail(std::string("tokenizer could not encode the chat template piece \"") + text + "\"");
    }
    return ids;
  };

  // prefix = [<|im_start|>] + bpe("user\nTranscribe the following: ")
  out.prompt_prefix_ids.clear();
  out.prompt_prefix_ids.push_back(out.chat_tokens.im_start);
  {
    const auto ids = encode("user\nTranscribe the following: ");
    out.prompt_prefix_ids.insert(out.prompt_prefix_ids.end(), ids.begin(), ids.end());
  }
  // suffix = [<|im_end|>] + bpe("\n") + [<|im_start|>] + bpe("assistant\n")
  out.prompt_suffix_ids.clear();
  out.prompt_suffix_ids.push_back(out.chat_tokens.im_end);
  {
    const auto ids = encode("\n");
    out.prompt_suffix_ids.insert(out.prompt_suffix_ids.end(), ids.begin(), ids.end());
  }
  out.prompt_suffix_ids.push_back(out.chat_tokens.im_start);
  {
    const auto ids = encode("assistant\n");
    out.prompt_suffix_ids.insert(out.prompt_suffix_ids.end(), ids.begin(), ids.end());
  }
}

// The arch's load_common::read_f32_tensor_checked: absent is fine, present
// with the wrong element count is a malformed file.
std::vector<float> optional_frontend_tensor(const engine::assets::TensorSource &source,
                                            const char *name, size_t expected_elems) {
  if (!source.has_tensor(name)) {
    return {};
  }
  std::vector<float> values = source.require_f32(name, std::optional<std::vector<int64_t>>{});
  if (values.size() != expected_elems) {
    fail(std::string(name) + " has " + std::to_string(values.size()) + " elements, expected " +
         std::to_string(expected_elems));
  }
  return values;
}

} // namespace

int canary_qwen_context_ceiling(int n_ctx_knob, const CanaryQwenHParams &hp) {
  int ceiling = hp.dec_max_position;
  if (n_ctx_knob > 0 && n_ctx_knob < ceiling) {
    ceiling = n_ctx_knob;
  }
  return ceiling;
}

int64_t canary_qwen_max_audio_ms(const CanaryQwenHParams &hp) {
  if (hp.dec_max_position <= 0 || hp.enc_subsampling_factor <= 0 || hp.fe_hop_length <= 0 ||
      hp.fe_sample_rate <= 0) {
    return 0;
  }
  const int max_audio_tokens = hp.dec_max_position - kPromptOverhead - kGenReserve;
  if (max_audio_tokens <= 0) {
    return 0;
  }
  const int64_t mel_frames = static_cast<int64_t>(max_audio_tokens) * hp.enc_subsampling_factor;
  return mel_frames * hp.fe_hop_length * 1000 / hp.fe_sample_rate;
}

std::vector<int32_t> CanaryQwenAssets::encode(std::string_view text) const {
  return tokenizer->encode(text);
}

std::string CanaryQwenAssets::decode(const std::vector<int32_t> &ids) const {
  return tokenizer->decode(ids);
}

bool looks_like_canary_qwen_gguf(const std::filesystem::path &path) {
  if (read_magic(path) != kGgufMagic) {
    return false;
  }
  try {
    const auto meta = engine::assets::GgufMetadata::open(path);
    return meta.find_string("general.architecture").value_or("") == "canary_qwen";
  } catch (const std::exception &) {
    return false;
  }
}

std::shared_ptr<const CanaryQwenAssets>
load_canary_qwen_assets(const std::filesystem::path &model_path) {
  if (!std::filesystem::exists(model_path)) {
    fail("model file not found: " + model_path.string());
  }
  if (read_magic(model_path) != kGgufMagic) {
    fail("not a GGUF file: " + model_path.string());
  }
  auto out = std::make_shared<CanaryQwenAssets>();
  out->model_path = model_path;

  const auto meta = engine::assets::GgufMetadata::open(model_path);
  if (meta.find_string("general.architecture").value_or("") != "canary_qwen") {
    fail("GGUF general.architecture is not \"canary_qwen\": " + model_path.string());
  }

  // Capabilities (read_capability_kv / read_languages_kv; absent = off).
  out->languages = meta.find_string_array("general.languages").value_or(std::vector<std::string>{});
  out->supports_translate = meta.find_bool("stt.capability.translate").value_or(false);
  out->supports_language_detect = meta.find_bool("stt.capability.lang_detect").value_or(false);

  // Tokenizer (Qwen2 byte-level BPE; encode needs the merges).
  out->tokenizer = text::load_tokenizer_from_gguf(meta.context());
  if (!out->tokenizer || out->tokenizer->model() != text::TokenizerModel::ByteLevelBpe) {
    fail("GGUF tokenizer is not a byte-level BPE (tokenizer.ggml.model = \"gpt2\")");
  }

  auto &hp = out->hparams;
  read_hparams(meta, hp);

  out->max_audio_ms = canary_qwen_max_audio_ms(hp);
  out->ms_per_audio_token = static_cast<double>(hp.enc_subsampling_factor) * hp.fe_hop_length *
                            1000.0 / hp.fe_sample_rate;

  hp.vocab_size = static_cast<int32_t>(out->tokenizer->vocab_size());
  hp.bos_token_id = out->tokenizer->specials().bos;
  hp.eos_token_id = out->tokenizer->specials().eos;
  if (hp.vocab_size != hp.dec_vocab_size) {
    fail("tokenizer vocab (" + std::to_string(hp.vocab_size) + ") != decoder vocab_size (" +
         std::to_string(hp.dec_vocab_size) + ")");
  }
  if (hp.eos_token_id < 0) {
    fail("GGUF tokenizer has no eos_token_id");
  }

  build_prompt_segments(*out);

  out->variant = meta.find_string("stt.variant").value_or(kDefaultVariant);
  if (out->variant.empty()) {
    out->variant = kDefaultVariant;
  }

  out->source = engine::assets::open_tensor_source(model_path);
  const size_t fb_elems =
      static_cast<size_t>(hp.fe_num_mels) * static_cast<size_t>(hp.fe_n_fft / 2 + 1);
  out->mel_filterbank = optional_frontend_tensor(*out->source, "frontend.mel_filterbank", fb_elems);
  out->window = optional_frontend_tensor(*out->source, "frontend.window",
                                         static_cast<size_t>(hp.fe_win_length));
  return out;
}

} // namespace engine::models::canary_qwen
