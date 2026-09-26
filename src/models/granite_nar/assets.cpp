// engine/models/granite_nar/assets.cpp - model resources for the native
// engine Granite Speech NAR package, from parent transcribe.cpp's GGUF.
//
// Hyperparameters: exactly the KVs of src/runtime/arch/granite_nar/weights.cpp
// read_granite_nar_hparams, in the same order, with the same optional
// defaults and cross-field invariants. Tokenizer, capability / language KVs
// and frontend buffers: the arch's model.cpp load().

#include "engine/models/granite_nar/assets.h"

#include "engine/framework/assets/gguf_metadata.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::granite_nar {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747u; // "GGUF" little-endian
// The converter's gguf_writer(..., "granite_speech_nar") and the arch's
// Arch::name (model.cpp) - NOT the family directory name "granite_nar".
constexpr const char *kArchitecture = "granite_speech_nar";
// arch model.cpp k_default_variant.
constexpr const char *kDefaultVariant = "granite-speech-nar";

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("granite_nar: " + message);
}

uint32_t read_magic(const std::filesystem::path &path) {
  std::ifstream f(path, std::ios::binary);
  uint32_t magic = 0;
  if (f) {
    f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  }
  return f ? magic : 0u;
}

// read_granite_nar_hparams, key for key.
void read_hparams(const engine::assets::GgufMetadata &meta, GraniteNarHParams &hp) {
  const std::string p = "stt.granite_nar.";

  // Encoder.
  hp.enc_n_layers = meta.require_i32(p + "encoder.n_layers");
  hp.enc_hidden = meta.require_i32(p + "encoder.hidden");
  hp.enc_n_heads = meta.require_i32(p + "encoder.n_heads");
  hp.enc_head_dim = meta.require_i32(p + "encoder.head_dim");
  hp.enc_input_dim = meta.require_i32(p + "encoder.input_dim");
  hp.enc_output_dim = meta.require_i32(p + "encoder.output_dim");
  hp.enc_bpe_output_dim = meta.require_i32(p + "encoder.bpe_output_dim");
  hp.enc_bpe_pool_window = meta.require_i32(p + "encoder.bpe_pool_window");
  // Optional; the arch used read_required_u32_kv and fell back to 0 on ANY
  // failure (absent or mistyped), so a mistyped key is 0 here too.
  try {
    hp.enc_bpe_blank_id = meta.find_i32(p + "encoder.bpe_blank_id").value_or(0);
  } catch (const std::exception &) {
    hp.enc_bpe_blank_id = 0;
  }
  hp.enc_self_cond_layer = meta.require_i32(p + "encoder.self_cond_layer");
  hp.enc_feedforward_mult = meta.require_i32(p + "encoder.feedforward_mult");
  hp.enc_conv_kernel_size = meta.require_i32(p + "encoder.conv_kernel_size");
  hp.enc_conv_expansion = meta.require_i32(p + "encoder.conv_expansion");
  hp.enc_max_pos_emb = meta.require_i32(p + "encoder.max_pos_emb");
  hp.enc_context_size = meta.require_i32(p + "encoder.context_size");
  // read_int32_array_kv: absent -> {-1}, wrong type -> error (find_* throws).
  hp.enc_layer_indices =
      meta.find_i32_array(p + "encoder.layer_indices").value_or(std::vector<int32_t>{-1});

  // Projector.
  hp.prj_n_layers = meta.require_i32(p + "projector.n_layers");
  hp.prj_hidden = meta.require_i32(p + "projector.hidden");
  hp.prj_mlp_ratio = meta.require_i32(p + "projector.mlp_ratio");
  hp.prj_n_heads = meta.require_i32(p + "projector.n_heads");
  hp.prj_encoder_dim = meta.require_i32(p + "projector.encoder_dim");
  hp.prj_num_encoder_layers = meta.require_i32(p + "projector.num_encoder_layers");
  hp.prj_block_size = meta.require_i32(p + "projector.block_size");
  hp.prj_downsample_rate = meta.require_i32(p + "projector.downsample_rate");
  hp.prj_llm_dim = meta.require_i32(p + "projector.llm_dim");
  hp.prj_layernorm_eps = meta.require_float(p + "projector.layernorm_eps");
  hp.prj_attn_bias = meta.find_bool(p + "projector.attn_bias").value_or(true);
  hp.prj_mlp_bias = meta.find_bool(p + "projector.mlp_bias").value_or(true);
  hp.scale_projected_embeddings = meta.find_bool(p + "scale_projected_embeddings").value_or(true);

  // Text LM.
  hp.dec_n_layers = meta.require_i32(p + "text.n_layers");
  hp.dec_hidden = meta.require_i32(p + "text.hidden");
  hp.dec_intermediate = meta.require_i32(p + "text.intermediate");
  hp.dec_n_heads = meta.require_i32(p + "text.n_heads");
  hp.dec_n_kv_heads = meta.require_i32(p + "text.n_kv_heads");
  hp.dec_head_dim = meta.require_i32(p + "text.head_dim");
  hp.dec_hidden_act = meta.require_string(p + "text.hidden_act");
  hp.dec_rms_norm_eps = meta.require_float(p + "text.rms_norm_eps");
  hp.dec_rope_theta = meta.require_float(p + "text.rope_theta");
  hp.dec_max_pos_emb = meta.require_i32(p + "text.max_position_embeddings");
  hp.dec_tie_word_embeddings = meta.find_bool(p + "text.tie_word_embeddings").value_or(true);
  hp.dec_vocab_size = meta.require_i32(p + "text.vocab_size");
  hp.dec_embedding_multiplier = meta.require_float(p + "text.embedding_multiplier");
  hp.dec_logits_scaling = meta.require_float(p + "text.logits_scaling");
  hp.dec_attention_multiplier = meta.require_float(p + "text.attention_multiplier");
  hp.dec_residual_multiplier = meta.require_float(p + "text.residual_multiplier");
  hp.dec_bos_id = meta.require_i32(p + "text.bos_id");
  hp.dec_eos_id = meta.require_i32(p + "text.eos_id");
  hp.dec_pad_id = meta.require_i32(p + "text.pad_id");

  // Frontend.
  hp.fe_type = meta.require_string("stt.frontend.type");
  hp.fe_sample_rate = meta.require_i32("stt.frontend.sample_rate");
  hp.fe_num_mels = meta.require_i32("stt.frontend.num_mels");
  hp.fe_n_fft = meta.require_i32("stt.frontend.n_fft");
  hp.fe_win_length = meta.require_i32("stt.frontend.win_length");
  hp.fe_hop_length = meta.require_i32("stt.frontend.hop_length");
  hp.fe_window = meta.require_string("stt.frontend.window");
  hp.fe_normalize = meta.require_string("stt.frontend.normalize");
  hp.fe_pad_mode = meta.find_string("stt.frontend.pad_mode").value_or("reflect");
  hp.fe_mel_norm = meta.find_string("stt.frontend.mel_norm").value_or("htk");

  hp.ctc_chars = meta.find_string_array(p + "ctc_chars").value_or(std::vector<std::string>{});

  // The arch's two cross-field invariants.
  if (hp.prj_num_encoder_layers != static_cast<int32_t>(hp.enc_layer_indices.size())) {
    fail("prj_num_encoder_layers (" + std::to_string(hp.prj_num_encoder_layers) +
         ") != len(enc_layer_indices) (" + std::to_string(hp.enc_layer_indices.size()) + ")");
  }
  if (hp.dec_hidden != hp.prj_llm_dim) {
    fail("dec_hidden (" + std::to_string(hp.dec_hidden) + ") != prj_llm_dim (" +
         std::to_string(hp.prj_llm_dim) + ")");
  }

  // Not checked by the arch at load, but it divided by / indexed with every
  // one of these while building graphs (projector.cpp checks prj_hidden %
  // n_heads per run); refuse the file up front instead.
  if (hp.enc_n_layers < 2 || hp.enc_hidden <= 0 || hp.enc_n_heads <= 0 || hp.enc_head_dim <= 0 ||
      hp.enc_input_dim <= 0 || hp.enc_output_dim <= 0 || hp.enc_context_size <= 0 ||
      hp.enc_max_pos_emb <= 0 || hp.enc_conv_kernel_size <= 0 || hp.enc_conv_expansion <= 0 ||
      hp.enc_feedforward_mult <= 0 || hp.enc_bpe_pool_window <= 0) {
    fail("non-positive encoder geometry in stt.granite_nar.encoder.*");
  }
  if (hp.prj_n_layers <= 0 || hp.prj_hidden <= 0 || hp.prj_n_heads <= 0 ||
      hp.prj_hidden % hp.prj_n_heads != 0 || hp.prj_block_size <= 0 ||
      hp.prj_downsample_rate <= 0 || hp.prj_block_size % hp.prj_downsample_rate != 0 ||
      hp.prj_encoder_dim != hp.enc_hidden) {
    fail("invalid projector geometry in stt.granite_nar.projector.*");
  }
  if (hp.dec_n_layers <= 0 || hp.dec_n_heads <= 0 || hp.dec_n_kv_heads <= 0 ||
      hp.dec_n_heads % hp.dec_n_kv_heads != 0 || hp.dec_head_dim <= 0 ||
      hp.dec_max_pos_emb <= 0 || hp.dec_vocab_size <= 0) {
    fail("invalid text-LM geometry in stt.granite_nar.text.*");
  }
  if (hp.fe_sample_rate <= 0 || hp.fe_num_mels <= 0 || hp.fe_n_fft <= 0 ||
      hp.fe_win_length <= 0 || hp.fe_win_length > hp.fe_n_fft || hp.fe_hop_length <= 0 ||
      hp.enc_input_dim != 2 * hp.fe_num_mels) {
    fail("invalid frontend geometry in stt.frontend.* (encoder.input_dim must be 2 * num_mels)");
  }
}

// transcribe::load_common::read_f32_tensor_checked: absent is fine, present
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

int granite_nar_context_ceiling(int n_ctx_knob, const GraniteNarHParams &hp) {
  int ceiling = hp.dec_max_pos_emb;
  if (n_ctx_knob > 0 && n_ctx_knob < ceiling) {
    ceiling = n_ctx_knob;
  }
  return ceiling;
}

int granite_nar_num_queries(const GraniteNarHParams &hp) {
  if (hp.prj_downsample_rate > 0 && hp.prj_block_size > 0) {
    return hp.prj_block_size / hp.prj_downsample_rate;
  }
  return 3;
}

int64_t granite_nar_max_audio_ms(const GraniteNarHParams &hp) {
  const int num_queries = granite_nar_num_queries(hp);
  if (num_queries <= 0 || hp.prj_block_size <= 0 || hp.fe_hop_length <= 0 ||
      hp.fe_sample_rate <= 0 || hp.dec_max_pos_emb <= 0) {
    return 0;
  }
  const int max_audio_tokens = hp.dec_max_pos_emb - kTextBudget;
  if (max_audio_tokens <= 0) {
    return 0;
  }
  // tokens = nblocks * num_queries ; t_enc <= nblocks * block_size ;
  // mel_frames = t_enc * 2 ; ms = mel_frames * hop * 1000 / sample_rate.
  const int64_t nblocks = max_audio_tokens / num_queries;
  const int64_t t_enc = nblocks * hp.prj_block_size;
  const int64_t frames = t_enc * 2;
  return frames * hp.fe_hop_length * 1000 / hp.fe_sample_rate;
}

std::vector<int32_t> GraniteNarAssets::encode(std::string_view text) const {
  return tokenizer->encode(text);
}

std::string GraniteNarAssets::decode(const std::vector<int32_t> &ids) const {
  return tokenizer->decode(ids);
}

bool looks_like_granite_nar_gguf(const std::filesystem::path &path) {
  if (read_magic(path) != kGgufMagic) {
    return false;
  }
  try {
    const auto meta = engine::assets::GgufMetadata::open(path);
    return meta.find_string("general.architecture").value_or("") == kArchitecture;
  } catch (const std::exception &) {
    return false;
  }
}

std::shared_ptr<const GraniteNarAssets>
load_granite_nar_assets(const std::filesystem::path &model_path) {
  if (!std::filesystem::exists(model_path)) {
    fail("model file not found: " + model_path.string());
  }
  if (read_magic(model_path) != kGgufMagic) {
    fail("not a GGUF file: " + model_path.string());
  }
  auto out = std::make_shared<GraniteNarAssets>();
  out->model_path = model_path;

  const auto meta = engine::assets::GgufMetadata::open(model_path);
  if (meta.find_string("general.architecture").value_or("") != kArchitecture) {
    fail(std::string("GGUF general.architecture is not \"") + kArchitecture +
         "\": " + model_path.string());
  }

  // read_capability_kv / read_languages_kv (absent = off / unknown). The
  // family invariants set translate off first; the KV may override it.
  out->languages = meta.find_string_array("general.languages").value_or(std::vector<std::string>{});
  out->supports_translate = meta.find_bool("stt.capability.translate").value_or(false);
  out->supports_language_detect = meta.find_bool("stt.capability.lang_detect").value_or(false);

  // Tokenizer (granite-4 byte-level BPE, "gpt2" tag).
  out->tokenizer = text::load_tokenizer_from_gguf(meta.context());
  if (!out->tokenizer || out->tokenizer->vocab_size() == 0) {
    fail("GGUF has no tokenizer.ggml.tokens");
  }

  auto &hp = out->hparams;
  read_hparams(meta, hp);

  // model.cpp load(): bos / eos come from the tokenizer, overriding the
  // stt.granite_nar.text.{bos,eos}_id KVs; eos is the insertion-slot filler.
  hp.dec_bos_id = out->tokenizer->specials().bos;
  hp.dec_eos_id = out->tokenizer->specials().eos;
  if (hp.dec_eos_id < 0) {
    fail("tokenizer has no eos_token_id");
  }
  if (static_cast<int64_t>(out->tokenizer->vocab_size()) != hp.dec_vocab_size) {
    fail("tokenizer vocab (" + std::to_string(out->tokenizer->vocab_size()) +
         ") != dec_vocab_size (" + std::to_string(hp.dec_vocab_size) + ")");
  }

  out->max_audio_ms = granite_nar_max_audio_ms(hp);

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

} // namespace engine::models::granite_nar
