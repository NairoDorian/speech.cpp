// engine/models/voxtral/assets.cpp - load a transcribe.cpp voxtral GGUF into
// VoxtralAssets (see assets.h).
//
// Ported from src/runtime/arch/voxtral/weights.cpp (read_voxtral_hparams) and
// the host half of src/runtime/arch/voxtral/model.cpp (load, resolve_specials,
// voxtral_max_audio_ms, voxtral_context_ceiling). The GPU half of load() (the
// tensor catalog, the backend buffer, the gate+up packing) lives in
// runtime.cpp, which binds weights per session like every engine package.

#include "engine/models/voxtral/assets.h"

#include "engine/framework/assets/gguf_metadata.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::voxtral {

namespace {

constexpr const char *kArchitecture = "voxtral";
constexpr const char *kDefaultVariant = "voxtral-mini-3b-2507";

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("voxtral: " + message);
}

// The arch's read_required_u32_kv: required, integer, non-negative.
int32_t require_u32(const assets::GgufMetadata &meta, const char *key) {
  if (!meta.has(key)) {
    fail(std::string("required KV \"") + key + "\" is missing from " + meta.path().string());
  }
  const int32_t value = meta.require_i32(key);
  if (value < 0) {
    fail(std::string("KV \"") + key + "\" is negative (" + std::to_string(value) + ")");
  }
  return value;
}

float require_f32(const assets::GgufMetadata &meta, const char *key) {
  if (!meta.has(key)) {
    fail(std::string("required KV \"") + key + "\" is missing from " + meta.path().string());
  }
  return meta.require_float(key);
}

std::string require_string(const assets::GgufMetadata &meta, const char *key) {
  if (!meta.has(key)) {
    fail(std::string("required KV \"") + key + "\" is missing from " + meta.path().string());
  }
  return meta.require_string(key);
}

// read_voxtral_hparams, KV for KV, with the same cross-field invariants.
VoxtralHParams read_hparams(const assets::GgufMetadata &meta) {
  VoxtralHParams hp;

  // Audio encoder.
  hp.enc_n_layers = require_u32(meta, "stt.voxtral.encoder.n_layers");
  hp.enc_d_model = require_u32(meta, "stt.voxtral.encoder.d_model");
  hp.enc_n_heads = require_u32(meta, "stt.voxtral.encoder.n_heads");
  hp.enc_head_dim = require_u32(meta, "stt.voxtral.encoder.head_dim");
  hp.enc_ffn_dim = require_u32(meta, "stt.voxtral.encoder.ffn_dim");
  hp.enc_num_mel_bins = require_u32(meta, "stt.voxtral.encoder.num_mel_bins");
  hp.enc_max_source_positions = require_u32(meta, "stt.voxtral.encoder.max_source_positions");
  hp.enc_activation = require_string(meta, "stt.voxtral.encoder.activation");

  // Projector.
  hp.proj_downsample = require_u32(meta, "stt.voxtral.projector.downsample_factor");
  hp.proj_in = require_u32(meta, "stt.voxtral.projector.input_dim");
  hp.proj_hidden_act = require_string(meta, "stt.voxtral.projector.hidden_act");

  // Text LM.
  hp.dec_n_layers = require_u32(meta, "stt.voxtral.decoder.n_layers");
  hp.dec_hidden = require_u32(meta, "stt.voxtral.decoder.hidden_size");
  hp.dec_intermediate = require_u32(meta, "stt.voxtral.decoder.intermediate_size");
  hp.dec_n_heads = require_u32(meta, "stt.voxtral.decoder.n_heads");
  hp.dec_n_kv_heads = require_u32(meta, "stt.voxtral.decoder.n_kv_heads");
  hp.dec_head_dim = require_u32(meta, "stt.voxtral.decoder.head_dim");
  hp.dec_hidden_act = require_string(meta, "stt.voxtral.decoder.hidden_act");
  hp.dec_rms_norm_eps = require_f32(meta, "stt.voxtral.decoder.rms_norm_eps");
  hp.dec_rope_theta = require_f32(meta, "stt.voxtral.decoder.rope_theta");
  hp.dec_max_position_embeddings = require_u32(meta, "stt.voxtral.decoder.max_position_embeddings");
  hp.dec_tie_word_embeddings =
      meta.find_bool("stt.voxtral.decoder.tie_word_embeddings").value_or(false);
  hp.dec_vocab_size = require_u32(meta, "stt.voxtral.decoder.vocab_size");

  // Audio-token injection id.
  hp.audio_token_id = require_u32(meta, "stt.voxtral.audio_token_id");

  // Frontend.
  hp.fe_type = require_string(meta, "stt.frontend.type");
  hp.fe_num_mels = require_u32(meta, "stt.frontend.num_mels");
  hp.fe_sample_rate = require_u32(meta, "stt.frontend.sample_rate");
  hp.fe_n_fft = require_u32(meta, "stt.frontend.n_fft");
  hp.fe_win_length = require_u32(meta, "stt.frontend.win_length");
  hp.fe_hop_length = require_u32(meta, "stt.frontend.hop_length");
  hp.fe_window = require_string(meta, "stt.frontend.window");
  hp.fe_normalize = require_string(meta, "stt.frontend.normalize");
  hp.fe_dither = require_f32(meta, "stt.frontend.dither");
  hp.fe_pre_emphasis = require_f32(meta, "stt.frontend.pre_emphasis");
  hp.fe_f_min = require_f32(meta, "stt.frontend.f_min");
  hp.fe_f_max = require_f32(meta, "stt.frontend.f_max");
  hp.fe_pad_mode = meta.find_string("stt.frontend.pad_mode").value_or("reflect");
  hp.fe_mel_norm = meta.find_string("stt.frontend.mel_norm").value_or("slaney");
  hp.fe_center = meta.find_bool("stt.frontend.center").value_or(true);
  // Optional u32s: absent keeps 0, a mistyped value throws (GgufMetadata).
  if (const auto v = meta.find_i32("stt.frontend.chunk_length")) {
    hp.fe_chunk_length = *v;
  }
  if (const auto v = meta.find_i32("stt.frontend.n_samples")) {
    hp.fe_n_samples = *v;
  }
  if (const auto v = meta.find_i32("stt.frontend.nb_max_frames")) {
    hp.fe_nb_max_frames = *v;
  }

  // Cross-field invariants (read_voxtral_hparams).
  if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 || hp.enc_ffn_dim <= 0 ||
      hp.enc_num_mel_bins <= 0 || hp.enc_max_source_positions <= 0) {
    fail("encoder hparams must be positive");
  }
  if (hp.enc_d_model % hp.enc_n_heads != 0) {
    fail("encoder d_model (" + std::to_string(hp.enc_d_model) + ") not divisible by n_heads (" +
         std::to_string(hp.enc_n_heads) + ")");
  }
  if (hp.enc_head_dim != hp.enc_d_model / hp.enc_n_heads) {
    fail("encoder head_dim (" + std::to_string(hp.enc_head_dim) + ") != d_model/n_heads (" +
         std::to_string(hp.enc_d_model / hp.enc_n_heads) + ")");
  }
  if (hp.proj_downsample <= 0 || hp.proj_in <= 0) {
    fail("projector hparams must be positive");
  }
  if (hp.proj_in != hp.enc_d_model * hp.proj_downsample) {
    fail("projector input_dim (" + std::to_string(hp.proj_in) + ") != enc_d_model*downsample (" +
         std::to_string(hp.enc_d_model * hp.proj_downsample) + ")");
  }
  if (hp.enc_max_source_positions % hp.proj_downsample != 0) {
    fail("max_source_positions (" + std::to_string(hp.enc_max_source_positions) +
         ") not divisible by downsample (" + std::to_string(hp.proj_downsample) + ")");
  }
  if (hp.dec_n_layers <= 0 || hp.dec_hidden <= 0 || hp.dec_n_heads <= 0 || hp.dec_n_kv_heads <= 0 ||
      hp.dec_head_dim <= 0 || hp.dec_intermediate <= 0) {
    fail("decoder hparams must be positive");
  }
  if (hp.dec_n_heads % hp.dec_n_kv_heads != 0) {
    fail("n_heads (" + std::to_string(hp.dec_n_heads) + ") not divisible by n_kv_heads (" +
         std::to_string(hp.dec_n_kv_heads) + ")");
  }
  if (hp.dec_hidden_act != "silu" && hp.dec_hidden_act != "swish") {
    fail("unsupported decoder hidden_act \"" + hp.dec_hidden_act + "\" (only silu/swish)");
  }
  if (hp.enc_activation != "gelu") {
    fail("unsupported encoder activation \"" + hp.enc_activation + "\" (only gelu)");
  }
  if (hp.proj_hidden_act != "gelu") {
    fail("unsupported projector hidden_act \"" + hp.proj_hidden_act + "\" (only gelu)");
  }
  if (hp.fe_type != "mel") {
    fail("unsupported frontend type \"" + hp.fe_type + "\"");
  }
  // The engine frontend ports exactly the MelFrontend path the arch ran for
  // this family (see mel_frontend.h); reject configurations it does not carry
  // rather than silently computing different features.
  if (hp.fe_normalize != "per_utterance") {
    fail("unsupported frontend normalize \"" + hp.fe_normalize +
         "\" (the Voxtral Whisper frontend is per_utterance)");
  }
  if (hp.fe_pad_mode != "reflect" && hp.fe_pad_mode != "constant") {
    fail("unsupported frontend pad_mode \"" + hp.fe_pad_mode + "\" (reflect or constant)");
  }
  if (hp.fe_num_mels != hp.enc_num_mel_bins) {
    // The arch caught this per run ("mel bins %d != %d"); it is a property of
    // the file, so fail at load.
    fail("frontend num_mels (" + std::to_string(hp.fe_num_mels) + ") != encoder num_mel_bins (" +
         std::to_string(hp.enc_num_mel_bins) + ")");
  }
  if (hp.fe_sample_rate <= 0 || hp.fe_hop_length <= 0 || hp.fe_n_fft <= 0 ||
      hp.fe_win_length <= 0 || hp.fe_win_length > hp.fe_n_fft) {
    fail("frontend sample_rate / hop_length / n_fft / win_length are invalid");
  }
  return hp;
}

// The arch's Tokenizer::load for a "gpt2" GGUF vocab plus resolve_specials.
void load_tokenizer(const assets::GgufMetadata &meta, VoxtralAssets &out) {
  const auto model = meta.find_string("tokenizer.ggml.model");
  if (!model.has_value()) {
    fail("GGUF has no tokenizer.ggml.model");
  }
  if (*model != "gpt2") {
    // convert-voxtral.py always writes "gpt2" (llama.cpp's MistralVocab
    // representation of tekken); nothing else reaches this family.
    fail("unsupported tokenizer.ggml.model \"" + *model + "\" (expected gpt2)");
  }
  auto tokens = meta.find_string_array("tokenizer.ggml.tokens");
  if (!tokens.has_value() || tokens->empty()) {
    fail("GGUF has no tokenizer.ggml.tokens");
  }
  if (const auto scores = meta.find_f32_array("tokenizer.ggml.scores")) {
    if (scores->size() != tokens->size()) {
      fail("tokenizer.ggml.scores length does not match tokenizer.ggml.tokens");
    }
  }
  if (const auto types = meta.find_i32_array("tokenizer.ggml.token_type")) {
    if (types->size() != tokens->size()) {
      fail("tokenizer.ggml.token_type length does not match tokenizer.ggml.tokens");
    }
  }
  const auto merges = meta.find_string_array("tokenizer.ggml.merges").value_or(std::vector<std::string>{});
  // Absent pre is "qwen2" in the arch; "tekken" (what the converter writes)
  // is an unknown flavor there and falls back to qwen2 as well.
  const std::string pre = meta.find_string("tokenizer.ggml.pre").value_or("qwen2");
  const auto bos = meta.find_int("tokenizer.ggml.bos_token_id");
  const auto eos = meta.find_int("tokenizer.ggml.eos_token_id");
  const int32_t bos_id = bos.has_value() ? static_cast<int32_t>(*bos) : -1;
  const int32_t eos_id = eos.has_value() ? static_cast<int32_t>(*eos) : -1;

  const size_t n_tokens = tokens->size();
  out.tokenizer = VoxtralTokenizer(std::move(*tokens), merges, pre, bos_id, eos_id);

  auto &hp = out.hparams;
  hp.vocab_size = static_cast<int32_t>(n_tokens);
  hp.bos_token_id = bos_id;
  hp.eos_token_id = eos_id;
  if (hp.vocab_size != hp.dec_vocab_size) {
    fail("tokenizer vocab (" + std::to_string(hp.vocab_size) + ") != decoder vocab_size (" +
         std::to_string(hp.dec_vocab_size) + ")");
  }
  if (hp.eos_token_id < 0) {
    fail("GGUF tokenizer has no eos_token_id");
  }
  if (hp.audio_token_id >= hp.vocab_size) {
    fail("audio_token_id (" + std::to_string(hp.audio_token_id) + ") is outside the vocabulary");
  }

  // resolve_specials: a vocab reorder fails loudly here, not mid-decode.
  struct PieceSlot {
    const char *piece;
    int32_t *slot;
  };
  PromptSpecials &s = out.specials;
  const PieceSlot pieces[] = {
      {"[INST]", &s.inst},
      {"[BEGIN_AUDIO]", &s.begin_audio},
      {"[/INST]", &s.end_inst},
      {"[TRANSCRIBE]", &s.transcribe},
  };
  for (const auto &p : pieces) {
    const int32_t id = out.tokenizer.find(p.piece);
    if (id < 0) {
      fail(std::string("control token \"") + p.piece + "\" not in tokenizer");
    }
    *p.slot = id;
  }
  s.bos = out.tokenizer.bos_id();
  s.eos = out.tokenizer.eos_id();
  if (s.bos < 0 || s.eos < 0) {
    fail("tokenizer missing bos/eos id");
  }
}

// The arch's read_f32_tensor_checked for the baked frontend buffers: absent is
// fine (the frontend computes the buffer), a size mismatch is fatal. The
// element order is the file's (row-major numpy [n_mels, n_freq] for the
// filterbank), exactly what the arch copied into MelConfig.
std::vector<float> read_frontend_buffer(const assets::TensorSource &source, const char *name,
                                        size_t expected_elems) {
  if (!source.has_tensor(name)) {
    return {};
  }
  std::vector<float> values = source.require_f32(name);
  if (values.size() != expected_elems) {
    fail(std::string("tensor \"") + name + "\" has " + std::to_string(values.size()) +
         " elements, expected " + std::to_string(expected_elems));
  }
  return values;
}

std::string read_magic(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  char magic[4] = {};
  if (!in.read(magic, 4)) {
    return {};
  }
  return std::string(magic, 4);
}

}  // namespace

int32_t VoxtralHParams::samples_per_chunk() const noexcept {
  // arch run(): stt.frontend.n_samples, else chunk_length (default 30 s) * sr.
  if (fe_n_samples > 0) {
    return fe_n_samples;
  }
  return (fe_chunk_length > 0 ? fe_chunk_length : 30) * fe_sample_rate;
}

int32_t VoxtralHParams::frames_per_chunk() const noexcept {
  if (fe_nb_max_frames > 0) {
    return fe_nb_max_frames;
  }
  return fe_hop_length > 0 ? samples_per_chunk() / fe_hop_length : 0;
}

int voxtral_context_ceiling(int n_ctx_knob, const VoxtralHParams &hp) {
  int ceiling = hp.dec_max_position_embeddings;
  if (n_ctx_knob > 0 && n_ctx_knob < ceiling) {
    ceiling = n_ctx_knob;
  }
  return ceiling;
}

int64_t voxtral_max_audio_ms(const VoxtralHParams &hp) {
  const int tokens_per_chunk = hp.audio_tokens_per_chunk();
  if (tokens_per_chunk <= 0 || hp.fe_nb_max_frames <= 0 || hp.fe_hop_length <= 0 ||
      hp.fe_sample_rate <= 0 || hp.dec_max_position_embeddings <= 0) {
    return 0;
  }
  const int max_audio_tokens = hp.dec_max_position_embeddings - kPromptOverhead - kGenReserve;
  if (max_audio_tokens <= 0) {
    return 0;
  }
  // One 30 s chunk = fe_nb_max_frames mel frames; chunk_ms = frames * hop_ms.
  const int64_t chunk_ms = static_cast<int64_t>(hp.fe_nb_max_frames) * hp.fe_hop_length * 1000 /
                           hp.fe_sample_rate;
  return static_cast<int64_t>(max_audio_tokens) * chunk_ms / tokens_per_chunk;
}

bool looks_like_voxtral_gguf(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    return false;
  }
  if (read_magic(path) != "GGUF") {
    return false;
  }
  try {
    const auto meta = assets::GgufMetadata::open(path);
    return meta.find_string("general.architecture").value_or("") == kArchitecture;
  } catch (const std::exception &) {
    return false;
  }
}

std::shared_ptr<const VoxtralAssets> load_voxtral_assets(const std::filesystem::path &model_path) {
  if (read_magic(model_path) != "GGUF") {
    fail("not a GGUF file: " + model_path.string());
  }
  auto out = std::make_shared<VoxtralAssets>();
  out->model_path = model_path;

  const auto meta = assets::GgufMetadata::open(model_path);
  if (meta.find_string("general.architecture").value_or("") != kArchitecture) {
    fail("GGUF general.architecture is not \"voxtral\": " + model_path.string());
  }
  out->variant = meta.find_string("stt.variant").value_or(kDefaultVariant);
  if (out->variant.empty()) {
    out->variant = kDefaultVariant;
  }

  // Capability KVs (read_capability_kv / read_languages_kv): absent keeps the
  // family default.
  out->supports_translate = meta.find_bool("stt.capability.translate").value_or(true);
  out->supports_language_detect = meta.find_bool("stt.capability.lang_detect").value_or(false);
  out->languages = meta.find_string_array("general.languages").value_or(std::vector<std::string>{});
  out->translate_target_languages =
      meta.find_string_array("stt.translation.target_languages").value_or(std::vector<std::string>{});

  out->hparams = read_hparams(meta);
  load_tokenizer(meta, *out);

  // Advisory input limit and the rate the decode budget is derived from.
  const auto &hp = out->hparams;
  out->max_audio_ms = voxtral_max_audio_ms(hp);
  if (hp.audio_tokens_per_chunk() > 0 && hp.fe_nb_max_frames > 0) {
    const double chunk_ms = static_cast<double>(hp.fe_nb_max_frames) * hp.fe_hop_length * 1000.0 /
                            hp.fe_sample_rate;
    out->ms_per_audio_token = chunk_ms / hp.audio_tokens_per_chunk();
  }

  out->source = assets::open_tensor_source(model_path);
  const size_t fb_elems = static_cast<size_t>(hp.fe_num_mels) * static_cast<size_t>(hp.fe_n_fft / 2 + 1);
  out->mel_filterbank = read_frontend_buffer(*out->source, "frontend.mel_filterbank", fb_elems);
  out->window = read_frontend_buffer(*out->source, "frontend.window", static_cast<size_t>(hp.fe_win_length));
  return out;
}

}  // namespace engine::models::voxtral
