// engine/models/moonshine/assets.cpp - GGUF metadata + tokenizer + tensor
// catalog for the native engine Moonshine package.
//
// Hparams come from the same stt.moonshine.* GGUF metadata keys the arch
// loader reads (the pinned packages carry no embedded config.json sidecar);
// the tokenizer comes from the tokenizer.ggml.* vocab via the Phase-9
// TokenizerHub; weights stream through the engine TensorSource abstraction.

#include "engine/models/moonshine/assets.h"

#include "engine/framework/model_spec/package.h"

#include <gguf.h>

#include <cmath>
#include <string>
#include <vector>

namespace engine::models::moonshine {

namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("moonshine: " + message);
}

int32_t read_u32_kv(const gguf_context *gguf, const char *key,
                    int32_t default_value, bool required,
                    const std::string &where) {
  const int64_t kid = gguf_find_key(gguf, key);
  if (kid < 0) {
    if (required) {
      fail(where + ": missing required GGUF metadata key: " + key);
    }
    return default_value;
  }
  const gguf_type type = gguf_get_kv_type(gguf, kid);
  switch (type) {
  case GGUF_TYPE_UINT32:
  case GGUF_TYPE_INT32:
    return static_cast<int32_t>(gguf_get_val_u32(gguf, kid));
  case GGUF_TYPE_UINT64:
  case GGUF_TYPE_INT64:
    return static_cast<int32_t>(gguf_get_val_u64(gguf, kid));
  default:
    fail(where + ": GGUF metadata key " + key + " has unexpected type");
  }
}

float read_f32_kv(const gguf_context *gguf, const char *key,
                  float default_value, bool required,
                  const std::string &where) {
  const int64_t kid = gguf_find_key(gguf, key);
  if (kid < 0) {
    if (required) {
      fail(where + ": missing required GGUF metadata key: " + key);
    }
    return default_value;
  }
  const gguf_type type = gguf_get_kv_type(gguf, kid);
  if (type != GGUF_TYPE_FLOAT32 && type != GGUF_TYPE_FLOAT64) {
    fail(where + ": GGUF metadata key " + key + " has unexpected type");
  }
  return type == GGUF_TYPE_FLOAT64
             ? static_cast<float>(gguf_get_val_f64(gguf, kid))
             : gguf_get_val_f32(gguf, kid);
}

bool read_bool_kv(const gguf_context *gguf, const char *key, bool default_value,
                  bool required, const std::string &where) {
  const int64_t kid = gguf_find_key(gguf, key);
  if (kid < 0) {
    if (required) {
      fail(where + ": missing required GGUF metadata key: " + key);
    }
    return default_value;
  }
  const gguf_type type = gguf_get_kv_type(gguf, kid);
  if (type == GGUF_TYPE_BOOL) {
    return gguf_get_val_bool(gguf, kid);
  }
  if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
    return gguf_get_val_u32(gguf, kid) != 0;
  }
  fail(where + ": GGUF metadata key " + key + " has unexpected type");
}

std::string read_string_kv(const gguf_context *gguf, const char *key,
                           bool required, const std::string &where) {
  const int64_t kid = gguf_find_key(gguf, key);
  if (kid < 0) {
    if (required) {
      fail(where + ": missing required GGUF metadata key: " + key);
    }
    return {};
  }
  if (gguf_get_kv_type(gguf, kid) != GGUF_TYPE_STRING) {
    fail(where + ": GGUF metadata key " + key + " is not a string");
  }
  return std::string(gguf_get_val_str(gguf, kid));
}

std::vector<int32_t> read_u32_array_kv(const gguf_context *gguf,
                                       const char *key,
                                       const std::string &where) {
  const int64_t kid = gguf_find_key(gguf, key);
  if (kid < 0) {
    fail(where + ": missing required GGUF metadata key: " + key);
  }
  if (gguf_get_kv_type(gguf, kid) != GGUF_TYPE_ARRAY) {
    fail(where + ": GGUF metadata key " + key + " is not an array");
  }
  const gguf_type elem = gguf_get_arr_type(gguf, kid);
  if (elem != GGUF_TYPE_UINT32 && elem != GGUF_TYPE_INT32) {
    fail(where + ": GGUF metadata key " + key +
         " array element type is not u32/i32");
  }
  const size_t n = gguf_get_arr_n(gguf, kid);
  const void *data = gguf_get_arr_data(gguf, kid);
  std::vector<int32_t> out(n);
  if (elem == GGUF_TYPE_UINT32) {
    const auto *src = static_cast<const uint32_t *>(data);
    for (size_t i = 0; i < n; ++i) {
      out[i] = static_cast<int32_t>(src[i]);
    }
  } else {
    const auto *src = static_cast<const int32_t *>(data);
    for (size_t i = 0; i < n; ++i) {
      out[i] = src[i];
    }
  }
  return out;
}

MoonshineHParams read_hparams(const gguf_context *gguf) {
  constexpr const char *kTag = "moonshine";
  MoonshineHParams hp;

  hp.enc_n_layers =
      read_u32_kv(gguf, "stt.moonshine.encoder.n_layers", 0, true, kTag);
  hp.enc_d_model =
      read_u32_kv(gguf, "stt.moonshine.encoder.d_model", 0, true, kTag);
  hp.enc_n_heads =
      read_u32_kv(gguf, "stt.moonshine.encoder.n_heads", 0, true, kTag);
  hp.enc_n_kv_heads = read_u32_kv(gguf, "stt.moonshine.encoder.n_kv_heads",
                                  hp.enc_n_heads, false, kTag);
  hp.enc_ffn_dim =
      read_u32_kv(gguf, "stt.moonshine.encoder.ffn_dim", 0, true, kTag);
  hp.enc_activation =
      read_string_kv(gguf, "stt.moonshine.encoder.activation", true, kTag);

  hp.dec_n_layers =
      read_u32_kv(gguf, "stt.moonshine.decoder.n_layers", 0, true, kTag);
  hp.dec_d_model =
      read_u32_kv(gguf, "stt.moonshine.decoder.d_model", 0, true, kTag);
  hp.dec_n_heads =
      read_u32_kv(gguf, "stt.moonshine.decoder.n_heads", 0, true, kTag);
  hp.dec_n_kv_heads = read_u32_kv(gguf, "stt.moonshine.decoder.n_kv_heads",
                                  hp.dec_n_heads, false, kTag);
  hp.dec_ffn_dim =
      read_u32_kv(gguf, "stt.moonshine.decoder.ffn_dim", 0, true, kTag);
  hp.dec_activation =
      read_string_kv(gguf, "stt.moonshine.decoder.activation", true, kTag);
  hp.dec_vocab_size =
      read_u32_kv(gguf, "stt.moonshine.decoder.vocab_size", 0, true, kTag);
  hp.dec_max_position_embeddings = read_u32_kv(
      gguf, "stt.moonshine.decoder.max_position_embeddings", 0, true, kTag);
  hp.dec_tie_word_embeddings = read_bool_kv(
      gguf, "stt.moonshine.decoder.tie_word_embeddings", true, false, kTag);

  hp.bos_token_id =
      read_u32_kv(gguf, "stt.moonshine.bos_token_id", -1, true, kTag);
  hp.eos_token_id =
      read_u32_kv(gguf, "stt.moonshine.eos_token_id", -1, true, kTag);
  hp.pad_token_id =
      read_u32_kv(gguf, "stt.moonshine.pad_token_id", -1, true, kTag);
  hp.decoder_start_token_id =
      read_u32_kv(gguf, "stt.moonshine.decoder_start_token_id", -1, true, kTag);

  hp.partial_rotary_factor = read_f32_kv(
      gguf, "stt.moonshine.partial_rotary_factor", 0.9f, false, kTag);
  hp.rope_theta =
      read_f32_kv(gguf, "stt.moonshine.rope_theta", 10000.0f, false, kTag);
  hp.attention_bias =
      read_bool_kv(gguf, "stt.moonshine.attention_bias", false, false, kTag);
  hp.pad_head_dim_multiple = read_u32_kv(
      gguf, "stt.moonshine.pad_head_dim_to_multiple_of", 0, false, kTag);

  hp.conv_channels =
      read_u32_array_kv(gguf, "stt.moonshine.conv_stem.channels", kTag);
  hp.conv_kernel_sizes =
      read_u32_array_kv(gguf, "stt.moonshine.conv_stem.kernel_sizes", kTag);
  hp.conv_strides =
      read_u32_array_kv(gguf, "stt.moonshine.conv_stem.strides", kTag);
  hp.conv_groupnorm_num_groups = read_u32_kv(
      gguf, "stt.moonshine.conv_stem.groupnorm_num_groups", 1, false, kTag);
  hp.conv_groupnorm_eps = read_f32_kv(
      gguf, "stt.moonshine.conv_stem.groupnorm_eps", 1e-5f, false, kTag);

  hp.fe_type = read_string_kv(gguf, "stt.frontend.type", true, kTag);
  hp.fe_sample_rate =
      read_u32_kv(gguf, "stt.frontend.sample_rate", 0, true, kTag);

  // ----- cross-field invariants (mirrors the arch loader) -----
  if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 ||
      hp.enc_ffn_dim <= 0) {
    fail("encoder hparams must be positive");
  }
  if (hp.enc_d_model % hp.enc_n_heads != 0) {
    fail("encoder d_model not divisible by n_heads");
  }
  if (hp.dec_n_layers <= 0 || hp.dec_d_model <= 0 || hp.dec_n_heads <= 0 ||
      hp.dec_ffn_dim <= 0 || hp.dec_max_position_embeddings <= 0 ||
      hp.dec_vocab_size <= 0) {
    fail("decoder hparams must be positive");
  }
  if (hp.dec_d_model % hp.dec_n_heads != 0) {
    fail("decoder d_model not divisible by n_heads");
  }
  if (hp.enc_d_model != hp.dec_d_model) {
    fail("encoder d_model != decoder d_model");
  }
  if (hp.enc_activation != "gelu") {
    fail("only \"gelu\" encoder activation is supported (got \"" +
         hp.enc_activation + "\")");
  }
  if (hp.dec_activation != "silu") {
    fail("only \"silu\" decoder activation is supported (got \"" +
         hp.dec_activation + "\")");
  }
  if (!hp.dec_tie_word_embeddings) {
    fail("tie_word_embeddings=false is not supported (no separate lm_head "
         "tensor)");
  }
  if (hp.attention_bias) {
    fail("attention_bias=true is not supported (catalog has no attention bias "
         "slots)");
  }
  if (hp.partial_rotary_factor <= 0.0f || hp.partial_rotary_factor > 1.0f) {
    fail("invalid partial_rotary_factor");
  }
  if (hp.fe_type != "raw") {
    fail("unsupported frontend type \"" + hp.fe_type + "\" (only \"raw\")");
  }
  if (hp.fe_sample_rate != 16000) {
    fail("unsupported sample_rate (only 16000 Hz)");
  }
  if (hp.conv_channels.size() != 3 || hp.conv_kernel_sizes.size() != 3 ||
      hp.conv_strides.size() != 3) {
    fail("conv stem expected 3 entries each");
  }
  if (hp.conv_channels[0] != hp.enc_d_model ||
      hp.conv_channels[2] != hp.enc_d_model) {
    fail("conv stem channels disagree with encoder d_model");
  }
  if (hp.conv_groupnorm_num_groups != 1) {
    fail("only num_groups=1 GroupNorm is supported");
  }
  if (hp.bos_token_id < 0 || hp.eos_token_id < 0 || hp.pad_token_id < 0 ||
      hp.decoder_start_token_id < 0) {
    fail("special token IDs must be set");
  }
  return hp;
}

std::string read_variant(const gguf_context *gguf) {
  const int64_t kid = gguf_find_key(gguf, "stt.variant");
  if (kid >= 0 && gguf_get_kv_type(gguf, kid) == GGUF_TYPE_STRING) {
    return std::string(gguf_get_val_str(gguf, kid));
  }
  return "moonshine";
}

struct GgufHandle {
  gguf_context *ctx = nullptr;
  ggml_context *meta = nullptr;

  ~GgufHandle() {
    if (ctx != nullptr) {
      gguf_free(ctx);
    }
    if (meta != nullptr) {
      ggml_free(meta);
    }
  }
};

} // namespace

std::shared_ptr<const MoonshineAssets>
load_moonshine_assets(const std::filesystem::path &model_path) {
  // The spec's gguf source maps tensors from "$gguf" and keeps the model
  // root at the file itself. open_tensor_source gives us lazy native-type
  // access to every weight; a second lightweight no_alloc handle exposes
  // the stt.* / tokenizer.ggml.* metadata.
  auto bundle_assets = std::make_shared<MoonshineAssets>();
  bundle_assets->source = engine::model_spec::load_resource_bundle_for_family(
                              model_path, "moonshine")
                              .open_tensor_source("weights");

  GgufHandle handle;
  handle.ctx = gguf_init_from_file(
      bundle_assets->source->source_path().string().c_str(),
      gguf_init_params{/*no_alloc=*/true, /*ctx=*/&handle.meta});
  if (handle.ctx == nullptr) {
    fail("failed to open GGUF: " +
         bundle_assets->source->source_path().string());
  }

  bundle_assets->hparams = read_hparams(handle.ctx);
  bundle_assets->variant = read_variant(handle.ctx);

  bundle_assets->tokenizer = text::load_tokenizer_from_gguf(handle.ctx);
  if (!bundle_assets->tokenizer) {
    fail("failed to build tokenizer from GGUF vocab");
  }

  // Cheap structural probe before any weight upload: verify a couple of
  // sentinel tensors exist so malformed files fail fast with a clear error.
  for (const char *name : {"enc.conv.norm.weight", "dec.token_embd.weight"}) {
    if (!bundle_assets->source->has_tensor(name)) {
      fail(std::string("missing required tensor: ") + name);
    }
  }
  return bundle_assets;
}

} // namespace engine::models::moonshine
