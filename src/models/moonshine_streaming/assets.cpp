// engine/models/moonshine_streaming/assets.cpp - GGUF metadata + tokenizer +
// tensor catalog for the native engine Moonshine-Streaming package.
//
// Hparams come from the same stt.moonshine_streaming.* GGUF metadata keys the
// arch loader reads (the pinned packages carry no embedded config.json
// sidecar - verified with `audiocpp_gguf --inspect`, which reports
// embedded_sidecars=false, and model_specs/moonshine_streaming.json was
// corrected accordingly); the tokenizer comes from the tokenizer.ggml.* vocab
// via the Phase-9 TokenizerHub; weights stream through the engine TensorSource
// abstraction.

#include "engine/models/moonshine_streaming/assets.h"

#include "engine/framework/model_spec/package.h"

#include <gguf.h>

#include <cmath>
#include <string>
#include <vector>

namespace engine::models::moonshine_streaming {

namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("moonshine_streaming: " + message);
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
  switch (gguf_get_kv_type(gguf, kid)) {
  case GGUF_TYPE_UINT32:
    return static_cast<int32_t>(gguf_get_val_u32(gguf, kid));
  case GGUF_TYPE_INT32:
    return gguf_get_val_i32(gguf, kid);
  default:
    fail(where + ": GGUF metadata key " + key + " is not u32/i32");
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
  if (gguf_get_kv_type(gguf, kid) != GGUF_TYPE_FLOAT32) {
    fail(where + ": GGUF metadata key " + key + " is not f32");
  }
  return gguf_get_val_f32(gguf, kid);
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
  if (gguf_get_kv_type(gguf, kid) != GGUF_TYPE_BOOL) {
    fail(where + ": GGUF metadata key " + key + " is not bool");
  }
  return gguf_get_val_bool(gguf, kid);
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

MoonshineStreamingHParams read_hparams(const gguf_context *gguf) {
  constexpr const char *kTag = "moonshine_streaming";
  MoonshineStreamingHParams hp;

  // ----- encoder -----
  hp.enc_n_layers = read_u32_kv(
      gguf, "stt.moonshine_streaming.encoder.n_layers", 0, true, kTag);
  hp.enc_d_model = read_u32_kv(
      gguf, "stt.moonshine_streaming.encoder.d_model", 0, true, kTag);
  hp.enc_n_heads = read_u32_kv(
      gguf, "stt.moonshine_streaming.encoder.n_heads", 0, true, kTag);
  hp.enc_n_kv_heads =
      read_u32_kv(gguf, "stt.moonshine_streaming.encoder.n_kv_heads",
                  hp.enc_n_heads, false, kTag);
  hp.enc_head_dim = read_u32_kv(
      gguf, "stt.moonshine_streaming.encoder.head_dim", 0, true, kTag);
  hp.enc_ffn_dim = read_u32_kv(
      gguf, "stt.moonshine_streaming.encoder.ffn_dim", 0, true, kTag);
  hp.enc_activation = read_string_kv(
      gguf, "stt.moonshine_streaming.encoder.activation", true, kTag);
  hp.enc_frame_ms = read_f32_kv(
      gguf, "stt.moonshine_streaming.encoder.frame_ms", 0.0f, true, kTag);
  hp.enc_frame_len = read_u32_kv(
      gguf, "stt.moonshine_streaming.encoder.frame_len", 0, true, kTag);
  hp.enc_sliding_windows = read_u32_array_kv(
      gguf, "stt.moonshine_streaming.encoder.sliding_windows", kTag);

  // ----- decoder -----
  hp.dec_n_layers = read_u32_kv(
      gguf, "stt.moonshine_streaming.decoder.n_layers", 0, true, kTag);
  hp.dec_d_model = read_u32_kv(
      gguf, "stt.moonshine_streaming.decoder.d_model", 0, true, kTag);
  hp.dec_n_heads = read_u32_kv(
      gguf, "stt.moonshine_streaming.decoder.n_heads", 0, true, kTag);
  hp.dec_n_kv_heads =
      read_u32_kv(gguf, "stt.moonshine_streaming.decoder.n_kv_heads",
                  hp.dec_n_heads, false, kTag);
  hp.dec_head_dim = read_u32_kv(
      gguf, "stt.moonshine_streaming.decoder.head_dim", 0, true, kTag);
  hp.dec_ffn_dim = read_u32_kv(
      gguf, "stt.moonshine_streaming.decoder.ffn_dim", 0, true, kTag);
  hp.dec_max_position_embeddings =
      read_u32_kv(gguf, "stt.moonshine_streaming.decoder.max_position_embeddings",
                  0, true, kTag);
  hp.dec_vocab_size = read_u32_kv(
      gguf, "stt.moonshine_streaming.decoder.vocab_size", 0, true, kTag);
  hp.dec_activation = read_string_kv(
      gguf, "stt.moonshine_streaming.decoder.activation", true, kTag);
  hp.dec_tie_word_embeddings =
      read_bool_kv(gguf, "stt.moonshine_streaming.decoder.tie_word_embeddings",
                   false, false, kTag);

  // ----- special tokens -----
  hp.bos_token_id =
      read_u32_kv(gguf, "stt.moonshine_streaming.bos_token_id", -1, true, kTag);
  hp.eos_token_id =
      read_u32_kv(gguf, "stt.moonshine_streaming.eos_token_id", -1, true, kTag);
  hp.pad_token_id =
      read_u32_kv(gguf, "stt.moonshine_streaming.pad_token_id", -1, true, kTag);
  hp.decoder_start_token_id = read_u32_kv(
      gguf, "stt.moonshine_streaming.decoder_start_token_id", -1, true, kTag);

  // ----- attention / RoPE -----
  hp.partial_rotary_factor = read_f32_kv(
      gguf, "stt.moonshine_streaming.partial_rotary_factor", 0.8f, false, kTag);
  hp.rope_theta = read_f32_kv(gguf, "stt.moonshine_streaming.rope_theta",
                              10000.0f, false, kTag);
  hp.attention_bias = read_bool_kv(
      gguf, "stt.moonshine_streaming.attention_bias", false, false, kTag);
  hp.pad_head_dim_multiple = read_u32_kv(
      gguf, "stt.moonshine_streaming.pad_head_dim_to_multiple_of", 0, false,
      kTag);

  // ----- frontend / adapter -----
  hp.cmvn_eps =
      read_f32_kv(gguf, "stt.moonshine_streaming.cmvn_eps", 1e-6f, false, kTag);
  hp.encoder_hidden_size =
      read_u32_kv(gguf, "stt.moonshine_streaming.encoder_hidden_size",
                  hp.enc_d_model, false, kTag);
  hp.adapter_has_proj = read_bool_kv(
      gguf, "stt.moonshine_streaming.adapter_has_proj", false, false, kTag);

  hp.fe_type = read_string_kv(gguf, "stt.frontend.type", true, kTag);
  hp.fe_sample_rate =
      read_u32_kv(gguf, "stt.frontend.sample_rate", 0, true, kTag);

  // ----- cross-field invariants (mirrors the arch loader) -----
  if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 ||
      hp.enc_head_dim <= 0 || hp.enc_ffn_dim <= 0) {
    fail("encoder hparams must be positive");
  }
  if (hp.enc_frame_len <= 0) {
    fail("encoder frame_len must be positive");
  }
  if (static_cast<int32_t>(hp.enc_sliding_windows.size()) !=
      2 * hp.enc_n_layers) {
    fail("encoder.sliding_windows must hold exactly 2 entries per layer");
  }
  if (hp.dec_n_layers <= 0 || hp.dec_d_model <= 0 || hp.dec_n_heads <= 0 ||
      hp.dec_head_dim <= 0 || hp.dec_ffn_dim <= 0 ||
      hp.dec_max_position_embeddings <= 0 || hp.dec_vocab_size <= 0) {
    fail("decoder hparams must be positive");
  }
  if (hp.dec_d_model != hp.dec_n_heads * hp.dec_head_dim) {
    fail("decoder d_model != n_heads * head_dim");
  }
  if (hp.enc_activation != "gelu") {
    fail("only \"gelu\" encoder activation is supported (got \"" +
         hp.enc_activation + "\")");
  }
  if (hp.dec_activation != "silu") {
    fail("only \"silu\" decoder activation is supported (got \"" +
         hp.dec_activation + "\")");
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
  // The adapter is the only thing that can reconcile differing widths.
  if (hp.enc_d_model != hp.dec_d_model && !hp.adapter_has_proj) {
    fail("encoder d_model != decoder d_model but adapter has no projection");
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
  return "moonshine_streaming";
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

std::shared_ptr<const MoonshineStreamingAssets>
load_moonshine_streaming_assets(const std::filesystem::path &model_path) {
  auto bundle_assets = std::make_shared<MoonshineStreamingAssets>();
  bundle_assets->source =
      engine::model_spec::load_resource_bundle_for_family(model_path,
                                                          "moonshine_streaming")
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

  // Cheap structural probe before any weight upload. The lm_head sentinel is
  // deliberate: this family has an UNTIED lm_head, so its absence would only
  // surface much later as a decode-time null deref.
  for (const char *name : {"enc.embedder.linear.weight", "adapter.pos_emb.weight",
                           "dec.token_embd.weight", "dec.lm_head.weight"}) {
    if (!bundle_assets->source->has_tensor(name)) {
      fail(std::string("missing required tensor: ") + name);
    }
  }
  return bundle_assets;
}

} // namespace engine::models::moonshine_streaming
