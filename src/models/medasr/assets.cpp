// engine/models/medasr/assets.cpp - model resources for the native engine
// MedASR package (see assets.h).
//
// Ported from src/runtime/arch/medasr/weights.cpp (read_medasr_hparams, the
// tensor catalog's frontend half, fuse_batch_norm) and model.cpp::load() (the
// frontend buffers, the soft-window advisory). The KV names, the invariants
// and the BatchNorm fold are copied from the arch verbatim. Text joining goes
// through TokenizerHub (see MedAsrAssets::decode).

#include "engine/models/medasr/assets.h"

#include "engine/framework/assets/gguf_metadata.h"

#include <gguf.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::medasr {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747u; // "GGUF" little-endian
constexpr const char *kArchitecture = "medasr";

// PyTorch BatchNorm1d default eps. Independent of the encoder's LayerNorm eps
// (1e-6); LASR's batch_norm_momentum (0.01) only matters at training time.
constexpr float kBnEps = 1e-5f;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(std::string("medasr: ") + message);
}

uint32_t read_magic(const std::filesystem::path &path) {
  std::ifstream f(path, std::ios::binary);
  uint32_t magic = 0;
  if (f) {
    f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  }
  return f ? magic : 0u;
}

std::string lname(const char *fmt, int i) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), fmt, i);
  return std::string(buf);
}

// read_medasr_hparams, key for key. Every key is required, as in the arch.
void read_hparams(const assets::GgufMetadata &meta, MedAsrHParams &hp) {
  hp.enc_n_layers = meta.require_i32("stt.medasr.encoder.n_layers");
  hp.enc_hidden = meta.require_i32("stt.medasr.encoder.hidden");
  hp.enc_n_heads = meta.require_i32("stt.medasr.encoder.n_heads");
  hp.enc_n_kv_heads = meta.require_i32("stt.medasr.encoder.n_kv_heads");
  hp.enc_head_dim = meta.require_i32("stt.medasr.encoder.head_dim");
  hp.enc_intermediate = meta.require_i32("stt.medasr.encoder.intermediate");
  hp.enc_hidden_act = meta.require_string("stt.medasr.encoder.hidden_act");
  hp.enc_conv_kernel = meta.require_i32("stt.medasr.encoder.conv_kernel");
  hp.enc_conv_resid_w0 = meta.require_float("stt.medasr.encoder.conv_residual_w0");
  hp.enc_conv_resid_w1 = meta.require_float("stt.medasr.encoder.conv_residual_w1");
  hp.enc_ff_resid_w0 = meta.require_float("stt.medasr.encoder.ff_residual_w0");
  hp.enc_ff_resid_w1 = meta.require_float("stt.medasr.encoder.ff_residual_w1");
  hp.enc_layer_norm_eps = meta.require_float("stt.medasr.encoder.layer_norm_eps");
  hp.enc_max_pos_emb = meta.require_i32("stt.medasr.encoder.max_pos_emb");
  hp.enc_batch_norm_momentum = meta.require_float("stt.medasr.encoder.batch_norm_momentum");
  hp.enc_rope_theta = meta.require_float("stt.medasr.encoder.rope_theta");
  hp.enc_rope_type = meta.require_string("stt.medasr.encoder.rope_type");
  hp.enc_num_mel_bins = meta.require_i32("stt.medasr.encoder.num_mel_bins");
  hp.enc_sub_kernel = meta.require_i32("stt.medasr.encoder.sub_kernel");
  hp.enc_sub_stride = meta.require_i32("stt.medasr.encoder.sub_stride");
  hp.enc_sub_channels = meta.require_i32("stt.medasr.encoder.sub_channels");
  hp.enc_sub_n_layers = meta.require_i32("stt.medasr.encoder.sub_n_layers");

  // The arch's three invariants, same order, same meaning.
  if (hp.enc_n_heads <= 0 || hp.enc_hidden % hp.enc_n_heads != 0) {
    fail("invariant hidden % n_heads != 0 (hidden=" + std::to_string(hp.enc_hidden) +
         ", n_heads=" + std::to_string(hp.enc_n_heads) + ")");
  }
  if (hp.enc_sub_n_layers != 2 || hp.enc_sub_stride != 2) {
    fail("unsupported subsampling shape (n_layers=" + std::to_string(hp.enc_sub_n_layers) +
         ", stride=" + std::to_string(hp.enc_sub_stride) +
         ") - only 2 stride-2 convs implemented");
  }
  if (hp.enc_rope_type != "default") {
    fail("unsupported rope_type=" + hp.enc_rope_type + " (expected default)");
  }

  hp.ctc_vocab_size = meta.require_i32("stt.medasr.ctc.vocab_size");
  hp.ctc_blank_id = meta.require_i32("stt.medasr.ctc.blank_id");

  hp.fe_type = meta.require_string("stt.frontend.type");
  hp.fe_sample_rate = meta.require_i32("stt.frontend.sample_rate");
  hp.fe_num_mels = meta.require_i32("stt.frontend.num_mels");
  hp.fe_n_fft = meta.require_i32("stt.frontend.n_fft");
  hp.fe_win_length = meta.require_i32("stt.frontend.win_length");
  hp.fe_hop_length = meta.require_i32("stt.frontend.hop_length");
  hp.fe_window = meta.require_string("stt.frontend.window");
  hp.fe_normalize = meta.require_string("stt.frontend.normalize");
  hp.fe_pad_mode = meta.require_string("stt.frontend.pad_mode");
  hp.fe_mel_norm = meta.require_string("stt.frontend.mel_norm");
  hp.fe_log_clamp_min = meta.require_float("stt.frontend.log_clamp_min");
  hp.fe_mel_lower_hz = meta.require_float("stt.frontend.mel_lower_hz");
  hp.fe_mel_upper_hz = meta.require_float("stt.frontend.mel_upper_hz");

  // Not checked by the arch, but it would have indexed past its buffers or
  // divided by zero on any of these; refuse up front instead.
  if (hp.enc_n_layers <= 0 || hp.enc_hidden <= 0 || hp.enc_intermediate <= 0 ||
      hp.enc_conv_kernel <= 0 || hp.enc_sub_kernel <= 0 || hp.enc_sub_channels <= 0 ||
      hp.enc_num_mel_bins <= 0 || hp.ctc_vocab_size <= 0) {
    fail("non-positive encoder geometry in the GGUF");
  }
  if (hp.fe_sample_rate <= 0 || hp.fe_n_fft <= 0 || hp.fe_win_length <= 0 ||
      hp.fe_win_length > hp.fe_n_fft || hp.fe_hop_length <= 0 || hp.fe_num_mels <= 0) {
    fail("invalid frontend geometry in the GGUF");
  }
  if (hp.fe_num_mels != hp.enc_num_mel_bins) {
    fail("frontend num_mels (" + std::to_string(hp.fe_num_mels) +
         ") != encoder num_mel_bins (" + std::to_string(hp.enc_num_mel_bins) + ")");
  }
}

// transcribe::Tokenizer::load's contract for the keys it requires: a model tag
// it knows and a non-empty token list whose optional scores / types arrays are
// the same length.
void read_tokenizer(const assets::GgufMetadata &meta, MedAsrAssets &out) {
  const std::string model = meta.require_string("tokenizer.ggml.model");
  if (model == "unigram" || model == "bpe") {
    out.piece_decode = MedAsrPieceDecode::SentencePiece;
  } else if (model == "char") {
    out.piece_decode = MedAsrPieceDecode::RawConcat;
  } else if (model == "gpt2") {
    out.piece_decode = MedAsrPieceDecode::ByteLevel;
  } else {
    fail("unsupported tokenizer.ggml.model \"" + model + "\"");
  }
  out.tokenizer = text::load_tokenizer_from_gguf(meta.context());
  if (!out.tokenizer || out.tokenizer->vocab_size() == 0) {
    fail("GGUF has no tokenizer.ggml.tokens");
  }
  const auto tokens = meta.find_string_array("tokenizer.ggml.tokens");
  if (!tokens.has_value() || tokens->empty()) {
    fail("GGUF has no tokenizer.ggml.tokens");
  }
  out.pieces = *tokens;
  // Optional arrays: when present they must be typed arrays of the same
  // length as the vocabulary (the arch rejects the file otherwise).
  const gguf_context *ctx = meta.context();
  for (const char *key : {"tokenizer.ggml.scores", "tokenizer.ggml.token_type"}) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
      continue;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_n(ctx, id) != out.pieces.size()) {
      fail(std::string(key) + " is not an array as long as tokenizer.ggml.tokens");
    }
  }
}

} // namespace

int MedAsrAssets::encoder_frames_for(int n_mel_frames) const noexcept {
  const int k = hparams.enc_sub_kernel;
  const int s = hparams.enc_sub_stride;
  auto sub = [&](int t_in) { return (t_in < k) ? 0 : (t_in - k) / s + 1; };
  return sub(sub(n_mel_frames));
}

int MedAsrAssets::subsample_factor() const noexcept {
  if (hparams.enc_sub_stride <= 0 || hparams.enc_sub_n_layers <= 0) {
    return 4;
  }
  int factor = 1;
  for (int i = 0; i < hparams.enc_sub_n_layers; ++i) {
    factor *= hparams.enc_sub_stride;
  }
  return factor;
}

int64_t MedAsrAssets::ms_per_encoder_frame() const noexcept {
  if (hparams.fe_hop_length <= 0 || hparams.fe_sample_rate <= 0) {
    return 0;
  }
  return static_cast<int64_t>(hparams.fe_hop_length) * subsample_factor() * 1000 /
         hparams.fe_sample_rate;
}

int64_t MedAsrAssets::max_audio_ms() const noexcept {
  const int64_t ms_per_frame = ms_per_encoder_frame();
  if (hparams.enc_max_pos_emb <= 0 || ms_per_frame <= 0) {
    return 0;
  }
  return static_cast<int64_t>(hparams.enc_max_pos_emb) * ms_per_frame;
}

int64_t MedAsrAssets::min_audio_samples() const noexcept {
  // Smallest mel frame count whose double subsampling leaves one frame, then
  // the PyTorch center=False sample count for it: win + (frames - 1) * hop.
  int mel_frames = 1;
  while (encoder_frames_for(mel_frames) < 1 && mel_frames < (1 << 20)) {
    ++mel_frames;
  }
  return static_cast<int64_t>(hparams.fe_win_length) +
         static_cast<int64_t>(mel_frames - 1) * hparams.fe_hop_length;
}

std::string MedAsrAssets::piece(int32_t id) const {
  if (id < 0 || static_cast<size_t>(id) >= pieces.size()) {
    return {};
  }
  return pieces[static_cast<size_t>(id)];
}

std::string MedAsrAssets::decode(const std::vector<int32_t> &ids) const {
  switch (piece_decode) {
  case MedAsrPieceDecode::SentencePiece:
  case MedAsrPieceDecode::ByteLevel:
    // TokenizerHub: SentencePiece-marked "unigram" / "bpe" vocabs join pieces
    // (U+2581 -> ' ', <0xNN> -> byte) exactly as transcribe::Tokenizer does;
    // "gpt2" inverts the byte-level map.
    return tokenizer->decode(ids);
  case MedAsrPieceDecode::RawConcat: {
    // "char": the arch concatenates pieces verbatim; the hub would treat the
    // vocab as SentencePiece, so this one mode stays local.
    std::string out;
    for (const int32_t id : ids) {
      if (id >= 0 && static_cast<size_t>(id) < pieces.size()) {
        out += pieces[static_cast<size_t>(id)];
      }
    }
    return out;
  }
  }
  return {};
}

std::shared_ptr<const MedAsrAssets>
load_medasr_assets(const std::filesystem::path &model_path) {
  if (!std::filesystem::exists(model_path)) {
    fail("model file not found: " + model_path.string());
  }
  if (read_magic(model_path) != kGgufMagic) {
    fail("not a GGUF file: " + model_path.string());
  }
  auto out = std::make_shared<MedAsrAssets>();
  out->model_path = model_path;

  const auto meta = assets::GgufMetadata::open(model_path);
  if (meta.find_string("general.architecture").value_or("") != kArchitecture) {
    fail("GGUF general.architecture is not \"medasr\": " + model_path.string());
  }

  read_tokenizer(meta, *out);
  read_hparams(meta, out->hparams);
  const auto &hp = out->hparams;

  // read_languages_kv: general.languages when present; absent is an
  // information gap, not "no languages".
  out->language_codes =
      meta.find_string_array("general.languages").value_or(std::vector<std::string>{});
  out->variant = meta.find_string("stt.variant").value_or(kArchitecture);

  out->source = assets::open_tensor_source(model_path);
  const auto &source = *out->source;

  // Frontend buffers. The converter writes the filterbank as numpy
  // (n_freq, n_mels), i.e. ggml ne = [n_mels, n_freq]; MelExtractor wants
  // [n_mels][n_freq] row-major, so transpose once (as the arch did at load).
  const int n_freq = hp.fe_n_fft / 2 + 1;
  const int n_mels = hp.fe_num_mels;
  const std::vector<float> raw_fb = source.require_f32(
      "frontend.mel_filterbank", std::vector<int64_t>{n_freq, n_mels});
  out->mel_filterbank.assign(static_cast<size_t>(n_mels) * static_cast<size_t>(n_freq), 0.0f);
  for (int k = 0; k < n_freq; ++k) {
    for (int m = 0; m < n_mels; ++m) {
      out->mel_filterbank[static_cast<size_t>(m) * n_freq + k] =
          raw_fb[static_cast<size_t>(k) * n_mels + m];
    }
  }
  out->window = source.require_f32("frontend.window",
                                   std::vector<int64_t>{hp.fe_win_length});

  // fuse_batch_norm: scale = w / sqrt(var + eps), bias = b - mean * scale,
  // per channel, in f32 exactly as the arch computes it.
  const int d = hp.enc_hidden;
  out->bn_scale.assign(static_cast<size_t>(hp.enc_n_layers), std::vector<float>(d, 0.0f));
  out->bn_bias.assign(static_cast<size_t>(hp.enc_n_layers), std::vector<float>(d, 0.0f));
  const std::vector<int64_t> vec_shape{d};
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    const auto bn_w = source.require_f32(lname("enc.blocks.%d.conv_bn.weight", i), vec_shape);
    const auto bn_b = source.require_f32(lname("enc.blocks.%d.conv_bn.bias", i), vec_shape);
    const auto bn_mean =
        source.require_f32(lname("enc.blocks.%d.conv_bn.running_mean", i), vec_shape);
    const auto bn_var =
        source.require_f32(lname("enc.blocks.%d.conv_bn.running_var", i), vec_shape);
    auto &scale = out->bn_scale[static_cast<size_t>(i)];
    auto &bias = out->bn_bias[static_cast<size_t>(i)];
    for (int c = 0; c < d; ++c) {
      const float s = bn_w[static_cast<size_t>(c)] /
                      std::sqrt(bn_var[static_cast<size_t>(c)] + kBnEps);
      scale[static_cast<size_t>(c)] = s;
      bias[static_cast<size_t>(c)] =
          bn_b[static_cast<size_t>(c)] - bn_mean[static_cast<size_t>(c)] * s;
    }
  }
  return out;
}

bool looks_like_medasr_gguf(const std::filesystem::path &path) {
  if (read_magic(path) != kGgufMagic) {
    return false;
  }
  try {
    const auto meta = assets::GgufMetadata::open(path);
    return meta.find_string("general.architecture").value_or("") == kArchitecture;
  } catch (const std::exception &) {
    return false;
  }
}

} // namespace engine::models::medasr
