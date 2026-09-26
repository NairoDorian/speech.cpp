// engine/models/gigaam/assets.cpp - model resources for the native engine
// GigaAM package, from parent transcribe.cpp's GGUF layout.
//
// Hyperparameters: the stt.gigaam.* / stt.frontend.* KVs with the validation
// of the arch's read_gigaam_hparams (src/runtime/arch/gigaam/weights.cpp).
// Tensor catalog: every name, shape and dtype allowlist of the arch's
// build_gigaam_weights, checked up front so a bad file fails at load, as it
// did in the arch. Tokenizer: tokenizer.ggml.* (decode modes of the arch's
// transcribe::Tokenizer). Frontend buffers and the host head (RNN-T predictor
// + joint, or CTC head) are read here, once; the encoder weights are streamed
// by each session's runtime.

#include "engine/models/gigaam/assets.h"

#include "engine/framework/assets/gguf_metadata.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::gigaam {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747u; // "GGUF" little-endian
constexpr const char kSpSpace[] = "\xE2\x96\x81";
constexpr size_t kSpSpaceLen = 3;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("gigaam: " + message);
}

uint32_t read_magic(const std::filesystem::path &path) {
  std::ifstream f(path, std::ios::binary);
  uint32_t magic = 0;
  if (f) {
    f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  }
  return f ? magic : 0u;
}

std::string shape_string(const std::vector<int64_t> &shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += (i ? ", " : "") + std::to_string(shape[i]);
  }
  return out + "]";
}

std::string lname(const char *fmt, int i) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), fmt, i);
  return std::string(buf);
}

// The arch's dtype allowlists (transcribe-weights-util.h).
enum class TensorKind {
  F32,    // norms, biases, frontend buffers: GGML_TYPE_F32 only
  Linear, // TRANSCRIBE_QUANT_LINEAR_TYPES
  Conv,   // TRANSCRIBE_QUANT_CONV_TYPES
};

bool type_allowed(ggml_type type, TensorKind kind) {
  switch (kind) {
  case TensorKind::F32:
    return type == GGML_TYPE_F32;
  case TensorKind::Conv:
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
  case TensorKind::Linear:
    switch (type) {
    case GGML_TYPE_F32:
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
      return true;
    default:
      return false;
    }
  }
  return false;
}

// find_tensor: present, dtype in the allowlist, shape exact. `shape` is the
// TensorSource logical (row-major) shape, i.e. the arch's ggml ne reversed.
void check_tensor(const assets::TensorSource &source, const std::filesystem::path &path,
                  const std::string &name, const std::vector<int64_t> &shape, TensorKind kind) {
  if (!source.has_tensor(name)) {
    fail(path.filename().string() + ": missing tensor \"" + name + "\"");
  }
  const auto metadata = source.require_metadata(name);
  const ggml_type type = assets::ggml_type_for_tensor_dtype(metadata.dtype);
  if (!type_allowed(type, kind)) {
    fail(path.filename().string() + ": tensor \"" + name + "\" has unsupported type " +
         ggml_type_name(type));
  }
  if (metadata.shape != shape) {
    fail(path.filename().string() + ": tensor \"" + name + "\" has shape " +
         shape_string(metadata.shape) + ", expected " + shape_string(shape));
  }
}

// ---- hparams (read_gigaam_hparams) ----------------------------------------

void read_hparams(const assets::GgufMetadata &meta, GigaamHParams &hp) {
  const std::string file = meta.path().filename().string();
  const auto require = [&](bool ok, const std::string &what) {
    if (!ok) {
      fail(file + ": " + what);
    }
  };

  // Head discriminator.
  {
    const std::string head_kind = meta.require_string("stt.gigaam.head_kind");
    if (head_kind == "rnnt") {
      hp.head_kind = GigaamHeadKind::Rnnt;
    } else if (head_kind == "ctc") {
      hp.head_kind = GigaamHeadKind::Ctc;
    } else {
      fail(file + ": unsupported stt.gigaam.head_kind=" + head_kind);
    }
  }

  // Encoder.
  hp.enc_n_layers = meta.require_i32("stt.gigaam.encoder.n_layers");
  hp.enc_d_model = meta.require_i32("stt.gigaam.encoder.d_model");
  hp.enc_n_heads = meta.require_i32("stt.gigaam.encoder.n_heads");
  hp.enc_d_ff = meta.require_i32("stt.gigaam.encoder.d_ff");
  hp.enc_conv_kernel = meta.require_i32("stt.gigaam.encoder.conv_kernel");
  hp.enc_subsampling_factor = meta.require_i32("stt.gigaam.encoder.subsampling_factor");
  hp.enc_subs_kernel_size = meta.require_i32("stt.gigaam.encoder.subs_kernel_size");
  hp.enc_pos_emb_max_len = meta.require_i32("stt.gigaam.encoder.pos_emb_max_len");
  hp.enc_feat_in = meta.require_i32("stt.gigaam.encoder.feat_in");
  hp.enc_self_attention_model = meta.require_string("stt.gigaam.encoder.self_attention_model");
  hp.enc_conv_norm_type = meta.require_string("stt.gigaam.encoder.conv_norm_type");

  require(hp.enc_n_heads > 0 && hp.enc_d_model % hp.enc_n_heads == 0,
          "invariant d_model % n_heads != 0 (d_model=" + std::to_string(hp.enc_d_model) +
              ", n_heads=" + std::to_string(hp.enc_n_heads) + ")");
  require(hp.enc_self_attention_model == "rotary",
          "unsupported self_attention_model=" + hp.enc_self_attention_model + " (expected rotary)");
  require(hp.enc_conv_norm_type == "layer_norm",
          "unsupported conv_norm_type=" + hp.enc_conv_norm_type + " (expected layer_norm)");

  if (hp.head_kind == GigaamHeadKind::Rnnt) {
    hp.pred_hidden = meta.require_i32("stt.gigaam.predictor.hidden");
    hp.pred_n_layers = meta.require_i32("stt.gigaam.predictor.n_layers");
    hp.pred_vocab = meta.require_i32("stt.gigaam.predictor.vocab");
    hp.joint_hidden = meta.require_i32("stt.gigaam.joint.hidden");
    hp.joint_n_classes = meta.require_i32("stt.gigaam.joint.num_classes");
    hp.joint_activation = meta.require_string("stt.gigaam.joint.activation");
    require(hp.pred_vocab == hp.joint_n_classes,
            "pred_vocab (" + std::to_string(hp.pred_vocab) + ") != joint_n_classes (" +
                std::to_string(hp.joint_n_classes) + ")");
  } else {
    hp.head_feat_in = meta.require_i32("stt.gigaam.head.feat_in");
    hp.head_n_classes = meta.require_i32("stt.gigaam.head.num_classes");
    require(hp.head_feat_in == hp.enc_d_model,
            "ctc head.feat_in (" + std::to_string(hp.head_feat_in) + ") != encoder.d_model (" +
                std::to_string(hp.enc_d_model) + ")");
  }

  // Frontend (all required, as in the arch; center optional, default false).
  hp.fe_type = meta.require_string("stt.frontend.type");
  hp.fe_num_mels = meta.require_i32("stt.frontend.num_mels");
  hp.fe_sample_rate = meta.require_i32("stt.frontend.sample_rate");
  hp.fe_n_fft = meta.require_i32("stt.frontend.n_fft");
  hp.fe_win_length = meta.require_i32("stt.frontend.win_length");
  hp.fe_hop_length = meta.require_i32("stt.frontend.hop_length");
  hp.fe_window = meta.require_string("stt.frontend.window");
  hp.fe_normalize = meta.require_string("stt.frontend.normalize");
  hp.fe_dither = meta.require_float("stt.frontend.dither");
  hp.fe_pre_emphasis = meta.require_float("stt.frontend.pre_emphasis");
  hp.fe_f_min = meta.require_float("stt.frontend.f_min");
  hp.fe_f_max = meta.require_float("stt.frontend.f_max");
  hp.fe_center = meta.find_bool("stt.frontend.center").value_or(false);
  hp.fe_mel_norm = meta.require_string("stt.frontend.mel_norm");
  hp.fe_log_clamp_min = meta.require_float("stt.frontend.log_clamp_min");
  hp.fe_log_clamp_max = meta.require_float("stt.frontend.log_clamp_max");

  // Beyond the arch's checks: shapes its graph / decoder silently assume.
  // Every published variant satisfies them; a file that does not would have
  // crashed or decoded garbage in the arch, so fail loudly instead.
  require(hp.enc_n_layers > 0 && hp.enc_d_model > 0 && hp.enc_d_ff > 0 &&
              hp.enc_conv_kernel > 0 && hp.enc_conv_kernel % 2 == 1 &&
              hp.enc_subs_kernel_size > 0,
          "encoder geometry");
  require(hp.enc_subsampling_factor == 4,
          "subsampling_factor=" + std::to_string(hp.enc_subsampling_factor) +
              " (the pre-encode is two stride-2 convs)");
  require(hp.enc_feat_in == hp.fe_num_mels, "encoder.feat_in != frontend.num_mels");
  require(hp.n_classes() > 1, "num_classes");
  if (hp.head_kind == GigaamHeadKind::Rnnt) {
    require(hp.pred_hidden > 0 && hp.joint_hidden > 0, "predictor / joint geometry");
    require(hp.pred_n_layers == 1,
            "predictor.n_layers=" + std::to_string(hp.pred_n_layers) +
                " (the greedy decoder runs one LSTM layer)");
  }
  require(hp.fe_sample_rate > 0 && hp.fe_num_mels > 0 && hp.fe_hop_length > 0 &&
              hp.fe_n_fft > 0,
          "frontend geometry");
  // The arch windows x[t*hop, t*hop + win) and zero-pads to n_fft at the end.
  // The MelExtractor configuration used here reproduces that framing only
  // when n_fft == win_length and n_fft / 2 is a whole number of hops (320 /
  // 160 in every published variant; see runtime.cpp).
  require(hp.fe_win_length == hp.fe_n_fft, "win_length != n_fft");
  require((hp.fe_n_fft / 2) % hp.fe_hop_length == 0, "n_fft / 2 is not a multiple of hop");
}

// ---- tensor catalog (build_gigaam_weights) --------------------------------

void check_catalog(const assets::TensorSource &source, const std::filesystem::path &path,
                   const GigaamHParams &hp) {
  const int64_t d = hp.enc_d_model;
  const int64_t dff = hp.enc_d_ff;
  const int64_t dw_k = hp.enc_conv_kernel;
  const int64_t sub_k = hp.enc_subs_kernel_size;
  const int64_t n_freq = hp.fe_n_fft / 2 + 1;
  const auto t = [&](const std::string &name, std::vector<int64_t> shape, TensorKind kind) {
    check_tensor(source, path, name, shape, kind);
  };

  t("frontend.mel_filterbank", {hp.fe_num_mels, n_freq}, TensorKind::F32);
  t("frontend.window", {hp.fe_win_length}, TensorKind::F32);

  t("enc.pre_encode.conv.0.weight", {d, hp.enc_feat_in, sub_k}, TensorKind::Conv);
  t("enc.pre_encode.conv.0.bias", {d}, TensorKind::F32);
  t("enc.pre_encode.conv.2.weight", {d, d, sub_k}, TensorKind::Conv);
  t("enc.pre_encode.conv.2.bias", {d}, TensorKind::F32);

  for (int i = 0; i < hp.enc_n_layers; ++i) {
    const auto n = [&](const char *fmt) { return lname(fmt, i); };
    t(n("enc.blocks.%d.norm_ff1.weight"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.norm_ff1.bias"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.ff1.linear1.weight"), {dff, d}, TensorKind::Linear);
    t(n("enc.blocks.%d.ff1.linear1.bias"), {dff}, TensorKind::F32);
    t(n("enc.blocks.%d.ff1.linear2.weight"), {d, dff}, TensorKind::Linear);
    t(n("enc.blocks.%d.ff1.linear2.bias"), {d}, TensorKind::F32);

    t(n("enc.blocks.%d.norm_conv.weight"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.norm_conv.bias"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.conv.pointwise1.weight"), {2 * d, d, 1}, TensorKind::Conv);
    t(n("enc.blocks.%d.conv.pointwise1.bias"), {2 * d}, TensorKind::F32);
    t(n("enc.blocks.%d.conv.depthwise.weight"), {d, 1, dw_k}, TensorKind::Conv);
    t(n("enc.blocks.%d.conv.depthwise.bias"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.conv.ln.weight"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.conv.ln.bias"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.conv.pointwise2.weight"), {d, d, 1}, TensorKind::Conv);
    t(n("enc.blocks.%d.conv.pointwise2.bias"), {d}, TensorKind::F32);

    t(n("enc.blocks.%d.norm_attn.weight"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.norm_attn.bias"), {d}, TensorKind::F32);
    for (const char *proj : {"linear_q", "linear_k", "linear_v", "linear_out"}) {
      t(lname((std::string("enc.blocks.%d.attn.") + proj + ".weight").c_str(), i), {d, d},
        TensorKind::Linear);
      t(lname((std::string("enc.blocks.%d.attn.") + proj + ".bias").c_str(), i), {d},
        TensorKind::F32);
    }

    t(n("enc.blocks.%d.norm_ff2.weight"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.norm_ff2.bias"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.ff2.linear1.weight"), {dff, d}, TensorKind::Linear);
    t(n("enc.blocks.%d.ff2.linear1.bias"), {dff}, TensorKind::F32);
    t(n("enc.blocks.%d.ff2.linear2.weight"), {d, dff}, TensorKind::Linear);
    t(n("enc.blocks.%d.ff2.linear2.bias"), {d}, TensorKind::F32);

    t(n("enc.blocks.%d.norm_out.weight"), {d}, TensorKind::F32);
    t(n("enc.blocks.%d.norm_out.bias"), {d}, TensorKind::F32);
  }

  if (hp.head_kind == GigaamHeadKind::Rnnt) {
    const int64_t H = hp.pred_hidden;
    t("pred.embed.weight", {hp.pred_vocab, H}, TensorKind::Linear);
    for (int i = 0; i < hp.pred_n_layers; ++i) {
      t(lname("pred.lstm.%d.Wx", i), {4 * H, H}, TensorKind::Linear);
      t(lname("pred.lstm.%d.Wh", i), {4 * H, H}, TensorKind::Linear);
      t(lname("pred.lstm.%d.bias", i), {4 * H}, TensorKind::F32);
    }
    t("joint.enc.weight", {hp.joint_hidden, d}, TensorKind::Linear);
    t("joint.enc.bias", {hp.joint_hidden}, TensorKind::F32);
    t("joint.pred.weight", {hp.joint_hidden, H}, TensorKind::Linear);
    t("joint.pred.bias", {hp.joint_hidden}, TensorKind::F32);
    t("joint.out.weight", {hp.joint_n_classes, hp.joint_hidden}, TensorKind::Linear);
    t("joint.out.bias", {hp.joint_n_classes}, TensorKind::F32);
  } else {
    t("head.ctc.weight", {hp.head_n_classes, d, 1}, TensorKind::Conv);
    t("head.ctc.bias", {hp.head_n_classes}, TensorKind::F32);
  }
}

// ---- host head (build_host_decoder_weights) -------------------------------

void read_host_decoder(const assets::TensorSource &source, const GigaamHParams &hp,
                       GigaamHostDecoder &out) {
  // require_f32 dequantizes F16 / BF16 / Q* rows through ggml's to_float,
  // as the arch's readback_f32 did.
  const auto f32 = [&](const std::string &name, std::vector<int64_t> shape) {
    return source.require_f32(name, std::optional<std::vector<int64_t>>(std::move(shape)));
  };
  if (hp.head_kind == GigaamHeadKind::Ctc) {
    // ne [1, d_model, n_classes] is byte-identical to row-major [n_classes, d_model].
    out.ctc_w = f32("head.ctc.weight", {hp.head_n_classes, hp.enc_d_model, 1});
    out.ctc_b = f32("head.ctc.bias", {hp.head_n_classes});
    return;
  }
  const int64_t H = hp.pred_hidden;
  out.pred_embed = f32("pred.embed.weight", {hp.pred_vocab, H});
  out.lstm_Wx.resize(static_cast<size_t>(hp.pred_n_layers));
  out.lstm_Wh.resize(static_cast<size_t>(hp.pred_n_layers));
  out.lstm_b.resize(static_cast<size_t>(hp.pred_n_layers));
  for (int i = 0; i < hp.pred_n_layers; ++i) {
    out.lstm_Wx[static_cast<size_t>(i)] = f32(lname("pred.lstm.%d.Wx", i), {4 * H, H});
    out.lstm_Wh[static_cast<size_t>(i)] = f32(lname("pred.lstm.%d.Wh", i), {4 * H, H});
    out.lstm_b[static_cast<size_t>(i)] = f32(lname("pred.lstm.%d.bias", i), {4 * H});
  }
  out.joint_enc_w = f32("joint.enc.weight", {hp.joint_hidden, hp.enc_d_model});
  out.joint_enc_b = f32("joint.enc.bias", {hp.joint_hidden});
  out.joint_pred_w = f32("joint.pred.weight", {hp.joint_hidden, H});
  out.joint_pred_b = f32("joint.pred.bias", {hp.joint_hidden});
  out.joint_out_w = f32("joint.out.weight", {hp.joint_n_classes, hp.joint_hidden});
  out.joint_out_b = f32("joint.out.bias", {hp.joint_n_classes});
}

// ---- tokenizer (transcribe::Tokenizer::load) ------------------------------

void load_tokenizer(const assets::GgufMetadata &meta, GigaamAssets &out) {
  const std::string file = meta.path().filename().string();
  out.tokenizer_model = meta.require_string("tokenizer.ggml.model");
  const std::string &model = out.tokenizer_model;
  if (model == "bpe" || model == "unigram") {
    out.text_decode = GigaamTextDecode::SentencePiece;
  } else if (model == "char") {
    out.text_decode = GigaamTextDecode::Charwise;
  } else if (model == "gpt2") {
    out.text_decode = GigaamTextDecode::ByteLevel;
  } else {
    fail(file + ": unsupported tokenizer.ggml.model=" + model);
  }
  out.vocab = text::load_tokenizer_from_gguf(meta.context());
  if (!out.vocab || out.vocab->vocab_size() == 0) {
    fail(file + ": tokenizer.ggml.tokens is missing or empty");
  }
  if (out.text_decode == GigaamTextDecode::Charwise) {
    std::vector<std::string> pieces;
    pieces.reserve(out.vocab->vocab_size());
    for (size_t i = 0; i < out.vocab->vocab_size(); ++i) {
      pieces.emplace_back(out.vocab->piece(static_cast<int32_t>(i)));
    }
    out.text_encoder =
        text::load_tokenizer_from_tokens(std::move(pieces), out.vocab->specials(), /*raw_bytes=*/true);
  }
}

// SentencePiece decode of the arch (decode_sentencepiece): <0xHH> byte
// fallback pieces become the byte, U+2581 becomes ' ', everything else is
// copied byte for byte.
void append_sentencepiece(std::string &out, std::string_view p) {
  const auto hex_nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'A' && c <= 'F') {
      return 10 + (c - 'A');
    }
    if (c >= 'a' && c <= 'f') {
      return 10 + (c - 'a');
    }
    return -1;
  };
  if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
    const int hi = hex_nibble(p[3]);
    const int lo = hex_nibble(p[4]);
    if (hi >= 0 && lo >= 0) {
      out.push_back(static_cast<char>((hi << 4) | lo));
      return;
    }
  }
  size_t j = 0;
  while (j < p.size()) {
    if (j + kSpSpaceLen <= p.size() && std::memcmp(p.data() + j, kSpSpace, kSpSpaceLen) == 0) {
      out.push_back(' ');
      j += kSpSpaceLen;
    } else {
      out.push_back(p[j]);
      ++j;
    }
  }
}

} // namespace

std::string GigaamAssets::decode(const std::vector<int32_t> &ids) const {
  if (ids.empty() || !vocab) {
    return {};
  }
  if (text_decode == GigaamTextDecode::ByteLevel) {
    return vocab->decode(ids);
  }
  const size_t n_vocab = vocab->vocab_size();
  std::string out;
  out.reserve(ids.size() * 4);
  for (const int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= n_vocab) {
      continue;
    }
    const std::string_view p = vocab->piece(id);
    if (text_decode == GigaamTextDecode::Charwise) {
      out.append(p.data(), p.size());
    } else {
      append_sentencepiece(out, p);
    }
  }
  return out;
}

std::string GigaamAssets::piece(int32_t id) const {
  if (!vocab) {
    return {};
  }
  return std::string(vocab->piece(id));
}

bool looks_like_gigaam_gguf(const std::filesystem::path &path) {
  if (read_magic(path) != kGgufMagic) {
    return false;
  }
  try {
    const auto meta = engine::assets::GgufMetadata::open(path);
    return meta.find_string("general.architecture").value_or("") == "gigaam";
  } catch (const std::exception &) {
    return false;
  }
}

std::shared_ptr<const GigaamAssets> load_gigaam_assets(const std::filesystem::path &model_path) {
  if (!std::filesystem::exists(model_path)) {
    fail("model file not found: " + model_path.string());
  }
  if (read_magic(model_path) != kGgufMagic) {
    fail("not a GGUF file: " + model_path.string());
  }
  auto out = std::make_shared<GigaamAssets>();
  out->model_path = model_path;

  const auto meta = engine::assets::GgufMetadata::open(model_path);
  if (meta.find_string("general.architecture").value_or("") != "gigaam") {
    fail("GGUF general.architecture is not \"gigaam\": " + model_path.string());
  }

  // Order of the arch's load(): capabilities, languages, tokenizer, hparams.
  out->lang_detect = meta.find_bool("stt.capability.lang_detect").value_or(false);
  for (const auto &code :
       meta.find_string_array("general.languages").value_or(std::vector<std::string>{})) {
    if (!code.empty()) {
      out->languages.push_back(code);
    }
  }
  load_tokenizer(meta, *out);
  read_hparams(meta, out->hparams);
  out->variant = meta.find_string("stt.variant").value_or("gigaam-v3-e2e-rnnt");

  out->source = engine::assets::open_tensor_source(model_path);
  check_catalog(*out->source, model_path, out->hparams);

  const auto &hp = out->hparams;
  const int64_t n_freq = hp.fe_n_fft / 2 + 1;
  out->mel_filterbank = out->source->require_f32(
      "frontend.mel_filterbank", std::vector<int64_t>{hp.fe_num_mels, n_freq});
  out->window = out->source->require_f32("frontend.window",
                                         std::vector<int64_t>{hp.fe_win_length});
  read_host_decoder(*out->source, hp, out->host_decoder);
  // The encoder tensors are streamed by each session; drop the file bytes the
  // reads above pulled in (the source re-opens lazily).
  out->source->release_storage();
  return out;
}

} // namespace engine::models::gigaam
