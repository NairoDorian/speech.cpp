// engine/models/granite_nar/runtime.cpp - inference for the native engine
// Granite Speech NAR package: a port of src/runtime/arch/granite_nar/model.cpp
// (load() weight handling, fuse_batch_norm, run()) at transcribe.cpp HEAD's
// revision, i.e. with the two parent changes speech.cpp's arch has not
// adopted yet:
//   - 585b98f7: Shaw positional bias through the skew path (pos_rows +
//     rel_shift) instead of the ctx*ctx lookup - same math, different op
//     order (graphs.cpp shaw_block_attn);
//   - b174a427: the 100k-wide BPE-CTC head projects posterior-pooled encoder
//     states in bounded chunks of 512 windows and returns only per-window
//     argmaxes - linearity makes it the arch's "pool the frame logits" up to
//     float rounding, without a [vocab, T_enc] tensor (5.7 GiB at 5 min).
//
// Flow per utterance:
//   pcm -> per-utterance log-mel -> 2-frame stack -> encoder (cat_out,
//   mid-CTC blank posterior) -> bounded BPE-CTC -> initial hypothesis
//   (empty -> empty transcript) -> projector -> clip to T_enc / downsample
//   audio tokens -> insertion slots -> context gate -> editor forward ->
//   argmax + collapse + drop eos -> decode.
//
// Differences from the arch, none numeric on the CPU:
//   - one backend + ggml-alloc per graph (engine convention) instead of the
//     arch's multi-backend scheduler; the same kernels run on the same data;
//   - the folded BatchNorm scale / bias are uploaded with the weights (the
//     arch allocated them in a separate buffer);
//   - an early input gate: when even the shortest editor text (8 slots) cannot
//     fit next to the audio tokens, the utterance is rejected before the
//     encoder runs; the arch reached the same INPUT_TOO_LONG after encoding;
//   - audio yielding zero audio tokens with a non-empty hypothesis (T_enc < 5,
//     ~0.2 s) is std::invalid_argument; the arch returned ERR_GGUF from its
//     decoder builder;
//   - TRANSCRIBE_DEBUG dump taps and TRANSCRIBE_FORCE_FLASH / _NO_FLASH are
//     not carried (the arch read the flash flags but no graph consumed them);
//   - on Vulkan / Metal the weight store may hold BF16 linears as F16 (the
//     store's policy there); the arch kept every GGUF dtype.

#include "engine/models/granite_nar/runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/models/granite_nar/decoding.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::granite_nar {

namespace {

constexpr const char *kTag = "granite_nar";
// arch model.cpp kBnEps.
constexpr float kBnEps = 1e-5f;
// Parent model.cpp kBpeCtcWindowsPerChunk (b174a427).
constexpr int kBpeCtcWindowsPerChunk = 512;

enum WeightKind : int {
  kF32 = 0,    // GET_F32: norms, biases, BatchNorm stats
  kConv = 1,   // GET_CONV: F32 / F16, kept as stored
  kLinear = 2, // GET_LIN: TRANSCRIBE_QUANT_LINEAR_TYPES, kept as stored
  kEmbed = 3,  // prj.query / prj.window_positions: F32 / BF16 / F16
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(std::string(kTag) + ": " + message);
}

std::string lname(const char *fmt, int i) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), fmt, i);
  return std::string(buf);
}

// ggml ne (fastest first) -> TensorSource row-major shape: trailing unit dims
// dropped as ggml_n_dims does, then reversed.
std::vector<int64_t> shape_from_ne(std::initializer_list<int64_t> ne) {
  std::vector<int64_t> dims(ne);
  while (dims.size() > 1 && dims.back() == 1) {
    dims.pop_back();
  }
  std::reverse(dims.begin(), dims.end());
  return dims;
}

std::string shape_string(const std::vector<int64_t> &shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += (i ? ", " : "") + std::to_string(shape[i]);
  }
  return out + "]";
}

bool is_linear_type(ggml_type t) {
  switch (t) {
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

} // namespace

// ---------------------------------------------------------------------------
// GraphRun
// ---------------------------------------------------------------------------

void GraniteNarRuntime::GraphRun::free() {
  if (gallocr != nullptr) {
    ggml_gallocr_free(gallocr);
    gallocr = nullptr;
  }
  if (ctx != nullptr) {
    ggml_free(ctx);
    ctx = nullptr;
  }
}

void GraniteNarRuntime::GraphRun::init(const char *what) {
  free();
  ggml_init_params params{graph_context_bytes(), nullptr, /*no_alloc=*/true};
  ctx = ggml_init(params);
  if (ctx == nullptr) {
    fail(std::string("failed to init the ") + what + " compute context");
  }
}

void GraniteNarRuntime::GraphRun::allocate(ggml_backend_t backend, ggml_cgraph *graph,
                                           const char *what) {
  gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  if (gallocr == nullptr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
    fail(std::string(what) + " graph allocation failed - out of memory; shorten the audio or "
                             "lower the session n_ctx");
  }
}

// ---------------------------------------------------------------------------
// Weights (arch weights.cpp build_granite_nar_weights)
// ---------------------------------------------------------------------------

ggml_tensor *GraniteNarRuntime::load(const std::string &name, std::initializer_list<int64_t> ne,
                                     int kind) {
  const auto &source = *assets_->source;
  if (!source.has_tensor(name)) {
    fail("missing tensor " + name + " in " + assets_->model_path.string());
  }
  const auto metadata = source.require_metadata(name);
  const ggml_type type = assets::ggml_type_for_tensor_dtype(metadata.dtype);
  const auto shape = shape_from_ne(ne);
  if (metadata.shape != shape) {
    fail("tensor " + name + " has shape " + shape_string(metadata.shape) + ", expected " +
         shape_string(shape));
  }
  switch (kind) {
  case kF32:
    if (type != GGML_TYPE_F32) {
      fail("tensor " + name + " must be F32, got " + metadata.dtype);
    }
    return store_->load_f32_tensor(source, name, shape).tensor;
  case kConv:
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
      fail("conv tensor " + name + " must be F32 or F16, got " + metadata.dtype);
    }
    break;
  case kLinear:
    if (!is_linear_type(type)) {
      fail("linear tensor " + name + " has unsupported type " + metadata.dtype);
    }
    break;
  case kEmbed:
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_BF16 && type != GGML_TYPE_F16) {
      fail("tensor " + name + " must be F32, BF16 or F16, got " + metadata.dtype);
    }
    break;
  default:
    fail("internal: bad weight kind");
  }
  return store_->load_tensor(source, name, assets::TensorStorageType::Native, shape).tensor;
}

void GraniteNarRuntime::load_weights() {
  const auto &hp = assets_->hparams;
  const int64_t enc_h = hp.enc_hidden;
  const int64_t enc_in = hp.enc_input_dim;
  const int64_t enc_out = hp.enc_output_dim;
  const int64_t enc_bpe_out = hp.enc_bpe_output_dim;
  const int64_t enc_inner = static_cast<int64_t>(hp.enc_n_heads) * hp.enc_head_dim;
  const int64_t enc_ffn = enc_h * hp.enc_feedforward_mult;
  const int64_t conv_inner = enc_h * hp.enc_conv_expansion;
  const int64_t conv_up_out = conv_inner * 2;
  const int64_t conv_k = hp.enc_conv_kernel_size;
  const int64_t rel_pos_len = 2 * static_cast<int64_t>(hp.enc_max_pos_emb) + 1;

  const int64_t prj_h = hp.prj_hidden;
  const int64_t prj_im = static_cast<int64_t>(hp.prj_hidden) * hp.prj_mlp_ratio;
  const int64_t prj_kv_in = static_cast<int64_t>(hp.prj_num_encoder_layers) * hp.prj_encoder_dim;
  const int64_t prj_n_query = hp.prj_block_size / hp.prj_downsample_rate;

  const int64_t dec_h = hp.dec_hidden;
  const int64_t q_out = static_cast<int64_t>(hp.dec_n_heads) * hp.dec_head_dim;
  const int64_t kv_out = static_cast<int64_t>(hp.dec_n_kv_heads) * hp.dec_head_dim;
  const int64_t dec_im = hp.dec_intermediate;

  auto &et = weights_.enc_top;
  et.input_linear_w = load("enc.input_linear.weight", {enc_in, enc_h}, kLinear);
  et.input_linear_b = load("enc.input_linear.bias", {enc_h}, kF32);
  et.ctc_proj_w = load("enc.ctc_proj.weight", {enc_h, enc_out}, kLinear);
  et.ctc_proj_b = load("enc.ctc_proj.bias", {enc_out}, kF32);
  et.ctc_bypass_w = load("enc.ctc_bypass.weight", {enc_out, enc_h}, kLinear);
  et.ctc_bypass_b = load("enc.ctc_bypass.bias", {enc_h}, kF32);
  if (enc_bpe_out > 0) {
    et.ctc_bpe_w = load("enc.ctc_bpe.weight", {enc_h, enc_bpe_out}, kLinear);
    et.ctc_bpe_b = load("enc.ctc_bpe.bias", {enc_bpe_out}, kF32);
  }

  weights_.enc_blocks.assign(static_cast<size_t>(hp.enc_n_layers), EncBlockWeights{});
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &b = weights_.enc_blocks[static_cast<size_t>(i)];
    auto n = [i](const char *fmt) { return lname(fmt, i); };
    b.norm_ff1_w = load(n("enc.blocks.%d.norm_ff1.weight"), {enc_h}, kF32);
    b.norm_ff1_b = load(n("enc.blocks.%d.norm_ff1.bias"), {enc_h}, kF32);
    b.ff1_up_w = load(n("enc.blocks.%d.ff1_up.weight"), {enc_h, enc_ffn}, kLinear);
    b.ff1_up_b = load(n("enc.blocks.%d.ff1_up.bias"), {enc_ffn}, kF32);
    b.ff1_down_w = load(n("enc.blocks.%d.ff1_down.weight"), {enc_ffn, enc_h}, kLinear);
    b.ff1_down_b = load(n("enc.blocks.%d.ff1_down.bias"), {enc_h}, kF32);

    b.norm_attn_w = load(n("enc.blocks.%d.norm_attn.weight"), {enc_h}, kF32);
    b.norm_attn_b = load(n("enc.blocks.%d.norm_attn.bias"), {enc_h}, kF32);
    b.attn_q_w = load(n("enc.blocks.%d.attn_q.weight"), {enc_h, enc_inner}, kLinear);
    b.attn_kv_w = load(n("enc.blocks.%d.attn_kv.weight"), {enc_h, 2 * enc_inner}, kLinear);
    b.attn_out_w = load(n("enc.blocks.%d.attn_out.weight"), {enc_inner, enc_h}, kLinear);
    b.attn_out_b = load(n("enc.blocks.%d.attn_out.bias"), {enc_h}, kF32);
    b.attn_rel_pos_emb =
        load(n("enc.blocks.%d.attn_rel_pos_emb.weight"), {hp.enc_head_dim, rel_pos_len}, kLinear);

    b.norm_conv_w = load(n("enc.blocks.%d.norm_conv.weight"), {enc_h}, kF32);
    b.norm_conv_b = load(n("enc.blocks.%d.norm_conv.bias"), {enc_h}, kF32);
    b.conv_pointwise1_w =
        load(n("enc.blocks.%d.conv_pointwise1.weight"), {1, enc_h, conv_up_out}, kConv);
    b.conv_pointwise1_b = load(n("enc.blocks.%d.conv_pointwise1.bias"), {conv_up_out}, kF32);
    b.conv_depthwise_w = load(n("enc.blocks.%d.conv_depthwise.weight"), {conv_k, 1, conv_inner}, kConv);
    b.conv_pointwise2_w =
        load(n("enc.blocks.%d.conv_pointwise2.weight"), {1, conv_inner, enc_h}, kConv);
    b.conv_pointwise2_b = load(n("enc.blocks.%d.conv_pointwise2.bias"), {enc_h}, kF32);

    b.norm_ff2_w = load(n("enc.blocks.%d.norm_ff2.weight"), {enc_h}, kF32);
    b.norm_ff2_b = load(n("enc.blocks.%d.norm_ff2.bias"), {enc_h}, kF32);
    b.ff2_up_w = load(n("enc.blocks.%d.ff2_up.weight"), {enc_h, enc_ffn}, kLinear);
    b.ff2_up_b = load(n("enc.blocks.%d.ff2_up.bias"), {enc_ffn}, kF32);
    b.ff2_down_w = load(n("enc.blocks.%d.ff2_down.weight"), {enc_ffn, enc_h}, kLinear);
    b.ff2_down_b = load(n("enc.blocks.%d.ff2_down.bias"), {enc_h}, kF32);

    b.norm_post_w = load(n("enc.blocks.%d.norm_post.weight"), {enc_h}, kF32);
    b.norm_post_b = load(n("enc.blocks.%d.norm_post.bias"), {enc_h}, kF32);
  }

  auto &pt = weights_.proj_top;
  pt.layer_projector_w = load("prj.layer_projector.weight", {prj_kv_in, prj_h}, kLinear);
  pt.layer_projector_b = load("prj.layer_projector.bias", {prj_h}, kF32);
  pt.out_norm_w = load("prj.out_norm.weight", {prj_h}, kF32);
  pt.out_norm_b = load("prj.out_norm.bias", {prj_h}, kF32);
  pt.out_linear_w = load("prj.out_linear.weight", {prj_h, hp.prj_llm_dim}, kLinear);
  pt.out_linear_b = load("prj.out_linear.bias", {hp.prj_llm_dim}, kF32);
  pt.query = load("prj.query", {prj_h, prj_n_query, 1}, kEmbed);
  pt.window_positions = load("prj.window_positions", {prj_h, hp.prj_block_size, 1}, kEmbed);
  pt.layer_norms_w.assign(static_cast<size_t>(hp.prj_num_encoder_layers), nullptr);
  pt.layer_norms_b.assign(static_cast<size_t>(hp.prj_num_encoder_layers), nullptr);
  for (int j = 0; j < hp.prj_num_encoder_layers; ++j) {
    pt.layer_norms_w[static_cast<size_t>(j)] =
        load(lname("prj.layer_norms.%d.weight", j), {hp.prj_encoder_dim}, kF32);
    pt.layer_norms_b[static_cast<size_t>(j)] =
        load(lname("prj.layer_norms.%d.bias", j), {hp.prj_encoder_dim}, kF32);
  }

  weights_.proj_blocks.assign(static_cast<size_t>(hp.prj_n_layers), ProjBlockWeights{});
  for (int i = 0; i < hp.prj_n_layers; ++i) {
    auto &b = weights_.proj_blocks[static_cast<size_t>(i)];
    auto n = [i](const char *fmt) { return lname(fmt, i); };
    b.norm_attn_w = load(n("prj.blocks.%d.norm_attn.weight"), {prj_h}, kF32);
    b.norm_attn_b = load(n("prj.blocks.%d.norm_attn.bias"), {prj_h}, kF32);
    b.cross_attn_q_w = load(n("prj.blocks.%d.cross_attn_q.weight"), {prj_h, prj_h}, kLinear);
    b.cross_attn_q_b = load(n("prj.blocks.%d.cross_attn_q.bias"), {prj_h}, kF32);
    b.cross_attn_k_w = load(n("prj.blocks.%d.cross_attn_k.weight"), {prj_h, prj_h}, kLinear);
    b.cross_attn_k_b = load(n("prj.blocks.%d.cross_attn_k.bias"), {prj_h}, kF32);
    b.cross_attn_v_w = load(n("prj.blocks.%d.cross_attn_v.weight"), {prj_h, prj_h}, kLinear);
    b.cross_attn_v_b = load(n("prj.blocks.%d.cross_attn_v.bias"), {prj_h}, kF32);
    b.cross_attn_o_w = load(n("prj.blocks.%d.cross_attn_o.weight"), {prj_h, prj_h}, kLinear);
    b.cross_attn_o_b = load(n("prj.blocks.%d.cross_attn_o.bias"), {prj_h}, kF32);
    b.norm_ffn_w = load(n("prj.blocks.%d.norm_ffn.weight"), {prj_h}, kF32);
    b.norm_ffn_b = load(n("prj.blocks.%d.norm_ffn.bias"), {prj_h}, kF32);
    b.ffn_fc1_w = load(n("prj.blocks.%d.ffn_fc1.weight"), {prj_h, prj_im}, kLinear);
    b.ffn_fc1_b = load(n("prj.blocks.%d.ffn_fc1.bias"), {prj_im}, kF32);
    b.ffn_fc2_w = load(n("prj.blocks.%d.ffn_fc2.weight"), {prj_im, prj_h}, kLinear);
    b.ffn_fc2_b = load(n("prj.blocks.%d.ffn_fc2.bias"), {prj_h}, kF32);
  }

  weights_.dec_token_embd = load("dec.token_embd.weight", {dec_h, hp.dec_vocab_size}, kLinear);
  weights_.dec_blocks.assign(static_cast<size_t>(hp.dec_n_layers), DecBlockWeights{});
  for (int i = 0; i < hp.dec_n_layers; ++i) {
    auto &b = weights_.dec_blocks[static_cast<size_t>(i)];
    auto n = [i](const char *fmt) { return lname(fmt, i); };
    b.norm_attn_w = load(n("dec.blocks.%d.norm_attn.weight"), {dec_h}, kF32);
    b.norm_ffn_w = load(n("dec.blocks.%d.norm_ffn.weight"), {dec_h}, kF32);
    b.attn_q_w = load(n("dec.blocks.%d.attn.q.weight"), {dec_h, q_out}, kLinear);
    b.attn_k_w = load(n("dec.blocks.%d.attn.k.weight"), {dec_h, kv_out}, kLinear);
    b.attn_v_w = load(n("dec.blocks.%d.attn.v.weight"), {dec_h, kv_out}, kLinear);
    b.attn_o_w = load(n("dec.blocks.%d.attn.o.weight"), {q_out, dec_h}, kLinear);
    b.ffn_gate_w = load(n("dec.blocks.%d.ffn.gate.weight"), {dec_h, dec_im}, kLinear);
    b.ffn_up_w = load(n("dec.blocks.%d.ffn.up.weight"), {dec_h, dec_im}, kLinear);
    b.ffn_down_w = load(n("dec.blocks.%d.ffn.down.weight"), {dec_im, dec_h}, kLinear);
  }
  weights_.dec_output_norm = load("dec.output_norm.weight", {dec_h}, kF32);
}

// arch model.cpp fuse_batch_norm: scale = w / sqrt(var + 1e-5),
// bias = b - mean * scale, per channel, in f32 - host side, same arithmetic.
void GraniteNarRuntime::fuse_batch_norm() {
  const auto &source = *assets_->source;
  const int64_t inner =
      static_cast<int64_t>(assets_->hparams.enc_hidden) * assets_->hparams.enc_conv_expansion;
  auto read_f32 = [&](const std::string &name) {
    if (!source.has_tensor(name)) {
      fail("missing tensor " + name);
    }
    const auto metadata = source.require_metadata(name);
    if (assets::ggml_type_for_tensor_dtype(metadata.dtype) != GGML_TYPE_F32 ||
        metadata.shape != std::vector<int64_t>{inner}) {
      fail("tensor " + name + " must be F32 [" + std::to_string(inner) + "]");
    }
    return source.require_f32(name, std::optional<std::vector<int64_t>>(std::vector<int64_t>{inner}));
  };
  for (size_t i = 0; i < weights_.enc_blocks.size(); ++i) {
    const int li = static_cast<int>(i);
    const std::vector<float> bn_w = read_f32(lname("enc.blocks.%d.conv_bn.weight", li));
    const std::vector<float> bn_b = read_f32(lname("enc.blocks.%d.conv_bn.bias", li));
    const std::vector<float> rm = read_f32(lname("enc.blocks.%d.conv_bn.running_mean", li));
    const std::vector<float> rv = read_f32(lname("enc.blocks.%d.conv_bn.running_var", li));
    std::vector<float> fused_s(static_cast<size_t>(inner));
    std::vector<float> fused_b(static_cast<size_t>(inner));
    for (int64_t c = 0; c < inner; ++c) {
      const float s = bn_w[c] / std::sqrt(rv[c] + kBnEps);
      fused_s[c] = s;
      fused_b[c] = bn_b[c] - rm[c] * s;
    }
    auto &b = weights_.enc_blocks[i];
    b.conv_bn_fused_scale =
        store_->make_f32(core::TensorShape::from_dims({inner}), std::move(fused_s)).tensor;
    b.conv_bn_fused_bias =
        store_->make_f32(core::TensorShape::from_dims({inner}), std::move(fused_b)).tensor;
  }
}

GraniteNarRuntime::GraniteNarRuntime(std::shared_ptr<const GraniteNarAssets> model_assets,
                                     core::ExecutionContext &execution_context,
                                     GraniteNarRuntimeOptions options)
    : assets_(std::move(model_assets)), execution_context_(execution_context), options_(options) {
  if (!assets_ || !assets_->source) {
    fail("runtime requires loaded assets");
  }
  backend_ = execution_context_.backend();
  if (backend_ == nullptr) {
    fail("execution backend is not initialized");
  }
  store_ = std::make_shared<core::BackendWeightStore>(
      backend_, execution_context_.backend_type(), "granite_nar.weights", 512ull * 1024ull * 1024ull);
  load_weights();
  fuse_batch_norm();
  store_->upload();
  assets_->source->release_storage();

  conv_dw_direct_ = detect_conv_dw_direct();

  // arch model.cpp load(): the MelConfig. The NAR GGUF ships no
  // pre_emphasis / f_min / f_max KVs; the arch hardcodes the
  // WhisperFeatureExtractor defaults, and so does this.
  const auto &hp = assets_->hparams;
  GraniteMelConfig cfg;
  cfg.sample_rate = hp.fe_sample_rate;
  cfg.num_mels = hp.fe_num_mels;
  cfg.n_fft = hp.fe_n_fft;
  cfg.win_length = hp.fe_win_length;
  cfg.hop_length = hp.fe_hop_length;
  cfg.pre_emphasis = 0.0f;
  cfg.f_min = 0.0f;
  cfg.f_max = static_cast<float>(hp.fe_sample_rate) / 2.0f;
  cfg.pad_mode = hp.fe_pad_mode;
  cfg.window_type = hp.fe_window;
  cfg.normalize = hp.fe_normalize;
  cfg.filterbank = assets_->mel_filterbank;
  cfg.window = assets_->window;
  mel_.emplace(cfg);
}

GraniteNarRuntime::~GraniteNarRuntime() = default;

int32_t GraniteNarRuntime::sample_rate() const noexcept {
  return assets_ ? assets_->hparams.fe_sample_rate : 16000;
}

int GraniteNarRuntime::n_threads() const {
  return std::max(1, execution_context_.config().threads);
}

void GraniteNarRuntime::compute(ggml_cgraph *graph, const char *what) {
  core::set_backend_threads(backend_, n_threads());
  if (core::compute_backend_graph(backend_, graph, nullptr, what) != GGML_STATUS_SUCCESS) {
    fail(std::string(what) + " compute failed");
  }
  ggml_backend_synchronize(backend_);
}

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

void GraniteNarRuntime::encode(const std::vector<float> &mel_stacked, int T_enc,
                               std::vector<float> &enc_cat, int64_t &cat_h,
                               int64_t &final_offset, std::vector<float> &non_blank) {
  const auto &hp = assets_->hparams;
  GraphRun run;
  run.init("encoder");
  EncoderBuild eb =
      build_encoder_graph(run.ctx, weights_, hp, T_enc, options_.shaw_bias, conv_dw_direct_);
  if (eb.graph == nullptr || eb.cat_out == nullptr || eb.mid_blank_probs == nullptr) {
    fail("encoder graph build failed (the projector captures must include the final encoder "
         "layer and every index must name a layer)");
  }
  run.allocate(backend_, eb.graph, "encoder");

  // A fresh graph per call: every input is uploaded after allocation and
  // before the single compute, so nothing is reused across runs.
  ggml_backend_tensor_set(eb.mel_in, mel_stacked.data(), 0, mel_stacked.size() * sizeof(float));
  if (eb.pos_rows != nullptr) {
    const std::vector<int32_t> rows = precompute_pos_rows(hp.enc_context_size, hp.enc_max_pos_emb);
    ggml_backend_tensor_set(eb.pos_rows, rows.data(), 0, rows.size() * sizeof(int32_t));
  }
  if (eb.attention_dists != nullptr) {
    const std::vector<int32_t> dists =
        precompute_attention_dists(hp.enc_context_size, hp.enc_max_pos_emb);
    ggml_backend_tensor_set(eb.attention_dists, dists.data(), 0, dists.size() * sizeof(int32_t));
  }
  {
    const std::vector<float> mask =
        build_block_mask(hp.enc_context_size, eb.n_blocks_local, eb.last_block_rem);
    ggml_backend_tensor_set(eb.last_block_mask, mask.data(), 0, mask.size() * sizeof(float));
  }
  if (eb.zero_pad != nullptr) {
    const std::vector<float> zeros(static_cast<size_t>(ggml_nelements(eb.zero_pad)), 0.0f);
    ggml_backend_tensor_set(eb.zero_pad, zeros.data(), 0, zeros.size() * sizeof(float));
  }
  compute(eb.graph, "encoder");

  // ne [cat_h, T_enc] == host-row-major [T_enc][cat_h].
  cat_h = eb.cat_out->ne[0];
  enc_cat.resize(static_cast<size_t>(cat_h) * static_cast<size_t>(eb.cat_out->ne[1]));
  ggml_backend_tensor_get(eb.cat_out, enc_cat.data(), 0, enc_cat.size() * sizeof(float));
  final_offset = eb.final_capture_offset;

  std::vector<float> mid_blank(static_cast<size_t>(T_enc));
  ggml_backend_tensor_get(eb.mid_blank_probs, mid_blank.data(), 0, mid_blank.size() * sizeof(float));
  non_blank.resize(static_cast<size_t>(T_enc));
  for (int i = 0; i < T_enc; ++i) {
    non_blank[static_cast<size_t>(i)] = 1.0f - mid_blank[static_cast<size_t>(i)];
  }
}

// Parent model.cpp compute_bpe_ctc_initial_hypothesis (b174a427).
std::vector<int32_t> GraniteNarRuntime::bpe_hypothesis(const std::vector<float> &enc_cat,
                                                       int64_t cat_h, int T_enc,
                                                       int64_t final_offset,
                                                       const std::vector<float> &non_blank,
                                                       const runtime::RunControl &control) {
  const auto &hp = assets_->hparams;
  const int pool_window = hp.enc_bpe_pool_window;
  const int hidden = hp.enc_hidden;
  if (T_enc <= 0 || pool_window <= 0 || final_offset < 0 || final_offset + hidden > cat_h ||
      enc_cat.size() < static_cast<size_t>(cat_h) * static_cast<size_t>(T_enc) ||
      non_blank.size() < static_cast<size_t>(T_enc)) {
    fail("BPE-CTC: inconsistent encoder outputs");
  }
  const int n_windows = (T_enc + pool_window - 1) / pool_window;
  const int chunk_windows = std::min(kBpeCtcWindowsPerChunk, n_windows);

  GraphRun run;
  run.init("BPE-CTC");
  BpeCtcBuild bb = build_bpe_ctc_graph(run.ctx, weights_, hp, chunk_windows);
  if (bb.graph == nullptr || bb.hidden_in == nullptr || bb.token_ids == nullptr) {
    // The parent returns TRANSCRIBE_ERR_GGUF here (e.g. bpe_output_dim 0).
    fail("BPE-CTC graph build failed (the GGUF has no enc.ctc_bpe head)");
  }
  run.allocate(backend_, bb.graph, "BPE-CTC");

  std::vector<float> hidden_chunk(static_cast<size_t>(hidden) * static_cast<size_t>(bb.n_windows));
  std::vector<int32_t> chunk_ids(static_cast<size_t>(bb.n_windows), hp.enc_bpe_blank_id);
  std::vector<uint8_t> valid(static_cast<size_t>(bb.n_windows), 0);
  std::vector<int32_t> out;
  out.reserve(static_cast<size_t>(n_windows));
  int prev = -1;
  const int n_chunks = (n_windows + chunk_windows - 1) / chunk_windows;
  for (int w0 = 0, chunk = 0; w0 < n_windows; w0 += chunk_windows, ++chunk) {
    control.emit_progress("granite_nar.ctc", chunk, n_chunks);
    std::fill(hidden_chunk.begin(), hidden_chunk.end(), 0.0f);
    std::fill(valid.begin(), valid.end(), 0);
    const int actual = std::min(chunk_windows, n_windows - w0);
    for (int w = 0; w < actual; ++w) {
      const int t0 = (w0 + w) * pool_window;
      const int t1 = std::min(t0 + pool_window, T_enc);
      valid[static_cast<size_t>(w)] = pool_window_hidden(
          enc_cat.data(), cat_h, final_offset, hidden, non_blank.data(), t0, t1,
          hidden_chunk.data() + static_cast<size_t>(w) * hidden)
                                          ? 1
                                          : 0;
    }
    // This graph is computed once per chunk: its only input (hidden_in) is
    // re-uploaded before EVERY compute (the cached-graph rule), so the
    // allocator may reuse its memory freely between chunks.
    ggml_backend_tensor_set(bb.hidden_in, hidden_chunk.data(), 0,
                            hidden_chunk.size() * sizeof(float));
    compute(bb.graph, "BPE-CTC");
    ggml_backend_tensor_get(bb.token_ids, chunk_ids.data(), 0, chunk_ids.size() * sizeof(int32_t));
    collapse_bpe_ctc(chunk_ids.data(), valid.data(), actual, hp.enc_bpe_blank_id, prev, out);
  }
  return out;
}

std::vector<float> GraniteNarRuntime::project(const std::vector<float> &enc_cat, int T_enc,
                                              int &n_audio) {
  const auto &hp = assets_->hparams;
  GraphRun run;
  run.init("projector");
  ProjectorBuild pb = build_projector_graph(run.ctx, weights_, hp, T_enc);
  if (pb.graph == nullptr || pb.out == nullptr) {
    fail("projector graph build failed");
  }
  run.allocate(backend_, pb.graph, "projector");
  if (enc_cat.size() * sizeof(float) != ggml_nbytes(pb.enc_in)) {
    fail("projector input size mismatch");
  }
  ggml_backend_tensor_set(pb.enc_in, enc_cat.data(), 0, enc_cat.size() * sizeof(float));
  if (pb.enc_pad != nullptr) {
    const std::vector<float> zeros(static_cast<size_t>(ggml_nelements(pb.enc_pad)), 0.0f);
    ggml_backend_tensor_set(pb.enc_pad, zeros.data(), 0, zeros.size() * sizeof(float));
  }
  compute(pb.graph, "projector");

  // arch run(): the reference emits nblocks * n_query tokens but the LM only
  // consumes projected_length = T_enc / downsample_rate of them (dropping the
  // ones that came from the zero-padded window).
  n_audio = std::min(T_enc / hp.prj_downsample_rate, pb.n_audio_tokens);
  const int64_t llm_dim = pb.out->ne[0];
  std::vector<float> rows(static_cast<size_t>(llm_dim) * static_cast<size_t>(std::max(0, n_audio)));
  if (!rows.empty()) {
    ggml_backend_tensor_get(pb.out, rows.data(), 0, rows.size() * sizeof(float));
  }
  return rows;
}

std::vector<float> GraniteNarRuntime::edit(const std::vector<float> &audio_rows, int n_audio,
                                           const std::vector<int32_t> &text_ids, int &vocab) {
  const auto &hp = assets_->hparams;
  const int n_text = static_cast<int>(text_ids.size());
  GraphRun run;
  run.init("editor");
  ForwardBuild fb = build_forward_graph(run.ctx, weights_, hp, n_audio, n_text);
  if (fb.graph == nullptr || fb.out == nullptr) {
    fail("editor graph build failed");
  }
  run.allocate(backend_, fb.graph, "editor");

  ggml_backend_tensor_set(fb.audio_in, audio_rows.data(), 0, audio_rows.size() * sizeof(float));
  ggml_backend_tensor_set(fb.text_ids_in, text_ids.data(), 0, text_ids.size() * sizeof(int32_t));
  {
    std::vector<int32_t> positions(static_cast<size_t>(fb.T_total));
    for (int i = 0; i < fb.T_total; ++i) {
      positions[static_cast<size_t>(i)] = i;
    }
    ggml_backend_tensor_set(fb.positions_in, positions.data(), 0, positions.size() * sizeof(int32_t));
  }
  compute(fb.graph, "editor");

  // ne [vocab, n_text] == host-row-major [n_text][vocab].
  vocab = static_cast<int>(fb.out->ne[0]);
  std::vector<float> logits(static_cast<size_t>(vocab) * static_cast<size_t>(n_text));
  ggml_backend_tensor_get(fb.out, logits.data(), 0, logits.size() * sizeof(float));
  return logits;
}

// ---------------------------------------------------------------------------
// transcribe (arch run())
// ---------------------------------------------------------------------------

GraniteNarTranscription GraniteNarRuntime::transcribe(const std::vector<float> &pcm,
                                                      const runtime::RunControl &control) {
  const auto &hp = assets_->hparams;
  control.emit_progress("granite_nar.mel", 0, 1);
  if (pcm.empty()) {
    throw std::invalid_argument("granite_nar: empty audio input");
  }

  // Mel + 2-frame stack (arch encoder.cpp compute_mel_encoder_input).
  std::vector<float> mel;
  int n_mels = 0;
  int n_frames = 0;
  if (!mel_->compute(pcm.data(), pcm.size(), mel, n_mels, n_frames, n_threads()) ||
      n_mels != hp.fe_num_mels) {
    throw std::invalid_argument("granite_nar: audio too short for the mel frontend (" +
                                std::to_string(pcm.size()) + " samples)");
  }
  std::vector<float> stacked;
  const int T_enc = stack_mel_frames(mel, n_mels, n_frames, stacked);
  if (T_enc <= 0) {
    throw std::invalid_argument("granite_nar: audio too short (fewer than two mel frames)");
  }
  mel.clear();
  mel.shrink_to_fit();

  // Early gate (see the file header): the smallest editor text is 8 slots.
  const int ceiling = granite_nar_context_ceiling(options_.n_ctx, hp);
  {
    const int n_query = granite_nar_num_queries(hp);
    const int nblocks = (T_enc + hp.prj_block_size - 1) / hp.prj_block_size;
    const int n_audio_min = std::min(T_enc / hp.prj_downsample_rate, nblocks * n_query);
    if (n_audio_min + kMinEditorText > ceiling) {
      throw runtime::InputTooLong(
          "granite_nar: input too long - " + std::to_string(n_audio_min) + " audio + at least " +
          std::to_string(kMinEditorText) + " text tokens exceed the " + std::to_string(ceiling) +
          "-token context. Shorten audio (max_audio_ms = " + std::to_string(assets_->max_audio_ms) +
          ") or split it into segments.");
    }
  }

  control.emit_progress("granite_nar.encode", 0, 1);
  std::vector<float> enc_cat;
  std::vector<float> non_blank;
  int64_t cat_h = 0;
  int64_t final_offset = -1;
  encode(stacked, T_enc, enc_cat, cat_h, final_offset, non_blank);
  stacked.clear();
  stacked.shrink_to_fit();

  const std::vector<int32_t> hyp_ids =
      bpe_hypothesis(enc_cat, cat_h, T_enc, final_offset, non_blank, control);
  GraniteNarTranscription out;
  out.n_hypothesis_tokens = static_cast<int>(hyp_ids.size());
  if (hyp_ids.empty()) {
    // arch run(): no tokens -> empty transcript, OK.
    return out;
  }

  control.emit_progress("granite_nar.project", 0, 1);
  int n_audio = 0;
  std::vector<float> audio_rows = project(enc_cat, T_enc, n_audio);
  enc_cat.clear();
  enc_cat.shrink_to_fit();
  if (n_audio <= 0) {
    throw std::invalid_argument("granite_nar: audio too short - it yields no audio tokens (T_enc " +
                                std::to_string(T_enc) + " < downsample_rate " +
                                std::to_string(hp.prj_downsample_rate) + ")");
  }
  out.n_audio_tokens = n_audio;

  std::vector<int32_t> text_ids;
  add_insertion_slots(hyp_ids, hp.dec_eos_id, text_ids);
  const int n_text = static_cast<int>(text_ids.size());
  out.n_text_slots = n_text;

  // arch run() input-length gate: one bidirectional pass, every RoPE
  // position bounded by the ceiling; no output budget to reserve.
  if (n_audio + n_text > ceiling) {
    throw runtime::InputTooLong("granite_nar: input too long - " + std::to_string(n_audio) +
                                 " audio + " + std::to_string(n_text) +
                                 " text tokens exceed the " + std::to_string(ceiling) +
                                 "-token context. Shorten audio or split it into segments.");
  }

  // arch run(): divide by embedding_multiplier on the host so the in-graph
  // x emb_mul round-trips the audio rows.
  const float emb_mul = hp.dec_embedding_multiplier;
  if (emb_mul > 0.0f) {
    const float inv = 1.0f / emb_mul;
    for (auto &v : audio_rows) {
      v *= inv;
    }
  }

  control.emit_progress("granite_nar.edit", 0, 1);
  int vocab = 0;
  const std::vector<float> logits = edit(audio_rows, n_audio, text_ids, vocab);

  std::vector<int32_t> final_ids;
  argmax_collapse_drop_eos(logits, vocab, n_text, hp.dec_eos_id, final_ids);
  out.text = assets_->decode(final_ids);
  return out;
}

} // namespace engine::models::granite_nar
