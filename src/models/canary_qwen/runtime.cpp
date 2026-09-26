// engine/models/canary_qwen/runtime.cpp - inference for the native engine
// Canary-Qwen package: a port of src/runtime/arch/canary_qwen/model.cpp
// (load() weight transformations, run(), run_batch()).
//
// Flow per utterance (identical to the arch):
//   pcm -> NeMo mel (per_feature) -> FastConformer (32 blocks) -> perception
//   projection -> [hidden, T_enc] audio rows
//   prompt = prefix + <|audioplaceholder|> * T_enc + suffix; input gate
//   (T_prompt + 256 <= context ceiling), decode budget (#165)
//   prefill (audio rows spliced over the locator rows) -> greedy argmax step
//   loop until <|im_end|>, the budget, or the KV width.
//
// Weight regime (the arch's post-load transformations, reproduced at upload):
//   - conv pointwise1 / depthwise / pointwise2 kernels stored F16 are
//     promoted to F32 on every backend (ggml_conv_2d_dw_direct misbehaves on
//     F16 kernels);
//   - on a CPU backend every BF16 linear (encoder, projection, token
//     embedding / tied lm_head, decoder) is promoted to F32 (the reference's
//     F32 regime);
//   - BatchNorm is folded into (scale, bias) on the host, eps 1e-5;
//   - each decoder layer's gate and up projections are packed into one
//     [hidden, 2 * intermediate] tensor (gate bytes, then up bytes).
// Everything else keeps its GGUF dtype (quantized linears stay quantized).
//
// Graphs run on the session's single backend through ggml-alloc; every graph
// is built per run except the step graphs, whose every input (token,
// position, KV row, mask) is re-uploaded before each compute.

#include "engine/models/canary_qwen/runtime.h"

#include "engine/framework/core/backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace engine::models::canary_qwen {

namespace {

constexpr const char *kTag = "canary_qwen";
constexpr float kBnEps = 1e-5f;

enum WeightKind : int {
  kF32 = 0,        // GET_F32: norms, biases, pos_bias_u/v
  kPreConv = 1,    // GET_CONV in the pre_encode stem: kept as stored
  kBlockConv = 2,  // GET_CONV in a block: F16 promoted to F32
  kLinear = 3,     // GET_LIN: BF16 promoted to F32 on CPU
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(std::string(kTag) + ": " + message);
}

std::string lname(const char *fmt, int i) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), fmt, i);
  return std::string(buf);
}

// ggml ne (fastest first) -> the TensorSource's row-major shape: trailing
// unit dims dropped as ggml_n_dims does, then reversed.
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

int next_pow2_window(int need) {
  int n = 1024;
  while (n < need) {
    n *= 2;
  }
  return n;
}

int32_t host_argmax(const std::vector<float> &v) {
  int32_t best = 0;
  float best_v = v[0];
  for (int32_t i = 1; i < static_cast<int32_t>(v.size()); ++i) {
    if (v[i] > best_v) {
      best_v = v[i];
      best = i;
    }
  }
  return best;
}

// transcribe-decode-budget.h (#165), verbatim.
constexpr int k_transcript_tokens_per_sec = 12;

int predict_transcript_tokens(int audio_tokens, double ms_per_audio_token) {
  if (audio_tokens <= 0) {
    return 0;
  }
  if (!(ms_per_audio_token > 0.0)) {
    return audio_tokens;
  }
  const double seconds = static_cast<double>(audio_tokens) * ms_per_audio_token / 1000.0;
  const double predicted = seconds * k_transcript_tokens_per_sec;
  if (predicted <= 0.0) {
    return 0;
  }
  constexpr double k_int_max = 2147483647.0;
  return predicted >= k_int_max ? 2147483647 : static_cast<int>(predicted);
}

int pick_decode_budget(int predicted, int floor_tokens, int t_prompt, int ceiling) {
  int budget = std::max(floor_tokens, predicted);
  const int room = ceiling - t_prompt;
  if (budget > room) {
    budget = room;
  }
  return budget > 0 ? budget : 0;
}

std::string too_long_message(int T_enc, int prompt_tokens, int ceiling, int need) {
  return std::string(kTag) + ": input too long - " + std::to_string(T_enc) + " audio + " +
         std::to_string(prompt_tokens) + " prompt tokens leave no room for output within the " +
         std::to_string(ceiling) + "-token context (need " + std::to_string(need) +
         "). Shorten the audio or split it into segments.";
}

std::vector<ggml_fp16_t> causal_mask_f16(int T) {
  const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
  const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
  std::vector<ggml_fp16_t> mask(static_cast<size_t>(T) * T, neg_inf);
  for (int r = 0; r < T; ++r) {
    for (int c = 0; c <= r; ++c) {
      mask[static_cast<size_t>(r) * T + c] = zero;
    }
  }
  return mask;
}

} // namespace

void apply_flash_env_overrides(CanaryQwenRuntimeOptions &options) {
  if (conformer::env_flag("TRANSCRIBE_NO_FLASH")) {
    options.encoder_use_flash = false;
    options.decoder_use_flash = false;
  }
  if (conformer::env_flag("TRANSCRIBE_FORCE_FLASH")) {
    options.encoder_use_flash = true;
    options.decoder_use_flash = true;
  }
}

// ---------------------------------------------------------------------------
// GraphRun
// ---------------------------------------------------------------------------

void CanaryQwenRuntime::GraphRun::free() {
  if (gallocr != nullptr) {
    ggml_gallocr_free(gallocr);
    gallocr = nullptr;
  }
  if (ctx != nullptr) {
    ggml_free(ctx);
    ctx = nullptr;
  }
  graph = nullptr;
}

void CanaryQwenRuntime::GraphRun::init(size_t mem_bytes, const char *what) {
  free();
  ggml_init_params params{mem_bytes, nullptr, /*no_alloc=*/true};
  ctx = ggml_init(params);
  if (ctx == nullptr) {
    fail(std::string("failed to init the ") + what + " compute context");
  }
}

void CanaryQwenRuntime::GraphRun::allocate(ggml_backend_t backend, ggml_cgraph *g, const char *what) {
  graph = g;
  gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  if (gallocr == nullptr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
    fail(std::string(what) + " graph allocation failed (out of memory); shorten the audio or "
                             "lower the session n_ctx");
  }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

ggml_tensor *CanaryQwenRuntime::load(const std::string &name, std::initializer_list<int64_t> ne,
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

  assets::TensorStorageType storage = assets::TensorStorageType::Native;
  switch (kind) {
  case kF32:
    if (type != GGML_TYPE_F32) {
      fail("tensor " + name + " must be F32, got " + metadata.dtype);
    }
    break;
  case kPreConv:
  case kBlockConv:
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
      fail("conv tensor " + name + " must be F32 or F16, got " + metadata.dtype);
    }
    if (kind == kBlockConv && type == GGML_TYPE_F16) {
      storage = assets::TensorStorageType::F32;
    }
    break;
  case kLinear:
    if (!is_linear_type(type)) {
      fail("linear tensor " + name + " has unsupported type " + metadata.dtype);
    }
    if (cpu_backend_ && type == GGML_TYPE_BF16) {
      storage = assets::TensorStorageType::F32;
    }
    break;
  default:
    fail("internal: bad weight kind");
  }
  return store_->load_tensor(source, name, storage, shape).tensor;
}

void CanaryQwenRuntime::load_weights() {
  const auto &hp = assets_->hparams;
  const int64_t channels = hp.enc_subsampling_chans;
  const int64_t d_enc = hp.enc_d_model;
  const int64_t d_ff_e = hp.enc_d_ff;
  const int64_t n_heads = hp.enc_n_heads;
  const int64_t head_dim = hp.enc_d_model / hp.enc_n_heads;
  const int64_t k = hp.enc_conv_kernel;
  const int64_t out_dim = hp.perception_output_dim;
  const int64_t vocab = hp.dec_vocab_size;
  const int64_t hidden = hp.dec_hidden;
  const int64_t kv_dim = static_cast<int64_t>(hp.dec_n_kv_heads) * hp.dec_head_dim;
  const int64_t pre_encode_in = channels * (hp.fe_num_mels / hp.enc_subsampling_factor);

  auto &pe = weights_.pre_encode;
  pe.conv0_w = load("enc.pre_encode.conv.0.weight", {3, 3, 1, channels}, kPreConv);
  pe.conv0_b = load("enc.pre_encode.conv.0.bias", {channels}, kF32);
  pe.conv2_w = load("enc.pre_encode.conv.2.weight", {3, 3, 1, channels}, kPreConv);
  pe.conv2_b = load("enc.pre_encode.conv.2.bias", {channels}, kF32);
  pe.conv3_w = load("enc.pre_encode.conv.3.weight", {1, 1, channels, channels}, kPreConv);
  pe.conv3_b = load("enc.pre_encode.conv.3.bias", {channels}, kF32);
  pe.conv5_w = load("enc.pre_encode.conv.5.weight", {3, 3, 1, channels}, kPreConv);
  pe.conv5_b = load("enc.pre_encode.conv.5.bias", {channels}, kF32);
  pe.conv6_w = load("enc.pre_encode.conv.6.weight", {1, 1, channels, channels}, kPreConv);
  pe.conv6_b = load("enc.pre_encode.conv.6.bias", {channels}, kF32);
  pe.out_w = load("enc.pre_encode.out.weight", {pre_encode_in, d_enc}, kLinear);
  pe.out_b = load("enc.pre_encode.out.bias", {d_enc}, kF32);

  weights_.blocks.assign(static_cast<size_t>(hp.enc_n_layers), EncBlockWeights{});
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &b = weights_.blocks[static_cast<size_t>(i)];
    auto n = [i](const char *fmt) { return lname(fmt, i); };
    b.norm_ff1_w = load(n("enc.blocks.%d.norm_ff1.weight"), {d_enc}, kF32);
    b.norm_ff1_b = load(n("enc.blocks.%d.norm_ff1.bias"), {d_enc}, kF32);
    b.ff1_lin1_w = load(n("enc.blocks.%d.ff1.linear1.weight"), {d_enc, d_ff_e}, kLinear);
    b.ff1_lin1_b = load(n("enc.blocks.%d.ff1.linear1.bias"), {d_ff_e}, kF32);
    b.ff1_lin2_w = load(n("enc.blocks.%d.ff1.linear2.weight"), {d_ff_e, d_enc}, kLinear);
    b.ff1_lin2_b = load(n("enc.blocks.%d.ff1.linear2.bias"), {d_enc}, kF32);

    b.norm_attn_w = load(n("enc.blocks.%d.norm_attn.weight"), {d_enc}, kF32);
    b.norm_attn_b = load(n("enc.blocks.%d.norm_attn.bias"), {d_enc}, kF32);
    b.attn_q_w = load(n("enc.blocks.%d.attn.linear_q.weight"), {d_enc, d_enc}, kLinear);
    b.attn_q_b = load(n("enc.blocks.%d.attn.linear_q.bias"), {d_enc}, kF32);
    b.attn_k_w = load(n("enc.blocks.%d.attn.linear_k.weight"), {d_enc, d_enc}, kLinear);
    b.attn_k_b = load(n("enc.blocks.%d.attn.linear_k.bias"), {d_enc}, kF32);
    b.attn_v_w = load(n("enc.blocks.%d.attn.linear_v.weight"), {d_enc, d_enc}, kLinear);
    b.attn_v_b = load(n("enc.blocks.%d.attn.linear_v.bias"), {d_enc}, kF32);
    b.attn_out_w = load(n("enc.blocks.%d.attn.linear_out.weight"), {d_enc, d_enc}, kLinear);
    b.attn_out_b = load(n("enc.blocks.%d.attn.linear_out.bias"), {d_enc}, kF32);
    b.attn_pos_w = load(n("enc.blocks.%d.attn.linear_pos.weight"), {d_enc, d_enc}, kLinear);
    b.attn_pos_u = load(n("enc.blocks.%d.attn.pos_bias_u"), {head_dim, n_heads}, kF32);
    b.attn_pos_v = load(n("enc.blocks.%d.attn.pos_bias_v"), {head_dim, n_heads}, kF32);

    b.norm_conv_w = load(n("enc.blocks.%d.norm_conv.weight"), {d_enc}, kF32);
    b.norm_conv_b = load(n("enc.blocks.%d.norm_conv.bias"), {d_enc}, kF32);
    b.conv_pw1_w = load(n("enc.blocks.%d.conv.pointwise1.weight"), {1, d_enc, 2 * d_enc}, kBlockConv);
    b.conv_pw1_b = load(n("enc.blocks.%d.conv.pointwise1.bias"), {2 * d_enc}, kF32);
    b.conv_dw_w = load(n("enc.blocks.%d.conv.depthwise.weight"), {k, 1, d_enc}, kBlockConv);
    b.conv_dw_b = load(n("enc.blocks.%d.conv.depthwise.bias"), {d_enc}, kF32);
    b.conv_pw2_w = load(n("enc.blocks.%d.conv.pointwise2.weight"), {1, d_enc, d_enc}, kBlockConv);
    b.conv_pw2_b = load(n("enc.blocks.%d.conv.pointwise2.bias"), {d_enc}, kF32);

    b.norm_ff2_w = load(n("enc.blocks.%d.norm_ff2.weight"), {d_enc}, kF32);
    b.norm_ff2_b = load(n("enc.blocks.%d.norm_ff2.bias"), {d_enc}, kF32);
    b.ff2_lin1_w = load(n("enc.blocks.%d.ff2.linear1.weight"), {d_enc, d_ff_e}, kLinear);
    b.ff2_lin1_b = load(n("enc.blocks.%d.ff2.linear1.bias"), {d_ff_e}, kF32);
    b.ff2_lin2_w = load(n("enc.blocks.%d.ff2.linear2.weight"), {d_ff_e, d_enc}, kLinear);
    b.ff2_lin2_b = load(n("enc.blocks.%d.ff2.linear2.bias"), {d_enc}, kF32);

    b.norm_out_w = load(n("enc.blocks.%d.norm_out.weight"), {d_enc}, kF32);
    b.norm_out_b = load(n("enc.blocks.%d.norm_out.bias"), {d_enc}, kF32);
  }

  weights_.proj_w = load("enc.proj.weight", {d_enc, out_dim}, kLinear);
  weights_.proj_b = load("enc.proj.bias", {out_dim}, kF32);

  weights_.token_embd = load("dec.token_embd.weight", {hidden, vocab}, kLinear);
  weights_.output_norm = load("dec.output_norm.weight", {hidden}, kF32);

  weights_.dec_blocks.assign(static_cast<size_t>(hp.dec_n_layers), DecBlockWeights{});
  for (int i = 0; i < hp.dec_n_layers; ++i) {
    auto &b = weights_.dec_blocks[static_cast<size_t>(i)];
    auto n = [i](const char *fmt) { return lname(fmt, i); };
    b.norm_attn_w = load(n("dec.blocks.%d.norm_attn.weight"), {hidden}, kF32);
    b.norm_ffn_w = load(n("dec.blocks.%d.norm_ffn.weight"), {hidden}, kF32);
    b.attn_q_w = load(n("dec.blocks.%d.attn.q.weight"), {hidden, hidden}, kLinear);
    b.attn_k_w = load(n("dec.blocks.%d.attn.k.weight"), {hidden, kv_dim}, kLinear);
    b.attn_v_w = load(n("dec.blocks.%d.attn.v.weight"), {hidden, kv_dim}, kLinear);
    b.attn_o_w = load(n("dec.blocks.%d.attn.o.weight"), {hidden, hidden}, kLinear);
    b.attn_q_norm = load(n("dec.blocks.%d.attn.q_norm.weight"), {hp.dec_head_dim}, kF32);
    b.attn_k_norm = load(n("dec.blocks.%d.attn.k_norm.weight"), {hp.dec_head_dim}, kF32);
    b.ffn_down_w = load(n("dec.blocks.%d.ffn.down.weight"), {hp.dec_intermediate, hidden}, kLinear);
    // gate / up are packed by pack_gate_up() after the upload.
  }
}

// arch model.cpp: fuse_batch_norm. Same float arithmetic, host side.
void CanaryQwenRuntime::fuse_batch_norm() {
  const auto &source = *assets_->source;
  const int64_t d = assets_->hparams.enc_d_model;
  auto read_f32 = [&](const std::string &name) {
    if (!source.has_tensor(name)) {
      fail("missing tensor " + name);
    }
    const auto metadata = source.require_metadata(name);
    if (assets::ggml_type_for_tensor_dtype(metadata.dtype) != GGML_TYPE_F32 ||
        metadata.shape != std::vector<int64_t>{d}) {
      fail("tensor " + name + " must be F32 [" + std::to_string(d) + "]");
    }
    return source.require_f32(name, std::optional<std::vector<int64_t>>(std::vector<int64_t>{d}));
  };
  for (int i = 0; i < assets_->hparams.enc_n_layers; ++i) {
    auto &b = weights_.blocks[static_cast<size_t>(i)];
    const std::vector<float> bn_w = read_f32(lname("enc.blocks.%d.conv.bn.weight", i));
    const std::vector<float> bn_b = read_f32(lname("enc.blocks.%d.conv.bn.bias", i));
    const std::vector<float> rm = read_f32(lname("enc.blocks.%d.conv.bn.running_mean", i));
    const std::vector<float> rv = read_f32(lname("enc.blocks.%d.conv.bn.running_var", i));
    std::vector<float> fused_s(static_cast<size_t>(d));
    std::vector<float> fused_b(static_cast<size_t>(d));
    for (int64_t c = 0; c < d; ++c) {
      const float s = bn_w[c] / std::sqrt(rv[c] + kBnEps);
      fused_s[c] = s;
      fused_b[c] = bn_b[c] - rm[c] * s;
    }
    b.conv_bn_fused_scale = store_->make_f32(core::TensorShape::from_dims({d}), std::move(fused_s)).tensor;
    b.conv_bn_fused_bias = store_->make_f32(core::TensorShape::from_dims({d}), std::move(fused_b)).tensor;
  }
}

// arch model.cpp + causal_lm pack_gate_up: one [hidden, 2 * inter] tensor per
// layer holding gate's bytes then up's bytes, in the dtype the arch had after
// its BF16 -> F32 CPU promotion. Filled straight from the source (the arch
// additionally kept the unpacked copies resident; they are never read).
void CanaryQwenRuntime::pack_gate_up() {
  const auto &hp = assets_->hparams;
  const auto &source = *assets_->source;
  const int64_t hidden = hp.dec_hidden;
  const int64_t inter = hp.dec_intermediate;
  const auto shape = shape_from_ne({hidden, inter});
  const core::BackendType btype = core::backend_type(backend_);

  auto target_type = [&](ggml_type native) {
    if (native == GGML_TYPE_BF16) {
      if (cpu_backend_) {
        return GGML_TYPE_F32;
      }
      if (btype == core::BackendType::Vulkan || btype == core::BackendType::Metal) {
        return GGML_TYPE_F16;  // the weight store's policy for BF16 there
      }
    }
    return native;
  };
  auto check = [&](const std::string &name) {
    if (!source.has_tensor(name)) {
      fail("missing tensor " + name);
    }
    const auto metadata = source.require_metadata(name);
    const ggml_type type = assets::ggml_type_for_tensor_dtype(metadata.dtype);
    if (metadata.shape != shape) {
      fail("tensor " + name + " has shape " + shape_string(metadata.shape) + ", expected " +
           shape_string(shape));
    }
    if (!is_linear_type(type)) {
      fail("linear tensor " + name + " has unsupported type " + metadata.dtype);
    }
    return type;
  };

  const size_t n_layers = weights_.dec_blocks.size();
  std::vector<ggml_type> types(n_layers);
  for (size_t i = 0; i < n_layers; ++i) {
    const ggml_type g = target_type(check(lname("dec.blocks.%d.ffn.gate.weight", static_cast<int>(i))));
    const ggml_type u = target_type(check(lname("dec.blocks.%d.ffn.up.weight", static_cast<int>(i))));
    if (g != u) {
      fail("pack_gate_up: layer " + std::to_string(i) + " gate/up type mismatch");
    }
    types[i] = g;
  }

  ggml_init_params params{n_layers * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true};
  packed_ctx_ = ggml_init(params);
  if (packed_ctx_ == nullptr) {
    fail("pack_gate_up: ggml_init failed");
  }
  for (size_t i = 0; i < n_layers; ++i) {
    weights_.dec_blocks[i].ffn_gate_up_w = ggml_new_tensor_2d(packed_ctx_, types[i], hidden, 2 * inter);
  }
  packed_buffer_ = ggml_backend_alloc_ctx_tensors(packed_ctx_, backend_);
  if (packed_buffer_ == nullptr) {
    fail("pack_gate_up: backend buffer allocation failed");
  }
  ggml_backend_buffer_set_usage(packed_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

  auto bytes_for = [&](const std::string &name, ggml_type target) {
    auto raw = source.require_tensor_data(name);
    const ggml_type type = assets::ggml_type_for_tensor_dtype(raw.metadata.dtype);
    if (type == target) {
      return std::move(raw.bytes);
    }
    assets::TensorData data;
    data.shape = core::TensorShape::from_dims({inter, hidden});
    data.type = type;
    data.bytes = std::move(raw.bytes);
    const std::vector<float> values = assets::tensor_data_to_f32(name, data);
    std::vector<std::byte> out;
    if (target == GGML_TYPE_F32) {
      out.resize(values.size() * sizeof(float));
      std::memcpy(out.data(), values.data(), out.size());
    } else if (target == GGML_TYPE_F16) {
      std::vector<ggml_fp16_t> half(values.size());
      ggml_fp32_to_fp16_row(values.data(), half.data(), static_cast<int64_t>(values.size()));
      out.resize(half.size() * sizeof(ggml_fp16_t));
      std::memcpy(out.data(), half.data(), out.size());
    } else {
      fail("pack_gate_up: unsupported conversion for " + name);
    }
    return out;
  };

  for (size_t i = 0; i < n_layers; ++i) {
    ggml_tensor *packed = weights_.dec_blocks[i].ffn_gate_up_w;
    const auto gate = bytes_for(lname("dec.blocks.%d.ffn.gate.weight", static_cast<int>(i)), types[i]);
    const auto up = bytes_for(lname("dec.blocks.%d.ffn.up.weight", static_cast<int>(i)), types[i]);
    if (ggml_nbytes(packed) != gate.size() + up.size()) {
      fail("pack_gate_up: size mismatch at layer " + std::to_string(i));
    }
    ggml_backend_tensor_set(packed, gate.data(), 0, gate.size());
    ggml_backend_tensor_set(packed, up.data(), gate.size(), up.size());
  }
}

CanaryQwenRuntime::CanaryQwenRuntime(std::shared_ptr<const CanaryQwenAssets> model_assets,
                                     core::ExecutionContext &execution_context,
                                     CanaryQwenRuntimeOptions options)
    : assets_(std::move(model_assets)), execution_context_(execution_context), options_(options) {
  if (!assets_ || !assets_->source) {
    fail("runtime requires loaded assets");
  }
  backend_ = execution_context_.backend();
  if (backend_ == nullptr) {
    fail("execution backend is not initialized");
  }
  const char *name = ggml_backend_name(backend_);
  backend_name_ = name != nullptr ? name : "";
  cpu_backend_ = core::backend_type(backend_) == core::BackendType::Cpu;

  store_ = std::make_shared<core::BackendWeightStore>(
      backend_, execution_context_.backend_type(), "canary_qwen.weights", 512ull * 1024ull * 1024ull);
  try {
    load_weights();
    fuse_batch_norm();
    store_->upload();
    pack_gate_up();
  } catch (...) {
    // The destructor does not run for a throwing constructor.
    if (packed_buffer_ != nullptr) {
      ggml_backend_buffer_free(packed_buffer_);
      packed_buffer_ = nullptr;
    }
    if (packed_ctx_ != nullptr) {
      ggml_free(packed_ctx_);
      packed_ctx_ = nullptr;
    }
    throw;
  }
  assets_->source->release_storage();

  const auto &hp = assets_->hparams;
  NemoMelConfig cfg;
  cfg.sample_rate = hp.fe_sample_rate;
  cfg.num_mels = hp.fe_num_mels;
  cfg.n_fft = hp.fe_n_fft;
  cfg.win_length = hp.fe_win_length;
  cfg.hop_length = hp.fe_hop_length;
  cfg.pre_emphasis = hp.fe_pre_emphasis;
  cfg.f_min = hp.fe_f_min;
  cfg.f_max = hp.fe_f_max;
  cfg.pad_mode = hp.fe_pad_mode;
  // NeMo passes periodic=False (symmetric Hann); the GGUF's window wins when
  // present, exactly as in the arch.
  cfg.window_type = "hann_symmetric";
  cfg.normalize = hp.fe_normalize;
  cfg.filterbank = assets_->mel_filterbank;
  cfg.window = assets_->window;
  mel_.emplace(cfg);
}

CanaryQwenRuntime::~CanaryQwenRuntime() {
  kv_cache_.free();
  kv_cache_batch_.free();
  if (packed_buffer_ != nullptr) {
    ggml_backend_buffer_free(packed_buffer_);
    packed_buffer_ = nullptr;
  }
  if (packed_ctx_ != nullptr) {
    ggml_free(packed_ctx_);
    packed_ctx_ = nullptr;
  }
}

int32_t CanaryQwenRuntime::sample_rate() const noexcept {
  return assets_ ? assets_->hparams.fe_sample_rate : 16000;
}

int CanaryQwenRuntime::n_threads() const { return std::max(1, execution_context_.config().threads); }

void CanaryQwenRuntime::compute(ggml_cgraph *graph, const char *what) {
  core::set_backend_threads(backend_, n_threads());
  if (core::compute_backend_graph(backend_, graph, nullptr, what) != GGML_STATUS_SUCCESS) {
    fail(std::string(what) + " compute failed");
  }
  ggml_backend_synchronize(backend_);
}

bool CanaryQwenRuntime::compute_mel(const std::vector<float> &pcm, std::vector<float> &mel,
                                    int &n_frames, int threads) const {
  int n_mels = 0;
  n_frames = 0;
  if (pcm.empty()) {
    return false;
  }
  if (!mel_->compute(pcm.data(), pcm.size(), mel, n_mels, n_frames, threads)) {
    return false;
  }
  return n_mels == assets_->hparams.fe_num_mels && n_frames > 0;
}

std::vector<int32_t> CanaryQwenRuntime::build_prompt(int T_enc) const {
  std::vector<int32_t> ids;
  ids.reserve(assets_->prompt_prefix_ids.size() + static_cast<size_t>(T_enc) +
              assets_->prompt_suffix_ids.size());
  ids.insert(ids.end(), assets_->prompt_prefix_ids.begin(), assets_->prompt_prefix_ids.end());
  for (int i = 0; i < T_enc; ++i) {
    ids.push_back(assets_->hparams.audio_locator_id);
  }
  ids.insert(ids.end(), assets_->prompt_suffix_ids.begin(), assets_->prompt_suffix_ids.end());
  return ids;
}

void CanaryQwenRuntime::encode(const std::vector<float> &mel, int n_frames,
                               std::vector<float> &enc_host, int &T_enc) {
  const auto &hp = assets_->hparams;
  GraphRun run;
  run.init(32ull * 1024ull * 1024ull, "encoder");
  EncoderOptions eo;
  eo.use_flash = options_.encoder_use_flash;
  eo.backend_name = backend_name_.c_str();
  EncoderBuild eb = build_encoder_graph(run.ctx, weights_, hp, n_frames, eo);
  if (eb.graph == nullptr || eb.out == nullptr || eb.mel_in == nullptr) {
    fail("encoder graph build failed");
  }
  run.allocate(backend_, eb.graph, "encoder");

  // mel is mel-major ([n_mels, n_frames] row-major) == ne [n_frames, n_mels].
  ggml_backend_tensor_set(eb.mel_in, mel.data(), 0, mel.size() * sizeof(float));
  if (eb.pos_emb_in != nullptr) {
    const int pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);
    std::vector<float> pos_buf;
    build_relpos_emb_host(pos_buf, hp.enc_d_model, (pos_len + 1) / 2);
    ggml_backend_tensor_set(eb.pos_emb_in, pos_buf.data(), 0, pos_buf.size() * sizeof(float));
  }
  compute(eb.graph, "encoder");

  const int hidden = static_cast<int>(eb.out->ne[0]);
  T_enc = static_cast<int>(eb.out->ne[1]);
  enc_host.resize(static_cast<size_t>(hidden) * static_cast<size_t>(T_enc));
  ggml_backend_tensor_get(eb.out, enc_host.data(), 0, enc_host.size() * sizeof(float));
}

// ---------------------------------------------------------------------------
// transcribe (arch run())
// ---------------------------------------------------------------------------

CanaryQwenTranscription CanaryQwenRuntime::transcribe(const std::vector<float> &pcm,
                                                      const runtime::RunControl &control,
                                                      CanaryQwenTranscription *partial) {
  const auto &hp = assets_->hparams;
  control.emit_progress("canary_qwen.mel", 0, 1);

  std::vector<float> mel;
  int n_frames = 0;
  if (!compute_mel(pcm, mel, n_frames, n_threads())) {
    throw std::invalid_argument(std::string(kTag) + ": audio too short for the mel frontend (" +
                                std::to_string(pcm.size()) + " samples; need at least " +
                                std::to_string(hp.fe_n_fft / 2 + 1) + ")");
  }

  control.emit_progress("canary_qwen.encode", 0, 1);
  std::vector<float> enc_host;
  int T_enc = 0;
  encode(mel, n_frames, enc_host, T_enc);

  const std::vector<int32_t> prompt_ids = build_prompt(T_enc);
  const int prefix_len = static_cast<int>(assets_->prompt_prefix_ids.size());
  const int T_prompt = static_cast<int>(prompt_ids.size());
  const int suffix_len = T_prompt - prefix_len - T_enc;

  // Input-length gate (the arch's INPUT_TOO_LONG).
  const int ceiling = canary_qwen_context_ceiling(options_.n_ctx, hp);
  if (T_prompt + kGenReserve > ceiling) {
    throw runtime::InputTooLong(
        too_long_message(T_enc, prefix_len + suffix_len, ceiling, T_prompt + kGenReserve));
  }

  const int max_new = pick_decode_budget(
      predict_transcript_tokens(T_enc, assets_->ms_per_audio_token), kGenReserve, T_prompt, ceiling);

  // KV cache sized to prompt + budget, pow2 from 1024, clamped to the ceiling.
  int want_n_ctx = next_pow2_window(T_prompt + max_new);
  if (want_n_ctx > ceiling) {
    want_n_ctx = ceiling;
  }
  if (!causal_lm::kv_init(kv_cache_, backend_, want_n_ctx, hp.dec_n_kv_heads, hp.dec_head_dim,
                          hp.dec_n_layers, options_.kv_type)) {
    fail("KV cache allocation failed (n_ctx=" + std::to_string(want_n_ctx) +
         ") - out of memory; lower the session n_ctx or shorten the audio");
  }

  std::vector<int32_t> generated_ids;
  const int32_t eos_id = hp.eos_token_id;
  int32_t next_tok = 0;

  auto finish = [&](bool truncated) {
    std::vector<int32_t> ids = generated_ids;
    if (!ids.empty() && ids.back() == eos_id) {
      ids.pop_back();
    }
    CanaryQwenTranscription out;
    out.text = assets_->decode(ids);
    out.truncated = truncated;
    out.n_generated_tokens = static_cast<int>(generated_ids.size());
    return out;
  };

  try {
    // ---- prefill ----
    {
      GraphRun run;
      run.init(32ull * 1024ull * 1024ull, "prefill");
      PrefillBuild pb = build_prefill_graph(run.ctx, weights_, hp, kv_cache_, T_prompt, T_enc,
                                            prefix_len, suffix_len, options_.decoder_use_flash,
                                            /*slice_last=*/true);
      if (pb.graph == nullptr || pb.out == nullptr) {
        fail("prefill graph build failed");
      }
      run.allocate(backend_, pb.graph, "prefill");

      ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data(), 0,
                              prompt_ids.size() * sizeof(int32_t));
      if (T_enc > 0 && pb.audio_in != nullptr) {
        ggml_backend_tensor_set(pb.audio_in, enc_host.data(), 0, enc_host.size() * sizeof(float));
      }
      std::vector<int32_t> positions(static_cast<size_t>(T_prompt));
      for (int i = 0; i < T_prompt; ++i) {
        positions[static_cast<size_t>(i)] = i;
      }
      ggml_backend_tensor_set(pb.positions_in, positions.data(), 0,
                              positions.size() * sizeof(int32_t));
      const auto mask = causal_mask_f16(T_prompt);
      ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

      compute(pb.graph, "prefill");
      kv_cache_.n = T_prompt;
      kv_cache_.head = T_prompt;

      std::vector<float> logits(static_cast<size_t>(hp.dec_vocab_size));
      ggml_backend_tensor_get(pb.out, logits.data(), 0, logits.size() * sizeof(float));
      next_tok = host_argmax(logits);
      generated_ids.push_back(next_tok);
    }

    // ---- greedy step loop ----
    int cur_past = T_prompt;
    int max_n_kv = next_pow2_window(T_prompt + max_new);
    if (max_n_kv > kv_cache_.n_ctx) {
      max_n_kv = kv_cache_.n_ctx;
    }

    GraphRun run;
    run.init(16ull * 1024ull * 1024ull, "step");
    StepBuild sb = build_step_graph(run.ctx, weights_, hp, kv_cache_, max_n_kv,
                                    options_.decoder_use_flash);
    if (sb.graph == nullptr || sb.out == nullptr) {
      fail("step graph build failed");
    }
    run.allocate(backend_, sb.graph, "step");

    const ggml_fp16_t mask_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> step_mask(static_cast<size_t>(max_n_kv), mask_neg_inf);

    while (next_tok != eos_id && static_cast<int32_t>(generated_ids.size()) < max_new &&
           cur_past + 1 <= max_n_kv) {
      control.emit_progress("canary_qwen.decode", static_cast<int64_t>(generated_ids.size()), max_new);

      // Every step input is re-uploaded before each compute of the cached graph.
      ggml_backend_tensor_set(sb.input_id_in, &next_tok, 0, sizeof(int32_t));
      const int32_t pos_val = cur_past;
      ggml_backend_tensor_set(sb.position_in, &pos_val, 0, sizeof(int32_t));
      const int64_t kv_idx_val = cur_past;
      ggml_backend_tensor_set(sb.kv_idx_in, &kv_idx_val, 0, sizeof(int64_t));
      if (cur_past == T_prompt) {
        std::fill(step_mask.begin(), step_mask.begin() + cur_past + 1, mask_zero);
      } else {
        step_mask[static_cast<size_t>(cur_past)] = mask_zero;
      }
      ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0,
                              static_cast<size_t>(max_n_kv) * sizeof(ggml_fp16_t));

      compute(sb.graph, "step");

      int32_t argmax_tok = 0;
      ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
      next_tok = argmax_tok;
      generated_ids.push_back(next_tok);
      cur_past += 1;
    }
  } catch (const runtime::ProgressCanceled &) {
    if (partial != nullptr) {
      *partial = finish(/*truncated=*/false);
    }
    kv_cache_.free();
    throw;
  } catch (...) {
    kv_cache_.free();
    throw;
  }

  // Stopped at EOS (complete) or the budget / KV width (truncated).
  const bool truncated = next_tok != eos_id;
  kv_cache_.free();
  return finish(truncated);
}

// ---------------------------------------------------------------------------
// transcribe_batch (arch run_batch())
// ---------------------------------------------------------------------------

std::vector<CanaryQwenBatchItem>
CanaryQwenRuntime::transcribe_batch(const std::vector<const std::vector<float> *> &pcms,
                                    const runtime::RunControl &control) {
  const int n = static_cast<int>(pcms.size());
  std::vector<CanaryQwenBatchItem> items(static_cast<size_t>(n));
  if (n == 0) {
    return items;
  }

  // Serial fallback (the arch's run_batch_serial): non-flash decoder or a
  // single utterance.
  if (!options_.decoder_use_flash || n == 1) {
    for (int i = 0; i < n; ++i) {
      control.emit_progress("canary_qwen.batch", i, n);
      auto &item = items[static_cast<size_t>(i)];
      try {
        static const std::vector<float> kEmpty;
        item.result = transcribe(pcms[static_cast<size_t>(i)] != nullptr ? *pcms[static_cast<size_t>(i)] : kEmpty,
                                 control);
        item.ok = true;
      } catch (const runtime::ProgressCanceled &) {
        throw;
      } catch (const runtime::InputTooLong &e) {
        item.error = e.what();
        item.input_too_long = true;
      } catch (const std::invalid_argument &e) {
        item.error = e.what();
      } catch (const std::exception &e) {
        item.error = e.what();
      }
    }
    return items;
  }

  const auto &hp = assets_->hparams;
  const int ceiling = canary_qwen_context_ceiling(options_.n_ctx, hp);

  std::vector<char> valid(static_cast<size_t>(n), 0);
  std::vector<std::vector<float>> enc_hosts(static_cast<size_t>(n));
  std::vector<int> T_enc(static_cast<size_t>(n), 0);
  std::vector<std::vector<int32_t>> prompt_ids(static_cast<size_t>(n));
  std::vector<int> T_prompt(static_cast<size_t>(n), 0);
  const int prefix_len = static_cast<int>(assets_->prompt_prefix_ids.size());

  // Pass 0: mel per utterance, in parallel (frames are independent; the
  // thread split never changes a value).
  std::vector<std::vector<float>> mel_bufs(static_cast<size_t>(n));
  std::vector<int> mel_nf(static_cast<size_t>(n), 0);
  {
    const int workers = std::max(1, std::min(n, n_threads()));
    std::vector<std::thread> pool;
    auto worker = [&](int tid) {
      for (int b = tid; b < n; b += workers) {
        const auto *pcm = pcms[static_cast<size_t>(b)];
        int nf = 0;
        if (pcm != nullptr && compute_mel(*pcm, mel_bufs[static_cast<size_t>(b)], nf, 1)) {
          mel_nf[static_cast<size_t>(b)] = nf;
        }
      }
    };
    for (int t = 1; t < workers; ++t) {
      pool.emplace_back(worker, t);
    }
    worker(0);
    for (auto &th : pool) {
      th.join();
    }
  }

  // Pass 1: per-utterance encoder (serial) + prompt + input gate.
  for (int b = 0; b < n; ++b) {
    control.emit_progress("canary_qwen.encode", b, n);
    auto &item = items[static_cast<size_t>(b)];
    if (mel_nf[static_cast<size_t>(b)] <= 0) {
      item.error = std::string(kTag) + ": utterance " + std::to_string(b) +
                   " is empty or too short for the mel frontend";
      continue;
    }
    try {
      encode(mel_bufs[static_cast<size_t>(b)], mel_nf[static_cast<size_t>(b)],
             enc_hosts[static_cast<size_t>(b)], T_enc[static_cast<size_t>(b)]);
    } catch (const std::exception &e) {
      item.error = e.what();
      continue;
    }
    prompt_ids[static_cast<size_t>(b)] = build_prompt(T_enc[static_cast<size_t>(b)]);
    T_prompt[static_cast<size_t>(b)] = static_cast<int>(prompt_ids[static_cast<size_t>(b)].size());
    if (T_prompt[static_cast<size_t>(b)] + kGenReserve > ceiling) {
      item.error = too_long_message(T_enc[static_cast<size_t>(b)],
                                    T_prompt[static_cast<size_t>(b)] - T_enc[static_cast<size_t>(b)],
                                    ceiling, T_prompt[static_cast<size_t>(b)] + kGenReserve);
      item.input_too_long = true;
      continue;
    }
    valid[static_cast<size_t>(b)] = 1;
  }

  int max_T_prompt = 0;
  int T_enc_max = 0;
  for (int b = 0; b < n; ++b) {
    if (valid[static_cast<size_t>(b)]) {
      max_T_prompt = std::max(max_T_prompt, T_prompt[static_cast<size_t>(b)]);
      T_enc_max = std::max(T_enc_max, T_enc[static_cast<size_t>(b)]);
    }
  }
  if (max_T_prompt == 0) {
    return items;
  }
  T_enc_max = std::max(1, T_enc_max);
  const int max_new = pick_decode_budget(
      predict_transcript_tokens(T_enc_max, assets_->ms_per_audio_token), kGenReserve,
      max_T_prompt, ceiling);
  int max_n_kv = next_pow2_window(max_T_prompt + max_new);
  if (max_n_kv > ceiling) {
    max_n_kv = ceiling;
  }

  if (!causal_lm::kv_init_batched(kv_cache_batch_, backend_, max_n_kv, hp.dec_n_kv_heads,
                                  hp.dec_head_dim, hp.dec_n_layers, n, options_.kv_type)) {
    fail("batched KV cache allocation failed (n_ctx=" + std::to_string(max_n_kv) + " x " +
         std::to_string(n) + " utterances) - out of memory; lower the session n_ctx or the "
                             "batch size");
  }

  std::vector<int32_t> next_tok(static_cast<size_t>(n), 0);
  std::vector<int> n_past(static_cast<size_t>(n), 0);
  std::vector<std::vector<int32_t>> generated(static_cast<size_t>(n));
  const int32_t eos_id = hp.eos_token_id;
  std::vector<char> truncated(static_cast<size_t>(n), 0);

  try {
    // Pass 2: batched prefill.
    {
      GraphRun run;
      run.init(32ull * 1024ull * 1024ull, "batched prefill");
      PrefillBuildBatched pb = build_prefill_graph_batched(run.ctx, weights_, hp, kv_cache_batch_,
                                                           max_T_prompt, T_enc_max, n,
                                                           options_.decoder_use_flash);
      if (pb.graph == nullptr || pb.out == nullptr) {
        fail("batched prefill graph build failed");
      }
      run.allocate(backend_, pb.graph, "batched prefill");

      const int hidden = hp.dec_hidden;
      std::vector<int32_t> ids(static_cast<size_t>(max_T_prompt) * n, 0);
      std::vector<float> audio_dense(static_cast<size_t>(hidden) * max_T_prompt * n, 0.0f);
      std::vector<float> keep(static_cast<size_t>(max_T_prompt) * n, 1.0f);
      std::vector<int64_t> kidx(static_cast<size_t>(max_T_prompt) * n);
      std::vector<int32_t> lidx(static_cast<size_t>(n), 0);
      for (int b = 0; b < n; ++b) {
        const bool ok = valid[static_cast<size_t>(b)] != 0;
        const int ta = ok ? T_enc[static_cast<size_t>(b)] : 0;
        const int tp = ok ? T_prompt[static_cast<size_t>(b)] : 0;
        if (ok) {
          std::memcpy(ids.data() + static_cast<size_t>(b) * max_T_prompt,
                      prompt_ids[static_cast<size_t>(b)].data(), static_cast<size_t>(tp) * sizeof(int32_t));
          for (int j = 0; j < ta; ++j) {
            const size_t dst_col = static_cast<size_t>(b) * max_T_prompt + (prefix_len + j);
            std::memcpy(audio_dense.data() + dst_col * hidden,
                        enc_hosts[static_cast<size_t>(b)].data() + static_cast<size_t>(j) * hidden,
                        static_cast<size_t>(hidden) * sizeof(float));
            keep[dst_col] = 0.0f;
          }
        }
        for (int t = 0; t < max_T_prompt; ++t) {
          kidx[static_cast<size_t>(b) * max_T_prompt + t] = t;
        }
        lidx[static_cast<size_t>(b)] = ok ? (tp - 1) : 0;
      }
      ggml_backend_tensor_set(pb.input_ids_in, ids.data(), 0, ids.size() * sizeof(int32_t));
      ggml_backend_tensor_set(pb.audio_dense_in, audio_dense.data(), 0, audio_dense.size() * sizeof(float));
      ggml_backend_tensor_set(pb.keep_mask_in, keep.data(), 0, keep.size() * sizeof(float));
      {
        std::vector<int32_t> pos(static_cast<size_t>(max_T_prompt));
        for (int t = 0; t < max_T_prompt; ++t) {
          pos[static_cast<size_t>(t)] = t;
        }
        ggml_backend_tensor_set(pb.positions_in, pos.data(), 0, pos.size() * sizeof(int32_t));
      }
      const auto mask = causal_mask_f16(max_T_prompt);
      ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
      ggml_backend_tensor_set(pb.kv_idx_in, kidx.data(), 0, kidx.size() * sizeof(int64_t));
      ggml_backend_tensor_set(pb.last_idx_in, lidx.data(), 0, lidx.size() * sizeof(int32_t));

      compute(pb.graph, "batched prefill");
      std::vector<int32_t> amax(static_cast<size_t>(n), 0);
      ggml_backend_tensor_get(pb.out, amax.data(), 0, amax.size() * sizeof(int32_t));
      for (int b = 0; b < n; ++b) {
        if (!valid[static_cast<size_t>(b)]) {
          continue;
        }
        n_past[static_cast<size_t>(b)] = T_prompt[static_cast<size_t>(b)];
        next_tok[static_cast<size_t>(b)] = amax[static_cast<size_t>(b)];
        generated[static_cast<size_t>(b)].push_back(amax[static_cast<size_t>(b)]);
      }
    }

    // Pass 3: batched step loop (causal_lm run_batched_step_loop).
    GraphRun run;
    run.init(16ull * 1024ull * 1024ull, "batched step");
    StepBuildBatched sb = build_step_graph_batched(run.ctx, weights_, hp, kv_cache_batch_, max_n_kv,
                                                   n, options_.decoder_use_flash);
    if (sb.graph == nullptr || sb.out == nullptr) {
      fail("batched step graph build failed");
    }
    run.allocate(backend_, sb.graph, "batched step");

    std::vector<char> finished(static_cast<size_t>(n), 1);
    for (int b = 0; b < n; ++b) {
      if (valid[static_cast<size_t>(b)]) {
        finished[static_cast<size_t>(b)] = (next_tok[static_cast<size_t>(b)] == eos_id);
      }
    }
    std::vector<int32_t> ids_buf(static_cast<size_t>(n), 0), pos_buf(static_cast<size_t>(n), 0),
        out_buf(static_cast<size_t>(n), 0);
    std::vector<int64_t> kvidx_buf(static_cast<size_t>(n), 0);
    const ggml_fp16_t mz = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t mn = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask_buf(static_cast<size_t>(max_n_kv) * n, mn);
    for (int b = 0; b < n; ++b) {
      if (!valid[static_cast<size_t>(b)]) {
        continue;
      }
      const size_t base = static_cast<size_t>(b) * max_n_kv;
      for (int c = 0; c <= n_past[static_cast<size_t>(b)] && c < max_n_kv; ++c) {
        mask_buf[base + c] = mz;
      }
    }

    int n_steps = 0;
    bool all_done = false;
    while (!all_done) {
      control.emit_progress("canary_qwen.decode", n_steps, max_new);
      for (int b = 0; b < n; ++b) {
        ids_buf[static_cast<size_t>(b)] = next_tok[static_cast<size_t>(b)];
        pos_buf[static_cast<size_t>(b)] = n_past[static_cast<size_t>(b)];
        kvidx_buf[static_cast<size_t>(b)] = n_past[static_cast<size_t>(b)];
      }
      ggml_backend_tensor_set(sb.input_ids_in, ids_buf.data(), 0, ids_buf.size() * sizeof(int32_t));
      ggml_backend_tensor_set(sb.position_in, pos_buf.data(), 0, pos_buf.size() * sizeof(int32_t));
      ggml_backend_tensor_set(sb.kv_idx_in, kvidx_buf.data(), 0, kvidx_buf.size() * sizeof(int64_t));
      ggml_backend_tensor_set(sb.mask_in, mask_buf.data(), 0, mask_buf.size() * sizeof(ggml_fp16_t));

      compute(sb.graph, "batched step");
      ggml_backend_tensor_get(sb.out, out_buf.data(), 0, out_buf.size() * sizeof(int32_t));
      ++n_steps;

      all_done = true;
      for (int b = 0; b < n; ++b) {
        if (finished[static_cast<size_t>(b)] || !valid[static_cast<size_t>(b)]) {
          continue;
        }
        const int32_t tok = out_buf[static_cast<size_t>(b)];
        next_tok[static_cast<size_t>(b)] = tok;
        generated[static_cast<size_t>(b)].push_back(tok);
        n_past[static_cast<size_t>(b)] += 1;
        const size_t base = static_cast<size_t>(b) * max_n_kv;
        if (n_past[static_cast<size_t>(b)] < max_n_kv) {
          mask_buf[base + n_past[static_cast<size_t>(b)]] = mz;
        }
        if (tok == eos_id || static_cast<int>(generated[static_cast<size_t>(b)].size()) >= max_new ||
            n_past[static_cast<size_t>(b)] + 1 > max_n_kv) {
          finished[static_cast<size_t>(b)] = 1;
        } else {
          all_done = false;
        }
      }
    }
    for (int b = 0; b < n; ++b) {
      truncated[static_cast<size_t>(b)] =
          (valid[static_cast<size_t>(b)] && next_tok[static_cast<size_t>(b)] != eos_id) ? 1 : 0;
    }
  } catch (...) {
    kv_cache_batch_.free();
    throw;
  }
  kv_cache_batch_.free();

  for (int b = 0; b < n; ++b) {
    if (!valid[static_cast<size_t>(b)]) {
      continue;
    }
    std::vector<int32_t> gen = generated[static_cast<size_t>(b)];
    if (!gen.empty() && gen.back() == eos_id) {
      gen.pop_back();
    }
    auto &item = items[static_cast<size_t>(b)];
    item.ok = true;
    item.result.text = assets_->decode(gen);
    item.result.truncated = truncated[static_cast<size_t>(b)] != 0;
    item.result.n_generated_tokens = static_cast<int>(generated[static_cast<size_t>(b)].size());
  }
  return items;
}

} // namespace engine::models::canary_qwen
