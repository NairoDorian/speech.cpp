#include "engine/framework/modules/speech_encoders/sanm.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace engine::modules {
namespace {

void validate_config(const SanmBlockConfig &config) {
  if (config.input_size <= 0 || config.model_size <= 0 ||
      config.num_heads <= 0 || config.ffn_size <= 0 ||
      config.fsmn_kernel_size <= 0) {
    throw std::runtime_error("SAN-M block dimensions must be positive");
  }
  if (config.model_size % config.num_heads != 0) {
    throw std::runtime_error(
        "SAN-M model size must be divisible by the head count");
  }
  if (config.fsmn_kernel_size % 2 == 0) {
    throw std::runtime_error("SAN-M FSMN kernel size must be odd");
  }
  if (!std::isfinite(config.layer_norm_eps) ||
      !(config.layer_norm_eps > 0.0F)) {
    throw std::runtime_error(
        "SAN-M layer norm epsilon must be finite and positive");
  }
}

core::TensorValue reshape_heads(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                int64_t num_heads, int64_t head_dim) {
  const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
  return core::reshape_tensor(
      ctx, contiguous,
      core::TensorShape::from_dims(
          {input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
}

struct SanmAttentionBranch {
  core::TensorValue output;
};

SanmAttentionBranch build_attention_branch(core::ModuleBuildContext &ctx,
                                           const core::TensorValue &normalized,
                                           const SanmBlockWeightsView &weights,
                                           const SanmBlockConfig &config) {
  const int64_t head_dim = config.model_size / config.num_heads;
  const LinearModule output_projection(
      {config.model_size, config.model_size, true, GGML_PREC_F32});

  core::TensorValue query;
  core::TensorValue key;
  core::TensorValue value;

  if (weights.fused_qkv_projection.has_value()) {
    const LinearModule qkv_projection(
        {config.input_size, 3 * config.model_size, true, GGML_PREC_F32});
    auto qkv = qkv_projection.build(ctx, normalized, *weights.fused_qkv_projection);
    query = SliceModule({2, 0, config.model_size}).build(ctx, qkv);
    key = SliceModule({2, config.model_size, config.model_size}).build(ctx, qkv);
    value = SliceModule({2, 2 * config.model_size, config.model_size}).build(ctx, qkv);
  } else {
    const LinearModule q_projection(
        {config.input_size, config.model_size, true, GGML_PREC_F32});
    const LinearModule k_projection(
        {config.input_size, config.model_size, true, GGML_PREC_F32});
    const LinearModule v_projection(
        {config.input_size, config.model_size, true, GGML_PREC_F32});

    query = q_projection.build(ctx, normalized, weights.query_projection);
    key = k_projection.build(ctx, normalized, weights.key_projection);
    value = v_projection.build(ctx, normalized, weights.value_projection);
  }

  query = reshape_heads(ctx, query, config.num_heads, head_dim);
  key = reshape_heads(ctx, key, config.num_heads, head_dim);
  auto value_heads = reshape_heads(ctx, value, config.num_heads, head_dim);
  query = TransposeModule({{0, 2, 1, 3}, query.shape.rank}).build(ctx, query);
  key = TransposeModule({{0, 2, 1, 3}, key.shape.rank}).build(ctx, key);
  value_heads = TransposeModule({{0, 2, 1, 3}, value_heads.shape.rank})
                    .build(ctx, value_heads);

  core::TensorValue attention;
  if (config.attn_pad_mask.has_value()) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const MatMulModule matmul;
    auto key_transposed = TransposeModule({{0, 1, 3, 2}, key.shape.rank}).build(ctx, key);
    auto kq = matmul.build(ctx, query, key_transposed);
    kq = core::wrap_tensor(
        ggml_add(ctx.ggml, kq.tensor, config.attn_pad_mask->tensor),
        kq.shape, GGML_TYPE_F32);
    auto kq_soft = core::wrap_tensor(
        ggml_soft_max_ext(ctx.ggml, core::ensure_backend_addressable_layout(ctx, kq).tensor,
                          nullptr, scale, 0.0f),
        kq.shape, GGML_TYPE_F32);
    auto context = matmul.build(ctx, kq_soft, value_heads);
    context = TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    attention = core::ensure_backend_addressable_layout(ctx, context);
  } else {
    attention =
        ScaledDotProductAttentionModule({
                                            head_dim,
                                            config.attention_lowering,
                                            GGML_PREC_F32,
                                            AttentionCausality::NonCausal,
                                        })
            .build(ctx, query, key, value_heads);
    attention = core::ensure_backend_addressable_layout(ctx, attention);
  }

  attention =
      core::reshape_tensor(ctx, attention,
                           core::TensorShape::from_dims(
                               {normalized.shape.dims[0],
                                normalized.shape.dims[1], config.model_size}));
  attention = output_projection.build(ctx, attention,
                                      weights.attention_output_projection);

  auto fsmn = sanm_fsmn_branch(ctx, value, weights.fsmn_weight, config.fsmn_kernel_size, config.conv_pad_mask);
  return {AddModule{}.build(ctx, attention, fsmn)};
}

core::TensorValue build_ffn_residual(core::ModuleBuildContext &ctx,
                                     const core::TensorValue &residual,
                                     const SanmBlockWeightsView &weights,
                                     const SanmBlockConfig &config) {
  auto hidden =
      sanm_layer_norm(ctx, residual, weights.final_norm, config.layer_norm_eps);
  hidden =
      LinearModule({config.model_size, config.ffn_size, true, GGML_PREC_F32})
          .build(ctx, hidden, weights.ffn_input_projection);
  hidden = ReluModule{}.build(ctx, hidden);
  hidden =
      LinearModule({config.ffn_size, config.model_size, true, GGML_PREC_F32})
          .build(ctx, hidden, weights.ffn_output_projection);
  return AddModule{}.build(ctx, residual, hidden);
}

core::TensorValue build_block(core::ModuleBuildContext &ctx,
                              const core::TensorValue &input,
                              const SanmBlockWeightsView &weights,
                              const SanmBlockConfig &config,
                              bool add_input_residual) {
  validate_config(config);
  core::validate_rank_between(input, 3, 3, "SAN-M input");
  core::validate_last_dim(input, config.input_size, "SAN-M input");
  if (add_input_residual && config.input_size != config.model_size) {
    throw std::runtime_error(
        "SAN-M residual block requires input_size == model_size");
  }

  const auto normalized = sanm_layer_norm(
      ctx, input, weights.self_attention_norm, config.layer_norm_eps);
  auto residual =
      build_attention_branch(ctx, normalized, weights, config).output;
  if (add_input_residual) {
    residual = AddModule{}.build(ctx, input, residual);
  }
  return build_ffn_residual(ctx, residual, weights, config);
}

} // namespace

void build_sinusoidal_pe(std::vector<float> & out, int64_t depth, int64_t frames) {
  out.assign(static_cast<size_t>(frames * depth), 0.0f);
  if (depth <= 1 || frames <= 0) {
    return;
  }
  const int64_t half = depth / 2;
  if (half <= 1) {
    return;
  }
  const double log_increment = std::log(10000.0) / static_cast<double>(half - 1);
  std::vector<double> inv_ts(static_cast<size_t>(half));
  for (int64_t k = 0; k < half; ++k) {
    inv_ts[static_cast<size_t>(k)] = std::exp(static_cast<double>(k) * (-log_increment));
  }
  for (int64_t i = 0; i < frames; ++i) {
    const double pos = static_cast<double>(i + 1); // 1-based
    float *row = out.data() + static_cast<size_t>(i * depth);
    for (int64_t k = 0; k < half; ++k) {
      const double s = pos * inv_ts[static_cast<size_t>(k)];
      row[k] = static_cast<float>(std::sin(s));
      row[half + k] = static_cast<float>(std::cos(s));
    }
  }
}

std::vector<float> make_sinusoidal_positions(int64_t frames, int64_t channels) {
  std::vector<float> values;
  build_sinusoidal_pe(values, channels, frames);
  return values;
}

core::TensorValue sanm_layer_norm(core::ModuleBuildContext &ctx,
                                  const core::TensorValue &input,
                                  const NormWeights &weights, float epsilon) {
  core::validate_rank_between(input, 2, 4, "SAN-M layer norm input");
  if (!std::isfinite(epsilon) || !(epsilon > 0.0F)) {
    throw std::runtime_error(
        "SAN-M layer norm epsilon must be finite and positive");
  }
  return LayerNormModule({input.shape.last_dim(), epsilon, true, true})
      .build(ctx, input, weights);
}

core::TensorValue sanm_fsmn_branch(core::ModuleBuildContext &ctx,
                                   const core::TensorValue &value,
                                   const core::TensorValue &fsmn_weight,
                                   int64_t kernel_size,
                                   const std::optional<core::TensorValue> &conv_pad_mask) {
  auto v_in = value;
  if (conv_pad_mask.has_value()) {
    v_in = core::wrap_tensor(
        ggml_mul(ctx.ggml, v_in.tensor, conv_pad_mask->tensor),
        v_in.shape, v_in.type);
  }
  auto value_bct = TransposeModule({{0, 2, 1, 3}, v_in.shape.rank}).build(ctx, v_in);
  value_bct = core::ensure_backend_addressable_layout(ctx, value_bct);
  auto fsmn = DepthwiseConv1dModule(
                  {
                      v_in.shape.dims[2],
                      kernel_size,
                      1,
                      static_cast<int>((kernel_size - 1) / 2),
                      1,
                      false,
                  })
                  .build(ctx, value_bct, {fsmn_weight, std::nullopt});
  fsmn = TransposeModule({{0, 2, 1, 3}, fsmn.shape.rank}).build(ctx, fsmn);
  return AddModule{}.build(ctx, fsmn, value);
}

core::TensorValue sanm_projection_block(core::ModuleBuildContext &ctx,
                                        const core::TensorValue &input,
                                        const SanmBlockWeightsView &weights,
                                        const SanmBlockConfig &config) {
  return build_block(ctx, input, weights, config, false);
}

core::TensorValue sanm_residual_block(core::ModuleBuildContext &ctx,
                                      const core::TensorValue &input,
                                      const SanmBlockWeightsView &weights,
                                      const SanmBlockConfig &config) {
  return build_block(ctx, input, weights, config, true);
}

} // namespace engine::modules

