#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace engine::modules {

struct SanmBlockWeightsView {
  NormWeights self_attention_norm;
  LinearWeights query_projection;
  LinearWeights key_projection;
  LinearWeights value_projection;
  LinearWeights attention_output_projection;
  core::TensorValue fsmn_weight;
  NormWeights final_norm;
  LinearWeights ffn_input_projection;
  LinearWeights ffn_output_projection;
  std::optional<LinearWeights> fused_qkv_projection = std::nullopt;
};

struct SanmBlockConfig {
  int64_t input_size = 0;
  int64_t model_size = 0;
  int64_t num_heads = 0;
  int64_t ffn_size = 0;
  int64_t fsmn_kernel_size = 0;
  float layer_norm_eps = 1.0e-5F;
  ScaledDotProductAttentionLowering attention_lowering =
      ScaledDotProductAttentionLowering::Explicit;
  std::optional<core::TensorValue> attn_pad_mask;
  std::optional<core::TensorValue> conv_pad_mask;
  bool use_flash = true;
};

void build_sinusoidal_pe(std::vector<float> & out, int64_t depth, int64_t frames);
std::vector<float> make_sinusoidal_positions(int64_t frames, int64_t channels);

core::TensorValue sanm_layer_norm(core::ModuleBuildContext &ctx,
                                  const core::TensorValue &input,
                                  const NormWeights &weights, float epsilon);

core::TensorValue sanm_fsmn_branch(core::ModuleBuildContext &ctx,
                                   const core::TensorValue &value,
                                   const core::TensorValue &fsmn_weight,
                                   int64_t kernel_size,
                                   const std::optional<core::TensorValue> &conv_pad_mask = std::nullopt);

core::TensorValue sanm_projection_block(core::ModuleBuildContext &ctx,
                                        const core::TensorValue &input,
                                        const SanmBlockWeightsView &weights,
                                        const SanmBlockConfig &config);

core::TensorValue sanm_residual_block(core::ModuleBuildContext &ctx,
                                      const core::TensorValue &input,
                                      const SanmBlockWeightsView &weights,
                                      const SanmBlockConfig &config);

} // namespace engine::modules

