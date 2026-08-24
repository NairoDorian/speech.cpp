#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include "ggml.h"
#include <cstdint>
#include <optional>

namespace engine::modules {

struct ShawAttentionWeights {
  NormWeights norm_attn;
  LinearWeights attn_q;
  LinearWeights attn_kv; // fused K|V (2 * inner_dim)
  core::TensorValue rel_pos_emb; // [head_dim, 2 * max_pos_emb + 1]
  LinearWeights attn_out;
};

struct ShawAttentionConfig {
  int64_t num_heads = 0;
  int64_t head_dim = 0;
  int64_t context_size = 0;
  int64_t num_blocks = 0;
  int64_t sequence_length = 0;
  float layer_norm_eps = 1.0e-5f;
};

core::TensorValue build_shaw_block_attention(
    core::ModuleBuildContext &ctx,
    const core::TensorValue &input,
    const std::optional<core::TensorValue> &zero_pad,
    const core::TensorValue &dists,
    const core::TensorValue &pad_mask_3d,
    const ShawAttentionWeights &weights,
    const ShawAttentionConfig &config);

} // namespace engine::modules
