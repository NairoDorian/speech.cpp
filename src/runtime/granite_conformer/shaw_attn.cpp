// src/granite_conformer/shaw_attn.cpp - shared Shaw block-local attention.
//
// See shaw_attn.h for the contract.

#include "shaw_attn.h"
#include "engine/framework/modules/attention/shaw_attention.h"

#include "ggml.h"
#include "transcribe-log.h"

#include <cmath>
#include <cstdio>

namespace transcribe::granite_conformer {

ggml_tensor * shaw_block_attn(ggml_context *          ctx,
                              ggml_tensor *           x,
                              ggml_tensor *           zero_pad,
                              ggml_tensor *           dists,
                              ggml_tensor *           pad_mask_3d,
                              const ShawAttnWeights & w,
                              int                     n_heads,
                              int                     head_dim,
                              int                     context_size,
                              int                     num_blocks,
                              int                     T_enc,
                              float                   layer_norm_eps) {
    const int64_t T_pad = static_cast<int64_t>(context_size) * num_blocks;

    if (T_pad > T_enc && zero_pad == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "granite_conformer::shaw_block_attn: "
                "zero_pad is null but T_pad > T_enc");
        return nullptr;
    }

    engine::core::ModuleBuildContext module_ctx;
    module_ctx.ggml = ctx;
    module_ctx.backend_type = engine::core::BackendType::Cpu;

    engine::modules::ShawAttentionWeights ew;
    ew.norm_attn = {
        w.norm_attn_w != nullptr ? engine::core::wrap_tensor(w.norm_attn_w, engine::core::TensorShape::from_dims({w.norm_attn_w->ne[0]}), w.norm_attn_w->type) : engine::core::TensorValue{},
        w.norm_attn_b != nullptr ? std::optional(engine::core::wrap_tensor(w.norm_attn_b, engine::core::TensorShape::from_dims({w.norm_attn_b->ne[0]}), w.norm_attn_b->type)) : std::nullopt
    };
    ew.attn_q = {
        engine::core::wrap_tensor(w.attn_q_w, engine::core::TensorShape::from_dims({w.attn_q_w->ne[0], w.attn_q_w->ne[1]}), w.attn_q_w->type),
        std::nullopt
    };
    ew.attn_kv = {
        engine::core::wrap_tensor(w.attn_kv_w, engine::core::TensorShape::from_dims({w.attn_kv_w->ne[0], w.attn_kv_w->ne[1]}), w.attn_kv_w->type),
        std::nullopt
    };
    ew.rel_pos_emb = engine::core::wrap_tensor(w.attn_rel_pos_emb, engine::core::TensorShape::from_dims({w.attn_rel_pos_emb->ne[0], w.attn_rel_pos_emb->ne[1]}), w.attn_rel_pos_emb->type);
    ew.attn_out = {
        engine::core::wrap_tensor(w.attn_out_w, engine::core::TensorShape::from_dims({w.attn_out_w->ne[0], w.attn_out_w->ne[1]}), w.attn_out_w->type),
        w.attn_out_b != nullptr ? std::optional(engine::core::wrap_tensor(w.attn_out_b, engine::core::TensorShape::from_dims({w.attn_out_b->ne[0]}), w.attn_out_b->type)) : std::nullopt
    };

    engine::modules::ShawAttentionConfig config;
    config.num_heads = n_heads;
    config.head_dim = head_dim;
    config.context_size = context_size;
    config.num_blocks = num_blocks;
    config.sequence_length = T_enc;
    config.layer_norm_eps = layer_norm_eps;

    const int64_t B = x->ne[2];
    auto input_val = engine::core::wrap_tensor(x, engine::core::TensorShape::from_dims({B, T_enc, x->ne[0]}), x->type);
    std::optional<engine::core::TensorValue> zero_pad_val = std::nullopt;
    if (zero_pad != nullptr) {
        zero_pad_val = engine::core::wrap_tensor(zero_pad, engine::core::TensorShape::from_dims({B, zero_pad->ne[1], zero_pad->ne[0]}), zero_pad->type);
    }
    auto dists_val = engine::core::wrap_tensor(dists, engine::core::TensorShape::from_dims({dists->ne[0], dists->ne[1]}), dists->type);
    auto pad_mask_val = engine::core::wrap_tensor(pad_mask_3d, engine::core::TensorShape::from_dims({pad_mask_3d->ne[0], pad_mask_3d->ne[1], pad_mask_3d->ne[2]}), pad_mask_3d->type);

    auto result = engine::modules::build_shaw_block_attention(
        module_ctx, input_val, zero_pad_val, dists_val, pad_mask_val, ew, config);
    return result.tensor;
}

}  // namespace transcribe::granite_conformer

