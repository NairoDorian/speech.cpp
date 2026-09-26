// granite_speech weights: binds the transcribe.cpp GGUF tensor catalog.
//
// Ported from src/runtime/arch/granite/weights.cpp:build_granite_weights (the
// names and ggml shapes; here in the engine's PyTorch [out, in] order, i.e.
// the GGUF ne[] reversed with trailing 1-dims dropped, as TensorSource reports
// them) and src/runtime/arch/granite/model.cpp:fuse_batch_norm.
//
// Dtypes: every matmul weight keeps the GGUF's dtype under the default
// weight_type=native (BF16 / F16 / Q8_0 / K-quants exactly as the arch loads
// them; Vulkan / Metal store BF16 as F16, a BackendWeightStore policy). Norms,
// biases and BatchNorm statistics are F32 in the GGUF and loaded as F32.
// Three tensors always keep the native dtype because the arch never requants
// them: the Shaw rel_pos_emb table and the Q-Former query (both only read
// through get_rows / cast) and the depthwise kernel, which the arch casts to
// F32 for ggml_conv_2d_dw_direct and which is therefore loaded as F32 here.

#include "engine/models/granite_speech/model.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace engine::models::granite_speech {
namespace {

constexpr float kBatchNormEps = 1e-5f;  // arch model.cpp kBnEps

std::string layer(const char * pattern, int i) {
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), pattern, i);
    return buffer;
}

}  // namespace

std::unique_ptr<GraniteSpeechWeights> load_granite_speech_weights(
    const GraniteSpeechAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type) {
    const auto & hp = assets.hparams;
    const auto & source = *assets.source;
    auto out = std::make_unique<GraniteSpeechWeights>();
    out->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "granite_speech.weights", 8 * 1024 * 1024);
    auto & store = *out->store;

    const auto linear = [&](const std::string & name, int64_t in, int64_t out_features, bool bias) {
        modules::LinearWeights w;
        w.weight = store.load_tensor(source, name + ".weight", type, {out_features, in});
        if (bias) {
            w.bias = store.load_f32_tensor(source, name + ".bias", {out_features});
        }
        return w;
    };
    const auto norm = [&](const std::string & name, int64_t dim) {
        modules::NormWeights w;
        w.weight = store.load_f32_tensor(source, name + ".weight", {dim});
        w.bias = store.load_f32_tensor(source, name + ".bias", {dim});
        return w;
    };

    // Derived dims (arch build_granite_weights).
    const int64_t enc_h = hp.enc_hidden;
    const int64_t enc_inner = static_cast<int64_t>(hp.enc_n_heads) * hp.enc_head_dim;
    const int64_t enc_ffn = enc_h * hp.enc_feedforward_mult;
    const int64_t conv_inner = enc_h * hp.enc_conv_expansion;
    const int64_t conv_k = hp.enc_conv_kernel_size;
    const int64_t rel_pos_len = 2 * static_cast<int64_t>(hp.enc_max_pos_emb) + 1;
    const int64_t prj_h = hp.prj_hidden;
    const int64_t prj_kv_in = hp.prj_encoder_hidden_size;
    const int64_t dec_h = hp.dec_hidden;
    const int64_t q_out = static_cast<int64_t>(hp.dec_n_heads) * hp.dec_head_dim;
    const int64_t kv_out = static_cast<int64_t>(hp.dec_n_kv_heads) * hp.dec_head_dim;

    // ---- Encoder: top level ----
    out->enc_input_linear = linear("enc.input_linear", hp.enc_input_dim, enc_h, true);
    out->enc_ctc_proj = linear("enc.ctc_proj", enc_h, hp.enc_output_dim, true);
    out->enc_ctc_bypass = linear("enc.ctc_bypass", hp.enc_output_dim, enc_h, true);

    // ---- Encoder: Conformer blocks ----
    out->encoder.resize(static_cast<size_t>(hp.enc_n_layers));
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = out->encoder[static_cast<size_t>(i)];
        b.norm_ff1 = norm(layer("enc.blocks.%d.norm_ff1", i), enc_h);
        b.ff1_up = linear(layer("enc.blocks.%d.ff1.up", i), enc_h, enc_ffn, true);
        b.ff1_down = linear(layer("enc.blocks.%d.ff1.down", i), enc_ffn, enc_h, true);

        // Shaw block-local attention: q and the fused k|v carry no bias.
        b.attention.norm_attn = norm(layer("enc.blocks.%d.norm_attn", i), enc_h);
        b.attention.attn_q = linear(layer("enc.blocks.%d.attn.q", i), enc_h, enc_inner, false);
        b.attention.attn_kv = linear(layer("enc.blocks.%d.attn.kv", i), enc_h, 2 * enc_inner, false);
        b.attention.attn_out = linear(layer("enc.blocks.%d.attn.out", i), enc_inner, enc_h, true);
        b.attention.rel_pos_emb = store.load_tensor(source, layer("enc.blocks.%d.attn.rel_pos_emb.weight", i),
            assets::TensorStorageType::Native, {rel_pos_len, hp.enc_head_dim});

        // Conv module. The pointwise kernels are Conv1d (out, in, 1); bound as
        // their [out, in] matrices (the arch ggml_reshape_2d's them).
        b.norm_conv = norm(layer("enc.blocks.%d.norm_conv", i), enc_h);
        b.conv_pointwise1.weight = store.load_tensor_as_shape(source,
            layer("enc.blocks.%d.conv.pointwise1.weight", i), type, {2 * conv_inner, enc_h, 1},
            core::TensorShape::from_dims({2 * conv_inner, enc_h}));
        b.conv_pointwise1.bias = store.load_f32_tensor(source, layer("enc.blocks.%d.conv.pointwise1.bias", i),
            {2 * conv_inner});
        b.conv_depthwise = store.load_f32_tensor(source, layer("enc.blocks.%d.conv.depthwise.weight", i),
            {conv_inner, 1, conv_k});
        b.conv_pointwise2.weight = store.load_tensor_as_shape(source,
            layer("enc.blocks.%d.conv.pointwise2.weight", i), type, {enc_h, conv_inner, 1},
            core::TensorShape::from_dims({enc_h, conv_inner}));
        b.conv_pointwise2.bias = store.load_f32_tensor(source, layer("enc.blocks.%d.conv.pointwise2.bias", i),
            {enc_h});

        // BatchNorm fused on the host at load (arch model.cpp:fuse_batch_norm,
        // same float arithmetic): scale = gamma / sqrt(var + eps),
        // bias = beta - mean * scale.
        const auto gamma = source.require_f32(layer("enc.blocks.%d.conv.bn.weight", i), {conv_inner});
        const auto beta = source.require_f32(layer("enc.blocks.%d.conv.bn.bias", i), {conv_inner});
        const auto mean = source.require_f32(layer("enc.blocks.%d.conv.bn.running_mean", i), {conv_inner});
        const auto var = source.require_f32(layer("enc.blocks.%d.conv.bn.running_var", i), {conv_inner});
        std::vector<float> scale(static_cast<size_t>(conv_inner));
        std::vector<float> bias(static_cast<size_t>(conv_inner));
        for (size_t c = 0; c < scale.size(); ++c) {
            const float s = gamma[c] / std::sqrt(var[c] + kBatchNormEps);
            scale[c] = s;
            bias[c] = beta[c] - mean[c] * s;
        }
        b.conv_bn_scale = store.make_f32(core::TensorShape::from_dims({conv_inner}), std::move(scale));
        b.conv_bn_bias = store.make_f32(core::TensorShape::from_dims({conv_inner}), std::move(bias));

        b.norm_ff2 = norm(layer("enc.blocks.%d.norm_ff2", i), enc_h);
        b.ff2_up = linear(layer("enc.blocks.%d.ff2.up", i), enc_h, enc_ffn, true);
        b.ff2_down = linear(layer("enc.blocks.%d.ff2.down", i), enc_ffn, enc_h, true);
        b.norm_post = norm(layer("enc.blocks.%d.norm_post", i), enc_h);
    }

    // ---- Projector (BLIP-2 Q-Former) ----
    // proj.query is (1, num_queries, hidden) in PyTorch; TensorSource drops
    // the leading 1. The arch hard-codes 3 queries in its shape check; here
    // the count is window_size / downsample_rate, which is what its graph uses.
    out->proj_query = store.load_tensor(source, "proj.query", assets::TensorStorageType::Native,
        {hp.prj_num_queries, prj_h});
    out->proj_linear = linear("proj.linear", prj_h, dec_h, true);
    // "proj.qformer.final_norm" is a converter misnomer for
    // Blip2QFormerModel.layernorm, the Q-Former INPUT norm (arch projector.cpp).
    out->proj_qformer_norm = norm("proj.qformer.final_norm", prj_h);
    out->projector.resize(static_cast<size_t>(hp.prj_n_layers));
    for (int i = 0; i < hp.prj_n_layers; ++i) {
        auto & b = out->projector[static_cast<size_t>(i)];
        b.self_q = linear(layer("proj.qformer.blocks.%d.self_attn.q", i), prj_h, prj_h, true);
        b.self_k = linear(layer("proj.qformer.blocks.%d.self_attn.k", i), prj_h, prj_h, true);
        b.self_v = linear(layer("proj.qformer.blocks.%d.self_attn.v", i), prj_h, prj_h, true);
        b.self_out = linear(layer("proj.qformer.blocks.%d.self_attn.out", i), prj_h, prj_h, true);
        b.norm_self = norm(layer("proj.qformer.blocks.%d.norm_self_attn", i), prj_h);
        b.cross_q = linear(layer("proj.qformer.blocks.%d.cross_attn.q", i), prj_h, prj_h, true);
        b.cross_k = linear(layer("proj.qformer.blocks.%d.cross_attn.k", i), prj_kv_in, prj_h, true);
        b.cross_v = linear(layer("proj.qformer.blocks.%d.cross_attn.v", i), prj_kv_in, prj_h, true);
        b.cross_out = linear(layer("proj.qformer.blocks.%d.cross_attn.out", i), prj_h, prj_h, true);
        b.norm_cross = norm(layer("proj.qformer.blocks.%d.norm_cross_attn", i), prj_h);
        b.ffn_up = linear(layer("proj.qformer.blocks.%d.ffn.up", i), prj_h, hp.prj_intermediate, true);
        b.ffn_down = linear(layer("proj.qformer.blocks.%d.ffn.down", i), hp.prj_intermediate, prj_h, true);
        b.norm_ffn = norm(layer("proj.qformer.blocks.%d.norm_ffn", i), prj_h);
    }

    // ---- Granite-4 LM ----
    out->token_embedding = store.load_tensor(source, "dec.token_embd.weight", type, {hp.dec_vocab_size, dec_h});
    out->decoder.resize(static_cast<size_t>(hp.dec_n_layers));
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        auto & b = out->decoder[static_cast<size_t>(i)];
        b.norm_attn = store.load_f32_tensor(source, layer("dec.blocks.%d.norm_attn.weight", i), {dec_h});
        b.norm_ffn = store.load_f32_tensor(source, layer("dec.blocks.%d.norm_ffn.weight", i), {dec_h});
        b.q = store.load_tensor(source, layer("dec.blocks.%d.attn.q.weight", i), type, {q_out, dec_h});
        b.k = store.load_tensor(source, layer("dec.blocks.%d.attn.k.weight", i), type, {kv_out, dec_h});
        b.v = store.load_tensor(source, layer("dec.blocks.%d.attn.v.weight", i), type, {kv_out, dec_h});
        b.o = store.load_tensor(source, layer("dec.blocks.%d.attn.o.weight", i), type, {dec_h, q_out});
        b.gate = store.load_tensor(source, layer("dec.blocks.%d.ffn.gate.weight", i), type, {hp.dec_intermediate, dec_h});
        b.up = store.load_tensor(source, layer("dec.blocks.%d.ffn.up.weight", i), type, {hp.dec_intermediate, dec_h});
        b.down = store.load_tensor(source, layer("dec.blocks.%d.ffn.down.weight", i), type, {dec_h, hp.dec_intermediate});
    }
    out->output_norm = store.load_f32_tensor(source, "dec.output_norm.weight", {dec_h});
    if (hp.dec_tie_word_embeddings) {
        // Tied head (granite-speech-4.1-2b-plus): no dec.output.weight.
        out->output = out->token_embedding;
    } else {
        out->output = store.load_tensor(source, "dec.output.weight", type, {hp.dec_vocab_size, dec_h});
    }

    store.upload();
    return out;
}

}  // namespace engine::models::granite_speech
