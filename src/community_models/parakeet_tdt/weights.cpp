#include "engine/community_models/parakeet_tdt/weights.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/weight_binding.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = engine::modules::binding;

engine::modules::LinearWeights load_linear(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const std::string & prefix, engine::assets::TensorStorageType st,
    int64_t out_f, int64_t in_f, bool use_bias) {
    engine::modules::LinearWeights w;
    w.weight = store.load_tensor(source, prefix + ".weight", st, {out_f, in_f});
    if (use_bias) w.bias = store.load_f32_tensor(source, prefix + ".bias", {out_f});
    return w;
}

// Loads a weight with a constant activation-side scale folded in, so the graph
// does not need a separate ggml_scale pass over the activations at run time.
//
// Folding is exact whenever `scale` is a power of two — which is the only way
// it is used here (0.5 for the feed-forward half-step, sqrt(d_model)=32 for the
// subsampling xscaling). Multiplying a float by a power of two only adjusts the
// exponent, and since every product term is scaled identically the sum scales
// identically too: sum(0.5*w_i*x_i) == 0.5*sum(w_i*x_i) bit for bit. It also
// survives quantisation: q8_0 derives its block scale from max|w|, so halving
// every weight halves the block scale and leaves the stored integers unchanged.
std::vector<float> scaled_f32(std::vector<float> values, float scale) {
    for (auto & value : values) { value *= scale; }
    return values;
}

engine::modules::Conv2dWeights load_conv2d(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const std::string & prefix, engine::assets::TensorStorageType st,
    int64_t oc, int64_t ic, int64_t kh, int64_t kw, bool use_bias) {
    engine::modules::Conv2dWeights w;
    w.weight = store.load_tensor(source, prefix + ".weight", st, {oc, ic, kh, kw});
    if (use_bias) w.bias = store.load_f32_tensor(source, prefix + ".bias", {oc});
    return w;
}

ParakeetSubsamplingWeights load_subsampling(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, engine::assets::TensorStorageType matmul_st,
    engine::assets::TensorStorageType conv_st) {
    ParakeetSubsamplingWeights w;
    const auto & enc = config.encoder;
    const int64_t C = enc.subsampling_channels;
    const std::string sub = "encoder.subsampling";
    w.conv_in = load_conv2d(store, source, sub + ".layers.0", conv_st, C, 1, enc.subsampling_kernel, enc.subsampling_kernel, true);
    w.layers.resize(2);
    w.layers[0].depthwise_weight = store.load_tensor(source, sub + ".layers.2.weight", conv_st, {C, 1, enc.subsampling_kernel, enc.subsampling_kernel});
    w.layers[0].depthwise_bias = store.load_f32_tensor(source, sub + ".layers.2.bias", {C});
    w.layers[0].pointwise = load_conv2d(store, source, sub + ".layers.3", conv_st, C, C, 1, 1, true);
    w.layers[1].depthwise_weight = store.load_tensor(source, sub + ".layers.5.weight", conv_st, {C, 1, enc.subsampling_kernel, enc.subsampling_kernel});
    w.layers[1].depthwise_bias = store.load_f32_tensor(source, sub + ".layers.5.bias", {C});
    w.layers[1].pointwise = load_conv2d(store, source, sub + ".layers.6", conv_st, C, C, 1, 1, true);
    const int64_t ff = config.frontend.feature_size / enc.subsampling_factor;
    // RelPositionalEncoding's xscaling (multiply by sqrt(d_model)) folded into
    // the subsampling projection instead of running as its own ggml_scale over
    // the encoder input. sqrt(1024) = 32 exactly, so this is bit-exact; see
    // scaled_f32. Both weight and bias must be scaled, since the scale was
    // applied after the linear.
    const float xscale = std::sqrt(static_cast<float>(enc.hidden_size));
    w.linear.weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({enc.hidden_size, C * ff}),
        matmul_st,
        scaled_f32(source.require_f32(sub + ".linear.weight", {enc.hidden_size, C * ff}), xscale));
    w.linear.bias = store.make_f32(
        engine::core::TensorShape::from_dims({enc.hidden_size}),
        scaled_f32(source.require_f32(sub + ".linear.bias", {enc.hidden_size}), xscale));
    return w;
}

std::pair<std::vector<float>, std::vector<float>> fold_bn(
    const engine::assets::TensorSource & source, const std::string & dw_pfx, const std::string & bn_pfx,
    int64_t d_model, int64_t K) {
    auto bn_w = source.require_f32(bn_pfx + ".weight", {d_model});
    auto bn_b = source.require_f32(bn_pfx + ".bias", {d_model});
    auto bn_m = source.require_f32(bn_pfx + ".running_mean", {d_model});
    auto bn_v = source.require_f32(bn_pfx + ".running_var", {d_model});
    auto dw_w = source.require_f32(dw_pfx + ".weight", {d_model, 1, K});

    // Layout: dw_w is row-major [channels][1][kernel], so element (c,0,ki) is at offset c*K+ki
    // ggml stores [ne0=K, ne1=1, ne2=d_model] → consecutive K elements per channel
    std::vector<float> w_f(static_cast<size_t>(K * d_model));
    for (int64_t c = 0; c < d_model; ++c)
        for (int64_t ki = 0; ki < K; ++ki)
            w_f[static_cast<size_t>(ki + c * K)] = dw_w[static_cast<size_t>(c * K + ki)];

    std::vector<float> dw_b(static_cast<size_t>(d_model), 0.f);
    const float eps = 1e-5f;
    for (int64_t c = 0; c < d_model; ++c) {
        float s = bn_w[static_cast<size_t>(c)] / std::sqrt(std::max(bn_v[static_cast<size_t>(c)], eps) + eps);
        for (int64_t ki = 0; ki < K; ++ki)
            w_f[static_cast<size_t>(ki + c * K)] *= s;
        dw_b[static_cast<size_t>(c)] = -bn_m[static_cast<size_t>(c)] * s + bn_b[static_cast<size_t>(c)];
    }
    return {std::move(w_f), std::move(dw_b)};
}

ParakeetEncoderLayerWeights load_encoder_layer(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, int64_t idx, engine::assets::TensorStorageType matmul_st) {
    const std::string p = "encoder.layers." + std::to_string(idx);
    const auto & enc = config.encoder;
    const int64_t h = enc.hidden_size;
    const int64_t hd = h / enc.heads;
    ParakeetEncoderLayerWeights layer;
    layer.norm_ff1 = binding::norm_from_source(store, source, p + ".norm_feed_forward1", h);
    layer.ff1_linear1 = {store.load_tensor(source, p + ".feed_forward1.linear1.weight", matmul_st, {enc.intermediate_size, h}), std::nullopt};
    // The 0.5 half-step each feed-forward branch contributes to the residual is
    // folded into linear2 rather than run as a ggml_scale over [seq, hidden]
    // after every one of the 48 feed-forward blocks. Exact — see scaled_f32.
    layer.ff1_linear2 = {store.make_from_f32(
        engine::core::TensorShape::from_dims({h, enc.intermediate_size}),
        matmul_st,
        scaled_f32(source.require_f32(p + ".feed_forward1.linear2.weight", {h, enc.intermediate_size}), 0.5f)), std::nullopt};
    layer.norm_attn = binding::norm_from_source(store, source, p + ".norm_self_att", h);
    // Fused QKV: one [3h, h] matmul instead of three separate [h, h] matmuls per
    // layer. Read the raw F32 rows for q/k/v and concatenate along the
    // output-feature axis: HF/PyTorch Linear weight layout is [out_features,
    // in_features] row-major with in_features contiguous per row, so
    // concatenating the three already-flat [h, h] row-major buffers in q/k/v
    // order produces exactly the [3h, h] row-major layout of a single fused
    // Linear(h, 3h) — no interleaving needed, q_weight/k_weight/v_weight are
    // left unset (build_encoder_layer only ever reads qkv_weight).
    {
        auto q_f32 = source.require_f32(p + ".self_attn.q_proj.weight", {h, h});
        auto k_f32 = source.require_f32(p + ".self_attn.k_proj.weight", {h, h});
        auto v_f32 = source.require_f32(p + ".self_attn.v_proj.weight", {h, h});
        std::vector<float> qkv_f32;
        qkv_f32.reserve(q_f32.size() + k_f32.size() + v_f32.size());
        qkv_f32.insert(qkv_f32.end(), q_f32.begin(), q_f32.end());
        qkv_f32.insert(qkv_f32.end(), k_f32.begin(), k_f32.end());
        qkv_f32.insert(qkv_f32.end(), v_f32.begin(), v_f32.end());
        layer.self_attn.qkv_weight = store.make_from_f32(
            engine::core::TensorShape::from_dims({3 * h, h}), matmul_st, std::move(qkv_f32));
    }
    layer.self_attn.out_weight = store.load_tensor(source, p + ".self_attn.o_proj.weight", matmul_st, {h, h});
    layer.pos_weight = store.load_tensor(source, p + ".self_attn.relative_k_proj.weight", matmul_st, {h, h});
    layer.pos_bias_u = store.load_f32_tensor(source, p + ".self_attn.bias_u", {enc.heads, hd});
    layer.pos_bias_v = store.load_f32_tensor(source, p + ".self_attn.bias_v", {enc.heads, hd});
    layer.norm_conv = binding::norm_from_source(store, source, p + ".norm_conv", h);
    // pointwise_conv{1,2} are kernel_size=1 Conv1d — mathematically a Linear layer, and
    // that's exactly how build_fastconformer_conv_module runs them (via LinearModule/mul_mat,
    // not a real conv op). They belong in the matmul weight-storage bucket (which allows
    // q8_0) rather than the conv bucket (capped at f16 for actual conv ops) — leaving them
    // in the conv bucket silently excludes ~3*h^2 params/layer from matmul quantization.
    layer.conv_pw1 = {store.load_tensor_as_shape(source, p + ".conv.pointwise_conv1.weight", matmul_st, {2*h, h, 1}, engine::core::TensorShape::from_dims({2*h, h})), std::nullopt};
    layer.conv_pw2 = {store.load_tensor_as_shape(source, p + ".conv.pointwise_conv2.weight", matmul_st, {h, h, 1}, engine::core::TensorShape::from_dims({h, h})), std::nullopt};
    auto [fd_w, fd_b] = fold_bn(source, p + ".conv.depthwise_conv", p + ".conv.norm", h, enc.conv_kernel);
    layer.conv_dw_weight = store.make_from_f32(engine::core::TensorShape::from_dims({h, 1, enc.conv_kernel}), engine::assets::TensorStorageType::F32, std::move(fd_w));
    layer.conv_dw_bias = store.make_f32(engine::core::TensorShape::from_dims({h}), std::move(fd_b));
    layer.norm_ff2 = binding::norm_from_source(store, source, p + ".norm_feed_forward2", h);
    layer.ff2_linear1 = {store.load_tensor(source, p + ".feed_forward2.linear1.weight", matmul_st, {enc.intermediate_size, h}), std::nullopt};
    layer.ff2_linear2 = {store.make_from_f32(
        engine::core::TensorShape::from_dims({h, enc.intermediate_size}),
        matmul_st,
        scaled_f32(source.require_f32(p + ".feed_forward2.linear2.weight", {h, enc.intermediate_size}), 0.5f)), std::nullopt};
    layer.norm_out = binding::norm_from_source(store, source, p + ".norm_out", h);
    return layer;
}

// ---------------------------------------------------------------------------
// TranscribeGguf layout (the transcribe.cpp GGUF behind the rename view in
// assets.cpp). The graph is the same; what changes is how the weights are
// stored, so the numbers follow the retired arch:
//   - every matmul weight is bound in the GGUF's own type (Native) and never
//     requantized: no QKV fusion, no scale folding - the 0.5 macaron step and
//     the xscaling run as graph scales after the matmul + bias, as in
//     src/runtime/conformer/conformer.cpp macaron_ff_residual and
//     arch/parakeet/encoder.cpp build_encoder_graph;
//   - the linear/conv biases exist when stt.parakeet.encoder.use_bias;
//   - on a CPU backend the F16 pointwise convs are widened to F32
//     (arch/parakeet/model.cpp promote_conv_pw_to_f32_on_cpu);
//   - the predictor/joint/CTC head are F32 (arch/parakeet/decoder.cpp
//     build_host_decoder_weights dequantizes all of them).
// ---------------------------------------------------------------------------

engine::modules::LinearWeights load_native_linear(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const std::string & prefix, engine::assets::TensorStorageType st,
    int64_t out_f, int64_t in_f, bool use_bias) {
    engine::modules::LinearWeights w;
    w.weight = store.load_tensor(source, prefix + ".weight", st, {out_f, in_f});
    if (use_bias) w.bias = store.load_f32_tensor(source, prefix + ".bias", {out_f});
    return w;
}

// arch/parakeet/model.cpp promote_conv_pw_to_f32_on_cpu: F16 pointwise conv
// weights run as F32 on a CPU primary backend.
engine::assets::TensorStorageType transcribe_pointwise_storage(
    const engine::assets::TensorSource & source, const std::string & name,
    engine::assets::TensorStorageType requested, engine::core::BackendType backend_type) {
    if (requested != engine::assets::TensorStorageType::Native || backend_type != engine::core::BackendType::Cpu) {
        return requested;
    }
    const auto native = engine::assets::tensor_storage_type_for_dtype(source.require_metadata(name).dtype);
    return native == engine::assets::TensorStorageType::F16 ? engine::assets::TensorStorageType::F32 : requested;
}

// BatchNorm fused into the depthwise conv, with the arch's statistics:
// scale = gamma / sqrt(running_var + 1e-5) (arch/parakeet/model.cpp
// fuse_batch_norm). The arch runs y = (dw(x) + b_dw) * scale + (beta -
// mean * scale); folding it into the kernel is the same math up to rounding.
std::pair<std::vector<float>, std::vector<float>> fold_bn_transcribe(
    const engine::assets::TensorSource & source, const std::string & dw_pfx, const std::string & bn_pfx,
    int64_t d_model, int64_t K, bool dw_bias) {
    const auto bn_w = source.require_f32(bn_pfx + ".weight", {d_model});
    const auto bn_b = source.require_f32(bn_pfx + ".bias", {d_model});
    const auto bn_m = source.require_f32(bn_pfx + ".running_mean", {d_model});
    const auto bn_v = source.require_f32(bn_pfx + ".running_var", {d_model});
    const auto dw_w = source.require_f32(dw_pfx + ".weight", {d_model, 1, K});
    std::vector<float> dw_b(static_cast<size_t>(d_model), 0.0f);
    if (dw_bias) {
        dw_b = source.require_f32(dw_pfx + ".bias", {d_model});
    }
    constexpr float kBnEps = 1.0e-5f;
    std::vector<float> w_f(static_cast<size_t>(K * d_model));
    std::vector<float> b_f(static_cast<size_t>(d_model));
    for (int64_t c = 0; c < d_model; ++c) {
        const size_t ci = static_cast<size_t>(c);
        const float s = bn_w[ci] / std::sqrt(bn_v[ci] + kBnEps);
        for (int64_t ki = 0; ki < K; ++ki) {
            const size_t idx = static_cast<size_t>(c * K + ki);
            w_f[idx] = dw_w[idx] * s;
        }
        b_f[ci] = dw_b[ci] * s + (bn_b[ci] - bn_m[ci] * s);
    }
    return {std::move(w_f), std::move(b_f)};
}

ParakeetSubsamplingWeights load_subsampling_transcribe(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, engine::assets::TensorStorageType matmul_st,
    engine::assets::TensorStorageType conv_st) {
    ParakeetSubsamplingWeights w;
    const auto & enc = config.encoder;
    const int64_t C = enc.subsampling_channels;
    const int64_t k = enc.subsampling_kernel;
    const std::string sub = "encoder.subsampling";
    w.conv_in = load_conv2d(store, source, sub + ".layers.0", conv_st, C, 1, k, k, true);
    w.layers.resize(2);
    const char * dw_names[2] = {".layers.2", ".layers.5"};
    const char * pw_names[2] = {".layers.3", ".layers.6"};
    for (size_t i = 0; i < 2; ++i) {
        w.layers[i].depthwise_weight = store.load_tensor(source, sub + dw_names[i] + ".weight", conv_st, {C, 1, k, k});
        w.layers[i].depthwise_bias = store.load_f32_tensor(source, sub + dw_names[i] + ".bias", {C});
        w.layers[i].pointwise = load_conv2d(store, source, sub + pw_names[i], conv_st, C, C, 1, 1, true);
    }
    const int64_t ff = config.frontend.feature_size / enc.subsampling_factor;
    w.linear = load_native_linear(store, source, sub + ".linear", matmul_st, enc.hidden_size, C * ff, true);
    w.output_scale = enc.xscaling ? std::sqrt(static_cast<float>(enc.hidden_size)) : 1.0f;
    return w;
}

ParakeetEncoderLayerWeights load_encoder_layer_transcribe(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, int64_t idx, engine::assets::TensorStorageType matmul_st,
    engine::core::BackendType backend_type) {
    const std::string p = "encoder.layers." + std::to_string(idx);
    const auto & enc = config.encoder;
    const int64_t h = enc.hidden_size;
    const int64_t hd = h / enc.heads;
    const int64_t ff = enc.intermediate_size;
    const bool bias = enc.use_bias;
    ParakeetEncoderLayerWeights layer;
    layer.norm_ff1 = binding::norm_from_source(store, source, p + ".norm_feed_forward1", h);
    layer.ff1_linear1 = load_native_linear(store, source, p + ".feed_forward1.linear1", matmul_st, ff, h, bias);
    layer.ff1_linear2 = load_native_linear(store, source, p + ".feed_forward1.linear2", matmul_st, h, ff, bias);
    layer.norm_attn = binding::norm_from_source(store, source, p + ".norm_self_att", h);
    {
        const auto q = load_native_linear(store, source, p + ".self_attn.q_proj", matmul_st, h, h, bias);
        const auto k = load_native_linear(store, source, p + ".self_attn.k_proj", matmul_st, h, h, bias);
        const auto v = load_native_linear(store, source, p + ".self_attn.v_proj", matmul_st, h, h, bias);
        const auto o = load_native_linear(store, source, p + ".self_attn.o_proj", matmul_st, h, h, bias);
        layer.self_attn.q_weight = q.weight;
        layer.self_attn.q_bias = q.bias;
        layer.self_attn.k_weight = k.weight;
        layer.self_attn.k_bias = k.bias;
        layer.self_attn.v_weight = v.weight;
        layer.self_attn.v_bias = v.bias;
        layer.self_attn.out_weight = o.weight;
        layer.self_attn.out_bias = o.bias;
    }
    // linear_pos is bias-free in NeMo even when use_bias (weights.h).
    layer.pos_weight = store.load_tensor(source, p + ".self_attn.relative_k_proj.weight", matmul_st, {h, h});
    layer.pos_bias_u = store.load_f32_tensor(source, p + ".self_attn.bias_u", {enc.heads, hd});
    layer.pos_bias_v = store.load_f32_tensor(source, p + ".self_attn.bias_v", {enc.heads, hd});
    layer.norm_conv = binding::norm_from_source(store, source, p + ".norm_conv", h);
    {
        const std::string pw1 = p + ".conv.pointwise_conv1";
        const std::string pw2 = p + ".conv.pointwise_conv2";
        layer.conv_pw1.weight = store.load_tensor_as_shape(source, pw1 + ".weight",
            transcribe_pointwise_storage(source, pw1 + ".weight", matmul_st, backend_type),
            {2 * h, h, 1}, engine::core::TensorShape::from_dims({2 * h, h}));
        if (bias) layer.conv_pw1.bias = store.load_f32_tensor(source, pw1 + ".bias", {2 * h});
        layer.conv_pw2.weight = store.load_tensor_as_shape(source, pw2 + ".weight",
            transcribe_pointwise_storage(source, pw2 + ".weight", matmul_st, backend_type),
            {h, h, 1}, engine::core::TensorShape::from_dims({h, h}));
        if (bias) layer.conv_pw2.bias = store.load_f32_tensor(source, pw2 + ".bias", {h});
    }
    auto [fd_w, fd_b] = fold_bn_transcribe(source, p + ".conv.depthwise_conv", p + ".conv.norm", h, enc.conv_kernel, bias);
    layer.conv_dw_weight = store.make_from_f32(engine::core::TensorShape::from_dims({h, 1, enc.conv_kernel}),
        engine::assets::TensorStorageType::F32, std::move(fd_w));
    layer.conv_dw_bias = store.make_f32(engine::core::TensorShape::from_dims({h}), std::move(fd_b));
    layer.norm_ff2 = binding::norm_from_source(store, source, p + ".norm_feed_forward2", h);
    layer.ff2_linear1 = load_native_linear(store, source, p + ".feed_forward2.linear1", matmul_st, ff, h, bias);
    layer.ff2_linear2 = load_native_linear(store, source, p + ".feed_forward2.linear2", matmul_st, h, ff, bias);
    layer.norm_out = binding::norm_from_source(store, source, p + ".norm_out", h);
    layer.ff_residual_scale = 0.5f;
    return layer;
}

ParakeetDecoderWeights load_decoder_weights_transcribe(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, engine::assets::TensorStorageType matmul_st) {
    ParakeetDecoderWeights w;
    // F32 unless the caller explicitly asked for another matmul storage type.
    const auto st = matmul_st == engine::assets::TensorStorageType::Native
        ? engine::assets::TensorStorageType::F32 : matmul_st;
    const int64_t d_enc = config.encoder.hidden_size;
    if (config.head_kind == ParakeetHeadKind::Ctc) {
        const auto meta = source.require_metadata("ctc_head.weight");
        if (meta.shape.size() != 3 || meta.shape[0] != config.vocab_size || meta.shape[1] != d_enc ||
            meta.shape[2] != 1) {
            throw std::runtime_error(
                "Parakeet transcribe.cpp GGUF: head.ctc.weight must be [vocab + 1 = " +
                std::to_string(config.vocab_size) + ", d_model = " + std::to_string(d_enc) + ", 1]");
        }
        engine::modules::LinearWeights head;
        head.weight = store.load_tensor_as_shape(source, "ctc_head.weight", st, {config.vocab_size, d_enc, 1},
            engine::core::TensorShape::from_dims({config.vocab_size, d_enc}));
        head.bias = store.load_f32_tensor(source, "ctc_head.bias", {config.vocab_size});
        w.ctc_head = std::move(head);
        return w;
    }
    const int64_t H = config.decoder_hidden_size;
    const int64_t J = config.joint_hidden_size;
    w.embedding_host = source.require_f32("decoder.embedding.weight", {config.vocab_size, H});
    w.lstm_layers.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t l = 0; l < config.decoder_layers; ++l) {
        const std::string s = std::to_string(l);
        engine::modules::LSTMCellWeights cell;
        cell.weight_ih = store.load_tensor(source, "decoder.lstm.weight_ih_l" + s, st, {4 * H, H});
        cell.weight_hh = store.load_tensor(source, "decoder.lstm.weight_hh_l" + s, st, {4 * H, H});
        cell.bias_ih = store.load_f32_tensor(source, "decoder.lstm.bias_l" + s, {4 * H});  // bias_ih + bias_hh
        w.lstm_layers.push_back(std::move(cell));
    }
    w.decoder_projector = load_native_linear(store, source, "decoder.decoder_projector", st, J, H, true);
    w.joint_enc = load_native_linear(store, source, "encoder_projector", st, J, d_enc, true);
    const int64_t jout = config.vocab_size + static_cast<int64_t>(config.durations.size());
    w.joint_head = load_native_linear(store, source, "joint.head", st, jout, J, true);
    return w;
}

}  // namespace

ParakeetEncoderWeights load_encoder_weights(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, engine::assets::TensorStorageType matmul_st,
    engine::assets::TensorStorageType conv_st) {
    ParakeetEncoderWeights w;
    w.subsampling = load_subsampling(store, source, config, matmul_st, conv_st);
    w.layers.reserve(static_cast<size_t>(config.encoder.layers));
    for (int64_t i = 0; i < config.encoder.layers; ++i)
        w.layers.push_back(load_encoder_layer(store, source, config, i, matmul_st));
    return w;
}

ParakeetDecoderWeights load_decoder_weights(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, engine::assets::TensorStorageType st) {
    ParakeetDecoderWeights w;
    const int64_t H = config.decoder_hidden_size;
    w.embedding = store.load_tensor(source, "decoder.embedding.weight", st, {config.vocab_size, H});
    w.lstm_layers.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t l = 0; l < config.decoder_layers; ++l) {
        const std::string p = "decoder.lstm";
        w.lstm_layers.push_back({
            store.load_tensor(source, p + ".weight_ih_l" + std::to_string(l), st, {4*H, H}),
            store.load_tensor(source, p + ".weight_hh_l" + std::to_string(l), st, {4*H, H}),
            store.load_f32_tensor(source, p + ".bias_ih_l" + std::to_string(l), {4*H}),
            store.load_f32_tensor(source, p + ".bias_hh_l" + std::to_string(l), {4*H}),
        });
    }
    w.decoder_projector = load_linear(store, source, "decoder.decoder_projector", st, H, H, true);
    w.joint_enc = load_linear(store, source, "encoder_projector", st, H, config.encoder.hidden_size, true);
    const int64_t jout = config.vocab_size + static_cast<int64_t>(config.durations.size());
    w.joint_head = load_linear(store, source, "joint.head", st, jout, H, true);
    return w;
}

std::shared_ptr<const ParakeetWeights> load_parakeet_weights(
    const ParakeetTDTAssets & assets, ggml_backend_t backend, engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_st, engine::assets::TensorStorageType conv_st, size_t ctx_bytes) {
    if (!assets.source) throw std::runtime_error("Parakeet TDT requires a tensor source");
    const auto t0 = Clock::now();
    auto w = std::make_shared<ParakeetWeights>();
    w->store = std::make_shared<engine::core::BackendWeightStore>(backend, backend_type, "parakeet_tdt.weights", ctx_bytes);
    if (assets.transcribe_layout()) {
        const auto & config = assets.config;
        w->encoder.subsampling = load_subsampling_transcribe(*w->store, *assets.source, config, matmul_st, conv_st);
        w->encoder.layers.reserve(static_cast<size_t>(config.encoder.layers));
        for (int64_t i = 0; i < config.encoder.layers; ++i) {
            w->encoder.layers.push_back(
                load_encoder_layer_transcribe(*w->store, *assets.source, config, i, matmul_st, backend_type));
        }
        w->decoder = load_decoder_weights_transcribe(*w->store, *assets.source, config, matmul_st);
    } else {
        w->encoder = load_encoder_weights(*w->store, *assets.source, assets.config, matmul_st, conv_st);
        w->decoder = load_decoder_weights(*w->store, *assets.source, assets.config, matmul_st);
    }
    w->store->upload();
    assets.source->release_storage();
    debug::timing_log_scalar("parakeet_tdt.weights.upload_ms", engine::debug::elapsed_ms(t0, Clock::now()));
    return w;
}

}  // namespace engine::community_models::parakeet_tdt
