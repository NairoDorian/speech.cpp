#include "engine/models/sortformer_diar/graph.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace engine::models::sortformer_diar {

namespace {

constexpr size_t kInferenceGraphNodes = 1048576;

int64_t conv_output_dim(int64_t input, int64_t kernel, int64_t stride, int64_t padding) {
    return (input + 2 * padding - kernel) / stride + 1;
}

modules::RelativeConformerBlockWeights make_relative_block_weights(
    const SortformerConformerLayerWeights & layer) {
    modules::ConformerConvModuleWeights conv;
    conv.norm = layer.norm_conv;
    conv.pointwise_in = layer.conv_pointwise_conv1;
    conv.depthwise = layer.conv_depthwise_conv;
    conv.depthwise_norm.scale = layer.conv_norm.scale;
    conv.depthwise_norm.bias = layer.conv_norm.bias;
    conv.pointwise_out = layer.conv_pointwise_conv2;

    modules::RelativeConformerBlockWeights out;
    out.ffn1_norm = layer.norm_feed_forward1;
    out.ffn1_fc1 = layer.ff1_linear1;
    out.ffn1_fc2 = layer.ff1_linear2;
    out.norm1 = layer.norm_self_att;
    out.self_attention = layer.self_attn;
    out.conv = conv;
    out.norm2 = layer.norm_feed_forward2;
    out.ffn2_fc1 = layer.ff2_linear1;
    out.ffn2_fc2 = layer.ff2_linear2;
    out.final_norm = layer.norm_out;
    return out;
}

SortformerTransformerBlockWeights make_transformer_block_weights(
    const SortformerTransformerLayerWeights & layer) {
    SortformerTransformerBlockWeights out;
    out.self_attention_norm = layer.self_attn_layer_norm;
    out.self_attention.query = layer.self_attn_q_proj;
    out.self_attention.key = layer.self_attn_k_proj;
    out.self_attention.value = layer.self_attn_v_proj;
    out.self_attention.output = layer.self_attn_out_proj;
    out.feed_forward_norm = layer.final_layer_norm;
    out.feed_forward_fc1 = layer.fc1;
    out.feed_forward_fc2 = layer.fc2;
    return out;
}

std::vector<float> build_relative_positional_encoding(
    int64_t hidden_size,
    int64_t frames,
    int64_t max_positions) {
    if (frames > max_positions) {
        throw std::runtime_error("Sortformer relative positional encoding exceeds max_position_embeddings");
    }
    if (hidden_size % 2 != 0) {
        throw std::runtime_error("Sortformer relative positional encoding requires even hidden size");
    }
    const int64_t pos_frames = 2 * frames - 1;
    std::vector<float> values(static_cast<size_t>(pos_frames * hidden_size), 0.0f);
    constexpr double kBase = 10000.0;
    std::vector<double> inv_freq(static_cast<size_t>(hidden_size / 2), 0.0);
    for (int64_t i = 0; i < hidden_size / 2; ++i) {
        const double exponent = static_cast<double>(2 * i) / static_cast<double>(hidden_size);
        inv_freq[static_cast<size_t>(i)] = 1.0 / std::pow(kBase, exponent);
    }
    for (int64_t pos = 0; pos < pos_frames; ++pos) {
        const int64_t position_id = frames - 1 - pos;
        for (int64_t i = 0; i < hidden_size / 2; ++i) {
            const double phase = static_cast<double>(position_id) * inv_freq[static_cast<size_t>(i)];
            const size_t base = static_cast<size_t>(pos * hidden_size + 2 * i);
            values[base] = static_cast<float>(std::sin(phase));
            values[base + 1] = static_cast<float>(std::cos(phase));
        }
    }
    return values;
}

struct StemGeometry {
    int64_t kernel = 0;
    int64_t stride = 0;
    int64_t padding = 0;
    int64_t stage1_frames = 0;
    int64_t stage2_frames = 0;
    int64_t stage3_frames = 0;
    int64_t stage3_features = 0;
};

StemGeometry stem_geometry(const SortformerAssets & assets, int64_t feature_frames) {
    const auto & fc = assets.model_config.fc_encoder;
    StemGeometry g;
    g.kernel = fc.subsampling_conv_kernel_size;
    g.stride = fc.subsampling_conv_stride;
    g.padding = (g.kernel - 1) / 2;
    g.stage1_frames = conv_output_dim(feature_frames, g.kernel, g.stride, g.padding);
    const int64_t stage1_features = conv_output_dim(assets.feature_config.num_mel_bins, g.kernel, g.stride, g.padding);
    g.stage2_frames = conv_output_dim(g.stage1_frames, g.kernel, g.stride, g.padding);
    const int64_t stage2_features = conv_output_dim(stage1_features, g.kernel, g.stride, g.padding);
    g.stage3_frames = conv_output_dim(g.stage2_frames, g.kernel, g.stride, g.padding);
    g.stage3_features = conv_output_dim(stage2_features, g.kernel, g.stride, g.padding);
    return g;
}

ggml_context * init_graph_context(size_t graph_context_bytes, const char * what) {
    ggml_init_params params = {};
    params.mem_size = graph_context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        throw std::runtime_error(std::string("Failed to initialize Sortformer diar ggml context for ") + what);
    }
    return ggml;
}

// The dw_striding subsampling stem: mel [1, 1, feature_frames, n_mels] ->
// embeddings [1, output_frames, hidden]. No input scaling here (see
// build_body); the two masks zero the time frames past the valid length after
// the first and second stride-2 stages.
core::TensorValue build_stem(
    core::ModuleBuildContext & ctx,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    const core::TensorValue & input,
    const core::TensorValue & mask1,
    const core::TensorValue & mask2,
    const StemGeometry & g,
    int64_t output_frames) {
    const auto & fc = assets.model_config.fc_encoder;
    const int stride = static_cast<int>(g.stride);
    const int padding = static_cast<int>(g.padding);

    auto x = modules::Conv2dModule({
        1,
        fc.subsampling_conv_channels,
        g.kernel,
        g.kernel,
        stride,
        stride,
        padding,
        padding,
        1,
        1,
        true,
    }).build(ctx, input, weights.subsampling.conv0);
    x = modules::ReluModule().build(ctx, x);
    x = modules::TimeMask4dModule().build(ctx, x, mask1);
    x = modules::DepthwiseConv2dModule({
        fc.subsampling_conv_channels,
        g.kernel,
        g.kernel,
        stride,
        stride,
        padding,
        padding,
        1,
        1,
        true,
    }).build(ctx, x, {weights.subsampling.depthwise1_weight, weights.subsampling.depthwise1_bias});
    x = modules::Conv2dModule({
        fc.subsampling_conv_channels,
        fc.subsampling_conv_channels,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        true,
    }).build(ctx, x, weights.subsampling.pointwise1);
    x = modules::ReluModule().build(ctx, x);
    x = modules::TimeMask4dModule().build(ctx, x, mask2);
    x = modules::DepthwiseConv2dModule({
        fc.subsampling_conv_channels,
        g.kernel,
        g.kernel,
        stride,
        stride,
        padding,
        padding,
        1,
        1,
        true,
    }).build(ctx, x, {weights.subsampling.depthwise2_weight, weights.subsampling.depthwise2_bias});
    x = modules::Conv2dModule({
        fc.subsampling_conv_channels,
        fc.subsampling_conv_channels,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        true,
    }).build(ctx, x, weights.subsampling.pointwise2);
    x = modules::ReluModule().build(ctx, x);
    x = modules::SliceModule({2, 0, output_frames}).build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    x = modules::ReshapeModule({
        core::TensorShape::from_dims({1, output_frames, fc.subsampling_conv_channels * g.stage3_features}),
    }).build(ctx, x);
    x = modules::LinearModule(
            {fc.subsampling_conv_channels * g.stage3_features, fc.hidden_size, true})
            .build(ctx, x, weights.subsampling.linear);
    return x;
}

// Relative-position inputs shared by every graph that runs the conformer
// stack: the sinusoidal table for `frames` positions plus one projected copy
// per layer, computed once per graph on a side graph.
struct PosEmbSetup {
    core::TensorValue pos_emb;
    std::vector<core::TensorValue> projected;
    std::vector<core::TensorValue> projected_computed;
    ggml_cgraph * projection_graph = nullptr;
};

PosEmbSetup declare_pos_emb(
    core::ModuleBuildContext & ctx,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    int64_t frames) {
    const auto & fc = assets.model_config.fc_encoder;
    PosEmbSetup setup;
    setup.pos_emb = core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 2 * frames - 1, fc.hidden_size}));
    setup.projected.reserve(weights.conformer_layers.size());
    setup.projected_computed.reserve(weights.conformer_layers.size());
    for (const auto & layer : weights.conformer_layers) {
        setup.projected.push_back(
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2 * frames - 1, fc.hidden_size})));
        setup.projected_computed.push_back(
            modules::LinearModule({fc.hidden_size, fc.hidden_size, false}).build(
                ctx,
                setup.pos_emb,
                {layer.self_attn.pos_weight, std::nullopt}));
    }
    return setup;
}

void finish_pos_emb_projection(ggml_context * ggml, PosEmbSetup & setup) {
    setup.projection_graph = ggml_new_graph_custom(ggml, 4096, false);
    for (const auto & projected : setup.projected_computed) {
        ggml_build_forward_expand(setup.projection_graph, projected.tensor);
    }
}

void compute_pos_emb_projection(
    ggml_backend_t backend,
    const SortformerAssets & assets,
    int64_t frames,
    PosEmbSetup & setup) {
    core::write_tensor_f32(
        setup.pos_emb,
        build_relative_positional_encoding(
            assets.model_config.fc_encoder.hidden_size,
            frames,
            assets.model_config.fc_encoder.max_position_embeddings));
    engine::core::compute_backend_graph(backend, setup.projection_graph);
    for (size_t i = 0; i < setup.projected.size(); ++i) {
        ggml_backend_tensor_copy(setup.projected_computed[i].tensor, setup.projected[i].tensor);
    }
}

// Everything after the stem: optional input scaling (NeMo xscaling, applied to
// the full sequence the conformer sees - for the chunked path that is the
// concatenation, not the cached rows) -> conformer stack -> encoder_proj ->
// post-LN transformer -> diarization head -> sigmoid [1, frames, n_spk].
core::TensorValue build_body(
    core::ModuleBuildContext & ctx,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    const core::TensorValue & embeddings,
    const core::TensorValue & keep_mask,
    const core::TensorValue & attention_mask,
    const PosEmbSetup & pos) {
    const auto & fc = assets.model_config.fc_encoder;
    const auto & tf = assets.model_config.tf_encoder;
    const auto & head = assets.model_config.modules;

    auto x = embeddings;
    if (fc.scale_input) {
        x = core::wrap_tensor(ggml_scale(ctx.ggml, x.tensor, std::sqrt(static_cast<float>(fc.hidden_size))), x.shape, GGML_TYPE_F32);
    }

    for (size_t layer_index = 0; layer_index < weights.conformer_layers.size(); ++layer_index) {
        const auto & layer = weights.conformer_layers[layer_index];
        const auto layer_weights = make_relative_block_weights(layer);
        x = modules::RelativeConformerBlockModule({
            fc.hidden_size,
            fc.num_attention_heads,
            fc.intermediate_size,
            fc.conv_kernel_size,
            1.0e-5f,
            true,
            -1,
            -1,
            0,
        }).build(
            ctx,
            x,
            std::nullopt,
            layer_weights,
            attention_mask,
            keep_mask,
            keep_mask,
            pos.projected[layer_index]);
    }

    x = modules::LinearModule({head.fc_d_model, tf.hidden_size, true}).build(ctx, x, weights.head.encoder_proj);

    SortformerTransformerStackWeights transformer_weights;
    transformer_weights.layers.reserve(weights.transformer_layers.size());
    for (const auto & layer : weights.transformer_layers) {
        transformer_weights.layers.push_back(make_transformer_block_weights(layer));
    }
    x = SortformerPostNormTransformerStackModule(
            tf.hidden_size,
            tf.num_attention_heads,
            tf.intermediate_size,
            tf.num_hidden_layers,
            tf.layer_norm_eps)
            .build(ctx, x, attention_mask, transformer_weights);

    x = modules::ReluModule().build(ctx, x);
    x = modules::LinearModule({head.tf_d_model, head.tf_d_model, true}).build(ctx, x, weights.head.first_hidden_to_hidden);
    x = modules::ReluModule().build(ctx, x);
    x = modules::LinearModule({head.tf_d_model, head.num_speakers, true}).build(ctx, x, weights.head.single_hidden_to_spks);
    return modules::SigmoidModule().build(ctx, x);
}

template <typename Graph>
void allocate_and_plan(Graph & graph, const core::ExecutionContext & execution_context, const char * what) {
    graph.buffer = ggml_backend_alloc_ctx_tensors(graph.ggml, graph.backend);
    if (graph.buffer == nullptr) {
        throw std::runtime_error(std::string("Failed to allocate Sortformer diar backend tensors for ") + what);
    }
    if (execution_context.uses_host_graph_plan()) {
        graph.plan = engine::core::create_backend_graph_plan_if_host(graph.backend, graph.graph);
        if (graph.plan == nullptr) {
            throw std::runtime_error(std::string("Failed to create Sortformer diar backend graph plan for ") + what);
        }
    }
}

}  // namespace

SortformerInferenceGraph::~SortformerInferenceGraph() {
    if (plan != nullptr) {
        engine::core::free_backend_graph_plan(backend, plan);
    }
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    if (ggml != nullptr) {
        ggml_free(ggml);
    }
}

SortformerPreEncodeGraph::~SortformerPreEncodeGraph() {
    if (plan != nullptr) {
        engine::core::free_backend_graph_plan(backend, plan);
    }
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    if (ggml != nullptr) {
        ggml_free(ggml);
    }
}

SortformerBodyGraph::~SortformerBodyGraph() {
    if (plan != nullptr) {
        engine::core::free_backend_graph_plan(backend, plan);
    }
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    if (ggml != nullptr) {
        ggml_free(ggml);
    }
}

int64_t sortformer_conv_valid_length(int64_t valid, int64_t kernel, int64_t stride, int64_t padding) {
    if (valid <= 0) {
        return 0;
    }
    const int64_t numerator = valid + 2 * padding - kernel;
    if (numerator < 0) {
        return 0;
    }
    return numerator / stride + 1;
}

int64_t sortformer_subsampled_frames(const SortformerAssets & assets, int64_t feature_frames) {
    const auto & fc = assets.model_config.fc_encoder;
    const int64_t kernel = fc.subsampling_conv_kernel_size;
    const int64_t stride = fc.subsampling_conv_stride;
    const int64_t padding = (kernel - 1) / 2;
    const int64_t valid1 = sortformer_conv_valid_length(feature_frames, kernel, stride, padding);
    const int64_t valid2 = sortformer_conv_valid_length(valid1, kernel, stride, padding);
    return sortformer_conv_valid_length(valid2, kernel, stride, padding);
}

void fill_sortformer_keep_mask(std::vector<int32_t> & mask, int64_t frames, int64_t valid_frames) {
    mask.assign(static_cast<size_t>(frames), 0);
    const int64_t clamped = std::clamp<int64_t>(valid_frames, 0, frames);
    for (int64_t i = 0; i < clamped; ++i) {
        mask[static_cast<size_t>(i)] = 1;
    }
}

void fill_sortformer_transformer_attention_mask(
    std::vector<float> & mask,
    int64_t frames,
    int64_t valid_frames) {
    mask.assign(static_cast<size_t>(frames * frames), 0.0f);
    for (int64_t q = 0; q < frames; ++q) {
        for (int64_t k = valid_frames; k < frames; ++k) {
            mask[static_cast<size_t>(q * frames + k)] = -10000.0f;
        }
    }
}

void ensure_sortformer_inference_graph(
    std::unique_ptr<SortformerInferenceGraph> & graph,
    const core::ExecutionContext & execution_context,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    size_t graph_context_bytes,
    int64_t feature_frames,
    int64_t encoder_frames) {
    if (feature_frames <= 0 || encoder_frames <= 0) {
        throw std::runtime_error("Sortformer diar requires positive feature and encoder frame counts");
    }
    if (graph != nullptr &&
        graph->feature_frames == feature_frames &&
        graph->encoder_frames == encoder_frames &&
        graph->backend == execution_context.backend()) {
        return;
    }

    const StemGeometry g = stem_geometry(assets, feature_frames);

    auto next_graph = std::make_unique<SortformerInferenceGraph>();
    next_graph->feature_frames = feature_frames;
    next_graph->encoder_frames = encoder_frames;
    next_graph->backend = execution_context.backend();
    next_graph->compute_threads = std::max(1, execution_context.config().threads);
    next_graph->ggml = init_graph_context(graph_context_bytes, "the whole-window graph");

    core::ModuleBuildContext ctx = {};
    ctx.ggml = next_graph->ggml;
    ctx.backend_type = execution_context.backend_type();
    ctx.module_instance_name = "sortformer_diar";

    next_graph->input = core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 1, feature_frames, assets.feature_config.num_mel_bins}));
    next_graph->mask1 = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, g.stage1_frames}));
    next_graph->mask2 = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, g.stage2_frames}));
    next_graph->encoder_keep_mask =
        core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, encoder_frames}));
    next_graph->transformer_mask = core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({encoder_frames, encoder_frames}));
    PosEmbSetup pos = declare_pos_emb(ctx, assets, weights, encoder_frames);

    auto x = build_stem(ctx, assets, weights, next_graph->input, next_graph->mask1, next_graph->mask2, g, encoder_frames);
    next_graph->output_probabilities =
        build_body(ctx, assets, weights, x, next_graph->encoder_keep_mask, next_graph->transformer_mask, pos);

    finish_pos_emb_projection(next_graph->ggml, pos);
    next_graph->pos_projection_graph = pos.projection_graph;
    next_graph->graph = ggml_new_graph_custom(next_graph->ggml, kInferenceGraphNodes, false);
    ggml_build_forward_expand(next_graph->graph, next_graph->output_probabilities.tensor);
    allocate_and_plan(*next_graph, execution_context, "the whole-window graph");
    compute_pos_emb_projection(next_graph->backend, assets, encoder_frames, pos);
    next_graph->pos_emb = pos.pos_emb;
    next_graph->projected_pos_emb = std::move(pos.projected);
    next_graph->projected_pos_emb_computed = std::move(pos.projected_computed);

    std::vector<float> tf_mask;
    fill_sortformer_transformer_attention_mask(tf_mask, next_graph->encoder_frames, next_graph->encoder_frames);
    core::write_tensor_f32(next_graph->transformer_mask, tf_mask);

    graph = std::move(next_graph);
}

void ensure_sortformer_pre_encode_graph(
    std::unique_ptr<SortformerPreEncodeGraph> & graph,
    const core::ExecutionContext & execution_context,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    size_t graph_context_bytes,
    int64_t feature_capacity) {
    if (feature_capacity <= 0) {
        throw std::runtime_error("Sortformer diar pre-encode graph requires a positive feature capacity");
    }
    if (graph != nullptr &&
        graph->feature_capacity == feature_capacity &&
        graph->backend == execution_context.backend()) {
        return;
    }
    const StemGeometry g = stem_geometry(assets, feature_capacity);
    if (g.stage3_frames <= 0) {
        throw std::runtime_error("Sortformer diar pre-encode graph capacity is too small for the stem");
    }

    auto next_graph = std::make_unique<SortformerPreEncodeGraph>();
    next_graph->feature_capacity = feature_capacity;
    next_graph->embedding_capacity = g.stage3_frames;
    next_graph->backend = execution_context.backend();
    next_graph->compute_threads = std::max(1, execution_context.config().threads);
    next_graph->ggml = init_graph_context(graph_context_bytes, "the pre-encode graph");

    core::ModuleBuildContext ctx = {};
    ctx.ggml = next_graph->ggml;
    ctx.backend_type = execution_context.backend_type();
    ctx.module_instance_name = "sortformer_diar.pre_encode";

    next_graph->input = core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 1, feature_capacity, assets.feature_config.num_mel_bins}));
    next_graph->mask1 = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, g.stage1_frames}));
    next_graph->mask2 = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, g.stage2_frames}));
    next_graph->output = build_stem(
        ctx, assets, weights, next_graph->input, next_graph->mask1, next_graph->mask2, g, g.stage3_frames);

    next_graph->graph = ggml_new_graph_custom(next_graph->ggml, kInferenceGraphNodes, false);
    ggml_build_forward_expand(next_graph->graph, next_graph->output.tensor);
    allocate_and_plan(*next_graph, execution_context, "the pre-encode graph");
    graph = std::move(next_graph);
}

void ensure_sortformer_body_graph(
    std::unique_ptr<SortformerBodyGraph> & graph,
    const core::ExecutionContext & execution_context,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    size_t graph_context_bytes,
    int64_t frame_capacity) {
    if (frame_capacity <= 0) {
        throw std::runtime_error("Sortformer diar body graph requires a positive frame capacity");
    }
    if (graph != nullptr &&
        graph->frame_capacity == frame_capacity &&
        graph->backend == execution_context.backend()) {
        return;
    }

    auto next_graph = std::make_unique<SortformerBodyGraph>();
    next_graph->frame_capacity = frame_capacity;
    next_graph->backend = execution_context.backend();
    next_graph->compute_threads = std::max(1, execution_context.config().threads);
    next_graph->ggml = init_graph_context(graph_context_bytes, "the body graph");

    core::ModuleBuildContext ctx = {};
    ctx.ggml = next_graph->ggml;
    ctx.backend_type = execution_context.backend_type();
    ctx.module_instance_name = "sortformer_diar.body";

    const auto & fc = assets.model_config.fc_encoder;
    next_graph->input = core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, frame_capacity, fc.hidden_size}));
    next_graph->keep_mask = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, frame_capacity}));
    next_graph->transformer_mask = core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({frame_capacity, frame_capacity}));
    PosEmbSetup pos = declare_pos_emb(ctx, assets, weights, frame_capacity);

    next_graph->output_probabilities = build_body(
        ctx, assets, weights, next_graph->input, next_graph->keep_mask, next_graph->transformer_mask, pos);

    finish_pos_emb_projection(next_graph->ggml, pos);
    next_graph->pos_projection_graph = pos.projection_graph;
    next_graph->graph = ggml_new_graph_custom(next_graph->ggml, kInferenceGraphNodes, false);
    ggml_build_forward_expand(next_graph->graph, next_graph->output_probabilities.tensor);
    allocate_and_plan(*next_graph, execution_context, "the body graph");
    compute_pos_emb_projection(next_graph->backend, assets, frame_capacity, pos);
    next_graph->pos_emb = pos.pos_emb;
    next_graph->projected_pos_emb = std::move(pos.projected);
    next_graph->projected_pos_emb_computed = std::move(pos.projected_computed);
    graph = std::move(next_graph);
}

}  // namespace engine::models::sortformer_diar
