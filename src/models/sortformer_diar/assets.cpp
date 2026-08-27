#include "engine/models/sortformer_diar/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/weight_binding.h"

#include <gguf.h>

#include <cctype>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::sortformer_diar {

namespace {

// Pointwise convs are stored as Conv1d [out, in, 1] by the HF port and as
// plain Linear [out, in] by the NeMo GGUF; both are the same matrix, so read
// whatever shape the package holds and reshape onto the linear.
modules::LinearWeights load_linear_as_shape(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias = true) {
    modules::LinearWeights weights;
    const auto source_shape = source.require_metadata(prefix + ".weight").shape;
    int64_t elements = 1;
    for (const int64_t dim : source_shape) {
        elements *= dim;
    }
    if (elements != out_features * in_features) {
        throw std::runtime_error("Sortformer tensor " + prefix + ".weight has " + std::to_string(elements) +
                                 " elements, expected " + std::to_string(out_features * in_features));
    }
    weights.weight = store.load_tensor_as_shape(
        source,
        prefix + ".weight",
        storage_type,
        source_shape,
        core::TensorShape::from_dims({out_features, in_features}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

BatchNorm1dEvalWeights load_fused_batch_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    const auto weight = source.require_f32(prefix + ".weight", {channels});
    const auto bias = source.require_f32(prefix + ".bias", {channels});
    const auto running_mean = source.require_f32(prefix + ".running_mean", {channels});
    const auto running_var = source.require_f32(prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels), 0.0f);
    std::vector<float> fused_bias(static_cast<size_t>(channels), 0.0f);
    constexpr float eps = 1.0e-5f;
    for (int64_t i = 0; i < channels; ++i) {
        const auto index = static_cast<size_t>(i);
        const float channel_scale = weight[index] / std::sqrt(running_var[index] + eps);
        scale[index] = channel_scale;
        fused_bias[index] = bias[index] - running_mean[index] * channel_scale;
    }
    return {
        store.make_from_f32(core::TensorShape::from_dims({channels}), storage_type, std::move(scale)),
        store.make_from_f32(core::TensorShape::from_dims({channels}), storage_type, std::move(fused_bias)),
    };
}

modules::RelativeAttentionWeights load_relative_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden,
    int64_t heads,
    int64_t head_dim) {
    modules::RelativeAttentionWeights weights;
    weights.attention.q_weight = store.load_tensor(source, prefix + ".q_proj.weight", storage_type, {hidden, hidden});
    weights.attention.q_bias = store.load_f32_tensor(source, prefix + ".q_proj.bias", {hidden});
    weights.attention.k_weight = store.load_tensor(source, prefix + ".k_proj.weight", storage_type, {hidden, hidden});
    weights.attention.k_bias = store.load_f32_tensor(source, prefix + ".k_proj.bias", {hidden});
    weights.attention.v_weight = store.load_tensor(source, prefix + ".v_proj.weight", storage_type, {hidden, hidden});
    weights.attention.v_bias = store.load_f32_tensor(source, prefix + ".v_proj.bias", {hidden});
    weights.attention.out_weight = store.load_tensor(source, prefix + ".o_proj.weight", storage_type, {hidden, hidden});
    weights.attention.out_bias = store.load_f32_tensor(source, prefix + ".o_proj.bias", {hidden});
    weights.pos_weight = store.load_tensor(source, prefix + ".relative_k_proj.weight", storage_type, {hidden, hidden});
    weights.pos_bias_u = store.load_f32_tensor(source, prefix + ".bias_u", {heads, head_dim});
    weights.pos_bias_v = store.load_f32_tensor(source, prefix + ".bias_v", {heads, head_dim});
    return weights;
}

SortformerConformerLayerWeights load_conformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer_index,
    const SortformerModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    SortformerConformerLayerWeights layer;
    const auto & encoder = config.fc_encoder;
    const std::string prefix = "fc_encoder.layers." + std::to_string(layer_index);
    const int64_t hidden = encoder.hidden_size;
    const int64_t intermediate = encoder.intermediate_size;
    const int64_t heads = encoder.num_attention_heads;
    const int64_t head_dim = hidden / heads;
    const int64_t kernel = encoder.conv_kernel_size;

    layer.norm_feed_forward1 = modules::binding::norm_from_source(store, source, prefix + ".norm_feed_forward1", hidden);
    layer.norm_self_att = modules::binding::norm_from_source(store, source, prefix + ".norm_self_att", hidden);
    layer.norm_conv = modules::binding::norm_from_source(store, source, prefix + ".norm_conv", hidden);
    layer.norm_feed_forward2 = modules::binding::norm_from_source(store, source, prefix + ".norm_feed_forward2", hidden);
    layer.norm_out = modules::binding::norm_from_source(store, source, prefix + ".norm_out", hidden);

    layer.ff1_linear1 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward1.linear1", matmul_storage_type, intermediate, hidden, true);
    layer.ff1_linear2 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward1.linear2", matmul_storage_type, hidden, intermediate, true);
    layer.ff2_linear1 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward2.linear1", matmul_storage_type, intermediate, hidden, true);
    layer.ff2_linear2 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward2.linear2", matmul_storage_type, hidden, intermediate, true);

    layer.self_attn = load_relative_attention(
        store,
        source,
        prefix + ".self_attn",
        matmul_storage_type,
        hidden,
        heads,
        head_dim);

    layer.conv_pointwise_conv1 = load_linear_as_shape(
        store,
        source,
        prefix + ".conv.pointwise_conv1",
        conv_storage_type,
        2 * hidden,
        hidden);
    layer.conv_depthwise_conv = modules::binding::depthwise_conv1d_from_source(store, source, prefix + ".conv.depthwise_conv", conv_storage_type, hidden, kernel, true);
    layer.conv_norm = load_fused_batch_norm(store, source, prefix + ".conv.norm", hidden, conv_storage_type);
    layer.conv_pointwise_conv2 = load_linear_as_shape(
        store,
        source,
        prefix + ".conv.pointwise_conv2",
        conv_storage_type,
        hidden,
        hidden);
    return layer;
}

SortformerTransformerLayerWeights load_transformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer_index,
    const SortformerModelConfig & config,
    assets::TensorStorageType storage_type) {
    SortformerTransformerLayerWeights layer;
    const auto & encoder = config.tf_encoder;
    const int64_t hidden = encoder.hidden_size;
    const int64_t intermediate = encoder.intermediate_size;
    const std::string prefix = "tf_encoder.layers." + std::to_string(layer_index);

    layer.self_attn_layer_norm = modules::binding::norm_from_source(store, source, prefix + ".self_attn_layer_norm", hidden);
    layer.self_attn_q_proj = modules::binding::linear_from_source(store, source, prefix + ".self_attn.q_proj", storage_type, hidden, hidden, true);
    layer.self_attn_k_proj.weight =
        store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {hidden, hidden});
    if (source.has_tensor(prefix + ".self_attn.k_proj.bias")) {
        layer.self_attn_k_proj.bias = store.load_f32_tensor(source, prefix + ".self_attn.k_proj.bias", {hidden});
    }
    layer.self_attn_v_proj = modules::binding::linear_from_source(store, source, prefix + ".self_attn.v_proj", storage_type, hidden, hidden, true);
    layer.self_attn_out_proj = modules::binding::linear_from_source(store, source, prefix + ".self_attn.out_proj", storage_type, hidden, hidden, true);
    layer.final_layer_norm = modules::binding::norm_from_source(store, source, prefix + ".final_layer_norm", hidden);
    layer.fc1 = modules::binding::linear_from_source(store, source, prefix + ".fc1", storage_type, intermediate, hidden, true);
    layer.fc2 = modules::binding::linear_from_source(store, source, prefix + ".fc2", storage_type, hidden, intermediate, true);
    return layer;
}

// ---- NeMo-layout GGUF (nvidia/diar_streaming_sortformer_4spk-v2) -----------

struct GgufHandle {
    gguf_context * gguf = nullptr;
    ggml_context * tensors = nullptr;
    ~GgufHandle() {
        if (gguf != nullptr) gguf_free(gguf);
        if (tensors != nullptr) ggml_free(tensors);
    }
};

std::optional<std::filesystem::path> resolve_gguf_path(const std::filesystem::path & model_path) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(model_path, ec)) {
        std::string ext = model_path.extension().string();
        for (auto & c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext != ".gguf") return std::nullopt;
        return model_path;
    }
    if (std::filesystem::is_directory(model_path, ec)) {
        return assets::find_directory_gguf(model_path);
    }
    return std::nullopt;
}

bool open_gguf_metadata(const std::filesystem::path & path, GgufHandle & handle) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &handle.tensors;
    handle.gguf = gguf_init_from_file(path.string().c_str(), params);
    return handle.gguf != nullptr;
}

std::string kv_string(const gguf_context * gguf, const char * key, const std::optional<std::string> & fallback = std::nullopt) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
        if (fallback.has_value()) return *fallback;
        throw std::runtime_error(std::string("Sortformer NeMo GGUF is missing string KV ") + key);
    }
    return gguf_get_val_str(gguf, id);
}

int64_t kv_int(const gguf_context * gguf, const char * key, const std::optional<int64_t> & fallback = std::nullopt) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) {
        if (fallback.has_value()) return *fallback;
        throw std::runtime_error(std::string("Sortformer NeMo GGUF is missing integer KV ") + key);
    }
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(gguf, id);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(gguf, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf, id);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(gguf, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(gguf, id));
        case GGUF_TYPE_INT64: return gguf_get_val_i64(gguf, id);
        default: break;
    }
    throw std::runtime_error(std::string("Sortformer NeMo GGUF KV ") + key + " is not an integer");
}

float kv_float(const gguf_context * gguf, const char * key, const std::optional<float> & fallback = std::nullopt) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) {
        if (fallback.has_value()) return *fallback;
        throw std::runtime_error(std::string("Sortformer NeMo GGUF is missing float KV ") + key);
    }
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf, id);
        case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(gguf, id));
        default: break;
    }
    return static_cast<float>(kv_int(gguf, key));
}

bool kv_bool(const gguf_context * gguf, const char * key, bool fallback) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return fallback;
    if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_BOOL) return gguf_get_val_bool(gguf, id);
    return kv_int(gguf, key) != 0;
}

// The NeMo GGUF names its tensors after NeMo's module tree; the weight loader
// above is written against the HF port's names. Every tensor the loader reads
// has a counterpart, so the package is served through a renamed view rather
// than a second loader. Two tensors are deliberately dropped: the shipped
// relative-position table (encoder.pos_enc.pe - the engine computes it) and
// the mel filterbank (preprocessor.fb - likewise). Returns nullopt for them.
std::optional<std::string> hf_name_for_nemo_tensor(std::string_view nemo) {
    const std::string name(nemo);
    auto starts_with = [&](const char * prefix) { return name.rfind(prefix, 0) == 0; };
    auto replace_prefix = [&](const char * prefix, const char * with) {
        return std::string(with) + name.substr(std::strlen(prefix));
    };

    if (starts_with("encoder.pre_encode.conv.")) {
        return replace_prefix("encoder.pre_encode.conv.", "fc_encoder.subsampling.layers.");
    }
    if (starts_with("encoder.pre_encode.out.")) {
        return replace_prefix("encoder.pre_encode.out.", "fc_encoder.subsampling.linear.");
    }
    if (starts_with("encoder.layers.")) {
        std::string rest = name.substr(std::strlen("encoder.layers."));
        const auto dot = rest.find('.');
        if (dot == std::string::npos) return std::nullopt;
        const std::string layer = rest.substr(0, dot);
        std::string suffix = rest.substr(dot + 1);
        static const std::pair<const char *, const char *> attention_map[] = {
            {"self_attn.linear_q.", "self_attn.q_proj."},
            {"self_attn.linear_k.", "self_attn.k_proj."},
            {"self_attn.linear_v.", "self_attn.v_proj."},
            {"self_attn.linear_out.", "self_attn.o_proj."},
            {"self_attn.linear_pos.", "self_attn.relative_k_proj."},
            {"conv.batch_norm.", "conv.norm."},
        };
        for (const auto & [from, to] : attention_map) {
            if (suffix.rfind(from, 0) == 0) {
                suffix = std::string(to) + suffix.substr(std::strlen(from));
                break;
            }
        }
        if (suffix == "self_attn.pos_bias_u") suffix = "self_attn.bias_u";
        if (suffix == "self_attn.pos_bias_v") suffix = "self_attn.bias_v";
        return "fc_encoder.layers." + layer + "." + suffix;
    }
    if (starts_with("transformer.layers.")) {
        std::string rest = name.substr(std::strlen("transformer.layers."));
        const auto dot = rest.find('.');
        if (dot == std::string::npos) return std::nullopt;
        const std::string layer = rest.substr(0, dot);
        std::string suffix = rest.substr(dot + 1);
        static const std::pair<const char *, const char *> transformer_map[] = {
            {"layer_norm_1.", "self_attn_layer_norm."},
            {"layer_norm_2.", "final_layer_norm."},
            {"first_sub_layer.query_net.", "self_attn.q_proj."},
            {"first_sub_layer.key_net.", "self_attn.k_proj."},
            {"first_sub_layer.value_net.", "self_attn.v_proj."},
            {"first_sub_layer.out_projection.", "self_attn.out_proj."},
            {"second_sub_layer.dense_in.", "fc1."},
            {"second_sub_layer.dense_out.", "fc2."},
        };
        for (const auto & [from, to] : transformer_map) {
            if (suffix.rfind(from, 0) == 0) {
                suffix = std::string(to) + suffix.substr(std::strlen(from));
                return "tf_encoder.layers." + layer + "." + suffix;
            }
        }
        return std::nullopt;
    }
    if (starts_with("encoder_proj.")) {
        return replace_prefix("encoder_proj.", "sortformer_modules.encoder_proj.");
    }
    if (starts_with("head.first_hidden_to_hidden.")) {
        return replace_prefix("head.first_hidden_to_hidden.", "sortformer_modules.first_hidden_to_hidden.");
    }
    if (starts_with("head.single_hidden_to_spks.")) {
        return replace_prefix("head.single_hidden_to_spks.", "sortformer_modules.single_hidden_to_spks.");
    }
    // encoder.pos_enc.pe, preprocessor.fb, and anything this loader has no
    // slot for (e.g. a 2*hidden streaming head).
    return std::nullopt;
}

void read_nemo_model_config(const gguf_context * gguf, SortformerModelConfig & config,
                            SortformerFeatureExtractorConfig & feature) {
    config.model_type = "sortformer";
    config.variant = kv_string(gguf, "general.name", std::string("SortformerStreaming"));
    config.num_speakers = kv_int(gguf, "sortformer.num_speakers");
    config.pil_weight = 0.0f;
    config.ats_weight = 0.0f;

    auto & fc = config.fc_encoder;
    fc.hidden_size = kv_int(gguf, "sortformer.encoder.d_model");
    fc.intermediate_size = kv_int(gguf, "sortformer.encoder.d_ff");
    fc.num_attention_heads = kv_int(gguf, "sortformer.encoder.n_heads");
    fc.num_hidden_layers = kv_int(gguf, "sortformer.encoder.n_layers");
    fc.num_key_value_heads = fc.num_attention_heads;
    fc.num_mel_bins = kv_int(gguf, "sortformer.encoder.feat_in");
    fc.max_position_embeddings = kv_int(gguf, "sortformer.encoder.pos_emb_max_len");
    fc.conv_kernel_size = kv_int(gguf, "sortformer.encoder.conv_kernel_size");
    fc.subsampling_factor = kv_int(gguf, "sortformer.encoder.subsampling_factor");
    fc.subsampling_conv_channels = kv_int(gguf, "sortformer.encoder.subsampling_conv_channels");
    // NeMo dw_striding: 3x3 kernels, stride 2, log2(subsampling) stages.
    fc.subsampling_conv_kernel_size = kv_int(gguf, "sortformer.encoder.subsampling_conv_kernel_size", int64_t{3});
    fc.subsampling_conv_stride = kv_int(gguf, "sortformer.encoder.subsampling_conv_stride", int64_t{2});
    fc.attention_bias = kv_bool(gguf, "sortformer.encoder.use_bias", true);
    fc.scale_input = kv_bool(gguf, "sortformer.encoder.xscaling", true);
    fc.hidden_act = "silu";
    if (fc.subsampling_factor != 8) {
        throw std::runtime_error("Sortformer NeMo GGUF: only the 8x dw_striding subsampling is supported, got " +
                                 std::to_string(fc.subsampling_factor));
    }
    const std::string conv_norm = kv_string(gguf, "sortformer.encoder.conv_norm", std::string("batch_norm"));
    if (conv_norm != "batch_norm") {
        throw std::runtime_error("Sortformer NeMo GGUF: conv_norm '" + conv_norm + "' is not supported (batch_norm only)");
    }

    auto & tf = config.tf_encoder;
    tf.hidden_size = kv_int(gguf, "sortformer.transformer.hidden_size");
    tf.intermediate_size = kv_int(gguf, "sortformer.transformer.inner_size");
    tf.num_attention_heads = kv_int(gguf, "sortformer.transformer.n_heads");
    tf.num_hidden_layers = kv_int(gguf, "sortformer.transformer.n_layers");
    // The post-LN transformer carries no positional encoding, so its only
    // length limit is the conformer's relative-position table.
    tf.max_source_positions = fc.max_position_embeddings;
    tf.layer_norm_eps = 1.0e-5f;
    tf.activation_function = "relu";
    if (kv_bool(gguf, "sortformer.transformer.pre_ln", false)) {
        throw std::runtime_error("Sortformer NeMo GGUF: pre_ln=true transformer is not supported (post-LN only)");
    }

    auto & modules = config.modules;
    modules.num_speakers = config.num_speakers;
    modules.fc_d_model = fc.hidden_size;
    modules.tf_d_model = tf.hidden_size;
    modules.subsampling_factor = fc.subsampling_factor;
    modules.dropout_rate = 0.0f;

    auto & streaming = config.streaming;
    if (gguf_find_key(gguf, "sortformer.streaming.chunk_len") >= 0) {
        streaming.present = true;
        streaming.chunk_len = kv_int(gguf, "sortformer.streaming.chunk_len");
        streaming.spkcache_len = kv_int(gguf, "sortformer.streaming.spkcache_len", streaming.spkcache_len);
        streaming.fifo_len = kv_int(gguf, "sortformer.streaming.fifo_len", streaming.fifo_len);
        streaming.spkcache_update_period =
            kv_int(gguf, "sortformer.streaming.spkcache_update_period", streaming.spkcache_update_period);
        streaming.chunk_left_context = kv_int(gguf, "sortformer.streaming.chunk_left_context", streaming.chunk_left_context);
        streaming.chunk_right_context =
            kv_int(gguf, "sortformer.streaming.chunk_right_context", streaming.chunk_right_context);
    }

    auto & scoring = config.scoring;
    scoring.spkcache_sil_frames_per_spk =
        kv_int(gguf, "sortformer.scoring.spkcache_sil_frames_per_spk", scoring.spkcache_sil_frames_per_spk);
    scoring.sil_threshold = kv_float(gguf, "sortformer.scoring.sil_threshold", scoring.sil_threshold);
    scoring.pred_score_threshold = kv_float(gguf, "sortformer.scoring.pred_score_threshold", scoring.pred_score_threshold);
    scoring.scores_boost_latest = kv_float(gguf, "sortformer.scoring.scores_boost_latest", scoring.scores_boost_latest);
    scoring.strong_boost_rate = kv_float(gguf, "sortformer.scoring.strong_boost_rate", scoring.strong_boost_rate);
    scoring.weak_boost_rate = kv_float(gguf, "sortformer.scoring.weak_boost_rate", scoring.weak_boost_rate);
    scoring.min_pos_scores_rate = kv_float(gguf, "sortformer.scoring.min_pos_scores_rate", scoring.min_pos_scores_rate);

    // NeMo AudioToMelSpectrogramPreprocessor. window_size / window_stride are
    // seconds; normalize "NA" means no per-feature normalization; dither is a
    // training-time noise term and is not applied at inference; the amplitude
    // is never peak-normalized.
    feature.sample_rate = kv_int(gguf, "sortformer.preprocessor.sample_rate");
    feature.n_fft = kv_int(gguf, "sortformer.preprocessor.n_fft");
    feature.win_length = static_cast<int64_t>(std::llround(
        static_cast<double>(kv_float(gguf, "sortformer.preprocessor.window_size")) * static_cast<double>(feature.sample_rate)));
    feature.hop_length = static_cast<int64_t>(std::llround(
        static_cast<double>(kv_float(gguf, "sortformer.preprocessor.window_stride")) * static_cast<double>(feature.sample_rate)));
    feature.num_mel_bins = kv_int(gguf, "sortformer.preprocessor.features");
    feature.preemphasis = kv_float(gguf, "sortformer.preprocessor.preemph", 0.0f);
    feature.return_attention_mask = true;
    const std::string normalize = kv_string(gguf, "sortformer.preprocessor.normalize", std::string("NA"));
    if (normalize == "NA" || normalize == "none" || normalize.empty()) {
        feature.normalize = SortformerFeatureNormalize::None;
    } else if (normalize == "per_feature") {
        feature.normalize = SortformerFeatureNormalize::PerFeature;
    } else {
        throw std::runtime_error("Sortformer NeMo GGUF: preprocessor normalize '" + normalize + "' is not supported");
    }
    feature.peak_normalize = false;
    feature.frame_count = SortformerFrameCount::Ceil;
    if (feature.num_mel_bins != fc.num_mel_bins) {
        throw std::runtime_error("Sortformer NeMo GGUF: preprocessor features != encoder feat_in");
    }
}

std::shared_ptr<const SortformerAssets> load_nemo_sortformer_assets(const std::filesystem::path & gguf_path) {
    GgufHandle handle;
    if (!open_gguf_metadata(gguf_path, handle)) {
        throw std::runtime_error("failed to read Sortformer NeMo GGUF: " + gguf_path.string());
    }
    auto assets = std::make_shared<SortformerAssets>();
    assets->layout = SortformerPackageLayout::NemoGguf;
    assets->resources = assets::ResourceBundle(gguf_path.parent_path());
    assets->resources.add_tensor_source("weights", gguf_path);
    read_nemo_model_config(handle.gguf, assets->model_config, assets->feature_config);
    assets->model_weights = assets::make_renamed_tensor_source(
        assets->resources.open_tensor_source("weights"),
        hf_name_for_nemo_tensor);
    return assets;
}

}  // namespace

bool is_nemo_sortformer_gguf(const std::filesystem::path & model_path) noexcept {
    try {
        const auto gguf_path = resolve_gguf_path(model_path);
        if (!gguf_path.has_value()) return false;
        GgufHandle handle;
        if (!open_gguf_metadata(*gguf_path, handle)) return false;
        const int64_t id = gguf_find_key(handle.gguf, "general.architecture");
        if (id < 0 || gguf_get_kv_type(handle.gguf, id) != GGUF_TYPE_STRING) return false;
        return std::strcmp(gguf_get_val_str(handle.gguf, id), "sortformer") == 0;
    } catch (...) {
        return false;
    }
}

std::shared_ptr<const SortformerAssets> load_sortformer_assets(const std::filesystem::path & model_root) {
    if (const auto gguf_path = resolve_gguf_path(model_root); gguf_path.has_value() && is_nemo_sortformer_gguf(*gguf_path)) {
        return load_nemo_sortformer_assets(*gguf_path);
    }
    auto assets = std::make_shared<SortformerAssets>();
    assets->layout = SortformerPackageLayout::HuggingFace;
    assets->resources = engine::model_spec::load_resource_bundle(
        model_root,
        engine::model_spec::default_spec_path("sortformer_diar"));
    assets->model_config = parse_sortformer_model_config(assets->resources);
    assets->feature_config = parse_sortformer_feature_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("weights");
    return assets;
}

SortformerModelConfig parse_sortformer_model_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    SortformerModelConfig config;
    config.model_type = root.require("model_type").as_string();
    config.variant = root.find("architectures") != nullptr && !root.require("architectures").as_array().empty()
        ? root.require("architectures").as_array().front().as_string()
        : "SortformerOffline";
    config.num_speakers = root.require("num_speakers").as_i64();
    config.pil_weight = root.require("pil_weight").as_f32();
    config.ats_weight = root.require("ats_weight").as_f32();

    const auto & fc = root.require("fc_encoder_config");
    config.fc_encoder.hidden_size = fc.require("hidden_size").as_i64();
    config.fc_encoder.intermediate_size = fc.require("intermediate_size").as_i64();
    config.fc_encoder.num_attention_heads = fc.require("num_attention_heads").as_i64();
    config.fc_encoder.num_hidden_layers = fc.require("num_hidden_layers").as_i64();
    config.fc_encoder.num_key_value_heads = fc.require("num_key_value_heads").as_i64();
    config.fc_encoder.num_mel_bins = fc.require("num_mel_bins").as_i64();
    config.fc_encoder.max_position_embeddings = fc.require("max_position_embeddings").as_i64();
    config.fc_encoder.conv_kernel_size = fc.require("conv_kernel_size").as_i64();
    config.fc_encoder.subsampling_factor = fc.require("subsampling_factor").as_i64();
    config.fc_encoder.subsampling_conv_channels = fc.require("subsampling_conv_channels").as_i64();
    config.fc_encoder.subsampling_conv_kernel_size = fc.require("subsampling_conv_kernel_size").as_i64();
    config.fc_encoder.subsampling_conv_stride = fc.require("subsampling_conv_stride").as_i64();
    config.fc_encoder.attention_bias = fc.require("attention_bias").as_bool();
    config.fc_encoder.scale_input = fc.require("scale_input").as_bool();
    config.fc_encoder.hidden_act = fc.require("hidden_act").as_string();

    const auto & tf = root.require("tf_encoder_config");
    config.tf_encoder.hidden_size = tf.require("d_model").as_i64();
    config.tf_encoder.intermediate_size = tf.require("encoder_ffn_dim").as_i64();
    config.tf_encoder.num_attention_heads = tf.require("encoder_attention_heads").as_i64();
    config.tf_encoder.num_hidden_layers = tf.require("encoder_layers").as_i64();
    config.tf_encoder.max_source_positions = tf.require("max_source_positions").as_i64();
    config.tf_encoder.layer_norm_eps = tf.require("layer_norm_eps").as_f32();
    config.tf_encoder.activation_function = tf.require("activation_function").as_string();

    const auto & modules = root.require("modules_config");
    config.modules.num_speakers = modules.require("num_speakers").as_i64();
    config.modules.fc_d_model = modules.require("fc_d_model").as_i64();
    config.modules.tf_d_model = modules.require("tf_d_model").as_i64();
    config.modules.subsampling_factor = modules.require("subsampling_factor").as_i64();
    config.modules.dropout_rate = modules.require("dropout_rate").as_f32();

    // The HF port publishes the offline checkpoints; it carries no streaming
    // operating point and no AOSC scoring constants. The engine's defaults
    // (NeMo's) stay in place, and `streaming.present` stays false so the
    // whole-window forward remains the default product path for them.
    return config;
}

SortformerFeatureExtractorConfig parse_sortformer_feature_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("processor");
    const auto & feature = root.require("feature_extractor");
    SortformerFeatureExtractorConfig config;
    config.sample_rate = feature.require("sampling_rate").as_i64();
    config.n_fft = feature.require("n_fft").as_i64();
    config.win_length = feature.require("win_length").as_i64();
    config.hop_length = feature.require("hop_length").as_i64();
    config.num_mel_bins = feature.require("feature_size").as_i64();
    config.preemphasis = feature.require("preemphasis").as_f32();
    config.return_attention_mask = feature.require("return_attention_mask").as_bool();
    config.normalize = SortformerFeatureNormalize::PerFeature;
    config.peak_normalize = true;
    config.frame_count = SortformerFrameCount::Floor;
    return config;
}

std::shared_ptr<SortformerDiarWeights> load_sortformer_diar_weights(
    const SortformerAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("Sortformer tensor source must not be null");
    }
    const auto & tensor_source = *assets.model_weights;
    const auto & config = assets.model_config;
    auto weights = std::make_shared<SortformerDiarWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "sortformer diar weights",
        weight_context_bytes);

    const auto & fc = config.fc_encoder;
    weights->subsampling.conv0 = modules::binding::conv2d_from_source(
        *weights->store,
        tensor_source,
        "fc_encoder.subsampling.layers.0",
        conv_storage_type,
        fc.subsampling_conv_channels,
        1,
        fc.subsampling_conv_kernel_size,
        fc.subsampling_conv_kernel_size,
        true);
    weights->subsampling.depthwise1_weight = weights->store->load_tensor(
        tensor_source,
        "fc_encoder.subsampling.layers.2.weight",
        conv_storage_type,
        {fc.subsampling_conv_channels, 1, fc.subsampling_conv_kernel_size, fc.subsampling_conv_kernel_size});
    weights->subsampling.depthwise1_bias = weights->store->load_f32_tensor(
        tensor_source,
        "fc_encoder.subsampling.layers.2.bias", {fc.subsampling_conv_channels});
    weights->subsampling.pointwise1 = modules::binding::conv2d_from_source(
        *weights->store,
        tensor_source,
        "fc_encoder.subsampling.layers.3",
        conv_storage_type,
        fc.subsampling_conv_channels,
        fc.subsampling_conv_channels,
        1,
        1,
        true);
    weights->subsampling.depthwise2_weight = weights->store->load_tensor(
        tensor_source,
        "fc_encoder.subsampling.layers.5.weight",
        conv_storage_type,
        {fc.subsampling_conv_channels, 1, fc.subsampling_conv_kernel_size, fc.subsampling_conv_kernel_size});
    weights->subsampling.depthwise2_bias = weights->store->load_f32_tensor(
        tensor_source,
        "fc_encoder.subsampling.layers.5.bias", {fc.subsampling_conv_channels});
    weights->subsampling.pointwise2 = modules::binding::conv2d_from_source(
        *weights->store,
        tensor_source,
        "fc_encoder.subsampling.layers.6",
        conv_storage_type,
        fc.subsampling_conv_channels,
        fc.subsampling_conv_channels,
        1,
        1,
        true);
    const int64_t reduced_mels = fc.num_mel_bins / fc.subsampling_factor;
    weights->subsampling.linear = modules::binding::linear_from_source(
        *weights->store,
        tensor_source,
        "fc_encoder.subsampling.linear",
        matmul_storage_type,
        fc.hidden_size,
        fc.subsampling_conv_channels * reduced_mels,
        true);

    weights->conformer_layers.resize(static_cast<size_t>(fc.num_hidden_layers));
    for (int64_t i = 0; i < fc.num_hidden_layers; ++i) {
        weights->conformer_layers[static_cast<size_t>(i)] =
            load_conformer_layer(*weights->store, tensor_source, i, config, matmul_storage_type, conv_storage_type);
    }

    const auto & tf = config.tf_encoder;
    weights->transformer_layers.resize(static_cast<size_t>(tf.num_hidden_layers));
    for (int64_t i = 0; i < tf.num_hidden_layers; ++i) {
        weights->transformer_layers[static_cast<size_t>(i)] =
            load_transformer_layer(*weights->store, tensor_source, i, config, matmul_storage_type);
    }

    weights->head.encoder_proj = modules::binding::linear_from_source(
        *weights->store,
        tensor_source,
        "sortformer_modules.encoder_proj",
        matmul_storage_type,
        config.modules.tf_d_model,
        config.modules.fc_d_model,
        true);
    weights->head.first_hidden_to_hidden = modules::binding::linear_from_source(
        *weights->store,
        tensor_source,
        "sortformer_modules.first_hidden_to_hidden",
        matmul_storage_type,
        config.modules.tf_d_model,
        config.modules.tf_d_model,
        true);
    weights->head.single_hidden_to_spks = modules::binding::linear_from_source(
        *weights->store,
        tensor_source,
        "sortformer_modules.single_hidden_to_spks",
        matmul_storage_type,
        config.modules.num_speakers,
        config.modules.tf_d_model,
        true);
    weights->store->upload();
    return weights;
}

}  // namespace engine::models::sortformer_diar
