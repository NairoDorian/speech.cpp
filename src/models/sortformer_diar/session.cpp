#include "engine/models/sortformer_diar/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/sortformer_diar/frontend.h"
#include "engine/models/sortformer_diar/postprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::sortformer_diar {

namespace {
using engine::debug::measure_ms;

constexpr const char * kFamily = "sortformer_diar";
constexpr size_t kDefaultGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultWeightContextBytes = 128ull * 1024ull * 1024ull;

// Chunked-path graphs are built for a capacity tier rather than an exact
// size, so the handful of geometries a run passes through (FIFO ramp-up, the
// steady state, the tail chunk) share a few graphs instead of one each.
constexpr int64_t kCapacityTier = 64;
constexpr size_t kMaxCachedGraphs = 3;

int64_t round_up_to_tier(int64_t value) {
    return ((value + kCapacityTier - 1) / kCapacityTier) * kCapacityTier;
}

int64_t context_sample_capacity(
    const SortformerFixedContextContract & contract,
    const SortformerAssets & assets) {
    return static_cast<int64_t>(
        std::llround(contract.session_len_sec * static_cast<double>(assets.feature_config.sample_rate)));
}

std::shared_ptr<const SortformerAssets> require_assets(std::shared_ptr<const SortformerAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Sortformer diar session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Sortformer diar session requires a model contract");
    }
    return contract;
}

runtime::GraphCapacityMode default_graph_capacity_mode(const core::ExecutionContext & execution_context) {
    return execution_context.uses_host_graph_plan()
        ? runtime::GraphCapacityMode::Tiered
        : runtime::GraphCapacityMode::Fixed;
}

runtime::SessionOptions normalize_session_options(
    runtime::SessionOptions options,
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    options = runtime::apply_option_v1_compatibility(
        std::move(options),
        {
            {"graph_context_mb", "sortformer_diar.graph_arena_mb"},
            {"sortformer_diar.graph_context_mb", "sortformer_diar.graph_arena_mb"},
            {"weight_context_mb", "sortformer_diar.weight_context_mb"},
            {"weight_type", "sortformer_diar.weight_type"},
            {"matmul_weight_type", "sortformer_diar.matmul_weight_type"},
            {"conv_weight_type", "sortformer_diar.conv_weight_type"},
            {"session_len_sec", "sortformer_diar.session_len_sec"},
            {"speaker_threshold", "sortformer_diar.speaker_threshold"},
            {"speaker_min_frames", "sortformer_diar.speaker_min_frames"},
            {"speaker_pad_frames", "sortformer_diar.speaker_pad_frames"},
            {"offline_graph_capacity_mode", "sortformer_diar.graph_capacity_mode"},
            {"graph_capacity_mode", "sortformer_diar.graph_capacity_mode"},
            {"stream_preset", "sortformer_diar.stream_preset"},
            {"stream_chunk_len", "sortformer_diar.stream_chunk_len"},
            {"stream_left_context", "sortformer_diar.stream_left_context"},
            {"stream_right_context", "sortformer_diar.stream_right_context"},
            {"stream_fifo_len", "sortformer_diar.stream_fifo_len"},
            {"stream_spkcache_len", "sortformer_diar.stream_spkcache_len"},
            {"stream_update_period", "sortformer_diar.stream_update_period"},
        },
        "Sortformer diar");
    runtime::validate_spec_backed_session_options(
        options,
        *require_contract(contract),
        kFamily,
        "Sortformer diar");
    return options;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_sortformer_diar_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const SortformerAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<SortformerDiarSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

template <typename Graph>
Graph & touch_cached_graph(
    std::map<int64_t, std::unique_ptr<Graph>> & graphs,
    std::list<int64_t> & lru,
    int64_t capacity,
    const std::function<void(std::unique_ptr<Graph> &)> & build) {
    auto it = graphs.find(capacity);
    if (it == graphs.end()) {
        while (graphs.size() >= kMaxCachedGraphs && !lru.empty()) {
            graphs.erase(lru.back());
            lru.pop_back();
        }
        std::unique_ptr<Graph> graph;
        build(graph);
        it = graphs.emplace(capacity, std::move(graph)).first;
    } else {
        lru.remove(capacity);
    }
    lru.push_front(capacity);
    return *it->second;
}

}  // namespace

SortformerDiarSession::SortformerDiarSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SortformerAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(normalize_session_options(std::move(options), contract)),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      default_postprocess_(parse_sortformer_postprocess_config(RuntimeSessionBase::options())),
      graph_arena_bytes_(
          runtime::parse_size_mb_option(RuntimeSessionBase::options().options, {"sortformer_diar.graph_arena_mb"}, kDefaultGraphArenaBytes)),
      weight_context_bytes_(
          runtime::parse_size_mb_option(RuntimeSessionBase::options().options, {"sortformer_diar.weight_context_mb"}, kDefaultWeightContextBytes)),
      matmul_weight_storage_type_(runtime::parse_tensor_storage_option(
          RuntimeSessionBase::options().options,
          "sortformer_diar.matmul_weight_type",
          "sortformer_diar.weight_type",
          engine::assets::TensorStorageType::F32,
          {
              engine::assets::TensorStorageType::Native,
              engine::assets::TensorStorageType::F32,
              engine::assets::TensorStorageType::F16,
              engine::assets::TensorStorageType::BF16,
              engine::assets::TensorStorageType::Q8_0,
          })),
      conv_weight_storage_type_(runtime::parse_tensor_storage_option(
          RuntimeSessionBase::options().options,
          "sortformer_diar.conv_weight_type",
          "sortformer_diar.weight_type",
          engine::assets::TensorStorageType::F32,
          {
              engine::assets::TensorStorageType::Native,
              engine::assets::TensorStorageType::F32,
              engine::assets::TensorStorageType::F16,
              engine::assets::TensorStorageType::BF16,
              engine::assets::TensorStorageType::Q8_0,
          })) {
    weights_ = load_sortformer_diar_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        matmul_weight_storage_type_,
        conv_weight_storage_type_,
        weight_context_bytes_);
    assets_->model_weights->release_storage();
    const auto graph_capacity_mode = runtime::resolve_graph_capacity_mode(
        RuntimeSessionBase::options(),
        default_graph_capacity_mode(execution_context()),
        {"sortformer_diar.graph_capacity_mode"});
    if (graph_capacity_mode == runtime::GraphCapacityMode::Unsupported) {
        throw std::runtime_error("Sortformer diar graph_capacity_mode=unsupported is not implemented");
    }
    graph_capacity_controller_ = runtime::GraphCapacityController(graph_capacity_mode);
    base_context_ = parse_sortformer_fixed_context_contract(RuntimeSessionBase::options(), *assets_);
    // Validate the session-level streaming options up front, so a typo fails
    // at session creation rather than on the first run.
    (void)resolve_sortformer_run_plan(assets_->model_config, RuntimeSessionBase::options().options);
}

SortformerDiarSession::~SortformerDiarSession() = default;

std::string SortformerDiarSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind SortformerDiarSession::task_kind() const {
    return task_.task;
}

runtime::RunMode SortformerDiarSession::run_mode() const {
    return task_.mode;
}

int64_t SortformerDiarSession::base_graph_capacity_samples() const {
    return context_sample_capacity(base_context_, *assets_);
}

runtime::MappedGraphCapacityAdapter SortformerDiarSession::make_graph_capacity_adapter() {
    return runtime::MappedGraphCapacityAdapter(
        base_graph_capacity_samples(),
        base_graph_capacity_samples(),
        [](int64_t request_size) {
            if (request_size <= 0) {
                throw std::runtime_error("Sortformer graph capacity request size must be positive");
            }
            return request_size;
        },
        [this]() { return prepared_graph_capacities(); },
        [this](int64_t capacity) { prepare_graph_capacity(capacity); });
}

std::vector<int64_t> SortformerDiarSession::prepared_graph_capacities() const {
    std::vector<int64_t> capacities;
    capacities.reserve(inference_graphs_.size());
    for (const auto & [capacity, graph] : inference_graphs_) {
        if (graph) {
            capacities.push_back(capacity);
        }
    }
    return capacities;
}

void SortformerDiarSession::prepare_graph_capacity(int64_t capacity) {
    if (capacity <= 0) {
        throw std::runtime_error("Sortformer graph capacity must be positive");
    }
    if (inference_graphs_.find(capacity) != inference_graphs_.end()) {
        return;
    }
    const SortformerFixedContextContract context =
        capacity == base_graph_capacity_samples()
            ? base_context_
            : make_sortformer_fixed_context_contract_for_samples(capacity, *assets_);
    auto graph = std::make_unique<SortformerInferenceGraph>();
    ensure_sortformer_inference_graph(
        graph,
        execution_context(),
        *assets_,
        *weights_,
        graph_arena_bytes_,
        context.feature_frames,
        context.encoder_frames);
    prepared_contexts_[capacity] = context;
    inference_graphs_[capacity] = std::move(graph);
}

void SortformerDiarSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.audio.has_value()) {
        if (request.audio->sample_rate > 0 && request.audio->sample_rate != assets_->feature_config.sample_rate) {
            throw std::runtime_error("Sortformer diar prepare sample_rate mismatch");
        }
        if (request.audio->channels > 0 && request.audio->channels != 1) {
            throw std::runtime_error("Sortformer diar prepare currently requires mono audio");
        }
    }
    // The whole-window graph is prepared eagerly only when it is the path
    // this session will take; the chunked path builds its graphs per tier on
    // first use and must not be charged a whole-window graph it never runs.
    const auto plan = resolve_sortformer_run_plan(assets_->model_config, RuntimeSessionBase::options().options);
    if (plan.mode == SortformerRunMode::WholeWindow) {
        auto adapter = make_graph_capacity_adapter();
        const int64_t request_size = request.audio.has_value() ? request.audio->max_input_samples : 0;
        graph_capacity_controller_.ensure_prepared(adapter, request_size);
    }
    mark_prepared();
}

runtime::TaskResult SortformerDiarSession::run(const runtime::TaskRequest & request) {
    require_prepared("Sortformer run()");
    runtime::validate_spec_backed_request_options(
        request.options,
        *contract_,
        "Sortformer diar");
    if (task_.task != runtime::VoiceTaskKind::Diarization) {
        throw std::runtime_error("Sortformer diar session only supports --task diar");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Sortformer diar session only supports offline mode");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Sortformer diar offline run requires audio_input");
    }
    auto config = default_postprocess_;
    runtime::SessionOptions merged = options();
    for (const auto & [key, value] : request.options) {
        merged.options[key] = value;
    }
    if (!request.options.empty()) {
        config = parse_sortformer_postprocess_config(merged);
    }
    last_plan_ = resolve_sortformer_run_plan(assets_->model_config, merged.options);
    trace(engine::debug::LogLevel::Info, "sortformer_diar",
          std::string("run plan: ") +
              (last_plan_.mode == SortformerRunMode::Chunked ? "chunked" : "whole-window") +
              " (" + last_plan_.source + ")");

    SortformerRunTimings timings;
    const auto wall_started = std::chrono::steady_clock::now();
    runtime::TaskResult result;
    if (last_plan_.mode == SortformerRunMode::Chunked) {
        result = run_chunked_diarization(*request.audio_input, config, last_plan_.params, timings);
    } else {
        result = run_offline_diarization(*request.audio_input, config, timings);
    }
    const auto wall_ended = std::chrono::steady_clock::now();
    timings.wall_ms = std::chrono::duration<double, std::milli>(wall_ended - wall_started).count();
    emit_sortformer_timings(timings);
    return result;
}

runtime::TaskResult SortformerDiarSession::finish_result(
    const std::vector<float> & probabilities,
    int64_t frames,
    const SortformerPostprocessConfig & config,
    SortformerRunTimings & timings) {
    runtime::TaskResult result;
    const int64_t frame_step_samples =
        assets_->feature_config.hop_length * assets_->model_config.fc_encoder.subsampling_factor;
    timings.postprocess_ms = measure_ms([&]() {
        result.speaker_turns = decode_sortformer_speaker_turns(
            probabilities,
            frames,
            frames,
            assets_->model_config.modules.num_speakers,
            frame_step_samples,
            config);
    });
    return result;
}

runtime::TaskResult SortformerDiarSession::run_offline_diarization(
    const runtime::AudioBuffer & audio,
    const SortformerPostprocessConfig & config,
    SortformerRunTimings & timings) {
    emit_progress("sortformer_diar", 0, 3);

    SortformerFeatureBatch features;
    timings.frontend_ms = measure_ms([&]() {
        features = compute_sortformer_features(
            audio,
            *assets_,
            execution_context().config().threads,
            &timings);
    });

    emit_progress("sortformer_diar", 1, 3);
    const int64_t kernel = assets_->model_config.fc_encoder.subsampling_conv_kernel_size;
    const int64_t stride = assets_->model_config.fc_encoder.subsampling_conv_stride;
    const int64_t padding = (kernel - 1) / 2;
    const int64_t valid1 = sortformer_conv_valid_length(features.valid_frames, kernel, stride, padding);
    const int64_t valid2 = sortformer_conv_valid_length(valid1, kernel, stride, padding);
    const int64_t valid3 = sortformer_conv_valid_length(valid2, kernel, stride, padding);
    probabilities_.clear();
    probability_frames_ = 0;
    if (valid3 <= 0) {
        return runtime::TaskResult{};
    }
    int64_t selected_capacity = 0;
    auto adapter = make_graph_capacity_adapter();
    timings.graph_ensure_ms = measure_ms([&]() {
        selected_capacity = graph_capacity_controller_.ensure_prepared(
            adapter,
            static_cast<int64_t>(audio.samples.size()));
    });
    const auto context_it = prepared_contexts_.find(selected_capacity);
    const auto graph_it = inference_graphs_.find(selected_capacity);
    if (context_it == prepared_contexts_.end() || graph_it == inference_graphs_.end() || !graph_it->second) {
        throw std::runtime_error("Sortformer diar selected graph capacity was not prepared");
    }
    const SortformerFixedContextContract & prepared_context = context_it->second;
    if (features.frames > prepared_context.feature_frames || valid3 > prepared_context.encoder_frames) {
        // Never trim: the whole-window graph has a fixed context, and audio
        // past it is rejected here rather than silently cut (L11). The
        // chunked path (stream_preset) has no such limit.
        throw std::runtime_error(
            "Sortformer diar input exceeds prepared session context of " +
            std::to_string(prepared_context.session_len_sec) + " seconds; raise session_len_sec or run chunked (stream_preset)");
    }

    auto & graph = *graph_it->second;

    timings.graph_prepare_ms = measure_ms([&]() {
        std::vector<float> padded_input(
            static_cast<size_t>(prepared_context.feature_frames * assets_->feature_config.num_mel_bins),
            0.0f);
        std::copy(features.time_major.begin(), features.time_major.end(), padded_input.begin());
        core::write_tensor_f32(graph.input, padded_input);

        std::vector<int32_t> keep_mask;
        fill_sortformer_keep_mask(keep_mask, graph.mask1.shape.dims[1], std::min<int64_t>(graph.mask1.shape.dims[1], valid1));
        core::write_tensor_i32(graph.mask1, keep_mask);
        fill_sortformer_keep_mask(keep_mask, graph.mask2.shape.dims[1], std::min<int64_t>(graph.mask2.shape.dims[1], valid2));
        core::write_tensor_i32(graph.mask2, keep_mask);
        fill_sortformer_keep_mask(keep_mask, graph.encoder_keep_mask.shape.dims[1], valid3);
        core::write_tensor_i32(graph.encoder_keep_mask, keep_mask);
        std::vector<float> tf_mask;
        fill_sortformer_transformer_attention_mask(tf_mask, graph.encoder_frames, valid3);
        core::write_tensor_f32(graph.transformer_mask, tf_mask);
    });

    std::vector<float> padded_probabilities;
    timings.encoder_ms = measure_ms([&]() {
        timings.encoder_compute_ms = measure_ms([&]() {
            core::set_backend_threads(execution_context().backend(), graph.compute_threads);
            const ggml_status status =
                core::compute_backend_graph(execution_context().backend(), graph.graph, graph.plan);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Sortformer diar graph compute failed");
            }
        });
        timings.encoder_readback_ms = measure_ms([&]() {
            core::read_tensor_f32_into(graph.output_probabilities.tensor, padded_probabilities);
        });
    });

    emit_progress("sortformer_diar", 2, 3);
    const int64_t n_spk = assets_->model_config.modules.num_speakers;
    probabilities_.assign(padded_probabilities.begin(),
                          padded_probabilities.begin() + static_cast<std::ptrdiff_t>(valid3 * n_spk));
    probability_frames_ = valid3;
    return finish_result(probabilities_, probability_frames_, config, timings);
}

SortformerPreEncodeGraph & SortformerDiarSession::pre_encode_graph_for(int64_t window_frames) {
    const int64_t capacity = round_up_to_tier(window_frames);
    return touch_cached_graph<SortformerPreEncodeGraph>(
        pre_encode_graphs_, pre_encode_lru_, capacity,
        [&](std::unique_ptr<SortformerPreEncodeGraph> & graph) {
            ensure_sortformer_pre_encode_graph(
                graph, execution_context(), *assets_, *weights_, graph_arena_bytes_, capacity);
        });
}

SortformerBodyGraph & SortformerDiarSession::body_graph_for(int64_t frames) {
    const int64_t capacity = round_up_to_tier(frames);
    return touch_cached_graph<SortformerBodyGraph>(
        body_graphs_, body_lru_, capacity,
        [&](std::unique_ptr<SortformerBodyGraph> & graph) {
            ensure_sortformer_body_graph(
                graph, execution_context(), *assets_, *weights_, graph_arena_bytes_, capacity);
        });
}

runtime::TaskResult SortformerDiarSession::run_chunked_diarization(
    const runtime::AudioBuffer & audio,
    const SortformerPostprocessConfig & config,
    const SortformerStreamParams & params,
    SortformerRunTimings & timings) {
    SortformerFeatureBatch features;
    timings.frontend_ms = measure_ms([&]() {
        features = compute_sortformer_features(
            audio,
            *assets_,
            execution_context().config().threads,
            &timings);
    });

    const auto & fc = assets_->model_config.fc_encoder;
    const int64_t n_mels = assets_->feature_config.num_mel_bins;
    const int64_t n_spk = assets_->model_config.modules.num_speakers;
    const int64_t emb_dim = fc.hidden_size;
    const int64_t sub = fc.subsampling_factor;
    const int64_t kernel = fc.subsampling_conv_kernel_size;
    const int64_t stride = fc.subsampling_conv_stride;
    const int64_t padding = (kernel - 1) / 2;

    probabilities_.clear();
    probability_frames_ = 0;
    if (features.valid_frames <= 0 || sortformer_subsampled_frames(*assets_, features.valid_frames) <= 0) {
        return runtime::TaskResult{};
    }

    std::vector<int32_t> mask;
    std::vector<float> host_buffer;
    std::vector<float> tf_mask;

    // Graph A: the stem over one mel window. Rows past the window are zero
    // and masked, so a tier graph serves every window that fits.
    const SortformerPreEncodeFn pre_encode = [&](const float * mel_window, int64_t window_frames,
                                                 std::vector<float> & embeddings, int64_t & T_diar) {
        auto & graph = pre_encode_graph_for(window_frames);
        const int64_t valid1 = sortformer_conv_valid_length(window_frames, kernel, stride, padding);
        const int64_t valid2 = sortformer_conv_valid_length(valid1, kernel, stride, padding);
        T_diar = sortformer_conv_valid_length(valid2, kernel, stride, padding);
        double prepare_ms = measure_ms([&]() {
            host_buffer.assign(static_cast<size_t>(graph.feature_capacity * n_mels), 0.0f);
            std::copy(mel_window, mel_window + static_cast<std::ptrdiff_t>(window_frames * n_mels), host_buffer.begin());
            core::write_tensor_f32(graph.input, host_buffer);
            fill_sortformer_keep_mask(mask, graph.mask1.shape.dims[1], std::min<int64_t>(graph.mask1.shape.dims[1], valid1));
            core::write_tensor_i32(graph.mask1, mask);
            fill_sortformer_keep_mask(mask, graph.mask2.shape.dims[1], std::min<int64_t>(graph.mask2.shape.dims[1], valid2));
            core::write_tensor_i32(graph.mask2, mask);
        });
        timings.graph_prepare_ms += prepare_ms;
        timings.encoder_compute_ms += measure_ms([&]() {
            core::set_backend_threads(execution_context().backend(), graph.compute_threads);
            const ggml_status status =
                core::compute_backend_graph(execution_context().backend(), graph.graph, graph.plan);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Sortformer diar pre-encode graph compute failed");
            }
        });
        timings.encoder_readback_ms += measure_ms([&]() {
            core::read_tensor_f32_into(graph.output.tensor, host_buffer);
            embeddings.assign(host_buffer.begin(), host_buffer.begin() + static_cast<std::ptrdiff_t>(T_diar * emb_dim));
        });
    };

    // Graph B: the body over [spkcache | fifo | chunk]. Rows past T_concat
    // are zeroed before the convolutions, masked out of attention, and never
    // read back.
    const SortformerInferFn infer = [&](const std::vector<float> & concat, int64_t T_concat, std::vector<float> & preds) {
        auto & graph = body_graph_for(T_concat);
        timings.graph_prepare_ms += measure_ms([&]() {
            host_buffer.assign(static_cast<size_t>(graph.frame_capacity * emb_dim), 0.0f);
            std::copy(concat.begin(), concat.end(), host_buffer.begin());
            core::write_tensor_f32(graph.input, host_buffer);
            fill_sortformer_keep_mask(mask, graph.frame_capacity, T_concat);
            core::write_tensor_i32(graph.keep_mask, mask);
            fill_sortformer_transformer_attention_mask(tf_mask, graph.frame_capacity, T_concat);
            core::write_tensor_f32(graph.transformer_mask, tf_mask);
        });
        timings.encoder_compute_ms += measure_ms([&]() {
            core::set_backend_threads(execution_context().backend(), graph.compute_threads);
            const ggml_status status =
                core::compute_backend_graph(execution_context().backend(), graph.graph, graph.plan);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Sortformer diar body graph compute failed");
            }
        });
        timings.encoder_readback_ms += measure_ms([&]() {
            core::read_tensor_f32_into(graph.output_probabilities.tensor, host_buffer);
            preds.assign(host_buffer.begin(), host_buffer.begin() + static_cast<std::ptrdiff_t>(T_concat * n_spk));
        });
    };

    // One progress unit per chunk; RunControl unwinds the run between chunks.
    const SortformerChunkProgressFn progress = [&](int64_t chunk_index, int64_t n_chunks) {
        emit_progress("sortformer_diar", chunk_index, n_chunks);
    };

    SortformerChunkedResult chunked;
    timings.encoder_ms = measure_ms([&]() {
        chunked = run_sortformer_chunked(
            features, n_mels, sub, n_spk, emb_dim, params, pre_encode, infer, progress);
    });
    probabilities_ = std::move(chunked.probabilities);
    probability_frames_ = chunked.frames;
    return finish_result(probabilities_, probability_frames_, config, timings);
}

namespace {

// The spec-backed loader resolves a package through the family's model spec,
// which the NeMo GGUF layout cannot satisfy (no sidecars, no embedded spec).
// This wrapper answers can_load() for that layout itself and forwards
// everything else; load_sortformer_assets() then picks the loader by layout.
class SortformerDiarLoader final : public runtime::IVoiceModelLoader {
public:
    explicit SortformerDiarLoader(std::shared_ptr<runtime::IVoiceModelLoader> inner) : inner_(std::move(inner)) {}

    std::string family() const override { return inner_->family(); }
    std::vector<std::string> family_aliases() const override { return inner_->family_aliases(); }
    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value()) {
            const auto & hint = *request.family_hint;
            const auto aliases = family_aliases();
            if (hint != family() && std::find(aliases.begin(), aliases.end(), hint) == aliases.end()) {
                return false;
            }
        }
        if (is_nemo_sortformer_gguf(request.model_path)) {
            return true;
        }
        return inner_->can_load(request);
    }
    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        return inner_->inspect(request);
    }
    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return inner_->load(request);
    }
    runtime::CapabilitySet advertised_capabilities() const override { return inner_->advertised_capabilities(); }
    std::string advertised_instructions_policy() const override { return inner_->advertised_instructions_policy(); }
    std::vector<std::string> advertised_api_endpoints() const override { return inner_->advertised_api_endpoints(); }

private:
    std::shared_ptr<runtime::IVoiceModelLoader> inner_;
};

}  // namespace

// Loading adapter: Sortformer diarization uses the schema-v1 spec-backed loader,
// so loader wiring stays beside the session it constructs.
std::shared_ptr<runtime::IVoiceModelLoader> make_sortformer_diar_loader() {
    runtime::SpecBackedVoiceModelConfig<SortformerAssets> config;
    config.family = kFamily;
    config.aliases = {"sortformer", "sortformer-diar"};
    config.load_assets = load_sortformer_assets;
    config.create_session = create_sortformer_diar_session;
    return std::make_shared<SortformerDiarLoader>(runtime::make_spec_backed_voice_loader(std::move(config)));
}

}  // namespace engine::models::sortformer_diar
