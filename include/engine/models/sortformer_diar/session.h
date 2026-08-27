#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/sortformer_diar/assets.h"
#include "engine/models/sortformer_diar/graph.h"
#include "engine/models/sortformer_diar/streaming.h"
#include "engine/models/sortformer_diar/types.h"

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace engine::models::sortformer_diar {

std::shared_ptr<runtime::IVoiceModelLoader> make_sortformer_diar_loader();

class SortformerDiarSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    SortformerDiarSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SortformerAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~SortformerDiarSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    // The per-frame speaker-activity probabilities of the last run(),
    // row-major [last_probability_frames(), num_speakers]: the model's actual
    // output, before the speaker-turn post-processing. Diagnostic surface for
    // parity tools (transcribe.cpp dumped the same tensor as diar.probs); the
    // product reads TaskResult::speaker_turns.
    const std::vector<float> & last_probabilities() const noexcept { return probabilities_; }
    int64_t last_probability_frames() const noexcept { return probability_frames_; }
    // How the last run() executed (whole-window or chunked, and why).
    const SortformerRunPlan & last_run_plan() const noexcept { return last_plan_; }

private:
    runtime::MappedGraphCapacityAdapter make_graph_capacity_adapter();
    int64_t base_graph_capacity_samples() const;
    std::vector<int64_t> prepared_graph_capacities() const;
    void prepare_graph_capacity(int64_t capacity);
    runtime::TaskResult run_offline_diarization(
        const runtime::AudioBuffer & audio,
        const SortformerPostprocessConfig & config,
        SortformerRunTimings & timings);
    runtime::TaskResult run_chunked_diarization(
        const runtime::AudioBuffer & audio,
        const SortformerPostprocessConfig & config,
        const SortformerStreamParams & params,
        SortformerRunTimings & timings);
    SortformerPreEncodeGraph & pre_encode_graph_for(int64_t window_frames);
    SortformerBodyGraph & body_graph_for(int64_t frames);
    runtime::TaskResult finish_result(
        const std::vector<float> & probabilities,
        int64_t frames,
        const SortformerPostprocessConfig & config,
        SortformerRunTimings & timings);

    runtime::TaskSpec task_;
    std::shared_ptr<const SortformerAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<const SortformerDiarWeights> weights_;
    SortformerPostprocessConfig default_postprocess_;
    size_t graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    assets::TensorStorageType matmul_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType conv_weight_storage_type_ = assets::TensorStorageType::Native;
    runtime::GraphCapacityController graph_capacity_controller_;
    SortformerFixedContextContract base_context_;
    std::unordered_map<int64_t, SortformerFixedContextContract> prepared_contexts_;
    std::unordered_map<int64_t, std::unique_ptr<SortformerInferenceGraph>> inference_graphs_;
    // Chunked-path graphs, keyed by capacity tier; a small LRU keeps the
    // steady-state geometry resident and bounds memory during the FIFO ramp-up.
    std::map<int64_t, std::unique_ptr<SortformerPreEncodeGraph>> pre_encode_graphs_;
    std::list<int64_t> pre_encode_lru_;
    std::map<int64_t, std::unique_ptr<SortformerBodyGraph>> body_graphs_;
    std::list<int64_t> body_lru_;
    std::vector<float> probabilities_;
    int64_t probability_frames_ = 0;
    SortformerRunPlan last_plan_;
};

}  // namespace engine::models::sortformer_diar
