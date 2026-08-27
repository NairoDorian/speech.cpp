#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/sortformer_diar/assets.h"
#include "engine/models/sortformer_diar/modules.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

namespace engine::models::sortformer_diar {

// Whole-window forward: mel -> subsampling stem -> conformer stack ->
// encoder_proj -> post-LN transformer -> head, one graph. The offline product
// path for packages without a streaming operating point, and the parity
// reference for the chunked path (a clip that fits in one chunk must give the
// same probabilities either way).
struct SortformerInferenceGraph {
    int64_t feature_frames = 0;
    int64_t encoder_frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * pos_projection_graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int compute_threads = 1;

    core::TensorValue input;
    core::TensorValue mask1;
    core::TensorValue mask2;
    core::TensorValue encoder_keep_mask;
    core::TensorValue pos_emb;
    core::TensorValue transformer_mask;
    std::vector<core::TensorValue> projected_pos_emb;
    std::vector<core::TensorValue> projected_pos_emb_computed;
    core::TensorValue output_probabilities;

    ~SortformerInferenceGraph();
};

// Chunked (AOSC / FIFO) forward, graph A: the subsampling stem over one mel
// window of up to `feature_capacity` frames. Its output is the pre-encode
// embedding the speaker cache and FIFO store - before the encoder's input
// scaling, which NeMo applies to the concatenation, not to the cached rows.
struct SortformerPreEncodeGraph {
    int64_t feature_capacity = 0;
    int64_t embedding_capacity = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int compute_threads = 1;

    core::TensorValue input;
    core::TensorValue mask1;
    core::TensorValue mask2;
    core::TensorValue output;

    ~SortformerPreEncodeGraph();
};

// Chunked forward, graph B: everything after the stem, over the concatenated
// [speaker cache | FIFO | chunk] embeddings of up to `frame_capacity` rows.
// Rows past the valid count are masked out of attention, zeroed before the
// convolutions and ignored on readback, so one graph per capacity tier serves
// every chunk geometry that fits in it.
struct SortformerBodyGraph {
    int64_t frame_capacity = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * pos_projection_graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int compute_threads = 1;

    core::TensorValue input;
    core::TensorValue keep_mask;
    core::TensorValue pos_emb;
    core::TensorValue transformer_mask;
    std::vector<core::TensorValue> projected_pos_emb;
    std::vector<core::TensorValue> projected_pos_emb_computed;
    core::TensorValue output_probabilities;

    ~SortformerBodyGraph();
};

int64_t sortformer_conv_valid_length(int64_t valid, int64_t kernel, int64_t stride, int64_t padding);

// Subsampled frame count the stem yields for `feature_frames` input frames
// (the three stride-2 stages of the dw_striding stem).
int64_t sortformer_subsampled_frames(const SortformerAssets & assets, int64_t feature_frames);

void ensure_sortformer_inference_graph(
    std::unique_ptr<SortformerInferenceGraph> & graph,
    const core::ExecutionContext & execution_context,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    size_t graph_context_bytes,
    int64_t feature_frames,
    int64_t encoder_frames);

void ensure_sortformer_pre_encode_graph(
    std::unique_ptr<SortformerPreEncodeGraph> & graph,
    const core::ExecutionContext & execution_context,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    size_t graph_context_bytes,
    int64_t feature_capacity);

void ensure_sortformer_body_graph(
    std::unique_ptr<SortformerBodyGraph> & graph,
    const core::ExecutionContext & execution_context,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    size_t graph_context_bytes,
    int64_t frame_capacity);

void fill_sortformer_keep_mask(std::vector<int32_t> & mask, int64_t frames, int64_t valid_frames);

void fill_sortformer_transformer_attention_mask(
    std::vector<float> & mask,
    int64_t frames,
    int64_t valid_frames);

}  // namespace engine::models::sortformer_diar
