#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::sortformer_diar {

// Two published package layouts reach this family:
//
//   HuggingFace  audio.cpp GGUFs / safetensors converted from the HF
//                transformers port: config.json + processor_config.json
//                sidecars, tensors named fc_encoder.* / tf_encoder.* /
//                sortformer_modules.*. The v1 packages.
//   NemoGguf     the GGUF NVIDIA publishes for the streaming checkpoints
//                (nvidia/diar_streaming_sortformer_4spk-v2): no sidecars,
//                hparams in sortformer.* KVs, tensors named encoder.* /
//                transformer.* / head.*, and the streaming operating point
//                + AOSC scoring constants shipped as KVs. The catalogue's
//                default package. Before Phase 10.5 nothing in this repo
//                could open it - the transcribe.cpp arch read a third naming
//                (its own converter's stt.sortformer.* / enc.blocks.*).
enum class SortformerPackageLayout {
    HuggingFace,
    NemoGguf,
};

enum class SortformerFeatureNormalize {
    None,
    PerFeature,
};

// How many mel frames an utterance of n samples yields. The HF feature
// extractor reports floor(n / hop); NeMo's AudioToMelSpectrogramPreprocessor
// reports ceil(n / hop) (the transcribe.cpp frontend's nemo_seq_len_ceil).
// They differ by one frame exactly when n is a multiple of hop, which the
// streaming chunk geometry then sees.
enum class SortformerFrameCount {
    Floor,
    Ceil,
};

struct SortformerFeatureExtractorConfig {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t win_length = 0;
    int64_t hop_length = 0;
    int64_t num_mel_bins = 0;
    float preemphasis = 0.0f;
    bool return_attention_mask = true;
    // Layout-dependent frontend contract (see SortformerPackageLayout).
    SortformerFeatureNormalize normalize = SortformerFeatureNormalize::PerFeature;
    bool peak_normalize = true;
    SortformerFrameCount frame_count = SortformerFrameCount::Floor;
};

struct BatchNorm1dEvalWeights {
    core::TensorValue scale;
    core::TensorValue bias;
};

struct SortformerSubsamplingWeights {
    modules::Conv2dWeights conv0;
    core::TensorValue depthwise1_weight;
    core::TensorValue depthwise1_bias;
    modules::Conv2dWeights pointwise1;
    core::TensorValue depthwise2_weight;
    core::TensorValue depthwise2_bias;
    modules::Conv2dWeights pointwise2;
    modules::LinearWeights linear;
};

struct SortformerConformerLayerWeights {
    modules::NormWeights norm_feed_forward1;
    modules::NormWeights norm_self_att;
    modules::NormWeights norm_conv;
    modules::NormWeights norm_feed_forward2;
    modules::NormWeights norm_out;

    modules::LinearWeights ff1_linear1;
    modules::LinearWeights ff1_linear2;
    modules::LinearWeights ff2_linear1;
    modules::LinearWeights ff2_linear2;

    modules::RelativeAttentionWeights self_attn;

    modules::LinearWeights conv_pointwise_conv1;
    modules::DepthwiseConv1dWeights conv_depthwise_conv;
    BatchNorm1dEvalWeights conv_norm;
    modules::LinearWeights conv_pointwise_conv2;
};

struct SortformerTransformerLayerWeights {
    modules::NormWeights self_attn_layer_norm;
    modules::LinearWeights self_attn_q_proj;
    modules::LinearWeights self_attn_k_proj;
    modules::LinearWeights self_attn_v_proj;
    modules::LinearWeights self_attn_out_proj;
    modules::NormWeights final_layer_norm;
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
};

struct SortformerHeadWeights {
    modules::LinearWeights encoder_proj;
    modules::LinearWeights first_hidden_to_hidden;
    modules::LinearWeights single_hidden_to_spks;
};

struct SortformerFastConformerConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_key_value_heads = 0;
    int64_t num_mel_bins = 0;
    int64_t max_position_embeddings = 0;
    int64_t conv_kernel_size = 0;
    int64_t subsampling_factor = 0;
    int64_t subsampling_conv_channels = 0;
    int64_t subsampling_conv_kernel_size = 0;
    int64_t subsampling_conv_stride = 0;
    bool attention_bias = false;
    bool scale_input = false;
    std::string hidden_act;
};

struct SortformerTransformerConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t max_source_positions = 0;
    float layer_norm_eps = 1.0e-5f;
    std::string activation_function;
};

struct SortformerModulesConfig {
    int64_t num_speakers = 0;
    int64_t fc_d_model = 0;
    int64_t tf_d_model = 0;
    int64_t subsampling_factor = 0;
    float dropout_rate = 0.0f;
};

// The streaming operating point a checkpoint ships with (NeMo's
// sortformer_modules streaming cfg). All lengths are in diarization frames
// (80 ms at the shipped 8x subsampling). `present` is false for packages
// that publish none (the offline v1 checkpoints): those run whole-window by
// default and only run chunked when a caller names a preset.
struct SortformerStreamingDefaults {
    bool present = false;
    int64_t chunk_len = 188;
    int64_t chunk_left_context = 1;
    int64_t chunk_right_context = 1;
    int64_t fifo_len = 0;
    int64_t spkcache_len = 188;
    int64_t spkcache_update_period = 188;
};

// AOSC speaker-cache compression constants (NeMo SortformerModules). The
// defaults are the values every published checkpoint ships; a NeMo GGUF
// carries them as sortformer.scoring.* KVs.
struct SortformerScoringConfig {
    int64_t spkcache_sil_frames_per_spk = 3;
    float sil_threshold = 0.2f;
    float pred_score_threshold = 0.25f;
    float scores_boost_latest = 0.05f;
    float strong_boost_rate = 0.75f;
    float weak_boost_rate = 1.5f;
    float min_pos_scores_rate = 0.5f;
};

struct SortformerModelConfig {
    std::string model_type;
    std::string variant;
    int64_t num_speakers = 0;
    float pil_weight = 0.0f;
    float ats_weight = 0.0f;
    SortformerFastConformerConfig fc_encoder;
    SortformerTransformerConfig tf_encoder;
    SortformerModulesConfig modules;
    SortformerStreamingDefaults streaming;
    SortformerScoringConfig scoring;
};

struct SortformerDiarWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    SortformerSubsamplingWeights subsampling;
    std::vector<SortformerConformerLayerWeights> conformer_layers;
    std::vector<SortformerTransformerLayerWeights> transformer_layers;
    SortformerHeadWeights head;
};

struct SortformerAssets {
    assets::ResourceBundle resources;
    SortformerPackageLayout layout = SortformerPackageLayout::HuggingFace;
    SortformerModelConfig model_config;
    SortformerFeatureExtractorConfig feature_config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

// True when `model_path` is (or is a directory holding) a GGUF whose
// general.architecture is "sortformer" - the NeMo-layout package. Cheap: reads
// the GGUF header only. Never throws; unreadable paths answer false.
bool is_nemo_sortformer_gguf(const std::filesystem::path & model_path) noexcept;

std::shared_ptr<const SortformerAssets> load_sortformer_assets(const std::filesystem::path & model_root);
SortformerModelConfig parse_sortformer_model_config(const assets::ResourceBundle & resources);
SortformerFeatureExtractorConfig parse_sortformer_feature_config(const assets::ResourceBundle & resources);
std::shared_ptr<SortformerDiarWeights> load_sortformer_diar_weights(
    const SortformerAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::models::sortformer_diar
