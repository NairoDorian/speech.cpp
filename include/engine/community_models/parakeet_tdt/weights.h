#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/community_models/parakeet_tdt/assets.h"

#include <memory>
#include <optional>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct ParakeetSubsamplingLayerWeights {
    engine::core::TensorValue depthwise_weight;
    engine::core::TensorValue depthwise_bias;
    engine::modules::Conv2dWeights pointwise;
};

struct ParakeetSubsamplingWeights {
    engine::modules::Conv2dWeights conv_in;
    std::vector<ParakeetSubsamplingLayerWeights> layers;
    engine::modules::LinearWeights linear;
    // Scale applied to the projection output in the graph. 1 when the
    // xscaling is folded into `linear` (audio.cpp layout) or disabled; the
    // TranscribeGguf layout keeps the GGUF's own (often quantized) weight and
    // scales the activation like the arch (encoder.cpp ggml_scale).
    float output_scale = 1.0f;
};

struct ParakeetEncoderLayerWeights {
    engine::modules::NormWeights norm_ff1;
    engine::modules::LinearWeights ff1_linear1;
    engine::modules::LinearWeights ff1_linear2;
    engine::modules::NormWeights norm_attn;
    // Either qkv_weight (fused, audio.cpp layout) or q/k/v_weight with
    // optional biases (TranscribeGguf: kept separate and native so a
    // quantized GGUF is never requantized).
    engine::modules::AttentionWeights self_attn;
    engine::core::TensorValue pos_weight;
    engine::core::TensorValue pos_bias_u;
    engine::core::TensorValue pos_bias_v;
    engine::modules::NormWeights norm_conv;
    engine::modules::LinearWeights conv_pw1;
    engine::core::TensorValue conv_dw_weight;
    engine::core::TensorValue conv_dw_bias;
    engine::modules::LinearWeights conv_pw2;
    engine::modules::NormWeights norm_ff2;
    engine::modules::LinearWeights ff2_linear1;
    engine::modules::LinearWeights ff2_linear2;
    engine::modules::NormWeights norm_out;
    // Macaron half-step. 1 when folded into ff*_linear2 (audio.cpp layout);
    // 0.5 applied in the graph after linear2 + bias for TranscribeGguf
    // (conformer.cpp macaron_ff_residual).
    float ff_residual_scale = 1.0f;
};

struct ParakeetEncoderWeights {
    ParakeetSubsamplingWeights subsampling;
    std::vector<ParakeetEncoderLayerWeights> layers;
};

struct ParakeetDecoderWeights {
    engine::core::TensorValue embedding;
    std::vector<engine::modules::LSTMCellWeights> lstm_layers;
    engine::modules::LinearWeights decoder_projector;
    engine::modules::LinearWeights joint_enc;    // encoder_projector: hidden -> joint hidden
    engine::modules::LinearWeights joint_head;   // joint hidden -> vocab + durations
    // TranscribeGguf only. The arch's host decoder (arch/parakeet/decoder.cpp
    // build_host_decoder_weights) dequantizes every predictor/joint weight to
    // F32 and looks the embedding row up on the host (zeros for the start
    // state); this is that host copy [vocab, hidden]. lstm_layers[i].bias_ih
    // holds the converter's pre-summed bias and bias_hh is unused.
    std::vector<float> embedding_host;
    // TranscribeGguf CTC checkpoints: head.ctc (a 1x1 conv) as a Linear
    // [vocab + 1, d_model], F32 like the arch's host mirror.
    std::optional<engine::modules::LinearWeights> ctc_head;
};

struct ParakeetWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    ParakeetEncoderWeights encoder;
    ParakeetDecoderWeights decoder;
};

std::shared_ptr<const ParakeetWeights> load_parakeet_weights(
    const ParakeetTDTAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::community_models::parakeet_tdt
