#include "engine/models/cohere_asr/model.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/model_spec/package.h"

#include <array>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::cohere_asr {

namespace {

// ---------------------------------------------------------------------------
// The transcribe.cpp GGUF layout (general.architecture == "cohere_asr",
// published under huggingface.co/handy-computer/cohere-transcribe-*-gguf).
// The retired-arch path: the engine package must open the files the
// `cohere` arch in src/runtime/arch/cohere opened. The tensor names below are
// transcribe.cpp scripts/convert-cohere.py's tables read backwards
// (transcribe.cpp name -> HuggingFace safetensors name, which is what
// load_cohere_weights binds); the converter stores no transposes, so the view
// is a pure rename. The head weight is tied to the token embedding and absent
// from the file (load_cohere_weights falls back to the embedding).
// ---------------------------------------------------------------------------

constexpr std::array<std::pair<std::string_view, std::string_view>, 39> kEncoderBlock = {{
    {"norm_ff1.weight", "norm_feed_forward1.weight"},
    {"norm_ff1.bias", "norm_feed_forward1.bias"},
    {"ff1.linear1.weight", "feed_forward1.linear1.weight"},
    {"ff1.linear1.bias", "feed_forward1.linear1.bias"},
    {"ff1.linear2.weight", "feed_forward1.linear2.weight"},
    {"ff1.linear2.bias", "feed_forward1.linear2.bias"},
    {"norm_attn.weight", "norm_self_att.weight"},
    {"norm_attn.bias", "norm_self_att.bias"},
    {"attn.linear_q.weight", "self_attn.linear_q.weight"},
    {"attn.linear_q.bias", "self_attn.linear_q.bias"},
    {"attn.linear_k.weight", "self_attn.linear_k.weight"},
    {"attn.linear_k.bias", "self_attn.linear_k.bias"},
    {"attn.linear_v.weight", "self_attn.linear_v.weight"},
    {"attn.linear_v.bias", "self_attn.linear_v.bias"},
    {"attn.linear_out.weight", "self_attn.linear_out.weight"},
    {"attn.linear_out.bias", "self_attn.linear_out.bias"},
    {"attn.linear_pos.weight", "self_attn.linear_pos.weight"},
    {"attn.pos_bias_u", "self_attn.pos_bias_u"},
    {"attn.pos_bias_v", "self_attn.pos_bias_v"},
    {"norm_conv.weight", "norm_conv.weight"},
    {"norm_conv.bias", "norm_conv.bias"},
    {"conv.pointwise1.weight", "conv.pointwise_conv1.weight"},
    {"conv.pointwise1.bias", "conv.pointwise_conv1.bias"},
    {"conv.depthwise.weight", "conv.depthwise_conv.weight"},
    {"conv.depthwise.bias", "conv.depthwise_conv.bias"},
    {"conv.pointwise2.weight", "conv.pointwise_conv2.weight"},
    {"conv.pointwise2.bias", "conv.pointwise_conv2.bias"},
    {"conv.bn.weight", "conv.batch_norm.weight"},
    {"conv.bn.bias", "conv.batch_norm.bias"},
    {"conv.bn.running_mean", "conv.batch_norm.running_mean"},
    {"conv.bn.running_var", "conv.batch_norm.running_var"},
    {"norm_ff2.weight", "norm_feed_forward2.weight"},
    {"norm_ff2.bias", "norm_feed_forward2.bias"},
    {"ff2.linear1.weight", "feed_forward2.linear1.weight"},
    {"ff2.linear1.bias", "feed_forward2.linear1.bias"},
    {"ff2.linear2.weight", "feed_forward2.linear2.weight"},
    {"ff2.linear2.bias", "feed_forward2.linear2.bias"},
    {"norm_out.weight", "norm_out.weight"},
    {"norm_out.bias", "norm_out.bias"},
}};

constexpr std::array<std::pair<std::string_view, std::string_view>, 26> kDecoderBlock = {{
    {"norm_self.weight", "layer_norm_1.weight"},
    {"norm_self.bias", "layer_norm_1.bias"},
    {"self_attn.q.weight", "first_sub_layer.query_net.weight"},
    {"self_attn.q.bias", "first_sub_layer.query_net.bias"},
    {"self_attn.k.weight", "first_sub_layer.key_net.weight"},
    {"self_attn.k.bias", "first_sub_layer.key_net.bias"},
    {"self_attn.v.weight", "first_sub_layer.value_net.weight"},
    {"self_attn.v.bias", "first_sub_layer.value_net.bias"},
    {"self_attn.out.weight", "first_sub_layer.out_projection.weight"},
    {"self_attn.out.bias", "first_sub_layer.out_projection.bias"},
    {"norm_cross.weight", "layer_norm_2.weight"},
    {"norm_cross.bias", "layer_norm_2.bias"},
    {"cross_attn.q.weight", "second_sub_layer.query_net.weight"},
    {"cross_attn.q.bias", "second_sub_layer.query_net.bias"},
    {"cross_attn.k.weight", "second_sub_layer.key_net.weight"},
    {"cross_attn.k.bias", "second_sub_layer.key_net.bias"},
    {"cross_attn.v.weight", "second_sub_layer.value_net.weight"},
    {"cross_attn.v.bias", "second_sub_layer.value_net.bias"},
    {"cross_attn.out.weight", "second_sub_layer.out_projection.weight"},
    {"cross_attn.out.bias", "second_sub_layer.out_projection.bias"},
    {"norm_ff.weight", "layer_norm_3.weight"},
    {"norm_ff.bias", "layer_norm_3.bias"},
    {"ff.dense_in.weight", "third_sub_layer.dense_in.weight"},
    {"ff.dense_in.bias", "third_sub_layer.dense_in.bias"},
    {"ff.dense_out.weight", "third_sub_layer.dense_out.weight"},
    {"ff.dense_out.bias", "third_sub_layer.dense_out.bias"},
}};

constexpr std::array<std::pair<std::string_view, std::string_view>, 21> kTopLevel = {{
    {"enc.pre_encode.conv.0.weight", "encoder.pre_encode.conv.0.weight"},
    {"enc.pre_encode.conv.0.bias", "encoder.pre_encode.conv.0.bias"},
    {"enc.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.weight"},
    {"enc.pre_encode.conv.2.bias", "encoder.pre_encode.conv.2.bias"},
    {"enc.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.weight"},
    {"enc.pre_encode.conv.3.bias", "encoder.pre_encode.conv.3.bias"},
    {"enc.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.weight"},
    {"enc.pre_encode.conv.5.bias", "encoder.pre_encode.conv.5.bias"},
    {"enc.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.weight"},
    {"enc.pre_encode.conv.6.bias", "encoder.pre_encode.conv.6.bias"},
    {"enc.pre_encode.out.weight", "encoder.pre_encode.out.weight"},
    {"enc.pre_encode.out.bias", "encoder.pre_encode.out.bias"},
    {"enc_dec_proj.weight", "encoder_decoder_proj.weight"},
    {"enc_dec_proj.bias", "encoder_decoder_proj.bias"},
    {"dec.embed.token.weight", "transf_decoder._embedding.token_embedding.weight"},
    {"dec.embed.pos_enc", "transf_decoder._embedding.position_embedding.pos_enc"},
    {"dec.embed.norm.weight", "transf_decoder._embedding.layer_norm.weight"},
    {"dec.embed.norm.bias", "transf_decoder._embedding.layer_norm.bias"},
    {"dec.final_norm.weight", "transf_decoder._decoder.final_layer_norm.weight"},
    {"dec.final_norm.bias", "transf_decoder._decoder.final_layer_norm.bias"},
    {"head.bias", "log_softmax.mlp.layer0.bias"},
}};

std::unordered_map<std::string, std::string> transcribe_to_hf_names(int64_t encoder_layers, int64_t decoder_layers) {
    std::unordered_map<std::string, std::string> out;
    for (const auto & [from, to] : kTopLevel) {
        out.emplace(std::string(from), std::string(to));
    }
    for (int64_t i = 0; i < encoder_layers; ++i) {
        const std::string from_prefix = "enc.blocks." + std::to_string(i) + ".";
        const std::string to_prefix = "encoder.layers." + std::to_string(i) + ".";
        for (const auto & [from, to] : kEncoderBlock) {
            out.emplace(from_prefix + std::string(from), to_prefix + std::string(to));
        }
    }
    for (int64_t i = 0; i < decoder_layers; ++i) {
        const std::string from_prefix = "dec.blocks." + std::to_string(i) + ".";
        const std::string to_prefix = "transf_decoder._decoder.layers." + std::to_string(i) + ".";
        for (const auto & [from, to] : kDecoderBlock) {
            out.emplace(from_prefix + std::string(from), to_prefix + std::string(to));
        }
    }
    return out;
}

void require_value(int64_t got, int64_t want, const char * what) {
    if (got != want) {
        throw std::runtime_error(std::string("Cohere ASR requires the cohere-transcribe-03-2026 architecture (") +
            what + " is " + std::to_string(got) + ", expected " + std::to_string(want) + ")");
    }
}

// The frontend is fixed by the architecture; both layouts build it the same way
// (the official Transformers processor's F32 constants, not the checkpoint's
// BF16 preprocessor buffers or the transcribe.cpp GGUF's frontend.* tensors).
void build_frontend(CohereAssets & out) {
    out.window = audio::get_cached_stft_window({512, 160, 400, true, audio::STFTPadMode::Constant});
    out.filterbank = audio::MelFilterbank().build_sparse({16000, 512, 128, 0.0f, 8000.0f, true});
    audio::NemoMelFrontendConfig frontend_config;
    frontend_config.sample_rate = 16000;
    frontend_config.n_mels = 128;
    frontend_config.stft = {512, 160, 400, true, audio::STFTPadMode::Constant};
    frontend_config.preemphasis = 0.97f;
    frontend_config.dither_stddev = 1.0e-5f;
    frontend_config.dither_method = audio::DitherMethod::BoxMuller16;
    frontend_config.window = audio::MelWindow::FromArgument;
    frontend_config.mel_bank = audio::MelBank::FromArgument;
    frontend_config.norm = audio::MelNorm::PerBinF32;
    frontend_config.layout = audio::MelLayout::FeatureMajor;
    out.frontend = std::make_shared<audio::NemoMelFrontend>(frontend_config, out.window, out.filterbank.dense);
}

std::shared_ptr<const CohereAssets> load_transcribe_gguf_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<CohereAssets>();
    const auto meta = assets::GgufMetadata::open(path);
    const int64_t encoder_layers = meta.require_i32("stt.cohere.encoder.n_layers");
    const int64_t decoder_layers = meta.require_i32("stt.cohere.decoder.n_layers");
    require_value(encoder_layers, 48, "stt.cohere.encoder.n_layers");
    require_value(meta.require_i32("stt.cohere.encoder.d_model"), 1280, "stt.cohere.encoder.d_model");
    require_value(meta.require_i32("stt.cohere.encoder.subsampling_factor"), 8, "stt.cohere.encoder.subsampling_factor");
    require_value(meta.require_i32("stt.cohere.encoder.conv_kernel"), 9, "stt.cohere.encoder.conv_kernel");
    require_value(meta.require_i32("stt.cohere.decoder.hidden_size"), 1024, "stt.cohere.decoder.hidden_size");
    require_value(decoder_layers, 8, "stt.cohere.decoder.n_layers");
    if (!meta.require_bool("stt.cohere.head.tied_weights")) {
        throw std::runtime_error("Cohere ASR transcribe.cpp GGUF: expected a head tied to the token embedding");
    }

    auto names = transcribe_to_hf_names(encoder_layers, decoder_layers);
    out->source = assets::make_renamed_tensor_source(
        assets::open_tensor_source(path),
        [names = std::move(names)](std::string_view name) -> std::optional<std::string> {
            const auto it = names.find(std::string(name));
            if (it == names.end()) {
                return std::nullopt;  // frontend.* buffers: the frontend is rebuilt from constants
            }
            return it->second;
        });

    const auto tokens = meta.find_string_array("tokenizer.ggml.tokens");
    if (!tokens.has_value()) {
        throw std::runtime_error("Cohere ASR transcribe.cpp GGUF has no tokenizer.ggml.tokens");
    }
    const auto scores = meta.find_f32_array("tokenizer.ggml.scores");
    const auto types = meta.find_i32_array("tokenizer.ggml.token_type");
    out->vocabulary.reserve(tokens->size());
    for (size_t i = 0; i < tokens->size(); ++i) {
        tokenizers::SentencePiecePiece piece;
        piece.id = static_cast<int>(i);
        piece.text = (*tokens)[i];
        piece.score = (scores.has_value() && i < scores->size()) ? (*scores)[i] : 0.0F;
        // GGUF token types (llama.cpp: 1 normal, 2 unknown, 3 control,
        // 4 user-defined, 5 unused, 6 byte) are SentencePieceType's values.
        const int32_t type = (types.has_value() && i < types->size()) ? (*types)[i] : 1;
        piece.type = (type >= 1 && type <= 6) ? static_cast<tokenizers::SentencePieceType>(type)
                                              : tokenizers::SentencePieceType::Normal;
        out->vocabulary.push_back(std::move(piece));
    }
    if (out->vocabulary.size() != 16384) {
        throw std::runtime_error("Cohere vocabulary must contain 16384 entries");
    }
    build_frontend(*out);
    return out;
}

}  // namespace

bool looks_like_transcribe_cohere_gguf(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    if (!in.read(magic, 4) || std::string_view(magic, 4) != "GGUF") {
        return false;
    }
    in.close();
    return assets::GgufMetadata::open(path).find_string("general.architecture").value_or("") == "cohere_asr";
}

int32_t CohereAssets::special_token(const std::string & text) const {
    for (const auto & piece : vocabulary) {
        if (piece.text == text) {
            return piece.id;
        }
    }
    throw std::runtime_error("Cohere tokenizer is missing " + text);
}

std::shared_ptr<const CohereAssets> load_cohere_assets(const std::filesystem::path & path) {
    if (looks_like_transcribe_cohere_gguf(path)) {
        return load_transcribe_gguf_assets(path);
    }
    auto out = std::make_shared<CohereAssets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("cohere_asr"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    const auto & encoder = config.require("encoder");
    const auto & decoder = config.require("transf_decoder").require("config_dict");
    if (io::json::require_string(config, "prompt_format") != "cohere_asr" ||
        io::json::require_i64(encoder, "d_model") != 1280 ||
        io::json::require_i64(encoder, "n_layers") != 48 ||
        io::json::require_i64(encoder, "subsampling_factor") != 8 ||
        io::json::require_i64(encoder, "conv_kernel_size") != 9 ||
        io::json::require_i64(decoder, "hidden_size") != 1024 ||
        io::json::require_i64(decoder, "num_layers") != 8 ||
        !io::json::require_bool(decoder, "pre_ln")) {
        throw std::runtime_error("Cohere ASR requires the cohere-transcribe-03-2026 architecture");
    }
    out->vocabulary = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer"));
    if (out->vocabulary.size() != 16384) {
        throw std::runtime_error("Cohere vocabulary must contain 16384 entries");
    }
    build_frontend(*out);
    return out;
}

}  // namespace engine::models::cohere_asr
