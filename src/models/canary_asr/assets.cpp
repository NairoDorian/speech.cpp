#include "engine/models/canary_asr/model.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::canary_asr {

namespace {

// ---------------------------------------------------------------------------
// The transcribe.cpp GGUF layout (general.architecture == "canary",
// published under huggingface.co/handy-computer/canary-*-gguf).
// The retired-arch path: the engine package must open the files the `canary`
// arch in src/runtime/arch/canary opened. The tensor names below are
// transcribe.cpp scripts/convert-canary.py's tables read backwards
// (transcribe.cpp name -> NeMo state-dict name, which is what
// load_canary_weights binds); the converter stores every tensor as the
// state-dict array with no transpose, reshape, split or fusion, so the view is
// a pure rename. The head is untied and present (dec.head.*). The converter
// does not export the preprocessor buffers, so the frontend window and
// filterbank are rebuilt from the NeMo constants below.
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
    {"norm1.weight", "layer_norm_1.weight"},
    {"norm1.bias", "layer_norm_1.bias"},
    {"self_attn.q.weight", "first_sub_layer.query_net.weight"},
    {"self_attn.q.bias", "first_sub_layer.query_net.bias"},
    {"self_attn.k.weight", "first_sub_layer.key_net.weight"},
    {"self_attn.k.bias", "first_sub_layer.key_net.bias"},
    {"self_attn.v.weight", "first_sub_layer.value_net.weight"},
    {"self_attn.v.bias", "first_sub_layer.value_net.bias"},
    {"self_attn.o.weight", "first_sub_layer.out_projection.weight"},
    {"self_attn.o.bias", "first_sub_layer.out_projection.bias"},
    {"norm2.weight", "layer_norm_2.weight"},
    {"norm2.bias", "layer_norm_2.bias"},
    {"cross_attn.q.weight", "second_sub_layer.query_net.weight"},
    {"cross_attn.q.bias", "second_sub_layer.query_net.bias"},
    {"cross_attn.k.weight", "second_sub_layer.key_net.weight"},
    {"cross_attn.k.bias", "second_sub_layer.key_net.bias"},
    {"cross_attn.v.weight", "second_sub_layer.value_net.weight"},
    {"cross_attn.v.bias", "second_sub_layer.value_net.bias"},
    {"cross_attn.o.weight", "second_sub_layer.out_projection.weight"},
    {"cross_attn.o.bias", "second_sub_layer.out_projection.bias"},
    {"norm3.weight", "layer_norm_3.weight"},
    {"norm3.bias", "layer_norm_3.bias"},
    {"ffn.up.weight", "third_sub_layer.dense_in.weight"},
    {"ffn.up.bias", "third_sub_layer.dense_in.bias"},
    {"ffn.down.weight", "third_sub_layer.dense_out.weight"},
    {"ffn.down.bias", "third_sub_layer.dense_out.bias"},
}};

constexpr std::array<std::pair<std::string_view, std::string_view>, 22> kTopLevel = {{
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
    {"enc.proj.weight", "encoder_decoder_proj.weight"},
    {"enc.proj.bias", "encoder_decoder_proj.bias"},
    {"dec.embed.token.weight", "transf_decoder._embedding.token_embedding.weight"},
    {"dec.embed.pos_enc", "transf_decoder._embedding.position_embedding.pos_enc"},
    {"dec.embed.norm.weight", "transf_decoder._embedding.layer_norm.weight"},
    {"dec.embed.norm.bias", "transf_decoder._embedding.layer_norm.bias"},
    {"dec.norm.weight", "transf_decoder._decoder.final_layer_norm.weight"},
    {"dec.norm.bias", "transf_decoder._decoder.final_layer_norm.bias"},
    {"dec.head.weight", "log_softmax.mlp.layer0.weight"},
    {"dec.head.bias", "log_softmax.mlp.layer0.bias"},
}};

std::unordered_map<std::string, std::string> transcribe_to_nemo_names(int64_t encoder_layers, int64_t decoder_layers) {
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
        const std::string from_prefix = "dec.layer." + std::to_string(i) + ".";
        const std::string to_prefix = "transf_decoder._decoder.layers." + std::to_string(i) + ".";
        for (const auto & [from, to] : kDecoderBlock) {
            out.emplace(from_prefix + std::string(from), to_prefix + std::string(to));
        }
    }
    return out;
}

[[noreturn]] void fail_architecture(const std::string & what) {
    throw std::runtime_error("Canary ASR requires the Canary 180M Flash architecture (" + what + ")");
}

void require_value(int64_t got, int64_t want, const char * key) {
    if (got != want) {
        fail_architecture(std::string(key) + " is " + std::to_string(got) + ", expected " + std::to_string(want));
    }
}

void require_text(const std::string & got, const char * want, const char * key) {
    if (got != want) {
        fail_architecture(std::string(key) + " is \"" + got + "\", expected \"" + want + "\"");
    }
}

void require_close(float got, float want, const char * key) {
    if (!(std::fabs(got - want) <= 1.0e-6f * std::max(1.0f, std::fabs(want)))) {
        fail_architecture(std::string(key) + " is " + std::to_string(got) + ", expected " + std::to_string(want));
    }
}

// Both layouts run the same NeMo AudioToMelSpectrogramPreprocessor; only the
// source of the window and filterbank differs.
void build_frontend(CanaryAssets & out) {
    audio::NemoMelFrontendConfig frontend_config;
    frontend_config.sample_rate = 16000;
    frontend_config.n_mels = 128;
    frontend_config.stft = {512, 160, 400, true, audio::STFTPadMode::Constant};
    frontend_config.preemphasis = 0.97f;
    frontend_config.window = audio::MelWindow::FromArgument;
    frontend_config.mel_bank = audio::MelBank::FromArgument;
    frontend_config.mel_path = audio::MelPath::LogMelSpectrogram;
    frontend_config.norm = audio::MelNorm::PerBinF32;
    frontend_config.layout = audio::MelLayout::FeatureMajor;
    out.frontend = std::make_shared<audio::NemoMelFrontend>(
        frontend_config, out.window, out.filterbank);
}

// The engine looks special tokens up by text; the transcribe.cpp GGUF also
// declares their ids. They must agree, or the flat token table is not the
// aggregate vocabulary the decoder was trained on.
void check_special(const assets::GgufMetadata & meta, const CanaryAssets & assets,
    const std::string & key, const std::string & text, bool required) {
    const auto id = meta.find_int(key);
    if (!id.has_value()) {
        if (required) {
            throw std::runtime_error("Canary ASR transcribe.cpp GGUF is missing " + key);
        }
        return;
    }
    if (assets.special_token(text) != *id) {
        throw std::runtime_error("Canary ASR transcribe.cpp GGUF: " + key + " = " + std::to_string(*id) +
            " does not match the token table entry for " + text);
    }
}

std::shared_ptr<const CanaryAssets> load_transcribe_gguf_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<CanaryAssets>();
    const auto meta = assets::GgufMetadata::open(path);
    const int64_t encoder_layers = meta.require_i32("stt.canary.encoder.n_layers");
    const int64_t decoder_layers = meta.require_i32("stt.canary.decoder.n_layers");
    require_value(encoder_layers, 17, "stt.canary.encoder.n_layers");
    require_value(meta.require_i32("stt.canary.encoder.d_model"), 512, "stt.canary.encoder.d_model");
    require_value(meta.require_i32("stt.canary.encoder.n_heads"), 8, "stt.canary.encoder.n_heads");
    require_value(meta.require_i32("stt.canary.encoder.d_ff"), 2048, "stt.canary.encoder.d_ff");
    require_value(meta.require_i32("stt.canary.encoder.conv_kernel"), 9, "stt.canary.encoder.conv_kernel");
    require_value(meta.require_i32("stt.canary.encoder.subsampling_factor"), 8, "stt.canary.encoder.subsampling_factor");
    require_value(meta.require_i32("stt.canary.encoder.subsampling_channels"), 256,
        "stt.canary.encoder.subsampling_channels");
    require_value(decoder_layers, 4, "stt.canary.decoder.n_layers");
    require_value(meta.require_i32("stt.canary.decoder.d_model"), 1024, "stt.canary.decoder.d_model");
    require_value(meta.require_i32("stt.canary.decoder.n_heads"), 8, "stt.canary.decoder.n_heads");
    require_value(meta.require_i32("stt.canary.decoder.d_ff"), 4096, "stt.canary.decoder.d_ff");
    require_value(meta.require_i32("stt.canary.decoder.max_position"), 1024, "stt.canary.decoder.max_position");
    require_value(meta.require_i32("stt.canary.decoder.vocab_size"), 5248, "stt.canary.decoder.vocab_size");
    require_text(meta.require_string("stt.canary.decoder.activation"), "relu", "stt.canary.decoder.activation");
    if (!meta.find_bool("stt.canary.decoder.pre_ln").value_or(true)) {
        fail_architecture("stt.canary.decoder.pre_ln is false");
    }
    if (!meta.find_bool("stt.canary.decoder.encoder_decoder_proj").value_or(false)) {
        fail_architecture("stt.canary.decoder.encoder_decoder_proj is absent or false");
    }
    require_text(meta.require_string("stt.canary.tokenizer.prompt_format"), "canary2",
        "stt.canary.tokenizer.prompt_format");
    if (meta.find_bool("stt.canary.tokenizer.single_sp").value_or(false)) {
        fail_architecture("stt.canary.tokenizer.single_sp is true; the aggregate tokenizer is required");
    }
    require_text(meta.require_string("stt.frontend.type"), "mel", "stt.frontend.type");
    require_value(meta.require_i32("stt.frontend.num_mels"), 128, "stt.frontend.num_mels");
    require_value(meta.require_i32("stt.frontend.sample_rate"), 16000, "stt.frontend.sample_rate");
    require_value(meta.require_i32("stt.frontend.n_fft"), 512, "stt.frontend.n_fft");
    require_value(meta.require_i32("stt.frontend.win_length"), 400, "stt.frontend.win_length");
    require_value(meta.require_i32("stt.frontend.hop_length"), 160, "stt.frontend.hop_length");
    require_text(meta.require_string("stt.frontend.window"), "hann", "stt.frontend.window");
    require_text(meta.require_string("stt.frontend.normalize"), "per_feature", "stt.frontend.normalize");
    require_close(meta.require_float("stt.frontend.pre_emphasis"), 0.97f, "stt.frontend.pre_emphasis");
    require_close(meta.require_float("stt.frontend.f_min"), 0.0f, "stt.frontend.f_min");
    require_close(meta.require_float("stt.frontend.f_max"), 8000.0f, "stt.frontend.f_max");

    auto names = transcribe_to_nemo_names(encoder_layers, decoder_layers);
    out->source = assets::make_renamed_tensor_source(
        assets::open_tensor_source(path),
        [names = std::move(names)](std::string_view name) -> std::optional<std::string> {
            const auto it = names.find(std::string(name));
            if (it == names.end()) {
                return std::nullopt;  // optional frontend.* buffers: the frontend is rebuilt from constants
            }
            return it->second;
        });

    const auto tokens = meta.find_string_array("tokenizer.ggml.tokens");
    if (!tokens.has_value()) {
        throw std::runtime_error("Canary ASR transcribe.cpp GGUF has no tokenizer.ggml.tokens");
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
    if (out->vocabulary.size() != 5248) {
        throw std::runtime_error("Canary vocabulary must contain 5248 entries");
    }
    check_special(meta, *out, "stt.canary.special.startoftranscript_id", "<|startoftranscript|>", true);
    check_special(meta, *out, "stt.canary.special.endoftext_id", "<|endoftext|>", true);
    check_special(meta, *out, "stt.canary.special.pad_id", "<pad>", true);
    // The arch treats the prompt-slot ids as optional; the session's text
    // lookup still requires every piece of the canary2 prompt.
    check_special(meta, *out, "stt.canary.special.startofcontext_id", "<|startofcontext|>", false);
    check_special(meta, *out, "stt.canary.special.pnc_id", "<|pnc|>", false);
    check_special(meta, *out, "stt.canary.special.nopnc_id", "<|nopnc|>", false);
    check_special(meta, *out, "stt.canary.special.noitn_id", "<|noitn|>", false);
    check_special(meta, *out, "stt.canary.special.notimestamp_id", "<|notimestamp|>", false);
    check_special(meta, *out, "stt.canary.special.nodiarize_id", "<|nodiarize|>", false);
    for (const std::string code : {"en", "de", "es", "fr"}) {
        check_special(meta, *out, "stt.canary.special.lang." + code + "_id", "<|" + code + "|>", false);
    }

    // NeMo AudioToMelSpectrogramPreprocessor: symmetric Hann window
    // (torch.hann_window(periodic=False)) and librosa's Slaney-normalized mel
    // filterbank over 0..sample_rate/2.
    out->window = audio::get_cached_stft_window({512, 160, 400, true, audio::STFTPadMode::Constant});
    out->filterbank = audio::MelFilterbank().build({16000, 512, 128, 0.0f, 8000.0f, true});
    build_frontend(*out);
    return out;
}

}  // namespace

bool looks_like_transcribe_canary_gguf(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    if (!in.read(magic, 4) || std::string_view(magic, 4) != "GGUF") {
        return false;
    }
    in.close();
    return assets::GgufMetadata::open(path).find_string("general.architecture").value_or("") == "canary";
}

int32_t CanaryAssets::special_token(const std::string & text) const {
    for (const auto & piece : vocabulary) {
        if (piece.text == text) {
            return piece.id;
        }
    }
    throw std::runtime_error("Canary tokenizer is missing " + text);
}

std::shared_ptr<const CanaryAssets> load_canary_assets(const std::filesystem::path & path) {
    if (looks_like_transcribe_canary_gguf(path)) {
        return load_transcribe_gguf_assets(path);
    }
    auto out = std::make_shared<CanaryAssets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("canary_asr"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    const auto & encoder = config.require("encoder");
    const auto & decoder = config.require("transf_decoder").require("config_dict");
    if (io::json::require_string(config, "prompt_format") != "canary2" ||
        io::json::require_i64(encoder, "d_model") != 512 ||
        io::json::require_i64(encoder, "n_layers") != 17 ||
        io::json::require_i64(encoder, "subsampling_factor") != 8 ||
        io::json::require_i64(encoder, "conv_kernel_size") != 9 ||
        io::json::require_i64(decoder, "hidden_size") != 1024 ||
        io::json::require_i64(decoder, "num_layers") != 4 ||
        !io::json::require_bool(decoder, "pre_ln")) {
        throw std::runtime_error("Canary ASR requires the Canary 180M Flash architecture");
    }
    for (const auto & item : config.require("audio_cpp_tokenizers").as_array()) {
        const auto language = io::json::require_string(item, "language");
        const auto offset = io::json::require_i64(item, "offset");
        auto pieces = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer_" + language));
        if (offset != static_cast<int64_t>(out->vocabulary.size()) ||
            static_cast<int64_t>(pieces.size()) != io::json::require_i64(item, "size")) {
            throw std::runtime_error("Canary tokenizer offsets do not match the aggregate vocabulary");
        }
        for (auto & piece : pieces) {
            piece.id += static_cast<int>(offset);
            out->vocabulary.push_back(std::move(piece));
        }
    }
    if (out->vocabulary.size() != 5248) {
        throw std::runtime_error("Canary vocabulary must contain 5248 entries");
    }
    out->window = out->source->require_f32("preprocessor.featurizer.window", {400});
    out->filterbank = {out->source->require_f32("preprocessor.featurizer.fb", {1, 128, 257}), {128, 257}};
    build_frontend(*out);
    return out;
}

}  // namespace engine::models::canary_asr
