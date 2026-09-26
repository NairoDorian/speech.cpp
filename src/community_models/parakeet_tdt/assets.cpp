#include "engine/community_models/parakeet_tdt/assets.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace json = engine::io::json;
namespace {

void validate_config(const ParakeetConfig & config) {
    if (config.model_type != "parakeet_tdt") {
        throw std::runtime_error("Parakeet TDT expects model_type=parakeet_tdt");
    }
    if (config.vocab_size <= 0 || config.blank_token_id < 0 || config.blank_token_id >= config.vocab_size) {
        throw std::runtime_error("Parakeet TDT invalid vocab metadata");
    }
    if (config.decoder_hidden_size <= 0 || config.decoder_layers != 2) {
        throw std::runtime_error("Parakeet TDT expects 2-layer LSTM decoder");
    }
    if (config.encoder.hidden_size <= 0 || config.encoder.layers <= 0 || config.encoder.heads <= 0 ||
        config.encoder.hidden_size % config.encoder.heads != 0) {
        throw std::runtime_error("Parakeet TDT invalid encoder metadata");
    }
    if (config.encoder.subsampling_factor != 8 || config.encoder.subsampling_kernel != 3 ||
        config.encoder.subsampling_stride != 2) {
        throw std::runtime_error("Parakeet TDT expects factor-8 causal subsampling config");
    }
    if (config.frontend.sample_rate != 16000 || config.frontend.feature_size <= 0 ||
        config.frontend.n_fft <= 0 || config.frontend.win_length <= 0 || config.frontend.hop_length <= 0) {
        throw std::runtime_error("Parakeet TDT invalid frontend metadata");
    }
}

std::vector<uint8_t> parse_special_token_ids(const std::filesystem::path & tokenizer_json, int64_t vocab_size) {
    std::vector<uint8_t> special(static_cast<size_t>(vocab_size), 0);
    const auto root = json::parse_file(tokenizer_json);
    if (const auto * added = root.find("added_tokens"); added != nullptr && added->is_array()) {
        for (const auto & item : added->as_array()) {
            if (!json::optional_bool(item, "special", false)) {
                continue;
            }
            const int64_t id = json::require_i64(item, "id");
            if (id >= 0 && id < vocab_size) {
                special[static_cast<size_t>(id)] = 1;
            }
        }
    }
    return special;
}

ParakeetConfig parse_config(const assets::ResourceBundle & resources) {
    const auto config_root = resources.parse_json("config");

    ParakeetConfig config;
    config.model_type = json::require_string(config_root, "model_type");
    config.vocab_size = json::require_i64(config_root, "vocab_size");
    config.blank_token_id = json::require_i64(config_root, "blank_token_id");
    config.pad_token_id = json::require_i64(config_root, "pad_token_id");
    config.decoder_hidden_size = json::require_i64(config_root, "decoder_hidden_size");
    config.decoder_layers = json::require_i64(config_root, "num_decoder_layers");
    config.max_symbols_per_step = json::require_i64(config_root, "max_symbols_per_step");
    // The audio.cpp graphs size the joint by the decoder hidden size.
    config.joint_hidden_size = config.decoder_hidden_size;

    if (const auto * dur = config_root.find("durations"); dur != nullptr && dur->is_array()) {
        config.durations.clear();
        for (const auto & d : dur->as_array()) {
            if (!d.is_number()) continue;
            config.durations.push_back(static_cast<int32_t>(d.as_i64()));
        }
    }

    const auto & encoder = config_root.require("encoder_config");
    config.encoder.hidden_size = json::require_i64(encoder, "hidden_size");
    config.encoder.intermediate_size = json::require_i64(encoder, "intermediate_size");
    config.encoder.layers = json::require_i64(encoder, "num_hidden_layers");
    config.encoder.heads = json::require_i64(encoder, "num_attention_heads");
    config.encoder.conv_kernel = json::require_i64(encoder, "conv_kernel_size");
    config.encoder.subsampling_factor = json::require_i64(encoder, "subsampling_factor");
    config.encoder.subsampling_channels = json::require_i64(encoder, "subsampling_conv_channels");
    config.encoder.subsampling_kernel = json::require_i64(encoder, "subsampling_conv_kernel_size");
    config.encoder.subsampling_stride = json::require_i64(encoder, "subsampling_conv_stride");
    config.encoder.max_position_embeddings = json::require_i64(encoder, "max_position_embeddings");

    // processor_config.json is genuinely optional (not every model directory
    // layout provides it), so fall back to the ParakeetFrontendConfig
    // defaults when it's absent. But if the file IS present, any parse or
    // schema failure is a real problem worth failing loudly on: silently
    // keeping the defaults could load a variant model's weights against the
    // wrong frontend parameters (sample rate, FFT size, hop length, ...)
    // without any indication anything went wrong.
    if (resources.has_file("processor_config")) {
        const auto processor_root = resources.parse_json("processor_config");
        const auto & feature = processor_root.require("feature_extractor");
        config.frontend.sample_rate = json::require_i64(feature, "sampling_rate");
        config.frontend.feature_size = json::require_i64(feature, "feature_size");
        config.frontend.n_fft = json::require_i64(feature, "n_fft");
        config.frontend.win_length = json::require_i64(feature, "win_length");
        config.frontend.hop_length = json::require_i64(feature, "hop_length");
        if (const auto * p = feature.find("preemphasis"); p != nullptr) {
            config.frontend.preemphasis = json::require_f32(feature, "preemphasis");
        }
    }

    validate_config(config);
    return config;
}

// ---------------------------------------------------------------------------
// The transcribe.cpp GGUF layout (general.architecture == "parakeet",
// published under huggingface.co/handy-computer/parakeet-*-gguf).
//
// The retired-arch path: the engine package must open the files the
// `parakeet` arch in src/runtime/arch/parakeet opened. Tensor names are
// transcribe.cpp scripts/convert-parakeet.py's tables read backwards
// (transcribe.cpp name -> the HuggingFace name load_parakeet_weights binds).
// The converter stores every tensor in its NeMo layout (no transpose), so the
// view is a rename with three exceptions handled in weights.cpp:
//   - pred.lstm.<i>.bias is bias_ih + bias_hh pre-summed by the converter
//     (add_combined); it has no HF counterpart and is exposed as
//     decoder.lstm.bias_l<i>.
//   - head.ctc.* (CTC checkpoints) is exposed as ctc_head.*.
//   - the converter drops the preprocessor buffers; the mel frontend is
//     rebuilt from the stt.frontend.* KVs (Hann window + Slaney bank).
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
    {"attn.linear_q.weight", "self_attn.q_proj.weight"},
    {"attn.linear_q.bias", "self_attn.q_proj.bias"},
    {"attn.linear_k.weight", "self_attn.k_proj.weight"},
    {"attn.linear_k.bias", "self_attn.k_proj.bias"},
    {"attn.linear_v.weight", "self_attn.v_proj.weight"},
    {"attn.linear_v.bias", "self_attn.v_proj.bias"},
    {"attn.linear_out.weight", "self_attn.o_proj.weight"},
    {"attn.linear_out.bias", "self_attn.o_proj.bias"},
    {"attn.linear_pos.weight", "self_attn.relative_k_proj.weight"},
    {"attn.pos_bias_u", "self_attn.bias_u"},
    {"attn.pos_bias_v", "self_attn.bias_v"},
    {"norm_conv.weight", "norm_conv.weight"},
    {"norm_conv.bias", "norm_conv.bias"},
    {"conv.pointwise1.weight", "conv.pointwise_conv1.weight"},
    {"conv.pointwise1.bias", "conv.pointwise_conv1.bias"},
    {"conv.depthwise.weight", "conv.depthwise_conv.weight"},
    {"conv.depthwise.bias", "conv.depthwise_conv.bias"},
    {"conv.pointwise2.weight", "conv.pointwise_conv2.weight"},
    {"conv.pointwise2.bias", "conv.pointwise_conv2.bias"},
    {"conv.bn.weight", "conv.norm.weight"},
    {"conv.bn.bias", "conv.norm.bias"},
    {"conv.bn.running_mean", "conv.norm.running_mean"},
    {"conv.bn.running_var", "conv.norm.running_var"},
    {"norm_ff2.weight", "norm_feed_forward2.weight"},
    {"norm_ff2.bias", "norm_feed_forward2.bias"},
    {"ff2.linear1.weight", "feed_forward2.linear1.weight"},
    {"ff2.linear1.bias", "feed_forward2.linear1.bias"},
    {"ff2.linear2.weight", "feed_forward2.linear2.weight"},
    {"ff2.linear2.bias", "feed_forward2.linear2.bias"},
    {"norm_out.weight", "norm_out.weight"},
    {"norm_out.bias", "norm_out.bias"},
}};

constexpr std::array<std::pair<std::string_view, std::string_view>, 21> kTopLevel = {{
    {"enc.pre_encode.conv.0.weight", "encoder.subsampling.layers.0.weight"},
    {"enc.pre_encode.conv.0.bias", "encoder.subsampling.layers.0.bias"},
    {"enc.pre_encode.conv.2.weight", "encoder.subsampling.layers.2.weight"},
    {"enc.pre_encode.conv.2.bias", "encoder.subsampling.layers.2.bias"},
    {"enc.pre_encode.conv.3.weight", "encoder.subsampling.layers.3.weight"},
    {"enc.pre_encode.conv.3.bias", "encoder.subsampling.layers.3.bias"},
    {"enc.pre_encode.conv.5.weight", "encoder.subsampling.layers.5.weight"},
    {"enc.pre_encode.conv.5.bias", "encoder.subsampling.layers.5.bias"},
    {"enc.pre_encode.conv.6.weight", "encoder.subsampling.layers.6.weight"},
    {"enc.pre_encode.conv.6.bias", "encoder.subsampling.layers.6.bias"},
    {"enc.pre_encode.out.weight", "encoder.subsampling.linear.weight"},
    {"enc.pre_encode.out.bias", "encoder.subsampling.linear.bias"},
    {"pred.embed.weight", "decoder.embedding.weight"},
    {"joint.enc.weight", "encoder_projector.weight"},
    {"joint.enc.bias", "encoder_projector.bias"},
    {"joint.pred.weight", "decoder.decoder_projector.weight"},
    {"joint.pred.bias", "decoder.decoder_projector.bias"},
    {"joint.out.weight", "joint.head.weight"},
    {"joint.out.bias", "joint.head.bias"},
    {"head.ctc.weight", "ctc_head.weight"},
    {"head.ctc.bias", "ctc_head.bias"},
}};

std::unordered_map<std::string, std::string> transcribe_to_engine_names(int64_t encoder_layers, int64_t lstm_layers) {
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
    for (int64_t i = 0; i < lstm_layers; ++i) {
        const std::string l = std::to_string(i);
        out.emplace("pred.lstm." + l + ".Wx", "decoder.lstm.weight_ih_l" + l);
        out.emplace("pred.lstm." + l + ".Wh", "decoder.lstm.weight_hh_l" + l);
        out.emplace("pred.lstm." + l + ".bias", "decoder.lstm.bias_l" + l);
    }
    return out;
}

[[noreturn]] void fail_variant(const std::string & what) {
    throw std::runtime_error("Parakeet transcribe.cpp GGUF: " + what);
}

[[noreturn]] void fail_unsupported(const std::string & variant, const std::string & what) {
    throw std::runtime_error("Parakeet transcribe.cpp GGUF variant '" + variant +
        "' is not supported by the parakeet_tdt engine package: " + what);
}

void require_positive(int64_t value, const char * key) {
    if (value <= 0) {
        fail_variant(std::string(key) + " must be positive (got " + std::to_string(value) + ")");
    }
}

void require_close(float got, float want, const char * key, const std::string & variant) {
    if (!(std::fabs(got - want) <= 1.0e-3f * std::max(1.0f, std::fabs(want)))) {
        fail_unsupported(variant, std::string(key) + " is " + std::to_string(got) + ", the engine frontend implements " +
            std::to_string(want));
    }
}

// True for a multilingual language-tag piece "<ll-RR>" (2-3 lowercase letters,
// '-', 2-4 letters). Mirrors arch/parakeet/model.cpp is_lang_tag_piece: the
// arch strips these even when an older converter left them NORMAL-typed.
bool is_lang_tag_piece(const std::string & p) {
    const size_t n = p.size();
    if (n < 7 || p.front() != '<' || p.back() != '>') {
        return false;
    }
    size_t i = 1;
    const size_t end = n - 1;
    const size_t lang0 = i;
    while (i < end && p[i] >= 'a' && p[i] <= 'z') {
        ++i;
    }
    const size_t lang_len = i - lang0;
    if (lang_len < 2 || lang_len > 3 || i >= end || p[i] != '-') {
        return false;
    }
    ++i;
    const size_t reg0 = i;
    while (i < end && std::isalpha(static_cast<unsigned char>(p[i]))) {
        ++i;
    }
    const size_t reg_len = i - reg0;
    return reg_len >= 2 && reg_len <= 4 && i == end;
}

std::shared_ptr<const ParakeetTDTAssets> load_transcribe_gguf_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<ParakeetTDTAssets>();
    out->layout = ParakeetLayout::TranscribeGguf;
    const auto meta = assets::GgufMetadata::open(path);
    auto & config = out->config;
    auto & identity = out->identity;

    identity.variant = meta.find_string("stt.variant").value_or("tdt-0.6b-v2");  // arch k_default_variant
    identity.display_name = meta.find_string("general.name").value_or("Parakeet");
    identity.languages = meta.find_string_array("general.languages").value_or(std::vector<std::string>{});
    identity.lang_detect = meta.find_bool("stt.capability.lang_detect").value_or(false);
    const std::string & variant = identity.variant;

    // ----- Variants the engine does not implement: reject by capability, not
    // by variant string (the arch keys behavior off KVs, never stt.variant).
    if (meta.find_bool("stt.parakeet.diarizer.embedded").value_or(false)) {
        fail_unsupported(variant, "multitalker bundle with an embedded Sortformer diarizer "
            "(stt.parakeet.diarizer.embedded); speaker-attributed decoding is not ported");
    }
    if (!meta.find_i32_array("stt.parakeet.encoder.spk_kernel_layers").value_or(std::vector<int32_t>{}).empty()) {
        fail_unsupported(variant, "speaker-kernel injection (stt.parakeet.encoder.spk_kernel_layers, "
            "multitalker-parakeet-streaming) is not ported");
    }
    if (meta.find_int("stt.parakeet.prompt.num_prompts").value_or(0) > 0) {
        fail_unsupported(variant, "language-prompt MLP (stt.parakeet.prompt.*, nemotron-3.5-asr-streaming) "
            "is not ported");
    }
    const std::string att_style = meta.find_string("stt.parakeet.encoder.att_context_style").value_or("regular");
    if (att_style == "chunked_limited") {
        fail_unsupported(variant, "cache-aware streaming encoder (att_context_style=chunked_limited: causal "
            "subsampling, chunked attention mask, causal conv, cache-aware stream API) is not ported");
    }
    if (att_style != "regular" && att_style != "chunked_limited_with_rc") {
        fail_variant("unsupported stt.parakeet.encoder.att_context_style \"" + att_style + "\"");
    }
    const std::string conv_norm = meta.find_string("stt.parakeet.encoder.conv_norm_type").value_or("batch_norm");
    if (conv_norm != "batch_norm") {
        fail_unsupported(variant, "conv_norm_type=" + conv_norm + " (only batch_norm is implemented)");
    }

    // ----- Head.
    const std::string head = meta.find_string("stt.parakeet.head_kind").value_or("tdt");
    if (head == "tdt") {
        config.head_kind = ParakeetHeadKind::Tdt;
        config.model_type = "parakeet_tdt";
    } else if (head == "rnnt") {
        config.head_kind = ParakeetHeadKind::Rnnt;
        config.model_type = "parakeet_rnnt";
    } else if (head == "ctc") {
        config.head_kind = ParakeetHeadKind::Ctc;
        config.model_type = "parakeet_ctc";
    } else {
        fail_variant("unsupported stt.parakeet.head_kind \"" + head + "\" (allowed: tdt, rnnt, ctc)");
    }

    // ----- Encoder (read_parakeet_hparams contract).
    auto & enc = config.encoder;
    enc.layers = meta.require_i32("stt.parakeet.encoder.n_layers");
    enc.hidden_size = meta.require_i32("stt.parakeet.encoder.d_model");
    enc.heads = meta.require_i32("stt.parakeet.encoder.n_heads");
    enc.intermediate_size = meta.require_i32("stt.parakeet.encoder.d_ff");
    enc.conv_kernel = meta.require_i32("stt.parakeet.encoder.conv_kernel");
    enc.subsampling_factor = meta.require_i32("stt.parakeet.encoder.subsampling_factor");
    enc.subsampling_channels = meta.require_i32("stt.parakeet.encoder.subsampling_channels");
    (void) meta.require_i32("stt.parakeet.encoder.pos_emb_max_len");
    // The arch builds the relative table for whatever length it is handed
    // (NeMo extend_pe does the same); pos_emb_max_len is only the training
    // pre-allocation. Full-context length is bounded by memory, not by this.
    enc.max_position_embeddings = std::numeric_limits<int32_t>::max();
    enc.subsampling_kernel = 3;  // NeMo dw_striding: fixed k=3, s=2 stages
    enc.subsampling_stride = 2;
    enc.use_bias = meta.find_bool("stt.parakeet.encoder.use_bias").value_or(false);
    enc.xscaling = meta.find_bool("stt.parakeet.encoder.xscaling").value_or(false);
    enc.att_context_left = meta.find_int("stt.parakeet.encoder.att_context_left").value_or(-1);
    enc.att_context_right = meta.find_int("stt.parakeet.encoder.att_context_right").value_or(-1);
    enc.nemo_f32_positional_encoding = true;
    require_positive(enc.layers, "stt.parakeet.encoder.n_layers");
    require_positive(enc.hidden_size, "stt.parakeet.encoder.d_model");
    require_positive(enc.heads, "stt.parakeet.encoder.n_heads");
    require_positive(enc.intermediate_size, "stt.parakeet.encoder.d_ff");
    require_positive(enc.conv_kernel, "stt.parakeet.encoder.conv_kernel");
    require_positive(enc.subsampling_channels, "stt.parakeet.encoder.subsampling_channels");
    if (enc.hidden_size % enc.heads != 0) {
        fail_variant("encoder d_model (" + std::to_string(enc.hidden_size) + ") not divisible by n_heads (" +
            std::to_string(enc.heads) + ")");
    }
    if (enc.subsampling_factor != 8) {
        fail_unsupported(variant, "subsampling_factor=" + std::to_string(enc.subsampling_factor) +
            " (only the factor-8 dw_striding stack is implemented)");
    }
    if (enc.conv_kernel % 2 != 1) {
        fail_unsupported(variant, "even conformer conv_kernel " + std::to_string(enc.conv_kernel));
    }
    const bool full_attention = enc.att_context_left < 0 && enc.att_context_right < 0;
    const bool local_attention = enc.att_context_left >= 0 && enc.att_context_right >= 0;
    if (!full_attention && !local_attention) {
        fail_variant("att_context_left/right must both be -1 (full) or both >= 0 (local window)");
    }
    if (att_style == "chunked_limited_with_rc") {
        // parakeet-unified-en-0.6b: the offline path is plain full attention
        // (arch encoder.cpp: the chunked mask engages only for the buffered
        // stream driver); its chunked-with-rc stream API is a gap.
        if (!full_attention) {
            fail_variant("chunked_limited_with_rc expects full offline attention (att_context -1/-1)");
        }
        identity.streaming = true;
    }
    {
        const int64_t left = meta.find_int("stt.parakeet.encoder.conv_context_left").value_or(-1);
        const int64_t right = meta.find_int("stt.parakeet.encoder.conv_context_right").value_or(-1);
        const int64_t half = (enc.conv_kernel - 1) / 2;
        const bool centred = (left < 0 && right < 0) || (left == half && right == half);
        if (!centred) {
            fail_unsupported(variant, "asymmetric/causal conformer conv context [" + std::to_string(left) + ", " +
                std::to_string(right) + "] (only the centred (k-1)/2 padding is implemented)");
        }
    }

    // ----- Tokenizer (tokenizer.ggml.*; the +1 <blank> row is the last piece).
    const auto tokens = meta.find_string_array("tokenizer.ggml.tokens");
    if (!tokens.has_value() || tokens->empty()) {
        fail_variant("missing tokenizer.ggml.tokens");
    }
    out->vocab_pieces = *tokens;
    out->vocab_types = meta.find_i32_array("tokenizer.ggml.token_type").value_or(std::vector<int32_t>{});
    if (!out->vocab_types.empty() && out->vocab_types.size() != out->vocab_pieces.size()) {
        fail_variant("tokenizer.ggml.token_type length does not match tokenizer.ggml.tokens");
    }
    const int64_t n_tokens = static_cast<int64_t>(out->vocab_pieces.size());

    // ----- Predictor / joint / TDT.
    config.pad_token_id = -1;
    if (config.head_kind == ParakeetHeadKind::Ctc) {
        config.decoder_layers = 0;
        config.decoder_hidden_size = 0;
        config.joint_hidden_size = 0;
        config.durations.clear();
        config.max_symbols_per_step = 0;
        config.vocab_size = n_tokens;  // head.ctc rows == vocab + 1 (checked at weight load)
    } else {
        config.decoder_hidden_size = meta.require_i32("stt.parakeet.predictor.hidden");
        config.decoder_layers = meta.require_i32("stt.parakeet.predictor.n_layers");
        config.vocab_size = meta.require_i32("stt.parakeet.predictor.vocab");
        config.joint_hidden_size = meta.require_i32("stt.parakeet.joint.hidden");
        const int64_t extra = meta.require_i32("stt.parakeet.joint.num_extra_outputs");
        const std::string activation = meta.require_string("stt.parakeet.joint.activation");
        require_positive(config.decoder_hidden_size, "stt.parakeet.predictor.hidden");
        require_positive(config.decoder_layers, "stt.parakeet.predictor.n_layers");
        require_positive(config.joint_hidden_size, "stt.parakeet.joint.hidden");
        if (config.vocab_size <= 1) {
            fail_variant("stt.parakeet.predictor.vocab must be > 1");
        }
        if (activation == "relu") {
            config.joint_activation = ParakeetJointActivation::Relu;
        } else if (activation == "sigmoid") {
            config.joint_activation = ParakeetJointActivation::Sigmoid;
        } else if (activation == "tanh") {
            config.joint_activation = ParakeetJointActivation::Tanh;
        } else {
            fail_variant("unsupported stt.parakeet.joint.activation \"" + activation +
                "\" (relu, sigmoid, tanh are implemented)");
        }
        if (config.head_kind == ParakeetHeadKind::Tdt) {
            const auto durations = meta.find_i32_array("stt.parakeet.tdt.durations");
            if (!durations.has_value() || durations->empty()) {
                fail_variant("head_kind=tdt requires a non-empty stt.parakeet.tdt.durations");
            }
            config.durations = *durations;
            if (static_cast<int64_t>(config.durations.size()) != extra) {
                fail_variant("stt.parakeet.tdt.durations length (" + std::to_string(config.durations.size()) +
                    ") must equal joint.num_extra_outputs (" + std::to_string(extra) + ")");
            }
            for (const int32_t d : config.durations) {
                if (d < 0) {
                    fail_variant("stt.parakeet.tdt.durations contains a negative value");
                }
            }
            const int64_t max_symbols = meta.find_int("stt.parakeet.tdt.max_symbols").value_or(10);
            if (max_symbols < 0) {
                fail_variant("stt.parakeet.tdt.max_symbols must be >= 0");
            }
            config.max_symbols_per_step = max_symbols;
        } else {
            if (extra != 0) {
                fail_variant("head_kind=rnnt requires joint.num_extra_outputs=0 (got " + std::to_string(extra) + ")");
            }
            config.durations.clear();
            config.max_symbols_per_step = 10;  // arch weights.cpp: RNNT reuses the default stuck cap
        }
        if (config.vocab_size != n_tokens) {
            fail_variant("stt.parakeet.predictor.vocab (" + std::to_string(config.vocab_size) +
                ") does not match tokenizer.ggml.tokens (" + std::to_string(n_tokens) + ")");
        }
    }
    // Blank is the extra last class for every head (arch decoder.cpp
    // build_host_decoder_weights: blank_id = pred_vocab - 1 / n_classes - 1).
    config.blank_token_id = config.vocab_size - 1;
    if (const auto blank = meta.find_int("tokenizer.ggml.blank_token_id");
        blank.has_value() && *blank != config.blank_token_id) {
        fail_variant("tokenizer.ggml.blank_token_id (" + std::to_string(*blank) +
            ") is not the last class (" + std::to_string(config.blank_token_id) + ")");
    }

    // ----- Frontend (the complete stt.frontend.* contract).
    auto & fe = config.frontend;
    const std::string fe_type = meta.require_string("stt.frontend.type");
    if (fe_type != "mel") {
        fail_unsupported(variant, "frontend type \"" + fe_type + "\" (only mel)");
    }
    fe.feature_size = meta.require_i32("stt.frontend.num_mels");
    fe.sample_rate = meta.require_i32("stt.frontend.sample_rate");
    fe.n_fft = meta.require_i32("stt.frontend.n_fft");
    fe.win_length = meta.require_i32("stt.frontend.win_length");
    fe.hop_length = meta.require_i32("stt.frontend.hop_length");
    fe.preemphasis = meta.require_float("stt.frontend.pre_emphasis");
    (void) meta.require_float("stt.frontend.dither");  // training-only; the arch never dithers at inference
    require_positive(fe.feature_size, "stt.frontend.num_mels");
    require_positive(fe.n_fft, "stt.frontend.n_fft");
    require_positive(fe.win_length, "stt.frontend.win_length");
    require_positive(fe.hop_length, "stt.frontend.hop_length");
    if (fe.sample_rate != 16000) {
        fail_unsupported(variant, "stt.frontend.sample_rate " + std::to_string(fe.sample_rate) + " (16000 only)");
    }
    if (fe.win_length > fe.n_fft || (fe.n_fft & (fe.n_fft - 1)) != 0) {
        fail_variant("stt.frontend.n_fft must be a power of two >= win_length");
    }
    if (fe.feature_size % enc.subsampling_factor != 0) {
        fail_variant("stt.frontend.num_mels must be divisible by the subsampling factor");
    }
    if (const std::string window = meta.require_string("stt.frontend.window"); window != "hann") {
        fail_unsupported(variant, "frontend window \"" + window + "\" (only hann)");
    }
    const std::string normalize = meta.require_string("stt.frontend.normalize");
    if (normalize == "per_feature") {
        fe.normalize_per_feature = true;
    } else if (normalize == "none") {
        fe.normalize_per_feature = false;
    } else {
        fail_unsupported(variant, "frontend normalize \"" + normalize + "\" (per_feature or none)");
    }
    // The engine NeMo frontend hard-wires the Slaney bank over 0..sr/2, which
    // is what every converter wrote.
    require_close(meta.require_float("stt.frontend.f_min"), 0.0f, "stt.frontend.f_min", variant);
    require_close(meta.require_float("stt.frontend.f_max"), static_cast<float>(fe.sample_rate) / 2.0f,
        "stt.frontend.f_max", variant);
    fe.all_stft_frames_valid = true;

    // ----- Tokens the text drops by default (model.cpp is_strippable_special:
    // CONTROL-typed, or the <ll-RR> locale-tag shape).
    constexpr int32_t kTokenTypeControl = 3;
    out->special_token_ids.assign(static_cast<size_t>(n_tokens), 0);
    for (int64_t i = 0; i < n_tokens; ++i) {
        const bool control = !out->vocab_types.empty() && out->vocab_types[static_cast<size_t>(i)] == kTokenTypeControl;
        if (control || is_lang_tag_piece(out->vocab_pieces[static_cast<size_t>(i)])) {
            out->special_token_ids[static_cast<size_t>(i)] = 1;
        }
    }

    auto names = transcribe_to_engine_names(enc.layers, config.decoder_layers);
    out->source = assets::make_renamed_tensor_source(
        assets::open_tensor_source(path),
        [names = std::move(names)](std::string_view name) -> std::optional<std::string> {
            const auto it = names.find(std::string(name));
            if (it == names.end()) {
                return std::nullopt;  // tensors of unported variants were rejected above
            }
            return it->second;
        });
    return out;
}

}  // namespace

bool looks_like_transcribe_parakeet_gguf(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    if (!in.read(magic, 4) || std::string_view(magic, 4) != "GGUF") {
        return false;
    }
    in.close();
    const auto meta = assets::GgufMetadata::open(path);
    return meta.find_string("general.architecture").value_or("") == "parakeet" &&
        meta.has("stt.parakeet.encoder.n_layers");
}

std::string decode_transcribe_pieces(const ParakeetTDTAssets & assets, const int32_t * ids, size_t count) {
    // transcribe-tokenizer.cpp decode_sentencepiece, byte for byte.
    static constexpr char kSpace[] = "\xE2\x96\x81";
    constexpr size_t kSpaceLen = 3;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        return -1;
    };
    std::string out;
    out.reserve(count * 4);
    for (size_t i = 0; i < count; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || static_cast<size_t>(id) >= assets.vocab_pieces.size()) {
            continue;
        }
        const std::string & p = assets.vocab_pieces[static_cast<size_t>(id)];
        if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
            const int hi = hex(p[3]);
            const int lo = hex(p[4]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                continue;
            }
        }
        size_t j = 0;
        while (j < p.size()) {
            if (j + kSpaceLen <= p.size() && std::memcmp(p.data() + j, kSpace, kSpaceLen) == 0) {
                out.push_back(' ');
                j += kSpaceLen;
            } else {
                out.push_back(p[j]);
                ++j;
            }
        }
    }
    return out;
}

std::shared_ptr<const ParakeetTDTAssets> load_parakeet_assets(const std::filesystem::path & model_path) {
    if (looks_like_transcribe_parakeet_gguf(model_path)) {
        return load_transcribe_gguf_assets(model_path);
    }
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("parakeet_tdt"));
    auto assets = std::make_shared<ParakeetTDTAssets>();
    assets->resources = std::move(resources);
    assets->source = assets->resources.open_tensor_source("weights");
    assets->config = parse_config(assets->resources);
    const auto & tokenizer_json = assets->resources.require_file("tokenizer_json");
    assets->tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(tokenizer_json);
    assets->special_token_ids =
        parse_special_token_ids(tokenizer_json, assets->config.vocab_size);
    return assets;
}

}  // namespace engine::community_models::parakeet_tdt
