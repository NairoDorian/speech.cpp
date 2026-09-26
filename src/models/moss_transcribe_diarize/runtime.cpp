#include "engine/models/moss_transcribe_diarize/runtime.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_frontend.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/partial_text.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace engine::models::moss_transcribe_diarize {
namespace {

using Storage = assets::TensorStorageType;
namespace binding = modules::binding;
constexpr int64_t kHidden = 1024;
constexpr int64_t kAudioFrames = 375;
constexpr int64_t kLookupSteps = 512;
constexpr int64_t kChunkSamples = 480000;
constexpr int32_t kAudioToken = 151671;
constexpr int32_t kEosToken = 151645;
// The retired arch's decode budget (arch/moss/model.cpp run(): k_max_new and
// transcribe::pick_decode_budget(2 * T_enc + 128, k_max_new, T_prompt, ceiling)),
// the default for the transcribe.cpp GGUF, which carries no generation config.
constexpr int64_t kArchMinBudget = 256;

const char * const kDefaultPrompt =
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。";
const char * const kPromptPrefix =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|>";
const char * const kPromptSuffixHead = "<|audio_end|>\n";
const char * const kPromptSuffixTail = "<|im_end|>\n<|im_start|>assistant\n";

struct ContextDeleter {
    void operator()(ggml_context * value) const { ggml_free(value); }
};
struct AllocatorDeleter {
    void operator()(ggml_gallocr_t value) const { ggml_gallocr_free(value); }
};

// ---------------------------------------------------------------------------
// The transcribe.cpp GGUF layout (general.architecture == "moss", published
// under huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf). The
// retired-arch path: the engine package must open the files the `moss` arch in
// src/runtime/arch/moss opened. The tables are transcribe.cpp
// scripts/convert-moss.py's ENC_*/ADAPTOR/DEC_* tables read backwards
// (transcribe.cpp name -> HuggingFace name), with the Whisper encoder taking
// the OpenAI names the audio.cpp converter (tests/moss_transcribe_diarize/
// convert_gguf.py tensor_name) gives it. Neither converter transposes, splits
// or fuses a tensor, so the view is a pure rename; the tied lm_head is absent
// from both layouts (the runtime reuses the token embedding), and the
// frontend.* buffers are dropped (the Whisper log-mel frontend is rebuilt from
// constants the loader checks against the stt.frontend.* KVs).
// ---------------------------------------------------------------------------

constexpr std::array<std::pair<std::string_view, std::string_view>, 13> kTopLevel = {{
    {"enc.conv.0.weight", "encoder.conv1.weight"},
    {"enc.conv.0.bias", "encoder.conv1.bias"},
    {"enc.conv.1.weight", "encoder.conv2.weight"},
    {"enc.conv.1.bias", "encoder.conv2.bias"},
    {"enc.pos_emb.weight", "encoder.positional_embedding"},
    {"enc.final_norm.weight", "encoder.ln_post.weight"},
    {"enc.final_norm.bias", "encoder.ln_post.bias"},
    {"adaptor.fc1.weight", "model.vq_adaptor.layers.0.weight"},
    {"adaptor.fc1.bias", "model.vq_adaptor.layers.0.bias"},
    {"adaptor.fc2.weight", "model.vq_adaptor.layers.2.weight"},
    {"adaptor.fc2.bias", "model.vq_adaptor.layers.2.bias"},
    {"adaptor.norm_out.weight", "model.vq_adaptor.layers.3.weight"},
    {"adaptor.norm_out.bias", "model.vq_adaptor.layers.3.bias"},
}};

// Whisper block: q / v / out have a bias, k does not.
constexpr std::array<std::pair<std::string_view, std::string_view>, 15> kEncoderBlock = {{
    {"norm_attn.weight", "attn_ln.weight"},
    {"norm_attn.bias", "attn_ln.bias"},
    {"attn.q.weight", "attn.query.weight"},
    {"attn.q.bias", "attn.query.bias"},
    {"attn.k.weight", "attn.key.weight"},
    {"attn.v.weight", "attn.value.weight"},
    {"attn.v.bias", "attn.value.bias"},
    {"attn.out.weight", "attn.out.weight"},
    {"attn.out.bias", "attn.out.bias"},
    {"norm_ffn.weight", "mlp_ln.weight"},
    {"norm_ffn.bias", "mlp_ln.bias"},
    {"ffn.fc1.weight", "mlp.0.weight"},
    {"ffn.fc1.bias", "mlp.0.bias"},
    {"ffn.fc2.weight", "mlp.2.weight"},
    {"ffn.fc2.bias", "mlp.2.bias"},
}};

// Qwen3 block: per-head q/k RMSNorm, no attention biases. The arch packs
// gate/up into one tensor at load; the file stores them separately.
constexpr std::array<std::pair<std::string_view, std::string_view>, 11> kDecoderBlock = {{
    {"norm_attn.weight", "input_layernorm.weight"},
    {"norm_ffn.weight", "post_attention_layernorm.weight"},
    {"attn.q.weight", "self_attn.q_proj.weight"},
    {"attn.k.weight", "self_attn.k_proj.weight"},
    {"attn.v.weight", "self_attn.v_proj.weight"},
    {"attn.o.weight", "self_attn.o_proj.weight"},
    {"attn.q_norm.weight", "self_attn.q_norm.weight"},
    {"attn.k_norm.weight", "self_attn.k_norm.weight"},
    {"ffn.gate.weight", "mlp.gate_proj.weight"},
    {"ffn.up.weight", "mlp.up_proj.weight"},
    {"ffn.down.weight", "mlp.down_proj.weight"},
}};

std::unordered_map<std::string, std::string> transcribe_to_package_names(int64_t encoder_layers, int64_t decoder_layers) {
    std::unordered_map<std::string, std::string> out;
    out.emplace("dec.token_embd.weight", "model.language_model.embed_tokens.weight");
    out.emplace("dec.output_norm.weight", "model.language_model.norm.weight");
    for (const auto & [from, to] : kTopLevel) {
        out.emplace(std::string(from), std::string(to));
    }
    for (int64_t i = 0; i < encoder_layers; ++i) {
        const std::string from_prefix = "enc.blocks." + std::to_string(i) + ".";
        const std::string to_prefix = "encoder.blocks." + std::to_string(i) + ".";
        for (const auto & [from, to] : kEncoderBlock) {
            out.emplace(from_prefix + std::string(from), to_prefix + std::string(to));
        }
    }
    for (int64_t i = 0; i < decoder_layers; ++i) {
        const std::string from_prefix = "dec.blocks." + std::to_string(i) + ".";
        const std::string to_prefix = "model.language_model.layers." + std::to_string(i) + ".";
        for (const auto & [from, to] : kDecoderBlock) {
            out.emplace(from_prefix + std::string(from), to_prefix + std::string(to));
        }
    }
    return out;
}

void require_value(int64_t got, int64_t want, const char * what) {
    if (got != want) {
        throw std::runtime_error(std::string("Unsupported MOSS-Transcribe-Diarize architecture (") + what + " is " +
            std::to_string(got) + ", expected " + std::to_string(want) + ")");
    }
}

std::shared_ptr<const TranscribeAssets> load_transcribe_gguf_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<TranscribeAssets>();
    const auto meta = assets::GgufMetadata::open(path);
    // The runtime hard-wires the MOSS-Transcribe-Diarize geometry (Whisper-Medium
    // encoder, 4x merge, Qwen3-0.6B decoder); the KVs must describe exactly that.
    const int64_t encoder_layers = meta.require_i32("stt.moss.encoder.n_layers");
    const int64_t decoder_layers = meta.require_i32("stt.moss.decoder.n_layers");
    require_value(encoder_layers, 24, "stt.moss.encoder.n_layers");
    require_value(meta.require_i32("stt.moss.encoder.d_model"), 1024, "stt.moss.encoder.d_model");
    require_value(meta.require_i32("stt.moss.encoder.n_heads"), 16, "stt.moss.encoder.n_heads");
    require_value(meta.require_i32("stt.moss.encoder.ffn_dim"), 4096, "stt.moss.encoder.ffn_dim");
    require_value(meta.require_i32("stt.moss.encoder.num_mel_bins"), 80, "stt.moss.encoder.num_mel_bins");
    require_value(meta.require_i32("stt.moss.encoder.max_source_positions"), 1500,
        "stt.moss.encoder.max_source_positions");
    if (meta.require_string("stt.moss.encoder.activation") != "gelu") {
        throw std::runtime_error("Unsupported MOSS-Transcribe-Diarize architecture (encoder activation is not gelu)");
    }
    require_value(decoder_layers, 28, "stt.moss.decoder.n_layers");
    require_value(meta.require_i32("stt.moss.decoder.hidden_size"), kHidden, "stt.moss.decoder.hidden_size");
    const auto act = meta.require_string("stt.moss.decoder.hidden_act");
    if (act != "silu" && act != "swish") {
        throw std::runtime_error("Unsupported MOSS-Transcribe-Diarize architecture (decoder hidden_act is " + act + ")");
    }
    if (meta.require_float("stt.moss.decoder.rms_norm_eps") != 1e-6f ||
        meta.require_float("stt.moss.decoder.rope_theta") != 1000000.f) {
        throw std::runtime_error("Unsupported MOSS-Transcribe-Diarize architecture (decoder rms_norm_eps / rope_theta)");
    }
    if (!meta.find_bool("stt.moss.decoder.tie_word_embeddings").value_or(true)) {
        throw std::runtime_error("MOSS-Transcribe-Diarize transcribe.cpp GGUF: expected a head tied to the token embedding");
    }
    require_value(meta.require_i32("stt.moss.adaptor.input_dim"), 4096, "stt.moss.adaptor.input_dim");
    require_value(meta.require_i32("stt.moss.audio_merge_size"), 4, "stt.moss.audio_merge_size");
    require_value(meta.require_i32("stt.moss.audio_token_id"), kAudioToken, "stt.moss.audio_token_id");
    // The Whisper feature extractor the runtime rebuilds (WhisperLogMelExtractor,
    // 30 s chunks) instead of reading frontend.mel_filterbank / frontend.window.
    if (meta.require_string("stt.frontend.type") != "mel") {
        throw std::runtime_error("Unsupported MOSS-Transcribe-Diarize frontend (stt.frontend.type is not mel)");
    }
    require_value(meta.require_i32("stt.frontend.sample_rate"), 16000, "stt.frontend.sample_rate");
    require_value(meta.require_i32("stt.frontend.num_mels"), 80, "stt.frontend.num_mels");
    require_value(meta.require_i32("stt.frontend.n_fft"), 400, "stt.frontend.n_fft");
    require_value(meta.require_i32("stt.frontend.hop_length"), 160, "stt.frontend.hop_length");
    require_value(meta.find_i32("stt.frontend.n_samples").value_or(kChunkSamples), kChunkSamples,
        "stt.frontend.n_samples");

    auto & config = out->config;
    config.vocab_size = meta.require_i32("stt.moss.decoder.vocab_size");
    config.max_context = meta.require_i32("stt.moss.decoder.max_position_embeddings");
    config.num_attention_heads = meta.require_i32("stt.moss.decoder.n_heads");
    config.num_key_value_heads = meta.require_i32("stt.moss.decoder.n_kv_heads");
    config.head_dim = meta.require_i32("stt.moss.decoder.head_dim");
    config.intermediate_size = meta.require_i32("stt.moss.decoder.intermediate_size");
    config.audio_tokens_per_second = meta.require_float("stt.moss.audio_tokens_per_second");
    config.time_marker_every_seconds = meta.require_i32("stt.moss.time_marker_every_seconds");
    config.enable_time_marker = meta.find_bool("stt.moss.enable_time_marker").value_or(true);
    config.default_max_tokens = 0;  // the arch's adaptive budget (see kArchMinBudget)

    const auto require_ids = [&meta](const char * key) {
        auto ids = meta.find_i32_array(key);
        if (!ids.has_value() || ids->empty()) {
            throw std::runtime_error(std::string("MOSS-Transcribe-Diarize transcribe.cpp GGUF has no ") + key);
        }
        return std::move(*ids);
    };
    out->prompt_prefix = require_ids("stt.moss.prompt_prefix_tokens");
    out->prompt_suffix = require_ids("stt.moss.prompt_suffix_tokens");
    out->digit_tokens = require_ids("stt.moss.digit_tokens");
    if (out->digit_tokens.size() != 10) {
        throw std::runtime_error("MOSS-Transcribe-Diarize stt.moss.digit_tokens must have 10 entries");
    }

    // tokenizer.ggml.*: llama.cpp "gpt2" byte-level BPE with the qwen2 split,
    // padded by the converter to the 151936-wide logits with <|unused_N|> rows.
    if (meta.require_string("tokenizer.ggml.model") != "gpt2" ||
        meta.find_string("tokenizer.ggml.pre").value_or("") != "qwen2") {
        throw std::runtime_error("MOSS-Transcribe-Diarize transcribe.cpp GGUF: expected a gpt2/qwen2 BPE tokenizer");
    }
    out->gguf_tokenizer = text::load_tokenizer_from_gguf(meta.context());
    if (out->gguf_tokenizer == nullptr ||
        static_cast<int64_t>(out->gguf_tokenizer->vocab_size()) != config.vocab_size) {
        throw std::runtime_error("MOSS-Transcribe-Diarize tokenizer.ggml.tokens does not match the decoder vocab_size");
    }
    require_value(out->gguf_tokenizer->specials().eos, kEosToken, "tokenizer.ggml.eos_token_id");
    // HF splits on its added tokens (ids from <|endoftext|> through the audio
    // placeholders, special or not) before BPE; the converter types only the
    // special ones as control, so take every row from the first control id on,
    // minus the converter's <|unused_N|> padding.
    int64_t first_added = -1;
    for (int64_t id = 0; id < config.vocab_size; ++id) {
        if (out->gguf_tokenizer->is_control(static_cast<int32_t>(id))) {
            first_added = id;
            break;
        }
    }
    if (first_added < 0) {
        throw std::runtime_error("MOSS-Transcribe-Diarize tokenizer.ggml.token_type marks no control tokens");
    }
    for (int64_t id = first_added; id < config.vocab_size; ++id) {
        const std::string piece(out->gguf_tokenizer->piece(static_cast<int32_t>(id)));
        if (piece.empty() || piece == "<|unused_" + std::to_string(id) + "|>") {
            continue;
        }
        out->added_tokens.emplace_back(piece, static_cast<int32_t>(id));
    }
    std::stable_sort(out->added_tokens.begin(), out->added_tokens.end(),
        [](const auto & lhs, const auto & rhs) { return lhs.first.size() > rhs.first.size(); });

    auto names = transcribe_to_package_names(encoder_layers, decoder_layers);
    out->source = assets::make_renamed_tensor_source(
        assets::open_tensor_source(path),
        [names = std::move(names)](std::string_view name) -> std::optional<std::string> {
            const auto it = names.find(std::string(name));
            if (it == names.end()) {
                return std::nullopt;  // frontend.* buffers: the frontend is rebuilt from constants
            }
            return it->second;
        });
    return out;
}

std::shared_ptr<const TranscribeAssets> load_package_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<TranscribeAssets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("moss_transcribe_diarize"));
    const auto & resources = out->resources;
    const auto config = resources.parse_json("config");
    if (io::json::require_string(config, "model_type") != "moss_transcribe_diarize") {
        throw std::runtime_error("MOSS-Transcribe-Diarize model_type mismatch");
    }
    const auto & text = config.require("text_config");
    const auto & encoder = config.require("audio_config");
    if (io::json::require_i64(text, "hidden_size") != 1024 ||
        io::json::require_i64(text, "num_hidden_layers") != 28 ||
        io::json::require_i64(encoder, "encoder_layers") != 24 ||
        io::json::require_i64(encoder, "encoder_attention_heads") != 16 ||
        io::json::require_i64(config, "audio_merge_size") != 4 ||
        io::json::require_i64(config, "audio_token_id") != kAudioToken) {
        throw std::runtime_error("Unsupported MOSS-Transcribe-Diarize architecture");
    }
    auto & values = out->config;
    values.vocab_size = io::json::require_i64(text, "vocab_size");
    values.max_context = io::json::require_i64(text, "max_position_embeddings");
    values.num_attention_heads = io::json::require_i64(text, "num_attention_heads");
    values.num_key_value_heads = io::json::require_i64(text, "num_key_value_heads");
    values.head_dim = io::json::require_i64(text, "head_dim");
    values.intermediate_size = io::json::require_i64(text, "intermediate_size");
    const auto processor = resources.parse_json("processor_config");
    values.audio_tokens_per_second = io::json::require_f32(processor, "audio_tokens_per_second");
    values.time_marker_every_seconds = io::json::require_i64(processor, "time_marker_every_seconds");
    values.enable_time_marker = io::json::require_bool(processor, "enable_time_marker");
    values.default_max_tokens = io::json::require_i64(resources.parse_json("generation_config"), "max_new_tokens");
    tokenizers::LlamaBpeTokenizerSpec tokenizer_config;
    tokenizer_config.vocab_path = resources.require_file("vocab");
    tokenizer_config.merges_path = resources.require_file("merges");
    tokenizer_config.tokenizer_config_path = resources.require_file("tokenizer_config");
    tokenizer_config.tokenizer_json_path = resources.require_file("tokenizer_json");
    tokenizer_config.pre_type = tokenizers::LlamaBpePreTokenizer::Qwen2;
    out->bpe = tokenizers::load_llama_bpe_tokenizer(tokenizer_config);
    out->source = resources.open_tensor_source("weights");
    return out;
}

// HF-style encode over the GGUF table: split on the added tokens (leftmost,
// longest first), byte-level BPE the text between them.
std::vector<int32_t> encode_with_added_tokens(const TranscribeAssets & assets, const std::string & text) {
    std::vector<int32_t> out;
    size_t plain_begin = 0;
    const auto flush = [&](size_t end) {
        if (end > plain_begin) {
            const auto ids = assets.gguf_tokenizer->encode(std::string_view(text).substr(plain_begin, end - plain_begin));
            out.insert(out.end(), ids.begin(), ids.end());
        }
    };
    for (size_t i = 0; i < text.size();) {
        const std::pair<std::string, int32_t> * match = nullptr;
        for (const auto & token : assets.added_tokens) {
            if (text.compare(i, token.first.size(), token.first) == 0) {
                match = &token;
                break;
            }
        }
        if (match == nullptr) {
            ++i;
            continue;
        }
        flush(i);
        out.push_back(match->second);
        i += match->first.size();
        plain_begin = i;
    }
    flush(text.size());
    return out;
}

}  // namespace

bool looks_like_transcribe_moss_gguf(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    if (!in.read(magic, 4) || std::string_view(magic, 4) != "GGUF") {
        return false;
    }
    in.close();
    return assets::GgufMetadata::open(path).find_string("general.architecture").value_or("") == "moss";
}

std::shared_ptr<const TranscribeAssets> load_transcribe_assets(const std::filesystem::path & path) {
    if (looks_like_transcribe_moss_gguf(path)) {
        return load_transcribe_gguf_assets(path);
    }
    return load_package_assets(path);
}

class TranscribeRuntime::Impl {
public:
    Impl(const TranscribeAssets & assets, core::ExecutionContext & execution)
        : execution_(execution), assets_(assets), source_(assets.source),
          store_(execution.backend(), execution.backend_type(), "moss_transcribe_diarize.weights", 2 * 1024 * 1024),
          mel_({16000, 400, 160, 80, audio::STFTFamily::Default}) {
        vocab_ = assets.config.vocab_size;
        max_context_ = assets.config.max_context;
        audio_rate_ = assets.config.audio_tokens_per_second;
        marker_seconds_ = assets.config.time_marker_every_seconds;
        time_markers_ = assets.config.enable_time_marker;

        modules::WhisperFrontendComponentConfig whisper_config;
        whisper_config.name = "moss_transcribe_diarize.encoder";
        whisper_config.weight_context_bytes = 2 * 1024 * 1024;
        whisper_config.graph_context_bytes = 16 * 1024 * 1024;
        whisper_config.conv_weight_storage_type = Storage::F32;
        whisper_ = modules::WhisperFrontendComponent::load_openai_layout(
            source_, execution.config(), {80, 1500, 1024, 16, 24, 1e-5f}, whisper_config);

        modules::QwenCausalDecodeRuntimeConfig ar_config;
        ar_config.trace_name = "moss_transcribe_diarize.decoder";
        ar_config.prefill_graph_arena_bytes = 32 * 1024 * 1024;
        ar_config.decode_graph_arena_bytes = 32 * 1024 * 1024;
        ar_config.evict_cuda_graph_cache_on_release = true;
        auto & stack = ar_config.decoder.stack;
        stack.hidden_size = kHidden;
        stack.num_attention_heads = assets.config.num_attention_heads;
        stack.num_key_value_heads = assets.config.num_key_value_heads;
        stack.head_dim = assets.config.head_dim;
        stack.intermediate_size = assets.config.intermediate_size;
        stack.layers = 28;
        stack.rms_norm_eps = 1e-6f;
        stack.rope_theta = 1000000.f;
        stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
        ar_config.decoder.static_cache_type = GGML_TYPE_F16;
        ar_config.decoder.logits_size = vocab_;

        modules::QwenCausalDecodeRuntimeWeights ar_weights;
        const std::string prefix = "model.language_model";
        ar_weights.token_embedding = store_.load_tensor(*source_, prefix + ".embed_tokens.weight", Storage::Native, {vocab_, kHidden});
        for (int64_t i = 0; i < stack.layers; ++i) {
            const std::string p = prefix + ".layers." + std::to_string(i);
            modules::QwenDecoderLayerWeights layer;
            layer.input_norm = binding::norm_weight_from_source(store_, *source_, p + ".input_layernorm", kHidden);
            layer.post_norm = binding::norm_weight_from_source(store_, *source_, p + ".post_attention_layernorm", kHidden);
            layer.q_norm = binding::norm_weight_from_source(store_, *source_, p + ".self_attn.q_norm", stack.head_dim);
            layer.k_norm = binding::norm_weight_from_source(store_, *source_, p + ".self_attn.k_norm", stack.head_dim);
            layer.self_attention.q_weight = store_.load_tensor(*source_, p + ".self_attn.q_proj.weight", Storage::Native,
                {stack.num_attention_heads * stack.head_dim, kHidden});
            layer.self_attention.k_weight = store_.load_tensor(*source_, p + ".self_attn.k_proj.weight", Storage::Native,
                {stack.num_key_value_heads * stack.head_dim, kHidden});
            layer.self_attention.v_weight = store_.load_tensor(*source_, p + ".self_attn.v_proj.weight", Storage::Native,
                {stack.num_key_value_heads * stack.head_dim, kHidden});
            layer.self_attention.out_weight = store_.load_tensor(*source_, p + ".self_attn.o_proj.weight", Storage::Native,
                {kHidden, stack.num_attention_heads * stack.head_dim});
            layer.mlp.gate_proj = binding::linear_from_source(store_, *source_, p + ".mlp.gate_proj", Storage::Native,
                stack.intermediate_size, kHidden, false);
            layer.mlp.up_proj = binding::linear_from_source(store_, *source_, p + ".mlp.up_proj", Storage::Native,
                stack.intermediate_size, kHidden, false);
            layer.mlp.down_proj = binding::linear_from_source(store_, *source_, p + ".mlp.down_proj", Storage::Native,
                kHidden, stack.intermediate_size, false);
            ar_weights.stack.layers.push_back(std::move(layer));
        }
        ar_weights.final_norm = binding::norm_weight_from_source(store_, *source_, prefix + ".norm", kHidden);
        ar_weights.lm_head = modules::LinearWeights{ar_weights.token_embedding, std::nullopt};
        const auto first = binding::linear_from_source(store_, *source_, "model.vq_adaptor.layers.0", Storage::Native, kHidden, 4096, true);
        const auto second = binding::linear_from_source(store_, *source_, "model.vq_adaptor.layers.2", Storage::Native, kHidden, kHidden, true);
        const auto norm = binding::norm_from_source(store_, *source_, "model.vq_adaptor.layers.3", kHidden);
        store_.upload();
        source_->release_storage();
        ar_ = std::make_unique<modules::QwenCausalDecodeRuntime>(execution_, ar_config, ar_weights);

        ctx_.reset(ggml_init({2 * 1024 * 1024, nullptr, true}));
        if (!ctx_) {
            throw std::runtime_error("MOSS-Transcribe-Diarize adaptor context allocation failed");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "moss_transcribe_diarize", execution.backend_type()};
        adaptor_input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kAudioFrames, 4096}));
        ggml_set_input(adaptor_input_.tensor);
        auto projected = modules::LinearModule({4096, kHidden, true}).build(ctx, adaptor_input_, first);
        projected = modules::SiluModule{}.build(ctx, projected);
        projected = modules::LinearModule({kHidden, kHidden, true}).build(ctx, projected, second);
        adaptor_output_ = modules::LayerNormModule({kHidden, 1e-6f, true, true}).build(ctx, projected, norm);
        ggml_set_output(adaptor_output_.tensor);
        adaptor_graph_ = ggml_new_graph_custom(ctx_.get(), 256, false);
        ggml_build_forward_expand(adaptor_graph_, adaptor_output_.tensor);
        adaptor_allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(adaptor_allocator_.get(), adaptor_graph_)) {
            throw std::runtime_error("MOSS-Transcribe-Diarize adaptor allocation failed");
        }
        core::prepare_host_graph_plan(execution_, adaptor_graph_, adaptor_plan_);
        lookup_input_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({kLookupSteps}));
        ggml_set_input(lookup_input_.tensor);
        lookup_output_ = modules::EmbeddingModule({vocab_, kHidden}).build(ctx, lookup_input_, ar_weights.token_embedding);
        ggml_set_output(lookup_output_.tensor);
        lookup_graph_ = ggml_new_graph_custom(ctx_.get(), 32, false);
        ggml_build_forward_expand(lookup_graph_, lookup_output_.tensor);
        lookup_allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(lookup_allocator_.get(), lookup_graph_)) {
            throw std::runtime_error("MOSS-Transcribe-Diarize embedding allocation failed");
        }
        core::prepare_host_graph_plan(execution_, lookup_graph_, lookup_plan_);
    }

    ~Impl() {
        core::release_backend_graph_resources(execution_.backend(), adaptor_graph_, true);
        core::release_backend_graph_resources(execution_.backend(), lookup_graph_, true);
    }

    void start(const runtime::AudioBuffer & audio, const std::string & instruction, std::optional<int64_t> max_tokens,
               const TranscribePoll & poll) {
        reset();
        poll_ = poll;
        const auto started = std::chrono::steady_clock::now();
        auto samples = audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples, audio.sample_rate, audio.channels, audio.sample_rate);
        if (audio.sample_rate != 16000) {
            audio::SoxrResampleOptions options;
            options.output_length_policy = audio::SoxrOutputLengthPolicy::ExactExpected;
            options.require_full_input = true;
            auto resampled = audio::try_resample_mono_soxr(samples, audio.sample_rate, 16000, options);
            if (!resampled) {
                throw std::runtime_error("MOSS-Transcribe-Diarize requires SOXR for non-16-kHz audio");
            }
            samples = std::move(*resampled);
        }
        if (samples.empty() || (max_tokens.has_value() && *max_tokens <= 0)) {
            throw std::runtime_error("MOSS-Transcribe-Diarize requires nonempty audio and positive max_tokens");
        }
        const int64_t audio_tokens = (static_cast<int64_t>(samples.size()) + 1279) / 1280;
        auto ids = prompt_prefix();
        std::vector<size_t> audio_positions;
        audio_positions.reserve(audio_tokens);
        const int64_t marker_stride = static_cast<int64_t>(audio_rate_ * marker_seconds_);
        const int64_t markers = time_markers_ && marker_seconds_ > 0 && marker_stride > 0
            ? static_cast<int64_t>(audio_tokens / audio_rate_) / marker_seconds_ : 0;
        for (int64_t token = 0; token < audio_tokens; ++token) {
            audio_positions.push_back(ids.size());
            ids.push_back(kAudioToken);
            if (markers > 0 && (token + 1) % marker_stride == 0 && (token + 1) / marker_stride <= markers) {
                const auto marker = encode_marker((token + 1) / marker_stride * marker_seconds_);
                ids.insert(ids.end(), marker.begin(), marker.end());
            }
        }
        const auto suffix = prompt_suffix(instruction);
        ids.insert(ids.end(), suffix.begin(), suffix.end());
        // The decode budget, clamped to the context left after the prompt; running
        // it out is an honest truncation, not an error. An explicit max_tokens
        // (or the package's generation_config default) is used as given; the
        // transcribe.cpp GGUF defaults to the arch's adaptive budget, including
        // its input-too-long rule (prompt + 256 must fit the context).
        const int64_t prompt_tokens = static_cast<int64_t>(ids.size());
        const int64_t room = max_context_ - prompt_tokens;
        int64_t budget = 0;
        if (max_tokens.has_value() || assets_.config.default_max_tokens > 0) {
            if (room <= 0) {
                throw std::runtime_error("MOSS-Transcribe-Diarize audio/prompt exceeds the decoder context");
            }
            budget = std::min(max_tokens.value_or(assets_.config.default_max_tokens), room);
        } else {
            if (prompt_tokens + kArchMinBudget > max_context_) {
                throw engine::runtime::InputTooLong("MOSS-Transcribe-Diarize input too long: " + std::to_string(prompt_tokens) +
                    " prompt + " + std::to_string(kArchMinBudget) + " generation tokens exceed the " +
                    std::to_string(max_context_) + "-token context");
            }
            budget = std::min(std::max(kArchMinBudget, 2 * audio_tokens + 128), room);
        }
        std::vector<float> embeddings(ids.size() * kHidden);
        std::vector<int32_t> block(kLookupSteps, 0);
        for (size_t offset = 0; offset < ids.size(); offset += kLookupSteps) {
            const size_t count = std::min<size_t>(kLookupSteps, ids.size() - offset);
            std::copy_n(ids.begin() + offset, count, block.begin());
            core::write_tensor_i32(lookup_input_, block);
            if (core::compute_graph(execution_, lookup_graph_, lookup_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MOSS-Transcribe-Diarize embedding compute failed");
            }
            const auto values = core::read_tensor_f32(lookup_output_.tensor);
            std::copy_n(values.begin(), count * kHidden, embeddings.begin() + offset * kHidden);
        }
        std::vector<float> chunk(kChunkSamples, 0.f);
        size_t audio_offset = 0;
        const int64_t chunks = (static_cast<int64_t>(samples.size()) + kChunkSamples - 1) / kChunkSamples;
        for (size_t offset = 0; offset < samples.size(); offset += kChunkSamples) {
            if (poll_) {
                poll_("moss_transcribe_diarize.encode", static_cast<int64_t>(offset) / kChunkSamples, chunks);
            }
            const size_t count = std::min<size_t>(kChunkSamples, samples.size() - offset);
            std::fill(chunk.begin(), chunk.end(), 0.f);
            std::copy_n(samples.begin() + offset, count, chunk.begin());
            auto mel = mel_.compute(chunk, execution_.config().threads);
            const auto encoded = whisper_.encode_log_mel(mel.values);
            core::write_tensor_f32(adaptor_input_, encoded);
            if (core::compute_graph(execution_, adaptor_graph_, adaptor_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MOSS-Transcribe-Diarize adaptor compute failed");
            }
            const auto adapted = core::read_tensor_f32(adaptor_output_.tensor);
            const size_t tokens = (count + 1279) / 1280;
            for (size_t t = 0; t < tokens; ++t) {
                std::copy_n(adapted.begin() + t * kHidden, kHidden,
                    embeddings.begin() + audio_positions.at(audio_offset++) * kHidden);
            }
        }
        debug::timing_log_scalar("moss_transcribe_diarize.frontend_ms", debug::elapsed_ms(started));
        const auto prefill_started = std::chrono::steady_clock::now();
        logits_ = ar_->prefill_embeddings_into_cache(embeddings, static_cast<int64_t>(ids.size()),
            static_cast<int64_t>(ids.size()) + budget, 128).logits;
        debug::timing_log_scalar("moss_transcribe_diarize.prefill_ms", debug::elapsed_ms(prefill_started));
        max_tokens_ = budget;
        active_ = true;
        debug::trace_log_scalar("moss_transcribe_diarize.prompt_tokens", static_cast<int64_t>(ids.size()));
    }

    std::optional<std::string> next_text() {
        if (!active_) {
            return std::nullopt;
        }
        while (static_cast<int64_t>(generated_.size()) < max_tokens_) {
            if (poll_) {
                poll_("moss_transcribe_diarize.decode", static_cast<int64_t>(generated_.size()), max_tokens_);
            }
            if (!generated_.empty()) {
                logits_ = ar_->decode_token(generated_.back()).logits;
            }
            const int32_t token = sampling::HfLogitsProcessor::argmax(logits_.data(), logits_.size(), "MOSS-Transcribe-Diarize");
            if (token == kEosToken) {
                active_ = false;
                debug::trace_log_scalar("moss_transcribe_diarize.generated_tokens", static_cast<int64_t>(generated_.size()));
                if (partials_.published().size() != decoded_text_.size()) {
                    throw std::runtime_error("MOSS-Transcribe-Diarize ended with incomplete UTF-8 text");
                }
                return std::nullopt;
            }
            generated_.push_back(token);
            // BPE decoding is byte-concatenative; the publisher holds incomplete UTF-8 tails.
            decoded_text_ += decode_token(token);
            auto delta = partials_.publish(decoded_text_);
            if (!delta.empty()) {
                return delta;
            }
        }
        // Budget exhausted before EOS: the published text stands as the partial
        // result (an incomplete UTF-8 tail cut by the budget is dropped).
        active_ = false;
        truncated_ = true;
        debug::trace_log_scalar("moss_transcribe_diarize.generated_tokens", static_cast<int64_t>(generated_.size()));
        return std::nullopt;
    }

    bool truncated() const { return truncated_; }

    void reset() {
        active_ = false;
        truncated_ = false;
        generated_.clear();
        logits_.clear();
        decoded_text_.clear();
        partials_.reset();
        max_tokens_ = 0;
        poll_ = nullptr;
    }

private:
    // The two tokenizers: the audio.cpp package's HF BPE files encode the chat
    // template directly; the transcribe.cpp GGUF uses the converter's baked
    // default-prompt ids (what the arch fed the model) and digit ids, and
    // encodes a caller instruction over its tokenizer.ggml.* table.
    std::vector<int32_t> prompt_prefix() const {
        if (assets_.bpe) {
            return assets_.bpe->encode(kPromptPrefix);
        }
        return assets_.prompt_prefix;
    }

    std::vector<int32_t> prompt_suffix(const std::string & instruction) const {
        const std::string text = std::string(kPromptSuffixHead) + (instruction.empty() ? kDefaultPrompt : instruction) +
            kPromptSuffixTail;
        if (assets_.bpe) {
            return assets_.bpe->encode(text);
        }
        return instruction.empty() ? assets_.prompt_suffix : encode_with_added_tokens(assets_, text);
    }

    std::vector<int32_t> encode_marker(int64_t seconds) const {
        const auto digits = std::to_string(seconds);
        if (assets_.bpe) {
            return assets_.bpe->encode(digits);
        }
        std::vector<int32_t> out;
        for (const char digit : digits) {
            out.push_back(assets_.digit_tokens.at(static_cast<size_t>(digit - '0')));
        }
        return out;
    }

    std::string decode_token(int32_t token) const {
        if (assets_.bpe) {
            return assets_.bpe->decode({token}, true);
        }
        return assets_.gguf_tokenizer->is_control(token) ? std::string() : assets_.gguf_tokenizer->decode(&token, 1);
    }

    core::ExecutionContext & execution_;
    const TranscribeAssets & assets_;
    std::shared_ptr<const assets::TensorSource> source_;
    core::BackendWeightStore store_;
    audio::WhisperLogMelExtractor mel_;
    modules::WhisperFrontendComponent whisper_;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> ar_;
    int64_t vocab_ = 0;
    bool active_ = false;
    bool truncated_ = false;
    TranscribePoll poll_;
    int64_t max_tokens_ = 0;
    std::string decoded_text_;
    runtime::PartialTextPublisher partials_;
    std::vector<int32_t> generated_;
    std::vector<float> logits_;
    int64_t max_context_ = 0;
    float audio_rate_ = 12.5f;
    int64_t marker_seconds_ = 0;
    bool time_markers_ = false;
    std::unique_ptr<ggml_context, ContextDeleter> ctx_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> adaptor_allocator_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> lookup_allocator_;
    core::HostGraphPlan adaptor_plan_;
    core::HostGraphPlan lookup_plan_;
    core::TensorValue adaptor_input_, adaptor_output_, lookup_input_, lookup_output_;
    ggml_cgraph * adaptor_graph_ = nullptr;
    ggml_cgraph * lookup_graph_ = nullptr;
};

TranscribeRuntime::TranscribeRuntime(const TranscribeAssets & assets, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(assets, execution)) {}
TranscribeRuntime::~TranscribeRuntime() = default;
std::string TranscribeRuntime::transcribe(const runtime::AudioBuffer & audio, const std::string & instruction, std::optional<int64_t> max_tokens,
                                          const TranscribePoll & poll) {
    start(audio, instruction, max_tokens, poll);
    std::string text;
    while (auto delta = next_text()) {
        text += *delta;
    }
    return text;
}
void TranscribeRuntime::start(const runtime::AudioBuffer & audio, const std::string & instruction, std::optional<int64_t> max_tokens,
                              const TranscribePoll & poll) {
    impl_->start(audio, instruction, max_tokens, poll);
}
std::optional<std::string> TranscribeRuntime::next_text() { return impl_->next_text(); }
bool TranscribeRuntime::truncated() const { return impl_->truncated(); }
void TranscribeRuntime::reset() { impl_->reset(); }

}  // namespace engine::models::moss_transcribe_diarize
