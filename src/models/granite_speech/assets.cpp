// granite_speech assets: the transcribe.cpp GGUF contract.
//
// Ported from src/runtime/arch/granite/weights.cpp:read_granite_hparams (every
// KV, its required/optional status and the cross-field invariants) and
// src/runtime/arch/granite/model.cpp:load (capability KVs, tokenizer, chat
// tokens, frontend buffers, the decode-budget rate basis). The GGUF is written
// by transcribe.cpp scripts/convert-granite.py; the published files are
// handy-computer/granite-4.0-1b-speech-gguf, granite-speech-4.1-2b-gguf and
// granite-speech-4.1-2b-plus-gguf.

#include "engine/models/granite_speech/model.h"

#include "engine/framework/assets/gguf_metadata.h"

#include "gguf.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace engine::models::granite_speech {
namespace {

constexpr const char * kArchitecture = "granite_speech";

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("Granite Speech GGUF: " + message);
}

int32_t positive(const assets::GgufMetadata & meta, std::string_view key) {
    const int32_t value = meta.require_i32(key);
    if (value <= 0) {
        fail(std::string(key) + " must be positive (got " + std::to_string(value) + ")");
    }
    return value;
}

// stt.granite.encoder.cat_hidden_layers is an optional uint32 array; absent or
// empty both mean "no concat" (arch read_int32_array_kv Absent/Ok). An empty
// array may carry any element type, so check the length before the type.
std::vector<int32_t> read_cat_hidden_layers(const assets::GgufMetadata & meta) {
    constexpr const char * key = "stt.granite.encoder.cat_hidden_layers";
    const gguf_context * ctx = meta.context();
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return {};
    }
    if (gguf_get_kv_type(ctx, id) == GGUF_TYPE_ARRAY && gguf_get_arr_n(ctx, id) == 0) {
        return {};
    }
    return meta.find_i32_array(key).value_or(std::vector<int32_t>{});
}

GraniteSpeechHParams read_hparams(const assets::GgufMetadata & meta) {
    GraniteSpeechHParams hp;
    hp.variant = meta.find_string("stt.variant").value_or("");

    hp.enc_n_layers = positive(meta, "stt.granite.encoder.n_layers");
    hp.enc_hidden = positive(meta, "stt.granite.encoder.hidden");
    hp.enc_n_heads = positive(meta, "stt.granite.encoder.n_heads");
    hp.enc_head_dim = positive(meta, "stt.granite.encoder.head_dim");
    hp.enc_input_dim = positive(meta, "stt.granite.encoder.input_dim");
    hp.enc_output_dim = positive(meta, "stt.granite.encoder.output_dim");
    hp.enc_feedforward_mult = positive(meta, "stt.granite.encoder.feedforward_mult");
    hp.enc_conv_kernel_size = positive(meta, "stt.granite.encoder.conv_kernel_size");
    hp.enc_conv_expansion = positive(meta, "stt.granite.encoder.conv_expansion");
    hp.enc_max_pos_emb = positive(meta, "stt.granite.encoder.max_pos_emb");
    hp.enc_context_size = positive(meta, "stt.granite.encoder.context_size");
    hp.enc_cat_hidden_layers = read_cat_hidden_layers(meta);

    hp.prj_n_layers = positive(meta, "stt.granite.projector.n_layers");
    hp.prj_hidden = positive(meta, "stt.granite.projector.hidden");
    hp.prj_intermediate = positive(meta, "stt.granite.projector.intermediate");
    hp.prj_n_heads = positive(meta, "stt.granite.projector.n_heads");
    hp.prj_encoder_hidden_size = positive(meta, "stt.granite.projector.encoder_hidden_size");
    hp.prj_cross_attn_freq = meta.require_i32("stt.granite.projector.cross_attn_frequency");
    const auto prj_act = meta.require_string("stt.granite.projector.hidden_act");
    hp.prj_layer_norm_eps = meta.require_float("stt.granite.projector.layer_norm_eps");
    (void) meta.require_i32("stt.granite.projector.max_pos_emb");
    const auto prj_pos = meta.require_string("stt.granite.projector.position_embedding_type");

    hp.dec_n_layers = positive(meta, "stt.granite.decoder.n_layers");
    hp.dec_hidden = positive(meta, "stt.granite.decoder.hidden_size");
    hp.dec_intermediate = positive(meta, "stt.granite.decoder.intermediate_size");
    hp.dec_n_heads = positive(meta, "stt.granite.decoder.n_heads");
    hp.dec_n_kv_heads = positive(meta, "stt.granite.decoder.n_kv_heads");
    hp.dec_head_dim = positive(meta, "stt.granite.decoder.head_dim");
    const auto dec_act = meta.require_string("stt.granite.decoder.hidden_act");
    hp.dec_rms_norm_eps = meta.require_float("stt.granite.decoder.rms_norm_eps");
    hp.dec_rope_theta = meta.require_float("stt.granite.decoder.rope_theta");
    hp.dec_max_position_embeddings = positive(meta, "stt.granite.decoder.max_position_embeddings");
    hp.dec_tie_word_embeddings = meta.find_bool("stt.granite.decoder.tie_word_embeddings").value_or(false);
    hp.dec_vocab_size = positive(meta, "stt.granite.decoder.vocab_size");
    hp.dec_embedding_multiplier = meta.require_float("stt.granite.decoder.embedding_multiplier");
    hp.dec_logits_scaling = meta.require_float("stt.granite.decoder.logits_scaling");
    hp.dec_attention_multiplier = meta.require_float("stt.granite.decoder.attention_multiplier");
    hp.dec_residual_multiplier = meta.require_float("stt.granite.decoder.residual_multiplier");

    hp.audio_token_id = meta.require_i32("stt.granite.audio_token_id");
    hp.downsample_rate = positive(meta, "stt.granite.downsample_rate");
    hp.window_size = positive(meta, "stt.granite.window_size");

    // Cross-field invariants (arch read_granite_hparams).
    if (hp.enc_n_heads * hp.enc_head_dim != hp.enc_hidden) {
        fail("encoder n_heads * head_dim (" + std::to_string(hp.enc_n_heads) + " * " +
             std::to_string(hp.enc_head_dim) + ") != hidden (" + std::to_string(hp.enc_hidden) + ")");
    }
    if (hp.enc_context_size > hp.enc_max_pos_emb) {
        fail("encoder context_size (" + std::to_string(hp.enc_context_size) + ") > max_pos_emb (" +
             std::to_string(hp.enc_max_pos_emb) + ")");
    }
    const int32_t expected_kv_dim =
        hp.enc_hidden * (static_cast<int32_t>(hp.enc_cat_hidden_layers.size()) + 1);
    if (expected_kv_dim != hp.prj_encoder_hidden_size) {
        fail("projector encoder_hidden_size (" + std::to_string(hp.prj_encoder_hidden_size) +
             ") != encoder hidden * (1 + |cat_hidden_layers|) (" + std::to_string(expected_kv_dim) + ")");
    }
    for (const int32_t idx : hp.enc_cat_hidden_layers) {
        if (idx < 0 || idx >= hp.enc_n_layers) {
            fail("cat_hidden_layers entry " + std::to_string(idx) + " outside [0, " +
                 std::to_string(hp.enc_n_layers) + ")");
        }
    }
    if (hp.prj_hidden % hp.prj_n_heads != 0) {
        fail("projector hidden (" + std::to_string(hp.prj_hidden) + ") not divisible by n_heads (" +
             std::to_string(hp.prj_n_heads) + ")");
    }
    if (prj_act != "gelu") {
        fail("unsupported projector hidden_act \"" + prj_act + "\" (only gelu)");
    }
    if (prj_pos != "absolute") {
        fail("unsupported projector position_embedding_type \"" + prj_pos + "\" (only absolute)");
    }
    if (hp.prj_cross_attn_freq != 1) {
        // Every shipped variant (and the arch's projector graph) puts a
        // cross-attention in every Q-Former layer.
        fail("unsupported projector cross_attn_frequency " + std::to_string(hp.prj_cross_attn_freq) + " (only 1)");
    }
    if (hp.dec_n_heads % hp.dec_n_kv_heads != 0) {
        fail("decoder n_heads (" + std::to_string(hp.dec_n_heads) + ") not divisible by n_kv_heads (" +
             std::to_string(hp.dec_n_kv_heads) + ")");
    }
    if (hp.dec_n_heads * hp.dec_head_dim != hp.dec_hidden) {
        fail("decoder n_heads * head_dim != hidden");
    }
    if (dec_act != "silu" && dec_act != "swish") {
        fail("unsupported decoder hidden_act \"" + dec_act + "\" (only silu/swish)");
    }
    if (!(hp.dec_embedding_multiplier > 0.0f) || !(hp.dec_logits_scaling > 0.0f) ||
        !(hp.dec_attention_multiplier > 0.0f) || !(hp.dec_residual_multiplier > 0.0f)) {
        // Each Granite-4 multiplier is silently load-bearing.
        fail("decoder embedding/logits/attention/residual multipliers must be > 0");
    }
    if (hp.window_size % hp.downsample_rate != 0) {
        fail("window_size (" + std::to_string(hp.window_size) + ") not divisible by downsample_rate (" +
             std::to_string(hp.downsample_rate) + ")");
    }
    hp.prj_num_queries = hp.window_size / hp.downsample_rate;
    return hp;
}

GraniteSpeechFrontend read_frontend(const assets::GgufMetadata & meta, const assets::TensorSource & source) {
    if (meta.require_string("stt.frontend.type") != "mel") {
        fail("unsupported stt.frontend.type (only mel)");
    }
    const auto normalize = meta.require_string("stt.frontend.normalize");
    if (normalize != "per_utterance") {
        // GraniteSpeechFeatureExtractor = log10 -> (max - 8) floor -> /4 + 1.
        fail("unsupported stt.frontend.normalize \"" + normalize + "\" (only per_utterance)");
    }
    const auto mel_norm = meta.find_string("stt.frontend.mel_norm").value_or("htk");
    if (mel_norm != "htk") {
        fail("unsupported stt.frontend.mel_norm \"" + mel_norm + "\" (only htk)");
    }
    GraniteSpeechFrontend fe;
    fe.n_mels = positive(meta, "stt.frontend.num_mels");
    fe.sample_rate = positive(meta, "stt.frontend.sample_rate");
    fe.n_fft = positive(meta, "stt.frontend.n_fft");
    fe.win_length = positive(meta, "stt.frontend.win_length");
    fe.hop_length = positive(meta, "stt.frontend.hop_length");
    const auto window_type = meta.require_string("stt.frontend.window");
    (void) meta.require_float("stt.frontend.dither");  // required KV; the arch never dithers
    fe.pre_emphasis = meta.require_float("stt.frontend.pre_emphasis");
    (void) meta.require_float("stt.frontend.f_min");
    (void) meta.require_float("stt.frontend.f_max");
    fe.pad_mode = meta.find_string("stt.frontend.pad_mode").value_or("reflect");
    const bool center = meta.find_bool("stt.frontend.center").value_or(true);

    if (fe.sample_rate != 16000) {
        fail("stt.frontend.sample_rate must be 16000");
    }
    if ((fe.n_fft & (fe.n_fft - 1)) != 0) {
        // The arch's non-power-of-two branch is the fp32 mixed-radix FFT;
        // every granite checkpoint is n_fft = 512 and only the fp64 radix-2
        // branch is ported.
        fail("stt.frontend.n_fft must be a power of two (got " + std::to_string(fe.n_fft) + ")");
    }
    if (fe.win_length > fe.n_fft) {
        fail("stt.frontend.win_length exceeds n_fft");
    }
    if (!center || (fe.pad_mode != "reflect" && fe.pad_mode != "constant")) {
        fail("unsupported frontend padding (need center=true with pad_mode reflect or constant)");
    }

    // Window: the converter-baked frontend.window (bit-identical to the
    // reference), centred inside n_fft; else torch.hann_window(win_length,
    // periodic=(window == "hann_periodic")) (MelFrontend ctor).
    fe.window.assign(static_cast<size_t>(fe.n_fft), 0.0);
    const int left_pad = (fe.n_fft - fe.win_length) / 2;
    if (source.has_tensor("frontend.window")) {
        const auto window = source.require_f32("frontend.window");
        if (window.size() != static_cast<size_t>(fe.win_length)) {
            fail("frontend.window has " + std::to_string(window.size()) + " values, expected win_length");
        }
        for (int i = 0; i < fe.win_length; ++i) {
            fe.window[static_cast<size_t>(left_pad + i)] = static_cast<double>(window[static_cast<size_t>(i)]);
        }
    } else {
        const bool periodic = window_type == "hann_periodic";
        const double denom = periodic ? static_cast<double>(fe.win_length) : static_cast<double>(fe.win_length - 1);
        for (int k = 0; k < fe.win_length; ++k) {
            fe.window[static_cast<size_t>(left_pad + k)] = 0.5 - 0.5 * std::cos(2.0 * M_PI * k / denom);
        }
    }

    // Filterbank: the converter-baked librosa HTK bank. The arch would fall
    // back to a Slaney bank when absent - wrong for an htk checkpoint - so it
    // is required here.
    const size_t n_freq = static_cast<size_t>(fe.n_fft / 2 + 1);
    if (!source.has_tensor("frontend.mel_filterbank")) {
        fail("missing frontend.mel_filterbank");
    }
    fe.filterbank = source.require_f32("frontend.mel_filterbank");
    if (fe.filterbank.size() != static_cast<size_t>(fe.n_mels) * n_freq) {
        fail("frontend.mel_filterbank has " + std::to_string(fe.filterbank.size()) +
             " values, expected num_mels * (n_fft/2 + 1)");
    }
    // Per-band nonzero support (trailing zeros trimmed first so an all-zero
    // band collapses to an empty span).
    fe.fb_begin.assign(static_cast<size_t>(fe.n_mels), 0);
    fe.fb_end.assign(static_cast<size_t>(fe.n_mels), 0);
    for (int m = 0; m < fe.n_mels; ++m) {
        const float * row = fe.filterbank.data() + static_cast<size_t>(m) * n_freq;
        int lo = 0;
        while (lo < static_cast<int>(n_freq) && row[lo] == 0.0f) {
            ++lo;
        }
        int hi = static_cast<int>(n_freq);
        while (hi > lo && row[hi - 1] == 0.0f) {
            --hi;
        }
        fe.fb_begin[static_cast<size_t>(m)] = lo;
        fe.fb_end[static_cast<size_t>(m)] = hi;
    }
    return fe;
}

}  // namespace

bool looks_like_transcribe_granite_speech_gguf(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    if (!in.read(magic, 4) || std::string_view(magic, 4) != "GGUF") {
        return false;
    }
    in.close();
    return assets::GgufMetadata::open(path).find_string("general.architecture").value_or("") == kArchitecture;
}

std::shared_ptr<const GraniteSpeechAssets> load_granite_speech_assets(const std::filesystem::path & path) {
    if (!looks_like_transcribe_granite_speech_gguf(path)) {
        throw std::runtime_error("Granite Speech expects a transcribe.cpp GGUF with general.architecture \"" +
                                 std::string(kArchitecture) + "\": " + path.string());
    }
    auto out = std::make_shared<GraniteSpeechAssets>();
    out->resources = assets::ResourceBundle(path.parent_path());
    out->path = path;
    const auto meta = assets::GgufMetadata::open(path);
    out->source = assets::open_tensor_source(path);
    out->hparams = read_hparams(meta);
    out->frontend = read_frontend(meta, *out->source);
    const auto & hp = out->hparams;
    if (hp.enc_input_dim != 2 * out->frontend.n_mels) {
        fail("encoder input_dim (" + std::to_string(hp.enc_input_dim) + ") != 2 * num_mels (2-frame stack)");
    }

    // Capability KVs (arch model.cpp:load + read_capability_kv).
    out->can_translate = meta.find_bool("stt.capability.translate").value_or(false);
    out->has_word_timestamps = meta.find_bool("stt.capability.word_timestamps").value_or(false);
    out->has_speaker_attribution = meta.find_bool("stt.capability.speaker_diarization").value_or(false);
    // general.languages (arch transcribe-meta.cpp:read_languages_kv).
    out->languages = meta.find_string_array("general.languages").value_or(std::vector<std::string>{});

    // Tokenizer + chat tokens (arch resolve_chat_tokens: audio / end_of_text /
    // pad are required on every variant; the role markers only exist on -plus).
    out->tokenizer.load(path);
    if (out->tokenizer.size() != hp.dec_vocab_size) {
        fail("tokenizer vocab (" + std::to_string(out->tokenizer.size()) + ") != decoder vocab_size (" +
             std::to_string(hp.dec_vocab_size) + ")");
    }
    auto & chat = out->chat_tokens;
    const auto require_piece = [&](const char * piece) {
        const int32_t id = out->tokenizer.find(piece);
        if (id < 0) {
            fail(std::string("tokenizer is missing required piece ") + piece);
        }
        return id;
    };
    chat.audio = require_piece("<|audio|>");
    chat.end_of_text = require_piece("<|end_of_text|>");
    chat.pad = require_piece("<|pad|>");
    chat.start_of_role = out->tokenizer.find("<|start_of_role|>");
    chat.end_of_role = out->tokenizer.find("<|end_of_role|>");
    out->chat_template = meta.find_string("tokenizer.chat_template").value_or("");

    // ms of audio per LM audio token (arch model.cpp:load limits basis):
    // num_queries tokens per window_size encoder frames, 2 mel frames per
    // encoder frame, hop_length samples per mel frame.
    out->ms_per_audio_token = (static_cast<double>(hp.window_size) / hp.prj_num_queries) * 2.0 *
                              out->frontend.hop_length * 1000.0 / out->frontend.sample_rate;
    return out;
}

}  // namespace engine::models::granite_speech
