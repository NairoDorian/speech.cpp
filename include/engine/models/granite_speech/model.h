#pragma once

// engine `granite_speech` package: IBM Granite Speech, the autoregressive
// audio-LLM (granite-4.0-1b-speech, granite-speech-4.1-2b,
// granite-speech-4.1-2b-plus).
//
// Ported from the transcribe.cpp arch src/runtime/arch/granite (the retirement
// path of roadmap v6: the engine package must open the SAME GGUF files the
// arch opened and produce the same transcripts). The GGUF layout is the one
// transcribe.cpp scripts/convert-granite.py writes:
//   general.architecture == "granite_speech", stt.granite.* / stt.frontend.*
//   KVs, enc.* / proj.* / dec.* tensors, frontend.mel_filterbank +
//   frontend.window buffers, a GGUF "gpt2" byte-level BPE tokenizer.
//
// Pipeline (op-for-op with the arch; see each .cpp for the ported function):
//   frontend.cpp  host log-mel (torchaudio MelSpectrogram + whisper-style
//                 per-utterance normalisation) and the 2-frame stack
//   runtime.cpp   Conformer encoder (Shaw block-local attention via the shared
//                 engine::modules::build_shaw_block_attention, GLU conv module
//                 with fused BatchNorm, self-conditioned CTC bypass, optional
//                 cat_hidden_layers) -> BLIP-2 Q-Former window projector ->
//                 Granite-4 causal LM (embedding / attention / residual /
//                 logits multipliers, NeoX RoPE, GQA) greedy decode
//   prompt.cpp    chat-template affixes and the -plus result post-processing
//                 ("[T:N]" word timestamps, "[Speaker N]:" turns)
//   tokenizer.cpp GGUF byte-level BPE with the granite / gpt2 pretokenizers
//
// The NAR editor variant (granite-speech-4.1-2b-nar, arch granite_nar) is a
// different architecture and not handled here.

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/attention/shaw_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/runtime/model.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::granite_speech {

// ---------------------------------------------------------------------------
// Hyperparameters (arch weights.h GraniteHParams, read by
// weights.cpp:read_granite_hparams with the same required/optional split).
// ---------------------------------------------------------------------------
struct GraniteSpeechHParams {
    // stt.variant: selects the model-card prompt (empty = historical default).
    std::string variant;

    // Encoder (Conformer).
    int32_t enc_n_layers = 0;
    int32_t enc_hidden = 0;
    int32_t enc_n_heads = 0;
    int32_t enc_head_dim = 0;
    int32_t enc_input_dim = 0;
    int32_t enc_output_dim = 0;
    int32_t enc_feedforward_mult = 0;
    int32_t enc_conv_kernel_size = 0;
    int32_t enc_conv_expansion = 0;
    int32_t enc_max_pos_emb = 0;
    int32_t enc_context_size = 0;
    std::vector<int32_t> enc_cat_hidden_layers;  // [3] on -plus, empty otherwise

    // Projector (BLIP-2 Q-Former).
    int32_t prj_n_layers = 0;
    int32_t prj_hidden = 0;
    int32_t prj_intermediate = 0;
    int32_t prj_n_heads = 0;
    int32_t prj_encoder_hidden_size = 0;
    int32_t prj_cross_attn_freq = 0;
    float prj_layer_norm_eps = 0.0f;
    int32_t prj_num_queries = 0;  // window_size / downsample_rate (proj.query ne[1])

    // Granite-4 LM.
    int32_t dec_n_layers = 0;
    int32_t dec_hidden = 0;
    int32_t dec_intermediate = 0;
    int32_t dec_n_heads = 0;
    int32_t dec_n_kv_heads = 0;
    int32_t dec_head_dim = 0;
    float dec_rms_norm_eps = 0.0f;
    float dec_rope_theta = 0.0f;
    int32_t dec_max_position_embeddings = 0;
    bool dec_tie_word_embeddings = false;
    int32_t dec_vocab_size = 0;
    float dec_embedding_multiplier = 0.0f;
    float dec_logits_scaling = 0.0f;
    float dec_attention_multiplier = 0.0f;
    float dec_residual_multiplier = 0.0f;

    // Audio fusion.
    int32_t audio_token_id = 0;
    int32_t downsample_rate = 0;
    int32_t window_size = 0;
};

// Host log-mel frontend state (arch transcribe::MelConfig as granite's
// model.cpp:load fills it, plus the MelFrontend constructor's derived tables).
struct GraniteSpeechFrontend {
    int32_t sample_rate = 0;
    int32_t n_mels = 0;
    int32_t n_fft = 0;
    int32_t win_length = 0;
    int32_t hop_length = 0;
    float pre_emphasis = 0.0f;
    std::string pad_mode;               // "reflect" | "constant"
    std::vector<double> window;         // [n_fft], the GGUF window zero-padded (centred)
    std::vector<float> filterbank;      // [n_mels * (n_fft/2+1)] row-major (GGUF frontend.mel_filterbank)
    std::vector<int32_t> fb_begin;      // per-band nonzero support (MelFrontend ctor)
    std::vector<int32_t> fb_end;
};

// GGUF "gpt2" byte-level BPE (arch transcribe-tokenizer.cpp Tokenizer, load +
// encode + decode for model == "gpt2"). Encodes plain text only: special
// tokens are always spliced as ids by the prompt builder, never parsed.
class GraniteSpeechTokenizer {
public:
    void load(const std::filesystem::path & gguf_path);

    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const int32_t * ids, size_t count) const;
    int32_t find(const std::string & piece) const;  // -1 when absent
    int32_t eos_id() const noexcept { return eos_id_; }
    int32_t size() const noexcept { return static_cast<int32_t>(tokens_.size()); }
    const std::string & pretokenizer() const noexcept { return pre_; }

private:
    std::vector<std::string> pretokenize(const std::string & text) const;

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;
    std::string pre_;
    int32_t eos_id_ = -1;
};

// Chat-template token ids (arch granite.h ChatTokens), resolved at load.
struct GraniteSpeechChatTokens {
    int32_t audio = -1;          // <|audio|>
    int32_t end_of_text = -1;    // <|end_of_text|>
    int32_t pad = -1;            // <|pad|>
    int32_t start_of_role = -1;  // <|start_of_role|> (-plus only)
    int32_t end_of_role = -1;    // <|end_of_role|>   (-plus only)
};

struct GraniteSpeechAssets {
    // Model root for SpecBackedVoiceModelLoader::inspect (the GGUF's folder);
    // the self-contained GGUF needs no sidecar resources.
    assets::ResourceBundle resources;
    std::filesystem::path path;
    std::shared_ptr<const assets::TensorSource> source;
    GraniteSpeechHParams hparams;
    GraniteSpeechFrontend frontend;
    GraniteSpeechTokenizer tokenizer;
    GraniteSpeechChatTokens chat_tokens;
    std::string chat_template;

    // Per-variant capability KVs (stt.capability.*). The spec contract is per
    // family, so the session gates variant-specific tasks on these.
    bool can_translate = false;
    bool has_word_timestamps = false;
    bool has_speaker_attribution = false;
    std::vector<std::string> languages;

    // Decode-budget basis (arch model.cpp:load `limits`): ms of audio per
    // audio token; 0 when the rate fields are missing.
    double ms_per_audio_token = 0.0;
};

// True for the transcribe.cpp GGUF layout this package reads
// (general.architecture == "granite_speech").
bool looks_like_transcribe_granite_speech_gguf(const std::filesystem::path & path);
std::shared_ptr<const GraniteSpeechAssets> load_granite_speech_assets(const std::filesystem::path & path);

// ---------------------------------------------------------------------------
// Weights (arch weights.h slot structs, bound in the engine's PyTorch
// [out, in] shape convention).
// ---------------------------------------------------------------------------
struct GraniteSpeechEncoderBlock {
    modules::NormWeights norm_ff1, norm_conv, norm_ff2, norm_post;
    modules::LinearWeights ff1_up, ff1_down, ff2_up, ff2_down;
    modules::ShawAttentionWeights attention;  // norm_attn, q, fused kv, rel_pos_emb, out
    modules::LinearWeights conv_pointwise1;   // [2*inner, hidden]
    modules::LinearWeights conv_pointwise2;   // [hidden, inner]
    core::TensorValue conv_depthwise;         // F32 [inner, 1, k]
    core::TensorValue conv_bn_scale;          // F32 [inner] fused gamma / sqrt(var + eps)
    core::TensorValue conv_bn_bias;           // F32 [inner] fused beta - mean * scale
};

struct GraniteSpeechProjectorBlock {
    modules::LinearWeights self_q, self_k, self_v, self_out;
    modules::NormWeights norm_self;
    modules::LinearWeights cross_q, cross_k, cross_v, cross_out;
    modules::NormWeights norm_cross;
    modules::LinearWeights ffn_up, ffn_down;
    modules::NormWeights norm_ffn;
};

struct GraniteSpeechDecoderBlock {
    core::TensorValue norm_attn, norm_ffn;  // RMSNorm gains
    core::TensorValue q, k, v, o;
    core::TensorValue gate, up, down;
};

struct GraniteSpeechWeights {
    std::unique_ptr<core::BackendWeightStore> store;

    modules::LinearWeights enc_input_linear, enc_ctc_proj, enc_ctc_bypass;
    std::vector<GraniteSpeechEncoderBlock> encoder;

    core::TensorValue proj_query;             // [num_queries, prj_hidden]
    modules::NormWeights proj_qformer_norm;   // Blip2QFormerModel.layernorm (INPUT norm)
    modules::LinearWeights proj_linear;       // prj_hidden -> dec_hidden
    std::vector<GraniteSpeechProjectorBlock> projector;

    core::TensorValue token_embedding;        // [vocab, dec_hidden]
    std::vector<GraniteSpeechDecoderBlock> decoder;
    core::TensorValue output_norm;
    core::TensorValue output;                 // == token_embedding when tied (-plus)
};

std::unique_ptr<GraniteSpeechWeights> load_granite_speech_weights(
    const GraniteSpeechAssets & assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType type);

// ---------------------------------------------------------------------------
// Frontend: 16 kHz mono PCM -> [t_enc, 2 * n_mels] row-major encoder input
// (arch encoder.cpp:compute_mel_encoder_input over transcribe-mel.cpp's
// MelFrontend::compute in "per_utterance" mode).
// ---------------------------------------------------------------------------
struct GraniteSpeechFeatures {
    std::vector<float> values;  // [t_enc, input_dim] row-major
    int32_t t_enc = 0;
};

GraniteSpeechFeatures compute_granite_speech_features(
    const std::vector<float> & pcm, const GraniteSpeechFrontend & frontend, int threads);

// ---------------------------------------------------------------------------
// Prompt + result post-processing (arch model.cpp build_granite_affixes,
// parse_granite_word_timestamps, finalize_granite_result; diarize.cpp).
// ---------------------------------------------------------------------------
enum class GraniteSpeechTask {
    Transcribe,
    Translate,
    WordTimestamps,      // -plus: "[T:N]" markers
    SpeakerAttribution,  // -plus: "[Speaker N]:" tags
};

struct GraniteSpeechPrompt {
    GraniteSpeechTask task = GraniteSpeechTask::Transcribe;
    std::string target_language_name;  // Translate only, e.g. "German"
};

// Maps a BCP-47 code / English name to the instruction's language name;
// empty when unsupported (arch model.cpp:granite_target_language_name).
std::string granite_speech_target_language_name(const std::string & code_or_name);

void build_granite_speech_affixes(
    const GraniteSpeechAssets & assets,
    const GraniteSpeechPrompt & prompt,
    std::vector<int32_t> & prefix_ids,
    std::vector<int32_t> & suffix_ids);

struct GraniteSpeechWord {
    std::string text;
    int64_t t0_ms = 0;
    int64_t t1_ms = 0;
};

// Parses "<word> [T:N]" markers; returns the marker-free transcript (empty
// `words` when no marker was recognised).
std::string parse_granite_speech_word_timestamps(
    const std::string & raw, int64_t audio_ms, std::vector<GraniteSpeechWord> & words);

struct GraniteSpeechTurn {
    int32_t speaker = 0;  // 0 = text before the first marker
    std::string text;
};

// Splits at "[Speaker N]" markers; false (outputs untouched) when none.
bool split_granite_speech_speaker_turns(
    const std::string & raw, std::vector<GraniteSpeechTurn> & turns, std::string & full_text);

// ---------------------------------------------------------------------------
// Runtime: encoder + projector + greedy LM decode for one utterance.
// ---------------------------------------------------------------------------
struct GraniteSpeechDecodeLimits {
    int32_t n_ctx = 0;         // session context ceiling; 0 = the model maximum
    int64_t max_tokens = 0;    // generation cap; 0 = the arch's decode budget
    ggml_type kv_type = GGML_TYPE_F16;
};

struct GraniteSpeechDecodeResult {
    std::vector<int32_t> tokens;  // generated ids, EOS excluded
    bool truncated = false;       // stopped at the budget / context before EOS
    int32_t n_audio_tokens = 0;
};

class GraniteSpeechRuntime {
public:
    GraniteSpeechRuntime(const GraniteSpeechAssets & assets,
                         const GraniteSpeechWeights & weights,
                         core::ExecutionContext & execution);
    ~GraniteSpeechRuntime();

    // `poll(step, total)` runs at stage boundaries and every decode step and
    // may throw (RunControl cancellation).
    GraniteSpeechDecodeResult transcribe(
        const std::vector<float> & pcm_16k,
        const std::vector<int32_t> & prefix_ids,
        const std::vector<int32_t> & suffix_ids,
        const GraniteSpeechDecodeLimits & limits,
        const std::function<void(int64_t, int64_t)> & poll = {});

private:
    struct KvCache;
    std::vector<float> encode_audio(const GraniteSpeechFeatures & features, int32_t & n_audio_tokens);

    const GraniteSpeechAssets & assets_;
    const GraniteSpeechWeights & weights_;
    core::ExecutionContext & execution_;
    std::unique_ptr<KvCache> kv_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_granite_speech_loader();

}  // namespace engine::models::granite_speech
