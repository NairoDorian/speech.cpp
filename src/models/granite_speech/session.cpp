// granite_speech session + loader.
//
// Request handling ported from src/runtime/arch/granite/model.cpp:run (task
// selection, the diarize / word-timestamp exclusivity, truncation) and
// finalize_granite_result (plain / word-timestamp / speaker-attribution
// results). Per-variant capability gating stands in for the C-ABI
// dispatcher's checks against the arch's per-model capabilities (the engine
// spec contract is per family).

#include "engine/models/granite_speech/model.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::granite_speech {
namespace {

constexpr int kSampleRate = 16000;

enum class TimestampRequest { None, Segment, Word };

// "none" / "auto" / "segment" / "word" / "token", or the C ABI's numeric
// transcribe_timestamps value the adapter forwards (0..4).
TimestampRequest parse_timestamps(const std::string & value) {
    if (value.empty() || value == "none" || value == "0" || value == "auto" || value == "1") {
        return TimestampRequest::None;  // AUTO does not request the timestamp task (arch)
    }
    if (value == "segment" || value == "2") {
        return TimestampRequest::Segment;
    }
    if (value == "word" || value == "3") {
        return TimestampRequest::Word;
    }
    if (value == "token" || value == "4") {
        throw std::invalid_argument("Granite Speech does not produce token timestamps");
    }
    throw std::invalid_argument("Granite Speech timestamps must be none, auto, segment or word");
}

class GraniteSpeechSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    GraniteSpeechSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                         std::shared_ptr<const GraniteSpeechAssets> assets,
                         std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "granite_speech", "Granite Speech");
        if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Granite Speech requires an offline ASR session");
        }
        const auto type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"granite_speech.weight_type", "weight_type"}).value_or("native"));
        // kv_type / n_ctx: the C-ABI adapter forwards transcribe_session_params
        // under the bare names; the spec declares them as granite_speech.*.
        const auto kv = runtime::find_option(options.options, {"granite_speech.kv_type", "kv_type"}).value_or("f16");
        if (kv == "f16") {
            limits_.kv_type = GGML_TYPE_F16;
        } else if (kv == "f32") {
            limits_.kv_type = GGML_TYPE_F32;
        } else {
            throw std::invalid_argument("Granite Speech kv_type must be f16 or f32");
        }
        const auto n_ctx = runtime::parse_i64_option(options.options, {"granite_speech.n_ctx", "n_ctx"}).value_or(0);
        if (n_ctx < 0) {
            throw std::invalid_argument("Granite Speech n_ctx must be >= 0");
        }
        limits_.n_ctx = static_cast<int32_t>(std::min<int64_t>(n_ctx, assets_->hparams.dec_max_position_embeddings));
        weights_ = load_granite_speech_weights(*assets_, execution_context(), type);
        runtime_ = std::make_unique<GraniteSpeechRuntime>(*assets_, *weights_, execution_context());
    }

    std::string family() const override { return "granite_speech"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Granite Speech");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Granite Speech run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Granite Speech");
        if (!request.audio_input.has_value()) {
            throw std::invalid_argument("Granite Speech requires audio input");
        }
        const auto & opts = request.options;

        // ---- Task selection (arch build_granite_affixes precedence:
        // translate, then word timestamps, then speaker attribution) ----
        GraniteSpeechPrompt prompt;
        const auto task = runtime::find_option(opts, {"task"}).value_or("transcribe");
        const auto timestamps = parse_timestamps(runtime::find_option(opts, {"timestamps"}).value_or("none"));
        const bool diarize = runtime::parse_bool_option(
            runtime::find_option(opts, {"diarize"}).value_or("false"), "diarize");
        if (task == "translate") {
            if (!assets_->can_translate) {
                throw std::invalid_argument("Granite Speech variant '" + assets_->hparams.variant +
                                            "' does not support translation");
            }
            const auto target = runtime::find_option(opts, {"target_language"}).value_or("");
            if (target.empty()) {
                throw std::invalid_argument("Granite Speech translate task requires target_language");
            }
            prompt.target_language_name = granite_speech_target_language_name(target);
            if (prompt.target_language_name.empty()) {
                throw std::invalid_argument("Granite Speech target_language '" + target + "' is not advertised");
            }
            prompt.task = GraniteSpeechTask::Translate;
        } else if (task != "transcribe") {
            throw std::invalid_argument("Granite Speech task must be transcribe or translate");
        }
        if (timestamps != TimestampRequest::None && !assets_->has_word_timestamps) {
            // The arch advertises max_timestamp_kind NONE on 1b / 2b.
            throw std::invalid_argument("Granite Speech variant '" + assets_->hparams.variant +
                                        "' does not produce timestamps");
        }
        if (diarize && !assets_->has_speaker_attribution) {
            throw std::invalid_argument("Granite Speech variant '" + assets_->hparams.variant +
                                        "' does not support speaker attribution");
        }
        const bool want_words = prompt.task == GraniteSpeechTask::Transcribe && timestamps == TimestampRequest::Word;
        const bool want_speakers = prompt.task == GraniteSpeechTask::Transcribe && diarize;
        if (want_words && want_speakers) {
            // Separate tasks upstream (one instruction each); they do not compose.
            throw std::invalid_argument(
                "Granite Speech diarize and word timestamps are mutually exclusive tasks; request one");
        }
        if (want_words) {
            prompt.task = GraniteSpeechTask::WordTimestamps;
        } else if (want_speakers) {
            prompt.task = GraniteSpeechTask::SpeakerAttribution;
        }

        GraniteSpeechDecodeLimits limits = limits_;
        limits.max_tokens = runtime::parse_i64_option(opts, {"max_tokens"}).value_or(0);
        if (limits.max_tokens < 0) {
            throw std::invalid_argument("Granite Speech max_tokens must be >= 0");
        }

        std::vector<int32_t> prefix;
        std::vector<int32_t> suffix;
        build_granite_speech_affixes(*assets_, prompt, prefix, suffix);

        const auto & input = *request.audio_input;
        const auto pcm = audio::convert_interleaved_audio_to_mono_linear_resampled(
            input.samples, input.sample_rate, input.channels, kSampleRate);
        const auto decoded = runtime_->transcribe(pcm, prefix, suffix, limits,
            [this](int64_t step, int64_t total) {
                // Progress plus a cancellation point per decode step.
                emit_progress("granite_speech.decode", step, total);
            });

        const std::string raw = assets_->tokenizer.decode(decoded.tokens.data(), decoded.tokens.size());
        const int64_t n_samples = static_cast<int64_t>(pcm.size());
        const int64_t audio_ms = n_samples * 1000 / kSampleRate;
        trace(debug::LogLevel::Info, "granite_speech",
              "audio_tokens=" + std::to_string(decoded.n_audio_tokens) + " tokens=" +
                  std::to_string(decoded.tokens.size()) + (decoded.truncated ? " truncated" : ""));

        // ---- Result (arch finalize_granite_result) ----
        runtime::TaskResult result;
        result.truncated = decoded.truncated;
        std::string full_text;
        bool shaped = false;
        if (want_words) {
            std::vector<GraniteSpeechWord> words;
            full_text = parse_granite_speech_word_timestamps(raw, audio_ms, words);
            if (!words.empty()) {
                for (const auto & word : words) {
                    runtime::WordTimestamp wt;
                    wt.span = {word.t0_ms * kSampleRate / 1000, word.t1_ms * kSampleRate / 1000};
                    wt.word = word.text;
                    result.word_timestamps.push_back(std::move(wt));
                }
                runtime::SpeechSegment seg;
                seg.span = {words.front().t0_ms * kSampleRate / 1000, words.back().t1_ms * kSampleRate / 1000};
                seg.text = full_text;
                result.speech_segments.push_back(std::move(seg));
                shaped = true;
            }
        }
        if (!shaped && want_speakers) {
            std::vector<GraniteSpeechTurn> turns;
            if (split_granite_speech_speaker_turns(raw, turns, full_text)) {
                // The attribution task carries no timing: zero-length spans,
                // the arch's "absent" t0 == t1 == 0 sentinel.
                for (const auto & turn : turns) {
                    runtime::SpeechSegment seg;
                    seg.text = turn.text;
                    result.speech_segments.push_back(std::move(seg));
                    if (turn.speaker > 0) {
                        runtime::SpeakerTurn st;
                        st.speaker_id = std::to_string(turn.speaker);
                        st.text = turn.text;
                        result.speaker_turns.push_back(std::move(st));
                    }
                }
                shaped = true;
            }
        }
        if (!shaped) {
            full_text = raw;
            runtime::SpeechSegment seg;
            seg.span = {0, n_samples};
            seg.text = raw;
            result.speech_segments.push_back(std::move(seg));
        }
        // No language identification: the arch leaves detected_language empty.
        result.text_output = runtime::Transcript{full_text, ""};
        if (full_text != raw) {
            result.raw_text = raw;  // pre-parse marker text (transcribe_raw_text)
        }
        return result;
    }

private:
    std::shared_ptr<const GraniteSpeechAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    GraniteSpeechDecodeLimits limits_;
    std::unique_ptr<GraniteSpeechWeights> weights_;
    std::unique_ptr<GraniteSpeechRuntime> runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_granite_speech_loader() {
    runtime::SpecBackedVoiceModelConfig<GraniteSpeechAssets> config;
    config.family = "granite_speech";
    config.load_assets = load_granite_speech_assets;
    // The transcribe.cpp GGUF names its family only through
    // general.architecture; accept it without resolving a spec bundle.
    config.accepts_foreign_layout = looks_like_transcribe_granite_speech_gguf;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const GraniteSpeechAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<GraniteSpeechSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::granite_speech
