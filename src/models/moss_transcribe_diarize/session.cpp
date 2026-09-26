#include "engine/models/moss_transcribe_diarize/session.h"
#include "engine/models/moss_transcribe_diarize/runtime.h"

#include "engine/framework/io/text.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::moss_transcribe_diarize {
namespace {

class TranscribeSession final : public runtime::RuntimeSessionBase,
                                public runtime::IOfflineVoiceTaskSession,
                                public runtime::IStreamingVoiceTaskSession {
public:
    TranscribeSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                      std::shared_ptr<const TranscribeAssets> resources,
                      std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), contract_(std::move(contract)), resources_(std::move(resources)), mode_(task.mode) {
        runtime::validate_spec_backed_session_options(options, *contract_, "moss_transcribe_diarize", "MOSS-Transcribe-Diarize");
        if (task.task != runtime::VoiceTaskKind::Asr ||
            (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming)) {
            throw std::runtime_error("MOSS-Transcribe-Diarize supports offline and streaming ASR");
        }
        model_ = std::make_unique<TranscribeRuntime>(*resources_, execution_context());
    }

    std::string family() const override { return "moss_transcribe_diarize"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return mode_; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (!request.audio || request.audio->sample_rate <= 0 || request.audio->channels <= 0) {
            throw std::runtime_error("MOSS-Transcribe-Diarize requires an audio preparation contract");
        }
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("MOSS-Transcribe-Diarize run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "MOSS-Transcribe-Diarize");
        if (!request.audio_input) {
            throw std::runtime_error("MOSS-Transcribe-Diarize requires audio_input");
        }
        const auto started = std::chrono::steady_clock::now();
        const auto policy = parse_policy(request.options);
        const auto instruction = runtime::find_option(request.options, {"instruct"}).value_or("");
        // Parsed untrimmed, as the arch parsed its raw decode (a trailing
        // newline after the last end time changes the grammar's outcome).
        const auto raw = model_->transcribe(*request.audio_input, instruction, requested_max_tokens(request.options),
                                            progress_poll());
        auto result = build_result(raw, request.audio_input->sample_rate, audio_frames(*request.audio_input), policy);
        result.truncated = model_->truncated();
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

    runtime::StreamingPolicy streaming_policy() const override {
        runtime::StreamingPolicy policy;
        policy.input = runtime::StreamingInputKind::None;
        policy.output = runtime::StreamingOutputKind::PullEvents;
        return policy;
    }

    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("MOSS-Transcribe-Diarize start_stream");
        if (mode_ != runtime::RunMode::Streaming) {
            throw std::runtime_error("MOSS-Transcribe-Diarize start_stream requires streaming mode");
        }
        runtime::validate_spec_backed_request_options(request.options, *contract_, "MOSS-Transcribe-Diarize");
        if (!request.audio_input) {
            throw std::runtime_error("MOSS-Transcribe-Diarize streaming requires complete audio_input");
        }
        reset();
        stream_policy_ = parse_policy(request.options);
        // The published deltas spell the raw transcript, so the final text of a
        // stream stays raw (output=clean applies to offline runs); segments and
        // speaker rows follow the same policy as offline.
        stream_policy_.raw_text = true;
        const auto instruction = runtime::find_option(request.options, {"instruct"}).value_or("");
        model_->start(*request.audio_input, instruction, requested_max_tokens(request.options), progress_poll());
        stream_sample_rate_ = request.audio_input->sample_rate;
        stream_frames_ = audio_frames(*request.audio_input);
        stream_started_ = true;
    }

    std::optional<runtime::StreamEvent> next_stream_event() override {
        if (!stream_started_) {
            throw std::runtime_error("MOSS-Transcribe-Diarize stream has not been started");
        }
        auto delta = model_->next_text();
        if (!delta) {
            stream_complete_ = true;
            return std::nullopt;
        }
        stream_text_ += *delta;
        runtime::StreamEvent event;
        event.partial_text = runtime::Transcript{std::move(*delta), ""};
        return event;
    }

    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &) override {
        throw std::runtime_error("MOSS-Transcribe-Diarize supports text-output streaming, not incremental audio input");
    }

    runtime::TaskResult finalize() override {
        if (!stream_started_ || !stream_complete_) {
            throw std::runtime_error("MOSS-Transcribe-Diarize finalize requires completed text generation");
        }
        auto result = build_result(stream_text_, stream_sample_rate_, stream_frames_, stream_policy_);
        result.truncated = model_->truncated();
        reset();
        return result;
    }

    void reset() override {
        model_->reset();
        stream_started_ = false;
        stream_complete_ = false;
        stream_text_.clear();
        stream_sample_rate_ = 0;
        stream_frames_ = 0;
        stream_policy_ = ResultPolicy{};
    }

private:
    // Encoder-chunk and decode-step progress; each call is a cancellation point
    // (emit_progress throws ProgressCanceled on a cancel request).
    TranscribePoll progress_poll() {
        return [this](const char * stage, int64_t completed, int64_t total) {
            emit_progress(stage, completed, total);
        };
    }

    static int64_t audio_frames(const runtime::AudioBuffer & audio) {
        const auto samples = static_cast<int64_t>(audio.samples.size());
        return audio.channels > 0 ? samples / audio.channels : samples;
    }

    // An explicit max_tokens, else nullopt: the layout default (the package's
    // generation_config, or the retired arch's adaptive budget for the
    // transcribe.cpp GGUF).
    static std::optional<int64_t> requested_max_tokens(const std::unordered_map<std::string, std::string> & options) {
        if (!runtime::find_option(options, {"max_tokens"}).has_value()) {
            return std::nullopt;
        }
        return runtime::parse_positive_i64_option(options, {"max_tokens"}, 1);
    }

    // Result policy, read once per request: the arch's run axes (diarize,
    // timestamps) and which text `text_output` carries.
    struct ResultPolicy {
        bool diarize = false;
        bool timestamps = true;
        bool raw_text = false;
    };

    static ResultPolicy parse_policy(const std::unordered_map<std::string, std::string> & options) {
        ResultPolicy policy;
        if (const auto value = runtime::find_option(options, {"diarize"})) {
            policy.diarize = runtime::parse_bool_option(*value, "diarize");
        }
        // The C ABI forwards transcribe_timestamp_kind as its number (0 none,
        // 1 auto, 2 segment, 3 word); the model times turns, never words.
        const auto ts = runtime::find_option(options, {"timestamps"}).value_or("auto");
        if (ts == "none" || ts == "0") {
            policy.timestamps = false;
        } else if (ts == "auto" || ts == "1" || ts == "segment" || ts == "2" || ts.empty()) {
            policy.timestamps = true;
        } else {
            throw std::invalid_argument("MOSS-Transcribe-Diarize timestamps '" + ts + "' are not supported (segment only)");
        }
        const auto output = runtime::find_option(options, {"output"}).value_or("clean");
        if (output != "clean" && output != "raw") {
            throw std::invalid_argument("MOSS-Transcribe-Diarize output must be clean or raw");
        }
        policy.raw_text = output == "raw";
        return policy;
    }

    // The retired arch's grammar (src/runtime/arch/moss/diarize.cpp,
    // parse_diarized_transcript) over the model's emergent transcript
    //     [0.48][S01]Welcome.[1.66][1.70][S02]Thanks for having me.[3.20]
    // applied at each '[':
    //   TIME := '[' digits ('.' digits)? ']'   seconds, kept in ms (4th decimal rounds)
    //   SPK  := '[S' digits ']'                1-based speaker tag (S0 is text)
    // A TIME closes the open turn only when followed by SPK, TIME SPK, or end of
    // input; otherwise it, and any other '[' span, is verbatim text. A bare SPK is
    // a turn boundary with unknown times, patched from the neighbours (first
    // t0 -> 0, last t1 -> the clip end). Text before the first turn is a leading
    // unattributed segment. Nothing is dropped: empty turns stay.
    static constexpr int64_t kUnknownMs = -1;

    struct Turn {
        int64_t t0 = kUnknownMs;
        int64_t t1 = kUnknownMs;
        std::string speaker;  // the tag as written, e.g. "S01"
        std::string text;
    };

    static bool try_time(const std::string & raw, size_t i, int64_t & out_ms, size_t & out_next) {
        const size_t n = raw.size();
        if (i >= n || raw[i] != '[') {
            return false;
        }
        size_t j = i + 1;
        if (j >= n || !is_digit(raw[j])) {
            return false;
        }
        int64_t int_part = 0;
        while (j < n && is_digit(raw[j])) {
            if (int_part > (INT64_MAX - 9) / 10) {
                return false;
            }
            int_part = int_part * 10 + (raw[j] - '0');
            ++j;
        }
        int64_t frac_ms = 0;
        int n_frac = 0;
        bool round_up = false;
        if (j < n && raw[j] == '.') {
            ++j;
            if (j >= n || !is_digit(raw[j])) {
                return false;
            }
            while (j < n && is_digit(raw[j])) {
                if (n_frac < 3) {
                    frac_ms = frac_ms * 10 + (raw[j] - '0');
                    ++n_frac;
                } else if (n_frac == 3) {
                    round_up = raw[j] >= '5';
                    ++n_frac;
                }
                ++j;
            }
        }
        if (j >= n || raw[j] != ']') {
            return false;
        }
        while (n_frac < 3) {
            frac_ms *= 10;
            ++n_frac;
        }
        if (int_part > (INT64_MAX - frac_ms - 1) / 1000) {
            return false;
        }
        out_ms = int_part * 1000 + frac_ms + (round_up ? 1 : 0);
        out_next = j + 1;
        return true;
    }

    static bool try_speaker(const std::string & raw, size_t i, std::string & out_tag, size_t & out_next) {
        const size_t n = raw.size();
        if (i + 2 >= n || raw[i] != '[' || raw[i + 1] != 'S' || !is_digit(raw[i + 2])) {
            return false;
        }
        size_t j = i + 2;
        int64_t id = 0;
        while (j < n && is_digit(raw[j])) {
            id = id * 10 + (raw[j] - '0');
            if (id > INT32_MAX) {
                return false;
            }
            ++j;
        }
        if (j >= n || raw[j] != ']' || id <= 0) {
            return false;
        }
        out_tag = raw.substr(i + 1, j - i - 1);
        out_next = j + 1;
        return true;
    }

    static bool is_digit(char c) { return c >= '0' && c <= '9'; }

    static std::string trimmed(const std::string & s) {
        size_t b = 0;
        size_t e = s.size();
        while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) {
            ++b;
        }
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) {
            --e;
        }
        return s.substr(b, e - b);
    }

    // False when no turn is recognized (outputs untouched).
    static bool parse_turns(const std::string & raw, int64_t audio_ms, std::vector<Turn> & turns, std::string & lead) {
        std::string pre_text;
        bool has_open = false;
        Turn open;
        const auto sink = [&]() -> std::string & { return has_open ? open.text : pre_text; };
        const auto close_open = [&](int64_t t1) {
            open.t1 = t1;
            turns.push_back(std::move(open));
            open = Turn{};
            has_open = false;
        };
        const size_t n = raw.size();
        size_t i = 0;
        while (i < n) {
            if (raw[i] != '[') {
                sink().push_back(raw[i]);
                ++i;
                continue;
            }
            int64_t t_ms = 0;
            std::string tag;
            size_t j = 0;
            if (try_time(raw, i, t_ms, j)) {
                int64_t t2_ms = 0;
                size_t k = 0;
                if (try_speaker(raw, j, tag, k)) {
                    // TIME SPK: shared boundary (or the first turn's start).
                    if (has_open) {
                        close_open(t_ms);
                    }
                    open.t0 = t_ms;
                    open.speaker = tag;
                    has_open = true;
                    i = k;
                    continue;
                }
                size_t m = 0;
                if (try_time(raw, j, t2_ms, m) && try_speaker(raw, m, tag, k)) {
                    // TIME TIME SPK: end of the open turn, start of the next; with
                    // no open turn the first TIME is stray -> verbatim.
                    if (has_open) {
                        close_open(t_ms);
                        open.t0 = t2_ms;
                        open.speaker = tag;
                        has_open = true;
                        i = k;
                        continue;
                    }
                    sink().append(raw, i, j - i);
                    i = j;
                    continue;
                }
                if (j >= n && has_open) {
                    close_open(t_ms);  // TIME at end of input closes the open turn
                    i = j;
                    continue;
                }
                sink().append(raw, i, j - i);  // TIME followed by plain text
                i = j;
                continue;
            }
            if (try_speaker(raw, i, tag, j)) {
                // Bare SPK: turn boundary with unknown times.
                if (has_open) {
                    close_open(kUnknownMs);
                }
                open.t0 = kUnknownMs;
                open.speaker = tag;
                has_open = true;
                i = j;
                continue;
            }
            sink().push_back(raw[i]);  // unrecognized '[' span: literal text
            ++i;
        }
        if (has_open) {
            close_open(kUnknownMs);
        }
        if (turns.empty()) {
            return false;
        }
        // Forward pass: an unknown t0 inherits the previous turn's end (or start),
        // the first falls back to 0. Backward pass: an unknown t1 inherits the next
        // turn's start, the last falls back to the clip end. Known model values are
        // never reordered; only t1 >= t0 is enforced.
        for (size_t t = 0; t < turns.size(); ++t) {
            if (turns[t].t0 == kUnknownMs) {
                if (t == 0) {
                    turns[t].t0 = 0;
                } else if (turns[t - 1].t1 != kUnknownMs) {
                    turns[t].t0 = turns[t - 1].t1;
                } else {
                    turns[t].t0 = turns[t - 1].t0;
                }
            }
            if (turns[t].t0 < 0) {
                turns[t].t0 = 0;
            }
        }
        for (size_t t = turns.size(); t-- > 0;) {
            if (turns[t].t1 == kUnknownMs) {
                turns[t].t1 = (t + 1 < turns.size()) ? turns[t + 1].t0 : audio_ms;
            }
            if (turns[t].t1 < turns[t].t0) {
                turns[t].t1 = turns[t].t0;
            }
        }
        lead = trimmed(pre_text);
        return true;
    }

    // Turns -> TaskResult the way the arch installed its result: segments are the
    // leading text (if any) and one per turn; the clean text joins the non-empty
    // trimmed pieces with single spaces (what the WER gates scored); speaker rows
    // only with diarize; timestamps=none zeroes every time. Spans are ms converted
    // to input samples (exact at the C ABI's 16 kHz, so it reads the arch's ms
    // back). No recognized turn: the raw text as one whole-clip segment.
    static runtime::TaskResult build_result(const std::string & raw, int sample_rate, int64_t frames,
                                            const ResultPolicy & policy) {
        const int64_t audio_ms = frames * 1000 / sample_rate;
        const auto to_samples = [sample_rate](int64_t ms) {
            return static_cast<int64_t>(std::llround(static_cast<double>(ms) * sample_rate / 1000.0));
        };
        runtime::TaskResult result;
        result.output_artifacts.push_back(runtime::make_text_artifact(
            runtime::ArtifactKind::Custom, "moss_transcribe_diarize.raw_text", raw, {{"format", "text"}}));
        std::vector<Turn> turns;
        std::string lead;
        if (!parse_turns(raw, audio_ms, turns, lead)) {
            result.text_output = runtime::Transcript{raw, ""};
            result.speech_segments.push_back({runtime::TimeSpan{0, to_samples(audio_ms)}, 0.f, raw});
            return result;
        }
        std::string clean;
        const auto append_clean = [&clean](const std::string & piece) {
            if (piece.empty()) {
                return;
            }
            if (!clean.empty()) {
                clean.push_back(' ');
            }
            clean += piece;
        };
        const auto span_of = [&](int64_t t0, int64_t t1) {
            return policy.timestamps ? runtime::TimeSpan{to_samples(t0), to_samples(t1)} : runtime::TimeSpan{0, 0};
        };
        if (!lead.empty()) {
            result.speech_segments.push_back({span_of(0, turns.front().t0), 0.f, lead});
            append_clean(lead);
        }
        for (const auto & turn : turns) {
            const auto text = trimmed(turn.text);
            append_clean(text);
            result.speech_segments.push_back({span_of(turn.t0, turn.t1), 0.f, text});
            if (policy.diarize) {
                result.speaker_turns.push_back({span_of(turn.t0, turn.t1), turn.speaker, 0.f, text});
            }
        }
        result.text_output = runtime::Transcript{policy.raw_text ? raw : clean, ""};
        // transcribe_get_raw_text(): the arch's raw_text was the tagged decode.
        result.raw_text = raw;
        return result;
    }

private:
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::shared_ptr<const TranscribeAssets> resources_;
    std::unique_ptr<TranscribeRuntime> model_;
    runtime::RunMode mode_;
    bool stream_started_ = false;
    bool stream_complete_ = false;
    int stream_sample_rate_ = 0;
    int64_t stream_frames_ = 0;
    ResultPolicy stream_policy_;
    std::string stream_text_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_transcribe_diarize_loader() {
    runtime::SpecBackedVoiceModelConfig<TranscribeAssets> config;
    config.family = "moss_transcribe_diarize";
    config.load_assets = load_transcribe_assets;
    config.accepts_foreign_layout = looks_like_transcribe_moss_gguf;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const TranscribeAssets> resources,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<TranscribeSession>(task, options, std::move(resources), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::moss_transcribe_diarize
