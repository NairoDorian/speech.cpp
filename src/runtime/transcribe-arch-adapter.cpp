// transcribe-arch-adapter.cpp - bridge from audio.cpp's C++ model vtable into
// the transcribe C ABI `Arch` dispatch trait (V6 plan sub-task 0.H). See the
// sibling header for the lifetime and ABI mismatch rationale.
//
// STATUS (Phase 0.H, compile-validated since 0.L): implements the documented
// bridge surface so the unified C ABI can route audio.cpp families through
// find_arch(). Reachable families are those whose audio.cpp converter writes a
// GGUF carrying a `general.architecture` equal to the family id (e.g.
// citrinet_asr); safetensors families only reach here once the dispatcher gains
// a pre-Loader::open sniff (Phase 1). Backend binding is per-session inside the
// framework; the adapter reports the caller's requested surface label at model
// scope and resolves the concrete device when a session is eventually created.
//
// Two contract mismatches between the projects are resolved HERE, because
// neither side is wrong on its own terms:
//
//   1. Chunk granularity. The C ABI's transcribe_stream_feed() takes any
//      n_samples > 0; framework streaming sessions declare a required chunk
//      size via streaming_policy().preferred_audio_chunk_samples and reject
//      anything else (silero_vad: "chunk must contain exactly 512 samples")
//      plus demand contiguous start_sample. The adapter therefore buffers the
//      caller's PCM and dispatches only whole chunks, flushing the tail at
//      finalize. See StreamChunker below.
//
//   2. Cursor ownership. transcribe-session.h designates the audio cursors,
//      committed counts and stream_revision as HOOK-owned state that the
//      dispatcher reads back (transcribe.cpp's streaming-dispatcher comment).
//      The adapter writes the inherited base fields directly - it must never
//      declare same-named members, which would silently shadow them.

#include "transcribe-arch-adapter.h"

#include "transcribe-log.h"
#include "transcribe-loader.h"
#include "transcribe-model.h"
#include "transcribe-path.h"
#include "transcribe-session.h"
#include "transcribe/voxtral_realtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/stream_chunker.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using engine::core::BackendType;
using engine::core::BackendConfig;
using engine::runtime::CapabilitySet;
using engine::runtime::ILoadedVoiceModel;
using engine::runtime::IOfflineVoiceTaskSession;
using engine::runtime::IStreamingVoiceTaskSession;
using engine::runtime::IVoiceTaskSession;
using engine::runtime::ModelLoadRequest;
using engine::runtime::ModelRegistry;
using engine::runtime::RunMode;
using engine::runtime::AudioPreparationContract;
using engine::runtime::SessionOptions;
using engine::runtime::SessionPreparationRequest;
using engine::runtime::StreamEvent;
using engine::runtime::StreamingPolicy;
using engine::runtime::TaskResult;
using engine::runtime::TaskSpec;
using engine::runtime::TaskRequest;
using engine::runtime::VoiceTaskKind;
using engine::runtime::AudioBuffer;
using engine::runtime::AudioChunk;
using engine::runtime::build_preparation_request;
using engine::runtime::make_default_registry;

namespace transcribe {

namespace {

constexpr int k_native_sample_rate = 16000;

// Translate a C ABI backend request into the framework's BackendType. The
// concrete ggml backend is resolved by the framework session; we only hand the
// type enum through so the framework's device-selection policy runs.
static BackendType backend_type_from_request(transcribe_backend_request request) {
    switch (request) {
        case TRANSCRIBE_BACKEND_AUTO:        return BackendType::BestAvailable;
        case TRANSCRIBE_BACKEND_CPU:       return BackendType::Cpu;
        case TRANSCRIBE_BACKEND_CPU_ACCEL: return BackendType::Cpu;
        case TRANSCRIBE_BACKEND_METAL:     return BackendType::Metal;
        case TRANSCRIBE_BACKEND_VULKAN:    return BackendType::Vulkan;
        case TRANSCRIBE_BACKEND_CUDA:      return BackendType::Cuda;
        case TRANSCRIBE_BACKEND_ROCM:      return BackendType::Hip;
        default:                           return BackendType::BestAvailable;
    }
}

// Surface label for transcribe_model_backend(). AUTO resolves to "" (the C ABI
// contract's "no backend bound yet" sentinel); an explicit request reports the
// requested surface. The framework may fall back to CPU, which Phase 1 can
// report via a device accessor once the framework exposes its resolved handle.
static std::string backend_label_from_request(transcribe_backend_request request) {
    switch (request) {
        case TRANSCRIBE_BACKEND_CPU:       return "cpu";
        case TRANSCRIBE_BACKEND_CPU_ACCEL: return "cpu-accel";
        case TRANSCRIBE_BACKEND_METAL:     return "metal";
        case TRANSCRIBE_BACKEND_VULKAN:    return "vulkan";
        case TRANSCRIBE_BACKEND_CUDA:      return "cuda";
        case TRANSCRIBE_BACKEND_ROCM:      return "ROCm";
        default:                           return "";
    }
}

// Pick the TaskSpec the framework session should run for a given run mode. The
// C ABI is transcription-oriented, so prefer Asr; otherwise take the first task
// the family supports for that mode. Returns false when the family cannot run
// the requested mode.
static bool pick_task_spec(const CapabilitySet & caps, RunMode mode, TaskSpec & out) {
    const engine::runtime::TaskCapability * prefer_asr = nullptr;
    const engine::runtime::TaskCapability * any_mode = nullptr;
    for (const auto & tc : caps.supported_tasks) {
        if (std::find(tc.modes.begin(), tc.modes.end(), mode) == tc.modes.end()) {
            continue;
        }
        if (any_mode == nullptr) {
            any_mode = &tc;
        }
        if (tc.task == VoiceTaskKind::Asr && prefer_asr == nullptr) {
            prefer_asr = &tc;
        }
    }
    const auto * pick = prefer_asr != nullptr ? prefer_asr : any_mode;
    if (pick == nullptr) {
        return false;
    }
    out.task = pick->task;
    out.mode = mode;
    return true;
}

static bool caps_supports_streaming(const CapabilitySet & caps) {
    for (const auto & tc : caps.supported_tasks) {
        for (const auto & mode : tc.modes) {
            if (mode == RunMode::Streaming) {
                return true;
            }
        }
    }
    return false;
}

static bool caps_has_task(const CapabilitySet & caps, VoiceTaskKind task) {
    for (const auto & tc : caps.supported_tasks) {
        if (tc.task == task) {
            return true;
        }
    }
    return false;
}

static int64_t samples_to_ms(int64_t samples) {
    // 16 kHz → 1 ms per 16 samples.
    return samples / (k_native_sample_rate / 1000);
}

static int64_t samples_to_us(int64_t samples) {
    // Derived from the absolute sample count on every read rather than
    // accumulated per feed: 1000000/16000 is not an integer, so a running
    // += would truncate ~0.5 us per odd-length chunk and drift without
    // bound over a long stream.
    return samples * 1000000 / k_native_sample_rate;
}

// The framework identifies speakers by opaque string; the C ABI uses a 1-based
// int32 (0 = no attribution). Assign indices in first-appearance order so the
// numbering is stable within a result and reproducible across runs.
class SpeakerIndexer {
public:
    int32_t index_of(const std::string & id) {
        if (id.empty()) {
            return 0;
        }
        const auto it = ids_.find(id);
        if (it != ids_.end()) {
            return it->second;
        }
        const auto next = static_cast<int32_t>(ids_.size()) + 1;
        ids_.emplace(id, next);
        return next;
    }

private:
    std::unordered_map<std::string, int32_t> ids_;
};

// Attach words to the segment whose time span contains them, and stamp each
// segment's [first_word, n_words) window. Both vectors are assumed sorted by
// start time, which every framework family produces. Words that fall past the
// last segment stay attributed to it rather than being dropped, so the
// forward/backward indices the C ABI publishes always resolve.
static void tie_words_to_segments(std::vector<transcribe_session::SegmentEntry> & segments,
                                  std::vector<transcribe_session::WordEntry> &    words) {
    if (segments.empty() || words.empty()) {
        return;
    }
    size_t wi = 0;
    for (size_t si = 0; si < segments.size() && wi < words.size(); ++si) {
        const size_t first    = wi;
        const bool   last_seg = (si + 1 == segments.size());
        while (wi < words.size() && (last_seg || words[wi].t0_ms <= segments[si].t1_ms)) {
            words[wi].seg_index = static_cast<int>(si);
            ++wi;
        }
        segments[si].n_words = static_cast<int>(wi - first);
        segments[si].first_word = static_cast<int>(first);
    }
}

// Stamp each segment with the speaker whose turn overlaps it most. Speaker
// turns may overlap each other (crosstalk), so "largest overlap" is the only
// attribution that stays well defined; a segment no turn covers keeps 0.
static void attribute_segment_speakers(
    std::vector<transcribe_session::SegmentEntry> &             segments,
    const std::vector<transcribe_session::SpeakerSegmentEntry> & turns) {
    if (turns.empty()) {
        return;
    }
    for (auto & seg : segments) {
        int64_t best_overlap = 0;
        int32_t best_speaker = 0;
        for (const auto & turn : turns) {
            const int64_t lo = std::max(seg.t0_ms, turn.t0_ms);
            const int64_t hi = std::min(seg.t1_ms, turn.t1_ms);
            const int64_t overlap = hi - lo;
            if (overlap > best_overlap) {
                best_overlap = overlap;
                best_speaker = turn.speaker_id;
            }
        }
        seg.speaker_id = best_speaker;
    }
}

// Map a framework TaskResult into the family-agnostic result storage on
// transcribe_session. Single source of truth for the two result views; called
// by adapter_run() and by stream_finalize().
static void map_result_into(transcribe_session * ctx, const TaskResult & result) {
    ctx->tokens.clear();
    ctx->words.clear();
    ctx->segments.clear();
    ctx->speaker_segments.clear();
    ctx->full_text.clear();
    ctx->raw_text.clear();
    ctx->detected_language.clear();
    ctx->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    ctx->has_result = false;

    if (result.text_output.has_value()) {
        ctx->full_text = result.text_output->text;
        ctx->raw_text = result.text_output->text;
        ctx->detected_language = result.text_output->language;
    }

    for (const auto & seg : result.speech_segments) {
        transcribe_session::SegmentEntry entry;
        entry.t0_ms = samples_to_ms(seg.span.start_sample);
        entry.t1_ms = samples_to_ms(seg.span.end_sample);
        entry.text = seg.text;
        entry.first_word = static_cast<int>(ctx->words.size());
        entry.n_words = 0;
        ctx->segments.push_back(entry);
    }

    for (const auto & wt : result.word_timestamps) {
        transcribe_session::WordEntry entry;
        entry.text = wt.word;
        entry.t0_ms = samples_to_ms(wt.span.start_sample);
        entry.t1_ms = samples_to_ms(wt.span.end_sample);
        entry.seg_index = 0;
        entry.first_token = 0;
        entry.n_tokens = 0;
        ctx->words.push_back(entry);
    }

    tie_words_to_segments(ctx->segments, ctx->words);

    SpeakerIndexer speakers;
    for (const auto & turn : result.speaker_turns) {
        transcribe_session::SpeakerSegmentEntry entry;
        entry.t0_ms = samples_to_ms(turn.span.start_sample);
        entry.t1_ms = samples_to_ms(turn.span.end_sample);
        entry.speaker_id = speakers.index_of(turn.speaker_id);
        entry.p = turn.confidence;
        ctx->speaker_segments.push_back(entry);
    }

    attribute_segment_speakers(ctx->segments, ctx->speaker_segments);

    if (!ctx->words.empty()) {
        ctx->result_kind = TRANSCRIBE_TIMESTAMPS_WORD;
    } else if (!ctx->segments.empty()) {
        ctx->result_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    }
    ctx->has_result = result.text_output.has_value() || !result.speech_segments.empty() ||
                      !result.word_timestamps.empty() || !result.speaker_turns.empty() ||
                      (result.audio_output.has_value() && !result.audio_output->samples.empty());  // audio-only tasks (separation)
}

// Which typed extensions the adapter understands, per family and slot. The
// framework itself has no ext mechanism - family knobs are TaskRequest options
// - so every entry here is a translation, and a family with no entry keeps the
// old answer (no ext surface at all).
bool adapter_family_accepts_ext(const std::string & family, transcribe_ext_slot slot, uint32_t kind) {
    if (family == "voxtral_realtime") {
        return slot == TRANSCRIBE_EXT_SLOT_STREAM && kind == TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM;
    }
    return false;
}

// Validate a stream extension without touching session state; called from
// stream_validate, i.e. BEFORE the dispatcher clears the previous snapshot, so
// a rejected knob cannot destroy a transcript. Mirrors the arch's rules:
// num_delay_tokens -1 or 1..15 or 30; min_decode_interval_ms >= -1.
transcribe_status adapter_check_stream_ext(const std::string & family,
                                           const transcribe_stream_params * stream_params) {
    const transcribe_ext * fam = (stream_params != nullptr) ? stream_params->family : nullptr;
    if (fam == nullptr) {
        return TRANSCRIBE_OK;
    }
    if (family == "voxtral_realtime") {
        if (const transcribe_status st = transcribe_ext_check(
                fam, TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM,
                sizeof(transcribe_voxtral_realtime_stream_ext));
            st != TRANSCRIBE_OK) {
            return st;
        }
        const auto * vx = reinterpret_cast<const transcribe_voxtral_realtime_stream_ext *>(fam);
        const int32_t nt = vx->num_delay_tokens;
        if (nt != -1 && !((nt >= 1 && nt <= 15) || nt == 30)) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        if (vx->min_decode_interval_ms < -1) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        return TRANSCRIBE_OK;
    }
    // An extension pointed at a family with no surface: the generic contract
    // says probe first, so reject rather than silently ignore it.
    return TRANSCRIBE_ERR_INVALID_ARG;
}

// Translate an already-validated stream extension into the request options the
// framework session reads.
void adapter_apply_stream_ext(TaskRequest & request,
                              const std::string & family,
                              const transcribe_stream_params * stream_params) {
    const transcribe_ext * fam = (stream_params != nullptr) ? stream_params->family : nullptr;
    if (fam == nullptr || family != "voxtral_realtime") {
        return;
    }
    const auto * vx = reinterpret_cast<const transcribe_voxtral_realtime_stream_ext *>(fam);
    if (vx->num_delay_tokens >= 0) {
        request.options["num_delay_tokens"] = std::to_string(vx->num_delay_tokens);
    }
    if (vx->min_decode_interval_ms >= 0) {
        request.options["min_decode_interval_ms"] = std::to_string(vx->min_decode_interval_ms);
    }
}

// Forward run_params → TaskRequest.options using the transcribe ABI naming so a
// future per-family option bridge can consume them; unknown keys are passed
// through verbatim and ignored by families that do not recognize them.
static void apply_run_params(TaskRequest & request, const transcribe_run_params * params) {
    if (params == nullptr) {
        return;
    }
    request.options["language"] = params->language != nullptr ? params->language : "";
    if (params->target_language != nullptr) {
        request.options["target_language"] = params->target_language;
    }
    switch (params->task) {
        case TRANSCRIBE_TASK_TRANSCRIBE: request.options["task"] = "transcribe"; break;
        case TRANSCRIBE_TASK_TRANSLATE:  request.options["task"] = "translate";  break;
    }
    request.options["timestamps"] = std::to_string(static_cast<int>(params->timestamps));
    request.options["pnc"] = std::to_string(static_cast<int>(params->pnc));
    request.options["itn"] = std::to_string(static_cast<int>(params->itn));
    request.options["diarize"] = std::to_string(static_cast<int>(params->diarize));
    request.options["keep_special_tags"] = params->keep_special_tags ? "true" : "false";
    request.options["spec_k_drafts"] = std::to_string(params->spec_k_drafts);
}

// Re-blocks an arbitrary caller feed into the fixed-size, contiguous chunks a
// framework streaming session requires. Uses the unified engine::runtime::StreamChunker.
using StreamChunker = ::engine::runtime::StreamChunker;

// An adapter-wrapped model. Owns the framework ILoadedVoiceModel; the
// framework frees the gguf_context / weights / decode state via the
// ILoadedVoiceModel RAII destructor when this object is deleted by
// transcribe_model_free().
class AdapterModel final : public transcribe_model {
public:
    AdapterModel(std::unique_ptr<ILoadedVoiceModel> framework_model)
        : framework_model_(std::move(framework_model)) {}

    void configure(const transcribe::Arch * arch,
                   std::string variant,
                   BackendConfig backend_config,
                   std::string backend_label,
                   const CapabilitySet & caps) {
        this->arch = arch;
        this->variant = std::move(variant);
        this->backend_config_ = backend_config;
        this->backend = std::move(backend_label);
        caps_ = caps;

        // Apply capabilities into the public transcribe_capabilities + feature
        // bits. Ordering matters: zero-fill caps, set scalar gate fields, then
        // set_languages() (which republishes caps.languages + n_languages into
        // the model-owned pointer chain on the base class).
        transcribe_capabilities_init(&this->caps);
        this->caps.struct_size = sizeof(transcribe_capabilities);
        this->caps.native_sample_rate = k_native_sample_rate;
        this->caps.max_timestamp_kind =
            caps_has_task(caps_, VoiceTaskKind::Asr) && caps_.supports_timestamps
                ? TRANSCRIBE_TIMESTAMPS_WORD
                : (caps_has_task(caps_, VoiceTaskKind::Vad)
                       ? TRANSCRIBE_TIMESTAMPS_SEGMENT
                       : TRANSCRIBE_TIMESTAMPS_NONE);
        this->caps.supports_language_detect = !caps_.languages.empty();
        this->caps.supports_translate = false;
        this->caps.supports_streaming = caps_supports_streaming(caps_);
        this->caps.supports_spec_decode = caps_.supports_speculative_decode;
        this->caps.max_audio_ms = 0;  // framework sessions chunk internally; treat as unbounded.
        this->caps.n_translate_target_languages = 0;
        this->caps.translate_target_languages = nullptr;
        set_languages(caps_.languages);

        set_feature(this, TRANSCRIBE_FEATURE_CANCELLATION, caps_.supports_cancellation);
        set_feature(this, TRANSCRIBE_FEATURE_DIARIZATION,
                    caps_has_task(caps_, VoiceTaskKind::Diarization));
    }

    ILoadedVoiceModel * framework_model() const noexcept { return framework_model_.get(); }
    const CapabilitySet & framework_caps() const noexcept { return caps_; }
    const BackendConfig & backend_config() const noexcept { return backend_config_; }

private:
    std::unique_ptr<ILoadedVoiceModel> framework_model_;
    CapabilitySet caps_;
    BackendConfig backend_config_{BackendType::BestAvailable, 0, 1};
};

// An adapter-wrapped session. Owns a framework IVoiceTaskSession and lazily
// materializes an offline session (transcribe_run) and/or a streaming session
// (transcribe_stream_*) as each surface is exercised.
class AdapterSession final : public transcribe_session {
public:
    AdapterSession(AdapterModel * model, SessionOptions options)
        : model_(model), session_options_(std::move(options)) {
    }

    IOfflineVoiceTaskSession * ensure_offline() {
        if (offline_ == nullptr && model_ != nullptr) {
            TaskSpec spec;
            if (!pick_task_spec(model_->framework_caps(), RunMode::Offline, spec)) {
                return nullptr;
            }
            offline_ = model_->framework_model()->create_task_session(spec, session_options_);
        }
        return offline_ == nullptr ? nullptr
            : dynamic_cast<IOfflineVoiceTaskSession *>(offline_.get());
    }

    IStreamingVoiceTaskSession * ensure_streaming() {
        if (streaming_ == nullptr && model_ != nullptr) {
            TaskSpec spec;
            if (!pick_task_spec(model_->framework_caps(), RunMode::Streaming, spec)) {
                return nullptr;
            }
            streaming_ = model_->framework_model()->create_task_session(spec, session_options_);
        }
        return streaming_ == nullptr ? nullptr
            : dynamic_cast<IStreamingVoiceTaskSession *>(streaming_.get());
    }

    // The framework family id this session belongs to; the ext translators
    // key off it.
    std::string framework_family() const {
        return model_ != nullptr && model_->framework_model() != nullptr
            ? model_->framework_model()->metadata().family
            : std::string();
    }

    // Bridge transcribe_set_abort_callback onto the framework's progress
    // contract: the session polls RunControl at its stage / decode-step
    // boundaries and unwinds with ProgressCanceled when our callback declines.
    // Families that never poll simply run to completion (their capability
    // set says so: supports_cancellation == false).
    void install_abort_bridge(IVoiceTaskSession * session) {
        if (session == nullptr) {
            return;
        }
        session->set_progress_callback([this](const engine::runtime::ProgressInfo &) {
            return !this->poll_abort();
        });
    }

    transcribe_status run_offline(const float * pcm, int n_samples, const transcribe_run_params * params) {
        if (pcm == nullptr || n_samples <= 0) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        auto * offline = ensure_offline();
        if (offline == nullptr) {
            return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
        }
        try {
            TaskRequest request;
            request.audio_input = AudioBuffer{
                k_native_sample_rate,
                1,
                std::vector<float>(pcm, pcm + static_cast<std::size_t>(n_samples))};
            apply_run_params(request, params);
            offline->prepare(build_preparation_request(request));
            install_abort_bridge(offline);
            const TaskResult result = offline->run(request);
            clear_result();
            map_result_into(this, result);
            return TRANSCRIBE_OK;
        } catch (const engine::runtime::ProgressCanceled &) {
            // The transcribe abort callback fired (poll_abort set was_aborted);
            // same contract as a builtin arch: ERR_ABORTED, no partial result.
            clear_result();
            return TRANSCRIBE_ERR_ABORTED;
        } catch (const std::exception & e) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter run failed: %s", e.what());
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    transcribe_status run_batch(const float * const * pcm, const int * n_samples, int n,
                                const transcribe_run_params * params) {
        if (pcm == nullptr || n_samples == nullptr || n <= 0) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        auto * offline = ensure_offline();
        if (offline == nullptr) {
            return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
        }
        try {
            std::vector<TaskRequest> requests;
            requests.reserve(n);
            for (int i = 0; i < n; ++i) {
                TaskRequest req;
                if (pcm[i] != nullptr && n_samples[i] > 0) {
                    req.audio_input = AudioBuffer{
                        k_native_sample_rate,
                        1,
                        std::vector<float>(pcm[i], pcm[i] + static_cast<std::size_t>(n_samples[i]))};
                }
                apply_run_params(req, params);
                requests.push_back(std::move(req));
            }

            if (!requests.empty()) {
                offline->prepare(build_preparation_request(requests[0]));
            }

            install_abort_bridge(offline);
            const std::vector<TaskResult> results = offline->run_batch(requests);

            batch_results.clear();
            batch_results.reserve(n);
            for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
                ResultSet rs;
                if (i < results.size()) {
                    clear_result();
                    map_result_into(this, results[i]);
                    rs.full_text = full_text;
                    rs.raw_text = raw_text;
                    rs.segments = segments;
                    rs.words = words;
                    rs.tokens = tokens;
                    rs.has_result = has_result;
                    rs.result_kind = result_kind;
                    rs.status = TRANSCRIBE_OK;
                } else {
                    rs.has_result = false;
                    rs.status = TRANSCRIBE_ERR_BACKEND;
                }
                batch_results.push_back(std::move(rs));
            }

            if (!batch_results.empty() && batch_results[0].has_result) {
                clear_result();
                full_text = batch_results[0].full_text;
                raw_text = full_text;
                segments = batch_results[0].segments;
                words = batch_results[0].words;
                tokens = batch_results[0].tokens;
            }

            return TRANSCRIBE_OK;
        } catch (const engine::runtime::ProgressCanceled &) {
            clear_result();
            batch_results.clear();
            return TRANSCRIBE_ERR_ABORTED;
        } catch (const std::exception & e) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter run_batch failed: %s", e.what());
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    transcribe_status begin_stream(const transcribe_run_params * run_params,
                                   const transcribe_stream_params * stream_params) {
        auto * streaming = ensure_streaming();
        if (streaming == nullptr) {
            return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
        }
        try {
            TaskRequest request;
            apply_run_params(request, run_params);
            adapter_apply_stream_ext(request, framework_family(), stream_params);
            // prepare() BEFORE start_stream(): every framework session gates
            // its streaming entry points on require_prepared(), and the
            // default start_stream() delegates to reset() - which is itself
            // gated - so an unprepared begin throws on the very first call.
            //
            // A stream has no audio yet, so build_preparation_request() would
            // hand the session an EMPTY audio contract - and ASR families
            // reject that ("prepare() requires an audio contract"), which made
            // the C ABI streaming path unusable for every framework ASR family
            // reached through this adapter. Supply the contract the C ABI
            // itself guarantees instead: 16 kHz mono, length not yet known.
            SessionPreparationRequest prep = build_preparation_request(request);
            if (!prep.audio.has_value()) {
                prep.audio = AudioPreparationContract{k_native_sample_rate, 1, 0};
            }
            streaming->prepare(prep);
            install_abort_bridge(streaming);
            streaming->start_stream(request);
            chunker_.reset(preferred_chunk_samples(streaming));
            publish_cursors();
            return TRANSCRIBE_OK;
        } catch (const std::exception & e) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter stream_begin failed: %s", e.what());
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    transcribe_status feed_stream(const float * pcm, int n_samples, transcribe_stream_update * update) {
        auto * streaming = ensure_streaming();
        if (streaming == nullptr) {
            return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
        }
        if (n_samples < 0 || (n_samples > 0 && pcm == nullptr)) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        try {
            bool changed = false;
            // One caller feed can span many family chunks, so the abort poll
            // lives per chunk rather than per feed; the samples the abort
            // skipped stay buffered and resume contiguously on the next feed.
            chunker_.feed(pcm, n_samples, [&](const AudioChunk & chunk) {
                if (poll_abort()) {
                    return false;
                }
                const StreamEvent event = streaming->process_audio_chunk(chunk);
                changed = fold_stream_event(event) || changed;
                return true;
            });
            publish_cursors();
            publish_update(update, changed);
            if (was_aborted || poll_abort()) {
                return TRANSCRIBE_ERR_ABORTED;
            }
            return TRANSCRIBE_OK;
        } catch (const engine::runtime::ProgressCanceled &) {
            return TRANSCRIBE_ERR_ABORTED;
        } catch (const std::exception & e) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter stream_feed failed: %s", e.what());
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    transcribe_status finalize_stream(transcribe_stream_update * update) {
        auto * streaming = ensure_streaming();
        if (streaming == nullptr) {
            return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
        }
        try {
            // Push the sub-chunk tail through before asking for the final
            // result, otherwise the last partial block is silently dropped.
            bool changed = false;
            chunker_.flush([&](const AudioChunk & chunk) {
                const StreamEvent event = streaming->process_audio_chunk(chunk);
                changed = fold_stream_event(event) || changed;
                return true;
            });
            // finish_stream() is the streaming-surface entry point; it
            // defaults to finalize() but lets a family run stream-specific
            // teardown first.
            const TaskResult result = streaming->finish_stream();
            clear_result();
            map_result_into(this, result);
            publish_cursors();
            // Everything fed is committed once the stream is closed.
            stream_audio_committed_us = stream_audio_input_us;
            publish_update(update, changed || has_result);
            return TRANSCRIBE_OK;
        } catch (const engine::runtime::ProgressCanceled &) {
            return TRANSCRIBE_ERR_ABORTED;
        } catch (const std::exception & e) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter stream_finalize failed: %s", e.what());
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    void reset_stream() {
        if (streaming_ != nullptr) {
            auto * s = dynamic_cast<IStreamingVoiceTaskSession *>(streaming_.get());
            if (s != nullptr) {
                // reset() is prepared-gated like the rest of the streaming
                // surface; a session reset before any begin has nothing to
                // release, so skip rather than throw out of a void hook.
                try {
                    s->reset();
                } catch (const std::exception & e) {
                    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                                        "adapter stream_reset: framework reset skipped: %s", e.what());
                }
                chunker_.reset(preferred_chunk_samples(s));
            }
        }
        publish_cursors();
    }

private:
    static int64_t preferred_chunk_samples(IStreamingVoiceTaskSession * streaming) {
        if (streaming == nullptr) {
            return 0;
        }
        const StreamingPolicy policy = streaming->streaming_policy();
        if (policy.preferred_audio_chunk_samples > 0) {
            return policy.preferred_audio_chunk_samples;
        }
        if (policy.preferred_audio_chunk_seconds > 0.0) {
            return static_cast<int64_t>(policy.preferred_audio_chunk_seconds * k_native_sample_rate);
        }
        return 0;
    }

    // Fold one framework StreamEvent into the session's result snapshot.
    // Returns whether anything observable moved.
    //
    // Each category is replaced only when the event actually carries data for
    // it. A VAD event stream in particular emits bare SpeechStart/SpeechEnd
    // markers with no segment payload; wiping the accumulated segments on
    // those would drop the hypothesis mid-stream. result_kind and has_result
    // are likewise derived from what the session now HOLDS, not from what this
    // one event happened to contain.
    bool fold_stream_event(const StreamEvent & event) {
        bool changed = false;

        if (event.partial_text.has_value()) {
            full_text         = event.partial_text->text;
            raw_text          = full_text;
            detected_language = event.partial_text->language;
            changed           = true;
        }

        if (!event.word_timestamps.empty()) {
            words.clear();
            for (const auto & wt : event.word_timestamps) {
                WordEntry entry;
                entry.text  = wt.word;
                entry.t0_ms = samples_to_ms(wt.span.start_sample);
                entry.t1_ms = samples_to_ms(wt.span.end_sample);
                words.push_back(entry);
            }
            changed = true;
        }

        std::vector<SegmentEntry> event_segments;
        for (const auto & va : event.voice_activity) {
            if (!va.segment.has_value()) {
                continue;
            }
            const auto & seg = *va.segment;
            SegmentEntry entry;
            entry.t0_ms      = samples_to_ms(seg.span.start_sample);
            entry.t1_ms      = samples_to_ms(seg.span.end_sample);
            entry.text       = seg.text;
            entry.first_word = 0;
            entry.n_words    = 0;
            event_segments.push_back(std::move(entry));
        }
        if (!event_segments.empty()) {
            segments = std::move(event_segments);
            changed  = true;
        }

        if (!event.speaker_turns.empty()) {
            SpeakerIndexer speakers;
            speaker_segments.clear();
            for (const auto & turn : event.speaker_turns) {
                SpeakerSegmentEntry entry;
                entry.t0_ms      = samples_to_ms(turn.span.start_sample);
                entry.t1_ms      = samples_to_ms(turn.span.end_sample);
                entry.speaker_id = speakers.index_of(turn.speaker_id);
                entry.p          = turn.confidence;
                speaker_segments.push_back(entry);
            }
            changed = true;
        }

        if (changed) {
            tie_words_to_segments(segments, words);
            attribute_segment_speakers(segments, speaker_segments);
        }

        if (!words.empty()) {
            result_kind = TRANSCRIBE_TIMESTAMPS_WORD;
        } else if (!segments.empty()) {
            result_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;
        } else {
            result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        }
        has_result = !full_text.empty() || !words.empty() || !segments.empty() || !speaker_segments.empty();

        // event.is_final is informational here: the dispatcher owns the
        // ACTIVE -> FINISHED transition and forces update->is_final itself.
        return changed;
    }

    // Mirror the chunker's sample cursors onto the base session fields.
    // transcribe-session.h designates these as hook-owned state the dispatcher
    // reads back, so they must be the INHERITED members - a same-named member
    // here would shadow them and leave the dispatcher reading zeros.
    void publish_cursors() {
        stream_audio_input_us     = samples_to_us(chunker_.input_samples());
        stream_audio_committed_us = samples_to_us(chunker_.committed_samples());
    }

    // Fill the caller's update with the fields this hook owns. Revision,
    // result_changed, committed_changed, tentative_changed and is_final all
    // belong to the dispatcher (publish_observable_delta / publish_stream_-
    // update_tail); writing the two trailing bool fields here would also
    // overrun any caller whose struct_size stops at buffered_ms, which the ABI
    // explicitly permits.
    //
    // stream_revision is advanced only on a real change: publish_observable_-
    // delta forces result_changed true whenever the hook moved the counter, so
    // an unconditional bump would report every feed as a change.
    void publish_update(transcribe_stream_update * update, bool changed) {
        if (changed) {
            ++stream_revision;
        }
        n_committed_segments = static_cast<int>(segments.size());
        n_committed_words    = static_cast<int>(words.size());
        n_committed_tokens   = static_cast<int>(tokens.size());
        if (update == nullptr) {
            return;
        }
        update->result_changed     = changed;
        update->input_received_ms  = stream_audio_input_us / 1000;
        update->audio_committed_ms = stream_audio_committed_us / 1000;
        update->buffered_ms        = samples_to_us(chunker_.buffered_samples()) / 1000;
    }

    AdapterModel *                     model_ = nullptr;
    SessionOptions                     session_options_;
    std::unique_ptr<IVoiceTaskSession> offline_;
    std::unique_ptr<IVoiceTaskSession> streaming_;
    StreamChunker                      chunker_;
};

static SessionOptions build_session_options(const transcribe_session_params * params,
                                            const BackendConfig & backend_config) {
    struct transcribe_session_params defaults;
    transcribe_session_params_init(&defaults);
    const auto * p = params != nullptr ? params : &defaults;

    SessionOptions options;
    options.backend = backend_config;

    int threads = (p->n_threads > 0) ? p->n_threads : 1;
    options.backend.threads = threads;
    options.backend.device = 0;

    if (p->kv_type == TRANSCRIBE_KV_TYPE_F32) {
        options.options["kv_type"] = "f32";
    } else if (p->kv_type == TRANSCRIBE_KV_TYPE_F16) {
        options.options["kv_type"] = "f16";
    }
    const int32_t n_ctx = transcribe_session_params_n_ctx(p);
    if (n_ctx > 0) {
        options.options["n_ctx"] = std::to_string(n_ctx);
    }
    return options;
}

// The default registry instantiates every compiled-in family loader (~46 of
// them). The loaders are stateless factories and ModelRegistry::load() is
// const, so one shared instance serves every model load instead of rebuilding
// the catalog per transcribe_open(). Function-local static: thread-safe
// initialization, and no static-init-order dependency at library load.
static const ModelRegistry & default_registry() {
    static const ModelRegistry registry = make_default_registry();
    return registry;
}

// --- Static Arch hooks -------------------------------------------------------
// These and the per-family table below stay in the anonymous (internal)
// namespace: they reference AdapterModel/AdapterSession and are only ever
// reached through adapter_find_arch()'s external entry point.

transcribe_status adapter_load_impl(Loader & loader,
                                    const transcribe_model_load_params * params,
                                    struct transcribe_model ** out_model) {
    if (out_model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_model = nullptr;

    const char * family = loader.arch().c_str();
    if (family == nullptr || family[0] == '\0') {
        return TRANSCRIBE_ERR_UNSUPPORTED_ARCH;
    }

    transcribe_backend_request backend_request = TRANSCRIBE_BACKEND_AUTO;
    if (params != nullptr) {
        backend_request = params->backend;
    }

    try {
        ModelLoadRequest request;
        request.model_path = loader.path();
        request.family_hint = std::string(family);

        std::unique_ptr<ILoadedVoiceModel> framework_model = default_registry().load(request);
        if (framework_model == nullptr) {
            return TRANSCRIBE_ERR_UNSUPPORTED_ARCH;
        }

        const auto & meta = framework_model->metadata();
        const auto & caps = framework_model->capabilities();

        auto * model = new AdapterModel(std::move(framework_model));
        model->configure(
            adapter_find_arch(family),
            meta.variant,
            BackendConfig{backend_type_from_request(backend_request), 0, 1},
            backend_label_from_request(backend_request),
            caps);
        *out_model = model;
        return TRANSCRIBE_OK;
    } catch (const std::exception & e) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
            "adapter load failed for family '%s': %s", family, e.what());
        // No framework loader registered / spec unreadable under this build.
        return TRANSCRIBE_ERR_UNSUPPORTED_ARCH;
    }
}

transcribe_status adapter_init_context_impl(struct transcribe_model * model,
                                            const struct transcribe_session_params * params,
                                            struct transcribe_session ** out_session) {
    if (out_session == nullptr || model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_session = nullptr;

    auto * adapter_model = dynamic_cast<AdapterModel *>(model);
    if (adapter_model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    try {
        SessionOptions options = build_session_options(params, adapter_model->backend_config());
        auto * session = new AdapterSession(adapter_model, std::move(options));
        session->model = model;
        session->n_threads = (params != nullptr) ? params->n_threads : 0;
        session->kv_type = (params != nullptr) ? params->kv_type : TRANSCRIBE_KV_TYPE_AUTO;
        session->n_ctx = transcribe_session_params_n_ctx(params);
        *out_session = session;
        return TRANSCRIBE_OK;
    } catch (const std::exception & e) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter init_context failed: %s", e.what());
        return TRANSCRIBE_ERR_BACKEND;
    }
}

transcribe_status adapter_run_impl(struct transcribe_session * ctx, const float * pcm, int n_samples,
                                   const struct transcribe_run_params * params) {
    auto * session = static_cast<AdapterSession *>(ctx);
    return session->run_offline(pcm, n_samples, params);
}

transcribe_status adapter_run_batch_impl(struct transcribe_session * ctx,
                                         const float * const * pcm,
                                         const int * n_samples,
                                         int n,
                                         const struct transcribe_run_params * params) {
    auto * session = static_cast<AdapterSession *>(ctx);
    return session->run_batch(pcm, n_samples, n, params);
}

transcribe_status adapter_stream_validate_impl(const struct transcribe_session * ctx,
                                               const struct transcribe_run_params * run_params,
                                               const struct transcribe_stream_params * stream_params) {
    if (ctx == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (run_params != nullptr && run_params->struct_size > 0 &&
        run_params->struct_size < sizeof(transcribe_run_params)) {
        return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
    }
    if (stream_params != nullptr && stream_params->struct_size > 0 &&
        stream_params->struct_size < sizeof(transcribe_stream_params)) {
        return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
    }
    if (stream_params != nullptr && stream_params->family != nullptr) {
        const auto * model = dynamic_cast<const AdapterModel *>(ctx->model);
        const std::string family = (model != nullptr && model->framework_model() != nullptr)
            ? model->framework_model()->metadata().family
            : std::string();
        if (const transcribe_status st = adapter_check_stream_ext(family, stream_params);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    return TRANSCRIBE_OK;
}

transcribe_status adapter_run_validate_impl(const struct transcribe_session * ctx,
                                            const struct transcribe_run_params * params) {
    if (ctx == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (params != nullptr && params->struct_size > 0 &&
        params->struct_size < sizeof(transcribe_run_params)) {
        return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
    }
    return TRANSCRIBE_OK;
}

transcribe_status adapter_stream_begin_impl(struct transcribe_session * ctx,
                                            const struct transcribe_run_params * run_params,
                                            const struct transcribe_stream_params * stream_params) {
    auto * session = static_cast<AdapterSession *>(ctx);
    return session->begin_stream(run_params, stream_params);
}

transcribe_status adapter_stream_feed_impl(struct transcribe_session * ctx, const float * pcm, int n_samples,
                                           struct transcribe_stream_update * update) {
    auto * session = static_cast<AdapterSession *>(ctx);
    return session->feed_stream(pcm, n_samples, update);
}

transcribe_status adapter_stream_finalize_impl(struct transcribe_session * ctx,
                                               struct transcribe_stream_update * update) {
    auto * session = static_cast<AdapterSession *>(ctx);
    return session->finalize_stream(update);
}

void adapter_stream_reset_impl(struct transcribe_session * ctx) {
    auto * session = static_cast<AdapterSession *>(ctx);
    session->reset_stream();
}

bool adapter_accepts_ext_kind_impl(const struct transcribe_model * model,
                                   transcribe_ext_slot slot, uint32_t kind) {
    // Most framework families surface their knobs as TaskRequest options and
    // have no typed ext at all. The exceptions are families whose retired
    // transcribe.cpp arch published one: the adapter translates those so the
    // public surface survives the retirement (Phase 10.5).
    const auto * adapter_model = dynamic_cast<const AdapterModel *>(model);
    if (adapter_model == nullptr || adapter_model->framework_model() == nullptr) {
        return false;
    }
    return adapter_family_accepts_ext(adapter_model->framework_model()->metadata().family, slot, kind);
}

// Per-family Arch instances. Each carries a static-lifetime name matching the
// GGUF `general.architecture` an audio.cpp converter writes (identical to the
// framework family id); the function pointers are shared by design because the
// adapter is family-agnostic at the dispatch surface — load() recovers the
// family from loader.arch() and routes through the ModelRegistry. The table is
// consulted by adapter_find_arch() only AFTER the builtin transcribe.cpp table
// so the overlapping names still carried there (voxtral_realtime, moss) keep
// their transcribe.cpp handlers until Phase 10.5 retires them; qwen3_asr's
// was retired first, so that name resolves here alone.
static const Arch adapter_archs[] = {
    {"silero_vad",          &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"marblenet_vad",       &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"citrinet_asr",        &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"nemotron_asr",        &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"qwen3_asr",           &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"fun_asr_nano",        &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"sense_asr",           &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"hviske_asr",          &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"higgs_audio_stt",     &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"vibevoice_asr",       &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"kroko_asr",           &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"parakeet_tdt",        &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"voxtral_realtime",    &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"sortformer_diar",     &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"roformer",            &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
    {"moss",                &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl,
     &adapter_run_batch_impl, &adapter_stream_validate_impl, &adapter_stream_begin_impl,
     &adapter_stream_feed_impl, &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, &adapter_run_validate_impl},
};
constexpr size_t k_n_adapter_archs = sizeof(adapter_archs) / sizeof(adapter_archs[0]);

}  // namespace

// Public entry point called by find_arch() (in transcribe-arch.cpp) after the
// builtin transcribe.cpp families fail to match. `adapter_archs` lives in the
// anonymous namespace above but remains visible here via the enclosing
// `transcribe` namespace; only this function has external linkage.
std::string adapter_sniff_framework_family(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return {};
    }
    try {
        ModelLoadRequest request;
        request.model_path = transcribe::path_from_utf8(path);
        // For an audio.cpp GGUF the registry resolves the family from the
        // file's own `audiocpp.model_spec.family` KV; for a safetensors
        // directory it probes can_load(). Either way inspect() is the single
        // answer, and a family this build does not carry throws (caught below).
        const auto inspection = default_registry().inspect(request);
        const std::string & family = inspection.metadata.family;
        // Only report families the adapter table actually dispatches; a
        // framework loader with no Arch entry would otherwise resolve here and
        // then fail in find_arch() with a less useful status.
        if (family.empty() || adapter_find_arch(family.c_str()) == nullptr) {
            return {};
        }
        return family;
    } catch (const std::exception & e) {
        // No registered loader claims this path. Expected for GGUF and for
        // unrelated files; the caller falls back to the GGUF error contract.
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                            "adapter: no framework loader claims '%s': %s", path, e.what());
        return {};
    }
}

const Arch * adapter_find_arch(const char * name) {
    if (name == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < k_n_adapter_archs; ++i) {
        const Arch * a = &adapter_archs[i];
        if (a->name != nullptr && std::strcmp(a->name, name) == 0) {
            return a;
        }
    }
    return nullptr;
}

}  // namespace transcribe
