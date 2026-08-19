// transcribe-arch-adapter.cpp - bridge from audio.cpp's C++ model vtable into
// the transcribe C ABI `Arch` dispatch trait (V6 plan sub-task 0.H). See the
// sibling header for the lifetime and ABI mismatch rationale.
//
// NOTE (Phase 0 structural): this file is NOT compiled-validated until ggml
// converges (0.L). It implements the documented bridge surface so the
// unified C ABI can route audio.cpp families through find_arch(). Reachable
// families are those whose audio.cpp converter writes a GGUF carrying a
// `general.architecture` equal to the family id (e.g. citrinet_asr); safetensors
// families only reach here once the dispatcher gains a pre-Loader::open sniff
// (Phase 1). Backend binding is per-session inside the framework; the adapter
// reports the caller's requested surface label at model scope and resolves the
// concrete device when a session is eventually created.

#include "transcribe-arch-adapter.h"

#include "transcribe-log.h"
#include "transcribe-loader.h"
#include "transcribe-model.h"
#include "transcribe-session.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
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
using engine::runtime::SessionOptions;
using engine::runtime::StreamEvent;
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

    int seg_index = 0;
    std::vector<int> words_per_segment;
    for (const auto & seg : result.speech_segments) {
        transcribe_session::SegmentEntry entry;
        entry.t0_ms = samples_to_ms(seg.span.start_sample);
        entry.t1_ms = samples_to_ms(seg.span.end_sample);
        entry.text = seg.text;
        entry.first_word = static_cast<int>(ctx->words.size());
        entry.n_words = 0;
        ctx->segments.push_back(entry);
        words_per_segment.push_back(0);
        ++seg_index;
    }

    for (const auto & wt : result.word_timestamps) {
        transcribe_session::WordEntry entry;
        entry.text = wt.text;
        entry.t0_ms = samples_to_ms(wt.span.start_sample);
        entry.t1_ms = samples_to_ms(wt.span.end_sample);
        entry.seg_index = 0;
        entry.first_token = 0;
        entry.n_tokens = 0;
        ctx->words.push_back(entry);
    }

    // Tie words to segments by time order so the single-shot accessors stay
    // coherent; this is a best-effort alignment (Phase 0 structural).
    if (!ctx->segments.empty() && !ctx->words.empty()) {
        size_t wi = 0;
        for (size_t si = 0; si < ctx->segments.size() && wi < ctx->words.size(); ++si) {
            int first = static_cast<int>(wi);
            while (wi < ctx->words.size() && ctx->words[wi].t0_ms <= ctx->segments[si].t1_ms) {
                ctx->words[wi].seg_index = static_cast<int>(si);
                ++wi;
            }
            ctx->segments[si].n_words = static_cast<int>(wi) - first;
            if (ctx->segments[si].n_words > 0) {
                ctx->segments[si].first_word = first;
            }
        }
    }

    for (const auto & turn : result.speaker_turns) {
        transcribe_session::SpeakerSegmentEntry entry;
        entry.t0_ms = samples_to_ms(turn.span.start_sample);
        entry.t1_ms = samples_to_ms(turn.span.end_sample);
        // SpeakerTurn carries a string id; the C ABI speaker_segments use a
        // 1-based int32. Phase 1 assigns stable indices; for now attribute 0.
        entry.speaker_id = 0;
        entry.p = turn.confidence;
        ctx->speaker_segments.push_back(entry);
    }

    if (!ctx->words.empty()) {
        ctx->result_kind = TRANSCRIBE_TIMESTAMPS_WORD;
    } else if (!ctx->segments.empty()) {
        ctx->result_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    }
    ctx->has_result = result.text_output.has_value() || !result.speech_segments.empty() ||
                      !result.word_timestamps.empty() || !result.speaker_turns.empty() ||
                      (result.audio_output.has_value() && !result.audio_output->samples.empty());  // audio-only tasks (separation)
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
        this->caps.supports_spec_decode = false;
        this->caps.max_audio_ms = 0;  // framework sessions chunk internally; treat as unbounded.
        this->caps.n_translate_target_languages = 0;
        this->caps.translate_target_languages = nullptr;
        set_languages(caps_.languages);

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
            const TaskResult result = offline->run(request);
            clear_result();
            map_result_into(this, result);
            return TRANSCRIBE_OK;
        } catch (const std::exception & e) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter run failed: %s", e.what());
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    transcribe_status run_batch_offline(const float * const * pcm, const int * n_samples, int n,
                                        const transcribe_run_params * params) {
        if (pcm == nullptr || n_samples == nullptr || n <= 0) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        auto * offline = ensure_offline();
        if (offline == nullptr) {
            return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
        }
        try {
            std::vector<ResultSet> results;
            results.reserve(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                if (n_samples[i] <= 0 || pcm[i] == nullptr) {
                    ResultSet rs;
                    rs.has_result = false;
                    rs.status = TRANSCRIBE_ERR_INVALID_ARG;
                    results.push_back(std::move(rs));
                    continue;
                }
                TaskRequest request;
                request.audio_input = AudioBuffer{
                    k_native_sample_rate, 1,
                    std::vector<float>(pcm[i], pcm[i] + static_cast<std::size_t>(n_samples[i]))};
                apply_run_params(request, params);
                offline->prepare(build_preparation_request(request));
                const TaskResult result = offline->run(request);
                clear_result();
                map_result_into(this, result);
                results.push_back(capture_result());
                if (poll_abort()) {
                    batch_results = std::move(results);
                    return TRANSCRIBE_ERR_ABORTED;
                }
            }
            batch_results = std::move(results);
            return TRANSCRIBE_OK;
        } catch (const std::exception & e) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter run_batch failed: %s", e.what());
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    transcribe_status begin_stream(const transcribe_run_params * run_params) {
        auto * streaming = ensure_streaming();
        if (streaming == nullptr) {
            return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
        }
        try {
            TaskRequest request;
            apply_run_params(request, run_params);
            streaming->start_stream(request);
            stream_audio_input_us = 0;
            stream_audio_committed_us = 0;
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
            AudioChunk chunk;
            chunk.sample_rate = k_native_sample_rate;
            chunk.channels = 1;
            // start_sample is an absolute cursor into the 16 kHz stream.
            chunk.start_sample = (stream_audio_input_us / 1000) * (k_native_sample_rate / 1000);
            if (n_samples > 0) {
                chunk.samples.assign(pcm, pcm + static_cast<std::size_t>(n_samples));
                stream_audio_input_us +=
                    static_cast<int64_t>(n_samples) * 1000000 / k_native_sample_rate;
            }
            const StreamEvent event = streaming->process_audio_chunk(chunk);
            update_stream_from_event(event, update);
            if (poll_abort()) {
                return TRANSCRIBE_ERR_ABORTED;
            }
            return TRANSCRIBE_OK;
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
            const TaskResult result = streaming->finalize();
            clear_result();
            map_result_into(this, result);
            const bool changed = has_result || result_kind != TRANSCRIBE_TIMESTAMPS_NONE;
            if (update != nullptr) {
                update->result_changed = changed;
                update->is_final = true;
                update->revision = ++stream_revision;
                update->input_received_ms = stream_audio_input_us / 1000;
                update->audio_committed_ms = stream_audio_committed_us / 1000;
                update->buffered_ms = 0;
                update->committed_changed = changed;
                update->tentative_changed = false;
            }
            return TRANSCRIBE_OK;
        } catch (const std::exception & e) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "adapter stream_finalize failed: %s", e.what());
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    void reset_stream() {
        if (streaming_ != nullptr) {
            auto * s = dynamic_cast<IStreamingVoiceTaskSession *>(streaming_.get());
            if (s != nullptr) {
                s->reset();
            }
        }
        stream_audio_input_us = 0;
        stream_audio_committed_us = 0;
    }

private:
    void update_stream_from_event(const StreamEvent & event, transcribe_stream_update * update) {
        bool changed = false;

        if (event.partial_text.has_value()) {
            full_text = event.partial_text->text;
            raw_text = full_text;
            detected_language = event.partial_text->language;
            changed = true;
        }

        if (!event.speech_segments.empty()) {
            segments.clear();
            for (const auto & seg : event.speech_segments) {
                SegmentEntry entry;
                entry.t0_ms = samples_to_ms(seg.span.start_sample);
                entry.t1_ms = samples_to_ms(seg.span.end_sample);
                entry.text = seg.text;
                entry.first_word = static_cast<int>(words.size());
                entry.n_words = 0;
                segments.push_back(entry);
            }
            changed = true;
        }

        if (!event.word_timestamps.empty()) {
            words.clear();
            for (const auto & wt : event.word_timestamps) {
                WordEntry entry;
                entry.text = wt.text;
                entry.t0_ms = samples_to_ms(wt.span.start_sample);
                entry.t1_ms = samples_to_ms(wt.span.end_sample);
                words.push_back(entry);
            }
            changed = true;
        }

        if (!event.speaker_turns.empty()) {
            speaker_segments.clear();
            for (const auto & turn : event.speaker_turns) {
                SpeakerSegmentEntry entry;
                entry.t0_ms = samples_to_ms(turn.span.start_sample);
                entry.t1_ms = samples_to_ms(turn.span.end_sample);
                entry.speaker_id = 0;
                entry.p = turn.confidence;
                speaker_segments.push_back(entry);
            }
            changed = true;
        }

        if (event.is_final) {
            // The framework signals stream closure; the dispatcher owns the
            // FINISHED transition, so we just fold the final payload in.
        }

        if (!event.word_timestamps.empty()) {
            result_kind = TRANSCRIBE_TIMESTAMPS_WORD;
        } else if (!event.speech_segments.empty()) {
            result_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;
        } else {
            result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        }
        has_result = changed;

        if (update != nullptr) {
            ++stream_revision;
            update->revision = stream_revision;
            update->result_changed = changed;
            update->input_received_ms = stream_audio_input_us / 1000;
            update->audio_committed_ms = stream_audio_committed_us / 1000;
            update->buffered_ms = 0;
            update->committed_changed = changed;
            update->tentative_changed = changed;
        }
    }

    AdapterModel * model_ = nullptr;
    SessionOptions session_options_;
    std::unique_ptr<IVoiceTaskSession> offline_;
    std::unique_ptr<IVoiceTaskSession> streaming_;

    // The dispatcher owns the committed/tentative state machine; these cursors
    // are only the adapter's local progress counters mirrored into the update.
    int64_t stream_audio_input_us = 0;
    int64_t stream_audio_committed_us = 0;
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

        ModelRegistry registry = make_default_registry();
        std::unique_ptr<ILoadedVoiceModel> framework_model = registry.load(request);
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

transcribe_status adapter_run_batch_impl(struct transcribe_session * ctx, const float * const * pcm,
                                         const int * n_samples, int n,
                                         const struct transcribe_run_params * params) {
    auto * session = static_cast<AdapterSession *>(ctx);
    return session->run_batch_offline(pcm, n_samples, n, params);
}

transcribe_status adapter_stream_begin_impl(struct transcribe_session * ctx,
                                            const struct transcribe_run_params * run_params,
                                            const struct transcribe_stream_params * /*stream_params*/) {
    auto * session = static_cast<AdapterSession *>(ctx);
    return session->begin_stream(run_params);
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

bool adapter_accepts_ext_kind_impl(const struct transcribe_model * /*model*/,
                                   transcribe_ext_slot /*slot*/, uint32_t /*kind*/) {
    // The framework surfaces family knobs through TaskRequest.options, not the
    // C ABI extension mechanism; the adapter has no ext surface to probe.
    return false;
}

// Per-family Arch instances. Each carries a static-lifetime name matching the
// GGUF `general.architecture` an audio.cpp converter writes (identical to the
// framework family id); the function pointers are shared by design because the
// adapter is family-agnostic at the dispatch surface — load() recovers the
// family from loader.arch() and routes through the ModelRegistry. The table is
// consulted by adapter_find_arch() only AFTER the builtin transcribe.cpp table
// so overlapping names (qwen3_asr, voxtral_realtime, moss) keep their
// transcribe.cpp handlers until Phase 1 consolidation.
static const Arch adapter_archs[] = {
    {"silero_vad",          &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"marblenet_vad",       &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"citrinet_asr",        &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"nemotron_asr",        &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"qwen3_asr",           &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"fun_asr_nano",        &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"sense_asr",           &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"hviske_asr",          &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"higgs_audio_stt",     &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"vibevoice_asr",       &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"kroko_asr",           &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"parakeet_tdt",        &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"voxtral_realtime",    &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"sortformer_diar",     &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"roformer",            &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
    {"moss",                &adapter_load_impl,      &adapter_init_context_impl,   &adapter_run_impl, nullptr, nullptr,
     &adapter_stream_begin_impl,   &adapter_stream_feed_impl,   &adapter_stream_finalize_impl, &adapter_stream_reset_impl,
     &adapter_accepts_ext_kind_impl, nullptr},
};
constexpr size_t k_n_adapter_archs = sizeof(adapter_archs) / sizeof(adapter_archs[0]);

}  // namespace

// Public entry point called by find_arch() (in transcribe-arch.cpp) after the
// builtin transcribe.cpp families fail to match. `adapter_archs` lives in the
// anonymous namespace above but remains visible here via the enclosing
// `transcribe` namespace; only this function has external linkage.
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
