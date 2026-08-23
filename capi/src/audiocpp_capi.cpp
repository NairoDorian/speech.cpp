/**
 * audiocpp_capi.cpp — C ABI implementation for audio.cpp
 *
 * Wraps engine::runtime C++ interfaces into extern "C" functions.
 * All C++ exceptions are caught and converted to audiocpp_error_t.
 */

#include "audiocpp.h"

#include "audiocpp_build_info.h"

#include "engine/framework/assets/embedded.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/deepfilternet2.h"
#include "engine/framework/audio/flashsr.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/rnnoise.h"
#include "engine/framework/audio/zipenhancer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/audio/wav_writer.h"

#include "engine/framework/io/json.h"

#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "cJSON.h"
// cJSON.h is under external/cJSON/ — add include path if needed

#include "audiocpp_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

/* ======================================================================== */
/* Internal handle                                                           */
/* ======================================================================== */

struct audiocpp_model {
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> loaded_model;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
    engine::runtime::IOfflineVoiceTaskSession *offline = nullptr;
    // Backend config the model was loaded with — reused when creating streaming
    // sessions so they run on the same device the caller selected at load time.
    engine::core::BackendConfig backend_config;
    // Progress callback (installed once, persists across runs until replaced/cleared)
    audiocpp_progress_fn progress_fn = nullptr;
    void *progress_user = nullptr;
};

struct audiocpp_stream {
    std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
    engine::runtime::IStreamingVoiceTaskSession *streaming = nullptr;
    int64_t next_start_sample = 0;
    // Sink-collected events for models that emit via callback (nemotron ASR)
    std::vector<engine::runtime::StreamEvent> sink_events;
};

/* ======================================================================== */
/* Option number serialization (shared with unit tests)                      */
/* ======================================================================== */

namespace audiocpp::detail {

std::string option_number_to_string(double value) {
    // Integers within double's exact-integer range (< 2^53) render as plain
    // integers so the stoll option path accepts them. Beyond 2^53 a double can't
    // distinguish consecutive integers, so emit via the float serializer —
    // stoll then rejects the value instead of silently rounding it. Mirrors
    // app/cli/request.cpp json_option_string.
    constexpr double kMaxExactJsonInteger = 9007199254740992.0;  // 2^53
    if (std::isfinite(value) && std::trunc(value) == value
        && std::fabs(value) < kMaxExactJsonInteger) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(0) << value;
        return out.str();
    }
    return engine::io::json::stringify_number(value);
}

void collect_option_fields(
    const cJSON * root,
    std::unordered_map<std::string, std::string> & options) {
    if (root == nullptr) {
        return;
    }
    for (const cJSON * item = root->child; item; item = item->next) {
        if (item->string == nullptr) {
            continue;
        }
        const std::string key = item->string;
        // Same value rendering as the request-options path (apply_options):
        // numbers via option_number_to_string, booleans as "true"/"false".
        if (cJSON_IsString(item)) {
            options[key] = item->valuestring;
        } else if (cJSON_IsNumber(item)) {
            options[key] = option_number_to_string(item->valuedouble);
        } else if (cJSON_IsBool(item)) {
            options[key] = cJSON_IsTrue(item) ? "true" : "false";
        }
    }
}

bool parse_session_options_json(
    const char * options_json,
    std::unordered_map<std::string, std::string> & options) {
    if (options_json == nullptr || options_json[0] == '\0') {
        return true;  // defaults
    }
    cJSON * root = cJSON_Parse(options_json);
    if (root == nullptr || !cJSON_IsObject(root)) {
        if (root != nullptr) {
            cJSON_Delete(root);
        }
        return false;
    }
    collect_option_fields(root, options);
    cJSON_Delete(root);
    return true;
}

}  // namespace audiocpp::detail

/* ======================================================================== */
/* Error helpers                                                             */
/* ======================================================================== */

// Duplicate a string into malloc'd C string (caller frees via audiocpp_free_string)
static char *dup_cstr(const std::string &s) {
    char *p = static_cast<char *>(malloc(s.size() + 1));
    if (p) {
        memcpy(p, s.c_str(), s.size() + 1);
    }
    return p;
}

// Free the owned members of one audiocpp_artifact_t, but NOT the struct itself
// (elements may live inline in an array allocated as new[]). Leaves every field
// in a clean, NULL/0 state so a partially filled element is always re-freeable.
static void free_artifact_members(audiocpp_artifact_t *a) {
    if (!a) return;
    free(a->id);
    free(a->payload);
    if (a->meta_keys) {
        for (int i = 0; i < a->n_meta; ++i) free(a->meta_keys[i]);
        free(a->meta_keys);
    }
    if (a->meta_values) {
        for (int i = 0; i < a->n_meta; ++i) free(a->meta_values[i]);
        free(a->meta_values);
    }
    a->id = nullptr;
    a->payload = nullptr;
    a->payload_size = 0;
    a->meta_keys = nullptr;
    a->meta_values = nullptr;
    a->n_meta = 0;
}

// Free an artifact collection wrapper plus every element's owned data plus the
// element array. The array is allocated with new[] in collect_artifacts(). NULL-safe.
static void release_artifacts(audiocpp_artifacts_t *c) {
    if (!c) return;
    if (c->artifacts) {
        for (int64_t i = 0; i < c->n_artifacts; ++i) {
            free_artifact_members(&c->artifacts[i]);
        }
        delete[] c->artifacts;
    }
    delete c;
}

static void set_error(audiocpp_error_t *err, int code, const char *msg) {
    if (!err) return;
    audiocpp_clear_error(err);
    err->code = code;
    err->message = msg ? dup_cstr(msg) : nullptr;
}

// Catches all C++ exceptions in a lambda; returns false on exception.
// ProgressCanceled is reported with code 0 (cancellation is not a hard error)
// so callers can distinguish "user cancelled" from real failures.
#define AUDIOCPP_CATCH(err, body)                                  \
    do {                                                            \
        try {                                                       \
            body;                                                   \
        } catch (const engine::runtime::ProgressCanceled &e) {     \
            set_error((err), 0, e.what());                         \
        } catch (const std::exception &e) {                        \
            set_error((err), -1, e.what());                        \
        } catch (...) {                                             \
            set_error((err), -2, "unknown C++ exception");         \
        }                                                           \
    } while (0)

/* ======================================================================== */
/* Enum mapping                                                              */
/* ======================================================================== */

static engine::core::BackendType map_backend(int backend) {
    switch (backend) {
        case AUDIOCPP_BACKEND_CUDA:   return engine::core::BackendType::Cuda;
        case AUDIOCPP_BACKEND_VULKAN: return engine::core::BackendType::Vulkan;
        case AUDIOCPP_BACKEND_METAL:  return engine::core::BackendType::Metal;
        case AUDIOCPP_BACKEND_BEST:   return engine::core::BackendType::BestAvailable;
        default:                      return engine::core::BackendType::Cpu;
    }
}

// Task mapping lives in audiocpp::detail (not file-static) so the enum-sync
// unit test can link it directly; the using-declaration keeps the call sites
// in this file unchanged. The shared library's hidden-visibility preset plus
// the .def/version-script export whitelist keep it out of the dynamic symbol
// table, so the C ABI surface is unchanged.
namespace audiocpp::detail {
engine::runtime::VoiceTaskKind map_task(int task) {
    switch (task) {
        case AUDIOCPP_TASK_TTS:   return engine::runtime::VoiceTaskKind::Tts;
        case AUDIOCPP_TASK_ASR:   return engine::runtime::VoiceTaskKind::Asr;
        case AUDIOCPP_TASK_VAD:   return engine::runtime::VoiceTaskKind::Vad;
        case AUDIOCPP_TASK_DIAR:  return engine::runtime::VoiceTaskKind::Diarization;
        case AUDIOCPP_TASK_SEP:   return engine::runtime::VoiceTaskKind::SourceSeparation;
        case AUDIOCPP_TASK_GEN:   return engine::runtime::VoiceTaskKind::AudioGeneration;
        case AUDIOCPP_TASK_ALIGN: return engine::runtime::VoiceTaskKind::Alignment;
        case AUDIOCPP_TASK_VC:    return engine::runtime::VoiceTaskKind::VoiceConversion;
        case AUDIOCPP_TASK_CLON:  return engine::runtime::VoiceTaskKind::VoiceCloning;
        case AUDIOCPP_TASK_S2S:   return engine::runtime::VoiceTaskKind::SpeechToSpeech;
        case AUDIOCPP_TASK_VDES:  return engine::runtime::VoiceTaskKind::VoiceDesign;
        case AUDIOCPP_TASK_SPK:   return engine::runtime::VoiceTaskKind::SpeakerRecognition;
        case AUDIOCPP_TASK_SVC:   return engine::runtime::VoiceTaskKind::Svc;
        case AUDIOCPP_TASK_MIDI:  return engine::runtime::VoiceTaskKind::Midi;
        default:                  return engine::runtime::VoiceTaskKind::Tts;
    }
}
}  // namespace audiocpp::detail
using audiocpp::detail::map_task;

/* ======================================================================== */
/* Lifecycle                                                                 */
/* ======================================================================== */

audiocpp_model_t *audiocpp_load_model_ex(
    const char *model_path,
    const char *family_hint,
    int task,
    int backend,
    int device_id,
    int n_threads,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_model_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        // Session options are fixed at session creation and cannot change per
        // request. A malformed options JSON must fail loudly here — silently
        // falling back to defaults would surface later as confusing
        // "requires X" errors (e.g. miotts missing its codec model path).
        engine::runtime::SessionOptions opts;
        if (!audiocpp::detail::parse_session_options_json(options_json, opts.options)) {
            throw std::runtime_error(
                "failed to parse session options JSON (expected a JSON object, "
                "e.g. {\"miotts.codec_model_path\": \"...\"})");
        }
        const std::string hint = (family_hint && family_hint[0] != '\0') ? std::string(family_hint) : std::string{};
        const bool empty_path = (model_path == nullptr || model_path[0] == '\0');
        // Empty model_path is allowed ONLY for VAD families when the binary
        // was built with AUDIOCPP_EMBED_VAD_ASSETS (silero_vad/marblenet_vad
        // load from baked-in weights). Otherwise require a real path.
        const bool embedded_ok = (hint == "silero_vad" || hint == "marblenet_vad")
            && engine::assets::embedded::has_embedded_asset(hint);
        if (empty_path && !embedded_ok) {
            throw std::runtime_error(
                "model_path is null or empty (only allowed for silero_vad/"
                "marblenet_vad when built with AUDIOCPP_EMBED_VAD_ASSETS=ON)");
        }
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest req;
        req.model_path = empty_path ? std::filesystem::path{} : std::filesystem::path(model_path);
        if (!hint.empty()) {
            req.family_hint = hint;
        }
        auto loaded = registry.load(req);
        if (!loaded) {
            throw std::runtime_error("failed to load model: " + std::string(model_path));
        }
        engine::runtime::TaskSpec task_spec;
        task_spec.task = map_task(task);
        task_spec.mode = engine::runtime::RunMode::Offline;
        opts.backend.type = map_backend(backend);
        opts.backend.device = device_id;
        opts.backend.threads = n_threads > 0 ? n_threads : 0;
        auto session = loaded->create_task_session(task_spec, opts);
        if (!session) {
            throw std::runtime_error("failed to create task session");
        }
        result = new audiocpp_model{};
        result->loaded_model = std::move(loaded);
        result->session = std::move(session);
        result->backend_config = opts.backend;
        result->offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
            result->session.get());
        if (!result->offline) {
            throw std::runtime_error("model does not support offline task execution");
        }
    });
    return result;
}

audiocpp_model_t *audiocpp_load_model(
    const char *model_path,
    const char *family_hint,
    int task,
    int backend,
    int device_id,
    int n_threads,
    audiocpp_error_t *err
) {
    return audiocpp_load_model_ex(
        model_path, family_hint, task, backend, device_id, n_threads, nullptr, err);
}

void audiocpp_free_model(audiocpp_model_t *model) {
    delete model;
}

/* ======================================================================== */
/* Progress callback                                                         */
/* ======================================================================== */

// Install (or clear) the progress callback. The C function pointer + user data
// are stored on the model handle, and a C++ adapter std::function is registered
// on the session so it persists across runs until replaced.
void audiocpp_set_progress_callback(audiocpp_model_t *model,
                                    audiocpp_progress_fn fn,
                                    void *user_data) {
    if (!model) return;
    model->progress_fn = fn;
    model->progress_user = user_data;
    if (!model->session) return;
    if (fn == nullptr) {
        model->session->set_progress_callback(nullptr);
        return;
    }
    // Adapter: bridge C++ ProgressInfo -> C callback. Return false from C side
    // (non-zero) to request cancellation; the std::function returning false
    // makes emit_progress() throw ProgressCanceled inside run().
    audiocpp_model *m = static_cast<audiocpp_model *>(model);
    model->session->set_progress_callback(
        [m](const engine::runtime::ProgressInfo & info) -> bool {
            if (!m->progress_fn) return true;  // cleared mid-flight: continue
            const char *stage = info.stage.empty() ? "" : info.stage.c_str();
            int cont = m->progress_fn(info.progress, stage,
                                      info.completed_units, info.total_units,
                                      m->progress_user);
            return cont == 0;  // 0 == continue
        });
}

namespace audiocpp::detail {

// Pack an engine audio buffer into an owned audiocpp_audio_t, preserving the
// channel count. The engine may emit interleaved multi-channel audio (e.g.
// dramabox outputs stereo); dropping buf.channels here used to make callers
// write stereo data into mono WAVs (garbled playback) — the original bug.
// n_samples is the TOTAL sample count (channels * frames), matching the
// engine's AudioBuffer. Throws on allocation failure.
audiocpp_audio_t *pack_audio_output(const engine::runtime::AudioBuffer & buf) {
    auto * out = new audiocpp_audio_t{};
    out->n_samples = static_cast<int64_t>(buf.samples.size());
    out->sample_rate = buf.sample_rate;
    out->channels = std::max(1, buf.channels);
    if (buf.samples.empty()) {
        return out;
    }
    out->samples = static_cast<float *>(malloc(buf.samples.size() * sizeof(float)));
    if (!out->samples) {
        delete out;
        throw std::runtime_error("out of memory allocating audio output");
    }
    std::memcpy(out->samples, buf.samples.data(), buf.samples.size() * sizeof(float));
    return out;
}

// Map an engine ArtifactKind to the C ABI AUDIOCPP_ARTIFACT_* value. Kept in
// 1:1 sync with include/engine/framework/runtime/session.h ArtifactKind; the
// default falls back to CUSTOM so a future runtime enum value degrades safely.
int map_artifact_kind(engine::runtime::ArtifactKind kind) {
    switch (kind) {
        using K = engine::runtime::ArtifactKind;
        case K::SpeakerEmbedding:    return AUDIOCPP_ARTIFACT_SPEAKER_EMBEDDING;
        case K::StyleEmbedding:      return AUDIOCPP_ARTIFACT_STYLE_EMBEDDING;
        case K::PromptEmbedding:     return AUDIOCPP_ARTIFACT_PROMPT_EMBEDDING;
        case K::AcousticTokens:      return AUDIOCPP_ARTIFACT_ACOUSTIC_TOKENS;
        case K::Midi:                return AUDIOCPP_ARTIFACT_MIDI;
        case K::TranscriptAlignment: return AUDIOCPP_ARTIFACT_ALIGNMENT;
        case K::DiarizationState:    return AUDIOCPP_ARTIFACT_DIARIZATION_STATE;
        case K::VadState:            return AUDIOCPP_ARTIFACT_VAD_STATE;
        case K::Custom:              return AUDIOCPP_ARTIFACT_CUSTOM;
    }
    return AUDIOCPP_ARTIFACT_CUSTOM;
}

// Fill one audiocpp_artifact_t (already value-initialized, e.g. an array
// element) from a runtime VoiceArtifact. Every owned member is malloc'd so the
// element can later be freed by free_artifact_members(). On allocation failure
// the element is left in a state free_artifact_members() can still handle, then
// throws so AUDIOCPP_CATCH surfaces it. dup_cstr is NULL-on-OOM (no throw), so a
// failed string copy degrades to a NULL pointer rather than crashing the free path.
void pack_artifact_into(const engine::runtime::VoiceArtifact &src, audiocpp_artifact_t &dst) {
    dst.kind = map_artifact_kind(src.kind);
    dst.id = dup_cstr(src.id);
    dst.payload = nullptr;
    dst.payload_size = 0;
    if (!src.payload.empty()) {
        dst.payload = static_cast<uint8_t *>(malloc(src.payload.size()));
        if (!dst.payload) {
            throw std::runtime_error("out of memory allocating artifact payload");
        }
        std::memcpy(dst.payload, src.payload.data(), src.payload.size());
        dst.payload_size = static_cast<int64_t>(src.payload.size());
    }
    dst.n_meta = static_cast<int>(src.meta.size());
    dst.meta_keys = nullptr;
    dst.meta_values = nullptr;
    if (dst.n_meta > 0) {
        dst.meta_keys = static_cast<char **>(calloc(static_cast<size_t>(dst.n_meta), sizeof(char *)));
        dst.meta_values = static_cast<char **>(calloc(static_cast<size_t>(dst.n_meta), sizeof(char *)));
        if (!dst.meta_keys || !dst.meta_values) {
            throw std::runtime_error("out of memory allocating artifact metadata arrays");
        }
        int idx = 0;
        for (const auto &kv : src.meta) {
            dst.meta_keys[idx] = dup_cstr(kv.first);
            dst.meta_values[idx] = dup_cstr(kv.second);
            ++idx;
        }
    }
}

// Collect every artifact a run produced into one owned collection, mirroring
// app/server/runtime.cpp's JSON serialization order: the single optional
// artifact_output first, then the output_artifacts vector. Returns a non-NULL
// collection (n_artifacts may be 0). On a mid-build allocation failure, whatever
// was already packed is released before rethrowing.
audiocpp_artifacts_t *collect_artifacts(const engine::runtime::TaskResult &result) {
    auto *out = new audiocpp_artifacts_t{};
    try {
        const bool has_single = result.artifact_output.has_value();
        const int64_t total = (has_single ? int64_t{1} : int64_t{0})
            + static_cast<int64_t>(result.output_artifacts.size());
        out->n_artifacts = total;
        if (total == 0) return out;
        out->artifacts = new audiocpp_artifact_t[static_cast<size_t>(total)]();
        int64_t idx = 0;
        if (has_single) {
            pack_artifact_into(*result.artifact_output, out->artifacts[idx++]);
        }
        for (const auto &a : result.output_artifacts) {
            pack_artifact_into(a, out->artifacts[idx++]);
        }
    } catch (...) {
        release_artifacts(out);
        throw;
    }
    return out;
}

}  // namespace audiocpp::detail

/* ======================================================================== */
/* TTS                                                                       */
/* ======================================================================== */

// Helper: parse JSON options string into TaskRequest fields.
// Recognized keys:
//   voice_ref: path to reference WAV (loads audio + sets VoiceReference)
//   reference_text: transcript of reference audio
//   speed, language, emotion, speaking_rate, pitch_shift, energy_scale: scalar options
//   Any other key → passed through as option string
static void apply_options(engine::runtime::TaskRequest & req, const char * options_json) {
    if (!options_json || !options_json[0]) return;

    // Parse JSON (use cJSON which is already vendored)
    auto * root = cJSON_Parse(options_json);
    if (!root) return;

    // Voice cloning
    const char * voice_ref = cJSON_GetStringValue(cJSON_GetObjectItem(root, "voice_ref"));
    if (voice_ref && voice_ref[0]) {
        engine::runtime::VoiceCondition voice;
        voice.speaker = engine::runtime::VoiceReference{};
        auto wav = engine::audio::read_wav_f32(std::filesystem::path(voice_ref));
        engine::runtime::AudioBuffer ref_audio;
        ref_audio.sample_rate = wav.sample_rate;
        ref_audio.channels = wav.channels;
        ref_audio.samples = std::move(wav.samples);
        voice.speaker->audio = std::move(ref_audio);
        req.voice = std::move(voice);
    }

    // Transcript text(for forced aligner: audio + transcript contract)
    const char * text = cJSON_GetStringValue(cJSON_GetObjectItem(root, "text"));
    const char * lang = cJSON_GetStringValue(cJSON_GetObjectItem(root, "language"));
    if (text && text[0] && !req.text_input.has_value()) {
        req.text_input = engine::runtime::Transcript{};
        req.text_input->text = text;
        if (lang && lang[0]) {
            req.text_input->language = lang;
        }
    } else if (lang && lang[0] && req.text_input.has_value() && req.text_input->language.empty()) {
        req.text_input->language = lang;
    }

    // Iterate all string/number keys → options map
    for (auto * item = root->child; item; item = item->next) {
        if (!item->string) continue;
        std::string key = item->string;
        // Skip voice_ref (handled above)
        if (key == "voice_ref") continue;
        // String values
        if (cJSON_IsString(item)) {
            req.options[key] = item->valuestring;
        }
        // Number values: cJSON stores every JSON number in valuedouble.
        // Render via the shared helper so an integer like 2 becomes "2" (not
        // "2.000000", which the stoll option path rejects), and values beyond
        // double's exact-integer range aren't silently rounded. See
        // option_number_to_string / audiocpp_internal.h.
        else if (cJSON_IsNumber(item)) {
            req.options[key] = audiocpp::detail::option_number_to_string(item->valuedouble);
        }
        // Bool values → "true"/"false"
        else if (cJSON_IsBool(item)) {
            req.options[key] = cJSON_IsTrue(item) ? "true" : "false";
        }
    }

    cJSON_Delete(root);
}

audiocpp_audio_t *audiocpp_tts(
    const audiocpp_model_t *model,
    const char *text,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!text) {
            throw std::runtime_error("text is null");
        }
        engine::runtime::TaskRequest req;
        req.text_input = engine::runtime::Transcript{};
        req.text_input->text = text;
        apply_options(req, options_json);
        // Some models (e.g. Qwen3-ASR/TTS) require prepare() before run()
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        if (!task_result.audio_output || task_result.audio_output->samples.empty()) {
            throw std::runtime_error("TTS produced no audio output");
        }
        result = audiocpp::detail::pack_audio_output(*task_result.audio_output);
    });
    return result;
}

audiocpp_audio_t *audiocpp_tts_with_voice_ref(
    const audiocpp_model_t *model,
    const char *text,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!text) {
            throw std::runtime_error("text is null");
        }
        engine::runtime::TaskRequest req;
        req.text_input = engine::runtime::Transcript{};
        req.text_input->text = text;
        apply_options(req, options_json);

        // Inline voice reference PCM overrides file-based voice_ref
        if (voice_ref_pcm && voice_ref_n > 0) {
            engine::runtime::VoiceCondition voice;
            voice.speaker = engine::runtime::VoiceReference{};
            engine::runtime::AudioBuffer ref_audio;
            ref_audio.sample_rate = voice_ref_sr;
            ref_audio.channels = 1;
            ref_audio.samples.assign(voice_ref_pcm, voice_ref_pcm + voice_ref_n);
            voice.speaker->audio = std::move(ref_audio);
            req.voice = std::move(voice);
        }

        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        if (!task_result.audio_output || task_result.audio_output->samples.empty()) {
            throw std::runtime_error("TTS produced no audio output");
        }
        result = audiocpp::detail::pack_audio_output(*task_result.audio_output);
    });
    return result;
}

/* ======================================================================== */
/* Batch TTS                                                                 */
/* ======================================================================== */

audiocpp_audio_batch_t *audiocpp_tts_batch(
    audiocpp_model_t *model,
    const char *const *texts,
    int n_texts,
    const char *options_json,
    int merge_mode,
    audiocpp_error_t *err
) {
    audiocpp_audio_batch_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!texts || n_texts <= 0) {
            throw std::runtime_error("n_texts must be > 0");
        }

        // Prepare once for the batch using the first text (the prepare contract
        // is about graph capacity, which is sized per the worst chunk inside
        // any text; using the first text is sufficient for capacity signaling
        // — the engine re-prepares internally if a later chunk exceeds it).
        engine::runtime::TaskRequest first_req;
        first_req.text_input = engine::runtime::Transcript{};
        first_req.text_input->text = texts[0] ? texts[0] : "";
        apply_options(first_req, options_json);
        model->session->prepare(engine::runtime::build_preparation_request(first_req));

        // Suppress per-chunk progress during the batch so the installed
        // callback only sees request-level progress. Save & restore the
        // caller's callback around the whole batch.
        audiocpp_progress_fn saved_fn = model->progress_fn;
        void *saved_user = model->progress_user;
        model->session->set_progress_callback(nullptr);

        // Helper to emit request-level progress via the original C callback.
        auto emit_request_progress = [&](int completed) {
            if (!saved_fn) return;
            float p = (n_texts > 0) ? static_cast<float>(completed) / static_cast<float>(n_texts) : 1.0f;
            if (p < 0.0f) p = 0.0f;
            if (p > 1.0f) p = 1.0f;
            // Returning non-zero requests cancellation; we honor it by
            // throwing ProgressCanceled, which unwinds the batch cleanly.
            if (saved_fn(p, "batch_tts", completed, n_texts, saved_user) != 0) {
                throw engine::runtime::ProgressCanceled("batch_tts canceled by progress callback");
            }
        };

        // Synthesize each text independently. Individual failures do not abort
        // the whole batch — they are recorded per-item (empty audio).
        std::vector<engine::runtime::TaskResult> item_results(
            static_cast<size_t>(n_texts));
        int produced = 0;
        for (int i = 0; i < n_texts; ++i) {
            emit_request_progress(i);  // before each request
            engine::runtime::TaskRequest req;
            req.text_input = engine::runtime::Transcript{};
            req.text_input->text = texts[i] ? texts[i] : "";
            apply_options(req, options_json);
            try {
                item_results[static_cast<size_t>(i)] = model->offline->run(req);
                if (item_results[static_cast<size_t>(i)].audio_output &&
                    !item_results[static_cast<size_t>(i)].audio_output->samples.empty()) {
                    ++produced;
                }
            } catch (...) {
                // Leave this item's result empty; continue with the rest.
            }
        }
        emit_request_progress(n_texts);  // final 100%

        // Restore the caller's callback (even though we suppressed it, the
        // model handle's stored fn/user are unchanged — but the SESSION's
        // std::function was nulled, so reinstall the adapter if one was set).
        if (saved_fn != nullptr) {
            audiocpp_model *m = static_cast<audiocpp_model *>(model);
            m->session->set_progress_callback(
                [m](const engine::runtime::ProgressInfo & info) -> bool {
                    if (!m->progress_fn) return true;
                    const char *stage = info.stage.empty() ? "" : info.stage.c_str();
                    return m->progress_fn(info.progress, stage,
                                          info.completed_units, info.total_units,
                                          m->progress_user) == 0;
                });
        }

        if (produced == 0) {
            throw std::runtime_error("batch TTS produced no audio output for any text");
        }

        result = new audiocpp_audio_batch_t{};
        result->n_items = n_texts;
        result->items = static_cast<audiocpp_audio_t *>(
            calloc(static_cast<size_t>(n_texts), sizeof(audiocpp_audio_t)));
        if (!result->items) {
            delete result;
            result = nullptr;
            throw std::runtime_error("out of memory allocating batch items");
        }

        if (merge_mode == AUDIOCPP_BATCH_MERGE_CONCAT) {
            // Concatenate all non-empty outputs into a single buffer, recording
            // per-text [start,end) sample ranges.
            result->chapter_starts = static_cast<int64_t *>(
                calloc(static_cast<size_t>(n_texts), sizeof(int64_t)));
            result->chapter_ends = static_cast<int64_t *>(
                calloc(static_cast<size_t>(n_texts), sizeof(int64_t)));
            if (!result->chapter_starts || !result->chapter_ends) {
                free(result->chapter_starts);
                free(result->chapter_ends);
                free(result->items);
                delete result;
                result = nullptr;
                throw std::runtime_error("out of memory allocating chapters");
            }
            std::vector<float> merged;
            int sample_rate = 0;
            int channels = 0;
            for (int i = 0; i < n_texts; ++i) {
                const auto &tr = item_results[static_cast<size_t>(i)];
                result->chapter_starts[i] = static_cast<int64_t>(
                    merged.size() / static_cast<size_t>(std::max(1, channels)));
                if (!tr.audio_output || tr.audio_output->samples.empty()) {
                    result->chapter_ends[i] = result->chapter_starts[i];
                    continue;
                }
                const auto &buf = *tr.audio_output;
                if (sample_rate == 0) {
                    sample_rate = buf.sample_rate;
                    channels = std::max(1, buf.channels);
                }
                if (buf.sample_rate != sample_rate || buf.channels != channels) {
                    free(result->chapter_starts);
                    free(result->chapter_ends);
                    free(result->items);
                    delete result;
                    result = nullptr;
                    throw std::runtime_error(
                        "cannot concat batch audio with differing sample rates or channels");
                }
                merged.insert(merged.end(), buf.samples.begin(), buf.samples.end());
                result->chapter_ends[i] = static_cast<int64_t>(
                    merged.size() / static_cast<size_t>(channels));
            }
            // All texts' audio lives in items[0]; the rest stay empty.
            result->items[0].n_samples = static_cast<int64_t>(merged.size());
            result->items[0].sample_rate = sample_rate;
            result->items[0].channels = channels;
            result->items[0].samples = static_cast<float *>(
                malloc(merged.size() * sizeof(float)));
            if (!result->items[0].samples) {
                free(result->chapter_starts);
                free(result->chapter_ends);
                free(result->items);
                delete result;
                result = nullptr;
                throw std::runtime_error("out of memory allocating merged audio");
            }
            std::memcpy(result->items[0].samples, merged.data(),
                        merged.size() * sizeof(float));
        } else {
            // Independent: each item gets its own audio buffer (or stays empty).
            for (int i = 0; i < n_texts; ++i) {
                const auto &tr = item_results[static_cast<size_t>(i)];
                if (!tr.audio_output || tr.audio_output->samples.empty()) {
                    continue;  // leave samples=NULL, n_samples=0
                }
                const auto &buf = *tr.audio_output;
                result->items[i].n_samples = static_cast<int64_t>(buf.samples.size());
                result->items[i].sample_rate = buf.sample_rate;
                result->items[i].channels = std::max(1, buf.channels);
                result->items[i].samples = static_cast<float *>(
                    malloc(buf.samples.size() * sizeof(float)));
                if (!result->items[i].samples) {
                    // Free everything allocated so far before throwing.
                    for (int j = 0; j <= i; ++j) {
                        free(result->items[j].samples);
                    }
                    free(result->items);
                    delete result;
                    result = nullptr;
                    throw std::runtime_error("out of memory allocating audio output");
                }
                std::memcpy(result->items[i].samples, buf.samples.data(),
                            buf.samples.size() * sizeof(float));
            }
        }
    });
    return result;
}

void audiocpp_free_audio_batch(audiocpp_audio_batch_t *batch) {
    if (!batch) return;
    if (batch->items) {
        for (int i = 0; i < batch->n_items; ++i) {
            free(batch->items[i].samples);
        }
        free(batch->items);
    }
    free(batch->chapter_starts);
    free(batch->chapter_ends);
    delete batch;
}

/* ======================================================================== */
/* ASR                                                                       */
/* ======================================================================== */

audiocpp_text_t *audiocpp_asr(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_text_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        // Some models (e.g. Qwen3-ASR) require prepare() before run()
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        result = new audiocpp_text_t{};
        result->text = dup_cstr(
            task_result.text_output ? task_result.text_output->text : "");
        result->language = task_result.text_output && !task_result.text_output->language.empty()
            ? dup_cstr(task_result.text_output->language)
            : nullptr;
    });
    return result;
}

/* ======================================================================== */
/* Audio transform: audio → audio (Source Separation, Voice Conversion)      */
/* ======================================================================== */

audiocpp_audio_t *audiocpp_audio_transform(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        // VC/audio→audio:用 audio_output;SEP(HTDemucs):用 named_audio_outputs[0]
        // (多 stem 输出,返回第一个 — 通常是 "vocals")
        const engine::runtime::AudioBuffer *buf_ptr = nullptr;
        if (task_result.audio_output && !task_result.audio_output->samples.empty()) {
            buf_ptr = &(*task_result.audio_output);
        } else if (!task_result.named_audio_outputs.empty() &&
                   !task_result.named_audio_outputs[0].audio.samples.empty()) {
            buf_ptr = &task_result.named_audio_outputs[0].audio;
        }
        if (!buf_ptr) {
            throw std::runtime_error("audio transform produced no audio output");
        }
        result = audiocpp::detail::pack_audio_output(*buf_ptr);
    });
    return result;
}

audiocpp_audio_t *audiocpp_audio_transform_with_voice_ref(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);

        // Inline voice reference PCM (target speaker for VC)
        if (voice_ref_pcm && voice_ref_n > 0) {
            engine::runtime::VoiceCondition voice;
            voice.speaker = engine::runtime::VoiceReference{};
            engine::runtime::AudioBuffer ref_audio;
            ref_audio.sample_rate = voice_ref_sr;
            ref_audio.channels = 1;
            ref_audio.samples.assign(voice_ref_pcm, voice_ref_pcm + voice_ref_n);
            voice.speaker->audio = std::move(ref_audio);
            req.voice = std::move(voice);
        }

        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        const engine::runtime::AudioBuffer *buf_ptr = nullptr;
        if (task_result.audio_output && !task_result.audio_output->samples.empty()) {
            buf_ptr = &(*task_result.audio_output);
        } else if (!task_result.named_audio_outputs.empty() &&
                   !task_result.named_audio_outputs[0].audio.samples.empty()) {
            buf_ptr = &task_result.named_audio_outputs[0].audio;
        }
        if (!buf_ptr) {
            throw std::runtime_error("audio transform produced no audio output");
        }
        result = audiocpp::detail::pack_audio_output(*buf_ptr);
    });
    return result;
}

audiocpp_diar_t *audiocpp_diar(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_diar_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        const auto & turns = task_result.speaker_turns;
        result = new audiocpp_diar_t{};
        result->n_turns = static_cast<int64_t>(turns.size());
        if (turns.empty()) {
            result->turns = nullptr;
        } else {
            result->turns = static_cast<audiocpp_speaker_turn_t *>(
                calloc(turns.size(), sizeof(audiocpp_speaker_turn_t)));
            for (size_t i = 0; i < turns.size(); ++i) {
                result->turns[i].start_sample = turns[i].span.start_sample;
                result->turns[i].end_sample = turns[i].span.end_sample;
                result->turns[i].speaker_id = dup_cstr(turns[i].speaker_id);
                result->turns[i].confidence = turns[i].confidence;
            }
        }
    });
    return result;
}

/* ======================================================================== */
/* VAD                                                                       */
/* ======================================================================== */

audiocpp_vad_t *audiocpp_vad(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_vad_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        const auto & segs = task_result.speech_segments;
        result = new audiocpp_vad_t{};
        result->n_segments = static_cast<int64_t>(segs.size());
        if (segs.empty()) {
            result->segments = nullptr;
        } else {
            result->segments = static_cast<audiocpp_vad_segment_t *>(
                calloc(segs.size(), sizeof(audiocpp_vad_segment_t)));
            for (size_t i = 0; i < segs.size(); ++i) {
                result->segments[i].start_sample = segs[i].span.start_sample;
                result->segments[i].end_sample = segs[i].span.end_sample;
                result->segments[i].confidence = segs[i].confidence;
            }
        }
    });
    return result;
}

/* ======================================================================== */
/* Utilities                                                                 */
/* ======================================================================== */

// A self-contained build-ID string embedded verbatim into the binary so the
// full version (tag + commit + branch + date + backend) is recoverable from
// the file alone via `strings audiocpp.dll | grep audiocpp-build-id`, without
// loading or calling it. Kept distinct from audiocpp_version()'s runtime-built
// string: the macro literals there are merged/split by the optimizer, but this
// single constant survives as one contiguous ASCII run.
//
// Retention (Linux .so): keeping a metadata-only string through gcc/clang LTO
// + --gc-sections is hard. An earlier version used a `volatile char` anchor in
// an anonymous namespace; that internally-linked, only-self-referenced object
// was removed by the ELF linker even without --gc-sections, so the strings
// outlet was empty on .so (MSVC has no such DCE, so .dll worked). The triple
// defense here is: (1) external linkage (no anonymous namespace), (2) the GCC
// `used` attribute forbidding the compiler from dropping the symbol, (3) an
// exported referencing function (audiocpp_build_id) so the linker keeps it.
#if defined(__GNUC__)
__attribute__((used))
#endif
const char kAudiocppBuildId[] =
    "\naudiocpp-build-id: " AUDIOCPP_BUILD_VERSION " "
    AUDIOCPP_BUILD_COMMIT " " AUDIOCPP_BUILD_BRANCH " " AUDIOCPP_BUILD_DATE "\n";

// Exported accessor for the build-ID string. Roots the literal against dead-
// stripping on Linux (gcc/clang + --gc-sections / LTO): the earlier
// `volatile char` anchor in an anonymous namespace was removed by the linker
// even without --gc-sections, so .so lost the string (.dll was unaffected).
// The string is also grep-able via `strings <lib> | grep audiocpp-build-id`.
const char *audiocpp_build_id(void) {
    return kAudiocppBuildId;
}

const char *audiocpp_version(void) {
    // Report the build provenance captured at CMake configure time
    // (generated/audiocpp_build_info.h). Format:
    //   <version> <commit_short> <branch> <build_time> <backend>
    // e.g. "release-0.4.2-178-g3239acd 3239acd main 2026-08-03T10:50:06Z cuda"
    // The backend tag is a compile-time label only — use
    // audiocpp_backend_available() to probe what this build can actually run.
    // Full provenance (long commit) stays recoverable from the file alone via
    // `strings audiocpp.dll | grep audiocpp-build-id`.
#if defined(GGML_USE_CUDA)
    static constexpr const char *kBackend = "cuda";
#elif defined(GGML_USE_HIP)
    static constexpr const char *kBackend = "rocm";
#elif defined(GGML_USE_SYCL)
    static constexpr const char *kBackend = "sycl";
#elif defined(GGML_USE_VULKAN)
    static constexpr const char *kBackend = "vulkan";
#elif defined(GGML_USE_METAL)
    static constexpr const char *kBackend = "metal";
#else
    static constexpr const char *kBackend = "cpu";
#endif
    // Short commit: first 7 chars (also 7 for the "unknown" fallback).
    static const std::string kCommitShort =
        std::string(AUDIOCPP_BUILD_COMMIT).substr(0, 7);
    static const std::string version =
        std::string(AUDIOCPP_BUILD_VERSION) + " " +
        kCommitShort + " " +
        AUDIOCPP_BUILD_BRANCH + " " +
        AUDIOCPP_BUILD_DATE + " " +
        kBackend;
    return version.c_str();
}

void audiocpp_free_audio(audiocpp_audio_t *audio) {
    if (!audio) return;
    free(audio->samples);
    delete audio;
}

void audiocpp_free_text(audiocpp_text_t *text) {
    if (!text) return;
    free(text->text);
    free(text->language);
    delete text;
}

void audiocpp_free_diar(audiocpp_diar_t *diar) {
    if (!diar) return;
    if (diar->turns) {
        for (int64_t i = 0; i < diar->n_turns; ++i) {
            free(diar->turns[i].speaker_id);
        }
        free(diar->turns);
    }
    delete diar;
}

void audiocpp_free_vad(audiocpp_vad_t *vad) {
    if (!vad) return;
    free(vad->segments);
    delete vad;
}

audiocpp_vad_t *audiocpp_vad_energy(
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_vad_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!pcm) {
            throw std::runtime_error("pcm is null");
        }
        if (sample_rate <= 0) {
            throw std::runtime_error("sample_rate must be > 0");
        }

        // Defaults mirror plan_quiet_energy_audio_chunks' typical use (ASR
        // chunking at ~30s with a 2s boundary search and 0.1s energy window).
        double chunk_seconds = 30.0;
        double boundary_seconds = 2.0;
        double min_energy_seconds = 0.1;
        if (options_json && options_json[0] != '\0') {
            cJSON * root = cJSON_Parse(options_json);
            if (root != nullptr) {
                const cJSON * v = cJSON_GetObjectItem(root, "chunk_seconds");
                if (cJSON_IsNumber(v)) chunk_seconds = v->valuedouble;
                v = cJSON_GetObjectItem(root, "boundary_seconds");
                if (cJSON_IsNumber(v)) boundary_seconds = v->valuedouble;
                v = cJSON_GetObjectItem(root, "min_energy_seconds");
                if (cJSON_IsNumber(v)) min_energy_seconds = v->valuedouble;
                cJSON_Delete(root);
            }
        }
        if (chunk_seconds <= 0.0) {
            throw std::runtime_error("chunk_seconds must be > 0");
        }
        if (boundary_seconds < 0.0) {
            throw std::runtime_error("boundary_seconds must be >= 0");
        }
        if (min_energy_seconds <= 0.0) {
            throw std::runtime_error("min_energy_seconds must be > 0");
        }

        engine::audio::QuietEnergyAudioChunkOptions opts;
        opts.chunk_samples = static_cast<int64_t>(chunk_seconds * sample_rate);
        opts.boundary_context_samples = static_cast<int64_t>(boundary_seconds * sample_rate);
        opts.min_energy_window_samples = static_cast<int64_t>(min_energy_seconds * sample_rate);
        if (opts.chunk_samples <= 0) opts.chunk_samples = sample_rate;  // >= 1s fallback

        std::vector<float> mono(pcm, pcm + static_cast<size_t>(n_samples));
        const auto spans = engine::audio::plan_quiet_energy_audio_chunks(mono, opts);

        result = new audiocpp_vad_t{};
        result->n_segments = static_cast<int64_t>(spans.size());
        if (!spans.empty()) {
            result->segments = static_cast<audiocpp_vad_segment_t *>(
                calloc(spans.size(), sizeof(audiocpp_vad_segment_t)));
            if (!result->segments) {
                delete result;
                result = nullptr;
                throw std::runtime_error("out of memory allocating VAD segments");
            }
            for (size_t i = 0; i < spans.size(); ++i) {
                result->segments[i].start_sample = spans[i].start_sample;
                result->segments[i].end_sample = spans[i].end_sample;
                result->segments[i].confidence = 1.0f;  // heuristic, no probability
            }
        }
    });
    return result;
}

/* ======================================================================== */
/* Audio enhancement (denoise / super-resolve)                               */
/* ======================================================================== */
namespace {

// Process-level cache for loaded audio-utility models. Without this, every
// audiocpp_denoise / audiocpp_super_resolve call reloads + re-uploads the full
// model (8–11 MB weights + graph build), which dominates latency for repeated
// calls. Keyed by (resolved path, backend type, device) so different configs
// get distinct entries. Entries are shared_ptr so the model stays alive as long
// as any caller holds it; the cache itself holds a weak_ptr so unloaded models
// are reclaimed when the last reference drops.
template <typename ModelType>
std::shared_ptr<const ModelType> get_or_load_utility_model(
    const std::string & cache_key,
    std::function<std::shared_ptr<const ModelType>()> loader) {
    static std::mutex mtx;
    static std::unordered_map<std::string, std::weak_ptr<const ModelType>> cache;
    std::lock_guard<std::mutex> lock(mtx);
    if (auto it = cache.find(cache_key); it != cache.end()) {
        if (auto existing = it->second.lock()) {
            return existing;
        }
    }
    auto loaded = loader();
    cache[cache_key] = loaded;
    return loaded;
}

// Build a cache key from the resolved model path + backend config.
std::string utility_model_cache_key(
    const std::filesystem::path & model_path,
    const engine::core::BackendConfig & backend) {
    return model_path.string() + "|" +
           std::to_string(static_cast<int>(backend.type)) + ":" +
           std::to_string(backend.device);
}

// Pack a mono PCM vector + sample rate into an owned audiocpp_audio_t (or
// nullptr on OOM). Throws on allocation failure so AUDIOCPP_CATCH reports it.
// (The AudioBuffer overload used by the exported TTS/transform entry points
// lives at file scope above the TTS section.)
audiocpp_audio_t *pack_audio_output(const std::vector<float> & samples, int sample_rate) {
    auto * out = new audiocpp_audio_t{};
    out->n_samples = static_cast<int64_t>(samples.size());
    out->sample_rate = sample_rate;
    out->channels = 1;  // enhanced/utility outputs are mono
    if (samples.empty()) {
        return out;
    }
    out->samples = static_cast<float *>(malloc(samples.size() * sizeof(float)));
    if (!out->samples) {
        delete out;
        throw std::runtime_error("out of memory allocating enhanced audio output");
    }
    std::memcpy(out->samples, samples.data(), samples.size() * sizeof(float));
    return out;
}

// Parse backend/device from options_json (defaults CPU/device 0).
engine::core::BackendConfig parse_utility_backend(const char * options_json) {
    engine::core::BackendConfig cfg;
    cfg.type = engine::core::BackendType::Cpu;
    cfg.device = 0;
    cfg.threads = 1;
    if (options_json == nullptr || options_json[0] == '\0') {
        return cfg;
    }
    cJSON * root = cJSON_Parse(options_json);
    if (root == nullptr) {
        return cfg;
    }
    const cJSON * be = cJSON_GetObjectItem(root, "backend");
    if (cJSON_IsString(be)) {
        const std::string s = be->valuestring;
        if (s == "cuda" || s == "CUDA") cfg.type = engine::core::BackendType::Cuda;
        else if (s == "vulkan" || s == "Vulkan") cfg.type = engine::core::BackendType::Vulkan;
        else if (s == "metal" || s == "Metal") cfg.type = engine::core::BackendType::Metal;
    }
    const cJSON * dev = cJSON_GetObjectItem(root, "device");
    if (cJSON_IsNumber(dev)) cfg.device = dev->valueint;
    cJSON_Delete(root);
    return cfg;
}

// Resolve a model directory/file path. If model_path is NULL/empty, materialize
// the embedded asset (id) to a per-process temp file; for rnnoise (single-file
// loader) return the file path, for directory loaders return the parent dir.
// Returns empty path if no embedded asset is available (caller throws).
std::filesystem::path resolve_utility_model_path(
    const char * model_path, const char * embedded_id, const char * dest_filename,
    bool want_directory) {
    if (model_path != nullptr && model_path[0] != '\0') {
        return std::filesystem::path(model_path);
    }
    // Embedded path: materialize to temp, return dir or file as the loader wants.
    const auto file = engine::assets::embedded::embedded_asset_file(embedded_id, dest_filename);
    if (file.empty()) {
        return {};
    }
    return want_directory ? file.parent_path() : file;
}

}  // namespace

audiocpp_audio_t *audiocpp_denoise(
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *model_name,
    const char *model_path,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!pcm) {
            throw std::runtime_error("pcm is null");
        }
        if (sample_rate <= 0) {
            throw std::runtime_error("sample_rate must be > 0");
        }
        if (model_name == nullptr || model_name[0] == '\0') {
            throw std::runtime_error("model_name is required (deepfilternet2|rnnoise|zipenhancer)");
        }
        const std::string name = model_name;
        const auto backend = parse_utility_backend(options_json);
        std::vector<float> input(pcm, pcm + static_cast<size_t>(n_samples));

        if (name == "deepfilternet2") {
            const auto dir = resolve_utility_model_path(
                model_path, "deepfilternet2", "deepfilternet2.safetensors", /*want_directory=*/true);
            if (dir.empty()) {
                throw std::runtime_error("deepfilternet2: no model path and no embedded asset");
            }
            auto model = get_or_load_utility_model<engine::audio::DeepFilterNet2Model>(
                utility_model_cache_key(dir, backend),
                [&]() { return std::make_shared<engine::audio::DeepFilterNet2Model>(
                    engine::audio::DeepFilterNet2Model::load_from_directory(dir, backend)); });
            auto in48 = engine::audio::resample_mono_linear(input, sample_rate, 48000);
            auto out = model->run_mono_48k(in48);
            result = pack_audio_output(out.samples, out.sample_rate);
        } else if (name == "rnnoise") {
            // rnnoise loads a single safetensors FILE (not a dir). utility_api
            // hardcodes rnnoise10Gb_15.safetensors; we replicate that.
            const auto file = resolve_utility_model_path(
                model_path, "rnnoise", "rnnoise10Gb_15.safetensors", /*want_directory=*/false);
            if (file.empty()) {
                throw std::runtime_error("rnnoise: no model path and no embedded asset");
            }
            auto model = get_or_load_utility_model<engine::audio::RnnoiseModel>(
                utility_model_cache_key(file, backend),
                [&]() { return std::make_shared<engine::audio::RnnoiseModel>(
                    engine::audio::RnnoiseModel::load_from_safetensors(file, backend)); });
            auto in48 = engine::audio::resample_mono_linear(input, sample_rate, 48000);
            auto out = model->process_mono_48k(in48);
            result = pack_audio_output(out.samples, out.sample_rate);
        } else if (name == "zipenhancer") {
            const auto dir = resolve_utility_model_path(
                model_path, "zipenhancer", "zipenhancer.safetensors", /*want_directory=*/true);
            if (dir.empty()) {
                throw std::runtime_error("zipenhancer: no model path and no embedded asset");
            }
            auto model = get_or_load_utility_model<engine::audio::ZipEnhancerModel>(
                utility_model_cache_key(dir, backend),
                [&]() { return std::make_shared<engine::audio::ZipEnhancerModel>(
                    engine::audio::ZipEnhancerModel::load_from_directory(dir, backend)); });
            auto in16 = engine::audio::resample_mono_linear(input, sample_rate, 16000);
            auto out = model->denoise_mono_16k(in16);
            result = pack_audio_output(out.samples, out.sample_rate);
        } else {
            throw std::runtime_error(
                "unsupported denoise model: " + name +
                " (expected deepfilternet2|rnnoise|zipenhancer)");
        }
    });
    return result;
}

audiocpp_audio_t *audiocpp_super_resolve(
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *model_path,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!pcm) {
            throw std::runtime_error("pcm is null");
        }
        if (sample_rate <= 0) {
            throw std::runtime_error("sample_rate must be > 0");
        }
        const auto backend = parse_utility_backend(options_json);
        const auto dir = resolve_utility_model_path(
            model_path, "flashsr", "flashsr.safetensors", /*want_directory=*/true);
        if (dir.empty()) {
            throw std::runtime_error("flashsr: no model path and no embedded asset");
        }
        std::vector<float> input(pcm, pcm + static_cast<size_t>(n_samples));
        auto in16 = engine::audio::resample_mono_linear(input, sample_rate, 16000);
        auto model = get_or_load_utility_model<engine::audio::FlashSrModel>(
            utility_model_cache_key(dir, backend),
            [&]() { return std::make_shared<engine::audio::FlashSrModel>(
                engine::audio::FlashSrModel::load_from_directory(dir, backend)); });
        auto out = model->super_resolve_mono_16k(in16);
        result = pack_audio_output(out.samples, out.sample_rate);
    });
    return result;
}

/* ======================================================================== */
/* Forced Alignment                                                          */
/* ======================================================================== */

audiocpp_align_t *audiocpp_align(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *text,
    const char *language,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_align_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        if (!text || text[0] == '\0') {
            throw std::runtime_error("align requires non-empty transcript text");
        }
        if (!language || language[0] == '\0') {
            throw std::runtime_error("align requires a language code (e.g. \"en\", \"zh\")");
        }

        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        req.text_input = engine::runtime::Transcript{};
        req.text_input->text = text;
        req.text_input->language = language;
        apply_options(req, options_json);

        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        const auto &wt = task_result.word_timestamps;
        const double inv_rate = sample_rate > 0 ? 1.0 / static_cast<double>(sample_rate) : 0.0;

        result = new audiocpp_align_t{};
        result->language = dup_cstr(language);
        result->n_words = static_cast<int64_t>(wt.size());
        if (wt.empty()) {
            result->words = nullptr;
        } else {
            result->words = static_cast<audiocpp_word_t *>(
                calloc(wt.size(), sizeof(audiocpp_word_t)));
            for (size_t i = 0; i < wt.size(); ++i) {
                result->words[i].start_seconds = static_cast<double>(wt[i].span.start_sample) * inv_rate;
                result->words[i].end_seconds = static_cast<double>(wt[i].span.end_sample) * inv_rate;
                result->words[i].word = dup_cstr(wt[i].word);
                result->words[i].confidence = wt[i].confidence;
            }
        }
    });
    return result;
}

void audiocpp_free_align(audiocpp_align_t *align) {
    if (!align) return;
    if (align->words) {
        for (int64_t i = 0; i < align->n_words; ++i) {
            free(align->words[i].word);
        }
        free(align->words);
    }
    free(align->language);
    delete align;
}

/* ======================================================================== */
/* Multi-stem audio transform                                               */
/* ======================================================================== */

audiocpp_stems_t *audiocpp_transform_stems(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
) {
    audiocpp_stems_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        if (voice_ref_pcm && voice_ref_n > 0) {
            engine::runtime::VoiceCondition voice;
            voice.speaker = engine::runtime::VoiceReference{};
            engine::runtime::AudioBuffer ref_audio;
            ref_audio.sample_rate = voice_ref_sr;
            ref_audio.channels = 1;
            ref_audio.samples.assign(voice_ref_pcm, voice_ref_pcm + voice_ref_n);
            voice.speaker->audio = std::move(ref_audio);
            req.voice = std::move(voice);
        }
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        // Collect ALL named audio outputs as stems
        result = new audiocpp_stems_t{};
        auto collect_stem = [&](const char *name, const engine::runtime::AudioBuffer &buf) {
            audiocpp_stem_t stem;
            stem.name = dup_cstr(name);
            stem.sample_rate = buf.sample_rate;
            stem.channels = std::max(1, buf.channels);
            stem.n_samples = static_cast<int64_t>(buf.samples.size());
            stem.samples = static_cast<float *>(
                malloc(buf.samples.size() * sizeof(float)));
            if (stem.samples) {
                std::memcpy(stem.samples, buf.samples.data(),
                            buf.samples.size() * sizeof(float));
            }
            // Grow the stems array
            int64_t idx = result->n_stems;
            result->n_stems = idx + 1;
            result->stems = static_cast<audiocpp_stem_t *>(
                realloc(result->stems, result->n_stems * sizeof(audiocpp_stem_t)));
            result->stems[idx] = stem;
        };

        // If there's a primary audio_output, include it as "output"
        if (task_result.audio_output && !task_result.audio_output->samples.empty()) {
            collect_stem("output", *task_result.audio_output);
        }
        // Include all named outputs (vocals, drums, bass, instrumental, etc.)
        for (const auto &named : task_result.named_audio_outputs) {
            if (!named.audio.samples.empty()) {
                collect_stem(named.id.c_str(), named.audio);
            }
        }
    });
    return result;
}

void audiocpp_free_stems(audiocpp_stems_t *stems) {
    if (!stems) return;
    if (stems->stems) {
        for (int64_t i = 0; i < stems->n_stems; ++i) {
            free(stems->stems[i].name);
            free(stems->stems[i].samples);
        }
        free(stems->stems);
    }
    delete stems;
}

/* ======================================================================== */
/* Model inspection                                                         */
/* ======================================================================== */

int audiocpp_model_info(
    const audiocpp_model_t *model,
    audiocpp_model_info_t *out
) {
    if (!model || !model->loaded_model || !out) return -1;
    const auto &meta = model->loaded_model->metadata();
    out->family = dup_cstr(meta.family);
    out->variant = dup_cstr(meta.variant);
    out->description = dup_cstr(meta.description);
    return 0;
}

int audiocpp_model_capabilities(
    const audiocpp_model_t *model,
    audiocpp_model_capabilities_t *out
) {
    if (!model || !model->loaded_model || !out) return -1;
    memset(out, 0, sizeof(*out));
    const auto &caps = model->loaded_model->capabilities();
    out->supports_speaker_reference = caps.supports_speaker_reference ? 1 : 0;
    out->supports_style_condition = caps.supports_style_condition ? 1 : 0;
    out->supports_timestamps = caps.supports_timestamps ? 1 : 0;
    // Supported tasks
    out->n_supported_tasks = static_cast<int>(caps.supported_tasks.size());
    if (out->n_supported_tasks > 0) {
        out->supported_tasks = static_cast<int *>(
            calloc(out->n_supported_tasks, sizeof(int)));
        for (int i = 0; i < out->n_supported_tasks; ++i) {
            out->supported_tasks[i] = static_cast<int>(caps.supported_tasks[i].task);
        }
    }
    // Languages
    out->n_languages = static_cast<int>(caps.languages.size());
    if (out->n_languages > 0) {
        out->languages = static_cast<char **>(
            calloc(out->n_languages, sizeof(char *)));
        for (int i = 0; i < out->n_languages; ++i) {
            out->languages[i] = dup_cstr(caps.languages[i]);
        }
    }
    return 0;
}

void audiocpp_free_model_info(audiocpp_model_info_t *info) {
    if (!info) return;
    free(info->family);
    free(info->variant);
    free(info->description);
    memset(info, 0, sizeof(*info));
}

void audiocpp_free_capabilities(audiocpp_model_capabilities_t *caps) {
    if (!caps) return;
    free(caps->supported_tasks);
    if (caps->languages) {
        for (int i = 0; i < caps->n_languages; ++i) {
            free(caps->languages[i]);
        }
        free(caps->languages);
    }
    memset(caps, 0, sizeof(*caps));
}

/* ======================================================================== */
/* WAV I/O utilities                                                        */
/* ======================================================================== */

int audiocpp_read_wav(
    const char *path,
    float **out_samples,
    int64_t *out_n,
    int *out_rate
) {
    if (!path || !out_samples || !out_n || !out_rate) return -1;
    try {
        auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
        *out_n = static_cast<int64_t>(wav.samples.size());
        *out_rate = wav.sample_rate;
        *out_samples = static_cast<float *>(malloc(wav.samples.size() * sizeof(float)));
        if (!*out_samples) return -1;
        std::memcpy(*out_samples, wav.samples.data(), wav.samples.size() * sizeof(float));
        return 0;
    } catch (...) {
        return -1;
    }
}

int audiocpp_write_wav_ex(
    const char *path,
    const float *samples,
    int64_t n_samples,
    int sample_rate,
    int channels
) {
    if (!path || !samples || n_samples <= 0 || channels <= 0) return -1;
    // Interleaved data must hold a whole number of frames per channel.
    if (n_samples % channels != 0) return -1;
    try {
        std::vector<float> vec(samples, samples + n_samples);
        engine::audio::write_pcm16_wav(
            std::filesystem::path(path),
            sample_rate,
            channels,
            vec);
        return 0;
    } catch (...) {
        return -1;
    }
}

int audiocpp_write_wav(
    const char *path,
    const float *samples,
    int64_t n_samples,
    int sample_rate
) {
    return audiocpp_write_wav_ex(path, samples, n_samples, sample_rate, 1);
}

/* ======================================================================== */
/* Artifacts (reserved for future use)                                      */
/* ======================================================================== */

audiocpp_artifact_t *audiocpp_artifact_create(
    int kind,
    const char *id,
    const uint8_t *payload,
    int64_t payload_size
) {
    if (!id) return nullptr;
    auto *art = new audiocpp_artifact_t{};
    art->kind = kind;
    art->id = dup_cstr(id);
    art->n_meta = 0;
    art->meta_keys = nullptr;
    art->meta_values = nullptr;
    if (payload && payload_size > 0) {
        art->payload = static_cast<uint8_t *>(malloc(static_cast<size_t>(payload_size)));
        if (art->payload) {
            std::memcpy(art->payload, payload, static_cast<size_t>(payload_size));
            art->payload_size = payload_size;
        }
    }
    return art;
}

int audiocpp_artifact_set_meta(
    audiocpp_artifact_t *artifact,
    const char *key,
    const char *value
) {
    if (!artifact || !key || !value) return -1;
    // Check if key already exists
    for (int i = 0; i < artifact->n_meta; ++i) {
        if (std::strcmp(artifact->meta_keys[i], key) == 0) {
            free(artifact->meta_values[i]);
            artifact->meta_values[i] = dup_cstr(value);
            return 0;
        }
    }
    // Append new key-value pair
    int idx = artifact->n_meta;
    artifact->n_meta = idx + 1;
    artifact->meta_keys = static_cast<char **>(
        realloc(artifact->meta_keys, artifact->n_meta * sizeof(char *)));
    artifact->meta_values = static_cast<char **>(
        realloc(artifact->meta_values, artifact->n_meta * sizeof(char *)));
    artifact->meta_keys[idx] = dup_cstr(key);
    artifact->meta_values[idx] = dup_cstr(value);
    return 0;
}

void audiocpp_artifact_free(audiocpp_artifact_t *artifact) {
    if (!artifact) return;
    free_artifact_members(artifact);
    delete artifact;
}

void audiocpp_free_artifacts(audiocpp_artifacts_t *artifacts) {
    release_artifacts(artifacts);
}

/* ======================================================================== */
/* Music / audio generation (text → audio and/or artifacts)                 */
/* ======================================================================== */

audiocpp_gen_result_t *audiocpp_generate(
    const audiocpp_model_t *model,
    const char *prompt,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_gen_result_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        engine::runtime::TaskRequest req;
        if (prompt) {
            req.text_input = engine::runtime::Transcript{};
            req.text_input->text = prompt;
        }
        apply_options(req, options_json);
        // Some models (e.g. Qwen-family) require prepare() before run().
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        result = new audiocpp_gen_result_t{};
        // Audio generators (MiniMax-H3, stable_audio, ace_step) fill audio_output;
        // MuScriptor produces none. Following the audio_transform convention, a
        // present-but-empty buffer is treated as "no audio".
        if (task_result.audio_output && !task_result.audio_output->samples.empty()) {
            result->audio = audiocpp::detail::pack_audio_output(*task_result.audio_output);
        }
        // Artifacts: MuScriptor → MIDI bytes (artifact_output); MiniMax-H3 →
        // optional RGB24 video (output_artifacts) when options set return_video.
        // collect_artifacts merges both slots; surface a NULL pointer (not an
        // empty collection) when nothing was produced so callers can simple-
        // null-check res->artifacts.
        auto *arts = audiocpp::detail::collect_artifacts(task_result);
        if (arts->n_artifacts > 0) {
            result->artifacts = arts;
        } else {
            audiocpp_free_artifacts(arts);
        }
    });
    return result;
}

void audiocpp_free_gen_result(audiocpp_gen_result_t *res) {
    if (!res) return;
    audiocpp_free_audio(res->audio);
    audiocpp_free_artifacts(res->artifacts);
    delete res;
}

// Shared core for the MIDI entries: run MuScriptor on a TaskRequest whose
// audio_input is already populated, and collect the produced artifacts. Throws
// on error so each entry's AUDIOCPP_CATCH surfaces it via err.
static audiocpp_artifacts_t *run_midi_artifacts(
    const audiocpp_model_t *model,
    engine::runtime::TaskRequest &req) {
    if (!model || !model->offline) {
        throw std::runtime_error("invalid model handle");
    }
    // MuScriptor requires prepare() before run() (require_prepared).
    model->session->prepare(engine::runtime::build_preparation_request(req));
    auto task_result = model->offline->run(req);
    // MuScriptor emits artifact_output (MIDI/JSON) + text_output, never
    // audio_output. collect_artifacts merges both artifact slots; require at
    // least one so a no-op run surfaces as an error rather than NULL.
    auto *arts = audiocpp::detail::collect_artifacts(task_result);
    if (arts->n_artifacts == 0) {
        audiocpp_free_artifacts(arts);
        throw std::runtime_error("MIDI transcription produced no artifact");
    }
    return arts;
}

// MuScriptor is an audio→MIDI transcriber (it ignores text_input and throws
// "requires audio_input" without PCM). Unlike audiocpp_generate (text→audio),
// the MIDI entries take input audio and return the MIDI/JSON artifacts only.
audiocpp_artifacts_t *audiocpp_midi(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_artifacts_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        result = run_midi_artifacts(model, req);
    });
    return result;
}

audiocpp_artifacts_t *audiocpp_midi_from_wav(
    const audiocpp_model_t *model,
    const uint8_t *wav_data,
    int64_t wav_size,
    const char *options_json,
    audiocpp_error_t *err
) {
    // Convenience over audiocpp_midi: decode a .wav file's bytes in-memory (no
    // temp file) and downmix to mono, then run the same MuScriptor path. Lets a
    // caller that already holds the WAV in memory (HTTP upload, etc.) skip the
    // audiocpp_read_wav + deinterleave dance.
    audiocpp_artifacts_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!wav_data || wav_size <= 0) {
            throw std::runtime_error("invalid WAV data");
        }
        // read_wav_f32(string_view) parses RIFF from the byte buffer directly
        // (supports PCM16 / PCM24 / IEEE float32; throws on anything else).
        const auto wav = engine::audio::read_wav_f32(std::string_view(
            reinterpret_cast<const char *>(wav_data), static_cast<size_t>(wav_size)));
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = wav.sample_rate;
        req.audio_input->channels = 1;
        // target = native rate → pure downmix, no resample (MuScriptor resamples
        // internally). Averages all channels per frame for stereo/mono input.
        req.audio_input->samples =
            engine::audio::convert_wav_to_mono_linear_resampled(wav, wav.sample_rate);
        apply_options(req, options_json);
        result = run_midi_artifacts(model, req);
    });
    return result;
}

/* ======================================================================== */
/* Streaming (chunk-push model)                                             */
/* ======================================================================== */

audiocpp_stream_t *audiocpp_stream_start(
    const audiocpp_model_t *model,
    int task,
    const char *options_json,
    int64_t preferred_chunk_samples,
    audiocpp_error_t *err
) {
    audiocpp_stream_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->loaded_model) {
            throw std::runtime_error("invalid model handle");
        }
        // preferred_chunk_samples is accepted for forward compatibility — the
        // engine currently uses each model's own StreamingPolicy value. When a
        // caller override path lands in the engine, plumb this through
        // TaskRequest::options["preferred_audio_chunk_samples"].
        (void)preferred_chunk_samples;
        // Create a new streaming session on the same backend the model was
        // loaded with (so streaming runs on the caller's chosen GPU, not CPU).
        engine::runtime::TaskSpec task_spec;
        task_spec.task = map_task(task);
        task_spec.mode = engine::runtime::RunMode::Streaming;

        engine::runtime::SessionOptions opts;
        opts.backend = model->backend_config;

        auto session = model->loaded_model->create_task_session(task_spec, opts);
        if (!session) {
            throw std::runtime_error("failed to create streaming task session");
        }

        auto *streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(
            session.get());
        if (!streaming) {
            throw std::runtime_error("model does not support streaming for this task");
        }

        // Build an initial request from options, prepare the session, then start
        // the stream — all BEFORE taking ownership of the result handle.
        // start_stream can throw (e.g. empty text, model rejects the request); if
        // it does, we must return NULL so the caller sees the failure in `err`
        // instead of a half-initialized stream handle that blows up on the first
        // pull ("streaming has not been started").
        engine::runtime::TaskRequest req;
        apply_options(req, options_json);
        // Streaming sessions still need prepare() (e.g. supertonic's start_stream
        // calls require_prepared). The offline run() path prepares implicitly;
        // this streaming entry point must do it explicitly.
        streaming->prepare(engine::runtime::build_preparation_request(req));
        streaming->start_stream(req);

        result = new audiocpp_stream{};
        result->session = std::move(session);
        result->streaming = streaming;
        result->next_start_sample = 0;

        // Register a sink to collect events emitted via callback
        result->sink_events.clear();
        streaming->set_stream_event_sink([&result](const engine::runtime::StreamEvent &ev) {
            result->sink_events.push_back(ev);
        });
    });
    return result;
}

audiocpp_stream_event_t *audiocpp_stream_push(
    audiocpp_stream_t *stream,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    audiocpp_error_t *err
) {
    audiocpp_stream_event_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!stream || !stream->streaming) {
            throw std::runtime_error("invalid stream handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio chunk");
        }

        // For PullEvents-stream models (streaming TTS: input=None), process_audio_chunk
        // throws ("does not consume audio chunks"). Return an empty event so push is a
        // no-op for such models; callers should use audiocpp_stream_pull instead.
        const auto policy = stream->streaming->streaming_policy();
        if (policy.input == engine::runtime::StreamingInputKind::None) {
            result = new audiocpp_stream_event_t{};
            memset(result, 0, sizeof(*result));
        } else {

        engine::runtime::AudioChunk chunk;
        chunk.sample_rate = sample_rate;
        chunk.channels = 1;
        chunk.start_sample = stream->next_start_sample;
        chunk.samples.assign(pcm, pcm + n_samples);
        stream->next_start_sample += n_samples;

        auto ev = stream->streaming->process_audio_chunk(chunk);

        result = new audiocpp_stream_event_t{};
        memset(result, 0, sizeof(*result));
        result->is_final = ev.is_final ? 1 : 0;

        // Map VAD events
        if (!ev.voice_activity.empty()) {
            result->n_va_events = static_cast<int>(ev.voice_activity.size());
            result->va_events = static_cast<audiocpp_va_event_t *>(
                calloc(result->n_va_events, sizeof(audiocpp_va_event_t)));
            for (int i = 0; i < result->n_va_events; ++i) {
                result->va_events[i].kind = static_cast<int>(ev.voice_activity[i].kind);
                result->va_events[i].sample = ev.voice_activity[i].sample;
                result->va_events[i].probability = ev.voice_activity[i].probability;
            }
        }

        // Map partial text
        if (ev.partial_text && !ev.partial_text->text.empty()) {
            result->partial_text = dup_cstr(ev.partial_text->text);
        }

        // Map audio output
        if (ev.audio_output && !ev.audio_output->samples.empty()) {
            const auto &buf = *ev.audio_output;
            result->audio_sample_rate = buf.sample_rate;
            result->n_audio_samples = static_cast<int64_t>(buf.samples.size());
            result->audio_samples = static_cast<float *>(
                malloc(buf.samples.size() * sizeof(float)));
            if (result->audio_samples) {
                std::memcpy(result->audio_samples, buf.samples.data(),
                            buf.samples.size() * sizeof(float));
            }
        }
        }  // end else (input != None)
    });
    return result;
}

int audiocpp_stream_finish(
    audiocpp_stream_t *stream,
    audiocpp_text_t *out_text,
    audiocpp_error_t *err
) {
    if (!stream) return -1;
    AUDIOCPP_CATCH(err, {
        if (!stream->streaming) {
            throw std::runtime_error("invalid stream handle");
        }
        auto task_result = stream->streaming->finish_stream();

        // Some streaming ASR models (e.g. Nemotron) emit partial-text events to
        // the sink only inside finalize(), which runs as part of finish_stream()
        // ABOVE — after the caller's last stream_pull(). Those events would be
        // stranded. Recover them: if the final TaskResult has no text but sink
        // collected partial_text events, use the last partial as the result.
        if ((!task_result.text_output || task_result.text_output->text.empty())
            && out_text) {
            for (const auto &ev : stream->sink_events) {
                if (ev.partial_text.has_value() && !ev.partial_text->text.empty()) {
                    if (!task_result.text_output) {
                        task_result.text_output = engine::runtime::Transcript{};
                    }
                    task_result.text_output->text = ev.partial_text->text;
                    if (task_result.text_output->language.empty()) {
                        task_result.text_output->language = ev.partial_text->language;
                    }
                }
            }
        }

        // Clear the sink (events are now folded into task_result or were already
        // returned via stream_pull during the session).
        stream->sink_events.clear();
        stream->streaming->set_stream_event_sink(nullptr);

        // Extract final text (for ASR)
        if (out_text) {
            memset(out_text, 0, sizeof(*out_text));
            if (task_result.text_output) {
                out_text->text = dup_cstr(task_result.text_output->text);
                out_text->language = !task_result.text_output->language.empty()
                    ? dup_cstr(task_result.text_output->language)
                    : nullptr;
            }
        }
    });
    return err && err->code != 0 ? -1 : 0;
}

void audiocpp_free_stream_event(audiocpp_stream_event_t *event) {
    if (!event) return;
    free(event->va_events);
    free(event->partial_text);
    free(event->audio_samples);
    delete event;
}

audiocpp_stream_event_t *audiocpp_stream_pull(
    audiocpp_stream_t *stream,
    int timeout_ms,
    audiocpp_error_t *err
) {
    audiocpp_stream_event_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!stream || !stream->streaming) {
            throw std::runtime_error("invalid stream handle");
        }
        // next_stream_event() pulls one generated event (PullEvents output).
        // For streaming TTS (supertonic/omnivoice/voxcpm2) this returns audio
        // chunks; for input=None models process_audio_chunk would throw, so
        // callers MUST use stream_pull (not stream_push) for TTS.
        //
        // timeout_ms is accepted for forward compatibility but the current
        // implementation is synchronous: -1 / 0 / >0 all block until an event
        // is available or the stream is exhausted. Reject any other (invalid)
        // value explicitly so callers don't get silent wrong behavior.
        if (timeout_ms != -1 && timeout_ms < 0) {
            throw std::runtime_error("timeout_ms must be -1 or >= 0");
        }
        auto maybe_ev = stream->streaming->next_stream_event();
        if (!maybe_ev) {
            // Stream exhausted (no more data) — result stays nullptr, not an error
        } else {
        const auto &ev = *maybe_ev;
        result = new audiocpp_stream_event_t{};
        memset(result, 0, sizeof(*result));
        result->is_final = ev.is_final ? 1 : 0;

        // Map VAD events
        if (!ev.voice_activity.empty()) {
            result->n_va_events = static_cast<int>(ev.voice_activity.size());
            result->va_events = static_cast<audiocpp_va_event_t *>(
                calloc(result->n_va_events, sizeof(audiocpp_va_event_t)));
            for (int i = 0; i < result->n_va_events; ++i) {
                result->va_events[i].kind = static_cast<int>(ev.voice_activity[i].kind);
                result->va_events[i].sample = ev.voice_activity[i].sample;
                result->va_events[i].probability = ev.voice_activity[i].probability;
            }
        }

        // Map partial text
        if (ev.partial_text && !ev.partial_text->text.empty()) {
            result->partial_text = dup_cstr(ev.partial_text->text);
        }

        // Map audio output: prefer audio_output, fall back to first named_audio_output
        // (streaming TTS like supertonic uses named_audio_outputs)
        const engine::runtime::AudioBuffer *audio_buf = nullptr;
        if (ev.audio_output && !ev.audio_output->samples.empty()) {
            audio_buf = &(*ev.audio_output);
        } else if (!ev.named_audio_outputs.empty() && !ev.named_audio_outputs[0].audio.samples.empty()) {
            audio_buf = &ev.named_audio_outputs[0].audio;
        }
        if (audio_buf) {
            result->audio_sample_rate = audio_buf->sample_rate;
            result->n_audio_samples = static_cast<int64_t>(audio_buf->samples.size());
            result->audio_samples = static_cast<float *>(
                malloc(audio_buf->samples.size() * sizeof(float)));
            if (result->audio_samples) {
                std::memcpy(result->audio_samples, audio_buf->samples.data(),
                            audio_buf->samples.size() * sizeof(float));
            }
        }
        }  // end else (maybe_ev has value)
    });
    return result;
}

void audiocpp_stream_free(audiocpp_stream_t *stream) {
    if (!stream) return;
    // session unique_ptr cleans up the C++ session object
    delete stream;
}

void audiocpp_free_string(char *str) {
    free(str);
}

void audiocpp_clear_error(audiocpp_error_t *err) {
    if (!err) return;
    if (err->message) {
        free(err->message);
        err->message = nullptr;
    }
    err->code = 0;
}

/* ======================================================================== */
/* Device enumeration                                                        */
/* ======================================================================== */

namespace {

// Map a ggml backend registration to our AUDIOCPP_BACKEND_* enum.
int backend_reg_to_id(ggml_backend_reg_t reg) {
    // Match by registry name (not the deprecated direct ggml_backend_*_reg()
    // pointer compare). This works with dynamically-loaded backends and the
    // CUDA alias names used under HIP ("ROCm") and MUSA builds — mirroring the
    // engine's own reg_name_matches() logic in backend.cpp.
    if (!reg) return AUDIOCPP_BACKEND_CPU;
    const char * name = ggml_backend_reg_name(reg);
    if (name == nullptr) return AUDIOCPP_BACKEND_CPU;
    const auto matches = [&](std::initializer_list<const char *> aliases) {
        for (const char * a : aliases) {
            if (std::strcmp(name, a) == 0) return true;
        }
        return false;
    };
    if (matches({"CUDA", "ROCm", "MUSA"})) return AUDIOCPP_BACKEND_CUDA;
    if (std::strcmp(name, "Vulkan") == 0)  return AUDIOCPP_BACKEND_VULKAN;
    if (std::strcmp(name, "MTL") == 0)     return AUDIOCPP_BACKEND_METAL;
    if (std::strcmp(name, "SYCL") == 0)    return AUDIOCPP_BACKEND_SYCL;
    // CPU and any unknown backend
    return AUDIOCPP_BACKEND_CPU;
}

int dev_type_to_id(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return AUDIOCPP_DEVICE_GPU;
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return AUDIOCPP_DEVICE_IGPU;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return AUDIOCPP_DEVICE_ACCEL;
        case GGML_BACKEND_DEVICE_TYPE_META:  return AUDIOCPP_DEVICE_META;
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        default:                             return AUDIOCPP_DEVICE_CPU;
    }
}

void copy_cstr(char *dst, size_t dst_size, const char *src) {
    if (!src || dst_size == 0) {
        if (dst_size > 0) dst[0] = '\0';
        return;
    }
    size_t len = strlen(src);
    if (len >= dst_size) len = dst_size - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}

}  // namespace

int audiocpp_device_count(void) {
    return static_cast<int>(ggml_backend_dev_count());
}

int audiocpp_device_info(int index, audiocpp_device_info_t *out) {
    if (!out || index < 0 || static_cast<size_t>(index) >= ggml_backend_dev_count()) {
        return -1;
    }
    memset(out, 0, sizeof(*out));

    ggml_backend_dev_t dev = ggml_backend_dev_get(static_cast<size_t>(index));
    if (!dev) return -1;

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);

    copy_cstr(out->name, sizeof(out->name), props.name);
    copy_cstr(out->description, sizeof(out->description), props.description);
    out->type = dev_type_to_id(props.type);
    out->memory_total = static_cast<uint64_t>(props.memory_total);
    out->memory_free = static_cast<uint64_t>(props.memory_free);

    // Determine which backend this device belongs to.
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    out->backend = backend_reg_to_id(reg);

    // Compute per-backend device index: count how many earlier devices in the
    // global list share the same backend reg. This aligns with
    // ggml_backend_cuda_init(device) / ggml_backend_sycl_init(device) which
    // take backend-relative indices.
    int per_backend_idx = 0;
    for (int i = 0; i < index; ++i) {
        ggml_backend_dev_t earlier = ggml_backend_dev_get(static_cast<size_t>(i));
        if (earlier && ggml_backend_dev_backend_reg(earlier) == reg) {
            ++per_backend_idx;
        }
    }
    out->device_id = per_backend_idx;

    return 0;
}

void audiocpp_list_devices(void) {
    int count = audiocpp_device_count();
    printf("audio.cpp devices (%d):\n", count);
    printf("  %-4s  %-10s  %-6s  %-10s  %-30s\n",
           "idx", "backend", "type", "device_id", "name");
    printf("  %s\n", "---------------------------------------------------------------");
    static const char *backend_names[] = {
        "CPU", "CUDA", "Vulkan", "Metal", "SYCL", "BEST"
    };
    static const char *type_names[] = {"CPU", "GPU", "IGPU"};
    for (int i = 0; i < count; ++i) {
        audiocpp_device_info_t info;
        if (audiocpp_device_info(i, &info) == 0) {
            const char *bn = (info.backend >= 0 && info.backend <= 5)
                ? backend_names[info.backend] : "?";
            const char *tn = (info.type >= 0 && info.type <= 2)
                ? type_names[info.type] : "?";
            printf("  %-4d  %-10s  %-6s  %-10d  %-30s\n",
                   i, bn, tn, info.device_id, info.name);
        }
    }
}

/* ======================================================================== */
/* Backend availability probe                                                */
/* ======================================================================== */

namespace {

// Compile-time half of the availability check: was this backend baked into
// the build? Mirrors the backend tag logic in audiocpp_version() (the same
// GGML_USE_* macro set). CUDA also covers AMD ROCm/HIP and MUSA builds, whose
// ggml registries register under "ROCm"/"MUSA" (see backend_reg_to_id).
bool backend_compiled_in(int backend) {
    switch (backend) {
        case AUDIOCPP_BACKEND_CUDA:
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
            return true;
#else
            return false;
#endif
        case AUDIOCPP_BACKEND_VULKAN:
#if defined(GGML_USE_VULKAN)
            return true;
#else
            return false;
#endif
        case AUDIOCPP_BACKEND_METAL:
#if defined(GGML_USE_METAL)
            return true;
#else
            return false;
#endif
        case AUDIOCPP_BACKEND_SYCL:
#if defined(GGML_USE_SYCL)
            return true;
#else
            return false;
#endif
        default:
            // CPU is handled by the caller (always available); BEST/unknown
            // never match a single compiled-in backend.
            return false;
    }
}

// Runtime half: does at least one device of this backend exist right now
// (driver installed, hardware detected)? Device discovery is ggml's own
// graceful probe — a backend with no usable driver simply registers zero
// devices (e.g. ggml_backend_cuda_reg() returns nullptr when cuInit fails);
// it never aborts. Mirrors engine::core::list_backend_devices().
bool backend_has_device(int backend) {
    // Trigger registration of statically compiled backends (the ggml backend
    // registry constructor calls each *_reg()) and, in GGML_BACKEND_DL
    // builds, pick up dynamically loaded backends.
    if (ggml_backend_reg_count() == 0) {
        ggml_backend_load_all();
    }
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg == nullptr) {
            continue;
        }
        if (backend_reg_to_id(reg) == backend) {
            return true;
        }
    }
    return false;
}

// AUDIOCPP_BACKEND_BEST: available iff some GPU backend is compiled in AND
// has a device at runtime.
bool backend_has_gpu_device(void) {
    static const int kGpuBackends[] = {
        AUDIOCPP_BACKEND_CUDA,
        AUDIOCPP_BACKEND_VULKAN,
        AUDIOCPP_BACKEND_METAL,
        AUDIOCPP_BACKEND_SYCL,
    };
    for (int backend : kGpuBackends) {
        if (backend_compiled_in(backend) && backend_has_device(backend)) {
            return true;
        }
    }
    return false;
}

}  // namespace

int audiocpp_backend_available(int backend) {
    switch (backend) {
        case AUDIOCPP_BACKEND_CPU:
            return 1;  // CPU is always compiled in and always "present"
        case AUDIOCPP_BACKEND_CUDA:
        case AUDIOCPP_BACKEND_VULKAN:
        case AUDIOCPP_BACKEND_METAL:
        case AUDIOCPP_BACKEND_SYCL:
            return (backend_compiled_in(backend) && backend_has_device(backend)) ? 1 : 0;
        case AUDIOCPP_BACKEND_BEST:
            return backend_has_gpu_device() ? 1 : 0;
        default:
            return 0;  // unknown backend id — never abort
    }
}

