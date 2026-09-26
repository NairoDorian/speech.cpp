// engine/models/canary_qwen/session.cpp - offline ASR session + loader for the
// native engine Canary-Qwen package (port of src/runtime/arch/canary_qwen).
//
// Loads parent transcribe.cpp's GGUF (general.architecture = "canary_qwen",
// huggingface.co/handy-computer/canary-qwen-2.5b-gguf). can_load() sniffs the
// file rather than resolving a model-spec bundle: those GGUFs are third-party
// and embed no sidecars.
//
// Request options. The arch's run() read none of transcribe_run_params (the
// prompt is the fixed chat template "Transcribe the following: <audio>"):
//   language     accepted and ignored (English-only; the C ABI dispatcher
//                validates it against the published language list)
//   task         transcribe (translate -> std::invalid_argument; the model
//                cannot, and the C ABI rejects it before dispatch)
//   timestamps   accepted and ignored: the result is one untimed transcript
// Session options:
//   kv_type      f16 (default, the arch's AUTO) | f32 | auto
//   n_ctx        lowers the decoder context ceiling (0 = model maximum)
// Env (read exactly as the arch did): TRANSCRIBE_NO_FLASH / _FORCE_FLASH,
// TRANSCRIBE_CONV_{DIRECT,NO_DIRECT}_{PW,DW}.

#include "engine/models/canary_qwen/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/canary_qwen/runtime.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace engine::models::canary_qwen {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

CanaryQwenRuntimeOptions runtime_options_from(const runtime::SessionOptions &options) {
  CanaryQwenRuntimeOptions out;
  if (const auto kv = runtime::find_option(options.options, {"canary_qwen.kv_type", "kv_type"})) {
    const std::string v = lower(*kv);
    if (v.empty() || v == "auto" || v == "f16") {
      out.kv_type = GGML_TYPE_F16;
    } else if (v == "f32") {
      out.kv_type = GGML_TYPE_F32;
    } else {
      throw std::invalid_argument("canary_qwen.kv_type must be auto, f16 or f32");
    }
  }
  if (const auto n_ctx = runtime::parse_int_option(options.options, {"canary_qwen.n_ctx", "n_ctx"})) {
    if (*n_ctx < 0) {
      throw std::invalid_argument("canary_qwen.n_ctx must be >= 0");
    }
    out.n_ctx = *n_ctx;
  }
  apply_flash_env_overrides(out);
  return out;
}

void check_request_options(const runtime::TaskRequest &request) {
  if (const auto task = runtime::find_option(request.options, {"task"})) {
    const std::string t = lower(*task);
    if (!t.empty() && t != "transcribe") {
      throw std::invalid_argument("canary_qwen: task '" + *task +
                                  "' is not supported (English transcription only)");
    }
  }
}

std::vector<float> pcm_16k_mono(const runtime::AudioBuffer &audio, int32_t sample_rate) {
  if (audio.channels == 1 && audio.sample_rate == sample_rate) {
    return audio.samples;  // the C ABI's buffer, bit for bit
  }
  if (audio.sample_rate <= 0 || audio.channels <= 0) {
    throw std::invalid_argument("canary_qwen: audio needs a positive sample rate and channel count");
  }
  return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
      audio.samples, audio.sample_rate, audio.channels, sample_rate);
}

runtime::TaskResult to_task_result(const CanaryQwenTranscription &t) {
  runtime::TaskResult result;
  // No language detection: the arch never set a detected language.
  result.text_output = runtime::Transcript{t.text, ""};
  result.truncated = t.truncated;
  return result;
}

runtime::CapabilitySet base_capabilities() {
  runtime::CapabilitySet caps;
  caps.supported_tasks = {{runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
  caps.supports_timestamps = false;      // the arch's max_timestamp_kind NONE
  caps.supports_cancellation = true;     // RunControl polled per stage and decode step
  caps.supports_translate = false;
  caps.supports_language_detection = false;
  caps.supports_initial_prompt = false;  // fixed chat template
  caps.supports_temperature_fallback = false;
  caps.supports_long_form = false;       // hard context cap, no chunker
  return caps;
}

class CanaryQwenLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "canary_qwen"; }

  std::vector<std::string> family_aliases() const override {
    return {"canary-qwen", "canary_qwen_asr"};
  }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet caps = base_capabilities();
    caps.languages = {"en"};
    return caps;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value()) {
      const auto aliases = family_aliases();
      if (*request.family_hint != family() &&
          std::find(aliases.begin(), aliases.end(), *request.family_hint) == aliases.end()) {
        return false;
      }
    }
    return looks_like_canary_qwen_gguf(request.model_path);
  }

  runtime::ModelInspection inspect(const runtime::ModelLoadRequest &request) const override {
    runtime::ModelInspection inspection;
    inspection.model_root = request.model_path.parent_path();
    inspection.metadata.family = family();
    inspection.metadata.variant = "canary-qwen-2.5b";
    inspection.metadata.description =
        "NVIDIA Canary-Qwen-2.5B (NeMo SALM: FastConformer + Qwen3-1.7B), transcribe.cpp GGUF.";
    inspection.capabilities = advertised_capabilities();
    inspection.cli.request_options = {
        {"language", "en", "Accepted and ignored (English-only model).", false, "en"},
    };
    inspection.cli.session_options = {
        {"kv_type", "f16|f32|auto", "Decoder KV-cache dtype; auto = f16.", false, "f16"},
        {"n_ctx", "n", "Lower the decoder context ceiling (0 = model maximum).", false, "0", "0"},
    };
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    auto model_assets = load_canary_qwen_assets(request.model_path);
    runtime::ModelMetadata metadata;
    metadata.family = family();
    metadata.variant = model_assets->variant;
    metadata.description = "NVIDIA Canary-Qwen (NeMo SALM) speech recognition.";
    runtime::CapabilitySet caps = base_capabilities();
    caps.languages = model_assets->languages;
    caps.supports_translate = model_assets->supports_translate;
    caps.supports_language_detection = model_assets->supports_language_detect;
    return std::make_unique<CanaryQwenLoadedModel>(std::move(metadata), std::move(caps),
                                                   std::move(model_assets));
  }
};

} // namespace

CanaryQwenSession::CanaryQwenSession(runtime::TaskSpec task, runtime::SessionOptions options,
                                     std::shared_ptr<const CanaryQwenAssets> model_assets)
    : RuntimeSessionBase(std::move(options)), task_(std::move(task)),
      assets_(std::move(model_assets)) {
  if (task_.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("canary_qwen only supports VoiceTaskKind::Asr");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("canary_qwen supports offline mode only");
  }
  runtime_ = std::make_unique<CanaryQwenRuntime>(assets_, execution_context(),
                                                 runtime_options_from(RuntimeSessionBase::options()));
}

CanaryQwenSession::~CanaryQwenSession() = default;

std::string CanaryQwenSession::family() const { return "canary_qwen"; }

runtime::VoiceTaskKind CanaryQwenSession::task_kind() const { return task_.task; }

runtime::RunMode CanaryQwenSession::run_mode() const { return task_.mode; }

void CanaryQwenSession::prepare(const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error("canary_qwen prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::TaskResult CanaryQwenSession::run(const runtime::TaskRequest &request) {
  require_prepared("canary_qwen run()");
  if (!request.audio_input.has_value() || request.audio_input->samples.empty()) {
    throw std::invalid_argument("canary_qwen run() requires non-empty audio_input");
  }
  check_request_options(request);
  const std::vector<float> pcm = pcm_16k_mono(*request.audio_input, runtime_->sample_rate());

  CanaryQwenTranscription partial;
  try {
    return to_task_result(runtime_->transcribe(pcm, run_control(), &partial));
  } catch (const runtime::ProgressCanceled &canceled) {
    // The tokens decoded before the abort stay readable (ERR_ABORTED).
    runtime::ProgressCanceled with_partial(canceled.what());
    if (!partial.text.empty() || partial.n_generated_tokens > 0) {
      with_partial.partial = std::make_shared<const runtime::TaskResult>(to_task_result(partial));
    }
    throw with_partial;
  }
}

std::vector<runtime::TaskResult>
CanaryQwenSession::run_batch(const std::vector<runtime::TaskRequest> &requests) {
  require_prepared("canary_qwen run_batch()");
  std::vector<std::vector<float>> pcms(requests.size());
  std::vector<const std::vector<float> *> views(requests.size(), nullptr);
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto &request = requests[i];
    check_request_options(request);
    if (request.audio_input.has_value() && !request.audio_input->samples.empty()) {
      pcms[i] = pcm_16k_mono(*request.audio_input, runtime_->sample_rate());
      views[i] = &pcms[i];
    }
  }

  const auto items = runtime_->transcribe_batch(views, run_control());
  std::vector<runtime::TaskResult> results;
  results.reserve(items.size());
  for (const auto &item : items) {
    if (item.ok) {
      results.push_back(to_task_result(item.result));
    } else {
      // Rejected utterance: no transcript, its own status (the arch's
      // per-utterance INPUT_TOO_LONG / INVALID_ARG through the C ABI).
      runtime::TaskResult rejected;
      rejected.item_status = item.input_too_long ? runtime::TaskItemStatus::InputTooLong
                                                 : runtime::TaskItemStatus::InvalidArgument;
      rejected.item_error = item.error;
      results.push_back(std::move(rejected));
    }
  }
  return results;
}

CanaryQwenLoadedModel::CanaryQwenLoadedModel(runtime::ModelMetadata metadata,
                                             runtime::CapabilitySet capabilities,
                                             std::shared_ptr<const CanaryQwenAssets> model_assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(model_assets)) {}

const runtime::ModelMetadata &CanaryQwenLoadedModel::metadata() const noexcept { return metadata_; }

const runtime::CapabilitySet &CanaryQwenLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
CanaryQwenLoadedModel::create_task_session(const runtime::TaskSpec &task,
                                           const runtime::SessionOptions &options) const {
  return std::make_unique<CanaryQwenSession>(task, options, assets_);
}

std::optional<std::vector<int32_t>> CanaryQwenLoadedModel::tokenize(const std::string &text) const {
  return assets_->encode(text);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_canary_qwen_loader() {
  return std::make_shared<CanaryQwenLoader>();
}

} // namespace engine::models::canary_qwen
