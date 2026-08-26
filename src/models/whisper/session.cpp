// engine/models/whisper/session.cpp - offline ASR session + loader for the
// native engine Whisper package (Phase 11 Wave W2a).
//
// Unlike the moonshine packages, can_load() sniffs the legacy whisper.cpp
// `ggml` magic instead of resolving a model-spec resource bundle:
// model_specs/whisper.json is catalog-only (its 16 packages point at
// Whisper-*-GGUF paths that do not exist in audio-cpp/audio.cpp-gguf), so
// bundle resolution would fail for every real model. See assets.cpp.

#include "engine/models/whisper/session.h"

#include "engine/models/whisper/runtime.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::whisper {

namespace {

assets::TensorStorageType
weight_type_from_options(const runtime::SessionOptions &options) {
  // Legacy `.bin` weights are uploaded in the type the file stores; the option
  // is accepted and validated for forward compatibility with a GGUF path.
  const auto it = options.options.find("whisper.weight_type");
  if (it == options.options.end()) {
    return assets::TensorStorageType::Native;
  }
  const auto storage_type = assets::parse_tensor_storage_type(it->second);
  switch (storage_type) {
  case assets::TensorStorageType::Native:
  case assets::TensorStorageType::F32:
  case assets::TensorStorageType::F16:
    return storage_type;
  default:
    throw std::runtime_error(
        "whisper.weight_type currently supports only native, f32 and f16 "
        "(legacy .bin weights load in their stored type)");
  }
}

int32_t max_tokens_from_request(
    const std::unordered_map<std::string, std::string> &options) {
  const auto it = options.find("max_tokens");
  if (it == options.end() || it->second.empty()) {
    return 0;
  }
  try {
    const long long parsed = std::stoll(it->second);
    return parsed > 0 ? static_cast<int32_t>(parsed) : 0;
  } catch (...) {
    return 0;
  }
}

class WhisperLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "whisper"; }

  std::vector<std::string> family_aliases() const override {
    return {"whisper-offline"};
  }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet result;
    result.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
    result.languages = {"en"};
    result.supports_timestamps = false; // W2b
    return result;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value() && *request.family_hint != family() &&
        *request.family_hint != "whisper-offline") {
      return false;
    }
    return looks_like_whisper_bin(request.model_path);
  }

  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    runtime::ModelInspection inspection;
    inspection.model_root = request.model_path.parent_path();
    inspection.metadata.family = family();
    inspection.metadata.variant = "whisper";
    inspection.metadata.description =
        "Whisper encoder-decoder ASR (log-mel frontend, legacy whisper.cpp "
        ".bin weights).";
    inspection.capabilities = advertised_capabilities();
    inspection.cli.request_options = {
        {"max_tokens", "n", "Maximum generated transcript tokens.", false, "0",
         "0"},
    };
    inspection.cli.session_options = {
        {"whisper.weight_type", "native|f32|f16",
         "Shared model weight storage preference."},
    };
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    auto model_assets = load_whisper_assets(request.model_path);
    runtime::ModelMetadata metadata;
    metadata.family = family();
    metadata.variant = model_assets->variant;
    metadata.description =
        "Whisper encoder-decoder ASR (log-mel frontend, legacy whisper.cpp "
        ".bin weights).";
    return std::make_unique<WhisperLoadedModel>(
        std::move(metadata), advertised_capabilities(),
        std::move(model_assets));
  }
};

} // namespace

WhisperSession::WhisperSession(runtime::TaskSpec task,
                               runtime::SessionOptions options,
                               std::shared_ptr<const WhisperAssets> model_assets)
    : RuntimeSessionBase(std::move(options)), task_(std::move(task)),
      assets_(std::move(model_assets)),
      runtime_(std::make_unique<WhisperRuntime>(
          assets_, execution_context(),
          weight_type_from_options(RuntimeSessionBase::options()))) {
  if (task_.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("Whisper only supports VoiceTaskKind::Asr");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error(
        "Whisper supports offline mode only in this package (streaming and "
        "long-form seek are W2b)");
  }
}

WhisperSession::~WhisperSession() = default;

std::string WhisperSession::family() const { return "whisper"; }

runtime::VoiceTaskKind WhisperSession::task_kind() const { return task_.task; }

runtime::RunMode WhisperSession::run_mode() const { return task_.mode; }

void WhisperSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error("Whisper prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::TaskResult WhisperSession::run(const runtime::TaskRequest &request) {
  require_prepared("Whisper run()");
  if (!request.audio_input.has_value()) {
    throw std::runtime_error("Whisper run() requires audio_input");
  }

  const auto wall_start = std::chrono::steady_clock::now();
  const int32_t max_tokens = max_tokens_from_request(request.options);
  const auto transcription =
      runtime_->transcribe(*request.audio_input, run_control(), max_tokens);

  runtime::TaskResult result;
  result.text_output = runtime::Transcript{
      transcription.text,
      request.text_input.has_value() ? request.text_input->language : ""};
  engine::debug::timing_log_scalar("session.wall_ms",
                                   engine::debug::elapsed_ms(wall_start));
  return result;
}

std::vector<runtime::TaskResult>
WhisperSession::run_batch(const std::vector<runtime::TaskRequest> &requests) {
  std::vector<runtime::TaskResult> results;
  results.reserve(requests.size());
  for (const auto &request : requests) {
    try {
      results.push_back(run(request));
    } catch (const runtime::ProgressCanceled &) {
      throw;
    } catch (const std::exception &) {
      results.emplace_back();
    }
  }
  return results;
}

WhisperLoadedModel::WhisperLoadedModel(
    runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
    std::shared_ptr<const WhisperAssets> model_assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(model_assets)) {}

const runtime::ModelMetadata &WhisperLoadedModel::metadata() const noexcept {
  return metadata_;
}

const runtime::CapabilitySet &
WhisperLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
WhisperLoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  return std::make_unique<WhisperSession>(task, options, assets_);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_whisper_loader() {
  return std::make_shared<WhisperLoader>();
}

} // namespace engine::models::whisper
