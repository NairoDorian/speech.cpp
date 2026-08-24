// engine/models/moonshine/session.cpp - offline ASR session + loader for the
// native engine Moonshine package (Phase 11 W1).

#include "engine/models/moonshine/session.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/moonshine/runtime.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::moonshine {

namespace {

std::filesystem::path spec_path() {
  return engine::model_spec::default_spec_path("moonshine");
}

assets::TensorStorageType
weight_type_from_options(const runtime::SessionOptions &options) {
  const auto it = options.options.find("moonshine.weight_type");
  if (it == options.options.end()) {
    return assets::TensorStorageType::Native;
  }
  const auto storage_type = assets::parse_tensor_storage_type(it->second);
  switch (storage_type) {
  case assets::TensorStorageType::Native:
  case assets::TensorStorageType::F32:
  case assets::TensorStorageType::F16:
  case assets::TensorStorageType::BF16:
  case assets::TensorStorageType::Q8_0:
    return storage_type;
  default:
    throw std::runtime_error("moonshine.weight_type currently supports only "
                             "native, f32, f16, bf16, and q8_0");
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

class MoonshineLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "moonshine"; }

  std::vector<std::string> family_aliases() const override {
    return {"moonshine-offline"};
  }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet result;
    result.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
    result.languages = {"en"};
    result.supports_timestamps = false;
    return result;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value() && *request.family_hint != family() &&
        *request.family_hint != "moonshine-offline") {
      return false;
    }
    try {
      (void)engine::model_spec::load_resource_bundle(request.model_path,
                                                     spec_path());
      return true;
    } catch (...) {
      return false;
    }
  }

  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    const auto resources = engine::model_spec::load_resource_bundle(
        request.model_path, spec_path());
    runtime::ModelInspection inspection;
    inspection.model_root = resources.model_root();
    inspection.metadata.family = family();
    inspection.metadata.variant = "moonshine";
    inspection.metadata.description = "Moonshine lightweight English ASR "
                                      "(encoder-decoder over raw 16 kHz PCM).";
    inspection.capabilities = advertised_capabilities();
    inspection.discovered_configs =
        runtime::discover_named_assets_from_package_spec(
            request.model_path, spec_path(),
            engine::model_spec::ResourceKind::Files);
    inspection.discovered_weights =
        runtime::discover_named_assets_from_package_spec(
            request.model_path, spec_path(),
            engine::model_spec::ResourceKind::Tensors);
    inspection.cli.request_options = {
        {"max_tokens", "n", "Maximum generated transcript tokens.", false, "0",
         "0"},
    };
    inspection.cli.session_options = {
        {"moonshine.weight_type", "native|f32|f16|bf16|q8_0",
         "Shared model weight storage preference."},
    };
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    const auto assets = load_moonshine_assets(request.model_path);
    runtime::ModelMetadata metadata;
    metadata.family = family();
    metadata.variant = assets->variant;
    metadata.description = "Moonshine lightweight English ASR (encoder-decoder "
                           "over raw 16 kHz PCM).";
    return std::make_unique<MoonshineLoadedModel>(
        std::move(metadata), advertised_capabilities(), std::move(assets));
  }
};

} // namespace

MoonshineSession::MoonshineSession(
    runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const MoonshineAssets> assets)
    : RuntimeSessionBase(std::move(options)), task_(std::move(task)),
      assets_(std::move(assets)),
      runtime_(std::make_unique<MoonshineRuntime>(
          assets_, execution_context(),
          weight_type_from_options(RuntimeSessionBase::options()))) {
  if (task_.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("Moonshine only supports VoiceTaskKind::Asr");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error(
        "Moonshine only supports offline mode in this package "
        "(streaming lives in the moonshine_streaming family)");
  }
}

MoonshineSession::~MoonshineSession() = default;

std::string MoonshineSession::family() const { return "moonshine"; }

runtime::VoiceTaskKind MoonshineSession::task_kind() const {
  return task_.task;
}

runtime::RunMode MoonshineSession::run_mode() const { return task_.mode; }

void MoonshineSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error("Moonshine prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::TaskResult MoonshineSession::run(const runtime::TaskRequest &request) {
  require_prepared("Moonshine run()");
  if (!request.audio_input.has_value()) {
    throw std::runtime_error("Moonshine run() requires audio_input");
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
MoonshineSession::run_batch(const std::vector<runtime::TaskRequest> &requests) {
  std::vector<runtime::TaskResult> results;
  results.reserve(requests.size());
  for (const auto &request : requests) {
    // Per-utterance isolation: one failing utterance yields an empty
    // result rather than failing the batch; abort unwinds immediately.
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

MoonshineLoadedModel::MoonshineLoadedModel(
    runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
    std::shared_ptr<const MoonshineAssets> assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata &MoonshineLoadedModel::metadata() const noexcept {
  return metadata_;
}

const runtime::CapabilitySet &
MoonshineLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
MoonshineLoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  return std::make_unique<MoonshineSession>(task, options, assets_);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_loader() {
  return std::make_shared<MoonshineLoader>();
}

} // namespace engine::models::moonshine
