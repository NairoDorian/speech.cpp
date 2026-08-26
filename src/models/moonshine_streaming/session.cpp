// engine/models/moonshine_streaming/session.cpp - streaming + offline ASR
// session and loader for the native engine Moonshine-Streaming package
// (Phase 11 Wave W1b).

#include "engine/models/moonshine_streaming/session.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/moonshine_streaming/runtime.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::moonshine_streaming {

namespace {

std::filesystem::path spec_path() {
  return engine::model_spec::default_spec_path("moonshine_streaming");
}

assets::TensorStorageType
weight_type_from_options(const runtime::SessionOptions &options) {
  const auto it = options.options.find("moonshine_streaming.weight_type");
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
    throw std::runtime_error(
        "moonshine_streaming.weight_type currently supports only "
        "native, f32, f16, bf16, and q8_0");
  }
}

int32_t int_option(const std::unordered_map<std::string, std::string> &options,
                   const char *key, int32_t fallback) {
  const auto it = options.find(key);
  if (it == options.end() || it->second.empty()) {
    return fallback;
  }
  try {
    return static_cast<int32_t>(std::stoll(it->second));
  } catch (...) {
    return fallback;
  }
}

int32_t max_tokens_from_request(
    const std::unordered_map<std::string, std::string> &options) {
  const int32_t parsed = int_option(options, "max_tokens", 0);
  return parsed > 0 ? parsed : 0;
}

class MoonshineStreamingLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "moonshine_streaming"; }

  std::vector<std::string> family_aliases() const override {
    return {"moonshine-streaming"};
  }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet result;
    result.supported_tasks = {{runtime::VoiceTaskKind::Asr,
                               {runtime::RunMode::Offline,
                                runtime::RunMode::Streaming}}};
    result.languages = {"en"};
    result.supports_timestamps = false;
    result.supports_cancellation = true;  // RunControl polled per decode step
    return result;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value() && *request.family_hint != family() &&
        *request.family_hint != "moonshine-streaming") {
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
    inspection.metadata.variant = "moonshine_streaming";
    inspection.metadata.description =
        "Moonshine-Streaming low-latency English ASR (sliding-window encoder "
        "over raw 16 kHz PCM, incremental cross-KV).";
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
        {"moonshine_streaming.min_decode_interval_ms", "ms",
         "Minimum gap between per-feed partial decodes; -1 uses the family "
         "default (240), 0 decodes on every advance.",
         false, "-1", "-1"},
    };
    inspection.cli.session_options = {
        {"moonshine_streaming.weight_type", "native|f32|f16|bf16|q8_0",
         "Shared model weight storage preference."},
    };
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    auto model_assets = load_moonshine_streaming_assets(request.model_path);
    runtime::ModelMetadata metadata;
    metadata.family = family();
    metadata.variant = model_assets->variant;
    metadata.description =
        "Moonshine-Streaming low-latency English ASR (sliding-window encoder "
        "over raw 16 kHz PCM, incremental cross-KV).";
    return std::make_unique<MoonshineStreamingLoadedModel>(
        std::move(metadata), advertised_capabilities(),
        std::move(model_assets));
  }
};

} // namespace

MoonshineStreamingSession::MoonshineStreamingSession(
    runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const MoonshineStreamingAssets> model_assets)
    : StreamingSessionBase(options), task_(std::move(task)),
      assets_(std::move(model_assets)),
      runtime_(std::make_unique<MoonshineStreamingRuntime>(
          assets_, execution_context(),
          weight_type_from_options(RuntimeSessionBase::options()))) {
  if (task_.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error(
        "Moonshine-Streaming only supports VoiceTaskKind::Asr");
  }
  if (task_.mode != runtime::RunMode::Offline &&
      task_.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "Moonshine-Streaming supports offline and streaming modes only");
  }
  // Partial decodes re-run the AR decoder from BOS over a growing cross-KV, so
  // early tokens can still move. STABLE_PREFIX is exactly the right policy:
  // the base commits only the prefix that reproduced across N consecutive
  // hypotheses, which is the same guarantee the arch derived by hand from its
  // token-id history.
  set_commit_policy(runtime::StreamCommitPolicy::StablePrefix, /*agreement_n=*/3);
}

MoonshineStreamingSession::~MoonshineStreamingSession() = default;

std::string MoonshineStreamingSession::family() const {
  return "moonshine_streaming";
}

runtime::VoiceTaskKind MoonshineStreamingSession::task_kind() const {
  return task_.task;
}

runtime::RunMode MoonshineStreamingSession::run_mode() const {
  return task_.mode;
}

void MoonshineStreamingSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error(
        "Moonshine-Streaming prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::StreamingPolicy MoonshineStreamingSession::streaming_policy() const {
  runtime::StreamingPolicy policy;
  policy.input = runtime::StreamingInputKind::AudioChunks;
  policy.output = runtime::StreamingOutputKind::PullEvents;
  // One encoder frame is 4 * frame_len samples (320 at frame_len=80). Feeding
  // whole encoder frames keeps the runtime's frame algebra exact without it
  // having to re-block; the runtime buffers regardless, so this is a hint.
  policy.preferred_audio_chunk_samples = 1280;
  return policy;
}

bool MoonshineStreamingSession::validate_chunk(
    const runtime::AudioChunk &chunk) const {
  // Pure pre-clear validation: reject a malformed chunk BEFORE the base
  // mutates any snapshot state.
  if (chunk.channels != 1) {
    return false;
  }
  if (chunk.sample_rate != 0 &&
      chunk.sample_rate != runtime_->sample_rate()) {
    return false;
  }
  return true;
}

void MoonshineStreamingSession::on_start_stream(
    const runtime::TaskRequest &request) {
  stream_max_tokens_ = max_tokens_from_request(request.options);
  stream_language_ =
      request.text_input.has_value() ? request.text_input->language : "";
  const int32_t min_decode_interval_ms = int_option(
      request.options, "moonshine_streaming.min_decode_interval_ms", -1);
  if (min_decode_interval_ms < -1) {
    throw std::runtime_error(
        "moonshine_streaming.min_decode_interval_ms must be >= -1");
  }
  runtime_->stream_begin(min_decode_interval_ms, stream_max_tokens_);
}

runtime::StreamEvent MoonshineStreamingSession::on_process_audio_chunk(
    const runtime::AudioChunk &chunk) {
  const bool changed = runtime_->stream_feed(
      chunk.samples.data(), chunk.samples.size(), run_control());

  runtime::StreamEvent event;
  if (changed) {
    // The base owns the committed/tentative split and the revision counter.
    update_text(runtime_->stream_text(), /*is_finalize=*/false);
    event.partial_text = runtime::Transcript{full_text(), stream_language_};
  }
  return event;
}

runtime::TaskResult MoonshineStreamingSession::on_finalize() {
  const auto transcription = runtime_->stream_finalize(run_control());
  update_text(transcription.text, /*is_finalize=*/true);

  runtime::TaskResult result;
  result.text_output = runtime::Transcript{full_text(), stream_language_};
  return result;
}

void MoonshineStreamingSession::on_reset() {
  runtime_->stream_reset();
  stream_max_tokens_ = 0;
  stream_language_.clear();
}

runtime::TaskResult
MoonshineStreamingSession::run(const runtime::TaskRequest &request) {
  require_prepared("Moonshine-Streaming run()");
  if (!request.audio_input.has_value()) {
    throw std::runtime_error(
        "Moonshine-Streaming run() requires audio_input");
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

std::vector<runtime::TaskResult> MoonshineStreamingSession::run_batch(
    const std::vector<runtime::TaskRequest> &requests) {
  std::vector<runtime::TaskResult> results;
  results.reserve(requests.size());
  for (const auto &request : requests) {
    // Per-utterance isolation: one failing utterance yields an empty result
    // rather than failing the batch; abort unwinds immediately.
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

MoonshineStreamingLoadedModel::MoonshineStreamingLoadedModel(
    runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
    std::shared_ptr<const MoonshineStreamingAssets> model_assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(model_assets)) {}

const runtime::ModelMetadata &
MoonshineStreamingLoadedModel::metadata() const noexcept {
  return metadata_;
}

const runtime::CapabilitySet &
MoonshineStreamingLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
MoonshineStreamingLoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  return std::make_unique<MoonshineStreamingSession>(task, options, assets_);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_streaming_loader() {
  return std::make_shared<MoonshineStreamingLoader>();
}

} // namespace engine::models::moonshine_streaming
