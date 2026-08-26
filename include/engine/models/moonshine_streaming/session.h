#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/streaming_session_base.h"
#include "engine/models/moonshine_streaming/assets.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::moonshine_streaming {

class MoonshineStreamingRuntime;

// Streaming (and offline) ASR session over the native engine
// Moonshine-Streaming package.
//
// Lifecycle, revision counting and the committed/tentative text split are
// owned by engine::runtime::StreamingSessionBase; this class only implements
// the model-specific hooks. Per the W1b design map we drive
// update_text(full_text) on every partial decode and let the base's
// STABLE_PREFIX policy decide the commit boundary, rather than re-deriving a
// longest-common-prefix boundary here - one commit policy for every family.
class MoonshineStreamingSession final : public runtime::StreamingSessionBase,
                                        public runtime::IOfflineVoiceTaskSession {
public:
  MoonshineStreamingSession(
      runtime::TaskSpec task, runtime::SessionOptions options,
      std::shared_ptr<const MoonshineStreamingAssets> assets);
  ~MoonshineStreamingSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;

  runtime::StreamingPolicy streaming_policy() const override;

  // Offline one-shot over the same graphs, so streamed-vs-offline divergence
  // stays measurable on one session.
  runtime::TaskResult run(const runtime::TaskRequest &request) override;
  std::vector<runtime::TaskResult>
  run_batch(const std::vector<runtime::TaskRequest> &requests) override;

protected:
  void on_start_stream(const runtime::TaskRequest &request) override;
  runtime::StreamEvent
  on_process_audio_chunk(const runtime::AudioChunk &chunk) override;
  runtime::TaskResult on_finalize() override;
  void on_reset() override;

  bool validate_chunk(const runtime::AudioChunk &chunk) const override;

private:
  runtime::TaskSpec task_;
  std::shared_ptr<const MoonshineStreamingAssets> assets_;
  std::unique_ptr<MoonshineStreamingRuntime> runtime_;
  int32_t stream_max_tokens_ = 0;
  std::string stream_language_;
};

// Loaded model handle returned by the moonshine_streaming loader.
class MoonshineStreamingLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  MoonshineStreamingLoadedModel(
      runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
      std::shared_ptr<const MoonshineStreamingAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const MoonshineStreamingAssets> assets_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_streaming_loader();

} // namespace engine::models::moonshine_streaming
