#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/whisper/assets.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::whisper {

class WhisperRuntime;

// Offline ASR session over the native engine Whisper package (Phase 11 W2a).
class WhisperSession final : public runtime::RuntimeSessionBase,
                             public runtime::IOfflineVoiceTaskSession {
public:
  WhisperSession(runtime::TaskSpec task, runtime::SessionOptions options,
                 std::shared_ptr<const WhisperAssets> assets);
  ~WhisperSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

  std::vector<runtime::TaskResult>
  run_batch(const std::vector<runtime::TaskRequest> &requests) override;

private:
  runtime::TaskSpec task_;
  std::shared_ptr<const WhisperAssets> assets_;
  std::unique_ptr<WhisperRuntime> runtime_;
};

class WhisperLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  WhisperLoadedModel(runtime::ModelMetadata metadata,
                     runtime::CapabilitySet capabilities,
                     std::shared_ptr<const WhisperAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const WhisperAssets> assets_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_whisper_loader();

} // namespace engine::models::whisper
