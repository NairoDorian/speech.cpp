#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/moonshine/assets.h"

#include <memory>
#include <string>

namespace engine::models::moonshine {

class MoonshineRuntime;

// Offline ASR session over the native engine Moonshine package.
class MoonshineSession final : public runtime::RuntimeSessionBase,
                               public runtime::IOfflineVoiceTaskSession {
public:
  MoonshineSession(runtime::TaskSpec task, runtime::SessionOptions options,
                   std::shared_ptr<const MoonshineAssets> assets);
  ~MoonshineSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

  // Batched offline execution: serial per-utterance decode honoring abort
  // polling between utterances; a failing utterance yields an empty result
  // instead of failing the whole batch (Arch::run_batch contract).
  std::vector<runtime::TaskResult>
  run_batch(const std::vector<runtime::TaskRequest> &requests) override;

private:
  runtime::TaskSpec task_;
  std::shared_ptr<const MoonshineAssets> assets_;
  std::unique_ptr<MoonshineRuntime> runtime_;
};

// Loaded model handle returned by the moonshine loader.
class MoonshineLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  MoonshineLoadedModel(runtime::ModelMetadata metadata,
                       runtime::CapabilitySet capabilities,
                       std::shared_ptr<const MoonshineAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const MoonshineAssets> assets_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_loader();

} // namespace engine::models::moonshine
