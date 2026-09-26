#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/granite_nar/assets.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::granite_nar {

class GraniteNarRuntime;

// Offline ASR session over the native engine Granite Speech NAR package
// (port of src/runtime/arch/granite_nar).
class GraniteNarSession final : public runtime::RuntimeSessionBase,
                                public runtime::IOfflineVoiceTaskSession {
public:
  GraniteNarSession(runtime::TaskSpec task, runtime::SessionOptions options,
                    std::shared_ptr<const GraniteNarAssets> assets);
  ~GraniteNarSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

  // The arch had no run_batch (the C ABI ran it serially): serial here too,
  // with a per-item status (InvalidArgument / InputTooLong / OutOfMemory /
  // Failed) instead of failing the batch. An abort unwinds the whole call.
  std::vector<runtime::TaskResult>
  run_batch(const std::vector<runtime::TaskRequest> &requests) override;

private:
  runtime::TaskSpec task_;
  std::shared_ptr<const GraniteNarAssets> assets_;
  std::unique_ptr<GraniteNarRuntime> runtime_;
};

class GraniteNarLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  GraniteNarLoadedModel(runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
                        std::shared_ptr<const GraniteNarAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

  // The arch's transcribe::Tokenizer encodes "gpt2" vocabs; so does the hub.
  std::optional<std::vector<int32_t>> tokenize(const std::string &text) const override;

  const GraniteNarAssets &model_assets() const noexcept { return *assets_; }

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const GraniteNarAssets> assets_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_granite_nar_loader();

} // namespace engine::models::granite_nar
