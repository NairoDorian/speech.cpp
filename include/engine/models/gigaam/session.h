#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/gigaam/assets.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::gigaam {

class GigaamRuntime;

// Offline ASR session over the native engine GigaAM package (the port of
// src/runtime/arch/gigaam/). Russian-only, greedy RNN-T or CTC per variant.
class GigaamSession final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession {
public:
  GigaamSession(runtime::TaskSpec task, runtime::SessionOptions options,
                std::shared_ptr<const GigaamAssets> assets);
  ~GigaamSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

  // The arch's run_batch: one shared encoder dispatch when every request is
  // usable, per-utterance runs otherwise (a failed utterance yields an empty
  // result, the rest still decode).
  std::vector<runtime::TaskResult>
  run_batch(const std::vector<runtime::TaskRequest> &requests) override;

private:
  runtime::TaskSpec task_;
  std::shared_ptr<const GigaamAssets> assets_;
  std::unique_ptr<GigaamRuntime> runtime_;
};

class GigaamLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  GigaamLoadedModel(runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
                    std::shared_ptr<const GigaamAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

  // Charwise variants only (the arch's raw-bytes encoder); nullopt for the
  // SentencePiece variants, whose arch tokenizer has no encoder either.
  std::optional<std::vector<int32_t>> tokenize(const std::string &text) const override;

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const GigaamAssets> assets_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_gigaam_loader();

} // namespace engine::models::gigaam
