#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/medasr/assets.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::medasr {

class MedAsrRuntime;

// Offline ASR session over the native engine MedASR package (greedy CTC).
class MedAsrSession final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession {
public:
  MedAsrSession(runtime::TaskSpec task, runtime::SessionOptions options,
                std::shared_ptr<const MedAsrAssets> assets);
  ~MedAsrSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

  // The arch's batched encoder (one dispatch over every utterance); falls back
  // to per-utterance runs when any request is malformed, with an empty result
  // in that request's slot.
  std::vector<runtime::TaskResult>
  run_batch(const std::vector<runtime::TaskRequest> &requests) override;

private:
  runtime::TaskSpec task_;
  std::shared_ptr<const MedAsrAssets> assets_;
  std::unique_ptr<MedAsrRuntime> runtime_;
};

class MedAsrLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  MedAsrLoadedModel(runtime::ModelMetadata metadata,
                    runtime::CapabilitySet capabilities,
                    std::shared_ptr<const MedAsrAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

  // No tokenize(): the arch's SentencePiece-BPE tokenizer has no encoder
  // (transcribe::Tokenizer::encode returns NOT_IMPLEMENTED for "bpe"), so the
  // default nullopt is the faithful answer.

  const MedAsrAssets &model_assets() const noexcept { return *assets_; }

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const MedAsrAssets> assets_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_medasr_loader();

} // namespace engine::models::medasr
