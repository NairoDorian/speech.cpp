#pragma once

// engine/models/canary_qwen/session.h - offline ASR session and loader of the
// native engine Canary-Qwen package (NVIDIA Canary-Qwen-2.5B: NeMo SALM,
// FastConformer encoder + Qwen3-1.7B decoder), ported from
// src/runtime/arch/canary_qwen.

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/canary_qwen/assets.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::canary_qwen {

class CanaryQwenRuntime;

class CanaryQwenSession final : public runtime::RuntimeSessionBase,
                                public runtime::IOfflineVoiceTaskSession {
public:
  CanaryQwenSession(runtime::TaskSpec task, runtime::SessionOptions options,
                    std::shared_ptr<const CanaryQwenAssets> assets);
  ~CanaryQwenSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

  // The arch's transcribe_run_batch: lockstep batched decode. A rejected
  // utterance (too short, too long) yields an empty TaskResult (no
  // text_output); an abort throws ProgressCanceled for the whole batch.
  std::vector<runtime::TaskResult>
  run_batch(const std::vector<runtime::TaskRequest> &requests) override;

private:
  runtime::TaskSpec task_;
  std::shared_ptr<const CanaryQwenAssets> assets_;
  std::unique_ptr<CanaryQwenRuntime> runtime_;
};

class CanaryQwenLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  CanaryQwenLoadedModel(runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
                        std::shared_ptr<const CanaryQwenAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

  // HF-exact Qwen2 BPE ids (TokenizerHub), for transcribe_tokenize.
  std::optional<std::vector<int32_t>> tokenize(const std::string &text) const override;

  const CanaryQwenAssets &assets() const noexcept { return *assets_; }

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const CanaryQwenAssets> assets_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_canary_qwen_loader();

} // namespace engine::models::canary_qwen
