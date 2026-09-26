#pragma once

// engine/models/voxtral/model.h - public entry points of the native engine
// Voxtral (2507, offline) package: the loader the registry instantiates, the
// loaded model and the offline ASR session.
//
// Port of the transcribe.cpp `voxtral` arch (src/runtime/arch/voxtral/). The
// package reads exactly the arch's GGUF (general.architecture = "voxtral");
// see assets.h for the file contract, graphs.h for the numerics contract and
// runtime.h for the decode.

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/voxtral/assets.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::voxtral {

class VoxtralRuntime;

// Offline ASR session (greedy audio-LLM decode). Request options:
//   language         "" / "auto" = no hint; otherwise one of the GGUF's
//                    general.languages, added to the prompt as "lang:<code>"
//   task             transcribe (default) | translate
//   target_language  translate target (default English), one of
//                    stt.translation.target_languages when the GGUF lists them
//   timestamps       none | 0 | auto | 1 (the model has no timestamps)
// Session options (plain or "voxtral."-prefixed, as the C ABI adapter sends):
//   kv_type          f16 (default, the arch's AUTO) | f32 | auto
//   n_ctx            lowers the decoder context ceiling; 0 = model maximum
class VoxtralSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
  VoxtralSession(const runtime::TaskSpec &task, const runtime::SessionOptions &options,
                 std::shared_ptr<const VoxtralAssets> assets,
                 std::shared_ptr<const model_spec::ModelContract> contract);
  ~VoxtralSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;
  // run_batch: the IOfflineVoiceTaskSession default (serial run() per
  // request), which is the arch's run_batch_serial numerics.

private:
  std::shared_ptr<const VoxtralAssets> assets_;
  std::shared_ptr<const model_spec::ModelContract> contract_;
  std::unique_ptr<VoxtralRuntime> runtime_;
};

class VoxtralLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  VoxtralLoadedModel(runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
                     std::shared_ptr<const VoxtralAssets> assets,
                     std::shared_ptr<const model_spec::ModelContract> contract);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
      const runtime::TaskSpec &task, const runtime::SessionOptions &options) const override;

  // transcribe_tokenize(): the arch's Tokenizer::encode (Qwen2 pretokenizer
  // fallback + merge-rank BPE); nullopt when encode fails.
  std::optional<std::vector<int32_t>> tokenize(const std::string &text) const override;

  const VoxtralAssets &model_assets() const noexcept { return *assets_; }

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const VoxtralAssets> assets_;
  std::shared_ptr<const model_spec::ModelContract> contract_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_voxtral_loader();

}  // namespace engine::models::voxtral
