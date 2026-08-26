#pragma once

#include "engine/framework/audio/mel_extractor.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/whisper/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::whisper {

struct WhisperTranscription {
  std::string text;
  bool truncated = false;
};

// Per-session inference state: backend weight store + KV cache + graph
// scratch + the unified Phase-9 mel frontend.
//
// W2a runs a single 30 s window: PCM is padded/trimmed to fe_n_samples so the
// encoder always sees the 3000-frame geometry Whisper was trained on. Audio
// longer than the window is truncated with `truncated = true`; long-form seek
// continuation is W2b.
class WhisperRuntime {
public:
  WhisperRuntime(std::shared_ptr<const WhisperAssets> assets,
                 core::ExecutionContext &execution_context,
                 assets::TensorStorageType storage_type);
  ~WhisperRuntime();

  WhisperTranscription transcribe(const runtime::AudioBuffer &audio,
                                  const runtime::RunControl &control,
                                  int32_t max_tokens);

  int32_t sample_rate() const noexcept;

private:
  struct GraphRun {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;

    void free();
  };

  // Streams one legacy-named tensor's payload out of the `.bin` and hands it
  // to the weight store with the file's own ggml type.
  ggml_tensor *load_bin_tensor(const char *legacy_name, bool squeeze_leading);

  std::string decode_token_ids(const std::vector<int32_t> &ids) const;

  std::shared_ptr<const WhisperAssets> assets_;
  core::ExecutionContext &execution_context_;
  ggml_backend_t backend_ = nullptr;
  std::shared_ptr<core::BackendWeightStore> store_;
  WhisperWeights weights_{};
  WhisperKvCache kv_cache_;
  std::optional<audio::MelExtractor> mel_;

  GraphRun encoder_run_;
  GraphRun cross_kv_run_;
  GraphRun step_run_;

  // Suppression masks, resolved once from hparams.
  std::vector<int32_t> suppress_tokens_;
  std::vector<int32_t> begin_suppress_tokens_;
};

} // namespace engine::models::whisper
