#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/moonshine/assets.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::moonshine {

struct MoonshineTranscription {
  std::string text;
  bool truncated = false;
};

// Per-session inference state: backend weight store + KV cache + graph
// scratch. One instance lives inside each MoonshineSession.
class MoonshineRuntime {
public:
  MoonshineRuntime(std::shared_ptr<const MoonshineAssets> assets,
                   core::ExecutionContext &execution_context,
                   assets::TensorStorageType storage_type);
  ~MoonshineRuntime();

  MoonshineTranscription transcribe(const runtime::AudioBuffer &audio,
                                    const runtime::RunControl &control,
                                    int32_t max_tokens);

private:
  struct GraphRun {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;

    void free();
  };

  std::shared_ptr<const MoonshineAssets> assets_;
  core::ExecutionContext &execution_context_;
  ggml_backend_t backend_ = nullptr;
  std::shared_ptr<core::BackendWeightStore> store_;
  MoonshineWeights weights_{};
  MoonshineKvCache kv_cache_;

  // Host-side mirror of the encoder output (re-uploaded into the cross-KV
  // graph, which lives in a separate compute context).
  std::vector<float> enc_host_;
  int enc_T_ = 0;

  GraphRun encoder_run_;
  GraphRun cross_kv_run_;
  GraphRun step_run_;
};

} // namespace engine::models::moonshine
