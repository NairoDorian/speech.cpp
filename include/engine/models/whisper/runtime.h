#pragma once

#include "engine/framework/audio/mel_extractor.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/whisper/assets.h"
#include "engine/models/whisper/decoding.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::whisper {

struct WhisperTranscription {
  std::string text;
  std::string language;  // ISO code of the prompt's <|lang|> ("" for .en models)
  bool language_detected = false;
  struct Segment {
    int64_t t0_ms = 0;
    int64_t t1_ms = 0;
    std::string text;
  };
  std::vector<Segment> segments;                 // when timestamps were requested
  std::vector<runtime::DecodeTelemetry> traces;  // one per decoded window
  bool truncated = false;
};

// Per-session inference state: backend weight store + KV cache + graph
// scratch + the unified Phase-9 mel frontend.
//
// Since W2b the decode is the full Whisper recipe (decoding.h): any input
// length (the unified HF seek loop over 30 s windows - long-form is no longer
// truncated), segment timestamps, language detection / translate, prompts,
// the temperature-fallback ladder and its telemetry. This class only supplies
// the model to that policy.
class WhisperRuntime {
public:
  // `kv_type`: the KV-cache dtype; nullopt = F32. GGML_TYPE_COUNT selects the
  // arch / whisper.cpp AUTO policy (F16 cache for F16 / quantized decoders).
  WhisperRuntime(std::shared_ptr<const WhisperAssets> assets,
                 core::ExecutionContext &execution_context,
                 assets::TensorStorageType storage_type,
                 std::optional<ggml_type> kv_type = std::nullopt);
  ~WhisperRuntime();

  // `partial`, when given, receives the windows completed before an abort
  // (the run then rethrows the ProgressCanceled).
  WhisperTranscription transcribe(const runtime::AudioBuffer &audio,
                                  const runtime::RunControl &control,
                                  const WhisperDecodeOptions &options,
                                  WhisperTranscription *partial = nullptr);

  // The checkpoint's special-token contract, as the policy sees it.
  WhisperTokenContract token_contract() const;

  int32_t sample_rate() const noexcept;

private:
  struct GraphRun {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;

    void free();
  };

  // One weight, by its legacy whisper.cpp name, validated against the logical
  // (row-major) shape the graphs expect. A .bin conv bias stored [C, 1] is
  // loaded as [C].
  core::TensorValue load_weight(const std::string &legacy_name,
                                const std::vector<int64_t> &shape);

  void encode_window(const std::vector<float> &mel, int total_frames, int seek,
                     int n_frames);
  void decode_prompt(const std::vector<int32_t> &tokens, int n_past, int sot_row,
                     std::vector<float> *sot_logits, std::vector<float> &last_logits);
  // One greedy/beam step through the cached static step graph (W2b.3).
  void decode_static_step(int32_t token, int n_past, std::vector<float> &logits);
  // Drop the cached step graph; required whenever kv_cache_ is (re)allocated
  // because the graph holds views of the KV tensors.
  void release_static_step();

  std::shared_ptr<const WhisperAssets> assets_;
  core::ExecutionContext &execution_context_;
  assets::TensorStorageType storage_type_;
  ggml_backend_t backend_ = nullptr;
  std::shared_ptr<core::BackendWeightStore> store_;
  WhisperWeights weights_{};
  WhisperKvCache kv_cache_;
  std::optional<audio::MelExtractor> mel_;
  int T_enc_ = 0;
  ggml_type kv_type_ = GGML_TYPE_F32;

  GraphRun encoder_run_;
  GraphRun cross_kv_run_;
  GraphRun step_run_;

  // W2b.3: the single-token decoder step graph is built once per KV cache
  // (build_decoder_step_graph) and reused for every token, instead of
  // ggml_init + graph build + gallocr per token. Set
  // SPEECHCPP_WHISPER_STATIC_STEP=0 to fall back to the per-token graph
  // (parity debugging).
  bool static_step_enabled_ = true;
  GraphRun static_step_run_;
  DecoderBuild static_step_{};
  core::HostGraphPlan static_step_plan_;
  std::vector<float> static_step_mask_;
};

} // namespace engine::models::whisper
