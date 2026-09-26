#pragma once

// engine/models/canary_qwen/runtime.h - per-session inference state of the
// native engine Canary-Qwen package: the backend weight store, the ported
// mel frontend, and the greedy SALM decode (the arch's run() and
// run_batch()).

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/models/canary_qwen/assets.h"
#include "engine/models/canary_qwen/graphs.h"
#include "engine/models/canary_qwen/mel_frontend.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::canary_qwen {

struct CanaryQwenRuntimeOptions {
  // KV-cache dtype: the arch's AUTO / F16 -> F16, F32 on request.
  ggml_type kv_type = GGML_TYPE_F16;
  // Session context knob (transcribe_session_params.n_ctx): lowers, never
  // raises, the decoder context ceiling. 0 = the model's maximum.
  int n_ctx = 0;
  // The arch's flash defaults (encoder off, decoder on); both follow
  // TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH exactly as the arch did.
  bool encoder_use_flash = false;
  bool decoder_use_flash = true;
};

// Applies the arch's flash env overrides to `options` (transcribe::flash::
// apply_env_overrides): NO_FLASH clears both, FORCE_FLASH sets both.
void apply_flash_env_overrides(CanaryQwenRuntimeOptions &options);

struct CanaryQwenTranscription {
  std::string text;          // tokenizer decode of the generated ids (EOS dropped)
  bool truncated = false;    // stopped at the budget / KV width before EOS
  int n_generated_tokens = 0;
};

// One utterance of a batch: a transcription, or why it was rejected.
struct CanaryQwenBatchItem {
  bool ok = false;
  bool input_too_long = false;  // the arch's per-utterance INPUT_TOO_LONG
  std::string error;            // the rejection message when !ok
  CanaryQwenTranscription result;
};

class CanaryQwenRuntime {
public:
  CanaryQwenRuntime(std::shared_ptr<const CanaryQwenAssets> assets,
                    core::ExecutionContext &execution_context, CanaryQwenRuntimeOptions options);
  ~CanaryQwenRuntime();

  CanaryQwenRuntime(const CanaryQwenRuntime &) = delete;
  CanaryQwenRuntime &operator=(const CanaryQwenRuntime &) = delete;

  // One utterance (the arch's run()). `pcm` is 16 kHz mono. Throws
  // std::invalid_argument where the arch returned INVALID_ARG /
  // INPUT_TOO_LONG, runtime::ProgressCanceled on abort (with `partial`, when
  // given, holding the tokens decoded so far), std::runtime_error on backend
  // failures.
  CanaryQwenTranscription transcribe(const std::vector<float> &pcm,
                                     const runtime::RunControl &control,
                                     CanaryQwenTranscription *partial = nullptr);

  // The arch's run_batch(): lockstep batched prefill + step decode when the
  // decoder runs flash attention and there is more than one utterance,
  // otherwise serial transcribe() per utterance. Per-utterance rejections
  // come back as items; an abort throws ProgressCanceled for the batch.
  std::vector<CanaryQwenBatchItem> transcribe_batch(const std::vector<const std::vector<float> *> &pcms,
                                                    const runtime::RunControl &control);

  int32_t sample_rate() const noexcept;

private:
  struct GraphRun {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;

    GraphRun() = default;
    GraphRun(const GraphRun &) = delete;
    GraphRun &operator=(const GraphRun &) = delete;
    ~GraphRun() { free(); }
    void free();
    void init(size_t mem_bytes, const char *what);
    void allocate(ggml_backend_t backend, ggml_cgraph *g, const char *what);
  };

  void load_weights();
  ggml_tensor *load(const std::string &name, std::initializer_list<int64_t> ne, int kind);
  void fuse_batch_norm();
  void pack_gate_up();

  int n_threads() const;
  void compute(ggml_cgraph *graph, const char *what);

  // mel + FastConformer + perception projection -> [hidden, T_enc] host.
  void encode(const std::vector<float> &mel, int n_frames, std::vector<float> &enc_host,
              int &T_enc);
  bool compute_mel(const std::vector<float> &pcm, std::vector<float> &mel, int &n_frames,
                   int threads) const;
  std::vector<int32_t> build_prompt(int T_enc) const;

  std::shared_ptr<const CanaryQwenAssets> assets_;
  core::ExecutionContext &execution_context_;
  CanaryQwenRuntimeOptions options_;
  ggml_backend_t backend_ = nullptr;
  std::string backend_name_;
  bool cpu_backend_ = true;

  std::shared_ptr<core::BackendWeightStore> store_;
  CanaryQwenWeights weights_{};
  ggml_context *packed_ctx_ = nullptr;          // packed gate+up tensors
  ggml_backend_buffer_t packed_buffer_ = nullptr;

  std::optional<NemoMelFrontend> mel_;
  causal_lm::KvCache kv_cache_;
  causal_lm::KvCache kv_cache_batch_;
};

} // namespace engine::models::canary_qwen
