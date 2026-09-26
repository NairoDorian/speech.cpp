#pragma once

// engine/models/voxtral/runtime.h - per-session inference state of the native
// engine Voxtral (2507, offline) package: the backend weight store, the packed
// gate+up tensors, the ported Whisper mel frontend, the decoder KV cache and
// the greedy audio-LLM decode (the arch's run(), src/runtime/arch/voxtral/
// model.cpp).

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/voxtral/assets.h"
#include "engine/models/voxtral/graphs.h"
#include "engine/models/voxtral/mel_frontend.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::voxtral {

struct VoxtralRuntimeOptions {
  // KV-cache dtype: the arch's AUTO / F16 -> F16, F32 on request.
  ggml_type kv_type = GGML_TYPE_F16;
  // Session context knob (transcribe_session_params.n_ctx): lowers, never
  // raises, the decoder context ceiling. 0 = the model's maximum.
  int n_ctx = 0;
  // The arch's flash defaults (arch init_context): encoder on (Whisper
  // head_dim 64, unmasked) and decoder on (Llama head_dim 128). Both follow
  // TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH exactly as the arch did.
  bool encoder_use_flash = true;
  bool decoder_use_flash = true;
};

// transcribe::flash::apply_env_overrides: TRANSCRIBE_NO_FLASH clears both,
// TRANSCRIBE_FORCE_FLASH sets both (FORCE wins when both are set). A variable
// counts as set when it is non-empty and does not start with '0'.
void apply_flash_env_overrides(VoxtralRuntimeOptions &options);

// One utterance's prompt mode (the arch's run() parameters).
struct VoxtralRequest {
  // Spoken-language hint; "" = none (auto). Adds "lang:<code>" to the
  // transcription prompt, exactly as the arch did with params->language.
  std::string language;
  // TRANSCRIBE_TASK_TRANSLATE: the instruct template with the synthesized
  // "Translate this to {Language}." instruction.
  bool translate = false;
  // BCP-47 target for translate ("" -> English, the arch's lang_name_for).
  std::string target_language;
};

struct VoxtralTranscription {
  std::string text;      // whitespace-trimmed decode (the arch's full_text)
  std::string raw_text;  // pre-trim decode (the arch's raw_text)
  std::vector<int32_t> tokens;  // generated ids, trailing EOS dropped
  bool truncated = false;       // stopped at the budget / KV window before EOS
};

class VoxtralRuntime {
public:
  VoxtralRuntime(std::shared_ptr<const VoxtralAssets> assets, core::ExecutionContext &execution_context,
                 VoxtralRuntimeOptions options);
  ~VoxtralRuntime();

  VoxtralRuntime(const VoxtralRuntime &) = delete;
  VoxtralRuntime &operator=(const VoxtralRuntime &) = delete;

  // Mono PCM at the model rate from an engine AudioBuffer. Throws
  // std::invalid_argument on empty or malformed audio.
  std::vector<float> prepare_pcm(const runtime::AudioBuffer &audio) const;

  // One utterance (the arch's run()). `pcm` is mono at sample_rate().
  // Polls `control` (the cancellation point) at the top, per 30 s encoder
  // chunk, per prefill chunk and per decode step: an abort throws
  // runtime::ProgressCanceled. Throws std::invalid_argument where the arch
  // returned INVALID_ARG / INPUT_TOO_LONG, runtime::CapacityError where it
  // returned OOM (graph / KV allocation), std::runtime_error on backend
  // failures. Reaching the generation budget is NOT an error: the partial
  // transcript comes back with truncated = true.
  VoxtralTranscription transcribe(const std::vector<float> &pcm, const VoxtralRequest &request,
                                  const runtime::RunControl &control);

  // The arch's build_transcription_prompt / build_instruct_prompt: returns the
  // prompt ids with n_audio [AUDIO] placeholders at [prefix_len, prefix_len +
  // n_audio). Exposed for tests.
  std::vector<int32_t> build_prompt(const VoxtralRequest &request, int n_audio, int &prefix_len,
                                    int &suffix_len) const;

  int32_t sample_rate() const noexcept;
  const VoxtralRuntimeOptions &options() const noexcept { return options_; }

private:
  ggml_tensor *load_matrix(const std::string &name, const std::vector<int64_t> &shape);
  ggml_tensor *load_vector(const std::string &name, int64_t n);
  void load_weights();
  void pack_gate_up();

  int n_threads() const;
  void compute(ggml_cgraph *graph, const char *what);

  // Encoder + projector over every 30 s chunk of the (padded) mel; returns the
  // audio embeddings [n_chunks * tokens_per_chunk][hidden] (ggml [hidden, N]).
  std::vector<float> encode(const std::vector<float> &mel, int mel_n_frames, int n_chunks,
                            const runtime::RunControl &control);

  // Single-shot or chunked prefill; returns the last position's logits.
  std::vector<float> prefill(const std::vector<int32_t> &prompt_ids, const std::vector<float> &enc_host,
                             int prefix_len, int suffix_len, int n_audio, const runtime::RunControl &control);

  std::shared_ptr<const VoxtralAssets> assets_;
  core::ExecutionContext &execution_context_;
  VoxtralRuntimeOptions options_;
  ggml_backend_t backend_ = nullptr;

  std::shared_ptr<core::BackendWeightStore> store_;
  VoxtralWeights weights_{};
  ggml_context *packed_ctx_ = nullptr;  // packed gate+up tensors (own buffer)
  ggml_backend_buffer_t packed_buffer_ = nullptr;

  std::optional<WhisperMelFrontend> mel_;
  causal_lm::KvCache kv_cache_;
};

}  // namespace engine::models::voxtral
