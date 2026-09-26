#pragma once

// engine/models/medasr/runtime.h - inference for the native engine MedASR
// package: the arch's run() and run_batch() (src/runtime/arch/medasr/model.cpp)
// over engine machinery (BackendWeightStore, MelExtractor, RunControl).

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/mel_extractor.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/medasr/assets.h"
#include "engine/models/medasr/decoding.h"
#include "engine/models/medasr/graphs_internal.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::medasr {

struct MedAsrTranscription {
  std::string text;               // leading space trimmed (the arch's full_text)
  std::string raw_text;           // pre-trim decode (the arch's raw_text)
  std::vector<CtcToken> tokens;   // the arch's TOKEN rows (40 ms per CTC frame)
  bool over_trained_window = false; // RoPE range exceeded (warned, processed)
};

// Per-session inference state. Every encoder graph is built, allocated and
// uploaded per call (as the arch does), so nothing is carried from one run to
// the next except the weights.
class MedAsrRuntime {
public:
  MedAsrRuntime(std::shared_ptr<const MedAsrAssets> assets,
                core::ExecutionContext &execution_context,
                assets::TensorStorageType storage_type);
  ~MedAsrRuntime();

  // Mono PCM at the model rate from an engine AudioBuffer. Throws
  // std::invalid_argument on empty audio.
  std::vector<float> prepare_pcm(const runtime::AudioBuffer &audio) const;

  // Checks what the arch's run() would reject (or crash on) before any
  // compute: shorter than one STFT window, or too short to leave one encoder
  // frame after the subsampling stem. Throws std::invalid_argument.
  void validate_length(size_t n_samples) const;

  // Single utterance (the arch's run()). Polls `control` at the top and
  // between stages.
  MedAsrTranscription transcribe(const std::vector<float> &pcm,
                                 const runtime::RunControl &control);

  // The arch's run_batch() fast path: one encoder dispatch over every
  // utterance (batch on ne[3]), per-utterance host decode. Same-length
  // batches run mask-free; variable-length ones pad to the longest and apply
  // the key-padding + valid-frame masks. Every input must already pass
  // validate_length(). Polls `control` per utterance decode.
  std::vector<MedAsrTranscription>
  transcribe_batch(const std::vector<std::vector<float>> &pcm,
                   const runtime::RunControl &control);

  int32_t sample_rate() const noexcept;

private:
  struct GraphRun {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;

    void free();
  };

  ggml_tensor *load_matrix(const std::string &name, const std::vector<int64_t> &shape);
  ggml_tensor *load_vector(const std::string &name, int64_t n);

  // Frame-major mel ([T][n_mels]) - the byte layout of the ggml
  // [n_mels, T_mel] input. MelExtractor itself is mel-major.
  std::vector<float> compute_mel(const std::vector<float> &pcm, int &n_frames) const;

  // Runs the encoder over `mel` (n_batch slabs of [T_mel][n_mels]) and
  // returns the logits, [n_batch][T_enc][vocab].
  std::vector<float> encode(const std::vector<float> &mel, int T_mel, int n_batch,
                            const std::vector<int> &real_tenc, bool var_len,
                            int &T_enc, int &vocab);

  MedAsrTranscription decode_utterance(const float *logits, int vocab,
                                       int n_frames) const;

  void warn_if_over_window(int enc_frames, int utterance) const;

  std::shared_ptr<const MedAsrAssets> assets_;
  core::ExecutionContext &execution_context_;
  assets::TensorStorageType storage_type_;
  ggml_backend_t backend_ = nullptr;
  std::shared_ptr<core::BackendWeightStore> store_;
  MedAsrWeights weights_{};
  MedAsrGraphPolicy policy_{};
  std::optional<audio::MelExtractor> mel_;
  GraphRun encoder_run_;
};

} // namespace engine::models::medasr
