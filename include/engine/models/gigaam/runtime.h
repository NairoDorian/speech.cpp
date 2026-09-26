#pragma once

#include "engine/framework/audio/mel_extractor.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/gigaam/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::gigaam {

// One emitted token with the arch's timing rule: the encoder frame it was
// emitted on, one frame (subsampling * hop = 40 ms) long.
struct GigaamToken {
  int32_t id = 0;
  std::string piece; // raw vocabulary piece (SentencePiece keeps its U+2581)
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
};

struct GigaamTranscription {
  std::string text;     // one leading space trimmed (the arch's full_text)
  std::string raw_text; // untrimmed decode (the arch's raw_text; TaskResult::raw_text)
  std::vector<GigaamToken> tokens;
  // Always false, honestly: greedy RNN-T / CTC has no token budget, and audio
  // past the ~25 s training window is decoded whole (warn-and-proceed, as the
  // arch does), never cut.
  bool truncated = false;
};

// Per-session inference state: backend weight store for the encoder, the
// framework MelExtractor configured to reproduce the arch's GigaAM frontend,
// and the host heads from the shared assets.
class GigaamRuntime {
public:
  GigaamRuntime(std::shared_ptr<const GigaamAssets> assets,
                core::ExecutionContext &execution_context,
                assets::TensorStorageType storage_type);
  ~GigaamRuntime();

  // One utterance. Polls `control` at the mel / encoder / decode boundaries
  // and at decode steps (ProgressCanceled unwinds). Throws
  // std::invalid_argument for input the arch answered INVALID_ARG to (empty
  // or shorter than one analysis window, malformed channel layout).
  GigaamTranscription transcribe(const runtime::AudioBuffer &audio,
                                 const runtime::RunControl &control);

  // The arch's run_batch: every utterance through ONE encoder dispatch
  // (batch on ne[2]; same-length batches mask-free, variable-length batches
  // zero-padded to T_max with per-utterance masks), then each utterance's
  // slice decoded on the host. Throws std::invalid_argument if any input is
  // unusable (the caller then falls back to per-utterance runs, as the arch).
  std::vector<GigaamTranscription>
  transcribe_batch(const std::vector<const runtime::AudioBuffer *> &audios,
                   const runtime::RunControl &control);

  // Mono PCM at the model rate; throws std::invalid_argument when unusable.
  std::vector<float> to_model_pcm(const runtime::AudioBuffer &audio) const;

  // The arch's GigaamMelFrontend::compute output: MEL-MAJOR [n_mels, n_frames]
  // with n_frames = (n_samples - win_length) / hop + 1 (center=False).
  std::vector<float> log_mel(const std::vector<float> &pcm, int &n_frames) const;

  int32_t sample_rate() const noexcept;

private:
  // Runs the encoder over `mel` ([n_batch][n_mels][T_mel], mel-major per
  // utterance, zero-padded along time) and returns the host copy of the
  // encoder output ([n_batch][T_enc][d_model]).
  std::vector<float> encode(const std::vector<float> &mel, int T_mel,
                            const std::vector<int> &n_frames, int &T_enc);
  GigaamTranscription decode_utterance(const float *encoded, int T_enc,
                                       const runtime::RunControl &control) const;
  void warn_if_past_window(size_t n_samples) const;

  std::shared_ptr<const GigaamAssets> assets_;
  core::ExecutionContext &execution_context_;
  assets::TensorStorageType storage_type_;
  ggml_backend_t backend_ = nullptr;
  std::shared_ptr<core::BackendWeightStore> store_;
  GigaamEncoderWeights weights_{};
  GigaamGraphPolicy policy_{};
  std::optional<audio::MelExtractor> mel_;
  // MelExtractor frame index of the arch's frame 0 (see runtime.cpp).
  int mel_frame_offset_ = 0;
};

} // namespace engine::models::gigaam
