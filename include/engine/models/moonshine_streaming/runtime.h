#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/moonshine_streaming/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::moonshine_streaming {

struct MoonshineStreamingTranscription {
  std::string text;
  bool truncated = false;
};

// Per-session inference state: backend weight store + KV cache + graph
// scratch + the incremental streaming buffers. One instance lives inside each
// MoonshineStreamingSession.
//
// Streaming model (ported from src/runtime/arch/moonshine_streaming):
// each feed extends host-side committed buffers in lockstep -
//   adapter_committed_       post-adapter encoder hidden [dec_d_model, T]
//   cross_k/v_committed_[il] per decoder layer,          [dec_d_model, T]
// which are uploaded into the persistent KV cache on each partial decode and
// again at finalize.
//
// Per-feed slicing is numerically equivalent to a one-shot pass because the
// encoder is ergodic (no positional encoding) with per-layer sliding-window
// attention over a causal-conv frontend - output frame t depends only on
// conv-stack frames [t - L_total, t + R_total] - the adapter pos_emb is an
// ABSOLUTE-frame get_rows, and the cross-KV projection is per-frame linear.
class MoonshineStreamingRuntime {
public:
  MoonshineStreamingRuntime(
      std::shared_ptr<const MoonshineStreamingAssets> assets,
      core::ExecutionContext &execution_context,
      assets::TensorStorageType storage_type);
  ~MoonshineStreamingRuntime();

  // ---- offline one-shot (the run() path) ----
  MoonshineStreamingTranscription transcribe(const runtime::AudioBuffer &audio,
                                             const runtime::RunControl &control,
                                             int32_t max_tokens);

  // ---- streaming ----
  // min_decode_interval_ms < 0 selects the family default (240 ms).
  void stream_begin(int32_t min_decode_interval_ms, int32_t max_tokens);

  // Appends PCM and, when enough stable frames have accumulated and the
  // throttle allows, runs a partial decode. Returns true when the transcript
  // changed. The audio must already be mono at the model sample rate.
  bool stream_feed(const float *pcm, size_t n_samples,
                   const runtime::RunControl &control);

  // Flushes the trailing tail and produces the final transcript.
  MoonshineStreamingTranscription
  stream_finalize(const runtime::RunControl &control);

  void stream_reset();

  const std::string &stream_text() const noexcept { return stream_text_; }
  int32_t stream_frames_emitted() const noexcept { return T_emitted_; }
  int32_t sample_rate() const noexcept;

private:
  struct GraphRun {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;

    void free();
  };

  // ---- incremental pipeline steps ----
  void encode_window_to_host(const float *pcm, int n_samples,
                             std::vector<float> &out_enc_host,
                             int &out_T_enc);
  void apply_adapter_window(const float *enc_data, int n_frames,
                            int abs_frame_offset,
                            std::vector<float> &out_adapter);
  void project_cross_kv_window(const float *adapter_data, int n_frames,
                               std::vector<std::vector<float>> &k_out,
                               std::vector<std::vector<float>> &v_out);
  void flush_stable_frames(int win_start, int win_end, int emit_start,
                           int emit_end);
  void trim_pcm_buffer();

  void ensure_kv_cache_for_T(int T_enc);
  void commit_cross_kv_from_host(int T_enc);
  MoonshineStreamingTranscription
  decode_from_kv_cache(int T_enc, const runtime::RunControl &control,
                       int32_t max_tokens);

  // Runs a partial decode over the currently committed frames and updates
  // stream_text_ / stream_token_ids_. Returns true when the text changed.
  bool decode_partial(const runtime::RunControl &control);

  std::string decode_token_ids(const std::vector<int32_t> &ids) const;

  std::shared_ptr<const MoonshineStreamingAssets> assets_;
  core::ExecutionContext &execution_context_;
  ggml_backend_t backend_ = nullptr;
  std::shared_ptr<core::BackendWeightStore> store_;
  MoonshineStreamingWeights weights_{};
  MoonshineStreamingKvCache kv_cache_;

  GraphRun encoder_run_;
  GraphRun adapter_run_;
  GraphRun cross_kv_proj_run_;
  GraphRun cross_kv_commit_run_;
  GraphRun step_run_;

  // ---- streaming state ----
  std::vector<float> pcm_buffer_;
  int64_t pcm_start_sample_ = 0;
  std::vector<float> adapter_committed_;
  std::vector<std::vector<float>> cross_k_committed_;
  std::vector<std::vector<float>> cross_v_committed_;
  int32_t T_emitted_ = 0;
  int32_t last_decoded_T_ = 0;

  // Geometry, resolved at stream_begin (constant per stream).
  int32_t L_total_frames_ = 0;
  int32_t R_total_frames_ = 0;
  int32_t frontend_pad_frames_ = 0;
  int32_t samples_per_enc_frame_ = 0;
  int32_t min_decode_frames_ = 0;
  int32_t stream_max_tokens_ = 0;

  std::string stream_text_;
  std::vector<int32_t> stream_token_ids_;
  bool stream_truncated_ = false;
  bool streaming_ = false;
};

} // namespace engine::models::moonshine_streaming
