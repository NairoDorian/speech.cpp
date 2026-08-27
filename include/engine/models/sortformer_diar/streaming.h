#pragma once

// Chunked (streaming-core) diarization for Sortformer: the Arrival-Order
// Speaker Cache (AOSC) + FIFO scheduler NeMo's SortformerModules runs the
// streaming checkpoints with, ported host-for-host from
// src/runtime/arch/sortformer/stream.cpp (Phase 10.5, family 3).
//
// The scheduler is graph-agnostic: it consumes pre-encode embeddings and
// per-frame speaker probabilities through callbacks, so the session owns the
// ggml graphs and this file owns the exactness-critical bookkeeping
// (streaming_update, _get_silence_profile, _compress_spkcache).

#include "engine/models/sortformer_diar/assets.h"
#include "engine/models/sortformer_diar/types.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::models::sortformer_diar {

// One streaming operating point. All lengths are in diarization frames
// (80 ms at the shipped 8x subsampling of a 10 ms hop).
struct SortformerStreamParams {
    int64_t chunk_len = 188;
    int64_t chunk_left_context = 1;
    int64_t chunk_right_context = 1;
    int64_t fifo_len = 0;
    int64_t spkcache_len = 188;
    int64_t spkcache_update_period = 188;
    SortformerScoringConfig scoring;
    int64_t max_index = 99999;
};

// Named operating points. The first three are the publisher's jointly-tuned
// bundles (the public transcribe_sortformer_preset menu; only these are
// accuracy-validated, see docs/porting/families/sortformer.md); `small` is a
// diagnostic geometry that forces several chunks and a cache compression on
// a short clip. The menu is discrete, not a latency dial.
struct SortformerStreamPreset {
    const char * name;
    int64_t chunk_len;
    int64_t chunk_right_context;
    int64_t fifo_len;
    int64_t spkcache_update_period;
    int64_t spkcache_len;
};

const SortformerStreamPreset * find_sortformer_stream_preset(std::string_view name) noexcept;
const std::vector<SortformerStreamPreset> & sortformer_stream_presets();

enum class SortformerRunMode {
    // One graph over the whole recording. Bounded by the prepared session
    // context (session_len_sec); rejects longer input rather than trimming.
    WholeWindow,
    // Fixed chunks through the AOSC/FIFO scheduler. Unbounded length,
    // bounded memory.
    Chunked,
};

struct SortformerRunPlan {
    SortformerRunMode mode = SortformerRunMode::WholeWindow;
    SortformerStreamParams params;
    // What decided the plan: "default", "offline", or a preset name; env
    // overrides are appended so a log line says where a geometry came from.
    std::string source;
};

// Resolve how a run executes from the model's shipped configuration and the
// session/request options (lowest to highest precedence):
//
//   shipped streaming cfg (or the NeMo defaults when the package ships none)
//   < `stream_preset` option: default | offline | <preset name>
//   < per-field options: stream_chunk_len, stream_left_context,
//     stream_right_context, stream_fifo_len, stream_spkcache_len,
//     stream_update_period
//   < TRANSCRIBE_SORTFORMER_STREAM_PRESET env (validation hook)
//   < TRANSCRIBE_SORTFORMER_STREAM_{CHUNK_LEN,FIFO_LEN,SPKCACHE_LEN,
//     UPDATE_PERIOD,RC,LC} env (validation hooks)
//
// `default` runs chunked with the shipped geometry when the package ships one
// and whole-window otherwise; `offline` always runs whole-window; a named
// preset always runs chunked. Both `stream_preset` and
// `sortformer_diar.stream_preset` spellings are read (session options carry
// the family prefix, request options do not); the same holds for every
// per-field key. Throws std::runtime_error on an unknown preset or an
// out-of-range field.
SortformerRunPlan resolve_sortformer_run_plan(
    const SortformerModelConfig & config,
    const std::unordered_map<std::string, std::string> & options);

// Host-side streaming state (AOSC speaker cache + FIFO), mirroring NeMo's
// StreamingSortformerState. Embeddings are pre-encode outputs stored
// row-major [n_frames, emb_dim]; preds are row-major [n_frames, n_spk].
struct SortformerStreamState {
    std::vector<float> spkcache;                      // [spkcache_n * D]
    std::vector<float> spkcache_preds;                // [spkcache_n * S]
    int64_t spkcache_n = 0;
    bool spkcache_preds_init = false;                 // NeMo: spkcache_preds is None until the first compress
    std::vector<float> fifo;                          // [fifo_n * D]
    std::vector<float> fifo_preds;                    // [fifo_n * S]
    int64_t fifo_n = 0;
    std::vector<float> mean_sil_emb;                  // [D]
    int64_t n_sil_frames = 0;
    std::vector<float> total_preds;                   // [total_n * S], accumulated chunk preds
    int64_t total_n = 0;
    int64_t compress_count = 0;

    void reset(int64_t emb_dim);
};

// NeMo streaming_update (sync branch): append the chunk's middle frames to
// the FIFO, pop into the speaker cache, compress the cache when it overflows,
// and accumulate the chunk's predictions. `chunk_embs` holds T_diar rows for
// the whole window (left context + chunk + right context); `preds` holds
// spkcache_n + fifo_n + T_diar rows in that order.
void sortformer_streaming_update(
    SortformerStreamState & state,
    const SortformerStreamParams & params,
    int64_t n_spk,
    int64_t emb_dim,
    const std::vector<float> & chunk_embs,
    int64_t T_diar,
    const std::vector<float> & preds,
    int64_t lc,
    int64_t rc);

// NeMo _compress_spkcache: shrink an overflowing speaker cache to
// spkcache_len rows by log-pred scoring, boosting and top-k selection.
void sortformer_compress_spkcache(
    SortformerStreamState & state,
    const SortformerStreamParams & params,
    int64_t n_spk,
    int64_t emb_dim);

// One chunk of the mel feature sequence: the window rows [win_lo, win_hi)
// fed to the stem, and how many of the resulting subsampled frames are left
// / right context rather than the chunk itself.
struct SortformerChunkWindow {
    int64_t win_lo = 0;
    int64_t win_hi = 0;
    int64_t lc = 0;
    int64_t rc = 0;
};

// NeMo streaming_feat_loader windows over `feature_frames` mel frames.
std::vector<SortformerChunkWindow> plan_sortformer_chunks(
    int64_t feature_frames,
    int64_t subsampling_factor,
    const SortformerStreamParams & params);

struct SortformerChunkedResult {
    std::vector<float> probabilities;  // row-major [frames, n_spk]
    int64_t frames = 0;
};

// Run the stem over one mel window (row-major [frames, n_mels]); returns the
// embeddings row-major [T_diar, emb_dim] and T_diar.
using SortformerPreEncodeFn =
    std::function<void(const float * mel_window, int64_t window_frames, std::vector<float> & embeddings, int64_t & T_diar)>;
// Run the body over `T_concat` concatenated embedding rows; returns the
// per-frame speaker probabilities row-major [T_concat, n_spk].
using SortformerInferFn =
    std::function<void(const std::vector<float> & concat, int64_t T_concat, std::vector<float> & preds)>;
// Called before each chunk with (chunk_index, chunk_count); throwing unwinds
// the run (the session routes RunControl through this).
using SortformerChunkProgressFn = std::function<void(int64_t, int64_t)>;

// The scheduler: chunk the mel sequence, stem each window, run the body over
// [spkcache | fifo | chunk], update the host state, accumulate. The result
// is trimmed to ceil(feature_frames / subsampling) rows like NeMo's
// total_preds[:, :ceil(feat_len / sub)].
SortformerChunkedResult run_sortformer_chunked(
    const SortformerFeatureBatch & features,
    int64_t n_mels,
    int64_t subsampling_factor,
    int64_t n_spk,
    int64_t emb_dim,
    const SortformerStreamParams & params,
    const SortformerPreEncodeFn & pre_encode,
    const SortformerInferFn & infer,
    const SortformerChunkProgressFn & progress);

}  // namespace engine::models::sortformer_diar
