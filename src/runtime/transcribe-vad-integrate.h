// transcribe-vad-integrate.h - run_with_vad: the VAD chunk loop that wraps
// arch->run, plus the per-chunk result-merge helpers.

#ifndef TRANSCRIBE_VAD_INTEGRATE_H
#define TRANSCRIBE_VAD_INTEGRATE_H

#include "transcribe-vad.h"
#include "transcribe/transcribe.h"

#include <cstddef>
#include <vector>

struct transcribe_session;
struct transcribe_run_params;

namespace transcribe::vad {

// Run the VAD chunk loop. On success returns TRANSCRIBE_OK and the session's
// result fields hold the merged multi-chunk result. On VAD-load failure
// sets *degraded=true (and the caller should run the original full-buffer
// arch->run). On any other error returns the status; partial results are
// preserved on abort.
TRANSCRIBE_API transcribe_status run_with_vad(struct transcribe_session *          session,
                                               const float *                        pcm,
                                               int                                  n_samples,
                                               const struct transcribe_run_params * params,
                                               bool &                               degraded);

// ---- Testable merge helpers ---------------------------------------------

struct chunk_baseline {
    size_t n_segments = 0;
    size_t n_words    = 0;
    size_t n_tokens   = 0;
};

// Snapshot the session's entry counts (call before a chunk's arch->run).
TRANSCRIBE_API chunk_baseline snapshot(const struct transcribe_session & s);

// Fold this chunk's newly-added entries into the global timeline: shift
// their timestamps by chunk.keep_span.start_ms and fix cross-references
// (word/token seg_index, segment first_word/first_token) from chunk-local
// to global indices.
TRANSCRIBE_API void offset_chunk_results(struct transcribe_session & s, const chunk_baseline & base, const chunk_plan & chunk);

// Trim the three result vectors back to baseline (drop a failed chunk's
// partial entries while keeping earlier chunks' results).
TRANSCRIBE_API void rollback_to(struct transcribe_session & s, const chunk_baseline & base);

// Rebuild full_text / raw_text by space-joining segment texts.
TRANSCRIBE_API void rebuild_full_text(struct transcribe_session & s);

// Detect speech segments natively (using in-tree Silero VAD or Energy VAD)
TRANSCRIBE_API std::vector<time_span> detect_speech(const float * pcm, int n_samples, int sample_rate, const struct transcribe_vad_params & vp);

}  // namespace transcribe::vad

#endif  // TRANSCRIBE_VAD_INTEGRATE_H
