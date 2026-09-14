#pragma once

// Shared greedy decode loop for encoder-decoder ASR models.
//
// Eliminates the duplicated argmax + logit-suppression code that currently
// lives in each engine package (15 copies across Whisper, Moonshine,
// MoonshineStreaming, and the community ASR ports). This is the B-skeleton
// for the DecodeDriver layer of Phase 11a: a small set of free functions
// that a per-engine `run_step` callback (graph build + execute) composes with.
//
// Models that already do argmax in-graph (ggml_argmax) simply skip the
// argmax step and call emit_progress directly.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace engine::asr {

// Apply logit suppression for a greedy decode step.
//
// Sets every token id in `suppress_ids` (size `suppress_count`) to
// -infinity; when `first_step` is true, also sets every token id in
// `begin_step_ids` (size `begin_step_count`) to -infinity. Tokens already
// at -infinity are left untouched (idempotent for overlapping lists).
//
// Equivalent to the apply_suppression() lambdas that were duplicated
// in WhisperRuntime, MoonshineRuntime, and the community ASR sessions.
inline void suppress_logits(float * logits, int vocab_size,
                            const int * suppress_ids, int suppress_count,
                            bool first_step,
                            const int * begin_step_ids = nullptr,
                            int begin_step_count = 0) {
  for (int i = 0; i < suppress_count; ++i) {
    const int id = suppress_ids[i];
    if (id >= 0 && id < vocab_size) {
      logits[id] = -std::numeric_limits<float>::infinity();
    }
  }
  if (first_step && begin_step_count > 0) {
    for (int i = 0; i < begin_step_count; ++i) {
      const int id = begin_step_ids[i];
      if (id >= 0 && id < vocab_size) {
        logits[id] = -std::numeric_limits<float>::infinity();
      }
    }
  }
}

// Greedy argmax over a logits vector.
//
// Returns the index of the maximum value. If every element is -infinity
// (all tokens suppressed), returns 0 as a safe fallback — the caller's
// EOS check will then terminate the loop.
//
// Equivalent to the argmax_logits() lambdas duplicated in
// WhisperRuntime and the other ASR sessions.
inline int argmax_logits(const float * logits, int vocab_size) {
  int best = 0;
  float best_v = logits[0];
  for (int i = 1; i < vocab_size; ++i) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = i;
    }
  }
  return best;
}

}  // namespace engine::asr
