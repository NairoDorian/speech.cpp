// transcribe-vad.h - internal VAD types and pure planning functions.
//
// The public ABI types (transcribe_vad_mode, transcribe_vad_params,
// transcribe_vad_segment) live in include/transcribe/transcribe.h. This header holds
// the internal-only pure-function API (vad::plan) and the struct_size
// negotiation helpers used by run_one_inner to detect whether the caller's
// run_params carries the vad field at all.

#ifndef TRANSCRIBE_VAD_H
#define TRANSCRIBE_VAD_H

#include "transcribe/transcribe.h"

#include <cstdint>
#include <vector>

namespace transcribe::vad {

// Internal ms-resolution span (the public transcribe_vad_segment mirrors
// this but lives in the C ABI header).
struct time_span {
    int64_t start_ms   = 0;
    int64_t end_ms     = 0;
    float   confidence = 0.0f;
};

// A decode window produced by plan(). source_span is the PCM range fed to
// arch->run (includes padding/context); keep_span is the sub-range whose
// timestamps are attributed to this chunk (source_span minus padding).
struct chunk_plan {
    time_span source_span;
    time_span keep_span;
};

// Decide whether the caller's run_params actually carries the vad field.
// Returns false for callers built against an older header whose
// transcribe_run_params was shorter than the vad field's offset, so they
// silently get the pre-VAD full-buffer path. run_params may be NULL
// (means "all defaults" -> VAD off).
TRANSCRIBE_API bool params_present(const struct transcribe_run_params * run_params);

TRANSCRIBE_API transcribe_vad_mode effective_mode(const struct transcribe_run_params * run_params);

TRANSCRIBE_API std::vector<chunk_plan> plan(const std::vector<time_span> & speech,
                                            int64_t                        total_ms,
                                            int64_t                        max_chunk_ms,
                                            int64_t                        merge_gap_ms,
                                            int64_t                        padding_ms);

}  // namespace transcribe::vad

#endif  // TRANSCRIBE_VAD_H
