// transcribe-vad.cpp - struct_size negotiation + (Task 6) plan() impl.

#include "transcribe-vad.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace transcribe::vad {

bool params_present(const struct transcribe_run_params * run_params) {
    if (run_params == nullptr) {
        return false;
    }
    // struct_size is the caller's declared size. The vad field exists only
    // if the caller's struct is large enough to hold it. offsetof gives the
    // byte offset of vad within the (fully-defined) struct as this library
    // was compiled; a caller built against an older, shorter header sets a
    // smaller struct_size and is treated as not having the field.
    const size_t vad_off = offsetof(struct transcribe_run_params, vad);
    return run_params->struct_size >= vad_off + sizeof(struct transcribe_vad_params);
}

transcribe_vad_mode effective_mode(const struct transcribe_run_params * run_params) {
    if (!params_present(run_params)) {
        return TRANSCRIBE_VAD_OFF;
    }
    return run_params->vad.mode;
}

std::vector<chunk_plan> plan(const std::vector<time_span> & speech,
                             int64_t                        total_ms,
                             int64_t                        max_chunk_ms,
                             int64_t                        merge_gap_ms,
                             int64_t                        padding_ms) {
    std::vector<chunk_plan> out;
    if (total_ms <= 0 || speech.empty()) {
        return out;
    }

    // 1. Sort + clip speech to [0, total_ms], drop zero/negative-length.
    std::vector<time_span> segs;
    segs.reserve(speech.size());
    for (auto s : speech) {
        if (s.end_ms <= s.start_ms) {
            continue;
        }
        s.start_ms = std::max<int64_t>(s.start_ms, 0);
        s.end_ms   = std::min<int64_t>(s.end_ms, total_ms);
        if (s.end_ms <= s.start_ms) {
            continue;
        }
        segs.push_back(s);
    }
    if (segs.empty()) {
        return out;
    }
    std::sort(segs.begin(), segs.end(),
              [](const time_span & a, const time_span & b) { return a.start_ms < b.start_ms; });

    // 2+3. Greedily pack speech segments into windows up to max_chunk_ms,
    // splitting at internal silence gaps when a window would overflow.
    // Direct port of audio.cpp's plan_vad_audio_chunks (chunking.cpp),
    // verified correct across meeting audio.
    struct ChunkState {
        int64_t span_start_ms;
        int64_t span_end_ms;
        int64_t speech_end_ms;  // furthest speech seen in this window
    };

    std::vector<ChunkState> states;
    const int64_t           pad         = padding_ms > 0 ? padding_ms : 0;
    const bool              has_ceiling = max_chunk_ms > 0;

    auto padded_start = [&](int64_t s) {
        return std::max<int64_t>(s - pad, 0);
    };
    auto padded_end = [&](int64_t e) {
        return std::min<int64_t>(e + pad, total_ms);
    };
    auto start_chunk = [&](int64_t & span_start, int64_t span_end, int64_t speech_start, int64_t speech_end) {
        int64_t chunk_end = has_ceiling ? std::min<int64_t>(span_end, span_start + max_chunk_ms) : span_end;
        int64_t se        = (chunk_end > speech_start) ? std::min<int64_t>(speech_end, chunk_end) : span_start;
        states.push_back(ChunkState{ span_start, chunk_end, se });
        span_start = chunk_end;
    };

    for (const auto & seg : segs) {
        int64_t       span_start   = padded_start(seg.start_ms);
        const int64_t span_end     = padded_end(seg.end_ms);
        const int64_t speech_start = seg.start_ms;
        const int64_t speech_end   = seg.end_ms;
        while (span_start < span_end) {
            if (!states.empty()) {
                ChunkState & cur = states.back();
                if (span_end <= cur.span_end_ms) {
                    // segment fully inside current window
                    cur.speech_end_ms = std::max(cur.speech_end_ms, speech_end);
                    break;
                }
                if (span_start <= cur.span_end_ms) {
                    // overlaps current window
                    if (!has_ceiling) {
                        cur.span_end_ms   = span_end;
                        cur.speech_end_ms = std::max(cur.speech_end_ms, speech_end);
                        break;
                    }
                    const int64_t capacity_end = cur.span_start_ms + max_chunk_ms;
                    if (span_end <= capacity_end) {
                        // fits in budget: grow current window
                        cur.span_end_ms   = span_end;
                        cur.speech_end_ms = std::max(cur.speech_end_ms, speech_end);
                        break;
                    }
                    // overflows budget: prefer closing at the silence gap before
                    // this segment, then re-enter the loop to open a fresh window.
                    if (speech_start > cur.speech_end_ms && cur.speech_end_ms > cur.span_start_ms) {
                        const int64_t boundary = std::min<int64_t>(cur.span_end_ms, speech_start);
                        if (boundary >= cur.span_end_ms && boundary > cur.span_start_ms) {
                            cur.span_end_ms = boundary;
                            span_start      = boundary;
                            start_chunk(span_start, span_end, speech_start, speech_end);
                            continue;
                        }
                    }
                    // no usable gap: extend current to capacity, reprocess remainder
                    if (cur.span_end_ms < capacity_end) {
                        cur.span_end_ms = std::min<int64_t>(span_end, capacity_end);
                        if (cur.span_end_ms > speech_start) {
                            cur.speech_end_ms =
                                std::max(cur.speech_end_ms, std::min<int64_t>(speech_end, cur.span_end_ms));
                        }
                        span_start = cur.span_end_ms;
                        continue;
                    }
                } else if (has_ceiling) {
                    // disjoint from current window
                    const int64_t gap = span_start - cur.span_end_ms;
                    if (merge_gap_ms > 0 && gap <= merge_gap_ms && span_end - cur.span_start_ms <= max_chunk_ms) {
                        cur.span_end_ms   = span_end;
                        cur.speech_end_ms = std::max(cur.speech_end_ms, speech_end);
                        break;
                    }
                } else {
                    // disjoint, no ceiling: merge (one unbounded window)
                    cur.span_end_ms   = span_end;
                    cur.speech_end_ms = std::max(cur.speech_end_ms, speech_end);
                    break;
                }
            }
            start_chunk(span_start, span_end, speech_start, speech_end);
        }
    }

    // 4. Build chunk_plan. source_span is the merged padded window;
    // keep_span strips padding back to the speech core (clamped, non-empty).
    for (const auto & st : states) {
        chunk_plan cp;
        cp.source_span.start_ms   = st.span_start_ms;
        cp.source_span.end_ms     = st.span_end_ms;
        cp.source_span.confidence = 1.0f;
        cp.keep_span.start_ms     = std::max<int64_t>(st.span_start_ms + pad, 0);
        cp.keep_span.end_ms       = std::min<int64_t>(st.span_end_ms - pad, total_ms);
        if (cp.keep_span.end_ms <= cp.keep_span.start_ms) {
            // padding ate the whole window (very short speech); keep it all
            cp.keep_span.start_ms = st.span_start_ms;
            cp.keep_span.end_ms   = st.span_end_ms;
        }
        cp.keep_span.confidence = 1.0f;
        out.push_back(cp);
    }
    return out;
}

}  // namespace transcribe::vad
