// transcribe-family-ext.cpp - public extension initializers for families whose
// transcribe.cpp arch has been retired (Phase 10.5).
//
// A family's typed extension is part of the public C ABI: the struct, its kind
// constant and its init function are published in include/transcribe/<family>.h
// and callers link against them. The struct and the constant are header-only
// and survive on their own; the init function had a definition inside the arch
// that owned the family. When an arch is deleted the definition has to land
// somewhere that still ships, or the retirement silently breaks the ABI.
//
// That is what this file is. It holds no model code - only the initializers -
// and each entry names the family and the commit that retired its arch. The
// knobs themselves are served by the ArchAdapter, which translates the
// extension into the framework session's request options.

#include "transcribe/voxtral_realtime.h"

#include <cstring>

// voxtral_realtime: arch retired in Phase 10.5 (ledger B12). The adapter
// validates and applies both fields; see adapter_check_stream_ext /
// adapter_apply_stream_ext in transcribe-arch-adapter.cpp.
extern "C" void transcribe_voxtral_realtime_stream_ext_init(struct transcribe_voxtral_realtime_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size               = sizeof(*p);
    p->ext.kind               = TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM;
    p->num_delay_tokens       = -1;
    p->min_decode_interval_ms = -1;
}
