// transcribe-arch-adapter.h - bridge from audio.cpp's C++ model vtable into
// the transcribe C ABI `Arch` dispatch trait (V6 plan sub-task 0.H).
//
// INTERNAL. The audio.cpp framework loads models through a 3-level C++ vtable:
//   IVoiceModelLoader → ILoadedVoiceModel → IVoiceTaskSession
// while the unified C ABI dispatches through a flat struct-of-function-pointers
// (`Arch` in transcribe-arch.h). The ArchAdapter is the shim that owns the
// lifetime mismatch (C ABI = C structs, framework = std::unique_ptr + RAII) and
// translates the two result representations (TaskResult/StreamEvent ↔ the
// family-agnostic result vectors on transcribe_session).
//
// Dispatch wiring (src/runtime/transcribe-arch.cpp): find_arch() scans the
// builtin transcribe.cpp families first, then consults adapter_find_arch()
// so audio.cpp families can carry the same GGUF `general.architecture` name.
// A shared GGUF architecture name therefore resolves to the builtin handler
// when one exists (e.g. moss, qwen3_asr, voxtral_realtime keep the transcribe.cpp
// family); the adapter entry for that name remains registered but shadowed and
// is reclaimed by Phase 1 consolidation.

#pragma once

#include "transcribe-arch.h"

namespace transcribe {

// Look up an audio.cpp framework family by GGUF architecture-name string.
// Returns nullptr if no registered adapter family matches. The returned
// Arch's load()/init_context()/run()/stream_*() are shared generic handlers
// distinguished only by `name`. Consulted by find_arch() AFTER the builtin
// transcribe.cpp table so builtin families keep precedence on name collision.
const Arch * adapter_find_arch(const char * name);

}  // namespace transcribe