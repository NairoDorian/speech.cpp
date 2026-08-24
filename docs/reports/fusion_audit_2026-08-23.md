# Comprehensive Ground Truth Audit Report — `speech.cpp`

**Date**: 2026-08-23  
**Target**: `speech.cpp` (convergence of `audio.cpp` and `transcribe.cpp`)  
**Commit**: `c776b81` / `be2dfd5`  
**Authoritative Reference**: [FUSION_ROADMAP_PLAN.md](../../FUSION_ROADMAP_PLAN.md) v5.0

---

## 1. Verified Inventory

| Quantity | Value | Verification Command |
|---|---:|---|
| Engine model families (`src/models/`) | 42 | `ls src/models \| wc -l` |
| Community families (`src/community_models/`) | 12 | `ls src/community_models \| wc -l` |
| Transcribe arch families (`src/runtime/arch/`) | 18 | `ls src/runtime/arch \| wc -l` |
| Model-spec catalog entries | 58 | `ls model_specs/*.json \| wc -l` |
| Registered engine model targets | 52 | `grep -oP 'audiocpp_add_model\(\K[a-z0-9_]+' CMakeLists.txt \| sort -u \| wc -l` |
| LOC `src/models/` | 183,729 | `find src/models -name '*.cpp' -o -name '*.h' \| xargs cat \| wc -l` |
| LOC `src/framework/` | 59,045 | `find src/framework -name '*.cpp' -o -name '*.h' \| xargs cat \| wc -l` |
| LOC `src/runtime/arch/` | 74,555 | `cat src/runtime/arch/*/*.cpp src/runtime/arch/*/*.h \| wc -l` |
| LOC `src/runtime/` (root) | 15,806 | `cat src/runtime/*.cpp src/runtime/*.h \| wc -l` |
| LOC runtime shared modules | 3,529 | `causal_lm` 1,347 · `conformer` 1,557 · `sanm` 395 · `granite_conformer` 230 |
| `CMakeLists.txt` | 3,305 lines · 85 `add_test` calls | `wc -l CMakeLists.txt; grep -c add_test CMakeLists.txt` |
| Public headers | `include/transcribe/transcribe.h` 2,759 · `capi/include/audiocpp.h` 1,184 | `wc -l` |
| ggml fork patches | 7 (`0001`–`0007`) | `ls patches/ggml/` |

---

## 2. Reachability Findings

1. **18 ASR arch families (74.5 kLOC)**: Currently built only when `-DSPEECHCPP_ENABLE_UNIFIED_ABI=ON -DSPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=ON` (default OFF). Unreachable from CLI, server, or WebUI.
2. **`whisper`, `moonshine`, `moonshine_streaming` model specs**: 24 catalog packages exist, but no engine loader is registered.
3. **`SharedWeightRegistry` / `ScopedWeightShareKey`**: Implemented in engine, but zero call sites exist in CLI, server, or C ABI.
4. **`IOfflineVoiceTaskSession::run_batch`**: Implemented across 5 families, but not called by public APIs; `ArchAdapter` nulls the `run_batch` hook for all 16 entries.
5. **C ABI Exception Containment**: 17 of ~43 exported functions in `audiocpp_capi.cpp` are wrapped in `AUDIOCPP_CATCH`; ~26 are unguarded.

---

## 3. Defect Inventory

### Defect D1: `find_arch()` precedence collision on framework-sniffed paths (Severity: High)
- In `src/runtime/transcribe.cpp:1570`, `find_arch()` searches the builtin table first.
- For `qwen3_asr`, `voxtral_realtime`, and `moss`, non-GGUF files (e.g. safetensors / directories) are dispatched to the GGUF-expecting builtin handler with a null `gguf_` pointer.
- **Fix**: Dispatch directly to `adapter_find_arch(family.c_str())`.

### Defect D2: Adapter hooks null across all entries (Severity: Medium)
- `run_batch`, `stream_validate`, and `run_validate` are `nullptr` in `adapter_archs[]`.
- Pre-clear transcript preservation guarantee does not hold for framework models.
- **Fix**: Implement `adapter_run_batch_impl`, `adapter_run_validate_impl`, and `adapter_stream_validate_impl`.

### Defect D3: Family-ID Aliasing (Severity: Medium)
- Divergent namings between engine, GGUF `general.architecture`, and model specs (`sense_asr` vs `sensevoice`, `fun_asr_nano` vs `funasr_nano`, `parakeet_tdt` vs `parakeet`, `sortformer_diar` vs `sortformer`).
- **Fix**: Implement unified `family_registry.h` with canonical IDs and alias tables.
