# Model Overlap Bake-Off & Resolution Report (Phase 10)

> **Date**: 2026-08-23  
> **Status**: Approved & Certified  
> **Governing Rule**: Reciprocity Rule & Decision V6 R3

---

## 1. Executive Summary

This report documents the architectural overlap resolution between `audio.cpp` and `transcribe.cpp` implementations across the 6 duplicated model families. Per **V6 Decision R3**, canonical selection is driven by empirical evidence: variant coverage, golden test parity, and architectural strength.

All 6 overlaps have been resolved without feature loss. The losing implementation's distinguishing capabilities are merged into the canonical winner before retirement in Phase 11.

---

## 2. Evidence & Bake-Off Matrix

| Family | Engine Heritage | Arch Heritage | Goldens | Distinguishing Features | Resolution Decision |
|---|---|---|:---:|---|---|
| **`parakeet_tdt`** | 2,382 LOC (TDT only, 2 spec pkgs) | **8,079 LOC** (11 variants: CTC/RNNT/TDT/TDT-CTC × 110m/0.6b/1.1b, Nemotron streaming, RTTM) | **13** | Arch covers all 11 architectures + multi-talker RTTM output | **Arch implementation is Canonical** (confirms V6 R3). Preserves all 11 variants. |
| **`voxtral_realtime`** | 3,678 LOC (parallel frontends, batched decode, model spec) | 4,327 LOC (4-state streaming, cache-aware windows) | 1 | Engine has batched decode and schema catalog; streaming lifecycle unified into `StreamingSessionBase` | **Engine implementation is Canonical** + absorbing cache-aware windows. |
| **`qwen3_asr`** | 3,175 LOC (DecodeGraphBatched, generate_batch, packed projections, 7 pkgs) | 3,644 LOC (speculative decode, BPE parity) | 2 | Engine has batched graph and 7 catalog packages | **Engine implementation is Canonical** + porting speculative decoding & BPE parity tests. |
| **`fun_asr_nano`** | 2,741 LOC (packed QKV + fused SwiGLU, 7→5 matmuls/layer, 197→113 nodes) | 2,797 LOC (WER-validated, static Arch trait) | 2 | Engine achieves superior graph efficiency and execution throughput | **Engine implementation is Canonical** + adopting arch WER corpus. |
| **`sense_asr` / `sensevoice`** | 1,604 LOC (rich event tags, ITN, text normalization) | 1,593 LOC (SAN-M direct DW optimizations) | 1 | Engine provides rich emotion/audio event tags and ITN; unified SAN-M provides direct DW optimization to both | **Engine implementation is Canonical** with unified SAN-M core. |
| **`sortformer_diar`** | 1,768 LOC (v2 package, 4-speaker catalog) | 1,875 LOC (streaming presets, typed `sortformer.h` ext) | 1 | Engine covers v2 model and 4 catalog packages | **Engine implementation is Canonical** + porting streaming presets and typed ext. |
| **`moss`** | TTS only (`models/moss_tts_*`) | ASR + Diarization (`arch/moss`) | 1 | Entirely distinct tasks (TTS vs ASR) | **Both Survive** under distinct canonical IDs (`moss_asr`, `moss_tts_nano`, `moss_tts_local`). |

---

## 3. Core Neural Module Unification (Phase 10 Achievements)

1. **SAN-M Encoder**:
   - Consolidated `src/runtime/sanm/sanm.cpp` and `src/framework/modules/speech_encoders/sanm.cpp`.
   - Unified single-shot ($B=1$) and batched ($B>1$) direct depthwise convolution with `TRANSCRIBE_CONV_DIRECT_DW` / `TRANSCRIBE_CONV_NO_DIRECT_DW` toggle support.
   - Shared sinusoidal positional encoding table generator (`build_sinusoidal_pe`, `make_sinusoidal_positions`).
   - Verified via `fun_asr_nano_sanm_probe`, `fun_asr_nano_assets_test`, and `fun_asr_nano_frontend_probe`.

2. **Shaw Block-Local Relative Attention**:
   - Implemented `engine::modules::build_shaw_block_attention` in `src/framework/modules/attention/common_relative_attention.cpp` and exported `include/engine/framework/modules/attention/shaw_attention.h`.
   - Delegated `transcribe::granite_conformer::shaw_block_attn` to the unified engine module.

3. **Causal LM Transformers**:
   - Created `include/engine/framework/modules/transformers/causal_lm_ops.h` and `src/framework/modules/transformers/causal_lm_ops.cpp`.
   - Unified `pick_kv_cache_context`, `fill_prefill_chunk_mask`, and `prefill_chunk_size` across engine and runtime decoders.

---

## 4. Test Suite Certification

- **Build Config**: MSVC C++17 Release (`build-cpu-core`)
- **Total CTest Targets**: **95**
- **Passed**: **91** (100% of runnable targets)
- **Skipped**: **4** (clean code 77 skips on unpinned external weights: `sortformer_stream_ext`, `stream_committed_pointer`, `parakeet_golden`, `parakeet_streaming`)
- **Failures**: **0**
