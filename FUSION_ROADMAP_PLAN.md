# Master FUSION Roadmap Plan: The Ultimate Convergence of `audio.cpp` & `transcribe.cpp` into `speech.cpp`

> **Document Status**: Authoritative Master Plan & Architectural Blueprint  
> **Target Target**: `speech.cpp` — The Single, Fully-Fused Native Speech & Audio Intelligence Framework  
> **Date**: 2026-08-23  
> **Version**: 1.0 (Master Unified Edition)

---

## 1. Executive Vision: The "Best of Both Worlds" Fusion

`speech.cpp` is the definitive convergence of two groundbreaking C++ audio AI codebases:
- **`audio.cpp`**: Broad multi-modal coverage across 50+ model families spanning Text-to-Speech (TTS), Voice Cloning (VC), Source Separation, Speaker Diarization, Voice Activity Detection (VAD), Forced Alignment, Audio Codecs, and MIDI music generation, with rich HTTP/REST servers, CLI tooling, and WebUI.
- **`transcribe.cpp`**: Production-grade, high-accuracy Automatic Speech Recognition (ASR) engine with strict numerical parity, speculative decoding, size-aware C ABI discipline, exception safety, 4-state streaming machines, and golden WER regression gates.

### The Master Fusion Principle
> **"The two projects learn from each other in parallel — each is the other's teacher and student — and merging them improves them both at the same time."**

This roadmap defines the **complete, end-state structural fusion** of both systems. Rather than maintaining parallel runtimes behind a bridge adapter, `speech.cpp` systematically absorbs the superior components from each side, upgrades the entire engine with the lessons learned, and eliminates all redundant or duplicate code.

---

## 2. Current Architecture vs. Fully Fused Target Architecture

### 2.1 Current State: Coexistence with Inter-Engine Bridge (Phases 0–6)
Currently, `speech.cpp` shares a unified low-level GGML substrate, memory safety allocators, build toolchain, and dual public C ABIs (`audiocpp.dll` and `transcribe.dll`), but maintains two parallel model layers connected via `transcribe-arch-adapter.cpp`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT STATE: BRIDGED COEXISTENCE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Dual Public ABIs:  audiocpp.dll (Multi-Task)  │  transcribe.dll (STT)       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bridge:            src/runtime/transcribe-arch-adapter.cpp                  │
├───────────────────────────────────────┬─────────────────────────────────────┤
│ Parallel Models:   src/models/*       │  src/runtime/arch/*                 │
│                    (50+ Audio Models) │  (18 STT Architectures)             │
├───────────────────────────────────────┼─────────────────────────────────────┤
│ Parallel Frontends:src/framework/audio│  src/runtime/transcribe-mel.cpp     │
│                    (kaldi_fbank.cpp)  │  (transcribe-kaldi-fbank.cpp)       │
├───────────────────────────────────────┼─────────────────────────────────────┤
│ Parallel Toks:     external/llama_tok │  src/runtime/transcribe-tokenizer   │
│                    external/sentencep │  (<tok> JSON Tokenizer)             │
├───────────────────────────────────────┴─────────────────────────────────────┤
│ Unified Substrate: Single GGML • SharedWeightRegistry • BackendWeightStore  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Target State: Fully Fused Monolithic Engine (`libspeech`)
The target state eliminates the adapter layer, converges the frontends and tokenizers into unified shared modules, consolidates duplicate ASR families into canonical high-throughput implementations, and unifies the public surface into a single monolithic library (`libspeech`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TARGET STATE: FULLY FUSED SPEECH.CPP                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                 Unified Public C ABI: libspeech.so / speech.dll             │
│         (Single Monolithic Shared Library with Opaque Handles & C++ Guards) │
│         • speech_* Universal API (All 14 Audio Tasks + Universal Progress)  │
│         • Backward-Compatible Headers: audiocpp.h & transcribe.h (Shims)    │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Unified Task Session Layer                         │
│   • SpeechSession Base (Graph Execution + Stream State Machine + Lifecycle) │
│   • Multi-Task Pipeline Engine (VAD Chunking + ASR + TTS + Diarization)     │
├─────────────────────────────────────────────────────────────────────────────┤
│                      Canonical Fused Model Subsystems                       │
│   ┌───────────────────────────┬───────────────────────────┬─────────────┐   │
│   │        TTS & Voice        │        ASR & STT          │    Audio    │   │
│   │  • IndexTTS2, F5-TTS      │  • Whisper (HF 5.x Seek)  │ • HTDemucs  │   │
│   │  • Qwen3-TTS, CosyVoice   │  • Moonshine (Offline/Str)│ • RoFormer  │   │
│   │  • MiniMax-H3, MOSS       │  • Parakeet-TDT (Batched) │ • Sortformer│   │
│   │  • Kokoro, PocketTTS      │  • Qwen3-ASR (Fused)      │ • DFN2/RNNoi│   │
│   │  • Seed-VC (Gallocr)      │  • Voxtral (4-State Str)  │ • MMS Align │   │
│   └───────────────────────────┴───────────────────────────┴─────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Unified Framework Shared Modules                      │
│   • Mel & Audio Frontend: SIMD MelFrontend + Precomputed Kaldi Filterbanks  │
│   • Tokenizer: Unified Engine (BPE, Byte-Level BPE, SentencePiece, WordP)   │
│   • Audio Chunking & VAD: Native vad::plan + Deterministic Re-stitching     │
│   • Codecs: Mimi, MioCodec, EnCodec, SNAC, DAC, Vocos                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                      High-Performance Memory & Compute                      │
│   • SharedWeightRegistry (Process-Wide VRAM Sharing across All Sessions)    │
│   • ggml_gallocr Topological Arena Reuse (18x VRAM Peak Reduction)          │
│   • BackendWeightStore (16MB Metadata Context Pool Budget Cap)              │
│   • Fused SwiGLU & Packed QKV Linear Projections                            │
│   • Single Pinned GGML (CPU AVX2/FMA/AVX512 • CUDA sm_50-sm_120 • Metal)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Fusion Inventory & Elimination Matrix

To systematically eliminate duplication, every overlapping subsystem is analyzed and assigned a definitive fusion strategy.

### 3.1 Audio Frontends (Mel Spectrogram & Filterbanks)
- **Current Duplication**:
  - `speech.cpp/src/framework/audio/kaldi_fbank.cpp` & `mel_frontend.cpp` (audio.cpp).
  - `speech.cpp/src/runtime/transcribe-mel.cpp` & `transcribe-kaldi-fbank.cpp` (transcribe.cpp).
- **Evaluation & Choice**:
  - `transcribe.cpp`'s frontend features band-skip optimizations, precomputed filterbank tables, multi-threaded SIMD acceleration, and strict numerical tolerances certified by `asr_e2e_wer_test`.
- **Fusion Action**:
  - Consolidate into `include/engine/framework/audio/mel_frontend.h` and `src/framework/audio/mel_frontend.cpp`.
  - Delete `src/runtime/transcribe-mel.*` and `src/runtime/transcribe-kaldi-fbank.*`.
  - Update all models (TTS, ASR, VC) to consume the unified SIMD `MelFrontend`.

### 3.2 Tokenizer Engine
- **Current Duplication**:
  - `speech.cpp/external/llama_tokenizer/` + `speech.cpp/external/sentencepiece/` (audio.cpp).
  - `speech.cpp/src/runtime/transcribe-tokenizer.cpp` (transcribe.cpp).
- **Evaluation & Choice**:
  - `transcribe.cpp`'s `<tok>` JSON tokenizer is lightweight, self-contained, and handles complex byte fallbacks for multilingual ASR.
  - `audio.cpp`'s `sentencepiece` and `llama_tokenizer` support SentencePiece `.model` and HuggingFace `tokenizer.json` for 40+ TTS families.
- **Fusion Action**:
  - Create a unified `Tokenizer` interface in `include/engine/framework/text/tokenizer.h`.
  - Implement a single dispatcher supporting GGUF embedded tokenizer metadata, SentencePiece binary models, and HuggingFace JSON tokenizers.
  - Delete redundant helper wrappers and unneeded vendored files.

### 3.3 Overlapping Model Family Consolidation
Four major model families exist in both trees and will be fused into single canonical implementations:

| Family | `audio.cpp` Implementation | `transcribe.cpp` Implementation | Canonical Fused Architecture |
|---|---|---|---|
| **`Qwen3-ASR`** | `src/models/qwen3_asr/` (Batched decode graphs, thinker session) | `src/runtime/arch/qwen3_asr/` (Speculative decoding, frontend whisper) | Merge speculative decoding from transcribe into `src/models/qwen3_asr/`, add packed QKV, delete `src/runtime/arch/qwen3_asr/`. |
| **`Voxtral Realtime`** | `src/models/voxtral_realtime/` (Parallel host audio, reset graphs) | `src/runtime/arch/voxtral_realtime/` (4-state streaming machine) | Merge 4-state streaming machine into `src/models/voxtral_realtime/`, delete `src/runtime/arch/voxtral_realtime/`. |
| **`SenseVoice`** | `src/models/sense_asr/` (Rich CTC + event tags) | `src/runtime/arch/sensevoice/` (SAN-M block optimization) | Port SAN-M optimizations into `src/models/sense_asr/`, delete `src/runtime/arch/sensevoice/`. |
| **`FunASR Nano`** | `src/models/fun_asr_nano/` (Packed QKV & Gate/Up, Fused SwiGLU) | `src/runtime/arch/funasr_nano/` (Static Arch struct) | Canonicalize `src/models/fun_asr_nano/` with batched prefill, delete `src/runtime/arch/funasr_nano/`. |
| **`Parakeet-TDT`** | `src/community_models/parakeet_tdt/` (TDT greedy decoder loop) | `src/runtime/arch/parakeet/` (Frame-windowed joint batching) | Move to `src/models/parakeet_tdt/`, incorporate frame-windowed joint batching, delete `src/runtime/arch/parakeet/`. |

### 3.4 Standalone Transcribe Architectures (Whisper, Moonshine, Canary, Granite, Cohere, GigaAM)
- **Current Location**: `src/runtime/arch/<family>/`.
- **Fusion Action**:
  - Migrate all remaining standalone transcribe architectures into `src/models/<family>/` conforming to the unified `RuntimeSessionBase` / `IVoiceTaskSession` contract.
  - Register in `model_specs/<family>.json` with complete package metadata.
  - Delete `src/runtime/arch/` and `src/runtime/transcribe-arch-adapter.cpp` completely once all families are native.

### 3.5 Public C ABI Consolidation (`libspeech`)
- **Current Duplication**:
  - `capi/include/audiocpp.h` + `capi/src/audiocpp_capi.cpp` (`audiocpp.dll`).
  - `include/transcribe/transcribe.h` + `src/runtime/transcribe.cpp` (`transcribe.dll`).
- **Fusion Action**:
  - Create the unified `include/speech/speech.h` and `src/capi/speech_capi.cpp` exporting `libspeech.so` / `speech.dll`.
  - Provide thin backward-compatibility shim headers `include/audiocpp.h` and `include/transcribe.h` that forward directly to `speech_*` symbols.
  - Retain size-aware structs, opaque handles, and `speech_set_progress_callback`.

---

## 4. Master Phased Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MASTER FUSION ROADMAP PHASES                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 1: Engine Hardening & Allocator Guards                        [COMPLETED]
│ Phase 2: Toolchain Modernization & Build Provenance                 [COMPLETED]
│ Phase 3: Native Embedded VAD & Long-Form Pipeline                   [COMPLETED]
│ Phase 4: High-Throughput Batched ASR & SharedWeightRegistry         [COMPLETED]
│ Phase 5: Universal C ABI Subsystem & Progress Callbacks             [COMPLETED]
│ Phase 6: Whisper GPU Cleanup, Arch Sync & Model Spec Catalogs       [COMPLETED]
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 7: Common Audio Frontend & Tokenizer Unification              [NEXT]
│ Phase 8: Overlapping Model Family Fusion (Qwen3, Voxtral, Sense, Parakeet)
│ Phase 9: Migration of Remaining STT Arches to Engine Core (Whisper, Moonshine)
│ Phase 10: Unified C ABI (`libspeech`) & Backward Compatibility Shims
│ Phase 11: Zero-Dependency Language Bindings (Rust, Python, TS, Swift)
│ Phase 12: Comprehensive Regression Verification & Release 1.0
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 7: Common Audio Frontend & Tokenizer Unification
**Goal**: Eliminate duplicate mel-spectrogram extractors, filterbank tables, and tokenizer parsing routines.

1. **SIMD Mel Frontend Convergence**:
   - Merge `transcribe-mel.cpp` and `kaldi_fbank.cpp` into `src/framework/audio/mel_frontend.cpp`.
   - Support standard 80-band, 128-band Mel, and Kaldi 80-dimensional filterbanks with precomputed triangular weights.
   - Wire all TTS models (F5-TTS, CosyVoice, IndexTTS2) and ASR models (Whisper, Qwen3, Moonshine) to this shared frontend.
   - Delete `src/runtime/transcribe-mel.*` and `src/runtime/transcribe-kaldi-fbank.*`.
2. **Unified Tokenizer Dispatcher**:
   - Create `src/framework/text/unified_tokenizer.cpp` supporting BPE, WordPiece, SentencePiece, and `<tok>` JSON formats.
   - Validate with `tests/unittests/test_tokenizer_parity.cpp`.

---

### Phase 8: Overlapping Model Family Fusion
**Goal**: Merge dual implementations of Qwen3-ASR, Voxtral Realtime, SenseVoice, FunASR Nano, and Parakeet-TDT into single best-of-breed modules.

1. **`Qwen3-ASR` Fusion**:
   - Integrate speculative decoding from `arch/qwen3_asr/` into `src/models/qwen3_asr/session.cpp`.
   - Maintain `DecodeGraphBatched` and `generate_batch`.
   - Remove `src/runtime/arch/qwen3_asr/`.
2. **`Voxtral Realtime` Fusion**:
   - Integrate the 4-state streaming machine into `src/models/voxtral_realtime/session.cpp`.
   - Remove `src/runtime/arch/voxtral_realtime/`.
3. **`SenseVoice` & `FunASR Nano` Fusion**:
   - Consolidate SAN-M blocks and packed QKV projections under `src/models/`.
   - Remove `src/runtime/arch/sensevoice/` and `src/runtime/arch/funasr_nano/`.
4. **`Parakeet-TDT` Fusion**:
   - Move Parakeet from `community_models/` to core `src/models/parakeet_tdt/`.
   - Integrate frame-windowed greedy joint graph batching.
   - Remove `src/runtime/arch/parakeet/`.

---

### Phase 9: Migration of Remaining STT Arches to Engine Core
**Goal**: Move all remaining transcribe architectures (Whisper, Moonshine, Canary, Granite, Cohere, GigaAM, MedASR) into `src/models/` as first-class engine sessions.

1. **Native Engine Sessions**:
   - Create `src/models/whisper/` (`session.cpp`, `runtime.cpp`, `assets.cpp`) with HF 5.x seek continuation fix.
   - Create `src/models/moonshine/` and `src/models/moonshine_streaming/` with native `IVoiceTaskSession` integration.
   - Port Canary, Granite(+NAR), Cohere, GigaAM, and MedASR to `src/models/`.
2. **Decommission Bridge Adapter**:
   - Remove `src/runtime/arch/` directory completely.
   - Remove `src/runtime/transcribe-arch-adapter.cpp` and `transcribe-arch.cpp`.
   - Update CMake to build all models from `src/models/` under the unified engine target.

---

### Phase 10: Unified C ABI (`libspeech`) & Compatibility Shims
**Goal**: Deliver a single monolithic shared library exporting the entire speech intelligence surface.

1. **`libspeech` Implementation**:
   - Create `include/speech/speech.h` and `src/capi/speech_capi.cpp`.
   - Consolidate all 14 task APIs, VAD, Diarization, Streaming, Progress Callbacks, and Memory Management into `speech_*` symbols.
   - Enable `SPEECH_SHARED_EMBED=ON` for single-target DLL/SO compilation with hidden internal symbols.
2. **Backward-Compatibility Shims**:
   - `include/audiocpp.h` inline forwarding to `speech_*`.
   - `include/transcribe/transcribe.h` inline forwarding to `speech_*`.
   - Build compatibility import libraries (`audiocpp.lib`, `transcribe.lib`) so existing binaries link without modification.

---

### Phase 11: Zero-Dependency Language Bindings (`dynload`)
**Goal**: Deliver high-level language bindings with zero local compilation requirements.

1. **Rust Binding (`bindings/rust`)**:
   - Pure dynamic loading crate using `libloading` for `libspeech.so` / `speech.dll`.
   - Idiomatic Safe Rust wrappers for TTS, ASR, Streaming, and VAD.
2. **Python Binding (`bindings/python`)**:
   - Lightweight `ctypes` wrapper distributed via PyPI wheel with bundled shared library.
3. **TypeScript Binding (`bindings/typescript`)**:
   - Node.js / Bun / Deno binding using `koffi` (C-FFI without node-gyp native compilation).
4. **Swift Binding (`bindings/swift`)**:
   - Swift Package Manager (SPM) wrapper with C module map.

---

### Phase 12: Comprehensive Regression Verification & Release 1.0
**Goal**: 100% test coverage, strict WER/DER regression verification, and Release 1.0 tagging.

1. **Automated Verification Suite**:
   - **ASR Offline WER Gate**: `asr_e2e_wer_test` ≤ 1.50% corpus WER.
   - **ASR Streaming WER Gate**: `asr_stream_text_wer_test` ≤ 4.50% corpus WER, 0 divergence.
   - **TTS Parity Gate**: Audio waveform MSE & PESQ validation against Python baselines.
   - **Diarization DER Gate**: Speaker error rate validation on multi-talker fixtures.
   - **C ABI Integrity**: `capi_option_number_test`, `capi_session_options_test`, `capi_enum_sync_test`.
2. **Documentation & Release Tag**:
   - Publish full API reference documentation under `docs/`.
   - Tag `speech.cpp v1.0.0-stable`.

---

## 5. Verification & Acceptance Criteria

Every milestone must satisfy strict regression gates before landing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MASTER ACCEPTANCE CRITERIA                         │
├────────────────────────────────┬────────────────────────────────────────────┤
│ Metric / Gate                  │ Acceptance Requirement                     │
├────────────────────────────────┼────────────────────────────────────────────┤
│ Test Suite Pass Rate           │ 100% Green (All CTest targets passing)     │
│ Offline ASR WER (Moonshine)    │ ≤ 1.50% corpus WER on LibriSpeech fixtures │
│ Streaming ASR WER (Moonshine)  │ ≤ 4.50% corpus WER (0 word divergence)     │
│ VRAM Footprint (Multi-Session) │ ≤ 50 MB delta per additional session       │
│ Memory Safety Commit           │ 0 GB virtual memory over-commit spikes     │
│ Clean CUDA Compile Time        │ ≤ 4.0 minutes (auto-arch + ccache)         │
│ Incremental Rebuild Time       │ ≤ 10.0 seconds (ccache enabled)            │
│ Shared Library Artifact Count  │ Exactly 1 monolithic library (libspeech)   │
└────────────────────────────────┴────────────────────────────────────────────┘
```

---

## 6. Summary: The Final Unified Product

When this roadmap is fully executed, `speech.cpp` will stand as the **single most capable, efficient, and complete native speech framework in open source**:
- **All Voice Intelligence in One Binary**: TTS, Voice Cloning, STT/ASR, Streaming, VAD, Diarization, Separation, Alignment, Enhancement, Codecs, and MIDI.
- **Zero Redundant Code**: One unified Mel frontend, one tokenizer engine, one memory allocator, and one model execution graph runtime.
- **Unrivaled Efficiency**: Process-wide GPU weight sharing, topological graph arena reuse, and CUDA build acceleration.
- **Universal Ecosystem Support**: Direct support for C/C++, Rust, Python, TypeScript, and Swift.
