# Master FUSION Roadmap Plan: The Ultimate Convergence of `audio.cpp` & `transcribe.cpp` into `speech.cpp`

> **Document Status**: Authoritative Master Plan & Architectural Blueprint  
> **Target System**: `speech.cpp` — The Single, Fully-Fused Native Speech & Audio Intelligence Framework  
> **Date**: 2026-08-23  
> **Version**: 4.0 (Definitive Master Production Edition)

---

## 1. Executive Vision: The "Best of Both Worlds" Master Key

`speech.cpp` is the definitive convergence of two groundbreaking C++ audio AI codebases:
- **`audio.cpp`**: Broad multi-modal coverage across 50+ model families spanning Text-to-Speech (TTS), Voice Cloning (VC), Source Separation, Speaker Diarization, Voice Activity Detection (VAD), Forced Alignment, Audio Codecs, and MIDI music generation, with rich HTTP/REST servers, CLI tooling, and WebUI.
- **`transcribe.cpp`**: Production-grade, high-accuracy Automatic Speech Recognition (ASR) engine with strict numerical parity, speculative decoding, size-aware C ABI discipline, exception safety, 4-state streaming machines, and golden WER regression gates.

### The Master Fusion Principle
> **"The two projects learn from each other in parallel — each is the other's teacher and student — and merging them improves them both at the same time."**

This roadmap defines the **complete, end-state structural fusion** of both systems into `speech.cpp`. Rather than maintaining parallel runtimes behind a bridge adapter, `speech.cpp` systematically absorbs the superior components from each side, upgrades the entire engine with the lessons learned, and eliminates all redundant or duplicate code.

---

## 2. Architectural Evolution: From Bridged Coexistence to Monolithic Fusion

### 2.1 Current State: Coexistence with Inter-Engine Bridge (Phases 0–6)
Currently, `speech.cpp` shares a unified low-level GGML substrate, memory safety allocators, build toolchain, and dual public C ABIs (`audiocpp.dll` and `transcribe.dll`), but maintains two parallel model layers connected via `transcribe-arch-adapter.cpp`:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT STATE: BRIDGED COEXISTENCE                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Dual Public ABIs:  audiocpp.dll (Multi-Task)      │  transcribe.dll (STT + Stream)      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Bridge Layer:      src/runtime/transcribe-arch-adapter.cpp                              │
│                    (buffers PCM, adapts chunk granularity, maps vtables to Arch traits) │
├─────────────────────────────────────────┬───────────────────────────────────────────────┤
│ Parallel Models:   src/models/*         │  src/runtime/arch/*                           │
│                    (50+ Audio Models)   │  (18 STT Architectures)                       │
├─────────────────────────────────────────┼───────────────────────────────────────────────┤
│ Parallel Frontends:src/framework/audio/ │  src/runtime/transcribe-mel.cpp               │
│                    (kaldi_fbank.cpp)    │  (transcribe-kaldi-fbank.cpp)                 │
├─────────────────────────────────────────┼───────────────────────────────────────────────┤
│ Parallel Toks:     external/llama_tok   │  src/runtime/transcribe-tokenizer.cpp         │
│                    external/sentencep   │  (<tok> JSON Tokenizer)                       │
├─────────────────────────────────────────┼───────────────────────────────────────────────┤
│ Parallel Modules:  src/framework/modules│  src/runtime/{causal_lm, conformer, sanm}    │
├─────────────────────────────────────────┴───────────────────────────────────────────────┤
│ Unified Substrate: Single GGML • SharedWeightRegistry • BackendWeightStore (16MB Cap)   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Target State: Fully Fused Monolithic Engine (`libspeech`)
The target state eliminates the adapter layer, converges the frontends and tokenizers into unified shared modules, consolidates duplicate ASR families into canonical high-throughput implementations, and unifies the public surface into a single monolithic library (`libspeech`):

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          TARGET STATE: FULLY FUSED SPEECH.CPP                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                      Unified Public C ABI: libspeech.so / speech.dll                    │
│           (Single Monolithic Shared Library with Opaque Handles & C++ Guards)           │
│           • speech_* Universal API (All 14 Audio Tasks + Universal Progress)            │
│           • Backward-Compatible Shims: audiocpp.h & transcribe.h (Zero Breakdown)       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                               Unified Task Session Layer                                │
│   • SpeechSession Base (Graph Execution + 4-State Streaming Machine + RAII Cleanup)     │
│   • Multi-Task Pipeline Engine (VAD Chunking + ASR + TTS + Diarization Orchestration)   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                            Canonical Fused Model Subsystems                             │
│   ┌─────────────────────────────┬─────────────────────────────┬─────────────────────┐   │
│   │         TTS & Voice         │          ASR & STT          │ Audio Intelligence  │   │
│   │  • IndexTTS2, F5-TTS        │  • Whisper (HF 5.x Seek)    │ • HTDemucs (Sep)    │   │
│   │  • Qwen3-TTS (Top-P Guard)  │  • Moonshine (Offline/Str)  │ • RoFormer (Sep)    │   │
│   │  • CosyVoice, MiniMax-H3    │  • Parakeet-TDT (Batched)   │ • Sortformer v2(Dia)│   │
│   │  • MOSS, Kokoro, PocketTTS  │  • Qwen3-ASR (Fused SpecDec)│ • DFN2 / RNNoise    │   │
│   │  • Seed-VC (Gallocr 18x)    │  • Voxtral (4-State Stream) │ • MMS-300M (Align)  │   │
│   └─────────────────────────────┴─────────────────────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                         Unified Framework Shared Modules Layer                          │
│   • Mel & Audio Frontend: SIMD MelFrontend + Precomputed Kaldi Filterbanks (Single Path)│
│   • Tokenizer Hub: Unified Dispatcher (BPE, Byte-Level BPE, SentencePiece, WordPiece)   │
│   • Audio Chunking & VAD: Native vad::plan + Deterministic Global Timestamp Re-stitching│
│   • Codec Hub: Mimi, MioCodec, EnCodec, SNAC, DAC, Vocos (Shared Zero-Copy Modules)     │
│   • Core Modules: Shared Conformer, Causal-LM, SAN-M, RoPE, SDPA, SwiGLU, LayerNorm     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                        High-Performance Memory & Compute Layer                          │
│   • SharedWeightRegistry (Process-Wide VRAM Sharing across All Sessions: ~34MB/session) │
│   • ggml_gallocr Topological Arena Reuse (18x VRAM Reduction, Zero Dynamic Reallocs)    │
│   • BackendWeightStore 16MB Metadata Pool Budget Cap (Zero Virtual Memory Over-commit)  │
│   • Fused SwiGLU & Packed QKV Linear Projections (30% matmul reduction)                │
│   • Volta sm_70 to Blackwell sm_120 CUDA Graph Capture & Stream Paging                  │
│   • Single Pinned GGML (CPU AVX2/AVX512/ARM • CUDA • HIP/ROCm • Vulkan • Metal • SYCL)  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Exhaustive Component Fusion & Deduplication Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPONENT DEDUPLICATION MATRIX                            │
├──────────────────────┬──────────────────────┬──────────────────────┬───────────────────┤
│ Subsystem Domain     │ audio.cpp Heritage   │ transcribe Heritage  │ Target Fusion     │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **Audio Frontends**  │ kaldi_fbank.cpp      │ transcribe-mel.cpp   │ **SIMD MelFront** │
│                      │ mel_frontend.cpp     │ transcribe-kaldi     │ Precomputed filter│
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **Tokenizer Engine** │ llama_tokenizer      │ <tok> JSON parser    │ **Unified Tok**   │
│                      │ sentencepiece        │ (bpe/multilingual)   │ Single dispatcher │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **Qwen3-ASR**        │ Batched decode graph │ Speculative decode   │ **Fused Qwen3**   │
│                      │ src/models/qwen3/    │ arch/qwen3_asr/      │ Batch + Spec-Dec  │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **Voxtral Realtime** │ Parallel host front  │ 4-State Stream Mach  │ **Fused Voxtral** │
│                      │ src/models/voxtral/  │ arch/voxtral/        │ Cache-aware Stream│
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **Parakeet-TDT**     │ RNNT joint loop      │ Batched joint window │ **Fused Parakeet**│
│                      │ community_models/    │ arch/parakeet/       │ Multi-frame window│
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **SenseVoice**       │ Event tags / CTC     │ SAN-M optimization   │ **Fused Sense**   │
│                      │ src/models/sense/    │ arch/sensevoice/     │ Rich event tags   │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **FunASR Nano**      │ Packed QKV / SwiGLU  │ Static Arch trait    │ **Fused FunASR**  │
│                      │ src/models/funasr/   │ arch/funasr_nano/    │ Batched Prefill   │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **Neural Codecs**    │ Mimi / MioCodec      │ EnCodec / Vocos      │ **Unified Codec** │
│                      │ (Standalone runtime) │ (Arch-specific)      │ Modular Hub       │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **Conformers/SAN-M** │ framework/modules/   │ src/runtime/sanm/    │ **Shared Modules**│
│                      │ (Per-model copies)   │ src/runtime/conf/    │ Standard Library  │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────┤
│ **Public C ABI**     │ audiocpp.dll         │ transcribe.dll       │ **libspeech**     │
│                      │ (14 voice tasks)     │ (STT + WER gates)    │ Single artifact   │
└──────────────────────┴──────────────────────┴──────────────────────┴───────────────────┘
```

### 3.1 Audio Frontends (Mel Spectrogram & Filterbanks)
- **Current Duplicate Files**:
  - `src/framework/audio/kaldi_fbank.cpp` & `src/framework/audio/mel_frontend.cpp` (audio.cpp).
  - `src/runtime/transcribe-mel.cpp` & `src/runtime/transcribe-kaldi-fbank.cpp` (transcribe.cpp).
- **Superior Features Retained**:
  - `transcribe.cpp`'s frontend features SIMD vectorization, precomputed triangular filterbank matrices, band-skip optimizations, and threaded scalar paths certified by `asr_e2e_wer_test` (1.45% WER).
- **Consolidation Target**:
  - Fused into `include/engine/framework/audio/mel_frontend.h` and `src/framework/audio/mel_frontend.cpp`.
  - Delete `src/runtime/transcribe-mel.*` and `src/runtime/transcribe-kaldi-fbank.*`.
  - Wire all TTS (F5-TTS, CosyVoice, IndexTTS2) and ASR models to this unified implementation.

### 3.2 Tokenizer Engine
- **Current Duplicate Files**:
  - `external/llama_tokenizer/` + `external/sentencepiece/` (audio.cpp).
  - `src/runtime/transcribe-tokenizer.cpp` (transcribe.cpp).
- **Superior Features Retained**:
  - `transcribe.cpp`'s `<tok>` JSON tokenizer handles complex multilingual byte fallbacks with zero external dependencies.
  - `audio.cpp`'s tokenizer handles embedded GGUF tokenizer metadata, binary SentencePiece models, and HuggingFace `tokenizer.json` for 40+ TTS families.
- **Consolidation Target**:
  - Unified dispatcher in `include/engine/framework/text/tokenizer.h` and `src/framework/text/tokenizer.cpp`.
  - Single API parsing GGUF vocabulary, SentencePiece `.model`, and HuggingFace JSON tokenizers.

### 3.3 Neural Codecs Hub (`engine::codecs`)
- **Supported Codecs**:
  - **MimiCodec** (12.5Hz, 8 codebooks, Moshi/Mimi audio-language backbone).
  - **MioCodec** (24kHz acoustic tokenizer for MioTTS).
  - **EnCodec** (24kHz / 48kHz, multi-bandwidth RVQ for MOSS / AudioCraft).
  - **SNAC** (Multi-scale hierarchical neural codec for FishAudio / Kokoro / OuteTTS).
  - **DAC** (Descript Audio Codec, 44.1kHz high-fidelity vocoding).
  - **Vocos** (ISTFT neural vocoder for CosyVoice / F5-TTS / VibeVoice).
- **Consolidation Target**:
  - Standardized under `include/engine/framework/codecs/codec.h` with common `encode()` / `decode()` virtual interfaces.
  - Direct tensor sharing with `SharedWeightRegistry` to avoid duplicate codec VRAM allocations when running TTS pipelines.

### 3.4 Shared Core Neural Modules
- **Current Duplicate Files**:
  - `src/runtime/causal_lm/` <-> `src/framework/modules/self_attention/` & `rope/`.
  - `src/runtime/conformer/` & `granite_conformer/` <-> `src/framework/modules/convolutions/`.
  - `src/runtime/sanm/` <-> `src/framework/modules/layers/`.
- **Consolidation Target**:
  - Migrate transcribe's optimized conformer / SAN-M routines into `src/framework/modules/`.
  - Fused SwiGLU, RMSNorm, LayerNorm, Rotary Positional Embeddings (RoPE), and Scaled Dot-Product Attention (SDPA) standardized into single shared kernels.

---

## 4. Universal Public C ABI Specification (`speech.h` / `libspeech`)

The public interface consolidates both `audiocpp.h` and `transcribe.h` into a clean, size-aware, exception-contained C ABI.

```c
#ifndef SPEECH_H
#define SPEECH_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- API Export Macros ---
#if defined(_WIN32)
#  if defined(SPEECH_BUILD_DLL)
#    define SPEECH_API __declspec(dllexport)
#  else
#    define SPEECH_API __declspec(dllimport)
#  endif
#else
#  define SPEECH_API __attribute__((visibility("default")))
#endif

// --- Status and Error Handling ---
typedef enum speech_status {
    SPEECH_OK                    = 0,
    SPEECH_ERR_INVALID_ARG       = -1,
    SPEECH_ERR_OOM               = -2,
    SPEECH_ERR_BACKEND           = -3,
    SPEECH_ERR_GGUF              = -4,
    SPEECH_ERR_MODEL_NOT_FOUND   = -5,
    SPEECH_ERR_ABORTED           = -6,
    SPEECH_ERR_OUTPUT_TRUNCATED  = -7,
    SPEECH_ERR_UNSUPPORTED_TASK  = -8,
} speech_status;

// --- Task Enumeration ---
typedef enum speech_task {
    SPEECH_TASK_ASR              = 0,
    SPEECH_TASK_TTS              = 1,
    SPEECH_TASK_VAD              = 2,
    SPEECH_TASK_DIARIZATION      = 3,
    SPEECH_TASK_SEPARATION       = 4,
    SPEECH_TASK_ALIGNMENT        = 5,
    SPEECH_TASK_VOICE_CONVERSION = 6,
    SPEECH_TASK_DENOISE          = 7,
    SPEECH_TASK_SUPER_RESOLUTION = 8,
    SPEECH_TASK_MIDI             = 9,
    SPEECH_TASK_GENERATION       = 10,
} speech_task;

// --- Hardware Backends ---
typedef enum speech_backend_type {
    SPEECH_BACKEND_AUTO          = 0,
    SPEECH_BACKEND_CPU           = 1,
    SPEECH_BACKEND_CUDA          = 2,
    SPEECH_BACKEND_HIP           = 3,
    SPEECH_BACKEND_VULKAN        = 4,
    SPEECH_BACKEND_METAL         = 5,
    SPEECH_BACKEND_SYCL          = 6,
} speech_backend_type;

// --- Opaque Handles ---
typedef struct speech_model   speech_model;
typedef struct speech_session speech_session;
typedef struct speech_stream  speech_stream;
typedef struct speech_vad     speech_vad;

// --- Progress Callback ---
typedef struct speech_progress_info {
    size_t      struct_size;
    float       progress;          // 0.0 .. 1.0
    const char *stage_name;        // "encoding", "decoding", "synthesizing"
    int64_t     units_completed;
    int64_t     units_total;
} speech_progress_info;

typedef bool (*speech_progress_cb)(const speech_progress_info *info, void *user_data);

// --- Core Lifecycle APIs ---
SPEECH_API speech_model * speech_model_load(const char *model_path, const char *options_json);
SPEECH_API void           speech_model_free(speech_model *model);
SPEECH_API speech_session*speech_session_create(speech_model *model, speech_task task, const char *session_options_json);
SPEECH_API void           speech_session_free(speech_session *session);
SPEECH_API void           speech_session_set_progress_callback(speech_session *session, speech_progress_cb cb, void *user_data);

// --- Unified Inference APIs ---
SPEECH_API speech_status  speech_run_asr(speech_session *session, const float *samples, size_t n_samples, const char *run_options_json, char **out_text);
SPEECH_API speech_status  speech_run_tts(speech_session *session, const char *text, const char *run_options_json, float **out_samples, size_t *out_n_samples, int *out_sample_rate);
SPEECH_API speech_status  speech_run_vad(const float *samples, size_t n_samples, int sample_rate, const char *vad_options_json, float **out_segments, size_t *out_n_segments);
SPEECH_API speech_status  speech_run_diarize(speech_session *session, const float *samples, size_t n_samples, const char *options_json, char **out_rttm_json);
SPEECH_API speech_status  speech_run_separate(speech_session *session, const float *samples, size_t n_samples, const char *options_json, float ***out_stems, size_t *out_n_stems, size_t *out_samples_per_stem);
SPEECH_API speech_status  speech_run_align(speech_session *session, const float *samples, size_t n_samples, const char *text, const char *options_json, char **out_alignment_json);
SPEECH_API speech_status  speech_run_denoise(const float *samples, size_t n_samples, int sample_rate, const char *options_json, float **out_samples, size_t *out_n_samples);
SPEECH_API speech_status  speech_run_super_resolve(const float *samples, size_t n_samples, int in_sample_rate, int target_sample_rate, float **out_samples, size_t *out_n_samples);

// --- Streaming APIs (4-State Engine) ---
SPEECH_API speech_stream *speech_stream_start(speech_session *session, const char *stream_options_json);
SPEECH_API speech_status  speech_stream_push(speech_stream *stream, const float *samples, size_t n_samples);
SPEECH_API speech_status  speech_stream_pull(speech_stream *stream, char **out_partial_text, float **out_audio, size_t *out_n_samples);
SPEECH_API speech_status  speech_stream_finish(speech_stream *stream);
SPEECH_API void           speech_stream_free(speech_stream *stream);

// --- Hardware & Utility APIs ---
SPEECH_API int            speech_device_count(speech_backend_type backend);
SPEECH_API speech_status  speech_device_info(speech_backend_type backend, int device_id, char **out_info_json);
SPEECH_API bool           speech_backend_available(speech_backend_type backend);
SPEECH_API int            speech_read_wav(const char *path, float **out_samples, size_t *out_n_samples, int *out_sample_rate);
SPEECH_API int            speech_write_wav(const char *path, const float *samples, size_t n_samples, int sample_rate, int n_channels);

// --- Memory & String Management ---
SPEECH_API void           speech_free_string(char *str);
SPEECH_API void           speech_free_audio(float *samples);
SPEECH_API void           speech_free_stems(float **stems, size_t n_stems);
SPEECH_API const char *   speech_version(void);
SPEECH_API const char *   speech_build_id(void);

#ifdef __cplusplus
}
#endif
#endif // SPEECH_H
```

---

## 5. End-to-End Pipeline Orchestration Architectures

`speech.cpp` enables complex, composite multi-model pipelines within a single process without external IPC or disk serialization:

### 5.1 Real-Time Voice-to-Voice (S2S: ASR -> LLM -> Zero-Shot TTS)
```
  Microphone PCM Stream (16kHz Mono)
           │
           ▼
┌──────────────────────────────┐
│  Voxtral Realtime /          │ ──► Low-Latency Word Streaming Tokens
│  Moonshine Streaming ASR     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Causal Audio-LLM           │ ──► Synthesized Text Stream
│   Text Generation            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   IndexTTS2 / MOSS / Qwen3   │ ──► High-Fidelity Audio Codec Tokens
│   Zero-Shot TTS              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Mimi / MioCodec / Vocos     │ ──► Continuous Output PCM (24kHz/48kHz)
│  Neural Audio Vocoder        │
└──────────────────────────────┘
```

### 5.2 Long-Form Multi-Speaker Diarized Transcription
```
  Raw Long-Form Audio File (e.g. 2-Hour Meeting)
           │
           ▼
┌──────────────────────────────┐
│  Native Silero VAD           │ ──► Active Speech Spans ([start_ms, end_ms])
│   (vad::detect)              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Native vad::plan             │ ──► Bounded Utterances (≤ 30s) with 250ms Padding
│ Greedy Chunk Planner         │
└──────────────┬───────────────┘
               │
               ├───────────────────────────────────────────┐
               ▼                                           ▼
┌──────────────────────────────┐             ┌──────────────────────────────┐
│ Sortformer v2 Diarization    │             │ Whisper / Qwen3-ASR          │
│ Speaker Clustering           │             │ Batched Decode Graph         │
└──────────────┬───────────────┘             └─────────────┬────────────────┘
               │                                           │
               └─────────────────────┬─────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Deterministic Global Timestamp & Speaker Tag Re-stitching                 │
│ (`offset_chunk_results` + `rebuild_full_text`)                            │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  Diarized Transcript JSON / RTTM with Millisecond Precision
```

---

## 6. Phased Master Fusion Execution Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MASTER FUSION EXECUTION PLAN                       │
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
│ Phase 10: Unified C ABI (libspeech) & Backward Compatibility Shims
│ Phase 11: Zero-Dependency Language Bindings (Rust, Python, TS, Swift)
│ Phase 12: Comprehensive Golden Regression Verification & Release 1.0
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 7: Common Audio Frontend & Tokenizer Unification
**Target Completion**: Sprint 7  
**Goal**: Eliminate duplicate mel-spectrogram extractors, filterbank tables, and tokenizer parsing routines across the codebase.

- **Tasks**:
  1. **SIMD Mel Frontend Convergence**:
     - Fuse `transcribe-mel.cpp` and `kaldi_fbank.cpp` into `src/framework/audio/mel_frontend.cpp`.
     - Implement standard 80-band, 128-band Mel, and Kaldi 80-dimensional filterbanks with precomputed triangular weights and SIMD dot products.
     - Wire all TTS models (F5-TTS, CosyVoice, IndexTTS2) and ASR models to `include/engine/framework/audio/mel_frontend.h`.
     - Delete `src/runtime/transcribe-mel.*` and `src/runtime/transcribe-kaldi-fbank.*`.
  2. **Unified Tokenizer Dispatcher**:
     - Create `include/engine/framework/text/tokenizer.h` and `src/framework/text/tokenizer.cpp`.
     - Support embedded GGUF vocabularies, SentencePiece `.model` binaries, and HuggingFace `tokenizer.json` from a single unified class.
     - Delete `src/runtime/transcribe-tokenizer.*` and redundant wrappers.
  3. **Verification**:
     - Run `test_audio_dsp`, `test_subtitle_formatter`, and `test_tokenizer_parity` ensuring bit-exact parity.

---

### Phase 8: Overlapping Model Family Fusion
**Target Completion**: Sprint 8  
**Goal**: Eliminate duplicate implementations of Qwen3-ASR, Voxtral Realtime, SenseVoice, FunASR Nano, and Parakeet-TDT.

- **Tasks**:
  1. **`Qwen3-ASR` Canonicalization**:
     - Port speculative decoding from `src/runtime/arch/qwen3_asr/` into `src/models/qwen3_asr/session.cpp`.
     - Retain `DecodeGraphBatched`, `generate_batch`, and packed projections.
     - Delete `src/runtime/arch/qwen3_asr/`.
  2. **`Voxtral Realtime` Canonicalization**:
     - Port the 4-state streaming machine (`IDLE`, `FEEDING`, `FINALIZING`, `RESETTING`) into `src/models/voxtral_realtime/session.cpp`.
     - Delete `src/runtime/arch/voxtral_realtime/`.
  3. **`SenseVoice` & `FunASR Nano` Canonicalization**:
     - Port SAN-M blocks and packed QKV projections into `src/models/sense_asr/` and `src/models/fun_asr_nano/`.
     - Delete `src/runtime/arch/sensevoice/` and `src/runtime/arch/funasr_nano/`.
  4. **`Parakeet-TDT` Canonicalization**:
     - Promote Parakeet from `community_models/` to core `src/models/parakeet_tdt/`.
     - Integrate frame-windowed greedy joint batching (`JointGraphBatch`).
     - Delete `src/runtime/arch/parakeet/`.
  5. **Verification**:
     - Execute `asr_e2e_wer_test` and `asr_stream_text_wer_test` certifying zero WER regression.

---

### Phase 9: Migration of Remaining STT Arches to Engine Core
**Target Completion**: Sprint 9  
**Goal**: Move all remaining transcribe architectures (Whisper, Moonshine, Canary, Granite, Cohere, GigaAM, MedASR) into `src/models/` as first-class engine sessions.

- **Tasks**:
  1. **Native Model Sessions**:
     - Create `src/models/whisper/` (`session.cpp`, `runtime.cpp`, `assets.cpp`) with HF 5.x seek continuation fix.
     - Create `src/models/moonshine/` and `src/models/moonshine_streaming/` with native `IVoiceTaskSession` integration.
     - Migrate Canary, Canary-Qwen, Granite, Granite-NAR, Cohere, GigaAM, and MedASR to `src/models/`.
  2. **Decommission Bridge Adapter**:
     - Delete directory `src/runtime/arch/` completely.
     - Delete `src/runtime/transcribe-arch-adapter.cpp`, `transcribe-arch-adapter.h`, and `transcribe-arch.cpp`.
     - Update `CMakeLists.txt` so all model targets compile directly from `src/models/`.
  3. **Verification**:
     - Full test suite compile and execution on CPU and CUDA.

---

### Phase 10: Unified Public C ABI (`libspeech`) & Compatibility Shims
**Target Completion**: Sprint 10  
**Goal**: Deliver a single monolithic shared library exporting the entire speech intelligence surface.

- **Tasks**:
  1. **`libspeech` Implementation**:
     - Create `include/speech/speech.h` and `src/capi/speech_capi.cpp`.
     - Consolidate all 14 task APIs, VAD, Diarization, Streaming, Progress Callbacks, and Memory Management into `speech_*` symbols.
     - Configure `SPEECH_SHARED_EMBED=ON` compiling static GGML archives directly into `speech.dll` / `libspeech.so` with hidden internal symbols.
  2. **Backward-Compatibility Shims**:
     - Provide `include/audiocpp.h` inline forwarding to `speech_*`.
     - Provide `include/transcribe/transcribe.h` inline forwarding to `speech_*`.
     - Generate compatibility import libraries (`audiocpp.lib`, `transcribe.lib`) on Windows.
  3. **Verification**:
     - Link legacy `capi_test.exe`, `abi_bridge_hello.exe`, and new `speech_capi_test.exe` against `speech.dll`.

---

### Phase 11: Zero-Dependency Language Bindings (`dynload`)
**Target Completion**: Sprint 11  
**Goal**: Deliver high-level language bindings with zero local compilation requirements.

- **Tasks**:
  1. **Rust Binding (`bindings/rust/speech-rs`)**:
     - Pure dynamic loading crate using `libloading` for `libspeech.so` / `speech.dll`.
     - Safe Rust idiomatic wrappers for TTS, ASR, Streaming, VAD, and Progress reporting.
  2. **Python Binding (`bindings/python/speechcpp`)**:
     - Lightweight `ctypes` wrapper distributed via PyPI wheel with bundled binary.
  3. **TypeScript Binding (`bindings/typescript/speech-node`)**:
     - Node.js / Bun / Deno binding using `koffi` (pure C-FFI without node-gyp).
  4. **Swift Binding (`bindings/swift/SpeechKit`)**:
     - Swift Package Manager (SPM) wrapper with C module map.

---

### Phase 12: Comprehensive Golden Regression Verification & Release 1.0
**Target Completion**: Sprint 12  
**Goal**: 100% test coverage, strict WER/DER regression verification, and Release 1.0 tagging.

- **Tasks**:
  1. **Automated Verification Suite**:
     - **ASR Offline WER Gate**: `asr_e2e_wer_test` <= 1.50% corpus WER on LibriSpeech.
     - **ASR Streaming WER Gate**: `asr_stream_text_wer_test` <= 4.50% corpus WER, 0 divergence.
     - **TTS Parity Gate**: Audio waveform MSE & PESQ validation against PyTorch baselines.
     - **Diarization DER Gate**: Speaker error rate validation on multi-talker fixtures.
     - **C ABI Integrity**: `capi_option_number_test`, `capi_session_options_test`, `capi_enum_sync_test`.
  2. **Documentation & Release Tag**:
     - Publish complete API reference documentation under `docs/`.
     - Tag `speech.cpp v1.0.0-stable`.

---

## 7. Master Verification & Acceptance Criteria

Every milestone must satisfy strict regression gates before landing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MASTER ACCEPTANCE CRITERIA                         │
├────────────────────────────────┬────────────────────────────────────────────┤
│ Metric / Gate                  │ Acceptance Requirement                     │
├────────────────────────────────┼────────────────────────────────────────────┤
│ Test Suite Pass Rate           │ 100% Green (All CTest targets passing)     │
│ Offline ASR WER (Moonshine)    │ <= 1.50% corpus WER on LibriSpeech fixtures│
│ Streaming ASR WER (Moonshine)  │ <= 4.50% corpus WER (0 word divergence)    │
│ VRAM Footprint (Multi-Session) │ <= 50 MB delta per additional session      │
│ Memory Safety Commit           │ 0 GB virtual memory over-commit spikes     │
│ Clean CUDA Compile Time        │ <= 4.0 minutes (auto-arch + ccache)        │
│ Incremental Rebuild Time       │ <= 10.0 seconds (ccache enabled)           │
│ Shared Library Artifact Count  │ Exactly 1 monolithic library (libspeech)   │
└────────────────────────────────┴────────────────────────────────────────────┘
```

---

## 8. Summary: The Definitive Unified Speech Framework

When this roadmap is fully executed, `speech.cpp` will stand as the **single most capable, efficient, and complete native speech framework in open source**:
- **All Voice Intelligence in One Binary**: TTS, Voice Cloning, STT/ASR, Streaming, VAD, Diarization, Separation, Alignment, Enhancement, Codecs, and MIDI.
- **Zero Redundant Code**: One unified Mel frontend, one tokenizer engine, one memory allocator, and one model execution graph runtime.
- **Unrivaled Efficiency**: Process-wide GPU weight sharing, topological graph arena reuse, and CUDA build acceleration.
- **Universal Ecosystem Support**: Direct support for C/C++, Rust, Python, TypeScript, and Swift.
