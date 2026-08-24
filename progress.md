# Progress — Unified_Audio.cpp (speech.cpp ggml fork) merge & improve

Status snapshot: **Upstream audio.cpp main (d25ffac, 28 commits total) merged cleanly into speech.cpp. Phase 1 Allocator Hardening applied. Phase 2 Toolchain Modernization & Build Provenance landed. Phase 3 Native Long-Form VAD Chunk Planning & Public C ABI integrated. Phase 4 Process-Wide SharedWeightRegistry, Sortformer v2 Diarization package, Batched Offline ASR Decoders (Qwen3-ASR, Voxtral Realtime, Citrinet, VibeVoice, Higgs Audio), and universal audiocpp C ABI subsystem + progress callbacks fully implemented and verified. All 100 CPU core tests passing 100% green. Master architectural blueprint established in [FUSION_ROADMAP_PLAN.md](FUSION_ROADMAP_PLAN.md).** Date: 2026-08-25

## Repo layout (important, non-obvious)
`Unified_Audio.cpp/` is a **plain container directory with no git repo of its
own**. It holds exactly three things, each an independent repository:

| Folder | Role |
|---|---|
| `speech.cpp/` | the active development repo (the ggml/audio.cpp fork). **All merge work, and this log, live here.** Remote: `NairoDorian/speech.cpp`, upstream `0xShug0/audio.cpp`. |
| `audio.cpp/` | upstream reference — read from, not developed in (pulled to `d25ffac`) |
| `transcribe.cpp/` | merge source — read from, not developed in (pulled to `0df989a`). |
| `audio_cunba/` & `transcribe_cunba/` | hardened reference trees containing allocator fixes, VAD chunk planning, shared weights, batched decoders, C ABI, and build acceleration. |

Build trees are scratch dirs under `C:/Users/Z/AppData/Local/Temp/opencode/`:
`sp_bridge` (CPU, full model set, unified ABI + arches, tests), `sp_cuda`
(CUDA, core set, ABI/arches OFF), `audiocpp_flashsr` (audio.cpp reference),
`build-cpu-core` (local MSVC CPU core test suite), `build-cpu-asr` (local MSVC ASR test/executable build).

## Overall progress (toward "Unified_Audio transcribes on CPU")
| Area | Status | % |
|---|---|---|
| Merge: ggml convergence (pin 8c63e709 + patches 0001–0006), CPU+CUDA certified | Done (prior sessions) | 100% |
| Merge: Upstream audio.cpp main synchronization (`d25ffac`) | Done (clean merge, PR #299 & #301 resolved) | 100% |
| Memory: Phase 1 Allocator Hardening (16MB cap, WavLM gallocr, Qwen3 runaway, DFN2) | Done (certified in engine) | 100% |
| Toolchain: Phase 2 Modernization & Build Provenance (ccache, transcribe-build-info, version.rc) | Done (certified in build scripts & DLL) | 100% |
| Long-form: Phase 3 Native VAD Chunk Planning & Re-stitching (`vad_plan`, `vad_merge`) | Done (native Silero + Energy VAD, C ABI) | 100% |
| Optimization: Phase 7.4 Shared Weight Registry & Scoped Activation | Done & Verified (`test_shared_weight_vram`) | 100% |
| Scaling: Phase 7.5 Offline Batched ASR Decode across all 16 Arch Adapter slots | Done & Verified (`test_batch_dispatch`, C ABI) | 100% |
| Registry: Phase 7.8 Unified Family Registry v1 | Done & Verified (`family_registry_unit`) | 100% |
| Regression: Phase 7.1 Port transcribe.cpp Test Suite & Fixtures | Done (51 TUs, 87 golden fixtures, 13 GGUF fixtures) | 100% |
| Remediation: Phase 7.3 Defect D1 (sniff dispatch) & D2 (pre-clear) | Done & Verified (`test_adapter_sniff_dispatch`) | 100% |
| **Phase 8: Contract Convergence & Exception Boundary** | **Done & Verified (`StreamingSessionBase`, `RunControl`, `StreamChunker`, 100% C ABI Exception Guards)** | **100%** |
| **Phase 9: Unified Mel & Tokenizer Subsystems** | **Done & Verified (`MelExtractor`, `TokenizerHub`, `FrontendSpec`, `IAudioCodec`, Parity Tests)** | **100%** |
| **Phase 10: Attention & Conformer Module Fusion** | **Done & Verified (`sanm`, `shaw_attn`, `causal_lm_ops`, Bake-Off certified)** | **100%** |
| **Phase 11 W1a: Native Engine Moonshine (offline)** | **Done & Verified (`moonshine_engine_smoke_test`: engine-path WER 1.449% == arch 1/69 edits; batch + abort contracts)** | **100%** |
| Phase 11 W1b: Native Engine Moonshine-Streaming | Next increment — design map recorded in `walkthrough.md` | 0% |
| Specs: Phase 6 Whisper & Moonshine Model Spec Catalogs | Moonshine spec corrected + backed by native loader; Whisper still catalog-only (Phase 11 W2) | ~60% |
| ABI offline + streaming surface | Verified, real CTest gates | 100% |
| End-to-end ASR **offline text** (WER gate) | Done — 1.45% corpus WER (arch path); engine path now also 1/69 edits | 100% |
| **End-to-end ASR streaming text** | **Done — streamed 4.35% == offline 4.35%, divergence 0** | **100%** |
| Test suite status | **100/100 total (96 passed, 4 clean skips on unpinned weights) 100% green** | **100%** |
| **Completed increment** | **Upstream audio.cpp main synchronization (`d25ffac`) & Phase 11 W1a** | **DONE** |
| **Next increment** | **Phase 11 Wave W1b: Native Engine Moonshine-Streaming on `StreamingSessionBase`** | Ready |

## DONE this session (plan R12 records all of it)

### 0. Phase 11 Wave W1a — Native Engine Moonshine (offline), closed
First family migration of the arch layer onto the engine framework
(FUSION_ROADMAP_PLAN §8, §4.4 steps 5–7). New package `src/models/moonshine/`
(`graphs/assets/runtime/session` + internal headers in
`include/engine/models/moonshine/`): graph topology ported numerics-identically
from `src/runtime/arch/moonshine/`; weights via `core::BackendWeightStore`
(`SharedWeightRegistry` stays active); tokenizer via `TokenizerHub`
(`load_tokenizer_from_gguf`); abort/progress via `RunControl`.
Registered `audiocpp_add_model(moonshine ...)` + ASR composite; CLI/server/
WebUI/C ABI can now load `--family moonshine` GGUFs directly.
Findings: (1) pinned moonshine GGUFs carry **no** embedded config/tokenizer
sidecars — hparams live in `stt.*` metadata; fixed `model_specs/moonshine.json`
sources block (previous required mappings could never have loaded — a latent
Phase-6 defect surfaced by this wave). (2) Engine tensor shapes are logical
(PyTorch-order); `BackendWeightStore` reverses into ggml `ne`. (3)
TokenizerHub GGUF-BPE decode emits raw `▁` — the package post-processes to
spaces (hub change deferred). Gate `moonshine_engine_smoke_test`: registry
load by id + 4 LibriSpeech fixtures offline (WER ≤ 10% structural bound;
measured **1/69 edits == arch baseline**), ordered `run_batch`,
`request_abort()` unwinds `run()`. Arch copy untouched and still green.
Suite: 96/96 green.

### 1. Streaming ASR text validation — NEXT #1, closed
`tests/asr_stream_text_wer_test.cpp` + CTest gate `asr_stream_text_wer_test`:
streams the four LibriSpeech fixtures into **moonshine-streaming-tiny Q8_0**
(48 MB, MIT, `handy-computer/moonshine-streaming-tiny-gguf` — the exact model
the previous session hoped existed; arch already in
`src/runtime/arch/moonshine_streaming`) through the public C ABI with
odd-sized ~100–400 ms feeds, reads the final transcript from
`transcribe_stream_get_text().full_text`, and gates: streamed corpus WER
≤ 10%, offline corpus WER ≤ 10%, streamed-vs-offline divergence ≤ 3 words.

Measured (CPU): **streamed 4.35% == offline 4.35% (3/69: FORWARDED→VOTED +
"I AM"→"I'M"), divergence 0 words, streamed RTF 0.55.** Each fixture runs
offline then streaming on ONE session, so run/stream mode switching and
`stream_reset` are proven with real text. `scripts/fetch_asr_test_model.py`
became a pinned-model table (both gate models; `--only streaming` selects);
shared scoring lives in `tests/asr_test_text.h`. Report updated:
`docs/reports/asr_e2e_wer_gate.md`.

### 2. The three "environment/asset" failures — NEXT #2, all fixed, none was assets
- `model_spec_system_test` + `fun_asr_nano_assets_test`: model-spec
  resolution walks UP from the cwd, so they only ever passed from build
  trees inside the repo. Fixed with `WORKING_DIRECTORY` registrations.
- `server_model_installer_test` exposed **two real product bugs**, both fixed
  in `app/server/model_installer.{h,cpp}`:
  1. **`AUDIOCPP_PYTHON` never worked on Windows.** `cmd /c` strips the
     first+last quote when the command string starts with a quoted program
     path — proven with an isolated `std::system` repro. All helper commands
     now go through `run_shell_command()` (wraps the line in one extra quote
     pair; no-op for the unquoted default). The CTest registration passes
     CMake's `Python3_EXECUTABLE` via `AUDIOCPP_PYTHON` (on this machine bare
     `python` is the Microsoft Store stub).
  2. **Teardown raced detached workers.** A worker terminated by process
     exit inside `CreateProcess` leaves a permanently suspended child
     `cmd.exe` pinning inherited pipes — observed as CTest waiting 22 min on
     a 15 s test; the same race leaked ~40 `audiocpp-model-installer-*` temp
     dirs (all cleaned). Workers are now tracked + cancelled + joined in
     `~ModelInstaller` (first D7-teardown application to app-layer code).
     Test now passes in 2.7 s.

### 3. `asr_standalone_gguf_test` — NEXT #3, closed by correction
Filed for three sessions as "needs citrinet+hviske GGUFs". It does not: the
fixtures are synthetic (dummy safetensors → GGUF), and the failure was the
same cwd spec-resolution defect. `WORKING_DIRECTORY` registration fixed it;
the old download-and-pin recommendation is withdrawn. (A real citrinet/hviske
WER gate would be new, optional work — the plan's §5 Phase-5 corpus item.)

### 4. `scaled_dot_product_attention_test` skips without CUDA
It exists to pin the CUDA SDPA lowerings (R10) and hard-required a CUDA
device, failing CPU-only builds. Now probes `list_backend_devices()` and
skips (exit 2, `SKIP_RETURN_CODE 2`); stays a hard gate on CUDA builds.

### 5. Performance pass on transcribe.cpp runtime families
Closed the last depthwise-1D-conv im2col sites in the runtime and a couple
of decode/frontend hotspots, all with env-override kill switches and
numerical parity:
- **Granite + Granite-NAR encoders** (`src/runtime/arch/{granite,granite_nar}/encoder.cpp`):
  in-block depthwise conv now uses `ggml_conv_2d_dw_direct` instead of
  `conv_1d_dw_f32`'s B==1 im2col path (15x scratch expansion avoided; ~2.2 s
  of a 29 s CPU clip). Matches parakeet/canary/etc.
- **SAN-M FSMN branch** (`src/runtime/sanm/sanm.cpp`): B==1 single-shot path
  now uses direct dw conv (was im2col); B>1 already was. 11x expansion
  avoided for kernel=11. Affects sensevoice + fun_asr_nano arch.
- **Parakeet decoder** (`src/runtime/arch/parakeet/decoder.cpp`): frame-batched
  joint window (`JointGraphBatch`) fuses the RNN-T/TDT joint op across frames
  instead of per-frame dispatch.
- **Mel frontend** (`src/runtime/transcribe-mel.cpp`): nonzero mel-band skip
  (`fb_begin_`/`fb_end_`) across both FFT + scalar paths; scalar fallback
  threaded (`run_threaded`). Bit-exact.
This was the uncommitted `WORKSPACE` work, committed in two passes.

## Test suite state
**CPU build** (`sp_bridge`: full set, unified ABI + arches) — **58 tests,
0 failures, 100% green** (was 57 tests / 5 failures at session start).
Skips: `scaled_dot_product_attention_test` (no CUDA device — by design),
`parakeet_golden_transcription_test` / `parakeet_streaming_transcription_test`
(their own model-download skip contract). Newly green this session:
`model_spec_system_test`, `fun_asr_nano_assets_test`,
`server_model_installer_test`, `asr_standalone_gguf_test`, and the new
`asr_stream_text_wer_test`.

**CUDA build** (`sp_cuda`: core set, sm_89) — rebuilt with this session's
fixes; suite result recorded below when the run lands (expected: the same
three shared failures gone; WER gates not registered there — ABI/arches OFF).

## COMPLETED IN RECENT COMMITS
1. **Phase 1: Allocator Hardening & Memory Safety (`e760def`)**:
   - `BackendWeightStore`: Added `kMetadataPoolBudget = 16 MB` cap for `no_alloc=true` metadata pool.
   - `WavLMEncoder`: Switched to `ggml_gallocr` buffer reuse (18x peak memory reduction on 35s clips).
   - `Qwen3-TTS`: Added sampling runaway guard (`top_p = 0.8, temp = 0.8` defaults, `max_new_tokens = 1024` ceiling).
   - `DeepFilterNet2`: Lowered `segment_threshold` to `segment_samples` (48000) for overlap-add chunking.
2. **Phase 2: Toolchain Modernization & Build Provenance (`06962fb`)**:
   - Integrated `ccache` compiler launcher auto-detection into `scripts/build_windows.ps1` and `scripts/build_windows_hip.ps1`.
   - Configured `cmake/transcribe-build-info.h.in` and Windows `cmake/transcribe-version.rc.in` for `transcribe.dll`.
   - Implemented strong symbols `kTranscribeBuildId`, `transcribe_build_id()`, and `transcribe_version()`.
3. **Phase 3: Native Long-Form VAD Chunk Planning & Re-stitching (`5dcc27c`)**:
   - Implemented `src/runtime/transcribe-vad.h/.cpp` (`vad::plan`, `vad::params_present`, `vad::effective_mode`).
   - Implemented `src/runtime/transcribe-vad-integrate.h/.cpp` (`vad::detect_speech`, `vad::offset_chunk_results`, `vad::rollback_to`, `vad::rebuild_full_text`, `vad::run_with_vad`).
   - Added public C ABI `transcribe_vad` and `transcribe_free_vad`.
   - Added unit tests `tests/unittests/vad_plan_unit.cpp` and `tests/unittests/vad_merge_unit.cpp`.
4. **Phase 4: Shared Weight Registry, Sortformer v2 Package & Batched Offline Decoders (`13abbd7`, `fe1a307`, `4f0ae95`)**:
   - **`FunASR Nano` Packed QKV & Gate/Up Weights (`13abbd7`)**: Added `load_packed_rows` to `src/models/fun_asr_nano/decoder.cpp`, row-packing Q, K, V and Gate, Up projections at load time and switching decoder to `PackedQKV` + `PackedGateUp` for fused SwiGLU. Drops layer matmuls from 7 to 5 (197 to 113 in step graph).
   - **Process-Wide `SharedWeightRegistry` & `ScopedWeightShareKey` (`fe1a307`)**: Created `include/engine/framework/core/shared_weight_registry.h` and wired `BackendWeightStore` (`upload_shared`, `bind_to_shared`, `upload_pending_into`).
   - **`Sortformer` Diarization v2 Package (`fe1a307`)**: Added `sortformer_diar_4spk_v2_q8_0` package to `model_specs/sortformer_diar.json`.
   - **Offline Batched ASR Decoders (`4f0ae95`)**:
     - `IOfflineVoiceTaskSession::run_batch` interface added in `include/engine/framework/runtime/session.h`.
     - `Qwen3-ASR`: `DecodeGraphBatched` and `generate_batch` in `src/models/qwen3_asr/thinker.cpp` & `session.cpp`.
     - `Voxtral Realtime`: parallel frontends and compute graph resets in `src/models/voxtral_realtime/session.cpp`.
     - `Citrinet ASR`: batched CTC graph in `src/models/citrinet_asr/runtime.cpp` & `session.cpp`.
     - `VibeVoice ASR`: `VibeVoiceDecoderCachedStepGraphBatched` in `src/models/vibevoice_asr/text_decoder.cpp` & `session.cpp`.
     - `Higgs Audio STT`: `DecodeGraphBatched` in `src/models/higgs_audio_stt/text_decoder.cpp` & `session.cpp`.
5. **Phase 5: Universal `audiocpp` C ABI Subsystem & Progress Reporting (`b9f53ab`)**:
   - Integrated full C ABI surface (`capi/include/audiocpp.h` and `capi/src/audiocpp_capi.cpp`) with 46 exported C APIs covering all 14 audio tasks.
   - Built monolithic shared library `audiocpp.dll` / `libaudiocpp.so` with hidden internal GGML symbols and embedded Windows `VS_VERSION_INFO` resource.
   - Implemented `ProgressInfo`, `ProgressCallback`, and `ProgressCanceled` in `include/engine/framework/runtime/session.h`.
   - Implemented `RuntimeSessionBase::set_progress_callback` and `emit_progress` in `src/framework/runtime/session_base.cpp` and wired into `audiocpp_set_progress_callback`.
   - Integrated embedded asset subsystem (`include/engine/framework/assets/embedded.h` & `src/framework/assets/embedded.cpp`).
   - Added unit test suites `capi_option_number_test`, `capi_session_options_test`, and `capi_enum_sync_test` (56/56 tests passing 100% green).
6. **Phase 6: Whisper GPU Cleanup, Arch Sync, and Model Spec Catalog Integration**:
   - **GPU Buffer Cleanup across 18 Arches**: Integrated `cleanup_gpu` lambda across all `src/runtime/arch/*/model.cpp` files (Whisper, Moonshine, Parakeet, Canary, Voxtral, SenseVoice, etc.), eliminating GPU/KV-cache memory leaks across repeated runs on Windows.
   - **Whisper `bin_load.cpp` Token Tables**: Added separate English and Multilingual suppress-token tables and `synthesize_bin_suppress_tokens()`.
   - **Parakeet Batched Joint Window**: Added multi-frame greedy joint decoding graph amortization.
   - **Model Spec Catalog Additions**: Added schema-v1 catalog specifications for `whisper.json` (16 packages: tiny, base, small, medium, large-v3, large-v3-turbo, and .en variants), `moonshine.json` (6 packages: tiny, base, small in q8_0/f16), and `moonshine_streaming.json` (2 packages: streaming-tiny in q8_0/f16).
   - **Verified**: Full test suite passes 100% green (56/56 tests).

## NEXT (highest value first)
1. **Zero-Dependency Language Bindings (`dynload`)**:
   - Finalize single-artifact shared build (`SPEECH_SHARED_EMBED=ON`) and zero-dependency dynload bindings (Rust, Python ctypes, TypeScript koffi, Swift).
2. **HuggingFace 5.x Long-form Seek Continuation**:
   - Verify long-form chunked streaming continuation across multi-minute audio files with synthetic silence and early `<|t|>` termination guards.
3. **Parakeet TDT & Moonshine CLI / Server End-to-End Testing**:
   - Validate CLI invocation with newly cataloged `--model whisper`, `--model moonshine`, and `--model parakeet_tdt`.

## LEFT TO DO (small)
- [ ] Formalize root `.gitmodules` so `git submodule update` works for the 3 embedded repos.
- [ ] `LNK4217` warnings: arch static libs see `TRANSCRIBE_API` as `dllimport`
      for same-DLL symbols. Benign but noisy.
- [ ] `external/ggml` mixed line endings (1145 CRLF blobs vs `eol=lf`):
      unchanged posture — leave alone; sync matches committed state on
      Windows; a Linux run would produce LF + a large diff.

## Notes / decisions made this session
- **"Missing assets" is now 5-for-5 wrong in this repo.** flashsr (real
  regression), the WER gap (unfetched-by-design), and this session's three
  (two registration defects + product bugs). Treat the label as unverified
  until a failure is reproduced and root-caused once.
- A test that passes only when the build tree is inside the repo is a
  registration defect, not an environment issue: model-spec resolution walks
  up from the cwd, so declare `WORKING_DIRECTORY` when a test reads the
  production catalog.
- `std::system` on Windows: if the command string starts with a quoted
  program path, `cmd /c` strips the first and last quote on the line. Wrap
  the whole command in one extra pair of quotes (see `run_shell_command`).
  Any code that shell-quotes a configurable program path into `std::system`
  has this bug.
- Detached threads that spawn subprocesses are a teardown hazard beyond
  leaks: a thread terminated by `ExitProcess` inside `CreateProcess` leaves
  a permanently *suspended* child pinning inherited handles — visible only
  as a downstream consumer (CTest) hanging on a pipe. Track and join.
- The model-pinning pattern is now a table (`fetch_asr_test_model.py`); the
  next pinned model is one dataclass row, not a new script.
