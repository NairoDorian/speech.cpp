# Progress — Unified_Audio.cpp (speech.cpp ggml fork) merge & improve

Status snapshot: **Upstream audio.cpp main (62735ea, 26 commits total) merged cleanly into speech.cpp. Phase 1 Allocator Hardening applied (BackendWeightStore 16MB cap, WavLM gallocr buffer reuse, Qwen3-TTS runaway guard, DeepFilterNet2 threshold). Phase 3 Native Long-Form VAD Chunk Planning & Public C ABI integrated with dedicated unit tests. Build tooling accelerated with ccache compiler launcher auto-detection. All 53 CPU core tests passing 100% green.** Date: 2026-08-23

## Repo layout (important, non-obvious)
`Unified_Audio.cpp/` is a **plain container directory with no git repo of its
own**. It holds exactly three things, each an independent repository:

| Folder | Role |
|---|---|
| `speech.cpp/` | the active development repo (the ggml/audio.cpp fork). **All merge work, and this log, live here.** Remote: `NairoDorian/speech.cpp`, upstream `0xShug0/audio.cpp`. |
| `audio.cpp/` | upstream reference — read from, not developed in (pulled to `62735ea`) |
| `transcribe.cpp/` | merge source — read from, not developed in. |
| `audio_cunba/` & `transcribe_cunba/` | hardened reference trees containing allocator fixes, VAD chunk planning, and build acceleration. |

Build trees are scratch dirs under `C:/Users/Z/AppData/Local/Temp/opencode/`:
`sp_bridge` (CPU, full model set, unified ABI + arches, tests), `sp_cuda`
(CUDA, core set, ABI/arches OFF), `audiocpp_flashsr` (audio.cpp reference),
`build-cpu-core` (local MSVC CPU core test suite).

## Overall progress (toward "Unified_Audio transcribes on CPU")
| Area | Status | % |
|---|---|---|
| Merge: ggml convergence (pin 8c63e709 + patches 0001–0006), CPU+CUDA certified | Done (prior sessions) | 100% |
| Merge: Upstream audio.cpp main synchronization (`62735ea`) | Done (clean merge, test gating resolved) | 100% |
| Memory: Phase 1 Allocator Hardening (16MB cap, WavLM gallocr, Qwen3 runaway, DFN2) | Done (certified in engine) | 100% |
| Long-form: Phase 3 Native VAD Chunk Planning & Re-stitching (`vad_plan`, `vad_merge`) | Done (native Silero + Energy VAD, C ABI) | 100% |
| ABI offline + streaming surface | Verified, real CTest gates | 100% |
| End-to-end ASR **offline text** (WER gate) | Done — 1.45% corpus WER | 100% |
| **End-to-end ASR streaming text** | **Done — streamed 4.35% == offline 4.35%, divergence 0** | **100%** |
| Test suite status | **53/53 green (100% pass in ~22s)** | **100%** |
| **Project-wide (functional CPU transcribe + unified audio)** | Phase 0 & Phase 1 substrate complete, Phase 3 VAD landed | **~94%** |

## DONE this session (plan R12 records all of it)

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

## NEXT (highest value first)
1. **Family Consolidation & Bidirectional Upgrade**:
   - **Qwen3-ASR & FunASR Nano**: Consolidate engine models (`src/models/qwen3_asr/`, `src/models/fun_asr_nano/`) with transcribe.cpp's direct depthwise conv / packed projection optimizations, 4-state streaming machine, and WER test corpus.
   - **Whisper Full Pipeline & HF 5.x Seek Fix**: Port the complete Whisper engine session (16 variants) with HuggingFace 5.x seek continuation fix to eliminate tail speech truncation on early `<|t|>` closures.
   - **Parakeet TDT & Moonshine Engine Spec Integration**: Provide native engine sessions + `model_specs/*.json` catalogs for Moonshine and Parakeet (11 variants) so they are directly callable via CLI, server, WebUI, and C ABI.
   - **Sortformer v2.1 Streaming Diarization**: Upgrade from v1 to v2.1 streaming Sortformer (`diar_streaming_sortformer_4spk-v2.1`) with official NVIDIA CC-BY-4.0 GGUF.
2. **Process-Wide VRAM Optimization (`SharedWeightRegistry`)**:
   - Implement `SharedWeightRegistry` and `ScopedWeightShareKey` from `audio_cunba` for global reference-counted weight buffers, reducing multi-session server memory from ~3 GB to ~34 MB per session.
3. **Universal Multi-Task Progress Callback & Unified ABI (`libspeech`)**:
   - Standardize `speech_progress_callback(model, fn, user_data)` across all 42+ TTS and 18+ ASR models (reporting ratio `0.0..1.0`, stage name, units).
   - Finalize single-artifact shared build (`SPEECH_SHARED_EMBED=ON`) and zero-dependency dynload bindings (Rust, Python ctypes, TypeScript koffi, Swift).

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
