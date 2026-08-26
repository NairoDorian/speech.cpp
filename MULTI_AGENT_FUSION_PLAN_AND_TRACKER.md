# Multi-Agent Fusion Plan & Progress Tracker

> **The Master Key:** *The two projects (`audio.cpp` and `transcribe.cpp`) learn from each other in parallel — each is the other's teacher and student — and merging them improves them both at the same time.*
>
> **Target Repository:** `speech.cpp` is the **only** active development repository. `audio.cpp` and `transcribe.cpp` in `../Unified_Audio.cpp/` are read-only reference trees.

---

## 1. Operating Rules for AI Agents

Every agent working on this repository must strictly adhere to these 6 protocol rules:

1. **One Step / Phase at a Time**:
   - Follow the step-by-step roadmap methodically. Never skip phases or combine unverified refactors.
2. **Phase Exit & Verification Protocol**:
   - At the end of every phase, run the full verification test suite (`build_env.bat ctest --test-dir build-cpu-core --output-on-failure -C Release`).
   - All tests must pass 100% green (clean skips with code `77` are permitted only for unpinned external model weights).
3. **Documentation Update Requirement**:
   - When completing a phase, update all tracking documents before stopping:
     - `CHANGELOG.md` (record added features, activations, and bug fixes)
     - `progress.md` (update status table, completed items, test counts)
     - `FUSION_ROADMAP_PLAN.md` (check off phase gates)
     - `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md` (update overall progress)
     - `MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md` (mark checkboxes `[x]`, record handoff state)
     - Create or update the `walkthrough.md` artifact.
4. **Pause Rule**:
   - **STOP** at the end of each phase, present the verification results to the user, and wait for user review/approval before initiating the next phase.
5. **Windows MSVC Build Instructions**:
   - Run commands through `.\build_env.bat` (wraps Visual Studio `vcvars64.bat` environment).
   - Build command: `.\build_env.bat cmake --build build-cpu-core --config Release -j 8`
   - Test command: `.\build_env.bat ctest --test-dir build-cpu-core --output-on-failure -C Release`
   - Use `TRANSCRIBE_BUILD` compile definition for internal targets to prevent Windows `__declspec(dllimport)` link errors.
6. **Upstream Sync Protocol — merge, never content-copy**:
   - Syncing `0xShug0/audio.cpp:main` **must** end in a recorded merge (a real
     merge, or `git merge -s ours upstream/main` when every commit has been
     dispositioned by hand). Copying content without a merge leaves the
     merge-base behind, and `git` then re-reports commits that are already
     applied — this produced a phantom "6 commits behind" that survived two
     sessions (2 of the 6 were already in the tree verbatim).
   - Audit each upstream commit **by content, not by subject line**: grep the
     target symbols in this tree and dry-run with `git apply --check` before
     deciding. Fork divergence makes some upstream commits genuinely N/A —
     record those in the merge ledger so they are not re-triaged next session.
   - Check the sync is real with
     `git rev-list --left-right --count HEAD...upstream/main` (right operand
     must read `0`). `git fetch upstream` is always safe; never `git pull` this
     repo from upstream.

---

## 2. Master Phase Status Overview

| Phase | Description | Status | Verification Gate |
|---|---|---|---|
| **Phase 0** | ggml convergence & bridge test | **`[x] DONE`** | ggml patch pin 8c63e709, bridge test |
| **Phase 1** | Allocator hardening & memory safety | **`[x] DONE`** | 16MB metadata pool cap, WavLM gallocr |
| **Phase 2** | Toolchain modernization & build provenance | **`[x] DONE`** | ccache auto-detect, 3-outlet versioning |
| **Phase 3** | Long-form VAD chunk planning & C ABI | **`[x] DONE`** | `vad::plan`, `vad::merge`, `audiocpp_vad` |
| **Phase 4** | Shared weight registry & batched offline decoders | **`[x] DONE`** | 5 offline ASR decoders, Sortformer v2 |
| **Phase 5** | Universal `audiocpp` C ABI & progress callbacks | **`[x] DONE`** | 48 C ABI entry points, `ProgressCanceled` |
| **Phase 6** | Model spec catalogs (Whisper, Moonshine) | **`[x] DONE`** | `model_specs/*.json` catalogs validated |
| **Phase 7** | **Safety net, ground truth & activation** | **`[x] DONE`** | **88 CTest targets 100% green, D1 & D2 fixed** |
| **Phase 8** | **Contract convergence & exception boundary** | **`[x] DONE`** | **`StreamingSessionBase`, `RunControl`, ABI guards (92 targets)** |
| **Phase 9** | **Unified Mel & Tokenizer subsystems** | **`[x] DONE`** | **`MelExtractor`, `TokenizerHub`, parity tests (95 targets)** |
| **Phase 10** | **Attention & Conformer module fusion** | **`[x] DONE`** | **Unified `sanm`, `shaw_attn`, `causal_lm_ops`, bake-off certified** |
| **Phase 11** | Architectural family migration (Waves W1–W6) | `[~] IN PROGRESS — W1a DONE` | 18 families migrated to unified engine |
| **Phase 12** | Unified `speech.h` C ABI & language bindings | `[ ] PENDING` | Single unified DLL, Python/C#/Node bindings |
| **Phase 13** | End-to-end multi-stage pipeline composition | `[ ] PENDING` | VAD + Diarization + ASR + Alignment graphs |
| **Phase 14** | Cleanup, benchmarking, licensing & 1.0 release | `[ ] PENDING` | Deletion ledger cleared, zero deprecations |

---

## 3. Detailed Step-by-Step Execution Plan

---

### Phase 7: Safety Net, Ground Truth & Activation — `[x] COMPLETED`

- [x] **7.0**: Documentation audit correction (`CHANGELOG.md`, `progress.md`, `docs/reports/fusion_audit_2026-08-23.md`).
- [x] **7.1**: Ported 51 test translation units, 87 golden fixture files (19 dirs), 36 tolerance manifests, and 18 architecture porting docs into `tests/transcribe/`.
- [x] **7.2**: Integrated CI formatting lints and `AGENTS.md`.
- [x] **7.3**: Fixed Defect D1 (sniff precedence collision) in `src/runtime/transcribe.cpp` and added `test_adapter_sniff_dispatch`.
- [x] **7.4**: Activated `SharedWeightRegistry` across all call sites (`audiocpp_capi.cpp`, CLI, server `runtime.cpp`, workflow runner) and verified via `test_shared_weight_vram`.
- [x] **7.5**: Activated batched ASR decode in `ArchAdapter` across all 16 slots, added `audiocpp_asr_batch`/`audiocpp_free_text_batch`, and implemented pre-clear validation hooks (Defect D2 remediation).
- [x] **7.6**: Recalibrated WER gates: created `asr_e2e_edits_test` (`total_edits <= 1`) and tightened stream divergence bound to `0`.
- [x] **7.8**: Implemented Unified Family Registry v1 (`family_registry.h` & `family_registry.cpp`) and verified via `family_registry_unit`.
- [x] **Verification**: All 88 CTest targets 100% green on CPU core build.

---

### Phase 8: Contract Convergence & Exception Boundary — `[x] COMPLETED (100% Green)`

**Goal**: Raise `engine::runtime`'s session contract to `transcribe::Arch`'s level and establish total C ABI exception containment.

- [x] **8.1 — Implement `StreamingSessionBase` Lifecycle Engine**:
  - Created `include/engine/framework/runtime/streaming_session_base.h` and `src/framework/runtime/streaming_session_base.cpp`.
  - Implemented the 4-state machine: `IDLE → ACTIVE → FINISHED | FAILED` (state transitions managed solely by base).
  - Implemented `uint64_t stream_revision()`, incremented on any observable snapshot change.
  - Implemented commitment policies: `AUTO`, `ON_FINALIZE`, `STABLE_PREFIX` (with `stable_prefix_agreement_n`, default 3).
  - Implemented pure validation hooks `validate_stream()` and `validate_chunk()` called before any destructive state modification.
- [x] **8.2 — Implement `RunControl` (Unified Progress & Abort)**:
  - Created `include/engine/framework/runtime/run_control.h` providing `poll_abort()`, `emit_progress()`, `request_abort()`, and `reset_abort()`.
  - Connected `RunControl` to `RuntimeSessionBase` and `IVoiceTaskSession`.
- [x] **8.3 — Extract & Generalize `StreamChunker`**:
  - Lifted chunking logic into `include/engine/framework/runtime/stream_chunker.h` and `src/framework/runtime/stream_chunker.cpp`.
  - Maintained PCM ring buffering, exact chunk alignment (`preferred_audio_chunk_samples`), tail flushing on finalize, and contiguous timestamp offsets.
- [x] **8.4 — Integrate Streaming Infrastructure with Adapter**:
  - Replaced duplicate `StreamChunker` in `transcribe-arch-adapter.cpp` with unified `engine::runtime::StreamChunker`.
- [x] **8.5 — Close C ABI Exception Containment**:
  - Audited all exported `audiocpp_*` C ABI entry points in `capi/src/audiocpp_capi.cpp`.
  - Wrapped 100% of exported functions (device discovery, model info/capabilities, WAV I/O, artifacts, and all memory free functions) in `try { ... } catch (...) { ... }` exception guards.
- [x] **Phase 8 Verification**:
  - Added unit tests: `run_control_unit_test`, `stream_chunker_unit_test`, `streaming_session_base_unit_test`, `capi_exception_containment_test`.
  - Full CTest suite 92/92 test targets 100% green (88 passed, 4 clean skips on unpinned weights).

---

### Phase 9: Unified Mel & Tokenizer Subsystems — `[x] COMPLETED (100% Green)`

**Goal**: Merge duplicate Mel spectrogram extractors and Tokenizer implementations into high-performance, single-source engine subsystems.

- [x] **9.1 — Implement Unified `engine::audio::MelExtractor` & `FrontendSpec`**:
  - Created `include/engine/framework/audio/frontend_spec.h`, `include/engine/framework/audio/mel_extractor.h`, and `src/framework/audio/mel_extractor.cpp`.
  - Full support for `MelSpectrogram` (Whisper, Parakeet, GigaAM, NeMo), `KaldiFbank`, and `RawPcm` (Moonshine passthrough).
  - Supported Hann periodic/symmetric, Hamming, custom windows, Slaney filterbank area normalization, non-zero support span indexing (`fb_begin_`/`fb_end_`), and `PerFeature`, `PerUtterance`, `Global`, and `None` normalization.
- [x] **9.2 — Implement Model Spec Schema v2 Frontend Block**:
  - Updated `engine::model_spec::validate_spec` in `src/framework/model_spec/schema.cpp` to validate `"frontend"` objects while maintaining full backward compatibility for schema v1.
- [x] **9.3 — Implement Unified `engine::text::TokenizerHub`**:
  - Created `include/engine/framework/text/tokenizer_hub.h` and `src/framework/text/tokenizer_hub.cpp`.
  - Unified Unigram, BPE, Byte-Level BPE, Tiktoken raw bytes, SentencePiece, and HuggingFace JSON tokenizers with bidirectional `encode()` and `decode()`, $O(1)$ piece lookup, and GPT-2 byte mapping inversion.
- [x] **9.4 — Consolidate Neural Codec Interface**:
  - Updated `include/engine/framework/codecs/codec.h` (`IAudioCodec`) with unified `encode()` and `decode()` definitions.
- [x] **Phase 9 Verification**:
  - Added unit tests: `frontend_contract_test`, `frontend_parity_test`, `tokenizer_parity_test`.
  - Full CTest suite 95/95 test targets 100% green (91 passed, 4 clean skips on unpinned weights).

---

### Phase 10: Attention & Conformer Module Fusion — `[x] COMPLETED (100% Green)`

**Goal**: Unify duplicate neural building blocks (SANM, Conformer, RoPE, Relative Multi-Head Attention, Causal Transformers).

- [x] **10.1 — SANM Encoder Unification**:
  - Unified `src/framework/modules/speech_encoders/sanm.{cpp,h}` and `src/runtime/sanm/sanm.{cpp,h}`.
  - Implemented shared sinusoidal positional encoding (`build_sinusoidal_pe`, `make_sinusoidal_positions`), fused QKV branch support, and FSMN depthwise direct lowering with `TRANSCRIBE_CONV_DIRECT_DW` / `TRANSCRIBE_CONV_NO_DIRECT_DW` toggle support.
  - Verified with SenseVoice and FunASR-Nano probes (`fun_asr_nano_sanm_probe`, `fun_asr_nano_assets_test`, `fun_asr_nano_frontend_probe`).
- [x] **10.2 — Conformer Module & Shaw Attention Unification**:
  - Implemented unified Shaw block-local attention in `include/engine/framework/modules/attention/shaw_attention.h` and `src/framework/modules/attention/common_relative_attention.cpp`.
  - Re-routed `src/runtime/granite_conformer/shaw_attn.cpp` through `engine::modules::build_shaw_block_attention`.
  - Unified Conformer depthwise & pointwise conv lowering policies.
- [x] **10.3 — Causal LM / Transformer Decoder Unification**:
  - Implemented `include/engine/framework/modules/transformers/causal_lm_ops.h` and `src/framework/modules/transformers/causal_lm_ops.cpp`.
  - Unified `pick_kv_cache_context`, `fill_prefill_chunk_mask`, and `prefill_chunk_size` across engine and runtime decoders.
- [x] **10.4 — Overlap Bake-Off & Parity Certification**:
  - Published comprehensive bake-off report `docs/reports/overlap_bakeoff.md` covering all 6 overlapping families per V6 Decision R3.
- [x] **Phase 10 Verification**:
  - Full CTest suite 95/95 test targets 100% green (91 passed, 4 clean skips on unpinned weights).
  - **PAUSE & UPDATE**: Update markdown tracking files and request user approval.

---

### Phase 11: Architectural Family Migration (Waves W1–W6) — `[~] IN PROGRESS`

**Goal**: Migrate all 18 `transcribe.cpp` model families from `src/runtime/arch/` into native `src/models/` engine packages.

**Wave order** (aligned to FUSION_ROADMAP_PLAN §8 "Migration order — cheapest-risk first, highest-value first"; supersedes the earlier draft ordering in this tracker):

- [ ] **Wave W1 — Moonshine pair (raw-PCM, covered by both live WER gates)**:
  - [x] **W1a — `moonshine` (offline) → native engine package. DONE 2026-08-24.**
    - Created `src/models/moonshine/` (`graphs.cpp`, `assets.cpp`, `runtime.cpp`, `session.cpp`) with internal headers under `include/engine/models/moonshine/`.
    - Numerics-identical port of encoder / cross-KV / KV-cached decoder graphs + greedy loop from `src/runtime/arch/moonshine/`.
    - Engine integration: weights via `core::BackendWeightStore` (SharedWeightRegistry active), tokenizer via Phase-9 `TokenizerHub` (`load_tokenizer_from_gguf`), abort/progress via `RunControl`; session derives from `RuntimeSessionBase` + `IOfflineVoiceTaskSession` with contract-conformant `run_batch` (per-utterance isolation) and pre-requested-abort handling.
    - Registered: `audiocpp_add_model(moonshine ...)` + `AUDIOCPP_ASR_MODEL_TARGETS` (asr/full composites); family id/aliases already present in `family_registry` since Phase 8.7.
    - Fixed latent Phase-6 spec defect: pinned moonshine GGUFs carry no embedded config/tokenizer sidecars; `model_specs/moonshine.json` sources block corrected to tensors-only.
    - Gate: new CTest target **`moonshine_engine_smoke_test`** — registry load by canonical id, 4 LibriSpeech fixtures offline, corpus WER ≤ 10% structural bound (**measured 1.449% = 1/69 edits, identical to arch baseline**), ordered `run_batch`, `request_abort()` unwinds `run()`. Arch copy still builds and stays green (§4.4 coexistence).
    - Suite after W1a: **96/96 green** on `build-cpu-core`.
  - [ ] **W1b — `moonshine_streaming` → native engine package on `StreamingSessionBase`.** Design map recorded in `walkthrough.md` (incremental sliding-window encoder geometry L/R totals + frontend pad, adapter pos_emb absolute-frame add + optional proj, committed host cross-K/V buffers uploaded per decode, LCP-based commit ↔ base `STABLE_PREFIX` policy mapping, throttled partial decodes, finalize tail top-up).
  - [ ] W1 retirement step (after both green + user review): delete `src/runtime/arch/{moonshine,moonshine_streaming}/` as their own commits with Appendix B rows B16a/B16b.
- [ ] **Wave W2 — Whisper Universal Family** (`whisper`: tiny…large-v3-turbo, .en variants, legacy `.bin` loader, suppress tables, temperature-fallback ladder + DecodeTelemetry; unified MelExtractor + TokenizerHub; `model_specs/whisper.json` 16 packages validated).
- [ ] **Wave W3 — Acoustic & Single-Variant Models** (`gigaam`, `medasr`, `cohere`; gigaam retires the third private mel implementation).
- [ ] **Wave W4 — Transducer & Large Multi-Lingual Models** (`canary`, `canary_qwen`, `voxtral` offline).
- [ ] **Wave W5 — Conformer & Shaw Attention Models** (`granite`, `granite_nar`, `moss_asr` under its resolved id).
- [ ] **Wave W6 — Phase-10 overlap retirements & bridge decommissioning** (retire redundant arch dirs for engine winners; route remaining C ABI calls through ModelRegistry).
- [ ] **Phase 11 Verification (final)**:
  - All 18 families resolvable by canonical id through `ModelRegistry`; `family_registry_unit` green with zero orphan specs.
  - All 66 golden manifests green through the engine path; WER gates unchanged (`edits <= 1`, divergence 0).
  - CLI/server/WebUI reachability asserted per family; per-family RTF/VRAM within 10% of arch baseline.
  - Appendix B ledger complete for every deleted arch directory + the adapter.
  - **PAUSE & UPDATE**: Update markdown tracking files and request user approval.

---

### Phase 12: Unified Top-Level C ABI & Language Bindings — `[ ] PENDING`

**Goal**: Deliver a single clean C shared library (`speech.dll` / `libspeech.so`) with comprehensive multi-language bindings.

- [ ] **12.1 — Implement `speech.h` / `speech_capi.cpp`**:
  - Unified ABI surface exposing all 14 voice intelligence tasks with size-aware structs, typed extensions, and telemetry.
- [ ] **12.2 — Python Binding (`speechcpp`)**:
  - Fast ctypes/CFFI and NumPy integration; packaging via `pyproject.toml` and `uv`.
- [ ] **12.3 — Node.js / TypeScript Binding**:
  - N-API bindings supporting streaming audio buffers and async promises.
- [ ] **12.4 — C# / .NET & Rust Bindings**:
  - P/Invoke wrapper for .NET 8+ and safe idiomatic Rust crate.
- [ ] **Phase 12 Verification**:
  - Run multi-language test suites against unified library.
  - **PAUSE & UPDATE**: Update markdown tracking files and request user approval.

---

### Phase 13: End-to-End Multi-Stage Pipeline Composition — `[ ] PENDING`

**Goal**: Provide out-of-the-box composite pipelines connecting VAD, Diarization, ASR, Alignment, and Audio Enhancement in single executions.

- [ ] **13.1 — Full Meeting Transcription Pipeline**:
  - Long-Form Audio → Silero VAD Chunking → Sortformer v2 Diarization → Whisper/Parakeet ASR → Timestamp Alignment.
- [ ] **13.2 — Live Voice Agent / Streaming Pipeline**:
  - Streaming Mic PCM → Energy/Silero VAD → Streaming ASR (Moonshine/Parakeet/Voxtral) → TTS Voice Response.
- [ ] **13.3 — Audio Restoration Pipeline**:
  - Degraded Audio → RNNoise/DFN2 Denoise → FlashSR / ZipEnhancer Super-Resolution.
- [ ] **Phase 13 Verification**:
  - Composite pipeline end-to-end integration tests.
  - **PAUSE & UPDATE**: Update markdown tracking files and request user approval.

---

### Phase 14: Cleanup, Benchmarking, Licensing & 1.0 Release — `[ ] PENDING`

**Goal**: Final production certification, deletion ledger clearance, performance benchmarking, and 1.0 release.

- [ ] **14.1 — Deletion Ledger Final Audit**:
  - Verify all superseded files from `src/runtime/` have been safely deleted and recorded in `Appendix B`.
- [ ] **14.2 — Performance & Memory Benchmark Suite**:
  - Measure RTF, per-utterance latency, peak VRAM/RAM across CPU and CUDA backends.
- [ ] **14.3 — License & Attribution Hygiene**:
  - Ensure Apache-2.0 and MIT third-party notices are intact in `THIRD-PARTY-LICENSES.md`.
- [ ] **Phase 14 Verification**:
  - Full CI matrix clean build on Windows, Linux, and macOS (CPU, CUDA, Metal).
  - **FINAL 1.0 RELEASE SIGN-OFF**.

---

## 4. Current State & Handoff Summary for AI Agents

- **Current Timestamp**: 2026-08-26 (Upstream reconciliation to `c79e588` — **0 behind** — on top of the 2026-08-25 Phase 11 W1a close)
- **Last Completed Increments**:
  - **Upstream Reconciliation (`0xShug0/audio.cpp:main@c79e588`) — 63 ahead, 0 behind**:
    - Root-caused the recurring "6 commits behind": the prior sync (`9b34fd2`) content-copied instead of merging, so the merge-base stayed at `62735ea` and git re-reported already-applied commits. **2 of the 6 were phantoms** (`288a271` `--list-devices`, `d25ffac` chunk metadata — both verified present in the tree by symbol, not by subject line).
    - Cherry-picked `4ec485d` supertonic voice preset (→ `90659b1`) and `d03b957` IndexTTS2 HIP F16 KV/conv default (→ `3682698`); both applied clean with upstream authorship preserved.
    - Adapted `c79e588` tag-driven release CI (→ `a775463`): `.github/workflows/release.yml` + `docs/RELEASING.md`, artifacts rebranded `audio-<tag>-…` → `speech-<tag>-…`.
    - Declared `c6805de` (audio.cpp README 0.7 banner) **N/A** — fork README has no such banner; recorded in the merge ledger so it is not re-triaged.
    - Closed with `git merge -s ours upstream/main` carrying the full 6-commit disposition ledger. `git rev-list --left-right --count HEAD...upstream/main` now reads **`63  0`**. See Operating Rule 6.
    - Reference trees pulled: `audio.cpp`→`c79e588`, `transcribe.cpp`→`2102bca` (**ggml bumped to upstream master `36da5713` / v0.22.0**), `audio_cunba`→`8cf5136`, `transcribe_cunba`→`2345350`. `speech.cpp` was fetched only, never pulled.
    - Total CTest targets: **100/100 100% green** (96 passed, 4 clean model-dependent skips) — unchanged from baseline.
  - **Upstream Synchronization (`0xShug0/audio.cpp:main@d25ffac`) & `speech.cpp` Naming Aliases**:
    - Extracted device enumeration into `engine::core::print_backend_devices`.
    - Added `--list-devices` support to `audiocpp_server` and CLI.
    - Added CMake ALIAS targets (`speech_cli`, `speechcpp_cli`, `speech_server`, `speechcpp_server`, `speech_gguf`, `speechcpp_gguf`).
    - Handled out-of-span chunk speech metadata resilience in `chunking.cpp`.
    - Total CTest targets: **100/100 100% green** (96 passed, 4 clean model-dependent skips).
  - **Phase 11 Wave W1a — Native Engine Moonshine (offline)** — `100% DONE`
    - Package `src/models/moonshine/` + internal headers in `include/engine/models/moonshine/`.
    - Gate `moonshine_engine_smoke_test`: engine-path corpus WER **1.449% (1/69 edits) == arch baseline**.
- **IMMEDIATE NEXT TASK FOR AGENT**:
  - **Execute Phase 11 Wave W1b (`moonshine_streaming` → native engine package on `StreamingSessionBase`)**:
    - Follow the design map recorded in `walkthrough.md` §"W1b design map" (streaming geometry helpers, incremental encode→adapter→cross-KV-commit pipeline, LCP commit ↔ STABLE_PREFIX policy, throttled partial decodes, finalize tail top-up, reset semantics).
    - Port graph builders from `src/runtime/arch/moonshine_streaming/{encoder,decoder}.cpp`; session derives from `engine::runtime::StreamingSessionBase`.
    - Register `audiocpp_add_model(moonshine_streaming ...)`, correct `model_specs/moonshine_streaming.json`, add a streaming engine-path gate (feed odd-sized chunks through `IStreamingVoiceTaskSession`, gate streamed-vs-offline divergence == 0 against the offline engine package from W1a).
    - Then run full verification, update all tracking docs, and pause for user review before Wave W2.

