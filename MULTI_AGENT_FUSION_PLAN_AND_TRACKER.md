# Multi-Agent Fusion Plan & Progress Tracker

> **The Master Key:** *The two projects (`audio.cpp` and `transcribe.cpp`) learn from each other in parallel — each is the other's teacher and student — and merging them improves them both at the same time.* `speech.cpp` is the child of **both**; the audio.cpp fork base is a convenience of how the repo was created, not a statement of precedence.
>
> **Target Repository:** `speech.cpp` is the **only** active development repository. `audio.cpp` and `transcribe.cpp` in `../Unified_Audio.cpp/` are read-only *in the sense that we never commit there* — they are **both parents**, and both are equally authoritative sources of change. See Operating Rule 7.

---

## 1. Operating Rules for AI Agents

Every agent working on this repository must strictly adhere to these 7 protocol rules:

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

7. **Dual Parentage & the Dependency Sync Routine**:
   - **`speech.cpp` is equally a child of `audio.cpp` and of `transcribe.cpp`.**
     We forked `audio.cpp` because it was the larger tree to start from — a
     mechanical convenience, **not** a precedence claim. An improvement in
     `NairoDorian/transcribe.cpp` is **exactly as authoritative** as one in
     `0xShug0/audio.cpp` and gets the same audit-by-content and the same
     disposition ledger. Never treat transcribe.cpp commits as optional
     because it is labelled a "merge source".
   - Only `audio.cpp` has a git `upstream` remote here (it is the fork base),
     so only it produces a merge-base. **That is a tooling limitation, not a
     hierarchy** — transcribe.cpp drift must be tracked by hand.
   - **A dependency bump on either parent is a first-class upstream change for
     us.** transcribe.cpp moving ggml to `36da5713` (v0.22.0) moved *our* ggml
     floor; it was not a curiosity to note and defer.
   - **Before any release state, and regularly otherwise**, refresh all three
     sources and verify — see `AGENTS.md` § "Dependency Sync Routine" and
     `scripts/sync-deps.sh`:
     1. `audio.cpp` via `git fetch upstream` + audited merge (Rule 6);
     2. `transcribe.cpp` via the sibling checkout, triaged by hand;
     3. `ggml` via `scripts/sync-ggml.sh` (keep our pin **at or above**
        transcribe.cpp's `ggml/UPSTREAM` sha).
   - `external/ggml/` is generated: never hand-edit, always land deltas as
     `patches/ggml/NNNN-*.patch`. A bump that breaks a patch is normal — rebase
     the patch, do not drop it.
   - **An audio.cpp merge is a ggml event** *(added 2026-09-23)*. audio.cpp
     vendors its OWN hand-edited ggml at the same path, so its ggml commits can
     only reach us as untracked edits inside a merge. That is how ~1,800 lines in
     26 files (VibeASR INT8, Breeze bf16, CUDA stream priority, Vulkan dispatch)
     sat in the tree with no patch until they were captured as 0008-0011. After
     every audio.cpp merge that touches `external/ggml`, and before any sync:
     `scripts/sync-ggml.sh --check` (exit 1 = a delta no patch carries) — or
     `scripts/sync-deps.sh --verify-ggml`. Capture before you sync.
   - **transcribe.cpp's "merge-base" is a ledger**: the `Triage watermark:` line
     in `docs/upstream/transcribe_cpp_triage.md`, read by `sync-deps.sh`. Every
     commit gets a disposition row before the watermark moves past it. Taking a
     parent commit wholesale is never automatic: `6c767184` is labelled "VAD" but
     silently flips Whisper to a CPU static-decode path that decodes garbage
     (bisected 2026-09-23; see the ledger).

---

## 2. Master Phase Status Overview

| Phase | Description | Status | Verification Gate |
|---|---|---|---|
| **Phase 0** | ggml convergence & bridge test | **`[x] DONE`** | ggml pin `456172ec` (0.24.0, 2026-09-23; was 8c63e709 → 36da5713) + 11 tracked patches, `sync-ggml.sh --check` clean |
| **Phase 1** | Allocator hardening & memory safety | **`[x] DONE`** | 16MB metadata pool cap, WavLM gallocr |
| **Phase 2** | Toolchain modernization & build provenance | **`[x] DONE`** | ccache auto-detect, 3-outlet versioning |
| **Phase 3** | Long-form VAD chunk planning & C ABI | **`[x] DONE`** | `vad::plan`, `vad::merge`, `audiocpp_vad` |
| **Phase 4** | Shared weight registry & batched offline decoders | **`[x] DONE`** | 5 offline ASR decoders, Sortformer v2 |
| **Phase 5** | Universal `audiocpp` C ABI & progress callbacks | **`[x] DONE`** | 48 C ABI entry points, `ProgressCanceled` |
| **Phase 6** | Model spec catalogs (Whisper, Moonshine) | **`[x] DONE`** | `model_specs/*.json` catalogs validated |
| **Phase 7** | **Safety net, ground truth & activation** | **`[x] DONE`** | **88 CTest targets 100% green, D1 & D2 fixed** |
| **Phase 8** | **Contract convergence & exception boundary** | **`[x] DONE`** | **`StreamingSessionBase`, `RunControl`, ABI guards (92 targets)** |
| **Phase 9** | **Unified Mel & Tokenizer subsystems** | **`[x] DONE`** | **`MelExtractor`, `TokenizerHub`, parity tests (95 targets)** |
| **Phase 10** | **Attention & Conformer module fusion** | **`[x] DONE`** (modules) · **`[ ] VERDICTS ONLY`** (feature-merges + deletions — see 10.5) | **Unified `sanm`, `shaw_attn`, `causal_lm_ops`, bake-off certified; 0 of 5 loser-feature merges executed** |
| **Phase 10.5** | **Execute the Phase-10 verdicts** — 5 feature-merges, 5 arch deletions, ledger rows *(roadmap v6.0)* | `[x] DONE — all 5 overlapping families retired` (B11 9cc5457 / B12 fdaa9a5 / B13 e3e7eac1 / B14 1be9ac40 / B15 a8b7a03b); 110/110 CTest green on cpu-core | B11–B15 revert commits filled; no shadowed GGUF arch |
| **Phase 11a** | **ASR runtime layer** — `EncDecKVCache`, decode drivers, `AsrResult`/`AsrLimits`, long-form over `audio/chunking`; re-base W1a/W1b/W2a; fold Whisper onto `WhisperEmbeddingModule`; `.bin` as `TensorSource`; delete `transcribe-vad*` *(v6.0)* | `[~] MOSTLY DONE` | B29 VAD dedup (c59b15a0); `AsrResult`/`AsrLimits`; `TaskResult.truncated` (L11); B30 `EncDecKVCache` (a6d02849); `DecodeDriver` argmax/suppress + 7 packages migrated (d43f754e); B31 WhisperEmbeddingModule fold (5318d532) + `WhisperBinTensorSource` (f6a9d2b2). **Remaining**: long-form over `audio/chunking` for the non-Whisper families (Whisper's long-form is its own model-driven seek loop, W2b.1). *(2026-09-23: W2b.1 delivered the sampling / fallback primitives — `framework/asr/sampling` — and moved the engine Whisper's `.bin` weights onto `TensorSource`.)* |
| **Phase 12** | **`speech.h` thin over the engine** — pulled forward; `audiocpp.h` frozen → shim *(v6.0)* | `[ ] PENDING` | `abi_compat_test`; one artifact |
| **Phase 11b** | Remaining 9 families as **thin packages**, each deleting its arch dir in-wave | `[~] 3 of 18 DONE (W1a, W1b, W2a at exact arch parity)` | per-family golden + WER parity |
| **Phase 11c** | **Delete `src/runtime/` in full** *(v6.0 — reverses v5's "keep the dispatcher")* | `[ ] PENDING` | `ls src/runtime` absent; `lint_teardown` over `src/` → 0 |
| **Phase 13** | Bindings retarget — transcribe's 6 existing bindings → one generated IR over `speech.h` | `[ ] PENDING` | `generate.py --check` green ×6 |
| **Track M** | Methodology parity for audio.cpp's own families — goldens + tolerances + `validate.py` per phase quota *(v6.0)* | `[ ] CONTINUOUS` | families under `validate.py` count rises every phase |
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

- [x] **Wave W1 — Moonshine pair (raw-PCM, covered by both live WER gates)**:
  - [x] **W1a — `moonshine` (offline) → native engine package. DONE 2026-08-24.**
    - Created `src/models/moonshine/` (`graphs.cpp`, `assets.cpp`, `runtime.cpp`, `session.cpp`) with internal headers under `include/engine/models/moonshine/`.
    - Numerics-identical port of encoder / cross-KV / KV-cached decoder graphs + greedy loop from `src/runtime/arch/moonshine/`.
    - Engine integration: weights via `core::BackendWeightStore` (SharedWeightRegistry active), tokenizer via Phase-9 `TokenizerHub` (`load_tokenizer_from_gguf`), abort/progress via `RunControl`; session derives from `RuntimeSessionBase` + `IOfflineVoiceTaskSession` with contract-conformant `run_batch` (per-utterance isolation) and pre-requested-abort handling.
    - Registered: `audiocpp_add_model(moonshine ...)` + `AUDIOCPP_ASR_MODEL_TARGETS` (asr/full composites); family id/aliases already present in `family_registry` since Phase 8.7.
    - Fixed latent Phase-6 spec defect: pinned moonshine GGUFs carry no embedded config/tokenizer sidecars; `model_specs/moonshine.json` sources block corrected to tensors-only.
    - Gate: new CTest target **`moonshine_engine_smoke_test`** — registry load by canonical id, 4 LibriSpeech fixtures offline, corpus WER ≤ 10% structural bound (**measured 1.449% = 1/69 edits, identical to arch baseline**), ordered `run_batch`, `request_abort()` unwinds `run()`. Arch copy still builds and stays green (§4.4 coexistence).
    - Suite after W1a: **96/96 green** on `build-cpu-core`.
  - [x] **W1b — `moonshine_streaming` → native engine package on `StreamingSessionBase`. DONE 2026-08-26.**
    - Created `src/models/moonshine_streaming/` (`graphs.cpp`, `assets.cpp`, `runtime.cpp`, `session.cpp`) with internal headers under `include/engine/models/moonshine_streaming/`; registered via `audiocpp_add_model` + the ASR composite.
    - Numerics-identical port from `src/runtime/arch/moonshine_streaming/`: time-domain frontend (CMVN → `asinh(exp(log_k)·x)` → linear+SiLU → two causal stride-2 convs), encoder blocks with per-layer sliding-window masks and **no RoPE**, adapter with **absolute-frame** `pos_emb` get_rows (+ optional proj), **untied `lm_head`**, vanilla decoder LNs. The three conformer helpers the arch borrowed are reimplemented locally — the package has **no `src/runtime/` dependency**.
    - Session derives from `engine::runtime::StreamingSessionBase`; lifecycle, revision counting and the committed/tentative split are base-owned. Per the design map's preferred option we drive `update_text(full)` per partial and let the base's **`STABLE_PREFIX`** policy (agreement_n=3) choose the commit boundary rather than re-deriving an LCP per family.
    - Incremental pipeline per feed: encode window with L/R context → adapter the emit slice at absolute frame offsets → project cross-K/V → append to host committed buffers → trim unreachable PCM → throttled from-BOS AR re-decode (`moonshine_streaming.min_decode_interval_ms`, default 240 ms). Offline `run()` reuses the same graphs.
    - Fixed the same latent Phase-6 spec defect W1a found: `model_specs/moonshine_streaming.json`'s gguf source required sidecars the pinned GGUFs do not carry (`embedded_sidecars=false`) and could never resolve; now tensors-only.
    - Gate **`moonshine_streaming_engine_smoke_test`** (CTest #90): registry load by id + alias, family advertises ASR offline **and** streaming, per fixture offline + streamed on **one** session in odd non-frame-aligned chunks (~100–400 ms, throttle 0), asserting monotonic revision, append-only committed text, Active/Finished/Idle lifecycle, `reset()`, and abort unwinding. **Measured: streamed 4.34783% == offline 4.34783% (3/69), divergence 0 — exactly the arch `asr_stream_text_wer_test` baseline.**
    - Suite after W1b: **101/101 green** on `build-cpu-core`. Arch copy untouched and still building (§4.4 coexistence).
  - [x] **W1 retirement step (Wave W1 retirement: B16a & B16b). DONE 2026-09-17.**
    - Retired duplicate legacy arch implementations `src/runtime/arch/moonshine/` and `src/runtime/arch/moonshine_streaming/` (~7,136 LOC deleted).
    - Registered `"moonshine"` and `"moonshine_streaming"` in `adapter_archs` table in `src/runtime/transcribe-arch-adapter.cpp`, routing C ABI calls directly to native engine packages `engine::models::moonshine` and `engine::models::moonshine_streaming`.
    - Preserved public C ABI stream extension initializer `transcribe_moonshine_streaming_stream_ext_init` in `src/runtime/transcribe-family-ext.cpp`, with `moonshine_streaming.min_decode_interval_ms` mapped into task request options.
    - Verified 100% C ABI parity on `build-cpu-full`: `asr_e2e_wer_test` passed (1.449% WER = 1/69 edits, 1.39s), `asr_stream_text_wer_test` passed (4.348% WER, divergence 0, 25.11s), and `asr_e2e_edits_test` passed (1.38s).
    - Full CTest suite on `build-cpu-core`: 100% green (106 passed, 1 clean fixture skip, 0 failed).
- [ ] **Wave W2 — Whisper Universal Family** (`whisper`: tiny…large-v3-turbo, .en variants, legacy `.bin` loader, suppress tables, temperature-fallback ladder + DecodeTelemetry; unified MelExtractor + TokenizerHub).
  - [x] **W2 prerequisite — verification asset secured. DONE 2026-08-26.** *(Correction 2026-09-23: the claim below was only half true. `audio-cpp/audio.cpp-gguf` has no Whisper directory, but parent transcribe.cpp publishes every Whisper variant as a GGUF under `handy-computer/whisper-*-gguf` — its native format, `general.architecture = whisper`. `whisper-tiny.en-Q8_0.gguf` and `whisper-tiny-Q8_0.gguf` are now pinned too. The engine package cannot load them yet, which is why W2b owns GGUF loading.)* `model_specs/whisper.json` is catalog-only in the strong sense: its 16 packages point at `Whisper-*-GGUF` paths under `audio-cpp/audio.cpp-gguf`, and that repo has **no Whisper directory at all** — nothing is downloadable, so the family could not have been gated as catalogued. Pinned `ggml-tiny.en.bin` from `ggerganov/whisper.cpp` instead (canonical legacy ggml `.bin`, 77.7 MB, sha256 `921e4cf8…`), which also exercises `bin_load.cpp`. New gate **`asr_e2e_whisper_wer_test`** locks the arch baseline: **corpus WER 4.34783% (3/69), RTF 0.047** — the bar the engine package must match. Also removed a dead `librispeech-test-clean-500w.tar.gz` pin whose HF dataset now 401s and which broke the no-arg fetch for everyone.
  - [x] **W2a — `whisper` (offline core) → native engine package. DONE 2026-08-26.**
    - Created `src/models/whisper/` (`graphs.cpp`, `assets.cpp`, `runtime.cpp`, `session.cpp`) with internal headers under `include/engine/models/whisper/`; registered via `audiocpp_add_model` + the ASR composite; resolves by `whisper` / `whisper-offline`.
    - Numerics-identical port of the encoder (2-layer Conv1d stem → GELU(erf), learned absolute pos-emb, pre-LN blocks, final LN) and decoder (token + learned pos-emb, **no post-embed LN**, pre-LN blocks, **tied** logits head). The Whisper attention quirk is verbatim: **q/v/out carry bias, k does not**. The two conformer helpers the arch borrowed are reimplemented locally — **no `src/runtime/` dependency**.
    - **Frontend is reuse, not a port**: Phase-9 `engine::audio::MelExtractor` with Whisper `PerUtterance` normalization over the `.bin`'s own slaney filterbank. That is W2's "unified MelExtractor" scope item, delivered without a second private mel.
    - **Loads the legacy whisper.cpp `.bin`**, not a GGUF — `model_specs/whisper.json` is catalog-only and nothing it names is downloadable. Metadata-only parser + special-token/suppress-list synthesis in `assets.cpp`; payloads stream per tensor; `can_load()` sniffs the `ggml` magic rather than resolving a spec bundle.
    - **Bug found by the gate, recorded for every future family**: `MelExtractor` writes **mel-major**; a graph input `ggml_new_tensor_2d(ctx, F32, n_mels, n_frames)` needs **frame-major**. Feeding it straight through fed the encoder a transposed spectrogram — no crash, healthy-looking mel stats, and confident fluent unrelated text at **94% WER**. Only the end-to-end numeric gate caught it. **Transpose when wiring MelExtractor into a ggml graph.**
    - Gate **`whisper_engine_smoke_test`** is held to **parity with the arch**, not the 10% bound: fails above the 3 word edits `asr_e2e_whisper_wer_test` measures. **Measured: corpus WER 4.34783% (3/69), RTF 0.155 — exactly the arch baseline.**
    - Suite after W2a: **103/103 green**. Arch copy untouched (§4.4 coexistence).
  - [~] **W2b — remaining Whisper scope** *(re-scoped 2026-09-23; W2b.1 + W2b.2 + B16c DONE, W2b.3 open)*: **GGUF loading** (the family's real distribution format — the engine only reads `.bin` today), temperature-fallback ladder + DecodeTelemetry, timestamps, long-form seek continuation, language detection + `<|lang|>`/translate for multilingual variants, initial prompt / `condition_on_prev_tokens`, adapter translation of `transcribe_whisper_run_ext` + chunk traces, batched decode, and the static-topology step graph (W2a rebuilds the step graph per token, which is why its RTF is 0.155 vs the arch's 0.047 — a dispatch cost, not a numerics difference; **keep any static graph GPU-gated until a CPU gate proves it** — the parent's CPU flip in `6c767184` decodes garbage).
    - [x] **Retirement gate built (2026-09-23).** The four transcribe.cpp Whisper C-ABI tests were vendored in Phase 7.1 but never registered; now registered against pinned models and green on the arch: `transcribe_whisper_e2e_smoke` (language detection with no hint, SEGMENT timestamps, run ext: prompts / temperature ladder / thresholds, chunk traces, abort), `transcribe_whisper_tokenize_parity` (HF-identical BPE ids), and their `.bin` twins (multilingual `ggml-tiny.bin` now pinned). The glossary-prompt thresholds are capacity-calibrated (the parent's own build misses them with tiny/base/small), so the gate holds tiny to the arch's measured 0 → 2 hits. **B16c/B17 may not land until the engine passes all four through the C ABI.**
    - [x] **W2b.1 — engine Whisper at arch parity (2026-09-23).** Loads **both** formats through the shared `TensorSource` (GGUF via `GgufMetadata` + TokenizerHub; `.bin` via the now-streaming `WhisperBinTensorSource` — the private manifest streamer is gone, finishing B31). The full recipe lives in a graph-agnostic policy, `src/models/whisper/decoding.{h,cpp}` (HF timestamp rules, `_retrieve_segment`, temperature ladder + compression-ratio / logprob / no-speech gates, prompts + `condition_on_prev_tokens`, unified seek loop, language detection, translate), unit-tested under a scripted fake decoder (`whisper_decoding_test`); generic sampling lives in `framework/asr/sampling` (the arch now calls it too). Results: `speech_segments`, `decode_telemetry` (the chunk traces), `Transcript.language`, honest `truncated`. **Measured**: `.bin` 3/69 and GGUF 3/69 (== arch, identical hypotheses); `whisper_engine_arch_parity_test` — engine == arch word-for-word across language detection, translate, segment timestamps, initial prompt and 83 s long-form (25 segments, 3 windows); one case-only near-tie reported. KV cache stays F32 (arch AUTO policy = `whisper.kv_type=auto`; F16 flips more near-ties).
    - [x] **W2b.2 — C-ABI takeover + B16c. DONE 2026-09-23** (details in the next two items). *Original scope:* adapter translates `transcribe_whisper_run_ext` → `whisper.*` options (±INF → `inf`/`-inf`) and validates at run-validate time; `DecodeTelemetry` → `transcribe_get_whisper_chunk_*`; `CapabilitySet` gains translate / long-form / initial-prompt / temperature-fallback bits (the adapter hard-codes `supports_translate = false` today); `transcribe_tokenize` on adapter models via the hub; `model_specs/whisper.json` declares the `whisper.*` request options (else `prune_request_options_to_contract` drops them). Then re-apply B16c from the stash. Gate: the four `transcribe_whisper_*` tests green on the engine.
    - [x] **W2b.2 landed.** The ArchAdapter serves Whisper end to end:
      - `transcribe_whisper_run_ext`: validated in `run_validate` **before** the prior result is cleared (kind/size, `prompt_condition` in range, ALL_SEGMENTS requires `condition_on_prev_tokens`, no negative temperature / cap, no NaN — the arch only checked these after clearing); every one of its 12 fields forwarded under the option names `model_specs/whisper.json` declares, floats via `%.9g` with the ±INF `_DISABLED` sentinels spelled `inf`/`-inf` (round-trip bit-exact through `std::stof`), `seed` as u32 (the engine parsed it as `int`, so seeds > INT_MAX threw). Vocabulary-dependent rules (a leading `<|startofprev|>` in `prompt_tokens`, special-token literals in `initial_prompt`) stay in the engine, whose `std::invalid_argument` now maps to `ERR_INVALID_ARG` for every family (was `ERR_BACKEND`).
      - `CapabilitySet` gained `timestamp_granularity` (Whisper: Segment → `max_timestamp_kind = SEGMENT`, so WORD requests get `ERR_UNSUPPORTED_TIMESTAMPS`), `supports_translate`, an explicit optional `supports_language_detection` (tiny.en lists `{"en"}` but cannot detect), and `supports_initial_prompt` / `_temperature_fallback` / `_long_form` → the matching `TRANSCRIBE_FEATURE_*` bits.
      - `TaskResult::decode_telemetry` → `transcribe_session::decode_traces` → `transcribe_get_whisper_chunk_count/_trace` (moved with the init functions from `arch/whisper/public.cpp` to `transcribe-family-ext.cpp`); `transcribe_tokenize` on adapter models via a new `ILoadedVoiceModel::tokenize` (Whisper: TokenizerHub, HF-exact).
      - **Adapter-wide defects found on the way (all families)**: `TaskResult::truncated` never reached the C ABI (the moonshine pair has answered `OK` + flag false since B16a/b; it is now `ERR_OUTPUT_TRUNCATED` + flag, per utterance in a batch, as `transcribe.h` requires — Whisper's flag narrowed to windows whose text is actually lost, i.e. not re-entered by the seek); an aborted run discarded everything although `transcribe.h` promises the completed part stays readable (new `ProgressCanceled::partial`; Whisper attaches its decoded windows); `run_batch` left the top-level aliases with `has_result = false` and dropped each utterance's `detected_language` and speaker segments; a transcript without timed rows is now one untimed segment (the builtin arches' convention).
    - [x] **B16c landed 2026-09-23 (the arch is deleted).** `src/runtime/arch/whisper/` (12 files, 7,045 lines) is gone and `whisper` is an `adapter_archs` entry; GGUFs route by `general.architecture`, `.bin` files through the framework sniff. Its parallel `.bin` parser `src/runtime/transcribe-bin-loader.{h,cpp}` (557 lines) lost its last consumer and went too; `whisper_bin_parser_unit` now tests the engine loader and the C-ABI statuses. Contract notes: an unclaimed ggml container answers `ERR_UNSUPPORTED_ARCH` (as the arch did, not the GGUF reader's `ERR_GGUF`); a Whisper-shaped but malformed `.bin` now answers `ERR_UNSUPPORTED_ARCH` too (was `ERR_GGUF`) — the adapter convention `loader_smoke` pins for B13/B14. `.bin` variants are named `whisper-tiny[.en]` like the GGUF `stt.variant`.
      - **Gate, all green on the engine** (`build-cpu-asr-abi`, see below): `transcribe_whisper_e2e_smoke`, `_tokenize_parity`, `_bin_e2e_smoke`, `_bin_tokenize_parity`; `asr_e2e_whisper_wer_test` 3/69; `whisper_engine_smoke_test` / `_gguf_smoke_test` 3/69. `whisper_engine_arch_parity_test` became **`whisper_c_abi_parity_test`**: C ABI vs the engine called directly, now **byte-exact** (text, language, segments, every trace field, truncation) across the same six modes, plus the C-ABI contract (capabilities, WORD rejection, pre-clear rejection keeping the prior transcript, engine rejections → `ERR_INVALID_ARG`).
      - **Build-tree consequence**: the `core` model set links no engine models, so with the arch gone Whisper has no C-ABI path in `build-cpu-core` (moonshine has been in the same position since B16a/b). The Whisper C-ABI tests are guarded on `"whisper" IN_LIST AUDIOCPP_LINKED_MODELS`; they run in the new **`build-cpu-asr-abi`** (`MODEL_SET=asr`, UNIFIED_ABI + TRANSCRIBE_ARCHES + tests, server off). The pre-existing `build-cpu-asr` has tests and the ABI off.
      - `stash@{0}` (the premature 2026-09-23 B16c) is superseded by this; kept until the user okays dropping it.
    - [ ] **W2b.3 — performance (open)**: lockstep batched decode (the engine's `run_batch` is serial per utterance; the arch batched), and a static-topology step graph (the engine rebuilds the step graph per token: RTF 0.155 vs the arch's 0.047). **Keep any static graph GPU-gated until a CPU gate proves it** (parent `6c767184`).
    - [x] ~~**B16c parked (2026-09-23)**~~ *(resolved: landed properly, see above)*: an uncommitted working-tree deletion of `src/runtime/arch/whisper/` (routing `whisper` GGUF/.bin to the engine) was found and moved to `git stash` ("B16c premature…"). It would have dropped GGUF loading, long-form, timestamps, language detection and translate, and it accepted `transcribe_whisper_run_ext` without translating a single field (silently ignored knobs, L11). Recoverable; re-apply only after the gate above is green on the engine.
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

- **Current Timestamp**: 2026-09-23 (L13 dependency sync, W2b complete through B16c, **audio.cpp merged to `9bdd1d90` (v0.8.2)**; everything committed and pushed on `main`). `build-cpu-core` **121/121**, `build-cpu-asr-abi` **115/115** (1 clean skip each); `build-cuda-core` last measured 99/99 before B16c — CPU trees are the routine test target (user preference).
- **2026-09-23 increment — audio.cpp merge `c0b26a50..9bdd1d90` (70 commits), recorded as `38769d51`**: ggml deltas first ported as `patches/ggml/0012-0015` (`f4d8e31e`: lowering API, ssm_scan fusion API with the flag moved to slot 1 because 0.24.0 keeps `K` in slot 0, MUL_MAT_ACC + SNAKE_1D with CPU kernels, CPU refusal of ops it cannot compute); merge conflicts resolved per `docs/upstream/audio_cpp_merge_9bdd1d90.md`; one semantic fix the textual merge hid (`audio8_tts` passes `K = 1` to `ggml_ssm_scan`). Sortformer NeMo-GGUF layout unchanged (DER 0.3288, chunked == whole-window 1.8e-7). **Not ported**: the CUDA / Metal / Vulkan halves of audio.cpp's ggml work (LiveAvatar/Wan on CUDA and audio8/Breeze Metal fast paths wait on them).
- **2026-09-23 increment — Whisper C-ABI takeover (W2b.2) + B16c**: see the W2 block. Arch + parallel `.bin` parser deleted (7,602 lines); adapter translates the Whisper run ext, capabilities and chunk traces; four adapter-wide defects fixed (truncation flag, abort partial results, batch top-level aliasing + per-utterance language, `invalid_argument` → `ERR_INVALID_ARG`); new `build-cpu-asr-abi` tree runs the model-backed C-ABI gates.
- **2026-09-23 increment — dependency sync (L13) + Whisper retirement gate**:
  - **ggml `36da5713` (0.22.0) → `456172ec` (0.24.0)**, matching parent transcribe.cpp. First captured ~1,800 lines of UNTRACKED audio.cpp fork deltas as patches 0008-0011 (round-trip exact at the old pin), then rebased all 11 patches; 0011 shrank 4 → 1 hunk (upstream converged on #508). **0007 replaced by transcribe.cpp's CUDA pool patch** — ours never overrode `ggml_cuda_pool::clear()`, so `trim_pools` freed nothing. Details: `external/ggml/UPSTREAM` PIN HISTORY.
  - **Windows CUDA build fixed** (was broken for every MSVC + GGML_CUDA Release build): `AUDIOCPP_STRIP_DEAD_CODE` passed `/Gy /Gw /GF` to nvcc (now `COMPILE_LANGUAGE:C,CXX`-scoped), and four Phase 7/9 tests linked the `engine_core` OBJECT library directly, missing the CUDA iSTFT / torch-random kernels (now link `engine_runtime`).
  - **transcribe.cpp triaged `2102bca..0a67b65b`** (54 commits) → `docs/upstream/transcribe_cpp_triage.md`, watermark advanced. Actioned: ITN/PNC/diarize tri-state inversion in the C-ABI adapter (live bug), #165 decode budget (runtime arches), `sync-ggml.sh --check`. **Found and bisected a parent regression**: `6c767184` makes the parent's Whisper decode garbage on CPU (static step graph) — do not adopt that hunk.
  - **audio.cpp `c0b26a50..487800f5` (61 commits) trial-merged** in a scratch clone; **blocked on ggml** (upstream calls new audio.cpp ggml ops). Plan: `docs/upstream/audio_cpp_merge_9bdd1d90.md`.
  - **B16c (premature whisper-arch deletion) parked in `git stash`**; the Whisper C-ABI retirement gate (4 tests, pinned GGUF + `.bin`) is registered and green on the arch.
  - Tooling: `sync-deps.sh` now fetches by default (it reported "0 behind" at 61 behind), reads the transcribe.cpp watermark, and has `--verify-ggml`; shared `engine/framework/assets/gguf_metadata.h` for typed GGUF KVs.
- **Last Completed Increments (before 2026-09-23)**:
  - **Upstream Synchronization (`0xShug0/audio.cpp:main@6d530f4`) — 88 ahead, 0 behind**:
    - Audited, resolved, and merged 35 upstream commits from `c79e588` to `6d530f4`.
    - Integrated new model families: `AudioSR`, `ControlFoley`, `FireRedTTS3`, `FireRedAudio`, `MiDashengLM-Gen`, `Echo-TTS`, and `IBM Granite Speech 5.0 470M TurboCTC ASR`.
    - Added reusable modules & conditioners: `mel_latent_vae44k_runtime`, `redae_codec_runtime`, `fish_dac_codec_runtime`, `s3_tokenizer`, `hifigan_vocoder`, `cav_mae_st`, `clap_audio`, `musicgen_style`, `open_clip`, `synchformer`.
    - Extended WAV reader format coverage in `src/framework/audio/wav_reader.cpp` (PCM8/32, float64, A-law, $\mu$-law, `WAVEFORMATEXTENSIBLE`).
    - Added server `--idle-unload-ms` and pre-load memory guard check (`app/server/model_memory.cpp`, `app/server/runtime.cpp`).
    - Integrated Parakeet TDT VAD chunking, Qwen3 timestamp clamping & word diagnostics, ACE-Step caption rewriting, VibeVoice 7B improvements, and CUDA stream-k staging zeroing.
    - Updated WebUI with native themes, i18n localization, and Arena comparison workflow.
    - Resolved merge conflicts in `CMakeLists.txt`, `README.md`, `app/server/runtime.cpp`, `external/ggml/src/ggml-cuda/mmq.cuh`, `include/engine/models/qwen3_asr/types.h`, and `src/models/qwen3_asr/session.cpp`.
    - Preserved all `speech.cpp` native engine ASR ports (`moonshine`, `moonshine_streaming`, `whisper`), transcribe runtime, and ALIAS product targets.
    - Total CTest targets: **112/112 100% green** (108 passed, 4 clean skips on unpinned weights).
  - **Phase 10.5 — all 5 overlapping families retired** (B11 9cc5457 / B12 fdaa9a5 / B13 e3e7eac1 / B14 1be9ac40 / B15 a8b7a03b); the sortformer_diar step 2 feature-merge (111/111 green).
  - **Phase 11 Wave W2a — Native Engine Whisper (offline core)** — `100% DONE` (103/103 green).
  - **Dual Parentage recorded + ggml bumped to `36da5713` (v0.22.0)**:
    - **`speech.cpp` is equally a child of `audio.cpp` and of `transcribe.cpp`** — forking audio.cpp was a convenience, not precedence.
    - `scripts/sync-deps.sh` reporting drift over all three sources.
  - **Phase 11 Wave W1b — Native Engine Moonshine-Streaming** — `100% DONE` (Wave W1 complete, 101/101 green).
   - **Phase 11 Wave W1a — Native Engine Moonshine (offline)** — `100% DONE` (96/96 green).
   - **Phase 11a foundation (2026-09-14, commit c59b15a0)** — `AsrResult`/`AsrLimits` created; `TaskResult.truncated` added + propagated through all 3 sessions (L11/W2a fix); B29 VAD dedup complete (`transcribe-vad*` deleted → `plan_vad_audio_chunks`/`append_chunk_speech_metadata`).
- **IMMEDIATE NEXT TASK FOR AGENT** *(2026-09-23)*:
  0. **Get the user's go-ahead to commit** the 2026-09-23 working tree (suggested split: ggml capture 0008-0011 at the old pin → ggml 0.24.0 bump + 0007 swap → CUDA build fixes → adapter tri-state fix → #165 → Whisper gates + pins → TokenizerHub + test-integrity fixes → W2b.1 engine Whisper → W2b.2 adapter + adapter-wide fixes → B16c deletion → tooling/docs).
  1. ~~Phase 11 W2b.2 — C-ABI takeover + B16c~~ **DONE 2026-09-23**. W2b.3 (batched decode, static step graph) is performance work and can wait behind 2.
     1b. Open item found during B16c: `sense_asr`'s default Silero VAD path (`assets/framework/models/silero_vad`, spec default) is CWD-relative, so a C-ABI caller outside the repo root gets `ERR_BACKEND` on every run (its default `audio_chunk_mode=auto` needs the VAD). The gates now run from the source root; the real fix is resolving the VAD relative to the model / an asset root.
  2. ~~audio.cpp merge `487800f5`~~ **DONE 2026-09-23** as `38769d51`, extended to `9bdd1d90`. Follow-ups: (a) port the CUDA lowering kernels (62c664af, 82b4dc3a, bef51fcc) as patches 0016+ — needs a CUDA build, so schedule it with the user; (b) Metal (4bb1a204 + 712dd75a Metal, incl. a real `kernel_ssm_scan_f32` fix) and vulkan-shaders-gen hardening (5785167a, eb8e21bd); (c) register `confucius4_r2t2` and `nemotron_3_diar` in `family_registry.cpp` once their GGUF architecture names are confirmed (C-ABI routing); (d) CPU kernels for the five CUDA-only ops if LiveAvatar should run on CPU.
  3. transcribe.cpp open items in ledger order. **DONE 2026-09-24**: S3 conformer memory (-20% parakeet peak, transcripts byte-identical), `63baefe6` parakeet tail loss, `9aa6599f` scratch release, `e2f82cb6` Voxtral-RT offline delay 30, `6d7fc06f` UTF-8 intake - plus a real engine bug the parent's interleave test exposed (Voxtral-RT cached graphs decoded garbage on reuse; fixed). **Next**: S2 reclaim + S1 threading end state (backend / batch-util / load-common are 300+ lines behind the parent; S1 is performance, so measure on parakeet), engine halves of #165, R2T2 (S4) on the merged `confucius4_r2t2`, the granite_nar remainder (needs `585b98f7`).
  - Standing rules: no new entry points in `capi/audiocpp.h` (F14); every ASR result carries `truncated` honestly (L11); `scripts/sync-deps.sh` (+ `--verify-ggml`) at every phase boundary (L13).
