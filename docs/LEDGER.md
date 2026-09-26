# speech.cpp — decision, family and deletion ledger

> **What this file holds:** records only — decisions, family migrations, deletions and parent dispositions. It is the one ledger that [`PLAN.md`](../PLAN.md) points to.
> **Consolidated:** 2026-09-26, from `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md` (V6), `FUSION_ROADMAP_PLAN.md` v6.0 (RM) and `MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md`. The originals are in [`docs/archive/`](archive/).
> **Id namespaces** (old ids are kept so earlier citations still resolve):
> - `V6-Dn`: V6 decision log.
> - `MR-n`: V6 merge log (formerly "Rn").
> - `V7-Dn`: 2026-09-26 checkpoint decisions.
> - `Bn`: deletion rows (roadmap Appendix B).
> - `Ln`: laws (PLAN §3).
> - `RISK-n`: risks.
> **Rule (L7):** every deletion gets a row naming what was deleted, what replaced it, the gate that proves equivalence, and the revert commit.

---

## 1. Decisions in force

### 1.1 Checkpoint decisions (2026-09-26; taken by the user: "take all the decisions you suggested")

| Id | Decision | Consequence |
|---|---|---|
| **V7-D1** | **`speech.h` is the only first-party C ABI.** speech.cpp's own `capi/include/audiocpp.h` (1216 lines, `AUDIOCPP_BUILD_CAPI`) is **retired, not shimmed**, once `speech.h` reaches parity. Upstream audio.cpp's `include/audiocpp.h` (536 lines, `AUDIOCPP_BUILD_C_API`, default OFF) stays as the parent's file, untouched. `transcribe.h` stays as a compat shim until bindings move. | Supersedes the RM "freeze + shim `audiocpp.h`" plan and standing rule F14's end state. Deletion row **B32** becomes "retire". Phase S3 in PLAN. |
| **V7-D2** | **CrispASR is reference parent #3, mined by family, never merged.** Its code is read as a specification; individual functions may be copied with attribution (MIT). The oracle stays PyTorch. | [`docs/upstream/crispasr_triage.md`](upstream/crispasr_triage.md), watermark `dfcdacae`. Rules C-R1…C-R6 in PLAN §6. |
| **V7-D3** | **Upstream-first.** Before porting any family, check whether audio.cpp has it or is building it. Offer generic work (transcribe contracts, `framework/asr`, parity tooling) upstream so the fork delta shrinks. | Step 0 of every family wave. |
| **V7-D4** | **"And more" includes** text post-processing (punctuation, casing, segmentation), audio and text LID, generalized alignment, text MT (speech→translation), music analysis, and watermark/provenance. **Excluded:** a general text-chat LLM runtime (that is llama.cpp's job). | New TaskKinds, each contract-first (L1). Phase C1. |
| **V7-D5** | **Non-commercial or restricted-licence models** are allowed only as opt-in specs carrying `license` + `commercial_use` metadata, behind a pre-download licence gate. Never as defaults. | Spec lint in C0. |
| **V7-D7** | **Two profiles, one codebase** (user directives, 2026-09-26). **DEV (full):** every family plus audio.cpp's server + WebUI, CLI, GGUF tools, model manager, tests and benches. Kept and maintained at maximum functionality as the test bench for every model. **BUNDLE (embed):** what final apps ship. `libspeech` + `speech.h` + Rust crate with **only the selected families** (down to one STT or TTS family), like transcribe.cpp's `TRANSCRIBE_MODEL_SET` / Cargo features. No DEV component, no audio-device I/O, no Python, no network, no CWD-relative assets. **Primary consumers (both primary):** FreeSpeech desktop (#1; successor of ZER0 and S2B2S) and FreeSpeech Android (#2; Rust cdylib + JNI). The LLM brain (llama.cpp / LiteRT-LM) stays outside. Rule: DEV tools depend on the library, never the reverse. | PLAN §1 (profiles, consumers) and phase E (E0 model-set composites + Cargo features, E6 Android). SC3 is PCM push/pull; SC4 gates on Multi-STT concurrency. |
| **V7-D8** | **Lineage and priority** (user, 2026-09-26). The author's path is S2B2S → **NairoDorian/transcribe.cpp** (most engineering time; well optimized) → ZER0 → **speech.cpp** → FreeSpeech. audio.cpp is pure upstream with no work of the author's. speech.cpp is a new project based on both, and must absorb **all** optimizations and latest models of both. | An engine port must match the arch it replaces on RTF and peak memory (≤ 1.1×), not only WER (PLAN rule R5, 11b step 5). transcribe.cpp triage treats optimizations as must-adopt. |
| **V7-D9** | **Benchmark acceptance** (user directive, 2026-09-26). A port or numeric change must be **no slower and no less accurate** than both parents that have the family (WER/CER on a FLEURS multilingual subset; RTF), on the same GGUF, machine, backend and threads. Runs: ≥ 3; run 1 = warm-up (analyzed, never averaged); mean of runs 2..N with N = 3 in development, 5 for acceptance, ≤ 10. Keep suites small. Backends: CPU while developing (fastest build); CPU + CUDA + Vulkan-NVIDIA + Vulkan-Intel before acceptance; Android when E6 lands. | `docs/benchmarking.md`; PLAN rule R10, L12 updated, 11b step 5, C0.6 harness (port of transcribe.cpp `scripts/bench` + `scripts/wer`). |
| **V7-D10** | **Generated architecture map in the pre-commit habit** (user directive, 2026-09-26). `ARCHITECTURE.md` is generated with repomix (as in S2B2S and ZER0) and regenerated by `.githooks/pre-commit`. README and AGENTS send every reader, human or LLM agent, there first. | PLAN rule R11, C0.7 (done); `scripts/arch/`. |
| **V7-D6** | **Process diet:** one plan (`PLAN.md`), this ledger, `LESSONS.md`, and `CHANGELOG.md`. Status is generated by `scripts/status.sh`. Superseded plans are archived. | Done 2026-09-26 (see PLAN §0). |

### 1.1b Pending strategic decisions (proposed 2026-09-26; see PLAN §8)

| Id | Proposal | Status |
|---|---|---|
| SC1 | `speech.h` designed from FreeSpeech's use cases (spec in S1.8) | **pending user** |
| SC2 | Thin-fork topology; upstream-first PRs; fork delta in upstream-owned files measured and falling | **pending user** |
| SC3 | Real-time `ConversationSession` (AEC, VAD, KWS, streaming ASR, turn detection, streaming TTS, barge-in) + latency gates | **pending user** |
| SC4 | `ResourceManager` (residency, VRAM budget, shared backends, concurrent sessions) | **pending user** |
| SC5 | Catalogue tiers A/B/C, with FreeSpeech's needs deciding port order | **pending user** |
| SC6 | Test tiers (quick **adopted** 2026-09-26) + nightly scoreboard | quick tier done; scoreboard pending |
| SC7 | Family manifests → generated build; family plugins later | **pending user** |

### 1.2 V6 decisions still in force
(Superseded ones — V6-D2, D3, D4, D14, D15, D18, D19, D23 — are listed in §1.3.)

| Id | Decision |
|---|---|
| V6-D1 | The C ABI is the primary public surface; C++ headers are internal. (Now realized as `speech.h`, V7-D1.) |
| V6-D5 | `BackendPlan` + dynamic backend DLLs are part of the ABI surface. |
| V6-D6 | `transcribe_status` enum + `api_guard_*` at every ABI boundary (carried into `speech_status`). |
| V6-D7 | `safe_*` teardown + lint. Retrofit of the audio.cpp families is **partial** (sub-task 0.J open). |
| V6-D8 | Size-aware ABI structs (`struct_size`, `copy_out_prefix`). |
| V6-D9 | Golden manifests + per-tensor tolerances for all families (`1e-4×p99_abs` / `1e-5×rms`). |
| V6-D10 | Streaming state machine with a 3-checkpoint begin for all streaming tasks. |
| V6-D11 | Size-aware typed extension structs; the `options` map stays internal. |
| V6-D12 | Spec-decode sampler for all AR decoders. |
| V6-D13 | audio.cpp text normalization for PNC/ITN, exposed as ABI enums. |
| V6-D16 | Single vendored ggml pin. The pin is `456172ec` (0.24.0); **due to move to ≥ 0.25.3** (transcribe.cpp floor). |
| V6-D17 | Silero VAD pinned to the latest stable release. |
| V6-D20 | No `-Werror`; per-target warnings; vendored code exempt. |
| V6-D21 | transcribe-common WAV loader in `examples/common`. |
| V6-D22 | transcribe.cpp warning/visibility policy for runtime code (lapses with 11c). |
| V6-D24 | Ported families keep their GGUF `general.architecture` strings. **Arch strings are immutable ABI.** |
| MR-3 | One implementation per duplicated family, chosen by variant coverage, WER and integration depth. The loser's features merge into the winner. |
| MR-4 | Licensing: Apache-2.0 with MIT notices preserved. **Legal sign-off still open before 1.0.** |
| MR-8 | Sortformer pin: `nvidia/diar_streaming_sortformer_4spk-v2@5240a640`, q8_0 GGUF sha256 `0679cfeb…`, CC-BY-4.0. |

### 1.3 Superseded — do not act on these

| Id | Was | Replaced by |
|---|---|---|
| V6-D2 | `Arch` vtable as unified dispatch | engine session contract (RM Appendix F) |
| V6-D3 | transcribe `Loader` on the ABI path | `ModelRegistry` + `TensorSource` |
| V6-D4 | hybrid sessions | one `RuntimeSessionBase` |
| V6-D14 | transcribe `MelFrontend` / `KaldiFbank` | engine `MelExtractor` / `kaldi_fbank` |
| V6-D15 | transcribe `causal_lm` backbone | framework modules + `causal_lm_ops` |
| V6-D18 | `transcribe_task_*` ABI for all tasks | `speech_*` in `speech.h` |
| V6-D19 | CMake-generated `transcribe-arch.cpp` registry | `family_registry` + `ModelRegistry` |
| V6-D23 | ABI symbols stay `transcribe_*` | `speech_*`, with `transcribe_*` as a shim |
| V6-D25 | Freeze parents at Phase 4; `reference_commits.md` ledger | L13 continuous sync, `sync-deps.sh`, triage watermarks |
| RM F14 end state | `audiocpp.h` becomes a shim | V7-D1: retired |
| MR-1, 2, 5, 6, 7, 9–13 | historical closures (ggml convergence, WER validation, older syncs) | see `docs/archive/TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md` §0.0 |

---

## 2. transcribe.cpp arch families — migration state (2026-09-26)

"Uncommitted" means the code exists in the working tree after `d04fd552`. **Its build and test state is recorded in [`reports/checkpoint_2026-09-26.md`](reports/checkpoint_2026-09-26.md) §2**, not assumed.

### 2.1 Retired (arch deleted)

| Arch | Row | Commit | Engine package | Gates |
|---|---|---|---|---|
| qwen3_asr | B11 | `9cc5457` | `qwen3_asr` | `qwen3_asr_engine_smoke_test` (2/69), `qwen3_asr_bpe_parity_test`, `asr_e2e_qwen3_asr_wer_test` |
| voxtral_realtime | B12 | `fdaa9a5` | `voxtral_realtime` | `voxtral_realtime_engine_smoke_test` (2/69), `_delay_test`, `_ext_abi_test` |
| sensevoice | B13 | `e3e7eac1` | `sense_asr` (community) | `asr_e2e_sense_asr_wer_test`, `asr_stream_text_sense_asr_wer_test` |
| funasr_nano | B14 | `1be9ac40` | `fun_asr_nano` | `asr_e2e_fun_asr_nano_wer_test` |
| sortformer (standalone family only) | B15 | `a8b7a03b` | `sortformer_diar` | `sortformer_diar_streaming_engine_test`, `_scheduler_test`, `_ext_abi_test`. The core stays inside the parakeet arch. |
| moonshine | B16a | `f3c2ffca` | `moonshine` | `moonshine_engine_smoke_test` (1/69), `asr_e2e_wer_test`, `asr_e2e_edits_test` |
| moonshine_streaming | B16b | `f3c2ffca` | `moonshine_streaming` | `moonshine_streaming_engine_smoke_test`, `asr_stream_text_wer_test` (3/69, divergence 0) |
| whisper (+ bin loader) | B16c | `beb24538` | `whisper` | `transcribe_whisper_{e2e_smoke,tokenize_parity,bin_e2e_smoke,bin_tokenize_parity}`, `asr_e2e_whisper_wer_test` (3/69), `whisper_c_abi_parity_test` (byte-exact), `whisper_engine{,_gguf}_smoke_test` |

### 2.2 Live (arch dir present under `src/runtime/arch/`)

| Arch (LOC) | Engine target | State | Blockers / notes |
|---|---|---|---|
| canary (4176) | `canary_asr` (upstream #568) | verdict pending; dual layout uncommitted | Only 180m-flash loads (1b variants are rejected by a KV check). Reflect vs NeMo constant padding. |
| canary_qwen (3354) | `canary_qwen` (new) | port uncommitted | **Private KvCache**; NeMo mel copied verbatim, so MelExtractor PerFeature normalization needs fixing; peaks at ~9 GB. |
| cohere (4290) | `cohere_asr` (#568) | verdict pending; dual layout uncommitted | Byte-identical to the arch through the C ABI (2/69). RTF 0.455 vs 0.37. |
| gigaam (2967) | `gigaam` | port uncommitted | Retires arch mel (B3). MelExtractor `PadMode::None` non-pow2 fix untested. HF pin names unverified. |
| granite (5214) | `granite_speech` (canonical) | port uncommitted | **Private KvCache**. Real arch bug fixed: `shaw_attn.cpp` weight dims were swapped, so both granite arches could never encode (since `9b34fd2d`). Per-variant caps hook needed. |
| granite_nar (3045) | `granite_nar` | port uncommitted | Parent `585b98f7` skew Shaw bias + `b174a427` CTC head. Arch-side remainder open. Judge on WER (op order differs). |
| medasr (2373) | `medasr` | port uncommitted | Gated HF licence. The docs contradict each other on whether it is pinned — verify. |
| moss (3515) | `moss_transcribe_diarize` | verdict pending | Adapter: segment speaker should be the turn's own id; NULL-params defaults. Needs BF16 for the 2.08% bar. |
| parakeet (9443) | `parakeet_tdt` (community) | arch canonical; engine dual layout uncommitted | Engine lacks: multitalker, nemotron-3.5 prompt MLP, `chunked_limited` streaming, stream / buffered-stream exts, batched variable-length encode, `tokenize`. Row B9 (delete `parakeet_tdt`) is **void**: that package is now the winner. |
| sortformer (1757) | — | core kept for parakeet multitalker | Retires with parakeet. |
| voxtral (3587) | `voxtral` | port uncommitted | **Private KvCache**; own tekken tokenizer; serial `run_batch`. The arch fails on a second run of the same session. |

**L5 violation to resolve:** upstream `src/models/moonshine_asr/` sits beside our `moonshine` port and is missing from `family_registry.cpp`. MR-3 verdict needed: pick one, merge the other's features (upstream just added chunking), alias the loser.

**Other duplicates to decide under MR-3:**
- `granite_speech` vs community `granite5asr`
- `sortformer_diar` (models) vs community `sortformer_diar`
- `vibevoice_asr` vs `vibeasr`
- `nemotron_asr` / `parakeet_tdt` / arch parakeet (three NeMo transducer stacks)

---

## 3. Deletion ledger (B rows)

| Row | Target | Status |
|---|---|---|
| B1 | `transcribe-mel` (1077) | open |
| B2 | `transcribe-kaldi-fbank` (373) | open (only dropped from the OBJECT lib in B14) |
| B3 | `arch/gigaam/mel` | open; with the gigaam retirement |
| B4 | `transcribe-tokenizer` (1165) | open |
| B5 | `runtime/sanm` (395) | open |
| B6 | `runtime/conformer` (1557) | open |
| B7 | `granite_conformer` (230) | open |
| B8 | `runtime/causal_lm` (1347) | open |
| B9 | ~~`community_models/parakeet_tdt`~~ | **void**: that package is the dual-layout engine winner |
| B10 | `arch/parakeet` (9439) | open; blockers in §2.2 |
| B11–B15 | qwen3_asr, voxtral_realtime, sensevoice, funasr_nano, standalone sortformer | **done** (§2.1) |
| B16a / B16b | moonshine / moonshine_streaming | **done** `f3c2ffca` |
| B16c | whisper + bin loader (7602) | **done** `beb24538` |
| B17–B27 | voxtral, canary, canary_qwen, cohere, gigaam, granite, granite_nar, medasr, moss (~34k) | open (11b) |
| B28 | `transcribe-arch-adapter` (1884 + growing) | open (11c) |
| B29 | `transcribe-vad*` | **done** `c59b15a0` |
| B30 | private KvCaches + argmax impls | **done** (`c59b15a0`, `a6d02849`, `d43f754e`). **Regressed** by the uncommitted canary_qwen / granite_speech / voxtral ports. Re-close in PLAN phase S2. |
| B31 | Whisper private `.bin` parser + encoder | **done** `5318d532`, `f6a9d2b2` |
| B32 | speech.cpp `capi/` `audiocpp.h` + `audiocpp_capi.cpp` (~3.8k) | open. **Retire (not shim)** per V7-D1, after `speech.h` parity (S3). |
| B33 | rest of `src/runtime` (~20k) | open (11c) |

---

## 4. Parent dispositions (pointers)

| Parent | Ledger | Watermark / merge-base | Due |
|---|---|---|---|
| audio.cpp (`upstream`) | merge commit messages; last report [`archive/upstream/audio_cpp_merge_9bdd1d90.md`](archive/upstream/audio_cpp_merge_9bdd1d90.md) | merge-base `57186a82`; 21 behind at 2026-09-26 (`4d88768f` is a no-op re-release of `9bdd1d90`) | S0 |
| transcribe.cpp (`../transcribe.cpp`) | [`upstream/transcribe_cpp_triage.md`](upstream/transcribe_cpp_triage.md) (**parsed by `scripts/sync-deps.sh` — do not move or reword the `Triage watermark:` line**) | `0a67b65b`; 24 untriaged at 2026-09-26, including ggml 0.25.3 floor, TQ1_G128, error-code ABI `96b7c9c1` | S0 |
| transcribe_cunba | — | `883beea`. Unique content: sortformer push-audio streaming session (SFPS), build provenance, Rust dynload | S0: verify SFPS against `community_models/sortformer_diar/stream_schedule.cpp` |
| audio_cunba | — | 0 unique commits; dropped as a reference (V7-D6) | — |
| CrispASR (`../CrispASR`) | [`upstream/crispasr_triage.md`](upstream/crispasr_triage.md) | `dfcdacae` | per family, on demand |

---

## 5. Risks (live)

| Id | Risk | Mitigation |
|---|---|---|
| RISK-F4 | Upstream merge conflicts grow with fork delta | V7-D3 upstream-first; replace function bodies not files; new family dirs are conflict-free |
| RISK-F9 | Tolerances calibrated on one model | recalibrate per family; gates only ratchet (L8) |
| RISK-F10 | Licence: Apache-2.0 + MIT notices; model licences | MR-4 sign-off before 1.0; V7-D5 metadata |
| RISK-F11 | ggml patch drift (patches silently lost on sync) | `sync-ggml.sh --check` after every audio.cpp merge |
| RISK-F13 | Duplication recurs in new ports | PLAN S2 grep gates, enforced before any new port |
| RISK-F15 | `TRANSCRIBE_DUMP_DIR` dump points live in `src/runtime` | move them before 11c |
| RISK-N1 | Scope (~160 families) exceeds maintainer capacity | waves of 3–5 with gates; archive WIP families instead of porting them |
| RISK-N2 | Weak past agents left false claims or dead tests | checkpoint audit (2026-09-26); LESSONS A1–A3; generated status |
