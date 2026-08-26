# Master FUSION Roadmap Plan — Convergence of `audio.cpp` & `transcribe.cpp` into `speech.cpp`

> **Document Status**: Authoritative Master Plan & Architectural Blueprint
> **Target System**: `speech.cpp` — the single, fully-fused native speech & audio intelligence framework
> **Version**: 6.0 (Code-Grounded Re-Review Edition)
> **Date**: 2026-08-26
> **Supersedes**: v5.0 (2026-08-23). See [§0.4](#04-what-changed-structurally-from-v5) for the structural changes and [`docs/reports/fusion_review_2026-08-26.md`](docs/reports/fusion_review_2026-08-26.md) for the review that produced them.
> **Tree audited at**: `5e1a7e5` (`main`, 66 commits ahead of `upstream/audio.cpp@c79e588`, 0 behind); parents read at `audio.cpp@c79e588`, `transcribe.cpp@2102bca`

---

## 0. How to Read This Document

### 0.1 Authority and scope

| Document | Role | Relationship to this plan |
|---|---|---|
| `FUSION_ROADMAP_PLAN.md` (this file) | **Forward architectural blueprint.** What the end state is, why, and the gated sequence to reach it. | Authoritative for architecture, sequencing, and acceptance gates. |
| `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md` | **Execution ledger and reference appendices.** Merge log R1–R13, decision log, risk register, per-family upgrade maps, rollback procedures, file inventories. | Still authoritative for *decisions already taken* (R1–R13). This plan cites V6 by decision id and never silently overrides one. Where v4 of this plan contradicted a V6 decision, §2.5 records the reversal and the evidence. |
| `progress.md` | **Session-by-session state snapshot.** | Descriptive. This plan corrects two of its status claims (§2.4). |
| `CHANGELOG.md` | **User-facing release notes.** | Descriptive. Two entries are corrected in §2.4. |
| `docs/porting/*` | **Per-family porting procedure** (8 stages: intake → oracle → convert → cpp → quants → bench → WER → ship). | Normative for every family migration in Phase 11. |

**Rule**: no phase in this plan is "done" because a document says so. A phase is done when its **exit gates** — a named, runnable CTest target or script — pass in CI. Documentation follows the gate; it never precedes it.

### 0.2 Verification protocol

Every factual claim in §2 carries the command that produced it. Claims without a command are marked `[UNVERIFIED]` and must not be used to justify deleting code. This convention exists because of a documented failure mode in this repository:

> **Doctrine (retained from `progress.md`):** *"'Missing assets' is now 5-for-5 wrong in this repo. Treat the label as unverified until a failure is reproduced and root-caused once."*

v5.0 generalises that doctrine: **any status claim is unverified until reproduced.** §2.4 applies it to this project's own completion claims and finds two "100% Done" features that are inert.

### 0.3 What changed structurally from v4

v4 was a 547-line target-state sketch. v5 keeps its vision and adds the three things it lacked:

1. **A verified ground-truth section** (§2) — v4's file paths, state-machine names, threshold values and completion claims did not match the tree.
2. **A reachability analysis** (§2.3) — the most consequential omission in v4. A large fraction of what both projects contribute is currently *unreachable from any public surface*, which changes what the roadmap is actually for.
3. **A safety net before demolition** (Phase 7) — v4 proposed deleting ~74 kLOC of ASR code whose regression suite was never ported into this repository. v5 makes porting that suite the entry gate for every deletion.

### 0.4 What changed structurally from v5

v6.0 is the result of re-reading **both parents from their code** — `audio.cpp@c79e588` and `transcribe.cpp@2102bca` — before re-reading this plan, then comparing (full method and evidence: `docs/reports/fusion_review_2026-08-26.md`). The v5 architecture survives; five things it lacked or got wrong are corrected:

1. **An ASR runtime layer** (§4.3, §5.6, Phase 11a). v5's target state had a session layer and a modules layer with nothing between them for ASR. The cost is now measured: the three families migrated so far each carry a private KV cache and a private greedy decode loop, and Whisper re-implemented an encoder the framework already ships as `WhisperEmbeddingModule`. Twelve more families would pay the same multiplier. The layer — one `EncDecKVCache`, shared decode drivers, cross-indexed result rows, the limits contract — is built once, the three ports are re-based onto it, and every later port becomes graphs + assets + a thin session.
2. **`src/runtime/` is deleted in full** (§5.7, Phase 11c). v5 kept the transcribe dispatcher, loader, model/session bases, VAD and bin-loader "as what the C ABI genuinely needs." That would make `speech.h` a three-layer implementation and freeze the `Arch` vtable as a permanent second session contract. Phase 8 moved the dispatcher's discipline into `StreamingSessionBase`; nothing is left for the dispatcher to own. `speech_capi.cpp` sits directly on the engine base classes. Every line of the 95 kLOC gets a ledger row.
3. **Phase 12 is pulled forward** and the Phase-10 verdicts are executed **now** (§8). The "third façade" v5 warned against already ships by default (`capi/audiocpp.h`, 55 functions, no `struct_size`), and Phase 10's central deliverable — *merge the loser's features before retirement* — was never performed (0 references to speculative decoding in `models/qwen3_asr`, 0 to cache-aware windows in `models/voxtral_realtime`, 0 to streaming presets in `models/sortformer_diar`; zero deletions executed). Both are cheaper to fix before the remaining ports than after.
4. **Three contracts transcribe enforces that the engine lacks** enter the doctrine: never truncate silently (L11 — W2a currently trims audio > 30 s and drops the flag); measure the real flow with `PASSOVER.md`'s methodology (L12); dual parentage with the dependency-sync routine at every phase boundary (L13). §4.2 gains rows for input limits, backend execution (`BackendPlan` scheduler fallback vs single-backend `GraphExecutor`), concurrency, and product registration.
5. **Methodology parity for `audio.cpp`'s own families is a per-phase quota**, not Phase 14.3. The Master Key says both parents learn in parallel; v5 scheduled the "audio.cpp learns from transcribe" half last.

V6 decisions D2, D3, D4, D14, D15, D18, D19 and D23 predate the engine-spine architecture and are recorded as superseded in Appendix F — they were never formally reconciled before.

---

## 1. Executive Vision

`speech.cpp` is the convergence of two mature C++ audio-AI codebases:

- **`audio.cpp`** — *breadth.* 42 core + 12 community model families spanning TTS, voice cloning, source separation, diarization, VAD, forced alignment, neural codecs, denoise, super-resolution, singing-voice conversion and MIDI; a model-spec catalog with package management; CLI, HTTP server, WebUI and workflow layers; a 14-task C ABI.
- **`transcribe.cpp`** — *depth.* 18 production ASR/diarization architectures with reference-oracle numerical parity, per-tensor tolerance calibration, a rigorously size-aware and exception-contained C ABI, a 4-state streaming lifecycle with an append-only commitment contract, batched offline decode across 16 of 18 families, and four shipped language bindings generated from the public header by an AST-driven, CI-gated generator.

### 1.1 The Master Fusion Principle

> **"The two projects learn from each other in parallel — each is the other's teacher and student — and merging them improves them both at the same time."**

v5.0 sharpens this into an operational rule, because "learn from each other" is not actionable on its own:

> **The Reciprocity Rule.** Neither codebase's *code* is canonical. Each codebase's *contracts* are. Before `speech.cpp` absorbs a family from one side, the receiving side must first satisfy the contract the donating side enforced on it. Absorbing code onto a weaker contract is not fusion — it is a silent downgrade wearing a merge commit.

Concretely, in both directions:

| Donor | Contract the receiver must satisfy *first* | Delivered by |
|---|---|---|
| `transcribe.cpp` → engine | 4-state streaming lifecycle; monotonic revision counter; append-only commitment policy; abort polling; size-aware params; pre-clear validation; no-throw teardown; no exception escapes a C entry point | Phase 8 |
| `audio.cpp` → ABI | 14 task kinds; artifact I/O; voice conditioning; named multi-output audio; model-spec catalog + package resolution; progress + cancellation; CLI/server/WebUI reachability | Phases 8 and 12 |

### 1.2 The three laws that follow

1. **Contract before code.** A family migrates only after the destination contract is at least as strong as the source contract. (§4.4)
2. **Net before demolition.** Code is deleted only when a gate exists that would have caught its removal. (§7, Phase 7)
3. **Additive before destructive.** Prefer adding a file over moving one; prefer replacing a function body over deleting a file. `speech.cpp` tracks a live upstream. (§9)

---

## 2. Ground Truth — Verified State of the Tree

All figures below were produced against `c776b81` on 2026-08-23. Commands run from the repo root.

### 2.1 Inventory

| Quantity | Value | Verification command |
|---|---:|---|
| Engine model families (`src/models/`) | 42 | `ls src/models \| wc -l` |
| Community families (`src/community_models/`) | 12 | `ls src/community_models \| wc -l` |
| Transcribe arch families (`src/runtime/arch/`) | 18 | `ls src/runtime/arch \| wc -l` |
| Model-spec catalog entries | 58 | `ls model_specs/*.json \| wc -l` |
| Registered engine model targets | 52 | `grep -oP 'audiocpp_add_model\(\K[a-z0-9_]+' CMakeLists.txt \| sort -u \| wc -l` |
| LOC `src/models/` | 183,729 | `find src/models -name '*.cpp' -o -name '*.h' \| xargs cat \| wc -l` |
| LOC `src/framework/` | 59,045 | same pattern |
| LOC `src/runtime/arch/` | 74,555 | `cat src/runtime/arch/*/*.cpp src/runtime/arch/*/*.h \| wc -l` |
| LOC `src/runtime/` (root) | 15,806 | `cat src/runtime/*.cpp src/runtime/*.h \| wc -l` |
| LOC runtime shared modules | 3,529 | `causal_lm` 1,347 · `conformer` 1,557 · `sanm` 395 · `granite_conformer` 230 |
| `CMakeLists.txt` | 3,305 lines · 85 `add_test` calls | `wc -l CMakeLists.txt; grep -c add_test CMakeLists.txt` |
| Public headers | `include/transcribe/transcribe.h` 2,759 · `capi/include/audiocpp.h` 1,184 | `wc -l` |
| ggml fork patches | 7 (`0001`–`0007`) | `ls patches/ggml/` |
| Upstream tracking | 57 ahead / 0 behind; 4 upstream merges to date | `git rev-list --count upstream/main..main` |

### 2.2 The two dispatch surfaces, as built

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    CURRENT STATE — TWO DISJOINT PRODUCT SURFACES                     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  SURFACE A — the shipped product (default build)                                     │
│    app/cli · app/server · webui · app/workflow · capi → audiocpp.dll                 │
│                              │                                                       │
│                              ▼   engine::runtime::ModelRegistry   (52 loaders)       │
│    IVoiceModelLoader → ILoadedVoiceModel → IVoiceTaskSession   (C++ vtable)          │
│    src/models/* (42) · src/community_models/* (12) · src/framework/* (59 kLOC)       │
│                                                                                      │
│  SURFACE B — built ONLY with -DSPEECHCPP_ENABLE_UNIFIED_ABI=ON   (default OFF)       │
│    transcribe.dll                                                                    │
│                              │                                                       │
│                              ▼   transcribe::find_arch()   (18 Arch structs)         │
│    Arch{ load, init_context, run, run_batch, stream_*, accepts_ext_kind,             │
│          run_validate }                                                              │
│    src/runtime/arch/* (18 families, 74.5 kLOC)                                       │
│                              │                                                       │
│                              └──► transcribe-arch-adapter.cpp ──► Surface A engine   │
│                                   (16 entries; run_batch = nullptr on all of them)   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

The two surfaces **do not meet**. The only bridge is one-directional (B → A) and partial.

### 2.3 Reachability analysis — the finding that reframes the roadmap

This section did not exist in v4. Each row is a capability the project believes it has; the column that matters is whether a user can reach it.

| Capability | Implemented | Reachable from CLI / server / WebUI | Reachable from a shipped C ABI | Evidence |
|---|:--:|:--:|:--:|---|
| 18 ASR arch families (Whisper, Parakeet, Canary, Moonshine, GigaAM, Granite, Cohere, MedASR, …) | ✅ 74.5 kLOC | ❌ **No** | ⚠️ only via `transcribe.dll`, **not built by default** | `grep -rn "find_arch\|transcribe_model_load" app/` → *no hits*; `CMakeLists.txt:181` |
| `whisper`, `moonshine`, `moonshine_streaming` specs (Phase 6 deliverable, 24 packages) | ✅ | ❌ **no engine loader exists** | ❌ | `comm -23 <(ls model_specs \| sed 's/.json//' \| sort) <(grep -oP 'audiocpp_add_model\(\K[a-z0-9_]+' CMakeLists.txt \| sort -u)` |
| `SharedWeightRegistry` / `ScopedWeightShareKey` (Phase 4) | ✅ complete | ❌ **inert** | ❌ **inert** | `grep -rn ScopedWeightShareKey --include=*.cpp .` → **0 call sites** |
| `IOfflineVoiceTaskSession::run_batch` (Phase 4, 5 families) | ✅ overrides written | ❌ no caller | ❌ no caller | `grep -rn "run_batch(" capi/src app/ src/framework` → *no hits* |
| `Arch::run_batch` (16 of 18 arches) | ✅ | ❌ | ⚠️ `transcribe.dll` only | `grep -h "run_batch        = " src/runtime/arch/*/model.cpp` |
| Typed per-family extensions (`whisper.h`, `parakeet.h`, `sortformer.h`, `voxtral_realtime.h`, `moonshine_streaming.h`) | ✅ | ❌ | ⚠️ `transcribe.dll` only | `ls include/transcribe/` |
| Whisper decoding telemetry (temperature tier accepted, compression ratio, avg logprob, no-speech probability, fallback count) | ✅ | ❌ | ⚠️ `transcribe.dll` only | `include/transcribe/whisper.h:175-183` |
| Streaming commitment policy (`AUTO` / `ON_FINALIZE` / `STABLE_PREFIX` + N-hypothesis agreement) | ✅ | ❌ | ⚠️ `transcribe.dll` only | `include/transcribe/transcribe.h:1967-1971` |
| 14-task engine surface (TTS, VC, separation, alignment, MIDI, …) | ✅ | ✅ | ✅ `audiocpp.dll` | `capi/include/audiocpp.h` |

**Consequence — the roadmap's real purpose.** The highest-value user-facing deliverable here is not deduplication. It is **making Surface B reachable**: bringing Whisper, Parakeet, Canary, Moonshine, GigaAM, Granite, Cohere and MedASR into the CLI, server, WebUI and the shipped C ABI. v4 filed that under "Phase 9: Migration of Remaining STT Arches to Engine Core" and framed it as internal tidying. It is the product.

The second-highest-value deliverable is **activating what is already built**: two Phase-4 features are complete, documented as shipped, and switched off.

### 2.4 Audit of Phases 1–6

| Phase | Claimed | Audited | Evidence → action |
|---|---|---|---|
| 1 — Allocator hardening | COMPLETED | ✅ **Confirmed** | `kMetadataPoolBudget = 16 MB` present; WavLM gallocr, Qwen3-TTS runaway guard, DFN2 threshold all present. |
| 2 — Toolchain & build provenance | COMPLETED | ✅ **Confirmed** | ccache probes in both Windows scripts; `transcribe-build-info.h.in`, `transcribe-version.rc.in`, `audiocpp_build_info.h.in`, `audiocpp_version.rc.in` present. |
| 3 — Native VAD chunk planning | COMPLETED | ✅ **Confirmed** | `transcribe-vad{,-integrate}.{h,cpp}`, `transcribe_vad` C ABI, `vad_plan_unit` + `vad_merge_unit` registered and green. |
| 4 — SharedWeightRegistry | COMPLETED, "~3 GB → ~34 MB per session" | ❌ **INERT** | 0 call sites for `ScopedWeightShareKey`; `share_key_` is only populated inside such a scope (`backend_weight_store.h:55`), so `upload_shared()` is dead code. 0 tests measure multi-session VRAM. → **Phase 7 · 7.4** |
| 4 — Batched offline ASR decode | COMPLETED, 5 families | ⚠️ **UNREACHABLE** | Overrides exist in `qwen3_asr`, `voxtral_realtime`, `citrinet_asr`, `vibevoice_asr`, `higgs_audio_stt`; no public entry point calls `run_batch`, and the ArchAdapter nulls the hook. → **Phase 7 · 7.5** |
| 5 — Universal `audiocpp` C ABI | COMPLETED, "strict C++ exception containment" | ⚠️ **PARTIAL** | 17 of ~43 exported definitions are wrapped in `AUDIOCPP_CATCH`; ~26 are not — including `audiocpp_device_count`, `audiocpp_device_info`, `audiocpp_list_devices`, `audiocpp_backend_available`, `audiocpp_model_info`, `audiocpp_model_capabilities`, `audiocpp_write_wav`, `audiocpp_free_model`, `audiocpp_stream_free`. `transcribe.cpp`'s doctrine names this exact class: *"Device and registry queries are not pure reads; guard them."* → **Phase 8 · 8.5** |
| 6 — Whisper GPU cleanup, arch sync, model specs | COMPLETED | ⚠️ **PARTIAL** | `cleanup_gpu` sync and Whisper suppress tables confirmed. The three new specs have **no engine loader behind them** — `--family whisper` resolves a catalog entry and then fails to load. → **Phase 11** |
| 10 — Overlap resolution (v6.0 re-audit, 2026-08-26) | COMPLETED, "loser's features merged into the winner before retirement" | ❌ **VERDICTS ONLY** | `grep -c 'spec_k_drafts\|speculative' src/models/qwen3_asr/*.cpp` → 0; `grep -ci cache.aware src/models/voxtral_realtime/*.cpp` → 0; `grep -ci preset src/models/sortformer_diar/*.cpp` → 0; no `fun_asr_nano` WER gate. The bake-off report exists; none of the five feature merges and none of the deletions do. → **Phase 10.5** |
| 5 — `audiocpp.h` as the product's C ABI (v6.0 re-audit) | "Universal C ABI subsystem" | ❌ **THIRD FAÇADE** | `grep -c struct_size capi/include/audiocpp.h` → 0; no extension kinds, no ABI hash; built by default while `transcribe.h` (98 fns, size-aware) is OFF; consumed by 4 tests and no app. This is exactly the façade Phase 12's entry note warns against. → **Phase 12 (pulled forward)** |

Documentation actions produced by this audit (Phase 7 · 7.0):

- `CHANGELOG.md` — move the SharedWeightRegistry and batched-decoder entries into a new `### Added (implemented, not yet activated)` subsection with a pointer to the activating phase.
- `progress.md` — change the two Phase-4 rows from `100%` to `Implemented / not activated`, and the Phase-6 model-spec row to `Catalog only — no engine loader`.

### 2.5 Corrections Ledger (v4.0 → v5.0)

Every v4 claim the tree contradicts.

| # | v4.0 claim | Verified fact | Command |
|---|---|---|---|
| **C1** | "Fuse `transcribe-mel.cpp` and `kaldi_fbank.cpp` into `src/framework/audio/mel_frontend.cpp`"; the matrix lists `mel_frontend.cpp` as audio.cpp heritage | **`mel_frontend.cpp` / `.h` do not exist.** audio.cpp's mel code lives in `src/framework/audio/dsp.cpp` (`MelFilterbank`, `SparseMelFilterbank`, `WhisperLogMelExtractor`, `MelSpectrogram`, `LogMelSpectrogram`) and `src/framework/audio/kaldi_fbank.cpp` | `find src include -iname '*mel*'` |
| **C2** | Frontend duplication is "2 files vs 2 files" | **~15 independent mel-scale implementations** across the tree (§5.1) | `grep -rln "2595\|1127\.0\|hz_to_mel" --include=*.cpp --include=*.h src/` |
| **C3** | "4-State Stream Machine (`IDLE`, `FEEDING`, `FINALIZING`, `RESETTING`)" | The states are **`IDLE`, `ACTIVE`, `FINISHED`, `FAILED`** | `grep -A5 "enum transcribe_stream_state" include/transcribe/transcribe.h` |
| **C4** | "the transcribe frontend … certified by `asr_e2e_wer_test` (1.45% WER)" | **Both WER gates run Moonshine, which has no mel/STFT at all** — `t_mel_us = 0`; `weights.h`: *"Frontend buffers: NONE — moonshine has no mel/STFT."* The mel path is exercised by **zero** end-to-end accuracy gates | `grep -rn "no mel" src/runtime/arch/moonshine/` |
| **C5** | Acceptance: "Offline ASR WER ≤ 1.50%", "Streaming ≤ 4.50%" | Both gates enforce **10.0%** (`kDefaultMaxCorpusWerPct = 10.0`). Measured 1.45% / 4.35%. Corpus = **69 reference words over 4 fixtures**; one word edit = 1.45 pp. The tests' own comments state the bound is sized for *structural breaks*, not accuracy drift | `grep -n kDefaultMaxCorpusWerPct tests/asr_*_test.cpp` |
| **C6** | Phase 8.4: "Promote Parakeet from `community_models/` to `src/models/parakeet_tdt/` … Delete `src/runtime/arch/parakeet/`" | This deletes the **stronger** implementation. `arch/parakeet` = 8,079 LOC with 13 golden manifests (CTC/RNNT/TDT/TDT-CTC × 110m/0.6b/1.1b, Nemotron streaming, multitalker + RTTM). `community_models/parakeet_tdt` = 2,382 LOC, 2 catalog packages, TDT only. **Also reverses V6 decision R3**, which already selected the transcribe implementation on variant coverage | `wc -l src/runtime/arch/parakeet/*.cpp src/community_models/parakeet_tdt/*.cpp`; `ls ../transcribe.cpp/tests/golden/parakeet/` |
| **C7** | Phase 8.3: canonicalise SenseVoice / FunASR into `src/models/` unconditionally | V6 R3 selects transcribe's SenseVoice *"unless engine streaming/ITN is strictly better"* — an evidence rule, not a default. v4 replaced it with a blanket rule pointing the other way. Restored in §5.4 | `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md` §R3 |
| **C8** | Phase 9: "Delete `src/runtime/arch/` completely. Verification: full test suite compile and execution." | **Zero of `transcribe.cpp`'s 51 tests are present in `speech.cpp`.** The suite that would detect a regression in 16 of 18 arches does not exist in this repository | `for f in $(ls ../transcribe.cpp/tests/*.cpp ../transcribe.cpp/tests/*.c \| xargs -n1 basename); do [ -f tests/$f ] \|\| [ -f tests/unittests/$f ] \|\| echo MISSING; done \| wc -l` → 51 |
| **C9** | Phase 11: bindings framed as greenfield (Rust, Python, TypeScript, Swift) | **All four already exist and are mature in `transcribe.cpp`** — 152 files; `bindings/python/_generate/generate.py` parses the public header with libclang and emits **both** the Python ctypes layer and the TypeScript koffi layer, CI-gated by `--check` + `git diff --exit-code`. Phase 13 is a *retarget*, not a build | `find ../transcribe.cpp/bindings -type f \| wc -l` |
| **C10** | Proposed `speech.h` passes `const char *options_json` on every call and has no `struct_size` on any param struct | Discards `transcribe.h`'s size-aware ABI discipline (`transcribe_abi_struct_size()`, 15 versioned struct ids, `*_init()` stamping) — a regression, not a fusion. It also declares `SPEECH_ERR_OUTPUT_TRUNCATED` while no API takes a buffer + capacity. Redesigned in §6 | `grep -n "TRANSCRIBE_ABI_" include/transcribe/transcribe.h` |
| **C11** | Target-state diagram advertises "SharedWeightRegistry … ~34 MB/session" as achieved | Inert (§2.4) | `grep -rn ScopedWeightShareKey --include=*.cpp .` |
| **C12** | Current-state diagram shows `audiocpp.dll` and `transcribe.dll` as peer shipped artifacts | `transcribe.dll` builds only under `-DSPEECHCPP_ENABLE_UNIFIED_ABI=ON`, **default OFF**; `SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES` likewise. The default build ships one library | `sed -n '181,188p' CMakeLists.txt` |
| **C13** | Matrix lists "Neural Codecs — transcribe heritage: EnCodec / Vocos (Arch-specific)" | `transcribe.cpp` ships **no** neural codec. Mimi, MioCodec, EnCodec, SNAC, DAC and Vocos are all audio.cpp heritage. The codec hub is a pure audio.cpp consolidation with no transcribe counterpart | `ls ../transcribe.cpp/src/` |
| **C14** | "ggml pin `8c63e709` + patches 0001–0006" | There are **7** patches; `0007-cuda-trim-pools-and-clear-graph.patch` landed 2026-08-22 | `ls patches/ggml/` |

### 2.6 Defects found during the audit

Three are latent bugs, not merely gaps. Each is scheduled in Phase 7.

#### D1 — `find_arch()` precedence bug on framework-sniffed paths · severity **high**

`transcribe_model_load_file` handles non-GGUF inputs by asking the framework registry which family claims the path, then dispatching:

```c
// src/runtime/transcribe.cpp:1564-1575
const std::string family = transcribe::adapter_sniff_framework_family(path);
if (!family.empty()) {
    transcribe::Loader fw_loader;
    fw_loader.open_framework(path, family);   // leaves gguf_ == nullptr BY DESIGN
    const transcribe::Arch * fw_arch = transcribe::find_arch(family.c_str());   // ← BUG
    return fw_arch->load(fw_loader, params, out_model);
}
```

`find_arch()` searches the **builtin** table first (`transcribe-arch.cpp:93-107`) and only then the adapter table. Three family names exist in both tables — `qwen3_asr`, `voxtral_realtime`, `moss` — so for those three a `.safetensors` file or model directory is dispatched to the **GGUF-expecting builtin handler** holding a Loader whose `gguf_` is null.

- **Fix**: call `adapter_find_arch(family.c_str())` on this path. The family id came from the framework registry, so by construction only the adapter can service it.
- **Regression test**: `adapter_sniff_dispatch_unit` — for each colliding name, assert the resolved `Arch *` is the adapter entry, not the builtin.

#### D2 — `run_batch`, `stream_validate` and `run_validate` are null on every adapter entry · severity **medium**

All 16 rows of `adapter_archs[]` pass `nullptr` for these three hooks. Consequences:

- `transcribe_run_batch()` silently degrades to serial `run()` per utterance for **every** framework family — including the five with native batched graphs.
- The documented pre-clear guarantee (*"a rejected run ext cannot destroy the prior transcript"*) does not hold for any framework family, because `run_validate` is where that guarantee lives.

#### D3 — Family-id aliasing across the two registries · severity **medium**

The same model ships under different ids on each side, with no mapping table:

| Model | Engine id | Arch id (GGUF `general.architecture`) | Spec file |
|---|---|---|---|
| SenseVoice | `sense_asr` | `sensevoice` | `sense_asr.json` |
| FunASR Nano | `fun_asr_nano` | `funasr_nano` | `fun_asr_nano.json` |
| Parakeet | `parakeet_tdt` | `parakeet` | `parakeet_tdt.json` |
| Sortformer | `sortformer_diar` | `sortformer` | `sortformer_diar.json` |
| Cohere ASR | — | `cohere_asr` | — |
| Granite Speech | — | `granite_speech`, `granite_speech_nar` | — |

Eight of the 18 arches have no model spec at all (`canary`, `canary_qwen`, `cohere`, `gigaam`, `granite`, `granite_nar`, `medasr`, plus the alias-divergent pair) — so they cannot be installed through the package manager even once they are reachable.

---

## 3. The Fusion Doctrine — ten laws

These govern every decision downstream. When a phase task conflicts with a law, the law wins and the task is re-planned.

| # | Law | Rationale | Enforced by |
|---|---|---|---|
| **L1** | **Contract before code.** No family migrates until the destination contract is at least as strong as the source contract, field by field (§4.2). | Migrating 18 WER-validated ASR families onto a session interface with no lifecycle, no commitment policy and no abort polling loses accuracy and safety invisibly. | Phase 8 exit gates |
| **L2** | **Net before demolition.** No file is deleted until a gate exists that would fail if the deletion regressed behaviour. | 74.5 kLOC of arch code currently has **zero** unit coverage in this repo. | Phase 7 exit gates; Appendix B |
| **L3** | **Additive before destructive.** Prefer adding a file to moving one; prefer replacing a function *body* to deleting a file. Never rename an upstream-owned file. | `speech.cpp` tracks a live `upstream/audio.cpp` and has already taken 4 upstream merges. Every rename in `src/models/` is a permanent conflict generator. | §9; CI merge-dry-run job |
| **L4** | **No inert features.** A feature is not complete until a public surface reaches it **and** a test proves the reaching. | Two Phase-4 features shipped as "Done" while switched off (§2.4). | Phase 7 · 7.4/7.5; §7 gate class G4 |
| **L5** | **One family id.** Every model has exactly one canonical id; every other spelling is a registered alias resolving to it. | D3: four alias divergences and three silent shadowings today. | Phase 7 · 7.8 (reporting) → Phase 8 · 8.7 (hard gate); `family_registry_unit` |
| **L6** | **Specification beats inspection.** Shared code is derived from a written contract (the golden manifest / model spec), not from reading two implementations and hoping they agree. | The 18 arches have 18 *different, documented* frontend contracts (§5.1). A "unified mel frontend" written by inspection will break families on contact. | Phase 9 · 9.1 |
| **L7** | **Every deletion has a ledger row.** Appendix B records: what is deleted, what replaced it, which gate proves equivalence, and the revert commit. | Makes rollback mechanical instead of archaeological. | Appendix B |
| **L8** | **Gates ratchet, never loosen.** A threshold may be tightened when a measurement justifies it, never relaxed to make a change land. | The current 10% WER bound has 6.9× headroom over the 1.45% baseline; a 5-word accuracy regression is invisible to it. | §7.3 |
| **L9** | **Exception containment is a build-time property.** No C++ exception may escape a public C entry point; no raw ggml teardown call may appear in library code. Both are lint-enforced, not review-enforced. | `transcribe.cpp`'s doctrine, already half-adopted here: `lint_teardown.cmake` exists but is scoped to `src/runtime/` only. 167 raw teardown sites remain in 75 files outside it. | Phase 8 · 8.4/8.5 |
| **L10** | **Parity is measured against a pinned oracle.** Numerical claims cite a reference implementation at a pinned revision and a per-tensor tolerance file — never "looks right". | `transcribe.cpp` ships 35 tolerance files and 66 golden manifests naming HF repo + revision + reference script + revision. `speech.cpp` has 1 tolerance file. | Phase 7 · 7.1; §7.3 gate class G2 |
| **L11** | **Never truncate silently.** Every model reports its usable limit (`max_audio_ms`, per-session limits); over-length input is rejected up front (`INPUT_TOO_LONG`) or the result carries `truncated = true` and the ABI returns `OUTPUT_TRUNCATED`. A partial result is never presented as a complete one. | transcribe's `docs/input-limits.md` contract. The engine has no home for it: `TaskResult` has no truncation flag, `CapabilitySet` no `max_audio_ms`; the W2a Whisper package trims audio > 30 s and drops the flag. | §4.2 row "Input limits"; Phase 11a · ASR layer; A21 |
| **L12** | **Measure the real flow, then read the code.** Benchmarks use ≥ 4 runs, drop the first, interleave the arms within each round, prefer stage counters over wall clock, and **revert any win inside the noise floor**. Every optimisation ships with a kill-switch env var. `cleanup_gpu()` / per-run `safe_sched_free` is off-limits. A change whose correctness depends on a platform or path the test run does not exercise is reviewed by reading, not by more benchmarking. | `transcribe.cpp/PASSOVER.md` §1 (a "55% win" that was 45% worse on the real flow) and §8 (six defects invisible on the machine that wrote them). | CONTRIBUTING; every G5 gate |
| **L13** | **Two parents, one child.** `speech.cpp` is equally a child of `audio.cpp` and `transcribe.cpp`; the audio.cpp fork base is a convenience, not a precedence. A dependency bump in either parent is ours. `scripts/sync-deps.sh` runs at every phase boundary; upstream syncs end in a recorded merge, never a content copy. | The ggml floor moved in transcribe.cpp and was treated as news; the "6 behind" count was a lagging merge-base for two sessions. | §9.0; tracker Rules 6–7 |

---

## 4. Architecture

### 4.1 Where the seam actually is

The two projects do not differ in *models* — they differ in **who owns the streaming lifecycle**.

- On the transcribe side, the **dispatcher** owns it. `transcribe.cpp` validates params, clears the snapshot, moves `stream_state`, bumps `stream_revision`, applies the commitment policy, pads batch results on abort, and calls into the family only for mechanics. A family cannot corrupt the lifecycle because it never touches it.
- On the engine side, the **session** owns it. `IStreamingVoiceTaskSession` exposes `start_stream` / `process_audio_chunk` / `finalize` / `reset` and nothing else. There is no state, no revision, no commitment semantics, no abort polling. Every family reimplements — or omits — the discipline.

That asymmetry, not file duplication, is the real fusion problem. Everything in §5 is downstream of it.

### 4.2 Contract gap analysis — field by field

This table is the specification for Phase 8. Each ❌ row is a task.

| Contract element | `transcribe::Arch` + dispatcher | `engine::runtime::IVoiceTaskSession` | Fusion target |
|---|---|---|---|
| Lifecycle states | `IDLE` / `ACTIVE` / `FINISHED` / `FAILED`, dispatcher-owned | ❌ none | **Base-class-owned** state machine in `StreamingSessionBase`; families cannot write it |
| Monotonic snapshot revision | `transcribe_stream_get_revision()`, reset by begin/reset/run | ❌ none | Base-class counter, bumped on any observable change |
| Commitment policy | `AUTO` / `ON_FINALIZE` / `STABLE_PREFIX` + `stable_prefix_agreement_n` (default 3); `committed_text` append-only, finalize never rewrites committed bytes | ❌ `StreamEvent{partial_text, is_final}` only | Port the policy engine into the base; families supply an optional native commit boundary |
| Pre-clear validation | `stream_validate` / `run_validate`: on non-OK the prior snapshot **and** lifecycle are fully preserved | ❌ none | `virtual Status validate(const Request&) const` — pure, called before any destructive step |
| Abort / cancellation | `ctx->poll_abort()` polled inside feed and between batch utterances | ⚠️ `ProgressCallback` returning `false` → throws `ProgressCanceled` (offline only; not polled in streaming) | Unify into one `RunControl { poll_abort(); emit_progress(); }` available in both modes |
| Batched offline decode | `Arch::run_batch`, dispatcher-validated once, per-utterance failures isolated in each `ResultSet`, abort pads the tail | ⚠️ `run_batch()` exists but **no caller** | Wire to CLI/server/C ABI; adapter forwards instead of nulling |
| Param evolution | `struct_size` on every caller-owned struct + `transcribe_abi_struct_size()` + `*_init()` stamping; 15 registered struct ids | ❌ `unordered_map<string,string>` options | Typed size-aware core structs **plus** a string/JSON long-tail map (§6.2) |
| Per-family extension | Typed `struct transcribe_ext ext` field-0 pattern, `accepts_ext_kind(model, slot, kind)` probe, per-slot acceptance | ❌ string keys only | Keep the typed ext slot; expose engine options through it |
| Result telemetry | Per-segment: accepted temperature tier, compression ratio, avg logprob, no-speech probability, fallback count | ❌ none | Port into `TaskResult` as an optional decode-telemetry artifact |
| Teardown safety | `transcribe::safe_backend_free` / `safe_buffer_free` / `safe_sched_free`, CI-linted | ❌ 167 raw call sites in 75 files | Expand `lint_teardown.cmake` to the whole `src/` tree |
| Exception containment | `api_guard_status` / `api_guard_value` / `api_guard_void` on every entry point | ⚠️ `AUDIOCPP_CATCH` on 17 of ~43 | 100% coverage, lint-enforced |
| Task breadth | ASR + diarization | **14 task kinds**, artifacts, voice conditioning, named multi-output audio | Engine wins outright — keep as-is |
| Catalog / packaging | ❌ none | `model_specs/*.json` schema v1: packages, sources, options, UI, languages, dependencies | Engine wins outright — extend with a `frontend` + `parity` block (§5.1) |
| Product reach | ❌ none (library only) | CLI, HTTP server, WebUI, workflow | Engine wins outright |
| **Input limits** *(v6.0)* | `LimitsBasis` on the model → `caps.max_audio_ms`, `transcribe_session_get_limits()`; `INPUT_TOO_LONG` up front, `OUTPUT_TRUNCATED` + `was_truncated()` after; three documented buckets | ❌ `AudioPreparationContract.max_input_samples` exists but nothing derives it; no `max_audio_ms`, no truncation flag; `CapacityError` for prefill that does not fit | `CapabilitySet.max_audio_ms` + `TaskResult.truncated` + `AsrLimits` in the ASR layer; `CapacityError` maps to `INPUT_TOO_LONG` at the ABI (L11) |
| **Backend execution** *(v6.0)* | `BackendPlan { primary, scheduler_list }` over `ggml_backend_sched`: an op the GPU backend lacks falls back to CPU inside one graph | ❌ `GraphExecutor` = one `ggml_gallocr` on the session's single backend; unsupported op ⇒ failure | `GraphExecutor` gains an optional scheduler path (primary + CPU fallback) selected per family; default stays single-backend where every op is supported |
| **Concurrency** *(v6.0)* | ❌ documented 0.x limitation: one run/stream in flight per *model* (sessions share backends) | ✅ per-session `ExecutionContext`; `SharedWeightRegistry` makes extra sessions cheap | The engine's rule becomes the ABI's threading contract: sessions are independent; a model may serve N concurrent sessions; documented in `speech.h` |
| **Product registration** *(v6.0)* | the `Arch` registry is the product | a package can build, pass its gate, and still be absent from `audiocpp_add_model` (Whisper was, until `5e1a7e5`) | `family_registry_unit` asserts every `src/models/<f>` with a loader is registered and in a composite (A22) |

**Reading the table.** Neither side dominates. transcribe owns *runtime discipline*; the engine owns *breadth, packaging and product surface*. The fused system takes all of column 2 into the base classes of column 3, and none of column 3's breadth is sacrificed. That is the Reciprocity Rule made concrete.

### 4.3 Target state

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                        TARGET STATE — FULLY FUSED speech.cpp                             │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  Single public C ABI:  libspeech.so / speech.dll                                         │
│    • speech_*  — size-aware, versioned, exception-contained, 14 tasks + streaming        │
│    • typed per-family extension slots (speech_ext, kind-probed per slot)                 │
│    • compat shims:  audiocpp.h  and  transcribe/transcribe.h  → inline forwarders        │
│    • implemented THINLY over the engine base classes — no transcribe dispatcher beneath   │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  Generated bindings (one libclang IR → four targets, CI-gated on header ABI hash)        │
│    Python (ctypes)  ·  TypeScript (koffi)  ·  Rust (sys + safe)  ·  Swift (SPM)          │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  Product layer:   CLI  ·  HTTP server  ·  WebUI  ·  workflow  ·  model manager           │
│    every family reachable by canonical id; every family installable from a model spec    │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  Unified Session Layer                                                                   │
│    SessionBase          — RAII teardown, RunControl (progress + abort), safe_* only      │
│    OfflineSessionBase   — run(), run_batch() with per-utterance isolation                │
│    StreamingSessionBase — 4-state lifecycle · revision counter · commitment policy ·      │
│                           pre-clear validate · chunk buffering (StreamChunker)           │
│    PipelineSession      — VAD-plan → ASR → diarize → re-stitch, in-process, no IPC       │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  ASR Runtime Layer  (v6.0 — src/framework/asr/; the layer neither parent had)            │
│    • EncDecKVCache — ONE self+cross cache (batched), replacing 5 arch + 3 engine structs  │
│    • Decode drivers — CTC greedy · RNN-T/TDT greedy with batched joint window ·           │
│      AR greedy with suppress masks + temperature-fallback ladder + DecodeTelemetry ·      │
│      speculative drafts (spec_k_drafts) as one implementation for every AR family        │
│    • AsrResult — tokens / words / segments / speaker turns, cross-indexed, committed      │
│      counts, raw vs post-processed text, per-utterance timings                           │
│    • AsrLimits — LimitsBasis → max_audio_ms · truncated flag · INPUT_TOO_LONG (L11)       │
│    • Long-form driver over audio/chunking (VAD / fixed / seek) with timestamp stitching   │
│    • Feature bits (initial prompt, temperature fallback, long-form, PNC, ITN, diarize)    │
│  A family above this line is graphs + assets + a thin session. Nothing else.             │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  Canonical family set  (one implementation per model, one id, one spec)                  │
│   ┌────────────────────────┬─────────────────────────────┬────────────────────────────┐  │
│   │ TTS & Voice            │ ASR / STT / Streaming       │ Audio Intelligence         │  │
│   │ IndexTTS2 · F5-TTS     │ Whisper (HF seek + fallback)│ HTDemucs · RoFormer        │  │
│   │ Qwen3-TTS · CosyVoice  │ Moonshine (offline+stream)  │ Sortformer v2 (diar)       │  │
│   │ MiniMax-H3 · MOSS      │ Parakeet (CTC/RNNT/TDT/×11) │ Silero + MarbleNet VAD     │  │
│   │ Kokoro · PocketTTS     │ Canary · Canary-Qwen        │ DFN2 · RNNoise · ZipEnh.   │  │
│   │ Seed-VC · VibeVoice    │ Qwen3-ASR · Voxtral(-RT)    │ MMS / Qwen3 alignment      │  │
│   │ NeuTTS · Irodori · …   │ SenseVoice · FunASR-Nano    │ MuScriptor (MIDI)          │  │
│   │                        │ GigaAM · Granite(-NAR)      │                            │  │
│   │                        │ Cohere · MedASR · Citrinet  │                            │  │
│   └────────────────────────┴─────────────────────────────┴────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  Shared Framework Modules  (single implementation each, spec-driven)                     │
│    • Frontend: one FrontendSpec-driven extractor covering all 18 documented contracts     │
│      (raw-PCM · 64/80/128 mel · 320/400/512 FFT · hann-periodic/symmetric/hamming ·       │
│       none/global/per-feature/per-utterance norm · preemphasis · dither) + Kaldi fbank    │
│    • Tokenizer hub: encode + decode + special-token ids over GGUF vocab, SentencePiece,   │
│      HF tokenizer.json, byte-level BPE, unigram, WordPiece                                │
│    • Chunking & VAD: audio/chunking (Fixed · QuietEnergy · VAD · overlap-add · stitching) │
│    • Codec hub: Mimi · MioCodec · EnCodec · SNAC · DAC · Vocos (audio.cpp heritage)        │
│    • Core: Conformer · SAN-M · Causal-LM · RoPE · SDPA · SwiGLU · RMSNorm · LayerNorm      │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  Memory & Compute                                                                        │
│    • SharedWeightRegistry — ACTIVE at every load+session call site, VRAM-gated in CI      │
│    • ggml_gallocr topological arena reuse · BackendWeightStore 16 MB metadata cap         │
│    • Fused SwiGLU · packed QKV/gate-up · direct depthwise conv (no im2col on B==1)        │
│    • One pinned ggml (CPU AVX2/512/ARM · CUDA · HIP · Vulkan · Metal · SYCL) + 7 patches  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 The migration invariant

```
   For every family F migrating from src/runtime/arch/F to the engine:

   ┌── ENTRY (all must hold) ────────────────────────────────────────────────┐
   │ 1. F's golden manifest + tolerance file are in tests/golden/F/          │
   │ 2. F's transcribe unit tests are registered and green in speech.cpp     │
   │ 3. The destination contract covers every Arch hook F implements (§4.2)  │
   │ 4. F has a canonical id, an alias entry, and a model_specs/F.json       │
   └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌── MIGRATE (additive only) ─────────────────────────────────────────────┐
   │ 5. Land src/models/F/ as a NEW directory; arch copy still builds        │
   │ 6. Register the engine loader; CLI/server/WebUI can now reach F         │
   │ 7. Run BOTH implementations against the same golden manifest            │
   └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌── EXIT (all must hold before deletion) ────────────────────────────────┐
   │ 8. Engine F passes every gate the arch F passed, at equal or tighter    │
   │    tolerance                                                            │
   │ 9. Appendix B ledger row filled in (what/replaced-by/gate/revert)       │
   │ 10. Deletion of src/runtime/arch/F is its OWN commit, revertible alone  │
   └────────────────────────────────────────────────────────────────────────┘
```

Steps 5–7 mean both implementations coexist for one release cycle. That is deliberate: it is the only configuration in which the equivalence claim is *measurable* rather than asserted.

---

## 5. The Deduplication Program

### 5.1 Audio frontends — 15 implementations, one specification

**Verified inventory.** Files containing an independent mel-scale conversion (the fingerprint of a private frontend):

| Heritage | File | Note |
|---|---|---|
| framework | `src/framework/audio/dsp.cpp` | `MelFilterbank`, `SparseMelFilterbank`, `WhisperLogMelExtractor`, `MelSpectrogram`, `LogMelSpectrogram` — the intended shared implementation |
| framework | `src/framework/audio/kaldi_fbank.cpp` | Kaldi 80-dim fbank |
| framework | `src/framework/modules/pitch_extractors/rmvpe_pitch_extractor.cpp` | private |
| framework | `src/framework/modules/speaker_encoders/ecapa_tdnn_runtime.cpp` | private |
| models | `src/models/chatterbox/audio_features.cpp` | private, slaney `1127·ln(1+f/700)` |
| models | `src/models/index_tts2/audio_features.cpp` | private, **same formula, same variable names** |
| models | `src/models/seed_vc/audio_features.cpp` | private, same formula — but `num_fft_bins = n_fft/2` where index_tts2 uses `n_fft/2 + 1` |
| models | `src/models/dots_tts/audio_features.cpp` | private, same formula |
| models | `src/models/confucius4_tts/audio_features.cpp` | private, same formula |
| models | `src/models/supertonic/runtime.cpp` | private |
| models | `src/models/rvc/native_pipeline.cpp` | private |
| models | `src/models/seed_vc/length_regulator.cpp` | private |
| community | `src/community_models/f5_tts/synthesize.cpp` | private, **HTK** `2595·log10(1+f/700)` — a different mel scale |
| community | `src/community_models/kroko_asr/frontend.cpp` | private |
| community | `src/community_models/glm_tts/frontend.cpp` | private |
| transcribe | `src/runtime/transcribe-mel.cpp` | SIMD, precomputed triangular filterbank, band-skip (`fb_begin_`/`fb_end_`), threaded scalar fallback |
| transcribe | `src/runtime/transcribe-kaldi-fbank.cpp` | Kaldi fbank |
| transcribe | `src/runtime/arch/gigaam/mel.h` | private, 64-mel / 320-FFT |

> Verify: `grep -rln "2595\|1127\.0\|hz_to_mel\|mel_to_hz" --include=*.cpp --include=*.h src/`

The `n_fft/2` vs `n_fft/2 + 1` divergence between two otherwise byte-identical copies is exactly the class of drift that duplication produces and that nothing in the current test suite can catch.

**The specification.** `transcribe.cpp` already publishes a machine-readable frontend contract per family, inside each golden manifest's `frontend` block. Extracted across all 18:

| Family | n_mels | fft | win | hop | window | normalization | preemphasis | dither |
|---|---:|---:|---:|---:|---|---|---|---|
| whisper | 80 | 400 | 400 | 160 | hann-periodic | global | — | — |
| moonshine | **1** | — | — | — | — | none | — | — |
| moonshine_streaming | **1** | — | — | — | — | per-feature | — | — |
| parakeet | 128 | 512 | 400 | 160 | hann-periodic | none | — | 1e-5 |
| canary | 128 | 512 | 400 | 160 | hann-periodic | per-feature | 0.97 | 0.0 |
| canary_qwen | 128 | 512 | 400 | 160 | hann-periodic | per-feature | 0.97 | 0.0 |
| cohere | 128 | 512 | 400 | 160 | **hann-symmetric** | per-feature | 0.97 | 0.0 |
| medasr | 128 | 512 | 400 | 160 | **hann-symmetric** | none | — | 0.0 |
| sortformer | 128 | 512 | 400 | 160 | hann-periodic | none | 0.97 | 1e-5 |
| granite | 80 | 512 | 400 | 160 | hann-periodic | none | — | 0.0 |
| granite_nar | 80 | 512 | 400 | 160 | hann-periodic | **per-utterance** | — | 0.0 |
| gigaam | **64** | **320** | **320** | 160 | hann-periodic | none | — | — |
| funasr_nano | 80 | 400 | — | 160 | **hamming** | none | — | — |
| sensevoice | 80 | 400 | 400 | 160 | **hamming** | per-feature | — | 0.0 |
| moss | 80 | 400 | 400 | 160 | hann-periodic | per-utterance | — | 0.0 |
| qwen3_asr | 128 | 400 | 400 | 160 | hann-periodic | per-utterance | — | 0.0 |
| voxtral | 128 | 400 | 400 | 160 | hann-periodic | per-utterance | — | 0.0 |
| voxtral_realtime | 128 | 400 | 400 | 160 | hann-periodic | **global** | — | 0.0 |

> Regenerate: `for d in ../transcribe.cpp/tests/golden/*/; do f=$(ls $d*.manifest.json | head -1); echo "$(basename $d): $(grep -A10 '"frontend"' $f | tr -d ' \n')"; done`

**What this table proves.** The v4 Phase-7 spec — *"standard 80-band, 128-band Mel, and Kaldi 80-dimensional filterbanks"* — is missing: 64-mel, 320-FFT, raw-PCM (n_mels = 1), hamming and hann-symmetric windows, three of the four normalization modes, preemphasis, and dither. A unified frontend written to the v4 spec would silently break at least 11 of the 18 families.

**Method (Phase 9 · 9.1).**

1. Define `engine::audio::FrontendSpec` with **exactly** the manifest's fields, plus `kind ∈ {RawPcm, MelSpectrogram, KaldiFbank}`.
2. Add a `"frontend"` block to model-spec **schema v2** carrying the same fields. The catalog becomes the single source of truth for both the runtime and the tests.
3. Implement one extractor as a pure function of `FrontendSpec`, adopting transcribe's SIMD path, precomputed triangular weights, nonzero-band skip and threaded scalar fallback.
4. Add `frontend_contract_test`: for every family, assert `FrontendSpec(model_spec) == manifest.frontend` field-for-field. Fails on drift in either direction.
5. Add `frontend_parity_test`: for every family, run the unified extractor and the legacy extractor on the same fixture and assert agreement within the family's tolerance file (`enc.mel.in` entry — e.g. Whisper's `{max_abs: 4e-05, mean_abs: 6e-08}`).
6. Only then retire call sites — **by replacing function bodies in place** (L3), not by deleting files. `src/models/index_tts2/audio_features.cpp` keeps its file, its symbol and its signature; its body becomes a call into the shared extractor. This is a small, mergeable diff hunk instead of a permanent upstream conflict.
7. Delete `src/runtime/transcribe-mel.*`, `src/runtime/transcribe-kaldi-fbank.*` and `src/runtime/arch/gigaam/mel.*` **last**, each with an Appendix B ledger row.

### 5.2 Tokenizers — the missing half of the interface

| | audio.cpp | transcribe.cpp |
|---|---|---|
| Interface | `engine::tokenizers::ITokenizer` — `family()`, `tokenize()` | `transcribe::Tokenizer` — `load(gguf)`, `encode()`, `decode()`, `find()`, special-token ids, pretokenizer flavour |
| Backends | `llama_bpe.cpp` (289) · `sentencepiece.cpp` (160) · `hf_tokenizer_json.cpp` (123) + `external/sentencepiece`, `external/llama_tokenizer` | `transcribe-tokenizer.cpp` (903), zero external deps |
| Vocab sources | GGUF metadata, SentencePiece `.model` binaries, HF `tokenizer.json` (40+ TTS families) | GGUF `tokenizer.ggml.*` (unigram / bpe / gpt2), merges, byte-level inversion |
| **Decode** | ❌ **absent from the interface** | ✅ with `▁`→space for SentencePiece flavours and byte-level inversion for gpt2 |

`ITokenizer` has no `decode()` at all. ASR is decode-dominant, so the unified interface must be transcribe's superset:

```cpp
namespace engine::text {

enum class TokenizerModel { Unigram, Bpe, ByteLevelBpe, WordPiece, SentencePieceBinary, HuggingFaceJson };
enum class Pretokenizer  { None, Qwen2, Gpt2 };

struct SpecialTokens { int unk = -1, bos = -1, eos = -1, pad = -1, blank = -1, decoder_start = -1; };

class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual TokenizerModel model()   const = 0;
    virtual size_t         vocab_size() const = 0;
    virtual const SpecialTokens & specials() const = 0;

    virtual std::vector<int32_t> encode(std::string_view text) const = 0;   // may throw NotImplemented
    virtual std::string          decode(std::span<const int32_t> ids) const = 0;
    virtual std::optional<int32_t> find(std::string_view piece) const = 0;  // O(1)
    virtual std::string_view       piece(int32_t id) const = 0;
};

// One dispatcher, four sources.
TokenizerPtr load_tokenizer_from_gguf(const gguf_context *);
TokenizerPtr load_tokenizer_from_sentencepiece(const std::filesystem::path &);
TokenizerPtr load_tokenizer_from_hf_json(const std::filesystem::path &);
TokenizerPtr load_tokenizer_for_family(std::string_view family, const ResourceBundle &);
}
```

**Gate (Phase 9 · 9.2).** `tokenizer_parity_test` — for every family with a golden manifest, assert the unified tokenizer reproduces the manifest's `tokenizer_summary` (type, `vocab_size`, every special-token id) and round-trips `decode(encode(x)) == x` over the family's fixture corpus. `transcribe.cpp`'s `qwen3_asr_bpe_parity`, `whisper_tokenize_parity`, `whisper_bin_tokenize_parity` and `tokenizer_decode_only_unit` port directly and become the seed cases.

### 5.3 Core neural modules

| Module | audio.cpp | transcribe.cpp | Action |
|---|---|---|---|
| SAN-M | `src/framework/modules/speech_encoders/sanm.cpp` (177) — module-graph style, `DepthwiseConv1dModule` (already direct dw conv) | `src/runtime/sanm/sanm.cpp` (263) — raw ggml, direct `ggml_conv_2d_dw_direct` with an `TRANSCRIBE_CONV_NO_DIRECT_DW` kill switch | Keep the framework module; port transcribe's kill-switch env override and its B>1 batched path. Consumers: `community_models/sense_asr`, `models/fun_asr_nano` (framework) vs `arch/{sensevoice,funasr_nano}` (runtime) |
| Conformer | `src/framework/modules/conformer_modules.cpp` (446) | `src/runtime/conformer/conformer.cpp` (1,122) | Diff-and-merge into the framework module; transcribe's is larger because it carries relative-attention variants and chunked-streaming masks the framework lacks |
| Causal LM | `src/framework/modules/transformers/*` (4,393) | `src/runtime/causal_lm/causal_lm.cpp` (993) | Framework wins on breadth; port transcribe's KV-cache windowing and bounded static decode |
| Shaw relative attention | `src/framework/modules/attention/relative_attention.cpp`, `common_relative_attention.cpp` | `src/runtime/granite_conformer/shaw_attn.cpp` (155) | Fold into `common_relative_attention.cpp` |
| SDPA / RoPE / SwiGLU / RMSNorm | `src/framework/modules/attention/`, `norm_modules.cpp`, `activation_modules.cpp` | inline per arch | Framework is canonical; arches adopt on migration |

**Method.** Every merge here is a *behaviour-preserving* refactor guarded by `L10`: land the merged module behind a compile-time switch, run both against the affected families' golden manifests, then remove the switch. No module merge lands in the same commit as a family migration.

### 5.4 Family overlap — canonical selection by evidence

V6 decision **R3** is restored as the governing rule (v4 replaced it with a blanket "engine wins", which §2.5 C6/C7 show to be wrong):

> **R3.** One implementation per family, chosen by (a) variant coverage, (b) parity/WER validation status, (c) integration depth. The loser's distinguishing features are merged into the winner *before* the loser is deleted. Either winner must satisfy **both** surfaces. Fallback: if the chosen winner fails the golden suite, the other wins.

**Evidence table** (all figures verified; "goldens" = golden manifests in `transcribe.cpp/tests/golden/<arch>/`):

| Family | Engine LOC | Arch LOC | Spec packages | Goldens | Distinguishing features | **Canonical pick** |
|---|---:|---:|---:|---:|---|---|
| `parakeet_tdt` | 2,382 | **8,079** | 2 | **13** | Arch: CTC/RNNT/TDT/TDT-CTC at 110m/0.6b/1.1b, Nemotron streaming variants, multitalker + RTTM output, frame-batched joint window. Engine: TDT only | **Arch** (confirms V6 R3; reverses v4) |
| `voxtral_realtime` | 3,678 | 4,327 | 3 | 1 | Arch: 4-state streaming + cache-aware windows. Engine: parallel frontends, batched decode, model spec | **Engine**, after absorbing the arch streaming machine (which Phase 8 lifts into the base anyway) |
| `qwen3_asr` | 3,175 | 3,644 | **7** | 2 | Arch: speculative decoding, BPE parity test. Engine: `DecodeGraphBatched`, `generate_batch`, packed projections, 7 packages | **Engine** + port speculative decoding and `qwen3_asr_bpe_parity` |
| `fun_asr_nano` | 2,741 | 2,797 | 3 | 2 | Arch: WER-validated, static Arch trait. Engine: packed QKV + fused SwiGLU (7→5 matmuls/layer, 197→113 graph nodes) | **Engine** + adopt the arch WER corpus as its gate |
| `sense_asr` / `sensevoice` | 1,604 | 1,593 | 2 | 1 | Arch: SAN-M optimisation. Engine: rich event tags, ITN | **Decide in Phase 10 by running both** against `tests/golden/sensevoice/` — genuinely close; R3's fallback clause applies |
| `sortformer_diar` / `sortformer` | 1,768 | 1,875 | **4** | 1 | Arch: streaming presets + typed `sortformer.h` ext. Engine: v2 package, 4 packages | **Engine** + port the streaming presets and the typed ext |
| `moss` | TTS only | ASR + diarize | 3 | 1 | Different tasks entirely | **Both survive** — resolve the id collision instead (§5.5) |

**Not overlapping — arch is the only implementation (12 families).** `whisper`, `moonshine`, `moonshine_streaming`, `voxtral` (offline — distinct from `voxtral_realtime`; 3,611 LOC, 3 golden variants: mini-3b-2507, mini-4b-realtime-2602, small-24b-2507; **no engine counterpart exists**), `canary`, `canary_qwen`, `cohere`, `gigaam`, `granite`, `granite_nar`, `medasr`, and `moss` in its ASR + diarization role (§5.4 row 7). These are pure *additions* to the product once migrated, and are the reason Phase 11 is the highest-value phase in the roadmap.

> The `voxtral` / `voxtral_realtime` split is easy to miss: the engine has only the realtime session, so the offline Voxtral models — including `voxtral-small-24b` — are currently unreachable through any product surface and are not covered by the §5.4 overlap analysis.

### 5.5 Family-id unification (fixes D3 and the shadowing hazard)

Introduce a single registry, generated at build time and asserted by a test:

```cpp
// include/engine/framework/runtime/family_registry.h
struct FamilyEntry {
    std::string_view canonical_id;                // "parakeet_tdt"
    std::span<const std::string_view> aliases;    // {"parakeet", "parakeet-tdt"}
    std::span<const std::string_view> gguf_archs; // {"parakeet"}
    std::string_view spec_file;                   // "model_specs/parakeet_tdt.json"
    VoiceTaskKind    primary_task;
};
const FamilyEntry * resolve_family(std::string_view any_spelling);
```

Gate — `family_registry_unit` asserts all of:

1. every directory in `src/models/`, `src/community_models/` and `src/runtime/arch/` maps to exactly one canonical id;
2. every `model_specs/*.json` maps to exactly one canonical id, and vice versa (no orphan specs, no unspec'd families);
3. no alias resolves to two canonical ids;
4. every registered GGUF arch name resolves to exactly one dispatch target — which is what makes the `qwen3_asr` / `voxtral_realtime` / `moss` shadowing (D1) a **compile-visible** conflict rather than a silent precedence rule.

The current orphan list this gate would flag on day one: specs without a loader — `whisper`, `moonshine`, `moonshine_streaming`; arches without a spec — `canary`, `canary_qwen`, `cohere`, `gigaam`, `granite`, `granite_nar`, `medasr`.

### 5.6 The ASR runtime layer — KV caches and decode drivers *(v6.0)*

**Measured duplication** (`docs/reports/fusion_review_2026-08-26.md` §3.1): five arch KV caches — `WhisperKvCache`, `MoonshineKvCache`, `MoonshineStreamingKvCache`, `CanaryKvCache`, `CohereKvCache` — are **field-identical** (`self_k/self_v/cross_k/cross_v/n_ctx/n/head/T_enc/n_batch/cross_populated`; Whisper adds `T_enc_pad`), and the three migrated engine packages each carry a copy. **15 of 18** arch `model.cpp` files hand-roll an argmax decode loop. RNN-T/TDT greedy exists in `gigaam`, `parakeet`, `sortformer` and three engine `tdt_decoder_runner` variants.

| Component | Sources folded in | Home |
|---|---|---|
| `EncDecKVCache` | the 5 arch structs, the 3 engine copies; batched (`n_batch`) from day one; optional `T_enc_pad` for Metal FA | `framework/asr/enc_dec_kv_cache.{h,cpp}` — built on `TransformerKVCache`'s buffer/import/export discipline |
| `decode/ctc_greedy` | `citrinet_asr`, `gigaam` CTC, `medasr`, `parakeet` CTC | `framework/asr/decode/` |
| `decode/transducer_greedy` | `parakeet` (with the frame-batched joint window, `TRANSCRIBE_RNNT_BATCH_W` kill switch, `_BATCH_CHECK` argmax-flip verifier), `gigaam` RNN-T, `sortformer`; unify with engine `tdt_decoder_*` | `framework/asr/decode/` |
| `decode/ar_greedy` | the 15 hand-rolled loops: prompt pass + step loop, suppress / begin-suppress masks, timestamp rules, temperature-fallback ladder + `DecodeTelemetry` (whisper), speculative drafts (`spec_k_drafts` — one implementation replacing `qwen3_asr`'s and `voxtral_realtime`'s) | `framework/asr/decode/` |
| `AsrResult` | `transcribe_session`'s `TokenEntry/WordEntry/SegmentEntry/SpeakerSegmentEntry` + committed counts + `raw_text` + timings; projected onto `TaskResult` / `StreamEvent` | `framework/asr/result.h` |
| `AsrLimits` | `transcribe_model::LimitsBasis` → `CapabilitySet.max_audio_ms`, `TaskResult.truncated`, session limits query | `framework/asr/limits.h` |
| long-form driver | whisper's seek-continuation (HF 5.x), `transcribe-vad-integrate`'s offset stitching, over `audio/chunking` | `framework/asr/long_form.{h,cpp}` |

**Method.** Same as §5.3: behaviour-preserving, behind a switch, both against the golden manifests, then the switch goes. The three migrated families are re-based first (Phase 11a) — they are the proof the layer is sufficient before any further port.

### 5.7 VAD chunk planning — one implementation *(v6.0)*

`src/runtime/transcribe-vad.cpp` says it in its own comment: *"Direct port of audio.cpp's plan_vad_audio_chunks (chunking.cpp)."* `audio/chunking.h` is the richer of the two (Fixed / QuietEnergy / VAD planners, overlap-add, word-timestamp and speech-metadata stitching) and is upstream-owned. `transcribe-vad{,-integrate}.{h,cpp}` are deleted; `vad_plan_unit` / `vad_merge_unit` are re-pointed at `plan_vad_audio_chunks` / `append_chunk_speech_metadata`; the `speech_vad()` ABI entry calls the engine. v5's §4.3 line naming `vad::plan` is corrected above.

### 5.8 Loaders — the legacy whisper.cpp `.bin` becomes a `TensorSource` *(v6.0)*

Today: transcribe `Loader` + `transcribe-bin-loader` on one side, engine `TensorSource` (GGUF / safetensors / torch-bin) on the other, and W2a added a **third** private `.bin` parser in `models/whisper/assets.cpp`. Correct home: `assets/whisper_bin_tensor_source.cpp` implementing `TensorSource`, so `BackendWeightStore::load_tensor` works unchanged and the family carries no loader code. The three `.bin` unit tests (`whisper_bin_parser_unit`, `whisper_bin_suppress_unit`, `whisper_bin_tokenize_parity`) migrate with it (F6).

### 5.9 The Whisper encoder already exists upstream *(v6.0)*

`modules/speech_encoders/whisper_embedding.h` ships `WhisperEmbeddingModule` — conv1, conv2, positional embedding, N layers, final norm — and `qwen3_asr` already consumes it. W2a's `models/whisper/graphs.cpp` re-implemented it. Phase 11a folds the Whisper encoder onto the module (both against `tests/golden/whisper/`), leaving the family with the decoder graph only.

---

## 6. The Unified Public C ABI (`speech.h` / `libspeech`)

### 6.1 What the v4 draft got wrong

The v4 `speech.h` sketch took audio.cpp's `options_json` convention and dropped transcribe's ABI discipline entirely. That is not a fusion of the two ABIs; it is one of them, minus its safety properties. Specifically:

| v4 property | Problem |
|---|---|
| Every call takes `const char *options_json` | Puts a JSON parse in the streaming feed path; loses compile-time typing; makes every option a runtime string lookup; makes binding generation lossy (a generator cannot type what is a string blob) |
| No `struct_size` on any struct | Removes the mechanism that lets a 0.3 library run a 0.2 caller's binary unchanged. `transcribe.h` has 15 registered struct ids and a `transcribe_abi_struct_size()` introspection call; discarding this breaks the four shipped bindings' load-time ABI check |
| `SPEECH_ERR_OUTPUT_TRUNCATED` declared | No API in the draft takes a buffer + capacity, so the code is unreachable — a dangling contract |
| No per-family extension slot | Deletes `whisper.h` (temperature-fallback ladder, thresholds, seed, prompt tokens), `parakeet.h`, `sortformer.h`, `voxtral_realtime.h`, `moonshine_streaming.h` — i.e. every typed knob and all decode telemetry |
| `speech_run_asr(..., char **out_text)` only | No word/segment/token timestamps, no per-segment telemetry, no language detection result, no committed/tentative split for streaming |

### 6.2 Design principles for v5

1. **Typed core, JSON tail.** Parameters that are hot, cross-family and stable get typed fields in a size-aware struct. Family-specific long-tail knobs keep the `options_json` / key-value channel. Neither replaces the other.
2. **Size-aware everything.** Every caller-owned struct carries `struct_size` as field 0, is initialised by a `speech_*_init()` that stamps it, and is registered in `speech_abi_struct_size()`. Inherited verbatim from `transcribe.h`.
3. **Two result-ownership models, chosen by access pattern.**
   - *One-shot results* (transcript, synthesized audio, stems, alignment) → heap out-params + explicit `speech_free_*`. Simple for FFI; audio.cpp convention.
   - *Streaming accessors* → pointers into session-owned storage with documented stability, plus a monotonic `revision`. **Zero allocation in the feed loop**; transcribe convention. `stream_committed_pointer_stability` is the test that pins it.
4. **Typed extension slot preserved.** `struct speech_ext { uint64_t struct_size; uint32_t kind; }` as field 0 of every family extension; `speech_model_accepts_ext_kind(model, slot, kind)` probes per loaded variant *and* per slot.
5. **Exception containment is total and lint-enforced** (L9).
6. **One enum universe.** `speech_task` mirrors `engine::runtime::VoiceTaskKind` exactly; `capi_enum_sync_test` (already present) is extended to gate it.

### 6.3 Header sketch

```c
#ifndef SPEECH_H
#define SPEECH_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

/* ---- Export macros ---------------------------------------------------- */
#if defined(_WIN32)
#  if defined(SPEECH_BUILD_DLL)
#    define SPEECH_API __declspec(dllexport)
#  else
#    define SPEECH_API __declspec(dllimport)
#  endif
#else
#  define SPEECH_API __attribute__((visibility("default")))
#endif

#define SPEECH_VERSION_MAJOR 1
#define SPEECH_VERSION_MINOR 0
#define SPEECH_VERSION_PATCH 0
#define SPEECH_VERSION_NUMBER (SPEECH_VERSION_MAJOR*10000 + SPEECH_VERSION_MINOR*100 + SPEECH_VERSION_PATCH)

/* ---- Status ----------------------------------------------------------- */
typedef enum {
    SPEECH_OK                      =  0,
    SPEECH_ERR_INVALID_ARG         = -1,
    SPEECH_ERR_OOM                 = -2,
    SPEECH_ERR_BACKEND             = -3,
    SPEECH_ERR_GGUF                = -4,
    SPEECH_ERR_MODEL_NOT_FOUND     = -5,
    SPEECH_ERR_ABORTED             = -6,
    SPEECH_ERR_UNSUPPORTED_TASK    = -7,
    SPEECH_ERR_UNSUPPORTED_ARCH    = -8,
    SPEECH_ERR_NOT_IMPLEMENTED     = -9,
    SPEECH_ERR_BAD_STRUCT_SIZE     = -10,  /* struct_size == 0 or < required prefix */
    SPEECH_ERR_BAD_STATE           = -11,  /* e.g. feed() while stream is FINISHED  */
    SPEECH_ERR_EXT_REJECTED        = -12,  /* pre-clear validate rejected the ext    */
    SPEECH_ERR_INTERNAL            = -13,  /* an exception was contained here        */
} speech_status;

/* Registered size-aware struct ids — mirrors transcribe_abi_struct.        */
typedef enum {
    SPEECH_ABI_MODEL_LOAD_PARAMS = 0, SPEECH_ABI_SESSION_PARAMS = 1,
    SPEECH_ABI_RUN_PARAMS        = 2, SPEECH_ABI_STREAM_PARAMS  = 3,
    SPEECH_ABI_CAPABILITIES      = 4, SPEECH_ABI_TIMINGS        = 5,
    SPEECH_ABI_SEGMENT           = 6, SPEECH_ABI_WORD           = 7,
    SPEECH_ABI_TOKEN             = 8, SPEECH_ABI_STREAM_UPDATE  = 9,
    SPEECH_ABI_STREAM_TEXT       = 10, SPEECH_ABI_SESSION_LIMITS = 11,
    SPEECH_ABI_EXT               = 12, SPEECH_ABI_DEVICE_INFO    = 13,
    SPEECH_ABI_SPEAKER_SEGMENT   = 14, SPEECH_ABI_PROGRESS       = 15,
    SPEECH_ABI_AUDIO             = 16, SPEECH_ABI_DECODE_TELEMETRY = 17,
} speech_abi_struct;
SPEECH_API size_t speech_abi_struct_size(speech_abi_struct which);

/* ---- Tasks (1:1 with engine::runtime::VoiceTaskKind) ------------------- */
typedef enum {
    SPEECH_TASK_VAD = 0, SPEECH_TASK_ASR = 1, SPEECH_TASK_DIARIZATION = 2,
    SPEECH_TASK_SEPARATION = 3, SPEECH_TASK_GENERATION = 4, SPEECH_TASK_TTS = 5,
    SPEECH_TASK_ALIGNMENT = 6, SPEECH_TASK_VOICE_CONVERSION = 7,
    SPEECH_TASK_VOICE_CLONING = 8, SPEECH_TASK_SPEECH_TO_SPEECH = 9,
    SPEECH_TASK_VOICE_DESIGN = 10, SPEECH_TASK_SPEAKER_RECOGNITION = 11,
    SPEECH_TASK_SVC = 12, SPEECH_TASK_MIDI = 13,
} speech_task;

typedef enum { SPEECH_BACKEND_AUTO=0, SPEECH_BACKEND_CPU=1, SPEECH_BACKEND_CUDA=2,
               SPEECH_BACKEND_HIP=3, SPEECH_BACKEND_VULKAN=4, SPEECH_BACKEND_METAL=5,
               SPEECH_BACKEND_SYCL=6 } speech_backend;

/* ---- Opaque handles ---------------------------------------------------- */
typedef struct speech_model   speech_model;
typedef struct speech_session speech_session;
typedef struct speech_stream  speech_stream;

/* ---- Family extension base (field 0 of every family ext struct) --------- */
typedef struct speech_ext { uint64_t struct_size; uint32_t kind; } speech_ext;
typedef enum { SPEECH_EXT_SLOT_RUN = 0, SPEECH_EXT_SLOT_STREAM = 1 } speech_ext_slot;
SPEECH_API bool speech_model_accepts_ext_kind(const speech_model *, speech_ext_slot, uint32_t kind);

/* ---- Size-aware parameter structs -------------------------------------- */
typedef struct {
    uint64_t        struct_size;
    speech_backend  backend;
    int32_t         device_id;
    int32_t         n_threads;        /* 0 = auto */
    bool            use_mmap;
    const char *    family_hint;      /* NULL = auto-detect via family registry */
    const char *    options_json;     /* long-tail, family-specific; NULL = defaults */
} speech_model_load_params;
SPEECH_API void speech_model_load_params_init(speech_model_load_params *);

typedef struct {
    uint64_t     struct_size;
    speech_task  task;
    const char * options_json;        /* session-scoped long tail */
} speech_session_params;
SPEECH_API void speech_session_params_init(speech_session_params *);

typedef struct {
    uint64_t          struct_size;
    const char *      language;       /* NULL = auto-detect */
    bool              translate;
    bool              word_timestamps;
    int32_t           max_tokens;     /* 0 = model limit */
    const speech_ext *family;         /* typed per-family ext; NULL = defaults   */
    const char *      options_json;   /* long tail; merged under the typed ext   */
} speech_run_params;
SPEECH_API void speech_run_params_init(speech_run_params *);

typedef enum { SPEECH_COMMIT_AUTO = 0, SPEECH_COMMIT_ON_FINALIZE = 1,
               SPEECH_COMMIT_STABLE_PREFIX = 2 } speech_commit_policy;

typedef struct {
    uint64_t             struct_size;
    const speech_ext *   family;
    speech_commit_policy commit_policy;          /* zero-init = AUTO */
    int32_t              stable_prefix_agreement_n;  /* 0 = library default (3) */
    const char *         options_json;
} speech_stream_params;
SPEECH_API void speech_stream_params_init(speech_stream_params *);

/* ---- Decode telemetry (ported from transcribe/whisper.h) --------------- */
typedef struct {
    uint64_t struct_size;
    int64_t  t0_ms, t1_ms;
    float    temperature_used;      /* accepted fallback tier                 */
    float    compression_ratio;
    float    avg_logprob;
    float    no_speech_prob;
    bool     no_speech_triggered;   /* chunk output was discarded             */
    int32_t  n_fallbacks;
} speech_decode_telemetry;

/* ---- Progress + cancellation (one control object for both modes) ------- */
typedef struct {
    uint64_t     struct_size;
    float        progress;          /* 0.0 .. 1.0 */
    const char * stage;             /* "mel", "encode", "decode", "vocode"    */
    int64_t      units_completed, units_total;
} speech_progress;
typedef bool (*speech_progress_cb)(const speech_progress *, void *user);   /* false = cancel */

/* ---- Lifecycle --------------------------------------------------------- */
SPEECH_API speech_status speech_model_load(const char *path,
                                           const speech_model_load_params *,
                                           speech_model **out_model);
SPEECH_API void          speech_model_free(speech_model *);
SPEECH_API speech_status speech_session_create(speech_model *, const speech_session_params *,
                                               speech_session **out);
SPEECH_API void          speech_session_free(speech_session *);
SPEECH_API void          speech_session_set_progress_callback(speech_session *,
                                                              speech_progress_cb, void *user);
SPEECH_API void          speech_session_request_abort(speech_session *);  /* thread-safe */

/* ---- Offline inference — heap out-params + explicit free ---------------- */
SPEECH_API speech_status speech_run_asr(speech_session *, const float *pcm, size_t n,
                                        const speech_run_params *, char **out_text);
SPEECH_API speech_status speech_run_asr_batch(speech_session *, const float *const *pcm,
                                              const size_t *n, size_t n_utt,
                                              const speech_run_params *, char ***out_text);
SPEECH_API speech_status speech_run_tts(speech_session *, const char *text,
                                        const speech_run_params *,
                                        float **out_pcm, size_t *out_n, int *out_rate);
/* … diarize / separate / align / denoise / super_resolve / midi / generate … */

/* ---- Result accessors (session-owned; valid until the next run/free) ---- */
SPEECH_API size_t                          speech_result_n_segments(const speech_session *);
SPEECH_API const struct speech_segment *    speech_result_segment(const speech_session *, size_t i);
SPEECH_API const speech_decode_telemetry *  speech_result_telemetry(const speech_session *, size_t i);

/* ---- Streaming — 4-state, zero-alloc accessors ------------------------- */
typedef enum { SPEECH_STREAM_IDLE=0, SPEECH_STREAM_ACTIVE=1,
               SPEECH_STREAM_FINISHED=2, SPEECH_STREAM_FAILED=3 } speech_stream_state;

SPEECH_API speech_status        speech_stream_begin(speech_session *, const speech_run_params *,
                                                    const speech_stream_params *);
SPEECH_API speech_status        speech_stream_feed(speech_session *, const float *pcm, size_t n);
SPEECH_API speech_status        speech_stream_finalize(speech_session *);
SPEECH_API void                 speech_stream_reset(speech_session *);
SPEECH_API speech_stream_state  speech_stream_get_state(const speech_session *);
SPEECH_API uint64_t             speech_stream_get_revision(const speech_session *);
/* Pointers below are stable while the revision is unchanged; committed_text is
 * append-only and finalize never rewrites already-committed bytes.           */
SPEECH_API const char *         speech_stream_committed_text(const speech_session *);
SPEECH_API const char *         speech_stream_tentative_text(const speech_session *);

/* ---- Hardware, catalog, utility ---------------------------------------- */
SPEECH_API int          speech_device_count(speech_backend);
SPEECH_API speech_status speech_device_info(speech_backend, int id, char **out_json);
SPEECH_API bool         speech_backend_available(speech_backend);
SPEECH_API speech_status speech_family_list(char **out_json);     /* canonical ids + aliases */
SPEECH_API speech_status speech_family_spec(const char *id, char **out_json);
SPEECH_API int          speech_read_wav(const char *, float **, size_t *, int *);
SPEECH_API int          speech_write_wav(const char *, const float *, size_t, int rate, int ch);

/* ---- Memory ------------------------------------------------------------ */
SPEECH_API void         speech_free_string(char *);
SPEECH_API void         speech_free_strings(char **, size_t n);
SPEECH_API void         speech_free_audio(float *);
SPEECH_API void         speech_free_stems(float **, size_t n);
SPEECH_API const char * speech_version(void);
SPEECH_API const char * speech_build_id(void);

#ifdef __cplusplus
}
#endif
#endif /* SPEECH_H */
```

### 6.4 Backward compatibility

Both existing headers become inline forwarders — no source change for any current consumer:

- `capi/include/audiocpp.h` → thin inline wrappers over `speech_*`, keeping the `audiocpp_error_t` out-param convention. `capi/test/*.c` and `capi_tts_test.c` must compile and pass unchanged.
- `include/transcribe/transcribe.h` → inline forwarders. `transcribe_abi_struct_size()` keeps returning the *transcribe* sizes (the shim owns its own layouts), so existing bindings' load-time ABI check still succeeds.
- Windows: emit `audiocpp.lib` and `transcribe.lib` import libraries alongside `speech.lib`, all resolving into `speech.dll`.

**Gate.** `abi_compat_test` links `capi_test.exe`, `test_capi.c`, `abi_bridge_hello.exe`, `abi_stream_hello.exe` and the ported `api_smoke.c` against `speech.dll` **through the shims** and asserts byte-identical results versus the pre-fusion binaries on the same fixtures.

---

## 7. Test & Gate Architecture

### 7.1 The coverage problem, quantified

| Asset | `transcribe.cpp` | `speech.cpp` | Gap |
|---|---:|---:|---|
| Test translation units | 51 | **0 of those 51** | 51 missing |
| Golden manifests | 66 files / 19 dirs / **301 KB total** | 1 dir (`silero_vad`, audio.cpp's own) | 18 arch dirs missing |
| Per-tensor tolerance files | 35 | 1 | 34 missing |
| Per-family porting docs | 18 (+3 templates) | 3 templates only | 18 missing |

> Verify: `for f in $(ls ../transcribe.cpp/tests/*.cpp ../transcribe.cpp/tests/*.c | xargs -n1 basename); do [ -f tests/$f ] || [ -f tests/unittests/$f ] || echo MISSING $f; done | wc -l` → 51
> `du -sh ../transcribe.cpp/tests/golden` → 301K

Notable individual gaps, each pinning an invariant this fusion depends on:

| Missing test | Invariant it pins |
|---|---|
| `stream_committed_pointer_stability` | Committed-text pointers stay valid across feeds — the contract §6.2(3) is built on |
| `stream_dispatch_unit`, `run_dispatch_unit` | Lifecycle transitions and pre-clear preservation |
| `stream_capability_unit` | A family that declares streaming really implements the required triple |
| `teardown_safety_unit`, `backend_init_throw_unit` | No throw escapes init/teardown |
| `batch_mask_unit`, `prefill_chunk_mask_unit`, `qwen3_asr_batch_truncation`, `moonshine_streaming_batch_truncation` | Batched decode masks correctly; truncation is detected not silently accepted |
| `mel_unit` | The mel frontend itself — the subsystem Phase 9 rewrites |
| `tokenizer_smoke`, `tokenizer_decode_only_unit`, `*_tokenize_parity` | Tokenizer decode correctness |
| `conv_pw_promote_unit` | The pointwise-conv promotion that the depthwise optimisations depend on |
| `utf8_path_unit` | Windows UTF-8 path handling — this is a Windows-primary project |
| `thread_default_unit`, `threadpool_oversubscription` | The ggml patch `0001` invariant |

**And the sharpest one:** the only two end-to-end accuracy gates in `speech.cpp` both run **Moonshine**, which has no mel/STFT (`n_mels = 1`, raw PCM). The mel frontend — the subsystem Phase 9 exists to unify — is covered by **no** end-to-end gate at all. `mel_unit` is therefore not optional; it is the entry condition for Phase 9.

### 7.2 Gate ladder

Every phase's exit criteria name gates from this ladder. Higher classes subsume lower ones.

| Class | Name | What it proves | Cost | Runs on |
|---|---|---|---|---|
| **G0** | Structural | Compiles; links; lints pass (`lint_teardown`, `clang-format --check`, `extension_umbrella_check`, `capi_enum_sync`) | seconds | every commit |
| **G1** | Contract | Dispatch, lifecycle, ownership, ABI-size, family-registry invariants — no model weights needed | seconds | every commit |
| **G2** | Parity | Per-tensor agreement vs a pinned reference oracle, within a calibrated tolerance file | minutes | every PR (CPU) |
| **G3** | Accuracy | Corpus WER / DER / audio MSE against a pinned model + fixture corpus | minutes–hours | nightly + release |
| **G4** | Activation | A feature is reachable **and** measurably active (VRAM sharing engaged; batch path taken, not silently serialised) | minutes | every PR |
| **G5** | Performance | RTF, peak VRAM, clean/incremental build time | minutes | nightly |

### 7.3 Gate calibration — closing the 6.9× headroom

The present gate is `corpus_wer <= 10.0%` against a 1.45% baseline on 69 reference words. It can absorb five extra word errors without going red. Its own source comment is explicit that this is deliberate: it is sized to catch *structural* breaks (wrong mel layout, desynchronised decoder, tokenizer drift), not accuracy drift.

That is a fine G0/G1 gate. It is not an accuracy gate. Split it:

| Gate | Bound | Class | Rationale |
|---|---|---|---|
| `asr_e2e_wer_test` (structural) | corpus WER ≤ 10.0% — **unchanged** | G1 | Keep the existing fast structural tripwire exactly as-is |
| `asr_e2e_edits_test` (**new**) | `total_edits <= baseline_edits` (currently 1), env-overridable | G3 | At 69 words the *integer edit count* is the only sensitive statistic. One extra edit fails the build |
| `asr_stream_divergence_test` | streamed-vs-offline divergence `== 0` — **tightened from ≤ 3** | G3 | Measured value is already 0; a ratchet (L8), not a new requirement |
| `asr_corpus_wer_test` (**new**) | ≥ 500 reference words, corpus WER ≤ 1.10× recorded baseline | G3 | Restores statistical power. Extend the `fetch_asr_test_model.py` pinned table with a LibriSpeech `test-clean` subset — the table is already a dataclass list, so this is one row plus fixtures |
| `arch_golden_smoke_<family>` (**new** ×18) | Manifest-driven smoke per family: loads, produces a transcript, matches the golden's `transcript_compare: normalized` expectation | G2 | Turns 16 uncovered arches into covered ones |

### 7.4 Activation gates (class G4) — the antidote to inert features (L4)

Two new tests, both cheap, both currently red:

- **`shared_weight_vram_test`** — load one model, create N=3 sessions inside one `ScopedWeightShareKey`, and assert the per-session VRAM delta is below a threshold **and** that `SharedWeightRegistry::hit_count() >= N-1`. Without the hit-count assertion the test passes vacuously on CPU. This is the acceptance criterion "≤ 50 MB delta per additional session" made executable.
- **`batch_dispatch_test`** — run a 4-utterance batch and assert the native batched path was taken, via a counter the session increments (`n_batched_dispatches`). A silent fallback to serial `run()` fails the test rather than passing slowly.

### 7.5 CI matrix

| Job | Config | Gates | Cadence |
|---|---|---|---|
| `cpu-core` | `AUDIOCPP_MODEL_SET=core`, ABI+arches **ON** | G0, G1, G4 | every commit |
| `cpu-full` | `full`, ABI+arches ON | G0–G2 | every PR |
| `cuda` | `core`, CUDA, sm_89 | G0–G2, G5 | every PR |
| `parity-nightly` | `full`, CPU | G2, G3 (all 18 manifests) | nightly |
| `upstream-merge-dryrun` | `git merge --no-commit upstream/main` | conflict count vs the last run; fails on an increase attributable to a fork-side rename | nightly (§9) |
| `binding-abi` | header → libclang IR → 4 targets, `--check` | G1 | every PR touching `include/` |

**Note.** `SPEECHCPP_ENABLE_UNIFIED_ABI` and `SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES` default to OFF today, so the arch tree is not compiled in the default build. Phase 7 turns them ON in CI (not yet in the default product build); Phase 12 flips the product default once `libspeech` subsumes both.

---

## 8. Phased Execution Plan

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         MASTER FUSION EXECUTION PLAN (v6.0)                          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 1  Engine Hardening & Allocator Guards                            [COMPLETED]  │
│ Phase 2  Toolchain Modernization & Build Provenance                     [COMPLETED]  │
│ Phase 3  Native Embedded VAD & Long-Form Pipeline                       [COMPLETED]  │
│ Phase 4  Batched ASR + SharedWeightRegistry            [IMPLEMENTED · NOT ACTIVATED] │
│ Phase 5  Universal C ABI Subsystem & Progress Callbacks    [PARTIAL — guard coverage] │
│ Phase 6  Whisper GPU Cleanup, Arch Sync, Model Specs       [PARTIAL — no loaders]     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 7  SAFETY NET · GROUND TRUTH · ACTIVATION                          [COMPLETED] │
│ Phase 8  CONTRACT CONVERGENCE (the Reciprocity Rule, executed)           [COMPLETED] │
│ Phase 9  FRONTEND & TOKENIZER UNIFICATION (spec-driven)                  [COMPLETED] │
│ Phase 10 OVERLAP RESOLUTION — verdicts               [VERDICTS ONLY · merges pending] │
│ Phase 11 (W1a, W1b, W2a) — moonshine · moonshine_streaming · whisper   [3 of 18 DONE] │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 10.5 EXECUTE THE VERDICTS — 5 feature-merges, 5 deletions, ledger  ◄ NEXT      │
│ Phase 11a  ASR RUNTIME LAYER — build once; re-base the 3 ports; first dedup deletes  │
│ Phase 12   speech.h THIN OVER THE ENGINE — audiocpp.h frozen → shim   (pulled forward)│
│ Phase 11b  REMAINING 9 FAMILIES as thin packages, each deleting its arch dir         │
│ Phase 11c  DELETE src/runtime/ IN FULL — dispatcher, adapter, private subsystems     │
│ Phase 13   BINDINGS RETARGET (6 existing bindings → one generated IR)                │
│ Phase 14   CERTIFICATION & RELEASE 1.0                                               │
│ Track M    METHODOLOGY PARITY for audio.cpp's own families — a quota in EVERY phase  │
└──────────────────────────────────────────────────────────────────────────────────────┘

  Dependency graph (a phase may not start before its predecessors' exit gates pass):

     10.5 ──► 11a ──► 12 ──► 11b ──► 11c ──► 13 ──► 14
                                └── Track M runs alongside every arrow ──┘

  Why this order (v6.0): 10.5 is the cheapest real consolidation and proves the ledger;
  11a stops the per-family duplication multiplier before nine more families pay it;
  12 comes before 11b so every remaining port lands under its final ABI exactly once;
  11c is possible only once 11a+12 have given every dispatcher responsibility an
  engine home.
```

**Phase mapping from v4.0** — nothing is dropped, only re-sequenced and split:

| v4 phase | v5 phase | Change |
|---|---|---|
| 7 — Frontend & tokenizer unification | **9** | Unchanged in intent; the spec (§5.1) replaced the incomplete v4 spec; now depends on `mel_unit` existing (Phase 7) |
| 8 — Overlapping family fusion | **10** | Canonical picks corrected per V6 R3 (parakeet reversed); SenseVoice becomes a measured decision |
| 9 — STT arch migration | **11** | Reframed from cleanup to the roadmap's primary product deliverable; per-family gating added |
| 10 — Unified C ABI | **12** | ABI redesigned (§6); shims made a hard gate |
| 11 — Language bindings | **13** | Reframed from greenfield to retarget of 152 existing files |
| 12 — Regression & release | **14** | Gate ladder replaced (§7.2); thresholds recalibrated (§7.3) |
| *(none)* | **7** | New — safety net, ground truth, activation |
| *(none)* | **8** | New — contract convergence; the prerequisite v4 assumed away |

---

### Phase 7 — Safety Net, Ground Truth & Activation

**Goal.** Make every later phase *verifiable*, and switch on what is already built. Nothing is deleted in this phase.

**Why first.** Laws L2 and L4. 74.5 kLOC of ASR code has zero coverage here; two shipped features are inert; three latent defects are live. Every one of those is cheap to fix now and expensive to discover after a 74 kLOC refactor.

**Entry criteria.** None — this phase is unblocked today.

#### Tasks

**7.0 — Correct the record.** Apply the §2.4 documentation actions to `CHANGELOG.md` and `progress.md`. Add a `docs/reports/fusion_audit_2026-08-23.md` capturing §2.1–2.6 with the verification commands, so the next session does not re-derive them.

**7.1 — Port the regression suite (the safety net).**
- Copy all **51** test TUs from `transcribe.cpp/tests/` into `speech.cpp/tests/transcribe/`. Keep the subdirectory so the CMake glob is one block and future upstream syncs are a directory-level diff.
- Copy `tests/golden/` (66 files, 19 dirs, 301 KB) and `tests/tolerances/` (35 files). These are text/JSON; they belong in git.
- Copy `tests/fixtures/make_gguf_fixtures.py` and `qwen3_asr_bpe_parity.inc`.
- Copy `docs/porting/families/*.md` (18 files) — the design record for each arch, without which a migration is reverse-engineering.
- Copy `transcribe.cpp/docs/{extension-kinds,environment-variables,input-limits,model-family-testing,bindings}.md`.
- Register everything under `if (SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES)`; apply the `WORKING_DIRECTORY` registration rule from `progress.md` (a test that reads the production catalog must declare its working directory — a test that passes only from inside the repo is a registration defect, not an environment issue).
- **Expect failures.** Some tests will fail on first registration. Each failure is triaged as *port defect* / *real regression* / *fixture gap* and root-caused. Per the repository's own doctrine, "environment/asset issue" is not an acceptable classification until reproduced.

**7.2 — Adopt the two missing CI lints.**
- `.github/workflows/clang-format.yml` from `transcribe.cpp` (pinned formatter via `uvx`; scope excludes `external/`, `src/runtime/third_party/`, and the verbatim `transcribe-unicode-data.cpp`).
- `AGENTS.md` — the command/automation convention sheet (`uv run` discipline, build target, formatter invocation, validation entry points, git hygiene). `speech.cpp` currently has neither.

**7.3 — Fix D1 (dispatch precedence).** Change `transcribe.cpp:1570` to `adapter_find_arch(family.c_str())`. Add `adapter_sniff_dispatch_unit` asserting the adapter entry wins for `qwen3_asr`, `voxtral_realtime`, `moss` on a framework-sniffed path. *This is a correctness fix, not a refactor — it lands alone.*

**7.4 — Activate `SharedWeightRegistry` (L4).**
- Wrap load + session-create in a `ScopedWeightShareKey` at all four documented call sites the header names: `capi/src/audiocpp_capi.cpp` (`audiocpp_load_model_ex`), `app/cli/main.cpp`, `app/server/http.cpp`, `app/workflow/`.
- Key = canonical family id + absolute model path (the backend/device discriminator is appended by `BackendWeightStore` itself).
- Add `hit_count()` / `miss_count()` counters to the registry.
- Gate: `shared_weight_vram_test` (§7.4). **This test must be red before the change and green after** — that is the proof the feature was inert.

**7.5 — Make batched decode reachable (L4).**
- ArchAdapter: implement `adapter_run_batch_impl` forwarding to `IOfflineVoiceTaskSession::run_batch`, honouring the `Arch::run_batch` contract (push exactly `n` results in order; mirror slot 0; poll abort between utterances; per-utterance failure is a non-OK `ResultSet`, not a batch error). Replace `nullptr` in all 16 rows.
- Also implement `adapter_run_validate_impl` and `adapter_stream_validate_impl` so the pre-clear preservation guarantee (D2) holds for framework families.
- Add `speech`-side batch entry points: `audiocpp_asr_batch()` now, `speech_run_asr_batch()` in Phase 12; wire `--batch` in the CLI and a batch endpoint in the server.
- Gate: `batch_dispatch_test` (§7.4).

**7.6 — Split and recalibrate the WER gates** per §7.3: keep the 10% structural bound; add `asr_e2e_edits_test` (edits ≤ baseline), tighten stream divergence to 0, add the ≥500-word corpus row to `scripts/fetch_asr_test_model.py`.

**7.7 — Turn on the arch build in CI.** Add `-DSPEECHCPP_ENABLE_UNIFIED_ABI=ON -DSPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=ON` to the `cpu-core`, `cpu-full` and `cuda` CI jobs. Record clean and incremental build-time deltas — Phase 12 needs the baseline. Product defaults stay OFF until Phase 12.

**7.8 — Family registry v1 (read-only).** Land `family_registry.h` + `family_registry_unit` (§5.5) in *reporting* mode: the test prints orphan specs, unspec'd families and alias collisions but does not yet fail. It becomes a hard gate at the end of Phase 8, once Phase 7's inventory is reconciled.

#### Exit gates

| Gate | Requirement |
|---|---|
| G0 | `clang-format --check` green; `lint_teardown` green on `src/runtime/` |
| G1 | All 51 ported tests registered; **≥ 48 green**, every non-green one root-caused with a written cause and a tracking entry (no "environment issue" labels) |
| G1 | `adapter_sniff_dispatch_unit` green |
| G4 | `shared_weight_vram_test` green (was red) |
| G4 | `batch_dispatch_test` green (was red) |
| G3 | `asr_e2e_edits_test` green at `edits <= 1`; stream divergence gate at 0 |
| G0 | CI matrix builds with arches ON on CPU and CUDA |
| doc | `docs/reports/fusion_audit_2026-08-23.md` merged; `CHANGELOG.md` / `progress.md` corrected |

**Rollback.** Every task is an independent commit. 7.1 is additive (tests only). 7.3/7.4/7.5 are individually revertible.
**Risk.** *Ported tests fail in bulk* — likely, and the point. Mitigation: register in four batches (contract → tokenizer/mel → per-family smoke → e2e) so triage is bounded.

---

### Phase 8 — Contract Convergence & Exception Boundary `[COMPLETED]`

**Goal.** Raise `engine::runtime`'s session contract to `transcribe::Arch`'s level, establish unified `StreamingSessionBase`, `RunControl`, `StreamChunker`, `DecodeTelemetry`, and 100% C ABI exception containment.

**Status.** **Completed & 100% Verified** (92/92 CTest targets green).

**Entry.** Phase 7 gates green — specifically `stream_dispatch_unit`, `run_dispatch_unit`, `stream_capability_unit`, `stream_committed_pointer_stability` must be running against the arch tree, because they become the acceptance tests for the new base classes.

#### Tasks

**8.1 — `StreamingSessionBase` owns the lifecycle.**
New `include/engine/framework/runtime/streaming_session_base.h` + impl, providing:
- The 4-state machine `IDLE → ACTIVE → FINISHED | FAILED`, written **only** by the base.
- `uint64_t revision()`, bumped on any observable snapshot change; reset by begin/reset/run.
- Commitment policy engine: `AUTO` / `ON_FINALIZE` / `STABLE_PREFIX` with `stable_prefix_agreement_n` (default 3); `committed_text` append-only; finalize never rewrites committed bytes; families may supply an optional native commit boundary.
- `StreamChunker` — lifted out of `transcribe-arch-adapter.cpp`, where it already solves exactly this: buffer caller PCM, dispatch only whole chunks at `streaming_policy().preferred_audio_chunk_samples`, flush the tail at finalize, maintain contiguous `start_sample`.
- `virtual Status validate(const TaskRequest&, const StreamParams&) const` — pure, called **before** any destructive step; on non-OK the prior snapshot *and* lifecycle are preserved.

Existing streaming sessions (18 of them) are migrated to derive from it. Each migration is one commit with the family's golden smoke as its gate.

**8.2 — `RunControl` unifies progress and abort.**
Today `ProgressCallback` exists offline-only and cancellation is expressed by throwing `ProgressCanceled`; the arch side polls `ctx->poll_abort()` in feed loops and between batch utterances. Introduce one object available in both modes:

```cpp
class RunControl {
public:
    bool poll_abort() const noexcept;         // cheap, called in inner loops
    void emit_progress(const ProgressInfo &); // throws ProgressCanceled on cancel
    void request_abort() noexcept;            // thread-safe, from any thread
};
```
`SessionBase` owns one; `run()`, `run_batch()`, `process_audio_chunk()` and `finalize()` all receive it. `speech_session_request_abort()` (§6.3) is its public face — a capability neither parent exposed.

**8.3 — Batched decode contract in the base.** Move the dispatcher-level batch invariants into `OfflineSessionBase::run_batch()`: validate shared params once; isolate per-utterance failures inside each result; poll abort between utterances; pad the tail on abort. Families override only the batched graph.

**8.4 — Expand the teardown lint to the whole tree (L9).**
`tests/lint_teardown.cmake` documents its own limitation: *"currently scoped to `src/runtime/` … the broader `src/` tree still contains raw `ggml_backend_free`/`buffer_free`/`sched_free` calls inherited from the audio.cpp model sessions — converting those is Phase 0 sub-task 0.J."*

Current debt: **167 raw call sites across 75 files**
(`grep -rn "ggml_backend_free\s*(\|ggml_backend_buffer_free\s*(\|ggml_backend_sched_free\s*(" src/ | grep -v "^src/runtime/" | wc -l`).

Migrate file-by-file to `transcribe::safe_backend_free` / `safe_buffer_free` / `safe_sched_free`, in dependency order: `src/framework/core/` → `src/framework/audio/` → `src/framework/modules/` → `src/models/` → `src/community_models/`. Each batch flips a directory into the lint's `SRC_DIR` scope, so the gate ratchets and cannot regress. Land as ~8 mechanical commits.

**8.5 — 100% exception containment (L9).**
- Introduce `api_guard_status` / `api_guard_value` / `api_guard_void` in the capi layer (port from `transcribe.cpp:73-103`).
- Wrap the **~26** currently-unguarded exported definitions — notably `audiocpp_device_count`, `audiocpp_device_info`, `audiocpp_list_devices`, `audiocpp_backend_available`, `audiocpp_model_info`, `audiocpp_model_capabilities`, `audiocpp_write_wav`, `audiocpp_free_model`, `audiocpp_stream_free`, and every `audiocpp_free_*`.
- Enforce "non-OK ⇒ `*out == NULL`, nothing leaked" in every forwarder with an ownership out-param.
- New lint `tests/lint_api_guard.cmake`: parse `capi/src/*.cpp`, list every definition whose name matches an exported symbol in `capi/include/audiocpp.h`, fail if its body lacks a guard. Prevents silent regrowth.

**8.6 — Result telemetry into `TaskResult`.** Add an optional `DecodeTelemetry` per segment (accepted temperature tier, compression ratio, avg logprob, no-speech probability, no-speech triggered, fallback count) so Whisper's decoding quality signals survive migration and reach the CLI/server (`--json` output) and the C ABI.

**8.7 — Family registry becomes a hard gate.** Flip `family_registry_unit` from reporting to failing. Reconcile the orphans Phase 7 listed: give the 7 spec-less arches a `model_specs/*.json`; give `whisper` / `moonshine` / `moonshine_streaming` a registered loader (initially the ArchAdapter, permanently in Phase 11).

#### Exit gates

| Gate | Requirement |
|---|---|
| G1 | All 18 streaming sessions derive from `StreamingSessionBase`; `stream_dispatch_unit`, `stream_capability_unit`, `stream_committed_pointer_stability` green **against engine sessions**, not only arch ones |
| G1 | `family_registry_unit` green in failing mode; zero orphan specs, zero unspec'd families |
| G0 | `lint_teardown` green over the **entire** `src/` tree; 0 raw teardown calls outside `transcribe-backend.{h,cpp}` |
| G0 | `lint_api_guard` green; 100% of exported entry points guarded |
| G1 | `teardown_safety_unit`, `backend_init_throw_unit` green |
| G2 | All 18 `arch_golden_smoke_*` still green — this phase changes contracts, not numerics |
| G3 | WER gates unchanged: `edits <= 1`, divergence 0 |

**Rollback.** 8.1/8.2/8.3 land behind `SPEECH_UNIFIED_SESSION_BASE` (default ON in CI, OFF in the product) until the exit gates pass. 8.4/8.5 are mechanical and revertible per directory.
**Risk.** *Migrating 18 streaming sessions destabilises TTS streaming.* Mitigation: migrate ASR sessions first (they have golden smokes), TTS sessions last, one commit each.

---

### Phase 9 — Frontend & Tokenizer Unification — `[x] COMPLETED (100% Green)`

**Goal.** One frontend, one tokenizer hub, derived from a written specification (L6) and proved by parity tests.

**Entry.** Phase 8 green; **`mel_unit` ported and green** (Phase 7). Without `mel_unit` this phase is unverifiable — the two e2e gates run Moonshine, which has no mel path (§2.5 C4).

#### Tasks

- [x] **9.1 — `FrontendSpec` and the unified extractor** (`engine::audio::MelExtractor` & `FrontendSpec`).
- [x] **9.2 — Tokenizer hub** (`engine::text::TokenizerHub` & `ITokenizer`).
- [x] **9.3 — Unicode consolidation.**
- [x] **9.4 — Codec hub consolidation** (`include/engine/framework/codecs/codec.h`).

#### Exit gates

| Gate | Requirement | Status |
|---|---|---|
| G1 | `frontend_contract_test` green for all 18 arch families + every engine family with a frontend | `[x] PASSED` |
| G2 | `frontend_parity_test` green within each family's calibrated tolerance | `[x] PASSED` |
| G1 | `mel_unit` green against the unified extractor | `[x] PASSED` |
| G2 | `tokenizer_parity_test`, `qwen3_asr_bpe_parity`, `whisper_tokenize_parity`, `whisper_bin_tokenize_parity`, `tokenizer_decode_only_unit` green | `[x] PASSED` |
| G3 | WER gates unchanged (`edits <= 1`, divergence 0) — the frontend rewrite must be numerically invisible | `[x] PASSED` |
| G5 | Frontend RTF within 5% of the pre-unification baseline (transcribe's SIMD path must survive the merge) | `[x] PASSED` |

**Risk.** *A family's true frontend differs from its manifest.* Mitigation: `frontend_contract_test` fails loudly and the manifest is corrected first — the manifest is the spec, and a mismatch is a finding either way.

---

### Phase 10 — Overlap Resolution — `[x] COMPLETED (100% Green)`

**Goal.** One implementation per duplicated family, chosen by evidence (§5.4, V6 R3).

**Status.** **Completed & 100% Verified** (95/95 CTest targets green).

**Entry.** Phase 9 green. Both implementations of each overlapping family must be runnable against the same golden manifest — which is only true after Phases 7 and 9.

#### Tasks

- [x] **10.0 — Run the bake-off.** For each of the 6 overlaps, evaluate implementations against golden manifest and record metrics. Published `docs/reports/overlap_bakeoff.md`.
- [x] **10.1 — `parakeet_tdt` → the arch implementation is canonical.**
- [x] **10.2 — `qwen3_asr` → engine canonical.**
- [x] **10.3 — `voxtral_realtime` → engine canonical.**
- [x] **10.4 — `fun_asr_nano` → engine canonical.**
- [x] **10.5 — `sense_asr` / `sensevoice` → decided by 10.0.**
- [x] **10.6 — `sortformer_diar` → engine canonical.**
- [x] **10.7 — `moss` → both survive, id collision resolved.**

#### Exit gates

| Gate | Requirement | Status |
|---|---|---|
| G2 | For each resolved family: the surviving implementation passes **every** golden manifest the losing one passed, at equal or tighter tolerance | `[x] PASSED` |
| G3 | Parakeet: all 13 golden variants green; multitalker RTTM output byte-identical to the arch baseline | `[x] PASSED` |
| G1 | `family_registry_unit` green with canonical ids; zero aliases resolving to two targets | `[x] PASSED` |
| G4 | `batch_dispatch_test` green for every family that had a batched path on either side | `[x] PASSED` |
| ledger | Overlap resolutions recorded for all 6 pairs | `[x] PASSED` |
| doc | `docs/reports/overlap_bakeoff.md` published | `[x] PASSED` |

**Risk.** *Parakeet is the largest single port in the roadmap (8 kLOC, 11 variants).* Mitigation: port variant-by-variant, each gated by its own golden manifest; the arch copy stays buildable until all 13 are green.

---

### Phase 10.5 — Execute the Phase-10 Verdicts *(v6.0, next)*

**Goal.** Turn the bake-off's five engine-wins into the first real consolidation: merge each loser's distinguishing features into the winner (R3's precondition), then delete the five arch directories with ledger rows. ≈17 kLOC removed; the D1 shadowing hazard gone.

**Entry.** Phase 7 assets (goldens for all five families) in place — they are.

| Family | Feature to merge into the engine winner first | Gate before deletion |
|---|---|---|
| `qwen3_asr` | speculative drafts (`spec_k_drafts`, `supports_spec_decode`) — as the shared AR driver's option if 11a has landed, else a package-local port; `qwen3_asr_bpe_parity` | `tests/golden/qwen3_asr/` (2) through the engine; `qwen3_asr_batch_truncation` |
| `voxtral_realtime` | cache-aware streaming windows | `tests/golden/voxtral_realtime/` (1); streamed-vs-offline divergence 0 |
| `sortformer_diar` | streaming presets + the typed `sortformer.h` extension | `tests/golden/sortformer/` (1); DER unchanged |
| `sense_asr` | (SAN-M direct-dw already unified in Phase 10) | `tests/golden/sensevoice/` (1) |
| `fun_asr_nano` | adopt the arch WER corpus as the engine gate | new `fun_asr_nano_wer_test` at the arch baseline |

Deletions: `src/runtime/arch/{qwen3_asr, voxtral_realtime, sortformer, sensevoice, funasr_nano}/`, each its own commit, Appendix B rows B11–B15 filled in. `asr_e2e_wer_test` / `asr_stream_text_wer_test` are unaffected (their subjects are the moonshine pair).

**Exit.** Five ledger rows with revert commits; `family_registry_unit` reports no shadowed GGUF arch; suite green.

---

### Phase 11a — The ASR Runtime Layer, and Re-basing the First Three Ports *(v6.0)*

**Goal.** Build §5.6 once and prove it on the families already migrated, so that no later port re-creates a runner.

**Tasks.**

**11a.1 — `EncDecKVCache`** (§5.6). Land beside the existing structs behind a switch; W1a/W1b/W2a adopt it; their three private structs are deleted.
**11a.2 — Decode drivers.** `ar_greedy` first (it retires the three engine loops and is what Whisper, Moonshine, Canary, Cohere, Voxtral need), with suppress masks, timestamp rules, temperature-fallback ladder + `DecodeTelemetry`, and `spec_k_drafts`. Then `transducer_greedy` (unifying the engine's three `tdt_decoder_*` variants with parakeet's batched joint window) and `ctc_greedy`.
**11a.3 — `AsrResult` + `AsrLimits`.** Cross-indexed rows and committed counts projected onto `TaskResult`/`StreamEvent`; `CapabilitySet.max_audio_ms`; `TaskResult.truncated`; `CapacityError → INPUT_TOO_LONG` at the ABI. **Fixes W2a's silent truncation** (L11).
**11a.4 — Long-form driver** over `audio/chunking`; whisper's HF-5.x seek continuation lands here, not in the family. `transcribe-vad*` deleted (§5.7).
**11a.5 — Fold the Whisper encoder onto `WhisperEmbeddingModule`** (§5.9) and the `.bin` parser into a `TensorSource` (§5.8).
**11a.6 — `GraphExecutor` scheduler path** (§4.2 "Backend execution"): optional `primary + CPU` scheduler for families that need op fallback; single-backend remains the default.
**11a.7 — Track M quota:** two audio.cpp-native ASR families (`nemotron_asr`, `citrinet_asr`) gain golden manifests + tolerance files + `validate.py` support for the engine path.

**Exit gates.** `moonshine_engine_smoke_test`, `moonshine_streaming_engine_smoke_test`, `whisper_engine_smoke_test` unchanged at their arch baselines (1/69 · 3/69 divergence 0 · 3/69) **on the shared layer**; `grep -rn 'struct [A-Za-z]*KvCache' include/engine/models` → 0; a > 30 s clip through Whisper returns `truncated = true` and the ABI status says so; `vad_plan_unit` green against `plan_vad_audio_chunks`.

---

### Phase 11b — Arch Migration: the Remaining 9 Families as Thin Packages

> **Status (2026-08-26, v6.0)**: W1a, W1b and W2a are **done and gated at exact arch parity** (`moonshine` 1/69 · `moonshine_streaming` 3/69 with divergence 0 · `whisper` 3/69). They are re-based onto the ASR layer in Phase 11a before this phase resumes. Remaining: `canary`, `canary_qwen`, `cohere`, `gigaam`, `granite`, `granite_nar`, `medasr`, `voxtral` (offline), `moss` (ASR + diarization), plus `whisper` W2b scope (timestamps, language detection, batched decode) which is now mostly ASR-layer work.

**Goal.** Make Whisper, Moonshine, Moonshine-Streaming, Canary, Canary-Qwen, Cohere, GigaAM, Granite, Granite-NAR, MedASR (and the Phase-10 survivors) first-class engine families — reachable from the CLI, server, WebUI and the shipped C ABI, installable from the model manager.

**This is the highest-value phase in the roadmap** (§2.3). v4 framed it as internal cleanup; it is the reason a user would choose `speech.cpp` over either parent.

**Entry.** Phases 8, 9, 10 green. Per family, the four entry conditions of §4.4.

#### Migration order — cheapest-risk first, highest-value first

| Wave | Families | Rationale |
|---|---|---|
| **W1** | `moonshine`, `moonshine_streaming` | Both WER gates already cover them; raw-PCM frontend means Phase 9 cannot have perturbed them; smallest models. The safest possible first migration, and it converts the roadmap's own regression gates into engine-side gates |
| **W2** | `whisper` | Highest user demand; 16 catalog packages already written; 12 golden manifests; carries the HF 5.x seek-continuation fix, the `.bin` loader, suppress-token tables, and the temperature-fallback ladder + telemetry (Phase 8.6) |
| **W3** | `gigaam`, `medasr`, `cohere` | Small, single-variant, self-contained; `gigaam` also retires the third private mel implementation (`arch/gigaam/mel.h`) |
| **W4** | `canary`, `canary_qwen`, `voxtral` (offline) | Larger; `canary` pair shares a decoder lineage, both have batched paths. `voxtral` unlocks `voxtral-mini-3b`, `voxtral-mini-4b-realtime` and `voxtral-small-24b`, none of which any engine loader can reach today |
| **W5** | `granite`, `granite_nar`, `moss` (ASR + diarization) | `granite` pair depends on Shaw relative attention (§5.3) landing in the framework module. `moss_asr` lands under its new canonical id from §10.7, alongside the untouched `moss_tts_*` families |
| **W6** | Phase-10 survivors' arch copies retired | Bookkeeping close-out |

#### Per-family procedure

Follow `docs/porting/` stages 4→8 (the earlier stages are already done — these families are ported, just not integrated):

1. Create `src/models/<family>/` as a **thin package** *(v6.0)*: `graphs.cpp` (encoder/decoder topology only, on framework modules where one exists), `assets.cpp` (hparams + tokenizer via `TokenizerHub` + `TensorSource`), `session.cpp` (a thin session over the ASR layer's KV cache, decode driver, `AsrResult` and `AsrLimits`). **No private KV cache, no private decode loop, no private loader, no private mel.** A package that needs one of those has found a gap in the ASR layer — fix the layer.
2. Implement `IVoiceModelLoader` + `ILoadedVoiceModel` + a session deriving from `OfflineSessionBase` / `StreamingSessionBase`.
3. Register in `CMakeLists.txt` via `audiocpp_add_model(<family> SOURCES … LOADERS … ALIASES …)`; add to the `asr` and `full` composites.
4. Author or complete `model_specs/<family>.json` **schema v2** — including the `"frontend"` block from §5.1 and a `"parity"` block naming the golden manifest and tolerance file.
5. Register the family in `family_registry.h` with its GGUF arch name(s) as aliases.
6. Move `tests/golden/<family>/` and `tests/tolerances/<family>*.json` from arch-scoped to engine-scoped registration; add `<family>_golden_smoke` and, where a real model is pinned, `<family>_real_smoke`.
7. Verify reach: `audiocpp_cli --task asr --family <family> --model <path>` works; the server exposes it; the WebUI lists it; `audiocpp_load_model(..., "<family>", ...)` succeeds.
8. Delete `src/runtime/arch/<family>/` — **in the same wave** *(v6.0)*, its own commit, with a ledger row. Coexistence (§4.4 steps 5–7) is measured within the wave, not carried for a release cycle: the ABI gates now run through the engine path (Phase 12), so the arch copy has nothing left to prove once the engine copy matches it.
9. Retarget the family's e2e gates: `asr_e2e_*_test` binaries drive `speech.h`, whose implementation is the engine — the arch copy is no longer their subject.

#### Decommission the bridge — superseded by Phase 11c *(v6.0)*

v5 kept the transcribe dispatcher, loader, model/session bases, VAD, bin-loader and batch-util under `src/runtime/` "as what the C ABI genuinely needs." **That is reversed.** Once every family is a thin package on the ASR layer and `speech_capi.cpp` sits directly on the engine base classes (Phase 12), none of those files has a responsibility the engine does not already own. See Phase 11c.

#### Exit gates

| Gate | Requirement |
|---|---|
| G1 | Every one of the 18 families resolvable by canonical id through `ModelRegistry`; `family_registry_unit` green |
| G1 | `registry.families()` count increases by exactly the number migrated; zero orphan specs |
| G2 | All 66 golden manifests green through the **engine** path |
| G3 | `asr_corpus_wer_test` (≥ 500 words) green for `whisper`, `moonshine`, `parakeet` |
| G3 | `edits <= 1`, divergence 0, still green |
| G1 | CLI/server/WebUI reachability asserted per family by `cli_family_smoke` (extend `cli_output_smoke.cmake` from `transcribe.cpp`) |
| G5 | Per-family RTF and peak VRAM within 10% of the arch baseline recorded in Phase 7 |
| ledger | One Appendix B row per deleted arch directory + the adapter |

**Risk.** *Whisper is 7,231 LOC with a legacy `.bin` path, 12 variants and the most intricate decoding loop in the tree.* v6.0 status: the offline core (W2a) landed at exact arch parity; the `.bin` path becomes a `TensorSource` (§5.8) and the decode loop becomes the shared `ar_greedy` driver (§5.6) in Phase 11a — which is what makes the remaining Whisper scope (W2b) small.

---

### Phase 11c — Delete `src/runtime/` in Full *(v6.0)*

**Goal.** The end state v5 stopped short of: no second runtime. 95 kLOC deleted, every line with a ledger row.

**Entry.** Phase 11b complete (no arch dir left) **and** Phase 12 complete (`speech_capi.cpp` over the engine, all ABI gates green through it).

**What goes, and where its responsibility now lives.**

| Deleted | LOC (approx.) | Now owned by |
|---|---:|---|
| `transcribe.cpp` (dispatcher) | 3,300 | `StreamingSessionBase` (lifecycle, revision, commit policy, pre-clear validate), `RunControl`, `IOfflineVoiceTaskSession::run_batch` default, `speech_capi.cpp` (status mapping, result snapshot rules) |
| `transcribe-arch{,-adapter}.{h,cpp}` | 1,300 | `ModelRegistry` + `family_registry` |
| `transcribe-model.{h,cpp}`, `transcribe-session.h` | 900 | `ILoadedVoiceModel`, `RuntimeSessionBase`, `AsrResult`, `AsrLimits` |
| `transcribe-loader`, `transcribe-load-common`, `transcribe-bin-loader`, `transcribe-weights-util`, `transcribe-meta` | 3,400 | `TensorSource` (+ whisper `.bin` source), `BackendWeightStore`, `model_spec` |
| `transcribe-backend.{h,cpp}`, `transcribe-flash-policy` | 400 | `ExecutionContext` + the `GraphExecutor` scheduler path; `safe_*` wrappers live in `core/backend.h` |
| `transcribe-mel`, `transcribe-kaldi-fbank` | 1,450 | `audio/mel_extractor`, `audio/kaldi_fbank` (Phase 9) |
| `transcribe-tokenizer`, `transcribe-unicode{,-data}` | 3,000 | `text/tokenizer_hub`, `text/unicode_normalization` (Phase 9) |
| `transcribe-vad{,-integrate}` | 500 | `audio/chunking` (§5.7) |
| `transcribe-batch-util` | 600 | the ASR layer's batched decode drivers |
| `causal_lm/`, `conformer/`, `granite_conformer/`, `sanm/` | 3,500 | framework modules (Phase 10) |
| `transcribe-debug`, `transcribe-env`, `transcribe-log`, `transcribe-path` | 900 | `debug/trace`, `debug/profiler`, engine env/log conventions; the `TRANSCRIBE_DUMP_DIR` dump points move to engine tracing so `validate.py` keeps working |
| `arch/` (whatever remains after 11b) | — | thin packages |

`SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES` and `SPEECHCPP_ENABLE_UNIFIED_ABI` are retired; `lint_teardown` runs over all of `src/` (A12); the `tests/transcribe/` translation units that tested dispatcher behaviour are re-pointed at the base classes (they are contract tests, and the contract survives).

**Exit gates.** `ls src/runtime` → does not exist; A2 (all ported TUs green, re-pointed); A3; `lint_teardown` over `src/` → 0; every Appendix B row has a revert commit.

---

### Phase 12 — Unified `libspeech` ABI & Compatibility Shims *(pulled forward in v6.0)*

**Goal.** One shared library exporting the whole speech-intelligence surface, with zero breakage for existing `audiocpp` and `transcribe` consumers — **implemented thinly over the engine base classes**, not over the transcribe dispatcher.

**Entry (v6.0).** Phase 11a green. v5 gated this on Phase 11 ("a unified ABI over two parallel model layers would just be a third façade"); the third façade already exists (`capi/audiocpp.h`, default ON, §2.4), so the cheaper order is to freeze it now and build `speech.h` before the nine remaining ports — each then lands under its final ABI once. Rule from today: **no new entry points in `audiocpp.h`.**

#### Tasks

**12.1 — Implement `speech.h` / `speech_capi.cpp`** per §6.3, including size-aware structs + `speech_abi_struct_size()`, typed extension slots, decode telemetry, `RunControl`-backed progress + abort, and the two result-ownership models.
**12.2 — Port the family extension headers** to `include/speech/<family>.h` (`whisper`, `parakeet`, `sortformer`, `voxtral_realtime`, `moonshine_streaming`) plus a `speech/extensions.h` umbrella; keep `transcribe_extension_umbrella_check` as `speech_extension_umbrella_check`.
**12.3 — Single-artifact build.** `SPEECH_SHARED_EMBED=ON`: static ggml archives compiled into `speech.dll` / `libspeech.so` with hidden internal symbols; embedded `VS_VERSION_INFO`; a `.def`/version-script export list gated by a test that diffs actual exports against the header (extend `capi/test/exports_raw.txt`).
**12.4 — Compatibility shims** per §6.4, with `abi_compat_test` as the hard gate.
**12.5 — Flip the product defaults.** `SPEECHCPP_ENABLE_UNIFIED_ABI` becomes unconditional; the default build ships `speech.dll` plus the two import shims. Record and publish the binary-size and build-time delta.

#### Exit gates

| Gate | Requirement |
|---|---|
| G0 | Exactly **one** primary shared library artifact + two compatibility import libraries |
| G1 | `speech_capi_test`, `capi_option_number_test`, `capi_session_options_test`, `capi_enum_sync_test`, `speech_extension_umbrella_check`, `exports_match_header_test` green |
| G1 | `abi_compat_test` green — legacy binaries produce byte-identical results through the shims |
| G1 | `lint_api_guard` green over `speech_capi.cpp` |
| G3 | Full accuracy suite unchanged |
| G5 | Clean CUDA build ≤ 4.0 min (auto-arch + ccache); incremental ≤ 10 s; documented delta vs the Phase-7 baseline |

---

### Phase 13 — Bindings Retarget

**Goal.** Four language bindings over `libspeech`, generated from one parsed IR, CI-gated on header ABI drift. **Not greenfield** — §2.5 C9: 152 files already exist in `transcribe.cpp`.

#### Tasks

**13.1 — Port the generator.** `bindings/python/_generate/generate.py` (534 lines, libclang) + `check_version_sync.py` (249) → retarget from `include/transcribe/extensions.h` to `include/speech/extensions.h`. It already emits both the Python ctypes layer and the TypeScript koffi layer from one AST parse, embeds a `PUBLIC_HEADER_HASH`, and is semantic (comment/whitespace edits produce no diff; any real ABI change does).
**13.2 — Extend the generator to Rust.** `bindings/rust/sys/src/transcribe_sys.rs` (1,346 lines) is currently hand-maintained. Emitting it from the same IR removes a whole class of drift and is the natural payoff of owning the generator.
**13.3 — Port the four bindings.** Python (`speech_cpp`, 1,490-line safe layer + 13 test modules), TypeScript (koffi), Rust (`sys` + safe crate + `xtask`), Swift (SPM + 5 example targets + a demo app). Add the audio.cpp task surface — TTS, separation, alignment, MIDI, artifacts — which the transcribe bindings never had.
**13.4 — Native wheels.** Port `python-native` and `python-native-cu12` (bundled binary wheels, CPU and CUDA 12).
**13.5 — CI gate.** `binding-abi` job: regenerate, `git diff --exit-code`. A header ABI change that is not reflected in all four bindings fails the build.

#### Exit gates

| Gate | Requirement |
|---|---|
| G1 | `generate.py --check` green for Python, TypeScript **and** Rust |
| G1 | All 13 Python binding tests green; Rust `cargo test` green; Swift examples build; TS test suite green |
| G1 | `check_version_sync.py` green (`SPEECH_VERSION_*` == package versions across all four) |
| G3 | Each binding transcribes a fixture end-to-end and matches the C result exactly |

---

### Phase 14 — Golden Regression Certification & Release 1.0

#### Tasks

**14.1 — Certification matrix.** Run the full gate ladder across `{CPU, CUDA, HIP, Vulkan, Metal} × {core, asr, full}` and publish `docs/reports/certification_1.0.md` with every measured number and the command that produced it.
**14.2 — Close the parity debt.** Every family with a golden manifest gets a calibrated tolerance file (35 exist; the engine-only families need theirs authored via `scripts/calibrate_tolerances.py`, promoted out of the `tmp/` script referenced in the whisper tolerance comment).
**14.3 — TTS parity gate.** The audio.cpp side has no numerical-parity gate today. Apply transcribe's oracle method to TTS: pin reference implementation + revision per family, dump reference tensors, calibrate tolerances, add waveform MSE + a perceptual check. This is the clearest single instance of transcribe teaching audio.
**14.4 — Diarization DER gate** on multi-talker fixtures (`tests/golden/parakeet/multitalker-2spk-mix.rttm` is already the seed).
**14.5 — Docs.** Full `speech.h` API reference; migration guides for `audiocpp` and `transcribe` consumers; per-family docs merged from `docs/models/`, `docs/community_models/` and the ported `docs/porting/families/`.
**14.6 — Release.** Tag `speech.cpp v1.0.0`; publish wheels, crates, npm package and SPM tag; `RELEASING.md` ported from `transcribe.cpp`.

#### Exit gates — §10 Acceptance Criteria, in full, on all supported backends.

---

## 9. Upstream Merge Strategy

`speech.cpp` is a **live fork**: `upstream/audio.cpp` is an active remote, four upstream merges have already landed, and the tree is currently 57 ahead / 0 behind. v4 did not mention this once, yet almost every task it proposed edits or deletes files that upstream continues to develop.

### 9.0 Two parents, one child *(v6.0, L13)*

`speech.cpp` is equally a child of `audio.cpp` and of `transcribe.cpp`. Only `audio.cpp` has a git `upstream` remote here (it is the fork base), so only it yields a merge-base and a "N behind" count — **a tooling limitation, not a hierarchy**. transcribe.cpp drift is tracked by hand through the sibling checkout.

- `scripts/sync-deps.sh` is the first command of every phase: audio.cpp via the `upstream` remote, transcribe.cpp via the sibling, ggml against **both** upstream HEAD and transcribe.cpp's pin (our ggml floor is never below the parent's).
- An audio.cpp sync ends in a recorded merge (real, or `-s ours` after a by-content audit) — never a content copy; `git rev-list --left-right --count HEAD...upstream/main` must end in `0`.
- A transcribe.cpp commit gets the same by-content audit and the same disposition ledger as an audio.cpp commit. "Merge source" is not a lesser status.
- See tracker Operating Rules 6 and 7 and `AGENTS.md` § Dual Parentage.

### 9.1 The ownership map

| Region | Owner | Merge policy |
|---|---|---|
| `src/models/`, `src/community_models/`, `src/framework/`, `app/`, `webui/`, `model_specs/`, `tools/` | **Upstream** | Additive only. Never rename, never move. Replace function *bodies*; keep files, symbols and signatures |
| `src/runtime/`, `include/transcribe/`, `capi/`, `patches/ggml/`, `tests/transcribe/`, `tests/golden/`, `tests/tolerances/` | **Fork** | Free to restructure |
| `external/ggml/` | **Generated** | The tree is generated by `scripts/sync-ggml.sh`; **the 7 patches are the invariant**, not the tree. Never hand-edit; re-derive |
| `include/speech/`, `bindings/`, `src/models/<migrated arch families>/` | **Fork (new)** | New paths upstream does not have — conflict-free by construction |

### 9.2 Rules

1. **New family directories are conflict-free.** `src/models/whisper/` does not exist upstream, so it can never conflict. This is why Phase 11 is *cheap* to merge despite being the largest phase.
2. **Deduplication never deletes an upstream file.** Retiring the eight private mel builders means replacing each function's body with a call into the shared extractor — a 5–10 line hunk that git merges cleanly — not deleting `src/models/index_tts2/audio_features.cpp`.
3. **Batch fork-side edits to upstream files.** When Phase 8.4 converts 75 files to `safe_*` teardown, land them as ~8 directory-scoped commits so an upstream conflict is localised to one directory, not one 75-file commit.
4. **Merge before every phase boundary.** `git fetch upstream && git merge upstream/main` is the first task of each phase, not an afterthought. A phase that starts 200 upstream commits behind pays for it in the middle.
5. **The `upstream-merge-dryrun` CI job** (§7.5) runs `git merge --no-commit --no-ff upstream/main`, counts conflicted files, and fails when the count rises for a reason attributable to a fork-side rename. Conflict count becomes a tracked, visible metric rather than a periodic surprise.

### 9.3 `transcribe.cpp` divergence

`transcribe.cpp`'s upstream is `NairoDorian/transcribe.cpp` (itself forked from `handy-computer/transcribe.cpp`). The arch tree here is a point-in-time copy. Two options, decided in Phase 7:

- **(a) Freeze.** Treat `src/runtime/arch/` as vendored-at-a-pin; record the source commit in `patches/` or a `VENDOR` file; take no further transcribe updates. Simple; loses future upstream ASR work.
- **(b) Track until Phase 11 completes.** Keep a `transcribe` remote and cherry-pick arch fixes until each family migrates, then freeze that family.

**Recommendation: (b), with an explicit pin file.** Add `src/runtime/arch/VENDOR.md` recording the source repo, commit and date per family, updated when a family is synced and marked `FROZEN — migrated` when it is retired. Cost is near-zero; it preserves the option to pick up an upstream ASR fix during the 6–8 phases before a given family migrates.

---

## 10. Acceptance Criteria

Every criterion names its gate and how it is measured. A criterion with no runnable measurement is not a criterion.

| # | Criterion | Requirement | Gate | Measured by |
|---|---|---|---|---|
| A1 | Test suite | 100% green on `{CPU, CUDA} × {core, full}`; skips only for documented model-download contracts | G0–G3 | `ctest --output-on-failure` |
| A2 | Ported suite | All **51** transcribe TUs registered and green | G1 | `ctest -R "transcribe_"` |
| A3 | Golden coverage | All **66** manifests green through the engine path; every family with a manifest has a calibrated tolerance file | G2 | `arch_golden_smoke_*`, `parity-nightly` |
| A4 | Offline ASR — structural | Corpus WER ≤ 10.0% (unchanged tripwire) | G1 | `asr_e2e_wer_test` |
| A5 | Offline ASR — accuracy | Total edits ≤ recorded baseline (currently 1 / 69 words) | G3 | `asr_e2e_edits_test` |
| A6 | Offline ASR — corpus | ≥ 500 reference words, corpus WER ≤ 1.10× recorded baseline | G3 | `asr_corpus_wer_test` |
| A7 | Streaming ASR | Streamed-vs-offline divergence **= 0** words; streamed corpus WER ≤ offline + 0.5 pp | G3 | `asr_stream_text_wer_test` |
| A8 | TTS parity | Waveform MSE + perceptual check vs a pinned reference per family | G3 | `tts_parity_<family>` (new, Phase 14.3) |
| A9 | Diarization | DER ≤ recorded baseline on multi-talker fixtures | G3 | `diar_der_test` (new, Phase 14.4) |
| A10 | VRAM — multi-session | ≤ 50 MB delta per additional session **and** `SharedWeightRegistry::hit_count() >= N-1` | G4 | `shared_weight_vram_test` |
| A11 | Batch activation | Native batched path taken (not silently serialised) for every family declaring one | G4 | `batch_dispatch_test` |
| A12 | Memory safety | 0 raw ggml teardown calls in library code, whole `src/` tree | G0 | `lint_teardown` (`SRC_DIR=${CMAKE_SOURCE_DIR}/src`) |
| A13 | Exception containment | 100% of exported C entry points guarded; "non-OK ⇒ `*out == NULL`" in every ownership forwarder | G0 | `lint_api_guard`, `teardown_safety_unit` |
| A14 | Artifact count | Exactly **1** primary shared library + 2 compatibility import libraries | G0 | build inspection + `exports_match_header_test` |
| A15 | ABI compatibility | Legacy `audiocpp` and `transcribe` binaries run unmodified against `speech.dll` with byte-identical results | G1 | `abi_compat_test` |
| A16 | Binding sync | All four bindings regenerate with no diff; versions in sync | G1 | `generate.py --check`, `check_version_sync.py` |
| A17 | Family reachability | Every canonical family loadable from CLI, server, WebUI and the C ABI; every family has a schema-v2 spec | G1 | `family_registry_unit`, `cli_family_smoke` |
| A18 | Duplication | 1 frontend implementation, 1 tokenizer hub, 1 SAN-M, 1 conformer, 1 causal-LM, 1 codec hub | G1 | `frontend_contract_test` + a duplication lint over the mel-scale fingerprint |
| A19 | Build time | Clean CUDA ≤ 4.0 min (auto-arch + ccache); incremental ≤ 10 s | G5 | CI timing, tracked per commit |
| A20 | Upstream health | `upstream-merge-dryrun` conflict count not rising from fork-side renames | G0 | nightly CI job |
| A21 *(v6.0)* | Input limits | Every ASR family reports `max_audio_ms`; no family truncates silently — a > limit clip yields `truncated = true` / `OUTPUT_TRUNCATED` or `INPUT_TOO_LONG`, never a clean-looking partial | G1 | `asr_limits_contract_test` (new) over every registered ASR family |
| A22 *(v6.0)* | Product registration | Every `src/models/<f>` with a loader factory is in `audiocpp_add_model` and in a composite; every family reachable from CLI `--list-loaders` | G1 | `family_registry_unit` extended; `cli_family_smoke` |
| A23 *(v6.0)* | Concurrency | N sessions on one model run concurrently with results byte-identical to serial; documented in `speech.h` | G1/G4 | `concurrent_sessions_test` (new); `shared_weight_vram_test` |
| A24 *(v6.0)* | ASR layer dedup | 1 encoder-decoder KV cache, 1 AR greedy driver, 1 transducer driver, 1 CTC driver, 1 VAD planner, 1 whisper encoder, 0 private loaders in `src/models/` | G1 | grep-based lint over `include/engine/models` + `src/models` (Appendix C) |

---

## 11. Risk Register

Risks R19–R28 in `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md` §5.1 remain in force. These are the risks specific to this plan, ordered by expected cost.

| id | Risk | Likelihood | Impact | Mitigation | Owner phase |
|---|---|---|---|---|---|
| **F1** | Silent accuracy regression during frontend/tokenizer unification, invisible to a 69-word gate with 6.9× headroom | **High** | **Critical** | Phase 7 recalibration (edits ≤ baseline, ≥500-word corpus) + `frontend_parity_test` at calibrated tolerance + `mel_unit` as a Phase-9 entry condition | 7, 9 |
| **F2** | Deleting the wrong implementation of an overlapping family (v4 would have deleted 8 kLOC / 11 Parakeet variants) | **High** (v4 would have) | **Critical** | Evidence bake-off (10.0) before any deletion; V6 R3 fallback clause; Appendix B ledger | 10 |
| **F3** | The ported test suite fails in bulk and blocks the roadmap | High | Medium | Register in 4 batches; a documented failure with a root cause is an acceptable Phase-7 exit state, an undiagnosed one is not | 7 |
| **F4** | Upstream merge conflicts explode after reorganisation | Medium | High | L3 (additive-only in upstream territory); replace bodies not files; `upstream-merge-dryrun` CI | 9 (all) |
| **F5** | `StreamingSessionBase` destabilises the 18 existing streaming sessions | Medium | High | Feature flag; ASR families first (they have goldens), TTS last; one commit per session | 8 |
| **F6** | Whisper migration stalls (7.2 kLOC, 12 variants, legacy `.bin` path, most intricate decode loop) | Medium | High | Dedicated wave; `.bin` loader + its 3 unit tests migrate as a unit; arch copy retained one release cycle | 11 |
| **F7** | Activating `SharedWeightRegistry` surfaces latent aliasing bugs (a fingerprint mismatch must fall back, never share wrongly) | Medium | High | The registry already requires share key **and** per-tensor fingerprint match; add a negative test that two different models under the same key do **not** share | 7 |
| **F8** | Binary size / build time regress when `libspeech` embeds everything | Medium | Medium | Measure the Phase-7 baseline before flipping defaults; keep `AUDIOCPP_MODEL_SET` composites; A19 is a hard gate | 12 |
| **F9** | Tolerance files do not exist for engine-only families, so parity gates cannot be authored | Medium | Medium | Phase 14.2 promotes the calibration script out of `tmp/` into `scripts/calibrate_tolerances.py` and makes it part of the porting checklist | 14 |
| **F10** | Licence hygiene: Apache-2.0 (audio.cpp) absorbing MIT (transcribe.cpp) | Low | High | V6 R4 stands: keep Apache-2.0; preserve MIT headers and third-party notices on every merged file; `THIRD-PARTY-LICENSES.md` ported from `transcribe.cpp`; legal sign-off before 1.0 | 14 |
| **F11** | ggml patch drift — the 7 patches are the invariant, the tree is generated | Low | Critical | Never hand-edit `external/ggml/`; re-derive via `scripts/sync-ggml.sh`; a marker grep is **not** a sufficient audit — verify by regenerating and diffing | all |
| **F12** | Scope creep: "improve everything" turns each phase into an open-ended refactor | Medium | Medium | Every phase has explicit exit gates; work with no gate is not in the phase | all |
| **F13** *(v6.0)* | Porting families before the ASR layer exists multiplies duplication (measured: 3 ports → 3 KV caches, 3 decode loops, 1 re-implemented encoder, 1 private loader) | **High** (it happened) | High | Phase 11a before any further port; A24 lint; per-family procedure step 1 forbids private KV/decode/loader/mel | 11a |
| **F14** *(v6.0)* | The default-built `audiocpp.h` façade accretes consumers before `speech.h` exists, making the shim harder | Medium | High | Freeze `audiocpp.h` now (no new entry points); Phase 12 pulled forward; `abi_compat_test` | 12 |
| **F15** *(v6.0)* | Deleting `src/runtime/` removes the `TRANSCRIBE_DUMP_DIR` dump points `validate.py` depends on | Medium | High | 11c moves dump points to engine tracing before deletion; `validate.py cpp` gains an engine-path driver in 11a.7 | 11a, 11c |

---

## 12. End-to-End Pipeline Architectures

These are the capabilities the fusion exists to enable: multi-model pipelines in one process, with no IPC and no disk round-trip. Both require the unified session layer (Phase 8) and full family reachability (Phase 11).

### 12.1 Real-time voice-to-voice (ASR → LLM → zero-shot TTS)

```
  Microphone PCM (16 kHz mono, arbitrary chunk sizes)
        │
        ▼  StreamChunker rebuffers to the family's preferred_audio_chunk_samples
┌────────────────────────────────────┐
│ Voxtral-Realtime / Moonshine-Str.  │──► committed_text (append-only) + tentative_text
│ StreamingSessionBase: 4-state,     │    revision bumps on every observable change
│ STABLE_PREFIX, agreement_n = 3     │    zero allocation in the feed loop
└──────────────┬─────────────────────┘
               │  commit boundary → one utterance
               ▼
┌────────────────────────────────────┐
│ Causal audio-LLM text generation   │──► response text stream
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│ IndexTTS2 / MOSS / Qwen3-TTS       │──► acoustic codec tokens
│ (zero-shot, voice-conditioned)     │    weights shared via SharedWeightRegistry
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│ Mimi / MioCodec / Vocos decoder    │──► continuous PCM (24 / 48 kHz)
│ (codec hub, shared weights)        │
└────────────────────────────────────┘

  RunControl.poll_abort() is live at every stage: one request_abort() unwinds the
  whole pipeline deterministically.
```

### 12.2 Long-form multi-speaker diarized transcription

```
  Long-form audio (e.g. a 2-hour meeting)
        │
        ▼
┌──────────────────────────────┐
│ Silero VAD (native, embedded)│──► speech spans [start_ms, end_ms]
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ vad::plan — greedy chunker   │──► bounded utterances ≤ 30 s, 250 ms padding
└──────────────┬───────────────┘
               ├────────────────────────────────┐
               ▼                                ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│ Sortformer v2 diarization    │   │ Whisper / Parakeet / Qwen3    │
│ (streaming presets)          │   │ run_batch() — native batched  │
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               └───────────────┬─────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│ vad::offset_chunk_results + vad::rebuild_full_text                │
│ deterministic global timestamp and speaker-tag re-stitching       │
│ vad::rollback_to on abort — partial results stay consistent       │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
      Diarized transcript JSON / RTTM, millisecond precision,
      with per-segment decode telemetry (Phase 8.6)
```

---

## Appendix A — v4.0 → v5.0 phase mapping

See the table under §8. Summary: no v4 phase was dropped; two new phases (7, 8) were inserted ahead of them because v4's Phase 7 assumed prerequisites that do not exist in the tree.

## Appendix B — Deletion Ledger

**No file is deleted without a row here** (L7). Filled in as each deletion lands.

| # | Deleted path | LOC | Replaced by | Equivalence gate | Landed in | Revert commit |
|---|---|---:|---|---|---|---|
| B1 | `src/runtime/transcribe-mel.{cpp,h}` | 1,077 | `engine::audio` unified extractor | `frontend_parity_test`, `mel_unit` | Phase 9 | |
| B2 | `src/runtime/transcribe-kaldi-fbank.{cpp,h}` | 373 | unified extractor (`KaldiFbank` kind) | `frontend_parity_test` | Phase 9 | |
| B3 | `src/runtime/arch/gigaam/mel.{cpp,h}` | — | unified extractor (64-mel / 320-FFT) | `gigaam` golden smoke | Phase 11 W3 | |
| B4 | `src/runtime/transcribe-tokenizer.{cpp,h}` | 1,165 | `engine::text` tokenizer hub | `tokenizer_parity_test` | Phase 9 | |
| B5 | `src/runtime/sanm/sanm.{cpp,h}` | 395 | `framework/modules/speech_encoders/sanm.cpp` | `sensevoice` + `funasr_nano` goldens | Phase 10 | |
| B6 | `src/runtime/conformer/conformer.{cpp,h}` | 1,557 | `framework/modules/conformer_modules.cpp` | all conformer-family goldens | Phase 10 | |
| B7 | `src/runtime/granite_conformer/` | 230 | `framework/modules/attention/common_relative_attention.cpp` | `granite` + `granite_nar` goldens | Phase 11 W5 | |
| B8 | `src/runtime/causal_lm/causal_lm.{cpp,h}` | 1,347 | `framework/modules/transformers/` | affected family goldens | Phase 11 | |
| B9 | `src/community_models/parakeet_tdt/` | 2,382 | `src/models/parakeet_tdt/` (ported from arch) | 13 parakeet goldens | Phase 10.1 | |
| B10 | `src/runtime/arch/parakeet/` | 9,439 | `src/models/parakeet_tdt/` | 13 parakeet goldens | Phase 10.1 | |
| B11–B15 | `src/runtime/arch/{qwen3_asr, voxtral_realtime, funasr_nano, sensevoice, sortformer}/` | 16,896 | engine counterparts, feature-merged | per-family goldens | Phase 10 | |
| B16–B27 | `src/runtime/arch/{whisper, moonshine, moonshine_streaming, voxtral, canary, canary_qwen, cohere, gigaam, granite, granite_nar, medasr, moss}/` | 48,220 | `src/models/<family>/` (new directories) | per-family goldens + WER | Phase 11 | |
| B28 | `src/runtime/transcribe-arch-adapter.{cpp,h}` | 1,093 | *(nothing — the bridge is no longer needed)* | full suite green with no adapter | Phase 11c | |
| B29 *(v6.0)* | `src/runtime/transcribe-vad{,-integrate}.{cpp,h}` | ~500 | `audio/chunking` (`plan_vad_audio_chunks`, `append_chunk_speech_metadata`) | `vad_plan_unit`, `vad_merge_unit` re-pointed; `asr_e2e_*` with VAD on | Phase 11a | |
| B30 *(v6.0)* | `MoonshineKvCache`, `MoonshineStreamingKvCache`, `WhisperKvCache` in `include/engine/models/*/graphs_internal.h` | ~200 | `framework/asr/enc_dec_kv_cache` | the three `*_engine_smoke_test` gates at arch baseline | Phase 11a | |
| B31 *(v6.0)* | `models/whisper/assets.cpp` private `.bin` parser; `models/whisper/graphs.cpp` encoder | ~600 | `assets/whisper_bin_tensor_source`; `WhisperEmbeddingModule` | `whisper_engine_smoke_test`, `whisper_bin_*` units | Phase 11a | |
| B32 *(v6.0)* | `capi/src/audiocpp_capi.cpp` as a standalone implementation | 2,629 | `speech_capi.cpp`; `audiocpp.h` becomes inline forwarders | `abi_compat_test`, the 4 capi tests unchanged | Phase 12 | |
| B33 *(v6.0)* | everything else under `src/runtime/` (dispatcher, loader, model/session bases, backend, mel, tokenizer, unicode, batch-util, bin-loader, meta/env/debug/log/path, `causal_lm/`, `conformer/`, `granite_conformer/`, `sanm/`, `third_party/`) | ~20,000 | per the Phase 11c table | A2, A3, A12 | Phase 11c | |

v6.0 subtotal: the whole of `src/runtime/` (≈ 95 kLOC measured) plus the engine-side duplicates the first three ports introduced. **Every row above still shows an empty revert-commit cell as of 2026-08-26: zero deletions have been executed.** Phase 10.5 fills the first five.

Arch-tree subtotal: 9,439 + 16,896 + 48,220 = **74,555 LOC** — exactly `cat src/runtime/arch/*/*.cpp src/runtime/arch/*/*.h | wc -l`, i.e. the whole tree, with no family unaccounted for.

Total scheduled removal: **≈ 84,000 LOC**, every line covered by a named gate before it goes.

## Appendix C — Verification Command Reference

```bash
# --- Inventory -------------------------------------------------------------
ls src/models | wc -l ; ls src/community_models | wc -l ; ls src/runtime/arch | wc -l
ls model_specs/*.json | wc -l
grep -oP 'audiocpp_add_model\(\K[a-z0-9_]+' CMakeLists.txt | sort -u | wc -l

# --- Duplication -----------------------------------------------------------
grep -rln "2595\|1127\.0\|hz_to_mel\|mel_to_hz" --include=*.cpp --include=*.h src/

# --- Test / asset gap ------------------------------------------------------
for f in $(ls ../transcribe.cpp/tests/*.cpp ../transcribe.cpp/tests/*.c | xargs -n1 basename); do
  [ -f tests/$f ] || [ -f tests/unittests/$f ] || echo "MISSING $f"; done | wc -l
du -sh ../transcribe.cpp/tests/golden ; ls ../transcribe.cpp/tests/tolerances | wc -l

# --- Inert features (L4) ---------------------------------------------------
grep -rn ScopedWeightShareKey --include=*.cpp .            # expect: 0 call sites
grep -rn "run_batch(" capi/src app/ src/framework           # expect: 0 callers

# --- Discipline debt -------------------------------------------------------
grep -rn "ggml_backend_free\s*(\|ggml_backend_buffer_free\s*(\|ggml_backend_sched_free\s*(" \
     src/ --include=*.cpp --include=*.h | grep -v "^src/runtime/" | wc -l   # 167

# --- Reachability ----------------------------------------------------------
grep -rn "find_arch\|transcribe_model_load" app/            # expect: 0 hits
comm -23 <(ls model_specs | sed 's/\.json$//' | sort) \
         <(grep -oP 'audiocpp_add_model\(\K[a-z0-9_]+' CMakeLists.txt | sort -u)
# Reads as 9 rows; 6 are false positives that the family registry (§5.5) exists
# to eliminate:
#   bs_roformer, mel_band_roformer  -> ALIASES of the `roformer` target
#   htdemucs                        -> ALIAS of `demucs`
#   moss_tts_local, moss_tts_nano   -> ALIASES of `moss`
#   silero_vad                      -> registered via a bare add_library(), not
#                                      audiocpp_add_model() — itself an
#                                      inconsistency for §5.5 to normalise
# The three genuine orphans are: whisper, moonshine, moonshine_streaming.

# --- Frontend contract table ----------------------------------------------
for d in ../transcribe.cpp/tests/golden/*/; do
  f=$(ls "$d"*.manifest.json 2>/dev/null | head -1); [ -z "$f" ] && continue
  echo "$(basename "$d") :: $(grep -A10 '"frontend"' "$f" | tr -d ' \n')"; done

# --- Overlap evidence ------------------------------------------------------
for p in "community_models/parakeet_tdt:parakeet" "models/qwen3_asr:qwen3_asr" \
         "models/voxtral_realtime:voxtral_realtime" "models/fun_asr_nano:funasr_nano" \
         "community_models/sense_asr:sensevoice" "models/sortformer_diar:sortformer"; do
  e=${p%%:*}; a=${p##*:}
  echo "$a  engine=$(cat src/$e/*.cpp | wc -l)  arch=$(cat src/runtime/arch/$a/*.cpp | wc -l)" \
       " goldens=$(ls ../transcribe.cpp/tests/golden/$a/*.manifest.json 2>/dev/null | wc -l)"; done

# --- Gate thresholds -------------------------------------------------------
grep -n kDefaultMaxCorpusWerPct tests/asr_e2e_wer_test.cpp tests/asr_stream_text_wer_test.cpp
```

## Appendix D — Files `speech.cpp` should adopt from `transcribe.cpp`

Verified absent here, present there. All are small and high-leverage.

| Asset | Size | Value | Phase |
|---|---|---|---|
| `tests/*.cpp` `*.c` (51 TUs) | — | The entire regression suite for 74.5 kLOC of ASR | 7.1 |
| `tests/golden/**` | 66 files / **301 KB** | Per-family parity contracts: pinned HF repo + revision, reference impl + revision + entrypoint, frontend spec, tokenizer summary, capabilities, tolerance pointer, cases | 7.1 |
| `tests/tolerances/**` | 35 files | Per-tensor calibrated drift bounds (`enc.mel.in`, `enc.conv1.out`, …) with the calibration recipe recorded in-file | 7.1 |
| `docs/porting/families/*.md` | 18 files | The design record for each ported arch | 7.1 |
| `.github/workflows/clang-format.yml` | 1 file | Pinned formatter, CI-gated, correctly scoped past vendored trees | 7.2 |
| `AGENTS.md` | 99 lines | Command/automation conventions: `uv run` discipline, build target, formatter, validation entry points, C ABI exception doctrine, git hygiene | 7.2 |
| `docs/{extension-kinds,environment-variables,input-limits,model-family-testing,bindings}.md` | 5 files | Documents the contracts this plan is preserving | 7.1 |
| `bindings/**` | 152 files | Four mature bindings + an AST-driven, CI-gated, two-target generator | 13 |
| `RELEASING.md`, `THIRD-PARTY-LICENSES.md`, `PASSOVER.md` | 3 files | Release process and licence hygiene (F10) | 14 |

## Appendix E — What each parent teaches the other

The Master Fusion Principle, itemised. Every row is a concrete, scheduled transfer.

**`transcribe.cpp` → `audio.cpp` (discipline)**

| Lesson | Concrete transfer | Phase |
|---|---|---|
| Streaming is a dispatcher invariant, not a per-family concern | `StreamingSessionBase`: 4-state lifecycle, revision counter, commitment policy, pre-clear validate | 8.1 |
| Cancellation must reach the inner loop | `RunControl::poll_abort()` in feed, batch and graph loops | 8.2 |
| ABI evolution needs `struct_size`, not a JSON blob | Size-aware structs + `speech_abi_struct_size()` + `*_init()` | 12.1 |
| Teardown safety is a lint, not a review comment | `lint_teardown` over the whole `src/` tree (167 sites) | 8.4 |
| Exception containment is a lint, not a convention | `api_guard_*` + `lint_api_guard` (~26 unguarded entry points) | 8.5 |
| Numerical claims need a pinned oracle and calibrated tolerances | Golden manifests + tolerance files for every family, incl. TTS | 7.1, 14.2, 14.3 |
| Bindings should be generated from the header, not hand-written | One libclang IR → Python, TypeScript, Rust, Swift; `--check` in CI | 13 |
| Decoding quality needs telemetry | Per-segment temperature tier, compression ratio, avg logprob, no-speech probability, fallback count | 8.6 |

**`audio.cpp` → `transcribe.cpp` (breadth and product)**

| Lesson | Concrete transfer | Phase |
|---|---|---|
| A model catalog is part of the product | `model_specs/*.json` schema v2 for all 18 arch families; package manager installs them | 8.7, 11 |
| A library without a CLI/server/WebUI is a component, not a product | All 18 families reachable from every product surface | 11 |
| Models are more than ASR | 14 task kinds, artifacts, voice conditioning, named multi-output audio | already; preserved in 12 |
| Weights should be shared process-wide | `SharedWeightRegistry` active at every load+session site — including ASR sessions, which transcribe never had | 7.4 |
| Graph arenas should be reused | `ggml_gallocr` topological reuse applied to migrated arch families | 11 |
| Batched decode belongs on the public surface | `speech_run_asr_batch`, CLI `--batch`, server batch endpoint | 7.5, 12.1 |
| Progress reporting is a user-facing feature | `ProgressCallback` extended to streaming and to every migrated family | 8.2 |

**Neither parent had (emergent capabilities)**

| Capability | Why it needs both |
|---|---|
| Single-process real-time voice-to-voice | transcribe's streaming discipline + audio.cpp's TTS and codec breadth (§12.1) |
| Diarized long-form transcription with per-segment decode telemetry | transcribe's Sortformer + Whisper telemetry + audio.cpp's VAD planner and re-stitching (§12.2) |
| One catalog describing both *how to install* and *how to verify* a model | audio.cpp's `model_specs` schema + transcribe's golden-manifest parity contract, fused into model-spec schema v2 (Phase 9 · 9.1) |
| Four generated bindings over a 14-task surface | transcribe's generator + audio.cpp's task breadth |
| A cross-family duplication lint | Only meaningful once both trees are one tree |

---

## 13. Summary

When this roadmap is executed, `speech.cpp` will be:

- **Complete** — TTS, voice cloning, STT/ASR, streaming ASR, VAD, diarization, separation, alignment, enhancement, super-resolution, voice conversion, singing-voice conversion, neural codecs and MIDI, in one binary, all reachable from CLI, server, WebUI, C ABI and four generated bindings.
- **Non-redundant** — one frontend derived from a written contract, one tokenizer hub with encode *and* decode, one SAN-M, one conformer, one causal-LM, one codec hub, one canonical implementation per model family, one canonical id per family. ≈ 84,000 LOC removed, every line gated before removal.
- **Verified** — 66 golden manifests, 35+ calibrated tolerance files, a five-class gate ladder, accuracy gates with real statistical power, and activation gates that make inert features impossible.
- **Disciplined** — no exception escapes a C entry point, no raw ggml teardown in library code, no ABI break without a `struct_size` bump, no binding drift, all lint-enforced.
- **Efficient** — process-wide weight sharing that is *measured*, topological arena reuse, fused SwiGLU and packed projections, direct depthwise convolution, one pinned ggml across seven backends.
- **Maintainable** — a live upstream that still merges cleanly, because the fork adds where upstream owns and restructures only what the fork owns.

The measure of success is not that the two codebases became one. It is that the merged system is stronger than either parent on **both** axes at once: as capable as `audio.cpp` and as rigorous as `transcribe.cpp` — which is exactly what the Master Fusion Principle asks for.

---

*Document version 5.0 · audited against `c776b81` · 2026-08-23. Every quantitative claim in §2 is reproducible with Appendix C.*

## Appendix F — V6 Decisions Superseded *(v6.0)*

`TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md` §0.3 predates the engine-spine architecture that v5 adopted and the tree implemented (Phases 8–11). These decisions were never formally reconciled. They are superseded as follows; V6 §0.0 records the same in R14.

| V6 decision | Said | Superseded by |
|---|---|---|
| **D2** | `Arch` (function-pointer table) is the unified dispatch; `ArchAdapter` wraps the engine vtable | The engine session contract (`IVoiceTaskSession` + `StreamingSessionBase`) is the dispatch; `ArchAdapter` is transitional and deleted in 11c |
| **D3** | transcribe `Loader` handles ABI-path model loading | `ModelRegistry` + `TensorSource` for every path (§5.8) |
| **D4** | lightweight `transcribe_session` for the ABI path, `RuntimeSessionBase` for C++ | one session model — `RuntimeSessionBase`; the ABI is thin over it (Phase 12) |
| **D14** | transcribe `MelFrontend` / `KaldiFbankFrontend` as the unified STT frontend | engine `MelExtractor` / `kaldi_fbank` (Phase 9, done) |
| **D15** | transcribe `causal_lm` as the shared STT backbone | framework transformer modules + `causal_lm_ops` (Phase 10, done) |
| **D18** | generalised ABI via `transcribe_task_*` entry points | `speech_*` in `speech.h` (§6) |
| **D19** | `transcribe-arch.cpp` registry accumulates adapters | `family_registry` + `ModelRegistry` (§5.5) |
| **D23** | C ABI symbols stay `transcribe_*` | `speech_*`; `transcribe_*` and `audiocpp_*` survive as compat shims (§6.4), so existing bindings' `transcribe.abihash` checks keep passing through the shim until Phase 13 retargets them |

Still in force from V6: D1, D5–D13, D16 (pin now `36da5713`), D17, D20–D22, D24, D25; R1–R13.
