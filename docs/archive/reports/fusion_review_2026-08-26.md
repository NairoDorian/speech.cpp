# Fusion Review — `audio.cpp` + `transcribe.cpp` → `speech.cpp`, reconsidered from the code

> **Date**: 2026-08-26 · **Tree**: `5e1a7e5` (`main`, 66 ahead of `upstream/audio.cpp@c79e588`, 0 behind)
> **Parents read at**: `audio.cpp@c79e588`, `transcribe.cpp@2102bca`
> **Purpose**: An independent architectural review of the merge, formed from the parents' *code* first and only then compared against the existing plans. It produces `FUSION_ROADMAP_PLAN.md` v6.0.

---

## 0. Method, and a bias disclosure

The brief was to reconsider the fusion architecture as if the existing `speech.cpp` plan documents had never been read, then compare. Honest caveat: by the time this review started, `progress.md`, the tracker and fragments of the roadmap had already been read during the session. What was deliberately **not** read until §5 below: `FUSION_ROADMAP_PLAN.md` §1–§12 in full and `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md` at all.

Order of study:

1. `transcribe.cpp` — public header, `Arch` trait, session/model bases, dispatcher, shared modules, loader, backend plan, bindings, tests, validation scripts, porting docs, `PASSOVER.md`.
2. `audio.cpp` (upstream, not the fork) — session/model/registry contracts, session base, KV cache and decode runtimes, module library, spec/catalog system, apps, tests, contributing discipline.
3. `speech.cpp` — the tree as built: every fusion commit, the framework delta vs upstream, the two C ABIs, the duplication census.
4. Only then: the existing plans, section by section, against the findings.

Every number below has a command behind it (Appendix). Where this review finds fault with work done earlier **in this same session** (the three family ports), it says so.

---

## 1. Parent A — `transcribe.cpp`, what the code says it is

**Identity**: a *library*. One-header C ABI (`include/transcribe.h`, 2,665 lines, **98** exported functions), six language bindings (python, python-native, python-native-cu12, rust, swift, typescript — 18.7 kLOC), a CLI, bench and quantize tools. ASR and diarization only: 18 families, 74.5 kLOC under `src/arch/`.

### 1.1 What it does exceptionally well

| Asset | Evidence |
|---|---|
| **The C ABI contract** | Every caller-owned struct carries `struct_size` as field 0 and is stamped by a `*_init()`; kind-tagged family extensions (`struct transcribe_ext`, FourCC registry in `docs/extension-kinds.md`, per-slot `accepts_ext_kind` probe); a written threading contract; exception containment via `api_guard_status/value/void` on every entry point; result-pointer lifetime rules; `transcribe.abihash` gates all bindings; `contract.json` + `transcribe-link.json` for non-CMake consumers. |
| **Streaming semantics** | Dispatcher-owned 4-state lifecycle (`IDLE/ACTIVE/FINISHED/FAILED`), monotonic revision, `committed_text` **append-only** with `AUTO / ON_FINALIZE / STABLE_PREFIX` policies, `stream_validate` / `run_validate` **pre-clear** validation so a caller typo cannot destroy the prior transcript. |
| **Input-limits doctrine** (`docs/input-limits.md`) | Three buckets (chunked/unbounded, hard-cap, soft-window); `caps.max_audio_ms` derived from real metadata; `transcribe_session_get_limits()`; **never truncates silently** — `INPUT_TOO_LONG` up front or `OUTPUT_TRUNCATED` + `was_truncated()` after. |
| **Validation methodology** | Per family: `docs/porting/families/<f>.md`, intake schema, `scripts/dump_reference_<f>_*.py` (18), `compare_tensors.py` against `tests/tolerances/<f>.json`, `validate.py all --family <f>` orchestrating ref→cpp→compare, `preflight.py` for cheap metadata gates, golden manifests (19 dirs) pinning HF repo + revision + reference script. Test contract in `docs/model-family-testing.md`: fixture smoke → real-model smoke → e2e transcript smoke (edit-distance budget) → numerical gate → benchmark gate; skip code 77 only for honest missing assets. |
| **Engineering culture** (`PASSOVER.md`) | Interleaved-arm benchmarking, drop-first-run, revert anything inside the noise floor, kill-switch env vars for every optimization, "read the code — every defect was invisible on the machine it was written on", `cleanup_gpu()` explicitly off-limits. |
| **Batch API with serial fallback** | `Arch::run_batch` optional; the dispatcher falls back to per-utterance `run()`, so every family supports the batch API regardless. Per-utterance isolation; abort pads the tail. |

### 1.2 Where the code is weaker than its contracts

- **Monolithic families.** `arch/whisper/model.cpp` is 3,502 lines; each family owns its runner loop. **15 of 18** `model.cpp` files hand-roll an argmax decode loop. RNN-T/TDT greedy exists in `gigaam`, `parakeet` and `sortformer` separately.
- **Five field-identical KV caches.** `WhisperKvCache`, `MoonshineKvCache`, `MoonshineStreamingKvCache`, `CanaryKvCache`, `CohereKvCache` all carry `{self_k, self_v, cross_k, cross_v, n_ctx, n, head, T_enc, n_batch, cross_populated}` (Whisper adds `T_enc_pad`). Plus `causal_lm::KvCache` for decoder-only families.
- **Concurrency**: the header states it — at most one `run`/stream in flight per *model*, because sessions share the model's backend instances.
- **Scope**: no TTS, VC, separation, alignment, codec, music, MIDI.
- **The `Arch` trait is a C-style function-pointer table**: simple and ABI-shaped, but not composable — no base-class reuse between families beyond free-function helpers.

---

## 2. Parent B — `audio.cpp`, what the code says it is

**Identity**: a *framework plus product*. C++ virtual-interface contracts (`IVoiceModelLoader → ILoadedVoiceModel → IVoiceTaskSession`), a 60-header module library, a JSON model-spec catalog with package management, and shipped apps: CLI (988 lines, ~40 flags), OpenAI-compatible HTTP server (16 endpoints incl. `/v1/audio/transcriptions/live`), native Svelte WebUI, JSON workflow pipelines, model manager, streaming driver. 45 core + 12 community families across **14 task kinds**, ~220 kLOC.

**Upstream `audio.cpp` has no `capi/` directory and no `bindings/`.** The C ABI in `speech.cpp` was added by the fusion, not inherited.

### 2.1 What it does exceptionally well

| Asset | Evidence |
|---|---|
| **Task breadth on one contract** | `TaskRequest` / `TaskResult` / `StreamEvent` cover text, audio, voice conditioning, artifacts, named multi-output audio, speech segments, speaker turns, word timestamps — one session interface for 14 tasks. |
| **A real module library** | `modules/`: attention (SDPA, GQA, relative, Longformer, cross, streaming frame-attention cache), conformer, zipformer, SAN-M, wav2vec2-BERT, HuBERT, WavLM, CAMPPlus, **`WhisperEmbeddingModule`** (the full Whisper encoder as a module), BigVGAN/HiFT vocoders, ECAPA/TitaNet speaker encoders, RMVPE pitch, T5 text encoders, Qwen/Gemma decoders, NeMo NanoCodec. |
| **Decode runtimes** | `TransformerKVCache` / `TransformerBatchedKVCache` with `import_state` / `export_state` / `retain_prefix`; `CachedDecodeRuntime<Graph, Policy>`; `BoundedStaticKVDecodePolicy`; `GraphCapacityController` (Fixed/Tiered/Grow/Double); `tdt_decoder_runner` (three greedy variants). |
| **Weights and packaging** | `TensorSource` (GGUF / safetensors / torch-bin) → `BackendWeightStore` with storage-type selection; `ModelContract` validates every option key against the spec; `spec_backed_model` template; `audiocpp_gguf` embeds sidecars and *fails* if it cannot; `model_specs/*.json` schema v1 + HF download repo. |
| **Audio framework** | `audio/chunking.h`: Fixed / QuietEnergy / **VAD** chunk planners, overlap-add with triangular/linear windows, word-timestamp and speech-metadata re-stitching across chunks; `kaldi_fbank` with LFR + CMVN; DSP/FFT/iSTFT; DeepFilterNet2, RNNoise, FlashSR, ZipEnhancer. |
| **Per-session execution** | `RuntimeSessionBase` owns an `ExecutionContext` (its own backend), `ArtifactStore`, `RuntimeCache`, `RuntimeWorkspace`, `GraphExecutor`. Sessions are independent by construction — the concurrency limitation transcribe documents does not exist here. |
| **Product discipline** | `docs/maintainers/loader_and_catalog.md`: loader registry and package catalog must agree, checked by `tools/check_loader_catalog_sync.py`; PR-evidence culture in `CONTRIBUTING.md`; experimental modules land as `xxxExp` beside the original, never inside it. |

### 2.2 Where the code is weaker than its contracts

- **No streaming discipline in the base.** `IStreamingVoiceTaskSession` is `start/process/finalize/reset` and nothing else — no state, no revision, no committed/tentative split, no abort polling. `qwen3_asr` carries eleven private `streaming_*` members to fill the gap; every streaming family re-invents it.
- **Error model is exceptions all the way down** (`CapacityError`, `std::runtime_error`). Fine inside C++; a C ABI must contain them at every boundary.
- **Single-backend graphs.** `GraphExecutor` is one `ggml_gallocr` on the session's one backend. There is no `ggml_backend_sched` path, so an op the GPU backend lacks fails instead of falling back to CPU — transcribe's `BackendPlan.scheduler_list` handles exactly that.
- **No input-limits contract.** `AudioPreparationContract.max_input_samples` exists; nothing derives it from model metadata, `CapabilitySet` has no `max_audio_ms`, `TaskResult` has no truncation flag.
- **Validation is per-PR evidence, not a convention.** 65 kLOC of tests organised per family, but no `validate.py`-style ref→cpp→compare pipeline, no per-tensor tolerance files, no golden manifests pinning reference revisions.
- **Raw ggml teardown everywhere**: 170 raw `ggml_backend_*_free` call sites in 78 files outside `src/runtime/`.

---

## 3. The child today — what `speech.cpp` actually is

Measured, not from the phase table:

| Region | LOC | What it is |
|---|---:|---|
| `src/models/` + `src/community_models/` | 190 k | audio.cpp's families **+ 3 migrated** (`moonshine`, `moonshine_streaming`, `whisper`) |
| `src/framework/` | 61 k | audio.cpp framework **+ 2,747 lines added** by the fusion (`MelExtractor`, `TokenizerHub`, `StreamingSessionBase`, `RunControl`, `StreamChunker`, `SharedWeightRegistry`, `family_registry`, `shaw_attention`, `causal_lm_ops`, embedded assets) **+ 566 modified** |
| `src/runtime/` | 95 k | the **entire** transcribe.cpp runtime copied in (18 arch dirs, dispatcher, loader, model/session bases, mel, tokenizer, unicode, backend, batch-util, bin-loader) **+** fusion-added `transcribe-vad*` and a 1,093-line `ArchAdapter` bridge — built only when `SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=ON` (**default OFF**) |
| `capi/` | 3.8 k | a **third C ABI**, `audiocpp.h` — 55 functions, opaque handles, heap-returned results, **zero `struct_size` usage, no extension kinds, no ABI hash** — built by default (`AUDIOCPP_BUILD_CAPI=ON`) |
| `include/transcribe/` | 3.2 k | transcribe.h kept verbatim, built only with `SPEECHCPP_ENABLE_UNIFIED_ABI=ON` (**default OFF**) |
| `bindings/` | — | **does not exist** |
| `tests/transcribe/` | 54 files | ported; **29** registered as CTest targets |
| `docs/porting/` | 8 stages + **21** family docs | ported |
| `patches/ggml/` | 7 | the fork-delta invariant over ggml `36da5713` (0.22.0) |

### 3.1 The duplication census

| Subsystem | Copies in the tree today |
|---|---|
| Encoder-decoder KV cache | **8**: 5 arch structs + 3 engine-side copies (in the migrated packages) |
| Decoder-only KV cache | `causal_lm::KvCache` + `TransformerKVCache` / `TransformerBatchedKVCache` |
| Greedy AR decode loop | 15 arch `model.cpp` + 3 engine packages, each private |
| RNN-T / TDT greedy | `gigaam`, `parakeet`, `sortformer` arch + engine `tdt_decoder_runner` ×3 variants |
| Mel / frontend | `transcribe-mel` · `audio/mel_extractor` · `modules/whisper_frontend` + `whisper_embedding` · per-arch filterbank slots |
| Whisper encoder | `arch/whisper/encoder.cpp` · `modules::WhisperEmbeddingModule` (upstream, complete) · `models/whisper/graphs.cpp` (**re-implemented by W2a this session**) |
| VAD chunk planning | `audio/chunking.h::plan_vad_audio_chunks` (upstream) · `transcribe-vad.cpp::vad::plan` — whose own comment reads *"Direct port of audio.cpp's plan_vad_audio_chunks"* |
| Tokenizer | `transcribe-tokenizer` · `text/tokenizer_hub` · `tokenizers/{hf_tokenizer_json, llama_bpe, sentencepiece}` |
| Weight loading | transcribe `Loader` · `transcribe-bin-loader` · engine `TensorSource` · **a private `.bin` parser in `models/whisper/assets.cpp`** (W2a) |
| Conformer | `runtime/conformer/` · `modules/conformer_modules` |
| Causal LM | `runtime/causal_lm/` · `modules/transformers/{qwen,gemma}_decoder` · `causal_lm_ops` |
| Backend selection | transcribe `BackendPlan` (scheduler list, CPU fallback) · engine `ExecutionContext` (single backend) |
| Public C ABI | `transcribe.h` (98 fns, OFF) · `audiocpp.h` (55 fns, ON) |

**Deletions executed to date: zero.** Appendix B of the roadmap schedules ≈84 kLOC for removal; every "Revert commit" cell is empty; every file it names still exists.

### 3.2 Defects found, including in this session's own work

1. **Phase 10's central deliverable was never executed.** The bake-off decided five engine-wins on the condition *"the loser's distinguishing features are merged into the winner before retirement."* Measured: `src/models/qwen3_asr` has **0** references to speculative decoding; `voxtral_realtime` **0** to cache-aware windows; `sortformer_diar` **0** to streaming presets; `fun_asr_nano` has **no** WER gate. The verdicts exist; the merges do not. The tracker marks Phase 10 `[x] DONE`.
2. **The "third façade" the roadmap warns against already ships by default.** Roadmap v5 Phase 12: *"a unified ABI over two parallel model layers would just be a third façade."* `capi/audiocpp.h` is that façade — built by default, with a batch API added in 7.5, four tests, and none of transcribe's discipline. Nothing in `app/` consumes it; only tests do.
3. **W2a silently truncates audio > 30 s.** `WhisperRuntime::transcribe` trims to `fe_n_samples` and sets `truncated = true`; `WhisperSession::run` never reads the flag, and `TaskResult` has nowhere to carry it. This violates both parents: transcribe's *never truncate silently*, and the arch whisper's long-form seek. It is the direct consequence of the engine having no limits contract (§2.2).
4. **W1a/W1b/W2a each carry a private KV cache and a private decode loop** — reproducing transcribe's monolithic pattern inside the engine instead of converging on `TransformerKVCache`. W2a also re-implemented the Whisper encoder that `WhisperEmbeddingModule` already provides, and wrote a private `.bin` parser instead of a `TensorSource` backend. Each port was numerically exact and gated at arch parity; each also added to the duplication it was meant to reduce.
5. **The whisper package was test-only until `5e1a7e5`.** A per-family gate that links package sources directly cannot detect a missing `audiocpp_add_model`. There is no gate for product registration.
6. **Both transcribe options default OFF**, so the default product build contains none of the 15 un-migrated families and not the mature ABI.

---

## 4. The architecture this review would build (formed before reading the plans)

**Principle**: neither parent's *code* is canonical; each parent's *contracts* are. Concretely, the merged system takes audio.cpp's **framework, breadth, packaging and product surface** and transcribe.cpp's **ABI contract, streaming and limits semantics, validation methodology and engineering culture** — and adds one thing neither parent has: a shared ASR runtime layer, so that family code shrinks to graphs + assets + a thin session.

```
┌────────────────────────────────────────────────────────────────────────┐
│ speech.h — ONE C ABI, transcribe.h's discipline generalised to 14 tasks │
│   size-aware structs · *_init() · ext kinds · status codes · api_guard  │
│   abihash → 6 generated bindings · compat shims: audiocpp.h, transcribe.h│
│   implemented as a THIN layer over the engine base classes — no second  │
│   dispatcher underneath                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ Product: CLI · server (+ committed/tentative live transcription) · WebUI│
│          · workflow · model manager · bench/quantize tools              │
├────────────────────────────────────────────────────────────────────────┤
│ Session layer (exists): RuntimeSessionBase · StreamingSessionBase       │
│   (lifecycle, revision, STABLE_PREFIX, pre-clear validate) · RunControl │
├────────────────────────────────────────────────────────────────────────┤
│ ASR runtime layer (NEW — the missing piece)                             │
│   EncDecKVCache (one, batched) · decode drivers: CTC greedy, RNN-T/TDT  │
│   greedy + batched joint, AR greedy w/ suppress masks + temperature      │
│   ladder + telemetry · AsrResult rows (tokens/words/segments/speakers,  │
│   cross-indexed, committed counts) · LimitsBasis → max_audio_ms,        │
│   truncation flag, INPUT_TOO_LONG · long-form driver over audio/chunking│
│   · feature bits · timings                                              │
├────────────────────────────────────────────────────────────────────────┤
│ Families: one implementation each = graphs + assets + thin session      │
├────────────────────────────────────────────────────────────────────────┤
│ Framework: modules · MelExtractor/KaldiFbank · TokenizerHub · chunking  │
│   · TensorSource (+ whisper .bin source) · BackendWeightStore + Shared  │
│   WeightRegistry · GraphExecutor (+ sched fallback path) · ggml + 7 pat.│
└────────────────────────────────────────────────────────────────────────┘
```

**Rules that follow**

- **One ABI, now.** Freeze `audiocpp.h` (no new entry points), author `speech.h` from `transcribe.h`, implement it thinly over the engine. Do this *before* the remaining family ports so every port lands under its final ABI once.
- **Delete `src/runtime/` entirely at the end** — dispatcher, loader, model/session bases, backend, mel, tokenizer, unicode, batch-util, bin-loader, VAD, adapter, all 18 arch dirs. Their responsibilities all have engine-side homes once the ASR layer exists. Keeping the transcribe dispatcher as the ABI implementation would freeze the `Arch` vtable as a permanent second session contract.
- **Build the ASR runtime layer first, then port.** Every remaining family should be graphs + assets + a thin session. Re-base the three existing ports onto it as the proof.
- **Carry the limits contract into the engine.** `CapabilitySet.max_audio_ms`, `TaskResult.truncated`, `CapacityError` → `INPUT_TOO_LONG`. Nothing truncates silently.
- **Execute the Phase-10 verdicts before any new port.** Five feature-merges, five deletions, ~17 kLOC — the cheapest real consolidation available, and it proves the ledger.
- **Methodology for audio.cpp's own families is a continuous track**, not a Phase-14 bullet: each phase brings N native families under `validate.py` + tolerances + goldens.
- **Both parents stay live.** The sync routine (`scripts/sync-deps.sh`) runs at every phase boundary; a dependency bump in either parent is ours.

---

## 5. Comparison with the existing plans

Read after §4 was written: `FUSION_ROADMAP_PLAN.md` v5.0 (2026-08-23) and `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md`.

### 5.1 Where they already agree with this review

The v5 roadmap is better than expected. It independently reached most of §4: engine as the spine; `speech.h` with `struct_size`, ext kinds and exception containment (it caught its own v4 regression, C10); bindings as a *retarget* of transcribe's six (C9); the parakeet reversal (C6); the Reciprocity Rule; the migration invariant with one-release coexistence; the deletion ledger; L1–L10; per-family porting stages; the gate ladder G0–G5 and the 6.9× headroom recalibration. V6's D9 (methodology for all families), D10 (streaming lifecycle universal), D7 (safe teardown everywhere) also match.

### 5.2 Where the tree drifted from the roadmap

| Roadmap says | Tree does |
|---|---|
| Phase 12 after Phase 11, to avoid a third façade | The third façade (`audiocpp.h`) shipped in Phase 5 and grew in 7.5; it is the default |
| Phase 10: merge loser features *before* retirement | Zero feature merges executed; zero retirements |
| A12: `lint_teardown` over the whole `src/` | Still scoped to `src/runtime/`; 170 sites remain |
| A17: every family reachable from CLI/server/WebUI/ABI | Whisper was test-only until this review |
| `SPEECHCPP_ENABLE_*` ON in CI (7.5) | Both OFF; `build-cpu-core` turns them on locally only |

### 5.3 Gaps in the roadmap this review adds

1. **No ASR runtime layer.** §4.3's target diagram has a session layer and a modules layer with nothing between them for ASR; §5.3 mentions porting KV windowing into causal-LM but not the encoder-decoder cross-KV case. The Phase-11 per-family procedure (*"assets.cpp, weights.cpp, runtime.cpp, encoder.cpp, decoder.cpp, session.cpp"*) institutionalises the monolithic pattern. Cost is now measured: three ports, three private KV caches, three private decode loops.
2. **No input-limits row** in the §4.2 contract table; no engine-side home for `max_audio_ms` / truncation. W2a's silent truncation is the consequence.
3. **No backend-execution row**: `BackendPlan.scheduler_list` (CPU fallback for unsupported ops) vs single-backend `GraphExecutor` is undiscussed.
4. **No concurrency contract**: the merged ABI should state the *better* rule the engine already provides (independent sessions, `SharedWeightRegistry` making them cheap), not transcribe's 0.x limitation.
5. **`transcribe-vad*` treated as canonical** (Phase 11 keeps it in `src/runtime/`; §4.3 names `vad::plan`). It is a verbatim port of `audio/chunking.cpp`, which is the richer of the two. Delete it.
6. **`.bin` loading placement**: F6 migrates the loader "as a unit" but not *as a `TensorSource`*, which is what removes the special-casing.
7. **`WhisperEmbeddingModule`** (upstream) is not in the §5 dedup program; W2a duplicated it.
8. **No product-registration gate** (the whisper defect class).
9. **Methodology for audio.cpp's families deferred to Phase 14** — the "audio.cpp learns from transcribe" half is last, not parallel.
10. **`PASSOVER.md` culture** (measurement methodology, off-limits list, kill switches) is absent from the doctrine.
11. **Dual parentage and the sync routine** (added this session) are not in §9.

### 5.4 The one direction this review reverses

Roadmap v5, Phase 11 "Decommission the bridge": after migration, `src/runtime/` **keeps** the dispatcher, backend, loader, model, session, VAD, meta, batch-util and bin-loader "as what the C ABI genuinely needs", and the `Arch` struct "survives as the shim's vtable."

This review says: **no.** The whole point of Phase 8 was to move the dispatcher's lifecycle discipline *into* `StreamingSessionBase`. Once every family is an engine package and the ASR layer exists, the transcribe dispatcher has no responsibility left that the engine base classes do not already own. Keeping it makes the C ABI a three-layer implementation (`speech_capi → transcribe.cpp dispatcher → Arch shim → engine session`) and freezes `transcribe_session`/`transcribe_model` as a permanent second contract. The correct end state is `speech_capi.cpp` **directly** over `RuntimeSessionBase` / `StreamingSessionBase` / `ModelRegistry`, and `src/runtime/` deleted in full — 95 kLOC, every line with a ledger row.

### 5.5 V6 decisions that are now superseded

V6 (2026-08-19) predates the v5 reversal to an engine-spine architecture and was never reconciled. The following are superseded by v5 §4/§6 and by this review; v6.0 of the roadmap records them explicitly in Appendix F:

- **D2** (`Arch` as unified dispatch, `ArchAdapter` wraps the vtable) → the engine session contract is the dispatch; the adapter is transitional and deleted.
- **D3** (transcribe `Loader` for ABI-path loading) → `ModelRegistry` + `TensorSource` for every path.
- **D4** (lightweight `transcribe_session` for the ABI path) → one session model, `RuntimeSessionBase`.
- **D14** (transcribe `MelFrontend` as the STT frontend) → engine `MelExtractor` (Phase 9, done).
- **D15** (transcribe `causal_lm` as the STT backbone) → framework transformer modules + `causal_lm_ops` (Phase 10, done).
- **D18** (`transcribe_task_*` generalised ABI) and **D23** (symbols stay `transcribe_*`) → `speech_*` in `speech.h` with `transcribe_*` and `audiocpp_*` as compat shims (v5 §6.4).
- **D19** (`transcribe-arch.cpp` registry accumulates adapters) → `family_registry` + `ModelRegistry`.

---

## 6. What changes in the plan (summary; the detail is roadmap v6.0)

| # | Change | Why |
|---|---|---|
| 1 | **Phase 10.5 (new, next)**: execute the five Phase-10 verdicts — feature-merge, then delete, with ledger rows | Cheapest real consolidation; proves the ledger; removes the D1 shadowing hazard |
| 2 | **Phase 11a (new)**: the ASR runtime layer; re-base `moonshine`, `moonshine_streaming`, `whisper` onto it; delete their three private KV caches and W2a's `.bin` parser; fold W2a's encoder onto `WhisperEmbeddingModule` | Stops the duplication multiplier before 12 more families pay it |
| 3 | **Phase 12 pulled forward** (before the remaining ports): `speech.h` from `transcribe.h`, thin over the engine; `audiocpp.h` frozen then shimmed | Every remaining port lands under its final ABI once |
| 4 | **Phase 11b**: remaining families as thin packages, each deleting its arch dir in the same wave | Measured cost per family drops |
| 5 | **Phase 11c**: delete `src/runtime/` in full | Reverses v5's "keep the dispatcher" |
| 6 | Laws **L11** (never truncate silently), **L12** (measure the real flow; off-limits list), **L13** (dual parentage; sync at every phase boundary) | Contracts from transcribe not yet in the doctrine |
| 7 | §4.2 gains rows for input limits, backend execution, concurrency, product registration | Contract gaps found by code |
| 8 | Methodology-parity for audio.cpp families becomes a per-phase quota, not Phase 14.3 | Both parents learn in parallel, as the Master Key demands |
| 9 | Acceptance criteria A21–A24; risks F13–F14; Appendix B rows for `transcribe-vad*`, the three engine KV copies, `audiocpp.h`, all of `src/runtime/` | Ledger completeness |

---

## Appendix — commands behind the numbers

```bash
# parents
wc -l ../transcribe.cpp/include/transcribe.h                      # 2665
grep -c 'TRANSCRIBE_API' ../transcribe.cpp/include/transcribe.h   # 98 (in speech.cpp's copy)
ls ../transcribe.cpp/bindings                                      # 6
for d in ../transcribe.cpp/src/arch/*/; do cat $d/*.cpp $d/*.h | wc -l; done | paste -sd+ | bc  # 74555
ls ../audio.cpp/capi ../audio.cpp/bindings                         # both absent upstream
find ../audio.cpp/include/engine/framework/modules -name '*.h' | wc -l   # 60
# child
git diff --stat --diff-filter=A upstream/main HEAD -- include/engine/framework src/framework | tail -1   # 2747 insertions
grep -c struct_size capi/include/audiocpp.h                       # 0
grep -o '^AUDIOCPP_API' capi/include/audiocpp.h | wc -l           # 55
grep -rln 'audiocpp.h"' app webui tools                           # (none) — only tests consume it
grep -rn 'struct [A-Za-z]*KvCache\b\|class Transformer[A-Za-z]*KVCache' --include=*.h src include | wc -l
grep -ln argmax src/runtime/arch/*/model.cpp | wc -l              # 15
grep -n 'Direct port of audio.cpp' src/runtime/transcribe-vad.cpp # the VAD duplication admission
grep -c 'spec_k_drafts\|speculative' src/models/qwen3_asr/*.cpp   # 0 — Phase-10 merge unexecuted
grep -n truncated src/models/whisper/session.cpp                  # (none) — W2a drops the flag
grep -rn 'ggml_backend_free\s*(\|ggml_backend_buffer_free\s*(\|ggml_backend_sched_free\s*(' src --include=*.cpp --include=*.h | grep -v '^src/runtime/' | wc -l   # 170
for f in src/runtime/transcribe-mel.cpp src/runtime/transcribe-arch-adapter.cpp src/runtime/arch/parakeet; do [ -e $f ] && echo "STILL PRESENT $f"; done   # all present — 0 deletions
```
