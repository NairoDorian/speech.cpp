# Fusion rethink — three sources, one runtime (proposal for roadmap v7)

> **Date:** 2026-09-26 · **Status:** ADOPTED 2026-09-26 — the user took all six recommendations (recorded as V7-D1…V7-D6 in [`docs/LEDGER.md`](../LEDGER.md)); the executable plan is [`PLAN.md`](../../PLAN.md). This note keeps the reasoning.
> **Superseded in parts by later directives the same day.** Where this note differs from [`PLAN.md`](../../PLAN.md), **PLAN wins**:
> - V7-D7: two build profiles. DEV keeps the server, WebUI and CLI for testing; BUNDLE ships the library with only the chosen families.
> - FreeSpeech desktop (#1) and Android (#2) are the primary targets.
> - The strategy proposals SC1–SC7 are in PLAN §8.
> **Trees read (all freshly pulled today):**
> - `speech.cpp@d04fd552` plus two days of uncommitted work
> - `audio.cpp@955c8725`
> - `transcribe.cpp@145c96a6`
> - `audio_cunba@8f267029`
> - `transcribe_cunba@883beea`
> - `CrispASR@dfcdacae` (new clone, `../CrispASR`)
>
> **Inputs:** [`crispasr/crispasr_architecture_review_2026-09-26.md`](crispasr/crispasr_architecture_review_2026-09-26.md), [`crispasr/crispasr_model_catalog_2026-09-26.md`](crispasr/crispasr_model_catalog_2026-09-26.md), [`fusion_review_2026-08-26.md`](../archive/reports/fusion_review_2026-08-26.md) (the review that produced roadmap v6.0).
> **[inf]** marks opinion or inference; everything else was measured today.

---

## 0. Summary

1. **Keep the engine spine.**
   - audio.cpp's `IVoiceModelLoader → ILoadedVoiceModel → I*VoiceTaskSession` registry is the best architecture of the three sources.
   - CrispASR, with the most models, shows what happens without it: every backend implemented twice (CLI adapter plus 778 `CA_HAVE_` ABI arms), 86 private schedulers, 504 env knobs.
   - The v6 decision "engine is the spine, delete `src/runtime`" stands.
2. **Do not merge CrispASR's code. Mine it.**
   - Treat CrispASR as a **third reference parent**. It is a source of model knowledge: converters, tensor maps, 113 PyTorch reference-dump modules, 305 numerics lessons.
   - It is not a git parent. At ~25–60 commits a day, with no engine and an if-chain ABI, it cannot be tracked by merge, and its runtime code would import exactly the dual-path debt we are deleting.
   - It is MIT licensed, so porting individual functions is allowed wherever that is cheaper.
3. **The strategy needs correcting more than the architecture does.** Three facts from today's pull:
   - **Upstream audio.cpp is absorbing ASR itself.** It already has canary_asr, cohere_asr, moss_transcribe_diarize, nemotron, moonshine_asr chunking and R2T2, and it has **its own C ABI named `audiocpp.h`**.
   - speech.cpp now ships **two different `audiocpp.h`**: upstream's 536-line `include/audiocpp.h` and our 1216-line `capi/include/audiocpp.h`. They share one target name. This cannot survive continued merges.
   - The fork delta is growing, not shrinking: 128 commits ahead, framework +26 kLOC and CMake +2k lines since 08-23.
4. **Execution has drifted from the v6 plan.**
   - The six uncommitted 11b ports (gigaam, medasr, voxtral, canary_qwen, granite_speech, granite_nar) landed **before Phase 12 (`speech.h`)** and **without the 11a ASR layer**.
   - Three of them declare private `KvCache` structs (canary_qwen, granite_speech, voxtral). That is the exact duplication 11a exists to prevent.
5. **Dependency hygiene is red by our own law L13.**
   - ggml is 0.24.0; transcribe.cpp's ggml floor is **0.25.3** (`353b63b4`).
   - 24 transcribe.cpp commits have not been triaged (they include an error-code ABI change and a new TQ1_G128 quant type).
   - 21 audio.cpp commits are unmerged.
   - About two days of work are uncommitted.
6. **Scope target:** about 102 speech.cpp families plus about 60 CrispASR-only families, roughly 160 in total. It also adds **new task kinds** the engine lacks: text punctuation/truecasing, text LID, text MT, music analysis, watermarking.
   - These are the "and more" in "one runtime for STT, TTS and more".
   - Each new task kind needs a contract (L1) before any family lands in it.

---

## 1. Fresh-pull state (2026-09-26)

| Source | Before → now | Drift vs speech.cpp | Notes |
|---|---|---|---|
| audio.cpp | `487800f5 → 955c8725` (+109 files) | **21 behind** / 128 ahead (merge-base `57186a82`) | One of the 21 is `4d88768f`, a rewrite of the already-merged v0.8.2 `9bdd1d90`. No ggml changes in the range. New: MOSS-TTSD, ZipVoice streaming, peak-memory metrics, nemotron options moved into the spec, voxtral_realtime UTF-8 delta fix. |
| transcribe.cpp | `cf9c6013 → 145c96a6` (+245 files) | **24 commits** past watermark `0a67b65b`, none triaged | ggml 0.25.0 → **0.25.3**; ternary TQ1_G128 quant (patch 0003); CUDA decode KV trim (0004); parakeet-ultra/redux; CUDA graphs on by default; qwen3_asr packed QKV; **error-code ABI change `96b7c9c1`**. |
| audio_cunba | `8cf5136 → 8f267029` | 0 of the last 60 non-merge commits are unique | A mirror of audio.cpp about 5 days behind. **[inf]** It no longer adds reference value; consider dropping it from the workspace. |
| transcribe_cunba | `2345350 → 883beea` | unique: **sortformer push-audio streaming session (SFPS)**, build provenance, Rust dynload | SFPS is not yet verified against our `community_models/sortformer_diar/stream_schedule.cpp`. |
| CrispASR | new clone `dfcdacae` | not a git parent | See the two companion notes. |

---

## 2. Comparing the three sources

| | audio.cpp | transcribe.cpp | CrispASR |
|---|---|---|---|
| Families | ~100 (TTS-heavy, everything) | 19 ASR arches | ~131 backends (widest) |
| Dispatch | **one registry, typed sessions, model specs** | Arch vtable + loader | CLI adapter **and** a separate ABI if-chain |
| Shared runtime | framework (KV cache, decoders, codecs, tokenizer hub) | per-arch, strong ASR helpers | `src/core` headers; each model owns its scheduler, allocator and KV cache |
| C ABI | new opt-in `audiocpp.h` (#530/#544/#566) | **best discipline**: struct_size, ext slots, 4-state streaming, exception containment | 293 functions, additive only, string version |
| Parity culture | spec-backed, PASSOVER measurements | **pinned oracle, WER gates** | **stage diff harness plus reference GGUFs**, TTS → ASR round-trip, loose WER thresholds |
| Config | model_specs JSON | GGUF metadata | 504 env vars |
| Downloads | spec URLs | pinned test GGUFs | registry, **unpinned** `resolve/main`, licence gate |
| Velocity | ~485 commits since Aug 1 | ~130 since Aug 1 | ~1,290 since Aug 1 |
| What we take | the spine | the contracts (ABI, streaming, limits, parity) | the model knowledge, verification tooling and new task kinds |

"Master Key" restated for three parents: **audio.cpp contributes the skeleton, transcribe.cpp the contracts, CrispASR the catalogue.** No parent's code is canonical just because it exists.

---

## 3. Re-questioning the direction

### 3.1 How I would design it from scratch

Starting clean with today's knowledge [inf]:

1. **A small kernel first**, about 20–30 kLOC:
   - backend and device management;
   - weight store (mmap, TensorSource, foreign layout maps);
   - a scheduler/allocator owned by the engine, never by families;
   - a shared KV cache;
   - decode drivers (AR, transducer, CTC, NAR/CIF, flow/diffusion steps for TTS);
   - audio front-end (resample, mel, fbank, STFT);
   - tokenizer hub;
   - chunking and long-form;
   - streaming state machine;
   - RunControl and limits.
2. **One family contract**: `graphs + assets + spec + thin session`, with a capability bitmask. The contract forbids private caches, loops, loaders and mels (the v6 11a rule, enforced by grep gates from day one).
3. **One C ABI** (`speech.h`) designed with transcribe.h's discipline and generated bindings. The CLI, server and ABI are all thin clients of the same registry.
4. **Declarative family manifests** (`family.toml` or JSON next to the sources) that **generate** the CMake, alias tables, arch map and capability tables. There would be no 247 KB hand-written CMakeLists and no hand-maintained status docs.
5. **Parents are sources, not bases.** Each family has provenance (audio.cpp / transcribe.cpp / CrispASR / in-house), a pinned oracle and a parity gate. It is ported into the kernel once and re-synced by diffing the *source family* against its watermark.

### 3.2 Would I restart? No.

That design differs from speech.cpp in only two places: **point 5** (fork base vs sources) and **point 4** (hand-written build and docs). Everything else is already the v6 target. Restarting would throw away:
- ~100 working families;
- 121 + 115 green tests;
- the ggml patch stack;
- the merge channel to audio.cpp, the fastest-moving source.

**Keep the fork and correct the strategy:**

- **Fork base → upstream-first.**
  - For every ASR family, first check whether audio.cpp has it or is building it (it now builds ASR natively).
  - Where our work is generic (transcribe's contracts, `framework/asr`, AsrLimits, parity tests), **offer it upstream**, so the fork delta shrinks at each merge instead of growing.
  - speech.cpp's permanent delta should become: `speech.h` plus bindings, the transcribe-derived families audio.cpp does not want, the CrispASR-derived families, and the three-parent verification tooling.
- **Hand-written → generated.**
  - Family manifest → CMake/registry/capability tables.
  - `scripts/status.sh` → the status block (ctest label counts, grep gates, `ls src/runtime/arch`, drift numbers).
  - Humans write only decisions and lessons.

### 3.3 Is the current phase order right? Partly.

v6 said **11a → 12 → 11b** so that every port lands once under its final ABI and on the shared ASR layer. The uncommitted batch inverted this. The options are:
- **(a) Honour the plan:** finish the 11a layer, refactor the 6 new ports onto `framework/asr` (removing the 3 private KvCaches), and do Phase 12 before any more 11b ports.
- **(b) Downgrade the goal to "re-host now, dedup later"** and say so explicitly.

**Recommendation: (a).**
- The six ports are fresh and small, so fixing them now costs little. Fixing 30 later costs a lot.
- CrispASR ports will add ~20 more ASR families. Without an enforced ASR layer we would recreate CrispASR's architecture inside our engine.

### 3.4 What I would stop doing

Each of these costs more than it protects [inf]:

1. **Two master plans.** V6 is 205 KB and roadmap v6.0 is 168 KB; Appendix F exists only to reconcile them. Merge them into **one short living architecture doc (< 40 KB) plus one ledger**. Archive the rest under `docs/archive/`.
2. **Hand-written status in four places.** Generate it.
3. **Per-family ceremony for small families.** The ArchAdapter row, verdict harness, parity test, ledger row and five doc updates for medasr/gigaam-sized families is too much. **Keep the parity test; drop the rest.**
4. **Carrying audio_cunba** in the workspace (0 unique commits).
5. **Letting uncommitted work pile up** on a main-only workflow: commit after each green suite.

---

## 4. CrispASR absorption strategy

### 4.1 Rules

- **C-R1. Knowledge, not runtime.** A CrispASR family enters speech.cpp as an **engine package** (graphs + assets + spec + thin session) on `framework/*` layers. Their `src/<model>.cpp` is read as a *specification*. Individual functions (G2P tables, tokenizer quirks, DSP) may be copied with attribution under MIT. No CrispASR scheduler, allocator, KV cache or `whisper_params` coupling may enter.
- **C-R2. Oracle is PyTorch, not CrispASR.** Parity is measured against the original model. CrispASR is a **second opinion**: for overlapping families, run a three-way check (ours / CrispASR / PyTorch) to find bugs in either implementation. Their lesson L352 applies here: a reference that shares your assumption cannot falsify it.
- **C-R3. Watermark and triage ledger.** Add `docs/upstream/crispasr_triage.md` with watermark `dfcdacae`. Triage **by family on demand** (when porting or refreshing a family), not commit by commit. At ~40 commits a day, a full commit triage would consume the project.
- **C-R4. No unpinned models.** Every spec that points at a `cstr/*` GGUF pins the revision plus sha256. This fixes their weakness, not copies it.
- **C-R5. Licences are metadata.** Every spec carries `license` and `commercial_use`, plus a pre-download gate (their `--accept-license` idea). NC models are never defaults.
- **C-R6. Contract first for new task kinds** (L1). Text post-processing, text LID, MT, music analysis and watermarking each get a `TaskKind`, request/result types and a capability bit **before** the first family lands.

### 4.2 What to take, in priority order

**Tier 0 — verification and infrastructure.** Highest leverage, no new families.

| # | Item | CrispASR source | Why |
|---|---|---|---|
| C0.1 | Stage diff harness: reference GGUF of named intermediate tensors plus a C++ diff (cos_min + **magnitude** + max_abs) | `tools/dump_reference.py`, `tools/reference_backends/*` (113), `examples/crispasr-diff` | Generalizes our per-family parity tests into one tool; 113 reference modules are ready-made oracles. |
| C0.2 | TTS → ASR round-trip WER gate | nightly `regression.yml` | We have **no TTS numeric gate** (Phase 14.3 is pending). This is a cheap proxy. |
| C0.3 | ggml patch audit | fork patches 1–6, #10, #15 | Cross-check our 15 patches during the ≥ 0.25.3 bump. |
| C0.4 | Lesson mining | 305-entry `LEARNINGS-INDEX.md` | Turn the ~25 transferable lessons into lint/grep gates or tests where possible (siglu swap, F32⊙F16 broadcast, cached-graph inputs, sub-Q8 tower carve-outs). |
| C0.5 | Quantizer carve-out policy | `crispasr-quantize` | Audio tower ≥ Q8 and `token_embd` F16 below Q8, as a per-family spec field. |
| C0.6 | Single arch → family table plus test | `core/arch_backend_map.h` | We have the registry; add a test that every converter-written arch string resolves. |

**Tier 1 — new task kinds** ("and more"). Each needs a contract first.
- **Text post-processing:** PCS (47 languages: punctuation, casing and segmentation in one model), fireredpunc, fullstop, truecasers. Exposed as an ASR post-step, and also usable standalone.
- **Language identification:** audio (ecapa-lid 107 languages, silero-lid, firered-lid) and text (CLD3, GlotLID).
- **Alignment generalized:** wav2vec2 ×13 and fastconformer ×19 aligners behind one `align` task.
- **Diarization pipeline:** foxnose (WeSpeaker + clustering, 7.3% DER), pyannote-seg, a speaker DB.
- **Text MT:** m2m100 and madlad, for speech → text → translation.
- **Music analysis:** crepe, beat-this, basic-pitch, onsets-and-frames, hft-transformer, piano-transcription, mt3, tabcnn, btc-chords (NC).
- **Watermark and provenance:** AudioSeal plus C2PA. This is also an EU AI Act concern for TTS output.

**Tier 2 — ASR families we lack**, ranked by value against cost [inf]:
1. **wav2vec2 / hubert / data2vec.** A classic CTC family that also covers the aligners.
2. **stt_xx fastconformer-CTC ×19 languages.** Probably just specs on our existing parakeet-CTC / citrinet path.
3. **kyutai-stt.** Streaming Mimi; Mimi may already be in our codecs.
4. **paraformer.** Tiny, fast, zh; adds the CIF decoder driver to `framework/asr`.
5. **omniasr.** 1600+ languages.
6. **firered-asr + firered-lid.** Port *with* a KV cache; CrispASR's decoder has none.
7. **glm-asr.**
8. **gemma4 audio.**
9. **xasr.** Zipformer2; reuse kroko_asr's stack.
10. **Later:** dolphin, mimo-asr, moss-audio, hojo, lfm2-audio, mini-omni2, moss-transcribe.
11. **Skip:** ark-asr (WIP), tiron (experimental).

**Tier 3 — TTS we lack:**
- zonos, dia, csm, orpheus (fix CUDA first), parler, melotts + openvoice2, speecht5, fastpitch, bark, tada, kugelaudio, bananamind, sidon (restoration).
- NC-licensed and opt-in only: voxtral-tts, raon.

**Tier 4 — variants of families we already have.** These are spec and alias work only: parakeet (reazonspeech-ja, quds-fa, orukeet, ultra/redux, which transcribe.cpp also just added), cohere ja/ar fine-tunes, qwen3 mega-asr / ja-anime, moonshine-de (NC), chatterbox nano/finnish/kartoffelbox, and the pocket-tts language packs.

**Tier 5 — optional foreign-GGUF compatibility.** Load `cstr/*` GGUFs through per-family layout maps (same mechanism as the transcribe GGUF layout), pinned per C-R4. This gives users ~344 pre-quantized downloads at zero conversion cost.

**Out of scope:** the vendored llama.cpp text-chat ABI (`crispasr_chat_*`). A text LLM runtime is llama.cpp's job; our remit is local speech and audio models. Speech-LLM decoders inside ASR and TTS families are in scope, as they are today.

---

## 5. Target architecture (v7, delta from v6)

```
            CLI · server (OpenAI /v1/audio/*, /v1/realtime, Wyoming) · bindings (py, rust, dart, go, …)
                                         │
                          speech.h  (the ONLY first-party C ABI; struct_size, ext slots,
                                     4-state streaming, exception containment, error codes)
                                         │           [audiocpp.h / transcribe.h → shims, then removed]
                          Engine registry: IVoiceModelLoader → ILoadedVoiceModel → I*TaskSession
                          TaskKinds: asr · tts · s2s · vc · vad · diar · align · lid(audio|text)
                                     · textpost(punc|case|seg) · mt · separation · enhance
                                     · music_analysis · codec · watermark · generation
                                         │
      framework/  asr (KV cache, AR/transducer/CTC/CIF/NAR drivers, AsrResult, AsrLimits, long-form)
                  tts (sampling, flow/diffusion steppers, vocoders, ref-voice cache)
                  audio (resample polyphase, mel Params, fbank, STFT, chunking) · tokenizer hub
                  weights (TensorSource + foreign layout maps) · sched/alloc owned by engine
                                         │
      families/   <family>/ graphs · assets · spec · session   + family manifest (generates build)
                  provenance: audio.cpp | transcribe.cpp | CrispASR | in-house   + pinned oracle
                                         │
      external/ggml   pin ≥ max(parents' floors) + tracked patch stack (--check gate)
```

**Changes from v6:**
- **New task kinds.** Each needs its contract first.
- **A `framework/tts` layer.** This is the TTS equivalent of 11a. 36 TTS families across three sources share sampling, ref-voice caching and vocoders.
- **Family manifests generate the build.**
- **Provenance plus oracle** is required metadata for every family.
- **The ABI decision** (§8, V7-D1).

---

## 6. Proposed sequence (replaces v6 §5 ordering if adopted)

| Phase | Content | Exit gate |
|---|---|---|
| **S0 — Stabilize** | Run the suites on the uncommitted batch and commit it. Merge audio.cpp's 21 commits (recorded merge; `4d88768f` is a no-op re-release). Triage transcribe.cpp's 24 commits. **Bump ggml to ≥ 0.25.3** with the patch audit, including CrispASR's patches (C0.3). Decide the SFPS streaming session from transcribe_cunba. | `rev-list` shows 0 behind; watermark advanced; `--check` green; core + asr-abi suites green. |
| **S1 — Decide and diet** | Take decisions V7-D1…D6 (§8). Merge the two plans into one doc under 40 KB plus a ledger. Add `scripts/status.sh`. Drop audio_cunba. | One plan file; status is generated. |
| **S2 — Enforce 11a** | Move the 6 new ports onto `framework/asr` and delete the 3 private KvCaches. Add the CIF/NAR driver slot. | `grep 'struct *KvCache' include/engine/models` shows only forward declarations of the shared type; the 6 parity tests still pass. |
| **S3 — speech.h (Phase 12)** | The ABI per D1. Bindings are generated. | ABI conformance tests; `audiocpp.h` / `transcribe.h` shims byte-exact. |
| **C0 — Verification tooling** | C0.1–C0.6. Create `crispasr_triage.md`. | Diff harness runs on 3 existing families; TTS round-trip gate covers ≥ 5 TTS families. |
| **11b / 11c** | Remaining arch verdicts and deletion of `src/runtime`, as in v6. | As in v6. |
| **C1 — New task-kind contracts** | TaskKind, request/result and capability bits for textpost, lid, align, mt, music_analysis, watermark. One reference family each: PCS, ecapa-lid, wav2vec2 aligner, m2m100, beat-this, AudioSeal. | One green parity test per kind. |
| **C2 — `framework/tts`** | Extract shared TTS pieces from the overlapping families. | ≥ 3 families migrated; round-trip gate green. |
| **C3… — Family waves** | Tier 2, then Tier 3, then Tier 4, in batches of 3–5. Each wave is upstream-first checked against audio.cpp and ends with a sync-deps run. | Per family: PyTorch parity plus WER/round-trip gate; no private cache, loop or loader. |
| **C-opt — Foreign GGUF** | Tier 5. | Pinned sha256 per spec. |

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| The scope (~160 families) exceeds what one maintainer plus agents can keep green across three moving parents | Upstream-first (shrinks the delta); family waves with gates; archive rather than port low-value or WIP families; generated status. |
| Upstream audio.cpp's `audiocpp.h` diverges further from ours with every merge | Decide V7-D1 in S1, before the next merge after S0. |
| The CrispASR code style leaks in through copy-paste | C-R1 plus grep gates (`getenv(` count, `ggml_backend_sched_new` outside `framework/`, private `KvCache`). |
| NC-licensed weights become defaults | C-R5 plus a spec lint. |
| The TTS round-trip gate hides prosody or voice-quality regressions | It is a floor, not a ceiling. Keep stage-diff parity per family as the primary gate. |
| Stage-diff cosine is scale-invariant | The harness always reports magnitude (their L1396). |

---

## 8. Decisions (all taken 2026-09-26 as recommended — ids V7-D1…V7-D6)

- **V7-D1 — The ABI.**
  - (a) `speech.h` is the only first-party ABI. Our `capi/include/audiocpp.h` is **retired, not shimmed**, and upstream's `include/audiocpp.h` is carried untouched as the parent's.
  - (b) Adopt upstream's `audiocpp.h` as the product ABI and contribute transcribe's discipline to it upstream.
  - *Recommendation:* **(a)**, and additionally offer the struct_size/ext-slot discipline upstream. Either way, the two same-named headers must stop coexisting.
- **V7-D2 — CrispASR's role.** Reference parent #3, mined by family (recommended), or a code parent tracked commit by commit (not recommended: velocity plus architecture).
- **V7-D3 — Upstream-first.** Should we proactively offer generic work (transcribe contracts, `framework/asr`, parity tooling, ports audio.cpp lacks) to audio.cpp to shrink the fork?
- **V7-D4 — Scope of "and more".** Which new task kinds are in: text MT? music analysis? watermarking? Text-chat LLM is recommended out.
- **V7-D5 — Non-commercial models.** Allowed as opt-in specs behind a licence gate (recommended), or excluded entirely.
- **V7-D6 — Process diet.** Merge V6 and roadmap v6.0 into one doc and generate status (recommended).

---

## 9. Evidence commands (re-runnable)

```bash
git -C speech.cpp rev-list --left-right --count HEAD...upstream/main      # 128 21
git -C transcribe.cpp log --oneline 0a67b65b..HEAD | wc -l                # 24
head -4 transcribe.cpp/ggml/UPSTREAM speech.cpp/external/ggml/UPSTREAM    # 0.25.3 vs 0.24.0
wc -l speech.cpp/include/audiocpp.h speech.cpp/capi/include/audiocpp.h   # 536 / 1216
grep -rlE 'struct +KvCache' speech.cpp/include/engine/models              # canary_qwen, granite_speech, voxtral
grep -c CA_HAVE_ CrispASR/src/crispasr_c_api.cpp                          # 778
grep -rhoE 'getenv\("[A-Z0-9_]+"' CrispASR/src CrispASR/examples | sort -u | wc -l   # 504
grep -c sha256 CrispASR/src/crispasr_model_registry.cpp                   # 0
```
