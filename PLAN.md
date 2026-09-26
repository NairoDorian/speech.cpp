# speech.cpp — the plan

> **Version:** 7.0 (checkpoint re-steer), 2026-09-26. **This is the only plan.**
> - It replaces `FUSION_ROADMAP_PLAN.md` v6.0, `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md`, `MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md` and `progress.md`, all now in [`docs/archive/`](docs/archive/).
>
> **Companion files. Keep this set small; do not add new plan files:**
>
> | File | Holds |
> |---|---|
> | [`docs/LEDGER.md`](docs/LEDGER.md) | Decisions (V7-D1…D6 and the V6 ones still in force), family migration state, deletion rows B1–B33, parent dispositions, risks |
> | [`LESSONS.md`](LESSONS.md) | Every hard-won lesson, one heading each |
> | [`docs/reports/checkpoint_2026-09-26.md`](docs/reports/checkpoint_2026-09-26.md) | The audit behind this version: what was wrong, the evidence, and the corrections |
> | [`AGENTS.md`](AGENTS.md) | How to work here: commands, sync routine, rules for agents |
> | `CHANGELOG.md` | What changed, by date. Session narrative goes here and nowhere else |
> | `scripts/status.sh` | **Generated** status. Never hand-write counts |
> | [`ARCHITECTURE.md`](ARCHITECTURE.md) | **Generated** map of the tree (repomix, pre-commit). Read it first |
> | [`docs/benchmarking.md`](docs/benchmarking.md) + [`docs/glossary.md`](docs/glossary.md) | Port acceptance protocol (R10), and the vocabulary / metrics / benchmarks it uses |

---

## North star: the end goal and the rules (canonical; every other doc points here)

**The end goal.** speech.cpp is **the one local C++/ggml speech runtime behind FreeSpeech**. It serves FreeSpeech desktop (#1, the successor of ZER0 and S2B2S) and FreeSpeech Android (#2). **Both are primary.**

It runs every local speech model those apps need:
- STT offline and streaming, with committed and tentative partials;
- TTS offline and streaming, voice cloning and voice design;
- VAD, turn detection, diarization, alignment and language identification;
- text post-processing, speech translation, enhancement and separation, music analysis.

It does this through **one C ABI (`speech.h`) and a Rust crate**, linked in-process. Every supported model is **verified by gates**, and the shipped library is **as small as the app's model choice**.

**Where it comes from** (the author's path, V7-D8):

```
S2B2S (app) → transcribe.cpp fork (most of the engineering time: well optimized) → ZER0 (app, latest)
          → speech.cpp (now) → FreeSpeech desktop + Android
```

speech.cpp is a new project based on **audio.cpp** (pure upstream; none of the author's work) and **NairoDorian/transcribe.cpp** (the author's own fork, where most of the work went).
- It must carry **all** the optimizations and latest models of **both**.
- Losing a transcribe.cpp optimization while moving a family onto the engine is a regression of the author's own work.

**The rules.** They are not optional. Each carries the id of its source. **First read of any session: [`ARCHITECTURE.md`](ARCHITECTURE.md)**, the generated map of the whole tree.

| # | Rule | Source |
|---|---|---|
| R1 | **Two profiles, one codebase.** DEV (full) keeps everything, including the server + WebUI, CLI, GGUF tools, model manager, tests and benches, with maximum functionality to try and test every model. BUNDLE ships `libspeech` with only the chosen families, down to one STT or TTS family. DEV tools depend on the library, never the reverse. | V7-D7 |
| R2 | **Both targets, always.** A change that works on only desktop or only Android is not done. | V7-D7 |
| R3 | **One first-party ABI: `speech.h`.** The `transcribe.h` shim keeps ZER0, Android_FreeSpeech and the `transcribe-cpp` crate working until they migrate. Our `capi/` `audiocpp.h` retires; upstream's `include/audiocpp.h` is theirs. | V7-D1 |
| R4 | **Three sources, never a blind merge, and nothing left behind.** audio.cpp: recorded merges. transcribe.cpp (the author's fork): per-commit triage. CrispASR: mined by family, never merged. Every optimization and new model of both parents is absorbed. Offer generic work upstream first. A dependency bump in any parent is ours. | L13, V7-D2, V7-D3, V7-D8, L16 |
| R5 | **Thin families on shared layers, without losing speed.** No private KV cache, decode loop, loader, mel or scheduler in a family package. An engine port must match the transcribe.cpp arch it replaces on WER **and** on RTF / peak memory (within 10 %, or faster). | L15, V7-D8 |
| R6 | **Proof before status.** Tick a box only by running its gate. Gates only tighten. Counts come from `scripts/status.sh`. The quick test tier (one short clip per model) is the default. | L14, L8, LESSONS A1–A3, A14 |
| R10 | **Never slower, never less accurate.** A port or numeric change is accepted only if, against **both parents that have the family** (audio.cpp and transcribe.cpp), on the same GGUF, machine, backend and threads, it is not worse on WER/CER (FLEURS multilingual subset) and not slower on RTF. Protocol: ≥ 3 runs; run 1 = warm-up (reported, never averaged); mean of runs 2..N (N = 3 in development, 5 for acceptance, ≤ 10). Backends: **CPU while developing**; CPU + CUDA + Vulkan-NVIDIA + Vulkan-Intel before acceptance. Full rules: [`docs/benchmarking.md`](docs/benchmarking.md). | V7-D9, V7-D8, L12 |
| R7 | **A pure library.** No audio-device I/O, no Python, no network, no CWD-relative assets and no GPL component by default. The LLM "brain" stays outside (llama.cpp on desktop, LiteRT-LM on Android). | V7-D7, V7-D4 |
| R8 | **Commit at green:** one family per commit, with the user's go-ahead, on `main`. | L17, AGENTS.md |
| R9 | **The scope question** before any work: *does FreeSpeech need it, and in which profile (DEV or BUNDLE)?* | V7-D7, SC5 |
| R11 | **Keep the map current.** `ARCHITECTURE.md` is generated from the tree with repomix by the pre-commit hook (`git config core.hooksPath .githooks`). Read it first; never hand-edit it. | V7-D10 |

---

## 0. If you are the agent taking over: read this first

0. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) (generated map) and re-read the **North star** above (the end goal and rules R1–R11). Every decision you make must fit it.
1. Run `bash scripts/status.sh`. Compare it with §2.2. Anything that moved since then is news.
2. Read §2.3 (known-bad state) and §4 (the next phase). Do the **first unchecked item of the current phase**, and nothing from a later phase.
3. Read `LESSONS.md` section headings (5 minutes). Most mistakes in this repo have happened before.
4. Before trusting any "DONE" anywhere, even in this file, run the gate it cites (LESSONS A1). If no gate is cited, it is not done.
5. At a phase exit:
   - run the suites;
   - update §2.2 from `status.sh` output;
   - tick the boxes whose gate you ran;
   - write one dated CHANGELOG entry;
   - ask the user for the go-ahead to commit (on `main`).

   Do **not** update any other plan-like document; there are none.

---

## 1. Goal and target architecture

**Goal.** One local-only C++/ggml runtime for **all** speech and audio tasks: STT (offline and streaming), TTS and voice cloning, speech-to-speech, voice conversion, VAD, diarization, alignment, LID, text post-processing, speech translation, separation, enhancement, music analysis and generation, codecs, and watermarking. One C ABI, one registry and one set of shared runtime layers. Each model family is a thin package with a pinned oracle.

**Sources** (V7-D2; see LEDGER §4). No parent's *code* is canonical; each parent's *contracts* are.

| Source | Relationship | What we take |
|---|---|---|
| **audio.cpp** | git parent (`upstream` = `0xShug0/audio.cpp`) | The **skeleton**: engine sessions, registry, model specs, framework, ~100 families. Merged continuously. |
| **transcribe.cpp** | co-parent (sibling checkout, triage watermark) | The **contracts**: size-aware C ABI, 4-state streaming, limits, exception containment, oracle parity. |
| **CrispASR** | reference parent #3 (sibling checkout, triage by family) | The **catalogue**: ~60 families we lack, converters, 113 PyTorch reference-dump modules, a stage-diff harness, numerics lessons. Mined, never merged. |

```
 BUNDLE profile → host apps link libspeech in-process: FreeSpeech desktop (successor of ZER0 + S2B2S, Tauri/Rust)
                  and FreeSpeech Android (Rust cdylib + JNI). Apps own mic/speaker, UI and the LLM.
 bindings: rust (first; Cargo features pick the model set) · python · ts
 DEV profile   → everything above PLUS server + WebUI, CLI, GGUF tools, model manager, tests, benches
                  (the full test bench for every model; maintained, never a dependency of libspeech)
                              │
 speech.h — the ONLY first-party C ABI (V7-D1): struct_size, typed ext slots, 4-state streaming,
            exception containment, speech_status.   transcribe.h = shim until bindings move.
                              │
 Engine registry: IVoiceModelLoader → ILoadedVoiceModel → I{Offline,Batched,Streaming}VoiceTaskSession
 TaskKinds: asr tts s2s vc vad diar align lid textpost mt separation enhance music_analysis codec
            watermark generation   (new kinds are contract-first, V7-D4 / L1)
                              │
 framework/  asr  (EncDecKVCache, AR/transducer/CTC/CIF drivers, AsrResult, AsrLimits, long-form)
             tts  (sampling, flow/diffusion steppers, vocoders, ref-voice cache)       ← to build (C2)
             audio (polyphase resample, MelExtractor, fbank, STFT, chunking) · tokenizer hub
             weights (TensorSource + foreign-layout maps) · scheduler/allocator owned by the engine
                              │
 families/   graphs + assets + spec + thin session. No private KV cache, decode loop, loader or mel.
             Every family: provenance (audio.cpp | transcribe.cpp | CrispASR | in-house) + pinned oracle.
                              │
 external/ggml — generated: pin ≥ max(parents' floors) + patches/ggml/*.patch (sync-ggml.sh --check)
```

**Two build profiles, one codebase** (V7-D7, user directive 2026-09-26):

| Profile | What it contains | Who uses it |
|---|---|---|
| **DEV (full)** | Every family, plus audio.cpp's server + WebUI, CLI, GGUF tools, model manager, tests, benches, verdict/parity harnesses. **Maximum functionality: every way to try and test every model.** Maintained and extended (e.g. new families get WebUI/CLI coverage). | developers, model porting, QA |
| **BUNDLE (embed)** | `libspeech` + `speech.h` (+ Rust crate) and **only the selected families**, down to a single STT or TTS family. No server, WebUI, CLI, tools, tests or demo assets. Static by default; optionally one shared library with ggml embedded. | the final apps |

- **Model selection is a first-class build input**, like transcribe.cpp's `TRANSCRIBE_MODEL_SET` (`full | minimal-multilingual | none | custom` + `TRANSCRIBE_MODELS`, surfaced as Cargo features by its `build.rs`). speech.cpp has `AUDIOCPP_MODEL_SET` / `AUDIOCPP_MODELS`. Phase E turns them into named composites and Cargo features, so the bundled `.so`/`.dll` carries only what the app ships.
- **Later:** families and ggml backends as loadable plugins (`TRANSCRIBE_ARCH_DL`-style), so a zero-model core plus per-family plugins is also possible.
- **Rule:** the DEV tools may depend on the library, never the reverse. Nothing in the BUNDLE profile may require a DEV component.

**Not in scope:**
- A general text-chat LLM runtime (V7-D4).
- Cloud or remote inference.
- Shipping the DEV tools (server, WebUI, CLI, installer, demo assets) inside a bundled app (V7-D7). They are kept and maintained for testing, but only in the DEV profile.
- The LLM "brain": FreeSpeech uses llama.cpp on desktop and **LiteRT-LM** (Gemma 4) on Android. speech.cpp exposes hooks and callbacks for it and never embeds it (V7-D4).
- Audio device I/O inside the library: the host app owns mic and speaker; the library takes and returns PCM.

**The two primary consumers.** speech.cpp is built for both: desktop first, Android second, and neither is optional (details: [`docs/reports/strategy_freespeech_2026-09-26.md`](docs/reports/strategy_freespeech_2026-09-26.md) §2):

| Consumer | Today | What it needs from speech.cpp |
|---|---|---|
| **FreeSpeech desktop** (**primary target #1**; successor of **ZER0** `../../Handy_V2` and **S2B2S** `../../STT_BRAIN_TTS/S2B2S`) | Tauri/Rust. ZER0: transcribe-cpp crate, **Multi-STT** (a primary + up to 8 extra models concurrently, merged by concat or LLM), live mode, Earshot VAD. S2B2S: an audio.cpp server process + a Python TTS venv + ONNX VAD + a llama.cpp brain | one in-process library replacing all of that: STT, TTS, cloning, VAD, streaming both ways; **many concurrent sessions sharing one GPU** (ResourceManager, SC4); CUDA/Vulkan/Metal; the real-time conversation loop (SC3). |
| **FreeSpeech Android** (`../../Android_FreeSpeech`, **primary target #2**) | Rust `cdylib` + JNI; `transcribe-cpp` crate with `features = ["minimal-multilingual"]`; IME, RecognitionService, live subtitles, R2T2 streaming (`TRANSCRIBE_EXT_KIND_R2T2_STREAM`, 80–2000 ms cadence), committed + tentative partials | arm64 NEON/dotprod/fp16 with an Armv8-A fallback; Bionic-clean link (no libpthread; `c++_shared`); 16 KB page alignment; a small `.so` holding only the chosen families; mmap weights and a low RSS (the Low Memory Killer); the streaming contract carried unchanged into `speech.h`; later TTS on-device. |

**Migration path for both apps:** `transcribe.h` becomes a shim over `speech.h` (S3.2), and the Rust `transcribe-cpp` crate can then be pointed at speech.cpp with no app changes. After that, apps move to the `speech` crate at their own pace. What this means for the direction (proposed) is in §8.

---

## 2. State

### 2.1 How status is produced
`bash scripts/status.sh` (add `--no-ctest` to skip registration counts). It prints measured, diffable `key: value` lines. Registration counts are **not** results: a result comes only from running ctest.

### 2.2 Snapshot at the 2026-09-26 checkpoint
Output of `scripts/status.sh`, abridged:

```
head: d04fd552 2026-09-24   dirty_paths: 89   stashes: 1 (transcribe-sync-wip, superseded)
audio_cpp_ahead_behind: 128/21            transcribe_cpp_untriaged: 24 (watermark 0a67b65b)
ggml pin: 456172ec (0.24.0) + 15 patches  transcribe_cpp_floor: 353b63b4 0.25.3   ← L13 violated
src_models 254k (71 dirs)  community 102k (31)  framework 86k  runtime 63.5k
runtime_arch_dirs: canary canary_qwen cohere gigaam granite granite_nar medasr moss parakeet sortformer voxtral
private_kvcache_structs: canary_qwen, voxtral, granite_speech (+ upstream echo_tts: allow-listed)
c_abi_headers: capi/include/audiocpp.h  include/audiocpp.h  include/transcribe/transcribe.h   (speech.h: absent)
ctest registered: build-cpu-core 136, build-cpu-asr-abi 138
```

**Last measured suite results:**
- 2026-09-23, at `774cecf1`: core 121/121, asr-abi 115/115, CUDA 99/99 before B16c.
- The uncommitted batch: see [checkpoint §2](docs/reports/checkpoint_2026-09-26.md#2-true-state-of-the-uncommitted-batch).

### 2.3 Known-bad state (the correction list)
Evidence for each item is in the checkpoint report.

| # | Problem | Fixed in |
|---|---|---|
| K1 | ~75 files of engine work (6 ports, dual layouts, parity tests) are **uncommitted**, and the tracked `CMakeLists.txt` diff depends on them. **Measured 2026-09-26:** it builds, but 2 suite tests and 10 of 12 verdicts fail on real defects (cohere SentencePiece, parakeet graph overflow and segfault, granite routing, granite_nar arch shape, and one shared adapter contract gap). | S0.1 a–i |
| K13 | Model-backed tests ran the whole corpus several times per side (a >35 min run that never finished). | **Fixed** (S1.9) |
| K2 | **Two C ABIs named `audiocpp`** (`capi/include/audiocpp.h` and upstream `include/audiocpp.h`) share the library name, and 3 symbols have incompatible signatures (`audiocpp_stream_start/push/finish`). The options `AUDIOCPP_BUILD_C_API` and `AUDIOCPP_BUILD_CAPI` differ by one underscore. | S1.1 (interim), S3 (retire) |
| K3 | ggml 0.24.0 is below transcribe.cpp's floor 0.25.3; 24 transcribe.cpp commits are untriaged; 21 audio.cpp commits are unmerged; upstream rewrote v0.8.2 (`9bdd1d90` → `4d88768f`), and the `/utf-8` hunk for vieneu is missing. | S0.2–S0.4 |
| K4 | New ports violate 11a: 3 private KvCaches, 2 near-verbatim NeMo mel copies (canary_qwen and granite_nar, 700 lines each), and a new host argmax in granite_speech. | S2 |
| K5 | WER gates default to **10 %** while their comments say "the baseline is the bar". Only parakeet passes an explicit bound. | S1.3 |
| K6 | Test sources never compiled or registered: 13 in `tests/transcribe/`, `test_capi_shared_lib_surface.cpp`, `test-omtd.cpp` (which was "made live" but is never built), and `capi_test` (built, but no `add_test`; it prints SKIP and returns 0). `whisper_bin_parser_unit` returns 0 when skipping. | S1.4 |
| K7 | CI claims that are false: the clang-format job calls a missing `scripts/ci/clang-format.sh`; there is no `sync-ggml --check` job; no orphan-test or orphan-source lint. | S1.5 |
| K8 | The README described the fusion inverted (fixed 2026-09-26). It still has **20 dead `docs/models/*.md` links** (the cards exist only in transcribe.cpp), "16 STT families" over a 20-row table, and links to nonexistent files. | S1.6 |
| K9 | Orphans and junk: `src/runtime/transcribe-kaldi-fbank.*` is built by nothing; `capi/test/` holds scripts with foreign absolute paths; `tests/golden/batch/*.json` is keyed by `/Users/cj/...` paths; `patches/speech-cpp-music3-delta.patch` (339 K) has unknown status. | S1.7 |
| K10 | `DecodeDriver` overclaims: its header says it "eliminates 15 copies", but only argmax is shared, by 6 packages. All-suppressed logits return token 0, which is not EOS, so this is a latent emit loop. `<limits>` is missing. | S2.3 |
| K11 | L5 duplicates without a verdict: `moonshine` / upstream `moonshine_asr` (not in `family_registry`), `granite_speech` / `granite5asr`, two `sortformer_diar` packages, `vibevoice_asr` / `vibeasr`, three NeMo transducer stacks. | 11b waves |
| K12 | `capi/src/audiocpp_capi.cpp:724`: a batch-TTS item exception is swallowed with no per-item error. | retired with the ABI in S3; do not invest before then |

---

## 3. Laws

**Carried from v6:**

| Law | Rule |
|---|---|
| **L1** | **Contract before code.** No family migrates, and no task kind gets a family, until the destination contract is at least as strong as the source's, field by field. |
| **L2** | **Net before demolition.** No deletion without a gate that would fail if the deletion regressed behaviour. |
| **L3** | **Additive before destructive.** Replace function bodies, not files. Never rename an upstream-owned file (merge conflicts). |
| **L4** | **No inert features.** Done means a public surface reaches it **and** a test proves it. |
| **L5** | **One family id.** One canonical id per model; every other spelling is a registered alias. |
| **L6** | **Specification beats inspection.** Shared code derives from a written contract, not from reading two implementations. |
| **L7** | **Every deletion has a ledger row:** what, replacement, equivalence gate, revert commit (LEDGER §3). |
| **L8** | **Gates only ratchet tighter.** Never loosen a threshold to land a change. |
| **L9** | **Exception containment is a build property.** No exception escapes a C entry point, and no raw ggml teardown in library code. Both are lint-enforced. |
| **L10** | **Parity against a pinned oracle:** a reference implementation at a pinned revision plus a tolerance file. The oracle is the original model, not another port (LESSONS D9). |
| **L11** | **Never truncate silently.** Reject with `INPUT_TOO_LONG`, or return with `truncated=true`. |
| **L12** | **Measure the real flow** (C ABI + run_batch). Run ≥ 3 times; run 1 is the warm-up (reported, never averaged). Take the mean of runs 2..N (N = 3 in development, 5 for acceptance, ≤ 10), interleaving arms. Revert any win inside the noise floor. See [`docs/benchmarking.md`](docs/benchmarking.md). |
| **L13** | **Two parents, one child** (now three sources). A dependency bump in any parent is ours. `sync-deps.sh` runs at every phase boundary. Syncs end in a recorded merge. |

**New at the checkpoint (v7):**

| Law | Rule |
|---|---|
| **L14** | **No status without a gate.** A checkbox is ticked only by naming the ctest, script or grep that proves it and running it. Counts come from `scripts/status.sh`. |
| **L15** | **Thin family, enforced.** The §5 architecture gates are run before and after every port. A port that adds a private KV cache, decode loop, loader, scheduler or mel is not done, whatever its WER. |
| **L16** | **Upstream first** (V7-D3). Before porting or building anything generic, check audio.cpp (`git ls-tree upstream/main …`). Prefer changes that could be offered upstream. |
| **L17** | **Commit at green.** Work that passed its suite is committed (with the user's go-ahead) before new work starts. Never leave a session with unverified, uncommitted work without flagging it at the top of the handover. |

---

## 4. Phases

**Execution order:**

```
S0 → S1 → S2 → S3 → E (BUNDLE profile + composites + Android; E0–E4 may start after S0) → 13 (Rust first) → { 11b waves, C0 } → 11c → C1 → C2 → C3 waves … ; Track M continuous; 14 after S3
```

**Checkbox meaning:**
- `[x]` — the gate was run and is green;
- `[~]` — code is written but the gate has not been run;
- `[ ]` — not started.

**Pause rule:** stop at each phase exit and ask the user before starting the next phase.

### S0 — Stabilize (current phase)
- [ ] **S0.1 Fix, then commit, the uncommitted batch.**
  - Measured 2026-09-26 ([checkpoint §2](docs/reports/checkpoint_2026-09-26.md#2-true-state-of-the-uncommitted-batch)): both trees build, but gates fail.
  - Fix in this order, re-running `ctest -L quick` after each (~4 min for the whole tier):
    - [ ] a. `model_specs/granite_nar.json`: rename the session option `granite_nar.shaw_bias` → `shaw_bias`. *Gate:* `model_spec_system_test`.
    - [ ] b. `whisper_c_abi_parity_test`: pin the same `n_threads` on both sides (the adapter now defaults to up to 8). *Gate:* that test.
    - [ ] c. **One adapter fix for the shared contract gaps** (canary_qwen, voxtral, canary): segment end time when the family has no timestamps, `max_audio_ms` and prompt caps from the spec, `detected_language` policy. *Gate:* `verdict_canary_qwen`, `verdict_voxtral`, `verdict_canary`.
    - [ ] d. cohere engine: `SentencePiece processor cache miss` gives empty text. *Gate:* `verdict_cohere`.
    - [ ] e. parakeet dual layout: graph hash set full (0xc0000409) and a segfault in parity. Size the graph budget. *Gate:* `verdict_parakeet_*`, `parakeet_engine_arch_parity_*`.
    - [ ] f. granite routing: the `granite5asr` spec claims the Granite 4.0 GGUF (L5 collision). Make `granite_speech` claim `general.architecture=granite_speech`. *Gate:* `verdict_granite`.
    - [ ] g. granite_nar arch side broken by the uncommitted `shaw_attn.cpp` edit (shape [128,1025]). Verify by revert, then fix both arches consistently. *Gate:* `verdict_granite_nar`, `granite_nar_engine_arch_parity_test`.
    - [ ] h. moss: caps and diarization contract differences. Record them as decisions, or fix. *Gate:* `verdict_moss`.
    - [ ] i. `transcribe_stream_committed_pointer_stability`: pass `TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF` in its registration (harness gap, not an asset issue).
  - **Commit family by family, green ones first** (medasr, gigaam, canary_qwen after c, voxtral after c), with the user's go-ahead. Never one batch commit, and never a CMake hunk without its sources.
  - Drop `stash@{0}` (superseded premature B16c) with the user's OK.
  - *Gate:* `ctest -L quick` green in `build-cpu-asr-abi`; core suite green; `status.sh` `dirty_paths` covers only work in progress.
- [ ] **S0.2 audio.cpp merge (21 commits)** as a recorded merge.
  - Treat `4d88768f` as a re-release of the already-merged `9bdd1d90`; adopt its `/utf-8` hunk.
  - Then run `sync-ggml.sh --check`.
  - Conflict hot spots: `framework/runtime/kv_cache`, `greedy_qwen_decoder`, `model_spec/metadata`, `io/safetensors`, trace.
  - *Gate:* `rev-list … HEAD...upstream/main` shows `…/0`; `--check` clean; suites green.
- [ ] **S0.3 transcribe.cpp triage (24 commits past `0a67b65b`).** One row each in `docs/upstream/transcribe_cpp_triage.md`, then advance the watermark. Must-read items:
  - `96b7c9c1` error-code ABI;
  - `81022f38` TQ1_G128 quant (patch 0003);
  - `0611d23f` parakeet-ultra/redux;
  - CUDA-graphs default;
  - qwen3_asr packed QKV.

  *Gate:* `status.sh` shows `transcribe_cpp_untriaged: 0`.
- [ ] **S0.4 ggml → ≥ 0.25.3** (`353b63b4` or newer) with `sync-ggml.sh <40-char sha>`.
  - Rebase the 15 patches.
  - Audit transcribe.cpp patches 0003/0004 and CrispASR's fork patches 1–6, #10 and #15 (LEDGER §4) for overlap and need.
  - *Gate:* `--check` clean; core + asr-abi green; CUDA tree only if a CUDA patch changed (LESSONS F3).
- [ ] **S0.5 transcribe_cunba SFPS.** Compare its sortformer push-audio streaming session with `community_models/sortformer_diar/stream_schedule.cpp`, then adopt or record N/A.

### S1 — Hygiene corrections (from the audit)
- [ ] **S1.1 (K2, interim).**
  - Rename our library output (`audiocpp` → `speechcpp_capi_legacy`) and our option (`AUDIOCPP_BUILD_CAPI` → `SPEECHCPP_BUILD_LEGACY_CAPI`) so the two ABIs can never be mistaken for each other.
  - Add a test that no `audiocpp_*` symbol is declared with two signatures across `include/` and `capi/include/`.
  - The final fix is S3.
- [ ] **S1.2** Delete orphans: `src/runtime/transcribe-kaldi-fbank.*` (fill B2 row) and `capi/test/` (foreign scripts). Re-key or delete `tests/golden/batch/*.json`. Decide on `patches/speech-cpp-music3-delta.patch`.
- [ ] **S1.3 (K5)** Every WER gate registration passes an explicit bound = baseline edits + 1; remove the 10 % default (fail if no bound is given). Extend `asr_e2e_edits_test` beyond moonshine.
- [ ] **S1.4 (K6)** Register or delete every test source. Missing models return the skip code (`SKIP_RETURN_CODE`), never 0.
- [ ] **S1.5 (K7)** Add CI/lint gates:
  - an orphan test lint (every `tests/**/*.c*` is in a target);
  - an orphan source lint (every `src/**/*.cpp` is compiled);
  - a `sync-ggml.sh --check` job;
  - the §5 architecture gates;
  - a ban on absolute user paths in tracked non-fixture files;
  - a Markdown link check.

  Either vendor transcribe.cpp's clang-format script or delete the dead workflow.
- [ ] **S1.6 (K8)** README: fix the 20 dead model-card links (vendor the cards into `docs/models/stt/`, or link transcribe.cpp), fix the family count, and remove links to nonexistent files.
- [ ] **S1.7** Retire the `audio_cunba` checkout from the workspace (0 unique commits). This is a workspace action; ask the user.
- [x] **S1.9 Test tiers for model-backed tests** (done 2026-09-26 at the user's request; gate run).
  - Verdict and parity tests default to one 3.5 s clip (`assets/asr_validation/quick/`, label `quick`).
  - The multi-clip `*_full` variants are registered only with `-DSPEECHCPP_FULL_ASR_TESTS=ON`.
  - Verdicts register only when their engine family is linked.
  - *Gate run:* `ctest -L quick` in `build-cpu-asr-abi`: 19 tests in 4.4 min (was >35 min, unfinished).
  - **Extend the same pattern** to the remaining model-backed tests: the WER gates keep the 4-clip corpus (they measure WER), but smoke/e2e tests take one clip.
- [ ] **S1.8 (proposed SC1) `docs/api/speech_h_spec.md`**: the `speech.h` contract written from FreeSpeech's use cases (see §8), before S3 implements it.

### S2 — Enforce the shared ASR layer (the unfinished 11a, re-closing B30)
- [ ] **S2.1** Move canary_qwen, voxtral and granite_speech onto `framework/asr` `EncDecKVCache` / decoder-only KV. Delete their private `KvCache`. *Gate:* the §5 KvCache grep lists only the allow-list; the three `*_engine_arch_parity_test` pass unchanged.
- [ ] **S2.2** One `NemoMelFrontend` in `framework/audio`, arch-exact (PerFeature `1/(std+eps)`, reflect or constant pad as an option). canary_qwen and granite_nar use it, and B1 (`transcribe-mel`) can then go. *Gate:* the parity tests of both families pass unchanged.
- [ ] **S2.3 (K10)** Make `DecodeDriver` real: AR greedy loop + suppress + EOS set + max-tokens (L11), and never return a non-EOS token when everything is suppressed. Migrate whisper, moonshine and granite_speech onto it. Fix the header claim. *Gate:* the WER gates are unchanged; a unit test covers all-suppressed logits.
- [ ] **S2.4** Close the remaining 11a gates:
  - long-form driver over `engine/audio/chunking`;
  - a >30 s Whisper clip returns `truncated`, or is processed fully (new test `asr_limits_contract_test`);
  - transducer and CTC drivers (A24);
  - replace the stale `vad_plan_unit` reference with `audio_chunking_test`.
- [ ] **S2.5** Finish the cached-graph input audit across engine models (LESSONS D1). *Gate:* a "same clip twice" test for every ASR family in `build-cpu-asr-abi`.

### S3 — `speech.h`, and retire our `audiocpp.h` (V7-D1; was Phase 12)
- [ ] **S3.1** `include/speech/speech.h` + `src/capi/speech_capi.cpp`, directly on engine sessions. Designed from `transcribe.h`: `struct_size`, typed ext kinds, 4-state streaming, `speech_status`, api_guard. The header sketch is in archived roadmap §6.3; the shim design is in §6.4.
- [ ] **S3.2** `transcribe.h` becomes a shim over `speech.h`. *Gate:* `abi_compat_test` is byte-identical through the shim; also `exports_match_header_test`, `lint_api_guard`, `speech_capi_test`.
- [ ] **S3.3** **Retire** `capi/include/audiocpp.h` + `capi/src/audiocpp_capi.cpp` (B32). No shim. Move its tests to `speech.h` equivalents; the `cli`/`server` already use the engine. Upstream's `include/audiocpp.h` stays untouched (it is the parent's).
- [ ] **S3.4** Offer the ABI discipline upstream (V7-D3). Optional; ask the user.

### E — BUNDLE profile and model-set composites (V7-D7; can start after S0, must be done before F1)

The model is transcribe.cpp's embedding: `TRANSCRIBE_BUILD_SHARED`, `TRANSCRIBE_SHARED_EMBED`, `TRANSCRIBE_INSTALL` + `transcribe-link.json`, `TRANSCRIBE_GGML_BACKEND_DL`, `TRANSCRIBE_ARCH_DL`, and `bindings/rust/sys/build.rs`.

- [ ] **E1 `bundle` CMake preset (BUNDLE profile): library only.** A matching `dev` preset keeps everything ON (server + WebUI, CLI, tools, tests).
  - Settings:
    - `AUDIOCPP_BUILD_SERVER=OFF`, `AUDIOCPP_BUILD_CLI=OFF`, `AUDIOCPP_BUILD_GGUF_TOOL=OFF`, `AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF`;
    - tests, examples and warmbench OFF;
    - `AUDIOCPP_MODEL_SET=custom` with an explicit `AUDIOCPP_MODELS` list;
    - the unified ABI ON (`transcribe.h` until `speech.h` lands, then `speech.h`);
    - `AUDIOCPP_STRIP_DEAD_CODE=ON`.
  - It replaces the `client-*` presets, which ship the legacy `audiocpp` C API + CLI + GGUF tool (`docs/build/client_minimized_build.md`).
  - *Gate:*
    - a clean configure+build never touches `webui/`, Node or OpenSSL (prove `app/server/server_frontends.cmake`, included unconditionally, is inert);
    - the only artifacts are the library + public headers;
    - the exported symbols are only `speech_*` / `transcribe_*`;
    - `status.sh` records library size per model list.
- [ ] **E2 Install tree + link manifest.**
  - `SPEECHCPP_INSTALL=ON` produces `lib/`, `include/speech/` and `lib/speech-link.json`, following transcribe.cpp's `transcribe-link.json` convention, so a Rust `build.rs` can link a prebuilt tree or drive CMake itself.
  - Static by default; `SPEECHCPP_SHARED_EMBED=ON` gives one shared library with ggml inside.
  - *Gate:* a 20-line C consumer and a Rust `build.rs` example both link the install tree and run one quick clip.
- [ ] **E3 Per-model bundling.**
  - Each family declares its dependency closure: framework modules, tokenizers, bundled assets such as the silero VAD weights.
  - Selecting one family compiles and links only that closure.
  - Bundled assets resolve through `framework/assets/asset_paths` (next to the library, or embedded), **never CWD-relative**.
  - *Gate:* a single-family embed build (e.g. `parakeet_tdt` only) runs its quick test from a directory outside the source tree.
- [ ] **E4 Runtime independence.**
  - The library needs no Python, no network, no environment variables and no GPL component by default (eSpeak stays opt-in, `AUDIOCPP_STATIC_ESPEAK=OFF`).
  - The model catalogue API returns specs, URLs, revision and sha256, and **verifies** files. **Downloading is the host app's job**; an optional helper stays outside the core library.
  - *Gate:* quick tests pass with network off and no Python on PATH.
- [ ] **E5 (later) Plugins.** ggml backends as loadable modules (`GGML_BACKEND_DL`) and families as plugins (port `TRANSCRIBE_ARCH_DL`), so an app ships a small core and downloads model plugins with the weights.
- [ ] **E0 Model-set composites + Cargo features** (the transcribe.cpp `TRANSCRIBE_MODEL_SET` model).
  - Named composites:
    - `full`;
    - `android-stt` (e.g. parakeet_tdt, nemotron_asr, moonshine_streaming, whisper, confucius4_r2t2);
    - `desktop-freespeech` (Tier-A STT + TTS + VAD);
    - `none` (plugin core);
    - `custom` via `AUDIOCPP_MODELS`.
  - The Rust `build.rs` exposes them as Cargo features, plus `SPEECHCPP_MODELS` / `SPEECHCPP_CMAKE_ARGS` passthrough.
  - *Gate:* a one-family and a composite build each link and pass their quick test; `status.sh` prints library size per composite.
- [ ] **E6 Android (arm64-v8a) is a first-class target.**
  - NDK cross-build in CI.
  - CPU variants (NEON, dotprod, fp16, i8mm) with an Armv8-A runtime fallback.
  - The Bionic link line comes from our link manifest: no libpthread, `c++_shared`. This retires the `build.rs` workaround in Android_FreeSpeech.
  - 16 KB page-size alignment.
  - mmap'd weights; peak-RSS recorded.
  - *Gate:* Android_FreeSpeech builds against the speech.cpp crate (or the `transcribe-cpp` shim) and transcribes one quick clip on an emulator or device, with the `.so` size and peak RSS recorded.
- **Standing rule (V7-D7):** the DEV tools (`app/server` + WebUI, `app/cli`, GGUF tools, model manager) are **kept and maintained at full functionality**: every family should be testable through them. They live strictly *above* the library. A change that makes `libspeech` depend on any of them is rejected, and so is a BUNDLE build that pulls one in.

### 11b — Arch verdict and port waves (resume after S2; each family ends in a B row)
**Per-family procedure:**
1. Upstream-first check (L16).
2. Run the family through the C ABI and run_batch (L12).
3. Merge the loser's features as shared primitives.
4. Port as a thin package (L15).
5. Gates: `verdict_<f>`, `<f>_engine_arch_parity_test`, WER == baseline, **plus the R10 benchmark against both parents** ([`docs/benchmarking.md`](docs/benchmarking.md): FLEURS multilingual subset, WER ≤ and RTF ≤ the best parent, CPU first, all four backends before acceptance), **and RTF + peak memory ≤ 1.1 × the arch** on the same machine and threads (V7-D8: the transcribe.cpp optimizations must survive the port). Known case: the cohere engine is at RTF 0.455 vs the arch's 0.37 (+23 %), so it is **not** retirable until that gap is closed or the cause is recorded.
6. Delete the arch in its own commit, with the B row filled (L7).

Worked example: `docs/archive/reports/sortformer_diar_engine_port.md`, plus the phase-10.5 pattern in LESSONS A4.

- [~] gigaam, medasr, voxtral, canary_qwen, granite_speech, granite_nar: ports written (S0.1 verifies; S2 thins them).
- [ ] canary, cohere, moss: **verdicts** against the upstream packages `canary_asr`, `cohere_asr`, `moss_transcribe_diarize`, not ports.
- [ ] parakeet: close the engine blockers (LEDGER §2.2), then B10; the sortformer core goes with it.
- [ ] K11 L5 duplicates: one MR-3 verdict each.

### 11c — Delete `src/runtime/` in full
*Gate:* `src/runtime` absent; `lint_teardown` over all of `src/` shows 0; every B row has a revert commit; `TRANSCRIBE_DUMP_DIR` dump points moved (RISK-F15).

### C0 — Verification tooling from CrispASR (may run alongside 11b)
- [ ] C0.1 Stage diff harness: reference GGUF of named intermediate tensors + C++ diff reporting cos_min, magnitude and max_abs. Port `tools/dump_reference.py` conventions. *Gate:* runs on 3 existing families.
- [ ] C0.2 TTS→ASR round-trip WER gate. *Gate:* ≥5 TTS families (this is the first TTS numeric gate).
- [ ] C0.3 Quantizer carve-outs as a spec field (audio tower ≥ Q8, `token_embd` F16 below Q8).
- [ ] C0.4 Spec lint: `license` + `commercial_use` present; NC models are never default (V7-D5); every download has revision + sha256.
- [ ] C0.5 Arch-string resolution test: every converter-written `general.architecture` resolves through `family_registry`.
- [ ] **C0.6 Benchmark harness (R10, V7-D9). Needed before the next port is accepted, so do it right after S0.**
  - Port the author's transcribe.cpp `scripts/bench/` (run, suite, compare, report) and `scripts/wer/` (ingest FLEURS/LibriSpeech, subset, score WER/CER).
  - Add three arms on the same GGUF: speech.cpp, audio.cpp (`audiocpp_cli`) and transcribe.cpp (`transcribe-bench`).
  - Parent reference builds live in `../_ref_builds/`.
  - Add a `build-vulkan` tree; record the `GGML_VK_VISIBLE_DEVICES` index for Intel and for NVIDIA.
  - *Gate:* one family (e.g. parakeet_tdt) produces a `docs/reports/bench/<family>.md` with the three arms, the development subset, CPU, N = 3, and run 1 reported separately.
- [x] **C0.7 Generated architecture map + pre-commit hook** (V7-D10, 2026-09-26).
  - `ARCHITECTURE.md` is produced by `scripts/arch/gen-architecture.ts` with repomix 1.18.1, pinned.
  - `.githooks/pre-commit` regenerates and stages it, then checks staged files for control bytes, absolute user paths and unsynced `external/ggml` edits.
  - *Gate run:* the hook takes 1.2 s; a probe with a user path and a NUL byte was rejected; `gen-architecture.ts --check` passes.

### C1 — New task-kind contracts (V7-D4; one reference family each)
- [ ] textpost (PCS)
- [ ] lid audio (ecapa-lid) and text (CLD3)
- [ ] align (wav2vec2 aligner)
- [ ] mt (m2m100)
- [ ] music_analysis (beat-this)
- [ ] watermark (AudioSeal)

*Gate per kind:* TaskKind + request/result types + capability bit + one PyTorch-parity test.

### C2 — `framework/tts`
Extract sampling, ref-voice caching (CrispASR `tts_ref_cache` idea), and vocoder and codec steppers from overlapping families. *Gate:* ≥3 TTS families migrated with the C0.2 gate unchanged.

### C3… — Family waves from CrispASR (3–5 families per wave; each wave ends with `sync-deps.sh`)
Priority order is in [`docs/reports/fusion_rethink_2026-09-26.md`](docs/reports/fusion_rethink_2026-09-26.md) §4.2:
- **Tier 2 ASR:** wav2vec2 family, stt_xx fastconformer-CTC, kyutai-stt, paraformer (adds CIF driver), omniasr, firered-asr (with KV cache), glm-asr, gemma4-audio, xasr (on the kroko Zipformer2 stack), …
- **Tier 3 TTS:** zonos, dia, csm, orpheus, parler, melotts+openvoice2, speecht5, fastpitch, bark, tada, kugelaudio, sidon, …
- **Tier 4 variants:** spec and alias work only.
- **Tier 5:** optional pinned foreign-GGUF layout maps for `cstr/*` files.

*Per-family gate:* PyTorch parity (C0.1) + WER or round-trip gate + §5 gates clean.

### 13 — Bindings (after S3 and E2; **Rust first**)
Port transcribe.cpp's generator and `bindings/rust/sys/build.rs` model: build from source through CMake, or link a prebuilt E2 install tree.

Order: Rust (`speech-sys` / `speech`, what Handy, S2B2S and FreeSpeech link), then Python, then TypeScript.

*Gate:* `generate.py --check`; each binding matches the C result exactly; the Rust crate builds against the E2 install tree.

### 14 — Acceptance on all backends
A1–A24 (archived roadmap §10), including TTS parity (14.3) and `diar_der_test` (14.4). CUDA and Vulkan trees.

### Track M — methodology parity (continuous)
Bring families under `validate.py` with an engine-path driver (F15 open). This includes the nemotron_asr and citrinet_asr goldens and the qwen3 / voxtral_rt HF reference dumps. The count of families under `validate.py` rises every phase.

---

## 5. Architecture gates (run before and after every port; S1.5 puts them in CI)

| Gate | Command | Target |
|---|---|---|
| Private KV caches | `grep -rlE 'struct +[A-Za-z:]*KvCache *\{' include/engine/models src/models src/community_models` | only `src/community_models/echo_tts/dit.cpp` (upstream allow-list) |
| Private schedulers | `grep -rl ggml_backend_sched_new src/models src/community_models` | ≤ today's 2; never grows |
| Env-var config | `status.sh` `getenv_names_first_party` | ≤ 39; never grows (LESSONS C3) |
| Duplicate ABI symbols | S1.1 test | 0 |
| ggml invariant | `scripts/sync-ggml.sh --check` | clean |
| Teardown lint | `cmake -DSRC_DIR=src/runtime -P tests/lint_teardown.cmake` (widen to `src/` after sub-task 0.J) | 0 |
| Orphans | S1.5 lints | 0 |

---

## 6. CrispASR rules (V7-D2)

| Rule | Content |
|---|---|
| **C-R1** | Knowledge, not runtime. Families enter as engine packages. CrispASR `src/<model>.cpp` is read as a specification; MIT functions may be copied with attribution. None of its schedulers, KV caches, `whisper_params` or env knobs. |
| **C-R2** | The oracle is PyTorch. CrispASR is a second opinion for three-way checks on overlapping families. |
| **C-R3** | Triage by family, on demand: [`docs/upstream/crispasr_triage.md`](docs/upstream/crispasr_triage.md), watermark `dfcdacae`. |
| **C-R4** | No unpinned models. Revision + sha256 in every spec. |
| **C-R5** | Licences are spec metadata, with a pre-download gate (V7-D5). |
| **C-R6** | New task kinds are contract-first (L1). |

Background: [`docs/reports/crispasr/crispasr_architecture_review_2026-09-26.md`](docs/reports/crispasr/crispasr_architecture_review_2026-09-26.md), [`docs/reports/crispasr/crispasr_model_catalog_2026-09-26.md`](docs/reports/crispasr/crispasr_model_catalog_2026-09-26.md).

---

## 7. Reference material (linked, not inlined)

| Need | Where |
|---|---|
| Why v7 re-steered | [`docs/reports/fusion_rethink_2026-09-26.md`](docs/reports/fusion_rethink_2026-09-26.md), [`docs/reports/checkpoint_2026-09-26.md`](docs/reports/checkpoint_2026-09-26.md) |
| `speech.h` sketch and shim design (§6.3–6.4); acceptance A1–A24 (§10); gate ladder G0–G5 (§7.2); ownership map and merge rules (§9); frontend contract table (§5.1) | [`docs/archive/FUSION_ROADMAP_PLAN.md`](docs/archive/FUSION_ROADMAP_PLAN.md) |
| V6 decisions, merge log, rollback (Appendix J: `-DSPEECHCPP_ENABLE_UNIFIED_ABI=OFF -DSPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=OFF`) | [`docs/archive/TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md`](docs/archive/TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md). Appendix L baselines are **[UNVERIFIED]**: round numbers, no commands. |
| Session history to 2026-09-24 | [`docs/archive/progress.md`](docs/archive/progress.md), `CHANGELOG.md` |
| Porting stages 1–8 | [`docs/porting/`](docs/porting/) |
| ASR WER gates | [`docs/reports/asr_e2e_wer_gate.md`](docs/reports/asr_e2e_wer_gate.md) |

---

## 8. Strategic direction — proposed changes (awaiting the user's confirmation)

Full reasoning: [`docs/reports/strategy_freespeech_2026-09-26.md`](docs/reports/strategy_freespeech_2026-09-26.md).

**The finding.** Breadth is not a moat. audio.cpp now builds ASR families and its own C ABI, and CrispASR has more models than either. As planned, speech.cpp becomes "a better-tested audio.cpp" with a growing merge burden. The reason to exist is the **embeddable real-time voice runtime** that FreeSpeech needs and audio.cpp is not trying to be.

**The recommended strategy: upstream-first core, differentiated product layer.**
- Generic work goes to audio.cpp as PRs.
- speech.cpp keeps only what audio.cpp should not own.

| Id | Change | Where it lands |
|---|---|---|
| **SC1** | Design `speech.h` from FreeSpeech's use cases: sessions, bidirectional streaming, stable partials, events, cancellation/barge-in, model manager. Keep transcribe.h's discipline. | S1.8 spec → S3 implementation |
| **SC2** | Thin-fork topology. speech-only code lives in `src/speech/` + `include/speech/`. `status.sh` reports lines changed in upstream-owned files, and that number must fall. Every generic fix gets an upstream PR note. | continuous from S0 |
| **SC3** | Real-time `ConversationSession` **inside the library**: AEC → VAD → wake word → streaming ASR → turn detection → host LLM callback → streaming TTS → barge-in. The host pushes mic PCM, plus the far-end reference for AEC, and pulls TTS PCM. No device I/O in the library (V7-D7). Plus **latency gates**: time-to-first-partial, end-of-turn → final, text → first audio, barge-in reaction. | new phases R1–R2 after F1 |
| **SC4** | `ResourceManager`: model residency, VRAM/RAM budget and eviction, shared backends across sessions, truthful memory reporting. Makes concurrent sessions a tested contract (A23). *Gate:* ZER0's Multi-STT shape, 1 primary + 8 extra STT sessions on one GPU, and the Android low-RSS budget. | RM, alongside R |
| **SC5** | Curated catalogue tiers: **A** supported (gated, benchmarked, what FreeSpeech ships, ~15–25 families), **B** community (builds + quick smoke), **C** archived. New ports must answer "does FreeSpeech need it?" | ordering rule for 11b/C3 |
| **SC6** | Test tiers (quick = default, **done**; full = opt-in; nightly bench) plus a generated public `docs/scoreboard.md` (WER/RTF/VRAM/latency per Tier-A family per backend). | S1.9 done; bench after S3 |
| **SC7** | Family manifests generate CMake, aliases and capability tables. Later: families as loadable plugins. | after SC2 settles |

**Proposed order after S2:**

```
S3 (speech.h per spec) + E (BUNDLE profile: library only, composites, install tree, Android)
 → F1: Rust crate `speech-sys`/`speech` + FreeSpeech integration spike (replace the transcribe-cpp crate,
       the audio.cpp server and the Python TTS in one app build)
 → R1 ConversationSession → R2 AEC + wake word + latency gates   (RM in parallel)
 → Tier-A completion + scoreboard
 → 11b/11c underneath
 → C0–C3 only for Tier-A needs
 → SC7
```

**First Tier-A candidates** (from what S2B2S uses today):

| Role | Families |
|---|---|
| STT | parakeet, whisper, moonshine_streaming, qwen3_asr, voxtral_realtime |
| TTS | qwen3_tts, kitten_tts, kokoro, plus one voice-cloning family |
| VAD | silero_vad / pulsevad |
| Enhancement | rnnoise-class |
