# Checkpoint audit and re-steer — 2026-09-26

> **Why this exists:** the user asked for a mid-course checkpoint: re-question everything, find the errors earlier (sometimes weak) LLM agents introduced, correct the course, and prepare the handover.
> **Trees:**
> - `speech.cpp@d04fd552` plus ~89 dirty paths;
> - parents pulled fresh: `audio.cpp@955c8725`, `transcribe.cpp@145c96a6`;
> - `CrispASR@dfcdacae` (new reference parent).
>
> **Outputs of this checkpoint:**
> - [`PLAN.md`](../../PLAN.md) v7 — the only plan;
> - [`docs/LEDGER.md`](../LEDGER.md);
> - [`LESSONS.md`](../../LESSONS.md);
> - `scripts/status.sh`;
> - an archive of the four superseded plan docs.
>
> **Method:** four independent audits (build/test verification, AI-error hunt over the fusion commits, doc inventory, plan extraction), with the load-bearing findings re-verified by hand. Every finding below carries its evidence.

---

## 1. Verdict in one paragraph

**The target architecture is right; the execution drifted.** The engine spine, one ABI and a thin-family contract are the correct design. CrispASR, which lacks all three, shows the alternative: 778 ABI `#ifdef` arms and 86 private schedulers.

But four things went wrong over the last month:
1. **The plan's order was inverted.** Ports landed before the shared ASR layer and the final ABI existed.
2. **Status documents drifted away from the code.** Four hand-maintained plan docs held 14 different test counts and 3 patch counts, and marked phases DONE on tests that could not fail.
3. **Parallel surfaces were left unresolved.** Two `audiocpp.h` ABIs now carry conflicting symbols.
4. **Dependency hygiene lapsed.** ggml is below the parent floor, 45 parent commits are untriaged, and two days of work are uncommitted.

None of this is fatal, and all of it is listed with a fix in PLAN §2.3. The process changes (one plan, generated status, grep gates, L14–L17) target the *causes*, not only the symptoms.

---

## 2. True state of the uncommitted batch

*Measured 2026-09-26. Both trees were built from the working tree (uncommitted batch included), with real GGUFs; all models are present in `models/`, so nothing skipped for missing assets.*

**The six-port batch recorded in the tracker as "written + integrated" does not pass its own gates.** It builds, but it was never run.

### 2.1 Builds
- `build-cpu-core` builds: exit 0, 448 steps.
- `build-cpu-asr-abi` builds: exit 0, 135 steps.
- The new sources, parity tests and verdict harness all link.

### 2.2 Suites, excluding the model-backed verdict and parity tests
| Tree | Result | Failures |
|---|---|---|
| core | 123/124 passed, 1 skipped | **`model_spec_system_test`**: `model_specs/granite_nar.json` names its session option `"granite_nar.shaw_bias"`; the schema requires the local name `"shaw_bias"`. |
| asr-abi | 116/118 passed, 1 skipped | Same spec failure, plus **`whisper_c_abi_parity_test`** (7 diffs). |

The `whisper_c_abi_parity_test` failure comes from the uncommitted adapter change: `n_threads` 0 now maps to `default_n_threads()` (up to 8) instead of 1.
- The C-ABI side therefore runs multi-threaded while the direct-engine side runs single-threaded, and float drift flips punctuation tokens.
- It passes when pinned to one CPU.
- Fix: the test pins both sides to the same thread count. It also shows whisper-tiny output depends on thread count; record that in the family doc.

The skip is `transcribe_stream_committed_pointer_stability`. It is **not** an asset issue: the model exists, but ctest never passes `TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF` to this test (`cmake/transcribe-tests.cmake` sets it only for the interleave test). With the variable set, it passes in asr-abi. This is an old harness gap (LESSONS A2, sixth instance).

### 2.3 Model-backed verdicts (arch vs engine through the C ABI) and per-family parity tests
Quick tier, after the test-speed fix (§2.4):

| Test | Result | Root cause |
|---|---|---|
| verdict_medasr, medasr parity | **PASS** | — |
| verdict_gigaam_rnnt / _ctc (quick, ru.wav) | **PASS** | The full corpus shows token flips and a 40 ms timing shift on the English clips, which are out of domain for a Russian model. Judge on ru.wav. |
| canary_qwen parity | **PASS** | Text identical. |
| voxtral parity | **PASS** | quick: transcribe, re-run, tokenize. |
| verdict_canary_qwen | FAIL | Text identical, but engine segments have timing [0,0] where the arch has [0,3505]. Caps `max_audio_ms` differ. |
| verdict_canary | FAIL | Transcripts identical. Caps differ (max_audio_ms, lang_detect, translate, PNC); leading space in raw_text; detected_language "" vs "en". Contract mismatch, not numerics. |
| verdict_cohere | FAIL | **Engine bug:** `SentencePiece processor cache miss for the provided piece inventory`, so the engine returns empty text on every clip. |
| verdict_moss | FAIL | Caps and diarization contract differ; segment start 550 vs 510 ms. |
| verdict_parakeet_tdt_ctc + parity | **CRASH** 0xc0000409 | `ggml-impl.h:330` graph hash set full: the dual-layout engine graph exceeds its graph-size budget. |
| verdict_parakeet_unified | FAIL | Text identical. Token probabilities differ up to ~5e-3 vs the 1e-3 tolerance. parakeet_unified parity **SegFaults**. |
| verdict_granite | FAIL | **Routing bug:** the GGUF is claimed by the `granite5asr` spec (then fails on a missing `config.json`) instead of the new `granite_speech` family. An L5 duplicate-family collision. |
| verdict_granite_nar + parity | FAIL | **The arch side is broken:** `Wrapped ggml tensor shape does not match logical shape [128, 1025]`. Likely the uncommitted `shaw_attn.cpp` edit, which fixed granite_speech but breaks the NAR arch path (unverified by revert). The engine side transcribes. |
| verdict_voxtral | FAIL | Text identical. Engine caps `max_audio_ms` 0 (arch 10448640) and `INITIAL_PROMPT` 0; segment [0,0] vs [0,3505]; detected_language "en" vs "". |

**One shared defect, not N family bugs:** canary_qwen, voxtral and (in part) canary fail on the *same* contract gaps in the engine→C-ABI adapter path, with identical text:
- no segment end time when the family emits no timestamps;
- `max_audio_ms` and prompt caps not derived from the spec;
- `detected_language` set when the arch leaves it empty.

**Fix it once in `transcribe-arch-adapter.cpp` + spec caps, then re-run.** This is the "adapter result mapping is shared by every family" lesson (LESSONS A4) again.

The harness cannot pass by accident. With `SPEECHCPP_ENGINE_ARCHS=all` the builtin arch is removed from lookup, and the two sides print different arch strings.

**Consequence for S0.1:** commit only what is green. Candidates are medasr, gigaam, canary_qwen (after the timing fix) and voxtral parity. Fix the rest first, or commit them explicitly as `[~]` work-in-progress in their own commits, **with the user's OK**. Never mix them into a "batch" commit.

### 2.4 Test speed (fixed 2026-09-26, at the user's request)
**Before:** every model-backed verdict and parity test ran the whole 4-clip corpus (22 s of audio, including a 14.2 s clip) several times per side, plus batched passes. `canary_qwen` parity took 379 s and `verdict_moss` 206 s, and the run had not finished after more than 35 minutes.

**Now:**
- The default registration uses `assets/asr_validation/quick/`: one 3.5 s clip, `--max-clips 1 --no-batch` for verdicts.
- Parity tests run whichever listed fixtures exist.
- Voxtral parity takes `--quick`: plain + re-run; hint and translate modes are full-tier only.
- The multi-clip versions register as `*_full` only with `-DSPEECHCPP_FULL_ASR_TESTS=ON` (label `full`).
- Verdicts register only when their engine family is linked; the core tree used to fail them at load.

**Measured:** the whole quick tier (19 tests) runs in **4.4 min** wall-clock at `-j 2`. Most tests take 1–15 s. The floor is the 3B voxtral model on CPU (~50 s alone).

---

## 3. Findings (ranked)

Evidence is a path:line or a command. **V** means re-verified by hand during this checkpoint.

### CRITICAL
| # | Finding | Evidence | Correction |
|---|---|---|---|
| C1 | **Two C ABIs named `audiocpp`** with the same library name and symbol prefix, and incompatible signatures for `audiocpp_stream_start/push/finish`. A consumer built against one header that loads the other library crashes, and the linker gives no warning. The two options differ by one underscore (`AUDIOCPP_BUILD_C_API` vs `AUDIOCPP_BUILD_CAPI`). | `include/audiocpp.h:490` vs `capi/include/audiocpp.h:1006` **V**; CMakeLists.txt:202-207, 3326-3340 | PLAN S1.1 (rename output and option, add a duplicate-symbol test), then S3.3 (retire ours, V7-D1). |
| C2 | **~75 files of engine work uncommitted for 2 days**, and the tracked `CMakeLists.txt` diff depends on them: committing CMake alone breaks the build, and losing the tree loses 6 families. `stash@{0}` (premature B16c) is still held. | `git status`; `status.sh` `dirty_paths: 89` **V** | PLAN S0.1: commit family by family after verification (L17). |

### HIGH
| # | Finding | Evidence | Correction |
|---|---|---|---|
| H1 | **WER gates default to 10 %**, while their comments and memory say "the baseline is the bar". A 4–7× regression passes on a 69-word corpus. Only parakeet passes an explicit bound. | `tests/asr_e2e_wer_test.cpp:60` **V**, `tests/asr_stream_text_wer_test.cpp:66`; CMakeLists 2836-2847 | PLAN S1.3. |
| H2 | **Test sources never compiled or registered, and some were "fixed" anyway.** 13 files in `tests/transcribe/` (cohere/parakeet/loader/decoder/tokenizer smokes, …); `test_capi_shared_lib_surface.cpp` (it claims to guard the exported surface, which matters given C1); `test-omtd.cpp` was "made live" by `#undef NDEBUG` in `c562b7aa` but is built by no target; `capi_test` is built with no `add_test` and returns 0 on SKIP. | `grep -rn omtd` in all CMake files → only `tools/omtd` **V**; no `add_test` for `capi_test` **V** | PLAN S1.4, plus an orphan lint (S1.5). |
| H3 | **False or stale DONE claims in the (archived) tracker:** 7.2 "CI formatting lints integrated" (the script doesn't exist, so the job can't pass); Phase 0 "11 patches" (15); Phase 5 "48 entry points" (55); Phase 7.1 "51 TUs / 87 fixtures" (13 unregistered; `tests/golden/batch/*.json` unreferenced and keyed by `/Users/cj/...`); Phase 8 certified on tests "never compiled or registered until 2026-09-23" (`cmake/transcribe-tests.cmake:233-236`); the Phase 10 row contradicts Phase 10.5. | `docs/archive/MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md:94,99,102,123,124,134` | Archived. PLAN L14: a box is ticked only by a named gate. |
| H4 | **README described the fusion inverted**: it said transcribe.cpp's arch is canonical, the opposite of what happened. It linked a nonexistent `TO_DO_…_V5.md`, has 20 dead `docs/models/*.md` links (cards exist only in transcribe.cpp), and says "16 STT families" over a 20-row table. | README.md:262, 291-297 **V** | Inverted paragraph and plan links **fixed today**. The rest is in PLAN S1.6. |

### MEDIUM
| # | Finding | Evidence | Correction |
|---|---|---|---|
| M1 | **`DecodeDriver` overclaims.** The header says it "eliminates 15 copies", but 6 packages use only `argmax_logits`; whisper and moonshine don't use it; granite_speech adds another argmax. All-suppressed logits return token 0 ("caller's EOS check will terminate", but 0 is not EOS): a latent emit loop. `<limits>` is missing. | `include/engine/framework/asr/decode_driver.h:1-10,54`; `src/models/granite_speech/runtime.cpp:691` | PLAN S2.3. |
| M2 | **"Unified mel" marked DONE while new verbatim copies were added.** `canary_qwen/mel_frontend.cpp` (697 lines) and `granite_nar/mel_frontend.cpp` (703) differ by 46 lines, and both copy `src/runtime/transcribe-mel.cpp`. | diff after name normalisation | PLAN S2.2. |
| M3 | **11a regressed by new ports:** private `KvCache` in canary_qwen, voxtral, granite_speech (B30 had closed this class). | `status.sh` `private_kvcache_structs` **V** | PLAN S2.1. |
| M4 | **Orphan left by a retirement:** `src/runtime/transcribe-kaldi-fbank.*` is compiled and included by nothing (it served the retired sensevoice and funasr arches). | grep over all CMake and sources **V** | PLAN S1.2 (fill B2). |
| M5 | **Numbers contradict each other across docs.** CTest totals of 86, 88, 92, 95, 96, 100–103, 110, 111, 113, 114 and 121 are all claimed; patches are given as 7, 11 and 15; progress.md's snapshot ordering is confused. | archived docs | PLAN L14 plus `scripts/status.sh`. |
| M6 | **Upstream rewrote v0.8.2 after we merged it** (`9bdd1d90` → `4d88768f`, same parent and timestamp). Our tree lacks its `/utf-8` MSVC option for `engine_model_vieneu_v3_turbo`. No doc mentioned it. | `git reflog upstream/main` | PLAN S0.2. |
| M7 | **`whisper_c_abi_parity_test` proves ABI mapping, not model correctness**: since B16c both sides run the same engine package. That's fine, but it must not be cited as correctness parity (correctness = the WER gates vs the arch baseline). | test source | Wording in LEDGER §2.1 (listed as an ABI gate). |
| M8 | `catch (...)` swallowing in the legacy C ABI: batch-TTS item errors are dropped. | `capi/src/audiocpp_capi.cpp:724` | Dies with S3.3; do not invest. |
| M9 | Plan-order inversion: 11b ports landed before 11a closed and before Phase 12. | archived roadmap §8 vs working tree | PLAN S2 before any further port; L15. |

### LOW
- `capi/test/` holds ad-hoc scripts from another machine (`C:/Users/56579/...`, `D:\audiocpp_test\...`).
- `build_env.bat` hard-codes the VS 18 Community path.
- No CI job runs `sync-ggml.sh --check`.
- `patches/speech-cpp-music3-delta.patch` (339 K): status unknown.
- `transcribe_whisper_bin_parser_unit` returns 0 when skipping.

All of these are in PLAN S1.2 / S1.4 / S1.5.

### Verified sound (keep trusting these)
- `sync-ggml.sh --check` is clean: pin `456172ec` + 15 patches. The patch history is add-only.
- B29 is real: the `transcribe-vad*` files are gone.
- The verdict harness is strict: it compares status, text and every row, with p_tol 1e-3.
- `SPEECHCPP_ENGINE_ARCHS` exists and is used.
- All 11 verdict models are pinned with sha256 in `scripts/fetch_asr_test_model.py`.
- `sync-deps.sh` is read-only.
- The bare-`assert` problem is fixed wherever the file is actually built.
- Real bug found during the ports: `shaw_attn.cpp` had swapped weight dimensions, so **both granite arches could never encode** (since `9b34fd2d`). The fix is in the uncommitted batch.

---

## 4. Recurring mistake patterns, and the rule that now prevents each

| Pattern | Example | Prevention |
|---|---|---|
| "Written means done" | Phase 8 certified on uncompiled tests; test-omtd "fixed" but never built | L14; orphan-test lint (S1.5); LESSONS A1, A3 |
| Numbers copied forward instead of re-measured | 11 vs 15 patches; 110 vs 121 tests | `scripts/status.sh`; one plan (V7-D6) |
| Copying without dependencies | README from transcribe.cpp with 20 dead links; `capi/test` with foreign paths; goldens keyed by `/Users/cj` | link check; absolute-path ban (S1.5) |
| Headers and docs that overclaim | DecodeDriver "eliminates 15 copies"; "Unified mel DONE" | L4 + L14; S2 makes the claims true |
| Gates looser than their comments | 10 % default vs "3/69 is the bar" | S1.3; L8 |
| Parallel surfaces left side by side | two `audiocpp.h`; `C_API` vs `CAPI` options | V7-D1; S1.1 duplicate-symbol test |
| Plan order inverted under momentum | ports before 11a/12 | L15; §5 gates run before and after each port |
| Long-lived uncommitted work | 2 days, 7 families | L17 |
| Dependency drift unnoticed | ggml below parent floor; 45 parent commits pending | `status.sh` shows the drift every session; S0 |

---

## 5. What was re-questioned, and the answer

| Question | Answer |
|---|---|
| Should speech.cpp restart from a clean kernel? | **No.** A from-scratch design (rethink §3.1) differs from ours only in "parents as sources" and "generated build/docs". Both are adoptable incrementally (V7-D3, V7-D6, family manifests later). Restarting would lose ~100 families, the suites and the merge channel. |
| Is the audio.cpp engine the right spine? | **Yes.** It is the only one of the three sources with one registry for every surface. |
| Should we keep two ABIs and shim? | **No.** V7-D1: `speech.h` only; our `audiocpp.h` is retired. |
| Should CrispASR be merged? | **No.** Mine it by family (V7-D2). |
| Is the ceremony per family worth it? | Keep the parity test and the B row. The per-phase multi-document update is gone (V7-D6). |
| Are the recorded ASR baselines trustworthy? | The WER numbers come from runnable gates and are trusted. The V6 Appendix L performance baselines are **[UNVERIFIED]** (round numbers, no commands). |

---

## 6. Handover checklist for the next agent
1. `bash scripts/status.sh`
2. `PLAN.md` §0 → §2.3 → S0.1.
3. S0.1 needs the build result in §2 above. If it is green, ask the user for the go-ahead and commit family by family. If it is red, fix it first; never commit red.
4. Do not start S2 or any port before S0 and S1 are closed.
