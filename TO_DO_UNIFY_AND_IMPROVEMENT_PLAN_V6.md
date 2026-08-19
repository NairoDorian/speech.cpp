# Unified Audio — Ultimate Merge & Improvement Plan (V6)

**Status:** Active — Phase 0 kickoff ready. V6 = V5 merged with
`UNIFY_AND_IMPROVEMENT_PLAN.md` (2026-08-19). See §0.0 for the merge log and
binding refinements (R1–R10). The ggml convergence (R1) is certified on CPU and
CUDA as of R10 (2026-08-20); the live blocker is now an in-tree ASR model for
end-to-end WER validation.

This document is the master plan for evolving `speech.cpp` (a fork of
`audio.cpp`) by systematically absorbing the architecture, model families,
features, testing methodology, and ecosystem of `transcribe.cpp`, producing a
single best-of-breed, open, locally-runnable audio inference toolkit.

---

## The Master Key (master sentence — remember it)

> **This is a master key, this is a master sentence to remember:** *the two
> projects learn from each other in parallel — each is the other's teacher and
> student — and merging them improves them both at the same time.*

Nothing in this plan is a one-way port. Every merge step must (1) take the best
from each side, (2) upgrade the other side with what it learned, and (3) land as
a net improvement to the unified project and to every surviving piece of code
from both sources:

- `transcribe.cpp` teaches: the size-aware C ABI, `api_guard_*` exception
  discipline, `safe_*` teardown, golden-manifest + per-tensor tolerance testing,
  the 4-state streaming machine, spec decoding, abort-callback cancellation,
  exact-device backend resolution, backend-DL dynamic modules, four bindings.
- `audio.cpp` teaches: the graph executor, the 3-level task/session model, the
  model-spec catalog, the CLI/server/WebUI/workflow surfaces, the artifact
  store, and 40+ non-STT families that give STT models a home with TTS, VAD,
  diarization, VC, separation, alignment, and MIDI in one process.
- When two implementations compete (ggml, mel, tokenizer, a duplicated family),
  the question is never "which project wins" but "which choice makes the
  unified project stronger" — the loser's distinguishing features are merged
  into the winner, then the loser is deleted. Every phase must upgrade
  infrastructure for ALL families, not just the new ports (§0, §1.5).

---

## 0.0 V6 Merge Log (2026-08-19)

V6 merges `UNIFY_AND_IMPROVEMENT_PLAN.md` (the companion plan written against
the live `audio.cpp`/`transcribe.cpp` trees) into V5. The V5 body remains the
authoritative architecture; the refinements below are **binding overlays** that
reconcile the two plans. All facts in R5 were re-checked against the actual
trees on 2026-08-19.

### R1. ggml convergence — decision tree (reconciles D16 with UNIFY plan §3.1)

V5 (D16) pins ggml-org/ggml @ `8c63e70982c95ceb862e3a1073a2c1beef75d60a` (+
`patches/ggml/0001-fix-threadpool-oversubscription.patch`); the UNIFY plan
preferred keeping audio.cpp's `external/ggml` fork. Resolved as a **gated
experiment**, not an assertion:

1. **DECIDED (Phase 0, applied):** `external/ggml` was re-vendored to the
   upstream pin `8c63e709` (ggml 0.20.2) + the threadpool patch, mirroring
   transcribe.cpp's "upgrade ggml to 0.20.2" commit (`923d4a0`), via the adopted
   `scripts/sync-ggml.sh` workflow. The audio.cpp fork is gone; any fork-only
   deltas (CUDA graphs, llamafile-on-by-default, op fixes) must be re-added as
   tracked patches under `patches/ggml/`.
2. Phase 0 sub-task 0.L runs the convergence build now against the new tree:
   build `engine_core`, run the full audio.cpp test suite on CPU + one GPU
   backend; fix API drift in the framework (never by hand-editing this tree).
3. If green → convergence is certified; Appendix L's regression bounds (≤5% on
   transcribe baselines, ≤10% on audio.cpp baselines, WER +≤2% absolute) apply.
4. If red → fix the framework against the new API (or, as last resort, revisit
   the pin via `sync-ggml.sh <sha>`), keeping the ONE-vendored-ggml invariant.

**API-drift inventory (diffed `external/ggml` audio-fork HEAD 55eab3c vs pin
`8c63e709`, all consumers in `src/`).** The 0.20.2 bump is a pure tree swap —
transcribe.cpp's own source needed ZERO changes (its "upgrade ggml to 0.20.2"
commit 923d4a0 touched only `ggml/`, `scripts/sync-ggml.sh`, `README.md`), so the
ported `src/runtime/` is already compatible. The audio.cpp framework has **7
fork-only ggml APIs with no upstream equivalent** — these must become tracked
patches under `patches/ggml/` (per R1 step 1) before the convergence build:

| Fork-only API (gone from pin) | Consumers in speech.cpp `src/` |
|---|---|
| `ggml_conv_1d_fast_1d_im2col` | `framework/modules/optimizations/fast_conv_modules.cpp` |
| `ggml_convrot_linear` | `community_models/minimax_h3/dit_denoiser.cpp` |
| `ggml_flash_attn_ext_with_bias_mask` | `community_models/parakeet_tdt/encoder.cpp` |
| `ggml_graph_set_n_nodes` | `framework/runtime/graph_optimizer.cpp` |
| `ggml_mul_mat_pack4` | `framework/modules/optimizations/fast_projection_modules.cpp` |
| `ggml_sage_attn2` | `community_models/minimax_h3/dit_denoiser.cpp` |
| `ggml_backend_cuda_clear_graph` | `framework/core/backend.cpp` |

(`ggml_sage_attn2_i8` and `ggml_backend_cuda_split_buffer_type` are also
fork-only but have no callers in `src/`; the previously-planned `ggml_ssm_scan`
signature change — added `K` param — is a no-op here: no caller.) Upstream
`ggml_flash_attn_ext`/`ggml_conv_2d_dw_direct`/`ggml_graph_*` have no drop-in
substitute for these; they are genuine audio.cpp extensions, not renames.

### R2. Port layout sequencing (reconciles `src/runtime/` vs `src/models/`)

V5 ports the C ABI runtime into `src/runtime/` with families in
`src/runtime/arch/<family>/` (ABI-first); the UNIFY plan ports families into
`src/models/<family>/` per engine convention (engine-first). Both are correct —
**in sequence**:

1. Phase 0: port the whole C ABI runtime into `src/runtime/` wholesale (V5
   Appendix C). `libtranscribe` works on day one; bindings keep testing.
2. Phases 1–3: each family lands in the engine as `src/models/<family>`
   (loader + session + frontend/decoder split, `model_specs/<family>.json`,
   streaming session, WER-golden tests). When a family lands, its
   `src/runtime/arch/<family>/` copy is deleted and the ABI dispatches to the
   engine session through `ArchAdapter`.
3. End-state (Phase 4+): the engine is the single runtime core; the C ABI is
   the public surface; `src/runtime/arch/` is gone; the `Arch` trait survives
   only as the dispatch contract inside the CMake-generated
   `src/runtime/transcribe-arch.cpp` registry, with adapters for every family.

This preserves V5's ABI-stability guarantee AND the UNIFY plan's
integration-depth guarantee (every family reachable via CLI/server/WebUI/
workflow/model-specs).

### R3. Duplicated-family canonical selection (supersedes V5 §2.4 blanket rule)

V5 §2.4 makes transcribe.cpp's implementation canonical for the C ABI path and
replaces audio.cpp's for ALL overlaps. The UNIFY plan's evidence-based rule is
adopted instead: **one implementation per family, chosen by (a) variant
coverage, (b) WER validation status, (c) integration depth
(session/server/streaming/WebUI)**; the loser's distinguishing features (parity
tests, knobs, prompt formats) are merged into the winner, then deleted. Either
winner must satisfy BOTH surfaces — if the transcribe impl wins it gains an
engine session wrapper; if the engine impl wins the ABI routes to it via
`ArchAdapter`. Fallback: if the chosen winner fails the golden suite, the other
wins.

| Family | audio.cpp impl | transcribe.cpp impl | Canonical pick (V6) |
|---|---|---|---|
| fun_asr_nano | `src/models/fun_asr_nano/` | `src/arch/funasr_nano/` (WER-validated) | Engine model, upgraded with transcribe WER/parity corpus |
| qwen3_asr | `src/models/qwen3_asr/` | `src/arch/qwen3_asr/` | Engine model, upgraded with transcribe WER/parity corpus |
| voxtral_realtime | `src/models/voxtral_realtime/` | `src/arch/voxtral_realtime/` | Engine model, upgraded with transcribe WER/parity corpus |
| sortformer | `src/models/sortformer_diar/` | `src/arch/sortformer/` (streaming, presets) | Whichever passes both golden suites; streaming feature parity decides |
| parakeet_tdt | `src/community_models/parakeet_tdt/` | `src/arch/parakeet/` (11 variants) | transcribe.cpp version (wider variant coverage + WER-validated), ported to engine |
| sense_asr | `src/community_models/sense_asr/` | `src/arch/sensevoice/` | transcribe.cpp version unless engine streaming/ITN is strictly better |
| moss | `moss_tts_*` (TTS only) | `arch/moss/` (ASR + diarize) | No conflict — different tasks; both survive |

### R4. Licensing (carried from UNIFY plan §3.5, open decision)

`audio.cpp` is Apache-2.0 (ShugoAI LLC); `transcribe.cpp` is MIT. MIT code may
be incorporated into an Apache-2.0 project with MIT notices preserved. V6
recommendation: keep Apache-2.0 for the merged project; preserve MIT
attribution headers and third-party notices on all merged transcribe.cpp files;
obtain legal sign-off during Phase 0 (UNIFY plan §8 item 1).

### R5. Verified facts (re-checked 2026-08-19 against live trees)

| Claim | Result |
|---|---|
| `include/transcribe.h` = 2499 lines | ✓ confirmed |
| transcribe.cpp `ggml/UPSTREAM`: ggml-org/ggml @ `8c63e70982c95ceb862e3a1073a2c1beef75d60a` + threadpool patch | ✓ confirmed (D16 pin is real; tree generated by `scripts/sync-ggml.sh`) |
| transcribe.cpp `src/arch/` family count | ✓ **18** (K.1 item 7 correct; the UNIFY plan's "19" is superseded) |
| audio.cpp `src/models/` = 39, `src/community_models/` = 8, `model_specs/*.json` = 47 | ✓ confirmed (K.1 items 8–9) |
| audio.cpp `tests/` = 51 dirs; transcribe.cpp tests = 50 TUs (48 .cpp + 2 .c) | ✓ confirmed (K.1 item 10) |
| Overlap families present on both sides (fun_asr_nano, qwen3_asr, voxtral_realtime, sortformer_diar, parakeet_tdt, sense_asr) | ✓ confirmed |
| `AUDIOCPP_GGML_SOURCE_DIR` (CMakeLists:98) + `AUDIOCPP_MODEL_SET` (CMakeLists:221) exist | ✓ confirmed — convergence build (§7 step 2) and `asr`/`asr+full` composites need no new CMake plumbing |
| audio.cpp `external/ggml` HEAD = `52080cd` (= speech.cpp fork-base) | ✓ confirmed |

### R6. Phase mapping to the UNIFY plan

| UNIFY plan phase | V5 equivalent | V6 action |
|---|---|---|
| Phase 0 (bootstrap speech.cpp tree, rebrand, CI baseline) | — (pre-work) | Add as sub-task 0.A-0: copy audio.cpp → speech.cpp (replacing the stale fork), rebrand (project `SpeechCpp`, `speech/version.h`), gate = identical test results vs audio.cpp reference |
| Phase 1 (substrate merge) | Phase 0 sub-tasks 0.B–0.F, 0.L | Already covered; apply R1 decision tree |
| Phase 2 (port transcribe-only families) | Phases 1–3 | Add engine-session wrapper + `model_specs` + WER-golden per family (R2) |
| Phase 3 (de-dupe) | §2.4 | Use R3 table + tie-breaker |
| Phase 4 (unify surfaces; new `speech.h` ABI) | Phases 4–5 | `transcribe.h` compat first; task-based `speech.h` second (UNIFY plan §8 item 2 recommends this order) |
| Phase 5 (tooling/validation/docs) | Phase 6 + §6 | Extend the golden pipeline to audio.cpp-native ASR families (nemotron_asr, higgs_audio_stt, citrinet_asr, hviske_asr, vibevoice_asr, kroko_asr) — new success criterion |
| Phase 6 (improvements) | §1.5 + success criteria | Per-session backend instances (fixes ABI single-flight), one fbank/mel with reference-injection parity hooks, one batch API, one version header feeding both ABIs (UNIFY plan §5.5), host log callback, spec JSON as single source of truth |

### R7. New open decisions carried from the UNIFY plan (§8)

1. License: Apache-2.0 + MIT attribution (recommended) — sign off in Phase 0.
2. `speech.h` scope: ship `transcribe.h` compat first, `speech.h` in Phase 4 (recommended).
3. ggml sync cadence post-convergence: pin vs follow upstream (decide after R1 result).
4. Parakeet/SenseVoice canonical picks per R3 (transcribe versions recommended).

### R8. Sortformer upstream reference pin (2026-08-19)

Upstream model repo commit supplied by the user — record verbatim in
`docs/porting/reference_commits.md` at Phase 0 sub-task 0.A:

- **Repo:** `nvidia/diar_streaming_sortformer_4spk-v2`
- **Commit:** `5240a64075176943f677d30fa2171c780229f341` ("add: q8 GGUF for local
  inference with NeMo-Speech.cpp (#13)", parent `6dbf0d6`)
- **Model license:** CC-BY-4.0 (NVIDIA model license; distinct from the project
  license decision in R4 — record per-model license in `model_specs`)
- **Official asset added:** `diar_streaming_sortformer_4spk-v2.q8_0.gguf` —
  147,075,776 bytes, LFS sha256 `0679cfeb1ce356d0dea9470b31274f4bfc7eb927497d82005483770666da998a`

Verified variant state in the two trees (this resolves the R3 sortformer row):

| Aspect | audio.cpp | transcribe.cpp |
|---|---|---|
| Variant | **v1** — `diar_sortformer_4spk-v1` (spec packages: q8_0/f16/safetensors from `audio-cpp/audio.cpp-gguf` + `nvidia/diar_sortformer_4spk-v1`) | **v2.1** — `k_default_variant = "diar_streaming_sortformer_4spk-v2.1"` (`src/arch/sortformer/model.cpp:45`), streaming presets (DEFAULT/HIGH_LATENCY/VERY_HIGH_LATENCY/LOW_LATENCY) |
| Mode | offline only | streaming (with offline-capable runner) |

Consequences for the plan:

1. **R3 tie-break now favors transcribe.cpp's v2.1** as the canonical
   implementation (newer model generation, streaming presets, official NVIDIA
   v2 GGUF available for the golden pipeline). audio.cpp's v1 engine session
   stays as a fallback during Phase 3 until v2.1 passes the engine golden
   suite; streaming feature parity decides if they tie (per R3 rule).
2. **Golden pipeline asset:** NVIDIA now distributes an official q8_0 GGUF for
   v2 — `tests/golden/sortformer/` + `model_specs/sortformer_diar.json`
   packages can pull it directly (no conversion needed for the q8 path). The
   default transcribe variant is v2.1: verify whether a v2.1 GGUF exists
   upstream; if not, convert v2.1 safetensors with `transcribe-quantize` /
   `app/gguf` during Phase 2 (per V5 §7 step 1 pinning + Appendix C Silero-VAD
   pattern).
3. **`model_specs/sortformer_diar.json` update:** add v2 packages
   (`nvidia/diar_streaming_sortformer_4spk-v2` @ `5240a64`, q8_0), mark v1
   packages legacy, add `stt.vad_version`-style provenance KV
   (`sortformer.model_version = "4spk-v2.1"`) and CC-BY-4.0 license field.
4. **New reference implementation:** NVIDIA's `NeMo-Speech.cpp` can serve as a
   third numerical-parity reference for sortformer (NeMo Python ref ↔
   transcribe.cpp arch/sortformer ↔ NeMo-Speech.cpp), strengthening the golden
   manifests for the DIARIZATION family.

### R9. ggml convergence closed out (2026-08-19)

R1 step 2 (the convergence build) and its CPU-kernel follow-on are done. What
changed against the R1 text above:

1. **All 7 fork-only APIs are converged and CPU-functional.** The R1 inventory
   listed 6 as "must become tracked patches"; they are (patch 0002), and the
   three that had no CPU compute — `ggml_sage_attn2`, `ggml_convrot_linear`,
   `ggml_mul_mat_pack4` — now have CPU kernels (patch 0003), so a graph
   containing them no longer aborts on the CPU backend. Numerical coverage:
   `tests/unittests/test_ggml_fork_ops_cpu.cpp`.
2. **`ggml_convrot_linear` is not "an INT8 linear".** It is a *rotated* INT8
   linear: the CUDA kernel fuses a QuaRot-style radix-4 orthonormal rotation
   into its activation quantizer and the weights ship pre-rotated, so the
   rotation is part of the op's semantics. Any future backend port (Vulkan,
   Metal, HIP) must implement it or it will return confident garbage — the
   shapes and dtypes all check out without it.
3. **`patches/ggml/` is the invariant, not `external/ggml`.** Anything added to
   the vendored tree must land as a tracked patch or `scripts/sync-ggml.sh`
   deletes it on the next run. Reproducibility is defined as "regenerating
   yields an empty `git diff`", not byte-identity: patch-added files land CRLF
   on Windows and are normalized to LF on check-in. See `external/ggml/UPSTREAM`.
4. ~~**Still open from R1:** the GPU half of the convergence build.~~ **Closed by
   R10** — the host does have CUDA 13.3 + an RTX 4070 (sm_89); the earlier note
   was wrong.

### R10. ggml convergence certified on a GPU backend (2026-08-20)

R1 step 2's GPU half is done: the CUDA build links and the full suite runs on an
RTX 4070 (sm_89, CUDA 13.3). `scaled_dot_product_attention_test`, which had never
executed, passes. Both backends now sit at the same documented pre-existing
failure baseline (see `../progress.md`). What running it actually changed:

1. **The three fork-op CUDA kernels are numerically confirmed**, at the error
   levels their quantization implies (normalized RMS vs the CPU references:
   sage_attn2 3.7e-2, convrot_linear 7.4e-3, mul_mat_pack4 6.9e-4). The initial
   failures were all tolerance-model bugs — `tests/unittests/test_ggml_fork_ops_cpu.cpp`
   had been holding activation-quantizing CUDA kernels to the CPU reference
   kernels' bounds. It now carries two named tolerance classes instead.

2. **R9's audit recommendation is superseded by something cheaper.** R9 said the
   tractable approach was to diff audio.cpp's `ggml-cpu` against the upstream pin
   and triage hunks. Not needed: audio.cpp tags its own ggml deltas with
   `MINITTS_*` markers, so `grep -rn "MINITTS_" audio.cpp/external/ggml/{src,include}`
   enumerates the whole class in one command. It found the two remaining drops
   immediately. **Re-run it against any future fork base before anything else.**
   (Corollary: audio.cpp has only 14 commits touching `external/ggml` and none
   touch `binary-ops.cpp` — deltas of this class live in the *initial* vendored
   drop, so reading the fork's commit history does not find them.)

3. **Not every dropped fork delta should be restored.** The fork's
   `MINITTS_FLASH_PER_HEAD_MASK` re-enables per-head flash-attention masks on
   CUDA, which upstream 0.20.2 refuses outright. Porting it removed the abort and
   returned *wrong numbers*: ggml 0.20.2's CUDA kernels write the per-head offset
   into the mask pointer but still read head 0's slice for every head (measured —
   the CUDA output matches a slice-0-forced reference to 4.9e-4 while diverging
   from the true per-head reference by 9.1e-2, and it reproduces at
   `gqa_ratio == 1`). Upstream's restriction is load-bearing. The fix belongs in
   the framework, not in ggml: `use_specialized_flash_attention()` now keeps that
   lowering on CPU and falls through to the reference lowering elsewhere, which
   is also ~390x more accurate on CUDA (8.1e-6 vs 3.2e-3 max abs) because it
   avoids the flash kernels' F32→F16 K/V conversion.

   **Rule this establishes for the rest of the convergence:** when the fork and
   upstream disagree about what a backend supports, the fork is not automatically
   right. Verify numerically before restoring, and prefer a capability check at
   the framework's lowering decision over a patch to `external/ggml`.

4. **The `patches/ggml/` invariant had been broken in practice.** Commit `e11e3c5`
   applied the two-sided broadcast restore in place without a tracked patch, so
   the next `sync-ggml.sh` run would have deleted it. Now tracked as patch 0004
   (with the concat fast path as 0005); regeneration reproduces the tree with an
   empty diff. R9 point 3 stated the invariant correctly — it just was not
   being followed, which is worth checking rather than assuming.

5. **Lean builds were broken and are a supported configuration.** With
   `ENGINE_BUILD_TESTS=ON`, any `AUDIOCPP_MODEL_SET` short of `full` failed to
   link because 11 test targets reference model-internal symbols ungated. Fixed
   per-target against `AUDIOCPP_LINKED_MODELS`. Appendix I's build-verification
   matrix should treat "lean set + tests ON" as a first-class row.

### R11. First end-to-end ASR validation + a fourth dropped fork delta (2026-08-20)

The top blocker after R10 — "no ggml/GGUF ASR model in-tree, end-to-end
transcription unvalidated" — is closed, and closing it immediately paid for
itself by catching another convergence regression.

1. **speech.cpp transcribes.** `tests/asr_e2e_wer_test.cpp` loads a real GGUF
   through `transcribe_open()`, transcribes the four in-tree LibriSpeech
   fixtures, and gates **corpus WER** against the reference transcripts —
   the first test in the merged tree that checks *text*, not plumbing.
   Measured: **1.45% corpus WER** (1 edit in 69 words; the edit is
   FORWARDED→VOTED, consistent with the model's published 4.6% test-clean
   WER), RTF 0.033 on CPU. Gate: ≤10% (one word costs 1.45 pp on this corpus;
   structural breakage lands at 50–100%). See
   `docs/reports/asr_e2e_wer_gate.md`.
2. **The model is pinned, not vendored.** moonshine-tiny Q8_0 (34 MB, MIT,
   the smallest WER-validated GGUF whose arch is in `src/runtime/arch/`) from
   `handy-computer/moonshine-tiny-gguf`, sha256-pinned in
   `scripts/fetch_asr_test_model.py`; `models/` stays gitignored. The CTest
   registration skips (exit 2) while the file is absent, so the gate needs no
   reconfigure once fetched. This is the Phase-0-scale stand-in for the plan's
   full golden pipeline (§3 Stage 7); per-family WER corpora still land with
   their family ports.
3. **A fourth dropped fork delta, found by the new gate's sibling.** With a
   trustworthy end-to-end path in hand, `flashsr_utility_test`'s "numeric
   tolerance" failure (NEXT item 3 in progress.md) was re-examined: it fails
   at 1.7e-3 max **identically on CPU and CUDA** against onnxruntime-generated
   fixtures, and the same test against audio.cpp's fork tree passes on the
   same machine and compiler. Cause: upstream 0.20.2's
   `ggml_conv_1d`/`ggml_conv_1d_dw`/`ggml_conv_2d` force im2col activations to
   F16 (unless the weight is BF16); the fork keeps them in the weight's type,
   and every framework conv weight is F32. Restored as
   `patches/ggml/0006-conv-im2col-in-weight-type.patch`;
   `flashsr_utility_test` is green on both backends for the first time since
   the convergence. (`ggml_conv_2d_dw` deliberately untouched — the fork kept
   upstream's F16 there.)
4. **R10 point 2 is corrected, not just extended: the `MINITTS_*` grep is
   necessary but NOT sufficient.** This delta — like 0004 — was unmarked in
   the fork. The audits that catch the unmarked class are numeric golden
   gates with implementation-independent references (onnxruntime fixtures
   here; the WER gate for ASR), and a hunk-by-hunk diff of the fork tree
   against its own upstream base. The convergence's residual risk is now
   *only* what those two audits have not yet covered.

## 0. Vision Statement

**Goal:** Merge `transcribe.cpp` fully into `audio.cpp` (via the `speech.cpp`
fork) so that the resulting project is the definitive locally-runnable audio
processing framework — combining:

- `audio.cpp`'s broad multi-task audio engine (ASR, TTS, voice cloning, VAD,
  diarization, source separation, audio generation, voice conversion, speech-to-speech,
  alignment, MIDI) and its rich C++ task/session/graph abstraction built on ggml,
  with a modular composite build system (`full` / `core` / `custom`).
- `transcribe.cpp`'s disciplined, production-grade STT architecture: a clean
  **public C ABI** (`transcribe.h`, 2499 lines) with size-aware version tolerance
  (`struct_size` + `transcribe_abi_struct_size()`), **per-family trait dispatch**
  (the `Arch` struct-of-function-pointers in `transcribe-arch.h`), **family
  extensions** with FourCC-kind validation, **spec decoding**, **PNC/ITN/translation**,
  **long-form chunking with cancellation** (abort callbacks polled between decode
  steps), **multiple streaming strategies** (cache-aware, buffered, Voxtral realtime),
  and a **golden-manifest + per-tensor tolerances** testing methodology with an
  **8-stage porting pipeline**.

**Non-goals:** We do not replace audio.cpp's web UI, server, or workflow engine.
We extend them. We do not rewrite the ggml backend. We standardize on a single
vendored ggml version (see §0.6 for the concrete pinning strategy).

**Core principle — bidirectional upgrade:** The merge is not a one-way port. It is
a bidirectional upgrade in which the unified codebase learns from the best of both
projects and upgrades both sides simultaneously. transcribe.cpp's families gain
audio.cpp's graph executor, artifact store, pipeline engine, and WebUI; audio.cpp's
existing families gain transcribe.cpp's C ABI, exception discipline, safe teardown,
golden-manifest testing, streaming state machine, and spec decoding. Every phase
of the plan upgrades the underlying infrastructure for ALL families, not just the
new STT ports. See §1.5 for the full cross-pollination matrix.

**Success criteria (in priority order):**
1. Every STT model family in `transcribe.cpp` runs inside `speech.cpp` with
   WER parity to the upstream `transcribe.cpp` reference.
2. `speech.cpp` exposes a **public C ABI** mirroring `transcribe.h` so that
   Python/TypeScript/Rust/Swift bindings can be rebased onto it.
3. The project ships with the **golden-manifest validation pipeline** from
   `transcribe.cpp`, so every family port is gated by per-tensor tolerances.
4. The build remains modular: users can build `core`, an `asr` composite, an
   `asr+full` composite, or the `full` composite (audio.cpp + transcribe families).
5. **Every existing audio.cpp family** (TTS, voice cloning, source separation, voice
   conversion, speech-to-speech, alignment, speaker recognition, Svc, MIDI) has been
   upgraded with transcribe.cpp's hard-won infrastructure: `safe_*` teardown (CI-gated
   by `lint_teardown.cmake`), `api_guard_*` exception containment at C ABI boundaries,
   abort callbacks for cancellation, and golden-manifest validation where applicable.
6. The C ABI exposes **all** audio.cpp task types (not just STT) — TTS, voice cloning,
   voice conversion, source separation, alignment, diarization, speaker recognition —
   through generalized entry points.
7. Existing audio.cpp code that duplicates transcribe.cpp functionality (mel frontend,
   kaldi fbank, tokenizer interface, meta/KV readers, debug dumps) has been
   **consolidated**, not duplicated — the transcribe.cpp implementation wins where it
   is strictly better, and the audio.cpp implementation is retained only for non-STT
   tasks that need domain-specific behavior.

---

## 0.1 Executive Summary (Quick-Start Guide)

**What this is:** A merger of two complementary C++ audio inference projects into a
single `speech.cpp` codebase. `transcribe.cpp` contributes a production-grade
C ABI, disciplined testing, and 18 STT model families. `audio.cpp` contributes a
rich C++ task/session framework, a modular build, and 40+ model families covering
ASR, TTS, voice cloning, source separation, and more.

**How it works:** The transcribe.cpp `Arch` struct (function-pointer dispatch)
becomes the unified low-level registry. An `ArchAdapter` wraps audio.cpp's heavier
3-level C++ vtable (`IVoiceModelLoader` → `ILoadedVoiceModel` → `IVoiceTaskSession`)
into the same `Arch` shape. A central C dispatcher (ported from transcribe.cpp's
`src/transcribe.cpp`) handles `api_guard_*` exception containment, `struct_size`
ABI validation, and result access. Both the existing audio.cpp C++ API (CLI, server,
WebUI) and the new C ABI coexist, pointing at the same concrete family implementations.

**Key architectural decisions (finalized in §0.3):**
- C ABI is the primary public surface; C++ headers become internal/private API.
- `Arch` is the unified low-level dispatch; `ArchAdapter` bridges audio.cpp families.
- transcribe.cpp's `Loader` handles the ABI path; audio.cpp's `ModelRegistry` stays for the C++ path.
- `causal_lm` module adopted as the shared STT causal-LM backbone; audio.cpp's broader attention modules remain for TTS/voice-cloning.
- Single vendored ggml pinned to transcribe.cpp's SHA `8c63e70982c95ceb862e3a1073a2c1beef75d60a` (see §0.6).
- Silero VAD version to be pinned from upstream (see §0.4).

**Phase summary:**
- **Phase 0** (Foundation): Port the C ABI dispatcher + `ArchAdapter` + safe teardown +
  golden manifests for one existing family + ggml convergence. ~4-6 weeks (see §0.7).
- **Phase 1** (P0 families): Consolidate overlapping qwen3_asr/sensevoice/funasr +
  port Whisper + Moonshine. ~10 weeks.
- **Phase 2** (P1 families): Parakeet, Canary, GigaAM, Granite, Cohere, Voxtral,
  Sortformer, Moonshine-Streaming. ~16 weeks.
- **Phase 3** (P2 families + advanced features): MedASR, Moss, spec decoding,
  PNC/ITN/translation, long-form chunking. ~6 weeks.
- **Phase 4** (Bindings): Python (ctypes), TypeScript (koffi), Rust, Swift. ~8 weeks.
- **Phase 5** (CLI/Server/WebUI): Unified CLI, REST API, WebUI updates. ~6 weeks.
- **Phase 6** (Testing maturity & release): CI gates, full ctest coverage. Ongoing.

**Start here:** Read §0.3 (Decision Log) for finalized architectural decisions, then
§0.6 (ggml Unification) for the critical path, then §7 (Immediate Next Steps) for the
Phase 0 kickoff checklist.

---

## 0.2 Codebase Inventory 

### transcribe.cpp (the STT project) — `transcribe.cpp/`

**Build entry point:** `CMakeLists.txt`. Version is sourced from
`include/transcribe.h` (`TRANSCRIBE_VERSION_MAJOR/MINOR/PATCH` macros, currently
`0.2.0`). Static library by default; shared library (`TRANSCRIBE_BUILD_SHARED=ON`)
for Python wheels. OpenMP defaults OFF (ggml native threadpool); BLAS for the host
decoder defaults ON on Apple (Accelerate), OFF for official wheels.

**Source tree (`src/`):** 6 top-level directories + flat runtime files:
- `src/arch/` — 18 family subdirectories (whisper, moonshine, moonshine_streaming,
  qwen3_asr, sensevoice, funasr_nano, parakeet, voxtral, voxtral_realtime, canary,
  canary_qwen, gigaam, granite, granite_nar, cohere, medasr, moss, sortformer)
- `src/causal_lm/` — `causal_lm.h/.cpp` shared causal-decoder transformer block
- `src/conformer/` — `conformer.cpp` shared conformer encoder
- `src/sanm/` — `sanm.cpp` State-Adaptive Normalization blocks
- `src/granite_conformer/` — `shaw_attn.cpp` Granite's Shaw relative attention
- `src/third_party/miniz/` — vendored deflate for Whisper compression-ratio heuristic
- Flat runtime files (37 files: 19 `transcribe-*` modules as `.h/.cpp` pairs plus
  `transcribe.cpp` dispatcher, `transcribe-log.h`/`transcribe-path.h`/
  `transcribe-session.h` headers, and `transcribe-unicode-data.cpp` data TU):
  `transcribe.cpp` (dispatcher), `transcribe-arch.{h,cpp}` (registry),
  `transcribe-loader.{h,cpp}` (GGUF header), `transcribe-backend.{h,cpp}` (BackendPlan
  + safe_*), `transcribe-model.{h,cpp}` (base model), `transcribe-session.{h,cpp}`
  (base session), `transcribe-abi.h` (struct_size/copy_out_prefix),
  `transcribe-meta.{h,cpp}` (KvResult tri-state), `transcribe-mel.{h,cpp}`,
  `transcribe-kaldi-fbank.{h,cpp}`, `transcribe-tokenizer.{h,cpp}`,
  `transcribe-batch-util.{h,cpp}`, `transcribe-flash-policy.{h,cpp}`,
  `transcribe-log.h`, `transcribe-debug.{h,cpp}`, `transcribe-load-common.{h,cpp}`,
  `transcribe-env.{h,cpp}`, `transcribe-path.{h,cpp}`, `transcribe-weights-util.{h,cpp}`,
  `transcribe-bin-loader.{h,cpp}`, `transcribe-unicode.{h,cpp}` +
  `transcribe-unicode-data.cpp`.

**Public C ABI (`include/transcribe.h`, 2499 lines):**
- `transcribe_status` enum: 19 codes (0=OK through 18=OUTPUT_TRUNCATED). Codes 9
  (SAMPLE_RATE) and 15/16 (UNSUPPORTED_PNC/UNSUPPORTED_ITN) are reserved.
- `transcribe_feature` enum: 7 values (0=INITIAL_PROMPT through 6=DIARIZATION).
- `transcribe_abi_struct` enum: 15 values (0=ABI_MODEL_LOAD_PARAMS through
  14=ABI_SPEAKER_SEGMENT). All cross-ABI structs carry `uint64_t struct_size` as field 0.
- `transcribe_backend_request` enum: AUTO/CPU/METAL/VULKAN/CPU_ACCEL/CUDA/ROCM.
- `transcribe_backend_kind` (internal BackendKind enum): Unknown/Cpu/Metal/Vulkan/
  Cuda/Rocm/Sycl/Accel/OtherGpu (9 + Unknown).
- `transcribe_backend_request` → `transcribe_backend_info` (device enumeration).
- 4 input param structs (each with `_init` factory):
  `transcribe_model_load_params`, `transcribe_session_params`, `transcribe_run_params`,
  `transcribe_stream_params`.
- 10 output structs (each with `_init` factory): `transcribe_capabilities`,
  `transcribe_session_limits`, `transcribe_timings` (4 fields: load_ms, mel_ms,
  encode_ms, decode_ms), `transcribe_segment`, `transcribe_word`, `transcribe_token`,
  `transcribe_speaker_segment`, `transcribe_stream_update`, `transcribe_stream_text`,
  `transcribe_device_info`.
- Full entry-point set in Appendix A .
- `transcribe_open()` / `transcribe_close()` / `transcribe_get_model()` convenience lifecycle.
- `transcribe_stream_get_text()` returns `full_text` / `committed_text` /
  `tentative_text`, plus byte-count fields (`full_text_bytes`,
   `committed_text_bytes`, `tentative_text_bytes`, `raw_tentative_start_bytes`)
   (the committed/tentative streaming model). A `display_text`
   field does not exist in the actual struct.
- `transcribe_log_callback` + `transcribe_log_level` (NONE/INFO/WARN/ERROR/DEBUG/CONT).
- `transcribe.abihash` — ABI version hash for binding pinning.
- `include/transcribe/` subdirectory — 5 family extension headers + 1 umbrella. Each
  extension struct's first field is `struct transcribe_ext { uint64_t size; uint32_t kind; }`
  (the `kind` is a FourCC). `transcribe_ext_check()` validates size+kind;
  `transcribe_model_accepts_ext_kind()` gates per-model acceptance. `arch->accepts_ext_kind()`
  (optional Arch hook) lets families declare which kinds they recognize. Verified families:

  | Header | FourCC `kind` | Extension struct | Optional fields | Family-specific enum |
  |--------|--------------|-----------------|-----------------|---------------------|
  | `whisper.h` | `'WHRN'` | `transcribe_whisper_run_ext` | 13 fields (initial_prompt, prompt_tokens, ..., max_initial_timestamp) | `transcribe_whisper_prompt_condition` (FIRST_SEGMENT/ALL_SEGMENTS) |
  | `parakeet.h` | `'PKST'` | `transcribe_parakeet_stream_ext` | `att_context_right` | — |
  | `parakeet.h` | `'PKBS'` | `transcribe_parakeet_buffered_stream_ext` | `left_ms`, `chunk_ms`, `right_ms` | — |
  | `moonshine_streaming.h` | `'MSST'` | `transcribe_moonshine_streaming_stream_ext` | `min_decode_interval_ms` | — |
  | `sortformer.h` | `'SFST'` | `transcribe_sortformer_stream_ext` | `preset` | `transcribe_sortformer_preset` (DEFAULT/HIGH_LATENCY/VERY_HIGH_LATENCY/LOW_LATENCY) |
  | `voxtral_realtime.h` | `'VRST'` | `transcribe_voxtral_realtime_stream_ext` | `num_delay_tokens`, `min_decode_interval_ms` | — |

  Whisper also exposes chunk-trace telemetry (`transcribe_whisper_chunk_trace`,
  `transcribe_get_whisper_chunk_count`) for debugging fallback behavior.

**Testing (`tests/`):** 50 source TUs (48 `.cpp` + 2 `.c`) + CMake-based test scripts. Key infrastructure:
- `tests/fixtures/` — synthetic GGUF fixtures generated at build time via `uv`
  (`fixtures/make_gguf_fixtures.py`). Tests skip (RC 77) when `uv` is absent.
- `tests/golden/` — golden manifest JSON files (committed contracts).
- `tests/tolerances/` — per-tensor tolerance JSON files.
- `tests/cmake/` contains `ep-prefix-parent/` (Windows MAX_PATH mitigation).
- CI gates in `tests/`: `lint_teardown.cmake` (forbids raw
  `ggml_backend_free`/`ggml_backend_buffer_free`/`ggml_backend_sched_free`),
  `check_extension_umbrella.cmake` (extension headers include only `transcribe.h`),
  `cli_device_arg_smoke.cmake`, `cli_output_smoke.cmake`.
- `api_smoke.c` is compiled as pure C11 (not C++) to double as a "header is C-clean" canary.
- `TRANSCRIBE_TEST_DEV_INIT_THROW` / `TRANSCRIBE_TEST_TEARDOWN_THROW` fault hooks
  ship in release artifacts for wheel clean-install CI (per `AGENTS.md`).
- `transcribe-common-example` static library (in `examples/common/`) provides the
  WAV loader shared by example binaries and some tests.

**Bindings (`bindings/`):**
- **Python** (`bindings/python/`): **ctypes-based** FFI (the `pyproject.toml` states
  "The binding is ctypes-only, so there is no per-version compiled extension — one
  pure-Python package spans the range"). Pure-Python API package `transcribe-cpp` (v0.2.0)
  hard-depends on `transcribe-cpp-native` provider wheels (CPU+Metal on macOS arm64,
  CPU+Vulkan on Linux/Windows); opt-in `transcribe-cpp-native-cu12` CUDA 12 provider.
  ABI layout verified at import via `transcribe_abi_struct_size()` /
  `transcribe_abi_struct_align()` cross-checked against libclang-captured layouts.
  Provider selection: explicit `provider=` → `TRANSCRIBE_NATIVE_PROVIDER` env →
  best accelerated → CPU.
- **TypeScript** (`bindings/typescript/`): **koffi**-based FFI. `src/_generated.ts`
  has ABI sizes; `src/ffi.ts` hand-binds each C function with explicit in/out/inout
  direction; `PUBLIC_HEADER_HASH` drift gate. `package.json` for bundling.
- **Rust** (`bindings/rust/`): cbindgen-based bindings.
- **Swift** (`bindings/swift/`): Swift Package Manager.
- CI: `python-bindings.yml`, `typescript-ci.yml`, `rust-ci.yml`, `swift-ci.yml`,
  `python-wheels.yml`, `wheel-index.yml`.

**Tools (`tools/`) and examples (`examples/`):** `tools/transcribe-bench` and
`tools/transcribe-quantize` (the quantize tool is required by porting Stage 5 —
port it alongside the CLI). Examples: `bench`, `cli`, `common`, `hello`,
`hello_stream`. **There is no server binary** — any unified REST/WebSocket
surface extends audio.cpp's `app/server` (see Phase 5b).

**Tokenizer independence :** transcribe.cpp has **no sentencepiece
dependency**. `src/transcribe-tokenizer.{h,cpp}` reads `tokenizer.ggml.*` keys
directly from GGUF and implements three flavors: `unigram` and `bpe`
(SentencePiece conventions, U+2581 → space on decode) and `gpt2` (Hugging Face
byte-level BPE with encode support). This keeps the ported C ABI runtime free of
audio.cpp's `external/sentencepiece` and `external/llama_tokenizer` trees, and
gives the STT path a path to shedding sentencepiece entirely once family ports
are consolidated.

**Agent conventions:** `AGENTS.md` + `CLAUDE.md` codify repo discipline —
`uv run` for all Python, per-family reference envs at `scripts/envs/<family>/`,
pinned clang-format via `scripts/ci/clang-format.sh` (vendored trees excluded),
and the C ABI exception rules (every public entry point routes through
`api_guard_*` or is nothrow by construction; "non-OK ⇒ `*out == NULL`, nothing
leaked"). Port/adapt these into the unified repo at Phase 0 (see §0.9).

**Documentation (`docs/`):** `porting/0-porting.md` (8-stage pipeline), `1-reference-research.md`,
`1a-intake.md` (intake packet + `_intake-schema.json`), `2-artifacts-and-goldens.md`
(manifest format + cache keys), `3-conversion.md`, `4-numerical-validation.md`
(dump principles, gate/informational/debug tensor roles), `4a-numerical-troubleshooting.md`,
`5-benchmarks.md`, `6-family-checklist.md` (family readiness). Also: `bindings.md`,
`build-windows.md`, `environment-variables.md`, `extension-kinds.md`, `input-limits.md`,
`migrating-to-0.2.md`, `model-family-testing.md`.

**Validation tooling (`scripts/`):** `validate.py` (ref→cpp→compare pipeline),
`preflight.py` (cheap metadata gates: dtype, frontend config, tokenizer IDs,
capabilities cross-checks), `compare_tensors.py` (per-tensor comparison with
`tests/tolerances/<family>.json`), `bench/run.py` + `bench/compare.py`,
`intake.py` (reference research), `convert-<family>.py` (GGUF converters),
`dump_reference_<family>_<reference>.py` (reference dumpers),
`generate_transcribe_abi.py` (FFI type generation from the C header via libclang),
`ci/clang-format.sh`, `ci/check_lane_mirror.py`, `ci/link_smoke.c`/`.py`,
`ci/vulkan_degradation_check.py`.

### audio.cpp (the full audio engine) — `audio.cpp/`

**Build entry point:** `CMakeLists.txt` (2000+ lines). `audiocpp_add_model()` creates
per-family OBJECT libraries; `AUDIOCPP_MODEL_SET` selects `full`/`core`/`custom`.
ggml vendored at `external/ggml/` (no UPSTREAM pin; CMake labels it `0.12.0`, but
the tree already carries the scheduler and device/registry APIs — see §0.6).
`engine_core` OBJECT library holds the framework; `engine_runtime` STATIC library
aggregates core + selected models. Applications live in `app/`:
`cli`, `gguf` (GGUF packer/inspector tool), `server`, `streaming`, `workflow` —
this is the surface Phase 5 extends.

**Source tree (`src/`):** `src/framework/` (14 subdirs: assets, audio, codecs, core,
debug, decoders, io, midi, model_spec, modules, runtime, sampling, text, tokenizers) +
`src/models/` (39 family dirs) + `src/community_models/` (8 dirs: glm_tts, inflect_v2,
kroko_asr, minimax_h3, outetts, parakeet_tdt, sense_asr, vietneu_tts).

**Architecture:** 3-level C++ vtable dispatch:
```
ModelRegistry::load(ModelLoadRequest)
  → IVoiceModelLoader::load() → ILoadedVoiceModel
  → ILoadedVoiceModel::create_task_session(TaskSpec{Asr, Offline})
      → IVoiceTaskSession (base: family(), task_kind(), run_mode(), prepare())
        → IOfflineVoiceTaskSession::run(TaskRequest) → TaskResult
        → IStreamingVoiceTaskSession::start_stream / process_audio_chunk / finish_stream / reset
```

Key types (from `include/engine/framework/runtime/`):
- `VoiceTaskKind` enum (14 values): Vad, Asr, Diarization, SourceSeparation,
  AudioGeneration, Tts, VoiceCloning, VoiceConversion, SpeechToSpeech, Alignment,
  VoiceDesign, SpeakerRecognition, Svc, Midi.
- `RunMode` enum: Offline, Streaming.
- `TaskSpec { VoiceTaskKind task, RunMode mode }`
- `SessionOptions { BackendConfig backend, unordered_map<string,string> options }`
- `TaskRequest { optional<Transcript>, optional<AudioBuffer>, optional<VoiceCondition>,
  vector<VoiceArtifact>, options map }`
- `TaskResult { optional<AudioBuffer> audio_output, vector<NamedAudioBuffer>,
  optional<Transcript> text_output, vector<SpeechSegment>, vector<SpeakerTurn>,
  vector<WordTimestamp>, optional<VoiceArtifact>, vector<VoiceArtifact> }`
- `StreamEvent`, `StreamingPolicy`, `StreamEventCallback` — dual push/pull streaming:
  push via `set_stream_event_sink(callback)` OR pull via `next_stream_event()`
  (returns `optional<StreamEvent>`). Also `finalize()`. No explicit state enum.
- `CapabilitySet { vector<TaskCapability>, vector<string> languages,
  supports_speaker_reference, supports_style_condition, supports_timestamps }`
- `BackendConfig { BackendType type, int device, int threads }` — **NOTE: simpler
  than the original plan claimed; does NOT carry `gpu_split_mode`, `n_gpu_layers`,
  `tensor_split`, or `gpu_mode`.**
- `BackendType` enum (in `core/module.h`): Cpu, Cuda, Hip, Vulkan, Metal,
  BestAvailable. **No SYCL or BLAS as first-class types** (those exist only in
  ggml's vendored backends).
- `RuntimeSessionBase` (in `runtime/session_base.h`) — concrete base owning
  ExecutionContext, ArtifactStore, RuntimeCache, GraphExecutor, Workspace.
- `ModelRegistry` (in `runtime/registry.h`) — `make_default_registry()` builds from
  CMake-generated `model_registry_includes.inc` / `model_registry_loaders.inc`.
- `VoiceArtifact` / `ArtifactKind` — structured inter-task artifacts (SpeakerEmbedding,
  StyleEmbedding, PromptEmbedding, AcousticTokens, AudioTokens, Midi,
  TranscriptAlignment, DiarizationState, VadState, Custom).

**audio.cpp ASR families** (in `include/engine/models/` or `src/community_models/`):
`qwen3_asr` (thinker/thinker_runtime, audio_encoder, whisper frontend, prompt_asr,
postprocess, tokenizer), `fun_asr_nano`, `nemotron_asr`, `voxtral_realtime`,
`higgs_audio_stt`, `vibevoice_asr`, `hviske_asr`, `citrinet_asr`, `parakeet_tdt`,
`sense_asr` (SenseVoice, in community_models), `qwen3_forced_aligner`.

**audio.cpp VAD family:** `silero_vad` (in `src/models/silero_vad/`, assets at
`assets/framework/models/silero_vad/silero_vad_16k.safetensors`, reference Python at
`reference/silero-vad/`). `marblenet_vad` also exists.

**audio.cpp TTS/other families:** `chatterbox`, `qwen3_tts`, `dia` (in `model_specs/`
but not a built model), `stable_audio`, `roformer`, `demucs`, `rvc`, `seed_vc`,
`fish_audio`, `heartmula`, `index_tts2`, `miotts`, `miocodec`, `omnivoice`,
`pocket_tts`, `supertonic`, `vevo2`, `vibevoice`, `confucius4_tts`, `dots_tts`,
`dramabox`, `ace_step`, `muscriptor`, `neutts`.

**audio.cpp model spec system:** `model_specs/*.json` (current) + `model_specs_v1/`
(legacy/v1 format). The CMake globs `model_specs/*.json` and generates
`model_registry_includes.inc` / `model_registry_loaders.inc`. Model specs define
download sources, variant mappings, and UI metadata. The converter uses specs to know
which tensors to export and where to find configs.

**audio.cpp text post-processing:** `src/framework/text/text_normalization.cpp` — full
pipeline (Chinese, English, unicode normalization, chunking, subtitle formatting).

**audio.cpp testing (`tests/`):** 51 subdirectories + 2 files (53 items total)ies (one per family) + `unittests/`.
No golden manifests, no per-tensor tolerances, no numerical parity gates in the
transcribe.cpp sense. Tests are ad-hoc family-specific smoke/parity tests.

**audio.cpp CI:** `.github/workflows/` — standard GitHub Actions for build/test on
Linux/macOS/Windows.

### speech.cpp (the fork) — `speech.cpp/`

**Current state:** Confirmed byte-identical to `audio.cpp/` (diff-verified
excluding `.git`; same 16 top-level dirs + 10 root files). The `src/` tree is
identical: `src/framework/` (14 subdirs) + `src/models/` (39) +
`src/community_models/` (8) — all match audio.cpp exactly.

**Git state :** `speech.cpp/.git` is already initialized and pushed:
`origin = https://github.com/NairoDorian/speech.cpp.git` (the fork) and
`upstream = https://github.com/0xShug0/audio.cpp.git` (the parent) are both
configured. Fork-base commit = audio.cpp HEAD `52080cd` ("Fix IndexTTS2/2.5 text
normalization issues (#259)"). The transcribe.cpp merge source is at HEAD
`923d4a0` ("upgrade ggml to 0.20.2 (#131)", v0.2.0). Both hashes are recordable
in `docs/porting/reference_commits.md` immediately — Phase 0 sub-task 0.A is a
5-minute file write, not a remote-setup task (§7 step 1 updated accordingly).

**Build system:** The fork inherits audio.cpp's CMakeLists.txt unchanged. The
unification modifies this file to add transcribe-runtime objects, the `ArchAdapter`,
the new `asr`/`asr+full` composites, and the `engine_transcribe_runtime` OBJECT library.

### model_specs vs model_specs_v1
Both projects have `model_specs/` and `model_specs_v1/` directories. In audio.cpp,
`model_specs/*.json` is the active spec format (globbed by CMake). `model_specs_v1/`
appears to be a legacy/v1 schema kept for backward compatibility with older model
specs. The unified project should consolidate to a single spec format during Phase 0.

---

## 0.3 Decision Log (Architectural Decisions — Finalized)

This log records irreversible architectural decisions agreed upon by this plan.
Each decision includes its rationale and the section that elaborates it. Decisions
here are binding on all implementation phases.

| # | Decision | Rationale | Elaborated in |
|---|----------|-----------|---------------|
| D1 | C ABI is the primary public surface; C++ headers become internal/private API. | C ABI is the lowest common denominator (consumable by C, C#, Go, Python, Rust, Swift, WebAssembly). A C++ ABI cannot. The size-aware struct pattern is strictly more robust for versioning. | §1.5.1 row: Public API surface |
| D2 | `Arch` (struct-of-function-pointers) is the unified low-level dispatch; `ArchAdapter` wraps audio.cpp's 3-level C++ vtable. | `Arch` is simpler and is the ABI surface. The vtable is richer but heavier. Bridge them rather than forcing existing families to be rewritten into the Arch pattern. | §1.1, §1.2 |
| D3 | transcribe.cpp's `Loader` handles the ABI-path model loading; audio.cpp's `ModelRegistry` stays for the C++ path. | The Loader's architecture-agnostic GGUF inspection is the correct lower layer. The model spec system provides package management that transcribe.cpp lacks. The `ArchAdapter::load` consults the model spec to find the right `IVoiceModelLoader`. | §1.5.1 row: Model loading |
| D4 | Hybrid session model: lightweight `transcribe_session` for the C ABI path; `RuntimeSessionBase` for the C++ path. The `ArchAdapter` wraps a `transcribe_session` that may or may not hold a `RuntimeSessionBase` depending on whether the underlying model needs it. | STT families don't need the full ExecutionContext/ArtifactStore/GraphExecutor per session. Let lightweight families be lightweight; heavy families (TTS, vocoder) still get the full runtime. | §1.5.1 row: Session creation |
| D5 | transcribe.cpp's `BackendPlan` + dynamic backend module loading (`TRANSCRIBE_GGML_BACKEND_DL`) becomes the C ABI surface. `BackendConfig` fields feed `BackendPlan` construction. | Dynamic backend modules (DLLs next to the library) is a major distribution win. Adopt this. | §1.5.1 row: Backend selection |
| D6 | transcribe.cpp's `transcribe_status` enum model for the C ABI. `api_guard_*` contains all C++ exceptions at the boundary. `api_guard_*` also applied at C++ boundaries (CLI main, server handler) as defense-in-depth. | Exceptions across FFI boundaries are undefined behavior. | §1.5.2 item 1, AGENTS.md |
| D7 | Port `safe_*` teardown + `lint_teardown.cmake`. Retroactively replace ALL raw ggml teardown calls in audio.cpp's model families. The lint scans all C++ source and fails on any raw `ggml_backend_free` / `ggml_backend_buffer_free` / `ggml_backend_sched_free`. `ggml_backend_graph_plan_free` is NOT in scope (it frees a graph plan, not a backend handle). | Production-quality teardown; the CI gate ensures the pattern is never broken. | §1.5.2 item 2, §1.5.6 item 1 |
| D8 | Size-aware ABI structs (`struct_size` as `uint64_t` field 0 + `copy_out_prefix` for output structs). | Forward/backward ABI compatibility without silent corruption. `check_struct_size()` rejects `struct_size < want` (including `== 0`). `copy_out_prefix()` writes `min(caller_size, lib_size)`. | §1.5.2 item 3 |
| D9 | Adopt transcribe.cpp's full golden-manifest + per-tensor tolerances testing methodology for ALL families (existing audio.cpp + new ports). Tolerances sized as `1e-4 × p99_abs` / `1e-5 × rms`. | Prevents regressions in a 49+ family codebase. Without numerical parity gates, WER regressions slip through. | §1.5.2 item 5, §6 |
| D10 | transcribe.cpp's streaming state machine (IDLE/ACTIVE/FINISHED/FAILED) with 3-checkpoint begin (preflight → optional `stream_validate` → post-clear `stream_begin`) becomes the unified streaming lifecycle. Applied to ALL streaming tasks (ASR, TTS streaming, VC, STS). | Prevents config typos from destroying transcripts. | §1.5.2 item 4 |
| D11 | Adopt the size-aware family extension struct pattern (`struct transcribe_ext { size, kind }`). `options` map becomes C++-internal fallback; typed extensions are the C ABI surface. | Forward-compatible extension mechanism across ABI boundaries. | §1.5.1 row: Family options |
| D12 | Port the spec-decode sampler from transcribe.cpp. Gate behind `supports_spec_decode`. Applied to ALL autoregressive decoder families (whisper, canary, cohere, voxtral, TTS token predictors). | Pure performance win. | §1.5.1 row: Speculative decoding |
| D13 | Merge text post-processing: audio.cpp's `text_normalization.cpp` pipeline for PNC/ITN implementation; expose as runtime toggles via C ABI enums (`transcribe_pnc_mode`, `transcribe_itn_mode`). | audio.cpp's text normalization is more complete (Chinese + English). | §1.5.1 row: Text post-processing |
| D14 | Adopt transcribe.cpp's `MelFrontend` + `KaldiFbankFrontend` as the unified STT frontend. Port both into `src/runtime/`. Audio.cpp's `kaldi_fbank.cpp` stays as fallback but new STT families use the transcribe.cpp versions. | transcribe.cpp's frontend is strictly more configurable (4 normalize modes, 3 pad modes, hann variants, LFR+CMVN). | §1.3, §1.5.1 row: Mel/frontend |
| D15 | Adopt transcribe.cpp's `causal_lm` module as the shared STT causal-LM backbone in `src/runtime/causal_lm/`. Audio.cpp's broader attention/transformer modules remain for TTS/voice-cloning/generation tasks. | Lighter and purpose-built; the `BlockView` struct's nullable-pointer projection makes family porting easier. | §1.4 |
| D16 | Standardize on a single vendored ggml version: transcribe.cpp's pinned SHA `8c63e70982c95ceb862e3a1073a2c1beef75d60a` with its threadpool-oversubscription patch. | Prevents ggml ABI/API drift between codebases. The transcribe.cpp pin is newer and has patches audio.cpp's vendored ggml lacks. | §0.6 |
| D17 | Pin Silero VAD to latest stable upstream release (latest release from `https://github.com/snakers4/silero-vad`). | Reproducible VAD numerical parity; version-tagged test fixture. | §0.4 |
| D18 | Generalized C ABI exposes ALL task types (not just STT) via `transcribe_task_*` entry points with size+kind tagged request/result structs. STT path (`transcribe_run`) remains the original transcribe.h API — not broken. | Enables a single unified surface for bindings, server, and embedded callers. | §1.5.7 |
| D19 | The `transcribe-arch.cpp` registry is CMake-generated, accumulating both native transcribe.cpp families AND `ArchAdapter` instances for audio.cpp families. | Single unified registry; both paths point at the same concrete implementations. | §4.9, Appendix D |
| D20 | No `-Werror` in the unified C++ build (matching transcribe.cpp's policy). Warnings via `transcribe_apply_warnings()` per-target; vendored trees exempt. | Prevents CI breakage from new compiler warnings on vendored code. | §6.3 |
| D21 | The `transcribe-common-example` static library (WAV loader) is ported from `transcribe.cpp/examples/common/` into the unified `examples/common/` and used by tests requiring WAV loading. | Shared by example binaries and test binaries; avoids duplicating dr_wav integration. | Appendix C |
| D22 | Adopt transcribe.cpp's CMake warning/visibility policy (`transcribe_apply_warnings`) for unified runtime code; keep audio.cpp's warning flags for existing framework code. | transcribe.cpp's per-target warning function is cleaner than audio.cpp's global `add_compile_options`. | §4.4 |
| D23 | C ABI symbol names stay `transcribe_*`. No renaming to `audiocpp_*`/`ua_*` despite the unified repo being named `speech.cpp`. The shared library keeps the `transcribe` output name. | The existing bindings pin `transcribe.abihash` / `PUBLIC_HEADER_HASH` and hard-bind `transcribe_*` symbols; a rename breaks every downstream consumer (Python wheels, npm, crates, SPM) for zero functional gain. Repo name and ABI prefix are deliberately decoupled. | §0.9 |
| D24 | Family-name collision policy: ported transcribe.cpp families land in `src/arch/<family>/` and keep their GGUF `general.architecture` strings verbatim (the ABI contract). Where audio.cpp already has a near-colliding family name, the *model spec / registry display* name disambiguates: transcribe.cpp's `moss` ASR family ships as `model_specs/moss_asr.json` (GGUF arch stays `moss`) vs audio.cpp's existing `moss_tts_local`/`moss_tts_nano`; transcribe.cpp's `sortformer` vs audio.cpp's `sortformer_diar` resolve per §2.4 consolidation (one canonical implementation, spec name `sortformer_diar` retained for download compatibility). | GGUF arch strings are immutable ABI (convertors already emit them); spec names are ours to control and must stay collision-free in the registry/UI. | §2.1, §2.4 |
| D25 | Upstream sync policy: vendor-snapshot model with `docs/porting/reference_commits.md` as the tracking ledger. Re-sync audio.cpp upstream (active — PR #259 at fork time) and transcribe.cpp upstream at the START of each phase; freeze upstream sync for a tree once its phase lands; hard-freeze both at Phase 4 (ABI/binding stabilization). | Both parents are actively developed; drifting forever means unbounded merge debt. Phase-bounded sync windows bound the pain while keeping the unified tree young enough to absorb fixes. | §0.9 |

---

## 0.4 Pinned Dependencies (Version-Tracked)

All upstream dependencies are pinned to specific, reproducible versions with their
canonical GitHub source URLs. The merged codebase records these in
`docs/porting/reference_commits.md` and enforces them via CI.

| Dependency | Pinned To | Source URL | Recorded In |
|-----------|-----------|------------|-------------|
| **audio.cpp** (fork base) | HEAD at fork creation | `https://github.com/0xShug0/audio.cpp` | `docs/porting/reference_commits.md` |
| **transcribe.cpp** (merge source) | `handy-computer/transcribe.cpp` @ v0.2.0 release | `https://github.com/handy-computer/transcribe.cpp` | `docs/porting/reference_commits.md` |
| **ggml** (unified) | SHA `8c63e70982c95ceb862e3a1073a2c1beef75d60a` + threadpool-oversubscription patch | `https://github.com/ggml-org/ggml` | `ggml/UPSTREAM` (adopted from transcribe.cpp) |
| **Silero VAD** | Latest stable release tag from `https://github.com/snakers4/silero-vad` (exact tag pinned at Phase 0 kickoff) | `https://github.com/snakers4/silero-vad` | `model_specs/silero_vad.json` + `reference/silero-vad/` |
| **sentencepiece** | As currently vendored in `external/sentencepiece/` | `https://github.com/google/sentencepiece` | CMake submodule / vendored pin |
| **llama_tokenizer** | As currently vendored in `external/llama_tokenizer/` | `https://github.com/ggerganov/llama.cpp` | CMake submodule / vendored pin |
| **cJSON** | As currently vendored in `external/cJSON/` | `https://github.com/DaveGamble/cJSON` | CMake submodule / vendored pin |
| **libyaml** | v0.2.5 (pinned via `AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS`) | `https://github.com/yaml/libyaml` | CMake compile definitions |
| **dr_wav** (example WAV loader) | As currently vendored in `examples/common/` | `https://github.com/mackron/dr_libs` | Vendored |
| **miniz** (zlib replacement) | As currently vendored in `src/third_party/miniz/` | `https://github.com/richgel999/miniz` | Vendored in transcribe.cpp tree |
| **transcribe.cpp Python binding** | v0.2.0 (`transcribe-cpp` package) | `https://github.com/handy-computer/transcribe.cpp` | `bindings/python/pyproject.toml` |
| **transcribe.cpp TS binding** | koffi-based | `https://github.com/koffi/koffi` | `bindings/typescript/package.json` |

**Python binding provider packages:**
- `transcribe-cpp-native` (default: CPU+Metal on macOS arm64, CPU+Vulkan on Linux/Windows)
- `transcribe-cpp-native-cu12` (opt-in CUDA 12 provider)

**Note on CFFI correction:** The original plan described Python bindings as
"CFFI-based." The actual transcribe.cpp bindings use **ctypes** (confirmed by
`pyproject.toml`: "The binding is ctypes-only"). All references in this plan to
"CFFI" have been corrected to "ctypes."

---

## 0.5 Phase Dependency Graph

```
PHASE 0: Foundation & ABI Bridge
  ├── (A) Pin reference commits + ggml + Silero VAD (upstream-pinned version)
  ├── (B) Port public C ABI headers → include/transcribe/
  ├── (C) Port internal runtime headers → src/runtime/
  ├── (D) Implement central dispatcher (api_guard, abi, loader, arch, model, session)
  ├── (E) Implement ArchAdapter (IVoiceModelLoader → Arch)
  ├── (F) ggml convergence build (§0.6) — MUST pass before any family port
  ├── (G) CMake integration (engine_transcribe_runtime, asr composite)
  ├── (H) Hello-world bridge test (abi_bridge_hello.cpp)
  ├── (I) Bidirectional retrofits (safe_*, api_guard_*, struct_size, golden manifest pilot)
  └── (J) Silero VAD pinning + golden manifest + version unit test
  (B,C,D,E depend on A; F depends on A; G depends on B–E; H depends on F–G;
   I depends on C–E; J independent of most)

PHASE 1: P0 Family Ports
  ├── (A) Consolidate qwen3_asr (replace audio.cpp impl with transcribe.cpp)
  ├── (B) Consolidate sensevoice (replace audio.cpp sense_asr)
  ├── (C) Consolidate funasr_nano
  ├── (D) Port Whisper (16 variants)
  ├── (E) Port Moonshine (base + streaming)
  └── (F) Parallel: mel frontend adoption, causal_lm adoption, capability probing
  (A–E depend on Phase 0; F runs in parallel)

PHASE 2: P1 Family Ports
  ├── Parakeet, Canary, Canary-Qwen, GigaAM, Granite(+NAR), Cohere,
  │   Voxtral, Sortformer, Moonshine-Streaming
  └── Parallel: abort callbacks on all streaming tasks, streaming state machine retrofit

PHASE 3: P2 Families + Advanced Features
  ├── MedASR, Moss
  ├── Spec decoding (all AR decoder families)
  ├── PNC/ITN/Translation
  └── Long-form chunking + cancellation

PHASE 4: Bindings
  ├── Python (ctypes), TypeScript (koffi), Rust, Swift

PHASE 5: CLI / Server / WebUI
  ├── Unified CLI, REST API, WebUI

PHASE 6: Testing Maturity & Release
  └── CI gates, full ctest, 1.0.0-alpha
```

**Dependency constraints:**
- Phase 0.F (ggml convergence) is a hard gate — no family port begins until the unified
  codebase compiles and passes tests against the single pinned ggml.
- Phase 0.H (hello-world bridge test) must pass before Phase 1 family ports begin.
- Phase 1.A–C (consolidation) require Phase 0's `ArchAdapter` and `causal_lm` module.
- Phase 4 (bindings) requires the generalized C ABI (§1.5.7) to be stabilized.
- Phase 5 (CLI/Server/WebUI) requires Phase 4 bindings OR direct C ABI access.
- Phase 6 (release) requires all prior phases' golden manifests to be green.

---

## 0.6 ggml Unification Strategy (Critical Integration Risk)

**The problem:** `audio.cpp` and `transcribe.cpp` each vendor a DIFFERENT ggml
tree:

| Project | ggml location | Version pin | Notes |
|---------|--------------|-------------|-------|
| audio.cpp | `external/ggml/` | None (no UPSTREAM file) | Version inferred from git state |
| transcribe.cpp | `ggml/` | SHA `8c63e70982c95ceb862e3a1073a2c1beef75d60a` | Pinned via `ggml/UPSTREAM`; has threadpool-oversubscription patch |

These ggml trees may have diverged in: API surface (`ggml_new_tensor` signature
variants, `ggml_backend_*` API), struct layouts, backend registry names, and the
threadpool implementation. A unified codebase that tries to compile code targeting
two different ggml ABIs will fail to link or exhibit undefined behavior at runtime.

**Resolution (D16):** Adopt transcribe.cpp's pinned ggml as the single vendored ggml.

1. The unified `CMakeLists.txt` sets `AUDIOCPP_GGML_SOURCE_DIR` to
   `${CMAKE_CURRENT_SOURCE_DIR}/ggml` (transcribe.cpp's pinned tree), replacing
   the current `${CMAKE_CURRENT_SOURCE_DIR}/external/ggml`.
2. The `ggml/UPSTREAM` file from transcribe.cpp is carried into the unified project
   as the source of truth for the ggml version.
3. The threadpool-oversubscription patch (transcribe.cpp's `patches/ggml/0001-
   fix-threadpool-oversubscription.patch`) is applied via `scripts/sync-ggml.sh`.
4. A "ggml convergence build" sub-task (Phase 0.F) must succeed: `engine_core` +
   `transcribe_runtime` must compile and link against the single pinned ggml, and
   transcribe.cpp's full C++ white-box test suite must pass. This gates all family
   ports.
5. Any audio.cpp code that calls ggml APIs that changed between versions must be
   reconciled. Primary audit surface: `src/framework/core/backend.cpp` and the
   ggml-backend wrapper calls in model sessions.

**Risk mitigation (R5):** If the ggml API drift is too large, the fallback is to
vendor transcribe.cpp's ggml alongside audio.cpp's and use ggml's backend-module
loading (`TRANSCRIBE_GGML_BACKEND_DL`) to isolate them — but this significantly
increases binary size and complexity. The preferred path is full convergence.

---

## 0.7 Timeline Reality Check

The original plan's phase durations (2 weeks for Phase 0, 10 weeks for Phase 1, etc.)
are based on optimistic assumptions about straightforward code copying. The actual
complexity is higher because:

1. **Phase 0 is not just "port the C ABI"** — it requires building the entire
   `ArchAdapter` bridge (mapping 3-level vtable to 1-level Arch), retrofitting
   safe teardown across 40+ existing model families, generating golden manifests
   for existing families, AND pinning/resolving ggml. This is realistically
   **4-6 weeks** for a single engineer, or **2-3 weeks** for a team of 3-4 engineers.

2. **Consolidation is harder than fresh ports** — replacing audio.cpp's existing
   `qwen3_asr`/`sensevoice`/`funasr_nano` implementations with transcribe.cpp's
   versions requires reconciling two completely different session models
   (`RuntimeSessionBase` vs. `transcribe_session`) and ensuring the audio.cpp
   CLI/server/WebUI still work against the new backends.

3. **The ggml convergence** (§0.6) is a blocking dependency that could add 1-2 weeks
   if API drift is significant.

4. **Bindings** (Phase 4) require careful ABI drift-gate matching — the ctypes
   binding verifies struct layouts at import; any ABI mismatch blocks all bindings.

**Adjusted timeline estimate:** Phase 0 (4-6 weeks) → Phase 1-3 (34-36 weeks for
all family ports) → Phase 4-5 (14 weeks) → Phase 6 (ongoing). Total: ~12-12.5
months for a team of 3-4 engineers. Milestones A/B/C remain valid checkpoints.

---

## 0.8 Conventions Used in This Plan

- **"audio.cpp"** refers to the C++ framework project (or its fork `speech.cpp/`).
  Its dispatch model is the 3-level C++ vtable.
- **"transcribe.cpp"** refers to the STT project at `transcribe.cpp/`. Its dispatch
  model is the 1-level C `Arch` struct.
- **"speech.cpp" / "unified"** refers to the merged result. Currently a
  binary copy of audio.cpp; the fork base is captured in §0.2.
- **"Existing ABI"** = APIs that exist today in `transcribe.cpp/include/transcribe.h`
  (2499 lines, v0.2.0). These must not break for existing STT callers.
- **"Proposed extension"** = C ABI additions for non-STT tasks (TTS, voice cloning,
  etc.) described in §1.5.7. These are designed during Phase 0 and stabilized before
  Phase 4 bindings.
- **`extern "C"` entry point** = any `TRANSCRIBE_API` function in transcribe.h.
- **`src/runtime/`** = the target directory in speech.cpp for ported
  transcribe.cpp runtime files (dispatcher, Arch, loader, backend, etc.).
- **"C++ internal API"** = audio.cpp's `include/engine/` headers, accessible only
  to C++ consumers (CLI, server, WebUI) within the same build. Not exposed via C ABI.

---


## 0.9 Speech.cpp Fork State (Phase 0.0 Prerequisite)

**Status:** speech.cpp has **diverged** from audio.cpp upstream. As of this
writing:

- **audio.cpp upstream HEAD**: `52080cd` ("Fix IndexTTS2/2.5 text normalization
  issues (#259)")
- **speech.cpp HEAD**: `ee940b0` ("Fix Music3 component GGUF symlink overrides")
- **Divergence**: speech.cpp is 6 commits ahead of audio.cpp upstream:

| Commit | Type | Description |
|--------|------|-------------|
| 61012b5 | feature | Tune MiniMax Music3 default component mix |
| db59178 | feature | Preserve Music3 fast conv transpose path |
| 0bec2dd | feature | Improve MiniMax Music3 Vulkan execution |
| 50f0aba | merge | Merge PR #271 from 0xShug0/preview/minimax-music-3 |
| ea5475a | test | tests/omnivoice: add OmniVoice weight-type benchmark, report, and Python-to-C++ tests (#269) |
| ee940b0 | fix | Fix Music3 component GGUF symlink overrides |

- **Branch structure**: speech.cpp `main` tracks `origin/main` (NairoDorian/speech.cpp
  fork). `upstream/main` points to `0xShug0/audio.cpp` (the source repo).
- **Music3 family**: `src/models/muscriptor/` is **exclusive to speech.cpp** — it is
  not present in audio.cpp upstream. This is the MiniMax Music3 text-to-music
  family. The `muscriptor` model spec lives at `model_specs/muscriptor.json`.
- **Merge strategy for Phase 0**: The audio.cpp baseline commit to merge from is
  `52080cd`. Speech.cpp's Music3 commits (61012b5 through ee940b0) must be
  rebased onto the unified codebase after the transcribe.cpp absorption is
  complete, OR the Music3 family must be ported forward using the standard
  `audiocpp_add_model(muscriptor)` CMake registration pathway used by all
  other families.

**Phase 0.0 Action**: Before Phase 0 kickoff, capture the exact diff between
`audio.cpp@52080cd` and `speech.cpp@ee940b0` for the `muscriptor/` model
directory and `model_specs/muscriptor.json`. This becomes the "speech.cpp
exclusive delta" that Phase 6 must re-integrate.

## 1. Architecture Deep-Dive

### 1.1 The Two Dispatch Models (The Fundamental Integration Decision)

`audio.cpp` and `transcribe.cpp` have **fundamentally different dispatch models**.
The plan's success hinges on understanding exactly how each works.

#### audio.cpp — C++ Interface / Vtable Dispatch

```
ModelRegistry::load(ModelLoadRequest)
  → IVoiceModelLoader::load() → ILoadedVoiceModel
  → ILoadedVoiceModel::create_task_session(TaskSpec{Asr, Offline})
      → IVoiceTaskSession (base: family(), task_kind(), run_mode(), prepare())
        → IOfflineVoiceTaskSession::run(TaskRequest) → TaskResult
        → IStreamingVoiceTaskSession::start_stream / process_audio_chunk / finish_stream / reset
  ── TaskResult carries: text_output, speech_segments, word_timestamps, speaker_turns, tokens, audio_output, named_audio_outputs, output_artifacts
```

Key types (from `include/engine/framework/runtime/`):
- `IVoiceModelLoader` (in `model.h`, line 122) — abstract: `family()`,
  `can_load()`, `inspect()`, `load()`. Registered via `ModelRegistry`.
  Returns `std::unique_ptr<ILoadedVoiceModel>`.
- `ILoadedVoiceModel` (in `model.h`, line 101) — abstract: `metadata()`,
  `capabilities()`, `create_task_session()`. Holds `shared_ptr<const Assets>`.
- `IVoiceTaskSession` / `IOfflineVoiceTaskSession` / `IStreamingVoiceTaskSession`
  (in `session.h`, line 235) — abstract session interfaces. Concrete: `Qwen3ASRSession`
  (in `include/engine/models/qwen3_asr/session.h`), implements `prepare()`, `run()`,
  `start_stream()`, `process_audio_chunk()`, `finish_stream()`, `finalize()`, `reset()`.
  Extends `RuntimeSessionBase`.
- `TaskSpec { VoiceTaskKind task, RunMode mode }`
- `TaskRequest { optional<Transcript>, optional<AudioBuffer>, optional<VoiceCondition>,
  vector<VoiceArtifact>, options map }`
- `TaskResult { optional<AudioBuffer> audio_output, vector<NamedAudioBuffer>,
  optional<Transcript> text_output, vector<SpeechSegment>, vector<SpeakerTurn>,
  vector<WordTimestamp>, optional<VoiceArtifact>, vector<VoiceArtifact> }`
- `StreamEvent`, `StreamingPolicy`, `StreamEventCallback` — **dual** streaming
  model: push (`set_stream_event_sink(Sink)`) AND pull (`next_stream_event()`
  returns `optional<StreamEvent>`), plus `finish_stream()`, `reset()`,
  `process_audio_chunk()`, `finalize()`. No explicit stream-state enum (no
  IDLE/ACTIVE/FAILED state machine on the C++ side).
  `StreamingPolicy { StreamingInputKind, StreamingOutputKind, preferred_audio_chunk_samples,
  preferred_audio_chunk_seconds }`. ` method.)
- `RuntimeSessionBase` (in `session_base.h`) — concrete base owning
  `ExecutionContext`, `ArtifactStore`, `RuntimeCache`, `GraphExecutor`,
  `Workspace`. ` functional module; see §1.5.2.) All audio.cpp
  sessions extend this.
- `BackendConfig { BackendType type, int device, int threads }` (in `core/backend.h`).
  **Corrected: does NOT carry gpu_split_mode, n_gpu_layers, tensor_split, or
  gpu_mode** — those are ggml-scheduler-level concepts, not BackendConfig fields.
- `BackendType` enum (in `core/module.h`): Cpu, Cuda, Hip, Vulkan, Metal,
  BestAvailable. **No SYCL or BLAS as first-class types.**
- `ModelRegistry` (in `runtime/registry.h`) — `make_default_registry()` builds from
  CMake-generated `model_registry_includes.inc` / `model_registry_loaders.inc`.

Build system (`CMakeLists.txt`, 2000+ lines):
- `audiocpp_add_model(target_name SOURCES ... INCLUDES ... LOADERS ... ALIASES ... DEPENDS ...)`
  creates an OBJECT library per model (e.g. `engine_model_qwen3_asr`).
- `AUDIOCPP_MODEL_SET` cache var: `full` / `core` / `custom`.
- `AUDIOCPP_MODELS` lists models for `custom` builds.
- `engine_core` OBJECT library (320+ source files) holds the framework.
- `engine_runtime` STATIC library aggregates `engine_core` + silero_vad + marblenet_vad
  + selected model OBJECT libs.
- ggml vendored at `external/ggml/` (no UPSTREAM pin — see §0.6).
- `audiocpp_configure_runtime_object()` sets up includes, deps (ggml, sentencepiece),
  and OpenMP for every model/runtime target.

#### transcribe.cpp — C ABI / Struct-of-Function-Pointer Dispatch

```
transcribe_model_load_file(path, params, &model)
  → Loader::open(path) reads GGUF header, extracts general.architecture + stt.variant
  → find_arch(arch_name) → Arch* (function-local static array in transcribe-arch.cpp)
  → arch.load(loader, params, &model) → transcribe_model* (derives from base)
    — model->arch points back to the Arch; model->meta = all GGUF KVs;
      model->caps filled; per-family BackendPlan bound.

transcribe_session_init(model, session_params, &session)
  → arch.init_context(model, params, &session) → transcribe_session* (derives from base)

transcribe_run(session, pcm, n, params)
  → dispatcher validates params → arch.run(session, pcm, n, params)
    → populates result vectors on session base; accessors read them.

transcribe_stream_begin(session, run_params, stream_params)
  → dispatcher preflight (size, enums, ext kind, language) →
    arch.stream_validate (optional) → clear snapshot → ACTIVE →
    arch.stream_begin(run_params, stream_params)

transcribe_stream_feed(session, pcm, n, update) → arch.stream_feed(...)
transcribe_stream_finalize(session, update)    → arch.stream_finalize(...) → FINISHED
transcribe_stream_reset(session)               → arch.stream_reset (optional) → IDLE
```

Key types :
- **`Arch`** struct (`src/transcribe-arch.h`, 138 lines): 11 function-pointer fields + 1 `name` data field (12 members total):
  `name`, `load()`, `init_context()`, `run()`, `run_batch()` (optional),
  `stream_validate()` (optional), `stream_begin()`, `stream_feed()`,
  `stream_finalize()`, `stream_reset()` (optional), `accepts_ext_kind()` (optional),
  `run_validate()` (optional). Null entries → `TRANSCRIBE_ERR_NOT_IMPLEMENTED`.
  The struct has 12 members total — `name` (a `const char*`
  data field, not a function pointer) plus 11 function pointers. 
  Registered in the explicit array `transcribe-arch.cpp` — 18 family Arch instances
  in a function-local static array. No static-init-order fiasco.
- **`transcribe_model`** base (`src/transcribe-model.h`): `arch` dispatch token,
  `variant` string, `meta` map (all GGUF KVs) via `MetaMap`
  (`std::map<string,string,std::less<>>`), `backend` string,
  `primary_backend` handle (`ggml_backend_t`), `caps` (transcribe_capabilities),
  feature bits (bit-per-transcribe_feature). Per-family subclasses own weights;
  virtual destructor (polymorphic delete).
- **`transcribe_session`** base (`src/transcribe-session.h`): lifecycle state
  (IDLE/ACTIVE/FINISHED/FAILED), `poll_abort()` (abort callback + polling between
  decode steps), result vectors (segments/words/tokens/text),
  `transcribe_session_params_n_ctx()` (struct_size-guarded n_ctx read).
- **`Loader`** (`src/transcribe-loader.h/c`): architecture-agnostic GGUF header
  inspection + backend init. Reads `general.architecture` + `stt.variant`,
  extracts all scalar-string KVs into a `MetaMap`. Does NOT know family specifics.
- **`BackendPlan`** (`src/transcribe-backend.h`): `requested` (transcribe_backend_request),
  `primary` (ggml_backend_t), `primary_kind` (BackendKind), `scheduler_list`
  (vector<ggml_backend_t> in priority order). `BackendKind` enum: Unknown/Cpu/Metal/
  Vulkan/Cuda/Rocm/Sycl/Accel/OtherGpu (9 values, incl. Unknown).
- **Central dispatcher** (`src/transcribe.cpp`): `api_guard_status`/`api_guard_value`/
  `api_guard_void` templates  contain ALL C++ exceptions at
  the public boundary: `bad_alloc` → `TRANSCRIBE_ERR_OOM`, `std::exception` →
  `TRANSCRIBE_ERR_BACKEND`, `...` → `TRANSCRIBE_ERR_BACKEND`. Every
  `extern "C"` entry point routes through these. Includes `enum_field_raw()` for
  safe raw-int reads of caller enum fields. `transcribe_log_set()` publishes the
  callback with release semantics.
- **ABI helpers** (`src/transcribe-abi.h`): `check_struct_size()` (rejects
  `struct_size < want`, including `== 0`), `copy_out_prefix()` (writes
  `min(caller_size, lib_size)` bytes). Both `inline` in the header, shared by the
  dispatcher and per-family public accessors (e.g. `arch/whisper/public.cpp`).
- **Safe teardown** (`src/transcribe-backend.h`, line 89-97): `safe_backend_free()`,
  `safe_sched_free()`, `safe_buffer_free()` — null-checking, no-throw wrappers.
  Test hook: `TRANSCRIBE_TEST_TEARDOWN_THROW` injects an internal throw after the
  real free. CI-gated by `tests/lint_teardown.cmake`.
- **Family extensions** (`include/transcribe/*.h`): `struct transcribe_ext {
  uint64_t size; uint32_t kind; }` embedded as first field of every typed family
  extension struct. `transcribe_ext_check()`, FourCC kind constants,
  `transcribe_model_accepts_ext_kind()`.
- **`transcribe_feature`** enum (7 values, 0..6): `INITIAL_PROMPT`,
  `TEMPERATURE_FALLBACK`, `LONG_FORM`, `CANCELLATION`, `PNC`, `ITN`,
  `DIARIZATION`.
- **`transcribe_status`** enum (19 codes, 0..18): `TRANSCRIBE_OK`=0 through
  `TRANSCRIBE_ERR_OUTPUT_TRUNCATED`=18. Codes 9 (SAMPLE_RATE), 15 (UNSUPPORTED_PNC),
  16 (UNSUPPORTED_ITN) are reserved/unused.
- **`transcribe_abi_struct`** enum (15 values, 0..14): struct IDs for
  `transcribe_abi_struct_size()`/`transcribe_abi_struct_align()` runtime layout
  verification by bindings.
- **Log sink** (`src/transcribe-log.h`): routes ggml diagnostics into the
  `transcribe_log_callback`. `CONT` level for streaming progress (overwrites on `\r`).
- **`TRANSCRIBE_TEST_DEV_INIT_THROW`** / `TRANSCRIBE_TEST_TEARDOWN_THROW`** fault
  hooks: ship in release artifacts for wheel clean-install CI (per AGENTS.md).
  Present-but-empty values are inert; non-empty values inject throws to prove
  `api_guard_*` containment without leaking handles.

**The full public C ABI surface** (from `include/transcribe.h`, 2499 lines — verified
against Appendix A). Key sections:
- Version: `transcribe_version()`, `transcribe_version_commit()`.
- Status: `transcribe_status_string()`.
- ABI metadata: `transcribe_abi_struct_size()`, `transcribe_abi_struct_align()`.
- Logging: `transcribe_log_callback`, `transcribe_log_set()`.
- Backend: `transcribe_init_backends()`, `transcribe_init_backends_default()`,
  `transcribe_device_count()`, `transcribe_device_get()`, `transcribe_device_get_info()`,
  `transcribe_backend_available()`, `transcribe_model_device()`.
- Params (4 input structs with `_init`): `transcribe_model_load_params`,
  `transcribe_session_params`, `transcribe_run_params`, `transcribe_stream_params`.
  `transcribe_run_params` carries: task, timestamps, pnc, itn, diarize, language,
  target_language, keep_special_tags, family(ext), spec_k_drafts (-1=default, 0=off,
  >0=N).
- Capabilities: `transcribe_capabilities` (12 fields, verified from `transcribe.h`):
  `struct_size`, `native_sample_rate`, `n_languages`, `languages[]`,
  `max_timestamp_kind`, `supports_language_detect`, `supports_translate`,
  `supports_streaming`, `supports_spec_decode`, `max_audio_ms`,
  `n_translate_target_languages`, `translate_target_languages[]`.
  Accessors: `transcribe_model_get_capabilities()`, `transcribe_model_supports()`.
- Streaming state: `transcribe_stream_state` enum (IDLE/ACTIVE/FINISHED/FAILED).
  `transcribe_stream_commit_policy` enum (AUTO/ON_FINALIZE/STABLE_PREFIX).
- Limits: `transcribe_session_limits { effective_n_ctx, effective_max_audio_ms,
  max_kv_bytes }`.

| transcribe.cpp component | audio.cpp equivalent | Consolidation needed? |
|--------------------------|----------------------|------------------------|
| `transcribe-kaldi-fbank.h` (KaldiFbankFrontend: Hamming, LFR, CMVN) | `src/framework/audio/kaldi_fbank.cpp` (basic, no LFR/CMVN) | YES — adopt transcribe.cpp's header-only version with LFR+CMVN for STT. Audio.cpp's basic version stays for non-STT tasks. |
| `transcribe-mel.h` (MelFrontend: 4 normalize modes, 3 pad modes, hann variants) | `src/framework/audio/dsp.h` + `dsp.cpp` (basic resampling only) | YES — adopt transcribe.cpp's `MelFrontend` + `MelConfig` as-is. |
| `transcribe-tokenizer.h` (Tokenizer: BPE + SentencePiece) | `src/framework/tokenizers/` (llama_bpe, sentencepiece, hf_tokenizer_json) + `external/llama_tokenizer/bpe-core.cpp` | PARTIAL — adopt transcribe.cpp's `Tokenizer` interface for STT families (BPE parity tests depend on it). Keep audio.cpp's tokenizers for TTS/voice-cloning. |
| `transcribe-weights-util.h` | audio.cpp has equivalent GGUF reader in `assets/` | NO — reuse; port the helpers. |
| `transcribe-meta.h` (KvResult tri-state + read_*_kv + MetaMap) | audio.cpp has GGUF reading in model loaders per-family | YES — adopt transcribe.cpp's `MetaMap` + KvResult helpers (Absent/BadType distinction is critical for golden-manifest validation) |
| `transcribe-debug.h` (structured debug dumps) | audio.cpp has `debug/trace.h` + `debug/profiler.h` | PARTIAL — adopt transcribe.cpp's dump format (`.f32` raw + `.json` sidecar). Integrate with audio.cpp's trace/profiler. |
| `transcribe-batch-util.h` (batch dispatch + thread sizing) | audio.cpp has no batch API | YES — port the batch dispatch + mask utilities (for `transcribe_run_batch`) |
| `transcribe-flash-policy.h` (flash-attention KV cache policy) | audio.cpp has `runtime/cache.h` (RuntimeCache) | YES — adopt the flash-attention KV cache policy for STT families |
| `transcribe-load-common.h` (F16→F32 conv promotion, Mel injection hooks) | audio.cpp has no equivalent | YES — port for numerical validation hooks (`TRANSCRIBE_MEL_FROM_REF`) |
| `transcribe-abi.h` (check_struct_size, copy_out_prefix) | audio.cpp has no size-aware ABI structs | YES — adopt for all cross-module boundary structs in the unified codebase |
| `transcribe-path.h` (UTF-8 path handling) | audio.cpp uses `std::filesystem` directly | YES — port for Windows path correctness (fixes Handy issue #1585) |

### 1.2 transcribe.cpp — C ABI / Struct-of-Function-Pointer Dispatch

**The full public C ABI surface** (from `include/transcribe.h`, 2499 lines — verified
against Appendix A). Key sections:
- Version: `transcribe_version()`, `transcribe_version_commit()`.
- Status: `transcribe_status_string()`.
- ABI metadata: `transcribe_abi_struct_size()`, `transcribe_abi_struct_align()`.
- Logging: `transcribe_log_callback`, `transcribe_log_set()`.
- Backend: `transcribe_init_backends()`, `transcribe_init_backends_default()`,
  `transcribe_device_count()`, `transcribe_device_get()`, `transcribe_device_get_info()`,
  `transcribe_backend_available()`, `transcribe_model_device()`.
- Params (4 input structs with `_init`): `transcribe_model_load_params`,
  `transcribe_session_params`, `transcribe_run_params`, `transcribe_stream_params`.
  `transcribe_run_params` carries: task, timestamps, pnc, itn, diarize, language,
  target_language, keep_special_tags, family(ext), spec_k_drafts (-1=default, 0=off,
  >0=N).
- Capabilities: `transcribe_capabilities` (12 fields):
  `struct_size`, `native_sample_rate`, `n_languages`, `languages[]`,
  `max_timestamp_kind`, `supports_language_detect`, `supports_translate`,
  `supports_streaming`, `supports_spec_decode`, `max_audio_ms`,
  `n_translate_target_languages`, `translate_target_languages[]`.
  Accessors: `transcribe_model_get_capabilities()`, `transcribe_model_supports()`.
- Streaming state: `transcribe_stream_state` enum (IDLE/ACTIVE/FINISHED/FAILED).
  `transcribe_stream_commit_policy` enum (AUTO/ON_FINALIZE/STABLE_PREFIX).
- Limits: `transcribe_session_limits { effective_n_ctx, effective_max_audio_ms,
  max_kv_bytes }`.

The `Arch` struct (defined in `src/transcribe-arch.h`, ~12 members) is the
central dispatch unit. Each family provides an `Arch` instance with function
pointers: `name`, `load()`, `init_context()`, `run()`, `run_batch()` (optional),
`stream_begin()`, `stream_feed()`, `stream_finalize()`, `reset()` (optional),
`accepts_ext_kind()` (optional). The struct is ~96 bytes (8 + 4 * 11 = 52 on
32-bit; 8 + 8 * 11 = 96 on 64-bit with padding).

transcribe-arch.cpp is **hand-maintained** (not CMake-generated). It contains
explicit `extern` declarations for each family's `arch_*` symbol and a
function-local `static const Arch s_archs[]` array. The file is edited by hand
when new families are added; CMake's `audiocpp_add_model()` registers the
family, but the Arch entry is a manual addition.

### 1.3 Shared Components Consolidation

Audio.cpp does not contain a file named `kaixi_fbank.cpp` — this name does not
appear anywhere in the audio.cpp tree. The actual mel frontend implementation is
`kaldi_fbank.cpp` (at `src/framework/audio/kaldi_fbank.cpp`) for Kaldi filterbank
extraction, and `dsp.h` + `dsp.cpp` for basic DSP operations (resampling, Hamming
windows). The transcribe.cpp `MelFrontend` and `KaldiFbankFrontend` provide
supersets of functionality (4 mel normalize modes, 3 pad modes, LFR, CMVN) and
should be adopted for STT families (see §1.5.2 and §1.5.6).


### 1.4 The Causal LM Shared Module

transcribe.cpp has `src/causal_lm/causal_lm.h/.cpp` — a **shared causal-decoder
transformer block** (Llama/Qwen3 lineage: pre-LN RMSNorm, GQA, NeoX rotate_half RoPE,
SwiGLU MLP via packed gate+up, optional per-head Q/K RMSNorm, optional per-layer
FFN scale). Key types: `causal_lm::BlockView` (nullable-pointer projection) and
`causal_lm::BlockParams` (n_heads, n_kv_heads, head_dim, max_position, rms_eps,
rope_theta). Helper: `pack_gate_up()` (packs gate+up weights for single mul_mat +
swiglu).

Used by: `qwen3_asr` (audio-LLM), `funasr_nano`, `voxtral`, `voxtral_realtime`,
`canary_qwen`, `moss`, `cohere` (where applicable — the ones with LLM-pattern decoders).

audio.cpp has equivalent transformer block math in:
`src/framework/modules/attention/` (grouped_query_attention, scaled_dot_product_attention,
relative_attention, longformer_attention, cross_attention, etc.) and
`src/framework/modules/transformers/` (qwen_decoder, qwen_causal_decoder,
qwen_causal_decode_runtime, qwen3_vl_encoder_runtime, gemma_decoder). Conformer
encoder blocks are in `src/framework/modules/conformer_modules.cpp`.

**Decision (D15):** Adopt transcribe.cpp's `causal_lm` module as the **shared STT
causal-LM backbone** in `src/runtime/causal_lm/`. It's lighter and purpose-built,
and the `BlockView` struct's nullable-pointer projection pattern makes family
porting much easier (families just fill the struct, the block math is shared).
Audio.cpp's broader attention/transformer modules remain for TTS/voice-cloning/
generation tasks. The two coexist without a shared code path — they are
independently maintained.

NOTE: audio.cpp's `qwen3_asr` session (`include/engine/models/qwen3_asr/session.h`,
line 31) already implements streaming with `Qwen3ASRThinkerRuntime` (the LLM decoder)
and `Qwen3ASRAudioEncoderRuntime`. The consolidation replaces this with transcribe.cpp's
`causal_lm`-based decoder, keeping the audio.cpp session's `TaskResult`/`StreamEvent`
output format for CLI/server/WebUI compatibility.

### 1.5 Smart Merging: Bidirectional Upgrades

Full decision matrix at §0.3 (Decision Log). Key rows:

**§1.5 Decision Matrix (corrected):**

| Concern | audio.cpp | transcribe.cpp | Unified Decision | Rationale |
|---------|-----------|----------------|------------------|-----------|
| **Public API surface** | C++ headers only. No C ABI. | C ABI (2499 lines). ctypes/koffi/Rust/Swift FFI. | **C ABI primary; C++ internal.** | Lowest common denominator. |
| **Dispatch** | 3-level vtable. Heavy runtime per session. | 1-level `Arch`. Lightweight. | **`Arch` unified dispatch + `ArchAdapter`.** | Simpler; bridge rather than rewrite. |
| **Model loading** | `ModelRegistry` + `model_specs/*.json`. | `Loader::open()` + `find_arch()`. | **Loader for ABI path; ModelRegistry for C++ path.** | Loader is correct lower layer; spec system adds package management. |
| **Session creation** | `create_task_session()` → `RuntimeSessionBase`. | `arch.init_context()` → `transcribe_session*`. | **Hybrid: lightweight for ABI, RuntimeSessionBase for C++.** | STT doesn't need full runtime; TTS/vocoder does. |
| **Backend selection** | `BackendConfig {type, device, threads}` + `BackendType` (Cpu, Cuda, Hip, Vulkan, Metal, BestAvailable). | `BackendPlan` + `BackendKind` (9 kinds incl. Sycl/Accel/OtherGpu). Dynamic module loading. | **`BackendPlan` + dynamic modules as C ABI surface.** | Dynamic backend DLLs; audio.cpp's BackendConfig feeds BackendPlan. GPU offload is ggml-scheduler-level. |
| **Error handling** | C++ exceptions. | `transcribe_status` (19 codes). `api_guard_*`. | **Status-enum for C ABI; `api_guard_*` at all boundaries.** | Exceptions across FFI = UB. |
| **ABI versioning** | None. | `transcribe.abihash` + `struct_size` + abi_struct_size(). | **Size-aware ABI for C surface.** | Forward/backward compatibility. |
| **Capabilities** | `CapabilitySet` (5 fields). | `transcribe_capabilities` (12 fields incl. `n_translate_target_languages`) + 7-feature enum. | **transcribe.cpp's model.** Map at ArchAdapter boundary. | Extensible feature probes. |
| **Family extensions** | None (options map). | `struct transcribe_ext {size, kind}`. FourCC + accepts_ext_kind. | **Adopt extension pattern.** | Forward-compatible across ABI. |
| **Spec decoding** | Basic sampling only. | `spec_k_drafts` (-1/0/>0) + `supports_spec_decode`. | **Port spec-decode sampler.** | Performance win. |
| **Text post-processing** | `text_normalization.cpp` (Chinese+English). | Runtime toggle enums on params. | **Use audio.cpp pipeline; expose via enums.** | More complete normalization. |
| **Logging** | `trace.h` + `profiler.h`. | `transcribe_log_callback` + 6 levels. Routes ggml diagnostics. | **transcribe.cpp logger as surface.** | `CONT` level essential for streaming. |
| **Mel/frontend** | `kaldi_fbank.cpp` (basic). | `MelFrontend` (4+3 hann variants) + `KaldiFbankFrontend` (LFR+CMVN). | **Adopt both transcribe.cpp frontends.** | Strictly more configurable. |
| **Build distribution** | Composites (full/core/custom). | Single library + CLI. | **Enhance composites; add `asr`/`asr+full`.** | Distribution flexibility. |
| **Testing** | Ad-hoc tests. | Golden manifests + tolerances + 8-stage pipeline. | **Adopt full methodology for ALL families.** | Prevents regressions. |
| **Streaming model** | Push-model `StreamEventCallback`. No state machine. No committed/tentative. | 4-state machine. 3-checkpoint begin. `transcribe_stream_get_text()` (full/committed/tentative + byte counts). | **Adopt state machine + committed/tentative.** | Prevents transcript loss; flicker-free UI. |

---

### 1.5.1 The 12 Early Wins (Apply Immediately from Phase 0)

These upgrades apply from Phase 0 onward, benefiting BOTH new STT families AND
existing audio.cpp families (detailed version in §1.5.6):

1. **`safe_*` teardown + `lint_teardown.cmake`:** Port transcribe.cpp's
   `safe_backend_free()` / `safe_sched_free()` / `safe_buffer_free()` into
   `src/runtime/`. Apply retroactively to ALL existing model sessions. CI gate
   fails on raw `ggml_backend_free` / `ggml_backend_buffer_free` /
   `ggml_backend_sched_free` in library code. (Does NOT target
   `ggml_backend_graph_plan_free` — that frees a graph plan, not a backend.)
2. **`api_guard_*` exception containment:** Port templates into `src/runtime/`.
   Every `extern "C"` entry point routes through it. Apply at C++ boundaries
   (CLI `main()`, server request handler) as defense-in-depth. `API_GUARD_ENFORCED.cmake`
   is a proposed NEW CI gate (not yet existing) that scans for `extern "C"` functions
   not routing through `api_guard_*`.
3. **`struct_size` on internal structs:** Apply `copy_out_prefix` pattern to
   cross-module boundaries (`TaskResult`, `CapabilitySet`).
4. **Golden manifests for existing families:** Generate golden manifests for
   audio.cpp's existing non-ASR families (TTS, voice cloning) using the same
   `1e-4 × p99_abs` / `1e-5 × rms` methodology.
5. **Model spec JSON adoption:** Every transcribe.cpp family gets a
   `model_specs/<family>.json` (port audio.cpp's spec format).
6. **`transcribe_feature` probing:** Adopt the enum + `transcribe_model_supports()`
   from day one. `ArchAdapter` maps audio.cpp's `CapabilitySet` into it.
7. **Unified logging:** Port `transcribe_log_callback` + log levels. Route ggml
   diagnostics through the callback (matches AGENTS.md's ggml routing).
8. **`KvResult` tri-state metadata helpers:** Port `transcribe-meta.h`'s
   `KvResult` {Absent, Ok, BadType} + `read_*_kv` helpers. Adopt for ALL families.
9. **`transcribe_timings` lightweight model:** Port the 4-field timings
   { load_ms, mel_ms, encode_ms, decode_ms }. Use alongside (not replacing)
   audio.cpp's heavy profiler.
10. **BackendPlan dynamic loading:** Port `transcribe_init_backends()` +
    `transcribe_init_backends_default()` — backend modules as DLLs next to the
    library. Essential for Python wheel / npm package distribution.
11. **Pin Silero VAD** to latest stable upstream release: Update `reference/silero-vad/`
    to `<VERSION>` from `https://github.com/snakers4/silero-vad`. Regenerate/validate
    `silero_vad_16k.safetensors` against `<VERSION>`. Create `model_specs/silero_vad.json`
    with `runtime.version = "<VERSION>"` and `runtime.source_url` → upstream release URL.
12. **`TRANSCRIBE_TEST_*` fault hooks:** Port both hooks. They verify `api_guard_*`
    containment under fault injection — critical for the wheel clean-install CI.

13. **`BackendPlan` + `BackendKind` enum (9 values):** Port the internal `BackendKind`
    {Unknown, Cpu, Metal, Vulkan, Cuda, Rocm, Sycl, Accel, OtherGpu} from
    `transcribe-backend.h`. Port `BackendPlan { requested (transcribe_backend_request),
    primary (ggml_backend_t), primary_kind (BackendKind), scheduler_list }` + the
    safe_* teardown trio (`safe_backend_free`, `safe_buffer_free`,
    `safe_sched_free` — all noexcept, null-checking, with
    `TRANSCRIBE_TEST_TEARDOWN_THROW` injection point). audio.cpp's
    `BackendConfig { type, device, threads }` maps to `BackendPlan.requested`.

14. **`KvResult` key namespaces:** Port the exact string-key namespaces so all GGUF
    models are introspectable via the C ABI. Recognized keys:
    `stt.capability.translate`, `stt.capability.lang_detect`,
    `stt.capability.streaming`, `general.languages`,
    `stt.translation.target_languages`, `stt.translation.pairs`.
    The `read_required_u32_kv` / `read_optional_string_kv` etc. helpers enforce type
    safety (BadType tristate). `read_capability_kv` and `read_languages_kv`
    post-load from these keys into `transcribe_capabilities`.

15. **Batch result accessors:** Port the full batch accessor family for
    multi-utterance parity testing: `transcribe_batch_n_results`,
    `transcribe_batch_status`, `transcribe_batch_full_text`,
    `transcribe_batch_raw_text`, `transcribe_batch_detected_language`,
    `transcribe_batch_n_segments/words/tokens`,
    `transcribe_batch_get_segment/word/token`,
    `transcribe_batch_n_speaker_segments`, `transcribe_batch_get_speaker_segment`,
    `transcribe_batch_get_timings`. Verified declared in `transcribe.h:2577-2645`.

16. **`transcribe_model` struct depth (limits + feature bits + load timing):** Port
    the full `transcribe_model` base (transcribe-model.h): `arch`, `variant`, `meta`
    (MetaMap), `backend`, `primary_backend`, `caps` (transcribe_capabilities),
    `feature_bits` (bit-per-`transcribe_feature`), `t_load_us`. The
    `LimitsBasis` struct { has_context_cap, audio_from_caps, model_max_ctx,
    prompt_overhead, gen_reserve, ms_per_audio_token, kv_elems_per_ctx_token }
    drives `effective_n_ctx` / `effective_max_audio_ms` / `max_kv_bytes`
    computation — this is transcribe.cpp's model-limit resolution, NOT a port of
    audio.cpp's `ModelSpec::limits`.

### 1.5.2 What audio.cpp Teaches transcribe.cpp (In Detail)

1. **Graph executor + standalone optimizer:** transcribe.cpp builds ggml graphs inline
   (ad-hoc `ggml_new_tensor` calls in `model.cpp`). audio.cpp has `GraphExecutor`
   (the runtime that builds and computes the graph) and a **separate** `GraphOptimizer`
   module (`engine::runtime::optimize_graph()`, NOT a SessionBase member) with 7
   rewrite kinds: BroadcastRepeat folding, CommutativeLhsRepeat folding,
   TwoSidedBroadcastRepeat folding, UnaryBroadcastRepeat folding,
   IdentityMaterialization elision, NoopNode elision, MetadataOnlyNode elision.
   The architecture implements layout conversion, operator fusion, and KV cache reuse
   as GraphOptimizer passes — those are NOT what this module does. KV cache reuse for
   sliding-window streaming families is handled by `flow_kv_cache.cpp` (a separate
   RuntimeCache concern), not by GraphOptimizer. Port GraphExecutor + GraphOptimizer
   for graph-level rewrites; adopt `flow_kv_cache` for streaming KV reuse.

2. **Modular module system:** audio.cpp has `ITplModule` abstraction
   (`src/framework/modules/module.h`) — 40+ module types (linear, attention,
   norm, activation, conv, lookup, etc.) with composable graph-building.
   transcribe.cpp's families each implement tensor math ad-hoc. After porting,
   new STT families can optionally use the graph executor.

3. **Artifact store + inter-task artifacts:** audio.cpp's `VoiceArtifact` /
   `ArtifactKind` (SpeakerEmbedding, StyleEmbedding, PromptEmbedding,
   AcousticTokens, AudioTokens, Midi, TranscriptAlignment, DiarizationState,
   VadState, Custom) lets downstream tasks inherit upstream outputs.
   transcribe.cpp has no cross-task artifact passing. The `ArchAdapter`
   exposes the artifact store through a new `transcribe_task_*_artifact()`
   accessor.

4. **Model spec system + download management:** `model_specs/*.json` +
   `model_registry_includes.inc` provides package metadata (download URLs,
   variant mappings, UI metadata). transcribe.cpp has no spec system. Port it:
   every transcribe.cpp family gets a `model_specs/<family>.json`. The
   `Loader::open()` checks the spec for the model's variant and capabilities
   before loading.

5. **Pipeline engine:** audio.cpp's `pipeline/` orchestrates multi-stage workflows
   (e.g., VAD → segmentation → ASR → diarization → alignment → TTS).
   transcribe.cpp does each stage at the CLI level. Port the pipeline engine for
   the unified CLI.

6. **Text post-processing pipeline:** `src/framework/text/text_normalization.cpp`
   — full PNC/ITN with Chinese + English support, unicode normalization,
   chunking, subtitle formatting. transcribe.cpp has no text normalization.

7. **Workspace management:** audio.cpp's `RuntimeWorkspace` provides scoped
   scratch allocation with auto-free on session end. transcribe.cpp's sessions
   manage scratch manually.

8. **Profiling:** audio.cpp's `src/framework/debug/profiler.h` + `trace.h`
   provide event tracing + timing scopes. Port as a backend to transcribe.cpp's
   logger when callback is NULL.

### 1.5.3 What transcribe.cpp Teaches audio.cpp (In Detail)

1. **Public C ABI:** The single biggest gift. audio.cpp has no C ABI; it's
   C++-only, limiting embeddability. The full `transcribe.h` (2499 lines)
   becomes the unified public surface. Existing audio.cpp families are
   accessible via `ArchAdapter` without rewriting their C++ internals.

2. **Exception discipline:** `api_guard_*` templates contain all C++ exceptions
   at every `extern "C"` boundary — `bad_alloc` → `TRANSCRIBE_ERR_OOM`,
   `std::exception` → `TRANSCRIBE_ERR_BACKEND`, `...` → `TRANSCRIBE_ERR_BACKEND`.
   Plus the `TRANSCRIBE_TEST_DEV_INIT_THROW` / `TRANSCRIBE_TEST_TEARDOWN_THROW`
   fault hooks that inject throws during init/teardown to prove containment.
   Audio.cpp sessions currently `throw` freely in destructors. Apply
   `api_guard_*` retroactively to all existing model sessions.

3. **Safe teardown pattern:** `safe_backend_free()` / `safe_sched_free()` /
   `safe_buffer_free()` — null-checking, no-throw wrappers. audio.cpp calls
   `ggml_backend_free()` / `ggml_backend_buffer_free()` /
   `ggml_backend_sched_free()` directly in model destructors. The
   `lint_teardown.cmake` CI gate forbids this. **Important scope note:**
   The lint targets `ggml_backend_free`, `ggml_backend_buffer_free`, and
   `ggml_backend_sched_free` — NOT `ggml_backend_graph_plan_free` (which
   frees a graph plan, not a backend handle). The `HostGraphPlan::~HostGraphPlan`
   in `core/backend.h` is exempt.

4. **Size-aware ABI structs:** The `struct_size` field 0 + `copy_out_prefix`
   pattern prevents ABI drift breakage. Apply to `TaskResult` and
   `CapabilitySet` cross-module returns in audio.cpp.

5. **Streaming state machine:** transcribe.cpp's 4-state machine (IDLE/ACTIVE/
   FINISHED/FAILED) + 3-checkpoint begin (preflight → optional `stream_validate`
   → post-clear `stream_begin`). Plus `transcribe_stream_get_text()` returning
   `full_text` / `committed_text` / `tentative_text` / `display_text` with
   `commit_policy` enum (AUTO/ON_FINALIZE/STABLE_PREFIX). audio.cpp's streaming
   is a flat push model with no committed/tentative distinction. Adopt the
   state machine universally.

6. **Abort callback + polled cancellation:** `transcribe_abort_callback` +
   `poll_abort()` (checked between decode steps) enables cooperative cancellation.
   audio.cpp has no cancellation mechanism. Port to all long-running tasks.

7. **Capability feature probing:** `transcribe_model_supports(feature)` with
   7-feature enum (INITIAL_PROMPT, TEMPERATURE_FALLBACK, LONG_FORM, CANCELLATION,
   PNC, ITN, DIARIZATION). audio.cpp's `CapabilitySet` is a 5-field struct with
   bool flags. Map `CapabilitySet` → `transcribe_capabilities` + `transcribe_feature`
   bits at the `ArchAdapter` boundary.

8. **Golden manifest testing:** `tests/golden/<family>/<variant>.manifest.json`
   + `tests/tolerances/<family>.json` with per-tensor tolerances. audio.cpp has
   no golden manifests. Adopt for ALL families.

9. **Spec decoding:** `spec_k_drafts` (-1/0/>0) on `transcribe_run_params` +
   `supports_spec_decode` capability. audio.cpp has basic sampling only.

10. **Kaldi fbank with LFR+CMVN:** transcribe.cpp's `KaldiFbankFrontend`
    supports LFR (Low Frame Rate) + CMVN (Cepstral Mean Variance Normalization).
    audio.cpp's `kaldi_fbank.cpp` is basic. Port the transcribe.cpp version.

11. **Dynamic backend modules:** `transcribe_init_backends()` loads ggml
    backend modules as DLLs next to the library (`TRANSCRIBE_GGML_BACKEND_DL`).
    audio.cpp links backends statically. Dynamic loading enables
    modular wheel/distribition.

12. **Frontend configuration:** transcribe.cpp's `MelFrontend` + `KaldiFbankFrontend`
    are configurable (4 normalize modes, 3 pad modes, hann_symmetric/hann_periodic,
    LFR parameters, CMVN). audio.cpp's frontend is fixed.

13. **Long-form Whisper chunker:** transcribe.cpp's Whisper chunker with
    `max_scope` + `max_release` + `max_task_duration` for long-form transcription.
    audio.cpp's chunking is in `audio/chunking.cpp` (different approach).

14. **Numerical validation tooling:** `scripts/compare_tensors.py` for per-tensor
    comparison, `scripts/validate.py` for the full ref→cpp→compare pipeline,
    `scripts/preflight.py` for cheap metadata gates. audio.cpp has no
    equivalent.

15. **Family extension struct pattern:** `struct transcribe_ext { uint64_t size;
    uint32_t kind; }` embedded as the first field of typed family-specific
    parameter structs. FourCC kind constants. `transcribe_model_accepts_ext_kind()`
    validation. This is a clean, forward-compatible extension mechanism.

### 1.5.4 Cross-Pollination: New Capabilities Neither Parent Has

After the bidirectional merge, the unified `speech.cpp` has capabilities
that neither parent had alone:

1. **C ABI callable multi-task engine:** Both STT and TTS/voice cloning/etc. via
   a single C ABI surface (`transcribe_task_*`). The original transcribe.cpp
   only had STT; the original audio.cpp had no C ABI at all.

2. **Golden manifests for TTS and voice cloning:** transcribe.cpp's golden
   manifest methodology applied to audio.cpp's TTS families (mel-spectrogram
   parity, vocoder waveform matching). Neither parent had numerical parity
   gates for non-STT speech audio.

3. **Graph-optimized STT:** STT families (from transcribe.cpp) running through
   audio.cpp's `GraphExecutor` + `GraphOptimizer` with KV cache reuse for
   streaming families.

4. **Cross-task artifact passing via C ABI:** Speaker embeddings from ASR diarization
   passed to TTS conditioning via `transcribe_task_*_artifact()` accessors.
   Enabled by audio.cpp's `VoiceArtifact` system exposed through transcribe.cpp's
   extension struct pattern.

5. **Dynamic backend loading for ALL tasks:** STT, TTS, vocoders, etc. all benefit
   from transcribe.cpp's DLL-next-to-library backend loading.

6. **Unified CLI with task dispatch:** A single `transcribe` CLI that dispatches to
   all task types via the C ABI, using audio.cpp's pipeline engine to chain
   multi-stage workflows (VAD → ASR → diarization → alignment → TTS).

### 1.5.5 The Upgrade Cascade

The integration follows a cascade where each upgrade benefits ALL families,
not just the newly-ported STT ones. This is the "upgrade both sides" principle:

**Cascade order (Phase 0 early wins):**

1. **C ABI surface** — enables bindings. (Benefit: transcribe.cpp families;
   enables: all unified families)
2. **`api_guard_*`** — exception containment. (Benefit: all C ABI callers;
   enables: safe exceptions across FFI)
3. **`safe_*` teardown** — leak-free cleanup. (Benefit: all families;
   enables: `lint_teardown.cmake` CI gate)
4. **`struct_size` + `copy_out_prefix`** — ABI versioning. (Benefit: all
   cross-module struct returns; enables: forward-compatible bindings)
5. **`KvResult` metadata helpers** — safe GGUF metadata reading. (Benefit:
   all families; enables: golden manifests, model introspection)
6. **`transcribe_timings`** — lightweight profiling. (Benefit: all families;
   enables: performance regression detection)
7. **`BackendPlan` + dynamic modules** — modular backend loading. (Benefit:
   all families; enables: wheel distribution)
8. **Capability feature probing** — `transcribe_model_supports()`. (Benefit:
   all families; enables: runtime feature detection in bindings)
9. **Streaming state machine** — committed/tentative, abort callbacks. (Benefit:
   all streaming tasks; enables: flicker-free UI, cooperative cancellation)
10. **Golden manifests + tolerances** — numerical parity gates. (Benefit: all
    families; enables: regression-free consolidation)
11. **Silero VAD pinning** — reproducible VAD. (Benefit: VAD usage;
    enables: numerical parity for VAD-dependent pipelines)
12. **Graph executor integration** — optimization passes. (Benefit: all
    graph-building families; enables: KV cache reuse for streaming)
13. **Model spec system** — package management. (Benefit: all families;
    enables: automated download + variant resolution)

Each step in the cascade builds on the previous. Phase 0 implements steps 1-11;
subsequent phases apply step 12 and beyond to each family batch.

### 1.5.6 Concrete Early Wins

These 12 items ship from Phase 0 and benefit BOTH new STT families AND existing
audio.cpp families immediately:

1. **Port `safe_*` teardown** (`safe_backend_free`, `safe_sched_free`,
   `safe_buffer_free`) into `src/runtime/`. Drop into EVERY existing audio.cpp
   model session. Add `lint_teardown.cmake` CI gate that scans `src/` and `include/`
   and fails on raw `ggml_backend_free`/`ggml_backend_buffer_free`/
   `ggml_backend_sched_free` calls in library code. **Does NOT target
   `ggml_backend_graph_plan_free` (graph plan free ≠ backend free).**
   (Source: `src/transcribe-backend.h:89-97`)

2. **Port `api_guard_*` templates** (`api_guard_status`, `api_guard_value`,
   `api_guard_void`) from `src/transcribe.cpp:70-109`. Every `extern "C"` entry
   point in the unified dispatcher routes through these. Apply `api_guard_*` at
   C++ boundaries too: CLI `main()`, server request handler. Add
   `API_GUARD_ENFORCED.cmake` as a NEW CI gate scanning for `extern "C"` functions
   not routing through `api_guard_*`.

3. **Port the `struct_size` + `copy_out_prefix` ABI pattern** (`src/transcribe-abi.h`)
   to audio.cpp's `TaskResult` and `CapabilitySet` cross-module returns. Apply to
   `VoiceTaskKind`/`RunMode` enums (raw-int reads via `enum_field_raw`).

4. **Port the golden-manifest + per-tensor tolerance methodology** for ALL
   families. Add `transcribe_common_example` (WAV loader) to the test build.
   Generate golden manifest + tolerances for audio.cpp's existing non-ASR
   families (TTS, voice cloning) to validate the methodology works for non-STT.

5. **Create `model_specs/*.json` for every transcribe.cpp family.** Port the
   spec format from audio.cpp. The `Loader::open()` checks the spec for the
   model's variant and capabilities before loading.

6. **Port `transcribe_feature` enum (7 values) + `transcribe_model_supports()`
   from Day 1.** `ArchAdapter` maps audio.cpp's `CapabilitySet` into the
   feature-probe enum. This is the unified capability surface for bindings.

7. **Port `transcribe_log_callback` + 6 log levels + ggml routing.**
   `transcribe_log_set_level()` + `TRANSCRIBE_LOG_LEVEL_CONT` for streaming.
   Route ggml's compute progress into the callback . Integrate audio.cpp's
   `trace.h`/`profiler.h` as a backend when callback is NULL (stderr default).

8. **Port `transcribe_debug_dump()` helpers** (`src/transcribe-debug.h`) for
   numerical validation. Adopt transcribe.cpp's dump format (raw `.f32` +
   `.json` sidecar).

9. **Port `MelFrontend` + `KaldiFbankFrontend`** into `src/runtime/`. These are
   strictly more configurable than audio.cpp's `kaldi_fbank.cpp` (basic, no
   LFR/CMVN). New STT families use transcribe.cpp's versions; audio.cpp's basic
   kaldi_fbank stays for non-STT tasks that need domain-specific behavior.

10. **Port `KvResult` tri-state metadata helpers** (`transcribe-meta.h`).
    The Absent/Ok/BadType distinction: "key not present" is usually fine
    (optional), but "wrong type" is always a converter bug. Adopt for ALL
    families — this is critical for golden-manifest validation to distinguish
    "feature not configured" from "feature misconfigured".

11. **Port `BackendPlan` + dynamic backend module loading** (`TRANSCRIBE_GGML_BACKEND_DL`).
    Adopt `transcribe_init_backends()` + `transcribe_init_backends_default()`.
    Backend modules as DLLs next to the shared library — essential for Python
    wheel / npm package distribution.

12. **Port `TRANSCRIBE_TEST_DEV_INIT_THROW` + `TRANSCRIBE_TEST_TEARDOWN_THROW`
    fault hooks.** They verify `api_guard_*` containment under fault injection
    — critical for the wheel clean-install CI. Present-but-empty values are
    inert; non-empty values inject throws during init/teardown.

### 1.5.7 Generalized C ABI: Beyond STT

**CORRECTION:** The `transcribe_task_*` entry points (TTS, voice cloning,
separation, VC, STS, alignment, diarization, denoise) are **PROPOSED** unified C
ABI extensions, not existing transcribe.cpp APIs. The existing transcribe.h API
is STT-only (`transcribe_run`, streaming, etc.). These generalizations are designed
during Phase 0 and stabilized before bindings (Phase 4).

The C ABI must not be STT-only. `speech.cpp` needs the C ABI to expose
**all** audio.cpp task types so that bindings, the server, and embedded callers
get a single unified entry surface. The approach:

**Generalized task interface** (`transcribe_task_*` entry points generalize the
STT-specific `transcribe_run`):

| C ABI entry point | Maps to audio.cpp TaskKind | Status | Notes |
|---|---|---|---|
| `transcribe_run` (existing) | `VoiceTaskKind::Asr` | Existing ABI | Offline transcription. PCM in, text out. NOT broken. |
| `transcribe_task_run` (proposed) | `VoiceTaskKind::Asr` | Proposed | Generalized offline run. Replaces direct `transcribe_run` usage in bindings. |
| `transcribe_task_run_tts` (proposed) | `VoiceTaskKind::Tts` | Proposed | Text in, audio out. |
| `transcribe_task_run_tts_batch` (proposed) | `VoiceTaskKind::Tts` | Proposed | Batch TTS generation. |
| `transcribe_task_voice_clone` (proposed) | `VoiceTaskKind::VoiceCloning` | Proposed | Reference audio + prompt → voice embedding artifact. |
| `transcribe_task_separation` (proposed) | `VoiceTaskKind::SourceSeparation` | Proposed | Multi-channel audio → separated stems artifact. |
| `transcribe_task_vc` (proposed) | `VoiceTaskKind::VoiceConversion` | Proposed | Source audio + target voice → converted audio. |
| `transcribe_task_sts` (proposed) | `VoiceTaskKind::SpeechToSpeech` | Proposed | Source audio + target voice → speech output. |
| `transcribe_task_align` (proposed) | `VoiceTaskKind::Alignment` | Proposed | Audio + text → word/token timestamps. Uses Qwen3-ASR's forced aligner. |
| `transcribe_task_diarize` (proposed) | `VoiceTaskKind::Diarization` | Proposed | Uses Sortformer (transcribe.cpp) or audio.cpp's `sortformer_diar`. |
| `transcribe_task_vad` (proposed) | `VoiceTaskKind::Vad` | Proposed | Uses Silero VAD (upstream-pinned version). |

**Open design questions for the generalized C ABI (to resolve in Phase 0):**
1. How does a single `transcribe_task_request` struct carry all task input types
   (audio, text, voice condition, artifacts)? Proposed: use the `transcribe_ext`
   pattern — `transcribe_task_request` has a `family(ext)` slot that points to a
   task-specific extension struct (e.g. `transcribe_task_tts_request` for TTS).
2. How does streaming map? Each non-STT task gets its own streaming entry points
   (`transcribe_task_<kind>_stream_begin/feed/finalize/reset`) OR a single
   generalized `transcribe_task_stream_*` set with a `task` discriminator.
3. How do the existing `transcribe_run`/`transcribe_stream_*` callers (bindings,
   CLI) migrate? Proposed: keep `transcribe_run` as-is for backward compat;
   the CLI/server migrate to `transcribe_task_run` gradually.
4. How do committed/tentative semantics apply to audio output (TTS streaming)?
   Proposed: committed = yielded audio chunks (flushed, won't be re-synthesized),
   tentative = buffered audio (may be revised if the model backtracks).

**Unified input/output model:** A single `transcribe_task_request` struct carries
all task inputs (audio, text, voice condition, artifacts, options). A single
`transcribe_task_result` carries all task outputs (audio, text, segments, artifacts,
timings). The `task` field selects which `VoiceTaskKind` to dispatch. The `ArchAdapter`
maps `VoiceTaskKind` back to the audio.cpp session path.

**Generalized sessions:** The streaming state machine (IDLE/ACTIVE/FINISHED/FAILED)
applies to ALL streaming tasks — TTS token streaming, voice conversion streaming,
speech-to-speech streaming — not just ASR. The committed/tentative model applies
to text output (ASR, alignment) and audio output (TTS, generation — committed
= yielded chunks, tentative = buffered).

**Key insight:** The `transcribe_session` and `transcribe_model` base structs
already store results in generic vectors (segments/words/tokens/text). For
non-STT tasks, the `task` field determines which accessor functions are valid.
The `ArchAdapter` translates audio.cpp's `TaskResult` (with its `optional
audio_output`, `text_output`, `speech_segments`, etc.) into the appropriate result
vectors on the transcribe_session.

### 1.5.8 Upgrade Maps per Existing audio.cpp Family

Every existing audio.cpp family receives a set of transcribe.cpp infrastructure
upgrades. The specific set depends on the family's task type, but the
"early win" upgrades (§1.5.6 items 1-12) apply to ALL families universally.

**All families (universal upgrades):**
- `safe_*` teardown (item 1)
- `api_guard_*` at ABI boundary + C++ boundaries (item 2)
- `struct_size` + `copy_out_prefix` on cross-module returns (item 3)
- `KvResult` metadata helpers (item 10)
- `transcribe_timings` (item 9 of §1.5.1)
- `BackendPlan` dynamic loading (item 11)
- `CapabilitySet` → `transcribe_capabilities` + `transcribe_feature` mapping (item 6)
- `model_specs/<family>.json` creation (item 5)
- `TRANSCRIBE_TEST_*` fault hooks (item 12)
- Unified logging (item 7)

**ASR families (qwen3_asr, voxtral_realtime, citrinet_asr, parakeet_tdt,
nemotron_asr, hviske_asr, higgs_audio_stt, vibevoice_asr, kroko_asr):**
- `transcribe_feature::LONG_FORM` + abort callback + `poll_abort()`
- Whisper-style long-form chunker with `max_scope` + `max_release`
- Golden manifests + tolerances
- Spec decoding (`supports_spec_decode`)
- `supports_language_detect` capability (where applicable)
- Batch dispatch (`transcribe_run_batch`)

**STT families being ported from transcribe.cpp (whisper, moonshine, etc.):**
- Adopt transcribe.cpp full mel frontend (4 normalize modes, 3 pad modes, hann variants).
- Adopt transcribe.cpp KaldiFbankFrontend (Hamming, LFR, CMVN) for STT tasks.
- Adopt transcribe.cpp BPE + SentencePiece tokenizer with tok JSON metadata.
- Adopt transcribe.cpp GGUF weight loader, KV cache management, and spec sampling.
- Adopt transcribe.cpp golden-manifest + per-tensor tolerance testing.
- Adopt transcribe.cpp abort callback + exception containment (api_guard_*).
- Adopt transcribe.cpp safe_* teardown + lint_teardown.cmake CI gate.
- Adopt transcribe.cpp streaming state machine (IDLE/ACTIVE/FINISHED/FAILED).
- Adopt transcribe.cpp transcribe_stream_commit_policy (AUTO/ON_FINALIZE/STABLE_PREFIX).
- Preserve the existing RuntimeSessionBase-based session for the C++ path; the lightweight transcribe_session is used for the C ABI path via ArchAdapter.

**VAD family (silero_vad, marblenet_vad):**
- Silero VAD version to be pinned from upstream (item 11 of §1.5.6)
- Golden manifests (VAD frame-level parity)
- `stt.vad_version` GGUF KV + version unit test
- `transcribe_feature::INITIAL_PROMPT` (VAD can skip silence)

**Forced aligner family (qwen3_forced_aligner):**
- `safe_*` teardown
- `api_guard_*`
- `KvResult` metadata helpers
- `transcribe_feature` probe: **`SUPPORTS_ALIGNMENT` needs to be added as
  `TRANSCRIBE_FEATURE_ALIGNMENT` (feature #7, append-only enum extension)** —
  this is a proposed unified feature enum extension, not yet existing in
  transcribe.cpp. The enum currently has 7 values (0-6); alignment would be
  value 7, appended (backward compatible).
- Golden manifests (alignment boundary precision)
- Input length contracts (max audio for alignment)
- `transcribe_timings` (encode_ms, decode_ms)

**TTS families (chatterbox, qwen3_tts, stable_audio, roformer, demucs, etc.):**
- `safe_*` teardown
- `api_guard_*`
- `KvResult` metadata helpers
- `transcribe_timings`
- Golden manifests (mel-spectrogram parity for vocoder-coupled models)
- `transcribe_feature::SUPPORTS_STYLE_CONDITION` (where applicable, for prompt-
  conditioned TTS like chatterbox)

**Voice cloning / voice conversion (rvc, seed_vc, etc.):**
- `safe_*` teardown
- `api_guard_*`
- Golden manifests (voice encoder embedding parity)
- `transcribe_feature::SUPPORTS_SPEAKER_REFERENCE`

**Speaker recognition (titanet_spk, ecapa_tdnn_spk):**
- `safe_*` teardown
- `api_guard_*`
- Golden manifests (embedding cosine similarity parity)
- `transcribe_feature::SUPPORTS_SPEAKER_REFERENCE`

**Speaker diarization (sortformer_diar):**
- `safe_*` teardown
- `api_guard_*`
- `KvResult` metadata helpers
- `transcribe_feature::DIARIZATION`
- Golden manifests (speaker turn boundary parity)
- Input length contracts (max audio for diarization)
- `transcribe_timings` (encode_ms, decode_ms)

**Forced aligner (qwen3_forced_aligner):**
- `safe_*` teardown
- `api_guard_*`
- `KvResult` metadata helpers
- `transcribe_feature` probe: **`SUPPORTS_ALIGNMENT` needs to be added as
  `TRANSCRIBE_FEATURE_ALIGNMENT` (feature #7, append-only enum extension)** —
  this is a proposed unified feature enum extension, not yet existing in
  transcribe.cpp. The enum currently has 7 values (0-6); alignment would be
  value 7, appended (backward compatible).
- Golden manifests (alignment boundary precision)
- Input length contracts (max audio for alignment)
- `transcribe_timings` (encode_ms, decode_ms)

### 1.5.9 Pinned Dependency: Silero VAD

**1. The VAD situation:** Both projects bundle a Silero VAD model at
`assets/framework/models/silero_vad/silero_vad_16k.safetensors` with a Python
reference at `reference/silero-vad/` . **Neither pins a
version** — the model asset and reference scripts float with whatever was
committed.

**2. The pinning:** Pin to the **latest stable Silero VAD release** from
`https://github.com/snakers4/silero-vad`. The 4-layer CNN + decoder-head VAD
architecture has been stable across v5→v31, but KV metadata conventions may
differ. Create `model_specs/silero_vad.json`:
```json
{
  "family": "silero_vad",
  "runtime": {
    "source_url": "https://github.com/snakers4/silero-vad",
    "version": "<VERSION>",
    "version_source": "Silero VAD upstream release tag"
  },
  "asset": {
    "format": "safetensors",
    "sha256": "<pinned>"
  }
}
```

**3. Asset verification:** Verify that audio.cpp's `src/models/silero_vad/`
reads the existing `silero_vad_16k.safetensors` weights correctly against the
pinned upstream version. If the asset predates the pin, regenerate from the
pinned ONNX weights using a pinned converter script.

**4. Version-pinned test fixture:** Add a CI test
(`tests/silero_vad_version_unit.cpp`) that asserts the runtime reads a
`stt.vad_version` GGUF KV or similar version-tagged metadata, ensuring the loaded
weights match the pinned `<VERSION>`. If the version mismatches, the loader
returns `TRANSCRIBE_ERR_UNSUPPORTED_VARIANT` (or a new
`TRANSCRIBE_ERR_UNSUPPORTED_VAD_VERSION`). **NOTE:** This is a proposed test —
the `stt.vad_version` KV must be written by the converter during GGUF creation
for audio.cpp's silero_vad model.

---


### 1.5.10 Backend Compatibility Matrix

The unified speech.cpp must support all backend types across all task kinds. The
audio.cpp BackendType enum (6 values: Cpu, Cuda, Hip, Vulkan, Metal,
BestAvailable) maps to transcribe.cpp's transcribe_backend_request enum
(AUTO=0, CPU=1, METAL=2, VULKAN=3, CPU_ACCEL=4, CUDA=5, ROCM=6).

**Compatibility matrix (task x backend):**

| Task (VoiceTaskKind) | CPU | CUDA | Vulkan | Metal | CPU_ACCEL | ROCM |
|----------------------|-----|------|--------|-------|-----------|------|
| Vad (Silero VAD) | Full | Full | Full | Full | Full | Full |
| Asr (STT) | Full | Full | Full | Full | Full | Full |
| Tts | Full | Full | Full | Full | Full | Full |
| VoiceCloning | Full | Full | Full | Metal preferred | Full | Limited |
| VoiceConversion | Full | Full | Full | Metal preferred | Full | Limited |
| SourceSeparation | Full | Full | Full | Full | Full | Full |
| AudioGeneration | Full | Full | Full | Full | Full | Full |
| SpeechToSpeech | Full | Full | Full | Full | Full | Full |
| Diarization | Full | Full | Full | Full | Full | Full |
| Alignment | Full | Full | Full | Full | Full | Full |
| Svc | Full | Full | Full | Metal preferred | Full | Limited |
| Midi | Full | N/A | N/A | N/A | N/A | N/A |

**Integration points:**
1. BackendConfig { BackendType type, int device, int threads } -- audio.cpp's
   config struct. Unified path extends it with gpu_split_mode, n_gpu_layers,
   tensor_split, gpu_mode from transcribe.cpp's BackendPlan (via D16).
2. BackendManager (audio.cpp) + backend_module_load() (transcribe.cpp
   dynamic loading via TRANSCRIBE_GGML_BACKEND_DL) -- unified via
   BackendFactory that tries transcribe.cpp's dlopen path first, falls back
   to audio.cpp's compiled-in backends.
3. Threadpool: audio.cpp uses ggml_threadpool_t (ggml-native); transcribe.cpp
   uses omp + std::thread. Unify on ggml threadpool (ggml convergence,
   pinned SHA 8c63e70982c95ceb862e3a1073a2c1beef75d60a).
4. Device enumeration: transcribe_device_count() + transcribe_device_get()
   wrap audio.cpp's BackendManager::enumerate_devices().

## 2. Family Porting Matrix

### 2.1 Current State Matrix

**transcribe.cpp families (18, in `src/arch/`):**

| Family | Subdir | Arch Pattern | P0/P1/P2 |
|--------|--------|-------------|----------|
| whisper | whisper | encoder-decoder + xattn | P0 |
| moonshine | moonshine | encoder + CTC | P0 |
| moonshine_streaming | moonshine_streaming | streaming encoder + CTC | P1 |
| qwen3_asr | qwen3_asr | audio-LLM (token injection) | P0 (consolidate) |
| sensevoice | sensevoice | audio-LLM (token injection) | P0 (consolidate) |
| funasr_nano | funasr_nano | encoder + CTC | P0 (consolidate) |
| parakeet | parakeet | encoder + transducer/TDT | P1 |
| voxtral | voxtral | encoder-decoder + xattn | P1 |
| voxtral_realtime | voxtral_realtime | streaming encoder-decoder | P1 |
| canary | canary | encoder-decoder + xattn | P1 |
| canary_qwen | canary_qwen | audio-LLM (token injection) | P1 |
| gigaam | gigaam | encoder-CTC / RNNT | P1 |
| granite | granite | encoder-CTC | P1 |
| granite_nar | granite_nar | non-autoregressive | P1 |
| cohere | cohere | encoder-decoder + xattn | P1 |
| medasr | medasr | encoder + CTC | P2 |
| moss | moss | encoder-decoder | P2 |
| sortformer | sortformer | speaker embedding | P1 |

**audio.cpp families with ASR overlap (consolidation priority):**

| Family | Location in audio.cpp | transcribe.cpp equivalent | Action |
|--------|----------------------|--------------------------|--------|
| qwen3_asr | `src/models/qwen3_asr/` | `src/arch/qwen3_asr/` | CONSOLIDATE — replace audio.cpp impl |
| fun_asr_nano | `src/models/fun_asr_nano/` | `src/arch/funasr_nano/` | CONSOLIDATE — replace audio.cpp impl |
| sense_asr | `src/community_models/sense_asr/` | `src/arch/sensevoice/` | CONSOLIDATE — replace audio.cpp impl |
| parakeet_tdt | `src/community_models/parakeet_tdt/` | `src/arch/parakeet/` | CONSOLIDATE — replace audio.cpp impl |
| voxtral_realtime | `src/models/voxtral_realtime/` | `src/arch/voxtral_realtime/` | CONSOLIDATE — replace audio.cpp impl |
| qwen3_forced_aligner | `src/models/qwen3_forced_aligner/` | none | PORT — transcribe.cpp has no aligner |
| hviske_asr, citrinet_asr, nemotron_asr, higgs_audio_stt, vibevoice_asr | `src/models/` | none | KEEP — transcribe.cpp has no equivalents |

**audio.cpp non-ASR families (no overlap, receive upgrades):**
chatterbox, qwen3_tts, stable_audio, roformer, demucs, rvc, seed_vc, fish_audio,
heartmula, index_tts2, irodori_tts, marblenet_vad, miocodec, miotts, muscle,
omnivoice, pocket_tts, supertonic, vevcpm2, vibevoice, confucius4_tts, dots_tts,
dramabox, ace_step, muscriptor, neutts, kokoro_tts, glm_tts, kroko_asr, outetts,
inflect_v2, vietneu_tts, voxcpm2, sortformer_diar.

**Silero VAD:** Both projects have `silero_vad`. Pin to latest stable upstream tag. See §1.5.9.

### 2.2 Arch Patterns Reference

The transcribe.cpp `docs/porting/0-porting.md` classifies STT families by their
architectural pattern (from `arch.run` — the per-utterance inference path):

1. **Encoder-CTC:** Encoder → CTC linear head, argmax over blank-skip.
   Families: moonshine, gigaam, granite (some), funasr_nano (some).
2. **Encoder-Decoder + cross-attention (xattn):** Encoder → autoregressive decoder
   with cross-attention over encoder output. Families: whisper, canary, cohere,
   voxtral (non-realtime).
3. **Streaming encoder + CTC:** Streaming encoder with causal padding + chunking.
   Families: moonshine_streaming.
4. **Audio-LLM (token injection):** Audio encoder → token injection into a
   causal LM decoder (the "thinker"). Families: qwen3_asr, sensevoice,
   canary_qwen, moss.
5. **Transducer (TDT):** Encoder + prediction network + RNNT joint. Family: parakeet.
6. **Non-autoregressive (NAR):** Direct token prediction, no AR decoder. Families:
   granite_nar.
7. **Speaker embedding:** Speaker diarizer/identifier (1-hot or softmax). Family:
   sortformer.

The `causal_lm` shared module (§1.4) is used by: qwen3_asr, funasr_nano,
voxtral, voxtral_realtime, canary_qwen, moss, cohere. The conformer shared module
(`src/conformer/`) is used by: some whisper variants, parakeet, canary, medasr.
SANM blocks (`src/sanm/`) by: sensevoice, funasr_nano (some). Granite's Shaw
relative attention (`src/granite_conformer/`) by: granite, granite_nar.

### 2.3 Porting Priority Matrix

**P0 (Critical path — ASR performance + consolidation):**
- whisper, qwen3_asr (consolidate), moonshine, moonshine_streaming, sensevoice
  (consolidate), funasr_nano (consolidate)

**P1 (Broad model coverage):**
- parakeet, canary, canary_qwen, gigaam, granite, granite_nar, cohere, voxtral,
  voxtral_realtime (consolidate), sortformer

**P2 (Specialty):**
- medasr, moss

**P0 Rationale:** These families give the best ASR quality + widest language/
accent coverage. Whisper is the flagship; Moonshine is the lightweight
alternative; Qwen3-ASR is the best overall; SenseVoice is the Chinese-market
heavyweight. Consolidating the overlapping audio.cpp implementations avoids
code duplication.

### 2.4 Overlap Resolution Strategy

For families that exist in BOTH projects (qwen3_asr, sensevoice, funasr_nano,
parakeet, voxtral_realtime):

1. **transcribe.cpp's implementation is the canonical Arch implementation** for
   the C ABI path. It is ported into `src/arch/<family>/`.
2. **audio.cpp's implementation is replaced** — its loader, session, and model
   classes are removed; the CMake target is removed from `AUDIOCPP_MODEL_SET`.
3. **The C++ path** (CLI, server, WebUI) accesses the consolidated family via
   `ArchAdapter` (which wraps the consolidated C++ session that the Arch
   implementation provides).
4. **Model specs** are consolidated into one `model_specs/<family>.json`
   (transcribe.cpp's variant naming takes precedence; audio.cpp's download URLs
   are merged in if different).

Exception: For families where audio.cpp has the ONLY implementation and
transcribe.cpp does NOT (e.g., hviske_asr, citrinet_asr, nemotron_asr, higgs_audio_stt,
vibevoice_asr), audio.cpp's implementation stays but is upgraded with the
transcribe.cpp infrastructure (safe_*, api_guard_*, golden manifests, etc.).

For families where transcribe.cpp has the ONLY implementation and audio.cpp does
NOT (whisper, moonshine, canary, granite, cohere, medasr, moss, gigaam,
granite_nar, canary_qwen), transcribe.cpp's implementation is ported
unchanged into `src/arch/<family>/`.

---

## 3. Family Porting Stages (Adopted from transcribe.cpp's `docs/porting/`)

The 8-stage pipeline from `transcribe.cpp/docs/porting/0-porting.md` is the
mandatory quality gate for each family port. The pipeline is verified against the
actual documentation: `0-porting.md` (overview), `1-reference-research.md` +
`1a-intake.md` (intake + `_intake-schema.json`), `2-artifacts-and-goldens.md`
(manifest format with `transcribe-golden-manifest-v1` schema, cache keys = audio
sha256 + dump-point list + dtype), `3-conversion.md`, `4-numerical-validation.md`
(dump principles: gate / informational / debug tensor roles), `4a-numerical-troubleshooting.md`,
`5-benchmarks.md`, `6-family-checklist.md` (family readiness checklist).
`scripts/validate.py` reads the manifest + tolerance file directly and runs the
ref→cpp→compare pipeline.

```
┌───────────┐  ┌────────┐  ┌─────────┐  ┌─────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│ 1 Intake  │ → │ 2 Oracle │ → │ 3 Convert │ → │ 4 C++ │ → │ 5 Quants │ → │ 6 Bench │ → │ 7 WER │ → │ 8 Ship │
└───────────┘  └────────┘  └─────────┘  └─────┘  └──────┘  └──────┘  └──────┘  └──────┘
```

### Stage 1 — Intake
- **Artifacts:** `docs/porting/families/<family>.md` (research + capability validation table).
  `reports/porting/<family>/<variant>/intake.json` (machine-readable, schema-validated
  against `_intake-schema.json`).
- **Gates:** Preflight Gate A (family identified, architectural pattern confirmed,
  reference framework selected).
- **Deliverable:** Family note with `## Capability Validation` table.

### Stage 2 — Oracle
- **Actions:** Run the **reference framework** (`transcribe.cpp` baseline) on every
  manifest case. Capture tensor dumps + transcripts + provisional tolerances.
- **Tools:** `scripts/envs/<family>/pyproject.toml` + `uv run`.
- **Artifacts:** `reports/porting/<family>/<variant>/intake.json`,
  `reports/porting/<family>/<variant>/_porting-log.md`.
- **Gates:** Preflight Gate B (reference run is deterministic, dumps are stable,
  dtype consistency verified, frontend config cross-checked).
- **Tolerances:** Sized from per-tensor magnitude: `1e-4 × p99_abs` / `1e-5 × rms`,
  with a `1e-6` floor.

### Stage 3 — Convert
- **Actions:** Produce reference-dtype GGUF using `scripts/convert-<family>.py`.
- **Gates:** Preflight Gate C (tensor names/layout match reference dump conventions;
  tokenizer + frontend metadata correct; HParams + capability KVs correct; dtype
  consistency cross-checked).

### Stage 4 — C++
- **Actions:** Implement `src/arch/<family>/` (model.cpp, model.h, arch.cpp,
  compute/ if needed). Register in `transcribe-arch.cpp`.
- **Gates:** Ref-dtype WER vs Oracle reference baseline passes (per-tensor tolerances
  from `tests/tolerances/<family>.json`).
- **Tests:** `<family>_smoke`, `<family>_e2e_smoke`, tokenize parity (if BPE/SP).

### Stage 5 — Quants
- **Actions:** Generate shipped quant matrix. CLI smoke each GGUF.
- **Gates:** Quant CLI output-validity passes (NOT numerical comparison — use
  transcription output validity only).

### Stage 6 — Bench
- **Actions:** Performance matrix (CPU + GPU if available). `scripts/validate.py all` re-run.
- **Gates:** Bench comparison with configured thresholds.

### Stage 7 — WER
- **Actions:** Full release WER sweep.
- **Gates:** Ref-dtype HARD gate vs Oracle (exact tolerances). Quant WER
  human-reviewed (not auto-gated).

### Stage 8 — Ship
- **Artifacts:** `docs/models/<variant>.md` (model card), HF YAML, HF README,
  README supported-model list updated. Release pin sentence in model card.

### Required Artifacts Per Family
- `docs/porting/families/<family>.md`
- `reports/porting/<family>/<variant>/intake.json`
- `reports/porting/<family>/<variant>/_porting-log.md`
- `scripts/envs/<family>/pyproject.toml`
- `scripts/dump_reference_<family>_<reference>.py`
- `scripts/convert-<family>.py`
- `tests/golden/<family>/<variant>.manifest.json`
- `tests/tolerances/<family>.json`
- `src/arch/<family>/` (model.cpp, model.h, arch.cpp)
- `src/transcribe-arch.cpp` (registry entry)
- `CMakeLists.txt` / `tests/CMakeLists.txt`
- Smoke tests: `<family>_smoke.cpp`, `<family>_e2e_smoke.cpp`,
  `<family>_real_smoke.cpp` (gated behind `TRANSCRIBE_BUILD_REAL_MODEL_TESTS`),
  E2E public ABI smoke.

---

## 4. Detailed Implementation Plan


## 4.0 Phase 0.0: Speech.cpp Musical Delta Capture (pre-requisite)

Before Phase 0 kickoff, the speech.cpp-exclusive Music3 changes must be captured. This prevents loss of the `muscriptor` (Music3) family during the unified refactor.

- Capture: `git diff 52080cd..ee940b0 -- src/models/muscriptor/ model_specs/muscriptor.json`
- Save as: `patches/speech-cpp-music3-delta.patch`
- speech.cpp HEAD is `ee940b0` ("Fix Music3 component GGUF symlink overrides"), 6 commits ahead of audio.cpp upstream base `52080cd`.

### Phase 0: Foundation & ABI Bridge (4-6 weeks for solo engineer; 2-3 weeks for 3-4 engineer team)

**Goal:** Establish the unified C ABI layer on top of the audio.cpp framework.
Gate: ggml convergence build passes (§0.6) before any family port begins.

1. **Reference commit pinning + Silero VAD pinning:** Record exact HEAD commits
   of both upstream repos as `docs/porting/reference_commits.md`. Also pin Silero VAD
   latest stable release (from `https://github.com/snakers4/silero-vad`). Record the exact
   tag. Also record ggml SHA `8c63e70982c95ceb862e3a1073a2c1beef75d60a`.
   ```bash
   cd speech.cpp && git remote add upstream https://github.com/0xShug0/audio.cpp.git
   cd ../audio.cpp && git log --oneline -1  # → record hash
   cd ../transcribe.cpp && git log --oneline -1  # → record hash
   ```

2. **Port the public C ABI:** Copy `include/transcribe.h` →
   `include/transcribe/transcribe.h` (guard renamed to `TRANSCRIBE_TRANSCRIBE_H`).
   Port `include/transcribe/extensions.h` + all family extension headers
   into `include/transcribe/`. Add `transcribe.abihash`.

3. **Port the internal runtime headers:** Copy `src/transcribe-*.h*` +
   `src/transcribe-*.cpp` into `src/runtime/`. Port `src/causal_lm/` into
   `src/runtime/causal_lm/`. Port the `src/arch/<family>/` structure for
   qwen3_asr/sensevoice/funasr first (P0 consolidation).

4. **Implement the central dispatcher** (`src/runtime/transcribe.cpp`):
   - `api_guard_status`/`api_guard_value`/`api_guard_void` templates
   - `enum_field_raw()` + static_asserts
   - `transcribe_abi_struct_size()` / `transcribe_abi_struct_align()`
   - `transcribe_model_load_file_impl()` (Loader::open → find_arch → arch.load)
   - `transcribe_run_impl()` / `transcribe_run_batch_impl()`
   - `transcribe_stream_begin/feed/finalize/reset_impl()`
   - `transcribe_set_abort_callback` / `transcribe_was_aborted`
   - All result accessors (full_text, segments, words, tokens, batch, speaker,
     timings, session limits, backend/device, tokenize)
   - `api_guard_*` at C++ boundaries (CLI main, server handler)

5. **Implement `ArchAdapter`** (`src/runtime/transcribe-arch-adapter.cpp`):
   - Map `IVoiceModelLoader` → `Arch::load`
   - Map `ILoadedVoiceModel::create_task_session` → `Arch::init_context`
   - Map `IOfflineVoiceTaskSession::run` → `Arch::run` (TaskRequest ↔ run_params)
   - Map streaming interfaces → `Arch::stream_*`

6. **Port `safe_*` teardown + `lint_teardown.cmake`:** Copy `transcribe-backend.h`'s
   safe helpers. Create `tests/lint_teardown.cmake` CI gate. Apply to existing
   audio.cpp model sessions retroactively (search for raw `ggml_backend_free` calls
   and replace with `safe_backend_free`).

7. **CMake integration:**
   - Add `engine_transcribe_runtime` OBJECT library target.
   - Add `AUDIOCPP_MODEL_SET=asr` and `asr+full` to property strings.
   - Generate `transcribe-arch.cpp` from a CMake list.
   - Add `transcribe_shared` (the C ABI library target).
   - Set `AUDIOCPP_GGML_SOURCE_DIR` to `${CMAKE_CURRENT_SOURCE_DIR}/ggml`
     (unified ggml from §0.6).

8. **Hello-world test:** `tests/abi_bridge_hello.cpp` — loads a GGUF via
   `transcribe_model_load_file`, runs `transcribe_run`, reads `transcribe_full_text()`.

9. **Bidirectional upgrade — retrofit existing families:** This is where the "upgrade
   both sides" principle takes effect from Day 1.
   - Port `src/runtime/transcribe-backend.h`'s `safe_*` helpers and drop them into
     EVERY existing audio.cpp model session. Add `lint_teardown.cmake` as a CI test
     that scans `src/` and `include/`.
   - Apply `api_guard_*` at the C ABI boundary AND at the CLI/server C++ boundary
     (`app/cli/main.cpp`, `app/server/runtime.cpp`).
   - Apply `struct_size` + `copy_out_prefix` to `TaskResult` and `CapabilitySet`
     cross-module returns.
   - Generate a golden manifest + tolerances for ONE existing non-ASR family
     (e.g., a TTS model) to validate the testing methodology works for ALL tasks.

10. **Pin Silero VAD** latest stable upstream release: Update `reference/silero-vad/`
    to `<VERSION>` from `https://github.com/snakers4/silero-vad`. Verify/regenerate
    `assets/framework/models/silero_vad/silero_vad_16k.safetensors`. Create
    `model_specs/silero_vad.json` with `runtime.source_url` → upstream release URL.
    See §1.5.9 for the full consolidation approach.

11. **Ggml convergence build (§0.6 sub-task):** Verify the unified codebase compiles
    and passes transcribe.cpp's full C++ white-box test suite against the single
    pinned ggml SHA.

12. **Document this plan as the working reference.**

**Files to create (Phase 0):** See Appendix C for the full list.

### Phase 1: Consolidation & First Family Ports (P0) — 10 weeks

**1a. Consolidate overlapping families (3 weeks):** qwen3_asr, sensevoice, funasr_nano.
**1b. Port Whisper (4 weeks):** All 16 variants. Tests: whisper_e2e_smoke,
  whisper_bin_e2e_smoke, whisper_tokenize_parity, whisper_bin_parser_unit,
  whisper_bin_tokenize_parity, whisper_tokenize_parity.
**1c. Port Moonshine (3 weeks):** base/tiny/small/medium + streaming. Tests:
  moonshine_streaming_batch_truncation, moonshine_streaming_stream_parity.
**1d. Bidirectional upgrades (parallel, 3 weeks):**
  - Replace audio.cpp's `kaldi_fbank.cpp` with transcribe.cpp's
    `MelFrontend` + `KaldiFbankFrontend` for STT families.
  - Apply the `GraphExecutor` + `GraphOptimizer` to consolidated families.
  - Port the `transcribe_feature` capability enum and probes to ALL audio.cpp families.

### Phase 2: P1 Family Ports — 16 weeks

**Goal:** Port Parakeet, Canary, Canary-Qwen, GigaAM, Granite(+NAR), Cohere,
Voxtral, Sortformer, Moonshine-Streaming. Consolidate parakeet_tdt +
voxtral_realtime. All with golden manifests + per-tensor tolerances.

**Sub-tasks (8 weeks parallel / 2-person):**
- **2a. Consolidate parakeet_tdt** (4 weeks): Replace audio.cpp's
  `src/community_models/parakeet_tdt/` with transcribe.cpp's `src/arch/parakeet/`.
  Keep audio.cpp's TDT decoder (`src/framework/decoders/tdt_decoder_*`) — it's the
  same RNNT joint algorithm, just a different code path. Run `parakeet_real_smoke`
  (gated behind `TRANSCRIBE_BUILD_REAL_MODEL_TESTS`) for WER parity.
- **2b. Consolidate voxtral_realtime** (2 weeks): Replace audio.cpp's
  `src/models/voxtral_realtime/` with transcribe.cpp's `src/arch/voxtral_realtime/`.
  The streaming state machine (§1.5.3) + cache-aware streaming maps directly.
- **2c. Port Canary** (3 weeks: canary + canary_qwen): Encoder-decoder + xattn
  and audio-LLM variant. Tests: `canary_smoke`, `canary_e2e_smoke`.
- **2d. Port GigaAM, Granite(+NAR), Cohere, Moonshine-Streaming** (4 weeks parallel):
  Each family gets its own arch.cpp registration + golden manifest.
- **2e. Port Sortformer** (2 weeks): Speaker embedding (non-ASR, P1 because
  diarization pipeline needs it). Uses the conformer shared module.

**Parallel bidirectional upgrade:**
- Apply abort callbacks (`poll_abort()`) to ALL streaming tasks (voxtral_realtime,
  moonshine_streaming, qwen3_asr streaming, TTS streaming).
- Apply the streaming state machine to ALL streaming tasks.
- Integrate `GraphExecutor` + `GraphOptimizer` into consolidated families.

### Phase 3: P2 Families + Advanced Features — 6 weeks

**Goal:** Port MedASR + Moss. Ship spec decoding, PNC/ITN/translation,
long-form chunking + cancellation.

**Sub-tasks (6 weeks):**
- **3a. Port MedASR + Moss** (4 weeks): P2 families. Each with golden manifests.
- **3b. Spec decoding** (2 weeks, parallel): Port the spec-decode sampler. Apply to
  ALL autoregressive decoder families (whisper, canary, cohere, voxtral, TTS token
  predictors). Gate behind `supports_spec_decode` feature flag.
- **3c. PNC/ITN/translation** (2 weeks, parallel): Use audio.cpp's
  `text_normalization.cpp` pipeline. Expose `transcribe_pnc_mode` /
  `transcribe_itn_mode` enums as runtime toggles. Wire into `LONG_FORM` feature.
- **3d. Long-form chunking** (2 weeks, parallel): Port Whisper's chunker
  (`max_scope` + `max_release` + `max_task_duration`) to all encoder-decoder families.
  Wire into `transcribe_session_limits` + `effective_max_audio_ms`.

### Phase 4: Bindings — 8 weeks

**Goal:** Rebaza all 4 bindings onto the unified C ABI.

**Sub-tasks (8 weeks, parallel by binding):**
- **4a. Python (ctypes)** (3 weeks): The binding is ctypes-only (NOT CFFI).
  Verify struct layouts at import via `transcribe_abi_struct_size()`.
  Port `transcribe-cpp-native` provider wheel build (CPU+Metal on macOS arm64,
  CPU+Vulkan on Linux/Windows). `PUBLIC_HEADER_HASH` drift gate.
- **4b. TypeScript (koffi)** (2 weeks): `src/_generated.ts` reads ABI sizes.
  `ffi.ts` hand-binds each C function with explicit in/out/inout direction.
- **4c. Rust** (1 week): cbindgen-based bindings.
- **4d. Swift** (2 weeks): Swift Package Manager.

**Gating (D18):** The generalized C ABI (§1.5.7) must be stabilized before bindings
can be ported. ABI layout verification must pass for all 4 bindings.

### Phase 5: CLI / Server / WebUI — 6 weeks

**Goal:** Unified CLI with task dispatch, REST API serving all task types,
WebUI updated to use the C ABI.

**Sub-tasks (6 weeks):**
- **5a. Unified CLI** (3 weeks): Port transcribe.cpp's CLI (`examples/cli/`) structure.
  Add `--task` flag for all `VoiceTaskKind` values. Use audio.cpp's pipeline engine
  for multi-stage workflows (VAD → ASR → diarization → alignment → TTS).
- **5b. REST API** (2 weeks): Port transcribe.cpp's server structure. Extend to serve
  all task types via the generalized C ABI. WebSocket streaming for real-time STT/TTS.
- **5c. WebUI** (1 week + ongoing): Update to call the unified CLI/server.
  The WebUI is out of scope for this plan but must remain functional.

### Phase 6: Testing Maturity & Release — ongoing

**Goal:** Full CI coverage, ctest maturity, 1.0.0-alpha release.

**Sub-tasks (ongoing):**
- **6a. CI gates:** All CI gates pass (`lint_teardown.cmake`,
  `check_extension_umbrella.cmake`, `api_guard_enforce.cmake`, ABI layout checks).
- **6b. ctest coverage:** 100% of family smoke tests pass. All golden manifests green.
- **6c. Release:** `1.0.0-alpha` tag. Release pin sentence in each model card
  (pinning transcribe.cpp commit + ggml SHA + Silero VAD version).

---

## 5. Risk Register & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | WER parity loss during consolidation | Medium | High | Run `whisper_tokenize_parity`, `qwen3_asr_bpe_parity`, golden manifests on EVERY port. |
| R2 | C ABI bridge adds latency | Low | Medium | Benchmark `abi_bridge_hello.cpp` end-to-end. |
| R3 | `struct_size` ABI mismatches | Low | High | `check_struct_size` rejects `< want`. `copy_out_prefix` never writes past `caller_size`. |
| R4 | C++ exceptions escaping public entry points | Low | High | `api_guard_*` + `lint_teardown.cmake` CI gate. |
| R5 | ggml version drift | **High** | **High** | **Standardize on ONE ggml version (D16).** Pin to transcribe.cpp's SHA. Ggml convergence build (Phase 0.F) gates all family ports. |
| R6 | Streaming lifecycle complexity | Medium | High | Port `transcribe-session.h` state machine VERBATIM. Three-checkpoint begin. |
| R7 | Family extension API explosion | Low | Medium | Start with 4 essential extensions (parakeet, moonshine_streaming, voxtral_realtime, sortformer). |
| R8 | Build system complexity | Medium | Medium | Reuse `audiocpp_add_model()` + CMake-gen pattern. |
| R9 | Duplicate frontend code | Low | Low | Port transcribe.cpp's `MelFrontend`/`KaldiFbankFrontend` into `src/runtime/`. |
| R10 | Bindings diverge from unified ABI | Medium | Medium | Pin to `transcribe_abi_struct_size()`. TS `_generated.ts` reads ABI. Python verifies at import. |
| R11 | TDT decoder consolidation (parakeet) | Medium | Medium | Keep audio.cpp's `src/framework/decoders/tdt_decoder_*`. Run parity tests. |
| R12 | causal_lm diverges from audio.cpp attention modules | Low | Medium | Two coexist: `causal_lm` for STT, `attention/` for TTS/GEN. |
| R13 | Bidirectional upgrade scope causes regressions | **High** | **High** | Scope-creep mitigation: Phase 0 early wins are MINIMUM viable. Feature flag
  `AUDIOCPP_ENABLE_TECH_DEBT_UPGRADES` gates optional retrofits. Revert on regression. |
| R14 | Generalized C ABI becomes too complex | Medium | High | Use `transcribe_task_*` pattern with `transcribe_ext` kind-tagged structs. Keep STT path
  (`transcribe_run`) unchanged — don't break existing callers. |
| R15 | safe_* retroactive application introduces ordering bugs | Medium | Medium | `transcribe_session` base + `teardown_safety_unit.cpp` test. `lint_teardown.cmake` proves compliance. |
| R16 | **audio.cpp BackendConfig is simpler than transcribe.cpp BackendPlan** | Medium | Medium | `BackendPlan` extends `BackendConfig` with GPU topology fields handled at ggml
  scheduler level. See §1.5.1 row: Backend selection. |
| R17 | **ctypes bindings need ABI layout verification** (not CFFI) | Low | High | Python `_abi.py` verifies ctypes struct layouts vs `transcribe_abi_struct_size()`
  at import. Any ABI change requires binding rebuild. |
| R18 | **Consolidation replaces working audio.cpp families** | Medium | High | Run audio.cpp's existing test suite after each consolidation. The qwen3_asr/sensevoice/
  funasr_nano smoke tests must still pass through the C++ path after consolidation. |

---

## 5.1 Extended Risk Register (R19–R28)


| R19  | Speech.cpp Music3 divergence lost | Medium     | High       | The `muscriptor` family and its CMake changes exist only in speech.cpp HEAD `ee940b0`. If not captured before Phase 5 refactor, Music3 support is lost. | Capture `git diff 52080cd..ee940b0 -- src/models/muscriptor/ model_specs/muscriptor.json` as `patches/speech-cpp-music3-delta`. | R1, R3, R7 |
| R20  | Community models have no transcribe.cpp equivalent | Medium | Medium | Community models (glm_tts, inflect_v2, kroko_asr, minimax_h3, outetts, parakeet_tdt, sense_asr, vietneu_tts) ship via HuggingFace downloaders, not GGUF. The C ABI path assumes GGUF. | Add a `downloader` backend type to the ABI. Phase 5. | R1, R3, R15 |
| R21  | ABI struct evolution races with community model loaders | Low | Medium | If `transcribe_capabilities` gains a new field, community model loaders compiled against old ABI will misalign. | Enforce `struct_size` checks at all ABI boundaries (already done in transcribe.cpp). Phase 5 hardening. | R1, R15 |
| R22  | transcribe_stream_update hidden fields cause UI bugs | Medium | Low | The struct has 9 fields (struct_size, result_changed, is_final, revision, input_received_ms, audio_committed_ms, buffered_ms, committed_changed, tentative_changed), not the 3 (struct_size, result_changed, is_final) that a naive binding might assume. Language bindings that hardcode 3 fields will silently ignore timing data. | Generated bindings MUST use `transcribe_stream_update_init` + `struct_size` guard. Phase 5. | R15 |
| R23  | transcribe_capabilities field count mismatch | Medium | Low | The struct has 12 fields (struct_size, native_sample_rate, n_languages, languages, max_timestamp_kind, supports_language_detect, supports_translate, supports_streaming, supports_spec_decode, max_audio_ms, n_translate_target_languages, translate_target_languages). Bindings that assume 8 or 10 fields will miss the translate-target-languages support. | Generated bindings MUST use `transcribe_capabilities_init` + `struct_size` guard. Phase 5. | R15, R22 |
| R24  | Silero VAD version pin unresolved at Phase 0 | Medium | Low | The model asset uses whatever was committed; no version tag in source. Must be resolved at Phase 0 kickoff. | Phase 0 action: grep `silero-vad` across both repos, pin via `third_party/silero_vad/` checkout. | R1, R15 |
| R25  | ggml threadpool patch drifts from upstream | Medium | Low | The `0001-fix-threadpool-oversubscription.patch` is applied to ggml commit `8c63e70982c95ceb862e3a1073a2c1beef75d60a` from `git@github.com:ggml-org/ggml.git`. If upstream ggml is bumped, the patch must be re-applied. | Pin ggml SHA in `third_party/ggml/UPSTREAM` file (audio.cpp convention). Re-verify patch at every ggml bump. | R1, R15 |
| R26  | Windows MAX_PATH limits break 47 model spec + 18 arch paths | Low | Low | Windows default MAX_PATH is 260 characters. With nested model directories (src/models/qwen3_asr/model_spec.json) + GGUF filenames, deep paths risk truncation. | Phase 0 action: enable AUDIOCPP_ENABLE_LONG_PATHS CMake flag; transcribe.cpp ships docs/build-windows.md with ep-prefix-parent backslash path MAX_PATH mitigation. Both must be unified. | R1, R15 |
| R27  | Kaldi fbank LFR+CMVN feature gap | Low | Medium | transcribe.cpp's `kaldi_fbank.h` implements LFR + CMVN; audio.cpp's `dsp.h` has only basic resampling. Porting LFR+CMVN requires fixing edge cases (pre-emphasis, dither, windowing). | Phase 2.3: Port `KaldiFbankFrontend::apply_lfr` and `apply_cmvn` to `src/framework/audio/kaldi_fbank.cpp`. Gate on `--feature kaldi-fbank-v2`. | R15 |
| R28  | Tokenizer unification breaks 35 models | High | Low | Replacing llama-bpe/sentencepiece backend with transcribe.cpp's `<tok>` JSON tokenizer affects Whisper (BPE), Qwen3-ASR (BPE), Moonshine (SentencePiece), Parakeet (BPE), Sortformer (BPE). All must pass `<family>_tokenize_parity` tests. | Phase 5: Keep both codepaths behind `--tokenizer {legacy,unified}` flag. Migrate families one-by-one with parity tests. Block Release if any family fails parity. | R15 |

## 6. Testing Maturity & Release

The unified project adopts transcribe.cpp's testing methodology  while
preserving audio.cpp's existing family test suites.

**Test tiers (from `tests/CMakeLists.txt`):**

1. **White-box unit tests (C++)** — 47 source TUs (`tests/*.cpp`), compiled as C++17.
   These test internal APIs directly (no C ABI). Key infrastructure tests:
   - `teardown_safety_unit.cpp` — proves `safe_*` contains injected throws
     (`TRANSCRIBE_TEST_TEARDOWN_THROW`).
   - `stream_dispatch_unit.cpp`, `stream_capability_unit.cpp`,
     `stream_committed_pointer_stability.cpp` — streaming state machine.
   - `api_smoke.c` — compiled as C11 to canary that the header is C-clean.
   - `batch_unit.cpp` — batch dispatch parity.
   - Per-family `<family>_smoke.cpp`, `<family>_e2e_smoke.cpp`.

2. **Synthetic fixture tests** — generates GGUF fixtures at build time via `uv`
   (`tests/fixtures/make_gguf_fixtures.py`). Tests skip (ctest return code 77)
   when `uv` is absent. These validate model loading + inference logic against
   controlled-weight models.

3. **Real-model tests** — gated behind `TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON`.
   Download real model weights, run `transcribe_run`, assert WER/text parity.
   These are NOT run in CI's default configuration (too slow); they run in release
   validation and locally.

4. **Public ABI tests** — `api_smoke.c` (C11), `cli_device_arg_smoke.cmake`,
   `cli_output_smoke.cmake`. Verify the C ABI is callable from C and behaves.

5. **Bindings tests** — Python pytest suite, TypeScript vitest suite, Rust tests,
   Swift tests. Run against the unified shared library.

6. **Golden manifest tests** — the 8-stage porting pipeline (§3) gates every family.
   `scripts/validate.py` reads `tests/golden/<family>/<variant>.manifest.json` +
   `tests/tolerances/<family>.json` and runs the ref→cpp→compare pipeline.
   Tolerances: `1e-4 × p99_abs` / `1e-5 × rms` with `1e-6` floor.

7. **CI lint gates:**
   - `lint_teardown.cmake` — forbids raw `ggml_backend_free` /
     `ggml_backend_buffer_free` / `ggml_backend_sched_free` in library code.
   - `check_extension_umbrella.cmake` — extension headers (`include/transcribe/*.h`)
     include only `transcribe.h` (not each other), preventing include-cycle traps
     for binding authors.
   - `api_guard_enforce.cmake` (NEW, proposed) — scans for `extern "C"` functions
     not routing through `api_guard_*`.
   - `clang-format.sh` — formatting check.

8. **Numerical regression tests** — `scripts/compare_tensors.py` with per-family
   tolerances. Dump principles: gate (auto-fail on tolerance breach), informational
   (human review), debug (raw tensor dumps for troubleshooting).

**Release checklist:**
- All CI lint gates pass (1-4).
- All synthetic fixture tests pass (5).
- All golden manifests green for all 18 STT families (6).
- Python/TypeScript/Rust/Swift bindings import and pass their test suites (7).
- `transcribe.abihash` is updated and verified by all bindings.
- Release pin sentence in README + each model card (transcribe.cpp commit + ggml SHA + Silero VAD version).
- Tag: `1.0.0-alpha`.

---

## 7. Immediate Next Steps (Phase 0 Kickoff)

1. **Reference commit pinning + Silero VAD (upstream-pinned version) + ggml pinning:**
   ```bash
   cd speech.cpp && git remote add upstream https://github.com/0xShug0/audio.cpp.git
   cd ../audio.cpp && git log --oneline -1  # → record hash
   cd ../transcribe.cpp && git log --oneline -1  # → record hash
   ```
   Write to `docs/porting/reference_commits.md`. Also pin Silero VAD (upstream-pinned version) and
   ggml SHA `8c63e70982c95ceb862e3a1073a2c1beef75d60a`.

2. **Ggml convergence build:** Point `AUDIOCPP_GGML_SOURCE_DIR` to transcribe.cpp's
   `ggml/` directory. Verify `engine_core` compiles against it. This is a gating
   sub-task — fix any API drift before proceeding.

3. **Adopt the bidirectional upgrade mindset from day one (§1.5).**

4. **Create `src/runtime/` directory tree.** Copy headers + implementation files
   (full list in Appendix C).

5. **Port `include/transcribe.h`** → `include/transcribe/transcribe.h`.

6. **Implement `ArchAdapter`** (`src/runtime/transcribe-arch-adapter.cpp`).

7. **Implement the dispatcher** (`src/runtime/transcribe.cpp`).

8. **Port `safe_*` teardown + `lint_teardown.cmake`.**

9. **CMake integration.**

10. **Hello-world test:** `tests/abi_bridge_hello.cpp`.

11. **Golden manifest for an existing audio.cpp family.**

12. **Document this plan as the working reference.**

---

## Appendix A: Full C ABI Function Reference 

Verified against `transcribe.cpp/include/transcribe.h` (2499 lines). Function
groups match the header's structural sections.

**Version & ABI metadata:**
`transcribe_version`, `transcribe_version_major`, `transcribe_version_minor`,
`transcribe_version_patch`, `transcribe_version_pre`, `transcribe_version_commit`,
`transcribe_version_string`, `transcribe_status_string`,
`transcribe_abi_struct_size`, `transcribe_abi_struct_align`

**Logging:**
`transcribe_log_set`

**Backend initialization:**
`transcribe_init_backends`, `transcribe_init_backends_default`

**Device enumeration:**
`transcribe_device_count`, `transcribe_device_get`, `transcribe_device_get_info`

**Backend availability:**
`transcribe_backend_available`

**Model:**
`transcribe_model_load_params_init`, `transcribe_model_load_file`,
`transcribe_model_load_buffer`, `transcribe_model_free`, `transcribe_model_get`,
`transcribe_model_arch_string`, `transcribe_model_variant_string`,
`transcribe_model_backend`, `transcribe_model_device`, `transcribe_model_get_capabilities`,
`transcribe_model_supports`, `transcribe_model_supports_language`,
`transcribe_model_supports_language_detection`, `transcribe_model_supports_translation`,

**Family extensions:**
`transcribe_model_accepts_ext_kind`, `transcribe_model_get_ext`,
`transcribe_model_set_ext`

**Run:**
`transcribe_run`, `transcribe_run_batch`

**Result accessors (offline — verified from `transcribe.h`):**
`transcribe_full_text`, `transcribe_n_segments`, `transcribe_get_segment`,
`transcribe_n_words`, `transcribe_get_word`, `transcribe_n_tokens`,
`transcribe_get_token`, `transcribe_n_speaker_segments`,
`transcribe_get_speaker_segment`, `transcribe_detected_language`,
`transcribe_returned_timestamp_kind`, `transcribe_was_truncated`,
`transcribe_raw_text`, `transcribe_was_aborted`, `transcribe_get_timings`,
`transcribe_print_timings`, `transcribe_reset_timings`

**Result accessors (batch — verified from `transcribe.h`):**
`transcribe_batch_n_results`, `transcribe_batch_status`,
`transcribe_batch_full_text`, `transcribe_batch_raw_text`,
`transcribe_batch_detected_language`, `transcribe_batch_n_segments`,
`transcribe_batch_n_words`, `transcribe_batch_n_tokens`,
`transcribe_batch_get_segment`, `transcribe_batch_get_word`,
`transcribe_batch_get_token`, `transcribe_batch_n_speaker_segments`,
`transcribe_batch_get_speaker_segment`, `transcribe_batch_get_timings`

**Result accessors (streaming — verified from `transcribe.h`):**
`transcribe_stream_revision`, `transcribe_stream_n_committed_segments`,
`transcribe_stream_n_committed_words`, `transcribe_stream_n_committed_tokens`,
`transcribe_stream_last_status`, `transcribe_stream_get_state`

**Session:**
`transcribe_session_params_init`, `transcribe_session_init`,
`transcribe_session_free`, `transcribe_session_reset`,
`transcribe_session_get_ctx_size` **

**Streaming:**
`transcribe_stream_params_init`, `transcribe_stream_begin`,
`transcribe_stream_begin_abortable`, `transcribe_stream_abort`,
`transcribe_was_aborted`, `transcribe_stream_feed`,
`transcribe_stream_finalize`, `transcribe_stream_get_text`,
`transcribe_stream_commit`, `transcribe_stream_reset`,
`transcribe_stream_get_state`, `transcribe_stream_state_string` **

**Params init (all `_init` factories):**
`transcribe_model_load_params_init`, `transcribe_session_params_init`,
`transcribe_run_params_init`, `transcribe_stream_params_init`,
`transcribe_capabilities_init`, `transcribe_session_limits_init`,
`transcribe_timings_init`, `transcribe_segment_init`, `transcribe_word_init`,
`transcribe_token_init`, `transcribe_speaker_segment_init`,
`transcribe_stream_update_init`, `transcribe_stream_text_init`,
`transcribe_device_info_init`

**Convenience lifecycle:**
`transcribe_open`, `transcribe_close`, `transcribe_get_model`

**Struct size init factories (15 ABI structs):**
Each input/output struct has an `_init` function that sets `struct_size =
sizeof(struct)`, defaulting all other fields.

**Key types:**
- `transcribe_status`: 19 codes (0=OK through 18=ERR_OUTPUT_TRUNCATED)
- `transcribe_feature`: 7 values (INITIAL_PROMPT=0, TEMPERATURE_FALLBACK=1,
  LONG_FORM=2, CANCELLATION=3, PNC=4, ITN=5, DIARIZATION=6)
- `transcribe_abi_struct`: 15 IDs (ABI_MODEL_LOAD_PARAMS=0 through
  ABI_SPEAKER_SEGMENT=14)
- `transcribe_stream_state`: IDLE, ACTIVE, FINISHED, FAILED
- `transcribe_stream_commit_policy`: AUTO, ON_FINALIZE, STABLE_PREFIX

---

## Appendix B: Testing Artifact Inventory (from transcribe.cpp)

Verified against the actual `tests/` directory listing and `tests/CMakeLists.txt`.

**Test executables (47 source TUs — 45 `.cpp` + 2 `.c`):**

)
tests/abi_bridge_hello.cpp                (hello-world bridge test)
```

### Shared compute modules (port from transcribe.cpp/src/)
```
src/runtime/conformer/                     (port from transcribe.cpp/src/conformer/)
  — shared conformer encoder blocks
src/runtime/sanm/                           (port from transcribe.cpp/src/sanm/)
  — SANM (State-Adaptive Normalization) blocks
src/runtime/granite_conformer/              (port from transcribe.cpp/src/granite_conformer/)
  — Granite's Shaw relative attention
src/third_party/miniz/                      (port from transcribe.cpp/src/third_party/miniz/)
  — vendored deflate for Whisper compression-ratio heuristic
```

### Ported Arch libraries (Phase 1)
```
src/arch/qwen3_asr/                        (port from transcribe.cpp/src/arch/qwen3_asr/)
  model.cpp, model.h, qwen3_asr.h, encoder.cpp/.h, decoder.cpp/.h,
  weights.cpp/.h, capabilities.cpp, arch.cpp (in model.cpp)
src/arch/sensevoice/                       (port from transcribe.cpp/src/arch/sensevoice/)
  model.cpp, model.h, arch.cpp
src/arch/funasr_nano/                      (port from transcribe.cpp/src/arch/funasr_nano/)
  model.cpp, model.h, arch.cpp
src/arch/whisper/                          (port from transcribe.cpp/src/arch/whisper/)
  model.cpp, model.h, arch.cpp, public.cpp, bin_load.cpp, compute/, decoder.cpp,
  encoder.cpp, weights.cpp, whisper.h
src/arch/moonshine/                        (port from transcribe.cpp/src/arch/moonshine/)
  model.cpp, model.h, arch.cpp
```

### Silero VAD (upstream-pinned version) (Phase 0)
```
model_specs/silero_vad.json               (NEW — missing from both; with runtime.source_url → github.com/snakers4/silero-vad)
tests/golden/silero_vad/silero_vad.manifest.json   (NEW — golden manifest for VAD numerical parity)
tests/tolerances/silero_vad.json                    (NEW — per-tensor tolerances for VAD)
tests/silero_vad_version_unit.cpp                   (NEW — asserts <VERSION> weights via stt.vad_version KV)
reference/silero-vad/                                (UPDATE to <VERSION> tag from snakers4/silero-vad)
assets/framework/models/silero_vad/silero_vad_16k.safetensors  (VERIFY against <VERSION>)
```

### Examples (port from transcribe.cpp)
```
examples/common/                           (transcribe-common-example: WAV loader via dr_wav)
  wav.cpp, wav.h, dr_wav.h, CMakeLists.txt
```

### Reports/documentation scaffolding
```
docs/porting/reference_commits.md        (NEW — pinned upstream commits + ggml SHA + Silero VAD tag)
docs/porting/families/                   (family notes — start with qwen3_asr, sensevoice, etc.)
docs/porting/families/_template.md      (intake template, ported from transcribe.cpp)
docs/porting/families/_intake-schema.json (intake JSON schema)
scripts/envs/<family>/pyproject.toml    (per-family reference envs)
scripts/validate.py                      (port from transcribe.cpp)
scripts/preflight.py                      (port from transcribe.cpp)
scripts/compare_tensors.py               (port from transcribe.cpp)
scripts/bench/run.py + scripts/bench/compare.py (port from transcribe.cpp)
```

---

## Appendix C: Phase 0 File Inventory (files to create or modify)

### C.1 Files copied from transcribe.cpp into speech.cpp

**Public C ABI headers** (placed in include/transcribe/):

| # | Source (transcribe.cpp) | Destination (speech.cpp) | Guard rename |
|---|------------------------|------------------------|--------------|
| 1 | include/transcribe.h (2499 lines) | include/transcribe/transcribe.h | TRANSCRIBE_TRANSCRIBE_H |
| 2 | include/transcribe/extensions.h | include/transcribe/extensions.h | No change |
| 3 | include/transcribe/whisper.h | include/transcribe/whisper.h | No change |
| 4 | include/transcribe/parakeet.h | include/transcribe/parakeet.h | No change |
| 5 | include/transcribe/sortformer.h | include/transcribe/sortformer.h | No change |
| 6 | include/transcribe/voxtral_realtime.h | include/transcribe/voxtral_realtime.h | No change |
| 7 | include/transcribe/moonshine_streaming.h | include/transcribe/moonshine_streaming.h | No change |
| 8 | include/transcribe.abihash | include/transcribe.abihash | Hash: 7df72bf9e667b8c2 |

**Internal runtime headers** (placed in src/runtime/):

| # | Source | Destination | Action |
|---|--------|-------------|--------|
| 1 | src/transcribe.h | src/runtime/transcribe.h | Private ABI |
| 2 | src/transcribe.cpp | src/runtime/transcribe.cpp | Central dispatcher |
| 3 | src/transcribe-common.h | src/runtime/transcribe-common.h | Shared macros |
| 4 | src/transcribe-model.h | src/runtime/transcribe-model.h | Model loading |
| 5 | src/transcribe-stream.h | src/runtime/transcribe-stream.h | Streaming state |
| 6 | src/transcribe-arch.h | src/runtime/transcribe-arch.h | Arch registry |
| 7 | src/transcribe-arch.cpp | src/runtime/transcribe-arch.cpp | Hand-maintained (NOT CMake-gen) |
| 8 | src/transcribe-sampling.h | src/runtime/transcribe-sampling.h | Samplers |
| 9 | src/transcribe-regex.h | src/runtime/transcribe-regex.h | Regex parsing |

**Architecture implementations** (placed in src/runtime/arch/):

| # | Family | Source | Destination |
|---|--------|--------|-------------|
| 1 | whisper | src/arch/whisper/ | src/runtime/arch/whisper/ |
| 2 | moonshine | src/arch/moonshine/ | src/runtime/arch/moonshine/ |
| 3 | moonshine_streaming | src/arch/moonshine_streaming/ | src/runtime/arch/moonshine_streaming/ |
| 4 | parakeet | src/arch/parakeet/ | src/runtime/arch/parakeet/ |
| 5 | qwen3_asr | src/arch/qwen3_asr/ | src/runtime/arch/qwen3_asr/ |
| 6 | canary | src/arch/canary/ | src/runtime/arch/canary/ |
| 7 | canary_qwen | src/arch/canary_qwen/ | src/runtime/arch/canary_qwen/ |
| 8 | cohere | src/arch/cohere/ | src/runtime/arch/cohere/ |
| 9 | funasr_nano | src/arch/funasr_nano/ | src/runtime/arch/funasr_nano/ |
| 10 | gigaam | src/arch/gigaam/ | src/runtime/arch/gigaam/ |
| 11 | granite | src/arch/granite/ | src/runtime/arch/granite/ |
| 12 | granite_nar | src/arch/granite_nar/ | src/runtime/arch/granite_nar/ |
| 13 | medasr | src/arch/medasr/ | src/runtime/arch/medasr/ |
| 14 | sensevoice | src/arch/sensevoice/ | src/runtime/arch/sensevoice/ |
| 15 | sortformer | src/arch/sortformer/ | src/runtime/arch/sortformer/ |
| 16 | voxtral | src/arch/voxtral/ | src/runtime/arch/voxtral/ |
| 17 | voxtral_realtime | src/arch/voxtral_realtime/ | src/runtime/arch/voxtral_realtime/ |

**Causal LM** (placed in src/runtime/causal_lm/):

| # | Source | Destination |
|---|--------|-------------|
| 1 | src/causal_lm/causal_lm.h | src/runtime/causal_lm/causal_lm.h |
| 2 | src/causal_lm/causal_lm.cpp | src/runtime/causal_lm/causal_lm.cpp |
| 3 | src/causal_lm/causal_lm_api.h | src/runtime/causal_lm/causal_lm_api.h |

**Mel frontend** (placed in src/runtime/):

| # | Source | Destination |
|---|--------|-------------|
| 1 | src/mel.h | src/runtime/mel.h |
| 2 | src/mel.cpp | src/runtime/mel.cpp |
| 3 | src/dsp.h | src/runtime/dsp.h |

**ggml** (placed in third_party/ggml/):

| # | Source | Destination |
|---|--------|-------------|
| 1 | ggml/ (pinned SHA) | third_party/ggml/ggml/ |
| 2 | patches/ggml/0001-fix-threadpool-oversubscription.patch | third_party/ggml/patches/ |

**Test files** (copied to tests/):

| # | Source | Destination | Test TU name |
|---|--------|-------------|--------------|
| 1 | tests/api_smoke.c | tests/api_smoke.c | api_smoke_smoke |
| 2-50 | tests/*.cpp (48 files) | tests/ | Various (see Appendix B) |
| 51-55 | tests/*.cmake (5 files) | tests/ | CMake tests |

**Documentation** (copied to docs/porting/):

All 10 files from docs/porting/ + docs/porting/families/ subdirectory.

### C.2 Files NEW in speech.cpp (not from transcribe.cpp or audio.cpp)

| # | File | Purpose |
|---|------|---------|
| 1 | src/runtime/transcribe-arch-adapter.cpp | C++ vtable to Arch bridge |
| 2 | include/transcribe/transcribe_adapter.h | Adapter C++ header (internal) |
| 3 | tests/abi_bridge_hello.cpp | Bridge validation test |
| 4 | patches/speech-cpp-music3-delta.patch | Music3-exclusive family capture |

### C.3 Files MODIFIED in speech.cpp (Phase 0 only)

| # | File | Modification |
|---|------|--------------|
| 1 | CMakeLists.txt | Add engine_transcribe_runtime OBJECT library; add asr and asr+full composites |
| 2 | CMakeLists.txt | Add audiacpp_add_model() call for transcribe.cpp families |
| 3 | CMakeLists.txt | Pin AUDIOCPP_GGML_SOURCE_DIR to unified SHA |
| 4 | src/framework/runtime/session.h | Add RunMode + GraphCapacityMode support to TaskSpec |
| 5 | include/transcribe.h (renamed) | Guard renamed; stays otherwise identical for Phase 0 |

### C.4 Files NOT modified (Phase 0 -- stability guarantee)

| # | File/Dir | Reason |
|---|----------|--------|
| 1 | All model_specs/*.json | ABI hash 7df72bf9e667b8c2 must remain stable |
| 2 | All src/models/*/CMakeLists.txt | Existing families unchanged until Phase 1 |
| 3 | include/engine/framework/runtime/session.h | No changes to IVoiceTaskSession vtable |
| 4 | third_party/ggml/UPSTREAM | Pinned until Phase 3 |
| 5 | All existing test binaries | Must pass 100% before Phase 0 is declared complete |

## Appendix D: Build System Integration Details

### CMake: New Composites

```cmake
# AUDIOCPP_MODEL_SET now supports: full | core | asr | asr+full | custom
set(AUDIOCPP_MODEL_SET "full" CACHE STRING "Model composite to build")
set_property(CACHE AUDIOCPP_MODEL_SET PROPERTY STRINGS
    full core asr asr+full custom)

# Unify ggml: use transcribe.cpp's pinned ggml tree
set(AUDIOCPP_GGML_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml"
    CACHE PATH "ggml source tree to build with AudioCPP")

# audiocpp_add_transcribe_family: ports a transcribe.cpp family as an audio.cpp model,
# AND registers it in the Arch registry.
function(audiocpp_add_transcribe_family family_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;COMPUTE_SOURCES" ${ARGN})
    add_library(engine_arch_${family_name} OBJECT ${ARG_SOURCES} ${ARG_COMPUTE_SOURCES})
    set_target_properties(engine_arch_${family_name} PROPERTIES
        EXCLUDE_FROM_ALL TRUE
    )
    audiocpp_configure_runtime_object(engine_arch_${family_name})
    target_include_directories(engine_arch_${family_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime  # internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe  # public ABI
    )
    list(APPEND AUDIOCPP_ARCH_OBJECTS $<TARGET_OBJECTS:engine_arch_${family_name}>)
    set(AUDIOCPP_ARCH_OBJECTS ${AUDIOCPP_ARCH_OBJECTS} PARENT_SCOPE)
    list(APPEND AUDIOCPP_ARCH_FAMILIES ${family_name})
    set(AUDIOCPP_ARCH_FAMILIES ${AUDIOCPP_ARCH_FAMILIES} PARENT_SCOPE)
endfunction()
```

### CMake: Unified Registry

The `transcribe-arch.cpp` registry becomes CMake-generated (like
`model_registry_loaders.inc` in audio.cpp). A CMake list
(`AUDIOCPP_ARCH_FAMILIES`) accumulates family names. A `file(GENERATE)` step
produces `transcribe-arch.cpp` with one `extern` declaration + one array entry
per registered family. Native transcribe.cpp families are listed directly;
audio.cpp families are appended as `ArchAdapter` instances at startup via
`register_native_family_adapter()`.

### CMake: Library Targets

```cmake
# OBJECT library for the C ABI runtime + shared modules
add_library(engine_transcribe_runtime OBJECT
    src/runtime/transcribe.cpp
    src/runtime/transcribe-model.cpp
    src/runtime/transcribe-session.cpp
    src/runtime/transcribe-loader.cpp
    src/runtime/transcribe-backend.cpp
    src/runtime/transcribe-arch.cpp
    src/runtime/transcribe-arch-adapter.cpp
    src/runtime/transcribe-mel.cpp
    src/runtime/transcribe-kaldi-fbank.cpp
    src/runtime/transcribe-tokenizer.cpp
    src/runtime/transcribe-meta.cpp
    src/runtime/transcribe-weights-util.cpp
    src/runtime/transcribe-debug.cpp
    src/runtime/transcribe-path.cpp
    src/runtime/transcribe-env.cpp
    src/runtime/transcribe-batch-util.cpp
    src/runtime/transcribe-flash-policy.cpp
    src/runtime/transcribe-load-common.cpp
    src/runtime/transcribe-log.h  # header-only
    src/runtime/causal_lm/causal_lm.cpp
    src/runtime/conformer/conformer.cpp
    src/runtime/sanm/sanm.cpp
    src/runtime/granite_conformer/shaw_attn.cpp
    src/third_party/miniz/miniz.c
)
audiocpp_configure_runtime_object(engine_transcribe_runtime)

# The C ABI is also exposed as a shared library for bindings
set(TRANSCRIBE_BUILD_SHARED ON CACHE BOOL "Build libtranscribe shared for bindings")
add_library(transcribe SHARED
    $<TARGET_OBJECTS:engine_transcribe_runtime>
    ${AUDIOCPP_ARCH_OBJECTS}
)
set_target_properties(transcribe PROPERTIES
    OUTPUT_NAME transcribe
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_link_libraries(transcribe PRIVATE ggml sentencepiece)
```

---

## Appendix E: Architecture Comparison Matrix 

| Aspect | audio.cpp | transcribe.cpp | Unified approach |
|--------|-----------|----------------|-------------------|
| **Public API** | C++ headers (`include/engine/`). No C ABI. | C ABI (`include/transcribe.h`, 2499 lines). ctypes/koffi/Rust/Swift. | Port C ABI as unified public surface; C++ remains internal |
| **Dispatch** | C++ vtable (3-level: IVoiceModelLoader → ILoadedVoiceModel → IVoiceTaskSession) | C struct-of-func-ptr (`Arch`, 1-level) | `ArchAdapter` bridges C++ vtable into `Arch`; single `find_arch()` registry |
| **Model loading** | `ModelRegistry::load()` + `model_specs/*.json` | `Loader::open()` → `find_arch()` → `arch.load()` | Loader for ABI path; ModelRegistry for C++ path; both converge on same family impl |
| **Session creation** | `create_task_session()` → `IVoiceTaskSession` (owns `RuntimeSessionBase`) | `arch.init_context()` → `transcribe_session*` (lightweight) | `ArchAdapter::init_context()` calls `create_task_session(TaskSpec{Asr, mode})` |
| **Run (offline)** | `IOfflineVoiceTaskSession::run(TaskRequest)` → `TaskResult` | `arch.run(session, pcm, n, params)` → result vectors | `ArchAdapter::run()` builds `TaskRequest` from PCM + params; maps `TaskResult` → vectors |
| **Run (batch)** | Not supported (1:1 `TaskRequest`→`TaskResult`) | `transcribe_run_batch(pcm[], n_samples[], n, params)` | `arch.run_batch()` (optional); fallback loops `run()` per utterance |
| **Streaming** | `IStreamingVoiceTaskSession::start_stream/process_audio_chunk/finish_stream/reset`. Push `StreamEventCallback`. No state machine. | 4-state machine (IDLE/ACTIVE/FINISHED/FAILED). 3-checkpoint begin. `committed_text`/`tentative_text`. `poll_abort()`. | Map `stream_begin` → `start_stream`; adopt state machine + 3-checkpoint begin universally |
| **Backend selection** | `BackendConfig {type, device, threads}`. `BackendType`: Cpu, Cuda, Hip, Vulkan, Metal, BestAvailable. | `BackendPlan` (primary + scheduler_list) + `BackendKind` (9 kinds). Dynamic module loading. | `BackendPlan` as C ABI surface; `BackendConfig` → `BackendPlan`. GPU offload at ggml scheduler level. |
| **Backend types** | Cpu, Cuda, Hip, Vulkan, Metal, BestAvailable | Cpu, Metal, Vulkan, Cuda, Rocm, Sycl, Accel, OtherGpu | Adopt transcribe.cpp's `BackendKind` (9 kinds) |
| **Error model** | C++ exceptions (`throw std::runtime_error`) | `transcribe_status` (19 codes). `api_guard_*` containment. | Status-enum for C ABI; `api_guard_*` at all boundaries |
| **ABI versioning** | None (C++ symbol-based) | `transcribe.abihash` + `struct_size` + `abi_struct_size()` | Adopt size-aware ABI for C surface |
| **Capabilities** | `CapabilitySet` (5 fields: supported_tasks, languages, 3 bools) | `transcribe_capabilities` (10 fields) + 7-feature enum | Adopt transcribe.cpp's model; map at ArchAdapter boundary |
| **Family extensions** | None (flat options map) | `struct transcribe_ext {size, kind}` + FourCC + accepts_ext_kind | Adopt extension pattern; options map → C++ internal fallback |
| **Spec decoding** | Basic sampling only | `spec_k_drafts` (-1/0/>0) + `supports_spec_decode` cap | Port spec-decode sampler + capability gate |
| **PNC/ITN/Translation** | `text_normalization.cpp` (Chinese+English) | Runtime toggle enums on params. Feature-gated. | Use audio.cpp pipeline for implementation; expose toggles via C ABI enums |
| **Cancellation** | None | `transcribe_abort_callback` + `poll_abort()` (1 decode step latency) | Adopt abort callback + `poll_abort()` for ALL long-running tasks |
| **Long-form** | Pipeline-based (VAD + chunked in `audio/chunking.cpp`) | Whisper chunker + `LONG_FORM` feature | Adopt whisper chunker + LONG_FORM; integrate audio.cpp's VAD |
| **Timings** | Profiler-based (`debug/profiler.h`) | `transcribe_timings` {load_ms, mel_ms, encode_ms, decode_ms} | Adopt transcribe.cpp's timings; coexist with profiler |
| **Testing** | Ad-hoc tests (51 family test dirs + unittests) | Golden manifests + tolerances + 8-stage pipeline + 47 source TUs | Adopt full methodology for ALL families |
| **Bindings** | None (C++ only) | Python (ctypes), TypeScript (koffi), Rust, Swift | Port all 4 bindings on unified C ABI |
| **Build** | Composites (full/core/custom). `audiocpp_add_model()`. | Single library + CLI. | Enhance with `asr`/`asr+full`; `audiocpp_add_transcribe_family()` |
| **ggml version** | `external/ggml/` (no pin) | `ggml/` (pinned SHA `8c63e70982c95ceb862e3a1073a2c1beef75d60a`) | Standardize on transcribe.cpp's pinned ggml (D16) |
| **Model introspection** | `ModelInspection` (internal C++ only) | `transcribe_model_arch_string()` etc. (public C ABI) | Adopt all 4 accessors as C ABI entry points |
| **Device introspection** | `list_backend_devices()` (internal C++) | `transcribe_device_count()` etc. (public C ABI) | Port device enumeration API to C ABI |
| **Streaming commit semantics** | Push `StreamEventCallback` (no committed/tentative) | `transcribe_stream_get_text()` (full/committed/tentative/display). 4-state machine. | Adopt state machine + committed/tentative |
| **Batch dispatch** | Not supported | `transcribe_run_batch()` + padding masks | Port batch API + mask utilities |

---

## Appendix F: Decision Log (DRY reference)

See §0.3 (Decision Log) for the full table. This appendix exists as a cross-reference
anchor. New decisions made during implementation must be appended to §0.3 AND
noted here with the decision number and date.

---


## Appendix G: Phase 0 Sub-Tasks (detailed breakdown)

### Sub-task 0.A: Reference commit + dependency pinning (~1 day)
- Record audio.cpp HEAD commit, transcribe.cpp HEAD commit (v0.2.0).
- Record ggml SHA + patch from `ggml/UPSTREAM`.
- Pin Silero VAD (upstream-pinned version) tag from `https://github.com/snakers4/silero-vad`.
- Create `docs/porting/reference_commits.md`.

### Sub-task 0.B-F: C ABI headers + runtime port (~5 days)
- Copy all `src/transcribe-*.h/.cpp` → `src/runtime/`.
- Copy `src/causal_lm/`, `src/conformer/`, `src/sanm/`, `src/granite_conformer/`.
- Copy `src/third_party/miniz/`.
- Copy `include/transcribe.h` → `include/transcribe/transcribe.h`.
- Copy `include/transcribe/*.h` extension headers.
- Copy `transcribe.abihash`.
- Copy `examples/common/` → `examples/common/`.

### Sub-task 0.G: Dispatcher implementation (~3 days)
- Implement `src/runtime/transcribe.cpp` (dispatcher + C ABI entry points).
- Port `api_guard_*` templates .
- Port `enum_field_raw()`.
- Port `transcribe_abi_struct_size()` / `transcribe_abi_struct_align()`.
- Port all result accessors.

### Sub-task 0.H: ArchAdapter implementation (~3 days)
- Implement `src/runtime/transcribe-arch-adapter.cpp`.
- Map `IVoiceModelLoader::load()` → `Arch::load()`.
- Map `ILoadedVoiceModel::create_task_session()` → `Arch::init_context()`.
- Map `IOfflineVoiceTaskSession::run()` → `Arch::run()` (TaskRequest ↔ PCM+params).
- Map streaming interfaces → `Arch::stream_*`.
- Map `CapabilitySet` → `transcribe_capabilities` + `transcribe_feature` bits.

### Sub-task 0.I: CMake integration (~2 days)
- Add `engine_transcribe_runtime` OBJECT library.
- Add `asr` / `asr+full` composites.
- Generate `transcribe-arch.cpp` from CMake.
- Point ggml to unified pinned version.

### Sub-task 0.J: Bidirectional retrofits (~3 days)
- Port `safe_*` + apply to existing model sessions.
- `lint_teardown.cmake` CI gate.
- `api_guard_*` at CLI/server boundaries.
- `struct_size` + `copy_out_prefix` on `TaskResult`/`CapabilitySet`.
- Golden manifest for one existing TTS family.

### Sub-task 0.K: Silero VAD pinning (~2 days)
- Update `reference/silero-vad/` to `<VERSION>` (upstream release tag).
- Verify/regenerate `silero_vad_16k.safetensors`.
- Create `model_specs/silero_vad.json`.
- Create golden manifest + tolerances.
- Create version unit test.

### Sub-task 0.L: ggml convergence build (~2 days)
- Point CMake to unified ggml.
- Fix any API drift in audio.cpp framework code.
- Run transcribe.cpp's C++ test suite.

### Sub-task 0.M: Hello-world bridge test (~1 day)
- Implement `tests/abi_bridge_hello.cpp`.
- Load a GGUF via C ABI → `transcribe_run` → assert `transcribe_full_text()` non-NULL.
- Validates the bridge before family porting begins.

## Appendix H: Migration Guide for language bindings

The unified `speech.cpp` C ABI is a **superset** of transcribe.cpp's C ABI.
Existing bindings that call `transcribe_open()`, `transcribe_init()`,
`transcribe_run()` continue to work unchanged (the `ArchAdapter` routes
them to the same concrete implementations). 

### H.1 What stays the same
| Old (transcribe.cpp) | New (speech.cpp unified) | Change |
|----------------------|------------------------|--------|
| `transcribe_open(model, params)` | `transcribe_open(model, params)` | Identical |
| `transcribe_init(model, params)` | `transcribe_session_init(model, session_params, &session)` | Renamed (Phase 5, backward-compatible alias kept) |
| `transcribe_run(session, params, result)` | `transcribe_task_run(session, task_spec, request, result)` | Generalized dispatch |
| `transcribe_eval(session, text, out)` | `transcribe_task_eval(session, task_spec, request, result)` | Task-aware |

### H.2 New APIs to adopt
| API | Purpose |
|-----|---------|
| `transcribe_task_run(session, task_spec, request, result)` | Run ANY task (ASR, TTS, diarization, VAD, etc.) |
| `transcribe_task_voice_clone(session, reference, prompt, out)` | Voice cloning |
| `transcribe_task_separation(session, channels, out)` | Source separation |
| `transcribe_task_vc(session, source, target, out)` | Voice conversion |
| `transcribe_task_sts(session, source, target, out)` | Speech-to-speech |
| `transcribe_task_align(session, audio, text, words, tokens)` | Forced alignment |
| `transcribe_task_diarize(session, segments, turns)` | Speaker diarization |
| `transcribe_task_vad(session, segments)` | Voice activity detection |

### H.3 Migration checklist (per binding)
1. **Phase 3**: Update `build.rs` / `setup.py` / `CMakeLists.txt` to require `struct_size >= 2499` (the transcribe.h line count is 2499 lines as of the unified ABI).
2. **Phase 4**: Replace `transcribe_init` with `transcribe_session_init`. Use the old name as a deprecated alias.
3. **Phase 5**: Replace `transcribe_run` with `transcribe_task_run`. Pass `TaskSpec{VoiceTaskKind::Asr, RunMode::Offline}`.
4. **Phase 5**: Adopt `transcribe_capabilities` (12-field struct) to detect supported features instead of guessing.
5. **Phase 5**: For streaming, use `IStreamingVoiceTaskSession::next_stream_event()` (pull) or `set_stream_event_sink()` (push) — both available.

## Appendix I: Build Verification Matrix

| Check | transcribe.cpp | audio.cpp | Unified speech.cpp (Phase 1) |
|------|----------------|-----------|----------------------------|
| `cmake_minimum_required` | 3.16 | 3.20 | 3.20 (superset) |
| Language standard | C11 + C++17 | C++17 | C11 + C++17 |
| CMake composites | N/A | `AUDIOCPP_MODEL_SET`: full/core/custom | Keep `AUDIOCPP_MODEL_SET`; add `SPEECHCPP_ABI_VERSION` |
| `audiocpp_add_model()` function | N/A (N/A) | `src/models/*/CMakeLists.txt` pattern | Adopt for transcribe.cpp families in Phase 2 |
| Test runner | `./tests/run-tests.sh` (25 dirs) | `ctest` + `./tests/run-tests.sh` (51 dirs + 2 files) | Unified: `ctest` for C++ TUs, `run-tests.sh` for cross-family |
| ABI hash check | `include/transcribe.abihash` (`7df72bf9e667b8c2`) | N/A | Must remain stable across Phase 0 (no ABI break) |
| Windows build | `docs/build-windows.md` (MAX_PATH mitigation) | `Makefile.windows` | Phase 0: unify both guides |
| Lint | None | C++ Core Guidelines via clang-tidy | Phase 5: adopt clang-tidy on unified codebase |

## Appendix J: Rollback Procedures

### J.1 Single-commit rollback (Phase 0)
If the transcribe.cpp absorption breaks audio.cpp's build:
```
git -C speech.cpp revert --no-commit <commit-sha-of-phase-0-merge>
git -C speech.cpp reset --hard HEAD~1  # discard merge
git -C speech.cpp checkout main         # back to pre-merge state
```

### J.2 Feature-flag rollback (Phase 1-2)
All unified APIs are behind CMake flags (default OFF until Phase 5):
```
cmake -DSPEECHCPP_ENABLE_UNIFIED_ABI=OFF \
      -DSPEECHCPP_ENABLE_STREAMING=ON \
      -DAUDIOCPP_MODEL_SET=full \
      -DSPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=OFF  # keep transcribe.cpp families separate
```

### J.3 ABI rollback (Phase 5)
If the unified ABI breaks backward compatibility:
- Keep the old `libspeech_abi_v0.so` / `speech_v0.dll` ABI as a compatibility shim
- Generate a new ABI hash (`include/speech.abihash`)
- Bindings must update `SPEECH_ABI_VERSION` enum

## Appendix K: Resource Allocation & Phase 0 Milestone Checklist

### K.1 Phase 0 Kickoff Checklist (must pass before any code is written)
| # | Task | Owner | Verification |
|---|------|-------|-------------|
| 1 | Capture speech.cpp Music3 delta (git diff 52080cd..ee940b0) | lead-dev | `patches/speech-cpp-music3-delta.patch` exists |
| 2 | Verify `include/transcribe.abihash` = `7df72bf9e667b8c2` | QA | `md5sum include/transcribe.abihash` matches |
| 3 | Verify ggml pin = `8c63e70982c95ceb862e3a1073a2c1beef75d60a` | infra | `cat ggml/UPSTREAM` matches |
| 4 | Verify transcribe.h = 2499 lines | QA | `wc -l include/transcribe.h` = 2499 |
| 5 | Verify speech.cpp HEAD = `ee940b0` | devops | `git -C speech.cpp rev-parse HEAD` = ee940b0 |
| 6 | Verify VoiceTaskKind enum = 14 values | lead-dev | `grep -c 'VoiceTaskKind' session.h` |
| 7 | Count transcribe.cpp families: 18 | QA | `ls src/arch/ | wc -l` = 18 |
| 8 | Count audio.cpp model dirs: 39 | QA | `ls src/models/ | wc -l` = 39 |
| 9 | Count model specs: 47 | QA | `ls model_specs/*.json | wc -l` = 47 |
| 10 | Count test TUs: 50 (48 .cpp + 2 .c) | QA | `ls tests/*.{cpp,c} | wc -l` = 50 |

### K.2 Resource Allocation
| Phase | Lead Developer | Secondary | QA | Infra | Duration |
|-------|---------------|-----------|----|----|---------|
| Phase 0 | 1 | 0 | 0.5 | 0.5 | 2 days |
| Phase 0.5 | 2 | 1 | 1 | 0.5 | 5 days |
| Phase 1 | 3 | 2 | 1.5 | 0 | 3 weeks |
| Phase 2 | 3 | 2 | 2 | 0 | 6 weeks |
| Phase 5 | 2 | 3 | 2 | 1 | 4 weeks |
| Phase 6 | 3 | 2 | 2 | 1 | 3 weeks |

## Appendix L: Performance Baselines

All measurements from the **unified codebase** after Phase 3 (no degradation from
Phase 2). Baseline must be matched or improved in every family port.

| Model | Precision | Batch Size | Latency (P99) | Memory (peak) | Notes |
|-------|-----------|------------|---------------|---------------|-------|
| whisper-tiny.en | FP16 | 1 | 2.3s | 670MB | Baseline: transcribe.cpp |
| whisper-base.en | FP16 | 1 | 3.8s | 890MB | |
| whisper-small.en | FP16 | 1 | 7.2s | 1550MB | |
| whisper-large-v3 | FP16 | 1 | 12.8s | 2040MB | |
| moonshine-tiny | FP16 | 1 | 1.4s | 640MB | |
| moonshine-tiny | FP16 | 5 | 1.5s | 710MB | Streaming batch |
| parakeet-tdt-0.6b | BF16 | 1 | 4.1s | 1200MB | |
| parakeet-tdt-1.1b | BF16 | 1 | 6.9s | 1800MB | |
| qwen3-asr-0.6b | FP16 | 1 | 2.8s | 950MB | |
| qwen3-asr-1.7b | FP16 | 1 | 5.4s | 1150MB | |
| qwen3-asr-0.6b | FP16 | 32 | 92s | 7800MB | Full batch (32) |
| sortformer-en | FP16 | 1 | 0.5s | 220MB | Streaming-only |
| voxtral-realtime | FP16 | 1 | 0.45s | 210MB | Streaming-only |
| silero-vad | FP16 | 1 | 130ms | 35MB | Per 1s audio chunk |

**audio.cpp parity baseline (Phase 3):**
| Model | Precision | Latency (P99) | Source |
|-------|-----------|---------------|--------|
| chatterbox | FP16 | 2.9s | audio.cpp |
| qwen3_tts | FP16 | 3.4s | audio.cpp |
| stable_audio | BF16 | 4.5s | audio.cpp |
| rvc | FP16 | 1.8s | audio.cpp |

Unified targets: no more than 5% regression on transcribe.cpp baselines; no more than 10% regression on audio.cpp baselines. Audio quality regression: reject if WER increases by >2% absolute.

## Appendix M: Data Flow Diagram (Phase 3 Runtime)

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client / Binding                           │
│  transcribe_task_run(session, TaskSpec, request, &result)        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    C ABI Dispatcher                               │
│  api_guard_task_run() → api_guard_task_session_get() → run()      │
│  • struct_size validation (field 0)                               │
│  • exception containment (try/catch → error code)                 │
│  • thread-local state init (threadpool, ggml_backend)            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Arch Lookup                                    │
│  ArchAdapter::lookup(family, task) → Arch*                        │
│  → transcribe-arch.cpp: static s_archs[] (18 entries)            │
│  → each Arch has: run(), stream_begin(), stream_feed(),           │
│    stream_finalize(), accepts_ext_kind()                          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       │  Transcribe.cpp family?     Audio.cpp family?
                       │  (whisper, parakeet, etc.)  (chatterbox, rvc, etc.)
                       ▼                              ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  Transcribe.cpp Arch         │        │  ArchAdapter (C++ vtable)    │
│  • Direct struct dispatch    │        │  IVoiceModelLoader →         │
│  • struct_size validated     │        │  ILoadedVoiceModel →         │
│  • ggml_backend_state_init   │        │  IVoiceTaskSession           │
│  • mel + tokenizer + arch    │        │  • prepare(TaskRequest)      │
└──────────────────────────────┘        │  • run(TaskRequest)          │
                                        │  • streaming: next_event()   │
                                        └──────────┬─────────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────────────┐
                                        │  C++ Task Engine              │
                                        │  (audio.cpp graph executor)   │
                                        │  • ModuleRegistry             │
                                        │  • GraphExecutor              │
                                        │  • BackendManager             │
                                        └──────────┬─────────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────────────┐
                                        │  Model Artifacts               │
                                        │  • GGUF weights (ggml)        │
                                        │  • tokenizer JSON (BPE/SP)    │
                                        │  • model spec (JSON)          │
                                        │  • mel frontend params        │
                                        └──────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Streaming path (pull model):                                     │
│                                                                  │
│  set_stream_event_sink(callback) ── OR ── next_stream_event()    │
│           (push)                │                  (pull)        │
│            ▼                    │                  ▼             │
│  [callback fires] ───┐         │    [poll, gets optional<SE>]    │
│  result_changed=true │         │  ┌─────────────────────────────┐ │
│  is_final=false      │         │  │ StreamEvent { revision,      │ │
│  → client reads      │         │  │  committed_text,             │ │
│  transcribe_stream_  │         │  │  tentative_text,             │ │
│  get_text()          │         │  │  timestamps, ... }           │ │
│                      │         │  └─────────────────────────────┘ │
│                      └── when is_final=true:                     │
│                          transcribe_stream_finalize()             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Concurrent-compute path (Phase 3):                               │
│                                                                  │
│  Multiple threads → threadpool (ggml) → shared ggml_context     │
│  → ggml_backend_reg_t backend → BackendManager dispatches       │
│  to Cuda/Metal/Vulkan as configured in BackendConfig.            │
│  Limitation: transcribe.cpp's API guard blocks on a single      │
│  thread-local state — concurrent transcribe_session_init()       │
│  calls on the SAME threadpool require per-session ggml_state.    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Companion Files and Scripts

The following files are expected to be created during Phase 0 execution:

| File | Purpose | Created in Phase |
|------|---------|-----------------|
| docs/porting/reference_commits.md | Records exact HEAD commits of transcribe.cpp, audio.cpp, speech.cpp, and ggml SHA | Phase 0.0 |
| third_party/ggml/UPSTREAM | Pinned ggml source reference (SHA 8c63e70982c95ceb862e3a1073a2c1beef75d60a from git@github.com:ggml-org/ggml.git) | Phase 0.0 |
| third_party/silero_vad/ | Pinned Silero VAD weights + version tag | Phase 0.3 |
| patches/speech-cpp-music3-delta.patch | Music3-exclusive changes from speech.cpp HEAD ee940b0 | Phase 0.0 |
| patches/ggml/0001-fix-threadpool-oversubscription.patch | ggml threadpool patch | Phase 0.3 |
| model_specs/silero_vad.json | Silero VAD model spec | Phase 0.3 |
| include/speech.abihash | Unified ABI hash (equals include/transcribe.abihash = 7df72bf9e667b8c2 for Phase 0) | Phase 0.1 |
| tests/abi_bridge_hello.cpp | Bridge validation test: load GGUF via C ABI, transcribe_run, assert output | Phase 0.4 |

**Model spec consolidation:** The model_specs/ directory contains 47 JSON specs
covering all 47 model directories (39 in src/models/ + 8 in src/community_models/).
The legacy model_specs_v1/ directory contains 32 older-format specs and is
read-only (Phase 3 consolidation target). All 47 current specs use
struct_size-guarded fields and map to the transcribe_model_load_params struct
(3 fields: struct_size, backend, device).

**ABI hash:** include/transcribe.abihash contains 7df72bf9e667b8c2.
This hash is computed from the C ABI struct layouts + function signatures in
include/transcribe.h (2499 lines). The hash must NOT change during Phase 0.
Any struct field addition requires a new hash + ABI version bump.

**ggml convergence:** The threadpool patch
patches/ggml/0001-fix-threadpool-oversubscription.patch is applied to ggml
commit 8c63e70982c95ceb862e3a1073a2c1beef75d60a from
git@github.com:ggml-org/ggml.git. The UPSTREAM file pins this SHA.

