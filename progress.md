# Progress — Unified_Audio.cpp (speech.cpp ggml fork) merge & improve

Status snapshot: **The ggml convergence is now certified on CPU *and* CUDA.
The GPU half of R1 step 2 ran on real hardware, which immediately surfaced two
more dropped fork deltas and one upstream CUDA limitation that had been hiding
behind "no GPU on this host". The `patches/ggml/` invariant, which had actually
been broken, is restored and re-verified.**
Date: 2026-08-20 (work session log)

## Repo layout (important, non-obvious)
`Unified_Audio.cpp/` is a **plain container directory with no git repo of its
own**. It holds exactly three things, each an independent repository:

| Folder | Role |
|---|---|
| `speech.cpp/` | the active development repo (the ggml/audio.cpp fork). **All merge work, and this log, live here.** Remote: `NairoDorian/speech.cpp`, upstream `0xShug0/audio.cpp`. |
| `audio.cpp/` | upstream reference — read from, not developed in |
| `transcribe.cpp/` | merge source — read from, not developed in |

History (2026-08-20): the container was briefly `git init`-ed as a superproject
tracking the three as gitlinks, with `progress.md` and a duplicate
`scripts/sync-ggml.sh` at its root. That was undone — the superproject `.git`,
the root script and this log's old location are gone, and nothing outside the
three repos is versioned. If you find a reference to a root-level
`scripts/sync-ggml.sh`, it is stale: the canonical one is
`speech.cpp/scripts/sync-ggml.sh`, which is the better implementation
(`--dry-run`, `--check` before applying, staged swap) and is tracked.

## Overall progress (toward "Unified_Audio transcribes on CPU")
| Area | Status | % |
|---|---|---|
| Merge: ggerganov/ggml → bundled `external/ggml` (pin 8c63e709) + fork patches | Done | 100% |
| Build host: green default-OFF **and** default-On-OpenMP (auto-detected) | Done | 100% |
| Reproducibility (`sync-ggml.sh` + patches 0001–0005 → empty diff) | **Re-verified after a real break** | 100% |
| **Phase-2: CPU compute for SAGE_ATTN2 / CONVROT_LINEAR / MUL_MAT_PACK4** | Implemented + numerically validated + patch-tracked | 100% |
| **CUDA parity run for the same three ops** | **Executed on RTX 4070 / CUDA 13.3** | **100%** |
| **Convergence build certified on a GPU backend (R1 step 2, GPU half)** | **Done** | **100%** |
| Fork-delta audit (behavioural relaxations, not just API drift) | **Done — marker-based, 3 found** | 100% |
| Lean-build health (`AUDIOCPP_MODEL_SET` != full with tests ON) | **Fixed — was broken** | 100% |
| ABI offline + streaming paths (`transcribe_open`/`run`/`full_text`, `stream_*`) | Verified, real CTest gates | 100% |
| End-to-end **ASR** transcription validation (needs a real ASR model) | Blocked | 0% |
| **Project-wide (functional CPU transcribe)** | | **~70%** |

## DONE this session

### 1. The CUDA parity run — executed, and it changed conclusions
`ggml-cuda.lib` was already built from the prior session; linking
`ggml_fork_ops_cpu_test` against it and running produced the first real
CPU↔CUDA numbers for the three fork-only ops.

Five checks failed on the first run. **All five were tolerance-model bugs, not
kernel bugs**, and each cause was verified in the CUDA sources rather than
assumed:

- `sage_attn2` and `convrot_linear` were being held to the *CPU reference*
  tolerances. The CPU kernels deliberately keep activations in F32; the CUDA
  kernels quantize them — INT8 Q·Kᵀ with FP8 PV
  (`sage-attn2/qattn/qk_int_sv_f8_cuda_sm89.cuh`) and per-token INT8 by
  `max_abs/127` (`convrot-linear.cu:112`). Judging a quantizing kernel by a
  reference kernel's max-abs bound measures the quantizer.
- `mul_mat_pack4` CPU↔CUDA differed by 6.9e-4 against a 1e-4 bound. Cause:
  ggml-cuda sets `CUBLAS_TF32_TENSOR_OP_MATH` on every cuBLAS handle
  (`common.cuh:1502`), so an F32 GEMM there runs on TF32 tensor cores — a
  10-bit mantissa.

The test now carries two tolerance classes (exact-activation vs
quantized-activation), reports max-abs **and** normalized RMS for every
comparison, and — the useful part — cross-checks plain `mul_mat` alongside
`pack4` and asserts the two cross-backend deltas are **equal**. They are, to
1e-9: both are 6.938e-04. That turns a loosened bound into a real statement,
namely that the pack4 wrapper contributes nothing of its own.

Measured CPU↔CUDA (RTX 4070, sm_89, CUDA 13.3), normalized RMS:
sage_attn2 3.7e-2 / 3.5e-2, convrot_linear 7.4e-3 / 7.8e-3, mul_mat 6.9e-4.
All consistent with the published SageAttention2 and INT8 error levels.

### 2. Two more lost fork deltas — found cheaply, by grepping for markers
The prior session found the two-sided-broadcast loss via a crashing test and
noted this class "has no new symbol to grep for". That is true of the *upstream*
tree but not of the fork: audio.cpp tags its own ggml deltas with `MINITTS_*`
markers. Grepping `audio.cpp/external/ggml` for them enumerates the class in
one command, and should be the first step on any future fork base:

| Marker | What | Status |
|---|---|---|
| `MINITTS_FLASH_BIAS_WRAPPER` | `ggml_flash_attn_ext_with_bias_mask` | already converged (patch 0002) |
| `MINITTS_CONCAT_FASTPATH` | contiguous concat fast paths on CPU | **was dropped → patch 0005** |
| `MINITTS_FLASH_PER_HEAD_MASK` | per-head flash-attention masks on CUDA | **was dropped → deliberately not restored, see §3** |

Also confirmed: audio.cpp's ggml history has only 14 commits touching
`external/ggml`, and none of them touch `binary-ops.cpp` — so the deltas of this
class live in the *initial* vendored drop, which is why a commit-by-commit read
of the fork's history misses them.

**`MINITTS_CONCAT_FASTPATH` was not cosmetic.** Upstream still walks concat
element-by-element ("TODO: smarter multi-theading"); the fork copies contiguous
planes/rows with two memcpys. The framework concatenates per-head attention
outputs in a loop, and restoring the fast path is what put
`supertonic_vector_convnext_exp_test` — which asserts a ≥5% speedup — back in
the green.

### 3. A dropped delta that must NOT be restored — and the right fix instead
`MINITTS_FLASH_PER_HEAD_MASK` looked like a straightforward port: upstream
0.20.2 refuses any mask with `ne[2] != 1`
(`ggml_cuda_get_best_fattn_kernel` → `BEST_FATTN_KERNEL_NONE`), which made
`encoder_module_test` abort at `fattn.cu` on the CUDA build, while all three
CUDA kernels already index the mask per head as `nb32*(head % ne32)`.

Porting it removed the abort — and produced **wrong numbers**. A new test case
pins it: with a per-head bias mask, the CUDA output matches a reference forced
to use *bias slice 0 for every head* to within 4.9e-4 (exactly the F16 floor
measured on the shared-mask control) while diverging from the true per-head
reference by 9.1e-2. It reproduces with `gqa_ratio == 1`, so head-grouping is
not the cause: **ggml 0.20.2's CUDA flash-attention simply does not implement
per-head masks**, and upstream's blanket refusal is load-bearing.

So the ggml patch was reverted and the decision moved one level up:
`use_specialized_flash_attention()` in
`src/framework/modules/attention/common_relative_attention.cpp` now keeps that
lowering on CPU and falls through to the reference lowering elsewhere — the
same shape as the existing `ggml_backend_supports_op` probe in
`minimax_h3/dit_denoiser.cpp`.

This is an accuracy **win**, not a workaround. Against the F32 reference, the
CUDA relative-attention output improved from 3.2e-3 max / 8.0e-4 mean to
**8.1e-6 / 1.3e-6** — roughly 390x — because the fallback also avoids the CUDA
flash kernels' conversion of F32 K/V to F16. The test's CUDA bounds were
tightened to ~10x the measured values so that silently re-enabling the flash
path fails loudly.

**`encoder_module_test` had a second bug of its own:** it never set
`ctx.backend_type`, which production populates from
`execution_context.backend_type()`. Its "cuda" case was therefore building
CPU-configured graphs and merely running them on a CUDA device — so every
`ctx.backend_type == Cuda` branch in the framework was untested by it. Fixed in
`set_runner_backend`, and the two hand-rolled runner setups now go through it.

### 4. The `patches/ggml/` invariant had actually been broken
`external/ggml` is generated and the UPSTREAM manifest warns that anything added
by hand is deleted on the next `sync-ggml.sh` run. Commit `e11e3c5` (the
two-sided broadcast restore: `binary-ops.cpp`, `ops.cpp`, `binbcast.cu`,
`scale.cu`) was applied **in place and never tracked as a patch** — the next
sync would have silently deleted it. That is exactly the failure mode the
manifest describes, and it had already happened.

Now tracked as `patches/ggml/0004-restore-two-sided-broadcast.patch`, with the
concat fast path as `0005-cpu-concat-fastpath.patch`. Both were generated by
regenerating a pristine tree and diffing against staged intermediates, so the
split is exact rather than hand-authored. Re-verified end to end:
`sync-ggml.sh` + patches 0001–0005 reproduces the tree with an **empty diff**.

Two further traps in the same area, both found rather than assumed:

- **`.gitignore` had `/patches/` ignored wholesale.** Patches 0001–0003 are in
  the tree only because someone force-added them, so `git add` on a new patch was
  *silently a no-op* — the same failure mode one level up. Narrowed to
  `/patches/*` + `!/patches/ggml/`; the directory form cannot be negated, because
  git does not descend into an excluded directory.
- **`scripts/sync-ggml.sh` destroyed the UPSTREAM notes.** Only the header
  (repo/sha/patch list) is generated; everything below it is hand-written — the
  audit recipe, the per-op status, the deliberately-not-restored decision. The
  script regenerated the whole file, so one sync deleted 91 lines of exactly the
  knowledge a sync exists to carry forward. It now carries those notes over.
  Found by running the script rather than reading it, and only because the root
  container's *duplicate* copy (which did preserve them) was about to be deleted
  — the two copies had silently diverged, and the surviving one was the lossy one.

### 5. Lean builds were broken with `ENGINE_BUILD_TESTS=ON`
`AUDIOCPP_MODEL_SET=core` links **no** models at all, but 11 test targets
reference model-internal symbols unconditionally, so any lean build failed at
link. This matters because lean configs are the fast-iteration path.

Gated each on its owning model, following the `vibevoice` precedent already in
the file: `moss` (4 targets), `dots_tts`, `inflect_v2`, `voxtral_realtime`,
`supertonic`, `outetts`, `qwen3_forced_aligner`, `citrinet_asr` + `hviske_asr`,
`parakeet_tdt`. The lean CUDA build now links clean with zero failures.

### 6. `ggml_flash_attn_ext_with_bias_mask` gained numerical coverage
Of the six converged fork-only APIs, the wrapper had none — the manifest listed
it as "CPU-functional", which meant "it links". It now has an independent
reference (softmax over `scale*(QK + bias)`, with the bias read back through F16
so the mask's own rounding is not counted as kernel error), covering both a
per-head and a shared mask, plus the "does NOT collapse to bias slice N" guard
described in §3.

## Test suite state
Both backends sit at the documented pre-existing baseline; nothing regressed.

**CUDA build** (`AUDIOCPP_MODEL_SET=core`, `ENGINE_ENABLE_CUDA=ON`, sm_89) —
46 tests, 4 failures, all pre-existing environment/asset issues:
`flashsr_utility_test` (numeric tolerance), `model_spec_system_test`,
`fun_asr_nano_assets_test`, `server_model_installer_test`.
Newly green here: **`scaled_dot_product_attention_test`** (needs a CUDA backend
— it had never run), `encoder_module_test`, `supertonic_vector_convnext_exp_test`.

**CPU build** (`AUDIOCPP_MODEL_SET=full`, unified ABI + transcribe arches) —
56 tests, 6 failures, all pre-existing: the four above plus
`scaled_dot_product_attention_test` (needs CUDA, absent from this build) and
`asr_standalone_gguf_test` (missing assets).

Note: `supertonic_vector_convnext_exp_test` asserts a ≥5% wall-clock speedup and
does fail when another build or test suite is running on the machine. Run it
uncontended.

## NEXT (highest value first)
1. **End-to-end ASR validation** — now the top blocker with no workaround. No
   ggml/GGUF ASR model is in-tree; only the silero_vad safetensors.
   `abi_bridge_hello` / `abi_stream_hello` already accept a model path and would
   become real transcription gates (today they exercise VAD segments, not text),
   and `assets/asr_validation/librispeech/*.txt` is ready for WER scoring.
2. **Re-run the marker sweep against any future ggml base.**
   `grep -rn "MINITTS_" audio.cpp/external/ggml/{src,include}` is the whole
   audit; it took one command to find what a function-level API inventory
   missed twice. Recorded in `external/ggml/UPSTREAM`.
3. **`flashsr_utility_test`** fails identically on CPU and CUDA
   (max_diff 1.7e-3, mean 3.3e-4 vs its bound) — listed as "missing assets" in
   the previous log, but it is actually a numeric-tolerance failure and worth a
   look on its own terms.
4. **Consider a per-head-mask CUDA kernel fix upstream.** The framework fallback
   is correct and currently faster/more accurate, so this is optional — but if
   ggml ever fixes `nb32` indexing in its flash kernels, the CPU-only gate in
   `common_relative_attention.cpp` should be revisited against measurements.

## LEFT TO DO (small)
- [ ] Optionally add `CMakePresets.json` (clang + default-On OpenMP one-command).
- [ ] Formalize root `.gitmodules` so `git submodule update` works for the 3 embedded repos.
- [ ] `LNK4217` warnings: the arch static libs see `TRANSCRIBE_API` as
      `dllimport` for symbols defined in the same DLL. Benign but noisy.
- [ ] `external/ggml` has 1145 of 2163 files with CRLF blobs while
      `.gitattributes` declares `* text=auto eol=lf`. Left alone deliberately —
      renormalizing is a 1145-file sweep, and `sync-ggml.sh` currently matches
      the committed state on Windows. It does mean the *checkout* step is
      platform-dependent; a Linux run would produce LF and a large diff.
- [ ] The testing-infrastructure port from the prior session
      (`tests/lint_teardown.cmake`, `tests/check_extension_umbrella.cmake`,
      `tests/golden/`, `tests/tolerances/`, `scripts/dump_reference_silero_vad.py`)
      is still untracked/uncommitted alongside this session's work.

## Notes / decisions made this session
- Tolerances are a property of *what a kernel computes*, not of which backend
  runs it. The fork-op test now names its two classes explicitly rather than
  branching on the backend name, so a future exact-activation GPU kernel gets
  held to the reference bounds and passes them.
- When a bound has to be loosened, prefer adding a second measurement that
  makes the loosening falsifiable — the `pack4 == plain mul_mat` cross-backend
  equality is worth more than the 5e-3 bound it sits behind.
- The convergence's remaining risk is no longer API drift. Both remaining
  classes are behavioural: fork relaxations that were dropped (findable via
  `MINITTS_*`), and upstream restrictions that are *correct* and must be
  respected in the framework rather than patched away in ggml (§3).
