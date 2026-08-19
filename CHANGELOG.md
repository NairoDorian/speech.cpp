# Changelog

All notable changes to `speech.cpp` — the fork of [audio.cpp](https://github.com/0xShug0/audio.cpp)
that is absorbing [transcribe.cpp](https://github.com/) per
`TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md`.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are the work-session dates recorded in the plan.

## [Unreleased]

### Fixed

- **The `patches/ggml/` invariant was broken in practice, not just at risk.**
  `external/ggml` is generated, and `e11e3c5` applied the two-sided broadcast
  restore (`binary-ops.cpp`, `ops.cpp`, `binbcast.cu`, `scale.cu`) **in place
  without a tracked patch** — the next `scripts/sync-ggml.sh` run would have
  deleted it silently, which is exactly the failure mode `external/ggml/UPSTREAM`
  warns about. Now tracked as
  `patches/ggml/0004-restore-two-sided-broadcast.patch`.
- **Per-head flash-attention masks are wrong on CUDA; the framework now avoids
  them.** `ggml_flash_attn_ext_with_bias_mask` folds a dense per-head additive
  bias into the mask, giving it `ne[2] == n_head`. Upstream 0.20.2 refuses such
  masks outright (`ggml_cuda_get_best_fattn_kernel` → `BEST_FATTN_KERNEL_NONE`),
  so `encoder_module_test` aborted at `fattn.cu` on a CUDA build. Restoring the
  fork's `MINITTS_FLASH_PER_HEAD_MASK` delta removed the abort and returned
  *wrong numbers*: ggml's CUDA kernels write the per-head offset into the mask
  pointer (`nb32*(head % ne32)`) yet still read head 0's slice for every head.
  Measured — the CUDA result matches a reference forced to slice 0 to 4.9e-4
  (the F16 floor of the shared-mask control) while diverging from the true
  per-head reference by 9.1e-2, and it reproduces at `gqa_ratio == 1`, so head
  grouping is not the cause. Upstream's restriction is therefore load-bearing.
  `use_specialized_flash_attention()` now keeps that lowering on CPU and falls
  through to the reference lowering elsewhere, the same shape as the
  `ggml_backend_supports_op` probe in `minimax_h3/dit_denoiser.cpp`. This is also
  an accuracy win: CUDA relative-attention drift against the F32 reference
  improved from 3.2e-3 max / 8.0e-4 mean to **8.1e-6 / 1.3e-6** (~390x), because
  the fallback additionally avoids the CUDA flash kernels' F32→F16 K/V
  conversion.
- **`ENGINE_BUILD_TESTS=ON` broke every lean build.** `AUDIOCPP_MODEL_SET=core`
  links *no* models, but eleven test targets reference model-internal symbols
  unconditionally, so any set short of `full` failed at link. Each is now gated
  on its owning model in `AUDIOCPP_LINKED_MODELS`, following the `vibevoice`
  precedent already in `CMakeLists.txt`: the four `moss_tts_local` harnesses plus
  `dots_tts_vocoder_parity`, `inflect_v2_frontend_test`,
  `voxtral_realtime_stream_chunking_test`, `unicode_normalization_test`,
  `outetts_generation_budget_test`, `qwen3_forced_aligner_processor_test`,
  `asr_standalone_gguf_test` and `parakeet_parity_dump`.
- **`encoder_module_test` never set `ctx.backend_type`.** Production builds a
  `ModuleBuildContext` as `{ggml, name, execution.backend_type()}` and modules
  branch on it to pick lowerings, so the test's "cuda" case was building
  CPU-configured graphs and merely *running* them on a CUDA device — every
  `ctx.backend_type == Cuda` branch in the framework went untested by it.
  `set_runner_backend()` now keeps the two in step, and the two hand-rolled
  runner setups in the flash-parity case go through it.
- **`tests/unittests/test_ggml_fork_ops_cpu.cpp` held CUDA to CPU tolerances.**
  The first execution of the CPU↔CUDA pass produced five failures, all of them
  tolerance-model bugs rather than kernel bugs: the CPU kernels are deliberate
  references that keep activations in F32, while the CUDA kernels quantize them
  (INT8 Q·Kᵀ with FP8 PV in
  `sage-attn2/qattn/qk_int_sv_f8_cuda_sm89.cuh`; per-token INT8 by `max_abs/127`
  in `convrot-linear.cu:112`), so a reference kernel's max-abs bound measures the
  quantizer. The remaining failure, a 6.9e-4 `mul_mat_pack4` CPU↔CUDA delta
  against a 1e-4 bound, is TF32: ggml-cuda sets `CUBLAS_TF32_TENSOR_OP_MATH` on
  every cuBLAS handle (`ggml-cuda/common.cuh:1502`), giving F32 GEMMs a 10-bit
  mantissa.
- **ggml: restored the two-sided broadcast fork delta lost in the 0.20.2
  convergence.** `audio.cpp`'s ggml let *both* operands of a binary op broadcast
  up to `dst`, and let `scale`'s `src0` broadcast up to `dst`; upstream 0.20.2
  allows only `src1` to broadcast and requires `src0` to already match `dst`.
  The framework graph optimizer depends on the relaxed form — it folds
  `add(repeat(row, full), col)` to `add(row, col)` and `scale(repeat(x, full))`
  to `scale(x)`, leaving the broadcast to the kernel — so after the convergence
  those folds emitted graphs no backend could execute (`encoder_module_test`
  aborted on `GGML_ASSERT(ggml_can_repeat(src1, src0) && ...)`). Restored on both
  CPU (`binary-ops.cpp`, `ops.cpp`) and CUDA (`binbcast.cu`, `scale.cu`), each as
  a separate kernel so future ggml upgrades need not re-merge them.
  R1's API-drift inventory missed this because it is a behavioural relaxation
  inside existing kernels, with no new symbol to grep for.
- **`ggml_compute_forward_convrot_linear` omitted the rotation.** The CUDA kernel
  fuses a QuaRot-style radix-4 orthonormal rotation into its activation
  quantizer, and the weights ship *pre-rotated*, so `W_rot · x` is not the
  layer's output — only `W_rot · (R x) == W · x` is. Shapes and dtypes validate
  without the rotation, so nothing but numerics catches it. The CPU kernel now
  applies the identical rotation and keeps activations in F32, which is strictly
  more accurate than the CUDA path (it drops only the lossy activation
  quantization) — what a golden reference should be.
- **`GGML_OP_SAGE_ATTEN2` → `GGML_OP_SAGE_ATTN2`** at six dispatch sites across
  the CPU, meta and CUDA backends. A hard compile error, which is why the
  Phase-2 CPU kernels had never actually been built.
- **`-DENGINE_BUILD_TESTS=ON` did not build.** The three `moss_tts_local` codec
  parity harnesses passed a HuggingFace snapshot *directory* into constructors
  that now take an `engine::assets::TensorSource`, and
  `engine::assets::open_tensor_source()` opens a weights *file*. Added
  `tests/moss_tts_local/codec_weights_path.h` to resolve either form, and dropped
  hardcoded `C:/Users/justi/...` defaults that made the harnesses unusable
  anywhere else. Pre-existing — the same three targets fail to compile at
  `213ac74`.
- **`ggml_backend_cpu_device_supports_op` claimed support it did not have.**
  `SAGE_ATTN2`, `CONVROT_LINEAR` and `MUL_MAT_PACK4` fell through to
  `default: return true`, so a mis-shaped graph aborted mid-compute instead of
  being rejected while the scheduler could still act on it.
- **MiniMax-H3's DiT hard-threw `"requires CUDA backend"`** for INT8 ConvRot
  weights, making the new CPU kernel unreachable. It now asks
  `ggml_backend_supports_op`, matching the pattern the `sage_attn2` site in the
  same file already used.
- **`audio_dsp_test` flakiness.** `test_istft_matches_reference_across_configs_and_variants`
  seeded its waveform from a per-run `steady_clock::now()` (`WaveformMode::PerRun`), so the
  test mutated its own input every invocation and intermittently tripped the ISTFT
  tolerance (~1 in 6 runs; reproduced 3/30 in a baseline loop). The profile is now pinned
  to a fixed seed and is deterministic — 50/50 clean in a repeat loop. The time-based
  randomization is retained behind the `AUDIO_DSP_TEST_RANDOMIZE` env var as an opt-in
  stress mode that still surfaces the historical failure (~2/15 in a stress loop).
- **MiniMax Music3 LM-head input precision on Metal** (port from upstream audio.cpp
  `4e973b1`). The Metal path now uses `GGML_TYPE_F32` for the LM head input instead of
  the Vulkan-shared `GGML_TYPE_F16`, eliminating BF16 convolution corruption on Metal;
  now identical to audio.cpp's converged path.

### Added

- **`patches/ggml/0005-cpu-concat-fastpath.patch`** — restores the fork's
  `MINITTS_CONCAT_FASTPATH` deltas, which the convergence dropped. Upstream still
  walks concat element by element, indexing all four dimensions per element
  ("TODO: smarter multi-theading"); the fork copies each contiguous `(ne0 x ne1)`
  plane (dim 1) or logical row (dim 0) with two bulk `memcpy`s. Not cosmetic: the
  framework concatenates per-head attention outputs in a loop (the
  `ExplicitCpuPerHead` lowering), and restoring the fast path is what returned
  `supertonic_vector_convnext_exp_test` — which asserts a ≥5% speedup — to green.
- **Numerical coverage for `ggml_flash_attn_ext_with_bias_mask`**, the fourth
  converged fork-only API, which had none; `external/ggml/UPSTREAM` listed it as
  "CPU-functional", meaning only that it linked. Checked against an independent
  reference — softmax over `scale*(QK + bias)` with the bias read back through
  F16 so the mask's own rounding is not scored as kernel error — for both a
  per-head and a shared (`ne[2] == 1`) mask, the latter acting as the control
  that isolates head indexing from precision. Includes a "does NOT collapse to
  bias slice N" guard: a backend that accepts a per-head mask and then serves one
  slice to every head produces a *plausible* answer, so the check has to be
  pinned from both sides.
- **R10 in `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md`**, certifying the ggml
  convergence on a GPU backend and recording three things the rest of the merge
  depends on: audio.cpp tags its own ggml deltas with `MINITTS_*` markers, so one
  grep enumerates the class R1's function-level API inventory kept missing (and
  those deltas live in the *initial* vendored drop — audio.cpp has only 14
  commits touching `external/ggml` and none touch `binary-ops.cpp`, so reading
  its history does not find them); a dropped fork delta is not automatically
  worth restoring, and must be verified numerically first; and lean model sets
  are a supported configuration that Appendix I should test.
- **CPU compute kernels for the three fork-only ggml ops** that were CUDA-only
  after the convergence: `GGML_OP_SAGE_ATTN2` (F16 scaled dot-product attention
  with GQA and causal masking), `GGML_OP_CONVROT_LINEAR` (rotated INT8 linear),
  and `GGML_OP_MUL_MAT_PACK4` (dispatches to `ggml_compute_forward_mul_mat`, as
  both the audio.cpp fork and the CUDA backend do). Graphs containing them no
  longer abort on the CPU backend.
- **`tests/unittests/test_ggml_fork_ops_cpu.cpp`** — checks each op against an
  independent in-test reference, on the CPU backend at 1 and 4 threads and, when
  the build has one, on a GPU device, then cross-checks the two. That CPU↔GPU
  delta is the per-tensor tolerance the plan's golden-manifest methodology wants.
  The convrot case pins the rotation via its orthogonality: rotating the float
  weight rows with the same transform must reproduce the un-rotated matmul, which
  it does to 0.7% normalized RMS (the INT8 weight-quantization floor) where a
  kernel skipping the rotation lands ~140% off.
- **`tests/abi_stream_hello.cpp`** — exercises the C ABI streaming surface
  (`transcribe_stream_begin/feed/finalize/reset`), which Phase 0.H fixed but
  nothing ran. Feed sizes cycle through 1/337/7/4099/63/1531/2/911, none of which
  divide into any plausible family chunk size, so chunk boundaries land at a
  different offset inside every feed. Asserts per feed that `input_received_ms`
  is monotonic and equals the audio fed, that `audio_committed_ms <=
  input_received_ms`, and that `buffered_ms` is their difference; after finalize
  that `committed == input`, so the sub-chunk tail is not stranded. Two
  begin/feed/finalize cycles per session prove `reset` clears per-utterance
  state. Streaming-vs-offline agreement against `silero_vad` on the four in-tree
  LibriSpeech fixtures: 0%, 0.51%, 0.86%, 0.98% drift in detected speech.
  Links only `libtranscribe` — no `engine_runtime`, no `ggml` — so it also proves
  the shipped shared library is self-sufficient for a streaming consumer.
- **`patches/ggml/0003-cpu-kernels-for-fork-only-ops.patch`** — `external/ggml`
  is generated and the Phase-2 kernels had been applied in place, so the next
  `scripts/sync-ggml.sh` run would have silently deleted them. Verified by
  regenerating into a scratch directory: `sync-ggml.sh` plus patches 0001–0003
  reproduces `external/ggml` with an empty `git diff`.
- **R9 in `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md`**, closing out the ggml
  convergence and recording the two facts a future backend port most needs:
  `convrot_linear` is a *rotated* INT8 linear, and `patches/ggml/` is the
  invariant rather than `external/ggml`.
- **This changelog.**

### Changed

- **The ggml convergence is certified on a GPU backend, not just on CPU.** The
  full suite builds and runs against CUDA 13.3 on an RTX 4070 (sm_89);
  `scaled_dot_product_attention_test`, which had never executed anywhere, passes.
  Both backends now sit at the same pre-existing failure baseline. Measured
  CPU↔CUDA agreement for the fork-only ops, as normalized RMS: `sage_attn2`
  3.7e-2 / 3.5e-2, `convrot_linear` 7.4e-3 / 7.8e-3, `mul_mat_pack4` 6.9e-4 —
  all consistent with the quantization each CUDA kernel performs.
- `test_ggml_fork_ops_cpu` now carries two named tolerance classes,
  exact-activation and quantized-activation, chosen by what a kernel computes
  rather than by which backend runs it — so a future exact-activation GPU kernel
  is held to the reference bounds and passes them. Every comparison reports
  max-abs *and* normalized RMS whether or not it is bounded, since a number that
  is merely reported today is what shows the next reader that a backend drifted.
  It also cross-checks plain `mul_mat` alongside `mul_mat_pack4` and asserts the
  two cross-backend deltas are **equal** — they are, to 1e-9, both 6.938e-04 —
  which turns the loosened TF32 bound into an actual statement that the `pack4`
  wrapper contributes no error of its own.
- `encoder_module_test` prints its measured drift on success as well as failure,
  and its CUDA bounds are tightened from 4.0e-3 / 9.0e-4 to 1.0e-4 / 1.5e-5,
  roughly 10x the measured values. Taking the flash lowering on CUDA lands at
  3.2e-3 / 8.0e-4, so the bounds are now what turns a silent regression back onto
  that path into a test failure. A CPU-vs-CUDA comparison of the lowering each
  backend actually selects was added alongside the per-backend reference checks.
- `abi_bridge_hello` and `abi_stream_hello` are now real CTest gates. Both were
  registered with no arguments, so they exited 2 and skipped on every run despite
  the `silero_vad` safetensors and LibriSpeech fixtures being in-tree. They now
  point at those assets, with `SKIP_RETURN_CODE=2` so exit 2 still reads as
  "skipped" where the assets are absent.
- `external/ggml/UPSTREAM` per-op status refreshed — it still described
  `SAGE_ATTN2` and `CONVROT_LINEAR` as CUDA-only with no CPU kernel and
  `MUL_MAT_PACK4` as deferred — and the stale closing note asking for patch 0002
  to be authored was dropped; 0002 has been authored and tracked since `60e603b`.

### Known issues

- **`flashsr_utility_test` fails identically on CPU and CUDA** (case 0:
  `max_diff` 1.7e-3, `mean_diff` 3.3e-4 against its bound). Pre-existing, and
  previously filed under "missing assets" — it is in fact a numeric-tolerance
  failure and needs looking at on its own terms.
- **`supertonic_vector_convnext_exp_test` asserts a ≥5% wall-clock speedup**, so
  it fails when another build or test suite is running on the same machine. Run
  it uncontended. (Contrast `audio_dsp_test`, whose flakiness was a seeding bug
  and was fixed rather than documented.)
- **Per-head flash-attention masks remain CPU-only.** The framework fallback is
  correct and currently both faster and more accurate on CUDA, so this is not
  urgent — but if ggml ever fixes `nb32` indexing in its flash kernels, the
  CPU-only gate in `common_relative_attention.cpp` should be revisited against
  fresh measurements rather than assumed obsolete.
- **No in-tree ASR model**, so end-to-end transcription is still unvalidated. The
  bridge tests currently exercise VAD segments, not text; both already accept a
  model path, and `assets/asr_validation/librispeech/*.txt` is ready for WER
  scoring.
- **`external/ggml` line endings are mixed** — 1145 of 2163 files have CRLF blobs
  while `.gitattributes` declares `* text=auto eol=lf`. Left alone deliberately:
  renormalizing is a 1145-file sweep, and `sync-ggml.sh` currently matches the
  committed state on Windows. It does mean the script's *checkout* step is
  platform-dependent — a Linux run would produce LF and a large diff.

## Earlier (pre-changelog)

Recorded here for continuity; see `git log` for detail.

- `213ac74` Phase 0.H/0.N — made the `ArchAdapter` reachable (framework sniff on
  the load path) and fixed its stream contract (`prepare()` before
  `start_stream()`, tail flush, contiguous cursors, an out-of-bounds write, and
  private-member shadowing).
- `66b824b`, `fcbbd4f` OpenMP DX — auto-detect LLVM `libomp` on Clang and
  `clang-cl` on Windows so the default `-DENGINE_ENABLE_OPENMP=ON` configures and
  builds out of the box, plus a configure-time note that `libomp.dll` must be on
  `PATH` at run time.
- `60e603b` Phase 0.L — converged the bundled ggml to upstream 0.20.2
  (`8c63e709`) with patches 0001/0002 and `scripts/sync-ggml.sh`.
- `e99e291`, `cc95c15`, `0df43b0` Phase 0 — integrated the transcribe.cpp runtime,
  implemented the `ArchAdapter`, and added the `abi_bridge_hello` smoke test.
