# audio.cpp upstream merge `c0b26a50..487800f5` (then `..9bdd1d90`) — trial-merge report, plan, and outcome

> **Status (2026-09-23, later): MERGED as `38769d51`**, extended to
> `9bdd1d90` (Release v0.8.2) because upstream moved 8 commits past the trial
> target while the ggml work was done (Nemotron 3 diarization, native batch
> transcription endpoint, a moss codec window fix, README). None of the 8
> touch ggml; all merged clean. `git rev-list --left-right --count
> main...upstream/main` = `122 0`.
>
> **What was done, in the order below:**
> 1. ggml deltas ported as `patches/ggml/0012-0015` (`f4d8e31e`), section (c):
>    c.1 (07490d83) API surface -> 0012; c.3 (82b4dc3a) API only -> 0013, with
>    the fusion flag moved to op_params slot 1 because 0.24.0's `SSM_SCAN`
>    keeps `K` in slot 0; c.6 (712dd75a) API + CPU kernels -> 0014; plus 0015,
>    which makes the CPU backend refuse the ops it cannot compute.
>    **Not ported (need CUDA / Metal / Vulkan builds):** c.2, the CUDA half of
>    c.3, c.4, c.5, the Metal half of c.6, c.7. Functional cost: LiveAvatar /
>    Wan S2V on CUDA (its five new ops have no kernel anywhere yet), audio8_tts
>    and Breeze fast paths on Metal. Everything else treats the lowerings as
>    hints.
> 2. The merge was redone in speech.cpp itself, reusing this report's
>    resolutions for files neither side had touched since (cli/main.cpp,
>    sortformer frontend + assets, family_registry.cpp) and resolving
>    `CMakeLists.txt`, `backend_weight_store.h` and `README.md` afresh the same
>    way (both had moved on our side). One defect the source-level trial could
>    not see: upstream `audio8_tts/ar.cpp` calls `ggml_ssm_scan` without 0.24.0's
>    `K` argument -> passes `K = 1`.
> 3. Verified on CPU (the routine test target): `build-cpu-core` 121/121,
>    `build-cpu-asr-abi` 115/115 (1 skip each); all ASR WER gates at baseline;
>    sortformer on the NeMo-GGUF layout unchanged — oracle DER 0.3288 and
>    chunked == whole-window at 1.8e-7, the pre-merge values (risk d.2 closed
>    for that layout; the HF layout's smoke test passes). CUDA not built (user
>    preference: CPU-only routine testing).
> 4. Still open from (d): R2T2 fixes from transcribe.cpp on `confucius4_r2t2`;
>    registering `confucius4_r2t2` / `nemotron_3_diar` in `family_registry`
>    (C-ABI routing) once their GGUF architecture names are confirmed; the
>    Metal / CUDA ggml ports; `kokoro_word_timings_test` / moss parity
>    executables in reduced presets (d.8).
>
> The original plan follows, unchanged, for the record.

---


- Scratch clone: `<SCRATCH>/speech-merge` (a `--no-local` clone of speech.cpp, with `core.autocrlf=false`; remote `upstream` = https://github.com/0xShug0/audio.cpp.git)
- Branch: `merge-upstream-487800f5`
- Merge commit: `8557474449e752c95534547afe19169ff6d49100` (parents `f3c2ffca`, `487800f5`)
- `git rev-list --left-right --count HEAD...upstream/main` gives `115 0`, so nothing on upstream is left unmerged.
- Merge base: `c0b26a50`, which is the upstream tip that `0686aab3` reconciled. That makes this a clean 61-commit window.
- The real speech.cpp repo was not modified: its HEAD is still `f3c2ffca` on `main`. Its working tree already had uncommitted `external/ggml/*` edits before I started. I did not touch them, and they are not part of this merge.
- EOL settings: speech.cpp uses `core.autocrlf=true`, and its `.gitattributes` has `* text=auto eol=lf`. The clone uses autocrlf=false and keeps that `.gitattributes`.
- **Nothing was built or run.** All checks were done at the source level.

## Headline risk: the merged tree does not compile until the ggml patch lands

`external/ggml` was kept at OURS, as instructed. Upstream engine code now calls ggml APIs that exist only in audio.cpp's own ggml fork. I diffed every `ggml_*`/`GGML_*` symbol in upstream-changed files against OURS headers. The only undeclared ones are the new upstream APIs:

| Caller (merged tree) | Missing ggml API | Target that breaks |
|---|---|---|
| src/framework/codecs/wan_video_vae_runtime.cpp | `ggml_im2col_2d_set_lowering`, `ggml_im2col_3d_set_lowering`, `ggml_mul_mat_set_lowering`, `ggml_rms_norm_channels{,_silu,_add_bias_silu}`, `ggml_rms_norm_channels_set_lowering`, `ggml_conv_3d_concat_pad_spatial_gemm_ex`, `ggml_conv_3d_concat_pad_spatial_gemm_set_lowering`, and the matching `GGML_*_LOWERING_*` enums | **engine_core** |
| src/framework/modules/linear_module.cpp:95 | `ggml_mul_mat_set_lowering`, `GGML_MUL_MAT_LOWERING_CUDA_NVFP4_F16_ACTIVATION` | **engine_core** |
| src/framework/modules/primitive_modules.cpp:326 | `ggml_rope_interleaved_pairs` | **engine_core** |
| src/framework/modules/structural_modules.cpp:341 | `ggml_concat_set_lowering`, `GGML_CONCAT_LOWERING_CUDA_CONTIGUOUS_4D` | **engine_core** |
| src/framework/modules/conv_modules.cpp:253 | `ggml_mul_mat_acc` | **engine_core** |
| src/community_models/audio8_tts/codec.cpp:517 | `ggml_snake_1d` | engine_model_audio8_tts |
| tests/unittests/test_conv1d_pertap_fast_path.cpp:235 | `GGML_OP_MUL_MAT_ACC` | conv1d_pertap_fast_path_test |

Even `build-cpu-core` fails until `patches/ggml/0008-*` (or several patches) adds at least the API surface from 07490d83, 82b4dc3a and 712dd75a. The other option is to temporarily guard these call sites, but I did not do that. The only non-ggml ABI-shaped difference I found is `ggml_get_to_fp16_nc_cuda`. The CUDA test uses it, and it already exists in OURS `ggml-cuda/convert.cuh` with the same signature.

## (a) Per-commit disposition table

Totals across 61 commits:

| Disposition | Count |
|---|---|
| MERGED-CLEAN | 45 |
| MERGED-RESOLVED | 6 |
| NEEDS-GGML-PATCH only | 5 |
| Mixed: engine merged + ggml hunks need a patch | 3 |
| N/A | 2 |
| ALREADY-PRESENT | 0 |

I checked the ALREADY-PRESENT count mechanically. `git apply --check -R` of each commit's non-ggml diff against a HEAD worktree matched nothing.

| # | sha | subject | disposition |
|---|---|---|---|
| 1 | 994c0cfc | fix(clap): contiguous layout before mel-projection matmul | MERGED-CLEAN |
| 2 | 5752a66f | cli: --out-format for WAV sample format | MERGED-RESOLVED: `app/cli/main.cpp` conflict, both sides kept (conflict 1) |
| 3 | 542bb4ea | Release (pre) v0.8.1 (docs, README news, model_specs) | MERGED-RESOLVED: README Projects date took upstream's (conflict 5b). The rest merged clean, including the v0.8.1 news line in speech.cpp's fused README. |
| 4 | f2b49373 | Release v0.8.1: drop `* text=auto eol=lf` from .gitattributes | N/A: kept OURS (post-merge fix A) |
| 5 | 550557a6 | tests(yue2): standalone NAR parity probe | MERGED-CLEAN (the CMake block is gated on `yue2 IN_LIST AUDIOCPP_LINKED_MODELS`) |
| 6 | 07490d83 | Add explicit ggml lowering APIs | NEEDS-GGML-PATCH (see c.1) |
| 7 | 62c664af | Add CUDA lowering implementations | NEEDS-GGML-PATCH (c.2) |
| 8 | 82b4dc3a | Opt-in CUDA SSM gate fusion and tiled F32 im2col | NEEDS-GGML-PATCH (c.3) |
| 9 | 51855086 | Reusable Wan S2V framework components | MERGED-CLEAN, but it won't compile without c.1 (wan_video_vae_runtime, linear/primitive/structural modules) |
| 10 | a074d6b8 | LiveAvatar native model support | MERGED-RESOLVED: `backend_weight_store.h` (conflict 2) |
| 11 | 456c8e78 | fix(scripts): bash 3.2 empty arrays | MERGED-CLEAN |
| 12 | 4bb1a204 | breeze: bf16 activation rounding on Metal | Mixed: engine parts (breeze generator/session/spec/docs) MERGED-CLEAN; ggml-metal hunks NEEDS-GGML-PATCH (c.4) |
| 13 | 4b9a39fe | zipvoice community TTS | MERGED-CLEAN |
| 14 | a7b58a6d | Confucius4-R2T2 streaming ASR | MERGED-CLEAN. Not added to `AUDIOCPP_ASR_MODEL_TARGETS`, following the precedent that canary_asr, cohere_asr and granite5asr aren't in it either. Your call. |
| 15 | 9d82abd7 | vibevoice_asr_streaming 1.5B | MERGED-CLEAN |
| 16 | 219a7e0e | AuK Base / AuK-Flash | MERGED-RESOLVED: CMake (conflict 4). AuK tests are gated on `auk` being linked. |
| 17 | 9ba88417 | speech speed across server + TTS | MERGED-CLEAN |
| 18 | 9c00066a | Rebuild oversized cached ASR encoder graph | MERGED-CLEAN |
| 19 | a045e690 | fix(voxcpm2): left context | MERGED-CLEAN |
| 20 | 40ffe9cd | prompt query param on live transcription route | MERGED-CLEAN |
| 21 | 6c70f32d | fix(voxcpm2): drop right context | MERGED-CLEAN |
| 22 | 29441016 | Piper + KittenTTS | MERGED-CLEAN |
| 23 | 93d76313 | fix(voxcpm1): stateful streaming decode on Vulkan | MERGED-CLEAN |
| 24 | 5785167a | vulkan-shaders-gen: retry empty compile | NEEDS-GGML-PATCH (c.5) |
| 25 | 2d7b7bc5 | webui: reproducible bundle | MERGED-CLEAN |
| 26 | 1a2f73ab | ci: run CI jobs locally | MERGED-CLEAN |
| 27 | e3de8e3f | Shared configurable NeMo mel frontend | MERGED-RESOLVED: `sortformer_diar/frontend.cpp` (conflict 3), plus the NeMo-GGUF loader wiring in `assets.cpp` |
| 28 | f7f5dd11 | "#1 repository of the day" badge | N/A: audio.cpp README badge (conflict 5a) |
| 29 | 62e44c54 | moss_tts_v15 + promote moss_tts_delay runtimes | MERGED-CLEAN |
| 30 | 9aa53fe2 | Improve Nemotron ASR streaming | MERGED-CLEAN. Auto-merged with speech.cpp's `argmax_logits` change in decoder.cpp; I confirmed `argmax_index` was not reintroduced. |
| 31 | 779714db | Shared mel + reference audio frontends | MERGED-CLEAN |
| 32 | 2792c657 | kokoro_tts word timings | MERGED-CLEAN |
| 33 | f0e3a900 | yue2: export semantic token stream | MERGED-CLEAN |
| 34 | 741fe030 | perf(roformer): pipeline CUDA chunks | MERGED-CLEAN |
| 35 | 39342d1d | webui: YuE2 NAR LoRA | MERGED-CLEAN |
| 36 | 17cc8980 | server memory guard Windows encoding (CI yml) | MERGED-CLEAN |
| 37 | 2fff3c8c | yue2: stop_after | MERGED-CLEAN |
| 38 | 5d786f97 | yue2: tile eager NAR attention | MERGED-CLEAN |
| 39 | eb8e21bd | vulkan-shaders-gen: retry unstartable compiler / back off | NEEDS-GGML-PATCH (c.5) |
| 40 | 712dd75a | audio8_tts Falcon-H1 + codec/AR perf | Mixed: engine parts (ar.cpp, codec.cpp, conv_modules, tests, CMake) MERGED-CLEAN; ggml hunks NEEDS-GGML-PATCH (c.6) |
| 41 | bef51fcc | perf(cuda): narrow RoFormer ops | Mixed: CMake CUDA tests MERGED-RESOLVED (conflict 4); ggml-cuda hunks NEEDS-GGML-PATCH (c.7) |
| 42 | 23f466af | docs: model weight license overview | MERGED-CLEAN |
| 43 | 61e062d3 | /utf-8 for Windows engine_core | MERGED-CLEAN |
| 44 | 8d5e10aa | vieneu_v3_turbo (formerly vietneu_tts) | MERGED-RESOLVED: textually clean, but needed post-merge fix B in `family_registry.cpp` |
| 45 | d32a6a01 | yue2: continue from semantic tokens | MERGED-CLEAN |
| 46 | f6e24957 | gguf: quantize rows in parallel | MERGED-CLEAN (tensor_source.cpp auto-merged with speech.cpp's own additions) |
| 47 | 1ee4ce82 | moss_tts_v15: chunking options in spec | MERGED-CLEAN |
| 48 | 4e8be77d | Fix Seed-VC V1 F0 conditioning | MERGED-CLEAN |
| 49 | 0f9306e7 | YuE2 empty lyrics | MERGED-CLEAN |
| 50 | c40e7810 | fix(workflow): escape control chars in file-sink JSON | MERGED-CLEAN |
| 51 | 7130f25b | Update WebUI for v0.8.2 | MERGED-CLEAN |
| 52 | 066c6704 | KittenTTS + Piper in WebUI | MERGED-CLEAN |
| 53 | 1865fe96 | MOSS-TTS v1.5 in WebUI | MERGED-CLEAN |
| 54 | 84764929 | sanoTTS voice labels | MERGED-CLEAN |
| 55 | 1ff4c958 | LiveAvatar in WebUI (+ pipeline_state.cpp) | MERGED-CLEAN |
| 56 | 88cd5712 | Instrumental YuE2 in WebUI | MERGED-CLEAN |
| 57 | abe55464 | AuK in native UI | MERGED-CLEAN |
| 58 | 639de869 | Merge PR #655 (release/v0.8.2-ui) | MERGED-CLEAN (merge commit; its content is rows 51-57) |
| 59 | a6049dae | "Merge PR #658" vieneu follow-up (single-parent) | MERGED-CLEAN |
| 60 | ab268906 | Moonshine ASR audio chunking | MERGED-CLEAN. This touches upstream's `moonshine_asr` family, not speech.cpp's native `moonshine`/`moonshine_streaming` packages (see d.9). |
| 61 | 487800f5 | Clarify VAD audio chunking errors | MERGED-CLEAN |

## (b) Conflicts and how each was resolved

Git reported 22 content conflicts and 1 modify/delete conflict. 17 of them were under `external/ggml`; 5 were outside it.

### Outside external/ggml

1. **app/cli/main.cpp** (5752a66f vs speech.cpp's weight sharing): kept both.
   - `wav_options = wav_write_options_from_cli(...)` runs first, so a bad `--out-format` still fails before the model loads.
   - speech.cpp's `ScopedWeightShareKey share_scope(share_key)` then wraps `registry.load` and `create_task_session` exactly as before.
   - `wav_options` is consumed further down, at line 859.

2. **include/engine/framework/core/backend_weight_store.h** (a074d6b8 `buffer_type` vs speech.cpp's `kMetadataPoolBudget` cap and SharedWeightRegistry):
   - Kept speech.cpp's comment block and `kMetadataPoolBudget`.
   - Adopted upstream's constructor, which takes `ggml_backend_buffer_type_t buffer_type = nullptr` and adds the `buffer_type_` member.
   - Added a private `alloc_weight_buffer()` helper that uses `ggml_backend_alloc_ctx_tensors_from_buft(buffer_type_)` when a buffer type was given, and the backend default otherwise.
   - All three allocation sites now call the helper: the independent `upload()` path, the shared-provider lambda, and the shared fingerprint-conflict fallback. Upstream only changed the first; without this, a sharing scope would silently drop LiveAvatar's host buffer type.
   - **Addition of my own:** when `buffer_type_` is set, `"|buft:<name>"` is appended to `share_key_`. This stops a host-buffer store and a device-buffer store with an identical tensor set from binding to each other's buffers.

3. **src/models/sortformer_diar/frontend.cpp** (e3de8e3f shared NemoMelFrontend vs speech.cpp's Phase-10.5 layout-dependent frontend). This is the riskiest resolution.
   - Upstream replaced the hand-rolled mel frontend with `assets.frontend->extract_audio(...)`, configured for HF defaults only.
   - speech.cpp's version has three layout knobs from `SortformerFeatureExtractorConfig`: HF uses `peak_normalize`, per_feature normalization and floor frames; the NeMo GGUF uses no peak norm, `"NA"` normalization and ceil frames.
   - Resolution: I took upstream's `NemoMelFrontend` and mapped each knob onto it. `make_sortformer_frontend` maps `peak_normalize` to `WaveScale::DivideByMaxPlusEps` or `None`, and `normalize` to `MelNorm::PerBinF32` or `None`. At run time `frame_count` becomes `ValidFrameRule::FloorHops` or `CeilHops`.
   - Kept speech.cpp's up-front error messages ("positive sample rate", "requires 16 kHz"), and added a null-frontend guard.
   - **Behavior difference I compensated for:** `NemoMelFrontend` copies `min(frames, raw_frames)` frames, including raw frames past `valid_frames`. With `MelNorm::None`, those frames would no longer be zero. speech.cpp's old code zero-padded everything from `valid_frames` on. The resolved code now explicitly zero-fills `[valid_frames, frames)` after extraction. With PerFeature normalization `FeatureNormalizer` already zeroes them, so that is a no-op there.
   - **Upstream bug for speech.cpp, fixed:** upstream only builds `assets->frontend` in the HF loader. speech.cpp's `load_nemo_sortformer_assets` (NeMo GGUF) would have hit a null `frontend` dereference. I added the same `make_shared<NemoMelFrontend>(make_sortformer_frontend(*assets))` line there (`src/models/sortformer_diar/assets.cpp`, right after `model_weights`).
   - One small remaining difference: `NemoMelFrontend::prepare_audio` throws on empty audio. The old code may have produced a zero-frame batch. I did not verify how the old path behaved on empty input.

4. **CMakeLists.txt** (the test block): speech.cpp's `audiocpp_test_link_model_objects()` helper sat where upstream put the AuK tests (219a7e0e) and the CUDA unit tests (bef51fcc).
   - Kept the helper.
   - Kept upstream's three `ENGINE_ENABLE_CUDA` tests unchanged: `cuda_bias_fusion_test`, `cuda_rope_rows_test`, `cuda_head_convert_test`.
   - Wrapped the three AuK test executables in `if (auk IN_LIST AUDIOCPP_LINKED_MODELS)`. They compile against auk model objects, which need `yaml_vendor`, so they would fail to link under the `asr`/`custom`/client presets. I chose gating over `audiocpp_test_link_model_objects` because of that yaml_vendor dependency.
   - Everything else in CMake auto-merged: engine_core sources, the 7 new `audiocpp_add_model` families, the `vietneu_tts`→`vieneu_v3_turbo` rename, `/utf-8` on engine_core, and the new tests.
   - I checked mechanically that every `.cpp` upstream added under `src/` and `tests/` is referenced in CMakeLists.txt. None are missing.

5. **README.md**:
   - (a) The title block conflicted with upstream's trendshift badges (f7f5dd11). Kept speech.cpp's `# speech.cpp` intro; the badges point at audio.cpp's own repo and don't apply here.
   - (b) The Projects "Last update" line: took upstream's 2026-09-17 (542bb4ea), because the AIRI entries moved into that list in the same change.
   - Auto-merged into speech.cpp's README: the v0.8.1 news line, the model-license sentence (it says "audio.cpp's own license"; you may want to reword it), and the new model table rows.

6. **external/ggml/\*\***: 16 content conflicts (ggml.h, ggml.c, ggml-cpu.c, 9 ggml-cuda files, ggml-metal-device.h, ggml-metal-ops.cpp, vulkan-shaders-gen.cpp) and 1 modify/delete (`ggml-metal.metal`, which is deleted in OURS because 0.22.0 splits shaders into `ggml-metal/kernels/*.metal`).
   - Resolved everything with `git checkout HEAD -- external/ggml`.
   - That also reverted hunks that had auto-merged into 10 files (binbcast.cuh, convert.cu, im2col.cu, norm.cu/.cuh, ssm-scan.cuh, unary.cu/.cuh, ggml-metal-device.cpp/.m, ggml-metal-impl.h, ggml-metal-ops.h).
   - Removed the two files upstream added: `ggml-cuda/conv3d.cu` and `conv3d.cuh`.
   - Verified with `git diff --name-only f3c2ffca HEAD -- external/ggml`, which returns 0 files.

### Post-merge fixes (no textual conflict)

- **A. `.gitattributes`:** upstream f2b49373 deletes `* text=auto eol=lf`. I kept OURS. The Windows repo runs with `core.autocrlf=true`, and sync-ggml's LF-normalized patch application depends on the rule, so dropping it would change every checkout's line endings.
- **B. `src/framework/runtime/family_registry.cpp`** (speech.cpp-only catalog): the entry `{"vietneu_tts", ..., "model_specs/vietneu_tts.json"}` pointed at a spec upstream renamed. I changed it to `{"vieneu_v3_turbo", aliases {"vieneu","vietneu","vietneu_tts"}, "model_specs/vieneu_v3_turbo.json"}`.
  - The upstream loader already lists `vietneu_tts` in `family_aliases()`, and remaps `vietneu_tts.*` session options. So `capi/test/*.ps1/.bat`, which still pass `--family vietneu_tts`, should keep resolving. I did not run them.
  - `test_family_registry` checks for alias uniqueness; the new aliases don't collide with any existing entry.

## (c) NEEDS-GGML-PATCH list, with file and function detail

Across 8 commits, none of these hunks apply to OURS: `git apply --check` fails on every commit, forward and reverse. They have to be ported onto 0.22.0 by hand. Metal hunks target the monolithic `ggml-metal.metal`; in OURS the kernels live in `src/ggml-metal/kernels/{unary,ssm,mul_mm,...}.metal`.

**c.1 07490d83: Add explicit ggml lowering APIs.** This is the API surface engine_core needs.

`include/ggml.h`:
- New enums: `ggml_mul_mat_lowering`, `ggml_concat_lowering`, `ggml_im2col_2d_lowering`, `ggml_im2col_3d_lowering`, `ggml_rms_norm_channels_lowering`, `ggml_conv_3d_concat_pad_spatial_gemm_lowering`.
- `ggml_mul_mat_lowering` values are 0, 2 and 3. They skip 1 because they share op_params slot 1 with `ggml_mul_mat_set_hint` (`GGML_HINT_SRC0_IS_HADAMARD=1`). Keep that numbering; OURS also puts the hint in slot 1.
- New ops appended after `GGML_OP_IM2COL_ASYM`: `CONV_3D_CONCAT_PAD_SPATIAL_GEMM`, `RMS_NORM_CHANNELS`, `RMS_NORM_CHANNELS_SILU`, `RMS_NORM_CHANNELS_ADD_BIAS_SILU`, `ROPE_INTERLEAVED_PAIRS`.
- New declarations: `ggml_rope_interleaved_pairs`, `ggml_concat_set_lowering`, `ggml_conv_3d_concat_pad_spatial_gemm{,_ex,_set_lowering}`, `ggml_rms_norm_channels{,_silu,_add_bias_silu,_set_lowering}`, `ggml_mul_mat_set_lowering`, `ggml_im2col_2d_set_lowering`, `ggml_im2col_3d_set_lowering`.

`src/ggml.c`:
- Extend `GGML_OP_NAME` and `GGML_OP_SYMBOL`, and bump the `static_assert(GGML_OP_COUNT == ...)` (+5; upstream went 107→112).
- `ggml_add_cast_impl`: the assert was relaxed from `ggml_can_repeat_rows(b,a)` to `ggml_can_repeat(b,a)`. wan_video_vae_runtime.cpp:187 relies on it. Check that the CPU and CUDA add_cast kernels really handle full broadcast.
- New builders placed after `ggml_concat`, `ggml_rms_norm_inplace`, `ggml_mul_mat_set_hint`, `ggml_im2col` and `ggml_im2col_3d`.
- Op-param slots used: concat 1; conv3d gemm 0-5 for padding, 6 for lowering; rms_norm_channels eps and 1 for lowering; mul_mat 1; im2col_2d 7 (it asserts `is_2D` in slot 6); im2col_3d 10.

**Caution:** upstream adds **no CPU kernels** for RMS_NORM_CHANNELS*, ROPE_INTERLEAVED_PAIRS or CONV_3D_CONCAT_PAD_SPATIAL_GEMM. They are CUDA-only (c.2), and wan/LiveAvatar is CUDA-only. speech.cpp's patch-0003 doctrine gives fork ops CPU kernels. Decide whether this patch does the same or just declares them unsupported on CPU.

**c.2 62c664af: CUDA lowering implementations.** Everything is under `src/ggml-cuda/`:
- binbcast.cu: `k_bin_bcast_unravel`, `launch_bin_bcast_pack`, `k_repeat_back`, `ggml_cuda_op_mul`, `ggml_cuda_op_div`. This is a large rewrite of about 1000 lines.
  - **Conflicts with OURS patch 0004 (restore-two-sided-broadcast).** Reconcile the two carefully.
- binbcast.cuh: declaration change.
- concat.cu: `concat_cuda`, `concat_cuda_typed`. This adds the CONTIGUOUS_4D lowering. Watch for interaction with OURS patch 0005, the concat fast path (CPU only, but check).
- **New files:** conv3d.cu and conv3d.cuh (the CONV_3D_CONCAT_PAD_SPATIAL_GEMM kernels).
- ggml-cuda.cu:
  - `ggml_cuda_op_mul_mat_cublas`, `ggml_cuda_mul_mat`: the TILE_F16_ACCUM_OUTPUT and NVFP4_F16_ACTIVATION lowerings.
  - `ggml_cuda_compute_forward`: dispatch for the new ops.
  - `ggml_backend_cuda_device_supports_op`.
- im2col.cu: `im2col_kernel`, `im2col_cuda{,_f16,_f32}`, `ggml_cuda_op_im2col`, `im2col_3d_kernel`, `im2col_3d_cuda{,_f16,_f32}`, `ggml_cuda_op_im2col_3d`. The im2col lowerings may collide with OURS patch 0006 (conv-im2col-in-weight-type).
- mmq.cu: `ggml_cuda_mul_mat_q`.
- norm.cu and norm.cuh: `rms_norm_f32`, `ggml_cuda_op_rms_norm`, and the new rms_norm_channels kernels.
- quantize.cu and quantize.cuh: `compute_e8m0_scale`, `quantize_mmq_nvfp4`, `quantize_mmq_mxfp4`, `quantize_mmq_fp4_cuda`.

**c.3 82b4dc3a: Opt-in CUDA SSM gate fusion and tiled F32 im2col.**
- `include/ggml.h`: `enum ggml_ssm_scan_fusion` and `ggml_ssm_scan_set_fusion`. The `ggml_ssm_scan` declaration change is whitespace only.
- `src/ggml.c`: `ggml_ssm_scan_set_fusion` (op_params slot 0); `ggml_ssm_conv` and `ggml_ssm_scan` context.
- `ggml-cuda/ggml-cuda.cu`: `ggml_cuda_can_fuse`, `ggml_cuda_try_fuse` (the SSM gate fusion).
- `ggml-cuda/im2col.cu`: `im2col_cuda_f16` and `im2col_cuda_f32` (the `F32_K3_TILED` lowering).
- `ggml-cuda/ssm-scan.cu` and `ssm-scan.cuh`: the `ssm_scan_f32` kernel, `ssm_scan_f32_cuda`, `ggml_cuda_op_ssm_scan`.

**c.4 4bb1a204: Metal bf16 for breeze.**
- `ggml-metal-device.cpp`: `ggml_metal_library_get_pipeline_unary`.
- `ggml-metal-device.m`: `ggml_metal_device_supports_op`, which should accept F16↔BF16 for CPY/SET/DUP/CONT.
- `ggml-metal-impl.h`: `OP_UNARY_NUM_ROUND_BF16`.
- `ggml-metal.metal` (in OURS, `kernels/unary.metal` and the cpy kernels): `kernel_unary_impl` gains ROUND_BF16. It adds the instantiations `kernel_unary_{f16,bf16}_f32(_4)`, `kernel_cpy_contig_{f16_bf16,bf16_f16}` and `kernel_cpy_{f16_bf16,bf16_f16}`.
- OURS already has `GGML_UNARY_OP_ROUND_BF16` in ggml.h but no Metal implementation. The new breeze code enables bf16 rounding on Metal, so **Breeze on Metal will fail at runtime until this lands.**

**c.5 5785167a and eb8e21bd: vulkan-shaders-gen robustness** (build tooling only). `src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`:
- `execute_command`: now returns int, closes pipe fds on error, and puts errno in the exception.
- New `compile_count_guard` / `acquire_compile_slot`, with the cap halved on spawn failure.
- `string_to_spv_func`: judges success by a non-empty SPIR-V artefact; retries 3x with backoff of 250ms→2s; the try block moves inside the retry loop.
- `write_output_files`: never declares a symbol whose artefact is empty.
- `main`.

This is optional, but cheap to carry for Vulkan CI.

**c.6 712dd75a: Falcon-H1 / audio8 codec ops.**
- `include/ggml.h`: `GGML_OP_MUL_MAT_ACC`, `GGML_OP_SNAKE_1D`, `ggml_mul_mat_acc`, `ggml_snake_1d`.
- `src/ggml.c`: NAME/SYMBOL tables (+2; upstream went 112→114) and `ggml_mul_mat_acc`, which returns a `ggml_view_tensor(acc)` with src[0..2] = a, b, acc. `ggml_snake_1d` is added after `ggml_mul_mat`.
- `src/ggml-cpu/ggml-cpu.c`: `ggml_compute_forward_mul_mat_id` area (new forward functions), `ggml_compute_forward` dispatch, `ggml_get_n_tasks`. **CPU kernels are provided for both ops.**
- Metal:
  - `ggml-metal-device.cpp`: `ggml_metal_library_get_pipeline_unary`, `ggml_metal_library_get_pipeline_mul_mm` (mul_mm with accumulation).
  - `ggml-metal-device.h`: prototypes.
  - `ggml-metal-device.m`: `supports_op`.
  - `ggml-metal-impl.h`: kargs struct.
  - `ggml-metal-ops.cpp`: `ggml_metal_op_encode_impl`, `ggml_metal_op_unary`, `ggml_metal_op_mul_mat`.
  - `ggml-metal-ops.h`.
  - `ggml-metal.metal`: `kernel_unary_impl` (snake), **`kernel_ssm_scan_f32` (a real bug fix: reduction garbage for d_state>32, n_t<sgptg)**, `kernel_mul_mm` and `kernel_mul_mm_f32_f32` (acc).
- **CUDA has no MUL_MAT_ACC or SNAKE_1D.** conv_modules gates `use_acc` only on shape (`in_channels >= 64 && output_frames > 8`), and codec.cpp gates snake on Metal. Check that the conv_modules per-tap path is only reached on CPU/Metal; otherwise CUDA graphs containing MUL_MAT_ACC will be rejected.

**c.7 bef51fcc: narrow RoFormer CUDA ops.** Everything is under `src/ggml-cuda/`:
- convert.cu: `convert_unary`, `convert_unary_cuda`, used by the fp16 nc head-convert.
- ggml-cuda.cu: `ggml_cuda_can_fuse` and `ggml_cuda_try_fuse` (bias add + GELU/residual fusion).
- rope.cu: `rope_yarn`, `rope_norm`, `rope_norm_cuda` (packed rows).
- unary.cu: `ggml_cuda_op_gelu`.
- unary.cuh.

These are performance-only. The three new CUDA unit tests (`cuda_bias_fusion_test`, `cuda_rope_rows_test`, `cuda_head_convert_test`) compare against CPU, so they should pass without the patch. I did not verify that.

## (d) Risks the main session must verify by building and testing

1. **The build is blocked until ggml patch 0008 lands.** At minimum that means the c.1, c.3 and c.6 API surface. Then build `build-cpu-core`, the full CPU build, and CUDA.
   - The targets that fail today are listed in the headline table: engine_core, engine_model_audio8_tts and conv1d_pertap_fast_path_test.
   - After the patch, run `ctest`. The previous baseline was 110 passed and 1 skipped.
2. **Sortformer (conflict 3, highest risk).** Run:
   - `sortformer_diar_engine_smoke_test`
   - `sortformer_diar_streaming_engine_test`
   - `sortformer_diar_scheduler_test`
   - `sortformer_diar_ext_abi_test`
   - `sortformer_v2_aosc_test` and `sortformer_v2_schedule_test`
   - the `nemo_mel_migration_probe` executable
   - the engine-vs-arch decision-parity gate: previously 0/600 flips, max |dp| 7e-3, and chunked == whole-window at 1.8e-7

   Run all of these on **both layouts**: the HF package, and the NeMo GGUF, where normalize may be "NA" and frames are ceil. The NeMo GGUF path is the one I changed beyond upstream.
3. **BackendWeightStore (conflict 2).** Check CLI weight sharing (`ScopedWeightShareKey`) with a multi-store model (moss, IndexTTS2). Also run LiveAvatar with host-offloaded blocks, which exercises the `buffer_type` + share-key path. `tests/unittests/test_shared_weight_vram.cpp` exists, but I found no registration of it in CMakeLists.txt or cmake/.
4. **ASR WER gates.** Upstream rewrote the NeMo mel frontends of citrinet, canary, cohere, hviske, nemotron, granite5asr and parakeet_tdt, and reworked Nemotron streaming. Run:
   - `asr_e2e_wer_test`, `asr_e2e_whisper_wer_test`, `asr_e2e_qwen3_asr_wer_test`, `asr_e2e_sense_asr_wer_test`, `asr_e2e_fun_asr_nano_wer_test`, `asr_e2e_edits_test`
   - the three pinned gates (moonshine offline/streaming, whisper .bin)
   - a Nemotron streaming run
   - the new `asr_graph_capacity_test`
5. **family_registry:** run `family_registry_unit`; `vieneu_v3_turbo` replaces `vietneu_tts` in the catalog.
6. **CLI:** `cli_request_options_test` (upstream extended it for `--out-format`) and `file_sink_json_test`.
7. **New tests from upstream** to run once built:
   - `mel_spectrogram_frontend_test`
   - `quantize_chunking_test`
   - `kokoro_word_timings_test`: it links `engine_model_kokoro_tts` directly, not model-set gated, so it may fail under `asr` or client presets
   - `conv1d_pertap_fast_path_test`
   - `audio8_tts_falcon_kv_cache_test`
   - the `moss_tts_v15_text_normalization*` tests
   - the CUDA tests
8. **Reduced composites** (`AUDIOCPP_MODEL_SET=asr`, `custom`, and the client-compact presets): configure and link.
   - The new extended-test executables `moss_voicegen_codec_encode_parity`, `moss_tts_delay_backbone_parity` and `moss_tts_v15_prompt_parity` are not model-gated. Neither are the existing moss_voicegen executables in OURS; the c3b41731 gating was lost in an earlier merge.
   - `kokoro_word_timings_test` has the same issue.
9. **Follow-up decisions, not merge defects:**
   - Should `confucius4_r2t2` join `AUDIOCPP_ASR_MODEL_TARGETS`?
   - ab268906's Moonshine chunking landed only in upstream's `moonshine_asr`. speech.cpp's native `moonshine` package does not get it.
   - README wording ("audio.cpp's own license").
10. **Metal:** Breeze bf16 (c.4) and the audio8 snake/acc and ssm_scan fix (c.6) need the Metal patch. They can't be tested on Windows.

## Things I was unsure of

- I did not verify that `NemoMelFrontend`'s `LogMelSpectrogram` path with `STFTFamily::Default` and `MelWindow::StftHann` is bit-identical to speech.cpp's old direct `LogMelSpectrogram().compute(...)` call for the NeMo GGUF layout. Upstream's migration report claims parity for the HF layout only.
- The `"|buft:"` share-key suffix is my addition, not upstream's.
- Gating the AuK tests departs from upstream's unconditional registration.
- I did not check whether the CUDA unit tests compile against 0.22.0's internal headers. `cuda_head_convert_test` declares `ggml_get_to_fp16_nc_cuda` by hand; the signature matches OURS convert.cuh.
