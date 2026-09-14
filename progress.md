# Progress — Unified_Audio.cpp (speech.cpp ggml fork) merge & improve

Status snapshot: **Upstream audio.cpp main fully merged and synchronized at `3eccab50` — 0 behind (30 commits merged from baseline `78d47706`). All conflicts resolved across CMakeLists.txt (fused client build toggles with upstream's opt-in `AUDIOCPP_BUILD_C_API` facade, mutually exclusive with the Universal C ABI `AUDIOCPP_BUILD_CAPI` — both define target `audiocpp`; upstream server frontend module wired into the gated `AUDIOCPP_BUILD_SERVER` block) and README.md (news fused). New in-tree header shadow resolved: upstream `include/audiocpp.h` facade vs Universal `capi/include/audiocpp.h` — `test_batch_dispatch` include order fixed. 100% CTest pass rate (110 passed, 2 clean fixture skips, 0 failed out of 112 on `build-cpu-core`). Client compact minimized preset re-verified post-merge.** Date: 2026-09-14

## Repo layout (important, non-obvious)
`Unified_Audio.cpp/` is a **plain container directory with no git repo of its
own**. It holds five independent repositories (three primary, two hardened reference trees):

| Folder | Role |
|---|---|
| `speech.cpp/` | the active development repo (the ggml/audio.cpp fork). **All merge work, and this log, live here.** Remote: `NairoDorian/speech.cpp`, upstream `0xShug0/audio.cpp`. |
| `audio.cpp/` | **parent** — read from, never committed to (pulled to `6d530f4`, 2026-08-28). Has a git `upstream` remote here, so it is the only source that yields a merge-base. |
| `transcribe.cpp/` | **parent, equally authoritative** — read from, never committed to (pulled to `2102bca`, 2026-08-26 — carried the ggml bump to `36da5713` / v0.22.0 that we then adopted). No remote here, so its drift is invisible to git and must be triaged by hand — that is a tooling limit, **not** a hierarchy. See AGENTS.md "Dual Parentage". |
| `audio_cunba/` (pulled to `8cf5136`) & `transcribe_cunba/` (pulled to `2345350`) | hardened reference trees containing allocator fixes, VAD chunk planning, shared weights, batched decoders, C ABI, and build acceleration. |

Build trees are scratch dirs under `C:/Users/Z/AppData/Local/Temp/opencode/`:
`sp_bridge` (CPU, full model set, unified ABI + arches, tests), `sp_cuda`
(CUDA, core set, ABI/arches OFF), `audiocpp_flashsr` (audio.cpp reference),
`build-cpu-core` (local MSVC CPU core test suite), `build-cpu-asr` (local MSVC ASR test/executable build).

## Overall progress (toward "Unified_Audio transcribes on CPU")
| Area | Status | % |
|---|---|---|
| Dependency: ggml pin `36da5713` (v0.22.0) + patches 0001–0007, matched to parent transcribe.cpp | Done 2026-08-26 — **CPU 102/102 + CUDA 57/57 green**, re-sync reproduces exactly (0 paths) | 100% |
| Gates: Whisper arch baseline locked (`asr_e2e_whisper_wer_test`, pinned `ggml-tiny.en.bin`) | Done 2026-08-26 — **corpus WER 4.34783% (3/69), RTF 0.047**; the bar the W2 engine port must match | 100% |
| Doctrine: dual parentage (transcribe.cpp is a co-parent) + `scripts/sync-deps.sh` routine | Done 2026-08-26 (AGENTS.md, tracker Rule 7) | 100% |
| Merge: Upstream audio.cpp main synchronization (`6d530f4`) | Done — 0 behind, 88 ahead, all 35 commits dispositioned & merged | 100% |
| Memory: Phase 1 Allocator Hardening (16MB cap, WavLM gallocr, Qwen3 runaway, DFN2) | Done (certified in engine) | 100% |
| Toolchain: Phase 2 Modernization & Build Provenance (ccache, transcribe-build-info, version.rc) | Done (certified in build scripts & DLL) | 100% |
| Long-form: Phase 3 Native VAD Chunk Planning & Re-stitching (`vad_plan`, `vad_merge`) | Done (native Silero + Energy VAD, C ABI) | 100% |
| Optimization: Phase 7.4 Shared Weight Registry & Scoped Activation | Done & Verified (`test_shared_weight_vram`) | 100% |
| Scaling: Phase 7.5 Offline Batched ASR Decode across all 16 Arch Adapter slots | Done & Verified (`test_batch_dispatch`, C ABI) | 100% |
| Registry: Phase 7.8 Unified Family Registry v1 | Done & Verified (`family_registry_unit`) | 100% |
| Regression: Phase 7.1 Port transcribe.cpp Test Suite & Fixtures | Done (51 TUs, 87 golden fixtures, 13 GGUF fixtures) | 100% |
| Remediation: Phase 7.3 Defect D1 (sniff dispatch) & D2 (pre-clear) | Done & Verified (`test_adapter_sniff_dispatch`) | 100% |
| **Phase 8: Contract Convergence & Exception Boundary** | **Done & Verified (`StreamingSessionBase`, `RunControl`, `StreamChunker`, 100% C ABI Exception Guards)** | **100%** |
| **Phase 9: Unified Mel & Tokenizer Subsystems** | **Done & Verified (`MelExtractor`, `TokenizerHub`, `FrontendSpec`, `IAudioCodec`, Parity Tests)** | **100%** |
| **Phase 10: Attention & Conformer Module Fusion** | **Done & Verified (`sanm`, `shaw_attn`, `causal_lm_ops`, Bake-Off certified)** | **100%** |
| **Phase 11 W1a: Native Engine Moonshine (offline)** | **Done & Verified (`moonshine_engine_smoke_test`: engine-path WER 1.449% == arch 1/69 edits; batch + abort contracts)** | **100%** |
| **Phase 11 W1b: Native Engine Moonshine-Streaming** | **Done & Verified (`moonshine_streaming_engine_smoke_test`: streamed 4.348% == offline 4.348% == arch baseline 3/69, divergence 0; lifecycle + abort contracts)** | **100%** |
| **Phase 11 W2a: Native Engine Whisper (offline core)** | **Done & Verified (`whisper_engine_smoke_test`: 4.34783% == arch baseline 3/69, RTF 0.155; legacy `.bin` loader + unified MelExtractor)** | **100%** |
| Specs: Phase 6 Whisper & Moonshine Model Spec Catalogs | Moonshine spec corrected + backed by native loader. **Whisper: `whisper.json` is catalog-only in the strong sense — its 16 packages point at `Whisper-*-GGUF` paths that do NOT exist in `audio-cpp/audio.cpp-gguf` (no Whisper dir at all), so none are downloadable.** Family now gated via the legacy `.bin` instead (see W2 prerequisite). | ~70% |
| ABI offline + streaming surface | Verified, real CTest gates | 100% |
| End-to-end ASR **offline text** (WER gate) | Done — 1.45% corpus WER (arch path); engine path now also 1/69 edits | 100% |
| **End-to-end ASR streaming text** | **Done — streamed 4.35% == offline 4.35%, divergence 0** | **100%** |
| Test suite status | **110/110 green** on `build-cpu-core`; `build-cpu-full` unlocked (firered_audio fix) — ASR smoke tests (`whisper`, `moonshine`, `moonshine_streaming`, `qwen3_asr`, `voxtral_realtime`, `hviske_asr`, `nemotron_asr`) + WER gates (`whisper`, `qwen3_asr`) green | **100%** |
| **Completed increment** | **Upstream `78d47706` (0 behind), ggml 0.22.0 (CPU+CUDA certified), Phase 11 W1a + W1b + W2a** | **DONE** |
| **Phase 10.5, family 3 of 5: `sortformer_diar` step 2 (feature-merge)** | **Done 2026-08-27** — chunked AOSC/FIFO scheduler + presets + typed RUN ext in the engine; the catalogue's default (NeMo-layout) v2 package opens in the engine (neither parent could); chunked == whole-window to 1.8e-7; 0/600 decision flips vs the arch on identical weights; **111/111** core, C-ABI ext gate OK. Report: `docs/reports/sortformer_diar_engine_port.md` | 100% |
| **Next increment** | **Phase 11a: DecodeDriver + WhisperEmbeddingModule (B31)** | **IN PROGRESS** |

## DONE this session

### Phase 11a: DecodeDriver + B30 migration (2026-09-14/15, commits a6d02849, d43f754e)

Created the shared `framework/asr/` layer and collapsed all private duplicates onto it:

- **DecodeDriver** (`include/engine/framework/asr/decode_driver.h`): introduced shared `argmax_logits()` and `suppress_logits()` utilities. Migrated 7 ASR engine packages (whisper W2a, qwen3_asr B11, voxtral_realtime B12, hviske_asr, nemotron_asr, higgs_audio_stt, fun_asr_nano B14) off their private `argmax_index`/`argmax_index_ptr` implementations — 7 private functions (63 LOC) + 13 call-site duplications eliminated. All migrated packages compile and verify green on `cpu-full`: WER gates (whisper, qwen3_asr), smoke tests (whisper, qwen3_asr, voxtral_realtime, hviske_asr, nemotron_asr), and probe tests (fun_asr_nano decoder/encoder/session).
- **B30 EncDecKVCache** (commits ea326212, a6d02849): created `framework/asr/enc_dec_kv_cache.{h,cpp}` with `EncDecKVCache` struct + `kv_cache_init()`. All three ASR engine packages (whisper, moonshine, moonshine_streaming) now `using` alias `engine::asr::EncDecKVCache` — removed 3 byte-for-byte identical struct definitions, 3 `kv_cache_init()` definitions, and 3 `free()` methods (271 LOC). No graph code changes needed (fields accessed identically).
- **firered_audio fix** (3de02cdc): unblocked the cpu-full build by passing `K=1` to `ggml_gated_delta_net` (ggml v0.22.0 added the `K` parameter). Without this fix, the full 375-target ASR build could not be verified.

### Phase 11a B30 — Collapse 3 private KV caches onto shared `EncDecKVCache` (2026-09-14, commit a6d02849)

The three ASR engine packages each carried a byte-for-byte identical KV cache struct (`MoonshineKvCache`, `MoonshineStreamingKvCache`, `WhisperKvCache`) and a duplicate `kv_cache_init()` + `::free()` implementation. Replaced all three with `using` aliases for `engine::asr::EncDecKVCache`.

- **`include/engine/framework/asr/enc_dec_kv_cache.h`** — created (2026-09-14, commit ea326212). Defines `EncDecKVCache` and `kv_cache_init()` with the 4D tensor layout shared across all encoder-decoder families: `self_k`/`self_v` as `[d_model, n_layer * n_ctx]`, `cross_k`/`cross_v` as `[d_model, n_layer * T_enc]`, allocated via `ggml_backend_alloc_ctx_tensors`.
- **Graph headers** (`include/engine/models/{whisper,moonshine,moonshine_streaming}/graphs_internal.h`): `struct WhisperKvCache { … }` → `using WhisperKvCache = engine::asr::EncDecKVCache;` (and same for Moonshine/MoonshineStreaming). The graph builders access KV fields (`self_k`, `self_v`, `cross_k`, `cross_v`, `n`, `head`, `n_ctx`, `T_enc`, `cross_populated`) identically — no graph code changes needed.
- **Graph sources** (`src/models/*/graphs.cpp`): removed the three duplicate `kv_cache_init()` definitions and `::free()` methods (271 lines deleted total).
- **Runtimes** (`src/models/*/runtime.cpp`): `kv_cache_init(...)` → `engine::asr::kv_cache_init(...)` (3 call sites).
- **firered_audio fix** (`3de02cdc`): the cpu-full ASR build was blocked by a ggml v0.22.0 API change — `ggml_gated_delta_net` gained a `K` (state-snapshot count) parameter. Added `K=1` at both call sites in `src/models/firered_audio/qwen35_runtime.cpp`, matching the old API's default behavior (keep only final state).
- **Verification**: full cpu-full reconfigure + build clean (91 targets). All three ASR engine smoke tests pass: `moonshine_engine_smoke_test` (PASS, 4.76s), `moonshine_streaming_engine_smoke_test` (PASS, 25.78s), `whisper_engine_smoke_test` (PASS, 13.74s). WER gate: `asr_e2e_whisper_wer_test` PASS (3.17s) — encoder port still matches arch baseline after the KV cache unification. `test_adapter_sniff_dispatch` PASS.

### Phase 10.5, family 4 of 5 — `fun_asr_nano` retirement (2026-08-27, commit 1be9ac40)
Completed the final retirement of the Phase 10.5 family wave, dropping the parallel `transcribe.cpp` arch in favor of the engine `fun_asr_nano` family.

- **`src/runtime/arch/funasr_nano/`**: deleted entirely (~3,372 LOC across 11 files). The engine `fun_asr_nano` frontend already ships its own LFR + KaldiFbank + CMVN + frontend ports (via `engine/framework/audio/kaldi_fbank.h`); nothing else in the arch tree included from this dir.
- **`src/runtime/transcribe-arch.cpp`**: dropped `&funasr_nano::arch` + the namespace declaration from the builtin dispatch table.
- **`src/runtime/transcribe-kaldi-fbank.{cpp,h}`**: removed `.cpp` from the `engine_transcribe_runtime` OBJECT library (the last two arch consumers — funasr_nano and sensevoice — are retired); updated the dangling "still used by arch/funasr_nano" comment.
- **`CMakeLists.txt`**: added `asr_e2e_fun_asr_nano_wer_test` WER gate against the pinned `models/fun-asr-nano-2512-f16.gguf` (gated on `fun_asr_nano` linked + file present → SKIP otherwise).
- **`tests/transcribe/loader_smoke.cpp` + `tests/fixtures/make_gguf_fixtures.py`**: the synthetic `arch_funasr_nano.gguf` (still carries `general.architecture = "funasr_nano"`) now sniffs through the family-registry alias map to the engine `fun_asr_nano` family, whose loader rejects the missing-tensor payload → `TRANSCRIBE_ERR_UNSUPPORTED_ARCH` (same path as B13's sensevoice).
- **Verification**: `cpu-core` build clean (engine_transcribe_runtime, asr_e2e_wer_test, test_adapter_sniff_dispatch compile + link); `test_adapter_sniff_dispatch` passes (adapter routes "funasr_nano" GGUF → fun_asr_nano family correctly). `cpu-full` has a pre-existing firered_audio ggml API mismatch (upstream `3eccab50` / v0.22.0) unrelated to this change.

### Phase 11a foundation — AsrResult/AsrLimits, TaskResult.truncated, VAD dedup (B29) (2026-09-14, commit c59b15a0)

Established the shared ASR runtime layer that all three engine ports (moonshine offline W1a, moonshine_streaming W1b, whisper offline W2a) will converge onto.

- **L11 fix**: Added `bool truncated = false` to `TaskResult` (`include/engine/framework/runtime/session.h`). Previously `TaskResult` had no `truncated` field, violating the "every ASR result must carry truncated honestly" rule. Propagated through the three ASR sessions:
  - `MoonshineSession::run()` — `result.truncated = transcription.truncated;`
  - `MoonshineStreamingSession::on_finalize()` — `result.truncated = transcription.truncated;` (uses `MoonshineStreamingTranscription::truncated` + `stream_truncated_`)
  - `WhisperSession::run()` — `result.truncated = transcription.truncated;`
  This also fixes the W2a defect where Whisper > 30 s audio was silently truncated with no signal to the caller.

- **Shared types**: Created `include/engine/framework/asr/asr_result.h` with `AsrResult { text, truncated }` and `AsrLimits { max_audio_ms, max_output_tokens, sample_rate }`. All three per-family transcription structs (WhisperTranscription, MoonshineTranscription, MoonshineStreamingTranscription) carry the same `{text, truncated}` shape — `AsrResult` is the canonical unification target.

- **VAD dedup (B29)**: Deleted `src/runtime/transcribe-vad{,-integrate}.{h,cpp}` (~500 LOC), which were verbatim ports of `audio/chunking.cpp`'s `plan_vad_audio_chunks` and `append_chunk_speech_metadata`. Consolidated the arch-level VAD runner (`detect_speech`, `run_with_vad`, merge helpers) into `transcribe.cpp` as `namespace transcribe::vad`. The local `plan()` clone is now replaced by `engine::audio::plan_vad_audio_chunks()` — the shared sampler-based chunker. Merge helpers (`offset_chunk_results`, `rollback_to`, `rebuild_full_text`) retained since they operate on `transcribe_session` (arch-level), not `TaskResult` (engine-level). Removed orphaned `transcribe-kaldi-fbank.cpp` from the runtime OBJECT library (last arch consumers funasr_nano + sensevoice retired in Phase 10.5).

- **Tests**: Deleted `vad_plan_unit.cpp` + `vad_merge_unit.cpp` (coverage subsumed by `test_audio_chunking`).
- **Verification**: `cpu-core` build clean (engine_transcribe_runtime, transcribe.dll, test_adapter_sniff_dispatch compile + link); `test_adapter_sniff_dispatch` passes; `audio_chunking_test` passes. `cpu-full` has a pre-existing firered_audio ggml API mismatch (upstream v0.22.0) unrelated to this change.

### Client compact minimized build profile (2026-09-14)
New build posture for client-side desktop GUI embedding without the HTTP server, WebUI assets, or unused model families.
- **CMake toggles**: `AUDIOCPP_BUILD_SERVER` / `AUDIOCPP_BUILD_CLI` / `AUDIOCPP_BUILD_GGUF_TOOL` (default ON), `AUDIOCPP_STRIP_DEAD_CODE` (Release `/Gy /Gw /GF` + `/OPT:REF /OPT:ICF` on MSVC; `-ffunction-sections/-fdata-sections` + `--gc-sections` elsewhere).
- **Registry fix**: model registry include/loader generation moved after the dependency scan — registry content now covers `AUDIOCPP_LINKED_MODELS` (dependency closure, e.g. `qwen3_forced_aligner` auto-linked for `qwen3_asr`), not just the user-selected set.
- **Test gating**: unified-ABI bridge/unit tests, `capi_test`, and `model_perf` now gated behind `ENGINE_BUILD_TESTS`. `speech`/`speechcpp` alias targets added for the `audiocpp` C API library.
- **Presets**: `client-compact-minimized-cpu` / `-cuda` (server OFF, curated set: supertonic, nemotron_asr, granite5asr, qwen3_asr, parakeet_tdt). Docs: `docs/build/client_minimized_build.md` + README section.

### Upstream audio.cpp reconciliation — `3eccab50`, 0 behind (2026-09-14)
Synchronized 30 commits from upstream `0xShug0/audio.cpp:main` (from baseline `78d47706` up to `3eccab50`). Highlights: opt-in C ABI for in-process embedding (#530), optional server frontend module (#330 series: HTTPS listener, pipeline decoupling), Moonshine ASR GGUF (#510/#531), Kokoro refactoring + multilingual GGUF CPU (#515/#496), Niagara ASR family (#517), Parakeet TDT path-sized encoder graph (#536), VibeVoice ASR flash suffix attention (#526), vocoder 32-bit overflow fixes (#512), model config size_t overflow fix (#513), portable Linux release artifacts (#529), v0.7.4 release, WebUI sentence splitting (#539).
- **Conflict & Fusion Resolution**:
  - `CMakeLists.txt` conflict 1 (options): fused speech.cpp's four client toggles + unified-ABI options with upstream's `AUDIOCPP_BUILD_C_API` option and `AUDIOCPP_C_API_MODEL_ROOT` cache.
  - `CMakeLists.txt` conflict 2 (C ABI / CLI region): kept speech.cpp's build-info setup; dropped upstream's *unguarded* `audiocpp_cli` and WebUI hex-embedding blocks (speech.cpp's gated versions already exist) — the gating IS the client-build feature.
  - `CMakeLists.txt` conflict 3 (C ABI block / server region): kept speech.cpp's Universal C ABI block (`.def`/`.map` export control, aliases, tests) verbatim; re-homed upstream's new C API block **after** it with a `FATAL_ERROR` mutual-exclusion guard (both define target `audiocpp` and export `audiocpp_*` symbols; Universal ABI wins by default, facade opt-in requires `AUDIOCPP_BUILD_CAPI=OFF`).
  - Server frontend: fused upstream's `app/server/frontend.cpp` source and `audiocpp_configure_server_frontends(audiocpp_server)` call into speech.cpp's gated `AUDIOCPP_BUILD_SERVER` block; `server_frontends.cmake` include auto-merged (module system defaults OFF, self-contained; `external/audio.cpp-server-frontends` submodule is opt-in only).
  - `README.md`: fused news — upstream 0.7.4 entry + speech.cpp 09-10 sync note + 0.7.2 entry, chronological.
  - Header shadow: upstream's new `include/audiocpp.h` facade shadowed the Universal `capi/include/audiocpp.h` for any target with `include/` first on the path — `test_batch_dispatch` include order fixed in `cmake/transcribe-tests.cmake` (`capi/include` precedes `include/`). Verified the Universal `audiocpp` target itself and the `capi_*` tests resolve `capi/include` first via their own explicit dirs.
- **Verification**:
  - MSVC x64 Release: `cpu-core` preset configure + build clean (134 ninja targets, ccache-warm); `client-compact-minimized-cpu` configure clean post-merge with correct dependency closure.
  - CTest: **112 tests — 110 passed, 2 skipped (known unpinned-weight fixture skips: `transcribe_sortformer_stream_ext_unit`, `transcribe_stream_committed_pointer_stability`), 0 failed.** New upstream tests all green: `transformer_kv_ring_test`, `transpose_module_test`, `unicode_normalization_test`, `tdt_decoder_duration_loop_test`, `asr_standalone_gguf_test`, `kokoro_cpu_kernel_test`, `i8_s_fused_ops_test`, `i2_s_mul_mat_test`.
  - `git rev-list --count main..upstream/main` reads `0` after merge.
- **Follow-up (future increment)**: converge the two C ABI surfaces — upstream's thin `src/capi` facade (`include/audiocpp.h`, engine::runtime facade, 69 exports) and speech.cpp's Universal C ABI (`capi/`, 14 tasks, 50+ symbols) share the `audiocpp` target name and `audiocpp_*` symbol prefix but are separate APIs. Currently mutually exclusive by guard; true single-surface convergence is open.

### Upstream audio.cpp reconciliation — `78d47706`, 0 behind (2026-09-11)
Synchronized 8 commits from upstream `0xShug0/audio.cpp:main` (from baseline `3174e6b2` up to `78d47706`).
- **Commit Ledger**:
  - `014bd372`: Added Colab WebUI notebook (`Notebooks/colab_audio_cpp.ipynb`) and release workflow for CUDA T4 prebuilts. Added Colab badge under `## WebUI` in `README.md`.
  - `4a2c403c`: Updated `README.md` latest model news (Yue2 3B song generation dev testing; consolidated VibeVoice ASR Streaming 7B and Irodori-TTS v4.1 Anime announcement).
  - `183b0c98` & `efb04233`: Integrated NVIDIA Sortformer v2.1 streaming diarization community model (`sortformer_diar_v2`) under `src/community_models/sortformer_diar/`, `include/engine/community_models/sortformer_diar/`, `model_specs/sortformer_diar_v2.json`, and converter `tools/community_models/convert_sortformer_v2_1.py`. Completely non-conflicting with native engine `src/models/sortformer_diar/`.
  - `cd2fe109`: Fixed warmbench `voxtral_realtime` streaming flag resolution by reading from `warmup_case.get("streaming", False)`.
  - `43833e87`: Applied Vulkan fill dispatch fix in `external/ggml/src/ggml-vulkan/ggml-vulkan.cpp` and `fill.comp` splitting task distribution into a 2D grid to respect `maxComputeWorkGroupCount`. Fully compatible with pinned ggml `36da5713` (v0.22.0).
  - `eb1a569a`: Added shared eSpeak-ng phonemizer (`src/framework/audio/espeak_*`), data packing/caching, CMake static eSpeak option, updated `sanotts` and `inflect_v2` frontends, and added unit tests (`espeak_data_test`, `espeak_phonemizer_test`, `espeak_frontend_probe`).
  - `78d47706`: Kept Fish Audio Fast-AR inputs on HIP stream (`src/models/fish_audio/ar.cpp`, `hip_fast_sampler.*`).
- **Conflict & Build Resolution**:
  - `README.md`: Reconciled news header and WebUI sections preserving `speech.cpp`'s FUSION Roadmap blueprint alert and custom embedded WebUI documentation.
  - `CMakeLists.txt`: Moved `audiocpp_test_link_model_objects` higher up so `sortformer_v2_aosc_test` and `sortformer_v2_schedule_test` link `engine_model_sortformer_diar_v2` objects cleanly even when `AUDIOCPP_MODEL_SET=core`.
- **Verification**:
  - MSVC x64 Release build clean (290 targets).
  - CTest suite: 110 tests total, 108 Passed, 2 Skipped (expected fixture skips), 0 Failed (100% pass rate).
  - New tests verified individually: `sortformer_v2_aosc_test` (PASS), `sortformer_v2_schedule_test` (PASS), `espeak_data_test` (PASS), `espeak_phonemizer_test` (PASS).
  - `git rev-list --left-right --count HEAD...upstream/main` reads `92 0` (0 behind).

### Upstream audio.cpp reconciliation — `fa5aaac9`, 0 behind (2026-09-10)
Synchronized 91 commits from upstream `0xShug0/audio.cpp:main` (from baseline `6d530f4` up to `fa5aaac9`).
- **Conflict Resolution**:
  - `external/ggml/`: Kept pinned invariant `36da5713` (v0.22.0) matching `transcribe.cpp`. Reconciled `vulkan-shaders-gen.cpp`, `ggml-vulkan.cpp` (cleanly adding BF16 round op & copy pipelines), `ggml.c` (MSVC C2371 fix for `ggml_calc_conv_output_size`), and `ggml-cpu/ops.cpp` (`ggml_i8_s_quantize_range`/`requantize`).
  - `CMakeLists.txt`: Integrated upstream 3-tier test structure (`ENGINE_BUILD_TESTS`, `ENGINE_BUILD_EXTENDED_TESTS`, `ENGINE_BUILD_MODEL_TESTS`), preserved dual parentage (`TRANSCRIBE_VERSION` + `AUDIOCPP_VERSION`), preserved Phase 10.5/11 test additions, and added MSVC `/utf-8` compile option.
  - `README.md`: Integrated 0.7.2 upstream notes, community models (`sanotts`, `sopro_tts`, `soprano_tts`, `vibeasr`, `voxcpm1`), and binary distribution table while preserving `speech.cpp` dual-parentage notices.
  - `include/engine/framework/model_spec/schema.h` & `src/framework/model_spec/schema.cpp`: Aligned `kModelSpecSchemaVersion = 1` matching upstream and all catalog model specs.
  - `tests/unittests/test_http_live_body.cpp`: Updated test expectation for chunked uploads on non-live routes to `buffered: 4` reflecting commit `9c2f238a3b58` ("Buffer chunked HTTP uploads (#479)").
- **Verification**:
  - Built Release mode on MSVC x64: 573/573 targets cleanly built.
  - CTest test suite: 106 tests total, 104 Passed, 2 Skipped (expected fixture skips), 0 Failed (100% pass rate).

### 0. Upstream audio.cpp reconciliation — `c79e588`, now 0 behind (2026-08-26)

The repo had been reporting **"6 commits behind `0xShug0/audio.cpp:main`"**
every session. Two of the six were phantoms: the prior sync (`9b34fd2`)
**content-copied** upstream rather than merging, so the merge-base never left
`62735ea` and git kept re-listing commits already applied verbatim. Audited all
six **by content, not by subject**:

| upstream | disposition |
|---|---|
| `288a271` `--list-devices` (#299) | already present — `print_backend_devices` at `backend.h:33` / `backend.cpp:213`, both call sites, plus our `speech_*_list_devices` tests |
| `d25ffac` out-of-span chunk metadata (#301) | already present — `chunking.cpp:652` + `test_chunk_speech_metadata_merge_drops_outside_spans` |
| `4ec485d` supertonic voice preset (#302) | **cherry-picked** `90659b1` |
| `d03b957` IndexTTS2 HIP F16 KV/conv (#305) | **cherry-picked** `3682698` |
| `c6805de` README 0.7 banner | **N/A** — patches an audio.cpp README banner; our README is a speech.cpp rewrite with no such banner, and the content is upstream project news |
| `c79e588` tag-driven release CI (#286) | **adapted** `a775463` — artifacts renamed `audio-<tag>-…` → `speech-<tag>-…` |

Closed with a `-s ours` reconciliation merge carrying the full disposition
ledger: base advanced to `c79e588`, tree untouched, and
`git rev-list --left-right --count HEAD...upstream/main` now reads **`63  0`**.
**Future upstream syncs must merge, not content-copy** — otherwise the phantom
count returns.

Reference trees pulled the same day: `audio.cpp` → `c79e588`, `transcribe.cpp`
→ `2102bca` (**ggml bumped to upstream master `36da5713` / v0.22.0** — relevant
to our pin at `8c63e709` + `patches/ggml/0001…0006`; see the ggml-patch
invariant before acting on it), `audio_cunba` → `8cf5136`, `transcribe_cunba` →
`2345350`. `speech.cpp` itself was never pulled — only `git fetch upstream`.

Verification: `build-cpu-core` rebuilt (`engine_model_supertonic` and
`engine_model_index_tts2` compile clean); **CTest 100/100** (96 passed, 4
documented skips), unchanged from baseline. Caveat recorded honestly: at
`MODEL_SET=core` both packages are OBJECT libraries with **no consumer**, so
neither fix is runtime-exercised by this suite, and the IndexTTS2 change is
HIP-only (needs an AMD GPU to observe).


### 1. Dual parentage recorded + ggml bumped to 0.22.0 (2026-08-26)

**Doctrine first.** `speech.cpp` is **equally a child of `audio.cpp` and of
`transcribe.cpp`**. Forking audio.cpp was a convenience — it was the larger
tree to start from — not a statement of precedence. The distinction had been
quietly eroding because only audio.cpp has a git `upstream` remote, so only it
yields a merge-base and an "N behind" number; transcribe.cpp drift is invisible
to git here, and the docs reinforced it by calling transcribe.cpp a "merge
source" / "read-only reference tree". The concrete failure: transcribe.cpp
bumped ggml to `36da5713` (v0.22.0) while we sat on `8c63e709` (0.20.2), and
that was treated as a curiosity to note rather than as **our own dependency
floor moving**. Now recorded in `AGENTS.md` § "Dual Parentage" and tracker
**Operating Rule 7**.

**Routine.** `scripts/sync-deps.sh` reports drift across all three sources —
audio.cpp via the `upstream` remote, transcribe.cpp via the sibling checkout,
ggml compared both to upstream HEAD **and to parent transcribe.cpp's pin**.
`--fetch` fast-forwards the siblings. It never pulls/merges/re-vendors
speech.cpp; it prints the remediation command per stale item. Run it regularly
and **always before a release state**.

**The bump.** `8c63e709` (0.20.2) → `36da5713` (0.22.0). All 7 tracked patches
apply; 88 paths changed (the big deletion is upstream splitting
`ggml-metal.metal`, 11,820 lines, into `ggml-metal/kernels/` — restructuring,
not loss). Two patches needed real work:

- **0005 concat fast paths — rebased.** Upstream **rewrote**
  `ggml_compute_forward_concat_any` into a row-wise `memcpy` loop, i.e. it has
  *converged on most of this fork delta* (which existed to replace 0.20.2's
  scalar element walk). The `len` local the patch's byte math used is gone;
  byte counts now go through `ggml_row_size`, which is block-aware — this also
  removes a latent over-count for quantized types in our own delta.
  `concat_f32` is untouched upstream and keeps its full win. Remaining
  `concat_any` win is memcpy *count* only; flagged to benchmark before the
  next bump.
- **0007 CUDA trim-pools/clear-graph — regenerated.** The old file was
  hand-written and had **never been round-trip verified**: a hunk declared a
  21-line new side for a 35-line body, and there was no trailing newline, so
  `git apply` called it corrupt. Its content was in the vendored tree anyway —
  **the delta was one sync away from silent loss**, precisely what
  `patches/ggml/` exists to prevent. It is also the only patch whose targets
  carry CRLF blobs upstream, which is why it never got a clean run.

API drift is **purely additive** (`ggml_clamp_inplace`, `ggml_rope_set_offset`,
`ggml_backend_cuda_allreduce_tensor`); **no engine source changes were needed.**

Verified: clean build (444 targets), **CTest 100/100** unchanged from the
0.20.2 baseline — `moonshine_engine_smoke_test` and `ggml_fork_ops_cpu_test`
both green — `lint_teardown` green at its `src/runtime` scope, and a re-sync
reports **`0 path(s) changed`**: `sync + patches == committed tree`, exactly.
**Not covered:** CUDA/HIP/Metal/Vulkan are not built in `cpu-core`, so patch
0007's CUDA entry points and the 0.22.0 CUDA kernel churn are compile-
unverified pending an `sp_cuda` run.


### 2. Phase 11 Wave W1b — Native Engine Moonshine-Streaming, closed (2026-08-26)

Second family off the arch layer, onto the engine framework. New package
`src/models/moonshine_streaming/` (`graphs/assets/runtime/session` + internal
headers under `include/engine/models/moonshine_streaming/`), registered via
`audiocpp_add_model` and in the ASR composite, so CLI / server / WebUI / C ABI
resolve it by canonical id or the `moonshine-streaming` alias.

Ported numerics-identically from `src/runtime/arch/moonshine_streaming/`: the
time-domain frontend (CMVN → `asinh(exp(log_k)·x)` → linear+SiLU → two causal
stride-2 convs), encoder blocks with **per-layer sliding-window masks and no
RoPE**, the adapter (**absolute-frame** `pos_emb` get_rows + optional proj), the
**untied `lm_head`**, vanilla decoder LNs. The three conformer helpers the arch
borrowed are reimplemented locally — the package has **no `src/runtime/`
dependency**.

**Lifecycle is base-owned.** Session derives from `StreamingSessionBase`; we
implement only `on_start_stream` / `on_process_audio_chunk` / `on_finalize` /
`on_reset` + a pure `validate_chunk`. Per the design map's preferred option we
drive `update_text(full)` per partial and let the base's **STABLE_PREFIX**
policy (agreement_n=3) pick the commit boundary — one commit policy for every
family, instead of a per-family LCP.

**Result — the port reproduces the arch numerics exactly:**

| | offline | streamed | divergence |
|---|---|---|---|
| engine package (W1b) | **4.34783% (3/69)** | **4.34783% (3/69)** | **0 words** |
| arch `asr_stream_text_wer_test` baseline | 4.35% (3/69) | 4.35% (3/69) | 0 words |

Gate `moonshine_streaming_engine_smoke_test` (CTest #90, 12.9 s): registry load
by id + alias, family must advertise ASR offline **and** streaming, then each
fixture is run offline and streamed **on the same session** in odd,
non-frame-aligned chunks (1601/3203/6397/2477/4801 ≈ 100–400 ms) with the
throttle at 0. Also asserts monotonic revision, append-only committed text,
Active/Finished/Idle transitions, `reset()` clearing state, and abort unwinding.

Two findings worth carrying forward:
- **`model_specs/moonshine_streaming.json` had the same latent Phase-6 defect
  W1a found in `moonshine.json`** — its gguf source required `config.json` /
  `tokenizer.json` sidecars the pinned packages do not carry
  (`audiocpp_gguf --inspect` → `embedded_sidecars=false`), so it could never
  resolve. Fixed to tensors-only. **Scope note (corrected after a catalog
  sweep):** a `files:` block on a gguf source is the *norm and is correct* —
  `audiocpp_gguf` embeds sidecars by default and fails conversion if it cannot
  find them, and 54+ specs legitimately declare one. This is not a
  catalog-wide defect; it hits only families pinned to **third-party GGUFs
  this pipeline did not produce**, which is exactly the moonshine pair
  (`handy-computer/moonshine-*-gguf`). Check per family with `--inspect`
  before porting; do not rewrite `files` blocks on the pattern alone.
- `StreamingSessionBase` guarantees **append-only** committed text, *not* that
  `committed_text()` stays a live prefix of `full_text()`. A from-BOS re-decode
  may revise an already-committed region; the base then keeps the old commit
  rather than rewriting it. My first gate asserted the stronger property and
  failed — the contract, not the code, was wrong.

Suite: **101/101 green**. Arch copy untouched and still building (§4.4
coexistence); retiring both moonshine arch dirs stays the separate gated W1
retirement step (Appendix B B16a/B16b).
### 3. Phase 11 Wave W2a — Native Engine Whisper (offline core), closed (2026-08-26)

Third family off the arch layer. New package `src/models/whisper/`
(`graphs/assets/runtime/session` + internal headers), registered via
`audiocpp_add_model` and in the ASR composite; resolves by `whisper` /
`whisper-offline`.

**The frontend is reuse, not a port.** The Phase-9 `engine::audio::MelExtractor`
runs Whisper's `PerUtterance` normalization over the exact slaney filterbank
shipped inside the `.bin` — the "unified MelExtractor" half of W2's scope,
delivered without a second private mel implementation.

**Weights load from the legacy whisper.cpp `.bin`.** `model_specs/whisper.json`
is catalog-only: its 16 packages point at `Whisper-*-GGUF` paths that do not
exist in `audio-cpp/audio.cpp-gguf` (no Whisper directory at all), so no GGUF
is obtainable. `assets.cpp` carries a metadata-only parser plus the
special-token and suppress-list synthesis whisper.cpp derives rather than
stores; payloads stream per tensor at upload. `can_load()` sniffs the `ggml`
magic instead of resolving a spec bundle, which would fail for every real model.

**Result — parity with the arch, same bar as W1a/W1b:**

| | corpus WER | edits | RTF |
|---|---|---|---|
| engine package (W2a) | **4.34783%** | **3/69** | 0.155 |
| arch `asr_e2e_whisper_wer_test` | 4.34783% | 3/69 | 0.047 |

The gate fails if the engine exceeds the arch's 3 edits — parity, not the 10%
structural bound. (RTF is higher than the arch's because W2a rebuilds the
step graph per token; the arch has a static-topology step graph. That is a
W2b optimization, not a numerics difference.)

**The bug the gate caught — remember this one.** `MelExtractor` writes
**mel-major** (element (m,t) at `m*n_frames + t`), but a graph input declared
`ggml_new_tensor_2d(ctx, F32, n_mels, n_frames)` has `n_mels` as its *fastest*
axis and therefore needs **frame-major**. Passing the extractor's buffer
straight through fed the encoder a **transposed spectrogram**. Nothing crashed;
the mel statistics looked perfectly healthy (range ≈ [-0.95, +1.05],
audio-dependent mean); the decoder produced confident, fluent English unrelated
to the audio — "You", "So", "I'm going to go to the next one." — at **94% WER**.
Only the end-to-end numeric gate surfaced it, and only host-side instrumentation
localized it. **Any family wiring MelExtractor into a ggml graph must
transpose.**

Suite: **103/103 green**. Arch copy untouched (§4.4 coexistence).

W2b remains: temperature-fallback ladder + DecodeTelemetry, timestamps,
long-form seek continuation, language detection for multilingual variants,
batched decode, and the static-topology step graph.

### 4. Full fusion review from the parents' code → roadmap v6.0 (2026-08-26)

On request, the merge was reconsidered from scratch: both parents re-read from
code (`audio.cpp@c79e588`, `transcribe.cpp@2102bca`) *before* re-reading the
plans, an independent architecture formed, then compared. Deliverables:
`docs/reports/fusion_review_2026-08-26.md` (the review, every number with its
command), `FUSION_ROADMAP_PLAN.md` **v6.0** (the updated authoritative plan),
V6 **R14** (superseded decisions), tracker phase table + next task.

**The v5 architecture survives.** Engine as the spine, `speech.h` with
transcribe's `struct_size`/ext-kind/exception discipline, bindings as a
retarget of transcribe's six, methodology for every family, migration
invariant, ledger — all confirmed from the code. What changed:

| # | Finding (measured) | Plan change |
|---|---|---|
| 1 | **Phase 10's feature-merges were never executed** — 0 refs to spec-decode in `models/qwen3_asr`, 0 cache-aware in `voxtral_realtime`, 0 presets in `sortformer_diar`; **zero deletions in the whole project** | **Phase 10.5, next**: 5 merges, 5 deletions, first ledger rows |
| 2 | **No ASR runtime layer** — 5 arch KV caches field-identical + 3 engine copies (mine); 15/18 arch `model.cpp` hand-roll argmax; W2a re-implemented `WhisperEmbeddingModule` and wrote a private `.bin` parser | **Phase 11a**: `EncDecKVCache`, decode drivers, `AsrResult`/`AsrLimits`; re-base the 3 ports; A24 dedup lint |
| 3 | **`capi/audiocpp.h` is a third C ABI** — 55 fns, **0 `struct_size`**, default ON; `transcribe.h` (98 fns) OFF; no app consumes either | **Phase 12 pulled forward**; `audiocpp.h` frozen (F14) |
| 4 | **`transcribe-vad*` is a verbatim port of `audio/chunking.cpp`** (its own comment) | delete (§5.7) |
| 5 | **Engine has no input-limits contract**; W2a silently truncates > 30 s (I wrote that) | law **L11**, §4.2 row, A21; fixed in 11a.3 |
| 6 | `GraphExecutor` is single-backend; no `BackendPlan`-style CPU fallback | §4.2 row; 11a.6 scheduler path |
| 7 | v5 kept the transcribe dispatcher under `src/runtime/` forever | **reversed**: Phase 11c deletes `src/runtime/` in full (95 kLOC) |
| 8 | "audio.cpp learns from transcribe" scheduled last (14.3) | **Track M**: a per-phase quota |
| 9 | V6 D2/D3/D4/D14/D15/D18/D19/D23 contradict the built architecture | roadmap Appendix F: superseded |

Honest accounting of this session's own ports: numerically exact, gated at
arch parity, product-registered — and each added a private KV cache and decode
loop, one re-implemented an existing module, one truncates silently. That is
the multiplier Phase 11a exists to stop.

### 5. Phase 10.5, family 1 of 5 — `qwen3_asr` verdict executed (2026-08-26)

The first real consolidation of the fusion: the transcribe.cpp arch for
Qwen3-ASR is gone and the engine package carries everything it had.
Feature-merge commit `89758cf`, deletion commit `9cc5457` (ledger B11).

**Measured before anything was changed (L12):**

- The C ABI could not open the family's only downloadable GGUF at all:
  `asr_e2e_wer_test models/qwen3-asr-0.6b-q8_0.gguf` → `status=5`
  (UNSUPPORTED_ARCH). audio.cpp GGUFs carry `general.architecture =
  "audiocpp"`; `transcribe_model_load_file` only consulted the framework
  registry for non-GGUF paths, so *no* audio.cpp package was reachable
  through `transcribe.h`. Not a qwen3_asr problem — every family.
- The arch could not be run on that GGUF either (it reads
  `stt.qwen3_asr.*` KVs, a transcribe.cpp converter layout that no public
  package uses), so there is no arch-vs-engine number for this family. The
  engine's own first measurement is the baseline, and the ledger says so.
- The engine's batched decode threw `positions shape mismatch: expected
  [1], got [2]` for any batch ≥ 2: `build_with_static_cache_tail_batched`
  rotated `[n_seqs, 1, heads, dim]` heads with `[n_seqs]` positions. The
  bake-off's "engine has batched decode" was structural, never run. Same
  code upstream — a candidate audio.cpp contribution (affects
  `higgs_audio_stt`, `vibevoice_asr` too).
- `request_abort()` was not honoured by `run()`; the arch advertised
  `TRANSCRIBE_FEATURE_CANCELLATION`.

**Merged into the engine package** (all gated):

| Arch feature | Where it lives now |
|---|---|
| 1-gram-lookup speculative decode (`spec_k_drafts`, `supports_spec_decode`) | `NgramLookupDrafter` in `modules/transformers/causal_lm_ops` (family-agnostic, unit-tested); `VerifyGraph` in `models/qwen3_asr/thinker.cpp` over a new shared primitive: multi-token static-cache tail (`QwenDecoderLayerModule::build_with_static_cache_tail`, steps ≥ 1) + multi-row `FastKVSetRows`. Request option `spec_k_drafts` (−1/0/1..8, transcribe convention), `CapabilitySet::supports_speculative_decode`, adapter maps it to `supports_spec_decode` |
| HF BPE parity (`qwen3_asr_bpe_parity`) | `qwen3_asr_bpe_parity_test` on the engine tokenizer, same fixture file: 37 strings + 30 language prefixes HF-exact |
| BCP-47 language hints (`encode_language_prefix`) | `include/engine/models/qwen3_asr/languages.h`; `build_prompt` accepts "en" and "English" |
| Cancellation | `RunControl` polled at stage boundaries and every decode step; `CapabilitySet::supports_cancellation`; the adapter bridges `transcribe_set_abort_callback` onto the progress callback → `TRANSCRIBE_ERR_ABORTED`, and now advertises `TRANSCRIBE_FEATURE_CANCELLATION` for the engine families that poll (qwen3_asr, whisper, moonshine ×2) |

**Fixed on the way:** batched RoPE positions for all Qwen-decoder families;
`transcribe_model_load_file` routes `general.architecture == "audiocpp"`
GGUFs through the framework registry; and `ModelRegistry` auto-detection
now reads a GGUF's own `audiocpp.model_spec.family` instead of taking the
first loader whose `can_load()` says yes (`bf0e68a` — the mis-detection that
decoded a Qwen3-ASR file as Silero VAD; it affected `audiocpp_cli --model`
and the server too, not only the C ABI).

**Gates (new, all green on the pinned `qwen3-asr-0.6b-q8_0.gguf`, sha
`6c44ec2f…`, 1,151,272,416 bytes — fourth row of `fetch_asr_test_model.py`):**

| Gate | Result |
|---|---|
| `qwen3_asr_engine_smoke_test` | corpus WER **2.89855% (2/69)**, RTF 0.81 (CPU); `spec_k_drafts=4`: divergence **0** words on all four fixtures, 2 edits, RTF 1.23 (slower on short clips, exactly as transcribe.cpp documented — the default stays 0); `run_batch(2)` ordered/non-empty; abort honoured |
| `qwen3_asr_bpe_parity_test` | 37 + 30 fixtures, HF-exact |
| `ngram_lookup_drafter_test` | pure-logic drafter semantics |
| `asr_e2e_qwen3_asr_wer_test` | the same fixtures through `transcribe.h` in a product tree that links the family (`-DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=qwen3_asr`): **corpus WER 2.89855% (2/69), RTF 0.763** — the C ABI and the engine session produce the same text |

**Deleted:** `src/runtime/arch/qwen3_asr/` (4,329 LOC) and the six
`tests/transcribe/qwen3_asr_*.cpp` (1,650 LOC) that were copied from
transcribe.cpp but never registered in any CMake file — the "validation
methodology" they represented was already absent from CI.

**What the goldens could not do:** `tests/golden/qwen3_asr/` (2 manifests)
need `scripts/dump_reference_qwen3_asr_author.py` and the HF checkpoint;
neither is in this repo. The manifests stay; running them is Track M work.

**Lessons for families 2–5:**

- Measure the C ABI path and `run_batch` first; two of the four defects
  above were invisible to every existing gate.
- A "loser feature" can be a *contract* (cancellation, BCP-47), not only
  an algorithm. Read the arch's `capabilities.cpp` and the transcribe.h
  fields it sets before deciding what to merge.
- The engine's model sets matter: `core` links no ASR family, so C-ABI
  gates must be guarded on `AUDIOCPP_LINKED_MODELS` and measured in a
  `custom` tree.

### 5b. Phase 10.5, family 3 of 5 — `sortformer_diar` step 2, the feature-merge (2026-08-27)

Full report with every number and command:
`docs/reports/sortformer_diar_engine_port.md`. The short version:

**Step 1's premise did not survive reproduction.** The catalogue's default
package (`diar_streaming_sortformer_4spk-v2.q8_0.gguf`,
`general.architecture = "sortformer"`) was recorded as "a transcribe.cpp-flavoured
GGUF the arch can load". It is **NVIDIA's own layout** (`sortformer.encoder.*`
KVs, `encoder.layers.N.*` / `transformer.layers.N.first_sub_layer.*` tensors);
the arch fails on it exactly like the engine did (`missing KV
stt.sortformer.max_speakers`). Nothing in the repo could open the default
package. "Answer for v2 support" therefore meant a third-layout loader in the
engine — done through a renaming `TensorSource` view over the existing
HF-named weight loader (new generic helper `make_renamed_tensor_source`), a
KV → config reader, a per-layout frontend contract, and a loader wrapper so
registry auto-detection reaches it without a hint.

**The scheduler port.** `src/models/sortformer_diar/streaming.{h,cpp}` is a
host-for-host port of `arch/sortformer/stream.cpp` (FIFO / speaker-cache
update, silence profile, the compression stack, NeMo's chunk windows, preset
table, option + env precedence), graph-agnostic through callbacks. The
whole-window graph was split into stem + body; the chunked path keeps one
stem graph and one body graph per 64-frame capacity tier in a small LRU
instead of rebuilding per chunk (the arch's `low_latency` ran below realtime
for exactly that reason). Packages that ship a streaming operating point run
chunked by default; the HF v1 checkpoints stay whole-window; `stream_preset`
and the per-field `stream_*` options choose explicitly; the typed RUN-slot
ext goes through the adapter.

**Measured (v2, CPU):** single-speaker fixtures → 1 speaker; the 2-speaker
oracle → 2 speakers under all five operating points, DER 0.31–0.34 (the
arch's own probabilities score 0.315–0.334 on this synthetic clip, so the
gate holds parity with the arch, not absolute quality); chunked ==
whole-window to **1.8e-7** on a single-chunk clip; vs the arch on identical
weights (the pinned package re-keyed into the arch's layout) max |Δp| 7.4e-3,
mean 1.7e-4, **0/600 decision flips** on the three published operating points
(1/600 on `low_latency` and `small`, the compression-heavy ones). The residual
is not weight storage (native Q8_0 gives the same size); it is op ordering in
the engine's modules — a Track M `validate.py` item. `low_latency` on 12 s:
16 s CPU (25 chunks) with cached graphs.

**Four defects fixed on the way, none in the family's code:** (1) the
adapter forwarded `language` / `task` / … as request options and every family
that validates request options strictly (`sortformer_diar`, `sense_asr`,
`hviske_asr`, `higgs_audio_stt`) rejected them — unusable through the C ABI
until now; (2) a package's embedded spec outranked the build's, so options
added after conversion (and step 1's `cancellation` capability, which never
reached the v1 package) were rejected by the session that implements them —
the build's copy now wins when present; (3) a GGUF without an embedded spec
was refused a contract outright, a converter-output rule applied to every
GGUF — foreign layouts now resolve like a safetensors checkout; (4) the schema
validator had no `cancellation` capability, so the spec step 1 edited was
invalid and nothing noticed; (5) registry auto-detection still decoded a
foreign GGUF as Silero VAD (`bf0e68a` only covered audio.cpp packages) —
`find_loader` now maps a non-`audiocpp` `general.architecture` through
`family_registry` before probing, so `audiocpp_cli --model <v2.gguf>` works
without `--family`.

**Step 3 scope, from the consumers:** the parakeet multitalker arch includes
the sortformer arch's embedded-diarizer surface and parakeet's arch is the
canonical one, so only the standalone family retires (dispatcher entry, `Arch`
hooks, the ext initializer's move); the embedded core stays with parakeet.
Dropping the dispatcher entry is also what routes the NeMo v2 package to the
engine through the C ABI.

Gates: `sortformer_diar_streaming_engine_test`, `sortformer_diar_scheduler_test`
(core, 111/111), `sortformer_diar_ext_abi_test` (custom tree, OK). Not in the
repo: a formatter — `scripts/ci/clang-format.sh` referenced by AGENTS.md and the
CI workflow does not exist in this tree.

### 6. Phase 10.5, family 2 of 5 — `voxtral_realtime` verdict executed (2026-08-26)

Feature-merge commit `61ab725`, deletion commit `fdaa9a5` (ledger B12).

**What the bake-off row got wrong.** It recorded the arch's edge as
"cache-aware streaming windows". The engine package already had those: a
sliding-window encoder cache (`make_causal_sliding_mask`,
`config.audio.sliding_window`) and a sliding decoder KV
(`stream_decode_cache_steps`), with incremental streaming state for both.
Reading the two implementations instead of the summary, the real gap was the
**typed stream extension** (`include/transcribe/voxtral_realtime.h`) and the
contracts around it:

| Arch feature | State in the engine before | Where it lives now |
|---|---|---|
| `num_delay_tokens` (transcription delay, 1..15 or 30 per the publisher) | Hardwired in **four** places: the offline right-pad, both first-chunk geometry helpers, and the prompt — the last as a literal `0.480` seconds | One validated value threaded through all four; session option `voxtral_realtime.num_delay_tokens`, request option `num_delay_tokens`, fixed for the life of a stream |
| `min_decode_interval_ms` | absent | A partial-emission throttle. Honest difference, recorded in the code: the arch needed it because its partial decode reprocessed the whole buffer; the engine decodes incrementally, so the knob bounds how often the delta is *published*, never how much is decoded, and never the final transcript |
| Cancellation (`TRANSCRIBE_FEATURE_CANCELLATION`) | advertised by the arch, not honoured by the engine | `RunControl` polled at stage boundaries and every decode step; `CapabilitySet::supports_cancellation` |
| The typed extension itself | the adapter answered `accepts_ext_kind` → false for everything | The **ArchAdapter now accepts it**: `adapter_check_stream_ext` validates it pre-clear with the arch's exact rules, `adapter_apply_stream_ext` translates it into request options. Deleting an arch must not delete a public surface |

**A trap worth remembering.** `transcribe_voxtral_realtime_stream_ext_init` —
part of the public C ABI — was *defined inside the arch*. Deleting the arch
would have removed a published symbol while its header kept declaring it. The
initializers of retired families now live in
`src/runtime/transcribe-family-ext.cpp`; check for this in every remaining
retirement (parakeet, moonshine_streaming, sortformer and whisper all define
theirs the same way).

**The defect the gate found — silent truncation, again (L11).** The first
streaming measurement came back one word short: streamed
`" I'm from the cutter lying off the"` against offline
`" I'm from the cutter lying off the coast."`. The streaming loop consumes
fixed-size chunks and stops when the remainder is smaller than one, and
`finalize()` ran that same loop — so a caller who stopped feeding mid-chunk
lost the tail, with no flag and no error. transcribe.cpp's arch never had this
bug: its `stream_finalize` re-ran the whole offline forward. `finalize()` now
zero-extends the buffer and decodes the remainder (the pad is what the model
would see anyway — a stream that ends is silence afterwards, which is exactly
what the delay tokens are defined against). **Second family in a row where the
engine truncated silently and the arch did not** — the W2a Whisper package
does the same for audio > 30 s, and 11a's `AsrLimits` is what fixes that one.

**What the C ABI test found, and why it was worth writing.** It exercises the
extension with a **non-default** delay (4) — and that is the only reason three
further defects surfaced:

1. **The C ABI streaming path was broken for every framework ASR family.** The
   adapter prepared a stream with `build_preparation_request()`, which carries
   no audio contract because a stream has no audio yet — and ASR sessions
   reject that (`prepare() requires an audio contract`). The adapter's own
   comment claimed families "default the sample rate to 16 kHz in that case";
   they do not. It now supplies the contract the C ABI actually guarantees
   (16 kHz mono, length unknown). Nothing had caught this because the only
   streaming C-ABI tests run either a VAD model or a GGUF that still resolved
   to a builtin arch.
2. **The first streaming chunk's length ignored the requested delay.** The
   session asked for a delay-sized first chunk while the frontend padded to
   the model default, so the encoder produced a different number of audio
   tokens than the prompt was built for and the decoder refused the stream.
   Only a non-default delay could expose it — the default matched by
   construction.
3. **`stat()` overflows above 2 GB on MSVC**, so the test reported SKIP on a
   model that was right there. `std::filesystem` throughout now.

**Also fixed on the way:** the batched path passed the delay to the frontend
but not to the prompt (found by reading my own diff, not by a test — the two
must agree or the decoder gets the wrong token budget), and a cancel landing
mid-chunk now reports `TRANSCRIBE_ERR_ABORTED` from
`stream_feed`/`stream_finalize` rather than a generic backend error.

**Gates** (pinned `voxtral-mini-4b-realtime-2602-q4_k.gguf`, sha `8cafef18…`,
3,097,662,432 bytes — the smallest catalogued package for this 4B family):

| Gate | Result |
|---|---|
| `voxtral_realtime_engine_smoke_test` | corpus WER **2.89855% (2/69)**, RTF 5.49; streamed-vs-offline divergence **0 word(s)**; `num_delay_tokens=6` reproduces the default exactly and 0/16/31 are refused before audio moves; abort honoured |
| `voxtral_realtime_delay_test` | the publisher's validated set (1..15, 30) pinned as a unit test, no model needed |
| `voxtral_realtime_ext_abi_test` | the typed extension through `transcribe.h`: probe accepts STREAM+kind, rejects RUN and foreign kinds; out-of-range values return `INVALID_ARG` with the stream still IDLE; a valid extension begins/feeds/finalizes |

**Deleted:** `src/runtime/arch/voxtral_realtime/` (4,996 LOC) and the
never-registered `tests/transcribe/voxtral_realtime_real_smoke.cpp`.

### 7. Phase 11 Wave W1a — Native Engine Moonshine (offline), closed
First family migration of the arch layer onto the engine framework
(FUSION_ROADMAP_PLAN §8, §4.4 steps 5–7). New package `src/models/moonshine/`
(`graphs/assets/runtime/session` + internal headers in
`include/engine/models/moonshine/`): graph topology ported numerics-identically
from `src/runtime/arch/moonshine/`; weights via `core::BackendWeightStore`
(`SharedWeightRegistry` stays active); tokenizer via `TokenizerHub`
(`load_tokenizer_from_gguf`); abort/progress via `RunControl`.
Registered `audiocpp_add_model(moonshine ...)` + ASR composite; CLI/server/
WebUI/C ABI can now load `--family moonshine` GGUFs directly.
Findings: (1) pinned moonshine GGUFs carry **no** embedded config/tokenizer
sidecars — hparams live in `stt.*` metadata; fixed `model_specs/moonshine.json`
sources block (previous required mappings could never have loaded — a latent
Phase-6 defect surfaced by this wave). (2) Engine tensor shapes are logical
(PyTorch-order); `BackendWeightStore` reverses into ggml `ne`. (3)
TokenizerHub GGUF-BPE decode emits raw `▁` — the package post-processes to
spaces (hub change deferred). Gate `moonshine_engine_smoke_test`: registry
load by id + 4 LibriSpeech fixtures offline (WER ≤ 10% structural bound;
measured **1/69 edits == arch baseline**), ordered `run_batch`,
`request_abort()` unwinds `run()`. Arch copy untouched and still green.
Suite: 96/96 green.

### 8. Streaming ASR text validation — NEXT #1, closed
`tests/asr_stream_text_wer_test.cpp` + CTest gate `asr_stream_text_wer_test`:
streams the four LibriSpeech fixtures into **moonshine-streaming-tiny Q8_0**
(48 MB, MIT, `handy-computer/moonshine-streaming-tiny-gguf` — the exact model
the previous session hoped existed; arch already in
`src/runtime/arch/moonshine_streaming`) through the public C ABI with
odd-sized ~100–400 ms feeds, reads the final transcript from
`transcribe_stream_get_text().full_text`, and gates: streamed corpus WER
≤ 10%, offline corpus WER ≤ 10%, streamed-vs-offline divergence ≤ 3 words.

Measured (CPU): **streamed 4.35% == offline 4.35% (3/69: FORWARDED→VOTED +
"I AM"→"I'M"), divergence 0 words, streamed RTF 0.55.** Each fixture runs
offline then streaming on ONE session, so run/stream mode switching and
`stream_reset` are proven with real text. `scripts/fetch_asr_test_model.py`
became a pinned-model table (both gate models; `--only streaming` selects);
shared scoring lives in `tests/asr_test_text.h`. Report updated:
`docs/reports/asr_e2e_wer_gate.md`.

### 9. The three "environment/asset" failures — NEXT #2, all fixed, none was assets
- `model_spec_system_test` + `fun_asr_nano_assets_test`: model-spec
  resolution walks UP from the cwd, so they only ever passed from build
  trees inside the repo. Fixed with `WORKING_DIRECTORY` registrations.
- `server_model_installer_test` exposed **two real product bugs**, both fixed
  in `app/server/model_installer.{h,cpp}`:
  1. **`AUDIOCPP_PYTHON` never worked on Windows.** `cmd /c` strips the
     first+last quote when the command string starts with a quoted program
     path — proven with an isolated `std::system` repro. All helper commands
     now go through `run_shell_command()` (wraps the line in one extra quote
     pair; no-op for the unquoted default). The CTest registration passes
     CMake's `Python3_EXECUTABLE` via `AUDIOCPP_PYTHON` (on this machine bare
     `python` is the Microsoft Store stub).
  2. **Teardown raced detached workers.** A worker terminated by process
     exit inside `CreateProcess` leaves a permanently suspended child
     `cmd.exe` pinning inherited pipes — observed as CTest waiting 22 min on
     a 15 s test; the same race leaked ~40 `audiocpp-model-installer-*` temp
     dirs (all cleaned). Workers are now tracked + cancelled + joined in
     `~ModelInstaller` (first D7-teardown application to app-layer code).
     Test now passes in 2.7 s.

### 10. `asr_standalone_gguf_test` — NEXT #3, closed by correction
Filed for three sessions as "needs citrinet+hviske GGUFs". It does not: the
fixtures are synthetic (dummy safetensors → GGUF), and the failure was the
same cwd spec-resolution defect. `WORKING_DIRECTORY` registration fixed it;
the old download-and-pin recommendation is withdrawn. (A real citrinet/hviske
WER gate would be new, optional work — the plan's §5 Phase-5 corpus item.)

### 11. `scaled_dot_product_attention_test` skips without CUDA
It exists to pin the CUDA SDPA lowerings (R10) and hard-required a CUDA
device, failing CPU-only builds. Now probes `list_backend_devices()` and
skips (exit 2, `SKIP_RETURN_CODE 2`); stays a hard gate on CUDA builds.

### 12. Performance pass on transcribe.cpp runtime families
Closed the last depthwise-1D-conv im2col sites in the runtime and a couple
of decode/frontend hotspots, all with env-override kill switches and
numerical parity:
- **Granite + Granite-NAR encoders** (`src/runtime/arch/{granite,granite_nar}/encoder.cpp`):
  in-block depthwise conv now uses `ggml_conv_2d_dw_direct` instead of
  `conv_1d_dw_f32`'s B==1 im2col path (15x scratch expansion avoided; ~2.2 s
  of a 29 s CPU clip). Matches parakeet/canary/etc.
- **SAN-M FSMN branch** (`src/runtime/sanm/sanm.cpp`): B==1 single-shot path
  now uses direct dw conv (was im2col); B>1 already was. 11x expansion
  avoided for kernel=11. Affects sensevoice + fun_asr_nano arch.
- **Parakeet decoder** (`src/runtime/arch/parakeet/decoder.cpp`): frame-batched
  joint window (`JointGraphBatch`) fuses the RNN-T/TDT joint op across frames
  instead of per-frame dispatch.
- **Mel frontend** (`src/runtime/transcribe-mel.cpp`): nonzero mel-band skip
  (`fb_begin_`/`fb_end_`) across both FFT + scalar paths; scalar fallback
  threaded (`run_threaded`). Bit-exact.
This was the uncommitted `WORKSPACE` work, committed in two passes.

## Test suite state
**CPU build** (`sp_bridge`: full set, unified ABI + arches) — **58 tests,
0 failures, 100% green** (was 57 tests / 5 failures at session start).
Skips: `scaled_dot_product_attention_test` (no CUDA device — by design),
`parakeet_golden_transcription_test` / `parakeet_streaming_transcription_test`
(their own model-download skip contract). Newly green this session:
`model_spec_system_test`, `fun_asr_nano_assets_test`,
`server_model_installer_test`, `asr_standalone_gguf_test`, and the new
`asr_stream_text_wer_test`.

**CUDA build** (`sp_cuda`: core set, sm_89) — rebuilt with this session's
fixes; suite result recorded below when the run lands (expected: the same
three shared failures gone; WER gates not registered there — ABI/arches OFF).

## COMPLETED IN RECENT COMMITS
1. **Phase 1: Allocator Hardening & Memory Safety (`e760def`)**:
   - `BackendWeightStore`: Added `kMetadataPoolBudget = 16 MB` cap for `no_alloc=true` metadata pool.
   - `WavLMEncoder`: Switched to `ggml_gallocr` buffer reuse (18x peak memory reduction on 35s clips).
   - `Qwen3-TTS`: Added sampling runaway guard (`top_p = 0.8, temp = 0.8` defaults, `max_new_tokens = 1024` ceiling).
   - `DeepFilterNet2`: Lowered `segment_threshold` to `segment_samples` (48000) for overlap-add chunking.
2. **Phase 2: Toolchain Modernization & Build Provenance (`06962fb`)**:
   - Integrated `ccache` compiler launcher auto-detection into `scripts/build_windows.ps1` and `scripts/build_windows_hip.ps1`.
   - Configured `cmake/transcribe-build-info.h.in` and Windows `cmake/transcribe-version.rc.in` for `transcribe.dll`.
   - Implemented strong symbols `kTranscribeBuildId`, `transcribe_build_id()`, and `transcribe_version()`.
3. **Phase 3: Native Long-Form VAD Chunk Planning & Re-stitching (`5dcc27c`)**:
   - Implemented `src/runtime/transcribe-vad.h/.cpp` (`vad::plan`, `vad::params_present`, `vad::effective_mode`).
   - Implemented `src/runtime/transcribe-vad-integrate.h/.cpp` (`vad::detect_speech`, `vad::offset_chunk_results`, `vad::rollback_to`, `vad::rebuild_full_text`, `vad::run_with_vad`).
   - Added public C ABI `transcribe_vad` and `transcribe_free_vad`.
   - Added unit tests `tests/unittests/vad_plan_unit.cpp` and `tests/unittests/vad_merge_unit.cpp`.
4. **Phase 4: Shared Weight Registry, Sortformer v2 Package & Batched Offline Decoders (`13abbd7`, `fe1a307`, `4f0ae95`)**:
   - **`FunASR Nano` Packed QKV & Gate/Up Weights (`13abbd7`)**: Added `load_packed_rows` to `src/models/fun_asr_nano/decoder.cpp`, row-packing Q, K, V and Gate, Up projections at load time and switching decoder to `PackedQKV` + `PackedGateUp` for fused SwiGLU. Drops layer matmuls from 7 to 5 (197 to 113 in step graph).
   - **Process-Wide `SharedWeightRegistry` & `ScopedWeightShareKey` (`fe1a307`)**: Created `include/engine/framework/core/shared_weight_registry.h` and wired `BackendWeightStore` (`upload_shared`, `bind_to_shared`, `upload_pending_into`).
   - **`Sortformer` Diarization v2 Package (`fe1a307`)**: Added `sortformer_diar_4spk_v2_q8_0` package to `model_specs/sortformer_diar.json`.
   - **Offline Batched ASR Decoders (`4f0ae95`)**:
     - `IOfflineVoiceTaskSession::run_batch` interface added in `include/engine/framework/runtime/session.h`.
     - `Qwen3-ASR`: `DecodeGraphBatched` and `generate_batch` in `src/models/qwen3_asr/thinker.cpp` & `session.cpp`.
     - `Voxtral Realtime`: parallel frontends and compute graph resets in `src/models/voxtral_realtime/session.cpp`.
     - `Citrinet ASR`: batched CTC graph in `src/models/citrinet_asr/runtime.cpp` & `session.cpp`.
     - `VibeVoice ASR`: `VibeVoiceDecoderCachedStepGraphBatched` in `src/models/vibevoice_asr/text_decoder.cpp` & `session.cpp`.
     - `Higgs Audio STT`: `DecodeGraphBatched` in `src/models/higgs_audio_stt/text_decoder.cpp` & `session.cpp`.
5. **Phase 5: Universal `audiocpp` C ABI Subsystem & Progress Reporting (`b9f53ab`)**:
   - Integrated full C ABI surface (`capi/include/audiocpp.h` and `capi/src/audiocpp_capi.cpp`) with 46 exported C APIs covering all 14 audio tasks.
   - Built monolithic shared library `audiocpp.dll` / `libaudiocpp.so` with hidden internal GGML symbols and embedded Windows `VS_VERSION_INFO` resource.
   - Implemented `ProgressInfo`, `ProgressCallback`, and `ProgressCanceled` in `include/engine/framework/runtime/session.h`.
   - Implemented `RuntimeSessionBase::set_progress_callback` and `emit_progress` in `src/framework/runtime/session_base.cpp` and wired into `audiocpp_set_progress_callback`.
   - Integrated embedded asset subsystem (`include/engine/framework/assets/embedded.h` & `src/framework/assets/embedded.cpp`).
   - Added unit test suites `capi_option_number_test`, `capi_session_options_test`, and `capi_enum_sync_test` (56/56 tests passing 100% green).
6. **Phase 6: Whisper GPU Cleanup, Arch Sync, and Model Spec Catalog Integration**:
   - **GPU Buffer Cleanup across 18 Arches**: Integrated `cleanup_gpu` lambda across all `src/runtime/arch/*/model.cpp` files (Whisper, Moonshine, Parakeet, Canary, Voxtral, SenseVoice, etc.), eliminating GPU/KV-cache memory leaks across repeated runs on Windows.
   - **Whisper `bin_load.cpp` Token Tables**: Added separate English and Multilingual suppress-token tables and `synthesize_bin_suppress_tokens()`.
   - **Parakeet Batched Joint Window**: Added multi-frame greedy joint decoding graph amortization.
   - **Model Spec Catalog Additions**: Added schema-v1 catalog specifications for `whisper.json` (16 packages: tiny, base, small, medium, large-v3, large-v3-turbo, and .en variants), `moonshine.json` (6 packages: tiny, base, small in q8_0/f16), and `moonshine_streaming.json` (2 packages: streaming-tiny in q8_0/f16).
   - **Verified**: Full test suite passes 100% green (56/56 tests).

## NEXT (highest value first)
1. **Phase 11 Wave W1 retirement step** (optional, gated): with W1a + W1b both
   green, `src/runtime/arch/{moonshine,moonshine_streaming}/` can be deleted as
   their own commits with Appendix B rows B16a/B16b. Both engine packages now
   reproduce the arch numerics exactly, so the ledger evidence is in hand.
2. **Phase 11 Wave W2 — Whisper Universal Family**: tiny…large-v3-turbo, `.en`
   variants, legacy `.bin` loader, suppress tables, temperature-fallback ladder
   + DecodeTelemetry; unified MelExtractor + TokenizerHub; validate the 16
   packages in `model_specs/whisper.json`.
   - **Check `whisper.json`'s gguf source against the actual pinned GGUF
     before porting** — but scope it correctly. A `files:` block on a gguf
     source is the **norm and is correct**: `audiocpp_gguf` embeds sidecars by
     default and fails conversion if it cannot find them, so 54+ catalog specs
     legitimately declare one. The defect only hits families pinned to
     **third-party GGUFs this pipeline did not produce** — that is what bit
     `moonshine.json` (W1a) and `moonshine_streaming.json` (W1b), both of
     which pull from `handy-computer/moonshine-*-gguf`. Decide per family with
     `audiocpp_gguf --inspect <gguf>`: `embedded_sidecars=false` means drop the
     `files` block; `true` means leave the spec alone. Do **not** rewrite
     `files` blocks across the catalog on the pattern alone.
3. **Zero-Dependency Language Bindings (`dynload`)**:
   - Finalize single-artifact shared build (`SPEECH_SHARED_EMBED=ON`) and zero-dependency dynload bindings (Rust, Python ctypes, TypeScript koffi, Swift).
4. **HuggingFace 5.x Long-form Seek Continuation**:
   - Verify long-form chunked streaming continuation across multi-minute audio files with synthetic silence and early `<|t|>` termination guards.
5. **Parakeet TDT & Moonshine CLI / Server End-to-End Testing**:
   - Validate CLI invocation with newly cataloged `--model whisper`, `--model moonshine`, `--model moonshine_streaming`, and `--model parakeet_tdt`.

## LEFT TO DO (small)
- [ ] Formalize root `.gitmodules` so `git submodule update` works for the 3 embedded repos.
- [ ] `LNK4217` warnings: arch static libs see `TRANSCRIBE_API` as `dllimport`
      for same-DLL symbols. Benign but noisy.
- [ ] `external/ggml` mixed line endings (1145 CRLF blobs vs `eol=lf`):
      unchanged posture — leave alone; sync matches committed state on
      Windows; a Linux run would produce LF + a large diff.

## Notes / decisions made this session
- **"Missing assets" is now 5-for-5 wrong in this repo.** flashsr (real
  regression), the WER gap (unfetched-by-design), and this session's three
  (two registration defects + product bugs). Treat the label as unverified
  until a failure is reproduced and root-caused once.
- A test that passes only when the build tree is inside the repo is a
  registration defect, not an environment issue: model-spec resolution walks
  up from the cwd, so declare `WORKING_DIRECTORY` when a test reads the
  production catalog.
- `std::system` on Windows: if the command string starts with a quoted
  program path, `cmd /c` strips the first and last quote on the line. Wrap
  the whole command in one extra pair of quotes (see `run_shell_command`).
  Any code that shell-quotes a configurable program path into `std::system`
  has this bug.
- Detached threads that spawn subprocesses are a teardown hazard beyond
  leaks: a thread terminated by `ExitProcess` inside `CreateProcess` leaves
  a permanently *suspended* child pinning inherited handles — visible only
  as a downstream consumer (CTest) hanging on a pipe. Track and join.
- The model-pinning pattern is now a table (`fetch_asr_test_model.py`); the
  next pinned model is one dataclass row, not a new script.
