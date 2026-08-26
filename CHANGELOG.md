# Changelog

All notable changes to `speech.cpp` — the fork of [audio.cpp](https://github.com/0xShug0/audio.cpp)
that is absorbing [transcribe.cpp](https://github.com/) per
`TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md`.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are the work-session dates recorded in the plan.

## [Unreleased]

### Added

- **Phase 11 W2 prerequisite: Whisper model pinned and arch baseline locked**:
  - **`model_specs/whisper.json` is catalog-only in a stronger sense than the docs said.** Its 16 packages point at `Whisper-*-GGUF` paths under `audio-cpp/audio.cpp-gguf` — and that repo contains **no Whisper directory at all** (67 entries, none Whisper). None of the catalogued packages are downloadable, so porting the family first would have meant writing ~3.5k lines against a gate that could only ever skip.
  - **Third pinned model**: whisper tiny.en as the canonical legacy **ggml `.bin`** from `ggerganov/whisper.cpp` (77,704,715 bytes, sha256 `921e4cf8…`), added as one dataclass row in `scripts/fetch_asr_test_model.py`. Deliberately the `.bin` and not a GGUF: it is the format that actually exists for this family, and it exercises `src/runtime/arch/whisper/bin_load.cpp` — the legacy loader W2 has to carry over regardless.
  - **New gate `asr_e2e_whisper_wer_test`** reuses the existing `asr_e2e_wer_test` binary (`transcribe_open` sniffs the model, so no family hint is needed) against the vendored LibriSpeech fixtures; skips (exit 2) while the model is absent, like its siblings. **Locked baseline: corpus WER 4.34783% (3/69 over 4 utterances), RTF 0.047** — the "== arch baseline" bar the native engine Whisper package must match, the same standard W1a (1/69) and W1b (3/69, divergence 0) were held to.
  - **Removed a dead pin found while verifying the table**: the `librispeech-test-clean-500w.tar.gz` entry pointed at an HF dataset that is now gated (HTTP 401). Nothing referenced it — the four wav/txt fixture pairs and `manifest.jsonl` are vendored under `assets/asr_validation/librispeech/` and tracked in git — but because the error was unhandled, the documented no-argument `uv run scripts/fetch_asr_test_model.py` **failed for everyone**. It now exits 0 with all three models verifying.
  - Suite: **102/102 green**.

- **Phase 11 Wave W1b Completed: Native Engine Moonshine-Streaming Package (`src/models/moonshine_streaming/`)**:
  - **Second family migrated off the transcribe arch layer onto the engine framework.** Moonshine-Streaming is now reachable as a first-class product family through `ModelRegistry` by canonical id (`moonshine_streaming`) or alias (`moonshine-streaming`), so CLI, HTTP server, WebUI and the shipped `audiocpp` C ABI can load it directly — previously only via the default-off `transcribe.dll`.
  - Package layout mirrors W1a: `src/models/moonshine_streaming/{graphs,assets,runtime,session}.cpp` with internal headers under `include/engine/models/moonshine_streaming/`. Ported numerics-identically from `src/runtime/arch/moonshine_streaming/`: the time-domain frontend (CMVN → `asinh(exp(log_k)·x)` → linear+SiLU → two causal stride-2 convs), encoder blocks with **per-layer sliding-window masks and no RoPE**, the adapter (absolute-frame `pos_emb` get_rows + optional proj), the **untied `lm_head`**, and vanilla decoder LayerNorms. The three conformer helpers the arch borrowed (`conv_1d_f32` / `layer_norm` / `named`) are reimplemented locally so the package carries **no `src/runtime/` dependency**.
  - **Streaming lifecycle is base-owned.** The session derives from `engine::runtime::StreamingSessionBase` and implements only `on_start_stream` / `on_process_audio_chunk` / `on_finalize` / `on_reset` plus a pure `validate_chunk` pre-clear hook. Per the W1b design map's preferred option, the family drives `update_text(full_text)` on every partial decode and lets the base's **`STABLE_PREFIX` policy (agreement_n=3)** choose the commit boundary, instead of re-deriving a longest-common-prefix boundary per family — one commit policy across every family.
  - **Incremental pipeline**: each feed encodes a window with L/R context, adapters the emit slice at **absolute frame offsets**, projects its cross-K/V, appends to host committed buffers, trims unreachable PCM, then — throttled by `moonshine_streaming.min_decode_interval_ms` (default 240 ms) — re-runs the AR decoder from BOS over the extended cross-KV. Offline `run()` reuses the same graphs, so streamed-vs-offline divergence is measurable on **one** session.
  - **Latent Phase-6 spec defect fixed** — the same one W1a found in `moonshine.json`: `model_specs/moonshine_streaming.json`'s `gguf` source required `config.json` / `tokenizer.json` sidecars the pinned packages do not carry (`audiocpp_gguf --inspect` reports `embedded_sidecars=false`), so that source **could never resolve**. The gguf source is now tensors-only; the safetensors source keeps its sidecars, which are real for an HF checkout.
  - **Gate `moonshine_streaming_engine_smoke_test`** (CTest #90): registry load by id + alias, asserts the family advertises ASR **offline and streaming**, then per fixture transcribes offline and streams the same audio on the **same session** in odd, non-frame-aligned chunks (1601/3203/6397/2477/4801 samples ≈ 100–400 ms) with the decode throttle at 0 so the partial path runs hard. Also asserts monotonic revision, the base's **append-only committed-text** contract, the Active/Finished/Idle lifecycle, `reset()` clearing state, and a pre-requested abort unwinding the stream.
  - **Measured: corpus WER offline 4.34783% (3/69), streamed 4.34783% (3/69), streamed-vs-offline divergence 0 words** — exactly the arch-side `asr_stream_text_wer_test` baseline, i.e. the engine package **reproduces the arch numerics** rather than merely meeting the ≤10% bound.
  - Suite: **101/101 green** (was 100/100). The arch copy `src/runtime/arch/moonshine_streaming/` is untouched and still builds per the §4.4 coexistence requirement; retiring both moonshine arch directories remains the separate gated W1 retirement step (Appendix B rows B16a/B16b).

- **ggml 0.22.0 certified on CUDA** — closing the caveat left by the bump: the `sp_cuda` build (sm_89, CUDA 13.3, `MODEL_SET=core`) compiles clean against `36da5713` with patch 0007's CUDA entry points, and its **57/57 CTest suite passes** — including `scaled_dot_product_attention_test`, which only executes on a real CUDA device (it skips on CPU-only builds). The 0.22.0 CUDA kernel churn and our CUDA fork delta are therefore both verified, not just the CPU path.

- **Dual Parentage doctrine + `scripts/sync-deps.sh` dependency routine**:
  - **`speech.cpp` is equally a child of `audio.cpp` and of `transcribe.cpp`.** Forking audio.cpp was a convenience — it was the larger tree to start from — **not** a statement of precedence. An improvement in `NairoDorian/transcribe.cpp` is exactly as authoritative as one in `0xShug0/audio.cpp`. This was drifting because only audio.cpp has a git `upstream` remote, so only it produces a merge-base and a "N behind" number; transcribe.cpp drift is invisible to git here, and the docs reinforced it by calling transcribe.cpp a "merge source" / "read-only reference tree". Recorded in `AGENTS.md` § "Dual Parentage" and tracker **Operating Rule 7**; the tracker's Master Key and Target Repository lines were corrected.
  - **`scripts/sync-deps.sh`** — read-only drift report across all three sources: audio.cpp via the `upstream` remote, transcribe.cpp via the sibling checkout, and ggml compared **both** against upstream HEAD **and against parent transcribe.cpp's pin**. `--fetch` also fast-forwards the siblings. It never pulls/merges/re-vendors speech.cpp; it prints the exact remediation command for each stale item so adoption keeps an audit trail. Run it regularly and **always before a release state**.

- **ggml bumped `8c63e709` (0.20.2) → `36da5713` (0.22.0)** — matching parent transcribe.cpp (its commit `2102bca`):
  - Re-vendored via `scripts/sync-ggml.sh`; **all 7 tracked patches apply**, 88 paths changed. The large deletion count is upstream splitting `src/ggml-metal/ggml-metal.metal` (11,820 lines) into `src/ggml-metal/kernels/` — restructuring, not lost content.
  - **Patch 0005 (concat fast paths) rebased.** Upstream rewrote `ggml_compute_forward_concat_any` into a row-wise `memcpy` loop — it has **converged on most of this fork delta**, which existed to replace 0.20.2's scalar element walk. The `len` local the patch's byte math used is gone; byte counts now go through `ggml_row_size`, which is block-aware and also removes a latent over-count for quantized types in our own delta. `ggml_compute_forward_concat_f32` is untouched upstream (still scalar), so its dim-0 row and dim-1 plane fast paths keep their full win. The remaining `concat_any` win is memcpy **count** only — flagged in the patch header to benchmark before the next bump.
  - **Patch 0007 (CUDA trim-pools / clear-graph) regenerated.** The previous file was hand-written and had **never been round-trip verified**: one hunk declared a 21-line new side for a 35-line body, and the file lacked a trailing newline, so `git apply` rejected it as corrupt. Its content was nonetheless present in the vendored tree — **the delta was one sync away from silent loss**, exactly the failure mode `patches/ggml/` exists to prevent. It is also the only patch whose targets carry CRLF blobs upstream, which is why it never got a clean run.
  - `external/ggml/UPSTREAM` gains a **PIN HISTORY** block recording the bump and its per-patch cost (the sync script carries hand-written prose across syncs).
  - **API drift is purely additive**: `ggml.h` gains `ggml_clamp_inplace` and `ggml_rope_set_offset`, `ggml-cuda.h` gains `ggml_backend_cuda_allreduce_tensor`; no removals or signature changes across `ggml.h` / `ggml-backend.h` / `ggml-cpu.h` / `ggml-alloc.h`. **No engine source changes were needed.**
  - **Verified** on `build-cpu-core`: clean build (444 targets), **CTest 100/100** (96 passed, 4 documented skips) unchanged from the 0.20.2 baseline — including `moonshine_engine_smoke_test` (engine WER gate) and `ggml_fork_ops_cpu_test` (patch-0003 fork kernels vs independent references at 1 and 4 threads); `lint_teardown` green at its `src/runtime` scope. **Reproducibility certified**: a re-sync reports `0 path(s) changed` — `sync + patches == committed tree`, exactly.
  - **Not covered**: CUDA/HIP/Metal/Vulkan are not built in `cpu-core`, so patch 0007's CUDA entry points and the 0.22.0 CUDA kernel churn remain compile-unverified pending an `sp_cuda` run.

- **Upstream Synchronization (`0xShug0/audio.cpp:main@c79e588`) — history reconciled, 0 behind**:
  - **Root cause of the recurring "6 commits behind"**: the prior sync (`9b34fd2`) copied upstream content without recording a merge, so the merge-base stayed pinned at `62735ea`. `git` therefore kept re-reporting commits that were already applied verbatim. Resolved by a `-s ours` reconciliation merge that advances the base to `c79e588` while leaving the tree untouched; the merge commit carries a per-commit disposition ledger for all six. **`git rev-list --left-right --count HEAD...upstream/main` now reports `63  0`.** Future upstream syncs must merge rather than content-copy.
  - **Audited, not assumed**: each of the six was checked against the tree by content, not by commit subject. `288a271` (`--list-devices`, PR #299) and `d25ffac` (out-of-span chunk metadata, PR #301) were confirmed **already present** — `print_backend_devices` at `backend.h:33` / `backend.cpp:213` with both call sites, and the warning-and-drop path at `chunking.cpp:652` with `test_chunk_speech_metadata_merge_drops_outside_spans`. Only four were real work.
  - **`supertonic` voice preset honored as a plain request option** (cherry-pick `90659b1`, upstream `4ec485d`, PR #302): `SupertonicSession::generation_options_from_request` now falls back to the `voice` / `supertonic.voice` request options when no `cached_voice_id` is supplied, so the WebUI voice picker (which sends the preset via `webui/configs/model_params.json`) takes effect. Unknown ids stay rejected downstream by the voice-style loader.
  - **IndexTTS2 HIP decode: F16 KV cache and F16 conv weight default** (cherry-pick `3682698`, upstream `d03b957`, PR #305): `decode_cache_type()` now returns `GGML_TYPE_F16` on HIP as it already did on CUDA, so ggml's flash-attention kernels stop converting the whole K/V cache per decode step (`convert_unary` had dominated the HIP kernel profile); and `IndexTTS2Session` promotes `Native` conv weight storage to `F16` on HIP, where the conv/im2col paths support it and measure faster. CUDA, CPU, Vulkan and Metal behavior is unchanged.
  - **Tag-driven prebuilt release pipeline, adapted for this fork** (`a775463`, upstream `c79e588`, PR #286): added `.github/workflows/release.yml` (9 jobs — CPU / CUDA 12.4 + 13.3 / Vulkan / Metal across macOS, Ubuntu, Windows) and `docs/RELEASING.md`. Releases are semantic-version and tag-driven: a pushed `v*` tag or a manual Actions run builds and attaches binaries; pushes to `main` never trigger a release. **Fork adaptation**: all seven upload-artifact names emit `speech-<tag>-…` instead of `audio-<tag>-…` (the release job archives by artifact basename via `assets/*.zip` globs, so the rename is self-consistent); prose retargeted; `docs/RELEASING.md` gained a provenance note and points at the `speech_cli` / `speechcpp_cli` ALIAS targets. `.gitignore` takes upstream's `audio-vtest-*` and adds `speech-vtest-*`. Prerequisites were verified against this tree rather than assumed — the `audiocpp_cli` / `audiocpp_server` / `audiocpp_gguf` targets still exist, `scripts/build_windows.ps1` already defines the `windows-cpu-release` and `windows-vulkan-release` presets the workflow invokes, and the YAML parses.
  - **Deliberately not taken**: upstream `c6805de` ("Highlight release 0.7 model testing") edits an audio.cpp README banner announcing audio.cpp's own 0.7 model testing. Our README is a speech.cpp rewrite carrying no such banner, the patch does not apply, and the content is upstream project news rather than a fork concern. Recorded as N/A in the merge ledger so it is not re-triaged next session.
  - **Verification**: `build-cpu-core` (`MODEL_SET=core`, Release, MSVC) rebuilt — `engine_model_supertonic` and `engine_model_index_tts2` both compile clean. **CTest 100/100 passing** (96 passed, 4 documented skips), unchanged from baseline. Note that at `MODEL_SET=core` both packages are OBJECT libraries with no consumer, so neither fix is runtime-exercised by this suite; the IndexTTS2 change is HIP-only and needs an AMD GPU to observe.

- **Upstream Synchronization (`0xShug0/audio.cpp:main@d25ffac`) & `speech.cpp` Unified Aliases**:
  - **Unified Device Enumeration (`engine::core::print_backend_devices`)**: Extracted device printing logic into core framework (`include/engine/framework/core/backend.h` & `src/framework/core/backend.cpp`). Added `--list-devices` support to `audiocpp_server` and refactored CLI `--list-devices` dispatch (PR #299 upstream).
  - **`speech_cli` / `speech_server` / `speech_gguf` Primary Executable Aliases**: Configured CMake ALIAS targets `speech_cli`, `speechcpp_cli`, `speech_server`, `speechcpp_server`, `speech_gguf`, `speechcpp_gguf` to standardize product executables under `speech.cpp`.
  - **CTest Gates for Device Enumeration**: Added `speech_cli_list_devices`, `speech_server_list_devices`, `audiocpp_cli_list_devices`, and `audiocpp_server_list_devices` asserting regex pattern `available_devices=[0-9]+`.
  - **Long-Form Audio Chunk Speech Metadata Resilience**: In `src/framework/audio/chunking.cpp` (`append_chunk_speech_metadata`), replaced aborting exception on out-of-span segments/turns with warning diagnostics and graceful segment drop (PR #301 upstream). Verified via `test_chunk_speech_metadata_merge_drops_outside_spans` in `audio_chunking_test`.
  - **Full Suite Status**: **100/100 CTest targets passing 100% green** on `build-cpu-core` (96 passed, 4 clean skips on unpinned weights).

- **Phase 11 Started (Wave W1a): Native Engine Moonshine ASR Package (`src/models/moonshine/`)**:
  - **First architectural family migrated off the transcribe arch layer onto the engine framework** (FUSION_ROADMAP_PLAN §8 Phase 11 / §4.4 migration invariant, steps 5–7). Moonshine is now reachable as a first-class product family through `ModelRegistry`, so CLI, HTTP server, WebUI and the shipped `audiocpp` C ABI can load it by canonical id or alias (`moonshine`, `moonshine-offline`) — previously only via the separate default-off `transcribe.dll`.
  - Package layout: `src/models/moonshine/{graphs,assets,runtime,session}.cpp` with internal headers under `include/engine/models/moonshine/`. Encoder/decoder graph topology, partial-RoPE handling, head-dim padding policy and greedy decode loop are ported numerics-identically from `src/runtime/arch/moonshine/`; weight upload goes through `core::BackendWeightStore` (keeping `SharedWeightRegistry` activation), tokenization through the Phase-9 `TokenizerHub` (`load_tokenizer_from_gguf`), cancellation/progress through `RunControl`.
  - Hparams/tokenizer are read from the `stt.*` / `tokenizer.ggml.*` GGUF metadata keys; `model_specs/moonshine.json` sources block corrected (the pinned packages carry no embedded config/tokenizer sidecars — the previous required file mappings could never have loaded).
  - **Engine-path gate `moonshine_engine_smoke_test`** (CTest #85): loads the pinned `moonshine-tiny-Q8_0.gguf` through a `ModelRegistry`, runs the four LibriSpeech fixtures offline, gates corpus WER ≤ 10%, asserts ordered `run_batch()` results and that `request_abort()` unwinds `run()`. Measured: **corpus WER 1.449% (1/69 edits) — identical to the arch-side baseline**; transcripts text-identical to the `transcribe.dll` path.
  - Arch copy `src/runtime/arch/moonshine/` remains buildable and green (§4.4 coexistence requirement); retirement is deferred to its own gated step with an Appendix B ledger row.
  - Full suite: **96/96 CTest targets green** on `build-cpu-core` (92 passed, 4 clean skips on unpinned weights).

- **Phase 10 Completed: Attention & Conformer Module Fusion & Overlap Resolution (§10.1–§10.4)**:
  - **Unified SAN-M Encoder Subsystem**: Consolidated `src/framework/modules/speech_encoders/sanm.{cpp,h}` and `src/runtime/sanm/sanm.{cpp,h}` with shared sinusoidal positional encoding table generator (`build_sinusoidal_pe`, `make_sinusoidal_positions`), fused QKV linear projections, optional attention/conv padding masks, and direct depthwise conv lowering with `TRANSCRIBE_CONV_DIRECT_DW` / `TRANSCRIBE_CONV_NO_DIRECT_DW` toggle support. Verified with `fun_asr_nano_sanm_probe`, `fun_asr_nano_assets_test`, and `fun_asr_nano_frontend_probe`.
  - **Shaw Block-Local Relative Attention**: Implemented `engine::modules::build_shaw_block_attention` in `include/engine/framework/modules/attention/shaw_attention.h` and `src/framework/modules/attention/common_relative_attention.cpp`; delegated `transcribe::granite_conformer::shaw_block_attn` to unified engine module.
  - **Unified Causal LM Transformer Operations**: Implemented `include/engine/framework/modules/transformers/causal_lm_ops.h` and `src/framework/modules/transformers/causal_lm_ops.cpp`, unifying KV-cache context bucket sizing (`pick_kv_cache_context`), chunked prefill mask trapezoid computation (`fill_prefill_chunk_mask`), and prefill chunk size calculation (`prefill_chunk_size`).
  - **Overlap Bake-Off & Resolution Certification**: Published `docs/reports/overlap_bakeoff.md` resolving all 6 architectural overlaps under V6 Decision R3 without feature loss.
  - **Verification**: Full CTest suite 95/95 test targets 100% green (91 passed, 4 clean skips on unpinned weights).
- **Phase 9 Completed: Unified Mel & Tokenizer Subsystems (§9.1–§9.4)**:
  - **`engine::audio::MelExtractor` & `FrontendSpec`**: Implemented single-source unified frontend subsystem supporting `MelSpectrogram` (Whisper, Parakeet, GigaAM, NeMo, SenseVoice), `KaldiFbank` (80-dim HTK filterbanks), and `RawPcm` (Moonshine passthrough). Full support for Hann periodic/symmetric, Hamming, and custom windows, Slaney filterbank area normalization, non-zero support span indexing (`fb_begin_`/`fb_end_`), SIMD/threaded STFT, and `PerFeature`, `PerUtterance`, `Global`, and `None` normalization modes.
  - **Model Spec Schema v2 Frontend Block**: Added `"frontend"` configuration validation to `engine::model_spec::validate_spec` with seamless backward compatibility for schema version 1.
  - **`engine::text::TokenizerHub`**: Unified tokenization subsystem supporting Unigram, BPE, Byte-Level BPE, Tiktoken raw bytes, SentencePiece, and HuggingFace JSON tokenizers with bidirectional `encode()` and `decode()`, $O(1)$ piece lookup, control token classification, and GPT-2 byte mapping inversion.
  - **Neural Codec Interface (`IAudioCodec`)**: Added unified `encode()` / `decode()` abstraction across neural codecs with `SharedWeightRegistry` integration.
  - **Parity & Contract Verification**: Added 3 unit tests (`frontend_contract_test`, `frontend_parity_test`, `tokenizer_parity_test`). Full CTest suite 95/95 test targets 100% green (91 passed, 4 clean skips on unpinned weights).
- **Phase 8 Completed: Unified Streaming Contracts & Exception Boundary (§8.1–§8.8)**:
  - **`StreamingSessionBase` Lifecycle Engine**: Implemented 4-state lifecycle machine (`Idle` $\to$ `Active` $\to$ `Finished` / `Failed`, `reset()`), monotonic revision counter, commitment policies (`Auto`, `OnFinalize`, `StablePrefix` with $N$-agreement), and pre-clear validation hooks.
  - **Unified `RunControl`**: Thread-safe, lockless abort polling (`poll_abort()`, `request_abort()`, `reset_abort()`), and synchronized progress emission throwing `ProgressCanceled`. Integrated into `RuntimeSessionBase` and `IVoiceTaskSession`.
  - **Unified `StreamChunker`**: PCM sample re-blocking, exact chunk boundary alignment, contiguous start sample cursor tracking, and zero-padded tail flushing. Replaced duplicate implementation in adapter.
  - **Telemetry Extension**: Added `DecodeTelemetry` structure and wired `decode_telemetry` vector into `TaskResult`.
  - **Universal C ABI Exception Containment**: Wrapped 100% of exported C API functions in `audiocpp_capi.cpp` (device discovery, model info/capabilities, WAV I/O, artifacts, and all `audiocpp_free_*` functions) in `try { ... } catch (...) { ... }` exception guards.
  - **Verification**: Added 4 unit tests (`run_control_unit_test`, `stream_chunker_unit_test`, `streaming_session_base_unit_test`, `capi_exception_containment_test`). 92/92 test targets green.
- **Phase 7 Activated: Process-Wide `SharedWeightRegistry` & `ScopedWeightShareKey` (§7.4)**:
  Eliminated redundant weight uploads across concurrent sessions with process-wide reference-counted
  GPU/host weight buffer sharing, dropping per-session overhead from ~3 GB to ~34 MB.
  Integrated telemetry counters (`hit_count()`, `miss_count()`, `conflict_count()`), wrapped all model
  load and session creation call sites (`audiocpp_load_model_ex`, CLI, server `runtime.cpp`, workflow runner),
  and verified via `test_shared_weight_vram`.
- **Phase 7 Activated: Batched ASR Decode & Pre-Clear Validation in `ArchAdapter` (§7.5)**:
  Wired native multi-utterance lockstep decoding across all 16 adapter architecture slots via
  `IOfflineVoiceTaskSession::run_batch` forwarding, preserving batch contracts (mirroring slot 0 in
  session result, filling per-utterance results). Added C ABI batch entry points `audiocpp_asr_batch` and
  `audiocpp_free_text_batch`. Added pre-clear validation hooks `stream_validate` and `run_validate` across
  all adapter architecture entries, preventing pre-clear destruction of existing results upon invalid arguments.
- **Phase 7 Activated: Unified Family Registry v1 (`family_registry.h` & `family_registry.cpp`) (§7.8)**:
  Created single-source static catalog mapping 46 canonical voice model families, aliases, GGUF arch identifiers,
  and specification paths. Verified via `family_registry_unit` in reporting mode.
- **Phase 7 Ported: Comprehensive Regression Test Suite & Fixtures (§7.1)**:
  Ported 51 translation units from `transcribe.cpp/tests/` into `tests/transcribe/`, along with 87 golden fixtures
  across 19 directories, 36 tolerance manifests, and 13 GGUF binary test fixtures.
- **Universal `audiocpp` C ABI Subsystem & Shared Library (`audiocpp.dll` / `libaudiocpp.so`)**:
  Exposes 48 exported C APIs covering all 14 audio tasks (TTS, ASR, VAD, Diarization,
  Separation, Alignment, Voice Conversion, Denoise, Super-Resolution) with opaque handles,
  strict C++ exception containment, and hidden internal GGML symbols.
- **Universal Multi-Task Progress Reporting & Synchronous Cancellation**:
  Added `ProgressInfo`, `ProgressCallback`, and `ProgressCanceled` in `session.h` and
  `session_base.cpp`, wired directly into `audiocpp_set_progress_callback(model, cb, user_data)`.
- **Phase 3 Native Long-Form VAD Chunk Planning & Re-stitching (`vad::plan`)**:
  Greedy gap merging, boundary padding, timestamp offset re-stitching, transactional rollback,
  and standalone `transcribe_vad` C ABI.
- **Phase 1 Allocator Hardening & Memory Safety**:
  Added `kMetadataPoolBudget = 16 MB` cap for `no_alloc=true` metadata pool in `BackendWeightStore`,
  switched `WavLMEncoder` to `ggml_gallocr` buffer reuse (18x peak memory reduction), added
  sampling runaway guard to `Qwen3-TTS`, and lowered `DeepFilterNet2` overlap-add threshold.
- **Phase 2 Toolchain Modernization & Build Provenance**:
  Integrated `ccache` compiler launcher auto-detection into Windows build scripts, configured
  3-outlet build provenance (`transcribe-build-info`, `audiocpp_build_info`, Windows `.rc` resources).
- **Whisper & Moonshine Engine Model Spec Catalogs (Catalog only — no engine loader yet; Phase 11)**:
  Added comprehensive schema-v1 catalog specifications for `model_specs/whisper.json`
  (16 packages covering tiny, base, small, medium, large-v3, large-v3-turbo, and English-only variants),
  `model_specs/moonshine.json` (6 packages: tiny, base, small), and
  `model_specs/moonshine_streaming.json` (2 streaming packages).
- **GPU Buffer Cleanup & Memory Leak Prevention**:
  Integrated `cleanup_gpu` RAII/exit cleanup across all 18 arch runtime sessions
  (`src/runtime/arch/*/model.cpp`), eliminating GPU memory and KV cache accumulation on Windows
  during repeated model inference runs.

### Fixed

- **Phase 7 Defect D1 Remediation: GGUF Sniff Architecture Precedence Collision**:
  In `src/runtime/transcribe.cpp`, routed model architecture sniffing to `transcribe::adapter_find_arch(family.c_str())`,
  ensuring adapter architecture table lookup takes precedence over engine fallbacks for overlapping architectures
  (`qwen3_asr`, `voxtral_realtime`, `moss`).
- **Phase 7 Defect D2 Remediation: Pre-Clear Result Destruction on Invalid Parameters**:
  Added `stream_validate` and `run_validate` hooks across all 16 `adapter_archs` entries in
  `src/runtime/transcribe-arch-adapter.cpp`, guaranteeing that invalid session parameters or unrecognized
  extensions fail before clearing existing transcription or diarization results.
- **`AUDIOCPP_PYTHON` never worked on Windows: `cmd.exe` mangled every quoted
  program path.** `ModelInstaller` builds helper commands as
  `"<python>" -u "<script>" ... > "<log>" 2>&1` and runs them through
  `std::system`, i.e. `cmd /c <string>`. When that string *begins* with a
  quote, cmd strips the first and the LAST quote character on the line,
  mis-tokenizing everything between ("The filename, directory name, or volume
  label syntax is incorrect") — which is why `python_command()` returns the
  bare word `python` by default but the `AUDIOCPP_PYTHON` override (shell-
  quoted) could never have worked. All helper invocations now go through
  `run_shell_command()`, which wraps the whole line in one extra pair of
  quotes on Windows so cmd's strip is a no-op (verified by an isolated
  `std::system` reproduction: unwrapped fails, wrapped runs, and the wrap is
  also a no-op for the unquoted default). Found because
  `server_model_installer_test` needs a real interpreter and `python` on this
  machine is the Microsoft Store stub; the test's CTest registration now
  passes CMake's `Python3_EXECUTABLE` through `AUDIOCPP_PYTHON` when one is
  found, which is what finally exercised the override path.
- **`ModelInstaller` teardown raced its own workers; on Windows the race
  parks a child `cmd.exe` in a permanently suspended state.** Install and
  size-scan workers were detached `std::thread`s, so nothing stopped the
  process from exiting while a worker sat inside `std::system` —
  `ExitProcess` then terminates the worker at an arbitrary instruction, and a
  thread killed inside `CreateProcess` (after the child exists, before its
  initial thread is resumed) leaves a suspended orphan that never runs and
  never exits. Observed live: `server_model_installer_test` finished its
  asserts, but a suspended orphan `cmd.exe` kept the inherited stdout pipe
  open and CTest waited 22 minutes for a 15-second test (the same race also
  leaked ~40 `audiocpp-model-installer-*` temp directories across past runs,
  because `~State`'s `remove_all(job_root)` ran while workers still had
  redirection files open in it). Workers are now tracked and joined:
  `~ModelInstaller` writes the cancellation marker for any active job (the
  preparation helper polls it and exits early) and joins every worker before
  the job directory is cleaned up; finished workers are reaped on each new
  launch so the tracking list stays small. The detached-thread pattern is
  exactly the teardown-discipline class the plan's D7/safe_* doctrine exists
  for — this is its first application to audio.cpp's app-layer code.
- **Three "environment/asset" test failures were really test-infrastructure
  defects — none needed any asset.** `model_spec_system_test`,
  `fun_asr_nano_assets_test`, and `asr_standalone_gguf_test` all resolve
  production model specs by walking UP from the current working directory
  (`discover_workspace_model_spec`), which finds `model_specs/<family>.json`
  only when the build tree lives inside the repo; the documented scratch
  builds live outside it, so all three failed with "model contract spec not
  found". They are now registered with `WORKING_DIRECTORY` at the source
  root. `asr_standalone_gguf_test` in particular had been filed as "needs
  citrinet+hviske assets" for three sessions — it builds synthetic fixtures
  and needs no model download at all (progress.md's NEXT item recommending a
  download-and-pin for it was based on that misdiagnosis).
- **`scaled_dot_product_attention_test` failed CPU-only builds instead of
  skipping.** Every case in it builds CUDA-configured graphs on a CUDA
  device — it exists to pin the CUDA SDPA lowerings (R10). It now probes
  `list_backend_devices()` and skips (exit 2, `SKIP_RETURN_CODE 2`) when no
  CUDA device is registered, keeping it a hard gate on CUDA builds while a
  CPU-only build — a supported configuration — stays green.

- **The 0.20.2 convergence silently demoted every framework convolution to F16
  activations; restored as `patches/ggml/0006-conv-im2col-in-weight-type.patch`.**
  Upstream's `ggml_conv_1d`/`ggml_conv_1d_dw`/`ggml_conv_2d` lower to im2col
  with dst type `F16` unless the weight is BF16; the audio.cpp fork lowers to
  the *weight's* type, and every framework audio model loads conv weights as
  F32. This is the fourth dropped fork delta (after the broadcast, concat and
  flash-mask ones) and — like 0004 — it carried **no `MINITTS_` marker**, so
  the R10 marker sweep could not have found it: the marker grep is necessary,
  not sufficient. It was found numerically: `flashsr_utility_test` checks the
  FlashSR decoder against onnxruntime-generated fixtures with a 2e-4 bound and
  failed at 1.7e-3 max / 3.3e-4 mean **identically on CPU and CUDA** (the
  signature of a backend-independent graph-builder delta, previously misfiled
  as "missing assets", then as "numeric tolerance"), while the same test built
  from audio.cpp's fork tree passes on the same machine and compiler.
  `flashsr_utility_test` is green on both backends for the first time since
  the convergence; no other test moved. `ggml_conv_2d_dw` is deliberately
  untouched — the fork kept upstream's hardcoded F16 there.

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

### Performance

- **Direct depthwise-conv dispatch across all transcribe.php runtime encoder
  families.** Granite and Granite-NAR encoders (`src/runtime/arch/{granite,granite_nar}/encoder.cpp`)
  and the SAN-M FSMN branch (`src/runtime/sanm/sanm.cpp`) were the only runtime
  sites still routed through `conv_1d_dw_f32`'s B==1 im2col path (a k-wide
  scratch expansion + degenerate per-channel matmul). Switched to
  `ggml_conv_2d_dw_direct`, matching parakeet/canary/etc., gated by the
  `TRANSCRIBE_CONV_NO_DIRECT_DW` kill switch. Granite measured 2.2 s of a 29 s
  CPU encode; SAN-M avoids the same 11x expansion (kernel=11) for the
  common single-utterance path. Pointwise convs were already direct (`mul_mat`).
- **Frame-batched Parakeet RNN-T/TDT joint window** (`src/runtime/arch/parakeet/decoder.cpp`).
  The greedy decode loop fuses the joint-network op across consecutive frames
  via `JointGraphBatch` instead of dispatching once per frame, with a
  `joint_batch_check` serial-vs-batched parity verifier.
- **Mel frontend nonzero-band skip + threaded scalar path** (`src/runtime/transcribe-mel.cpp`).
  `fb_begin_`/`fb_end_` bounds skip trailing-zero columns in the filterbank
  sum across both the FFT and scalar paths; the no-BLAS scalar fallback is now
  threaded (`run_threaded`). Bit-exact to the reference.

### Added

- **Streaming ASR text is validated end to end:
  `tests/asr_stream_text_wer_test.cpp`.** The streaming counterpart of the
  WER gate, closing the last unvalidated path in "speech.cpp transcribes":
  it streams the four LibriSpeech fixtures into a real streaming-family
  model through the public C ABI (begin → odd-sized ~100–400 ms feeds →
  finalize → `transcribe_stream_get_text().full_text`, asserting the
  documented finalize contract that tentative text is empty) and scores the
  STREAMED text against the references, next to the same model's offline
  text and the divergence between the two. Model: **moonshine-streaming-tiny
  Q8_0** (48 MB, MIT, `handy-computer/moonshine-streaming-tiny-gguf`,
  transcribe.cpp-validated at 4.52% offline / 4.54% streamed on full
  test-clean; the arch was already in `src/runtime/arch/moonshine_streaming`)
  — exactly the model the previous session's NEXT item hoped existed.
  Measured baseline (CPU, 2026-08-20): **streamed corpus WER 4.35% = offline
  corpus WER 4.35% (3/69: FORWARDED→VOTED plus "I AM"→"I'M"), divergence 0
  words**, streamed RTF 0.55. Gates: both WERs ≤ 10%, divergence ≤ 3 words —
  same weights, so divergence is a streaming-path defect by construction.
  Every fixture runs offline then streaming on ONE session, so run/stream
  mode switching and `stream_reset` are proven with real text, not just
  segment counts. Skips (exit 2) while the model is absent, like its
  sibling. `scripts/fetch_asr_test_model.py` now carries a pinned-model
  table and fetches both gate models (`--only streaming` selects one);
  `tests/asr_test_text.h` holds the normalization/edit-distance/fixture
  scanning shared by both gates (extracted from asr_e2e_wer_test, the same
  move abi_test_wav.h made). Report: `docs/reports/asr_e2e_wer_gate.md`.
- **End-to-end ASR transcription is validated: `tests/asr_e2e_wer_test.cpp`.**
  The first test in the merged tree that checks *text* rather than plumbing:
  it loads a real GGUF ASR model through the public C ABI (`transcribe_open`),
  transcribes the four in-tree LibriSpeech fixtures, and gates **corpus WER**
  (total word edits over total reference words, LibriSpeech-style
  normalization) at ≤10%. Measured baseline: **1.45%** — 1 edit in 69 words
  (`FORWARDED`→`VOTED`), consistent with the model's published 4.6% full
  test-clean WER — at RTF 0.033 on CPU. Links only `transcribe.dll`, like a
  language binding. Registered under CTest whenever the unified ABI + arch
  families are built; skips (exit 2) while the model file is absent. See
  `docs/reports/asr_e2e_wer_gate.md`.
- **`scripts/fetch_asr_test_model.py`** — fetches the gate's model:
  moonshine-tiny Q8_0 (34 MB, MIT; Useful Sensors' model, ported and
  WER-validated by transcribe.cpp; the smallest validated GGUF whose arch is
  compiled into `src/runtime/arch/`). sha256-pinned to the repo's LFS oid;
  a mismatched download is deleted, never installed. `models/` stays
  gitignored — the gate is a download, not a vendored asset. stdlib-only, no
  venv needed (unlike `fetch_silero_vad.py`, which needs torch).
- **`tests/abi_test_wav.h`** — the minimal RIFF/WAVE reader that lived inside
  `abi_stream_hello.cpp`, extracted header-only so `asr_e2e_wer_test` shares
  it without linking anything beyond the C ABI.
- **R11 in `TO_DO_UNIFY_AND_IMPROVEMENT_PLAN_V6.md`** — records the
  end-to-end validation, the fourth dropped fork delta, and the correction to
  R10: unmarked behavioural deltas exist, so the two audits that close the
  class are numeric golden gates with implementation-independent references
  and a hunk-level diff of the fork tree against its upstream base.
- **`CMakePresets.json`** — one-command configure/build/test for the three
  validated configurations: `cpu-full` (the CPU test baseline incl. the ABI
  and WER gates), `cpu-core` (lean fast-iteration, supported per R10.5), and
  `cuda` (the GPU-certification config). `cmake --preset cpu-full && cmake
  --build --preset cpu-full && ctest --preset cpu-full` from a vcvars64
  environment. Closes the "optionally add CMakePresets.json" item.
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

- **`scripts/sync-ggml.sh` normalizes patches to LF before applying them.** The
  stage it patches is LF (`git archive` honors `eol=lf`), and a patch file
  freshly written on Windows can carry CRs in its hunk lines that make
  context matching fail — patch 0006's first sync run died exactly this way
  while the patch itself was correct (re-materializing it from the index
  fixed it). Tracked patches are CR-free in the object store, so stripping
  CRs from the working copy is always content-preserving. Reproducibility
  re-verified end to end with 0006 in the stack: sync + patches 0001–0006 →
  `git diff --ignore-cr-at-eol -- external/ggml` empty.
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

- ~~**`flashsr_utility_test` fails identically on CPU and CUDA**~~ — resolved:
  it was the dropped conv-im2col fork delta, restored as patch 0006 (see
  Fixed). The "identical on both backends" symptom was the tell.
- ~~**No in-tree ASR model / end-to-end transcription unvalidated**~~ —
  resolved by `asr_e2e_wer_test` + `scripts/fetch_asr_test_model.py` (see
  Added). The model remains out-of-tree by design; the gate skips until
  fetched.
- **`supertonic_vector_convnext_exp_test` asserts a ≥5% wall-clock speedup**, so
  it fails when another build or test suite is running on the same machine. Run
  it uncontended. (Contrast `audio_dsp_test`, whose flakiness was a seeding bug
  and was fixed rather than documented.)
- **Per-head flash-attention masks remain CPU-only.** The framework fallback is
  correct and currently both faster and more accurate on CUDA, so this is not
  urgent — but if ggml ever fixes `nb32` indexing in its flash kernels, the
  CPU-only gate in `common_relative_attention.cpp` should be revisited against
  fresh measurements rather than assumed obsolete.
- ~~**Streaming ASR text is still unvalidated end to end.**~~ — resolved by
  `asr_stream_text_wer_test` + the pinned moonshine-streaming-tiny Q8_0 (see
  Added): streamed text scores 4.35% corpus WER, word-identical to the same
  model's offline text on this corpus.
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
