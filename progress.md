# Progress — Unified_Audio.cpp (speech.cpp ggml fork) merge & improve

Status snapshot: **Upstream audio.cpp main fully reconciled at `c79e588` — 63 ahead, 0 behind (the recurring "6 behind" was a lagging merge-base, now fixed). Phase 1 Allocator Hardening applied. Phase 2 Toolchain Modernization & Build Provenance landed. Phase 3 Native Long-Form VAD Chunk Planning & Public C ABI integrated. Phase 4 Process-Wide SharedWeightRegistry, Sortformer v2 Diarization package, Batched Offline ASR Decoders (Qwen3-ASR, Voxtral Realtime, Citrinet, VibeVoice, Higgs Audio), and universal audiocpp C ABI subsystem + progress callbacks fully implemented and verified. **ggml bumped to `36da5713` (v0.22.0)** to match parent transcribe.cpp, with the 7-patch stack rebased and reproducibility certified. **Phase 11 W1b closed: native engine Moonshine-Streaming reproduces the arch baseline exactly (streamed 4.348% == offline 4.348%, divergence 0).** All 101 CPU core tests passing 100% green on the new ggml; CUDA suite 57/57 green. Master architectural blueprint established in [FUSION_ROADMAP_PLAN.md](FUSION_ROADMAP_PLAN.md).** Date: 2026-08-26

## Repo layout (important, non-obvious)
`Unified_Audio.cpp/` is a **plain container directory with no git repo of its
own**. It holds five independent repositories (three primary, two hardened reference trees):

| Folder | Role |
|---|---|
| `speech.cpp/` | the active development repo (the ggml/audio.cpp fork). **All merge work, and this log, live here.** Remote: `NairoDorian/speech.cpp`, upstream `0xShug0/audio.cpp`. |
| `audio.cpp/` | **parent** — read from, never committed to (pulled to `c79e588`, 2026-08-26). Has a git `upstream` remote here, so it is the only source that yields a merge-base. |
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
| Merge: Upstream audio.cpp main synchronization (`c79e588`) | Done — 0 behind, merge-base reconciled, all 6 dispositioned | 100% |
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
| Specs: Phase 6 Whisper & Moonshine Model Spec Catalogs | Moonshine spec corrected + backed by native loader. **Whisper: `whisper.json` is catalog-only in the strong sense — its 16 packages point at `Whisper-*-GGUF` paths that do NOT exist in `audio-cpp/audio.cpp-gguf` (no Whisper dir at all), so none are downloadable.** Family now gated via the legacy `.bin` instead (see W2 prerequisite). | ~70% |
| ABI offline + streaming surface | Verified, real CTest gates | 100% |
| End-to-end ASR **offline text** (WER gate) | Done — 1.45% corpus WER (arch path); engine path now also 1/69 edits | 100% |
| **End-to-end ASR streaming text** | **Done — streamed 4.35% == offline 4.35%, divergence 0** | **100%** |
| Test suite status | **102/102 total (98 passed, 4 clean skips on unpinned weights) 100% green** | **100%** |
| **Completed increment** | **Upstream reconciliation `c79e588` (0 behind), ggml 0.22.0 (CPU+CUDA certified), Phase 11 W1a & W1b** | **DONE** |
| **Next increment** | **Phase 11 Wave W2: Whisper Universal Family** (W1 retirement step first, if taking it) | Ready |

## DONE this session (plan R12 records all of it)

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
### 3. Phase 11 Wave W1a — Native Engine Moonshine (offline), closed
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

### 4. Streaming ASR text validation — NEXT #1, closed
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

### 5. The three "environment/asset" failures — NEXT #2, all fixed, none was assets
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

### 6. `asr_standalone_gguf_test` — NEXT #3, closed by correction
Filed for three sessions as "needs citrinet+hviske GGUFs". It does not: the
fixtures are synthetic (dummy safetensors → GGUF), and the failure was the
same cwd spec-resolution defect. `WORKING_DIRECTORY` registration fixed it;
the old download-and-pin recommendation is withdrawn. (A real citrinet/hviske
WER gate would be new, optional work — the plan's §5 Phase-5 corpus item.)

### 7. `scaled_dot_product_attention_test` skips without CUDA
It exists to pin the CUDA SDPA lowerings (R10) and hard-required a CUDA
device, failing CPU-only builds. Now probes `list_backend_devices()` and
skips (exit 2, `SKIP_RETURN_CODE 2`); stays a hard gate on CUDA builds.

### 8. Performance pass on transcribe.cpp runtime families
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
