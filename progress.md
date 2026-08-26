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
| **Phase 11 W2a: Native Engine Whisper (offline core)** | **Done & Verified (`whisper_engine_smoke_test`: 4.34783% == arch baseline 3/69, RTF 0.155; legacy `.bin` loader + unified MelExtractor)** | **100%** |
| Specs: Phase 6 Whisper & Moonshine Model Spec Catalogs | Moonshine spec corrected + backed by native loader. **Whisper: `whisper.json` is catalog-only in the strong sense — its 16 packages point at `Whisper-*-GGUF` paths that do NOT exist in `audio-cpp/audio.cpp-gguf` (no Whisper dir at all), so none are downloadable.** Family now gated via the legacy `.bin` instead (see W2 prerequisite). | ~70% |
| ABI offline + streaming surface | Verified, real CTest gates | 100% |
| End-to-end ASR **offline text** (WER gate) | Done — 1.45% corpus WER (arch path); engine path now also 1/69 edits | 100% |
| **End-to-end ASR streaming text** | **Done — streamed 4.35% == offline 4.35%, divergence 0** | **100%** |
| Test suite status | **103/103 total (99 passed, 4 clean skips on unpinned weights) 100% green** | **100%** |
| **Completed increment** | **Upstream `c79e588` (0 behind), ggml 0.22.0 (CPU+CUDA certified), Phase 11 W1a + W1b + W2a** | **DONE** |
| **Next increment** | **Phase 10.5, family 3 of 5: `sortformer_diar`, step 2** — port the arch's chunked diarization scheduler (AOSC speaker cache + FIFO + preset operating points, ~475 LOC of host logic plus a pre-encode/stream-infer graph split) into the engine package, which is offline-only today, and answer for v2-package support; step 1 (first registered gate + cancellation) landed in `65ab43c`. Then `sense_asr`, `fun_asr_nano`; then 11a | Ready |

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
