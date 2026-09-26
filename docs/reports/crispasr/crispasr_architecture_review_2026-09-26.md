# CrispASR — architecture & engineering-quality review

> **Date:** 2026-09-26 · **Tree read:** `CrispASR@dfcdacae` (v0.8.37), cloned to `../CrispASR` next to `audio.cpp/` and `transcribe.cpp/`
> **Status:** analysis only — no code was changed in any repo.
> **Companion notes:** [`crispasr_model_catalog_2026-09-26.md`](crispasr_model_catalog_2026-09-26.md) (every family + overlap with speech.cpp) and [`../fusion_rethink_2026-09-26.md`](../fusion_rethink_2026-09-26.md) (what this means for speech.cpp's plan).
> Items marked **[inf]** are inference, not measurement.

---

## 1. What CrispASR is

- A **whisper.cpp fork**, renamed wholesale (`whisper` → `crispasr`, including the find/replace artefact "ggml-org/crispasr" in UPSTREAM.md). Upstream history was squashed; the first commit is 2026-03-29 ("whisper.cpp fork with Cohere Transcribe Conformer support").
- **6,931 commits in six months.** Monthly: Apr 1006 · May 1273 · Jun 1469 · Jul 1883 · Aug 671 · Sep 622. Peak days hit ~200 commits.
- **Overwhelmingly agent-authored:** ~3,840 commits carry `Co-Authored-By: Claude …`. There is one human maintainer (CrispStrobe, plus a "crispasr integration" identity). The workflow is the same kind as ours, at roughly 5× our commit rate.
- **License:** MIT, inherited from whisper.cpp. Absorbing its code is legally clean.
  - Avoid linking espeak-ng. It is dlopen'd by default; `-DCRISPASR_WITH_ESPEAK_NG=ON` makes the binary a GPL-3 derivative.
  - libgomp is GPL-3 with the runtime exception.
- **Size:**
  - `src/` is ~386 kLOC (144 .cpp + 169 .h).
  - `src/core/` is 58 kLOC (152 files, mostly header-only).
  - `examples/cli` is 54 kLOC.
  - ~135 Python converters in `models/`.
  - 113 PyTorch reference-dump modules in `tools/reference_backends/`.
- **Scope:** ~131 backend names — 36 ASR runtimes, ~36 TTS, plus VAD, diarization, LID (audio and text), punctuation, truecasing, text MT, source separation, 9 music-analysis models, codecs, watermarking and a llama.cpp chat ABI. Full list in the catalog note.

---

## 2. Layering

```
examples/cli/   crispasr binary: cli.cpp, crispasr_run.cpp (5.2k LOC dispatch),
                85 crispasr_backend_<x>.cpp adapters behind class CrispasrBackend,
                server (OpenAI /v1/audio/*, WebSocket, Wyoming), VAD, output writers,
                model manager (-m auto)
src/<model>.{h,cpp}   one C-style runtime per model (<model>_init/_transcribe/_free),
                      each its own static lib (124 add_library in src/CMakeLists.txt)
src/crispasr_c_api.cpp   13.5k-LOC session ABI (crispasr_session_*, crispasr_stream_*, mic, …)
src/core/        shared primitives, namespace core_* (mel, attention, ffn, gguf_loader, bpe, …)
ggml/            git submodule → CrispStrobe/ggml fork (@2f5a80d2, "v0.23" bump on 2026-09-07)
```

### 2.1 Backend interface (CLI side)

`class CrispasrBackend` (`examples/cli/crispasr_backend.h`, 26 KB):

- **Required:** `name()`, `capabilities()` (a 30-bit `CAP_*` mask), `init(const whisper_params&)`, `transcribe(samples, n, t_offset_cs, params)`, `shutdown()`.
- **Optional hooks:** `encode_slice` / `decode_slice` / `repair_slice`, `synthesize` / `synthesize_streaming`, `speech_to_speech`, `translate_text`, `detect_language`, `transcribe_streaming`, `create_realtime_session → {append, reset}`, `prefers_vad`, `vad_slice_cap_seconds`, `warmup`.
- **Factory:** a string if-chain with 83 `crispasr_make_*` arms.
- **Smell:** every call takes `whisper_params`, the CLI flag struct (704 lines, ~172 fields). The model interface is coupled to the CLI.

### 2.2 The central flaw: every backend is implemented twice

`src/crispasr_c_api.cpp` does **not** use the CLI adapters. Those adapters compile into the executable, not into the library. The ABI re-implements dispatch with **778 `CA_HAVE_*` ifdefs** and ~144 `s->backend == …` if-chains for open, free, synthesize, rate and set_voice. Measured and confirmed.

- Capabilities are copied into the library by a generator (`backend_caps_table.h`), with a CI drift check.
- Their own LEARNINGS § "The session ABI re-implements every backend" lists the resulting bugs:
  - hard-coded 24 kHz rates;
  - a missing rate arm that reported 0 Hz;
  - aliases accepted by the CLI but rejected by the ABI.
- Issue #335: **113 architecture strings** were known to the CLI and not to the ABI. That drift is what `core/arch_backend_map.h` was created to stop.
- 31 lessons in their "Multi-surface wiring" group are debt from this design.

**Lesson for us:** speech.cpp's engine already has the right shape here. `IVoiceModelLoader` → `ILoadedVoiceModel` → `I*VoiceTaskSession` is one registry that the CLI, server and ABI all consume. Never regress to per-surface dispatch. This is also the strongest argument **against** importing CrispASR's code wholesale.

### 2.3 No central engine

- 86 model files each call `ggml_backend_sched_new`, and 36 call `ggml_gallocr_new`.
- KV caches are per model (53 files).
- Graph caching is re-derived per model. The same cached-graph and gallocr traps recur in 5+ lessons. Our own `speechcpp-cached-graph-inputs` bug (Voxtral-RT) is lesson #6 in their list.
- **504 distinct `getenv(...)` knobs** (measured). The 82 KB `docs/environment-variables.md` is the de facto config system.

### 2.4 `src/core/` — what is actually shared

- **Load-bearing headers**, by number of includers: `gguf_loader.h` 126 · `attention.h` 47 · `bpe.h` 44 · `ffn.h` 42 · `mel.h` 38 · `beam_decode.h` 23 · `conv.h` 22 · `fft.h` 20 · `greedy_decode.h` 9.
- **Incomplete sharing:**
  - Only `gemma4_e2b` uses `greedy_decode.h`; the other audio-LLMs keep private decode loops.
  - About 14–28 local FFTs remain.
  - Whisper deliberately keeps its own mel, attention and loader as the "reference path".
  - `lid_cld3.cpp` and `text_lid_dispatch.cpp` are byte-identical copies of `crisp_lid/`.
- **"core" is partly a dumping ground:** `omnivoice_*`, `chatterbox_*`, `cosyvoice3_*`, `qwen3_stream.h`, `hojo_asr_frames.h` and similar are model-specific tables.

### 2.5 Public ABI and bindings

- `include/crispasr.h`: the whisper.h API renamed.
- `include/crispasr_session.h`: generated from `CA_EXPORT` markers, 293 functions (with some duplicate prototypes).
- `include/crispasr_chat.h`.
- **Versioning:** `crispasr_c_api_version()` returns `"0.7.0"`. Growth is additive only, and some params structs carry an `abi_version`. There is no struct_size discipline and no ABI checker.
- **Bindings:** Rust (sys + safe crate), Python (ctypes), Dart/Flutter, Go, Java, JS (emscripten, with 6 MB prebuilt JS committed), Ruby, C#.

### 2.6 Runtime details worth knowing

| Area | Finding |
|---|---|
| Weights | Two-pass loader; mmap only during load, then `tensor_set` + unmap (no persistent zero-copy on GPU). |
| Backends | CI builds CUDA, Metal, Vulkan, HIP, OpenBLAS and `GGML_BACKEND_DL`. GPU correctness is proven on Kaggle P100/T4 notebooks, not in CI. `gpu_backend_pref.h` replaces `ggml_backend_init_best()`, which always prefers CUDA. |
| Workarounds | `load_weights_split` forces LLM halves to CPU to avoid sched NaNs (20 uses). |
| Concurrency | The server serializes on one mutex. `--server-workers N` runs N model copies. No batched multi-stream inference. |
| Streaming | Real incremental sessions exist for nemotron, qwen3 (r2t2 prefix rollback), vibevoice-streaming, xasr and moonshine-streaming. **voxtral4b is chunk-and-retranscribe** (their TODO.md says so), and the default `transcribe_streaming` fakes partials. |
| Model manager | 253 registry rows / ~344 HF URLs under `cstr/*`. **`resolve/main`, no sha256, no revision pins** (0 hits, measured). curl/wget shell-out on POSIX. It does have a **licence gate before download** (`--accept-license`). |
| Quantizer | Per-model carve-outs: audio tower ≥ Q8, `token_embd` F16 below Q8, imatrix support. |

---

## 3. ggml fork: compare against our patch stack

Their fork (`CrispStrobe/ggml@2f5a80d2`) marks local patches with `// CrispASR patch|fork`. Their "bump hygiene" is a grep snapshot, which is weaker than our `--check` patch gate (they have lost a patch on a bump). Candidates to **cross-check against speech.cpp's 15 patches**:

1. CPU `MUL_MAT(F16,F32)`: `vec_dot_type = F32`. Activations above 65504 otherwise become Inf/NaN.
2. CUDA `im2col` with OW > 65535 (llama.cpp#22944).
3. CUDA `cpy_scalar_transpose` grid_y < USHRT_MAX.
4. Metal `conv_transpose_1d` O(IL) loop (merged upstream as ggml#1477 — check that our 0.24+ has it).
5. `ggml_conv_*` casts the kernel to F32.
6. cuBLAS `use_fp16` excludes F16×F32.
7. #10: `ggml_backend_sched_split_graph` leaves dangling `input_cpy` after `ggml_free(sched->ctx)`.
8. #15: dual-backend `[CUDA,CPU]` sched gives Inf/NaN in a Qwen2 LLM.

→ Tracked as work item **C1.3** in the rethink note. We do not adopt their fork; we audit each patch against our pin, now due to move to ≥ 0.25.3.

---

## 4. Testing and CI

**What they have**
- 311 test `.cpp` files: Catch2, `unit` and `live` labels, fuzzers (gguf meta, audio load, tokenizer), an ASan job.
- **Stage diff harness:**
  - `tools/dump_reference.py` plus 113 per-model PyTorch modules write a **reference GGUF of named intermediate tensors**.
  - `crispasr-diff <backend> <model> <ref.gguf> <audio>` reports `cos_min` / `max_abs` per stage.
  - This is their parity backbone, and the most transferable asset in the repo.
- **Nightly regression:** 66 manifest entries with pinned revisions, plus a **TTS → ASR round-trip WER** gate for ~21 TTS backends (synthesize, transcribe with parakeet, compute WER).

**Weaknesses**
- 13 test sources are never registered in CMake. This is the same failure class as our `speechcpp-test-integrity` memory.
- 38 of 66 regression entries skip the diff.
- `wer_max` is loose on many entries (0.3–0.5, one at 1.0).
- The harness uses cosine similarity, which is scale-invariant (their own lesson L1396). Always pair it with a magnitude check.

---

## 5. Documentation culture

- LEARNINGS.md (1.1 MB), HISTORY.md (1.1 MB) and PLAN.md (325 KB) are append-only.
- `docs/LEARNINGS-INDEX.md` is **generated and CI-checked**, with 305 lessons. Every heading states the whole lesson — copy this practice.
- AGENTS.md points to an **untracked** dev guide that "wins" over repo docs, so the real method is not in the repo.
- AUDIT.md is an *ops* audit and contains a VPS IP and access path in a public repo. Do not copy.
- Their own warning, which applies to us too: *"PLAN OPEN items are frequently already shipped — audit against the CODE."*

### 5.1 Top transferable engineering lessons (their LEARNINGS line refs)

1. `ggml_siglu` is `sigmoid(a)*b`, while PyTorch GLU is `a*sigmoid(b)` — use `ggml_siglu_swapped` (L3174).
2. `flash_attn_ext` accumulates KQ in F16, and `set_prec(F32)` is ignored on sm_60. Manual F32 attention by default, flash opt-in (L19087).
3. Binary broadcast F32 ⊙ F16 aborts on every backend. Cast 1-D norm affines to F32 (L18623).
4. CPU and cuBLAS F16×F32 matmul saturates above 65504 (ggml patches 1 and 6).
5. A cached cgraph is not re-entrant with `ggml_backend_sched`. Any other graph on the same sched invalidates it (L2458, L2112, L14056).
6. gallocr doesn't protect input tensors. Re-set every input before every compute (L14367) — same as our Voxtral-RT bug.
7. Never feed `ggml_graph_get_tensor` output straight to `tensor_set` (L3145, L13277).
8. Step-graph caching sized to max_ctx is 3.6× *slower*. Use narrow Lk buckets, and measure prep as a share of the step first (`core/step_graph_cache.h`).
9. Metal cached decode is dispatch-bound; split host-encode from GPU execute (L2500).
10. Metal q8_0 mat-vec requantizes activations; MoltenVK downconverts src0 to F16 (L2960, L2675).
11. Vulkan has no F16→F16 `REPEAT`. Cast K/V to F32 before the GQA repeat (L2686).
12. CUDA graphs are disabled below Ampere, so Ampere-only bugs don't reproduce on P100/T4 (L1941).
13. Cosine, correlation and peak-match are scale-invariant; add magnitude checks (L1396).
14. A reference that shares the runtime's assumption cannot falsify it (L352, L906).
15. If the engine and PyTorch agree on the same wrong output, the harness input (usually the prompt) is the bug (L2355).
16. Whole words deterministically dropped usually means a zeroed weight block. Scan converted GGUFs for zero-norm rows (L1636).
17. Sub-Q8 audio towers collapse behaviourally with no per-layer cliff; `token_embd` must stay F16 below Q8 (L14237, L3306).
18. Exact greedy parity is unreachable for a quantized audio AR LM. Validate with a step-0 logit-rank probe (L14731).
19. Linear-interpolation resampling gives −10 dB alias rejection; polyphase gives −89 dB (L260).
20. Mel layout traps show up as cos ≈ 0.3 or −0.15, not as drift (L2058, L2103). This is our `speechcpp-mel-layout-trap` memory.
21. SentencePiece: greedy longest-match is wrong for Unigram and BPE, `byte_fallback` is mandatory, and never `try/except: pass` on missing merges (L13872, L14129, L2332).
22. Streaming conv needs cached left context, not zero pad. Mimi's transformer must be causal (L3225, L2134).
23. A TTS that "never stops" usually has wrong sampling or generation-config defaults, not a bad port (L1681, L14395).
24. A capability flag is a promise: an unimplemented CAP disables the safety nets keyed off it (L1266, L3250).
25. Never benchmark on a busy box, a nearly full disk or a throttling laptop GPU (L1850, L2709).

---

## 6. Verdict

**Adopt (ideas and components):**
1. The stage diff harness and reference dumps.
2. The TTS → ASR round-trip gate.
3. A single arch → family table pinned by a test.
4. A capability bitmask on the family contract.
5. The quantizer carve-out policy.
6. A licence field plus a pre-download licence gate.
7. The generated lessons index.
8. Specific core components: `mel.h` Params (NeMo and HF clusters behind one struct), the polyphase resampler, `repeat_break.h` / `generation_health.h` loop detectors, the Aho-Corasick hotword trie (`asr_context_bias.h`), `realtime_turn_buffer.h` (OpenAI realtime turns), `gpu_backend_pref.h`, and the Wyoming and OpenAI server protocols.
9. Their **model knowledge**: converters, tensor maps and reference modules for ~60 families we lack.

**Do not copy:**
1. Per-surface dual implementation.
2. The `whisper_params` coupling.
3. Env-var configuration.
4. Per-model schedulers, allocators and KV caches.
5. Mega-functions (2.6k-line `crispasr_run_backend`).
6. Unpinned downloads.
7. Pseudo-streaming behind streaming APIs.
8. A grep-maintained ggml fork.
9. Append-only prose as the plan of record.

**Bottom line:** CrispASR has the widest model knowledge of the three sources and the weakest architecture. It is a **reference parent** to mine family by family into speech.cpp's engine, not a code base to merge. See the rethink note §4.
