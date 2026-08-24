# Upstream Synchronization (`d25ffac`) & Phase 11 Migration: Walkthrough

> Session artifact per `MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md` §1 rule 3.
> Covers:
> 1. **Upstream Synchronization (`0xShug0/audio.cpp:main@d25ffac`)**: device enumeration (`--list-devices`), chunk speech metadata resilience, and test suite gates.
> 2. **`speech.cpp` Progressive Naming Unification**: CMake executable aliases (`speech_cli`, `speechcpp_cli`, `speech_server`, `speechcpp_server`, `speech_gguf`, `speechcpp_gguf`) and test names (`speech_cli_list_devices`, `speech_server_list_devices`).
> 3. **Phase 11 Wave W1a**: native engine Moonshine (offline ASR).
> Baseline before this session: 96/96 targets green; **Current suite: 100/100 targets green** on `build-cpu-core`.

---

## 0. Upstream Synchronization & `speech.cpp` Aliases (2026-08-25)

### 0.1 Merged Changes from Upstream `audio.cpp:main`
- **Core Backend Device Enumeration**: Added `engine::core::print_backend_devices(std::ostream & out)` in `include/engine/framework/core/backend.h` & `src/framework/core/backend.cpp`.
- **HTTP Server `--list-devices` Support**: `app/server/main.cpp` now supports `--list-devices`, prints backend devices to `std::cout` and exits 0 before server init.
- **CLI Clean Refactor**: `app/cli/main.cpp` replaced its inlined loop with `engine::core::print_backend_devices(std::cout)`.
- **Chunk Speech Metadata Resilience**: In `src/framework/audio/chunking.cpp` (`append_chunk_speech_metadata`), out-of-span metadata segments/turns now emit a warning diagnostic and return `std::nullopt` instead of aborting with an uncaught `std::runtime_error`. Verified via `test_chunk_speech_metadata_merge_drops_outside_spans` in `tests/unittests/test_audio_chunking.cpp`.

### 0.2 `speech.cpp` Progressive Rebranding & Aliases
- Added CMake executable aliases:
  - `speech_cli` and `speechcpp_cli` $\to$ `audiocpp_cli`
  - `speech_server` and `speechcpp_server` $\to$ `audiocpp_server`
  - `speech_gguf` and `speechcpp_gguf` $\to$ `audiocpp_gguf`
- Added CTest targets `speech_cli_list_devices` and `speech_server_list_devices`.
- Total suite status: **100/100 test targets 100% green** on MSVC CPU core build (96 passed, 4 clean skips on unpinned models).

---

## 1. What was built

### 1.1 New files

| File | LOC | Purpose |
|---|---:|---|
| `include/engine/models/moonshine/graphs_internal.h` | ~250 | Hparams, weight-slot catalog, KV cache, graph-builder declarations |
| `src/models/moonshine/graphs.cpp` | ~700 | Encoder, cross-KV, KV-cached decoder graph builders (ported numerics-identically from `src/runtime/arch/moonshine/{encoder,decoder}.cpp`) |
| `include/engine/models/moonshine/assets.h` + `src/models/moonshine/assets.cpp` | ~330 | GGUF `stt.*` hparams reader with arch-identical invariants; tokenizer via `engine::text::load_tokenizer_from_gguf`; tensor source via ResourceBundle |
| `include/engine/models/moonshine/runtime.h` + `src/models/moonshine/runtime.cpp` | ~380 | Per-session backend weights (`core::BackendWeightStore`), encode → cross-KV → greedy decode orchestration, `RunControl` progress/abort |
| `include/engine/models/moonshine/session.h` + `src/models/moonshine/session.cpp` | ~300 | `MoonshineSession` (`RuntimeSessionBase` + `IOfflineVoiceTaskSession`), `MoonshineLoadedModel`, `MoonshineLoader`, factory `make_moonshine_loader()` |
| `tests/unittests/test_moonshine_engine.cpp` | ~210 | Engine-path gate (see §3) |

### 1.2 Modified files

| File | Change |
|---|---|
| `CMakeLists.txt` | `audiocpp_add_model(moonshine ...)` (graphs/assets/runtime/session sources); added to `AUDIOCPP_ASR_MODEL_TARGETS`; registered `moonshine_engine_smoke_test` (sources linked directly — core model set does not carry ASR families in `engine_runtime`; `WORKING_DIRECTORY` pinned to source root for spec resolution; `SKIP_RETURN_CODE 2` on missing pinned model) |
| `model_specs/moonshine.json` | Sources block corrected: removed required `config` / `tokenizer_json` file mappings (and the safetensors branch) — see finding F-1 below |

Not modified: everything under `src/runtime/arch/moonshine/` (arch copy stays
green per §4.4 coexistence), all transcribe-side tests.

## 2. How the port works

```
ModelRegistry.load(family_hint="moonshine")        registry.cpp find_loader()
  └─ MoonshineLoader::load                         session.cpp
       └─ load_moonshine_assets                    assets.cpp
            ├─ ResourceBundle("weights") → GgufTensorSource (lazy, native types)
            ├─ gguf_init_from_file(no_alloc) → stt.moonshine.* hparams + invariants
            └─ TokenizerHub::load_tokenizer_from_gguf
create_task_session → MoonshineSession             RuntimeSessionBase owns RunControl
  └─ MoonshineRuntime                              BackendWeightStore.upload() once
       run(): encoder graph → host readback → KV cache (F32 default)
              → cross-KV precompute → greedy single-token loop
              (fresh step graph per step = arch CPU path; argmax int32 readback)
```

Key integration points:

- **Weights**: every slot is `BackendWeightStore::load_tensor(...)` with a
  *logical* (PyTorch-order) expected shape; the store reverses into ggml
  `ne` internally (`core::to_ggml_dims`). Native Q8_0 storage is preserved
  so quantized matmuls stay native. `upload()` participates in
  `SharedWeightRegistry` when inside a `ScopedWeightShareKey` (the C ABI /
  CLI wrap load+session-create).
- **Tokenizer**: hub GGUF-BPE decode emits raw `▁` (U+2581); the package
  post-processes `▁ → space` and trims the leading space (mirrors the arch
  `Tokenizer::decode`). Hub-level fix deferred (finding F-3).
- **Abort/progress**: no auto-reset of the abort latch between runs;
  `request_abort()` therefore honors pre-run cancellation requests, and the
  decode loop checks it at every step via `emit_progress`.
- **Batch**: `run_batch` serializes with per-utterance isolation
  (empty result, batch continues) and immediate unwind on abort — the
  engine equivalent of the `Arch::run_batch` contract. (The arch's GPU-only
  native batched step graph is not ported yet; on CPU both serialize.)

## 3. Verification

Baseline first: full rebuild + ctest on untouched tree → 95/95 green.

New gate `moonshine_engine_smoke_test` (CTest #85):

1. Registry resolves canonical id `moonshine` and alias `moonshine-offline`.
2. Loads pinned `models/moonshine-tiny-Q8_0.gguf` (35 MB, skip-contract if absent).
3. Runs the four LibriSpeech fixtures offline through
   `IOfflineVoiceTaskSession::run` after `prepare(...)`.
4. Gates corpus WER ≤ 10% (structural bound shared with the arch gates).

Measured transcripts (engine path):

```
librispeech_test_clean_6930-75918-0000.wav: Concord returned to its place amidst the tents.
librispeech_test_clean_6930-75918-0001.wav: The English voted to the French baskets of flowers, …
librispeech_test_other_7902-96591-0000.wav: I am from the cutter lying off the coast.
librispeech_test_other_7902-96591-0001.wav: Don't cry, he said. I was obliged to come.
corpus WER: 1.44928% (1/69 edits)
```

**1/69 edits — identical to the arch-side baseline** (`asr_e2e_wer_test` /
`asr_e2e_edits_test` measure the same 1 edit on the same fixtures through
`transcribe.dll`). The one edit is the same FORWARDED→VOTED-class
substitution class the arch path produces; text is otherwise word-identical.

Additional assertions in the gate: ordered non-empty `run_batch` results;
`request_abort()` before `run()` unwinds with `ProgressCanceled`.

Product-surface reachability (roadmap §8 Phase 11, per-family step 7):

```powershell
.\build_env.bat .\build-cpu-full\bin\audiocpp_cli.exe --task asr --family moonshine `
    --model .\models\moonshine-tiny-Q8_0.gguf --backend cpu `
    --audio .\assets\asr_validation\librispeech\librispeech_test_other_7902-96591-0000.wav
# => family=moonshine / task=asr / mode=offline
#    text_output=I am from the cutter lying off the coast.
```

`engine_model_moonshine` also compiles under the full model set and
`audiocpp.dll` links with the loader wired into the generated registry
(server/WebUI/C ABI share this registry path).

Full suite after W1a: **96/96 targets green** (92 passed, 4 clean skips:
sortformer stream ext + committed-pointer stability [need streaming model
contract], parakeet ×2 [download contract]).

Commands:

```powershell
.\build_env.bat cmake --build build-cpu-core --config Release -j 8
.\build_env.bat ctest --test-dir build-cpu-core --output-on-failure -C Release
.\build_env.bat ctest -R moonshine_engine_smoke_test --output-on-failure -C Release
```

Formatting: all new files pass pinned `uvx clang-format`.

## 4. Findings (root-caused during W1a)

- **F-1 — Latent Phase-6 spec defect.** The pinned moonshine GGUFs carry
  **no embedded `config.json` / `tokenizer.json` sidecars** (verified by
  scanning the binary: only `stt.*`, `tokenizer.ggml.*`, `general.*`
  metadata keys exist). `model_specs/moonshine.json` required those sidecar
  mappings in its gguf source block, so any loader following the spec could
  never have loaded the shipped packages. Fixed to a tensors-only source
  block; hparams/tokenizer now come from GGUF metadata exactly like the
  arch loader.
- **F-2 — Engine shape convention.** `GgufTensorSource` reports *logical*
  (row-major, PyTorch-order) shapes; `BackendWeightStore` converts into
  ggml `ne` order itself. Migrated families must pass logical shapes to
  `load_tensor` (e.g. conv `[out, in, K]`, linear `[out, in]`,
  embedding `[vocab, d_model]`).
- **F-3 — TokenizerHub GGUF-BPE decode gap.** `decode()` returns pieces
  with raw `▁`. The moonshine package compensates locally; a hub-level
  SentencePiece-flavour option should be added later (small follow-up, not
  blocking).
- **F-4 — Abort latch semantics.** `RunControl` has `reset_abort()`, but an
  abort requested before `run()` must not be silently cleared by the next
  call. The package never auto-resets; consumers get deterministic cancel
  semantics (documented contract test included).

## 5. Deferred within Wave W1 (next increment)

- **W1b — `moonshine_streaming` native package.** Design map follows in §6.
- Retirement of `src/runtime/arch/{moonshine,moonshine_streaming}/`: its own
  commit(s) + Appendix B rows B16a/B16b, only after W1b green **and** user
  review approval (L2/L3/L7).
- Optional perf parity work: static-topology GPU step graph + native batched
  decode (arch has them; CPU behavior identical without them).

## 6. W1b design map (recorded for the next session)

Source of truth: `src/runtime/arch/moonshine_streaming/`
(`model.cpp` 2,252 lines; encoder 372; decoder 737; weights ~700).

Streaming architecture facts (verified against the code):

- **Encoder** differs from offline: embedder (CMVN → compact → linear →
  two causal stride-2 convs) then blocks with **sliding-window attention**
  using per-layer masks (`enc_sliding_windows[2i], [2i+1]` = L/R frames);
  builder exposes `per_layer_masks[i]` inputs filled by
  `build_sliding_window_mask(T_enc, L, R, ...)`.
- **Adapter**: absolute-frame `pos_emb` get_rows + add (+ optional proj)
  applied per window slice; positions start at `abs_frame_offset` so slices
  concatenate to the one-shot result (encoder ergodicity argument, comment
  lines 10–21 of model.cpp).
- **Incremental pipeline per feed**:
  geometry helpers `cumulative_right_context` / `cumulative_left_context`
  (sum of R_i−1 / L_i−1), `k_frontend_pad_enc_frames = 4`,
  `samples_per_encoder_frame = 4 * enc_frame_len`;
  `encode_window_to_host` → `apply_adapter_window` →
  `project_cross_kv_window` append into committed host buffers
  (`stream_adapter_committed`, `stream_cross_k/v_committed[il]`);
  PCM older than `(T_emitted − L_total − frontend_pad)` frames dropped
  (`stream_pcm_buffer` + `stream_pcm_start_sample`).
- **Partial decode** (`decode_partial`, throttled by
  `stream_min_decode_frames` / min-decode-interval-ms):
  `ensure_kv_cache_for_T` → `commit_cross_kv_from_host` (per-layer upload
  graph with F32→F16 cpy support) → `decode_from_kv_cache` greedy loop
  bounded by `dec_max_position_embeddings` AND
  `decode_generation_budget(T_enc)` (13/2 tokens/sec + floor 24).
  Commit = longest common prefix across `stream_token_id_history`.
- **Finalize**: flush remaining stable frames, top up trailing tail, final
  AR decode only if T advanced past last partial; otherwise commit last
  partial as-is.
- **Engine mapping**:
  - Session derives from `engine::runtime::StreamingSessionBase`;
    `on_process_audio_chunk` implements feed (chunker already re-blocks PCM),
    `on_finalize` implements finalize; lifecycle/revision/commit policy are
    base-owned.
  - Map LCP-commit to base `STABLE_PREFIX` policy (`agreement_n` from the
    family extension default), or drive `update_text(full)` per partial and
    let the base split committed/tentative — prefer the latter for contract
    uniformity; keep history-based LCP as the native boundary provider if
    agreement-N diverges.
  - Offline `run()` reuses the one-shot inner path
    (`run_one_shot_inner`) so streamed-vs-offline divergence stays
    measurable against the W1a package.
  - Gate: new `moonshine_streaming_engine_smoke_test` feeding odd-sized
    chunks (~100–400 ms) through `IStreamingVoiceTaskSession`, asserting
    streamed == offline transcript (divergence 0) vs the W1a engine package
    on the same fixture set, mirroring `asr_stream_text_wer_test`.
  - Spec: correct `model_specs/moonshine_streaming.json` sources block the
    same way as F-1; register `audiocpp_add_model(moonshine_streaming ...)`.

## 7. Handoff state

- Suite: 96/96 green on `build-cpu-core` (MODEL_SET=core, unified ABI +
  transcribe arches ON).
- Working tree intentionally left uncommitted (repo rule: commit only on
  explicit user request). All W1a changes are in the working tree.
- Next agent task: execute §6 verbatim, verify, update docs, pause.
