# Phase 10.5, family 3 of 5 — `sortformer_diar` step 2: Walkthrough

> Session artifact per `MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md` §1 rule 3.
> Date: 2026-08-27. Baseline before this session: 109/109 on `build-cpu-core`
> (step 1, `65ab43c`). After: **111/111** core, plus `sortformer_diar_ext_abi_test`
> green in the custom C-ABI tree. Full measurements:
> `docs/reports/sortformer_diar_engine_port.md`.

---

## 0. The first thing that changed: the premise

Step 1 recorded the catalogue's default Sortformer package as a
"transcribe.cpp-flavoured GGUF the arch can load". Reproducing that
(`abi_bridge_hello.exe models/diar_streaming_sortformer_4spk-v2.q8_0.gguf`)
gave `missing KV stt.sortformer.max_speakers`. A KV/tensor dump showed why:
NVIDIA publishes its **own** GGUF layout (`sortformer.encoder.*`,
`sortformer.streaming.*`, `sortformer.scoring.*` KVs; `encoder.layers.N.*`,
`transformer.layers.N.first_sub_layer.query_net.*`, `head.*` tensors). Neither
parent read it. So the increment had three parts, not two: the scheduler, the
typed ext, **and a third-layout loader** — otherwise the retirement would leave
the default package as unreachable as before.

## 1. What was built

| File | Purpose |
|---|---|
| `include/engine/framework/assets/tensor_source.h`, `src/framework/assets/tensor_source.cpp` | `make_renamed_tensor_source(source, rename_fn)` — a view exposing a source's tensors under other names (generic; later ports will need it too) |
| `include/engine/models/sortformer_diar/assets.h`, `src/models/sortformer_diar/assets.cpp` | `SortformerPackageLayout {HuggingFace, NemoGguf}`, streaming defaults + AOSC scoring constants in the model config, per-layout frontend contract, `is_nemo_sortformer_gguf`, KV → config reader, HF↔NeMo tensor-name map, shape-tolerant pointwise-conv loading |
| `src/models/sortformer_diar/frontend.cpp` | Honours the layout contract: normalization mode, peak scaling, floor/ceil framing |
| `include/engine/models/sortformer_diar/graph.h`, `src/models/sortformer_diar/graph.cpp` | Stem/body split; `SortformerPreEncodeGraph` (graph A) and `SortformerBodyGraph` (graph B) per capacity tier with masks; the whole-window graph is stem + body |
| `include/engine/models/sortformer_diar/streaming.h`, `src/models/sortformer_diar/streaming.cpp` | The scheduler: presets, option/env resolution (`resolve_sortformer_run_plan`), `SortformerStreamState`, `sortformer_streaming_update`, `sortformer_compress_spkcache`, `plan_sortformer_chunks`, `run_sortformer_chunked` (graph-agnostic, callback-driven) |
| `include/engine/models/sortformer_diar/session.h`, `src/models/sortformer_diar/session.cpp` | Run-plan dispatch, tiered graph caches (LRU 3), per-chunk progress/abort, `last_probabilities()` / `last_run_plan()`, `SortformerDiarLoader` wrapper (NeMo `can_load`), alias `sortformer` |
| `model_specs/sortformer_diar.json` | `stream_preset` + six `stream_*` geometry options (request and session) |
| `src/runtime/transcribe-arch-adapter.cpp` | RUN-slot `transcribe_sortformer_stream_ext` (accept / validate pre-clear / translate); `prune_request_options_to_contract` |
| `src/framework/model_spec/package.cpp` | Contract resolution: build's spec outranks a package's embedded copy; non-audiocpp GGUFs resolve like a safetensors checkout |
| `src/framework/model_spec/schema.cpp` | `cancellation` as a cross-task capability |
| `src/framework/runtime/registry.cpp` | Foreign `general.architecture` resolved through the family registry before `can_load()` probing |
| `tests/unittests/test_sortformer_diar_streaming.cpp` | The v2 gate (auto-detect, chunked-by-default, oracle under every preset with DER vs the golden RTTM, chunked == whole-window, determinism, cancellation, rejections; `--dump-probs`, `--weight-type` diagnostics) |
| `tests/unittests/test_sortformer_diar_scheduler.cpp` | Host-logic unit test: chunk planning, FIFO/cache update + compression invariants, plan resolution + precedence, the chunk driver on a stand-in model |
| `tests/sortformer_diar_ext_abi_test.cpp` | The C-ABI ext gate (custom tree) |
| `CMakeLists.txt` | `streaming.cpp` in the model; alias; three test registrations |
| `docs/reports/sortformer_diar_engine_port.md` | The report |

## 2. How the chunked path runs

```
run(request)
  resolve_sortformer_run_plan(config, session ⊕ request options, env)
    ├─ WholeWindow → run_offline_diarization (stem + body, fixed session context; over-length is REJECTED, never trimmed)
    └─ Chunked     → run_chunked_diarization
         compute_sortformer_features (layout contract)
         run_sortformer_chunked(features, params, pre_encode, infer, progress)
           for each NeMo window [win_lo, win_hi):
             progress(chunk, n)            ← RunControl unwinds here
             pre_encode: graph A @ tier(M)  → embeddings [T_diar, 512]
             concat [spkcache | fifo | chunk]
             infer:      graph B @ tier(T_concat) → preds [T_concat, 4]
             sortformer_streaming_update (FIFO pop → silence profile → cache append → compress)
           trim to ceil(feat_len / 8)
         decode_sortformer_speaker_turns (the engine post-processing, both paths)
```

NeMo's `xscaling` is applied inside graph B on the concatenation (cached rows
are raw pre-encode output), as the arch did. Padding rows are zero, masked out
of attention, zeroed before the depthwise convs and never read back — which is
what lets one graph per 64-frame tier serve every chunk geometry, and what the
1.8e-7 whole-window-vs-chunked agreement verifies.

## 3. Verification

```powershell
.\build_env.bat cmake --build build-cpu-core --config Release -j 12
.\build_env.bat ctest --test-dir build-cpu-core --output-on-failure -C Release   # 111/111
.\build-cpu-core\bin\sortformer_diar_streaming_engine_test.exe models\diar_streaming_sortformer_4spk-v2.q8_0.gguf assets\asr_validation\librispeech samples\sortformer-2spk-mix.wav tests\golden\sortformer\sortformer-2spk-mix.rttm
# custom tree (links the family): -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=qwen3_asr,voxtral_realtime,sortformer_diar
.\build-cpu-qwen3\bin\sortformer_diar_ext_abi_test.exe models\sortformer-diar-4spk-v1-q8_0.gguf assets\asr_validation\librispeech
.\build-cpu-qwen3\bin\audiocpp_cli.exe --task diar --model models\diar_streaming_sortformer_4spk-v2.q8_0.gguf --backend cpu --audio samples\sortformer-2spk-mix.wav --turns-out turns.json --metrics
```

Headline numbers (CPU, i9-13900H): fixtures 1 speaker each; oracle 2 speakers
under all five operating points, DER 0.31–0.34 (arch on the same weights:
0.315–0.334); chunked == whole-window 1.8e-7; vs the arch 0/600 decision flips
on the published operating points (max |Δp| 7.4e-3); CLI RTF 0.21 on the
oracle with the shipped operating point.

## 4. Findings to carry forward

- **Reproduce a package-layout claim with a KV dump** before planning around
  it. `uv run --with gguf python -c "from gguf import GGUFReader; ..."` costs a
  minute; step 1's claim cost the plan a wrong scope.
- **Two contracts were in play through the C ABI**: the session validated
  request options against the spec embedded in the package (stale), the
  adapter pruned against the workspace spec. Whenever a family option is added,
  ask which spec the running binary will resolve.
- **Strict request-option validation is a contract that the adapter has to
  honour**, not a family quirk — four families were unusable through the C ABI.
- **Read the arch's consumers before scoping a deletion.** `grep -rn sortformer
  src/runtime/arch/parakeet` bounded step 3 to the standalone family.
- **`scripts/ci/clang-format.sh` does not exist in this tree** although
  `AGENTS.md` and the CI workflow reference it; formatting was done by hand to
  the surrounding style.

## 5. Handoff state

- Working tree: all step-2 changes, committed as one feature-merge commit (see
  `git log`).
- Next: step 3 — retire the standalone `sortformer` family from the transcribe
  dispatcher (scope in the tracker's IMMEDIATE NEXT TASK block and roadmap
  Appendix B row B15), which is also what routes the NeMo v2 package to the
  engine through the C ABI; re-point `sortformer_diar_ext_abi_test` at v2 then.
