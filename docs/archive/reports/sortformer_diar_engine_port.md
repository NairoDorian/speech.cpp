# Sortformer diarization: the chunked scheduler and the NeMo package, in the engine

> **Date**: 2026-08-27 · **Phase 10.5, family 3 of 5, step 2** · tree: `main` after `2bf2a32`
> **Subject**: `src/models/sortformer_diar/` after absorbing the transcribe.cpp arch's distinguishing feature — the AOSC speaker-cache + FIFO chunked scheduler and its operating-point presets — and learning to open the catalogue's default package.
> **Method**: measure first (L12), on real audio, through the public surfaces; every number below has a command.

## 1. What step 1 got wrong, and what this step therefore had to do

Step 1 (`65ab43c`) recorded that the catalogue's default package,
`nvidia/diar_streaming_sortformer_4spk-v2` (`diar_streaming_sortformer_4spk-v2.q8_0.gguf`,
`general.architecture = "sortformer"`, 971 tensors), was "a transcribe.cpp-flavoured
GGUF" that the arch could load and the engine could not. Reproducing that
claim (asset doctrine: reproduce before accepting) showed otherwise:

```
$ build-cpu-core/bin/abi_bridge_hello.exe models/diar_streaming_sortformer_4spk-v2.q8_0.gguf
sortformer: missing KV stt.sortformer.max_speakers
abi_bridge_hello: transcribe_open failed: status=4
```

The file is a **third layout** — NVIDIA's own: hparams under `sortformer.encoder.*` /
`sortformer.transformer.*` / `sortformer.preprocessor.*`, the streaming operating
point under `sortformer.streaming.*`, the AOSC scoring constants under
`sortformer.scoring.*`, tensors named `encoder.layers.N.*`,
`transformer.layers.N.first_sub_layer.query_net.*`, `head.*`, `encoder_proj.*`.
The arch reads its converter's `stt.sortformer.*` / `enc.blocks.*` / `tf.blocks.*`;
the engine read the HuggingFace port's `fc_encoder.*` / `tf_encoder.*` plus JSON
sidecars. **Neither parent could open the package the catalogue installs by
default.** The retirement therefore had to deliver a NeMo-layout loader as well as
the scheduler — otherwise deleting the arch would leave v2 exactly as unreachable
as it was.

## 2. What landed

| Piece | Where | Notes |
|---|---|---|
| NeMo-layout loader | `assets.cpp` (`is_nemo_sortformer_gguf`, `read_nemo_model_config`, `hf_name_for_nemo_tensor`), `session.cpp` (`SortformerDiarLoader`) | KV → config; the HF-named weight loader is served through a renaming `TensorSource` view, so there is one weight loader, not two. `encoder.pos_enc.pe` and `preprocessor.fb` are dropped (the engine computes both). Registry auto-detection (no family hint) reaches it through `can_load()`. |
| `make_renamed_tensor_source` | `framework/assets/tensor_source.{h,cpp}` | Generic: any family whose loader is written against one checkpoint naming can read a package converted by another tool. |
| Frontend contract per layout | `assets.h` (`SortformerFeatureExtractorConfig`), `frontend.cpp` | HF packages keep per-feature normalization, peak scaling and floor(n/hop) framing; NeMo packages get `normalize=NA`, no peak scaling, ceil(n/hop) framing (the transcribe.cpp frontend's `nemo_seq_len_ceil`). |
| Graph split | `graph.{h,cpp}` | The monolithic whole-window graph is now stem + body; the chunked path builds a **pre-encode graph** (stem over one mel window) and a **body graph** (input scaling → 17 conformer blocks → `encoder_proj` → 18 post-LN transformer blocks → head) per capacity tier, padded and masked. NeMo's `xscaling` is applied to the concatenation, not to the cached rows — as the arch did. |
| Scheduler | `streaming.{h,cpp}` | Host-for-host port of `arch/sortformer/stream.cpp`: `streaming_update` (sync), `_get_silence_profile`, the `_compress_spkcache` stack (log-pred scores → disable-low → boost-latest → strong/weak top-k → silence pad → top-k gather), NeMo's `streaming_feat_loader` windows, the preset table, option/env precedence. Graph-agnostic: it drives the stem and body through callbacks. |
| Run plan | `streaming.cpp` (`resolve_sortformer_run_plan`), `session.cpp` | A package that ships a streaming operating point runs **chunked by default** (what the arch shipped); one that does not (the HF v1 checkpoints) runs whole-window. `stream_preset` = `default` / `offline` / `very_high_latency` / `high_latency` / `low_latency` / `small`; per-field `stream_*` options; `TRANSCRIBE_SORTFORMER_STREAM_*` env as validation hooks with the arch's precedence. |
| Product surface | `model_specs/sortformer_diar.json`, `transcribe-arch-adapter.cpp` | Request + session options for the presets and geometry; the arch's `transcribe_sortformer_stream_ext` (RUN slot) is accepted, validated pre-clear and translated by the ArchAdapter, so `transcribe/sortformer.h` keeps working after the arch is gone. |
| Progress + cancellation | `session.cpp` | One progress unit per chunk; `RunControl` unwinds between chunks. |
| Diagnostics | `SortformerDiarSession::last_probabilities()` / `last_run_plan()` | The model's actual output (transcribe.cpp dumped the same tensor as `diar.probs`) for parity tools; the product reads `speaker_turns`. |

### Graphs are cached per capacity tier, not rebuilt per chunk

The arch rebuilt both graphs for every chunk, which is why its `low_latency`
throughput was ~0.85× realtime on CPU ("per-chunk graph rebuild dominates at
0.5 s chunks"). The engine builds a graph per 64-frame capacity tier and keeps a
small LRU (3 per kind); rows past the valid count are zero, masked out of
attention, and never read back. The steady state of any preset is one tier.

## 3. Measurements

Machine: i9-13900H, CPU backend, `build-cpu-core` (MODEL_SET=core, unified ABI +
arches ON), pinned `diar_streaming_sortformer_4spk-v2.q8_0.gguf` (Q8_0, 147 MB),
oracle `samples/sortformer-2spk-mix.wav` (12.0 s, two synthetic speakers,
golden RTTM `tests/golden/sortformer/sortformer-2spk-mix.rttm`).

### 3.1 The gate — `sortformer_diar_streaming_engine_test`

```
build-cpu-core/bin/sortformer_diar_streaming_engine_test.exe \
    models/diar_streaming_sortformer_4spk-v2.q8_0.gguf assets/asr_validation/librispeech \
    samples/sortformer-2spk-mix.wav tests/golden/sortformer/sortformer-2spk-mix.rttm
```

| Input | Result |
|---|---|
| LibriSpeech fixtures (4, single speaker), shipped operating point | 1 speaker each, coverage 0.73–0.91, plan `chunked (default)`; 0.42–1.81 s for 2.1–14.2 s of audio |
| oracle, `default` (chunk 188 / rc 1 / fifo 0 / cache 188) | 2 speakers, DER **0.3288** |
| oracle, `very_high_latency` (340 / 40 / 40 / 188) | 2 speakers, DER 0.3288 |
| oracle, `high_latency` (124 / 1 / 124 / 188) | 2 speakers, DER 0.3352 |
| oracle, `low_latency` (6 / 7 / 188 / 188) | 2 speakers, DER 0.3224, 16.3 s (25 chunks) |
| oracle, `small` (20 / 1 / 10 / 24 — forces compression) | 2 speakers, DER 0.3096 |
| oracle, `offline` (whole-window graph) vs `default` (chunked, single chunk) | **max \|Δp\| = 1.79e-7** over 150 frames × 4 speakers |
| determinism | identical probabilities across runs |
| cancellation | declining progress after chunk 1 unwinds with `ProgressCanceled`; `request_abort()` before `run()` unwinds |
| unknown preset | refused; `stream_chunk_len=50` override applied and reported in the plan |

DER: 10 ms grid, no collar, overlap counted, best hypothesis→reference label
mapping. **The absolute DER is a property of the model on this synthetic clip,
not of the port** — see §3.2; the gate bound (0.35) is parity with the arch, not
absolute quality.

### 3.2 Parity with the transcribe.cpp arch on identical weights

The arch cannot read NVIDIA's layout, so its parity was measured by re-keying the
pinned package into the arch's layout (KVs and tensor names only; tensor bytes
copied verbatim; the arch's unused 2×hidden head slot zero-filled) and running
both implementations on the oracle under every operating point. The arch's own
port was validated against NeMo at max_abs 4.5e-4 on this clip
(`tests/tolerances/sortformer.json`), so it stands in for the reference here.

```
# scratch tools (not shipped): nemo_to_transcribe_gguf.py, then
TRANSCRIBE_DUMP_DIR=<dir> TRANSCRIBE_SORTFORMER_STREAM_PRESET=<preset> \
    build-cpu-core/bin/asr_e2e_wer_test.exe <re-keyed.gguf> <dir with the oracle>   # dumps diar.probs
build-cpu-core/bin/sortformer_diar_streaming_engine_test.exe ... --dump-probs <dir>  # probs_<preset>.f32
```

| Operating point | arch DER @0.5 | engine DER | max \|Δp\| | mean \|Δp\| | decision flips @0.5 |
|---|---|---|---|---|---|
| `default` | 0.3280 | 0.3288 | 7.4e-3 | 1.7e-4 | **0 / 600** |
| `very_high_latency` | 0.3280 | 0.3288 | 7.4e-3 | 1.7e-4 | **0 / 600** |
| `high_latency` | 0.3344 | 0.3352 | 8.2e-3 | 2.6e-4 | **0 / 600** |
| `low_latency` | 0.3152 | 0.3224 | 1.4e-2 | 2.1e-4 | 1 / 600 |
| `small` | 0.3152 | 0.3096 | 4.4e-2 | 9.3e-4 | 1 / 600 |

Read honestly: the port is at **decision parity** with the arch on the three
published operating points, not at tensor parity to the arch's 4.5e-4 tolerance.
The residual (max ~7e-3 on a sigmoid output, mean 1.7e-4) is not the weight
storage type — with `sortformer_diar.weight_type=native` (Q8_0 matmuls, F16
convs, what the arch runs) the numbers are the same size (max 1.1e-2, 0 flips) —
so it lives in op ordering between the engine's frontend / conformer / attention
modules and transcribe.cpp's. The compression-heavy geometries (`low_latency`,
`small`) amplify it into single-frame flips, exactly the chaotic sensitivity the
family doc records for the cache picks. Bringing the family under `validate.py`
against the NeMo reference dumps, with the existing tolerance file, is the Track M
item that would localise it; it is not a blocker for the retirement, because the
arch's own tolerance regime is per-tensor parity and this gate holds the product
output at parity with the arch.

Two more numbers from the same run: the whole-window engine graph vs the arch's
default is 7.5e-3 (same residual, so it is not in the scheduler), and the engine's
chunked path reproduces the whole-window graph to 1.8e-7 (so the split is exact).

### 3.3 The C ABI — `sortformer_diar_ext_abi_test`

Runs where the product links the family (`-DAUDIOCPP_MODEL_SET=custom
-DAUDIOCPP_MODELS=...,sortformer_diar`), on the v1 package (an audio.cpp GGUF,
which the C ABI routes through the adapter today; the NeMo v2 package still
names the builtin `sortformer` arch and reaches the engine only once that arch is
retired — the step-3 dependency).

Holds: the kind is accepted on RUN only and a foreign kind on neither slot;
`transcribe_sortformer_stream_ext_init` stamps size / kind / DEFAULT; a wrong-kind
ext and preset 99 are refused with `INVALID_ARG` **pre-clear** (the previous
speaker segments survive); ext `VERY_HIGH_LATENCY` produces the same rows as
`TRANSCRIBE_SORTFORMER_STREAM_PRESET=very_high_latency`; `LOW_LATENCY` runs.

Result: **OK** (§5).

## 4. Defects found on the way (none of them in the family's own code)

1. **Every strictly spec-validated family was unusable through the C ABI.**
   `apply_run_params` forwards `language`, `task`, `timestamps`, `pnc`, `itn`,
   `diarize`, `keep_special_tags`, `spec_k_drafts` as request options; families
   that validate request options against their contract (`sortformer_diar`,
   `sense_asr`, `hviske_asr`, `higgs_audio_stt`) rejected the first one:
   `unknown request option: language`. Qwen3-ASR and Voxtral passed only because
   they do not validate. The adapter now prunes to the family's contract.
2. **A package's embedded spec outranked the build's.** For an audio.cpp GGUF
   the contract was resolved from the spec embedded at conversion time, so an
   option added to `model_specs/<family>.json` after the package was published
   — `stream_preset` here, and step 1's `cancellation` capability, which never
   actually reached the v1 package — was rejected by the very session that
   implements it, while the adapter pruned against the workspace spec. The
   build's copy (workspace, else builtin) now wins when present; the embedded
   copy remains the fallback that keeps a package self-describing without a
   catalog.
3. **A GGUF without an embedded spec was refused a contract outright**
   ("Published GGUF packages must embed a model spec") — a rule for audio.cpp's
   own converter output, applied to every GGUF. A package whose
   `general.architecture` is not `audiocpp` now resolves its contract like a
   safetensors checkout. (`minimax_h3` already had a private carve-out for the
   same reason.)
4. **The schema validator had no `cancellation` capability**, so the workspace
   spec step 1 edited was invalid — unnoticed because the v1 path never
   validated it (defect 2). `cancellation` is now a cross-task capability.
5. **Registry auto-detection decoded the foreign GGUF as Silero VAD.** The
   product registry probes `can_load()` in registration order for a GGUF that
   embeds no family; Silero accepts any GGUF whose bundle resolves. `bf0e68a`
   fixed this for audio.cpp packages (embedded family); NVIDIA's package has
   none, so `audiocpp_cli --model <v2.gguf>` without `--family` died with
   `missing tensor: stft_conv.weight`. `find_loader` now maps a non-`audiocpp`
   `general.architecture` through `family_registry` (`sortformer` →
   `sortformer_diar`) before probing.

## 5. Suite and product reach

- `build-cpu-core`: **111/111** (107 passed, 4 documented skips); new gates
  `sortformer_diar_streaming_engine_test`, `sortformer_diar_scheduler_test`;
  step 1's `sortformer_diar_engine_smoke_test` unchanged (v1 through the new
  loader wrapper).
- Custom tree (`-DAUDIOCPP_MODELS=qwen3_asr,voxtral_realtime,sortformer_diar`):
  `sortformer_diar_ext_abi_test` **OK** (probe, init, pre-clear rejection,
  ext == env rows, low-latency run), and the qwen3 / voxtral C-ABI gates,
  the four `capi_*` tests and `model_spec_system_test` unchanged after the
  adapter and contract-resolution changes.
- `audiocpp_cli --task diar --model models/diar_streaming_sortformer_4spk-v2.q8_0.gguf
  --audio samples/sortformer-2spk-mix.wav` (no `--family`): both speakers,
  RTF 0.21 with the shipped operating point, 0.42 with
  `--request-option stream_preset=low_latency`.

## 6. What the retirement (step 3) still has to answer for

- Route `general.architecture == "sortformer"` GGUFs to the adapter: today the
  builtin arch owns the name, which is why the NeMo v2 package is unreachable
  through the C ABI until the arch is deleted. The adapter's family table
  already lists `sortformer_diar`; `family_registry` maps `sortformer` to it.
- Move `transcribe_sortformer_stream_ext_init` from `arch/sortformer/model.cpp`
  to `src/runtime/transcribe-family-ext.cpp` in the same commit (it is a
  published symbol; see the voxtral_realtime retirement).
- Re-point `sortformer_diar_ext_abi_test` at the v2 package once the routing
  lands, and delete `tests/transcribe/sortformer_stream_ext_unit.cpp`
  (registered as `transcribe_sortformer_stream_ext_unit`, env-gated, skipped in
  every run on record).
- The parakeet multitalker bundle path (`init_embedded_diarizer`,
  `run_diar_streaming_core` driven from `arch/parakeet`) is the arch's other
  consumer; it stays with the parakeet arch until Phase 11 retires that family,
  so the sortformer arch directory cannot be deleted before the parakeet arch
  stops including it — check `grep -rn "sortformer" src/runtime/arch/parakeet`.
