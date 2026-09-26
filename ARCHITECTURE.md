# speech.cpp — architecture map

> **Generated** by `scripts/arch/gen-architecture.ts` from the working tree with [repomix](https://github.com/yamadashy/repomix) (pinned in `scripts/arch/package.json`). **Do not edit by hand.** The pre-commit hook regenerates it (`.githooks/pre-commit`). To change a description, edit `DIR_DESCRIPTIONS` in the generator.
>
> **Reading order for any agent or new contributor:** this map → [`PLAN.md`](PLAN.md) *North star* (the end goal + rules R1–R11) → [`AGENTS.md`](AGENTS.md) → [`LESSONS.md`](LESSONS.md) headings. For file contents, generate a pack with `bun scripts/arch/repomix.ts [paths…]` (gitignored `repomix-output.xml`).

**Scope measured:** 3147 first-party files, ~883k lines, ~9.96M tokens (o200k). Excludes `external/` (vendored ggml), build trees, models, binaries and `docs/archive/`; see `repomix.config.json`. Counts are rounded to 3 significant figures.

## 1. Layers

```
host apps (FreeSpeech desktop #1, FreeSpeech Android #2)      DEV tools: app/ (cli, server+WebUI, gguf, model manager)
        │  link in-process (BUNDLE profile)                           │ depend on the library, never the reverse (R1)
        ▼                                                             ▼
C ABI: include/transcribe/transcribe.h today → include/speech/speech.h (target, V7-D1)   [capi/audiocpp.h: legacy, retiring]
        │  src/runtime/ (transcribe runtime + ArchAdapter; deleted in 11c)
        ▼
engine contract: include/engine/framework/runtime (registry, loaders, task sessions)
        ▼
src/framework/ shared layers (asr, audio, tokenizers, modules, codecs, runtime, assets)
        ▼
families: src/models/<f>, src/community_models/<f> (+ model_specs/<f>.json)   [src/runtime/arch/<f>: retiring]
        ▼
external/ggml (generated: pin + patches/ggml)
```

## 2. Key entry points

| File | Why you open it |
|---|---|
| [`PLAN.md`](PLAN.md) | North star (end goal + rules R1–R11), current state, phases. |
| [`AGENTS.md`](AGENTS.md) | How to work here: build/test commands, sync routine, test tiers, git rules. |
| [`LESSONS.md`](LESSONS.md) | Every hard-won lesson; headings state the lesson. |
| [`docs/LEDGER.md`](docs/LEDGER.md) | Decisions (V7-D*), family migration state, deletion rows. |
| [`docs/benchmarking.md`](docs/benchmarking.md) | Port acceptance: speed + WER vs both parents, run protocol, backend matrix. |
| [`docs/glossary.md`](docs/glossary.md) | Vocabulary (ASR/TTS/streaming/runtime), metrics (WER, CER, RTF, TTFA, DER, SIM, ...) and benchmarks per task. |
| [`CMakeLists.txt`](CMakeLists.txt) | The build: options, MODEL_SET selection, audiocpp_add_model blocks, tests. |
| [`include/engine/framework/runtime/model.h`](include/engine/framework/runtime/model.h) | Engine contract: IVoiceModelLoader / ILoadedVoiceModel. |
| [`include/engine/framework/runtime/session.h`](include/engine/framework/runtime/session.h) | Engine contract: task sessions (offline, batched, streaming). |
| [`include/transcribe/transcribe.h`](include/transcribe/transcribe.h) | The C ABI apps embed today. |
| [`src/runtime/transcribe-arch-adapter.cpp`](src/runtime/transcribe-arch-adapter.cpp) | Bridge: C ABI → engine families (deleted in 11c). |
| [`scripts/status.sh`](scripts/status.sh) | Generated project status (never hand-write counts). |
| [`scripts/sync-deps.sh`](scripts/sync-deps.sh) | Parent drift report (audio.cpp, transcribe.cpp, ggml, CrispASR). |

## 3. Directory map

| Directory | Files | Lines | Tokens | What it is |
|---|--:|--:|--:|---|
| _(root files)_ | 11 | 9.16k | 148k | Top-level docs and build entry points (README, PLAN, AGENTS, LESSONS, CHANGELOG, CMakeLists.txt, presets). |
| `app/` | 51 | 13.9k | 137k | DEV-profile tools (never a library dependency, R1): CLI, server + WebUI, GGUF tool, model manager, workflow. |
| &nbsp;&nbsp;`app/cli/` | 8 | 2.09k | 21.2k | audiocpp_cli: run any family from the command line (DEV). |
| &nbsp;&nbsp;`app/common/` | 3 | 74 | 581 | Code shared by the DEV apps. |
| &nbsp;&nbsp;`app/gguf/` | 1 | 804 | 8.82k | audiocpp_gguf: GGUF conversion / inspection tool (DEV). |
| &nbsp;&nbsp;`app/model_manager/` | 1 | 114 | 1.08k | Model download / install manager (DEV). |
| &nbsp;&nbsp;`app/server/` | 26 | 8.41k | 83k | audiocpp_server: HTTP API + embedded WebUI (DEV). |
| &nbsp;&nbsp;`app/streaming/` | 4 | 352 | 2.95k | Streaming helpers for the DEV apps. |
| &nbsp;&nbsp;`app/workflow/` | 8 | 2.08k | 19.1k | JSON pipeline / workflow runner (DEV). |
| `capi/` | 30 | 5.69k | 63.4k | Legacy speech.cpp `audiocpp.h` C ABI (frozen; retired after speech.h, V7-D1). |
| &nbsp;&nbsp;`capi/include/` | 1 | 1.22k | 12.6k | Legacy audiocpp.h header (frozen). |
| &nbsp;&nbsp;`capi/src/` | 4 | 2.75k | 25.9k | Legacy audiocpp C ABI implementation. |
| &nbsp;&nbsp;`capi/test/` | 24 | 1.07k | 16.8k | Ad-hoc scripts from another machine with foreign absolute paths (delete: PLAN S1.2). |
| `cmake/` | 3 | 320 | 4.51k | CMake helper modules (transcribe test registration, ...). |
| `docs/` | 166 | 30.9k | 464k | Documentation. Plan-level docs: LEDGER.md; reports/; upstream/ triage ledgers; porting/ guides. |
| &nbsp;&nbsp;`docs/build/` | 6 | 1.3k | 17.1k | Platform build guides. |
| &nbsp;&nbsp;`docs/community_models/` | 39 | 6.91k | 93.5k | Model cards for community families. |
| &nbsp;&nbsp;`docs/maintainers/` | 2 | 358 | 3.56k | Maintainer notes (loader, catalog, model specs). |
| &nbsp;&nbsp;`docs/models/` | 39 | 5.47k | 67.4k | Model cards for audio.cpp-origin families. |
| &nbsp;&nbsp;`docs/porting/` | 31 | 6.63k | 109k | Staged porting guides (from transcribe.cpp; read their speech.cpp notes: engine packages, not src/arch). |
| &nbsp;&nbsp;`docs/proposals/` | 2 | 333 | 3.67k | Design proposals. |
| &nbsp;&nbsp;`docs/reports/` | 22 | 4.09k | 70.2k | Dated reports: checkpoint audit, strategy, CrispASR analysis, validation and performance. |
| &nbsp;&nbsp;`docs/superpowers/` | 2 | 633 | 7.58k | Upstream agent plans/specs (audio.cpp). |
| &nbsp;&nbsp;`docs/upstream/` | 2 | 196 | 13.1k | Parent triage ledgers (transcribe_cpp_triage.md is parsed by sync-deps.sh). |
| `examples/` | 61 | 4.14k | 34.6k | Example programs (DEV). |
| &nbsp;&nbsp;_3 more under `examples/`_ | 27 | 2.8k | 22.9k | e.g. browser_extension, model_spec_demo, yue2_style_change (undescribed) |
| &nbsp;&nbsp;`examples/docker/` | 25 | 535 | 4.47k | Docker example. |
| &nbsp;&nbsp;`examples/xcode/` | 7 | 662 | 5.83k | Apple/Xcode integration example. |
| `include/` | 905 | 69.2k | 578k | Public and internal headers. |
| &nbsp;&nbsp;`include/engine/` | 896 | 65.3k | 538k | Engine C++ API: framework contracts and per-family headers. |
| &nbsp;&nbsp;&nbsp;&nbsp;`include/engine/community_models/` | 195 | 12.7k | 109k | Per-family headers for src/community_models. |
| &nbsp;&nbsp;&nbsp;&nbsp;`include/engine/framework/` | 200 | 18.2k | 139k | Engine framework headers (runtime contract: IVoiceModelLoader / ILoadedVoiceModel / I*VoiceTaskSession). |
| &nbsp;&nbsp;&nbsp;&nbsp;`include/engine/models/` | 501 | 34.4k | 291k | Per-family headers for src/models. |
| &nbsp;&nbsp;`include/transcribe/` | 7 | 3.33k | 35k | transcribe.h C ABI + family extension headers. The ABI apps embed today; becomes a shim over speech.h (V7-D1). |
| `model_specs/` | 104 | 22.8k | 174k | Model package specifications (JSON): variants, files, options, capabilities, licences. |
| `patches/` | 15 | 9.69k | 130k | Downstream patches; patches/ggml is the ggml invariant (sync-ggml.sh --check). |
| &nbsp;&nbsp;`patches/ggml/` | 15 | 9.69k | 130k | ggml patch stack re-applied on every sync, in filename order. |
| `scripts/` | 24 | 7.12k | 75.4k | Maintenance scripts: sync-deps.sh, sync-ggml.sh, status.sh, fetch_asr_test_model.py, converters. |
| &nbsp;&nbsp;`scripts/arch/` | 5 | 462 | 7.54k | This generator, the repomix pack script and the pre-commit routine (bun). |
| `src/` | 1062 | 535k | 6.04M | Library implementation (C++17). |
| &nbsp;&nbsp;`src/capi/` | 3 | 1.24k | 11.4k | C ABI glue on the engine side. |
| &nbsp;&nbsp;`src/community_models/` | 205 | 103k | 1.12M | Community model families (engine packages; lighter review). See the family table. |
| &nbsp;&nbsp;`src/framework/` | 185 | 111k | 1.43M | Shared engine layers every family builds on (R5: families must not re-implement these). |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/asr/` | 2 | 253 | 2.2k | ASR runtime layer: EncDecKVCache, decode drivers, sampling, AsrResult/AsrLimits (PLAN S2). |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/assets/` | 9 | 4.68k | 47.3k | Bundled-asset resolution (asset_paths: never CWD-relative). |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/audio/` | 28 | 17.3k | 195k | Audio front-ends: MelExtractor, fbank, resampling, chunking, STFT. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/codecs/` | 9 | 9.91k | 103k | Neural audio codecs shared by TTS/S2S families. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/conditioners/` | 5 | 3.6k | 36.9k | Conditioning modules for generative TTS/music (speaker, text, prompt encoders). |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/core/` | 6 | 1.37k | 12k | Backend/device management, tensors, weights, common utilities. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/debug/` | 2 | 383 | 2.62k | Debug dump helpers. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/decoders/` | 8 | 1.44k | 14.2k | Shared decoder building blocks. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/io/` | 7 | 1.65k | 11.9k | Audio / file I/O and serialization (wav, safetensors, ...). |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/midi/` | 1 | 266 | 2.61k | MIDI output for music-transcription families. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/model_spec/` | 4 | 1.93k | 19.6k | Model-spec schema, loading and validation (model_specs/*.json). |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/modules/` | 62 | 30.8k | 321k | Reusable ggml graph modules (attention, conformer, convs, ...). |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/package_manager/` | 1 | 882 | 8.58k | Model package resolution (spec → files). |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/runtime/` | 20 | 4.92k | 46.9k | Engine contract implementation: registry, sessions, KV cache, run control. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/sampling/` | 7 | 2.2k | 21.2k | Token sampling (temperature, top-k/p, penalties) for AR decoders. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/text/` | 10 | 29k | 581k | Text processing: normalization, G2P helpers. |
| &nbsp;&nbsp;&nbsp;&nbsp;`src/framework/tokenizers/` | 4 | 775 | 7.77k | TokenizerHub: BPE / SentencePiece / tekken / ... shared tokenizers. |
| &nbsp;&nbsp;`src/models/` | 517 | 256k | 2.75M | Core model families as engine packages (graphs + assets + spec + thin session). See the family table. |
| &nbsp;&nbsp;`src/runtime/` | 152 | 63.5k | 723k | transcribe.cpp runtime + C ABI implementation (transcribe.h) + remaining arch families. Deleted in PLAN phase 11c. |
| `tests/` | 605 | 142k | 1.73M | Unit, parity, verdict and WER gates. Model-backed tests use the quick tier by default (AGENTS.md). |
| &nbsp;&nbsp;_84 more under `tests/`_ | 362 | 71.3k | 967k | e.g. ace_step, apollo, audio, audio8_asr, audio_utilities, auk, … (per-family test dirs) |
| &nbsp;&nbsp;`tests/capi/` | 4 | 1.49k | 16.1k | Legacy audiocpp C ABI tests. |
| &nbsp;&nbsp;`tests/community_models/` | 2 | 245 | 3.1k | Tests for community families. |
| &nbsp;&nbsp;`tests/core/` | 2 | 385 | 3.39k | Framework core tests. |
| &nbsp;&nbsp;`tests/fixtures/` | 3 | 1.86k | 22.4k | Test fixtures. |
| &nbsp;&nbsp;`tests/perf/` | 4 | 3.5k | 33k | Performance probes. |
| &nbsp;&nbsp;`tests/tolerances/` | 36 | 3.18k | 50.3k | Per-family numeric tolerance files (L10). |
| &nbsp;&nbsp;`tests/transcribe/` | 49 | 19.9k | 202k | Tests vendored from transcribe.cpp (some unregistered: PLAN S1.4). |
| &nbsp;&nbsp;`tests/unittests/` | 129 | 32.5k | 348k | Framework + family unit/parity tests (incl. *_engine_arch_parity, abi_arch_engine_verdict). |
| `tools/` | 79 | 23.7k | 254k | Developer tools: CLI path tests, model manager, converters (DEV). |
| &nbsp;&nbsp;_2 more under `tools/`_ | 6 | 345 | 3.23k | e.g. cmake, omtd (undescribed) |
| &nbsp;&nbsp;`tools/audiocpp_cli/` | 9 | 6.27k | 84.5k | CLI path-test harness (run_audiocpp_cli_path_tests.py) (DEV). |
| &nbsp;&nbsp;`tools/community_models/` | 53 | 11.4k | 112k | Community-family tooling (converters, checks). |
| &nbsp;&nbsp;`tools/streaming/` | 2 | 1.01k | 9.04k | Streaming test tools. |
| `webui/` | 31 | 9.07k | 135k | WebUI sources and configs (DEV profile; served by audiocpp_server). |
| &nbsp;&nbsp;`webui/configs/` | 2 | 805 | 41.8k | WebUI model catalog / parameter configs. |
| &nbsp;&nbsp;`webui/native/` | 28 | 8.17k | 91.9k | WebUI app sources (SvelteKit) + demo voices (DEV). |

## 4. Model families

**71** core · **31** community · **11** transcribe arch directories. "spec" = `model_specs/<family>.json` exists. transcribe arch rows retire one by one (state: [`docs/LEDGER.md`](docs/LEDGER.md) §2). Tiers A/B/C (PLAN SC5) will be added here once decided.

| Family | Kind | Files | Lines | spec |
|---|---|--:|--:|:-:|
| `ace_step` | core | 20 | 12.1k | ✓ |
| `apollo` | core | 3 | 536 | ✓ |
| `audio8_asr` | community | 4 | 1.09k | ✓ |
| `audio8_tts` | community | 8 | 5.97k | ✓ |
| `audiosr` | core | 8 | 2.81k | ✓ |
| `auk` | community | 5 | 2.01k | ✓ |
| `breeze_tts` | core | 7 | 4.69k | ✓ |
| `builtin_audio_utils` | core | 1 | 280 | ✓ |
| `canary` | transcribe arch | 9 | 4.18k |  |
| `canary_asr` | core | 4 | 929 | ✓ |
| `canary_qwen` | core | 7 | 3.73k | ✓ |
| `canary_qwen` | transcribe arch | 9 | 3.35k | ✓ |
| `chatterbox` | core | 22 | 9.12k | ✓ |
| `chatterbox_turbo` | community | 9 | 1.85k | ✓ |
| `citrinet_asr` | core | 3 | 1.23k | ✓ |
| `cohere` | transcribe arch | 9 | 4.29k |  |
| `cohere_asr` | core | 4 | 946 | ✓ |
| `confucius4_r2t2` | community | 8 | 2.49k | ✓ |
| `confucius4_tts` | core | 10 | 4.79k | ✓ |
| `controlfoley` | core | 6 | 3.44k | ✓ |
| `cosyvoice3` | core | 7 | 2.53k | ✓ |
| `demucs` | core | 5 | 2.63k |  |
| `dots_tts` | core | 13 | 8.81k | ✓ |
| `dramabox` | core | 9 | 4.92k | ✓ |
| `echo_tts` | community | 6 | 2.38k | ✓ |
| `f5_tts` | community | 6 | 3.14k | ✓ |
| `firered_audio` | core | 10 | 4.44k | ✓ |
| `fireredtts3` | core | 7 | 3.63k | ✓ |
| `fish_audio` | core | 10 | 3.19k | ✓ |
| `fun_asr_nano` | core | 9 | 2.71k | ✓ |
| `gigaam` | core | 5 | 2.17k | ✓ |
| `gigaam` | transcribe arch | 11 | 2.97k | ✓ |
| `glm_tts` | community | 8 | 2.82k | ✓ |
| `granite` | transcribe arch | 13 | 5.21k |  |
| `granite_nar` | core | 6 | 2.83k | ✓ |
| `granite_nar` | transcribe arch | 11 | 3.04k | ✓ |
| `granite_speech` | core | 7 | 2.53k | ✓ |
| `granite5asr` | community | 4 | 1.2k | ✓ |
| `heartmula` | core | 6 | 5.32k | ✓ |
| `higgs_audio_stt` | core | 8 | 2.58k | ✓ |
| `higgs_audio_tts` | core | 9 | 4.95k | ✓ |
| `hviske_asr` | core | 6 | 3.13k | ✓ |
| `index_tts2` | core | 13 | 8.74k | ✓ |
| `inflect_v2` | community | 4 | 2.53k | ✓ |
| `irodori_tts` | core | 7 | 6.03k | ✓ |
| `kitten_tts` | community | 6 | 4.81k | ✓ |
| `kokoro_tts` | core | 8 | 6.35k | ✓ |
| `kroko_asr` | community | 7 | 3.86k | ✓ |
| `liveavatar` | community | 12 | 10.7k | ✓ |
| `magpie_tts` | core | 5 | 3.15k | ✓ |
| `marblenet_vad` | core | 3 | 1.07k |  |
| `meanvc2` | core | 7 | 3.38k | ✓ |
| `medasr` | core | 5 | 1.69k | ✓ |
| `medasr` | transcribe arch | 7 | 2.37k | ✓ |
| `midashenglm_gen` | core | 7 | 2.05k | ✓ |
| `minimax_h3` | community | 9 | 6.19k | ✓ |
| `minimax_music3` | community | 11 | 4.99k | ✓ |
| `miocodec` | core | 8 | 2.81k | ✓ |
| `miotts` | core | 5 | 2.2k | ✓ |
| `mira_tts` | community | 7 | 2.25k | ✓ |
| `mms_forced_aligner` | community | 5 | 1.31k | ✓ |
| `moonshine` | core | 4 | 1.76k | ✓ |
| `moonshine_asr` | core | 4 | 1.98k | ✓ |
| `moonshine_streaming` | core | 4 | 2.55k | ✓ |
| `moss` | core | 15 | 4.23k |  |
| `moss` | transcribe arch | 11 | 3.52k |  |
| `moss_transcribe_diarize` | core | 2 | 1.22k | ✓ |
| `moss_tts_v15` | community | 4 | 732 | ✓ |
| `moss_voicegen` | community | 4 | 619 | ✓ |
| `muscriptor` | core | 5 | 2.82k | ✓ |
| `nemotron_3_diar` | core | 5 | 1.48k | ✓ |
| `nemotron_asr` | core | 7 | 3.03k | ✓ |
| `neutts` | core | 6 | 1.18k | ✓ |
| `niagara_asr` | core | 6 | 1.23k | ✓ |
| `omnivoice` | core | 10 | 7.68k | ✓ |
| `outetts` | community | 5 | 3.03k | ✓ |
| `parakeet` | transcribe arch | 10 | 9.44k |  |
| `parakeet_tdt` | community | 6 | 4.15k | ✓ |
| `personaplex` | core | 5 | 2.16k | ✓ |
| `piper_tts` | community | 4 | 2.15k | ✓ |
| `pocket_tts` | core | 13 | 6.37k | ✓ |
| `pulsevad` | core | 2 | 349 | ✓ |
| `qwen3_asr` | core | 9 | 3.6k | ✓ |
| `qwen3_forced_aligner` | core | 3 | 842 | ✓ |
| `qwen3_tts` | core | 11 | 5.61k | ✓ |
| `roformer` | core | 3 | 1.79k |  |
| `rvc` | core | 9 | 3.4k | ✓ |
| `sanotts` | community | 6 | 3.66k | ✓ |
| `seed_vc` | core | 10 | 5.33k | ✓ |
| `sense_asr` | community | 4 | 1.61k | ✓ |
| `sheetsage` | core | 4 | 3.77k |  |
| `silero_vad` | core | 3 | 1k | ✓ |
| `soprano_tts` | community | 7 | 2.88k | ✓ |
| `sopro_tts` | community | 9 | 4.86k | ✓ |
| `sortformer` | transcribe arch | 5 | 1.76k |  |
| `sortformer_diar` | community | 7 | 1.88k | ✓ |
| `sortformer_diar` | core | 7 | 3.21k | ✓ |
| `stable_audio` | core | 14 | 7.07k | ✓ |
| `supertonic` | core | 5 | 3.2k | ✓ |
| `universr` | core | 5 | 1.01k | ✓ |
| `vevo2` | core | 9 | 5.64k | ✓ |
| `vibeasr` | community | 4 | 1.7k | ✓ |
| `vibevoice` | core | 12 | 7.91k | ✓ |
| `vibevoice_asr` | core | 10 | 7.17k | ✓ |
| `vieneu_v3_turbo` | community | 12 | 6.83k | ✓ |
| `voxcpm1` | community | 7 | 6.5k | ✓ |
| `voxcpm2` | core | 8 | 5.65k | ✓ |
| `voxtral` | core | 6 | 2.99k | ✓ |
| `voxtral` | transcribe arch | 9 | 3.59k | ✓ |
| `voxtral_realtime` | core | 7 | 3.84k | ✓ |
| `whisper` | core | 5 | 2.84k | ✓ |
| `yue2` | core | 9 | 3.45k | ✓ |
| `zipvoice` | community | 7 | 3.29k | ✓ |

## 5. Largest files (by tokens)

Big files are where review and merge cost concentrate; think twice before growing them.

| File | Lines | Tokens |
|---|--:|--:|
| `src/framework/text/unicode_nfkd_data.inc` | 22.3k | 517k |
| `tests/moss_tts_v15/reference/ref_hidden_en_instruction.json` | 1 | 119k |
| `tests/moss_voicegen/reference/ref_hidden_en_radio_voice.json` | 1 | 81.2k |
| `tests/transcribe/dr_wav.h` | 6.94k | 72.7k |
| `patches/ggml/0003-cpu-kernels-for-fork-only-ops.patch` | 5.04k | 66.5k |
| `CMakeLists.txt` | 5.43k | 61.1k |
| `tests/warmbench.py` | 5.98k | 60k |
| `src/framework/audio/detail/speech_fft_internal.h` | 3.85k | 44.5k |
| `src/runtime/arch/parakeet/model.cpp` | 3.48k | 40.6k |
| `tools/audiocpp_cli/audiocpp_cli_path_cases.json` | 4.17k | 40.5k |
| `src/community_models/audio8_tts/ar.cpp` | 3.15k | 39.4k |
| `src/community_models/liveavatar/pipeline_liveavatar.cpp` | 3.46k | 39.1k |
| `src/runtime/transcribe.cpp` | 3.78k | 37.5k |
| `src/community_models/minimax_h3/dit_denoiser.cpp` | 3.08k | 36.8k |
| `app/server/runtime.cpp` | 3.69k | 35.6k |
