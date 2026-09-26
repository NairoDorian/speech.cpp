# Strategy: what speech.cpp should become, and why anyone would choose it

> **Date:** 2026-09-26 · **Status:** RECOMMENDATIONS (the user asked: "if you had to make up to ~5 changes midcourse, what would they be?"). Each change is marked in [`PLAN.md`](../../PLAN.md) §8 as **proposed**; confirm or reject them there.
> **Context read:**
> - speech.cpp (checkpoint audit), audio.cpp, transcribe.cpp, CrispASR;
> - the consumer: **S2B2S** (`../../../STT_BRAIN_TTS/S2B2S`), to be replaced by **FreeSpeech** with speech.cpp as its only speech backend.

---

## 1. The honest starting point

**Today, speech.cpp is not a reason to switch away from audio.cpp.** Look at what a developer sees:
- audio.cpp has ~100 families.
- It is now adding ASR families itself (canary, cohere, moss-transcribe, nemotron).
- It ships its own C ABI (`include/audiocpp.h`), a server, a WebUI, and releases.
- It moves faster: about 485 commits since August 1.

What speech.cpp adds on top: transcribe.cpp's arches re-hosted on the same engine, a second (legacy) C ABI, and a stricter test culture. That is a **better-tested audio.cpp with transcribe.cpp's models**. It is a reasonable fork, but not a destination.

If the plan stays "port everything from both parents, then CrispASR", speech.cpp remains a strict superset with a larger merge burden. **Breadth is not a moat.** audio.cpp can add any model we add, and CrispASR already has more.

**The moat has to be what audio.cpp is not trying to be: an embeddable runtime for real-time voice applications.** That is exactly what FreeSpeech needs.

## 1b. User directives (V7-D7, 2026-09-26): two profiles, two primary consumers

**DEV and BUNDLE profiles:**
- The **DEV** profile keeps *everything*, for maximum functionality and every way to test the models: audio.cpp's WebUI and server, CLI, tools, tests, benches.
- The **BUNDLE** profile is what apps ship: an embeddable library like transcribe.cpp. It is stripped to the selected families, down to a single STT or TTS family, the way transcribe.cpp's `TRANSCRIBE_MODEL_SET` / Cargo features shrink its library.

**The consumers:**
- **FreeSpeech desktop** (**primary target #1**) succeeds **ZER0** (`Handy_V2`: a Handy fork on the transcribe-cpp crate with **Multi-STT**, up to 9 concurrent STT engines merged by concat or LLM, plus live mode and Earshot VAD) and **S2B2S**.
- **FreeSpeech Android** (`Android_FreeSpeech`) is **primary target #2**. Today it is a Rust cdylib + JNI on the transcribe-cpp crate with `minimal-multilingual`: IME, RecognitionService, live subtitles, R2T2 streaming with committed and tentative partials. It will grow toward desktop features (multi-STT, TTS, a brain via LiteRT-LM / Gemma 4, which stays outside speech.cpp).
- speech.cpp is meant for **both**: desktop first, Android second.

**Consequences:**
- Android is a first-class build target (PLAN E6).
- Model-set composites become Cargo features (E0).
- Many concurrent sessions sharing one device is a gated contract (SC4).
- The `transcribe.h` / `transcribe-cpp` crate shim lets both apps switch backend without code changes.

## 2. What FreeSpeech needs (evidence from S2B2S)

S2B2S is a Tauri app (Rust + React), forked from Handy. Today it assembles its speech stack from five different technologies:

| Need | How S2B2S does it today | Pain point (from its STATUS / TAKEOVER docs) |
|---|---|---|
| STT | `transcribe-cpp` Rust crate (git dependency on transcribe.cpp) | Streaming STT is "Partial: chunk boundary token edges" |
| Native TTS / audio models | spawns an **audio.cpp server process** (`src-tauri/src/audiocpp_server/`) | Process lifecycle, IPC, ports |
| More TTS (Qwen3-TTS, KittenTTS) | a **Python venv** provisioned at onboarding (torch, CUDA graphs) | "Python venv fragility"; Nix/cross-platform rated C+ |
| VAD | in-app TripleVAD: RMS → RNNoise → **Silero ONNX** | a third inference runtime (onnxruntime) |
| Conversation loop | Rust: VAD → STT → LLM (llama.cpp server) → TTS, with barge-in | "Continuous voice: limited echo cancellation"; "wake word: VAD-energy only" |
| Resources | GPU/VRAM and RAM indicators in the UI | STT + TTS + VAD + LLM compete for one GPU with no shared budget |

**The implication:** FreeSpeech wants **one in-process library** with a Rust crate that does all of the following. No Python, no ONNX runtime, no side-car server:
- STT: offline, plus streaming with stable partials.
- TTS: offline, plus streaming with low time to first audio.
- Voice cloning and voice design.
- VAD, turn detection, wake word and echo cancellation.
- Barge-in and cancellation.
- Model management with a shared resource budget.

**speech.cpp already holds most of the models.** It has qwen3_tts, kitten_tts, kokoro, silero_vad, pulsevad, marblenet_vad, the rnnoise-class enhancers, 20+ ASR families and streaming sessions. What is missing is the **product layer** that turns them into that library.

## 3. Would PRs to audio.cpp achieve the same result?

| Goal | PRs to audio.cpp only | speech.cpp fork |
|---|---|---|
| transcribe.cpp's ASR families on the engine | **Yes**, if accepted. audio.cpp is already absorbing ASR on its own. | Yes, but every family is merge burden forever. |
| transcribe.cpp's contracts (struct_size ABI, 4-state streaming, limits, exception containment) | **Partly.** audio.cpp just designed its own C ABI. A competing ABI discipline is a hard sell to its maintainer. | Yes, fully under our control (`speech.h`, V7-D1). |
| Parity and quality culture (pinned oracles, WER gates, verdicts) | Partly: tests can be contributed, but gates are only enforced if the maintainer wants them. | Yes. |
| Real-time conversation runtime, AEC, wake word, resource manager, Rust crate for FreeSpeech | **Unlikely.** That is app-runtime scope, outside a model-zoo project's charter. | **Yes: this is the reason to exist.** |
| Pace and priorities driven by FreeSpeech | No | Yes |

**Conclusion: hybrid, "upstream-first core, differentiated product layer".**
- Everything generic goes to audio.cpp as PRs: model families, engine fixes, framework primitives such as the ASR layer (`framework/asr`), numeric bug fixes, and tests. That shrinks our merge burden, earns goodwill, and makes audio.cpp's developers *our* developers, because they review and maintain the shared core.
- speech.cpp keeps **only** what audio.cpp should not own: `speech.h` + bindings, the real-time conversation runtime, the resource manager, the curated and verified catalogue, and the published scoreboard. These live in **separate directories**, so the fork stays mostly additive and merges stay cheap.

This is the llama.cpp / Ollama relationship, with one difference: a fork plus upstream PRs instead of a separate repo, so we can still patch the core when we must.

## 4. Seven recommended changes

### SC1 — Design the API from the app, not from the parents (contract first, L1)
- Write `docs/api/speech_h_spec.md` from FreeSpeech's use cases *before* implementing S3:
  - sessions for STT, TTS, VAD and conversation;
  - push-audio streaming in both directions;
  - stable-partial semantics;
  - cancellation and barge-in;
  - an event callback model (partials, finals, speech start/end, TTS chunks);
  - a model manager (list, download, verify, load, unload, residency);
  - errors.
- transcribe.h's discipline stays (struct_size, typed ext slots, status codes). The surface is shaped by the consumer.
- **Change to the plan:** add S1.8 "speech.h spec from FreeSpeech use cases", and make S3 implement that spec.

### SC2 — Thin-fork topology, with the fork delta as a measured number
- speech.cpp-only code moves under `src/speech/` (capi, realtime, resources) and `include/speech/`.
- Upstream-owned files are edited only by patches we intend to upstream, and each such edit is logged.
- `scripts/status.sh` reports **lines changed in upstream-owned files** (`git diff upstream/main -- <upstream paths>`). That number must go *down* over time.
- Every generic fix gets an upstream PR candidate note in the LEDGER.
- *Why:* merge cost is what kills forks. Today the delta grows every week (checkpoint §1).

### SC3 — A real-time conversation runtime in C++ (the differentiator)
A `ConversationSession` in `src/speech/realtime/` that does what S2B2S does in Rust today, natively and with gates:
- **Input:** mic frames → **AEC** → **VAD** → **wake word / KWS** (optional) → **streaming ASR** with stable partials → **turn detection / endpointing** (a semantic end-of-turn model, not only silence).
- **Output:** text from the host's LLM (FreeSpeech keeps llama.cpp; we expose a callback, V7-D4) → **streaming TTS** with sentence-level chunking → playback buffer.
- **Barge-in:** user speech during TTS cancels synthesis within one frame and reports what was actually spoken.
- **Echo cancellation:** a far-end reference from the TTS output feeds AEC. This is the hard, valuable part; S2B2S lists it as a gap.
- **Latency gates** (new; the first latency gates in this project), per backend, in the scoreboard:
  - time-to-first-partial;
  - end-of-turn → final;
  - text → first TTS audio;
  - barge-in reaction time.
- No ggml project has this. It is what makes a front-end developer pick speech.cpp over assembling five runtimes.

### SC4 — A resource manager (many models, one GPU)
- A process-wide `ResourceManager`: a model residency cache, a VRAM and RAM budget with eviction policy, shared backend and device contexts across sessions, warm-up and preload, and truthful memory reporting (FreeSpeech shows VRAM in its footer).
- It makes concurrent sessions a tested contract (acceptance A23 `concurrent_sessions_test`; it doesn't exist yet).
- audio.cpp just added peak-memory metrics (`1bfb9c54`); build on them.

### SC5 — A curated catalogue in three tiers, prioritized by FreeSpeech
- **Tier A — "supported"** (~15–25 families):
  - parity gate + WER or round-trip gate + latency numbers + binding tests;
  - **these are what FreeSpeech ships**;
  - every new port first answers: *does FreeSpeech need it?*
- **Tier B — "community"**: builds, quick smoke test, no guarantees.
- **Tier C — "archived"**: WIP, redundant or NC-by-default; kept out of the default build.
- **First Tier-A targets:**
  - replace S2B2S's Python TTS: qwen3_tts, kitten_tts, kokoro, plus a voice-cloning family;
  - its VAD: silero / pulsevad;
  - its STT: parakeet, whisper, moonshine_streaming, qwen3_asr, voxtral_realtime;
  - one S2S or dialogue family for later.
- The ~160-family union (rethink §4) is the long tail, not the roadmap.

### SC6 — Three test tiers and a published scoreboard
- **Quick** (default ctest): one short clip per model, seconds per test. **Done 2026-09-26** for the ASR verdict/parity tests; extend the pattern to every model-backed test.
- **Full** (`-DSPEECHCPP_FULL_ASR_TESTS=ON`, label `full`): the corpus, batched passes, all modes, before a merge or release.
- **Nightly bench:** generates `docs/scoreboard.md`: WER / RTF / peak VRAM / latency per Tier-A family per backend (CPU, CUDA, Vulkan).
- The scoreboard is the public trust signal audio.cpp does not offer: "every number on this page is regenerated nightly by a gate".

### SC7 — Build: generated from family manifests, and pluggable
- A `family.json` next to each package generates the `audiocpp_add_model` blocks, alias tables, arch map and capability table. This removes the ~68 KB hand-written registration section of `CMakeLists.txt`. The test section follows.
- Medium term: families as loadable plugins (like `GGML_BACKEND_DL`), so FreeSpeech ships one small core library and downloads model plugins with their weights.
- Lowest priority of the seven; do it when the upstream topology (SC2) is settled.

## 5. Why a developer would pick speech.cpp (the pitch this plan must make true)

> **speech.cpp is the embeddable voice runtime.** One small in-process C library that bundles only the models you pick (plus Rust, Python and TypeScript bindings) that runs every local speech model you need, and the real-time conversation loop around them: echo-cancelled, barge-in-aware, streaming in both directions, sharing one GPU budget. Every supported model has a published, nightly-regenerated accuracy and latency score. It tracks audio.cpp and transcribe.cpp upstream, so you get their models too.

Each clause maps to a change:

| Clause | Change |
|---|---|
| "one C library + bindings" | S3 + SC1 + Phase 13 (Rust first) |
| "real-time loop" | SC3 |
| "one GPU budget" | SC4 |
| "published score" | SC6 |
| "tracks upstream" | SC2 + L13 |

## 6. Revised order (proposed, replaces PLAN §4 order after S2)

```
S0 stabilize → S1 hygiene (+S1.8 speech.h spec, SC1) → S2 shared ASR layer
 → S3 speech.h (implements the spec) + retire our audiocpp.h
 → F1 Rust crate `speech-sys`/`speech` + FreeSpeech integration spike (replace transcribe-cpp crate,
      audio.cpp server and Python TTS in one app build)                         ← proves the product
 → R1 realtime ConversationSession (VAD, streaming ASR, endpointing, TTS streaming, barge-in)
 → R2 AEC + wake word + latency gates;   RM resource manager (SC4) in parallel
 → Tier-A catalogue completion (SC5) with scoreboard (SC6)
 → 11b/11c arch retirement continues underneath; C0–C3 CrispASR mining only for Tier-A needs
 → SC7 manifests/plugins
```

Upstream PRs (SC2) run continuously from S0 onward: every generic fix is offered upstream as it lands.

## 7. What NOT to do
- Don't race audio.cpp or CrispASR on model count.
- Don't build a text LLM runtime (FreeSpeech keeps llama.cpp).
- Don't let the DEV tools (server, WebUI, CLI) leak into the BUNDLE profile. Keep them fully working for testing, strictly above the library (V7-D7).
- Don't put audio-device I/O in the library. The host app owns mic and speaker; the library takes and returns PCM.
- Don't keep porting families FreeSpeech doesn't need until Tier A is solid.
