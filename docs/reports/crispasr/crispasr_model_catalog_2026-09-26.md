# CrispASR — model-family catalog and overlap with speech.cpp

> **Date:** 2026-09-26 · **Trees:** `CrispASR@dfcdacae`, `speech.cpp@d04fd552` plus the uncommitted working tree
> **Sources:** `src/core/arch_backend_map.h`, `src/crispasr_model_registry.cpp` (253 rows), `docs/feature-matrix.md` (131 backend names, auto-generated; README's "119" is stale), `PERFORMANCE.md` (Kaggle CUDA sweeps 2026-06-03 / 06-20), `models/convert-*.py`.
> **Caveats:**
> - "Absent in speech.cpp" is based on a keyword grep plus directory lists, not a full code read.
> - Commit counts (total/last 30 days) come from `git log -- <file>` without `--follow`, so renamed files are undercounted.
> - "JFK" is the 11 s `samples/jfk.wav` smoke clip. It is **not** a WER benchmark.

---

## 1. ASR (36 runtimes)

| Family (aliases) | Task | Upstream | Runtime LOC | Streaming | Evidence | Activity total/30d | In speech.cpp? |
|---|---|---|---|---|---|---|---|
| whisper (+tiron, experimental) | ASR, translation, LID | OpenAI Whisper, distil | 10385 | example | byte-identical to whisper.cpp | 39/3 | ✓ |
| parakeet (+reazonspeech, quds-fa, orukeet, ultra, redux, ja, rnnt) | TDT/RNNT | nvidia parakeet-* | 4203+801 | chunked | FLEURS sweeps, MAES beam study | 90/5 | ✓ (fewer fine-tunes) |
| nemotron | cache-aware RNNT streaming | nemotron-3.5-asr-streaming-0.6b | 3586 | **native** | JFK 9.1% | 69/11 | ✓ nemotron_asr |
| canary | ASR plus translation (25 languages) | canary-1b-v2 | 2453 | chunked, LCS merge | boundary duplicates at 60 s | 52/0 | ✓ (more variants) |
| canary-qwen | SALM | canary-qwen-2.5b | 1467 | – | live test | 15/0 | ✓ |
| fastconformer-ctc (stt_xx hybrid ×19 languages, canary-ctc aligner) | CTC, forced alignment | nvidia stt_*_fastconformer_hybrid | 1394 | – | JFK 0% | 29/0 | partial (parakeet-ctc, citrinet) |
| gigaam | ASR, Russian | ai-sage/GigaAM-v3 | 1392 | – | byte-identical to PyTorch | 4/0 | ✓ |
| dolphin | ASR, zh dialects | DataoceanAI dolphin | 860 | offline | none | 2/2 (new) | ✗ |
| xasr | Zipformer2 transducer, zh/en | X-ASR-zh-en | 1081 | **native** | none | 4/4 (new) | ✗ (but kroko_asr is Zipformer2) |
| cohere (+ja, ar) | ASR, LID probe | cohere-transcribe-03-2026 | 3711 | chunked | "lowest EN WER" claim | 87/0 | ✓ (base only) |
| granite 4.0/4.1/plus | ASR plus AST | granite-speech | 2857 | – | JFK 0% | 81/1 | ✓ |
| granite-4.1-nar | non-autoregressive | granite-speech-4.1-2b-nar | 2095 | – | JFK 0% | 26/1 | ✓ |
| voxtral | audio-LLM ASR | Voxtral-Mini-3B | 1622 | 30 s chunks | clean long-form | 38/2 | ✓ |
| voxtral4b | realtime | Voxtral-Mini-4B-Realtime | 2694 | incremental encoder (decode is chunked) | stream = batch | 54/1 | ✓ voxtral_realtime |
| qwen3 (+1.7b, mega-asr, ja-anime, raon-9B **NC**, r2t2 custom licence) | ASR, 30 languages | Qwen3-ASR | 2611 | r2t2 session | encoder cos 0.99953 | 55/7 | ✓ (+confucius4_r2t2) |
| higgs-stt | LLM ASR | higgs-audio-v3-stt | 2360 | – | live | 14/0 | ✓ |
| wav2vec2 / hubert / data2vec (13-language aligners) | CTC ASR, alignment | xlsr-53 etc. | 2148 | – | JFK 0–4.5% | 57/0 | ✗ (only MMS aligner) |
| glm-asr | LLM ASR zh/en/yue | GLM-ASR-Nano-2512 | 1673 | – | matches HF | 41/1 | ✗ |
| kyutai-stt (1b, 2.6b) | Mimi plus LM | kyutai stt | 1768 | streaming-capable | JFK 0% | 29/1 | ✗ |
| firered-asr (+firered-lid, 120 languages) | AED zh/en | FireRedASR2-AED | 3019 | – | decoder has no KV cache | 73/3 | ✗ |
| moonshine (+de fine-tunes **NC-SA**) | ASR | UsefulSensors | 1562 | – | JFK 9.1% | 35/0 | ✓ |
| moonshine-streaming | streaming | moonshine-streaming | 1387 | **native** | JFK 0% | 27/0 | ✓ |
| gemma4-e2b / e4b | ASR plus translation (Gemma terms) | gemma-4-E2B/E4B-it | 2749 | – | JFK 9.1% | 51/0 | ✗ |
| omniasr (CTC and LLM, 1600+ languages) | ASR | facebook omniASR | 1787 | 15 s segment protocol | JFK 4.5–22.7% | 46/0 | ✗ |
| mimo-asr | Qwen2-7.5B plus RVQ | MiMo-V2.5-ASR | 2010+1054 | – | JFK 0% | 41/0 | ✗ |
| ark-asr ⚠ WIP | ASR, 19 languages | ARK-ASR-3B | 1320 | – | placeholder URL | 17/0 | ✗ (skip) |
| moss-audio | audio understanding plus ASR | MOSS-Audio-4B | 2364 | – | JFK 0% | 40/0 | ✗ |
| hojo-asr | ASR, 9 languages | Hojo-ASR | 2280 | – | none | 10/10 (new) | ✗ |
| moss-transcribe | ASR zh/en | MOSS-Transcribe-2B | 1638 | – | live | 15/0 | ✗ (diarize variant ✓) |
| moss-diarize | ASR plus diarization | MOSS-Transcribe-Diarize | 1654 | – | none | 9/0 | ✓ |
| funasr (+mlt-nano) | LLM ASR | Fun-ASR-Nano | 2520 | – | JFK 0% | 38/2 | ✓ |
| paraformer | NAR CIF zh | funasr/paraformer-zh | 1132 | – | JFK 0% | 21/1 | ✗ |
| sensevoice | ASR, LID, events | SenseVoiceSmall | 938 | – | 17–20× RT | 16/2 | ✓ sense_asr |
| vibevoice-asr (+streaming, bitnet) | ASR plus diarization | microsoft VibeVoice-ASR | 6135 (shared) | **native** | JFK 4.5% | 103/3 | ✓ (+vibeasr) |
| lfm2-audio | ASR, TTS, S2S (LFM Open) | LFM2.5-Audio-1.5B | 2812 | – | live | 38/0 | ✗ |
| mini-omni2 | ASR, TTS, S2S | gpt-omni/mini-omni2 | 1568 | – | Q4_K = F16 | 12/0 | ✗ |

## 2. TTS, voice cloning and voice conversion

**Both repos have these** (17+): qwen3-tts, chatterbox (+turbo), cosyvoice3, omnivoice, voxcpm2, pocket-tts, irodori, dots-tts, fireredtts3, confucius4-tts, supertonic, kokoro, piper, f5-tts, moss-tts-local / v1.5, breeze-tts-2, vibevoice-tts, miotts. rvc also overlaps, as voice conversion.
- These overlap families are a **three-implementation cross-check opportunity**, not port work.
- **Version differences:** outetts is 0.3 in CrispASR vs 1.0 in ours; indextts is 1.5 vs our index_tts2.

**Only in CrispASR (candidates):**

| Family | Notes | LOC | Licence flag |
|---|---|---|---|
| orpheus (+DE) | 3B LM + SNAC; **CUDA failing** as of 06-20 | 1231 | Llama 3.2 |
| csm (sesame) | Mimi | 2478 | – |
| dia | dialogue TTS + DAC | 2228 | – |
| zonos | + DAC; very active (12 commits/30d) | 3174 | – |
| bark | .npz prompts | 2427 | – |
| parler-tts | prompt-described voice | 2219 | – |
| speecht5 | x-vector | 1610 | – |
| fastpitch | NVIDIA 60M | 1253 | – |
| melotts + openvoice2 | TTS + tone-colour voice conversion | 3048+1434 | – |
| tada (1b / 3b) | + codec + encoder | 3624+ | Llama 3.2 |
| kugelaudio | 7B + DiT | 1855 | – |
| voxtral-tts | 20 presets | 1933 | **NC** |
| bananamind-tts | 13M | 1561 | – |
| raon-opentts | via f5 runtime | – | **NC** |
| sidon | speech restoration | 1461 | MIT |
| voxcpm2-vae | audio upscaler | (in voxcpm2) | – |
| beatrice | voice conversion, **WIP stub** — skip | 538+866 | – |

## 3. Non-ASR, non-TTS (mostly absent from speech.cpp — new task kinds)

| Task | CrispASR families | speech.cpp today |
|---|---|---|
| **Text punctuation / capitalization / segmentation** | fireredpunc, fullstop (XLM-R), punctuate-all, **PCS** (47 languages) | ✗ |
| **Truecasing** | statistical, CRF, BiLSTM (German-focused) | ✗ |
| **Audio LID** | whisper-tiny, silero-lid (95), **ecapa-lid** (107), firered-lid (120), cohere probe | partial (LID inside some ASR) |
| **Text LID** | CLD3 (109), GlotLID-V3 (2102), fastText-176 (**NC**) | ✗ |
| **Text MT** | m2m100 (418M; wmt21 4.7B), madlad (T5-3B, 419 languages) | ✗ |
| **Forced alignment** | canary-ctc, qwen3-forced-aligner, wav2vec2 ×13, fastconformer ×19 | qwen3_forced_aligner, mms_forced_aligner |
| **Diarization** | sortformer (nemotron-3), pyannote-seg, **foxnose** (WeSpeaker + clustering, 7.3% DER VoxConverse dev), TitaNet / ECAPA embedders, speaker_db | nemotron_3_diar, sortformer_diar |
| **VAD** | silero, firered-vad (F1 97.6%), marblenet, whisper-vad-encdec (experimental), webrtc | silero, marblenet, pulsevad |
| **Separation** | htdemucs, mel-band-roformer | demucs, roformer (+bs) |
| **Music analysis** | crepe (F0), btc-chords (**NC-SA**), tabcnn (tabs), beat-this, basic-pitch, piano-transcription, onsets-and-frames (69.0% note F1), hft-transformer (70.5%), mt3 | muscriptor, sheetsage only |
| **Watermark / provenance** | AudioSeal + spread-spectrum + C2PA | ✗ |
| **Enhancement** | RNNoise | ✓ (more: deepfilternet2, gtcrn, zipenhancer, flashsr, audiosr, apollo, universr) |
| **Codecs** | SNAC, DAC, Mimi, WavTokenizer, MioCodec, TADA, MOSS, MiMo tokenizer; **glint** (MIT clean-room MP3/AAC/FLAC library) | codecs in framework; no MP3/AAC encoder |
| **Text chat LLM** | vendored llama.cpp chat ABI | ✗ (out of scope — see rethink §5) |

## 4. Only in speech.cpp (we already lead here)

- **ASR:** citrinet, hviske, niagara, medasr, kroko (Zipformer2), audio8, granite5asr, multitalker-parakeet, firered_audio.
- **TTS:** fish_audio, dramabox, higgs_audio_tts, magpie, neutts, glm_tts, zipvoice, echo, kitten, inflect_v2, sanotts, soprano, sopro, mira, vieneu, voxcpm1, audio8_tts, moss_tts_nano, moss_voicegen, auk.
- **Speech-to-speech:** personaplex.
- **Voice conversion:** seed_vc, meanvc2.
- **Restoration:** audiosr, apollo, universr, deepfilternet2, gtcrn, zipenhancer, flashsr.
- **Music and generation:** ace_step, heartmula, yue2, stable_audio, minimax_music3 / h3, midashenglm_gen, controlfoley, vevo2, liveavatar.
- **Other:** muscriptor, sheetsage, pulsevad, bs_roformer.

**Union estimate [inf]:** about 102 speech.cpp families plus about 60 CrispASR-only families gives **~160 families**. That is the realistic ceiling for "one runtime for all local speech".

## 5. Licence flags (must reach our model_specs as metadata)

- **Non-commercial:** quds, raon-speech / opentts, voxtral-tts, bt2-tts, btc-chords, moonshine-de, outetts-0.3, lid-fasttext176.
- **Custom or restricted:** confucius4 (Youdao), lfm2 (LFM Open), gemma4 (Gemma terms), orpheus / tada / kartoffel (Llama 3.2), pocket-tts (gated-use, anti-impersonation), supertonic (OpenRAIL-M), FunASR (attribution).
- **Attribution:** NVIDIA (CC-BY-4.0), wespeaker, tabcnn.

## 6. Pre-quantized GGUF ecosystem

CrispASR publishes ~344 GGUF URLs under 188 `cstr/*` HF repos. Their layout is llama.cpp-style: `general.architecture` plus `<arch>.*` hparams plus an embedded `tokenizer.*`.
- Accepting those files through a per-family **foreign-layout TensorSource map** would give users instant downloads. speech.cpp already does this for transcribe GGUFs; the parakeet_tdt dual layout is the precedent.
- The files are **unpinned** (`resolve/main`). If we accept them, we must pin revision plus sha256 in our specs.
