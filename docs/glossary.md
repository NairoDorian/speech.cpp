# Glossary: speech, audio and speech.cpp vocabulary, metrics and benchmarks

> A reference for humans and agents working on speech.cpp.
>
> - The **rules** for measuring are in [`benchmarking.md`](benchmarking.md) (rule R10).
> - This file explains **what the words and numbers mean**.
> - Terms specific to this repository are marked **(speech.cpp)**.

**Contents:**
1. Audio basics
2. Model families and architectures
3. Inference and runtime
4. Streaming and real-time conversation
5. speech.cpp project terms
6. Metrics (how they are computed and how to read them)
7. Benchmarks and datasets, by task
8. Quick "which metric for which task" table

---

## 1. Audio basics

| Term | Meaning |
|---|---|
| **PCM** | Raw audio samples. speech.cpp's library API takes **mono float32 PCM in [-1, 1]**. ASR models almost all expect **16 kHz**. TTS output is often 22.05 / 24 / 44.1 / 48 kHz. |
| **Sample rate** | Samples per second (Hz). Resampling converts between rates. Quality matters: linear interpolation aliases (about −10 dB rejection), polyphase is about −89 dB (LESSONS D11). |
| **Frame / hop / window** | Short-time analysis slices audio into overlapping **windows** (e.g. 25 ms) advanced by a **hop** (e.g. 10 ms). One hop gives one **frame** of features. |
| **STFT / iSTFT** | Short-time Fourier transform: time-frequency representation. The inverse (iSTFT) turns a spectrogram back into audio; many vocoders end with it. |
| **Mel spectrogram / log-mel** | STFT magnitudes pooled into mel-scale bands (typically 80 or 128), then log-compressed. The standard ASR input feature. **Layout trap:** mel-major vs frame-major (LESSONS D3). |
| **Fbank (Kaldi fbank)** | Log mel filterbank computed the Kaldi way (pre-emphasis, povey window, dither). Used by SenseVoice, Paraformer, FunASR and others. Not interchangeable with Whisper or NeMo mels. |
| **CMVN / per-feature normalization** | Mean/variance normalization of features: global (CMVN stats), per utterance, or per feature (NeMo `per_feature`). Must match the training recipe exactly. |
| **dB, dBFS, LUFS** | Level units. dBFS is relative to digital full scale; LUFS is perceived loudness (used to normalize TTS output). |
| **SNR** | Signal-to-noise ratio. Noisy test sets are grouped by SNR. |
| **Clip / utterance / segment** | A **clip** is an audio file. An **utterance** is one spoken unit with a reference transcript. A **segment** is a time span of output (with start/end times). |

## 2. Model families and architectures

### 2.1 Speech recognition (STT / ASR)

**STT** (speech-to-text) and **ASR** (automatic speech recognition) are used interchangeably.

| Term | Meaning |
|---|---|
| **Encoder** | Turns acoustic features into hidden states (Conformer, FastConformer, Zipformer, Transformer, Whisper encoder). Usually most of the compute on long audio. |
| **Conformer / FastConformer** | Convolution-augmented Transformer blocks. FastConformer (NVIDIA) subsamples 8× early for speed. It is the encoder behind Parakeet, Canary, Nemotron and Sortformer. |
| **Zipformer** | k2/icefall encoder with multi-rate stacks; used by streaming transducers (e.g. kroko, X-ASR). |
| **CTC** | *Connectionist Temporal Classification.* The encoder emits one label per frame (including *blank*); collapse repeats and blanks to get text. Fast, non-autoregressive, no language-model context. Also used for **forced alignment**. |
| **Transducer / RNN-T** | Encoder + **prediction network** (a small LM over previous tokens) + **joint network**. Streams naturally. **TDT** (Token-and-Duration Transducer, Parakeet-TDT) also predicts how many frames to skip, which makes it faster. |
| **AED / encoder–decoder** | Attention encoder–decoder: an autoregressive text decoder cross-attends to the encoder (Whisper, Canary, Cohere, FireRed-AED). Strong accuracy; decoding is sequential. |
| **Speech-LLM / audio-LLM / SALM** | An audio encoder + a projector into an LLM that decodes text (Qwen3-ASR, Voxtral, Granite-Speech, Canary-Qwen). Handles instructions, translation and context; heavier. |
| **NAR / CIF** | Non-autoregressive decoding: all tokens at once. **CIF** (continuous integrate-and-fire, Paraformer) predicts token boundaries from the encoder. |
| **Hybrid** | A model with two heads (e.g. TDT + CTC). Either can decode. |
| **Language model (LM) fusion / hotwords / context biasing** | Steering decoding toward domain words (external LM, n-gram, or a boosted hotword list). |
| **Decoding: greedy / beam / MAES** | **Greedy** takes the best token each step. **Beam search** keeps the k best hypotheses. **MAES** is a beam search variant for transducers. |
| **Speculative decoding** | A cheap drafter (n-gram lookup or a small model) proposes tokens that the big model verifies in one pass. Same output, faster. |
| **ITN / PNC** | *Inverse text normalization* ("twenty three" → "23") and *punctuation and capitalization*. They change WER unless normalized away (§6.1). |
| **Timestamps** | **Segment**, **word** or **token** start/end times. Produced natively (TDT/CTC) or by **forced alignment** (CTC aligner, Qwen3-ForcedAligner). |
| **LID** | *Language identification*, spoken (audio) or written (text). |
| **Diarization** | "Who spoke when": speaker turns labelled SPK0, SPK1… (Sortformer, pyannote, clustering of speaker embeddings). |
| **Speaker embedding** | A fixed vector summarizing a voice (ECAPA-TDNN, TitaNet, WavLM, CAM++). Used for diarization, verification and voice cloning. |
| **VAD** | *Voice activity detection*: speech vs non-speech per frame (Silero, MarbleNet, PulseVAD, FireRedVAD). |

### 2.2 Speech generation (TTS) and conversion

| Term | Meaning |
|---|---|
| **TTS** | Text-to-speech. Pipeline stages: **text normalization** → **G2P** (grapheme-to-phoneme) or direct text tokens → **acoustic model** → **vocoder / codec decoder** → audio. |
| **G2P / phonemizer** | Converts spelling to phonemes (espeak-ng (GPL; opt-in here), rule tables, neural G2P). |
| **Acoustic model** | Predicts intermediate acoustics: mel frames (FastPitch, VITS-style) or **codec tokens** (LM-based TTS). |
| **Vocoder** | Mel → waveform (HiFi-GAN, BigVGAN, Vocos). |
| **Neural audio codec / RVQ** | Compresses audio into discrete **tokens** using **residual vector quantization** codebooks (EnCodec, DAC, SNAC, Mimi, WavTokenizer). LM-based TTS predicts these tokens. |
| **AR vs NAR TTS** | **Autoregressive** TTS generates tokens one by one (Orpheus, CosyVoice LM stage, Qwen3-TTS). **Non-autoregressive** TTS generates in parallel (FastPitch, VITS/Piper, Kokoro). |
| **Flow matching / diffusion** | Iterative generative refinement over **N steps** (F5-TTS, CosyVoice flow stage, VibeVoice diffusion head, Stable Audio). Fewer steps is faster but lower quality. |
| **Voice cloning / zero-shot TTS** | Speak in a new voice from a short **reference clip** (plus its transcript for some models). |
| **Voice design** | Create a voice from a text description ("warm, low, British"). |
| **Voice conversion (VC) / SVC** | Change the voice of existing speech, or singing (RVC, Seed-VC, OpenVoice). |
| **Prosody** | Rhythm, stress and intonation. Duration, pitch (**F0**) and energy. |
| **S2S / speech-to-speech** | Speech in, speech out, directly (duplex dialogue models) or via ASR → LLM → TTS (the cascade FreeSpeech uses). |

### 2.3 Other audio tasks

| Term | Meaning |
|---|---|
| **Enhancement / denoising** | Remove noise or reverb (RNNoise, DeepFilterNet, GTCRN). |
| **Source separation** | Split a mix into stems: vocals, drums, bass, other (HTDemucs, BS/Mel-Band RoFormer). |
| **Super-resolution / bandwidth extension** | Restore high frequencies (AudioSR, FlashSR). |
| **Watermarking / provenance** | Imperceptible marks in generated audio (AudioSeal), signed metadata (C2PA). |
| **Music transcription** | Audio → notes/MIDI (onsets, offsets, pitch). Also chords, beats and tabs. |

## 3. Inference and runtime

| Term | Meaning |
|---|---|
| **ggml** | The C tensor library speech.cpp runs on. **Backends:** CPU, CUDA (NVIDIA), Vulkan (any GPU: NVIDIA, Intel, AMD, mobile), Metal (Apple), HIP (AMD). |
| **GGUF** | ggml's single-file model format: tensors + metadata (`general.architecture`, hyper-parameters, tokenizer). **A container, not a universal adapter:** each family needs its own tensor layout map. |
| **Quantization (F32, F16, BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ*, TQ*)** | Storing weights in fewer bits. `Q8_0` = 8-bit blocks (near-lossless). `Q4_K_M` = 4-bit k-quant, a medium mix. Smaller and often faster, but accuracy can drop, and audio towers are sensitive (keep them ≥ Q8, LESSONS D7). **Always benchmark the quant you ship.** |
| **Weights vs activations; mmap** | Weights are the stored parameters. Activations are the intermediate values. **mmap** maps weights from the file into memory lazily (lower RSS; important on Android). |
| **Graph / cached graph** | ggml builds a compute graph, then runs it. Reusing a built graph saves rebuild time, but **every input must be re-uploaded each run** (LESSONS D1). |
| **Scheduler / allocator (sched, gallocr)** | ggml components that place graph nodes on backends and assign memory. **(speech.cpp)** They are owned by the framework, never by families (R5). |
| **KV cache** | Stored attention keys/values of past tokens, so an autoregressive decoder does not recompute them. **(speech.cpp)** The shared `EncDecKVCache` / decoder KV only; no private copies (R5). |
| **Batch / batched decode** | Processing several utterances in one pass. Throughput goes up; per-item latency may go up. |
| **Threads** | CPU worker threads. **Can change numeric results** through reduction order (whisper-tiny did), so benchmarks fix the count. |
| **Warm-up / cold start** | The first run pays one-time costs: loading, graph build, kernel JIT or compile, caches. **Run 1 is the warm-up** (benchmarking.md §2). |
| **Offload / residency** | Which models or tensors live on the GPU vs CPU, and which models stay loaded. A future `ResourceManager` (SC4) handles this for many concurrent models. |

## 4. Streaming and real-time conversation

| Term | Meaning |
|---|---|
| **Offline vs streaming ASR** | **Offline** transcribes a complete clip. **Streaming** takes audio incrementally (**feeds** / **chunks**) and emits results as it goes. |
| **Chunk size / lookahead / left context** | Streaming models process fixed chunks (e.g. 160–560 ms) with limited **right context** (lookahead: more is more accurate but adds latency) and cached **left context**. |
| **Cache-aware streaming** | The encoder keeps per-layer caches between chunks instead of re-encoding (Nemotron and Parakeet streaming). |
| **Partial vs final; committed vs tentative text** | A **partial** is an interim hypothesis that may change. **Committed** (stable) text will not change; **tentative** text still can. The Android IME shows tentative text as composing text. |
| **Endpointing / turn detection / EOU** | Deciding that the user **finished a turn** (end of utterance): silence-based, or a model predicting semantic end-of-turn. |
| **Barge-in** | The user starts speaking while TTS is playing. The system must stop synthesis quickly. |
| **AEC** | *Acoustic echo cancellation*: removes the device's own TTS playback (the **far-end reference**) from the mic signal, so the ASR does not transcribe the assistant. |
| **Wake word / KWS** | Keyword spotting to start listening ("hey …"). |
| **Streaming TTS** | Emitting audio chunks before the whole utterance is synthesized, measured by **time to first audio**. |

## 5. speech.cpp project terms

| Term | Meaning |
|---|---|
| **Family** | One model architecture: a canonical id (e.g. `parakeet_tdt`) plus its **variants** (sizes, languages) and quants. |
| **Engine package** | A family implemented on the audio.cpp engine: `src/models/<f>/` or `src/community_models/<f>/`, plus `include/engine/...` and `model_specs/<f>.json`. The target form of every family. |
| **Arch** | A transcribe.cpp family implementation under `src/runtime/arch/<f>/`. **Retires** after its **verdict** (docs/LEDGER.md §2). |
| **Verdict** | `abi_arch_engine_verdict`: the same GGUF run through the arch and through the engine via the C ABI, comparing every result field. |
| **Parity test** | Engine output vs a reference (the arch, the C ABI, or PyTorch) on the same input. **Proves equality, not quality or speed**; that is what benchmarks are for. |
| **Oracle / reference / golden** | **Oracle:** the original implementation (usually PyTorch) at a pinned revision. **Golden:** stored expected output. **Tolerance file:** allowed numeric error per tensor. |
| **Quick / full tier** | ctest tiers. **Quick** (default) loads each model and runs one short clip. **Full** (opt-in) runs the corpus, batches and all modes. |
| **DEV / BUNDLE profile** | **DEV** is the full build with every family, the server, WebUI, CLI and tests. **BUNDLE** is the embeddable library with only the selected families (V7-D7). |
| **Composite / model set** | A named family selection for BUNDLE builds (e.g. `android-stt`), exposed as Cargo features. |
| **C ABI / shim** | The C interface apps link (`transcribe.h` today, `speech.h` target). A **shim** keeps an old ABI working on top of the new one. |
| **TaskKind** | The engine's task type: `asr`, `tts`, `vad`, `diar`, `align`, `lid`, … Each has its own request/result contract. |
| **Parents** | audio.cpp (git fork base), transcribe.cpp (the author's fork) and CrispASR (reference only). See PLAN *North star*. |

---

## 6. Metrics

### 6.1 Recognition accuracy

**WER: word error rate.** The core ASR metric.

```
WER = (S + D + I) / N
  S = substituted words, D = deleted words, I = inserted words
  N = number of words in the reference
```

It is computed from the minimum edit (Levenshtein) alignment between the hypothesis and the reference.

- **Lower is better.** 0 % is perfect. It **can exceed 100 %**, because insertions are unbounded; hallucinated long outputs do this.
- **Normalization decides the number.** Before scoring, both texts are normalized: lowercase, strip punctuation, unify numbers and spellings (e.g. the Whisper English normalizer). Different normalizers give different WERs on the same output, so **only compare WERs computed with the same normalizer**. speech.cpp uses one scorer per benchmark: transcribe.cpp `scripts/wer/score.py`, and `tests/asr_test_text.h` for the in-tree LibriSpeech gate.
- **Corpus vs average WER.** **Corpus (micro) WER** sums all errors over all words (long utterances weigh more). **Average (macro) WER** averages per-utterance or per-language WERs. Always say which. In speech.cpp, per-language scores are corpus WER within the language; the multilingual headline is the macro average over languages.
- **Small corpora are coarse.** On the 69-word LibriSpeech quick corpus, one word = 1.45 points. It is a tripwire, not a benchmark.
- **Rule of thumb for clean read English:** below 3 % is excellent, 5–10 % is usable, above 15 % needs a reason. Noisy, accented or conversational speech scores much higher; compare only on the same data.

Related accuracy metrics:

| Metric | Formula / meaning | Use |
|---|---|---|
| **CER**: character error rate | Same formula over characters | Languages without spaces (Chinese, Japanese, Thai) and very short outputs. FLEURS zh/ja/yue/th use CER. |
| **MER / WIL** | Match error rate / word information lost; bounded variants of WER | Rarely reported; ignore unless a paper uses them. |
| **SER**: sentence error rate | Fraction of utterances with any error | Strictness check (commands, dictation). |
| **Token flips** **(speech.cpp)** | Tokens that differ between two runs or implementations of the same model | Parity diagnostics; any flip should be explained (float ties, threads). |
| **Streamed-vs-offline divergence** **(speech.cpp)** | Word edits between the streaming result and the offline result of the same clip | Streaming quality (the gate allows ≤ 3 words; target 0). |
| **Partial revision rate / flicker** | How often tentative text changes before it commits | Streaming UX. Lower means a calmer IME and subtitles. |

### 6.2 Speed and latency

**RTF: real-time factor.** The core speed metric.

```
RTF = processing time / audio duration
```

- **Lower is better.** RTF 0.1 means 10 s of audio take 1 s. **RTF < 1 is faster than real time** (required for live use).
- **RTFx** (inverse RTF, used by the Hugging Face Open ASR Leaderboard) = audio duration / processing time. RTF 0.05 = RTFx 20 ("20× real time"). Bigger is better. **Don't mix RTF and RTFx in one table.**
- **What the time includes must be stated:**
  - **(a) inference only**, with the model already loaded (the default for benchmark cells);
  - **(b) end to end**, including feature extraction and decoding;
  - **(c) with load**.

  speech.cpp reports (a) as RTF mean over runs 2..N, and reports **load time** and **run-1 (cold) RTF** separately.
- RTF depends on hardware, backend, threads, quant and clip length. Compare only like with like (same GGUF, machine, backend and threads: R10).

| Metric | Meaning | Where it matters |
|---|---|---|
| **Load time** | Open the GGUF and set up weights and backends | App start, model switching (ZER0 Multi-STT loads many models) |
| **Cold start / run-1 latency** | The first inference after load, including graph build and kernel JIT | The first dictation after launch |
| **Throughput** | Audio seconds processed per wall second across concurrent streams or batches | Server, Multi-STT, file transcription |
| **Time to first partial** (streaming ASR) | Speech onset → first partial text | Live captions, the IME |
| **Finalization latency** (streaming ASR) | End of speech → final text | Dictation responsiveness |
| **TTFT / TTFA** (TTS) | Time to first token / **first audio chunk** after the text arrives | Conversational feel. A key streaming TTS metric |
| **Tokens/s** (autoregressive decoders) | Decoding speed of speech-LLM or TTS-LM tokens | Diagnosing where time goes |
| **Barge-in reaction time** | User speech onset → TTS stopped | Real-time conversation (SC3) |
| **End-to-end turn latency** | End of user turn → first reply audio (ASR + LLM + TTS) | The number FreeSpeech users feel |

**Noise floor:** timing varies run to run. The spread of runs 2..N defines what counts as a real difference (≥ 2 %; benchmarking.md §1).

### 6.3 Resources

| Metric | Meaning |
|---|---|
| **Peak RSS** | Peak resident host memory. Android's low-memory killer watches this. |
| **Peak VRAM** | Peak GPU memory. It decides how many models fit at once (Multi-STT). |
| **Library / bundle size** | Size of the shipped `.so` / `.dll` for a composite (BUNDLE profile). |
| **Model size** | GGUF bytes per quant. |

### 6.4 Speech generation (TTS) quality

| Metric | Meaning | Notes |
|---|---|---|
| **MOS**: mean opinion score | Human 1–5 ratings of naturalness | The gold standard; expensive. **CMOS/SMOS** compare two systems or rate similarity. |
| **UTMOS / DNSMOS / NISQA** (predicted MOS) | Neural predictors of MOS | Cheap and automatic; good for regressions, not absolute claims. |
| **Round-trip / intelligibility WER** | Synthesize text, transcribe it with a **fixed** strong ASR, compute WER/CER against the input text | speech.cpp's first TTS gate (C0.2). The ASR model must be fixed and stated, because its own errors are included. |
| **SIM: speaker similarity** | Cosine similarity between speaker embeddings of the reference voice and the generated speech (e.g. WavLM-large fine-tuned for verification, as in the Seed-TTS eval) | Voice cloning quality. Report the embedding model. |
| **F0 RMSE / V/UV error** | Pitch accuracy vs a reference | Prosody, singing |
| **MCD**: mel-cepstral distortion | Spectral distance to a reference recording (dB) | Only meaningful with a time-aligned reference |
| **TTFA, RTF** | See §6.2 | Streaming TTS |

### 6.5 Diarization, VAD, LID, alignment

| Metric | Meaning |
|---|---|
| **DER**: diarization error rate | `(missed speech + false alarm + speaker confusion) / total speech time`. Usually scored with a 0.25 s **collar** around boundaries; state whether overlap is scored. Lower is better. |
| **JER**: Jaccard error rate | A per-speaker overlap metric (DIHARD); less dominated by long speakers. |
| **VAD precision / recall / F1, ROC-AUC** | Frame-level speech detection quality. Also **onset/offset delay**, which matters for endpointing. |
| **LID accuracy / top-k** | Fraction of clips whose language is identified correctly, with a confusion matrix for close languages. |
| **Word boundary error / alignment accuracy** | Mean absolute error (ms) of predicted word start/end vs reference, or % within a tolerance (e.g. 50 ms). |

### 6.6 Enhancement, separation, translation, music, text

| Metric | Task | Meaning |
|---|---|---|
| **PESQ, STOI / ESTOI** | Enhancement | Perceptual quality (−0.5 to 4.5) and intelligibility (0–1) vs the clean reference |
| **SI-SDR / SDR** (dB) | Enhancement, separation | Signal-to-distortion ratio; higher is better. **SDR** (museval) is the separation standard. |
| **LSD** | Super-resolution | Log-spectral distance to the full-band reference |
| **BLEU, chrF, COMET** | Translation (speech→text) | n-gram overlap, character F-score, and a learned quality metric (COMET correlates best with humans) |
| **Punctuation / capitalization F1** | Text post-processing | Per-mark precision/recall |
| **Note F1** (onset, and onset+offset) | Music transcription | mir_eval note matching with tolerance windows |
| **Chord accuracy (WCSR)** | Chords | Weighted chord symbol recall (MIREX) |
| **Beat F-measure** | Beats | Beat hits within ±70 ms |

### 6.7 Numeric parity (implementation correctness, not model quality)

| Metric | Meaning |
|---|---|
| **max_abs / p99_abs / mean_abs** | Largest / 99th-percentile / mean absolute difference between two tensors |
| **Relative RMS** | RMS of the difference divided by RMS of the reference |
| **Cosine similarity** | Direction agreement. **Scale-invariant: always pair it with a magnitude check** (LESSONS D8). |
| **Tolerance (V6-D9)** | `1e-4 × p99_abs` / `1e-5 × rms` defaults, per tensor, in `tests/tolerances/<family>.json` |

---

## 7. Benchmarks and datasets by task

**What speech.cpp uses** is marked **→ used**; the rest are reference points for reading papers and leaderboards. Check each dataset's licence before redistributing audio (most are research-only or CC-BY).

### 7.1 ASR

| Dataset | Content | Notes |
|---|---|---|
| **FLEURS** (google/fleurs) | Read Wikipedia sentences, **102 languages**, ~12 h per language | **→ used**: speech.cpp's multilingual benchmark (`core6` subset: en, fr, de, es, ru, zh (CER)). Also used for LID and speech translation. CC-BY-4.0. |
| **LibriSpeech** test-clean / test-other | Read English audiobooks; *clean* vs harder *other* | **→ used**: English benchmark, plus the in-tree 4-clip smoke corpus. The most quoted English number (a "2 %" test-clean WER is modern-model territory). |
| **Common Voice** | Crowd-sourced read speech, 100+ languages, many accents and mics | Robustness to accents and devices |
| **MLS** (Multilingual LibriSpeech) | Audiobooks in 8 languages | Long-form, multilingual read speech |
| **VoxPopuli** | European Parliament speeches | Spontaneous-ish, multilingual |
| **TED-LIUM 3** | TED talks | Long-form, prepared speech |
| **AMI** (IHM / SDM) | Meetings (close-talk / distant mic) | Conversational + far-field; hard; also diarization |
| **Earnings-21/22, SPGISpeech** | Financial calls | Domain vocabulary, numbers, long-form |
| **GigaSpeech** | Podcasts and YouTube (English) | Diverse, large |
| **CHiME-6/7/8** | Dinner-party far-field | Very hard, multi-speaker |
| **AISHELL-1/2, WenetSpeech** | Mandarin | CER |
| **ReazonSpeech, CSJ** | Japanese | CER |
| **Golos, Russian LibriSpeech (RuLS)** | Russian | GigaAM's home turf |
| **HF Open ASR Leaderboard** | Average WER over AMI, Earnings22, GigaSpeech, LibriSpeech clean/other, SPGISpeech, TED-LIUM, VoxPopuli, plus **RTFx** | A common public ranking of English ASR; multilingual tracks exist. Reports **RTFx**, not RTF. |

**Streaming ASR** is evaluated on the same sets, fed in chunks. Report the chunk size and lookahead with the WER.

### 7.2 TTS

| Benchmark | Content | Metrics |
|---|---|---|
| **Seed-TTS eval** (test-en, test-zh, test-hard) | Zero-shot cloning prompts + target texts | WER (Whisper-large-v3 for en, Paraformer for zh) + SIM (WavLM-large) |
| **LibriSpeech / LibriTTS test-clean** prompts | English cloning and continuation | WER, SIM, UTMOS |
| **Multilingual sets** (MiniMax multilingual, FLEURS texts) | Many languages | Round-trip WER/CER, SIM |
| **EmergentTTS-Eval and similar** | Hard prosody, emotion, questions | LLM-judged or human-judged |

**speech.cpp's TTS gate** (benchmarking.md §3, C0.2): fixed sentences per language → round-trip WER/CER with one fixed ASR model, plus RTF and TTFA. SIM is added for cloning families.

### 7.3 Other tasks

| Task | Standard sets |
|---|---|
| Diarization | AMI, **VoxConverse**, CALLHOME, DIHARD III (DER, JER) |
| VAD | AVA-Speech, plus custom noisy / far-field sets (F1, detection delay) |
| LID | FLEURS-LID, VoxLingua107 (accuracy) |
| Speech translation | FLEURS X→en, CoVoST 2 (BLEU, COMET) |
| Enhancement | VoiceBank+DEMAND, DNS Challenge (PESQ, STOI, SI-SDR, DNSMOS) |
| Separation | MUSDB18-HQ (SDR) |
| Music transcription | MAESTRO (piano), GuitarSet (tabs), Isophonics / Billboard (chords), GTZAN / Ballroom (beats) |

---

## 8. Which metric for which task (speech.cpp defaults)

| Task | Accuracy | Speed / latency | Resource |
|---|---|---|---|
| Offline ASR | WER (CER for zh/ja) on the FLEURS subset + LibriSpeech | RTF (runs 2..N), load time, run-1 latency | peak RSS / VRAM |
| Streaming ASR | WER + streamed-vs-offline divergence + partial revision rate | time to first partial, finalization latency, RTF | peak RSS / VRAM |
| TTS | round-trip WER/CER (+ SIM for cloning; UTMOS when available) | RTF, TTFA (streaming) | peak RSS / VRAM |
| VAD | F1, onset/offset delay | RTF (tiny) | — |
| Diarization | DER (state the collar and overlap handling) | RTF | peak memory |
| Alignment | word boundary error (ms) | RTF | — |
| Conversation loop (SC3) | end-to-end task success | end-of-turn → first audio, barge-in reaction | total VRAM of resident models |
