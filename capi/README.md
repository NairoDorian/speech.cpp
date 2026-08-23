# audio.cpp C ABI (`libaudiocpp`)

A minimal C interface exposing audio.cpp's model loading, inference, and
streaming capabilities to foreign-language runtimes (Rust FFI, Python
ctypes, Go cgo, etc.).

## Quick Start

### Build

The C ABI shared library is built when `AUDIOCPP_BUILD_CAPI=ON` (default).
CI produces prebuilt binaries for 8 platform/backend combinations:

| Artifact | Platform | Backend |
|---|---|---|
| `audiocpp-windows-cuda-cpu.zip` | Windows | CUDA + CPU fallback |
| `audiocpp-windows-rocm-cpu.zip` | Windows | ROCm (HIP) |
| `audiocpp-windows-sycl-cpu.zip` | Windows | Intel SYCL |
| `audiocpp-windows-vulkan-cpu.zip` | Windows | Vulkan + CPU fallback |
| `audiocpp-linux-cuda-cpu.zip` | Linux | CUDA + CPU fallback |
| `audiocpp-linux-rocm-cpu.zip` | Linux | ROCm (HIP) |
| `audiocpp-linux-sycl-cpu.zip` | Linux | Intel SYCL |
| `audiocpp-linux-vulkan-cpu.zip` | Linux | Vulkan + CPU fallback |

Each zip contains:
- `audiocpp.dll` / `libaudiocpp.so` — the shared library
- `audiocpp.lib` — Windows import library (not needed for runtime loading)
- `audiocpp.h` — the C header

**Build options** (CMake):
- `AUDIOCPP_BUILD_CAPI=ON` — build the shared library (default ON).
- `AUDIOCPP_EMBED_VAD_ASSETS=ON` — bake silero_vad + marblenet_vad weights into
  the binary (+~2 MB) so VAD works with no external asset files. With this on,
  `audiocpp_load_model(NULL, "silero_vad", ...)` loads from the embedded bytes.
  Off by default — without it, ship `assets/framework/models/{silero,marblenet}_vad/`.
- `AUDIOCPP_EMBED_AUDIO_UTILITIES=ON` — bake deepfilternet2/rnnoise/zipenhancer/
  flashsr weights into the binary (+~28 MB) so `audiocpp_denoise`/
  `audiocpp_super_resolve` work with `model_path = NULL`. Off by default —
  without it, ship `assets/framework/audio_utilities/` and pass the model dir.

### Local Build

#### Windows (CUDA, local GPU)

```powershell
# Build audiocpp.dll for your local NVIDIA GPU
.\scripts\build_windows.ps1 `
    -Preset windows-cuda-release `
    -Target audiocpp `
    -CudaArchitectures "120-real" `
    -DeploymentBuild `
    -Ccache

# Output: build/windows-cuda-release/bin/audiocpp.dll
```

Parameters:
- `-CudaArchitectures "120-real"` — target sm_120 (Blackwell RTX 50-series).
  Use `"89-real"` for RTX 40-series, `"86-real"` for RTX 30-series.
  Multiple: `"75;80;86;89;120"` (matches CI config).
- `-DeploymentBuild` — embed model specs into the DLL (required for standalone
  deployment without external `model_specs/` directory).
- `-Ccache` — enable ccache for faster rebuilds.

#### Windows (CPU only)

```powershell
.\scripts\build_windows.ps1 `
    -Preset windows-cpu-release `
    -Target audiocpp `
    -Ccache

# Output: build/windows-cpu-release/bin/audiocpp.dll
```

#### Linux

```bash
# CUDA
scripts/build_linux.sh --backend cuda --target audiocpp

# CPU
scripts/build_linux.sh --backend cpu --target audiocpp
```

### Minimal Usage (Rust via libloading)

```rust
use libloading::Library;

let lib = Library::new("audiocpp.dll")?;

// Enumerate devices → pick GPU
let device_count: Symbol<unsafe extern "C" fn() -> i32> =
    unsafe { lib.get(b"audiocpp_device_count") }?;

// Load model
let model = load_model(path, "qwen3_asr", TASK_ASR, BACKEND_CUDA, 0, 4, &mut err);

// Run ASR
let text = asr(model, pcm, n_samples, 16000, "en", &mut err);
println!("{}", text.text);
```

---

## API Reference

The library exports **46 functions** across these categories:

### 1. Backend & Device Selection

#### Enums

```c
// Backend type
AUDIOCPP_BACKEND_CPU    = 0   // CPU (always available)
AUDIOCPP_BACKEND_CUDA   = 1   // NVIDIA CUDA (also AMD ROCm/HIP, MUSA)
AUDIOCPP_BACKEND_VULKAN = 2   // Vulkan (NVIDIA/AMD/Intel/Apple-MoltenVK)
AUDIOCPP_BACKEND_METAL  = 3   // Apple Metal (macOS/iOS)
AUDIOCPP_BACKEND_SYCL   = 4   // Intel oneAPI SYCL
AUDIOCPP_BACKEND_BEST   = 5   // auto-select best available

// Device type (mirrors ggml_backend_dev_type)
AUDIOCPP_DEVICE_CPU   = 0
AUDIOCPP_DEVICE_GPU   = 1
AUDIOCPP_DEVICE_IGPU  = 2
AUDIOCPP_DEVICE_ACCEL = 3   // hardware accelerator (e.g. Apple Metal)
AUDIOCPP_DEVICE_META  = 4   // meta device (ggml internal, rare)
```

#### Functions

| Function | Description |
|---|---|
| `audiocpp_backend_available(backend)` | 1 if `backend` is compiled in **and** has a device at runtime (CPU always 1; `AUDIOCPP_BACKEND_BEST` = any GPU available). Never aborts. |
| `audiocpp_device_count()` | Count all compute devices across compiled backends |
| `audiocpp_device_info(index, *out)` | Get device name, backend, memory, per-backend device_id |
| `audiocpp_list_devices()` | Print all devices to stdout (convenience) |

**Device selection flow**: enumerate → find GPU → pass `backend` + `device_id`
to `audiocpp_load_model`. The `device_id` from `audiocpp_device_info` is
per-backend (e.g. `device_id=0` for the first CUDA GPU, independent from
the first SYCL GPU). For a `--backend gpu` request, resolve it with
`audiocpp_backend_available()` (CUDA → Vulkan → SYCL → Metal → fall back to
CPU) instead of parsing the backend tag from `audiocpp_version()`. The
`device_id` to pass is 0 for the first device of the resolved backend.

---

### 2. Model Lifecycle

```c
audiocpp_model_t *audiocpp_load_model(
    const char *model_path,       // model directory or GGUF file
    const char *family_hint,      // e.g. "qwen3_asr", NULL = auto-detect
    int task,                     // AUDIOCPP_TASK_*
    int backend,                  // AUDIOCPP_BACKEND_*
    int device_id,                // GPU index (from device enumeration)
    int n_threads,                // CPU threads (0 = auto)
    audiocpp_error_t *err         // optional error output
);
void audiocpp_free_model(audiocpp_model_t *model);
```

**Session options**: `audiocpp_load_model_ex` is the same call plus a
session-options JSON argument. Session options are fixed at session creation
and apply to every request on the handle — unlike request options (per-call
`{"voice_ref": ...}` passed to `audiocpp_tts`). Use them for model-family
configuration that the engine reads at load time, e.g.:

```c
audiocpp_model_t *model = audiocpp_load_model_ex(
    "D:/models/miotts-1.7b-q8_0.gguf", "miotts", AUDIOCPP_TASK_TTS,
    AUDIOCPP_BACKEND_CUDA, 0, 8,
    "{\"miotts.codec_model_path\": \"D:/models/miocodec-25hz-44khz-v2-q8_0.gguf\"}",
    &err);
```

Recognized keys are family-specific: `miotts.codec_model_path`,
`miotts.best_of_n_asr_model_path`, `qwen3_asr.vad_model_path` (model
dependency paths), `*.weight_type` / `*.mem_saver` / `*.perf_mode` /
`*.audio_tokenizer_*_graph_arena_mb` (weight precision and memory/performance
policy). Values render like request options: JSON integers become `"123"`,
floats keep their fraction, booleans become `"true"`/`"false"`. Malformed
JSON is a hard load error (never silently ignored).

**Important**: Always pass `family_hint` when known. Without it, the loader
iterates all 35 loaders' `can_load` probes, which is slower and may hit
side effects from incompatible models probing the same directory.

---

### 3. Task Types

All 13 engine task kinds are exposed:

```c
AUDIOCPP_TASK_VAD   = 0    // Voice Activity Detection
AUDIOCPP_TASK_ASR   = 1    // Speech-to-Text
AUDIOCPP_TASK_DIAR  = 2    // Speaker Diarization
AUDIOCPP_TASK_SEP   = 3    // Source Separation
AUDIOCPP_TASK_GEN   = 4    // Audio/Music Generation
AUDIOCPP_TASK_TTS   = 5    // Text-to-Speech
AUDIOCPP_TASK_ALIGN = 6    // Forced Alignment
AUDIOCPP_TASK_VC    = 7    // Voice Conversion
AUDIOCPP_TASK_CLON  = 8    // Voice Cloning (TTS + speaker reference)
AUDIOCPP_TASK_S2S   = 9    // Speech-to-Speech (codec-based)
AUDIOCPP_TASK_VDES  = 10   // Voice Design (prompt-based)
AUDIOCPP_TASK_SPK   = 11   // Speaker Recognition
AUDIOCPP_TASK_SVC   = 12   // Singing Voice Conversion
```

---

### 4. Inference Functions (Offline)

Each returns a result handle (or NULL on error). The caller owns the result
and must free it with the matching `audiocpp_free_*` function.

| Function | Task | Input | Output | Free with |
|---|---|---|---|---|
| `audiocpp_tts` | TTS | text | audio | `free_audio` |
| `audiocpp_tts_with_voice_ref` | TTS/Clone | text + inline PCM voice ref | audio | `free_audio` |
| `audiocpp_asr` | ASR | audio PCM | text + language | `free_text` |
| `audiocpp_diar` | Diar | audio PCM | speaker turns | `free_diar` |
| `audiocpp_vad` | VAD | audio PCM | speech segments | `free_vad` |
| `audiocpp_vad_energy` | VAD (model-free) | audio PCM | speech segments | `free_vad` |
| `audiocpp_denoise` | Denoise | audio PCM | denoised audio | `free_audio` |
| `audiocpp_super_resolve` | Super-resolve | audio PCM | upsampled audio | `free_audio` |
| `audiocpp_align` | Align | audio PCM + text + language | word timestamps | `free_align` |
| `audiocpp_audio_transform` | SEP/VC | audio PCM | single audio output | `free_audio` |
| `audiocpp_audio_transform_with_voice_ref` | VC | audio PCM + inline voice ref | audio | `free_audio` |
| `audiocpp_transform_stems` | SEP/GEN | audio PCM (+ voice ref) | **all** named stems (vocals/drums/bass/...) | `free_stems` |
| `audiocpp_generate` | GEN | text prompt | audio and/or artifacts (video) | `free_gen_result` |
| `audiocpp_midi` | MIDI | audio PCM | artifacts (MIDI / event JSON) | `free_artifacts` |
| `audiocpp_midi_from_wav` | MIDI | WAV file bytes | artifacts (MIDI / event JSON) | `free_artifacts` |

**`audiocpp_transform_stems`** returns all named audio outputs (unlike
`audiocpp_audio_transform` which only returns the first). Use this for
source separation models (demucs, roformer) that emit multiple stems.

**Text → audio generation** — `audiocpp_generate(model, prompt, options, err)`
runs a text-conditioned generator and returns the audio output **and** any
artifacts together in one `audiocpp_gen_result_t` (free with
`audiocpp_free_gen_result`). Either field may be NULL; generation is expensive
and may be non-deterministic, so audio and artifacts are produced by a single
`run()` rather than two calls.

- **MiniMax-H3** (load with `AUDIOCPP_TASK_GEN`, family `"minimax_h3"`): emits
  audio (`res->audio`) and, when options set `"return_video": true`, an RGB24
  video artifact (`AUDIOCPP_ARTIFACT_CUSTOM`, id `minimax_h3_video_rgb24`; meta
  `width`/`height`/`frames`/`fps` as decimal strings). Options: `width`,
  `height`, `num_frames`, `num_inference_steps`, `guidance_scale`, `seed`, ...
- Also drives `stable_audio` / `ace_step` (text → audio).

**Audio → MIDI transcription** — `audiocpp_midi(model, pcm, n, sr, options, err)`
feeds input audio to **MuScriptor** (an audio→MIDI transcriber — it does *not*
consume text; a text-only call throws `"requires audio_input"`) and returns the
produced artifacts (free with `audiocpp_free_artifacts`). Load MuScriptor with
`AUDIOCPP_TASK_MIDI`, family `"muscriptor"`. The default artifact is
`AUDIOCPP_ARTIFACT_MIDI` (mime `audio/midi`, extension `mid`); set
`"output_format":"json"` for event JSON. Options: `temperature`,
`guidance_scale`, `max_tokens`, `instruments` (comma-separated group names),
`seed`, ...

`audiocpp_midi_from_wav(model, wav_bytes, wav_size, options, err)` is a
convenience that takes the **raw bytes of a .wav file** (RIFF header + PCM)
instead of decoded PCM — no temp file. The bytes are decoded in-memory (PCM16 /
PCM24 / IEEE float32) and downmixed to mono. Use it when the caller already has
the WAV in memory (e.g. an HTTP upload):

```c
/* MuScriptor from WAV bytes (e.g. read from a request body) */
uint8_t *wav = <the .wav bytes>;  size_t wav_n = <byte count>;
audiocpp_model_t *m = audiocpp_load_model_ex(
    model_path, "muscriptor", AUDIOCPP_TASK_MIDI, backend, 0, 0, "{}", err);
audiocpp_artifacts_t *arts = audiocpp_midi_from_wav(
    m, wav, (int64_t)wav_n, "{\"output_format\":\"midi\"}", err);
if (arts) {                                       /* arts->artifacts[0] */
    audiocpp_artifact_t *a = &arts->artifacts[0]; /* a->payload = .mid bytes */
    FILE *f = fopen("out.mid", "wb");
    fwrite(a->payload, 1, (size_t)a->payload_size, f); fclose(f);
}
audiocpp_free_artifacts(arts);
audiocpp_free_model(m);

/* Or, if you already have decoded PCM (e.g. via audiocpp_read_wav): */
/* audiocpp_midi(m, pcm, n_samples, sample_rate, options, err); */
```

**Energy VAD (model-free)** — `audiocpp_vad_energy(pcm, n, rate, options, err)`
segments audio by signal energy only (no model load, no weights). It splits the
audio into ~`chunk_seconds`-length pieces, snapping each boundary to the
lowest-RMS window near the nominal split point (so cuts land in silences, not
mid-word). Faster than `audiocpp_vad` and needs no files, but less accurate on
noisy input. Options JSON keys: `chunk_seconds` (default 30), `boundary_seconds`
(default 2), `min_energy_seconds` (default 0.1). Returns the same
`audiocpp_vad_t` shape as the model VAD (free with `audiocpp_free_vad`).

**Embedded VAD weights** — when the library is built with
`-DAUDIOCPP_EMBED_VAD_ASSETS=ON`, silero_vad and marblenet_vad weights are
compiled into the binary and you can load them with `model_path = NULL`:
```c
// No file needed — uses baked-in weights
audiocpp_model_t *vad = audiocpp_load_model(
    NULL, "silero_vad", AUDIOCPP_TASK_VAD, AUDIOCPP_BACKEND_CPU, 0, 4, &err);
```
Adds ~2 MB to the binary. Without the flag, VAD still works but you must ship
the `assets/framework/models/{silero_vad,marblenet_vad}/` files alongside the
library and pass their path. `audiocpp_vad_energy` needs neither (it's pure
signal energy).

**Denoise / super-resolve** — `audiocpp_denoise(pcm, n, rate, model_name,
model_path, options, err)` runs one of three standalone audio-enhancement models
(`deepfilternet2` / `rnnoise` / `zipenhancer`) to reduce noise. `audiocpp_super_
resolve(pcm, n, rate, model_path, options, err)` upsamples narrowband audio to
wideband via `flashsr` (16k → 48k). These are NOT model-registry tasks — they
load directly. With `AUDIOCPP_EMBED_AUDIO_UTILITIES=ON` (~28 MB: deepfilternet2 8.9 + rnnoise 11.0 + zipenhancer 7.9 + flashsr 0.3), pass
`model_path = NULL` to use baked-in weights; otherwise pass the model directory
(rnnoise takes a file path). The input sample rate is arbitrary (resampled to the
model's native rate internally). `options_json` keys: `backend` ("cpu"|"cuda"|
"vulkan"|"metal"|"sycl"), `device` (index).

> **Models are cached**: `audiocpp_denoise` / `audiocpp_super_resolve` cache the
> loaded model process-wide keyed by (path, backend, device). The first call
> pays the load cost (8–11 MB weights + graph build + GPU upload); subsequent
> calls with the same config reuse the cached model. The cache holds weak
> references, so models are reclaimed when no caller holds a result.

> **Arbitrary-length input is safe**: each model handles long audio internally
> via built-in overlap-add (deepfilternet2/zipenhancer/flashsr) or per-frame
> processing (rnnoise). No need to chunk on the caller side — pass the whole
> recording.

**Batch TTS** — `audiocpp_tts_batch` synthesizes N texts in one session
(reuses a single `prepare()`, far cheaper than N separate `audiocpp_tts`
calls). Each text is synthesized independently; individual failures do not
abort the batch. Two merge modes:

```c
const char *texts[] = {"第一段。", "第二段。", "第三段。"};
int n = 3;

// Mode 1: independent — N separate audio buffers
audiocpp_audio_batch_t *b = audiocpp_tts_batch(
    model, texts, n, "{\"voice_ref\":\"ref.wav\",\"reference_text\":\"...\"}",
    AUDIOCPP_BATCH_MERGE_NONE, &err);
for (int i = 0; i < b->n_items; ++i) {
    // b->items[i].samples  → audio for texts[i] (or empty if that text failed)
}
audiocpp_free_audio_batch(b);

// Mode 2: concat — one merged audio + per-text sample ranges
b = audiocpp_tts_batch(model, texts, n, opts,
                       AUDIOCPP_BATCH_MERGE_CONCAT, &err);
// b->items[0].samples   → the full concatenation
// b->chapter_starts[i], b->chapter_ends[i]  → [start,end) for texts[i]
audiocpp_free_audio_batch(b);
```

Progress: with a callback installed, it fires at **request granularity**
(stage `"batch_tts"`, `completed/total = (text_index)/n`) — one update per
text, not per internal chunk. Free with `audiocpp_free_audio_batch`.

---

### 5. Progress Callback (Offline Run)

Install a callback to observe progress during any offline inference function
(`audiocpp_tts`, `audiocpp_asr`, `audiocpp_vad`, `audiocpp_diar`, ...). The
callback fires synchronously on the calling thread from inside `run()` — at
chunk boundaries for chunked models (TTS text-chunk / ASR audio-chunk), or once
at start/end for single-shot models. Returning non-zero from the callback
requests cancellation: the in-flight `run()` aborts and the triggering function
returns NULL with `err->message = "canceled by progress callback"`
(`err->code` stays 0 — cancellation is not a hard error).

```c
// Signature: return 0 to continue, non-zero to cancel
typedef int (*audiocpp_progress_fn)(float progress,
                                    const char *stage,
                                    int64_t completed_units,
                                    int64_t total_units,
                                    void *user_data);

// A simple progress printer
int my_progress(float progress, const char *stage,
                int64_t completed, int64_t total, void *user) {
    printf("[%s] %d%% (%lld/%lld)\n",
           stage, (int)(progress * 100),
           (long long)completed, (long long)total);
    return 0;  // 0 = continue, non-zero = cancel
}

// Install once after load_model; persists across runs until cleared/replaced
audiocpp_set_progress_callback(model, my_progress, NULL);

// Subsequent runs fire the callback:
audiocpp_audio_t *audio = audiocpp_tts(model, "long text...", NULL, &err);
// console during run:
//   [qwen3_tts] 0% (0/7)
//   [qwen3_tts] 14% (1/7)
//   ...
//   [qwen3_tts] 100% (7/7)

// Cancel a long job by returning 1 from the callback:
int cancel_after_3(float p, const char *s, int64_t c, int64_t t, void *u) {
    return c >= 3 ? 1 : 0;  // abort once 3 chunks done
}

// Clear: pass fn = NULL
audiocpp_set_progress_callback(model, NULL, NULL);
```

| Field | Meaning |
|---|---|
| `progress` | Completion fraction in `[0.0, 1.0]` |
| `stage` | Model family name, e.g. `"qwen3_tts"` / `"qwen3_asr"`. Valid only during the callback. |
| `completed_units` | Chunks completed so far (`0..total_units`) |
| `total_units` | Total chunks (`1` for single-shot models) |

**Coverage**: all 37 model families. The 21 chunk-loop models (16 TTS, 4 ASR,
2 source-separation) report per-chunk progress; the 16 single-shot models
report `0%` at start and `100%` at completion. Thread-safety: set the callback
before calling a run function and do not change it mid-run.

---

### 6. Streaming (Chunk-Push Model)

For real-time / low-latency processing (streaming VAD, streaming ASR). The
streaming session inherits the backend (CPU/CUDA/...) from the model handle —
it runs on the same device selected at `audiocpp_load_model` time.

```c
// 1. Start stream (creates a new streaming session)
audiocpp_stream_t *stream = audiocpp_stream_start(
    model, TASK_VAD, NULL, 512, &err);

// 2. Push audio chunks → get events synchronously
while (have_audio) {
    audiocpp_stream_event_t *ev = audiocpp_stream_push(
        stream, pcm_chunk, chunk_len, 16000, &err);
    // ev->va_events[]      → VAD events (speech start/end)
    // ev->partial_text     → ASR partial transcript
    // ev->audio_samples    → streaming TTS output
    audiocpp_free_stream_event(ev);
}

// 3. Finish → get final result
audiocpp_text_t final;
audiocpp_stream_finish(stream, &final, &err);
audiocpp_free_text(&final);
audiocpp_stream_free(stream);
```

| Function | Description |
|---|---|
| `audiocpp_stream_start` | Create streaming session (RunMode::Streaming) |
| `audiocpp_stream_push` | Push audio chunk, get event (VAD/ASR). No-op for TTS (input=None). |
| `audiocpp_stream_pull` | Pull next generated event (streaming TTS). Use this, not push, for TTS. |
| `audiocpp_stream_finish` | End stream, get final accumulated result |
| `audiocpp_free_stream_event` | Free event returned by push/pull |
| `audiocpp_stream_free` | Free stream handle |

**Three streaming patterns** (StreamingPolicy distinguishes them):

- **VAD (silero_vad)** — `input=AudioChunks, output=PullEvents`: each `push`
  returns voice-activity events immediately (speech start/end).
- **ASR (nemotron_asr / higgs_audio_stt / voxtral_realtime)** — `input=AudioChunks`:
  `push` accumulates audio; `finish` triggers decode, returns final text.
- **TTS (supertonic / omnivoice / voxcpm2)** — `input=None, output=PullEvents`:
  `push` is a no-op (these models don't consume audio). The text to synthesize
  is passed in `stream_start`'s `options_json` as `{"text":"...","language":"zh"}`.
  Call `stream_pull` repeatedly to get synthesized audio chunks:
  ```c
  stream = audiocpp_stream_start(model, TASK_TTS,
      "{\"text\":\"你好世界\",\"language\":\"zh\"}", 0, &err);
  while (true) {
      audiocpp_stream_event_t *ev = audiocpp_stream_pull(stream, -1, &err);
      if (!ev) break;  // stream exhausted
      // ev->audio_samples  → synthesized PCM (use ev->audio_sample_rate)
      audiocpp_free_stream_event(ev);
  }
  ```

**Streamable models** (7): silero_vad (VAD), nemotron_asr/higgs_audio_stt/
voxtral_realtime (ASR), omnivoice/supertonic/voxcpm2 (TTS). Other models
(vibevoice_asr, etc.) reject `stream_start` → caller should fall back to offline.

**Notes on `stream_pull` and `stream_finish`:**

- `stream_pull`'s `timeout_ms` is currently **synchronous-blocking**: the call
  returns as soon as `next_stream_event()` produces an event or the stream is
  exhausted. Pass `-1`. The values `0` (non-blocking try) and `>0` (wait N ms)
  are accepted for forward compatibility but behave the same as `-1` today;
  any other negative value is rejected.
- For ASR models that decode only at finish time (nemotron_asr), `stream_finish`
  recovers partial-text events emitted during `finalize()` and folds the last
  one into `out_text` when the final `TaskResult` carries no text — so callers
  no longer lose the transcript that was previously stranded in the internal
  event sink.

---

### 7. Model Inspection

Query a loaded model's metadata and capabilities before running:

```c
audiocpp_model_info_t info;
audiocpp_model_info(model, &info);
// info.family, info.variant, info.description

audiocpp_model_capabilities_t caps;
audiocpp_model_capabilities(model, &caps);
// caps.supported_tasks[], caps.languages[]
// caps.supports_speaker_reference, supports_style_condition, supports_timestamps
```

| Function | Description |
|---|---|
| `audiocpp_model_info` | Get family/variant/description |
| `audiocpp_model_capabilities` | Get supported tasks, languages, feature flags |
| `audiocpp_free_model_info` | Free info struct |
| `audiocpp_free_capabilities` | Free capabilities struct |

---

### 8. WAV I/O Utilities

```c
// Read WAV → mono f32 PCM
float *samples; int64_t n; int rate;
audiocpp_read_wav("input.wav", &samples, &n, &rate);
// ... use samples ...
free(samples);

// Write mono f32 PCM → 16-bit WAV
audiocpp_write_wav("output.wav", samples, n, rate);

// Multi-channel results (check `audio->channels`, e.g. dramabox emits
// stereo) must be written with audiocpp_write_wav_ex so the WAV header
// matches the interleaved data:
audiocpp_write_wav_ex("stereo.wav", samples, n, rate, 2);
```

---

### 9. Artifacts

`audiocpp_artifact_t` carries opaque byte payloads + metadata (mime, format,
extension, dimensions, ...) — MIDI bytes, RGB24 video buffers, embeddings,
tokens, model state. Some shipping models **produce** artifacts: MuScriptor →
`AUDIOCPP_ARTIFACT_MIDI` (via `audiocpp_midi`, audio→MIDI); MiniMax-H3 → RGB24
video `AUDIOCPP_ARTIFACT_CUSTOM` (via `audiocpp_generate`, when
`return_video`). You may also **build** artifacts as inputs for forward
compatibility.

```c
/* Input: build one yourself (reserved — no current model consumes it) */
audiocpp_artifact_t *art = audiocpp_artifact_create(
    AUDIOCPP_ARTIFACT_SPEAKER_EMBEDDING, "spk_001",
    embedding_bytes, embedding_size);
audiocpp_artifact_set_meta(art, "dim", "256");
audiocpp_artifact_free(art);

/* Output: read what audiocpp_generate produced (see §4 for the full example) */
audiocpp_artifacts_t *arts = res->artifacts;      /* owned by res */
for (int64_t i = 0; i < arts->n_artifacts; ++i) {
    audiocpp_artifact_t *a = &arts->artifacts[i]; /* a->payload, a->payload_size */
    /* kind AUDIOCPP_ARTIFACT_MIDI → write .mid; CUSTOM video → meta width/height/... */
}
/* do NOT free individual output elements — freed by audiocpp_free_gen_result */
```

| Function | Description |
|---|---|
| `audiocpp_artifact_create` | Create input artifact with kind/id/payload |
| `audiocpp_artifact_set_meta` | Set metadata key-value pair |
| `audiocpp_artifact_free` | Free a single input artifact |
| `audiocpp_free_artifacts` | Free an artifact collection (output of `audiocpp_generate`) |

---

### 10. Memory Management

All result handles are owned by the caller. Free with the matching function:

| Result type | Free function |
|---|---|
| `audiocpp_model_t` | `audiocpp_free_model` |
| `audiocpp_audio_t` | `audiocpp_free_audio` |
| `audiocpp_text_t` | `audiocpp_free_text` |
| `audiocpp_diar_t` | `audiocpp_free_diar` |
| `audiocpp_vad_t` | `audiocpp_free_vad` |
| `audiocpp_align_t` | `audiocpp_free_align` |
| `audiocpp_stems_t` | `audiocpp_free_stems` |
| `audiocpp_stream_t` | `audiocpp_stream_free` |
| `audiocpp_stream_event_t` | `audiocpp_free_stream_event` |
| `audiocpp_model_info_t` | `audiocpp_free_model_info` |
| `audiocpp_model_capabilities_t` | `audiocpp_free_capabilities` |
| `audiocpp_artifact_t` | `audiocpp_artifact_free` |
| `audiocpp_artifacts_t` | `audiocpp_free_artifacts` |
| `audiocpp_gen_result_t` | `audiocpp_free_gen_result` |
| `char *` (from error.message) | `audiocpp_free_string` |

### Error Handling

All functions that take `audiocpp_error_t *err` catch C++ exceptions and
convert them to error codes:
- `code = 0`: success
- `code = -1`: `std::exception` (message in `err.message`)
- `code = -2`: unknown exception

Pass `NULL` for `err` to ignore errors (not recommended).

---

## Symbol Visibility

The shared library exports **only** `audiocpp_*` symbols. All internal
symbols (ggml, CUDA, sentencepiece, cJSON, libyaml) are hidden via:
- `__declspec(dllexport)` on exported functions (AUDIOCPP_API macro)
- `CXX_VISIBILITY_PRESET hidden` on the target + all static dependencies
- `.def` file (Windows) / version script (Linux) as secondary filter

Verified by CI: a post-build symbol check step fails if any non-`audiocpp_*`
symbol appears in the export table.

---

## Platform Notes

### Windows
- The DLL links the MSVC C++ runtime dynamically (`MSVCP140.dll`,
  `VCRUNTIME140.dll`, `VCOMP140.dll`). Deploy the VCRedist or ship these
  alongside.
- CUDA builds require CUDA runtime DLLs (`cudart64_12.dll`, `cublas64_12.dll`,
  `cufft64_11.dll`) in the same directory or on PATH.
- ROCm builds require the AMD HIP SDK runtime DLLs.
- SYCL builds require oneAPI runtime DLLs.

### Linux
- The `.so` is self-contained for CPU/SYCL builds.
- CUDA builds require CUDA runtime libraries on the system or `LD_LIBRARY_PATH`.

### GPU Architecture Support

| Backend | Compiled architectures |
|---|---|
| CUDA | sm_61, sm_70, sm_75, sm_80, sm_86, sm_89, sm_120 (Pascal–Blackwell) |
| ROCm (HIP) | gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1200, gfx1201 (RDNA3+4) |
| SYCL | All Intel GPU architectures (runtime JIT from SPIR-V) |
