/**
 * audiocpp.h — C ABI for audio.cpp
 *
 * Minimal C interface exposing audio.cpp's model loading, TTS, and ASR
 * capabilities to foreign-language runtimes (Rust FFI, Python ctypes, etc.).
 *
 * Design:
 *   - Opaque handles (audiocpp_model_t*) wrap C++ objects (ILoadedVoiceModel + session).
 *   - All C++ exceptions are caught at the boundary and converted to error codes.
 *   - Returned strings/audio are heap-allocated; caller MUST free with the
 *     matching audiocpp_free_* function.
 *   - The shared library hides all ggml/sentencepiece/etc. symbols; only
 *     audiocpp_* symbols are exported.
 *
 * Thread safety:
 *   - audiocpp_load_model is thread-safe (registry is read-only after init).
 *   - A single model handle is NOT thread-safe for concurrent run calls
 *     (each call mutates internal session state). Use separate model handles
 *     per thread, or serialize calls externally.
 */

#ifndef AUDIOCPP_H
#define AUDIOCPP_H

/* Export macro: when building audiocpp.dll, AUDIOCPP_EXPORTS is defined and
 * functions get __declspec(dllexport). Consumers just include the header
 * (no macro needed) or link dynamically. This works across MSVC, icx, and
 * clang — no .def file dependency. */
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef AUDIOCPP_EXPORTS
    #define AUDIOCPP_API __declspec(dllexport)
  #else
    #define AUDIOCPP_API __declspec(dllimport)
  #endif
#else
  #define AUDIOCPP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* ======================================================================== */
/* Types                                                                     */
/* ======================================================================== */

/** Opaque handle to a loaded model + task session. */
typedef struct audiocpp_model audiocpp_model_t;

/** Audio output (PCM f32 samples). Caller owns; free with audiocpp_free_audio. */
typedef struct {
    float *samples;      /**< PCM samples, interleaved when channels > 1, [-1.0, 1.0] */
    int64_t n_samples;   /**< Total number of samples (channels * frames) */
    int sample_rate;     /**< Sample rate in Hz (e.g. 24000) */
    int channels;        /**< Channel count (1 = mono; 2 = stereo, L/R interleaved).
                              Most TTS models emit mono; a few (e.g. dramabox) emit
                              stereo. Write with audiocpp_write_wav_ex so the WAV
                              header matches the data. */
} audiocpp_audio_t;

/** Text output. Caller owns; free with audiocpp_free_text. */
typedef struct {
    char *text;          /**< UTF-8 string (null-terminated) */
    char *language;      /**< Detected language code (may be NULL) */
} audiocpp_text_t;

/** Error info. */
typedef struct {
    int code;            /**< 0 = success, negative = error */
    char *message;       /**< Error message (may be NULL on success). Caller frees with audiocpp_free_string. */
} audiocpp_error_t;

/* ======================================================================== */
/* Backend type (mirrors engine::core::BackendType)                          */
/* ======================================================================== */

enum {
    AUDIOCPP_BACKEND_CPU    = 0,  /**< CPU (always available) */
    AUDIOCPP_BACKEND_CUDA   = 1,  /**< NVIDIA CUDA (also matches AMD ROCm/HIP and MUSA builds) */
    AUDIOCPP_BACKEND_VULKAN = 2,  /**< Vulkan (NVIDIA / AMD / Intel / Apple-over-MoltenVK) */
    AUDIOCPP_BACKEND_METAL  = 3,  /**< Apple Metal (macOS/iOS only) */
    AUDIOCPP_BACKEND_SYCL   = 4,  /**< Intel oneAPI SYCL */
    AUDIOCPP_BACKEND_BEST   = 5,  /**< auto-select best available */
};

/** Whether this build can actually use the given backend.
 *
 *  Returns true only if BOTH conditions hold:
 *    1. The backend was compiled into this build
 *       (e.g. a CUDA build returns false for Vulkan, true for CUDA);
 *    2. At least one device of that backend is present at runtime
 *       (driver installed, hardware detected).
 *
 *  CPU (AUDIOCPP_BACKEND_CPU) always returns true.
 *  Use this to probe which GPU backend a "gpu" request should resolve to,
 *  mirroring transcribe.cpp's transcribe_backend_available().
 *
 *  @param backend  One of AUDIOCPP_BACKEND_* (CPU/CUDA/Vulkan/Metal/SYCL).
 *                  AUDIOCPP_BACKEND_BEST returns true iff some GPU backend is available.
 *  @return 1 if available, 0 otherwise. Never aborts. */
AUDIOCPP_API int audiocpp_backend_available(int backend);

/* ======================================================================== */
/* Task type (mirrors engine::runtime::VoiceTaskKind)                        */
/* ======================================================================== */

enum {
    AUDIOCPP_TASK_VAD    = 0,  /**< Voice Activity Detection */
    AUDIOCPP_TASK_ASR    = 1,  /**< Speech-to-Text */
    AUDIOCPP_TASK_DIAR   = 2,  /**< Speaker Diarization */
    AUDIOCPP_TASK_SEP    = 3,  /**< Source Separation (vocal/accompaniment split) */
    AUDIOCPP_TASK_GEN    = 4,  /**< Audio/Music Generation */
    AUDIOCPP_TASK_TTS    = 5,  /**< Text-to-Speech */
    AUDIOCPP_TASK_ALIGN  = 6,  /**< Forced Alignment (audio + text → timestamps) */
    AUDIOCPP_TASK_VC     = 7,  /**< Voice Conversion (speaker timbre transfer) */
    AUDIOCPP_TASK_CLON   = 8,  /**< Voice Cloning (TTS with speaker reference) */
    AUDIOCPP_TASK_S2S    = 9,  /**< Speech-to-Speech (codec-based) */
    AUDIOCPP_TASK_VDES   = 10, /**< Voice Design (prompt-based voice creation) */
    AUDIOCPP_TASK_SPK    = 11, /**< Speaker Recognition (embeddings) */
    AUDIOCPP_TASK_SVC    = 12, /**< Singing Voice Conversion */
    AUDIOCPP_TASK_MIDI   = 13, /**< Music Generation → MIDI/event output (e.g. MuScriptor) */
};

/* ======================================================================== */
/* Lifecycle                                                                 */
/* ======================================================================== */

/**
 * Load a model and create a task session.
 *
 * @param model_path   Path to model directory or GGUF file. Pass NULL for
 *                     silero_vad / marblenet_vad when built with
 *                     AUDIOCPP_EMBED_VAD_ASSETS=ON (uses baked-in weights,
 *                     no external file needed).
 * @param family_hint  Model family hint (e.g. "qwen3_asr", "qwen3_tts"); NULL = auto-detect.
 * @param task         One of AUDIOCPP_TASK_* (TTS, ASR, VAD, etc.).
 * @param backend      One of AUDIOCPP_BACKEND_*.
 * @param device_id    GPU device index (0 for first GPU; ignored for CPU).
 * @param n_threads    Number of CPU threads (0 = auto).
 * @param err          Optional error output (pass NULL to ignore).
 * @return Model handle, or NULL on failure (check err).
 */
AUDIOCPP_API audiocpp_model_t *audiocpp_load_model(
    const char *model_path,
    const char *family_hint,
    int task,
    int backend,
    int device_id,
    int n_threads,
    audiocpp_error_t *err
);

/**
 * Load a model with session-level options (see audiocpp_load_model for the
 * base contract; this variant additionally configures the created session).
 *
 * Session options are fixed at session creation and apply to every request on
 * the returned handle — unlike request options, which are passed per call to
 * audiocpp_tts/audiocpp_asr/... Recognized keys are model-family specific, e.g.:
 *   {"miotts.codec_model_path": "D:/models/miocodec-25hz-44khz-v2-q8_0.gguf",
 *    "miotts.best_of_n_enabled": "false",
 *    "qwen3_tts.perf_mode": "flash_attention",
 *    "qwen3_tts.mem_saver": "true",
 *    "moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb": 512}
 * Values are rendered like request options: JSON integers become "123",
 * floats keep their fraction, booleans become "true"/"false".
 *
 * Pass NULL or "" for defaults. Malformed JSON is a hard error (the options
 * would otherwise be silently dropped and surface later as confusing
 * "requires X" failures).
 *
 * @param options_json JSON object of session options (NULL/"" = defaults).
 */
AUDIOCPP_API audiocpp_model_t *audiocpp_load_model_ex(
    const char *model_path,
    const char *family_hint,
    int task,
    int backend,
    int device_id,
    int n_threads,
    const char *options_json,
    audiocpp_error_t *err
);

/** Free a model handle. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_model(audiocpp_model_t *model);

/* ======================================================================== */
/* TTS: text → audio                                                         */
/* ======================================================================== */

/**
 * Synthesize speech from text.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_TTS).
 * @param text        UTF-8 input text.
 * @param options     JSON string of model-specific options, e.g.:
 *                      {"voice_ref": "/path/to/ref.wav",
 *                       "reference_text": "transcript of ref audio",
 *                       "speed": 1.0,
 *                       "language": "en",
 *                       "emotion": "happy"}
 *                    Pass NULL or "{}" for defaults.
 * @param err         Optional error output.
 * @return Audio output, or NULL on failure. Caller MUST free with audiocpp_free_audio.
 */
AUDIOCPP_API audiocpp_audio_t *audiocpp_tts(
    const audiocpp_model_t *model,
    const char *text,
    const char *options_json,
    audiocpp_error_t *err
);

/**
 * Synthesize speech with inline voice reference PCM (no temp file).
 *
 * @param model            Model handle (must be TTS).
 * @param text             UTF-8 input text.
 * @param options_json     Options (reference_text, speed, language, ...).
 *                         voice_ref in options_json is ignored when voice_ref_pcm is non-NULL.
 * @param voice_ref_pcm    Float32 mono PCM samples for voice cloning. NULL = no clone.
 * @param voice_ref_n      Number of samples in voice_ref_pcm (0 if NULL).
 * @param voice_ref_sr     Sample rate of voice_ref_pcm (e.g. 16000).
 * @param err              Optional error output.
 */
AUDIOCPP_API audiocpp_audio_t *audiocpp_tts_with_voice_ref(
    const audiocpp_model_t *model,
    const char *text,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
);

/* Batch output merge modes for audiocpp_tts_batch. */
enum {
    AUDIOCPP_BATCH_MERGE_NONE   = 0,  /**< Return N independent audio results */
    AUDIOCPP_BATCH_MERGE_CONCAT = 1,  /**< Concatenate into a single audio result */
};

/**
 * Batch of synthesized audio. Returned by audiocpp_tts_batch.
 * - When merge_mode = AUDIOCPP_BATCH_MERGE_NONE: items[] holds N separate
 *   audio buffers, one per input text (in order). items_failed marks which
 *   texts produced no audio (their entry has samples=NULL, n_samples=0).
 * - When merge_mode = AUDIOCPP_BATCH_MERGE_CONCAT: items[] holds a single
 *   audio buffer (the concatenation); chapters[] gives [start,end) sample
 *   ranges per input text so the client can locate each segment.
 *
 * Caller owns the result and MUST free with audiocpp_free_audio_batch.
 */
typedef struct {
    audiocpp_audio_t *items;   /**< Array of n_items audio buffers */
    int n_items;               /**< Number of items (== number of input texts) */
    /** Per-text sample ranges (concat mode only). [start,end) in samples.
     *  NULL in merge=NONE mode. Length n_items when present. */
    int64_t *chapter_starts;   /**< start sample per text (NULL if not concat) */
    int64_t *chapter_ends;     /**< end sample per text (NULL if not concat) */
} audiocpp_audio_batch_t;

/**
 * Synthesize a batch of texts in one session (reuses a single prepare()).
 *
 * Reuses the loaded model's prepared session across all texts — far cheaper
 * than calling audiocpp_tts N times when prepare() is expensive. Each text is
 * synthesized independently (internal text-chunking still applies per item),
 * then results are either returned separately or concatenated.
 *
 * Progress: if a callback is installed via audiocpp_set_progress_callback, it
 * fires once PER INPUT TEXT at request granularity — stage="batch_tts",
 * completed/total = (text index + 1)/(n_texts). Per-chunk progress from inside
 * each run() is suppressed during the batch so the caller sees clean
 * request-level progress only.
 *
 * @param model        Model handle (must be TTS).
 * @param texts        Array of n_texts UTF-8 strings.
 * @param n_texts      Number of input texts (>= 1).
 * @param options_json Options applied to EVERY text (voice_ref, speed, ...).
 *                     NULL or "{}" for defaults.
 * @param merge_mode   AUDIOCPP_BATCH_MERGE_NONE or AUDIOCPP_BATCH_MERGE_CONCAT.
 * @param err          Optional error output.
 * @return Batch result, or NULL on failure (e.g. all texts failed).
 *         Individual text failures do NOT fail the whole batch — they are
 *         recorded per-item (samples=NULL). Caller MUST free with
 *         audiocpp_free_audio_batch.
 */
AUDIOCPP_API audiocpp_audio_batch_t *audiocpp_tts_batch(
    audiocpp_model_t *model,
    const char *const *texts,
    int n_texts,
    const char *options_json,
    int merge_mode,
    audiocpp_error_t *err
);

/** Free a batch result. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_audio_batch(audiocpp_audio_batch_t *batch);

/* ======================================================================== */
/* ASR: audio → text                                                         */
/* ======================================================================== */

/**
 * Transcribe audio to text.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_ASR).
 * @param pcm         PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of PCM samples.
 * @param sample_rate Sample rate (e.g. 16000).
 * @param options     JSON string of model-specific options, e.g.:
 *                      {"language": "zh", "return_timestamps": "true"}
 *                    Pass NULL or "{}" for defaults.
 * @param err         Optional error output.
 * @return Text result, or NULL on failure. Caller MUST free with audiocpp_free_text.
 */
AUDIOCPP_API audiocpp_text_t *audiocpp_asr(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* Audio transform: audio → audio (SEP / VC)                                 */
/* ======================================================================== */

/**
 * Transform audio to audio (source separation, voice conversion).
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_SEP or AUDIOCPP_TASK_VC).
 * @param pcm         Input PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of PCM samples.
 * @param sample_rate Sample rate (e.g. 16000 or 44100).
 * @param options_json JSON options (NULL = defaults).
 * @param err         Optional error output.
 * @return Audio output, or NULL on failure. Caller MUST free with audiocpp_free_audio.
 */
AUDIOCPP_API audiocpp_audio_t *audiocpp_audio_transform(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/**
 * Transform audio with inline voice reference (for Voice Conversion).
 *
 * @param model            Model handle (must be AUDIOCPP_TASK_VC).
 * @param pcm              Input PCM (source audio to convert).
 * @param n_samples        Number of input samples.
 * @param sample_rate      Input sample rate.
 * @param options_json     Options (NULL = defaults).
 * @param voice_ref_pcm    Target speaker PCM (mono f32). NULL = no voice ref.
 * @param voice_ref_n      Number of voice_ref samples (0 if NULL).
 * @param voice_ref_sr     Voice ref sample rate.
 * @param err              Optional error output.
 * @return Audio output, or NULL on failure. Caller MUST free with audiocpp_free_audio.
 */
AUDIOCPP_API audiocpp_audio_t *audiocpp_audio_transform_with_voice_ref(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* Diarization: audio → speaker turns                                       */
/* ======================================================================== */

/** Speaker turn (one segment of one speaker). */
typedef struct {
    int64_t start_sample;   /**< Start position in samples */
    int64_t end_sample;     /**< End position in samples */
    char *speaker_id;       /**< Speaker ID string (may be NULL) */
    float confidence;       /**< Confidence score [0,1] */
} audiocpp_speaker_turn_t;

/** Diarization result. Caller owns; free with audiocpp_free_diar. */
typedef struct {
    audiocpp_speaker_turn_t *turns;  /**< Array of speaker turns */
    int64_t n_turns;                 /**< Number of turns */
} audiocpp_diar_t;

/**
 * Perform speaker diarization on audio.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_DIAR).
 * @param pcm         PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of PCM samples.
 * @param sample_rate Sample rate (e.g. 16000).
 * @param options_json JSON string of model-specific options (NULL = defaults).
 * @param err         Optional error output.
 * @return Diarization result, or NULL on failure. Caller MUST free with audiocpp_free_diar.
 */
AUDIOCPP_API audiocpp_diar_t *audiocpp_diar(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* VAD: audio → speech segments                                              */
/* ======================================================================== */

/** VAD speech segment. */
typedef struct {
    int64_t start_sample;
    int64_t end_sample;
    float confidence;
} audiocpp_vad_segment_t;

/** VAD result. Caller owns; free with audiocpp_free_vad. */
typedef struct {
    audiocpp_vad_segment_t *segments;
    int64_t n_segments;
} audiocpp_vad_t;

/**
 * Detect speech segments in audio.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_VAD).
 * @param pcm         PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of PCM samples.
 * @param sample_rate Sample rate (e.g. 16000).
 * @param options_json JSON options (NULL = defaults).
 * @param err         Optional error output.
 * @return VAD result, or NULL on failure. Caller MUST free with audiocpp_free_vad.
 */
AUDIOCPP_API audiocpp_vad_t *audiocpp_vad(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/**
 * Energy-based speech segmentation — NO model required.
 *
 * Splits the audio into roughly chunk_seconds-length segments, snapping each
 * boundary to the lowest-RMS window near the nominal split point (so cuts land
 * in silences, not mid-word). This is a pure signal-energy heuristic; it is
 * faster than model VAD and needs no weights, but is less accurate on noisy
 * input or quiet speech. Use it for quick pre-segmentation, or when you cannot
 * ship a VAD model.
 *
 * Recognized options_json keys (all optional):
 *   chunk_seconds        Target segment length in seconds (default 30.0).
 *   boundary_seconds     How far before/after the nominal boundary to search
 *                        for the lowest-energy split (default 2.0).
 *   min_energy_seconds   RMS window length for the energy minimum search
 *                        (default 0.1).
 *
 * @param pcm          PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples    Number of PCM samples.
 * @param sample_rate  Sample rate (e.g. 16000).
 * @param options_json Options (NULL = defaults).
 * @param err          Optional error output.
 * @return VAD result (segments with confidence=1.0; no model used), or NULL on
 *         failure. Caller MUST free with audiocpp_free_vad.
 */
AUDIOCPP_API audiocpp_vad_t *audiocpp_vad_energy(
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* Audio enhancement: denoise / super-resolve (no model registry)            */
/* ======================================================================== */

/**
 * Denoise / enhance audio with one of three models. NOT a model-registry task
 * — these are standalone audio-utility models loaded directly.
 *
 * @param pcm          Input PCM (mono f32, [-1.0, 1.0]).
 * @param n_samples    Number of input samples.
 * @param sample_rate  Input sample rate (any; resampled to the model's expected
 *                     rate internally: 48000 for deepfilternet2/rnnoise, 16000
 *                     for zipenhancer).
 * @param model_name   One of "deepfilternet2", "rnnoise", "zipenhancer".
 * @param model_path   Model directory (or file for rnnoise). NULL = use the
 *                     embedded asset (requires AUDIOCPP_EMBED_AUDIO_UTILITIES=ON).
 * @param options_json Options JSON. Recognized keys:
 *                       "backend": "cpu" | "cuda" | "vulkan" | "metal" | "sycl" (default cpu)
 *                       "device":  device index (default 0)
 * @param err          Optional error output.
 * @return Denoised audio (mono f32), or NULL on failure. The output sample rate
 *         matches the model's native rate (48k for deepfilternet2/rnnoise,
 *         16k for zipenhancer) — check result->sample_rate. Caller MUST free
 *         with audiocpp_free_audio.
 */
AUDIOCPP_API audiocpp_audio_t *audiocpp_denoise(
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *model_name,
    const char *model_path,
    const char *options_json,
    audiocpp_error_t *err
);

/**
 * Super-resolve (bandwidth-expand) narrowband audio to wideband with flashsr.
 *
 * @param pcm          Input PCM (mono f32, [-1.0, 1.0]).
 * @param n_samples    Number of input samples.
 * @param sample_rate  Input sample rate (any; resampled to 16000 internally).
 * @param model_path   flashsr model directory. NULL = use the embedded asset
 *                     (requires AUDIOCPP_EMBED_AUDIO_UTILITIES=ON).
 * @param options_json Options JSON (backend/device, same as audiocpp_denoise).
 * @param err          Optional error output.
 * @return Upsampled audio (mono f32 @ 48000 Hz), or NULL on failure.
 *         Caller MUST free with audiocpp_free_audio.
 */
AUDIOCPP_API audiocpp_audio_t *audiocpp_super_resolve(
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *model_path,
    const char *options_json,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* Forced Alignment: audio + text → word timestamps                         */
/* ======================================================================== */

/** Word alignment result (one word with its time span). */
typedef struct {
    double start_seconds;   /**< Word start time in seconds */
    double end_seconds;     /**< Word end time in seconds */
    char *word;             /**< UTF-8 word text (owned by result, freed with audiocpp_free_align) */
    float confidence;       /**< Confidence score [0,1] */
} audiocpp_word_t;

/** Alignment result. Caller owns; free with audiocpp_free_align. */
typedef struct {
    audiocpp_word_t *words;  /**< Array of word alignments */
    int64_t n_words;         /**< Number of words */
    char *language;          /**< Language code used (may be NULL) */
} audiocpp_align_t;

/**
 * Force-align audio to a transcript, producing per-word timestamps.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_ALIGN).
 * @param pcm         PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of PCM samples.
 * @param sample_rate Sample rate (e.g. 16000).
 * @param text        Transcript text to align (non-NULL, non-empty).
 * @param language    Language code, e.g. "en", "zh" (non-NULL, non-empty).
 * @param options_json JSON options (NULL = defaults).
 * @param err         Optional error output.
 * @return Alignment result, or NULL on failure. Caller MUST free with audiocpp_free_align.
 */
AUDIOCPP_API audiocpp_align_t *audiocpp_align(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *text,
    const char *language,
    const char *options_json,
    audiocpp_error_t *err
);

/** Free an alignment result. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_align(audiocpp_align_t *align);

/* ======================================================================== */
/* Multi-stem audio transform (SEP / music generation)                      */
/* ======================================================================== */

/** Named audio stem (e.g. "vocals", "drums", "bass", "other", "instrumental"). */
typedef struct {
    char *name;             /**< Stem name (owned by result) */
    float *samples;         /**< PCM samples, interleaved when channels > 1, [-1.0, 1.0] (owned) */
    int64_t n_samples;      /**< Total number of samples (channels * frames) */
    int sample_rate;        /**< Sample rate in Hz */
    int channels;           /**< Channel count (1 = mono; 2 = stereo, L/R interleaved) */
} audiocpp_stem_t;

/** Multi-stem audio result. Caller owns; free with audiocpp_free_stems. */
typedef struct {
    audiocpp_stem_t *stems;  /**< Array of stems */
    int64_t n_stems;         /**< Number of stems (may be 0) */
} audiocpp_stems_t;

/**
 * Transform audio → multi-stem audio (source separation, music generation).
 * Returns ALL named outputs (vocals, drums, bass, instrumental, etc.),
 * unlike audiocpp_audio_transform which only returns the first.
 *
 * @param model       Model handle (SEP or GEN task).
 * @param pcm         Input PCM (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of samples.
 * @param sample_rate Sample rate.
 * @param options_json Options (NULL = defaults).
 * @param voice_ref_pcm  Optional target speaker PCM for VC (NULL = none).
 * @param voice_ref_n    Voice ref sample count.
 * @param voice_ref_sr   Voice ref sample rate.
 * @param err         Optional error output.
 * @return Multi-stem result, or NULL on failure. Free with audiocpp_free_stems.
 */
AUDIOCPP_API audiocpp_stems_t *audiocpp_transform_stems(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
);

/** Free a multi-stem result. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_stems(audiocpp_stems_t *stems);

/* ======================================================================== */
/* Model inspection                                                         */
/* ======================================================================== */

/** Model capability info (what a loaded model can do). */
typedef struct {
    int  supports_speaker_reference;  /**< 1 if model accepts voice ref audio */
    int  supports_style_condition;    /**< 1 if model accepts style/emotion control */
    int  supports_timestamps;         /**< 1 if model produces word timestamps */
    int  n_supported_tasks;           /**< Number of entries in supported_tasks */
    int  *supported_tasks;            /**< Array of AUDIOCPP_TASK_* values (owned) */
    int  n_languages;                 /**< Number of language codes */
    char **languages;                 /**< Array of language code strings (owned) */
} audiocpp_model_capabilities_t;

/** Model metadata (family, variant, description). */
typedef struct {
    char *family;        /**< Model family (e.g. "qwen3_asr") */
    char *variant;       /**< Model variant (may be empty) */
    char *description;   /**< Human-readable description (may be empty) */
} audiocpp_model_info_t;

/**
 * Get metadata for a loaded model.
 * @param model  Model handle.
 * @param out    Receives metadata. Caller frees with audiocpp_free_model_info.
 * @return 0 on success, -1 on error.
 */
AUDIOCPP_API int audiocpp_model_info(
    const audiocpp_model_t *model,
    audiocpp_model_info_t *out
);

/**
 * Get capabilities for a loaded model (supported tasks, languages, etc.).
 * @param model  Model handle.
 * @param out    Receives capabilities. Caller frees with audiocpp_free_capabilities.
 * @return 0 on success, -1 on error.
 */
AUDIOCPP_API int audiocpp_model_capabilities(
    const audiocpp_model_t *model,
    audiocpp_model_capabilities_t *out
);

/** Free model info. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_model_info(audiocpp_model_info_t *info);

/** Free model capabilities. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_capabilities(audiocpp_model_capabilities_t *caps);

/* ======================================================================== */
/* WAV I/O utilities                                                        */
/* ======================================================================== */

/**
 * Read a WAV file into mono f32 PCM samples.
 * @param path         WAV file path.
 * @param out_samples  Receives malloc'd float array (caller frees with free()).
 * @param out_n        Receives sample count.
 * @param out_rate     Receives sample rate.
 * @return 0 on success, -1 on error.
 */
AUDIOCPP_API int audiocpp_read_wav(
    const char *path,
    float **out_samples,
    int64_t *out_n,
    int *out_rate
);

/**
 * Write mono f32 PCM samples to a 16-bit WAV file.
 * @param path         Output WAV path.
 * @param samples      PCM samples [-1.0, 1.0] (mono).
 * @param n_samples    Number of samples.
 * @param sample_rate  Sample rate.
 * @return 0 on success, -1 on error.
 */
AUDIOCPP_API int audiocpp_write_wav(
    const char *path,
    const float *samples,
    int64_t n_samples,
    int sample_rate
);

/**
 * Write f32 PCM samples to a 16-bit WAV file with an explicit channel count.
 *
 * Use this for results whose audiocpp_audio_t.channels > 1 (interleaved
 * multi-channel data): it writes a WAV header matching the data, so the file
 * plays back at the correct speed. audiocpp_write_wav is equivalent to calling
 * this with channels = 1.
 *
 * @param path         Output WAV path.
 * @param samples      PCM samples, interleaved for channels > 1, [-1.0, 1.0].
 * @param n_samples    Total number of samples (channels * frames).
 * @param sample_rate  Sample rate.
 * @param channels     Channel count (>= 1).
 * @return 0 on success, -1 on error.
 */
AUDIOCPP_API int audiocpp_write_wav_ex(
    const char *path,
    const float *samples,
    int64_t n_samples,
    int sample_rate,
    int channels
);

/* ======================================================================== */
/* Artifacts                                                                 */
/* ======================================================================== */
/* VoiceArtifact types for passing opaque data (MIDI/event bytes, embeddings,
 * tokens, model state, raw pixel buffers, ...) between models and callers.
 * Some shipping models produce artifacts: MuScriptor emits MIDI bytes
 * (AUDIOCPP_ARTIFACT_MIDI) and MiniMax-H3 may emit a raw RGB24 video
 * buffer (AUDIOCPP_ARTIFACT_CUSTOM, id "minimax_h3_video_rgb24"). Artifacts
 * produced by a run are surfaced via audiocpp_generate (see
 * audiocpp_gen_result_t); inputs may be built with audiocpp_artifact_create. */

/** Artifact kind (mirrors engine::runtime::ArtifactKind). */
enum {
    AUDIOCPP_ARTIFACT_SPEAKER_EMBEDDING = 0,
    AUDIOCPP_ARTIFACT_STYLE_EMBEDDING   = 1,
    AUDIOCPP_ARTIFACT_PROMPT_EMBEDDING  = 2,
    AUDIOCPP_ARTIFACT_ACOUSTIC_TOKENS   = 3,
    AUDIOCPP_ARTIFACT_MIDI              = 4,
    AUDIOCPP_ARTIFACT_ALIGNMENT         = 5,
    AUDIOCPP_ARTIFACT_DIARIZATION_STATE = 6,
    AUDIOCPP_ARTIFACT_VAD_STATE         = 7,
    AUDIOCPP_ARTIFACT_CUSTOM            = 8,
};

/** Opaque artifact (raw bytes + metadata key-value pairs). Maps 1:1 to
 *  engine::runtime::VoiceArtifact. As input: built with audiocpp_artifact_create
 *  and freed with audiocpp_artifact_free. As output: produced as elements of
 *  audiocpp_artifacts_t (from audiocpp_generate); free the whole collection
 *  with audiocpp_free_artifacts — do not call audiocpp_artifact_free on
 *  individual elements you did not create yourself. */
typedef struct {
    int kind;                /**< AUDIOCPP_ARTIFACT_* */
    char *id;                /**< Artifact identifier (owned) */
    uint8_t *payload;        /**< Raw byte payload (owned, may be NULL) */
    int64_t payload_size;    /**< Payload byte count (0 if no payload) */
    int n_meta;              /**< Number of metadata key-value pairs */
    char **meta_keys;        /**< Metadata keys array (owned) */
    char **meta_values;      /**< Metadata values array (owned) */
} audiocpp_artifact_t;

/** Collection of artifacts. Caller owns; free with audiocpp_free_artifacts.
 *  Elements are contiguous audiocpp_artifact_t values (array-of-struct).
 *  Well-known metadata keys: "mime", "format", "extension" (MuScriptor MIDI:
 *  mime "audio/midi", extension "mid"); MiniMax-H3 video also carries
 *  "width"/"height"/"frames"/"fps" as decimal strings alongside "format"
 *  = "rgb24". Payload byte count is each element's payload_size. */
typedef struct {
    audiocpp_artifact_t *artifacts;  /**< Contiguous array (n_artifacts elements) */
    int64_t n_artifacts;             /**< Number of artifacts (may be 0) */
} audiocpp_artifacts_t;

/**
 * Create an artifact (reserved for future use — no current model accepts it).
 * @param kind         AUDIOCPP_ARTIFACT_*
 * @param id           Artifact identifier (copied).
 * @param payload      Raw payload bytes (copied; NULL = no payload).
 * @param payload_size Payload byte count.
 * @return New artifact handle. Free with audiocpp_artifact_free.
 */
AUDIOCPP_API audiocpp_artifact_t *audiocpp_artifact_create(
    int kind,
    const char *id,
    const uint8_t *payload,
    int64_t payload_size
);

/**
 * Set a metadata key-value pair on an artifact.
 * If the key already exists, its value is replaced.
 * @return 0 on success, -1 on error.
 */
AUDIOCPP_API int audiocpp_artifact_set_meta(
    audiocpp_artifact_t *artifact,
    const char *key,
    const char *value
);

/** Free an artifact and all owned data. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_artifact_free(audiocpp_artifact_t *artifact);

/** Free an artifact collection (and every element's owned data).
 *  Safe to call with NULL. For collections returned by audiocpp_generate. */
AUDIOCPP_API void audiocpp_free_artifacts(audiocpp_artifacts_t *artifacts);

/* ======================================================================== */
/* Music / audio generation + MIDI transcription                            */
/* ======================================================================== */

/** Combined generation result: zero or one audio buffer plus zero or more
 *  artifacts, all from a single run() (generation is expensive and may be
 *  non-deterministic, so audio and artifacts are returned together rather
 *  than via two calls). Caller owns; free with audiocpp_free_gen_result.
 *  Either field may be NULL. Used by text→audio generators: MiniMax-H3
 *  (audio + optional RGB24 video when "return_video": true), stable_audio,
 *  ace_step (audio). MuScriptor does NOT use this — it is audio→MIDI, see
 *  audiocpp_midi. */
typedef struct {
    audiocpp_audio_t *audio;         /**< PCM output, NULL if the model emits none */
    audiocpp_artifacts_t *artifacts; /**< video/etc., NULL if none */
} audiocpp_gen_result_t;

/**
 * Run a text-conditioned generation task: text prompt → audio and/or artifacts.
 *
 * Load the model with AUDIOCPP_TASK_GEN and family_hint "minimax_h3"
 * (or "stable_audio" / "ace_step"). Then call with the text @p prompt;
 * model-specific knobs flow through @p options_json (e.g. MiniMax-H3:
 * {"return_video":true, "width":..., "height":..., "num_frames":...}).
 *
 * NOTE: this entry is text→output. MuScriptor is audio→MIDI transcription —
 * use audiocpp_midi for it, not this entry.
 *
 * @param model        Model handle (loaded via audiocpp_load_model[_ex]).
 * @param prompt       Text prompt (may be NULL if the model needs no text).
 * @param options_json Optional JSON object of model options (NULL/"" = none).
 * @param err          Optional error output (pass NULL to ignore).
 * @return Generation result, or NULL on failure (check err).
 *         Free with audiocpp_free_gen_result.
 */
AUDIOCPP_API audiocpp_gen_result_t *audiocpp_generate(
    const audiocpp_model_t *model,
    const char *prompt,
    const char *options_json,
    audiocpp_error_t *err
);

/** Free a generation result and both owned fields. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_gen_result(audiocpp_gen_result_t *res);

/**
 * Transcribe audio → MIDI (or event JSON). For MuScriptor: load with
 * AUDIOCPP_TASK_MIDI and family_hint "muscriptor", then feed the input audio
 * PCM. The model is an audio→MIDI transcriber (it does not consume text).
 * Returns the produced artifacts (MIDI bytes by default); free the collection
 * with audiocpp_free_artifacts.
 *
 * @param model        Model handle (MuScriptor, loaded with AUDIOCPP_TASK_MIDI).
 * @param pcm          Input audio PCM, mono, [-1.0, 1.0].
 * @param n_samples    Number of PCM samples in @p pcm.
 * @param sample_rate  Sample rate of @p pcm (the model resamples as needed).
 * @param options_json Optional JSON: {"output_format":"midi"|"json",
 *                     "temperature":1.0, "guidance_scale":1.0, "max_tokens":2000,
 *                     "instruments":"...", "seed":N, ...}.
 * @param err          Optional error output (pass NULL to ignore).
 * @return Artifacts (index 0 is the MIDI/JSON payload), or NULL on failure.
 *         Free with audiocpp_free_artifacts.
 */
AUDIOCPP_API audiocpp_artifacts_t *audiocpp_midi(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/**
 * Same as audiocpp_midi but takes the input audio as a WAV file's raw bytes
 * (RIFF header + PCM payload) instead of decoded PCM — no temp file needed.
 * Convenient when the caller already holds the WAV in memory (e.g. an HTTP
 * upload). The bytes are decoded in-memory (supports PCM16 / PCM24 / IEEE
 * float32) and downmixed to mono; @p options_json is the same as audiocpp_midi.
 *
 * @param model        Model handle (MuScriptor, loaded with AUDIOCPP_TASK_MIDI).
 * @param wav_data     Raw bytes of a .wav file.
 * @param wav_size     Number of bytes in @p wav_data.
 * @param options_json Optional JSON (see audiocpp_midi).
 * @param err          Optional error output (pass NULL to ignore).
 * @return Artifacts (index 0 is the MIDI/JSON payload), or NULL on failure
 *         (including unparseable WAV). Free with audiocpp_free_artifacts.
 */
AUDIOCPP_API audiocpp_artifacts_t *audiocpp_midi_from_wav(
    const audiocpp_model_t *model,
    const uint8_t *wav_data,
    int64_t wav_size,
    const char *options_json,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* Streaming (chunk-push model)                                             */
/* ======================================================================== */

/** Opaque handle to a streaming session. */
typedef struct audiocpp_stream audiocpp_stream_t;

/** Voice activity event kinds (mirrors VoiceActivityEvent::Kind). */
enum {
    AUDIOCPP_VA_SPEECH_START   = 0,  /**< Speech onset detected */
    AUDIOCPP_VA_SPEECH_END     = 1,  /**< Speech offset detected */
    AUDIOCPP_VA_SPEECH_SEGMENT = 2,  /**< Complete speech segment */
};

/** Voice activity event in a stream. */
typedef struct {
    int kind;               /**< AUDIOCPP_VA_* */
    int64_t sample;         /**< Sample position in the stream */
    float probability;      /**< Speech probability [0,1] */
} audiocpp_va_event_t;

/** Event returned by audiocpp_stream_push. Caller frees with
 *  audiocpp_free_stream_event. Fields that are NULL/0 mean "no data
 *  of this type in this chunk". */
typedef struct {
    /* VAD events (silero_vad: one or more per chunk) */
    audiocpp_va_event_t *va_events;
    int n_va_events;
    /* Partial text (nemotron_asr: incremental transcript, may be NULL) */
    char *partial_text;
    /* Streaming audio output (streaming TTS, may be NULL) */
    float *audio_samples;
    int64_t n_audio_samples;
    int audio_sample_rate;
    /* True when the stream signals completion */
    int is_final;
} audiocpp_stream_event_t;

/**
 * Start a streaming session from a loaded model.
 *
 * Creates a new streaming task session (separate from the model's
 * offline session). The model must support streaming for the given task.
 *
 * @param model       Loaded model handle.
 * @param task        AUDIOCPP_TASK_* (VAD, ASR, TTS, etc.).
 * @param options_json Optional JSON options (NULL = defaults).
 * @param preferred_chunk_samples Preferred audio chunk size in samples (0 = model default).
 * @param err         Optional error output.
 * @return Stream handle, or NULL on failure (check err).
 */
AUDIOCPP_API audiocpp_stream_t *audiocpp_stream_start(
    const audiocpp_model_t *model,
    int task,
    const char *options_json,
    int64_t preferred_chunk_samples,
    audiocpp_error_t *err
);

/**
 * Push an audio chunk to the stream and get the resulting event.
 *
 * For VAD: each chunk returns voice-activity events immediately.
 * For ASR: chunks accumulate audio; events are empty until finish.
 * For streaming TTS: may return audio output per chunk.
 *
 * @param stream      Stream handle from audiocpp_stream_start.
 * @param pcm         Audio chunk (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of samples in this chunk.
 * @param sample_rate Sample rate (e.g. 16000).
 * @param err         Optional error output.
 * @return Stream event (may have all-zero fields if nothing happened).
 *         Caller MUST free with audiocpp_free_stream_event. Returns NULL
 *         only on hard error (check err).
 */
AUDIOCPP_API audiocpp_stream_event_t *audiocpp_stream_push(
    audiocpp_stream_t *stream,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    audiocpp_error_t *err
);

/**
 * Finish the stream and get the final result.
 *
 * Triggers any deferred processing (e.g. ASR decode) and returns the
 * accumulated final output.
 *
 * @param stream   Stream handle (invalidated after this call).
 * @param out_text Optional: receives final transcript (for ASR). Pass NULL to skip.
 *                 Caller owns and must free with audiocpp_free_text.
 * @param err      Optional error output.
 * @return 0 on success, -1 on error.
 */
AUDIOCPP_API int audiocpp_stream_finish(
    audiocpp_stream_t *stream,
    audiocpp_text_t *out_text,
    audiocpp_error_t *err
);

/** Free a stream event. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_stream_event(audiocpp_stream_event_t *event);

/**
 * Pull the next generated event from a PullEvents stream (streaming TTS).
 *
 * For streaming TTS models (supertonic/omnivoice/voxcpm2) whose
 * StreamingPolicy.input is None, stream_push() throws because they don't
 * consume audio chunks. Use this function instead to pull synthesized audio
 * chunks one at a time. The text to synthesize is passed in stream_start's
 * options_json as {"text":"...","language":"zh"}.
 *
 * @param stream      Stream handle from audiocpp_stream_start.
 * @param timeout_ms  Reserved for future non-blocking support. The current
 *                    implementation is fully synchronous: this call blocks on
 *                    the calling thread until next_stream_event() returns.
 *                    Pass -1. The values 0 (non-blocking try) and >0 (wait N
 *                    ms) are accepted but currently behave the same as -1;
 *                    they will be honored once a background-pump mode lands.
 * @param err         Optional error output.
 * @return Stream event (typically audio_samples for TTS), or NULL if the
 *         stream is exhausted (no more data) or on timeout with no event.
 *         Caller MUST free non-NULL results with audiocpp_free_stream_event.
 */
AUDIOCPP_API audiocpp_stream_event_t *audiocpp_stream_pull(
    audiocpp_stream_t *stream,
    int timeout_ms,
    audiocpp_error_t *err
);

/** Free a stream handle. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_stream_free(audiocpp_stream_t *stream);

/* ======================================================================== */
/* Device enumeration                                                        */
/* ======================================================================== */

/** Device type (mirrors ggml_backend_dev_type). */
enum {
    AUDIOCPP_DEVICE_CPU   = 0,  /**< CPU */
    AUDIOCPP_DEVICE_GPU   = 1,  /**< Discrete GPU */
    AUDIOCPP_DEVICE_IGPU  = 2,  /**< Integrated GPU */
    AUDIOCPP_DEVICE_ACCEL = 3,  /**< Hardware accelerator (e.g. Apple Metal) */
    AUDIOCPP_DEVICE_META  = 4,  /**< Meta device (ggml internal, rare) */
};

/** Information about a compute device.
 *  Use audiocpp_device_count / audiocpp_device_info to enumerate before
 *  calling audiocpp_load_model, so the client can pick the right backend
 *  and device_id. */
typedef struct {
    char name[128];         /**< Device name (e.g. "NVIDIA GeForce RTX 4090") */
    char description[256];  /**< Longer description (may be empty) */
    int  backend;           /**< AUDIOCPP_BACKEND_* this device belongs to */
    int  device_id;         /**< Per-backend index — pass to audiocpp_load_model */
    int  type;              /**< AUDIOCPP_DEVICE_* (CPU/GPU/IGPU) */
    uint64_t memory_total;  /**< Total memory in bytes (0 if unknown) */
    uint64_t memory_free;   /**< Free memory in bytes (0 if unknown) */
} audiocpp_device_info_t;

/** Count available compute devices across all compiled backends. */
AUDIOCPP_API int audiocpp_device_count(void);

/** Get info for a device by index (0 .. count-1).
 *  @param index  Device index (global, not per-backend).
 *  @param out    Receives device info.
 *  @return 0 on success, -1 if index out of range or out is NULL. */
AUDIOCPP_API int audiocpp_device_info(int index, audiocpp_device_info_t *out);

/** Print all devices to stdout (for CLI / debugging convenience). */
AUDIOCPP_API void audiocpp_list_devices(void);

/* ======================================================================== */
/* Utilities                                                                 */
/* ======================================================================== */

/** Get the version string (static, do NOT free).
 *  Format: "<version> <commit_short> <branch> <build_time> <backend>"
 *  e.g. "release-0.4.2-178-g3239acd 3239acd main 2026-08-03T10:50:06Z cuda".
 *  Backend is a compile-time label (cpu/cuda/rocm/sycl/vulkan/metal); probe
 *  what this build can actually run with audiocpp_backend_available(). */
AUDIOCPP_API const char *audiocpp_version(void);

/**
 * Get the build-ID string (static, do NOT free): a single grep-able line
 * "audiocpp-build-id: <version> <commit> <branch> <date>". Also embedded
 * verbatim in the binary so it is recoverable via `strings <lib> | grep
 * audiocpp-build-id` without loading it.
 */
AUDIOCPP_API const char *audiocpp_build_id(void);

/** Free an audio result. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_audio(audiocpp_audio_t *audio);

/** Free a text result. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_text(audiocpp_text_t *text);

/** Free a diarization result. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_diar(audiocpp_diar_t *diar);

/** Free a VAD result. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_vad(audiocpp_vad_t *vad);

/** Free a string returned in audiocpp_error_t. Safe to call with NULL. */
AUDIOCPP_API void audiocpp_free_string(char *str);

/** Clear an error struct (frees message, resets code to 0). */
AUDIOCPP_API void audiocpp_clear_error(audiocpp_error_t *err);

/* ======================================================================== */
/* Progress callback (offline run progress + cancellation)                  */
/* ======================================================================== */

/**
 * Progress callback signature. Invoked synchronously on the calling thread
 * from inside an offline run (audiocpp_tts / audiocpp_asr / ...) at chunk
 * boundaries for chunked models, or once at start/end for single-shot models.
 *
 * @param progress       Completion fraction in [0.0, 1.0].
 * @param stage          Model family name, e.g. "qwen3_tts" / "qwen3_asr".
 *                       Valid for the duration of the callback only.
 * @param completed_units Chunks completed so far (0..total_units).
 * @param total_units    Total chunks (1 for single-shot models).
 * @param user_data      Opaque pointer passed to audiocpp_set_progress_callback.
 *
 * @return 0 to continue running; non-zero to request cancellation. When
 *         cancellation is requested, the in-flight run() aborts and the
 *         triggering audiocpp_* function returns NULL with
 *         err->message = "canceled by progress callback" (err->code stays 0;
 *         cancellation is not a hard error).
 */
typedef int (*audiocpp_progress_fn)(float progress,
                                    const char *stage,
                                    int64_t completed_units,
                                    int64_t total_units,
                                    void *user_data);

/**
 * Install a progress callback on a model handle. The callback fires during
 * subsequent offline runs (audiocpp_tts, audiocpp_asr, audiocpp_vad, ...).
 *
 * Pass fn=NULL to clear a previously installed callback.
 *
 * The callback is per-model-handle and persists across runs until cleared or
 * replaced. It is invoked on the same thread that calls the run function.
 *
 * Thread safety: do not call set_progress_callback concurrently with a run on
 * the same handle. Set it before invoking a run function.
 */
AUDIOCPP_API void audiocpp_set_progress_callback(audiocpp_model_t *model,
                                                 audiocpp_progress_fn fn,
                                                 void *user_data);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AUDIOCPP_H */
