#pragma once

// engine/models/whisper/decoding.h - Whisper's decoding recipe as a
// graph-agnostic policy (Phase 11 W2b).
//
// This is HF `WhisperForConditionalGeneration.generate()` with Whisper's own
// shipping recipe, ported from the transcribe.cpp arch
// (src/runtime/arch/whisper/model.cpp, whisper_run; deleted in B16c) where it was proven
// against HF: the WhisperTimeStampLogitsProcessor rules, `_retrieve_segment`,
// `generate_with_fallback` (temperature ladder + compression-ratio /
// avg-logprob / no-speech gates), prompt and `condition_on_prev_tokens`
// handling, and the unified seek loop that HF (5.x) runs for every input
// length. What it deliberately does NOT know about is ggml: the model is
// reached through `WhisperModelHooks` callbacks, so the whole algorithm runs
// under a scripted fake decoder in tests/unittests/test_whisper_decoding.cpp.
//
// Units: the mel frame rate is 100 Hz (hop 160 @ 16 kHz), an encoder frame is
// 2 mel frames, and a timestamp token step is 20 ms (one encoder frame).

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace engine::models::whisper {

// Token-id contract of one checkpoint. Built from WhisperHParams by the
// runtime; plain data so tests can describe a toy vocabulary.
struct WhisperTokenContract {
  int32_t vocab_size = 0;
  int32_t eot = -1;             // <|endoftext|>: ids below it are text
  int32_t sot = -1;             // <|startoftranscript|>
  int32_t transcribe = -1;      // multilingual only
  int32_t translate = -1;       // multilingual only
  int32_t prev_sot = -1;        // <|startofprev|>, -1 if absent
  int32_t no_timestamps = -1;   // <|notimestamps|>
  bool multilingual = false;

  int32_t timestamp_begin() const noexcept { return no_timestamps + 1; }
  int32_t no_speech() const noexcept { return no_timestamps - 1; }
  bool is_timestamp(int32_t id) const noexcept {
    return id >= timestamp_begin() && id < vocab_size;
  }
  bool is_text(int32_t id) const noexcept { return id >= 0 && id < eot; }
};

// Per-run knobs. Defaults are Whisper's shipping recipe (the defaults of
// transcribe_whisper_run_ext_init), not HF generate()'s library defaults.
// Thresholds use +/-INF to mean "disabled".
struct WhisperDecodeOptions {
  bool timestamps = false;           // SEGMENT timestamps (else <|notimestamps|>)
  bool translate = false;            // multilingual only
  // The <|lang|> token. Unset on a multilingual model means "detect on the
  // first window"; ignored (and must stay unset) on an English-only model.
  std::optional<int32_t> language_token;
  std::vector<int32_t> language_candidates;  // lang token ids for detection

  // Initial prompt, already tokenized, text ids only (no <|startofprev|>).
  std::vector<int32_t> prompt_ids;
  bool prompt_all_segments = false;  // ALL_SEGMENTS vs FIRST_SEGMENT
  bool condition_on_prev_tokens = false;
  int32_t max_prev_context_tokens = 223;

  float temperature = 0.0f;
  float temperature_inc = 0.2f;
  float compression_ratio_thold = 2.4f;
  float logprob_thold = -1.0f;
  float no_speech_thold = 0.6f;
  float max_initial_timestamp = 1.0f;  // seconds; negative / non-finite = off
  uint32_t seed = 0;                   // 0 = nondeterministic

  int32_t max_new_tokens = 256;        // per window (HF / arch value)
};

// One emitted segment: absolute times and the text ids inside it.
struct WhisperSegment {
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
  std::vector<int32_t> text_ids;
};

struct WhisperDecodeResult {
  std::vector<int32_t> text_ids;   // every accepted window, text only
  std::vector<int32_t> raw_ids;    // every accepted window, incl. timestamps
  std::vector<WhisperSegment> segments;         // only when options.timestamps
  std::vector<runtime::DecodeTelemetry> traces; // one per decoded window
  int32_t language_token = -1;     // the prompt's <|lang|>, -1 for .en models
  bool language_detected = false;  // true when it came from detection
  // L11: some window hit max_new_tokens / the decoder context before EOS and
  // the seek moved past it, so text was lost. (A cut window that re-enters at
  // its last closed timestamp pair loses nothing and does not set this.)
  bool truncated = false;
};

// The model, as the policy sees it. The runtime binds these to ggml graphs.
struct WhisperModelHooks {
  // Run the encoder on mel frames [seek, seek + n_frames) (zero-padded to the
  // window) and populate the cross-attention cache for the decoder.
  std::function<void(int seek_frame, int n_frames)> encode_window;
  // Decode `prompt` from position 0 (self-attention cache reset). Writes the
  // raw logits of the prompt's LAST position into `last_logits`, and - when
  // `sot_row` >= 0 - the raw logits at position `sot_row` into `sot_logits`.
  std::function<void(const std::vector<int32_t> & prompt, int sot_row,
                     std::vector<float> & sot_logits,
                     std::vector<float> & last_logits)>
      prefill;
  // Append `token` at position `n_past` and write the next raw logits.
  std::function<void(int32_t token, int n_past, std::vector<float> & logits)>
      step;
  // Called at window and token boundaries; throws (e.g. ProgressCanceled) to
  // abort. May be empty.
  std::function<void(int done_frames, int total_frames)> poll;
  // Called when any hook throws, with the windows accepted so far (the
  // in-flight window is not included); the exception then propagates
  // unchanged. Lets the caller publish a partial result on abort. May be empty.
  std::function<void(const WhisperDecodeResult & completed)> on_interrupted;
};

// ---- the policy's building blocks (exposed for unit tests) ----------------

// HF WhisperTimeStampLogitsProcessor. Mutates `logits` in place: always masks
// <|notimestamps|>; timestamps come in pairs and never decrease; the first
// generated token must be a timestamp, capped at `max_initial_timestamp_index`
// (-1 = uncapped); and a timestamp is forced when their total probability
// beats every single text token.
void apply_timestamp_rules(float *logits, const std::vector<int32_t> &generated,
                           const WhisperTokenContract &ids,
                           int max_initial_timestamp_index);

// HF _retrieve_segment. `seek_num_frames` is the window's real mel length.
struct RetrievedWindow {
  std::vector<WhisperSegment> segments;              // only if want_segments
  std::vector<std::vector<int32_t>> history_slices;  // prev-context carry
  int seek_advance_frames = 0;
};
RetrievedWindow retrieve_segments(const std::vector<int32_t> &generated,
                                  const WhisperTokenContract &ids,
                                  int64_t time_offset_ms, int seek_num_frames,
                                  bool want_segments);

// HF _prepare_decoder_input_ids for one window: the <|startofprev|> context,
// then [SOT, (<|lang|>, <|task|>), (<|notimestamps|>)]. Returns the prefix and
// the position of SOT in it (where the no-speech probability is read).
struct WindowPrefix {
  std::vector<int32_t> tokens;
  int sot_index = 0;
};
WindowPrefix build_window_prefix(
    const WhisperTokenContract &ids, const WhisperDecodeOptions &options,
    int32_t language_token,
    const std::vector<std::vector<int32_t>> &history, bool is_first_window,
    bool condition_on_prev);

// ---- the whole recipe -----------------------------------------------------

// Validates `options` against `ids` (throws std::invalid_argument, mapped to
// INVALID_ARG at the C ABI), then runs the seek loop over `total_mel_frames`
// in windows of `window_frames` (3000). Suppression lists are applied every
// step / on the first generated step only, before the timestamp rules.
WhisperDecodeResult run_whisper_decode(
    int total_mel_frames, int window_frames, int decoder_context,
    const WhisperTokenContract &ids, const std::vector<int32_t> &suppress,
    const std::vector<int32_t> &begin_suppress,
    const WhisperDecodeOptions &options, const WhisperModelHooks &hooks);

}  // namespace engine::models::whisper
