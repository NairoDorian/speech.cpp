// engine/models/whisper/decoding.cpp - see decoding.h.
//
// Ported from src/runtime/arch/whisper/model.cpp (deleted in B16c; whisper_run,
// apply_whisper_timestamp_rules, whisper_retrieve_segment), which matched HF
// transformers' generation_whisper.py. The structure follows the arch so the
// two can be diffed while both exist; keep the arithmetic and the order of
// operations identical - the retirement gate compares their outputs.

#include "engine/models/whisper/decoding.h"

#include "engine/framework/asr/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace engine::models::whisper {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Mel frames per timestamp position: the encoder halves the frame rate.
constexpr int kInputStride = 2;
// Milliseconds per mel frame (hop 160 at 16 kHz).
constexpr int64_t kMsPerMelFrame = 10;

void suppress(std::vector<float> &logits, const std::vector<int32_t> &ids) {
  const int vocab = static_cast<int>(logits.size());
  for (const int32_t id : ids) {
    if (id >= 0 && id < vocab) {
      logits[static_cast<size_t>(id)] = kNegInf;
    }
  }
}

// Softmax probability of `token` in a raw logits row (no-speech gate).
float softmax_probability(const std::vector<float> &logits, int32_t token) {
  float max_l = kNegInf;
  for (const float l : logits) {
    if (std::isfinite(l) && l > max_l) {
      max_l = l;
    }
  }
  if (!std::isfinite(max_l)) {
    return 0.0f;
  }
  double sum = 0.0;
  double hit = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    const float l = logits[i];
    if (std::isfinite(l)) {
      const double e = std::exp(static_cast<double>(l - max_l));
      sum += e;
      if (static_cast<int32_t>(i) == token) {
        hit = e;
      }
    }
  }
  return sum > 0.0 ? static_cast<float>(hit / sum) : 0.0f;
}

void validate(const WhisperTokenContract &ids,
              const WhisperDecodeOptions &options) {
  if (ids.vocab_size <= 0 || ids.eot < 0 || ids.sot < 0 ||
      ids.no_timestamps < 0 || ids.timestamp_begin() >= ids.vocab_size) {
    throw std::runtime_error("whisper: incomplete special-token contract");
  }
  if (options.prompt_all_segments && !options.condition_on_prev_tokens) {
    // HF raises ValueError for prompt_condition_type="all-segments" without
    // condition_on_prev_tokens.
    throw std::invalid_argument(
        "whisper: prompt_condition=all_segments requires "
        "condition_on_prev_tokens");
  }
  const bool wants_context =
      !options.prompt_ids.empty() || options.condition_on_prev_tokens;
  if (wants_context && ids.prev_sot < 0) {
    throw std::invalid_argument(
        "whisper: this model has no <|startofprev|> token, so initial prompts "
        "and condition_on_prev_tokens are unavailable");
  }
  if (!ids.multilingual) {
    if (options.translate) {
      throw std::invalid_argument(
          "whisper: translate needs a multilingual model (this is an "
          "English-only variant)");
    }
    if (options.language_token.has_value()) {
      throw std::invalid_argument(
          "whisper: an English-only model takes no language token");
    }
  } else {
    if (ids.transcribe < 0 || (options.translate && ids.translate < 0)) {
      throw std::runtime_error("whisper: task tokens missing from the vocab");
    }
    if (!options.language_token.has_value() &&
        options.language_candidates.empty()) {
      throw std::runtime_error(
          "whisper: multilingual model has no language tokens to detect with");
    }
  }
  for (const int32_t id : options.prompt_ids) {
    if (!ids.is_text(id)) {
      // HF get_prompt_ids rejects special tokens in prompt text.
      throw std::invalid_argument(
          "whisper: initial prompt contains a special token (id " +
          std::to_string(id) + ")");
    }
  }
}

}  // namespace

void apply_timestamp_rules(float *logits, const std::vector<int32_t> &generated,
                           const WhisperTokenContract &ids,
                           int max_initial_timestamp_index) {
  const int vocab = ids.vocab_size;
  const int ts_begin = ids.timestamp_begin();

  // 1. <|notimestamps|> is never generated in timestamp mode.
  if (ids.no_timestamps >= 0 && ids.no_timestamps < vocab) {
    logits[ids.no_timestamps] = kNegInf;
  }

  // 2. Pairing: after "<text> <ts>" the pair must close (a timestamp or EOT);
  //    after "<ts> <ts>" it is complete and the next token must be text.
  const bool last_ts = !generated.empty() && ids.is_timestamp(generated.back());
  const bool penult_ts = generated.size() < 2 ||
                         ids.is_timestamp(generated[generated.size() - 2]);
  if (last_ts) {
    if (penult_ts) {
      for (int i = ts_begin; i < vocab; ++i) {
        logits[i] = kNegInf;
      }
    } else {
      const int mask_hi = std::min(ids.eot, vocab);
      for (int i = 0; i < mask_hi; ++i) {
        logits[i] = kNegInf;
      }
    }
  }

  // 3. Monotonicity: timestamps never decrease (a closed pair may not repeat
  //    its own end).
  int last_ts_id = -1;
  for (auto it = generated.rbegin(); it != generated.rend(); ++it) {
    if (ids.is_timestamp(*it)) {
      last_ts_id = *it;
      break;
    }
  }
  if (last_ts_id >= 0) {
    const int ts_low = (last_ts && !penult_ts) ? last_ts_id : last_ts_id + 1;
    const int clamp_hi = std::min(ts_low, vocab);
    for (int i = ts_begin; i < clamp_hi; ++i) {
      logits[i] = kNegInf;
    }
  }

  // 4. The first generated token is a timestamp, at most
  //    max_initial_timestamp in.
  if (generated.empty()) {
    for (int i = 0; i < ts_begin; ++i) {
      logits[i] = kNegInf;
    }
    if (max_initial_timestamp_index >= 0) {
      const int last_allowed = ts_begin + max_initial_timestamp_index;
      for (int i = last_allowed + 1; i < vocab; ++i) {
        logits[i] = kNegInf;
      }
    }
  }

  // 5. Force a timestamp when their summed probability beats every text token.
  float max_ts = kNegInf;
  for (int i = ts_begin; i < vocab; ++i) {
    max_ts = std::max(max_ts, logits[i]);
  }
  float ts_logsumexp = kNegInf;
  if (std::isfinite(max_ts)) {
    double sum = 0.0;
    for (int i = ts_begin; i < vocab; ++i) {
      if (std::isfinite(logits[i])) {
        sum += std::exp(static_cast<double>(logits[i] - max_ts));
      }
    }
    if (sum > 0.0) {
      ts_logsumexp = max_ts + static_cast<float>(std::log(sum));
    }
  }
  float max_text = kNegInf;
  for (int i = 0; i < ts_begin; ++i) {
    max_text = std::max(max_text, logits[i]);
  }
  if (ts_logsumexp > max_text) {
    for (int i = 0; i < ts_begin; ++i) {
      logits[i] = kNegInf;
    }
  }
}

RetrievedWindow retrieve_segments(const std::vector<int32_t> &generated,
                                  const WhisperTokenContract &ids,
                                  int64_t time_offset_ms, int seek_num_frames,
                                  bool want_segments) {
  RetrievedWindow out;
  out.seek_advance_frames = seek_num_frames;

  const int ts_begin = ids.timestamp_begin();
  const int gn = static_cast<int>(generated.size());
  auto text_between = [&](int begin, int end) {
    std::vector<int32_t> text;
    for (int i = begin; i < end; ++i) {
      if (ids.is_text(generated[i])) {
        text.push_back(generated[i]);
      }
    }
    return text;
  };

  // Split points: after every consecutive <ts><ts> pair.
  std::vector<int> slices;
  for (int i = 0; i + 1 < gn; ++i) {
    if (ids.is_timestamp(generated[i]) && ids.is_timestamp(generated[i + 1])) {
      slices.push_back(i + 1);
    }
  }
  const bool single_timestamp_ending = gn >= 2 &&
                                       !ids.is_timestamp(generated[gn - 2]) &&
                                       ids.is_timestamp(generated[gn - 1]);

  if (!slices.empty()) {
    if (single_timestamp_ending) {
      slices.push_back(gn);
    } else {
      slices.back() += 1;
    }

    int last_slice = 0;
    for (size_t k = 0; k < slices.size(); ++k) {
      const int current_slice = slices[k];
      const bool is_last = (k + 1 == slices.size());
      if (current_slice <= last_slice) {
        last_slice = current_slice;
        continue;
      }
      const int32_t first_tok = generated[last_slice];
      if (!ids.is_timestamp(first_tok)) {
        last_slice = current_slice;
        continue;
      }
      const int end_idx = (!is_last || single_timestamp_ending)
                              ? current_slice - 1
                              : current_slice - 2;
      if (end_idx < last_slice || end_idx >= gn) {
        last_slice = current_slice;
        continue;
      }
      const int32_t last_tok = generated[end_idx];
      if (!ids.is_timestamp(last_tok)) {
        last_slice = current_slice;
        continue;
      }
      if (want_segments) {
        WhisperSegment seg;
        seg.t0_ms = time_offset_ms + static_cast<int64_t>(first_tok - ts_begin) * 20;
        seg.t1_ms = time_offset_ms + static_cast<int64_t>(last_tok - ts_begin) * 20;
        seg.t1_ms = std::max(seg.t1_ms, seg.t0_ms);
        seg.text_ids = text_between(last_slice, current_slice);
        if (!seg.text_ids.empty()) {
          out.segments.push_back(std::move(seg));
        }
      }
      out.history_slices.emplace_back(generated.begin() + last_slice,
                                      generated.begin() + current_slice);
      last_slice = current_slice;
    }

    if (single_timestamp_ending) {
      out.seek_advance_frames = seek_num_frames;
    } else if (last_slice >= 2 && last_slice - 2 < gn &&
               ids.is_timestamp(generated[last_slice - 2])) {
      out.seek_advance_frames =
          (generated[last_slice - 2] - ts_begin) * kInputStride;
    }
  } else {
    int64_t last_ts_pos = static_cast<int64_t>(seek_num_frames) / kInputStride;
    for (int i = gn - 1; i >= 0; --i) {
      if (ids.is_timestamp(generated[i]) && generated[i] != ts_begin) {
        last_ts_pos = generated[i] - ts_begin;
        break;
      }
    }
    if (want_segments && gn > 0) {
      WhisperSegment seg;
      seg.t0_ms = time_offset_ms;
      seg.t1_ms = time_offset_ms + last_ts_pos * 20;
      seg.text_ids = text_between(0, gn);
      if (!seg.text_ids.empty()) {
        out.segments.push_back(std::move(seg));
      }
    }
    if (gn > 0) {
      out.history_slices.emplace_back(generated.begin(), generated.end());
    }
    out.seek_advance_frames = seek_num_frames;
  }
  return out;
}

WindowPrefix build_window_prefix(
    const WhisperTokenContract &ids, const WhisperDecodeOptions &options,
    int32_t language_token,
    const std::vector<std::vector<int32_t>> &history, bool is_first_window,
    bool condition_on_prev) {
  WindowPrefix prefix;
  std::vector<int32_t> &out = prefix.tokens;
  const int max_prev = std::max(0, options.max_prev_context_tokens);

  if (condition_on_prev && !history.empty() && ids.prev_sot >= 0) {
    out.push_back(ids.prev_sot);
    if (options.prompt_all_segments) {
      out.insert(out.end(), options.prompt_ids.begin(), options.prompt_ids.end());
    }
    std::vector<int32_t> carried;
    for (const auto &slice : history) {
      size_t n = slice.size();
      // skip_ending_double_timestamps: a slice ending "<ts><ts>" carries only
      // the first of the pair.
      if (n > 2 && ids.is_timestamp(slice[n - 2])) {
        --n;
      }
      carried.insert(carried.end(), slice.begin(), slice.begin() + static_cast<ptrdiff_t>(n));
    }
    const int keep = std::min<int>(static_cast<int>(carried.size()), max_prev);
    out.insert(out.end(), carried.end() - keep, carried.end());
  } else if (!options.prompt_ids.empty() && ids.prev_sot >= 0 &&
             (is_first_window || options.prompt_all_segments)) {
    // FIRST_SEGMENT primes only the first window (OpenAI / whisper.cpp
    // behaviour; HF re-sends it - the arch chose this deliberately).
    out.push_back(ids.prev_sot);
    out.insert(out.end(), options.prompt_ids.begin(), options.prompt_ids.end());
  }

  prefix.sot_index = static_cast<int>(out.size());
  out.push_back(ids.sot);
  if (ids.multilingual) {
    out.push_back(language_token);
    out.push_back(options.translate ? ids.translate : ids.transcribe);
  }
  if (!options.timestamps) {
    out.push_back(ids.no_timestamps);
  }
  return prefix;
}

WhisperDecodeResult run_whisper_decode(
    int total_mel_frames, int window_frames, int decoder_context,
    const WhisperTokenContract &ids, const std::vector<int32_t> &suppress_ids,
    const std::vector<int32_t> &begin_suppress_ids,
    const WhisperDecodeOptions &options_in, const WhisperModelHooks &hooks) {
  validate(ids, options_in);
  if (total_mel_frames <= 0 || window_frames <= 0 || decoder_context <= 0) {
    throw std::invalid_argument("whisper: empty decode geometry");
  }

  WhisperDecodeOptions options = options_in;
  // The prompt keeps its most recent tokens when it exceeds the carry cap.
  const int max_prev = std::max(0, options.max_prev_context_tokens);
  if (static_cast<int>(options.prompt_ids.size()) > max_prev) {
    options.prompt_ids.erase(options.prompt_ids.begin(),
                             options.prompt_ids.end() - max_prev);
  }

  WhisperDecodeResult result;
  const int vocab = ids.vocab_size;
  std::vector<float> logits(static_cast<size_t>(vocab));
  std::vector<float> sot_logits(static_cast<size_t>(vocab));
  std::vector<double> sample_scratch;

  // One generator for the whole run: HF threads a single generator through
  // the seek loop, so reseeding per window would replay the same draws.
  std::mt19937 rng(options.seed != 0 ? options.seed : std::random_device{}());

  std::vector<std::vector<int32_t>> history;
  if (!options.prompt_all_segments && !options.prompt_ids.empty()) {
    history.push_back(options.prompt_ids);
  }
  bool condition_on_prev = options.condition_on_prev_tokens;

  int32_t language_token = ids.multilingual && options.language_token.has_value()
                               ? *options.language_token
                               : -1;

  // Temperature ladder [t0, t0 + dt, ...] up to 1.0.
  std::vector<float> temperatures{options.temperature};
  if (options.temperature_inc > 0.0f) {
    for (float t = options.temperature + options.temperature_inc; t <= 1.0f + 1e-4f;
         t += options.temperature_inc) {
      temperatures.push_back(t);
    }
  }
  const int max_initial_ts_index =
      (std::isfinite(options.max_initial_timestamp) &&
       options.max_initial_timestamp >= 0.0f)
          ? static_cast<int>(std::floor(
                static_cast<double>(options.max_initial_timestamp) / 0.02))
          : -1;

  auto poll = [&](int done) {
    if (hooks.poll) {
      hooks.poll(done, total_mel_frames);
    }
  };
  auto pick = [&](float temperature, float &logprob_out) {
    if (temperature <= 0.0f) {
      return engine::asr::argmax_with_logprob(logits.data(), vocab, &logprob_out);
    }
    const int id = engine::asr::sample_logits(logits.data(), vocab, temperature,
                                              rng, sample_scratch);
    logprob_out = engine::asr::token_logprob(logits.data(), vocab, id, temperature);
    return id;
  };

  int seek = 0;
  try {
    while (seek < total_mel_frames) {
      poll(seek);
      const bool is_first_window = (seek == 0);
      const int64_t time_offset_ms = static_cast<int64_t>(seek) * kMsPerMelFrame;
      const int seek_num_frames = std::min(total_mel_frames - seek, window_frames);

      hooks.encode_window(seek, seek_num_frames);

      // Language detection: once, on the first window, when no hint was given.
      if (is_first_window && ids.multilingual && language_token < 0) {
        hooks.prefill({ids.sot}, -1, sot_logits, logits);
        float best = kNegInf;
        for (const int32_t candidate : options.language_candidates) {
          if (candidate >= 0 && candidate < vocab && logits[candidate] > best) {
            best = logits[candidate];
            language_token = candidate;
          }
        }
        if (language_token < 0) {
          throw std::runtime_error("whisper: language detection found no candidate");
        }
        result.language_detected = true;
      }
      result.language_token = language_token;

      const WindowPrefix prefix = build_window_prefix(
          ids, options, language_token, history, is_first_window, condition_on_prev);
      const int prefix_len = static_cast<int>(prefix.tokens.size());
      if (prefix_len + 1 > decoder_context) {
        throw std::invalid_argument(
            "whisper: prompt prefix (" + std::to_string(prefix_len) +
            " tokens) does not fit the decoder context (" +
            std::to_string(decoder_context) + ")");
      }

      std::vector<int32_t> generated;
      std::vector<int32_t> accepted;
      float accepted_t = 0.0f;
      float accepted_ratio = 0.0f;
      float accepted_logprob = 0.0f;
      int accepted_fallbacks = 0;
      bool accepted_hit_eos = true;
      float no_speech_prob = 0.0f;
      bool no_speech_fired = false;

      for (size_t tier = 0; tier < temperatures.size(); ++tier) {
        const float t = temperatures[tier];
        generated.clear();
        bool hit_eos = false;
        double sum_logprob = 0.0;
        int n_logprob = 0;

        // Prompt pass. The no-speech probability is read from the raw SOT row
        // on tier 0 only (HF WhisperNoSpeechDetection), before any masking.
        hooks.prefill(prefix.tokens, tier == 0 ? prefix.sot_index : -1, sot_logits,
                      logits);
        if (tier == 0) {
          no_speech_prob = softmax_probability(sot_logits, ids.no_speech());
        }
        suppress(logits, suppress_ids);
        suppress(logits, begin_suppress_ids);
        if (options.timestamps) {
          apply_timestamp_rules(logits.data(), generated, ids, max_initial_ts_index);
        }
        float lp = 0.0f;
        int next = pick(t, lp);
        sum_logprob += lp;
        ++n_logprob;

        int n_past = prefix_len;
        for (int produced = 0; produced < options.max_new_tokens; ++produced) {
          if (next == ids.eot) {
            hit_eos = true;
            break;
          }
          poll(seek);
          generated.push_back(next);
          if (n_past + 1 > decoder_context) {
            break;
          }
          hooks.step(next, n_past, logits);
          ++n_past;
          suppress(logits, suppress_ids);
          if (options.timestamps) {
            apply_timestamp_rules(logits.data(), generated, ids, max_initial_ts_index);
          }
          next = pick(t, lp);
          sum_logprob += lp;
          ++n_logprob;
        }

        // HF _retrieve_compression_ratio scores the whole generated tail,
        // including timestamps and the EOS when one was produced.
        std::vector<int32_t> tail = generated;
        if (hit_eos) {
          tail.push_back(ids.eot);
        }
        const float ratio =
            engine::asr::token_compression_ratio(tail.data(), tail.size(), vocab);
        const float avg_logprob =
            n_logprob > 0 ? static_cast<float>(sum_logprob / n_logprob)
                          : -std::numeric_limits<float>::infinity();

        // HF _need_fallback: escalate strictly above / below the thresholds
        // (equality accepts); the +/-INF sentinels therefore disable them.
        const bool ratio_ok = ratio <= options.compression_ratio_thold;
        const bool logprob_ok = avg_logprob >= options.logprob_thold;
        const bool skip_no_speech = no_speech_prob > options.no_speech_thold &&
                                    avg_logprob < options.logprob_thold;

        // The last tier tried wins when none passes (generate_with_fallback).
        accepted = generated;
        accepted_t = t;
        accepted_ratio = ratio;
        accepted_logprob = avg_logprob;
        accepted_fallbacks = static_cast<int>(tier);
        accepted_hit_eos = hit_eos;
        if (skip_no_speech) {
          no_speech_fired = true;
          break;
        }
        if (ratio_ok && logprob_ok) {
          break;
        }
      }

      if (no_speech_fired) {
        accepted.clear();
      }
      // The window ran out of token budget before <|endoftext|>. Whether text
      // was lost depends on the seek below.
      const bool window_cut = !no_speech_fired && !accepted_hit_eos;

      runtime::DecodeTelemetry trace;
      trace.t0_ms = time_offset_ms;
      trace.t1_ms = time_offset_ms + static_cast<int64_t>(seek_num_frames) * kMsPerMelFrame;
      trace.temperature_used = accepted_t;
      trace.compression_ratio = accepted_ratio;
      trace.avg_logprob = accepted_logprob;
      trace.no_speech_prob = no_speech_prob;
      trace.no_speech_triggered = no_speech_fired;
      trace.n_fallbacks = accepted_fallbacks;
      result.traces.push_back(trace);

      // A hot accepted tier (>= 0.5) is presumed hallucinated: do not carry it.
      condition_on_prev = options.condition_on_prev_tokens && accepted_t < 0.5f;

      RetrievedWindow window = retrieve_segments(accepted, ids, time_offset_ms,
                                                 seek_num_frames, options.timestamps);
      for (auto &seg : window.segments) {
        result.segments.push_back(std::move(seg));
      }
      if (!no_speech_fired && !window.history_slices.empty()) {
        history.insert(history.end(), window.history_slices.begin(),
                       window.history_slices.end());
      }
      for (const int32_t id : accepted) {
        if (ids.is_text(id)) {
          result.text_ids.push_back(id);
        }
      }
      result.raw_ids.insert(result.raw_ids.end(), accepted.begin(), accepted.end());

      // Unified seek advance: a lone closing timestamp or no pairs advances the
      // whole window; a "<ts><ts>" ending re-enters at that timestamp. Guarded
      // against a zero advance.
      int advance = no_speech_fired ? seek_num_frames : window.seek_advance_frames;
      if (advance <= 0) {
        advance = seek_num_frames;
      }
      // L11, precisely: a cut window loses text only when the seek moves past
      // the whole window. When it re-enters at the last closed timestamp pair,
      // the unfinished segment is decoded again by the next window (HF's
      // generate_with_fallback behaves the same), so nothing is missing. The
      // C ABI turns `truncated` into TRANSCRIBE_ERR_OUTPUT_TRUNCATED, so a
      // false positive would fail runs the retired arch completed.
      if (window_cut && advance >= seek_num_frames) {
        result.truncated = true;
      }
      seek += advance;
    }
    poll(total_mel_frames);
  } catch (...) {
    // `result` holds only accepted windows: a window's segments / ids are
    // appended after its fallback loop settles.
    if (hooks.on_interrupted) {
      hooks.on_interrupted(result);
    }
    throw;
  }
  return result;
}

}  // namespace engine::models::whisper
