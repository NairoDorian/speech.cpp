// test_whisper_c_abi_parity.cpp - Whisper through the public C ABI against the
// engine `whisper` package called directly, in one process, on the same GGUF
// and audio (Phase 11 W2b).
//
// History: this began as whisper_engine_arch_parity_test, the evidence the
// arch retirement (B16c) needed - engine vs src/runtime/arch/whisper across
// language detection, translate, SEGMENT timestamps, an initial prompt and an
// 83 s long-form seek (words exact after normalization; 6/6 modes matched).
// With the arch gone both sides run the same engine code, so the comparison is
// now byte-exact and what it guards is the ArchAdapter: the run-extension ->
// request-option mapping, TaskResult -> segment / language / trace mapping,
// and the capability bits. Any difference is an adapter bug.
//
// It also pins the C-ABI contract the arch used to provide:
//   - capabilities: SEGMENT timestamps max, detection + translate on the
//     multilingual checkpoint, the INITIAL_PROMPT / TEMPERATURE_FALLBACK /
//     LONG_FORM / CANCELLATION feature bits
//   - WORD timestamps -> ERR_UNSUPPORTED_TIMESTAMPS
//   - a run extension the adapter rejects before the dispatcher clears the
//     result (ALL_SEGMENTS without condition_on_prev_tokens) -> ERR_INVALID_ARG
//     with the previous transcript intact
//   - a request the engine rejects on its merits (a special-token literal in
//     initial_prompt) -> ERR_INVALID_ARG, not ERR_BACKEND
//
// Usage: whisper_c_abi_parity_test <whisper-tiny.gguf> <samples_dir>
// Exit 2 (SKIP) while the model or the samples are absent.

#include "abi_test_wav.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/whisper/session.h"
#include "transcribe.h"
#include "transcribe/whisper.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace engine::runtime;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

struct Segment {
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
  std::string text;
  bool operator==(const Segment &o) const {
    return t0_ms == o.t0_ms && t1_ms == o.t1_ms && text == o.text;
  }
};

struct Trace {
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
  float temperature_used = 0.0f;
  float compression_ratio = 0.0f;
  float avg_logprob = 0.0f;
  float no_speech_prob = 0.0f;
  bool no_speech_triggered = false;
  int32_t n_fallbacks = 0;
  bool operator==(const Trace &o) const {
    return t0_ms == o.t0_ms && t1_ms == o.t1_ms && temperature_used == o.temperature_used &&
           compression_ratio == o.compression_ratio && avg_logprob == o.avg_logprob &&
           no_speech_prob == o.no_speech_prob && no_speech_triggered == o.no_speech_triggered &&
           n_fallbacks == o.n_fallbacks;
  }
};

struct Outcome {
  std::string text;
  std::string language;
  std::vector<Segment> segments;
  std::vector<Trace> traces;
  bool truncated = false;
};

struct Case {
  const char *name;
  const char *wav;
  const char *language;  // "" = detect
  bool translate = false;
  bool timestamps = false;
  const char *prompt = nullptr;
};

Outcome run_c_abi(transcribe_session *session, const std::vector<float> &pcm, const Case &c) {
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.language = c.language[0] != '\0' ? c.language : nullptr;
  rp.task = c.translate ? TRANSCRIBE_TASK_TRANSLATE : TRANSCRIBE_TASK_TRANSCRIBE;
  rp.timestamps = c.timestamps ? TRANSCRIBE_TIMESTAMPS_SEGMENT : TRANSCRIBE_TIMESTAMPS_NONE;
  transcribe_whisper_run_ext wp;
  transcribe_whisper_run_ext_init(&wp);
  if (c.prompt != nullptr) {
    wp.initial_prompt = c.prompt;
    rp.family = &wp.ext;
  }
  const transcribe_status st =
      transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
  // OUTPUT_TRUNCATED keeps the partial result readable; whether it matches the
  // engine's own `truncated` is checked in compare().
  if (st != TRANSCRIBE_OK && st != TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
    throw std::runtime_error(std::string("C ABI run failed: ") + transcribe_status_string(st));
  }
  Outcome out;
  out.text = transcribe_full_text(session);
  const char *lang = transcribe_detected_language(session);
  out.language = lang != nullptr ? lang : "";
  if (c.timestamps) {
    for (int i = 0; i < transcribe_n_segments(session); ++i) {
      transcribe_segment seg;
      transcribe_segment_init(&seg);
      if (transcribe_get_segment(session, i, &seg) == TRANSCRIBE_OK) {
        out.segments.push_back({seg.t0_ms, seg.t1_ms, seg.text != nullptr ? seg.text : ""});
      }
    }
  }
  for (int i = 0; i < transcribe_get_whisper_chunk_count(session); ++i) {
    transcribe_whisper_chunk_trace t;
    transcribe_whisper_chunk_trace_init(&t);
    if (transcribe_get_whisper_chunk_trace(session, i, &t) == TRANSCRIBE_OK) {
      out.traces.push_back({t.t0_ms, t.t1_ms, t.temperature_used, t.compression_ratio,
                            t.avg_logprob, t.no_speech_prob, t.no_speech_triggered,
                            t.n_fallbacks});
    }
  }
  out.truncated = transcribe_was_truncated(session);
  // transcribe.h: the flag is true exactly when the run said OUTPUT_TRUNCATED.
  if (out.truncated != (st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED)) {
    std::fprintf(stderr, "FAIL [%s] was_truncated disagrees with the run status\n", c.name);
    ++g_failures;
  }
  return out;
}

Outcome run_engine(IOfflineVoiceTaskSession &session, const AudioBuffer &audio, const Case &c) {
  session.prepare(build_preparation_request(audio));
  TaskRequest request;
  request.audio_input = audio;
  request.options["language"] = c.language;
  request.options["task"] = c.translate ? "translate" : "transcribe";
  request.options["timestamps"] = c.timestamps ? "segment" : "none";
  if (c.prompt != nullptr) {
    request.options["whisper.initial_prompt"] = c.prompt;
  }
  const TaskResult result = session.run(request);
  Outcome out;
  if (result.text_output.has_value()) {
    out.text = result.text_output->text;
    out.language = result.text_output->language;
  }
  for (const auto &seg : result.speech_segments) {
    // The adapter's samples -> ms conversion (16 kHz).
    out.segments.push_back({seg.span.start_sample * 1000 / audio.sample_rate,
                            seg.span.end_sample * 1000 / audio.sample_rate, seg.text});
  }
  for (const auto &t : result.decode_telemetry) {
    out.traces.push_back({t.t0_ms, t.t1_ms, t.temperature_used, t.compression_ratio,
                          t.avg_logprob, t.no_speech_prob, t.no_speech_triggered, t.n_fallbacks});
  }
  out.truncated = result.truncated;
  return out;
}

void compare(const Case &c, const Outcome &abi, const Outcome &engine) {
  bool ok = true;
  auto fail = [&](const std::string &what) {
    std::fprintf(stderr, "FAIL [%s] %s\n", c.name, what.c_str());
    ok = false;
    ++g_failures;
  };
  if (abi.text != engine.text) {
    fail("text\n    c abi : " + abi.text + "\n    engine: " + engine.text);
  }
  if (abi.language != engine.language) {
    fail("language: c abi '" + abi.language + "' vs engine '" + engine.language + "'");
  }
  if (abi.segments != engine.segments) {
    fail("segments: c abi " + std::to_string(abi.segments.size()) + " vs engine " +
         std::to_string(engine.segments.size()) + " (or different boundaries / text)");
  }
  if (abi.traces != engine.traces) {
    fail("window traces: c abi " + std::to_string(abi.traces.size()) + " vs engine " +
         std::to_string(engine.traces.size()) + " (or different values)");
  }
  if (abi.truncated != engine.truncated) {
    fail("truncated flag not propagated");
  }
  std::printf("[%s] %s  (%zu segment(s), %zu window(s)%s%s)\n", c.name, ok ? "match" : "DIFFER",
              engine.segments.size(), engine.traces.size(),
              engine.language.empty() ? "" : ", language ", engine.language.c_str());
  if (ok) {
    std::printf("    %s\n", engine.text.c_str());
  }
}

void check_capabilities(const transcribe_model *model) {
  CHECK(std::string(transcribe_model_arch_string(model)) == "whisper");
  transcribe_capabilities caps;
  transcribe_capabilities_init(&caps);
  CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
  CHECK(caps.max_timestamp_kind == TRANSCRIBE_TIMESTAMPS_SEGMENT);
  CHECK(caps.supports_language_detect);  // multilingual checkpoint
  CHECK(caps.supports_translate);
  CHECK(!caps.supports_streaming);
  CHECK(caps.n_languages > 1);
  CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_INITIAL_PROMPT));
  CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK));
  CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_LONG_FORM));
  CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION));
  CHECK(!transcribe_model_supports(model, TRANSCRIBE_FEATURE_DIARIZATION));
}

// Rejections, run after a successful jfk run so a surviving transcript is
// observable.
void check_rejections(transcribe_session *session, const std::vector<float> &pcm) {
  const std::string before = transcribe_full_text(session);
  CHECK(!before.empty());

  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.language = "en";

  // WORD timing is finer than Whisper's segment timestamps.
  rp.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
  CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) ==
        TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS);
  rp.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;

  // Rejected by run_validate, before the result is cleared.
  transcribe_whisper_run_ext wp;
  transcribe_whisper_run_ext_init(&wp);
  wp.prompt_condition = TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS;  // needs condition_on_prev_tokens
  rp.family = &wp.ext;
  CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) ==
        TRANSCRIBE_ERR_INVALID_ARG);
  CHECK(std::string(transcribe_full_text(session)) == before);

  // Rejected by the engine (needs the vocabulary): HF get_prompt_ids refuses
  // special tokens in prompt text.
  transcribe_whisper_run_ext_init(&wp);
  wp.initial_prompt = "hello <|en|> world";
  CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) ==
        TRANSCRIBE_ERR_INVALID_ARG);

  // A leading <|startofprev|> in prompt_tokens would be prepended twice.
  // 50361 is its id in every multilingual 51865-token checkpoint (tiny..large-v2).
  transcribe_whisper_run_ext_init(&wp);
  const int32_t ids[] = {50361, 440};
  wp.prompt_tokens = ids;
  wp.n_prompt_tokens = 2;
  CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) ==
        TRANSCRIBE_ERR_INVALID_ARG);

  // Traces read back zeroed out of range, and the session still works.
  transcribe_whisper_chunk_trace t;
  transcribe_whisper_chunk_trace_init(&t);
  t.temperature_used = 9.0f;
  CHECK(transcribe_get_whisper_chunk_trace(session, 1000, &t) == TRANSCRIBE_OK);
  CHECK(t.temperature_used == 0.0f);
  rp.family = nullptr;
  CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
  CHECK(std::string(transcribe_full_text(session)) == before);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <whisper-tiny.gguf> <samples_dir>\n", argv[0]);
    return 1;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path samples = argv[2];
  if (!std::filesystem::exists(model_path) || !std::filesystem::exists(samples / "jfk.wav")) {
    std::printf("SKIP: model or samples absent\n");
    return 2;
  }

  const std::vector<Case> cases = {
      {"jfk-detect", "jfk.wav", ""},
      {"jfk-segments", "jfk.wav", "en", false, true},
      {"jfk-prompt", "jfk.wav", "en", false, false, "Inaugural address, 1961."},
      {"german-detect", "german.wav", ""},
      {"german-translate", "german.wav", "de", true},
      {"whole-earth-longform-segments", "whole-earth.wav", "en", false, true},
  };

  try {
    // The C ABI (dispatcher -> ArchAdapter -> engine).
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    transcribe_model *abi_model = nullptr;
    if (transcribe_model_load_file(model_path.string().c_str(), &mp, &abi_model) != TRANSCRIBE_OK) {
      std::fprintf(stderr, "FAIL: C ABI could not load %s\n", model_path.string().c_str());
      return 1;
    }
    check_capabilities(abi_model);
    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    transcribe_session *abi_session = nullptr;
    if (transcribe_session_init(abi_model, &sp, &abi_session) != TRANSCRIBE_OK) {
      std::fprintf(stderr, "FAIL: C ABI session init\n");
      return 1;
    }

    // The engine directly, through the registry.
    ModelRegistry registry;
    registry.register_loader(engine::models::whisper::make_whisper_loader());
    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "whisper";
    auto loaded = registry.load(request);
    TaskSpec task;
    task.task = VoiceTaskKind::Asr;
    task.mode = RunMode::Offline;
    auto session = loaded->create_task_session(task, SessionOptions{});
    auto *engine = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    if (engine == nullptr) {
      std::fprintf(stderr, "FAIL: engine session is not offline\n");
      return 1;
    }

    std::vector<float> jfk;
    for (const auto &c : cases) {
      int rate = 0;
      AudioBuffer audio;
      audio.samples = abi_test::read_wav_mono_f32((samples / c.wav).string(), rate);
      audio.sample_rate = rate;
      audio.channels = 1;
      if (rate != 16000) {
        std::fprintf(stderr, "FAIL: %s is not 16 kHz\n", c.wav);
        ++g_failures;
        continue;
      }
      const Outcome abi = run_c_abi(abi_session, audio.samples, c);
      const Outcome eng = run_engine(*engine, audio, c);
      compare(c, abi, eng);
      if (std::strcmp(c.wav, "jfk.wav") == 0) {
        jfk = audio.samples;
      }
    }

    // Leave a known-good English transcript in the session, then probe the
    // rejection paths against it.
    {
      const Case c{"jfk-en", "jfk.wav", "en"};
      (void) run_c_abi(abi_session, jfk, c);
      check_rejections(abi_session, jfk);
    }

    transcribe_session_free(abi_session);
    transcribe_model_free(abi_model);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "whisper_c_abi_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("whisper_c_abi_parity_test: C ABI == engine in every mode; C-ABI contract holds\n");
  return 0;
}
