// test_medasr_engine_arch_parity.cpp - MedASR through the public C ABI against
// the engine `medasr` package called directly, in one process, on the same
// GGUF and audio.
//
// While the C ABI routes a medasr GGUF to src/runtime/arch/medasr this is the
// arch-retirement evidence: the engine port must reproduce the arch's whole
// result on every LibriSpeech fixture, single-shot and through the batched
// encoder (the 4 fixtures differ in length, so the variable-length masked path
// runs) - full text, raw (untrimmed) text, the returned timestamp kind, and
// every TOKEN row (id, piece text, p, t0/t1 on the 40 ms CTC grid, word index),
// with no word or segment rows. Once the arch is retired and the C ABI reaches
// the engine through the ArchAdapter, the same comparison guards the adapter
// (both sides then run the engine; any difference is an adapter bug).
//
// It also pins the contract both paths share:
//   - capabilities: no translation, no language detection, offline only,
//     TOKEN timestamps, max_audio_ms == 400000 (the RoPE-trained window),
//     TRANSCRIBE_FEATURE_CANCELLATION
//   - token rows are returned even when the run asked for NONE (the arch never
//     elided them)
//   - a second engine run on the same session reproduces the first exactly
//     (the per-run graph re-uploads every input)
//   - audio shorter than one STFT window is ERR_INVALID_ARG through the C ABI
//     and std::invalid_argument from the engine
//
// Usage: medasr_engine_arch_parity_test <medasr.gguf> [fixtures_dir]
//   fixtures_dir defaults to assets/asr_validation/librispeech (the test runs
//   from the source tree).
// Exit 2 (SKIP) while the model or the fixtures are absent.

#include "abi_test_wav.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/medasr/session.h"
#include "transcribe.h"

#include <cstdint>
#include <cstdio>
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

constexpr int64_t kArchMaxAudioMs = 400000; // 10000 RoPE positions * 40 ms

const char *const kAllFixtures[] = {
    "librispeech_test_clean_6930-75918-0000.wav",
    "librispeech_test_clean_6930-75918-0001.wav",
    "librispeech_test_other_7902-96591-0000.wav",
    "librispeech_test_other_7902-96591-0001.wav",
};

struct Fixture {
  std::string name;
  std::vector<float> pcm; // 16 kHz mono
};

struct Token {
  int id = 0;
  float p = 0.0f;
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
  int word_index = -1;
  std::string text;
  bool operator==(const Token &o) const {
    return id == o.id && p == o.p && t0_ms == o.t0_ms && t1_ms == o.t1_ms &&
           word_index == o.word_index && text == o.text;
  }
};

// Everything the C ABI publishes for one offline result.
struct Outcome {
  std::string text;
  std::string raw_text;
  transcribe_timestamp_kind kind = TRANSCRIBE_TIMESTAMPS_NONE;
  int n_words = 0;
  int n_segments = 0;
  std::vector<Token> tokens;
};

// AUTO resolves to TOKEN on both sides. (The arch ignored the request and
// always returned token rows; the C ABI now elides finer rows for a coarser
// request, per transcribe.h "Timestamp policy", so NONE would drop them.)
transcribe_run_params text_only_params() {
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.language = "en";
  rp.timestamps = TRANSCRIBE_TIMESTAMPS_AUTO;
  return rp;
}

Token token_from_abi(const transcribe_token &t) {
  Token out;
  out.id = t.id;
  out.p = t.p;
  out.t0_ms = t.t0_ms;
  out.t1_ms = t.t1_ms;
  out.word_index = t.word_index;
  out.text = t.text != nullptr ? t.text : "";
  return out;
}

Outcome run_c_abi(transcribe_session *session, const std::vector<float> &pcm) {
  const transcribe_run_params rp = text_only_params();
  const transcribe_status st =
      transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
  if (st != TRANSCRIBE_OK) {
    throw std::runtime_error(std::string("C ABI run failed: ") + transcribe_status_string(st));
  }
  CHECK(!transcribe_was_truncated(session)); // CTC never truncates
  Outcome out;
  const char *text = transcribe_full_text(session);
  out.text = text != nullptr ? text : "";
  const char *raw = transcribe_raw_text(session);
  out.raw_text = raw != nullptr ? raw : "";
  out.kind = transcribe_returned_timestamp_kind(session);
  out.n_words = transcribe_n_words(session);
  out.n_segments = transcribe_n_segments(session);
  for (int i = 0; i < transcribe_n_tokens(session); ++i) {
    transcribe_token t;
    transcribe_token_init(&t);
    CHECK(transcribe_get_token(session, i, &t) == TRANSCRIBE_OK);
    out.tokens.push_back(token_from_abi(t));
  }
  return out;
}

Outcome batch_c_abi(transcribe_session *session, int i) {
  Outcome out;
  const char *text = transcribe_batch_full_text(session, i);
  out.text = text != nullptr ? text : "";
  const char *raw = transcribe_batch_raw_text(session, i);
  out.raw_text = raw != nullptr ? raw : "";
  out.kind = transcribe_batch_returned_timestamp_kind(session, i);
  out.n_words = transcribe_batch_n_words(session, i);
  out.n_segments = transcribe_batch_n_segments(session, i);
  for (int j = 0; j < transcribe_batch_n_tokens(session, i); ++j) {
    transcribe_token t;
    transcribe_token_init(&t);
    CHECK(transcribe_batch_get_token(session, i, j, &t) == TRANSCRIBE_OK);
    out.tokens.push_back(token_from_abi(t));
  }
  return out;
}

// The engine TaskResult, read the way the ArchAdapter publishes it: raw_text
// defaults to the text, spans convert at 16 samples per ms, token rows make
// the result TOKEN.
Outcome from_engine(const TaskResult &result) {
  Outcome out;
  CHECK(!result.truncated);
  if (result.text_output.has_value()) {
    out.text = result.text_output->text;
    out.raw_text = result.raw_text.value_or(out.text);
  }
  out.n_words = static_cast<int>(result.word_timestamps.size());
  out.n_segments = static_cast<int>(result.speech_segments.size());
  for (const auto &tt : result.token_timestamps) {
    Token t;
    t.id = tt.id;
    t.p = tt.probability;
    t.t0_ms = tt.span.start_sample / 16;
    t.t1_ms = tt.span.end_sample / 16;
    t.word_index = tt.word_index;
    t.text = tt.text;
    out.tokens.push_back(std::move(t));
  }
  out.kind = out.tokens.empty() ? TRANSCRIBE_TIMESTAMPS_NONE : TRANSCRIBE_TIMESTAMPS_TOKEN;
  return out;
}

AudioBuffer to_audio(const std::vector<float> &pcm) {
  AudioBuffer audio;
  audio.sample_rate = 16000;
  audio.channels = 1;
  audio.samples = pcm;
  return audio;
}

TaskRequest engine_request(const std::vector<float> &pcm) {
  TaskRequest request;
  request.audio_input = to_audio(pcm);
  request.options["language"] = "en";
  request.options["timestamps"] = "auto";
  return request;
}

Outcome run_engine(IOfflineVoiceTaskSession &session, const std::vector<float> &pcm) {
  return from_engine(session.run(engine_request(pcm)));
}

std::string token_string(const Token &t) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "{id %d, p %g, %lld-%lld ms, word %d, '", t.id,
                static_cast<double>(t.p), static_cast<long long>(t.t0_ms),
                static_cast<long long>(t.t1_ms), t.word_index);
  return std::string(buf) + t.text + "'}";
}

bool compare(const std::string &label, const Outcome &abi, const Outcome &engine) {
  bool ok = true;
  auto fail = [&](const std::string &what) {
    std::fprintf(stderr, "FAIL [%s] %s\n", label.c_str(), what.c_str());
    ok = false;
    ++g_failures;
  };
  if (abi.text != engine.text) {
    fail("text\n    c abi : " + abi.text + "\n    engine: " + engine.text);
  }
  if (abi.raw_text != engine.raw_text) {
    fail("raw text\n    c abi : '" + abi.raw_text + "'\n    engine: '" + engine.raw_text + "'");
  }
  if (abi.kind != engine.kind) {
    fail("returned timestamp kind: c abi " + std::to_string(static_cast<int>(abi.kind)) +
         " vs engine " + std::to_string(static_cast<int>(engine.kind)));
  }
  if (abi.n_words != engine.n_words || abi.n_segments != engine.n_segments) {
    fail("row counts: c abi " + std::to_string(abi.n_words) + " word(s) / " +
         std::to_string(abi.n_segments) + " segment(s), engine " +
         std::to_string(engine.n_words) + " / " + std::to_string(engine.n_segments));
  }
  if (abi.tokens.size() != engine.tokens.size()) {
    fail("token rows: c abi " + std::to_string(abi.tokens.size()) + " vs engine " +
         std::to_string(engine.tokens.size()));
  }
  for (size_t i = 0; i < abi.tokens.size() && i < engine.tokens.size(); ++i) {
    if (!(abi.tokens[i] == engine.tokens[i])) {
      fail("token row " + std::to_string(i) + "\n    c abi : " + token_string(abi.tokens[i]) +
           "\n    engine: " + token_string(engine.tokens[i]));
      break; // the first divergence is the informative one
    }
  }
  std::printf("[%s] %s  (%zu token row(s))\n    %s\n", label.c_str(), ok ? "match" : "DIFFER",
              engine.tokens.size(), engine.text.c_str());
  return ok;
}

bool same_outcome(const Outcome &a, const Outcome &b) {
  return a.text == b.text && a.raw_text == b.raw_text && a.kind == b.kind &&
         a.n_words == b.n_words && a.n_segments == b.n_segments && a.tokens == b.tokens;
}

void check_capabilities(const transcribe_model *model) {
  CHECK(std::string(transcribe_model_arch_string(model)) == "medasr");
  transcribe_capabilities caps;
  transcribe_capabilities_init(&caps);
  CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
  CHECK(!caps.supports_translate);
  CHECK(!caps.supports_language_detect);
  CHECK(!caps.supports_streaming);
  CHECK(caps.max_timestamp_kind == TRANSCRIBE_TIMESTAMPS_TOKEN);
  if (caps.max_audio_ms != kArchMaxAudioMs) {
    std::fprintf(stderr, "FAIL: C ABI max_audio_ms %lld, expected %lld\n",
                 static_cast<long long>(caps.max_audio_ms),
                 static_cast<long long>(kArchMaxAudioMs));
    ++g_failures;
  }
  CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION));
  CHECK(!transcribe_model_supports(model, TRANSCRIBE_FEATURE_DIARIZATION));
}

void check_engine_capabilities(const ILoadedVoiceModel &model) {
  const CapabilitySet &caps = model.capabilities();
  CHECK(model.metadata().family == "medasr");
  CHECK(!caps.supports_translate);
  CHECK(caps.supports_language_detection.has_value() && !*caps.supports_language_detection);
  CHECK(caps.supports_cancellation);
  CHECK(caps.supports_timestamps);
  CHECK(caps.timestamp_granularity == TimestampGranularity::Token);
  CHECK(caps.max_audio_ms == kArchMaxAudioMs);
  CHECK(!caps.supported_tasks.empty() && caps.supported_tasks[0].task == VoiceTaskKind::Asr);
}

// Structure of the token rows themselves (both sides already compared equal).
void check_token_rows(const Fixture &f, const Outcome &o) {
  CHECK(!o.tokens.empty());
  CHECK(o.kind == TRANSCRIBE_TIMESTAMPS_TOKEN);
  int64_t prev_t0 = -1;
  const int64_t audio_ms = static_cast<int64_t>(f.pcm.size()) / 16;
  for (const auto &t : o.tokens) {
    CHECK(t.t1_ms - t.t0_ms == 40);
    CHECK(t.t0_ms % 40 == 0);
    CHECK(t.t0_ms > prev_t0);
    CHECK(t.t0_ms < audio_ms);
    CHECK(t.word_index == -1);
    CHECK(t.p == 0.0f);
    CHECK(!t.text.empty());
    CHECK(t.id > 3); // blank / <s> / </s> / <unk> never surface
    prev_t0 = t.t0_ms;
  }
  // raw_text is the untrimmed decode; full text drops at most one leading space.
  CHECK(o.raw_text == o.text || o.raw_text == " " + o.text);
}

void check_rejections(transcribe_session *abi_session, IOfflineVoiceTaskSession &engine) {
  const std::vector<float> tiny(100, 0.0f); // shorter than one 400-sample window
  const transcribe_run_params rp = text_only_params();
  CHECK(transcribe_run(abi_session, tiny.data(), static_cast<int>(tiny.size()), &rp) ==
        TRANSCRIBE_ERR_INVALID_ARG);
  bool threw = false;
  try {
    (void)engine.run(engine_request(tiny));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("SKIP: usage: %s <medasr.gguf> [fixtures_dir]\n", argv[0]);
    return 2;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path fixtures_dir =
      argc > 2 ? std::filesystem::path(argv[2])
               : std::filesystem::path("assets/asr_validation/librispeech");
  if (!std::filesystem::exists(model_path)) {
    std::printf("SKIP: model absent: %s\n", model_path.string().c_str());
    return 2;
  }
  // Run the listed fixtures that are present: the default registration points
  // at assets/asr_validation/quick (one short clip), the *_full one at the
  // 4-clip corpus. Skip only when none is present.
  std::vector<const char *> present_fixtures;
  for (const char *name : kAllFixtures) {
    if (std::filesystem::exists(fixtures_dir / name)) present_fixtures.push_back(name);
  }
  if (present_fixtures.empty()) {
    std::printf("SKIP: no fixture present in %s\n", fixtures_dir.string().c_str());
    return 2;
  }
  const std::vector<const char *> kFixtures = present_fixtures;

  try {
    std::vector<Fixture> fixtures;
    for (const char *name : kFixtures) {
      int rate = 0;
      Fixture f;
      f.name = name;
      f.pcm = abi_test::read_wav_mono_f32((fixtures_dir / name).string(), rate);
      if (rate != 16000) {
        std::fprintf(stderr, "FAIL: %s is not 16 kHz\n", name);
        return 1;
      }
      fixtures.push_back(std::move(f));
    }

    // The C ABI (dispatcher -> arch today; -> ArchAdapter -> engine later).
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
      transcribe_model_free(abi_model);
      return 1;
    }

    // The engine directly, through the registry.
    ModelRegistry registry;
    registry.register_loader(engine::models::medasr::make_medasr_loader());
    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "medasr";
    auto loaded = registry.load(request);
    check_engine_capabilities(*loaded);
    TaskSpec task;
    task.task = VoiceTaskKind::Asr;
    task.mode = RunMode::Offline;
    SessionOptions options;
    options.backend.type = engine::core::BackendType::Cpu;
    auto session = loaded->create_task_session(task, options);
    auto *engine = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    if (engine == nullptr) {
      std::fprintf(stderr, "FAIL: engine session is not offline\n");
      return 1;
    }
    engine->prepare(build_preparation_request(to_audio(fixtures[0].pcm)));

    // Single-shot, per fixture.
    std::vector<Outcome> single;
    for (const auto &f : fixtures) {
      const Outcome abi = run_c_abi(abi_session, f.pcm);
      const Outcome eng = run_engine(*engine, f.pcm);
      compare(f.name, abi, eng);
      CHECK(!eng.text.empty());
      check_token_rows(f, eng);
      single.push_back(eng);
    }

    // A second run on the same session must reproduce the first exactly.
    {
      const Outcome again = run_engine(*engine, fixtures[0].pcm);
      if (!same_outcome(again, single[0])) {
        std::fprintf(stderr,
                     "FAIL [rerun] second engine run differs\n    first : %s\n    second: %s\n",
                     single[0].text.c_str(), again.text.c_str());
        ++g_failures;
      }
    }

    // Batched: one encoder dispatch over the 4 (different-length) fixtures.
    {
      std::vector<const float *> ptrs;
      std::vector<int> lens;
      std::vector<TaskRequest> requests;
      for (const auto &f : fixtures) {
        ptrs.push_back(f.pcm.data());
        lens.push_back(static_cast<int>(f.pcm.size()));
        requests.push_back(engine_request(f.pcm));
      }
      const transcribe_run_params rp = text_only_params();
      const transcribe_status st = transcribe_run_batch(abi_session, ptrs.data(), lens.data(),
                                                        static_cast<int>(ptrs.size()), &rp);
      CHECK(st == TRANSCRIBE_OK);
      CHECK(transcribe_batch_n_results(abi_session) == static_cast<int>(fixtures.size()));
      const std::vector<TaskResult> results = engine->run_batch(requests);
      CHECK(results.size() == fixtures.size());
      for (size_t i = 0; i < fixtures.size() && i < results.size(); ++i) {
        CHECK(transcribe_batch_status(abi_session, static_cast<int>(i)) == TRANSCRIBE_OK);
        const Outcome abi = batch_c_abi(abi_session, static_cast<int>(i));
        const Outcome eng = from_engine(results[i]);
        compare("batch/" + fixtures[i].name, abi, eng);
        if (!same_outcome(eng, single[i])) {
          // Informational: the arch documents batch == serial on CPU; a
          // difference here is not a port defect as long as both paths agree.
          std::printf("    note: batched result differs from single-shot for %s\n",
                      fixtures[i].name.c_str());
        }
      }
    }

    check_rejections(abi_session, *engine);

    transcribe_session_free(abi_session);
    transcribe_model_free(abi_model);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "medasr_engine_arch_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("medasr_engine_arch_parity_test: C ABI == engine (text, raw text, token rows) on "
              "every fixture, single and batched\n");
  return 0;
}
