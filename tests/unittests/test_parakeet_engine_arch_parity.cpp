// test_parakeet_engine_arch_parity.cpp - Parakeet through the public C ABI
// against the engine `parakeet_tdt` package called directly, in one process,
// on the same transcribe.cpp GGUF and audio.
//
// Today the C ABI routes a general.architecture=parakeet GGUF to
// src/runtime/arch/parakeet, so this is the arch-retirement evidence: the
// engine must reproduce the arch's result on every LibriSpeech fixture -
// transcript, raw text, and the TOKEN rows (ids, 80 ms-grid t0/t1, word
// grouping) the arch publishes. Once the arch is retired and the C ABI reaches
// the engine through the ArchAdapter, the same comparison guards the adapter.
//
// It also pins the contract both paths share:
//   - capabilities: no translation, per-file language list / detection,
//     offline (streaming only where the GGUF says so), cancellation, TOKEN
//     timestamps
//   - a second engine run on the same session reproduces the first exactly
//     (every cached graph re-uploads its inputs)
//   - timestamps=none/segment/word elide the finer rows as the arch does
//   - a progress callback that returns false cancels the engine run
//   - audio shorter than one mel hop is ERR_INVALID_ARG through the C ABI and
//     std::invalid_argument from the engine
//
// Usage: parakeet_engine_arch_parity_test <parakeet.gguf> [fixtures_dir]
//   Pinned models (scripts/fetch_asr_test_model.py): parakeet-tdt_ctc-110m-
//   Q8_0.gguf (TDT head, use_bias, 80 mels, 1-layer predictor) and
//   parakeet-unified-en-0.6b-Q4_K_M.gguf (RNN-T head, xscaling, Q4_K).
// Exit 2 (SKIP) while the model or the fixtures are absent.

#include "abi_test_wav.h"
#include "engine/community_models/parakeet_tdt/session.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "transcribe.h"

#include <algorithm>
#include <cmath>
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

struct TokenRow {
  int id = 0;
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
  int word_index = -1;
  float p = 0.0f;
};

struct WordRow {
  std::string text;
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
};

struct Snapshot {
  std::string text;
  std::string raw_text;
  std::vector<TokenRow> tokens;
  std::vector<WordRow> words;
  int n_segments = 0;
  bool truncated = false;
};

int64_t samples_to_ms(int64_t samples) { return samples / 16; } // 16 kHz

transcribe_run_params run_params(transcribe_timestamp_kind kind) {
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.timestamps = kind;
  return rp;
}

Snapshot run_c_abi(transcribe_session *session, const std::vector<float> &pcm,
                   transcribe_timestamp_kind kind) {
  const transcribe_run_params rp = run_params(kind);
  const transcribe_status st =
      transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
  if (st != TRANSCRIBE_OK) {
    throw std::runtime_error(std::string("C ABI run failed: ") + transcribe_status_string(st));
  }
  Snapshot out;
  const char *text = transcribe_full_text(session);
  out.text = text != nullptr ? text : "";
  const char *raw = transcribe_raw_text(session);
  out.raw_text = raw != nullptr ? raw : "";
  out.truncated = transcribe_was_truncated(session);
  out.n_segments = transcribe_n_segments(session);
  for (int i = 0; i < transcribe_n_tokens(session); ++i) {
    transcribe_token t;
    transcribe_token_init(&t);
    CHECK(transcribe_get_token(session, i, &t) == TRANSCRIBE_OK);
    out.tokens.push_back({t.id, t.t0_ms, t.t1_ms, t.word_index, t.p});
  }
  for (int i = 0; i < transcribe_n_words(session); ++i) {
    transcribe_word w;
    transcribe_word_init(&w);
    CHECK(transcribe_get_word(session, i, &w) == TRANSCRIBE_OK);
    out.words.push_back({w.text != nullptr ? w.text : "", w.t0_ms, w.t1_ms});
  }
  return out;
}

AudioBuffer to_audio(const std::vector<float> &pcm) {
  AudioBuffer audio;
  audio.sample_rate = 16000;
  audio.channels = 1;
  audio.samples = pcm;
  return audio;
}

TaskRequest engine_request(const std::vector<float> &pcm, const char *timestamps) {
  TaskRequest request;
  request.audio_input = to_audio(pcm);
  request.options["timestamps"] = timestamps;
  return request;
}

Snapshot run_engine(IOfflineVoiceTaskSession &session, const std::vector<float> &pcm,
                    const char *timestamps) {
  const TaskResult result = session.run(engine_request(pcm, timestamps));
  Snapshot out;
  out.text = result.text_output.has_value() ? result.text_output->text : "";
  out.raw_text = result.raw_text.value_or(out.text);
  out.truncated = result.truncated;
  out.n_segments = static_cast<int>(result.speech_segments.size());
  for (const auto &t : result.token_timestamps) {
    out.tokens.push_back({t.id, samples_to_ms(t.span.start_sample), samples_to_ms(t.span.end_sample),
                          t.word_index, t.probability});
  }
  for (const auto &w : result.word_timestamps) {
    out.words.push_back({w.word, samples_to_ms(w.span.start_sample), samples_to_ms(w.span.end_sample)});
  }
  return out;
}

void compare(const std::string &label, const Snapshot &abi, const Snapshot &eng) {
  bool ok = true;
  if (abi.text != eng.text) {
    std::fprintf(stderr, "FAIL [%s] transcripts differ\n    c abi : %s\n    engine: %s\n",
                 label.c_str(), abi.text.c_str(), eng.text.c_str());
    ok = false;
  }
  if (abi.raw_text != eng.raw_text) {
    std::fprintf(stderr, "FAIL [%s] raw text differs\n    c abi : %s\n    engine: %s\n",
                 label.c_str(), abi.raw_text.c_str(), eng.raw_text.c_str());
    ok = false;
  }
  if (abi.truncated != eng.truncated) {
    std::fprintf(stderr, "FAIL [%s] truncation differs (c abi %d, engine %d)\n", label.c_str(),
                 abi.truncated ? 1 : 0, eng.truncated ? 1 : 0);
    ok = false;
  }
  if (abi.tokens.size() != eng.tokens.size()) {
    std::fprintf(stderr, "FAIL [%s] token rows %zu vs %zu\n", label.c_str(), abi.tokens.size(),
                 eng.tokens.size());
    ok = false;
  } else {
    double max_dp = 0.0;
    for (size_t i = 0; i < abi.tokens.size(); ++i) {
      const auto &a = abi.tokens[i];
      const auto &e = eng.tokens[i];
      if (a.id != e.id || a.t0_ms != e.t0_ms || a.t1_ms != e.t1_ms || a.word_index != e.word_index) {
        std::fprintf(stderr,
                     "FAIL [%s] token %zu: c abi {id %d, %lld-%lld ms, word %d} engine {id %d, "
                     "%lld-%lld ms, word %d}\n",
                     label.c_str(), i, a.id, static_cast<long long>(a.t0_ms),
                     static_cast<long long>(a.t1_ms), a.word_index, e.id,
                     static_cast<long long>(e.t0_ms), static_cast<long long>(e.t1_ms), e.word_index);
        ok = false;
        break;
      }
      if (std::isfinite(a.p)) {
        max_dp = std::max(max_dp, static_cast<double>(std::fabs(a.p - e.p)));
      }
    }
    // Confidence is a float softmax of logits that differ in the last bits
    // between the two implementations; informational only.
    std::printf("    [%s] max |p_abi - p_engine| = %.3g over %zu tokens\n", label.c_str(), max_dp,
                abi.tokens.size());
  }
  if (abi.words.size() != eng.words.size()) {
    std::fprintf(stderr, "FAIL [%s] word rows %zu vs %zu\n", label.c_str(), abi.words.size(),
                 eng.words.size());
    ok = false;
  } else {
    for (size_t i = 0; i < abi.words.size(); ++i) {
      const auto &a = abi.words[i];
      const auto &e = eng.words[i];
      if (a.text != e.text || a.t0_ms != e.t0_ms || a.t1_ms != e.t1_ms) {
        std::fprintf(stderr, "FAIL [%s] word %zu: c abi '%s' %lld-%lld, engine '%s' %lld-%lld\n",
                     label.c_str(), i, a.text.c_str(), static_cast<long long>(a.t0_ms),
                     static_cast<long long>(a.t1_ms), e.text.c_str(),
                     static_cast<long long>(e.t0_ms), static_cast<long long>(e.t1_ms));
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    std::printf("[%s] match (%zu tokens, %zu words)\n    %s\n", label.c_str(), eng.tokens.size(),
                eng.words.size(), eng.text.c_str());
  } else {
    ++g_failures;
  }
}

void check_capabilities(const transcribe_model *abi_model, const ILoadedVoiceModel &engine) {
  transcribe_capabilities caps;
  transcribe_capabilities_init(&caps);
  CHECK(transcribe_model_get_capabilities(abi_model, &caps) == TRANSCRIBE_OK);
  const CapabilitySet &ec = engine.capabilities();
  CHECK(engine.metadata().family == "parakeet_tdt");
  CHECK(!caps.supports_translate);
  CHECK(!ec.supports_translate);
  CHECK(ec.supports_cancellation);
  CHECK(transcribe_model_supports(abi_model, TRANSCRIBE_FEATURE_CANCELLATION));
  CHECK(caps.max_timestamp_kind == TRANSCRIBE_TIMESTAMPS_TOKEN);
  CHECK(ec.supports_timestamps && ec.timestamp_granularity == TimestampGranularity::Token);
  CHECK(ec.supports_language_detection.has_value() &&
        *ec.supports_language_detection == caps.supports_language_detect);
  CHECK(static_cast<int>(ec.languages.size()) == caps.n_languages);
  bool engine_streaming = false;
  for (const auto &task : ec.supported_tasks) {
    for (const auto mode : task.modes) {
      engine_streaming = engine_streaming || mode == RunMode::Streaming;
    }
  }
  CHECK(engine_streaming == caps.supports_streaming);
  CHECK(ec.max_audio_ms == caps.max_audio_ms);
}

void check_elision(IOfflineVoiceTaskSession &engine, const Fixture &f) {
  const TaskResult none = engine.run(engine_request(f.pcm, "none"));
  CHECK(none.speech_segments.empty() && none.word_timestamps.empty() && none.token_timestamps.empty());
  const TaskResult segment = engine.run(engine_request(f.pcm, "segment"));
  CHECK(segment.speech_segments.size() == 1 && segment.word_timestamps.empty() &&
        segment.token_timestamps.empty());
  const TaskResult word = engine.run(engine_request(f.pcm, "word"));
  CHECK(!word.word_timestamps.empty() && word.token_timestamps.empty());
  const TaskResult word_abi = engine.run(engine_request(f.pcm, "3"));  // C ABI spelling of WORD
  CHECK(!word_abi.word_timestamps.empty() && word_abi.token_timestamps.empty());
}

void check_cancellation(IOfflineVoiceTaskSession &engine, const Fixture &f) {
  engine.set_progress_callback([](const ProgressInfo &) { return false; });
  bool canceled = false;
  try {
    (void)engine.run(engine_request(f.pcm, "none"));
  } catch (const ProgressCanceled &) {
    canceled = true;
  }
  CHECK(canceled);
  engine.set_progress_callback(nullptr);
}

void check_rejections(transcribe_session *abi_session, IOfflineVoiceTaskSession &engine) {
  const std::vector<float> tiny(100, 0.0f); // shorter than one 160-sample hop
  const transcribe_run_params rp = run_params(TRANSCRIBE_TIMESTAMPS_NONE);
  CHECK(transcribe_run(abi_session, tiny.data(), static_cast<int>(tiny.size()), &rp) ==
        TRANSCRIBE_ERR_INVALID_ARG);
  bool threw = false;
  try {
    (void)engine.run(engine_request(tiny, "none"));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("SKIP: usage: %s <parakeet.gguf> [fixtures_dir]\n", argv[0]);
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
    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    sp.kv_type = TRANSCRIBE_KV_TYPE_F32; // the arch's F32 attention, closest to the engine's
    transcribe_session *abi_session = nullptr;
    if (transcribe_session_init(abi_model, &sp, &abi_session) != TRANSCRIBE_OK) {
      std::fprintf(stderr, "FAIL: C ABI session init\n");
      transcribe_model_free(abi_model);
      return 1;
    }

    // The engine directly, through the registry.
    ModelRegistry registry;
    registry.register_loader(engine::community_models::parakeet_tdt::make_parakeet_tdt_loader());
    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "parakeet_tdt";
    auto loaded = registry.load(request);
    check_capabilities(abi_model, *loaded);
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

    std::vector<Snapshot> single;
    for (const auto &f : fixtures) {
      const Snapshot abi = run_c_abi(abi_session, f.pcm, TRANSCRIBE_TIMESTAMPS_TOKEN);
      const Snapshot eng = run_engine(*engine, f.pcm, "token");
      compare(f.name, abi, eng);
      CHECK(!eng.text.empty());
      CHECK(!eng.truncated);
      CHECK(eng.n_segments == 1);
      single.push_back(eng);
    }

    // A second run on the same session must reproduce the first exactly.
    {
      const Snapshot again = run_engine(*engine, fixtures[0].pcm, "token");
      if (again.text != single[0].text || again.tokens.size() != single[0].tokens.size()) {
        std::fprintf(stderr, "FAIL [rerun] second engine run differs\n    first : %s\n    second: %s\n",
                     single[0].text.c_str(), again.text.c_str());
        ++g_failures;
      }
    }

    check_elision(*engine, fixtures[0]);
    check_cancellation(*engine, fixtures[1]);
    check_rejections(abi_session, *engine);

    transcribe_session_free(abi_session);
    transcribe_model_free(abi_model);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "parakeet_engine_arch_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("parakeet_engine_arch_parity_test: C ABI == engine on every fixture (text, raw text, "
              "token and word rows)\n");
  return 0;
}
