// test_gigaam_engine_arch_parity.cpp - GigaAM through the public C ABI (which,
// until the family is retired, routes to src/runtime/arch/gigaam) against the
// engine `gigaam` package called directly, in one process, on the same GGUF
// and audio. The evidence the gigaam arch retirement needs (Phase 11), in the
// shape of the whisper retirement gate (whisper_engine_arch_parity_test, now
// test_whisper_c_abi_parity.cpp).
//
// Checked, per input WAV:
//   - single-shot transcript and raw (untrimmed) text: C ABI == engine,
//     byte-exact
//   - per-token rows: the arch's TOKEN rows (id, piece text, t0 / t1) == the
//     engine's TaskResult::token_timestamps
//   - re-running the engine session reproduces its first transcript (a graph
//     or buffer reused across runs would decode garbage on the second run)
// Across all inputs:
//   - offline batch (transcribe_run_batch vs run_batch): a variable-length
//     batch of every input (masked path) and a same-length pair (mask-free)
//   - capabilities: languages, translate, language detection, cancellation,
//     TOKEN timestamps, max_audio_ms
//   - rejections: audio shorter than one analysis window is INVALID_ARG on
//     the C ABI and std::invalid_argument in the engine; translate and an
//     unsupported language are std::invalid_argument
//   - cancellation: an abort callback / declining progress callback unwinds
//     both sides
//
// The model is Russian-only. The LibriSpeech fixtures (English) still make a
// strict numerical test - the transcripts are phonetic Russian and have to
// agree exactly - but register a Russian clip (samples/ru.wav) too.
//
// Usage: gigaam_engine_arch_parity_test <gigaam.gguf> <wav-or-dir> [<wav-or-dir> ...]
// A directory contributes its *.wav files in name order. Exit 2 (SKIP) when
// the model or every input is absent.

#include "abi_test_wav.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/gigaam/session.h"
#include "transcribe.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace engine::runtime;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                                \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                         \
      ++g_failures;                                                                                \
    }                                                                                              \
  } while (0)

struct TokenRow {
  int id = 0;
  std::string text;
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
  bool operator==(const TokenRow &o) const {
    return id == o.id && text == o.text && t0_ms == o.t0_ms && t1_ms == o.t1_ms;
  }
};

struct Outcome {
  std::string text;
  std::string raw_text;
  std::vector<TokenRow> tokens;
};

struct Clip {
  std::string name;
  std::vector<float> pcm; // 16 kHz mono
};

std::vector<std::filesystem::path> collect_wavs(int argc, char **argv) {
  std::vector<std::filesystem::path> out;
  for (int i = 2; i < argc; ++i) {
    const std::filesystem::path p = argv[i];
    if (std::filesystem::is_directory(p)) {
      std::vector<std::filesystem::path> dir;
      for (const auto &entry : std::filesystem::directory_iterator(p)) {
        if (entry.is_regular_file() && entry.path().extension() == ".wav") {
          dir.push_back(entry.path());
        }
      }
      std::sort(dir.begin(), dir.end());
      out.insert(out.end(), dir.begin(), dir.end());
    } else if (std::filesystem::is_regular_file(p)) {
      out.push_back(p);
    }
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

Outcome run_c_abi(transcribe_session *session, const std::vector<float> &pcm) {
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.timestamps = TRANSCRIBE_TIMESTAMPS_TOKEN; // the arch's finest rows
  const transcribe_status st =
      transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
  if (st != TRANSCRIBE_OK) {
    throw std::runtime_error(std::string("C ABI run failed: ") + transcribe_status_string(st));
  }
  Outcome out;
  const char *text = transcribe_full_text(session);
  out.text = text != nullptr ? text : "";
  const char *raw = transcribe_raw_text(session);
  out.raw_text = raw != nullptr ? raw : "";
  for (int i = 0; i < transcribe_n_tokens(session); ++i) {
    transcribe_token tok;
    transcribe_token_init(&tok);
    if (transcribe_get_token(session, i, &tok) == TRANSCRIBE_OK) {
      out.tokens.push_back({tok.id, tok.text != nullptr ? tok.text : "", tok.t0_ms, tok.t1_ms});
    }
  }
  return out;
}

TaskRequest engine_request(const std::vector<float> &pcm) {
  TaskRequest request;
  request.audio_input = to_audio(pcm);
  request.options["language"] = "";
  request.options["task"] = "transcribe";
  request.options["timestamps"] = "token";
  return request;
}

Outcome to_outcome(const TaskResult &result) {
  Outcome out;
  if (result.text_output.has_value()) {
    out.text = result.text_output->text;
  }
  out.raw_text = result.raw_text.value_or(out.text);
  for (const auto &row : result.token_timestamps) {
    out.tokens.push_back({row.id, row.text, row.span.start_sample * 1000 / 16000,
                          row.span.end_sample * 1000 / 16000});
  }
  return out;
}

Outcome run_engine(IOfflineVoiceTaskSession &session, const std::vector<float> &pcm) {
  const TaskRequest request = engine_request(pcm);
  session.prepare(build_preparation_request(request));
  return to_outcome(session.run(request));
}

bool compare(const std::string &name, const Outcome &abi, const Outcome &engine,
             bool check_tokens) {
  bool ok = true;
  if (abi.text != engine.text) {
    std::fprintf(stderr, "FAIL [%s] text\n    c abi : %s\n    engine: %s\n", name.c_str(),
                 abi.text.c_str(), engine.text.c_str());
    ok = false;
  }
  if (abi.raw_text != engine.raw_text) {
    std::fprintf(stderr, "FAIL [%s] raw text\n    c abi : '%s'\n    engine: '%s'\n", name.c_str(),
                 abi.raw_text.c_str(), engine.raw_text.c_str());
    ok = false;
  }
  if (check_tokens && !(abi.tokens == engine.tokens)) {
    std::fprintf(stderr, "FAIL [%s] token rows: c abi %zu vs engine %zu (or different text / timing)\n",
                 name.c_str(), abi.tokens.size(), engine.tokens.size());
    const size_t n = std::min(abi.tokens.size(), engine.tokens.size());
    for (size_t i = 0; i < n; ++i) {
      if (!(abi.tokens[i] == engine.tokens[i])) {
        std::fprintf(stderr, "    first difference at %zu: c abi '%s' [%lld, %lld] vs engine '%s' [%lld, %lld]\n",
                     i, abi.tokens[i].text.c_str(), static_cast<long long>(abi.tokens[i].t0_ms),
                     static_cast<long long>(abi.tokens[i].t1_ms), engine.tokens[i].text.c_str(),
                     static_cast<long long>(engine.tokens[i].t0_ms),
                     static_cast<long long>(engine.tokens[i].t1_ms));
        break;
      }
    }
    ok = false;
  }
  if (!ok) {
    ++g_failures;
  }
  std::printf("[%s] %s  (%zu token(s))\n    %s\n", name.c_str(), ok ? "match" : "DIFFER",
              engine.tokens.size(), engine.text.c_str());
  return ok;
}

// The C-ABI batch against run_batch over the same utterances.
void compare_batch(const std::string &label, transcribe_session *abi_session,
                   IOfflineVoiceTaskSession &engine, const std::vector<const Clip *> &clips) {
  std::vector<const float *> ptrs;
  std::vector<int> lens;
  std::vector<TaskRequest> requests;
  for (const Clip *clip : clips) {
    ptrs.push_back(clip->pcm.data());
    lens.push_back(static_cast<int>(clip->pcm.size()));
    requests.push_back(engine_request(clip->pcm));
  }
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.timestamps = TRANSCRIBE_TIMESTAMPS_TOKEN;
  const transcribe_status st = transcribe_run_batch(abi_session, ptrs.data(), lens.data(),
                                                    static_cast<int>(clips.size()), &rp);
  if (st != TRANSCRIBE_OK) {
    std::fprintf(stderr, "FAIL [%s] C ABI run_batch: %s\n", label.c_str(),
                 transcribe_status_string(st));
    ++g_failures;
    return;
  }
  engine.prepare(build_preparation_request(requests[0]));
  const std::vector<TaskResult> results = engine.run_batch(requests);
  CHECK(static_cast<int>(results.size()) == transcribe_batch_n_results(abi_session));
  for (size_t i = 0; i < clips.size() && i < results.size(); ++i) {
    Outcome abi;
    const char *text = transcribe_batch_full_text(abi_session, static_cast<int>(i));
    abi.text = text != nullptr ? text : "";
    const char *raw = transcribe_batch_raw_text(abi_session, static_cast<int>(i));
    abi.raw_text = raw != nullptr ? raw : "";
    for (int k = 0; k < transcribe_batch_n_tokens(abi_session, static_cast<int>(i)); ++k) {
      transcribe_token tok;
      transcribe_token_init(&tok);
      if (transcribe_batch_get_token(abi_session, static_cast<int>(i), k, &tok) == TRANSCRIBE_OK) {
        abi.tokens.push_back({tok.id, tok.text != nullptr ? tok.text : "", tok.t0_ms, tok.t1_ms});
      }
    }
    CHECK(transcribe_batch_status(abi_session, static_cast<int>(i)) == TRANSCRIBE_OK);
    compare(label + "/" + clips[i]->name, abi, to_outcome(results[i]), /*check_tokens=*/true);
  }
}

void check_capabilities(const transcribe_model *abi_model, const CapabilitySet &engine_caps) {
  std::printf("C ABI arch: %s\n", transcribe_model_arch_string(abi_model));
  CHECK(std::string(transcribe_model_arch_string(abi_model)) == "gigaam");
  transcribe_capabilities caps;
  transcribe_capabilities_init(&caps);
  CHECK(transcribe_model_get_capabilities(abi_model, &caps) == TRANSCRIBE_OK);

  std::vector<std::string> abi_languages;
  for (int i = 0; i < caps.n_languages; ++i) {
    abi_languages.emplace_back(caps.languages[i] != nullptr ? caps.languages[i] : "");
  }
  if (abi_languages != engine_caps.languages) {
    std::fprintf(stderr, "FAIL capabilities: languages differ (c abi %d, engine %zu)\n",
                 caps.n_languages, engine_caps.languages.size());
    ++g_failures;
  }
  CHECK(caps.native_sample_rate == 16000);
  CHECK(caps.supports_translate == engine_caps.supports_translate);
  CHECK(caps.supports_language_detect == engine_caps.supports_language_detection.value_or(false));
  CHECK(!caps.supports_streaming);
  CHECK(transcribe_model_supports(abi_model, TRANSCRIBE_FEATURE_CANCELLATION) ==
        engine_caps.supports_cancellation);
  // TOKEN timestamps and the 25 s advisory window, which the adapter maps to
  // max_timestamp_kind / max_audio_ms once the C ABI routes to the engine.
  CHECK(caps.max_timestamp_kind == TRANSCRIBE_TIMESTAMPS_TOKEN);
  CHECK(engine_caps.supports_timestamps &&
        engine_caps.timestamp_granularity == TimestampGranularity::Token);
  CHECK(caps.max_audio_ms == engine_caps.max_audio_ms);
}

void check_rejections(transcribe_session *abi_session, IOfflineVoiceTaskSession &engine,
                      const std::vector<float> &good) {
  // Shorter than one 320-sample analysis window: no mel frame.
  const std::vector<float> tiny(100, 0.01f);
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  CHECK(transcribe_run(abi_session, tiny.data(), static_cast<int>(tiny.size()), &rp) ==
        TRANSCRIBE_ERR_INVALID_ARG);

  const auto expect_invalid = [&](TaskRequest request, const char *what) {
    try {
      engine.prepare(build_preparation_request(request));
      (void)engine.run(request);
      std::fprintf(stderr, "FAIL engine accepted %s\n", what);
      ++g_failures;
    } catch (const std::invalid_argument &) {
      // expected
    } catch (const std::exception &e) {
      std::fprintf(stderr, "FAIL engine rejected %s with the wrong exception: %s\n", what,
                   e.what());
      ++g_failures;
    }
  };
  expect_invalid(engine_request(tiny), "sub-window audio");
  {
    TaskRequest request = engine_request(good);
    request.options["task"] = "translate";
    expect_invalid(request, "task=translate");
  }
  {
    TaskRequest request = engine_request(good);
    request.options["language"] = "en";
    expect_invalid(request, "language=en");
  }
  {
    TaskRequest request = engine_request(good);
    request.options["timestamps"] = "phoneme";
    expect_invalid(request, "timestamps=phoneme");
  }
}

bool abort_now(void *) { return true; }

void check_cancellation(transcribe_session *abi_session, IOfflineVoiceTaskSession &engine,
                        const std::vector<float> &pcm) {
  transcribe_set_abort_callback(abi_session, &abort_now, nullptr);
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  CHECK(transcribe_run(abi_session, pcm.data(), static_cast<int>(pcm.size()), &rp) ==
        TRANSCRIBE_ERR_ABORTED);
  transcribe_set_abort_callback(abi_session, nullptr, nullptr);

  engine.set_progress_callback([](const ProgressInfo &) { return false; });
  bool canceled = false;
  try {
    (void)run_engine(engine, pcm);
  } catch (const ProgressCanceled &) {
    canceled = true;
  }
  CHECK(canceled);
  engine.set_progress_callback(nullptr);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <gigaam.gguf> <wav-or-dir> [<wav-or-dir> ...]\n", argv[0]);
    return 1;
  }
  const std::filesystem::path model_path = argv[1];
  if (!std::filesystem::exists(model_path)) {
    std::printf("SKIP: model absent: %s\n", model_path.string().c_str());
    return 2;
  }
  const auto wavs = collect_wavs(argc, argv);
  if (wavs.empty()) {
    std::printf("SKIP: no input WAV found\n");
    return 2;
  }

  try {
    std::vector<Clip> clips;
    for (const auto &path : wavs) {
      int rate = 0;
      Clip clip;
      clip.name = path.stem().string();
      clip.pcm = abi_test::read_wav_mono_f32(path.string(), rate);
      if (rate != 16000) {
        std::fprintf(stderr, "FAIL: %s is not 16 kHz\n", path.string().c_str());
        ++g_failures;
        continue;
      }
      clips.push_back(std::move(clip));
    }
    if (clips.empty()) {
      std::fprintf(stderr, "FAIL: no usable 16 kHz input\n");
      return 1;
    }

    // The C ABI (dispatcher -> the gigaam arch, until it is retired).
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
    transcribe_session *abi_session = nullptr;
    if (transcribe_session_init(abi_model, &sp, &abi_session) != TRANSCRIBE_OK) {
      std::fprintf(stderr, "FAIL: C ABI session init\n");
      transcribe_model_free(abi_model);
      return 1;
    }

    // The engine directly, through the registry.
    ModelRegistry registry;
    registry.register_loader(engine::models::gigaam::make_gigaam_loader());
    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "gigaam";
    auto loaded = registry.load(request);
    std::printf("engine variant: %s\n", loaded->metadata().variant.c_str());
    TaskSpec task;
    task.task = VoiceTaskKind::Asr;
    task.mode = RunMode::Offline;
    SessionOptions options;
    options.backend.threads = 4;
    auto session = loaded->create_task_session(task, options);
    auto *engine = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    if (engine == nullptr) {
      std::fprintf(stderr, "FAIL: engine session is not offline\n");
      return 1;
    }

    check_capabilities(abi_model, loaded->capabilities());

    // Single-shot, every clip.
    std::string first_engine_text;
    for (const auto &clip : clips) {
      const Outcome abi = run_c_abi(abi_session, clip.pcm);
      const Outcome eng = run_engine(*engine, clip.pcm);
      compare(clip.name, abi, eng, /*check_tokens=*/true);
      if (first_engine_text.empty()) {
        first_engine_text = eng.text;
      }
    }

    // Re-run: the session's second run of the first clip must reproduce it.
    {
      const Outcome again = run_engine(*engine, clips.front().pcm);
      if (again.text != first_engine_text) {
        std::fprintf(stderr, "FAIL [rerun] engine second run differs\n    first : %s\n    second: %s\n",
                     first_engine_text.c_str(), again.text.c_str());
        ++g_failures;
      }
    }

    // Offline batches: every clip (variable length -> masked encoder), and a
    // same-length pair (mask-free).
    {
      std::vector<const Clip *> all;
      for (const auto &clip : clips) {
        all.push_back(&clip);
      }
      compare_batch("batch-all", abi_session, *engine, all);
      compare_batch("batch-pair", abi_session, *engine, {&clips.front(), &clips.front()});
    }

    check_rejections(abi_session, *engine, clips.front().pcm);
    check_cancellation(abi_session, *engine, clips.front().pcm);

    transcribe_session_free(abi_session);
    transcribe_model_free(abi_model);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "gigaam_engine_arch_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("gigaam_engine_arch_parity_test: C ABI (arch) == engine on every input\n");
  return 0;
}
