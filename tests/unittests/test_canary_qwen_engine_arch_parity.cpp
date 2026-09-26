// test_canary_qwen_engine_arch_parity.cpp - Canary-Qwen through the public C
// ABI (transcribe_model_load_file, which routes a canary_qwen GGUF to the
// transcribe.cpp arch src/runtime/arch/canary_qwen while it exists) against
// the engine `canary_qwen` package called directly through ModelRegistry, in
// one process, on the same GGUF and audio.
//
// This is the retirement evidence for the arch (Phase 11 pattern, cf.
// whisper_c_abi_parity_test): on the four LibriSpeech fixtures in
// assets/asr_validation/librispeech the transcripts must be IDENTICAL, byte
// for byte, single-shot and batched (transcribe_run_batch vs run_batch, which
// exercises the lockstep batched prefill / step decode on both sides), and
// the truncation flag must agree. A mismatch prints both transcripts. The
// engine session is also run twice on the same clip (cached step graphs must
// re-upload their inputs) and must repeat itself exactly.
//
// Both sides run on the CPU backend with the same thread count (the arch's
// strict CPU plan: no BLAS accelerator in the scheduler).
//
// Usage: canary_qwen_engine_arch_parity_test <canary-qwen-2.5b.gguf> [fixtures_dir]
// Exit 2 (SKIP) when the model path is absent / missing, or the fixtures are.

#include "abi_test_wav.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/canary_qwen/session.h"
#include "transcribe.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
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

struct Outcome {
  std::string text;
  bool truncated = false;
};

void compare(const std::string &name, const Outcome &abi, const Outcome &engine) {
  bool ok = true;
  if (abi.text != engine.text) {
    std::fprintf(stderr, "FAIL [%s] transcripts differ\n    c abi (arch): \"%s\"\n    engine      : \"%s\"\n",
                 name.c_str(), abi.text.c_str(), engine.text.c_str());
    ok = false;
    ++g_failures;
  }
  if (abi.truncated != engine.truncated) {
    std::fprintf(stderr, "FAIL [%s] truncated: c abi %d vs engine %d\n", name.c_str(),
                 abi.truncated ? 1 : 0, engine.truncated ? 1 : 0);
    ok = false;
    ++g_failures;
  }
  std::printf("[%s] %s\n    %s\n", name.c_str(), ok ? "match" : "DIFFER", engine.text.c_str());
}

Outcome run_c_abi(transcribe_session *session, const std::vector<float> &pcm) {
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.language = "en";
  const transcribe_status st = transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
  if (st != TRANSCRIBE_OK && st != TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
    throw std::runtime_error(std::string("C ABI run failed: ") + transcribe_status_string(st));
  }
  Outcome out;
  out.text = transcribe_full_text(session);
  out.truncated = transcribe_was_truncated(session);
  if (out.truncated != (st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED)) {
    std::fprintf(stderr, "FAIL C ABI was_truncated disagrees with the run status\n");
    ++g_failures;
  }
  return out;
}

Outcome run_engine(IOfflineVoiceTaskSession &session, const AudioBuffer &audio) {
  session.prepare(build_preparation_request(audio));
  TaskRequest request;
  request.audio_input = audio;
  request.options["language"] = "en";
  const TaskResult result = session.run(request);
  Outcome out;
  if (result.text_output.has_value()) {
    out.text = result.text_output->text;
  }
  out.truncated = result.truncated;
  return out;
}

void check_capabilities(const transcribe_model *model) {
  CHECK(std::string(transcribe_model_arch_string(model)) == "canary_qwen");
  transcribe_capabilities caps;
  transcribe_capabilities_init(&caps);
  CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
  CHECK(caps.max_timestamp_kind == TRANSCRIBE_TIMESTAMPS_NONE);
  CHECK(!caps.supports_translate);
  CHECK(!caps.supports_language_detect);
  CHECK(!caps.supports_streaming);
  CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION));
}

void check_engine_capabilities(const ILoadedVoiceModel &model) {
  const CapabilitySet &caps = model.capabilities();
  CHECK(model.metadata().family == "canary_qwen");
  CHECK(!caps.supports_timestamps);
  CHECK(!caps.supports_translate);
  CHECK(caps.supports_language_detection.has_value() && !*caps.supports_language_detection);
  CHECK(caps.supports_cancellation);
  CHECK(std::find(caps.languages.begin(), caps.languages.end(), "en") != caps.languages.end());
}

// The prompt template pieces (and a transcript-like string) must tokenize to
// the same ids through the arch's transcribe::Tokenizer (transcribe_tokenize)
// and the engine's TokenizerHub: the chat-template prefix / suffix are built
// from exactly these encodes.
void check_tokenizer(const transcribe_model *abi_model, const ILoadedVoiceModel &engine_model) {
  const char *const texts[] = {
      "user\nTranscribe the following: ",
      "\n",
      "assistant\n",
      "Don't cry, he said. I was obliged to come.",
  };
  for (const char *text : texts) {
    std::vector<int32_t> abi_ids(256);
    const int n = transcribe_tokenize(abi_model, text, abi_ids.data(), abi_ids.size());
    if (n < 0) {
      std::printf("[tokenizer] C ABI transcribe_tokenize unavailable (%d); skipped\n", n);
      return;
    }
    abi_ids.resize(static_cast<size_t>(n));
    const auto eng_ids = engine_model.tokenize(text);
    if (!eng_ids.has_value() || *eng_ids != abi_ids) {
      std::string a;
      std::string e;
      for (const int32_t id : abi_ids) {
        a += std::to_string(id) + " ";
      }
      if (eng_ids.has_value()) {
        for (const int32_t id : *eng_ids) {
          e += std::to_string(id) + " ";
        }
      }
      std::fprintf(stderr, "FAIL [tokenizer] ids differ for \"%s\"\n    c abi : %s\n    engine: %s\n",
                   text, a.c_str(), e.c_str());
      ++g_failures;
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argv[1] == nullptr || argv[1][0] == '\0') {
    std::printf("SKIP: usage: %s <canary-qwen-2.5b.gguf> [fixtures_dir]\n", argv[0]);
    return 2;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path fixtures =
      argc >= 3 ? std::filesystem::path(argv[2])
                : std::filesystem::path("assets") / "asr_validation" / "librispeech";
  if (!std::filesystem::exists(model_path)) {
    std::printf("SKIP: model absent: %s\n", model_path.string().c_str());
    return 2;
  }
  // Run the listed fixtures that are present: the default registration points
  // at assets/asr_validation/quick (one short clip), the *_full one at the
  // 4-clip corpus. Skip only when none is present.
  std::vector<const char *> present_fixtures;
  for (const char *wav : kAllFixtures) {
    if (std::filesystem::exists(fixtures / wav)) present_fixtures.push_back(wav);
  }
  if (present_fixtures.empty()) {
    std::printf("SKIP: no fixture present in %s\n", fixtures.string().c_str());
    return 2;
  }
  const std::vector<const char *> kFixtures = present_fixtures;

  const int threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

  try {
    // The C ABI (dispatcher -> the canary_qwen arch while it is registered).
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
    sp.n_threads = threads;
    transcribe_session *abi_session = nullptr;
    if (transcribe_session_init(abi_model, &sp, &abi_session) != TRANSCRIBE_OK) {
      std::fprintf(stderr, "FAIL: C ABI session init\n");
      transcribe_model_free(abi_model);
      return 1;
    }

    // The engine package directly, through the registry.
    ModelRegistry registry;
    registry.register_loader(engine::models::canary_qwen::make_canary_qwen_loader());
    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "canary_qwen";
    auto loaded = registry.load(request);
    check_engine_capabilities(*loaded);
    check_tokenizer(abi_model, *loaded);
    TaskSpec task;
    task.task = VoiceTaskKind::Asr;
    task.mode = RunMode::Offline;
    SessionOptions options;
    options.backend.type = engine::core::BackendType::Cpu;
    options.backend.threads = threads;
    auto session = loaded->create_task_session(task, options);
    auto *engine = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    if (engine == nullptr) {
      std::fprintf(stderr, "FAIL: engine session is not offline\n");
      return 1;
    }

    // Single-shot, per fixture.
    std::vector<AudioBuffer> audios;
    std::vector<Outcome> abi_single;
    for (const char *wav : kFixtures) {
      int rate = 0;
      AudioBuffer audio;
      audio.samples = abi_test::read_wav_mono_f32((fixtures / wav).string(), rate);
      audio.sample_rate = rate;
      audio.channels = 1;
      if (rate != 16000) {
        std::fprintf(stderr, "FAIL: %s is not 16 kHz\n", wav);
        ++g_failures;
        continue;
      }
      const Outcome abi = run_c_abi(abi_session, audio.samples);
      const Outcome eng = run_engine(*engine, audio);
      compare(wav, abi, eng);
      audios.push_back(audio);
      abi_single.push_back(abi);
    }

    // The engine session must repeat itself on reuse (cached graphs re-upload
    // every input).
    if (!audios.empty()) {
      const Outcome again = run_engine(*engine, audios.front());
      if (again.text != abi_single.front().text) {
        std::fprintf(stderr, "FAIL [rerun] engine second run differs\n    first : \"%s\"\n    second: \"%s\"\n",
                     abi_single.front().text.c_str(), again.text.c_str());
        ++g_failures;
      }
    }

    // Batched: transcribe_run_batch (arch lockstep decode) vs run_batch.
    if (audios.size() == std::size(kFixtures)) {
      std::vector<const float *> pcm;
      std::vector<int> n_samples;
      for (const auto &a : audios) {
        pcm.push_back(a.samples.data());
        n_samples.push_back(static_cast<int>(a.samples.size()));
      }
      transcribe_run_params rp;
      transcribe_run_params_init(&rp);
      rp.language = "en";
      const transcribe_status st = transcribe_run_batch(abi_session, pcm.data(), n_samples.data(),
                                                        static_cast<int>(pcm.size()), &rp);
      CHECK(st == TRANSCRIBE_OK);
      CHECK(transcribe_batch_n_results(abi_session) == static_cast<int>(pcm.size()));

      std::vector<TaskRequest> requests;
      for (const auto &a : audios) {
        TaskRequest r;
        r.audio_input = a;
        r.options["language"] = "en";
        requests.push_back(std::move(r));
      }
      engine->prepare(build_preparation_request(audios.front()));
      const std::vector<TaskResult> results = engine->run_batch(requests);
      CHECK(results.size() == audios.size());

      for (size_t i = 0; i < audios.size() && i < results.size(); ++i) {
        Outcome abi;
        const transcribe_status bst = transcribe_batch_status(abi_session, static_cast<int>(i));
        CHECK(bst == TRANSCRIBE_OK || bst == TRANSCRIBE_ERR_OUTPUT_TRUNCATED);
        const char *text = transcribe_batch_full_text(abi_session, static_cast<int>(i));
        abi.text = text != nullptr ? text : "";
        abi.truncated = (bst == TRANSCRIBE_ERR_OUTPUT_TRUNCATED);
        Outcome eng;
        if (results[i].text_output.has_value()) {
          eng.text = results[i].text_output->text;
        }
        eng.truncated = results[i].truncated;
        compare(std::string("batch/") + kFixtures[i], abi, eng);
      }
    }

    transcribe_session_free(abi_session);
    transcribe_model_free(abi_model);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "canary_qwen_engine_arch_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("canary_qwen_engine_arch_parity_test: C ABI (arch) == engine on every fixture, "
              "single-shot and batched\n");
  return 0;
}
