// test_voxtral_engine_arch_parity.cpp - Voxtral (2507, offline) through the
// public C ABI against the engine `voxtral` package called directly, in one
// process, on the same GGUF and audio.
//
// Today the C ABI routes a voxtral GGUF to src/runtime/arch/voxtral (the
// builtin arch table is consulted before the adapter), so this is the
// arch-retirement evidence: the engine port must produce the SAME transcript
// as the arch - plain transcription, with a language hint, and translation.
// Once the arch is retired and the C ABI reaches the engine through the
// ArchAdapter, the same comparison guards the adapter instead.
//
// It also pins the contract both paths share:
//   - capabilities: translation advertised, no streaming, no timestamps,
//     TRANSCRIBE_FEATURE_CANCELLATION
//   - transcribe_tokenize() == ILoadedVoiceModel::tokenize() (the arch's
//     Qwen2-fallback pretokenizer + merge-rank BPE), incl. non-ASCII text
//   - a second engine run on the same session reproduces the first exactly
//     (every cached-graph input is re-uploaded; the KV cache is re-allocated)
//   - the truncation flag agrees (neither side truncates these fixtures)
//
// Every C ABI run uses a FRESH session: the arch's run() frees its KV cache at
// the end (cleanup_gpu) without resetting its size, so a second run() on one
// arch session that needs no KV growth fails to build its prefill graph. The
// engine has no such limit and is exercised on one session throughout.
//
// Usage: voxtral_engine_arch_parity_test <voxtral.gguf> [fixtures_dir]
//   fixtures_dir defaults to assets/asr_validation/librispeech (the test runs
//   from the source tree). Exit 2 (SKIP) while the model or fixtures are absent.

#include "abi_test_wav.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/voxtral/model.h"
#include "transcribe.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace engine::runtime;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

// Two LibriSpeech utterances (clean + other) keep a 3B CPU decode affordable.
const char *const kAllFixtures[] = {
    "librispeech_test_clean_6930-75918-0000.wav",
    "librispeech_test_other_7902-96591-0000.wav",
};

struct Fixture {
  std::string name;
  std::vector<float> pcm;  // 16 kHz mono
};

struct Mode {
  const char *label;
  const char *language;         // nullptr = no hint
  bool translate;
  const char *target_language;  // translate only
};

struct AbiOutput {
  std::string text;
  bool truncated = false;
};

AbiOutput run_c_abi(transcribe_model *model, const std::vector<float> &pcm, const Mode &mode) {
  transcribe_session_params sp;
  transcribe_session_params_init(&sp);
  transcribe_session *session = nullptr;
  if (transcribe_session_init(model, &sp, &session) != TRANSCRIBE_OK || session == nullptr) {
    throw std::runtime_error("C ABI session init failed");
  }
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.language = mode.language;
  rp.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
  if (mode.translate) {
    rp.task = TRANSCRIBE_TASK_TRANSLATE;
    rp.target_language = mode.target_language;
  }
  const transcribe_status st = transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
  AbiOutput out;
  if (st != TRANSCRIBE_OK && st != TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
    transcribe_session_free(session);
    throw std::runtime_error(std::string("C ABI run failed: ") + transcribe_status_string(st));
  }
  out.truncated = transcribe_was_truncated(session);
  const char *text = transcribe_full_text(session);
  out.text = text != nullptr ? text : "";
  transcribe_session_free(session);
  return out;
}

AudioBuffer to_audio(const std::vector<float> &pcm) {
  AudioBuffer audio;
  audio.sample_rate = 16000;
  audio.channels = 1;
  audio.samples = pcm;
  return audio;
}

TaskRequest engine_request(const std::vector<float> &pcm, const Mode &mode) {
  TaskRequest request;
  request.audio_input = to_audio(pcm);
  request.options["timestamps"] = "none";
  if (mode.language != nullptr) {
    request.options["language"] = mode.language;
  }
  if (mode.translate) {
    request.options["task"] = "translate";
    request.options["target_language"] = mode.target_language;
  }
  return request;
}

TaskResult run_engine(IOfflineVoiceTaskSession &session, const std::vector<float> &pcm, const Mode &mode) {
  return session.run(engine_request(pcm, mode));
}

std::string text_of(const TaskResult &r) { return r.text_output.has_value() ? r.text_output->text : ""; }

bool compare(const std::string &label, const std::string &abi, const std::string &engine) {
  if (abi == engine) {
    std::printf("[%s] match\n    %s\n", label.c_str(), engine.c_str());
    return true;
  }
  std::fprintf(stderr, "FAIL [%s] transcripts differ\n    c abi : %s\n    engine: %s\n", label.c_str(),
               abi.c_str(), engine.c_str());
  ++g_failures;
  return false;
}

void check_capabilities(const transcribe_model *model) {
  CHECK(std::string(transcribe_model_arch_string(model)) == "voxtral");
  transcribe_capabilities caps;
  transcribe_capabilities_init(&caps);
  CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
  CHECK(caps.supports_translate);
  CHECK(!caps.supports_streaming);
  CHECK(caps.max_timestamp_kind == TRANSCRIBE_TIMESTAMPS_NONE);
  CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION));
}

void check_engine_capabilities(const ILoadedVoiceModel &model) {
  const CapabilitySet &caps = model.capabilities();
  CHECK(model.metadata().family == "voxtral");
  CHECK(caps.supports_translate);
  CHECK(caps.supports_cancellation);
  CHECK(!caps.supports_timestamps);
  CHECK(!caps.supported_tasks.empty() && caps.supported_tasks[0].task == VoiceTaskKind::Asr);
}

void check_tokenize(const transcribe_model *abi_model, const ILoadedVoiceModel &engine_model) {
  const char *const texts[] = {
      "lang:en",
      "lang:fr",
      "Translate this to French.",
      "Gr\xC3\xB6\xC3\x9F" "e \xC3\xBC" "ber 3 H\xC3\xA4user \xE2\x80\x94 \xC3\xA7" "a va? 12345 don't",
  };
  for (const char *text : texts) {
    std::vector<int32_t> abi_ids(256);
    const int n = transcribe_tokenize(abi_model, text, abi_ids.data(), abi_ids.size());
    CHECK(n >= 0 && n != INT_MIN);
    if (n < 0) {
      continue;
    }
    abi_ids.resize(static_cast<size_t>(n));
    const auto engine_ids = engine_model.tokenize(text);
    CHECK(engine_ids.has_value());
    if (!engine_ids.has_value()) {
      continue;
    }
    if (*engine_ids != abi_ids) {
      std::fprintf(stderr, "FAIL [tokenize] \"%s\": c abi %zu ids, engine %zu ids differ\n", text,
                   abi_ids.size(), engine_ids->size());
      ++g_failures;
    } else {
      std::printf("[tokenize] \"%s\" -> %zu ids match\n", text, abi_ids.size());
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("SKIP: usage: %s <voxtral.gguf> [fixtures_dir] [--quick]\n", argv[0]);
    return 2;
  }
  // --quick (the default ctest registration): plain transcription + the
  // same-session re-run only; the language-hint and translate modes (two more
  // 3B decodes per side) run in the *_full registration.
  bool quick = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--quick") quick = true;
  }
  if (argc > 2 && std::string(argv[argc - 1]) == "--quick") --argc;
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path fixtures_dir = argc > 2 ? std::filesystem::path(argv[2])
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

    // The engine directly, through the registry.
    ModelRegistry registry;
    registry.register_loader(engine::models::voxtral::make_voxtral_loader());
    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "voxtral";
    auto loaded = registry.load(request);
    check_engine_capabilities(*loaded);
    check_tokenize(abi_model, *loaded);

    TaskSpec task;
    task.task = VoiceTaskKind::Asr;
    task.mode = RunMode::Offline;
    SessionOptions options;
    options.backend.type = engine::core::BackendType::Cpu;
    // Match the C ABI side (transcribe::default_n_threads(): usable CPUs, cap 8).
    options.backend.threads = static_cast<int>(std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
    auto session = loaded->create_task_session(task, options);
    auto *engine = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    if (engine == nullptr) {
      std::fprintf(stderr, "FAIL: engine session is not offline\n");
      transcribe_model_free(abi_model);
      return 1;
    }
    engine->prepare(build_preparation_request(to_audio(fixtures[0].pcm)));

    const Mode plain{"auto", nullptr, false, nullptr};
    const Mode hinted{"lang=en", "en", false, nullptr};
    const Mode translate{"translate->fr", "en", true, "fr"};

    // Plain transcription, per fixture.
    std::vector<std::string> first;
    for (const auto &f : fixtures) {
      const AbiOutput abi = run_c_abi(abi_model, f.pcm, plain);
      const TaskResult eng = run_engine(*engine, f.pcm, plain);
      compare(std::string(plain.label) + "/" + f.name, abi.text, text_of(eng));
      CHECK(abi.truncated == eng.truncated);
      CHECK(!text_of(eng).empty());
      first.push_back(text_of(eng));
    }

    // Language hint ("lang:en" in the prompt) and translation (instruct
    // template), on the first fixture.
    for (const Mode *mode : {&hinted, &translate}) {
      if (quick) break;
      const AbiOutput abi = run_c_abi(abi_model, fixtures[0].pcm, *mode);
      const TaskResult eng = run_engine(*engine, fixtures[0].pcm, *mode);
      compare(std::string(mode->label) + "/" + fixtures[0].name, abi.text, text_of(eng));
      CHECK(abi.truncated == eng.truncated);
    }

    // A second run of the same clip on the same engine session must
    // reproduce the first exactly.
    {
      const std::string again = text_of(run_engine(*engine, fixtures[0].pcm, plain));
      if (again != first[0]) {
        std::fprintf(stderr, "FAIL [rerun] second engine run differs\n    first : %s\n    second: %s\n",
                     first[0].c_str(), again.c_str());
        ++g_failures;
      }
    }

    transcribe_model_free(abi_model);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "voxtral_engine_arch_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("voxtral_engine_arch_parity_test: C ABI == engine (%s)\n",
              quick ? "quick: transcribe, re-run, tokenize" : "transcribe, re-run, lang hint, translate, tokenize");
  return 0;
}
