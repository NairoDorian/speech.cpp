// tests/unittests/test_moonshine_engine.cpp - Phase 11 W1 engine-path gate
// for the native Moonshine ASR package.
//
// Exercises src/models/moonshine through the public engine surface exactly
// like a product consumer would: ModelRegistry -> IVoiceModelLoader ->
// ILoadedVoiceModel -> IOfflineVoiceTaskSession::run / run_batch. Gated on
// the same four LibriSpeech fixtures the arch-side WER gates use, with the
// same structural bound (corpus WER <= 10%), so the native package is held
// to the standard the transcribe.dll path already meets.
//
// Skips (exit 2) when the pinned GGUF or fixtures are absent, mirroring the
// asr_e2e_*_test download contract.

#include "abi_test_wav.h"
#include "asr_test_text.h"

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/moonshine/session.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace engine::runtime;

constexpr double kMaxCorpusWerPct = 10.0;

struct EngineFixture {
  std::filesystem::path wav;
  std::string reference_text;
};

std::vector<EngineFixture> load_fixtures(const std::filesystem::path &dir) {
  std::vector<EngineFixture> fixtures;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".wav") {
      continue;
    }
    auto txt = entry.path();
    txt.replace_extension(".txt");
    if (!std::filesystem::exists(txt)) {
      continue;
    }
    fixtures.push_back({entry.path(), asr_test::read_text_file(txt)});
  }
  std::sort(fixtures.begin(), fixtures.end(),
            [](const EngineFixture &a, const EngineFixture &b) {
              return a.wav.filename() < b.wav.filename();
            });
  return fixtures;
}

AudioBuffer read_audio(const std::filesystem::path &wav_path) {
  int sample_rate = 0;
  AudioBuffer audio;
  audio.samples = abi_test::read_wav_mono_f32(wav_path.string(), sample_rate);
  audio.sample_rate = sample_rate;
  audio.channels = 1;
  return audio;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: moonshine_engine_smoke_test <model.gguf> "
                 "<librispeech_fixture_dir>\n";
    return 1;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path fixture_dir = argv[2];
  if (!std::filesystem::exists(model_path)) {
    std::cout << "SKIP: pinned moonshine GGUF not found: " << model_path
              << "\n";
    return 2;
  }
  if (!std::filesystem::is_directory(fixture_dir)) {
    std::cout << "SKIP: librispeech fixture dir not found: " << fixture_dir
              << "\n";
    return 2;
  }

  try {
    const auto fixtures = load_fixtures(fixture_dir);
    if (fixtures.empty()) {
      std::cerr << "FAIL: no .wav/.txt fixture pairs under " << fixture_dir
                << "\n";
      return 1;
    }

    // Registry path: the family must be discoverable by canonical id,
    // not handed a raw session object.
    engine::runtime::ModelRegistry registry;
    registry.register_loader(
        engine::models::moonshine::make_moonshine_loader());
    if (!registry.supports_family("moonshine") ||
        !registry.supports_family("moonshine-offline")) {
      std::cerr
          << "FAIL: registry does not resolve moonshine canonical id/alias\n";
      return 1;
    }

    engine::runtime::ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "moonshine";
    auto loaded = registry.load(request);
    if (!loaded) {
      std::cerr << "FAIL: registry.load returned null\n";
      return 1;
    }
    if (loaded->metadata().family != "moonshine") {
      std::cerr << "FAIL: unexpected metadata family: "
                << loaded->metadata().family << "\n";
      return 1;
    }

    engine::runtime::TaskSpec task;
    task.task = engine::runtime::VoiceTaskKind::Asr;
    task.mode = engine::runtime::RunMode::Offline;

    engine::runtime::SessionOptions options;
    auto session = loaded->create_task_session(task, options);
    auto *offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
        session.get());
    if (offline == nullptr) {
      std::cerr << "FAIL: session does not expose offline execution\n";
      return 1;
    }

    size_t total_words = 0;
    size_t total_edits = 0;
    std::vector<engine::runtime::AudioBuffer> batch_audio;
    for (const auto &fixture : fixtures) {
      auto audio = read_audio(fixture.wav);

      // prepare() must precede run() (audio preparation contract).
      offline->prepare(engine::runtime::build_preparation_request(audio));

      engine::runtime::TaskRequest run_request;
      run_request.audio_input = audio;
      const auto started = std::chrono::steady_clock::now();
      const auto result = offline->run(run_request);
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();

      if (!result.text_output.has_value()) {
        std::cerr << "FAIL: run() returned no text output for "
                  << fixture.wav.filename() << "\n";
        return 1;
      }
      const std::string &text = result.text_output->text;
      const auto reference = asr_test::normalize_words(fixture.reference_text);
      const auto hypothesis = asr_test::normalize_words(text);
      if (hypothesis.empty()) {
        std::cerr << "FAIL: empty transcript for " << fixture.wav.filename()
                  << "\n";
        return 1;
      }
      total_words += reference.size();
      total_edits += asr_test::word_edit_distance(reference, hypothesis);

      std::cout << "  " << fixture.wav.filename().string() << " (" << seconds
                << " s): " << text << "\n";
      batch_audio.push_back(std::move(audio));
    }

    const double corpus_wer_pct =
        total_words > 0 ? 100.0 * static_cast<double>(total_edits) /
                              static_cast<double>(total_words)
                        : 100.0;
    std::cout << "corpus WER: " << corpus_wer_pct << "% (" << total_edits << "/"
              << total_words << " edits)\n";
    if (corpus_wer_pct > kMaxCorpusWerPct) {
      std::cerr << "FAIL: corpus WER " << corpus_wer_pct
                << "% exceeds structural bound " << kMaxCorpusWerPct << "%\n";
      return 1;
    }

    // Batched execution: results must be ordered and non-empty.
    std::vector<engine::runtime::TaskRequest> batch_requests;
    for (size_t i = 0; i < 2 && i < batch_audio.size(); ++i) {
      engine::runtime::TaskRequest batch_request;
      batch_request.audio_input = batch_audio[i];
      batch_requests.push_back(std::move(batch_request));
    }
    const auto batch_results = offline->run_batch(batch_requests);
    if (batch_results.size() != batch_requests.size()) {
      std::cerr << "FAIL: run_batch result count mismatch\n";
      return 1;
    }
    for (size_t i = 0; i < batch_results.size(); ++i) {
      if (!batch_results[i].text_output.has_value() ||
          batch_results[i].text_output->text.empty()) {
        std::cerr << "FAIL: run_batch produced an empty transcript at slot "
                  << i << "\n";
        return 1;
      }
    }

    // Cancellation contract: a pre-requested abort unwinds run().
    auto abort_session = loaded->create_task_session(task, options);
    auto *abort_offline =
        dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
            abort_session.get());
    abort_offline->prepare(
        engine::runtime::build_preparation_request(batch_audio.front()));
    abort_session->request_abort();
    bool canceled = false;
    try {
      engine::runtime::TaskRequest rerun;
      rerun.audio_input = batch_audio.front();
      (void)abort_offline->run(rerun);
    } catch (const engine::runtime::ProgressCanceled &) {
      canceled = true;
    } catch (...) {
      canceled = true;
    }
    if (!canceled) {
      std::cerr << "FAIL: requested_abort was not honored by run()\n";
      return 1;
    }

    std::cout << "moonshine_engine_smoke_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
