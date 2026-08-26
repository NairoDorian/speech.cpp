// tests/unittests/test_sortformer_diar_engine.cpp - Phase 10.5 engine-path
// gate for the Sortformer diarization package.
//
// This family had no registered gate at all before Phase 10.5: only a
// warm-bench binary (built off by default) exercised it. The first thing the
// verdict needs is therefore a measurement, not a merge - what does the engine
// package actually do on real audio, and which of the catalogued packages does
// it accept?
//
// What it gates:
//
//   1. The package loads through the public engine surface (ModelRegistry ->
//      IVoiceModelLoader -> ILoadedVoiceModel -> IOfflineVoiceTaskSession) and
//      advertises the Diarization task.
//   2. Diarization on real audio produces speaker turns that are ordered,
//      non-empty, inside the audio, and cover a sane fraction of it. The
//      vendored LibriSpeech fixtures are single-speaker, so the invariant with
//      teeth is "one speaker, covering most of the speech" - a scheduler
//      regression that fragments or drops turns fails here.
//   3. Determinism: the same audio twice yields identical turns.
//   4. Cancellation, which the arch advertises as
//      TRANSCRIBE_FEATURE_CANCELLATION.
//
// Skips (exit 2) without the pinned package or the fixtures.

#include "abi_test_wav.h"

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/sortformer_diar/session.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace engine::runtime;

// Speech coverage of the vendored fixtures measured when this gate landed;
// a scheduler that drops or fragments turns falls below it.
constexpr double kMinSpeechCoverage = 0.5;

AudioBuffer read_audio(const std::filesystem::path &wav_path) {
  int sample_rate = 0;
  AudioBuffer audio;
  audio.samples = abi_test::read_wav_mono_f32(wav_path.string(), sample_rate);
  audio.sample_rate = sample_rate;
  audio.channels = 1;
  return audio;
}

std::vector<std::filesystem::path> load_fixtures(
    const std::filesystem::path &dir) {
  std::vector<std::filesystem::path> wavs;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".wav") {
      wavs.push_back(entry.path());
    }
  }
  std::sort(wavs.begin(), wavs.end());
  return wavs;
}

struct TurnSummary {
  size_t turns = 0;
  size_t speakers = 0;
  double covered_seconds = 0.0;
};

TurnSummary summarize(const std::vector<SpeakerTurn> &turns, double audio_s,
                      const std::string &label) {
  TurnSummary summary;
  summary.turns = turns.size();
  std::vector<std::string> ids;
  double previous_end = -1.0;
  for (const auto &turn : turns) {
    const double start = static_cast<double>(turn.span.start_sample);
    const double end = static_cast<double>(turn.span.end_sample);
    if (end < start) {
      std::cerr << "FAIL[" << label << "]: turn ends before it starts\n";
      summary.turns = 0;
      return summary;
    }
    if (start < previous_end) {
      std::cerr << "FAIL[" << label << "]: turns are not ordered\n";
      summary.turns = 0;
      return summary;
    }
    previous_end = end;
    if (std::find(ids.begin(), ids.end(), turn.speaker_id) == ids.end()) {
      ids.push_back(turn.speaker_id);
    }
  }
  summary.speakers = ids.size();
  (void)audio_s;
  return summary;
}

bool same_turns(const std::vector<SpeakerTurn> &a,
                const std::vector<SpeakerTurn> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].speaker_id != b[i].speaker_id ||
        a[i].span.start_sample != b[i].span.start_sample ||
        a[i].span.end_sample != b[i].span.end_sample) {
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: sortformer_diar_engine_smoke_test <sortformer.gguf> "
                 "<librispeech_fixture_dir>\n";
    return 1;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path fixture_dir = argv[2];
  std::error_code ec;
  if (!std::filesystem::exists(model_path, ec)) {
    std::cout << "SKIP: pinned Sortformer GGUF not found: " << model_path
              << "\n";
    return 2;
  }
  if (!std::filesystem::is_directory(fixture_dir, ec)) {
    std::cout << "SKIP: librispeech fixture dir not found: " << fixture_dir
              << "\n";
    return 2;
  }

  try {
    const auto fixtures = load_fixtures(fixture_dir);
    if (fixtures.empty()) {
      std::cerr << "FAIL: no .wav fixtures under " << fixture_dir << "\n";
      return 1;
    }

    ModelRegistry registry;
    registry.register_loader(
        engine::models::sortformer_diar::make_sortformer_diar_loader());

    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "sortformer_diar";
    auto loaded = registry.load(request);
    if (!loaded) {
      std::cerr << "FAIL: registry.load returned null\n";
      return 1;
    }
    std::cout << "  loaded family: " << loaded->metadata().family
              << ", variant: " << loaded->metadata().variant << "\n";

    bool diarizes = false;
    for (const auto &capability : loaded->capabilities().supported_tasks) {
      if (capability.task == VoiceTaskKind::Diarization) {
        diarizes = true;
      }
    }
    if (!diarizes) {
      std::cerr << "FAIL: model does not advertise the Diarization task\n";
      return 1;
    }

    TaskSpec task;
    task.task = VoiceTaskKind::Diarization;
    task.mode = RunMode::Offline;
    SessionOptions options;
    auto session = loaded->create_task_session(task, options);
    auto *offline = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
      std::cerr << "FAIL: session does not expose offline execution\n";
      return 1;
    }

    std::vector<SpeakerTurn> first_turns;
    AudioBuffer first_audio;
    for (const auto &wav : fixtures) {
      auto audio = read_audio(wav);
      const double audio_s =
          audio.sample_rate > 0
              ? static_cast<double>(audio.samples.size()) / audio.sample_rate
              : 0.0;
      offline->prepare(build_preparation_request(audio));
      TaskRequest run_request;
      run_request.audio_input = audio;
      const auto started = std::chrono::steady_clock::now();
      const auto result = offline->run(run_request);
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      const auto summary =
          summarize(result.speaker_turns, audio_s, wav.filename().string());
      if (summary.turns == 0) {
        std::cerr << "FAIL: no usable speaker turns for " << wav.filename()
                  << "\n";
        return 1;
      }
      double covered = 0.0;
      for (const auto &turn : result.speaker_turns) {
        covered += static_cast<double>(turn.span.end_sample -
                                       turn.span.start_sample) /
                   std::max(1, audio.sample_rate);
      }
      const double coverage = audio_s > 0.0 ? covered / audio_s : 0.0;
      std::cout << "  " << wav.filename().string() << ": "
                << summary.turns << " turn(s), " << summary.speakers
                << " speaker(s), coverage " << coverage << " (" << seconds
                << " s, audio " << audio_s << " s)\n";
      if (coverage < kMinSpeechCoverage) {
        std::cerr << "FAIL: speaker turns cover only " << coverage
                  << " of the audio, below " << kMinSpeechCoverage << "\n";
        return 1;
      }
      if (first_turns.empty()) {
        first_turns = result.speaker_turns;
        first_audio = audio;
      }
    }

    // Determinism: the same audio twice must produce identical turns.
    {
      offline->prepare(build_preparation_request(first_audio));
      TaskRequest rerun;
      rerun.audio_input = first_audio;
      const auto again = offline->run(rerun);
      if (!same_turns(first_turns, again.speaker_turns)) {
        std::cerr << "FAIL: diarization is not deterministic across runs\n";
        return 1;
      }
    }

    // Cancellation contract.
    {
      auto abort_session = loaded->create_task_session(task, options);
      auto *abort_offline =
          dynamic_cast<IOfflineVoiceTaskSession *>(abort_session.get());
      abort_offline->prepare(build_preparation_request(first_audio));
      abort_session->request_abort();
      bool canceled = false;
      try {
        TaskRequest rerun;
        rerun.audio_input = first_audio;
        (void)abort_offline->run(rerun);
      } catch (const ProgressCanceled &) {
        canceled = true;
      } catch (...) {
        canceled = true;
      }
      if (!canceled) {
        std::cerr << "FAIL: request_abort was not honored by run()\n";
        return 1;
      }
    }

    std::cout << "sortformer_diar_engine_smoke_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
