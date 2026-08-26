// tests/unittests/test_qwen3_asr_engine.cpp - Phase 10.5 engine-path gate for
// the canonical Qwen3-ASR package after the arch retirement.
//
// Exercises src/models/qwen3_asr through the public engine surface exactly
// like a product consumer: ModelRegistry -> IVoiceModelLoader ->
// ILoadedVoiceModel -> IOfflineVoiceTaskSession::run / run_batch.
//
// Three things are gated, in the order the Phase-10 verdict demanded:
//
//   1. Text on real audio. The four vendored LibriSpeech fixtures are scored
//      against their references; the corpus must stay within the structural
//      bound and at or below the edit count measured when this gate first
//      landed (the "== baseline" bar W1a/W1b/W2a were held to). There is no
//      arch baseline to compare against: the arch only loads transcribe.cpp-
//      flavored GGUFs and the only downloadable Qwen3-ASR GGUFs are audio.cpp
//      packages, so the engine's own first measurement is the reference.
//   2. The merged feature. spec_k_drafts=4 (transcribe.cpp's 1-gram-lookup
//      speculative decode) must reproduce the k=0 transcript on every
//      fixture within the divergence transcribe.cpp documented for the
//      mechanism - at most one flipped near-tie token per utterance, never a
//      different transcript - and must not cost corpus accuracy.
//   3. Product contracts: ordered non-empty run_batch results and a
//      pre-requested abort unwinding run().
//
// Skips (exit 2) while the pinned GGUF or the fixtures are absent
// (scripts/fetch_asr_test_model.py).

#include "abi_test_wav.h"
#include "asr_test_text.h"

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/qwen3_asr/loader.h"

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

constexpr double kMaxCorpusWerPct = 10.0;
// Corpus edits measured by this gate when it first landed (Qwen3-ASR-0.6B
// Q8_0, CPU, k=0). Regressions fail here rather than degrading silently.
constexpr size_t kEngineBaselineEdits = 2;
// transcribe.cpp measured exactly one flipped near-tie token in 255 for every
// k >= 1 (Qwen3-ASR-0.6B, CPU); a whole different transcript is a bug.
constexpr size_t kMaxSpecDivergencePerUtterance = 1;

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

struct RunOutcome {
  std::string text;
  double seconds = 0.0;
};

RunOutcome run_once(IOfflineVoiceTaskSession &offline, const AudioBuffer &audio,
                    const char *spec_k_drafts) {
  offline.prepare(build_preparation_request(audio));
  TaskRequest request;
  request.audio_input = audio;
  if (spec_k_drafts != nullptr) {
    request.options["spec_k_drafts"] = spec_k_drafts;
  }
  const auto started = std::chrono::steady_clock::now();
  const auto result = offline.run(request);
  RunOutcome outcome;
  outcome.seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started)
                        .count();
  if (!result.text_output.has_value()) {
    throw std::runtime_error("run() returned no text output");
  }
  outcome.text = result.text_output->text;
  return outcome;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: qwen3_asr_engine_smoke_test <qwen3-asr.gguf> "
                 "<librispeech_fixture_dir>\n";
    return 1;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path fixture_dir = argv[2];
  if (!std::filesystem::exists(model_path)) {
    std::cout << "SKIP: pinned Qwen3-ASR GGUF not found: " << model_path
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

    ModelRegistry registry;
    registry.register_loader(
        engine::models::qwen3_asr::make_qwen3_asr_loader());
    if (!registry.supports_family("qwen3_asr")) {
      std::cerr << "FAIL: registry does not resolve qwen3_asr\n";
      return 1;
    }

    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "qwen3_asr";
    auto loaded = registry.load(request);
    if (!loaded) {
      std::cerr << "FAIL: registry.load returned null\n";
      return 1;
    }
    if (loaded->metadata().family != "qwen3_asr") {
      std::cerr << "FAIL: unexpected metadata family: "
                << loaded->metadata().family << "\n";
      return 1;
    }
    if (!loaded->capabilities().supports_speculative_decode) {
      std::cerr << "FAIL: qwen3_asr must advertise speculative decode "
                   "(transcribe_capabilities::supports_spec_decode parity)\n";
      return 1;
    }
    std::cout << "  loaded variant: " << loaded->metadata().variant << "\n";

    TaskSpec task;
    task.task = VoiceTaskKind::Asr;
    task.mode = RunMode::Offline;

    SessionOptions options;
    auto session = loaded->create_task_session(task, options);
    auto *offline = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
      std::cerr << "FAIL: session does not expose offline execution\n";
      return 1;
    }

    size_t total_words = 0;
    size_t total_edits = 0;
    size_t total_spec_edits = 0;
    double total_audio_s = 0.0;
    double total_decode_s = 0.0;
    double total_spec_decode_s = 0.0;
    std::vector<AudioBuffer> batch_audio;

    for (const auto &fixture : fixtures) {
      auto audio = read_audio(fixture.wav);
      const double audio_s =
          audio.sample_rate > 0
              ? static_cast<double>(audio.samples.size()) / audio.sample_rate
              : 0.0;

      // 1. Reference path (k = 0).
      const auto plain = run_once(*offline, audio, nullptr);
      const auto reference = asr_test::normalize_words(fixture.reference_text);
      const auto hypothesis = asr_test::normalize_words(plain.text);
      if (hypothesis.empty()) {
        std::cerr << "FAIL: empty transcript for " << fixture.wav.filename()
                  << "\n";
        return 1;
      }
      const size_t edits = asr_test::word_edit_distance(reference, hypothesis);
      total_words += reference.size();
      total_edits += edits;
      total_audio_s += audio_s;
      total_decode_s += plain.seconds;
      std::cout << "  " << fixture.wav.filename().string() << ": " << edits
                << "/" << reference.size() << " edits (" << plain.seconds
                << " s): " << plain.text << "\n";

      // 2. Speculative path (k = 4) must reproduce the reference transcript.
      const auto spec = run_once(*offline, audio, "4");
      const auto spec_words = asr_test::normalize_words(spec.text);
      const size_t divergence =
          asr_test::word_edit_distance(hypothesis, spec_words);
      const size_t spec_edits =
          asr_test::word_edit_distance(reference, spec_words);
      total_spec_edits += spec_edits;
      total_spec_decode_s += spec.seconds;
      std::cout << "    spec k=4: divergence " << divergence << " word(s), "
                << spec_edits << " edits (" << spec.seconds << " s)"
                << (divergence > 0 ? ": " + spec.text : std::string()) << "\n";
      if (divergence > kMaxSpecDivergencePerUtterance) {
        std::cerr << "FAIL: spec_k_drafts=4 diverged from k=0 by "
                  << divergence << " words on " << fixture.wav.filename()
                  << " (budget " << kMaxSpecDivergencePerUtterance
                  << "). The acceptance rule is exact; a different "
                     "transcript means the verify graph or the KV "
                     "bookkeeping is wrong.\n";
        return 1;
      }
      batch_audio.push_back(std::move(audio));
    }

    const double corpus_wer_pct =
        total_words > 0 ? 100.0 * static_cast<double>(total_edits) /
                              static_cast<double>(total_words)
                        : 100.0;
    const double rtf =
        total_audio_s > 0.0 ? total_decode_s / total_audio_s : 0.0;
    const double spec_rtf =
        total_audio_s > 0.0 ? total_spec_decode_s / total_audio_s : 0.0;
    std::cout << "corpus WER: " << corpus_wer_pct << "% (" << total_edits << "/"
              << total_words << " edits), RTF " << rtf << "; spec k=4: "
              << total_spec_edits << " edits, RTF " << spec_rtf << "\n";

    if (corpus_wer_pct > kMaxCorpusWerPct) {
      std::cerr << "FAIL: corpus WER " << corpus_wer_pct
                << "% exceeds structural bound " << kMaxCorpusWerPct << "%\n";
      return 1;
    }
    if (total_edits > kEngineBaselineEdits) {
      std::cerr << "FAIL: engine path made " << total_edits
                << " word edits, above the baseline of "
                << kEngineBaselineEdits
                << " measured when this gate landed. A change that only "
                   "meets the 10% bound but drifts from that baseline is a "
                   "regression.\n";
      return 1;
    }
    if (total_spec_edits > total_edits + kMaxSpecDivergencePerUtterance) {
      std::cerr << "FAIL: spec_k_drafts=4 cost " << total_spec_edits
                << " corpus edits against " << total_edits
                << " for k=0; speculative decode may not cost accuracy.\n";
      return 1;
    }

    // 3a. Batched execution: results must be ordered and non-empty.
    std::vector<TaskRequest> batch_requests;
    for (size_t i = 0; i < 2 && i < batch_audio.size(); ++i) {
      TaskRequest batch_request;
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

    // 3b. Cancellation contract: a pre-requested abort unwinds run().
    {
      auto abort_session = loaded->create_task_session(task, options);
      auto *abort_offline =
          dynamic_cast<IOfflineVoiceTaskSession *>(abort_session.get());
      abort_offline->prepare(build_preparation_request(batch_audio.front()));
      abort_session->request_abort();
      bool canceled = false;
      try {
        TaskRequest rerun;
        rerun.audio_input = batch_audio.front();
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

    std::cout << "qwen3_asr_engine_smoke_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
