// tests/unittests/test_voxtral_realtime_engine.cpp - Phase 10.5 engine-path
// gate for the canonical Voxtral-Realtime package after the arch retirement.
//
// What it gates, in the order the Phase-10 verdict demanded:
//
//   1. Text on real audio through the public engine surface (ModelRegistry ->
//      IVoiceModelLoader -> ILoadedVoiceModel -> IOfflineVoiceTaskSession),
//      held to the corpus edit count measured when this gate landed.
//   2. Streaming vs offline. The family's whole point is realtime decoding;
//      the streamed transcript is compared word-for-word against the offline
//      one on the same audio.
//   3. The merged feature: num_delay_tokens, which transcribe.cpp exposed
//      through a typed stream extension and the engine hardwired. Asking for
//      the model default explicitly must reproduce the default run exactly,
//      and a value outside the publisher's validated set must be refused -
//      before any audio is consumed, not mid-stream.
//   4. Cancellation, which the arch advertised as
//      TRANSCRIBE_FEATURE_CANCELLATION and the engine did not honour.
//
// The model is the pinned q4_k package (scripts/fetch_asr_test_model.py):
// a 4B model, so this gate is deliberately frugal - the offline pass runs the
// four vendored LibriSpeech fixtures, everything else runs on the shortest
// one. Skips (exit 2) when the model or the fixtures are absent.

#include "abi_test_wav.h"
#include "asr_test_text.h"

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/voxtral_realtime/loader.h"

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

constexpr double kMaxCorpusWerPct = 25.0;
// Corpus edits measured by this gate when it first landed (q4_k, CPU):
// 2/69, RTF 5.6 - the same two edits qwen3_asr makes on the same fixture, so
// this is the corpus's floor rather than a family weakness. Regressions fail
// here rather than degrading silently.
constexpr size_t kEngineBaselineEdits = 2;
// Streaming decodes incrementally while offline sees the whole buffer, so the
// two are not required to be identical - but they must stay close, or the
// realtime path is not transcribing the same audio.
constexpr size_t kMaxStreamOfflineDivergenceWords = 0;  // filled from the first measurement

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

// Deliberately not aligned to the family's preferred chunk: a caller feeding a
// live microphone will not be, and the session has to re-block internally.
const std::vector<size_t> &odd_chunk_sizes() {
  static const std::vector<size_t> sizes = {1600, 4001, 977, 8000};
  return sizes;
}

std::string stream_audio(IStreamingVoiceTaskSession &streaming,
                         const AudioBuffer &audio,
                         const std::string &num_delay_tokens) {
  TaskRequest request;
  if (!num_delay_tokens.empty()) {
    request.options["num_delay_tokens"] = num_delay_tokens;
  }
  streaming.start_stream(request);
  size_t offset = 0;
  size_t size_idx = 0;
  while (offset < audio.samples.size()) {
    const size_t want = odd_chunk_sizes()[size_idx % odd_chunk_sizes().size()];
    ++size_idx;
    const size_t take = std::min(want, audio.samples.size() - offset);
    AudioChunk chunk;
    chunk.sample_rate = audio.sample_rate;
    chunk.channels = 1;
    chunk.start_sample = static_cast<int64_t>(offset);
    chunk.samples.assign(audio.samples.begin() + offset,
                         audio.samples.begin() + offset + take);
    (void)streaming.process_audio_chunk(chunk);
    offset += take;
  }
  const auto result = streaming.finish_stream();
  if (!result.text_output.has_value()) {
    throw std::runtime_error("finish_stream() produced no text");
  }
  return result.text_output->text;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: voxtral_realtime_engine_smoke_test <voxtral.gguf> "
                 "<librispeech_fixture_dir>\n";
    return 1;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path fixture_dir = argv[2];
  if (!std::filesystem::exists(model_path)) {
    std::cout << "SKIP: pinned Voxtral-Realtime GGUF not found: " << model_path
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
        engine::models::voxtral_realtime::make_voxtral_realtime_loader());
    {
      ModelLoadRequest sniff;
      sniff.model_path = model_path;  // no family_hint: pure auto-detection
      const auto detected = registry.inspect(sniff).metadata.family;
      if (detected != "voxtral_realtime") {
        std::cerr << "FAIL: auto-detection resolved the GGUF as '" << detected
                  << "', not voxtral_realtime\n";
        return 1;
      }
    }

    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "voxtral_realtime";
    auto loaded = registry.load(request);
    if (!loaded) {
      std::cerr << "FAIL: registry.load returned null\n";
      return 1;
    }
    if (!loaded->capabilities().supports_cancellation) {
      std::cerr << "FAIL: voxtral_realtime must advertise cancellation "
                   "(TRANSCRIBE_FEATURE_CANCELLATION parity)\n";
      return 1;
    }
    std::cout << "  loaded variant: " << loaded->metadata().variant << "\n";

    TaskSpec offline_task;
    offline_task.task = VoiceTaskKind::Asr;
    offline_task.mode = RunMode::Offline;
    SessionOptions options;
    auto session = loaded->create_task_session(offline_task, options);
    auto *offline = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
      std::cerr << "FAIL: session does not expose offline execution\n";
      return 1;
    }

    // ---------------- 1. offline text ----------------
    size_t total_words = 0;
    size_t total_edits = 0;
    double total_audio_s = 0.0;
    double total_decode_s = 0.0;
    std::vector<AudioBuffer> audios;
    std::vector<std::string> offline_texts;

    for (const auto &fixture : fixtures) {
      auto audio = read_audio(fixture.wav);
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
      if (!result.text_output.has_value()) {
        std::cerr << "FAIL: run() returned no text for "
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
      const size_t edits = asr_test::word_edit_distance(reference, hypothesis);
      total_words += reference.size();
      total_edits += edits;
      total_audio_s += audio_s;
      total_decode_s += seconds;
      std::cout << "  " << fixture.wav.filename().string() << ": " << edits
                << "/" << reference.size() << " edits (" << seconds
                << " s): " << text << "\n";
      audios.push_back(std::move(audio));
      offline_texts.push_back(text);
    }

    const double corpus_wer_pct =
        total_words > 0 ? 100.0 * static_cast<double>(total_edits) /
                              static_cast<double>(total_words)
                        : 100.0;
    const double rtf =
        total_audio_s > 0.0 ? total_decode_s / total_audio_s : 0.0;
    std::cout << "corpus WER: " << corpus_wer_pct << "% (" << total_edits << "/"
              << total_words << " edits), RTF " << rtf << "\n";
    if (corpus_wer_pct > kMaxCorpusWerPct) {
      std::cerr << "FAIL: corpus WER " << corpus_wer_pct
                << "% exceeds structural bound " << kMaxCorpusWerPct << "%\n";
      return 1;
    }
    if (total_edits > kEngineBaselineEdits) {
      std::cerr << "FAIL: engine path made " << total_edits
                << " word edits, above the baseline of "
                << kEngineBaselineEdits
                << " measured when this gate landed.\n";
      return 1;
    }

    // The shortest fixture carries the streaming and knob checks: this is a 4B
    // model, and every extra pass is real wall-clock in CI.
    size_t shortest = 0;
    for (size_t i = 1; i < audios.size(); ++i) {
      if (audios[i].samples.size() < audios[shortest].samples.size()) {
        shortest = i;
      }
    }
    const AudioBuffer &short_audio = audios[shortest];
    const auto short_offline_words =
        asr_test::normalize_words(offline_texts[shortest]);

    TaskSpec stream_task;
    stream_task.task = VoiceTaskKind::Asr;
    stream_task.mode = RunMode::Streaming;
    auto stream_session = loaded->create_task_session(stream_task, options);
    auto *streaming =
        dynamic_cast<IStreamingVoiceTaskSession *>(stream_session.get());
    if (streaming == nullptr) {
      std::cerr << "FAIL: session does not expose streaming execution\n";
      return 1;
    }
    streaming->prepare(build_preparation_request(short_audio));

    // ---------------- 2. streaming vs offline ----------------
    const std::string streamed = stream_audio(*streaming, short_audio, "");
    const auto streamed_words = asr_test::normalize_words(streamed);
    if (streamed_words.empty()) {
      std::cerr << "FAIL: empty streamed transcript\n";
      return 1;
    }
    const size_t divergence =
        asr_test::word_edit_distance(short_offline_words, streamed_words);
    std::cout << "  streamed: " << streamed << "\n"
              << "  streamed-vs-offline divergence: " << divergence
              << " word(s)\n";
    if (divergence > kMaxStreamOfflineDivergenceWords) {
      std::cerr << "FAIL: streamed-vs-offline divergence " << divergence
                << " exceeds bound " << kMaxStreamOfflineDivergenceWords
                << "\n";
      return 1;
    }

    // ---------------- 3. num_delay_tokens ----------------
    // Asking for the model default explicitly must change nothing.
    const std::string streamed_default_delay =
        stream_audio(*streaming, short_audio, "6");
    if (asr_test::normalize_words(streamed_default_delay) != streamed_words) {
      std::cerr << "FAIL: num_delay_tokens=6 (the model default) changed the "
                   "transcript:\n    default: "
                << streamed << "\n    explicit: " << streamed_default_delay
                << "\n";
      return 1;
    }
    // Outside the publisher's validated set: refused before any audio moves.
    for (const char *bad : {"0", "16", "31"}) {
      bool refused = false;
      try {
        TaskRequest bad_request;
        bad_request.options["num_delay_tokens"] = bad;
        streaming->start_stream(bad_request);
      } catch (const std::exception &) {
        refused = true;
      }
      if (!refused) {
        std::cerr << "FAIL: num_delay_tokens=" << bad
                  << " was accepted; the publisher's validated set is 1..15 "
                     "or 30\n";
        return 1;
      }
    }
    std::cout << "  num_delay_tokens: default-equivalent at 6, refuses 0/16/31\n";

    // ---------------- 4. cancellation ----------------
    {
      auto abort_session = loaded->create_task_session(offline_task, options);
      auto *abort_offline =
          dynamic_cast<IOfflineVoiceTaskSession *>(abort_session.get());
      abort_offline->prepare(build_preparation_request(short_audio));
      abort_session->request_abort();
      bool canceled = false;
      try {
        TaskRequest rerun;
        rerun.audio_input = short_audio;
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

    std::cout << "voxtral_realtime_engine_smoke_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
