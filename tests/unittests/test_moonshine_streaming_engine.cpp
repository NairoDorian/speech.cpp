// tests/unittests/test_moonshine_streaming_engine.cpp - Phase 11 W1b
// engine-path gate for the native Moonshine-Streaming ASR package.
//
// Exercises src/models/moonshine_streaming through the public engine surface
// exactly like a product consumer would: ModelRegistry -> IVoiceModelLoader ->
// ILoadedVoiceModel -> IStreamingVoiceTaskSession / IOfflineVoiceTaskSession.
//
// The core assertion mirrors the arch-side asr_stream_text_wer_test: each
// fixture is transcribed OFFLINE and then STREAMED on the SAME session with
// odd-sized (~100-400 ms) chunks, and the two transcripts must agree word for
// word (divergence 0). That is the property the whole incremental design rests
// on - per-feed encoder windows, absolute-frame adapter positions and per-frame
// cross-KV projection are supposed to reconstruct the one-shot result exactly -
// so any drift in the window geometry shows up here immediately. Running both
// modes on one session also proves mode switching and reset.
//
// Skips (exit 2) when the pinned GGUF or fixtures are absent, mirroring the
// asr_e2e_*_test download contract.

#include "abi_test_wav.h"
#include "asr_test_text.h"

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/streaming_session_base.h"
#include "engine/models/moonshine_streaming/session.h"

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
// The incremental path is supposed to reproduce the one-shot result exactly.
constexpr size_t kMaxStreamOfflineDivergenceWords = 0;

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

// Deliberately odd, non-frame-aligned chunk sizes spanning ~100-400 ms at
// 16 kHz. Cycling through them means chunk boundaries land mid-encoder-frame,
// which is exactly the case the runtime's PCM buffering + frame algebra has to
// absorb.
const std::vector<size_t> &odd_chunk_sizes() {
  static const std::vector<size_t> sizes = {1601, 3203, 6397, 2477, 4801};
  return sizes;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: moonshine_streaming_engine_smoke_test <model.gguf> "
                 "<librispeech_fixture_dir>\n";
    return 1;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path fixture_dir = argv[2];
  if (!std::filesystem::exists(model_path)) {
    std::cout << "SKIP: pinned moonshine-streaming GGUF not found: "
              << model_path << "\n";
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

    // Registry path: the family must be discoverable by canonical id and
    // alias, not handed a raw session object.
    ModelRegistry registry;
    registry.register_loader(
        engine::models::moonshine_streaming::make_moonshine_streaming_loader());
    if (!registry.supports_family("moonshine_streaming") ||
        !registry.supports_family("moonshine-streaming")) {
      std::cerr << "FAIL: registry does not resolve moonshine_streaming "
                   "canonical id/alias\n";
      return 1;
    }

    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "moonshine_streaming";
    auto loaded = registry.load(request);
    if (!loaded) {
      std::cerr << "FAIL: registry.load returned null\n";
      return 1;
    }
    if (loaded->metadata().family != "moonshine_streaming") {
      std::cerr << "FAIL: unexpected metadata family: "
                << loaded->metadata().family << "\n";
      return 1;
    }
    // The family must advertise BOTH modes - a streaming family that cannot
    // also answer an offline request is not a drop-in for the arch path.
    {
      const auto &caps = loaded->capabilities();
      bool has_stream = false;
      bool has_offline = false;
      for (const auto &cap : caps.supported_tasks) {
        if (cap.task != VoiceTaskKind::Asr) {
          continue;
        }
        for (const auto mode : cap.modes) {
          has_stream = has_stream || (mode == RunMode::Streaming);
          has_offline = has_offline || (mode == RunMode::Offline);
        }
      }
      if (!has_stream || !has_offline) {
        std::cerr << "FAIL: family does not advertise ASR offline+streaming\n";
        return 1;
      }
    }

    TaskSpec task;
    task.task = VoiceTaskKind::Asr;
    task.mode = RunMode::Streaming;

    SessionOptions options;
    auto session = loaded->create_task_session(task, options);
    auto *streaming =
        dynamic_cast<IStreamingVoiceTaskSession *>(session.get());
    auto *offline = dynamic_cast<IOfflineVoiceTaskSession *>(session.get());
    auto *base = dynamic_cast<StreamingSessionBase *>(session.get());
    if (streaming == nullptr || offline == nullptr || base == nullptr) {
      std::cerr << "FAIL: session does not expose streaming + offline + base\n";
      return 1;
    }
    if (base->commit_policy() != StreamCommitPolicy::StablePrefix) {
      std::cerr << "FAIL: expected STABLE_PREFIX commit policy\n";
      return 1;
    }

    size_t total_words = 0;
    size_t total_edits_offline = 0;
    size_t total_edits_streamed = 0;
    size_t total_divergence = 0;

    for (const auto &fixture : fixtures) {
      auto audio = read_audio(fixture.wav);
      const auto reference = asr_test::normalize_words(fixture.reference_text);

      // ---------------- offline pass ----------------
      offline->prepare(build_preparation_request(audio));
      TaskRequest run_request;
      run_request.audio_input = audio;
      const auto offline_result = offline->run(run_request);
      if (!offline_result.text_output.has_value()) {
        std::cerr << "FAIL: offline run() produced no text for "
                  << fixture.wav.filename() << "\n";
        return 1;
      }
      const std::string offline_text = offline_result.text_output->text;
      const auto offline_words = asr_test::normalize_words(offline_text);
      if (offline_words.empty()) {
        std::cerr << "FAIL: empty offline transcript for "
                  << fixture.wav.filename() << "\n";
        return 1;
      }

      // ---------------- streaming pass (same session) ----------------
      TaskRequest stream_request;
      // Decode on every advance so the partial path is exercised hard.
      stream_request.options["moonshine_streaming.min_decode_interval_ms"] =
          "0";
      streaming->start_stream(stream_request);
      if (base->stream_state() != StreamLifecycleState::Active) {
        std::cerr << "FAIL: stream did not enter Active\n";
        return 1;
      }

      const auto started = std::chrono::steady_clock::now();
      uint64_t last_revision = base->stream_revision();
      std::string last_committed = std::string(base->committed_text());
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
        (void)streaming->process_audio_chunk(chunk);
        offset += take;

        // Revision is monotonic at every observable point.
        const uint64_t revision = base->stream_revision();
        if (revision < last_revision) {
          std::cerr << "FAIL: stream revision went backwards\n";
          return 1;
        }
        last_revision = revision;

        // The base guarantees APPEND-ONLY committed text: once bytes are
        // committed they are never rewritten, even when a later from-BOS
        // re-decode revises that region (the base then keeps the old commit
        // rather than rewriting it). Assert that contract - not the stronger
        // "committed is a live prefix of full_text", which the base
        // deliberately does not promise.
        const std::string committed = base->committed_text();
        if (committed.size() < last_committed.size() ||
            committed.compare(0, last_committed.size(), last_committed) != 0) {
          std::cerr << "FAIL: committed text was rewritten (append-only "
                       "contract violated)\n"
                    << "  was: " << last_committed << "\n"
                    << "  now: " << committed << "\n";
          return 1;
        }
        last_committed = committed;
      }

      const auto final_result = streaming->finalize();
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      if (base->stream_state() != StreamLifecycleState::Finished) {
        std::cerr << "FAIL: stream did not reach Finished\n";
        return 1;
      }
      if (!final_result.text_output.has_value()) {
        std::cerr << "FAIL: finalize() produced no text for "
                  << fixture.wav.filename() << "\n";
        return 1;
      }
      const std::string streamed_text = final_result.text_output->text;
      const auto streamed_words = asr_test::normalize_words(streamed_text);
      if (streamed_words.empty()) {
        std::cerr << "FAIL: empty streamed transcript for "
                  << fixture.wav.filename() << "\n";
        return 1;
      }

      const size_t divergence =
          asr_test::word_edit_distance(offline_words, streamed_words);
      total_divergence += divergence;
      total_words += reference.size();
      total_edits_offline +=
          asr_test::word_edit_distance(reference, offline_words);
      total_edits_streamed +=
          asr_test::word_edit_distance(reference, streamed_words);

      std::cout << "  " << fixture.wav.filename().string() << " (" << seconds
                << " s stream)\n"
                << "    offline : " << offline_text << "\n"
                << "    streamed: " << streamed_text << "\n"
                << "    divergence: " << divergence << " word(s)\n";

      if (divergence > kMaxStreamOfflineDivergenceWords) {
        std::cerr << "FAIL: streamed transcript diverged from offline by "
                  << divergence << " word(s) (bound "
                  << kMaxStreamOfflineDivergenceWords << ") for "
                  << fixture.wav.filename() << "\n";
        return 1;
      }

      // reset() must return the session to a clean Idle state so the next
      // fixture starts from scratch on this same session.
      streaming->reset();
      if (base->stream_state() != StreamLifecycleState::Idle) {
        std::cerr << "FAIL: reset() did not return the stream to Idle\n";
        return 1;
      }
      if (!base->full_text().empty() || !base->committed_text().empty()) {
        std::cerr << "FAIL: reset() left text behind\n";
        return 1;
      }
    }

    const auto pct = [&](size_t edits) {
      return total_words > 0 ? 100.0 * static_cast<double>(edits) /
                                   static_cast<double>(total_words)
                             : 100.0;
    };
    const double offline_wer = pct(total_edits_offline);
    const double streamed_wer = pct(total_edits_streamed);
    std::cout << "corpus WER offline : " << offline_wer << "% ("
              << total_edits_offline << "/" << total_words << ")\n"
              << "corpus WER streamed: " << streamed_wer << "% ("
              << total_edits_streamed << "/" << total_words << ")\n"
              << "total streamed-vs-offline divergence: " << total_divergence
              << " word(s)\n";

    if (offline_wer > kMaxCorpusWerPct) {
      std::cerr << "FAIL: offline corpus WER " << offline_wer
                << "% exceeds structural bound " << kMaxCorpusWerPct << "%\n";
      return 1;
    }
    if (streamed_wer > kMaxCorpusWerPct) {
      std::cerr << "FAIL: streamed corpus WER " << streamed_wer
                << "% exceeds structural bound " << kMaxCorpusWerPct << "%\n";
      return 1;
    }
    if (total_divergence > kMaxStreamOfflineDivergenceWords) {
      std::cerr << "FAIL: total streamed-vs-offline divergence "
                << total_divergence << " exceeds bound "
                << kMaxStreamOfflineDivergenceWords << "\n";
      return 1;
    }

    // Cancellation contract: a pre-requested abort unwinds the stream.
    {
      auto abort_session = loaded->create_task_session(task, options);
      auto *abort_streaming =
          dynamic_cast<IStreamingVoiceTaskSession *>(abort_session.get());
      TaskRequest abort_request;
      abort_streaming->start_stream(abort_request);
      abort_session->request_abort();
      bool canceled = false;
      try {
        auto audio = read_audio(fixtures.front().wav);
        AudioChunk chunk;
        chunk.sample_rate = audio.sample_rate;
        chunk.channels = 1;
        chunk.samples.assign(audio.samples.begin(),
                             audio.samples.begin() +
                                 std::min<size_t>(16000, audio.samples.size()));
        (void)abort_streaming->process_audio_chunk(chunk);
      } catch (const ProgressCanceled &) {
        canceled = true;
      } catch (...) {
        canceled = true;
      }
      if (!canceled) {
        std::cerr << "FAIL: request_abort was not honored by the stream\n";
        return 1;
      }
    }

    std::cout << "moonshine_streaming_engine_smoke_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
