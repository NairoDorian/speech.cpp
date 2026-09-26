// test_granite_speech_engine_arch_parity.cpp - the engine `granite_speech`
// package against IBM Granite Speech through the public C ABI, in one process,
// on the same transcribe.cpp GGUF and audio.
//
// While src/runtime/arch/granite is in the builtin arch table the C ABI side
// IS the arch, so this is the retirement evidence (engine == arch, the Phase
// 10.5 B16c pattern). Once the arch is retired and "granite_speech" resolves
// through the ArchAdapter, the same comparison guards the adapter mapping.
//
// Cases (1b / 2b GGUF): jfk transcribe, german transcribe, german -> English
// translate (when the variant advertises translation), and jfk run twice on
// the same engine session (the cached-graph rule: a reused ggml graph that
// does not re-upload an input decodes garbage on the second run).
// Optional -plus GGUF: word-timestamp task and speaker-attribution task on a
// two-speaker mix (text + word rows + turn texts compared).
//
// Model-free checks of the ported "[T:N]" / "[Speaker N]:" parsers run first
// and fail the test on their own, so it can fail without a model.
//
// Usage: granite_speech_engine_arch_parity_test <granite.gguf> <samples_dir> [<granite-plus.gguf>]
// Exit 2 (SKIP) when the GGUF or the samples are absent (after the
// model-free checks pass).

#include "abi_test_wav.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/granite_speech/model.h"
#include "transcribe.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace engine::runtime;
namespace gs = engine::models::granite_speech;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

void check_parsers() {
  // Word timestamps: centisecond ends, 10 s rollover, "_" silence, trailing
  // word bounded by the audio duration.
  std::vector<gs::GraniteSpeechWord> words;
  const std::string clean =
      gs::parse_granite_speech_word_timestamps("hello [T:45] _ [T:60] world [T:982] again [T:10] tail", 25000, words);
  CHECK(clean == "hello world again tail");
  CHECK(words.size() == 4);
  if (words.size() == 4) {
    CHECK(words[0].t0_ms == 0 && words[0].t1_ms == 450);
    CHECK(words[1].t0_ms == 600 && words[1].t1_ms == 9820);
    CHECK(words[2].t1_ms == 10100);  // 10 < 982: one rollover
    CHECK(words[3].t0_ms == 10100 && words[3].t1_ms == 25000);
  }
  std::vector<gs::GraniteSpeechWord> none;
  CHECK(gs::parse_granite_speech_word_timestamps("no markers here", 1000, none) == "no markers here");
  CHECK(none.size() == 1);

  // Speaker turns: markers stripped, ids kept, [Speaker 0] not a marker.
  std::vector<gs::GraniteSpeechTurn> turns;
  std::string full;
  CHECK(gs::split_granite_speech_speaker_turns("[Speaker 1]: hi there [Speaker 2]: hello [Speaker 0] x", turns, full));
  CHECK(turns.size() == 2);
  if (turns.size() == 2) {
    CHECK(turns[0].speaker == 1 && turns[0].text == "hi there");
    CHECK(turns[1].speaker == 2 && turns[1].text == "hello [Speaker 0] x");
  }
  CHECK(full == "hi there hello [Speaker 0] x");
  CHECK(!gs::split_granite_speech_speaker_turns("plain text", turns, full));

  CHECK(gs::granite_speech_target_language_name("de") == "German");
  CHECK(gs::granite_speech_target_language_name("ZH") == "Mandarin");
  CHECK(gs::granite_speech_target_language_name("xx").empty());
}

struct Word {
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
  std::string text;
  bool operator==(const Word &o) const { return t0_ms == o.t0_ms && t1_ms == o.t1_ms && text == o.text; }
};

struct Outcome {
  std::string text;
  std::vector<Word> words;
  std::vector<std::string> turns;  // per-turn text (speaker attribution)
  bool truncated = false;
};

struct Case {
  const char *name;
  const char *wav;
  bool translate = false;
  bool words = false;
  bool diarize = false;
};

Outcome run_c_abi(transcribe_session *session, const std::vector<float> &pcm, const Case &c) {
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.task = c.translate ? TRANSCRIBE_TASK_TRANSLATE : TRANSCRIBE_TASK_TRANSCRIBE;
  rp.target_language = c.translate ? "en" : nullptr;
  rp.timestamps = c.words ? TRANSCRIBE_TIMESTAMPS_WORD : TRANSCRIBE_TIMESTAMPS_NONE;
  rp.diarize = c.diarize ? TRANSCRIBE_DIARIZE_MODE_ON : TRANSCRIBE_DIARIZE_MODE_DEFAULT;
  const transcribe_status st = transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
  if (st != TRANSCRIBE_OK && st != TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
    throw std::runtime_error(std::string("C ABI run failed: ") + transcribe_status_string(st));
  }
  Outcome out;
  out.text = transcribe_full_text(session);
  out.truncated = transcribe_was_truncated(session);
  if (c.words) {
    for (int i = 0; i < transcribe_n_words(session); ++i) {
      transcribe_word w;
      transcribe_word_init(&w);
      if (transcribe_get_word(session, i, &w) == TRANSCRIBE_OK) {
        out.words.push_back({w.t0_ms, w.t1_ms, w.text != nullptr ? w.text : ""});
      }
    }
  }
  if (c.diarize) {
    for (int i = 0; i < transcribe_n_segments(session); ++i) {
      transcribe_segment s;
      transcribe_segment_init(&s);
      if (transcribe_get_segment(session, i, &s) == TRANSCRIBE_OK) {
        out.turns.push_back(s.text != nullptr ? s.text : "");
      }
    }
  }
  return out;
}

Outcome run_engine(IOfflineVoiceTaskSession &session, const AudioBuffer &audio, const Case &c) {
  TaskRequest request;
  request.audio_input = audio;
  request.options["task"] = c.translate ? "translate" : "transcribe";
  if (c.translate) {
    request.options["target_language"] = "en";
  }
  request.options["timestamps"] = c.words ? "word" : "none";
  request.options["diarize"] = c.diarize ? "true" : "false";
  session.prepare(build_preparation_request(request));
  const TaskResult result = session.run(request);
  Outcome out;
  if (result.text_output.has_value()) {
    out.text = result.text_output->text;
  }
  for (const auto &w : result.word_timestamps) {
    // The adapter's samples -> ms conversion (16 kHz).
    out.words.push_back({w.span.start_sample * 1000 / 16000, w.span.end_sample * 1000 / 16000, w.word});
  }
  if (c.diarize) {
    for (const auto &seg : result.speech_segments) {
      out.turns.push_back(seg.text);
    }
  }
  out.truncated = result.truncated;
  return out;
}

void compare(const Case &c, const Outcome &abi, const Outcome &engine) {
  bool ok = true;
  const auto fail = [&](const std::string &what) {
    std::fprintf(stderr, "FAIL [%s] %s\n", c.name, what.c_str());
    ok = false;
    ++g_failures;
  };
  if (abi.text != engine.text) {
    fail("text\n    c abi : " + abi.text + "\n    engine: " + engine.text);
  }
  if (abi.words != engine.words) {
    fail("words: c abi " + std::to_string(abi.words.size()) + " vs engine " + std::to_string(engine.words.size()) +
         " (or different timing / text)");
  }
  if (abi.turns != engine.turns) {
    fail("speaker turns: c abi " + std::to_string(abi.turns.size()) + " vs engine " +
         std::to_string(engine.turns.size()));
  }
  if (abi.truncated != engine.truncated) {
    fail("truncated flag differs");
  }
  if (engine.text.empty()) {
    fail("empty transcript");
  }
  std::printf("[%s] %s  %s\n", c.name, ok ? "match" : "DIFFER", engine.text.c_str());
}

struct Pair {
  transcribe_model *abi_model = nullptr;
  transcribe_session *abi_session = nullptr;
  std::unique_ptr<ILoadedVoiceModel> engine_model;
  std::unique_ptr<IVoiceTaskSession> engine_session;
  ~Pair() {
    engine_session.reset();
    engine_model.reset();
    if (abi_session != nullptr) {
      transcribe_session_free(abi_session);
    }
    if (abi_model != nullptr) {
      transcribe_model_free(abi_model);
    }
  }
};

void open_pair(const std::filesystem::path &model_path, int threads, Pair &pair) {
  transcribe_model_load_params mp;
  transcribe_model_load_params_init(&mp);
  mp.backend = TRANSCRIBE_BACKEND_CPU;
  if (transcribe_model_load_file(model_path.string().c_str(), &mp, &pair.abi_model) != TRANSCRIBE_OK) {
    throw std::runtime_error("C ABI could not load " + model_path.string());
  }
  CHECK(std::string(transcribe_model_arch_string(pair.abi_model)) == "granite_speech");
  transcribe_session_params sp;
  transcribe_session_params_init(&sp);
  sp.n_threads = threads;
  if (transcribe_session_init(pair.abi_model, &sp, &pair.abi_session) != TRANSCRIBE_OK) {
    throw std::runtime_error("C ABI session init failed");
  }

  ModelRegistry registry;
  registry.register_loader(gs::make_granite_speech_loader());
  ModelLoadRequest request;
  request.model_path = model_path;
  request.family_hint = "granite_speech";
  pair.engine_model = registry.load(request);
  TaskSpec task;
  task.task = VoiceTaskKind::Asr;
  task.mode = RunMode::Offline;
  SessionOptions options;
  options.backend.threads = threads;
  pair.engine_session = pair.engine_model->create_task_session(task, options);
  if (dynamic_cast<IOfflineVoiceTaskSession *>(pair.engine_session.get()) == nullptr) {
    throw std::runtime_error("engine session is not offline");
  }
}

AudioBuffer load_audio(const std::filesystem::path &path) {
  int rate = 0;
  AudioBuffer audio;
  audio.samples = abi_test::read_wav_mono_f32(path.string(), rate);
  audio.sample_rate = rate;
  audio.channels = 1;
  if (rate != 16000) {
    throw std::runtime_error(path.string() + " is not 16 kHz");
  }
  return audio;
}

void run_cases(Pair &pair, const std::filesystem::path &samples, const std::vector<Case> &cases) {
  auto &engine = dynamic_cast<IOfflineVoiceTaskSession &>(*pair.engine_session);
  for (const auto &c : cases) {
    const AudioBuffer audio = load_audio(samples / c.wav);
    const Outcome abi = run_c_abi(pair.abi_session, audio.samples, c);
    const Outcome eng = run_engine(engine, audio, c);
    compare(c, abi, eng);
  }
}

}  // namespace

int main(int argc, char **argv) {
  check_parsers();
  if (g_failures != 0) {
    std::fprintf(stderr, "granite_speech_engine_arch_parity_test: %d model-free failure(s)\n", g_failures);
    return 1;
  }
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <granite.gguf> <samples_dir> [<granite-plus.gguf>]\n", argv[0]);
    return 1;
  }
  const std::filesystem::path model_path = argv[1];
  const std::filesystem::path samples = argv[2];
  if (!std::filesystem::exists(model_path) || !std::filesystem::exists(samples / "jfk.wav")) {
    std::printf("SKIP: model or samples absent (model-free checks passed)\n");
    return 2;
  }
  const int threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

  try {
    {
      Pair pair;
      open_pair(model_path, threads, pair);
      transcribe_capabilities caps;
      transcribe_capabilities_init(&caps);
      CHECK(transcribe_model_get_capabilities(pair.abi_model, &caps) == TRANSCRIBE_OK);
      std::vector<Case> cases = {
          {"jfk", "jfk.wav"},
          {"german", "german.wav"},
      };
      if (caps.supports_translate) {
        cases.push_back({"german-translate-en", "german.wav", true});
      }
      // Same clip twice on the same sessions: the second run must not differ.
      cases.push_back({"jfk-again", "jfk.wav"});
      run_cases(pair, samples, cases);
    }

    if (argc >= 4 && std::filesystem::exists(argv[3])) {
      Pair plus;
      open_pair(argv[3], threads, plus);
      const char *mix = std::filesystem::exists(samples / "multitalker-2spk-mix.wav") ? "multitalker-2spk-mix.wav"
                                                                                     : "jfk.wav";
      run_cases(plus, samples,
                {
                    {"plus-jfk", "jfk.wav"},
                    {"plus-jfk-words", "jfk.wav", false, true},
                    {"plus-mix-speakers", mix, false, false, true},
                });
    } else {
      std::printf("note: no -plus GGUF given; word-timestamp / speaker-attribution parity not exercised\n");
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "granite_speech_engine_arch_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("granite_speech_engine_arch_parity_test: engine == C ABI (arch) in every case\n");
  return 0;
}
