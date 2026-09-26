// test_granite_nar_engine_arch_parity.cpp - Granite Speech NAR through the
// public C ABI (transcribe_model_load_file, which routes a granite_speech_nar
// GGUF to the transcribe.cpp arch src/runtime/arch/granite_nar while it
// exists) against the engine `granite_nar` package called directly through
// ModelRegistry, in one process, on the same GGUF and audio.
//
// Retirement evidence for the arch (Phase 11b pattern, cf.
// canary_qwen_engine_arch_parity_test): on the four LibriSpeech fixtures in
// assets/asr_validation/librispeech the transcripts must be IDENTICAL, byte
// for byte, single-shot and batched (transcribe_run_batch - the arch has no
// run_batch, so the C ABI runs it serially - vs the engine run_batch).
//
// Two engine sessions are compared:
//   - default (Shaw skew bias, transcribe.cpp 585b98f7 + bounded BPE-CTC,
//     b174a427): the parent's current code, which speech.cpp's arch has not
//     adopted. Same math as the arch, different op order - a transcript flip
//     here (not in the direct session) is float drift from the adoption, to
//     be judged on the WER gate, not a port bug.
//   - granite_nar.shaw_bias=direct: the arch's own Shaw form; the only
//     remaining difference is the pooled-hidden BPE projection.
// The default session is also run twice on one clip (graphs are rebuilt and
// every input re-uploaded; the result must repeat exactly).
//
// Model-free checks run first and always (pure host functions: the skew row
// table must equal the direct lookup for every (query, key), insertion slots,
// CTC collapse, argmax collapse).
//
// Usage: granite_nar_engine_arch_parity_test <granite-speech-4.1-2b-nar.gguf> [fixtures_dir]
// Exit 2 (SKIP) when the model path is absent / missing, or the fixtures are.

#include "abi_test_wav.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/granite_nar/decoding.h"
#include "engine/models/granite_nar/session.h"
#include "transcribe.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace engine::runtime;
namespace gn = engine::models::granite_nar;

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

// ---------------------------------------------------------------------------
// Model-free host checks
// ---------------------------------------------------------------------------

void check_host_functions() {
  // Skew rows == direct lookup: dists[q * ctx + k] (c = query, r = key in
  // the arch's naming) must be the row pos_rows[k - q + ctx - 1].
  for (const int ctx : {1, 7, 200}) {
    const int max_pos = 512;
    const auto rows = gn::precompute_pos_rows(ctx, max_pos);
    const auto dists = gn::precompute_attention_dists(ctx, max_pos);
    CHECK(rows.size() == static_cast<size_t>(2 * ctx - 1));
    bool same = true;
    for (int q = 0; q < ctx && same; ++q) {
      for (int k = 0; k < ctx; ++k) {
        if (dists[static_cast<size_t>(q) * ctx + k] != rows[static_cast<size_t>(k - q + ctx - 1)]) {
          same = false;
          break;
        }
      }
    }
    CHECK(same);
  }

  // Insertion slots: eos around and between, floor 8.
  {
    std::vector<int32_t> out;
    gn::add_insertion_slots({5, 6, 7}, 0, out);
    CHECK((out == std::vector<int32_t>{0, 5, 0, 6, 0, 7, 0, 0}));
    gn::add_insertion_slots({1, 2, 3, 4, 5}, 9, out);
    CHECK((out == std::vector<int32_t>{9, 1, 9, 2, 9, 3, 9, 4, 9, 5, 9}));
  }

  // BPE-CTC collapse, both blank schemes; `prev` carries across chunks and an
  // invalid (all-blank) window counts as blank.
  {
    std::vector<int32_t> out;
    int prev = -1;
    const int32_t a[] = {3, 3, 0, 3, 4};
    const uint8_t v[] = {1, 1, 1, 1, 0};
    gn::collapse_bpe_ctc(a, v, 5, /*blank_id=*/0, prev, out);
    CHECK((out == std::vector<int32_t>{2, 2}));
    const int32_t b[] = {3, 5};
    const uint8_t vb[] = {1, 1};
    gn::collapse_bpe_ctc(b, vb, 2, 0, prev, out); // prev is blank(0) -> 3 re-emits
    CHECK((out == std::vector<int32_t>{2, 2, 2, 4}));
    out.clear();
    prev = -1;
    const int32_t c[] = {100257, 7, 7, 100257, 7};
    const uint8_t vc[] = {1, 1, 1, 1, 1};
    gn::collapse_bpe_ctc(c, vc, 5, /*blank_id=*/100257, prev, out);
    CHECK((out == std::vector<int32_t>{7, 7}));
  }

  // Editor collapse: eos dropped but still `prev`; first maximum wins.
  {
    const int vocab = 3;
    const std::vector<float> logits = {
        0, 5, 1,  // 1
        0, 5, 1,  // 1 (repeat)
        9, 0, 0,  // eos 0
        0, 5, 5,  // 1 (first max), after eos -> emitted
    };
    std::vector<int32_t> ids;
    gn::argmax_collapse_drop_eos(logits, vocab, 4, /*eos_id=*/0, ids);
    CHECK((ids == std::vector<int32_t>{1, 1}));
  }

  // Frame stacking: odd trailing frame dropped, [m(2t) | m(2t+1)].
  {
    // n_mels 2, n_frames 5: mel-major rows {0..4}, {10..14}.
    const std::vector<float> mel = {0, 1, 2, 3, 4, 10, 11, 12, 13, 14};
    std::vector<float> out;
    const int t = gn::stack_mel_frames(mel, 2, 5, out);
    CHECK(t == 2);
    CHECK((out == std::vector<float>{0, 10, 1, 11, 2, 12, 3, 13}));
  }
}

// ---------------------------------------------------------------------------
// C ABI vs engine
// ---------------------------------------------------------------------------

void compare(const std::string &name, const std::string &abi, const std::string &engine) {
  const bool ok = abi == engine;
  if (!ok) {
    std::fprintf(stderr, "FAIL [%s] transcripts differ\n    c abi (arch): \"%s\"\n    engine      : \"%s\"\n",
                 name.c_str(), abi.c_str(), engine.c_str());
    ++g_failures;
  }
  std::printf("[%s] %s\n    %s\n", name.c_str(), ok ? "match" : "DIFFER", engine.c_str());
}

std::string run_c_abi(transcribe_session *session, const std::vector<float> &pcm) {
  transcribe_run_params rp;
  transcribe_run_params_init(&rp);
  rp.language = "en";
  const transcribe_status st = transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
  if (st != TRANSCRIBE_OK) {
    throw std::runtime_error(std::string("C ABI run failed: ") + transcribe_status_string(st));
  }
  CHECK(!transcribe_was_truncated(session));
  return transcribe_full_text(session);
}

std::string run_engine(IOfflineVoiceTaskSession &session, const AudioBuffer &audio) {
  session.prepare(build_preparation_request(audio));
  TaskRequest request;
  request.audio_input = audio;
  request.options["language"] = "en";
  const TaskResult result = session.run(request);
  CHECK(!result.truncated);
  if (result.raw_text.has_value() && result.text_output.has_value()) {
    CHECK(*result.raw_text == result.text_output->text); // the arch's raw == full
  }
  return result.text_output.has_value() ? result.text_output->text : std::string();
}

void check_capabilities(const transcribe_model *model) {
  CHECK(std::string(transcribe_model_arch_string(model)) == "granite_speech_nar");
  transcribe_capabilities caps;
  transcribe_capabilities_init(&caps);
  CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
  CHECK(caps.max_timestamp_kind == TRANSCRIBE_TIMESTAMPS_NONE);
  CHECK(!caps.supports_translate);
  CHECK(!caps.supports_streaming);
  CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION));
}

void check_engine_capabilities(const ILoadedVoiceModel &model, const transcribe_model *abi_model) {
  const CapabilitySet &caps = model.capabilities();
  CHECK(model.metadata().family == "granite_nar");
  CHECK(!caps.supports_timestamps);
  CHECK(!caps.supports_translate);
  CHECK(caps.supports_cancellation);
  CHECK(std::find(caps.languages.begin(), caps.languages.end(), "en") != caps.languages.end());
  transcribe_capabilities abi_caps;
  transcribe_capabilities_init(&abi_caps);
  if (transcribe_model_get_capabilities(abi_model, &abi_caps) == TRANSCRIBE_OK) {
    if (caps.max_audio_ms != abi_caps.max_audio_ms) {
      std::fprintf(stderr, "FAIL max_audio_ms: c abi %lld vs engine %lld\n",
                   static_cast<long long>(abi_caps.max_audio_ms),
                   static_cast<long long>(caps.max_audio_ms));
      ++g_failures;
    }
  }
}

// Text -> ids through the arch's transcribe::Tokenizer (transcribe_tokenize)
// and the engine's TokenizerHub (the GGUF's pre is "granite").
void check_tokenizer(const transcribe_model *abi_model, const ILoadedVoiceModel &engine_model) {
  const char *const texts[] = {
      "He hoped there would be stew for dinner, turnips and carrots.",
      "Don't cry, he said. I was obliged to come.",
      "  numbers 1234 and émigré ",
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
      std::fprintf(stderr, "FAIL [tokenizer] ids differ for \"%s\"\n", text);
      ++g_failures;
    }
  }
}

std::unique_ptr<IVoiceTaskSession> make_session(const ILoadedVoiceModel &loaded, int threads,
                                                const char *shaw_bias) {
  TaskSpec task;
  task.task = VoiceTaskKind::Asr;
  task.mode = RunMode::Offline;
  SessionOptions options;
  options.backend.type = engine::core::BackendType::Cpu;
  options.backend.threads = threads;
  if (shaw_bias != nullptr) {
    options.options["granite_nar.shaw_bias"] = shaw_bias;
  }
  return loaded.create_task_session(task, options);
}

} // namespace

int main(int argc, char **argv) {
  check_host_functions();
  if (g_failures != 0) {
    std::fprintf(stderr, "granite_nar_engine_arch_parity_test: %d host-function failure(s)\n",
                 g_failures);
    return 1;
  }

  if (argc < 2 || argv[1] == nullptr || argv[1][0] == '\0') {
    std::printf("SKIP: usage: %s <granite-speech-4.1-2b-nar.gguf> [fixtures_dir]\n", argv[0]);
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

    ModelRegistry registry;
    registry.register_loader(gn::make_granite_nar_loader());
    ModelLoadRequest request;
    request.model_path = model_path;
    request.family_hint = "granite_nar";
    auto loaded = registry.load(request);
    check_engine_capabilities(*loaded, abi_model);
    check_tokenizer(abi_model, *loaded);

    auto skew_holder = make_session(*loaded, threads, nullptr);
    auto direct_holder = make_session(*loaded, threads, "direct");
    auto *skew = dynamic_cast<IOfflineVoiceTaskSession *>(skew_holder.get());
    auto *direct = dynamic_cast<IOfflineVoiceTaskSession *>(direct_holder.get());
    if (skew == nullptr || direct == nullptr) {
      std::fprintf(stderr, "FAIL: engine session is not offline\n");
      return 1;
    }

    std::vector<AudioBuffer> audios;
    std::vector<std::string> abi_single;
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
      const std::string abi = run_c_abi(abi_session, audio.samples);
      compare(std::string("direct/") + wav, abi, run_engine(*direct, audio));
      compare(std::string("skew/") + wav, abi, run_engine(*skew, audio));
      audios.push_back(audio);
      abi_single.push_back(abi);
    }

    if (!audios.empty()) {
      const std::string again = run_engine(*skew, audios.front());
      if (again != abi_single.front()) {
        std::fprintf(stderr, "FAIL [rerun] engine second run differs\n    first : \"%s\"\n    second: \"%s\"\n",
                     abi_single.front().c_str(), again.c_str());
        ++g_failures;
      }
    }

    // Batched: transcribe_run_batch (serial arch runs) vs engine run_batch.
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
      skew->prepare(build_preparation_request(audios.front()));
      const std::vector<TaskResult> results = skew->run_batch(requests);
      CHECK(results.size() == audios.size());
      for (size_t i = 0; i < audios.size() && i < results.size(); ++i) {
        CHECK(transcribe_batch_status(abi_session, static_cast<int>(i)) == TRANSCRIBE_OK);
        CHECK(results[i].item_status == TaskItemStatus::Ok);
        const char *text = transcribe_batch_full_text(abi_session, static_cast<int>(i));
        compare(std::string("batch/") + kFixtures[i], text != nullptr ? text : "",
                results[i].text_output.has_value() ? results[i].text_output->text : std::string());
      }
    }

    // Input gate: n_ctx 16 leaves no room for any utterance's audio tokens;
    // both sides must reject (the C ABI with INPUT_TOO_LONG, the engine
    // run_batch with an InputTooLong item).
    if (!audios.empty()) {
      transcribe_session_params sp_small;
      transcribe_session_params_init(&sp_small);
      sp_small.n_threads = threads;
      sp_small.n_ctx = 16;
      transcribe_session *abi_small = nullptr;
      if (transcribe_session_init(abi_model, &sp_small, &abi_small) == TRANSCRIBE_OK) {
        transcribe_run_params rp;
        transcribe_run_params_init(&rp);
        const transcribe_status st = transcribe_run(abi_small, audios.front().samples.data(),
                                                    static_cast<int>(audios.front().samples.size()), &rp);
        CHECK(st == TRANSCRIBE_ERR_INPUT_TOO_LONG);
        transcribe_session_free(abi_small);
      }
      TaskSpec task;
      task.task = VoiceTaskKind::Asr;
      task.mode = RunMode::Offline;
      SessionOptions options;
      options.backend.type = engine::core::BackendType::Cpu;
      options.backend.threads = threads;
      options.options["n_ctx"] = "16";
      auto small_holder = loaded->create_task_session(task, options);
      auto *small = dynamic_cast<IOfflineVoiceTaskSession *>(small_holder.get());
      CHECK(small != nullptr);
      if (small != nullptr) {
        small->prepare(build_preparation_request(audios.front()));
        TaskRequest r;
        r.audio_input = audios.front();
        const auto res = small->run_batch({r});
        CHECK(res.size() == 1 && res[0].item_status == TaskItemStatus::InputTooLong);
      }
    }

    transcribe_session_free(abi_session);
    transcribe_model_free(abi_model);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "granite_nar_engine_arch_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("granite_nar_engine_arch_parity_test: C ABI (arch) == engine (skew and direct) on "
              "every fixture, single-shot and batched\n");
  return 0;
}
