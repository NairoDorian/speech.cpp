// tests/unittests/test_sortformer_diar_streaming.cpp - Phase 10.5 (family 3,
// step 2) engine gate for the chunked Sortformer diarization scheduler and the
// NeMo GGUF package layout.
//
// The subject is the catalogue's DEFAULT package,
// nvidia/diar_streaming_sortformer_4spk-v2 (diar_streaming_sortformer_4spk-v2
// .q8_0.gguf). Before this step nothing in the repository could open it: the
// engine read the HF layout, the transcribe.cpp arch read its own converter's
// layout, and this file is NVIDIA's third one.
//
// What it gates, all through the public engine surface:
//
//   1. Loading: registry auto-detection (no family hint) routes the NeMo GGUF
//      to sortformer_diar; the alias "sortformer" does too.
//   2. The shipped streaming operating point runs CHUNKED by default and
//      diarizes the single-speaker LibriSpeech fixtures as one speaker.
//   3. The committed 2-speaker oracle mix (samples/sortformer-2spk-mix.wav,
//      tests/golden/sortformer/sortformer-2spk-mix.rttm) diarizes as two
//      speakers under every operating point, and the DER against the golden
//      RTTM stays under a measured bound.
//   4. Chunked == whole-window on a clip that fits one chunk: the graph split
//      (stem / body), the host concat and the masks reproduce the monolithic
//      graph's probabilities. This is the numerical parity leg of the port.
//   5. The `small` diagnostic preset forces several chunks and a speaker-cache
//      compression on the same clip and still separates both speakers.
//   6. Determinism, cancellation between chunks, and rejection of an unknown
//      preset.
//
// Skips (exit 2) without the pinned package, the fixtures or the oracle.

#include "abi_test_wav.h"

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/sortformer_diar/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace engine::runtime;
namespace sf = engine::models::sortformer_diar;

int popcount(uint32_t mask) {
    int count = 0;
    while (mask != 0) {
        count += static_cast<int>(mask & 1u);
        mask >>= 1;
    }
    return count;
}

// The oracle is a synthetic mix; its absolute DER is a property of the model,
// not of the port. The bar is parity with the transcribe.cpp arch on the same
// weights: its dumped diar.probs score 0.315-0.334 across the operating points
// (threshold 0.5, no collar, overlap counted), the engine 0.315-0.329. A
// scheduler regression that merges or drops a speaker lands far above this.
constexpr double kMaxOracleDer = 0.35;
constexpr double kMinSpeechCoverage = 0.5;
// F32 whole-window vs chunked: same ops, same weights, different padding.
constexpr float kMaxSingleChunkAbsDiff = 1.0e-3f;

AudioBuffer read_audio(const std::filesystem::path & wav_path) {
    int sample_rate = 0;
    AudioBuffer audio;
    audio.samples = abi_test::read_wav_mono_f32(wav_path.string(), sample_rate);
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    return audio;
}

std::vector<std::filesystem::path> load_fixtures(const std::filesystem::path & dir) {
    std::vector<std::filesystem::path> wavs;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".wav") {
            wavs.push_back(entry.path());
        }
    }
    std::sort(wavs.begin(), wavs.end());
    return wavs;
}

struct RttmTurn {
    double start = 0.0;
    double end = 0.0;
    std::string speaker;
};

std::vector<RttmTurn> read_rttm(const std::filesystem::path & path) {
    std::ifstream in(path);
    std::vector<RttmTurn> turns;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string type, file, channel, start, duration, na1, na2, speaker;
        if (!(fields >> type >> file >> channel >> start >> duration >> na1 >> na2 >> speaker)) {
            continue;
        }
        if (type != "SPEAKER") {
            continue;
        }
        RttmTurn turn;
        turn.start = std::stod(start);
        turn.end = turn.start + std::stod(duration);
        turn.speaker = speaker;
        turns.push_back(turn);
    }
    return turns;
}

// Diarization error rate on a 10 ms grid, no collar, overlap counted, with
// the best one-to-one mapping of hypothesis labels onto reference labels
// (a hypothesis label may also map to "no reference speaker").
double diarization_error_rate(
    const std::vector<RttmTurn> & reference,
    const std::vector<SpeakerTurn> & hypothesis,
    double audio_seconds) {
    constexpr double kStep = 0.01;
    const int64_t n_frames = static_cast<int64_t>(std::ceil(audio_seconds / kStep));
    std::vector<std::string> ref_labels;
    for (const auto & turn : reference) {
        if (std::find(ref_labels.begin(), ref_labels.end(), turn.speaker) == ref_labels.end()) {
            ref_labels.push_back(turn.speaker);
        }
    }
    std::vector<std::string> hyp_labels;
    for (const auto & turn : hypothesis) {
        if (std::find(hyp_labels.begin(), hyp_labels.end(), turn.speaker_id) == hyp_labels.end()) {
            hyp_labels.push_back(turn.speaker_id);
        }
    }
    const size_t R = ref_labels.size();
    const size_t H = hyp_labels.size();
    // ref_active[t] and hyp_active[t] are bitmasks over labels.
    std::vector<uint32_t> ref_active(static_cast<size_t>(n_frames), 0);
    std::vector<uint32_t> hyp_active(static_cast<size_t>(n_frames), 0);
    for (const auto & turn : reference) {
        const size_t r = static_cast<size_t>(std::find(ref_labels.begin(), ref_labels.end(), turn.speaker) - ref_labels.begin());
        const int64_t t0 = static_cast<int64_t>(std::llround(turn.start / kStep));
        const int64_t t1 = static_cast<int64_t>(std::llround(turn.end / kStep));
        for (int64_t t = std::max<int64_t>(0, t0); t < std::min(n_frames, t1); ++t) {
            ref_active[static_cast<size_t>(t)] |= 1u << r;
        }
    }
    for (const auto & turn : hypothesis) {
        const size_t h = static_cast<size_t>(std::find(hyp_labels.begin(), hyp_labels.end(), turn.speaker_id) - hyp_labels.begin());
        const int64_t t0 = static_cast<int64_t>(std::llround(static_cast<double>(turn.span.start_sample) / 16000.0 / kStep));
        const int64_t t1 = static_cast<int64_t>(std::llround(static_cast<double>(turn.span.end_sample) / 16000.0 / kStep));
        for (int64_t t = std::max<int64_t>(0, t0); t < std::min(n_frames, t1); ++t) {
            hyp_active[static_cast<size_t>(t)] |= 1u << h;
        }
    }
    double ref_total = 0.0;
    for (const uint32_t mask : ref_active) {
        ref_total += static_cast<double>(popcount(mask));
    }
    if (ref_total <= 0.0) {
        return 1.0;
    }
    // Enumerate mappings hyp label -> ref label or none (value R), injective
    // on ref labels.
    std::vector<size_t> mapping(H, R);
    double best_error = 1.0e300;
    std::vector<bool> used(R + 1, false);
    const std::function<void(size_t)> search = [&](size_t h) {
        if (h == H) {
            double error = 0.0;
            for (int64_t t = 0; t < n_frames; ++t) {
                const uint32_t ref = ref_active[static_cast<size_t>(t)];
                uint32_t mapped = 0;
                for (size_t k = 0; k < H; ++k) {
                    if ((hyp_active[static_cast<size_t>(t)] & (1u << k)) != 0 && mapping[k] < R) {
                        mapped |= 1u << mapping[k];
                    }
                }
                const int n_ref = popcount(ref);
                const int n_hyp = popcount(hyp_active[static_cast<size_t>(t)]);
                const int n_correct = popcount(ref & mapped);
                // NIST md-eval accounting per frame: miss + false alarm +
                // confusion, with overlap counted per speaker.
                const int miss = std::max(0, n_ref - n_hyp);
                const int fa = std::max(0, n_hyp - n_ref);
                const int conf = std::min(n_ref, n_hyp) - n_correct;
                error += static_cast<double>(miss + fa + conf);
            }
            best_error = std::min(best_error, error);
            return;
        }
        for (size_t r = 0; r <= R; ++r) {
            if (r < R && used[r]) {
                continue;
            }
            mapping[h] = r;
            if (r < R) used[r] = true;
            search(h + 1);
            if (r < R) used[r] = false;
        }
    };
    search(0);
    return best_error / ref_total;
}

size_t count_speakers(const std::vector<SpeakerTurn> & turns) {
    std::set<std::string> ids;
    for (const auto & turn : turns) {
        ids.insert(turn.speaker_id);
    }
    return ids.size();
}

double coverage_seconds(const std::vector<SpeakerTurn> & turns) {
    double covered = 0.0;
    for (const auto & turn : turns) {
        covered += static_cast<double>(turn.span.end_sample - turn.span.start_sample) / 16000.0;
    }
    return covered;
}

bool ordered(const std::vector<SpeakerTurn> & turns) {
    double previous = -1.0;
    for (const auto & turn : turns) {
        if (turn.span.end_sample < turn.span.start_sample) return false;
        if (static_cast<double>(turn.span.start_sample) < previous) return false;
        previous = static_cast<double>(turn.span.start_sample);
    }
    return true;
}

struct RunOutcome {
    TaskResult result;
    std::vector<float> probabilities;
    int64_t frames = 0;
    sf::SortformerRunPlan plan;
    double seconds = 0.0;
};

RunOutcome run_once(IVoiceTaskSession & session, const AudioBuffer & audio,
                    const std::unordered_map<std::string, std::string> & options) {
    auto * offline = dynamic_cast<IOfflineVoiceTaskSession *>(&session);
    auto * sortformer = dynamic_cast<sf::SortformerDiarSession *>(&session);
    if (offline == nullptr || sortformer == nullptr) {
        throw std::runtime_error("session is not a SortformerDiarSession");
    }
    offline->prepare(build_preparation_request(audio));
    TaskRequest request;
    request.audio_input = audio;
    request.options = options;
    const auto started = std::chrono::steady_clock::now();
    RunOutcome outcome;
    outcome.result = offline->run(request);
    outcome.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    outcome.probabilities = sortformer->last_probabilities();
    outcome.frames = sortformer->last_probability_frames();
    outcome.plan = sortformer->last_run_plan();
    return outcome;
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

}  // namespace

// Optional diagnostic: write the raw per-frame probabilities of every oracle
// run as <dir>/probs_<preset>.f32 (row-major [frames, n_spk]) so an external
// tool can compare them against another implementation's dump.
void dump_probabilities(const std::filesystem::path & dir, const std::string & label,
                        const std::vector<float> & probabilities) {
    if (dir.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::ofstream out(dir / ("probs_" + label + ".f32"), std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(probabilities.data()),
              static_cast<std::streamsize>(probabilities.size() * sizeof(float)));
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::cerr << "usage: sortformer_diar_streaming_engine_test <sortformer-v2.gguf> "
                     "<librispeech_fixture_dir> <oracle.wav> <oracle.rttm> [--dump-probs <dir>]\n";
        return 1;
    }
    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path fixture_dir = argv[2];
    const std::filesystem::path oracle_wav = argv[3];
    const std::filesystem::path oracle_rttm = argv[4];
    std::filesystem::path dump_dir;
    std::string weight_type;
    for (int i = 5; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--dump-probs") {
            dump_dir = argv[i + 1];
        } else if (flag == "--weight-type") {
            // Diagnostic: sortformer_diar.weight_type for every session this
            // test creates (e.g. "native" keeps the package's Q8_0 matmuls).
            weight_type = argv[i + 1];
        }
    }
    std::error_code ec;
    if (!std::filesystem::exists(model_path, ec)) {
        std::cout << "SKIP: pinned Sortformer v2 GGUF not found: " << model_path << "\n";
        return 2;
    }
    if (!std::filesystem::is_directory(fixture_dir, ec)) {
        std::cout << "SKIP: librispeech fixture dir not found: " << fixture_dir << "\n";
        return 2;
    }
    if (!std::filesystem::exists(oracle_wav, ec) || !std::filesystem::exists(oracle_rttm, ec)) {
        std::cout << "SKIP: oracle mix or RTTM not found\n";
        return 2;
    }

    try {
        ModelRegistry registry;
        registry.register_loader(sf::make_sortformer_diar_loader());

        // 1. Loading. No family hint: the file names no audiocpp family, so
        // the registry must reach this loader through can_load() alone.
        ModelLoadRequest request;
        request.model_path = model_path;
        auto loaded = registry.load(request);
        if (!loaded) {
            std::cerr << "FAIL: registry.load returned null\n";
            return 1;
        }
        if (loaded->metadata().family != "sortformer_diar") {
            std::cerr << "FAIL: NeMo GGUF loaded as family " << loaded->metadata().family << "\n";
            return 1;
        }
        {
            ModelLoadRequest aliased;
            aliased.model_path = model_path;
            aliased.family_hint = "sortformer";
            auto via_alias = registry.load(aliased);
            if (!via_alias || via_alias->metadata().family != "sortformer_diar") {
                std::cerr << "FAIL: alias 'sortformer' did not resolve to sortformer_diar\n";
                return 1;
            }
        }
        std::cout << "  loaded family: " << loaded->metadata().family << "\n";

        TaskSpec task;
        task.task = VoiceTaskKind::Diarization;
        task.mode = RunMode::Offline;
        SessionOptions options;
        if (!weight_type.empty()) {
            options.options["sortformer_diar.weight_type"] = weight_type;
        }
        auto session = loaded->create_task_session(task, options);
        if (dynamic_cast<sf::SortformerDiarSession *>(session.get()) == nullptr) {
            std::cerr << "FAIL: session is not a SortformerDiarSession\n";
            return 1;
        }

        // 2. Shipped operating point, chunked by default, single-speaker
        // fixtures.
        for (const auto & wav : load_fixtures(fixture_dir)) {
            const auto audio = read_audio(wav);
            const double audio_s = static_cast<double>(audio.samples.size()) / audio.sample_rate;
            const auto outcome = run_once(*session, audio, {});
            const auto & turns = outcome.result.speaker_turns;
            const double coverage = audio_s > 0.0 ? coverage_seconds(turns) / audio_s : 0.0;
            std::cout << "  " << wav.filename().string() << ": " << turns.size() << " turn(s), "
                      << count_speakers(turns) << " speaker(s), coverage " << coverage << ", plan "
                      << (outcome.plan.mode == sf::SortformerRunMode::Chunked ? "chunked" : "whole-window") << " ("
                      << outcome.plan.source << "), " << outcome.seconds << " s for " << audio_s << " s\n";
            if (outcome.plan.mode != sf::SortformerRunMode::Chunked || outcome.plan.source != "default") {
                std::cerr << "FAIL: a package that ships a streaming operating point must run chunked by default\n";
                return 1;
            }
            if (turns.empty() || !ordered(turns)) {
                std::cerr << "FAIL: no usable speaker turns for " << wav.filename() << "\n";
                return 1;
            }
            if (count_speakers(turns) != 1) {
                std::cerr << "FAIL: single-speaker fixture diarized as " << count_speakers(turns) << " speakers\n";
                return 1;
            }
            if (coverage < kMinSpeechCoverage) {
                std::cerr << "FAIL: speaker turns cover only " << coverage << " of the audio\n";
                return 1;
            }
        }

        // 3. The 2-speaker oracle under every operating point.
        const auto oracle = read_audio(oracle_wav);
        const double oracle_s = static_cast<double>(oracle.samples.size()) / oracle.sample_rate;
        const auto reference = read_rttm(oracle_rttm);
        if (reference.empty()) {
            std::cerr << "FAIL: empty oracle RTTM\n";
            return 1;
        }
        RunOutcome default_outcome;
        for (const char * preset : {"default", "very_high_latency", "high_latency", "low_latency", "small"}) {
            std::unordered_map<std::string, std::string> request_options;
            request_options["stream_preset"] = preset;
            const auto outcome = run_once(*session, oracle, request_options);
            dump_probabilities(dump_dir, preset, outcome.probabilities);
            const auto & turns = outcome.result.speaker_turns;
            const double der = diarization_error_rate(reference, turns, oracle_s);
            std::cout << "  oracle preset=" << preset << ": " << turns.size() << " turn(s), "
                      << count_speakers(turns) << " speaker(s), DER " << der << ", chunk_len "
                      << outcome.plan.params.chunk_len << ", rc " << outcome.plan.params.chunk_right_context
                      << ", fifo " << outcome.plan.params.fifo_len << ", spkcache "
                      << outcome.plan.params.spkcache_len << ", " << outcome.seconds << " s\n";
            if (outcome.plan.mode != sf::SortformerRunMode::Chunked) {
                std::cerr << "FAIL: preset " << preset << " did not run chunked\n";
                return 1;
            }
            if (!ordered(turns)) {
                std::cerr << "FAIL: turns not ordered under preset " << preset << "\n";
                return 1;
            }
            if (count_speakers(turns) != 2) {
                std::cerr << "FAIL: oracle mix diarized as " << count_speakers(turns) << " speakers under preset "
                          << preset << " (expected 2)\n";
                return 1;
            }
            if (der > kMaxOracleDer) {
                std::cerr << "FAIL: DER " << der << " exceeds " << kMaxOracleDer << " under preset " << preset << "\n";
                return 1;
            }
            if (std::string(preset) == "default") {
                default_outcome = outcome;
            }
        }

        // 4. Chunked == whole-window on a clip that fits one chunk.
        {
            std::unordered_map<std::string, std::string> request_options;
            request_options["stream_preset"] = "offline";
            const auto whole = run_once(*session, oracle, request_options);
            if (whole.plan.mode != sf::SortformerRunMode::WholeWindow) {
                std::cerr << "FAIL: stream_preset=offline did not run whole-window\n";
                return 1;
            }
            if (whole.frames != default_outcome.frames) {
                std::cerr << "FAIL: whole-window produced " << whole.frames << " frames, chunked "
                          << default_outcome.frames << "\n";
                return 1;
            }
            dump_probabilities(dump_dir, "offline", whole.probabilities);
            const float diff = max_abs_diff(whole.probabilities, default_outcome.probabilities);
            const double der = diarization_error_rate(reference, whole.result.speaker_turns, oracle_s);
            std::cout << "  whole-window vs chunked (single chunk): max |dp| = " << diff << " over "
                      << whole.frames << " frames; whole-window DER " << der << "\n";
            if (diff > kMaxSingleChunkAbsDiff) {
                std::cerr << "FAIL: chunked probabilities diverge from the whole-window graph by " << diff << "\n";
                return 1;
            }
        }

        // 5. Determinism.
        {
            const auto again = run_once(*session, oracle, {});
            if (max_abs_diff(again.probabilities, default_outcome.probabilities) != 0.0f) {
                std::cerr << "FAIL: chunked diarization is not deterministic across runs\n";
                return 1;
            }
        }

        // 6a. Cancellation between chunks: decline progress once the second
        // chunk starts; the run must unwind with ProgressCanceled.
        {
            auto abort_session = loaded->create_task_session(task, options);
            auto * abort_offline = dynamic_cast<IOfflineVoiceTaskSession *>(abort_session.get());
            abort_offline->prepare(build_preparation_request(oracle));
            int64_t chunks_seen = 0;
            abort_session->set_progress_callback([&](const ProgressInfo & info) {
                ++chunks_seen;
                return info.completed_units < 1;
            });
            TaskRequest rerun;
            rerun.audio_input = oracle;
            rerun.options["stream_preset"] = "low_latency";
            bool canceled = false;
            try {
                (void)abort_offline->run(rerun);
            } catch (const ProgressCanceled &) {
                canceled = true;
            }
            if (!canceled || chunks_seen < 2) {
                std::cerr << "FAIL: declining progress after the first chunk did not cancel the run\n";
                return 1;
            }
        }
        // 6b. request_abort() before run().
        {
            auto abort_session = loaded->create_task_session(task, options);
            auto * abort_offline = dynamic_cast<IOfflineVoiceTaskSession *>(abort_session.get());
            abort_offline->prepare(build_preparation_request(oracle));
            abort_session->request_abort();
            bool canceled = false;
            try {
                TaskRequest rerun;
                rerun.audio_input = oracle;
                (void)abort_offline->run(rerun);
            } catch (const ProgressCanceled &) {
                canceled = true;
            }
            if (!canceled) {
                std::cerr << "FAIL: request_abort was not honored by run()\n";
                return 1;
            }
        }
        // 6c. An unknown preset is refused, and a geometry override runs.
        {
            bool rejected = false;
            try {
                (void)run_once(*session, oracle, {{"stream_preset", "bogus"}});
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            if (!rejected) {
                std::cerr << "FAIL: unknown stream_preset was accepted\n";
                return 1;
            }
            const auto custom = run_once(*session, oracle,
                                         {{"stream_preset", "high_latency"}, {"stream_chunk_len", "50"}});
            if (custom.plan.params.chunk_len != 50 || custom.plan.source.find("+fields") == std::string::npos) {
                std::cerr << "FAIL: stream_chunk_len override was not applied\n";
                return 1;
            }
            if (count_speakers(custom.result.speaker_turns) != 2) {
                std::cerr << "FAIL: custom geometry lost a speaker\n";
                return 1;
            }
        }

        std::cout << "sortformer_diar_streaming_engine_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
