// tests/abi_stream_hello.cpp - streaming counterpart to abi_bridge_hello.
//
// abi_bridge_hello proves the offline surface of the C ABI bridge
// (open -> run -> full_text). This proves the streaming surface, which is a
// materially harder contract because the adapter has to reconcile two
// incompatible granularity rules:
//
//   - transcribe_stream_feed() accepts ANY n_samples, and
//   - framework streaming sessions declare a fixed chunk size via
//     streaming_policy() and enforce it hard (SileroRuntime::process_chunk
//     throws on both a wrong size AND a non-contiguous start_sample).
//
// StreamChunker in transcribe-arch-adapter.cpp bridges those. This test drives
// it with feed sizes chosen to never align with any plausible chunk size, so a
// broken re-blocker throws or silently desynchronizes the absolute sample
// cursor instead of quietly passing.
//
// What is asserted:
//   1. State machine: IDLE -> ACTIVE -> FINISHED, and reset returns to IDLE.
//   2. Cursor algebra, after every single feed:
//        input_received_ms is monotonic and tracks total audio fed
//        audio_committed_ms <= input_received_ms
//        buffered_ms == input_received_ms - audio_committed_ms
//      and after finalize, committed == input (nothing stranded in the tail).
//   3. Segment folding: the finalized stream result is readable through the
//      normal accessors.
//   4. Streaming/offline agreement: the same audio through transcribe_run()
//      and through the stream must agree on total detected speech duration.
//      This is the check that actually catches a desynchronized re-blocker --
//      cursor arithmetic alone can look right while the audio handed to the
//      model is shifted or duplicated.
//   5. A second begin/feed/finalize cycle on the same session works, so the
//      per-utterance state is really cleared.
//
// Usage:
//   abi_stream_hello <model> [audio.wav]
// or set TRANSCRIBE_BRIDGE_TEST_MODEL / TRANSCRIBE_BRIDGE_TEST_AUDIO.
// Without a model path it exits 2 (skipped), like abi_bridge_hello, so it is
// safe under CTest without shipping model assets. A model that reports
// supports_streaming == false also exits 2 rather than failing: not every
// family has a streaming surface, and that is not this test's business.

// Deliberately links ONLY the public C ABI — no engine_runtime, no ggml — so
// it exercises the same surface a language binding sees, and so the shipped
// shared library is proven self-sufficient for a streaming consumer. That is
// why the WAV parsing below is local rather than reusing
// engine/framework/audio/wav_reader.h.

#include "transcribe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi          = 3.14159265358979323846264338327950288;
constexpr int    kSampleRate  = 16000;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string env_or_arg(int argc, char ** argv, int index, const char * env_name) {
    if (argc > index && argv[index] != nullptr && argv[index][0] != '\0') {
        return argv[index];
    }
    const char * env = std::getenv(env_name);
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return std::string();
}

// Fallback audio when no WAV is supplied: alternating tone bursts and silence.
// A VAD will not necessarily call the tone "speech", which is fine — the
// cursor and re-blocking invariants hold regardless of what the model decides,
// and the streaming/offline agreement check compares the two paths against
// each other rather than against a fixed expectation.
std::vector<float> make_burst_pcm(int n_bursts, double burst_s, double gap_s) {
    std::vector<float> pcm;
    for (int b = 0; b < n_bursts; ++b) {
        const int burst_n = static_cast<int>(kSampleRate * burst_s);
        for (int i = 0; i < burst_n; ++i) {
            const double t = static_cast<double>(i) / kSampleRate;
            // Two partials so it is not a single pure tone.
            const double s = 0.35 * std::sin(2.0 * kPi * 220.0 * t)
                           + 0.20 * std::sin(2.0 * kPi * 517.0 * t);
            pcm.push_back(static_cast<float>(s));
        }
        pcm.insert(pcm.end(), static_cast<size_t>(kSampleRate * gap_s), 0.0f);
    }
    return pcm;
}

uint32_t read_u32(const unsigned char * p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t read_u16(const unsigned char * p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

// Minimal RIFF/WAVE reader: PCM16 or float32, any channel count, downmixed to
// mono. Enough for the fixtures in assets/asr_validation/.
std::vector<float> read_wav_mono_f32(const std::string & path, int & sample_rate_out) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "cannot open audio file: " + path);
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    require(buf.size() > 44 && std::memcmp(buf.data(), "RIFF", 4) == 0
                && std::memcmp(buf.data() + 8, "WAVE", 4) == 0,
            "not a RIFF/WAVE file: " + path);

    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    size_t   data_off = 0, data_len = 0;

    size_t pos = 12;
    while (pos + 8 <= buf.size()) {
        const char *   id   = reinterpret_cast<const char *>(buf.data() + pos);
        const uint32_t size = read_u32(buf.data() + pos + 4);
        const size_t   body = pos + 8;
        if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16 && body + 16 <= buf.size()) {
            format   = read_u16(buf.data() + body + 0);
            channels = read_u16(buf.data() + body + 2);
            rate     = read_u32(buf.data() + body + 4);
            bits     = read_u16(buf.data() + body + 14);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_off = body;
            data_len = std::min<size_t>(size, buf.size() - body);
        }
        pos = body + size + (size & 1u);  // chunks are word-aligned
    }
    require(data_off != 0 && channels > 0, "WAV has no usable fmt/data chunk: " + path);
    sample_rate_out = static_cast<int>(rate);

    std::vector<float> interleaved;
    if (format == 1 && bits == 16) {
        const size_t n = data_len / 2;
        interleaved.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const auto s = static_cast<int16_t>(read_u16(buf.data() + data_off + i * 2));
            interleaved.push_back(static_cast<float>(s) / 32768.0f);
        }
    } else if (format == 3 && bits == 32) {
        const size_t n = data_len / 4;
        interleaved.resize(n);
        std::memcpy(interleaved.data(), buf.data() + data_off, n * 4);
    } else {
        throw std::runtime_error("unsupported WAV encoding (format=" + std::to_string(format)
                                 + " bits=" + std::to_string(bits) + "): " + path);
    }

    if (channels == 1) {
        return interleaved;
    }
    std::vector<float> mono(interleaved.size() / channels);
    for (size_t i = 0; i < mono.size(); ++i) {
        float acc = 0.0f;
        for (uint16_t c = 0; c < channels; ++c) {
            acc += interleaved[i * channels + c];
        }
        mono[i] = acc / static_cast<float>(channels);
    }
    return mono;
}

std::vector<float> load_audio(const std::string & path) {
    if (path.empty()) {
        return make_burst_pcm(3, 0.6, 0.4);
    }
    int rate = 0;
    std::vector<float> pcm = read_wav_mono_f32(path, rate);
    require(rate == kSampleRate,
            "test audio must be 16 kHz (got " + std::to_string(rate) + "); v1 links no resampler");
    return pcm;
}

// Feed sizes deliberately coprime-ish with any plausible chunk size (Silero
// uses 512 at 16 kHz; others use 1024 / 1600 / 16000). None of these divide
// evenly into those, and the cycle length is odd, so chunk boundaries land at
// a different offset within each feed.
const int kFeedSizes[] = {1, 337, 7, 4099, 63, 1531, 2, 911};

// Total speech duration the session currently reports, in ms. Works for a VAD
// (segments are speech regions) and for ASR (segments are utterances).
int64_t total_segment_ms(const struct transcribe_session * session) {
    const int n = transcribe_n_segments(session);
    int64_t total = 0;
    for (int i = 0; i < n; ++i) {
        struct transcribe_segment seg;
        transcribe_segment_init(&seg);
        if (transcribe_get_segment(session, i, &seg) != TRANSCRIBE_OK) {
            continue;
        }
        if (seg.t1_ms > seg.t0_ms) {
            total += seg.t1_ms - seg.t0_ms;
        }
    }
    return total;
}

struct StreamOutcome {
    int64_t input_ms      = 0;
    int64_t committed_ms  = 0;
    int     n_segments    = 0;
    int64_t speech_ms     = 0;
    int     final_revision = 0;
};

// One begin -> many odd-sized feeds -> finalize cycle, checking the cursor
// algebra after every feed.
StreamOutcome run_stream(struct transcribe_session * session,
                         const std::vector<float> &  pcm,
                         const std::string &         label) {
    struct transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    struct transcribe_stream_params stream_params;
    transcribe_stream_params_init(&stream_params);

    require(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_IDLE,
            label + ": stream must start IDLE");

    const transcribe_status begin_st = transcribe_stream_begin(session, &run_params, &stream_params);
    require(begin_st == TRANSCRIBE_OK,
            label + ": transcribe_stream_begin failed: status=" + std::to_string(static_cast<int>(begin_st)));
    require(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_ACTIVE,
            label + ": stream must be ACTIVE after begin");

    size_t  offset       = 0;
    size_t  feed_index   = 0;
    int     n_feeds      = 0;
    int64_t fed_samples  = 0;
    int64_t last_input_ms = -1;

    while (offset < pcm.size()) {
        const size_t want = static_cast<size_t>(kFeedSizes[feed_index % (sizeof(kFeedSizes) / sizeof(kFeedSizes[0]))]);
        const size_t take = std::min(want, pcm.size() - offset);
        ++feed_index;

        struct transcribe_stream_update update;
        transcribe_stream_update_init(&update);
        const transcribe_status feed_st =
            transcribe_stream_feed(session, pcm.data() + offset, static_cast<int>(take), &update);
        require(feed_st == TRANSCRIBE_OK,
                label + ": transcribe_stream_feed failed at sample " + std::to_string(offset)
                    + " (n=" + std::to_string(take) + "): status=" + std::to_string(static_cast<int>(feed_st)));

        offset      += take;
        fed_samples += static_cast<int64_t>(take);
        ++n_feeds;

        // --- cursor algebra, checked on every single feed ---
        require(update.input_received_ms >= last_input_ms,
                label + ": input_received_ms went backwards at feed " + std::to_string(n_feeds));
        last_input_ms = update.input_received_ms;

        require(update.audio_committed_ms <= update.input_received_ms,
                label + ": audio_committed_ms (" + std::to_string(update.audio_committed_ms)
                    + ") exceeds input_received_ms (" + std::to_string(update.input_received_ms)
                    + ") at feed " + std::to_string(n_feeds));
        require(update.buffered_ms == update.input_received_ms - update.audio_committed_ms,
                label + ": buffered_ms (" + std::to_string(update.buffered_ms) + ") != input - committed at feed "
                    + std::to_string(n_feeds));

        // input_received_ms is derived from the fed sample count; allow 1 ms
        // for integer division of samples -> ms.
        const int64_t expect_ms = (fed_samples * 1000) / kSampleRate;
        require(std::llabs(update.input_received_ms - expect_ms) <= 1,
                label + ": input_received_ms (" + std::to_string(update.input_received_ms) + ") != fed audio ("
                    + std::to_string(expect_ms) + " ms) at feed " + std::to_string(n_feeds));
    }

    struct transcribe_stream_update final_update;
    transcribe_stream_update_init(&final_update);
    const transcribe_status fin_st = transcribe_stream_finalize(session, &final_update);
    require(fin_st == TRANSCRIBE_OK,
            label + ": transcribe_stream_finalize failed: status=" + std::to_string(static_cast<int>(fin_st)));
    require(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_FINISHED,
            label + ": stream must be FINISHED after finalize");
    require(final_update.is_final, label + ": final update must set is_final");

    // The tail below one chunk must not be stranded: finalize flushes it.
    require(final_update.audio_committed_ms == final_update.input_received_ms,
            label + ": finalize left " + std::to_string(final_update.input_received_ms - final_update.audio_committed_ms)
                + " ms uncommitted; the sub-chunk tail was dropped");
    require(final_update.buffered_ms == 0, label + ": finalize left audio buffered");

    StreamOutcome out;
    out.input_ms       = final_update.input_received_ms;
    out.committed_ms   = final_update.audio_committed_ms;
    out.n_segments     = transcribe_n_segments(session);
    out.speech_ms      = total_segment_ms(session);
    out.final_revision = transcribe_stream_revision(session);

    std::cout << "  " << label << ": " << n_feeds << " feeds, " << out.input_ms << " ms in, "
              << out.n_segments << " segment(s), " << out.speech_ms << " ms speech, revision "
              << out.final_revision << "\n";

    transcribe_stream_reset(session);
    require(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_IDLE,
            label + ": reset must return the stream to IDLE");
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string model_path = env_or_arg(argc, argv, 1, "TRANSCRIBE_BRIDGE_TEST_MODEL");
        if (model_path.empty()) {
            std::cerr << "abi_stream_hello: skipped (no model path; pass argv[1] or "
                         "TRANSCRIBE_BRIDGE_TEST_MODEL)\n";
            return 2;
        }
        const std::string audio_path = env_or_arg(argc, argv, 2, "TRANSCRIBE_BRIDGE_TEST_AUDIO");

        struct transcribe_model_load_params load_params;
        transcribe_model_load_params_init(&load_params);
        struct transcribe_session_params session_params;
        transcribe_session_params_init(&session_params);

        struct transcribe_session * session = nullptr;
        const transcribe_status load_st =
            transcribe_open(model_path.c_str(), &load_params, &session_params, &session);
        if (load_st != TRANSCRIBE_OK || session == nullptr) {
            std::cerr << "abi_stream_hello: transcribe_open failed: status=" << static_cast<int>(load_st)
                      << " path=" << model_path << "\n";
            return 1;
        }

        struct transcribe_capabilities caps;
        transcribe_capabilities_init(&caps);
        const transcribe_status caps_st = transcribe_model_get_capabilities(transcribe_get_model(session), &caps);
        require(caps_st == TRANSCRIBE_OK || caps_st == TRANSCRIBE_ERR_BAD_STRUCT_SIZE,
                "transcribe_model_get_capabilities failed");
        if (!caps.supports_streaming) {
            std::cerr << "abi_stream_hello: skipped (model reports supports_streaming == false)\n";
            transcribe_session_free(session);
            return 2;
        }

        const std::vector<float> pcm = load_audio(audio_path);
        require(!pcm.empty(), "test audio is empty");
        std::cout << "abi_stream_hello: " << pcm.size() << " samples ("
                  << (pcm.size() * 1000 / kSampleRate) << " ms) from "
                  << (audio_path.empty() ? std::string("synthesized bursts") : audio_path) << "\n";

        // Offline baseline first, on the same session, so the streaming result
        // has something independent to agree with.
        struct transcribe_run_params run_params;
        transcribe_run_params_init(&run_params);
        const transcribe_status run_st =
            transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &run_params);
        require(run_st == TRANSCRIBE_OK,
                "offline transcribe_run failed: status=" + std::to_string(static_cast<int>(run_st)));
        const int     offline_segments = transcribe_n_segments(session);
        const int64_t offline_speech   = total_segment_ms(session);
        std::cout << "  offline: " << offline_segments << " segment(s), " << offline_speech << " ms speech\n";

        // Two streaming cycles on the same session: the second proves the
        // per-utterance state really is cleared by reset.
        const StreamOutcome first  = run_stream(session, pcm, "stream #1");
        const StreamOutcome second = run_stream(session, pcm, "stream #2");

        require(first.input_ms == second.input_ms,
                "the two streaming cycles disagree on input length: " + std::to_string(first.input_ms)
                    + " vs " + std::to_string(second.input_ms));
        require(first.n_segments == second.n_segments && first.speech_ms == second.speech_ms,
                "streaming is not deterministic across cycles: " + std::to_string(first.n_segments) + " seg / "
                    + std::to_string(first.speech_ms) + " ms vs " + std::to_string(second.n_segments) + " seg / "
                    + std::to_string(second.speech_ms) + " ms — reset left per-utterance state behind");

        // Streaming/offline agreement. Not bit-exact by construction: a
        // streaming session sees a causal, chunk-quantized view of the audio
        // and finalize zero-pads the tail, so segment boundaries move by up to
        // a chunk. A desynchronized re-blocker, by contrast, shifts or
        // duplicates audio and moves this by a large fraction of the input.
        if (offline_speech > 0) {
            const double drift = std::fabs(static_cast<double>(first.speech_ms - offline_speech))
                               / static_cast<double>(offline_speech);
            std::cout << "  streaming vs offline speech drift: " << (drift * 100.0) << "%\n";
            require(drift <= 0.25,
                    "streaming detected " + std::to_string(first.speech_ms) + " ms of speech but offline detected "
                        + std::to_string(offline_speech) + " ms (" + std::to_string(drift * 100.0)
                        + "% drift) — the re-blocked audio does not match the contiguous audio");
        }

        std::cout << "abi_stream_hello: ok\n";
        transcribe_session_free(session);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "abi_stream_hello: " << ex.what() << "\n";
        return 1;
    }
}
