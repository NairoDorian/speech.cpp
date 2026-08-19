// tests/abi_test_wav.h - minimal RIFF/WAVE reader shared by the C-ABI bridge
// tests (abi_stream_hello, asr_e2e_wer_test).
//
// Header-only ON PURPOSE: these tests link ONLY the public C ABI — no
// engine_runtime, no ggml — so they exercise the same surface a language
// binding sees and prove the shipped shared library is self-sufficient. That
// is why this does not reuse engine/framework/audio/wav_reader.h.
//
// Supports PCM16 and float32, any channel count, downmixed to mono. Enough
// for the fixtures in assets/asr_validation/.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace abi_test {

inline uint32_t read_u32(const unsigned char * p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint16_t read_u16(const unsigned char * p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

inline std::vector<float> read_wav_mono_f32(const std::string & path, int & sample_rate_out) {
    auto fail = [&path](const std::string & what) -> std::vector<float> {
        throw std::runtime_error(what + ": " + path);
    };

    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return fail("cannot open audio file");
    }
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!(buf.size() > 44 && std::memcmp(buf.data(), "RIFF", 4) == 0
              && std::memcmp(buf.data() + 8, "WAVE", 4) == 0)) {
        return fail("not a RIFF/WAVE file");
    }

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
    if (data_off == 0 || channels == 0) {
        return fail("WAV has no usable fmt/data chunk");
    }
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

}  // namespace abi_test
