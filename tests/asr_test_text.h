// tests/asr_test_text.h - shared text-scoring helpers for the end-to-end ASR
// gates (asr_e2e_wer_test, asr_stream_text_wer_test).
//
// Header-only for the same reason as abi_test_wav.h: the gates link ONLY the
// public C ABI, like a language binding, so shared machinery must not pull in
// engine_runtime or ggml. Extracted verbatim from asr_e2e_wer_test.cpp when
// the streaming-text gate landed.

#ifndef SPEECHCPP_TESTS_ASR_TEST_TEXT_H
#define SPEECHCPP_TESTS_ASR_TEST_TEXT_H

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace asr_test {

// LibriSpeech-style normalization: uppercase, keep [A-Z0-9'], everything else
// is a separator. Typographic apostrophes (U+2018/U+2019) fold to ASCII '
// before filtering so "DON’T" == "DON'T"; other non-ASCII bytes separate.
// Tokens are stripped of leading/trailing apostrophes ("'CAUSE" from a
// quoting model still mismatches TRUE 'CAUSE — acceptable for a gate; the
// references here contain internal apostrophes only).
inline std::vector<std::string> normalize_words(const std::string & text) {
    std::string mapped;
    mapped.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == 0xE2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x80
            && (static_cast<unsigned char>(text[i + 2]) == 0x98
                || static_cast<unsigned char>(text[i + 2]) == 0x99)) {
            mapped.push_back('\'');
            i += 2;
            continue;
        }
        if (c >= 0x80) {
            mapped.push_back(' ');
        } else if (std::isalnum(c) != 0) {
            mapped.push_back(static_cast<char>(std::toupper(c)));
        } else if (c == '\'') {
            mapped.push_back('\'');
        } else {
            mapped.push_back(' ');
        }
    }

    std::vector<std::string> words;
    std::istringstream       iss(mapped);
    std::string              token;
    while (iss >> token) {
        size_t begin = 0, end = token.size();
        while (begin < end && token[begin] == '\'') {
            ++begin;
        }
        while (end > begin && token[end - 1] == '\'') {
            --end;
        }
        if (end > begin) {
            words.push_back(token.substr(begin, end - begin));
        }
    }
    return words;
}

// Word-level Levenshtein distance (substitutions + deletions + insertions),
// two-row DP. Utterances here are tens of words, so O(ref*hyp) is nothing.
inline size_t word_edit_distance(const std::vector<std::string> & ref,
                                 const std::vector<std::string> & hyp) {
    std::vector<size_t> prev(hyp.size() + 1);
    std::vector<size_t> cur(hyp.size() + 1);
    for (size_t j = 0; j <= hyp.size(); ++j) {
        prev[j] = j;
    }
    for (size_t i = 1; i <= ref.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= hyp.size(); ++j) {
            const size_t sub = prev[j - 1] + (ref[i - 1] == hyp[j - 1] ? 0 : 1);
            cur[j]           = std::min({ prev[j] + 1, cur[j - 1] + 1, sub });
        }
        std::swap(prev, cur);
    }
    return prev[hyp.size()];
}

inline std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        throw std::runtime_error("cannot open reference transcript: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline std::string join_words(const std::vector<std::string> & words) {
    std::string out;
    for (const auto & w : words) {
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += w;
    }
    return out;
}

struct Fixture {
    std::filesystem::path wav;
    std::filesystem::path txt;
};

// Scans dir for *.wav files with a sibling .txt reference, sorted by filename
// so per-fixture output is stable across platforms and runs.
inline std::vector<Fixture> collect_fixtures(const std::filesystem::path & dir) {
    std::vector<Fixture> fixtures;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".wav") {
            continue;
        }
        std::filesystem::path txt = entry.path();
        txt.replace_extension(".txt");
        if (std::filesystem::exists(txt)) {
            fixtures.push_back({ entry.path(), txt });
        }
    }
    std::sort(fixtures.begin(), fixtures.end(),
              [](const Fixture & a, const Fixture & b) { return a.wav.filename() < b.wav.filename(); });
    return fixtures;
}

}  // namespace asr_test

#endif  // SPEECHCPP_TESTS_ASR_TEST_TEXT_H
