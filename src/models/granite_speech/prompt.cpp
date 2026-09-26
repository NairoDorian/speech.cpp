// granite_speech prompt construction and result post-processing.
//
// Ported from src/runtime/arch/granite/model.cpp (granite_target_language_name,
// build_granite_affixes, parse_granite_word_timestamps) and
// src/runtime/arch/granite/diarize.cpp (k_saa_instruction,
// split_speaker_turns). The instruction strings are the per-variant
// model-card prompts: each variant's published WER was measured with its own
// prompt and a different one drifts it, and the -plus prompt's leading space
// changes the BPE ids.

#include "engine/models/granite_speech/model.h"

#include <cctype>
#include <cstdint>
#include <stdexcept>

namespace engine::models::granite_speech {
namespace {

// IBM's verbatim speaker-attribution instruction (arch diarize.cpp).
constexpr const char * kSpeakerAttributionInstruction =
    " Speaker attribution: Transcribe and denote who is speaking by adding "
    "[Speaker 1]: and [Speaker 2]: tags before speaker turns.";

// IBM's verbatim -plus word-timestamp instruction (arch build_granite_affixes).
constexpr const char * kWordTimestampsInstruction =
    " Timestamps: Transcribe the speech. After each word, add a timestamp tag "
    "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]";

void append(std::vector<int32_t> & dst, const std::vector<int32_t> & src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string trimmed(const std::string & s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && is_space(s[a])) {
        ++a;
    }
    while (b > a && is_space(s[b - 1])) {
        --b;
    }
    return s.substr(a, b - a);
}

// "[Speaker <digits>]" at raw[i], consuming an optional ':' and one optional
// following space; N >= 1 (arch diarize.cpp:try_speaker_marker).
bool try_speaker_marker(const std::string & raw, size_t i, int32_t & out_id, size_t & out_next) {
    static const std::string prefix = "[Speaker ";
    const size_t n = raw.size();
    if (raw.compare(i, prefix.size(), prefix) != 0) {
        return false;
    }
    size_t j = i + prefix.size();
    if (j >= n || !(raw[j] >= '0' && raw[j] <= '9')) {
        return false;
    }
    int64_t id = 0;
    while (j < n && raw[j] >= '0' && raw[j] <= '9') {
        id = id * 10 + (raw[j] - '0');
        if (id > INT32_MAX) {
            return false;
        }
        ++j;
    }
    if (j >= n || raw[j] != ']' || id <= 0) {
        return false;
    }
    ++j;
    if (j < n && raw[j] == ':') {
        ++j;
    }
    if (j < n && raw[j] == ' ') {
        ++j;
    }
    out_id = static_cast<int32_t>(id);
    out_next = j;
    return true;
}

}  // namespace

std::string granite_speech_target_language_name(const std::string & code_or_name) {
    std::string s = code_or_name;
    for (auto & c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (s == "de" || s == "ger" || s == "deu" || s == "german") return "German";
    if (s == "fr" || s == "fra" || s == "fre" || s == "french") return "French";
    if (s == "es" || s == "spa" || s == "spanish") return "Spanish";
    if (s == "pt" || s == "por" || s == "portuguese") return "Portuguese";
    if (s == "ja" || s == "jpn" || s == "japanese") return "Japanese";
    if (s == "it" || s == "ita" || s == "italian") return "Italian";
    if (s == "zh" || s == "zh-cn" || s == "cmn" || s == "mandarin" || s == "chinese") return "Mandarin";
    if (s == "en" || s == "eng" || s == "english") return "English";
    return {};
}

// arch model.cpp:build_granite_affixes. The audio tokens splice in between
// prefix and suffix; special tokens are emitted as ids, only plain text goes
// through the BPE.
void build_granite_speech_affixes(
    const GraniteSpeechAssets & assets,
    const GraniteSpeechPrompt & prompt,
    std::vector<int32_t> & prefix_ids,
    std::vector<int32_t> & suffix_ids) {
    const auto & tok = assets.tokenizer;
    const auto & variant = assets.hparams.variant;
    const auto & chat = assets.chat_tokens;
    prefix_ids.clear();
    suffix_ids.clear();

    std::string instruction;
    if (variant == "granite-speech-4.1-2b") {
        instruction = "transcribe the speech with proper punctuation and capitalization.";
    } else if (variant == "granite-speech-4.1-2b-plus") {
        instruction = " can you transcribe the speech into a written format?";
    } else {
        instruction = "can you transcribe the speech into a written format?";
    }
    switch (prompt.task) {
        case GraniteSpeechTask::Transcribe:
            break;
        case GraniteSpeechTask::Translate:
            if (prompt.target_language_name.empty()) {
                throw std::invalid_argument("Granite Speech translate task requires a target language");
            }
            instruction = "can you translate the speech into " + prompt.target_language_name + "?";
            break;
        case GraniteSpeechTask::WordTimestamps:
            instruction = kWordTimestampsInstruction;
            break;
        case GraniteSpeechTask::SpeakerAttribution:
            instruction = kSpeakerAttributionInstruction;
            break;
    }

    const bool use_granite4_chat = assets.chat_template.find("<|start_of_role|>") != std::string::npos &&
                                   chat.start_of_role >= 0 && chat.end_of_role >= 0;
    if (use_granite4_chat) {
        // Granite-4 system-role chat (-plus).
        const char * system_content = variant == "granite-speech-4.1-2b-plus"
            ? "Knowledge Cutoff Date: April 2024.\n"
              "Today's Date: December 19, 2024.\n"
              "You are Granite, developed by IBM. You are a helpful AI assistant"
            : "You are a helpful assistant. Please ensure responses are professional, accurate, and safe.";
        prefix_ids.push_back(chat.start_of_role);
        append(prefix_ids, tok.encode("system"));
        prefix_ids.push_back(chat.end_of_role);
        append(prefix_ids, tok.encode(system_content));
        prefix_ids.push_back(chat.end_of_text);
        append(prefix_ids, tok.encode("\n"));
        prefix_ids.push_back(chat.start_of_role);
        append(prefix_ids, tok.encode("user"));
        prefix_ids.push_back(chat.end_of_role);

        append(suffix_ids, tok.encode(instruction));
        suffix_ids.push_back(chat.end_of_text);
        append(suffix_ids, tok.encode("\n"));
        suffix_ids.push_back(chat.start_of_role);
        append(suffix_ids, tok.encode("assistant"));
        suffix_ids.push_back(chat.end_of_role);
    } else {
        // Bare "USER: <|audio|>{instruction}\n ASSISTANT:" (1b / 2b).
        prefix_ids = tok.encode("USER: ");
        suffix_ids = tok.encode(instruction + "\n ASSISTANT:");
    }
}

// arch model.cpp:parse_granite_word_timestamps. N is the preceding word's end
// in centiseconds, wrapping every 10 s; "_" items are silence placeholders
// (advance the clock, not emitted); a trailing word without a marker ends at
// the audio duration.
std::string parse_granite_speech_word_timestamps(
    const std::string & raw, int64_t audio_ms, std::vector<GraniteSpeechWord> & words) {
    std::string pending;
    int rollover = 0;
    long last_raw = -1;
    int64_t prev_end_ms = 0;

    const size_t n = raw.size();
    size_t i = 0;
    while (i < n) {
        if (raw[i] == '[' && i + 2 < n && raw[i + 1] == 'T' && raw[i + 2] == ':') {
            size_t j = i + 3;
            while (j < n && is_space(raw[j])) {
                ++j;
            }
            long val = 0;
            bool has_digit = false;
            while (j < n && raw[j] >= '0' && raw[j] <= '9') {
                val = val * 10 + (raw[j] - '0');
                has_digit = true;
                ++j;
            }
            while (j < n && is_space(raw[j])) {
                ++j;
            }
            if (has_digit && j < n && raw[j] == ']') {
                ++j;
                if (last_raw >= 0 && val < last_raw) {
                    ++rollover;
                }
                last_raw = val;
                const int64_t end_ms = (static_cast<int64_t>(val) + static_cast<int64_t>(rollover) * 1000) * 10;
                const std::string w = trimmed(pending);
                if (!w.empty() && w != "_") {
                    words.push_back({w, prev_end_ms, end_ms});
                }
                prev_end_ms = end_ms;
                pending.clear();
                i = j;
                continue;
            }
            // Not a well-formed marker: '[' is text.
        }
        pending.push_back(raw[i]);
        ++i;
    }
    if (const std::string w = trimmed(pending); !w.empty() && w != "_") {
        words.push_back({w, prev_end_ms, audio_ms > prev_end_ms ? audio_ms : prev_end_ms});
    }

    std::string clean;
    for (size_t k = 0; k < words.size(); ++k) {
        if (k != 0) {
            clean.push_back(' ');
        }
        clean += words[k].text;
    }
    return clean;
}

// arch diarize.cpp:split_speaker_turns. Text before the first marker becomes a
// speaker-0 turn (dropped when empty); unrecognised brackets pass through.
bool split_granite_speech_speaker_turns(
    const std::string & raw, std::vector<GraniteSpeechTurn> & turns, std::string & full_text) {
    std::vector<GraniteSpeechTurn> parsed;
    parsed.push_back(GraniteSpeechTurn{});
    const size_t n = raw.size();
    size_t i = 0;
    bool any_marker = false;
    while (i < n) {
        int32_t id = 0;
        size_t next = 0;
        if (raw[i] == '[' && try_speaker_marker(raw, i, id, next)) {
            parsed.push_back(GraniteSpeechTurn{id, std::string()});
            any_marker = true;
            i = next;
            continue;
        }
        parsed.back().text.push_back(raw[i]);
        ++i;
    }
    if (!any_marker) {
        return false;
    }
    turns.clear();
    full_text.clear();
    for (const auto & turn : parsed) {
        const std::string text = trimmed(turn.text);
        if (turn.speaker == 0 && text.empty()) {
            continue;
        }
        if (!full_text.empty() && !text.empty()) {
            full_text.push_back(' ');
        }
        full_text += text;
        turns.push_back(GraniteSpeechTurn{turn.speaker, text});
    }
    return true;
}

}  // namespace engine::models::granite_speech
