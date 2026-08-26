#pragma once

// Qwen3-ASR publisher language table (Phase 10.5, merged from transcribe.cpp's
// arch/qwen3_asr k_qwen3_asr_language_names). The chat template seeds a forced
// language as "language {Name}<asr_text>" with the publisher's display name;
// callers - the C ABI in particular - speak BCP-47. Both directions live here
// so the prompt builder can accept either form and a detected name can be
// reported back as a code.

#include <cctype>
#include <optional>
#include <string>
#include <string_view>

namespace engine::models::qwen3_asr {

struct Qwen3ASRLanguageName {
    const char * bcp47;
    const char * name;
};

// The 30 languages of the Qwen3-ASR model card, in publisher order. The
// tokenizer-parity fixture (tests/fixtures/qwen3_asr_bpe_parity.inc) carries
// the same 30 pairs; drift between the two surfaces in qwen3_asr_bpe_parity_test.
inline constexpr Qwen3ASRLanguageName kQwen3ASRLanguageNames[] = {
    {"zh", "Chinese"},    {"en", "English"},    {"yue", "Cantonese"}, {"ar", "Arabic"},
    {"de", "German"},     {"fr", "French"},     {"es", "Spanish"},    {"pt", "Portuguese"},
    {"id", "Indonesian"}, {"it", "Italian"},    {"ko", "Korean"},     {"ru", "Russian"},
    {"th", "Thai"},       {"vi", "Vietnamese"}, {"ja", "Japanese"},   {"tr", "Turkish"},
    {"hi", "Hindi"},      {"ms", "Malay"},      {"nl", "Dutch"},      {"sv", "Swedish"},
    {"da", "Danish"},     {"fi", "Finnish"},    {"pl", "Polish"},     {"cs", "Czech"},
    {"fil", "Filipino"},  {"fa", "Persian"},    {"el", "Greek"},      {"ro", "Romanian"},
    {"hu", "Hungarian"},  {"mk", "Macedonian"},
};

namespace detail {

inline bool qwen3_asr_iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace detail

// "en" -> "English" (case-insensitive on the code). Unknown codes -> nullopt.
inline std::optional<std::string> qwen3_asr_language_name_from_bcp47(std::string_view code) {
    for (const auto & entry : kQwen3ASRLanguageNames) {
        if (detail::qwen3_asr_iequals(code, entry.bcp47)) {
            return std::string(entry.name);
        }
    }
    return std::nullopt;
}

// "English" -> "en" (case-insensitive on the name). Unknown names -> nullopt.
inline std::optional<std::string> qwen3_asr_bcp47_from_language_name(std::string_view name) {
    for (const auto & entry : kQwen3ASRLanguageNames) {
        if (detail::qwen3_asr_iequals(name, entry.name)) {
            return std::string(entry.bcp47);
        }
    }
    return std::nullopt;
}

// The display name the prompt should carry for a caller-supplied language:
// a BCP-47 code maps to the publisher name; a name (or anything unknown)
// passes through unchanged so behaviour for existing callers is identical.
inline std::string qwen3_asr_prompt_language_name(const std::string & language) {
    if (auto name = qwen3_asr_language_name_from_bcp47(language)) {
        return *name;
    }
    return language;
}

}  // namespace engine::models::qwen3_asr
