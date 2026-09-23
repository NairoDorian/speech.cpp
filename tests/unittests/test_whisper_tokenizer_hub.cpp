// test_whisper_tokenizer_hub.cpp - the engine TokenizerHub must encode Whisper
// text exactly as HuggingFace does, from both distribution formats:
//
//   GGUF  (transcribe.cpp layout): tokenizer.ggml.model = "gpt2" with merges
//         -> byte-level BPE behind the GPT-2 pretokenizer.
//   .bin  (whisper.cpp layout): raw-bytes vocabulary whose ids are the merge
//         ranks -> tiktoken byte-pair merge behind the same pretokenizer.
//
// Expected ids are the HF reference (openai/whisper-tiny,
// add_special_tokens=False) carried by transcribe.cpp's
// whisper_tokenize_parity.cpp. The digit cases are load-bearing: GPT-2 merges
// whole digit runs (\p{N}+), which a Qwen2-style single-digit split would get
// wrong. This is what the engine Whisper package needs for initial prompts,
// and what transcribe_tokenize() must return once the arch retires (B16c).
//
// Usage: whisper_tokenizer_hub_test <whisper-tiny.gguf> <ggml-tiny.bin>
// Exit 2 (SKIP) while either model is absent.

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/text/tokenizer_hub.h"
#include "engine/models/whisper/assets.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct Case {
    const char * text;
    std::vector<int32_t> expected;
};

const std::vector<Case> kCases = {
    {"hello 123 world", {675, 1913, 34466, 1002}},
    {"The year is 2024.", {2278, 1064, 307, 45237, 13}},
    {"cost: $99.99", {27718, 25, 1848, 8494, 13, 8494}},
    {"  multi  spaces", {220, 4825, 220, 7673}},
    {"I'm going", {40, 478, 516}},
    {"she isn't here", {9611, 1943, 380, 510}},
    {"hello world", {675, 1913, 1002}},
    {"hello  world", {675, 1913, 220, 1002}},
    {"   triple   space", {220, 220, 15508, 220, 220, 1901}},
    {"42", {15628}},
    {"", {}},
};

int check(const char * label, const engine::text::ITokenizer & tok) {
    int failures = 0;
    for (const auto & c : kCases) {
        const std::vector<int32_t> got = tok.encode(c.text);
        if (got != c.expected) {
            std::fprintf(stderr, "FAIL [%s] encode(\"%s\"): got", label, c.text);
            for (int32_t id : got) {
                std::fprintf(stderr, " %d", id);
            }
            std::fprintf(stderr, "\n");
            ++failures;
            continue;
        }
        // Byte-exact round trip (the decode side of the same tables).
        if (tok.decode(got) != c.text) {
            std::fprintf(stderr, "FAIL [%s] decode(encode(\"%s\")) = \"%s\"\n", label, c.text,
                         tok.decode(got).c_str());
            ++failures;
        }
    }
    // Non-ASCII must survive a round trip (bytes 0x80..0xFF go through the
    // two-byte GPT-2 code points; this is the mojibake class fixed 2026-09-23).
    const std::string umlaut = "Gr\xC3\xBC\xC3\x9F" "e aus K\xC3\xB6ln";
    if (tok.decode(tok.encode(umlaut)) != umlaut) {
        std::fprintf(stderr, "FAIL [%s] non-ASCII round trip\n", label);
        ++failures;
    }
    std::printf("[%s] %zu HF parity cases + non-ASCII round trip: %s\n", label, kCases.size(),
                failures == 0 ? "ok" : "FAILED");
    return failures;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <whisper-tiny.gguf> <ggml-tiny.bin>\n", argv[0]);
        return 1;
    }
    const std::filesystem::path gguf_path = argv[1];
    const std::filesystem::path bin_path = argv[2];
    if (!std::filesystem::exists(gguf_path) || !std::filesystem::exists(bin_path)) {
        std::fprintf(stderr, "SKIP: pinned models absent (scripts/fetch_asr_test_model.py --only tiny)\n");
        return 2;
    }

    int failures = 0;
    try {
        const auto meta = engine::assets::GgufMetadata::open(gguf_path);
        const auto gguf_tok = engine::text::load_tokenizer_from_gguf(meta.context());
        if (!gguf_tok || gguf_tok->model() != engine::text::TokenizerModel::ByteLevelBpe) {
            std::fprintf(stderr, "FAIL: GGUF tokenizer is not byte-level BPE\n");
            return 1;
        }
        failures += check("gguf", *gguf_tok);

        const auto bin_assets = engine::models::whisper::load_whisper_assets(bin_path);
        const auto bin_tok = engine::text::load_tokenizer_from_tokens(bin_assets->vocab_tokens, {}, /*raw_bytes=*/true);
        failures += check("bin", *bin_tok);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }

    if (failures != 0) {
        std::fprintf(stderr, "whisper_tokenizer_hub_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("whisper_tokenizer_hub_test: ok\n");
    return 0;
}
