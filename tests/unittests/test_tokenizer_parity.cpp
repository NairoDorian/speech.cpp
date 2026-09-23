// These checks use assert() and the suite builds Release, where NDEBUG
// compiles every one of them out - until 2026-09-23 this test could not
// fail. Keep the checks live regardless of build type.
#undef NDEBUG

#include "engine/framework/text/tokenizer_hub.h"

#include "gguf.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace engine::text;

int main() {
    std::cout << "[tokenizer_parity_test] Starting tokenizer parity verification..." << std::endl;

    // 1. SentencePiece-style tokenizer with \xE2\x96\x81 leading spaces
    {
        std::vector<std::string> tokens = {
            "<unk>", "<s>", "</s>", "\xE2\x96\x81Hello", "\xE2\x96\x81world", "!"
        };
        SpecialTokens specials;
        specials.unk = 0;
        specials.bos = 1;
        specials.eos = 2;

        auto tok = load_tokenizer_from_tokens(tokens, specials, false);
        assert(tok != nullptr);
        assert(tok->vocab_size() == 6);
        assert(tok->specials().bos == 1);

        // Find
        auto id_hello = tok->find("\xE2\x96\x81Hello");
        assert(id_hello.has_value() && *id_hello == 3);

        // Decode
        std::vector<int32_t> seq = {3, 4, 5};
        std::string decoded = tok->decode(seq);
        assert(decoded == " Hello world!");
        std::cout << "  [PASS] SentencePiece token decode parity: '" << decoded << "'" << std::endl;
    }

    // 2. Tiktoken raw-bytes vocabulary (Whisper `.bin` layout: single bytes
    //    plus merged pieces, where a token's id IS its merge rank).
    //    Replaced 2026-09-23: the previous toy vocab ("Hello", " ", "world",
    //    "!") had no single-byte tokens and asserted a greedy longest-prefix
    //    split that tiktoken never produces - GPT-2 pretokenization makes
    //    " world" one piece, and BPE starts from bytes.
    {
        std::vector<std::string> tokens = {
            "H", "e", "l", "o", " ", "w", "r", "d", "!",  // 0..8 single bytes
            "Hello",                                      // 9: whole-piece hit
            " w", "or", " wor", "ld",                     // 10..13 merges
        };
        auto tok = load_tokenizer_from_tokens(tokens, {}, true);
        assert(tok != nullptr);
        assert(tok->model() == TokenizerModel::TiktokenRawBytes);

        // "Hello" | " world" | "!"; " world" merges " w"(10) -> "or"(11) ->
        // " wor"(12) -> "ld"(13): lowest-rank concatenation first.
        auto encoded = tok->encode("Hello world!");
        const std::vector<int32_t> expected = {9, 12, 13, 8};
        assert(encoded == expected);
        const std::string decoded = tok->decode(encoded);
        assert(decoded == "Hello world!");
        std::cout << "  [PASS] Tiktoken rank-BPE encode/decode round-trip: '" << decoded << "'" << std::endl;
    }

    // 3. GPT-2 byte-level BPE from a GGUF: pretokenization and UTF-8 tables.
    //    Both failed before 2026-09-23 (merges crossed word boundaries; bytes
    //    0xA1..0xFF mapped to raw single bytes instead of U+00A1.. UTF-8).
    {
        gguf_context * ctx = gguf_init_empty();
        // "\xC4\xA0" is U+0120 'G-dot', GPT-2's stand-in for the space byte.
        // "\xC3\x83" / "\xC2\xA9" are U+00C3 / U+00A9, the stand-ins for the two
        // UTF-8 bytes of "\xC3\xA9" (e-acute).
        const char * vocab[] = {"a", "b", "\xC4\xA0", "\xC4\xA0" "b", "a\xC4\xA0",
                                "\xC3\x83", "\xC2\xA9", "\xC3\x83\xC2\xA9"};
        // Rank 0 would glue "a" to the following space if BPE ran across the
        // pretokenizer boundary; HF never applies it to "a b".
        const char * merges[] = {"a \xC4\xA0", "\xC4\xA0 b", "\xC3\x83 \xC2\xA9"};
        gguf_set_val_str(ctx, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(ctx, "tokenizer.ggml.pre", "gpt2");
        gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", vocab, sizeof(vocab) / sizeof(vocab[0]));
        gguf_set_arr_str(ctx, "tokenizer.ggml.merges", merges, sizeof(merges) / sizeof(merges[0]));
        auto tok = load_tokenizer_from_gguf(ctx);
        gguf_free(ctx);
        assert(tok != nullptr);
        assert(tok->model() == TokenizerModel::ByteLevelBpe);

        const std::vector<int32_t> ab = tok->encode("a b");
        assert((ab == std::vector<int32_t>{0, 3}));  // "a" | "Gb", never "aG" + "b"
        assert(tok->decode(ab) == "a b");

        const std::vector<int32_t> e_acute = tok->encode("\xC3\xA9");
        assert((e_acute == std::vector<int32_t>{7}));
        assert(tok->decode(e_acute) == "\xC3\xA9");
        std::cout << "  [PASS] GPT-2 byte-level BPE: pretokenized merges + non-ASCII round-trip" << std::endl;
    }

    std::cout << "[tokenizer_parity_test] All tokenizer parity checks PASSED." << std::endl;
    return 0;
}
