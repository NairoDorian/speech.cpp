#include "engine/framework/text/tokenizer_hub.h"

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

    // 2. Tiktoken raw bytes vocabulary
    {
        std::vector<std::string> tokens = {
            "Hello", " ", "world", "!"
        };
        auto tok = load_tokenizer_from_tokens(tokens, {}, true);
        assert(tok != nullptr);
        assert(tok->model() == TokenizerModel::TiktokenRawBytes);

        std::vector<int32_t> seq = {0, 1, 2, 3};
        std::string decoded = tok->decode(seq);
        assert(decoded == "Hello world!");

        auto encoded = tok->encode("Hello world!");
        assert(encoded.size() == 4);
        assert(encoded[0] == 0 && encoded[1] == 1 && encoded[2] == 2 && encoded[3] == 3);
        std::cout << "  [PASS] Tiktoken raw bytes encode/decode round-trip: '" << decoded << "'" << std::endl;
    }

    std::cout << "[tokenizer_parity_test] All tokenizer parity checks PASSED." << std::endl;
    return 0;
}
