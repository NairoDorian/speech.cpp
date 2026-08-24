#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct gguf_context;

namespace engine::text {

enum class TokenizerModel {
    Unigram,
    Bpe,
    ByteLevelBpe,
    WordPiece,
    SentencePieceBinary,
    HuggingFaceJson,
    TiktokenRawBytes,
};

enum class Pretokenizer {
    None,
    Qwen2,
    Gpt2,
};

struct SpecialTokens {
    int unk           = -1;
    int bos           = -1;
    int eos           = -1;
    int pad           = -1;
    int blank         = -1;
    int decoder_start = -1;
};

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual TokenizerModel       model() const = 0;
    virtual size_t               vocab_size() const = 0;
    virtual const SpecialTokens & specials() const = 0;

    virtual std::vector<int32_t>   encode(std::string_view text) const = 0;
    virtual std::string            decode(const std::vector<int32_t> & ids) const = 0;
    virtual std::string            decode(const int32_t * ids, size_t count) const = 0;
    virtual std::optional<int32_t> find(std::string_view piece) const = 0;
    virtual std::string_view       piece(int32_t id) const = 0;
    virtual bool                   is_control(int32_t id) const { return false; }
    virtual bool                   has_encoder() const { return true; }
};

using TokenizerPtr = std::shared_ptr<ITokenizer>;

// Universal Tokenizer Hub Factory Dispatchers
TokenizerPtr load_tokenizer_from_gguf(const gguf_context * ctx);
TokenizerPtr load_tokenizer_from_sentencepiece(const std::string & model_path);
TokenizerPtr load_tokenizer_from_hf_json(const std::string & json_path);
TokenizerPtr load_tokenizer_from_tokens(std::vector<std::string> tokens,
                                         const SpecialTokens &    specials = {},
                                         bool                     raw_bytes = false);

}  // namespace engine::text
