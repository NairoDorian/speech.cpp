#include "engine/framework/text/tokenizer_hub.h"

#include "gguf.h"
#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/framework/tokenizers/hf_tokenizer_json.h"
#include "unicode.h"  // external/llama_tokenizer: HF-verified regex pretokenizers

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::text {

namespace {

constexpr const char k_sp_space[]   = "\xE2\x96\x81";
constexpr int        k_sp_space_len = 3;
constexpr char       k_merge_sep    = '\x1F';

// HF ByteLevel(use_regex=True) - the pretokenizer every GPT-2-family GGUF
// ("tokenizer.ggml.model" == "gpt2") assumes - splits text with this regex
// before BPE; merges never cross a piece boundary. llama.cpp's
// unicode_regex_split (external/llama_tokenizer) recognizes these exact
// strings and takes its hand-written, HF-verified fast paths.
constexpr const char * k_gpt2_split_regex =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
constexpr const char * k_qwen2_split_regex =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::string utf8_of_codepoint(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// GPT-2 bytes_to_unicode(): printable bytes map to the code point with the
// same value, the other 68 to U+0100 onward. Vocab pieces are the UTF-8 of
// those code points, so bytes 0xA1..0xFF map to TWO-byte sequences (U+00A1 is
// "\xC2\xA1"). Until 2026-09-23 they were stored as the raw single byte, which
// made every non-ASCII piece unmatchable on encode and decoded it as mojibake
// ("Ã©" for "é").
std::unordered_map<uint8_t, std::string> build_byte_to_unicode_map() {
    std::unordered_map<uint8_t, std::string> b2u;
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const uint32_t cp = printable ? static_cast<uint32_t>(b) : 256u + n++;
        b2u[static_cast<uint8_t>(b)] = utf8_of_codepoint(cp);
    }
    return b2u;
}

std::unordered_map<std::string, uint8_t> build_unicode_to_byte_map() {
    auto b2u = build_byte_to_unicode_map();
    std::unordered_map<std::string, uint8_t> u2b;
    for (const auto & [b, s] : b2u) {
        u2b[s] = b;
    }
    return u2b;
}

const std::unordered_map<uint8_t, std::string> & get_byte_to_unicode() {
    static const auto m = build_byte_to_unicode_map();
    return m;
}

const std::unordered_map<std::string, uint8_t> & get_unicode_to_byte() {
    static const auto m = build_unicode_to_byte_map();
    return m;
}

std::string decode_gpt2_bytes(const std::string & piece) {
    const auto & u2b = get_unicode_to_byte();
    std::string out;
    out.reserve(piece.size());

    size_t i = 0;
    while (i < piece.size()) {
        unsigned char c = static_cast<unsigned char>(piece[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0 && i + 1 < piece.size()) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < piece.size()) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < piece.size()) {
            len = 4;
        }

        std::string cp_str = piece.substr(i, len);
        auto it = u2b.find(cp_str);
        if (it != u2b.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            out.append(cp_str);
        }
        i += len;
    }
    return out;
}

// A SentencePiece byte-fallback piece: exactly "<0xHH>" (hex digits in
// either case). Returns the byte, or -1 for any other piece.
int sp_byte_fallback(const std::string & piece) {
    if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') {
        return -1;
    }
    const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    const int hi = hex(piece[3]);
    const int lo = hex(piece[4]);
    return (hi < 0 || lo < 0) ? -1 : hi * 16 + lo;
}

// A vocabulary marks word starts either the SentencePiece way (U+2581) or the
// GPT-2 byte-level way (U+0120). GGUF "tokenizer.ggml.model" does not say
// which for "bpe": transcribe.cpp's converters write "bpe" for NeMo /
// SentencePiece-BPE vocabs (parakeet, gigaam, medasr), whose pieces must be
// joined with U+2581 -> ' ', not GPT-2 byte-decoded (which leaked literal
// U+2581 characters into the text).
bool vocab_uses_sp_markers(const std::vector<std::string> & tokens) {
    bool sp_marker = false;
    for (const auto & t : tokens) {
        if (t.find("\xC4\xA0") != std::string::npos) {
            return false;
        }
        sp_marker = sp_marker || t.find(k_sp_space) != std::string::npos;
    }
    return sp_marker;
}

std::string decode_sp_piece(const std::string & piece) {
    if (const int byte = sp_byte_fallback(piece); byte >= 0) {
        return std::string(1, static_cast<char>(byte));
    }
    std::string out;
    out.reserve(piece.size());
    size_t i = 0;
    while (i < piece.size()) {
        if (i + k_sp_space_len <= piece.size() &&
            std::memcmp(piece.data() + i, k_sp_space, k_sp_space_len) == 0) {
            out.push_back(' ');
            i += k_sp_space_len;
        } else {
            out.push_back(piece[i]);
            ++i;
        }
    }
    return out;
}

}  // namespace

// Base generic tokenizer implementation
class GenericTokenizerImpl : public ITokenizer {
public:
    TokenizerModel model() const override { return model_; }
    size_t vocab_size() const override { return tokens_.size(); }
    const SpecialTokens & specials() const override { return specials_; }

    std::optional<int32_t> find(std::string_view piece) const override {
        auto it = piece_to_id_.find(std::string(piece));
        if (it != piece_to_id_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::string_view piece(int32_t id) const override {
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
            return {};
        }
        return tokens_[static_cast<size_t>(id)];
    }

    bool is_control(int32_t id) const override {
        if (id < 0 || static_cast<size_t>(id) >= token_type_.size()) {
            return false;
        }
        return token_type_[static_cast<size_t>(id)] == 3;
    }

    std::string decode(const std::vector<int32_t> & ids) const override {
        return decode(ids.data(), ids.size());
    }

    std::string decode(const int32_t * ids, size_t count) const override {
        if (ids == nullptr || count == 0) {
            return "";
        }

        std::string result;
        if (model_ == TokenizerModel::ByteLevelBpe || (model_ == TokenizerModel::Bpe && !sp_pieces_)) {
            for (size_t i = 0; i < count; ++i) {
                int32_t id = ids[i];
                if (id >= 0 && static_cast<size_t>(id) < tokens_.size()) {
                    result += decode_gpt2_bytes(tokens_[static_cast<size_t>(id)]);
                }
            }
        } else if (model_ == TokenizerModel::TiktokenRawBytes) {
            for (size_t i = 0; i < count; ++i) {
                int32_t id = ids[i];
                if (id >= 0 && static_cast<size_t>(id) < tokens_.size()) {
                    result += tokens_[static_cast<size_t>(id)];
                }
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                int32_t id = ids[i];
                if (id >= 0 && static_cast<size_t>(id) < tokens_.size()) {
                    result += decode_sp_piece(tokens_[static_cast<size_t>(id)]);
                }
            }
        }
        return result;
    }

    std::vector<int32_t> encode(std::string_view text) const override {
        if (model_ == TokenizerModel::TiktokenRawBytes) {
            return encode_tiktoken(text);
        }
        if (model_ == TokenizerModel::ByteLevelBpe) {
            return encode_bpe(text);
        }
        return encode_greedy(text);
    }

protected:
    TokenizerModel                           model_ = TokenizerModel::Unigram;
    Pretokenizer                             pre_   = Pretokenizer::None;
    SpecialTokens                            specials_;
    std::vector<std::string>                 tokens_;
    std::vector<float>                       scores_;
    std::vector<int32_t>                     token_type_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    // TokenizerModel::Bpe over a SentencePiece-marked vocabulary (see
    // vocab_uses_sp_markers): decoded by SentencePiece piece join.
    bool                                     sp_pieces_ = false;

    // Splits text into the pieces BPE runs on, per the tokenizer's declared
    // pretokenizer (byte-level models without one get HF's ByteLevel default,
    // the GPT-2 regex). Pieces are raw UTF-8 substrings of `text`. Text that is
    // not valid UTF-8 cannot be regex-split; it stays one piece, which is
    // still exactly representable by byte-level BPE.
    std::vector<std::string> pretokenize(std::string_view text) const {
        if (text.empty()) {
            return {};
        }
        const char * regex = pre_ == Pretokenizer::Qwen2 ? k_qwen2_split_regex : k_gpt2_split_regex;
        try {
            return unicode_regex_split(std::string(text), { regex }, /*byte_encode=*/false);
        } catch (const std::exception &) {
            return { std::string(text) };
        }
    }

    // tiktoken's byte-pair merge (Whisper's `.bin` vocabularies and other
    // raw-bytes tables): pretokenize with the GPT-2 regex, map a piece that is
    // itself a token straight to its id, otherwise start from single bytes and
    // repeatedly merge the adjacent pair whose concatenation has the LOWEST
    // token id (tiktoken ranks ARE ids), leftmost on ties. Replaces a greedy
    // longest-prefix match that produced ids HF / tiktoken never would.
    std::vector<int32_t> encode_tiktoken(std::string_view text) const {
        std::vector<int32_t> out;
        for (const std::string & word : pretokenize(text)) {
            if (const auto whole = piece_to_id_.find(word); whole != piece_to_id_.end()) {
                out.push_back(whole->second);
                continue;
            }
            std::vector<std::string> parts;
            parts.reserve(word.size());
            for (const char c : word) {
                parts.emplace_back(1, c);
            }
            while (parts.size() >= 2) {
                int32_t best_rank = std::numeric_limits<int32_t>::max();
                size_t best_idx = 0;
                for (size_t i = 0; i + 1 < parts.size(); ++i) {
                    const auto it = piece_to_id_.find(parts[i] + parts[i + 1]);
                    if (it != piece_to_id_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = i;
                    }
                }
                if (best_rank == std::numeric_limits<int32_t>::max()) {
                    break;
                }
                parts[best_idx] += parts[best_idx + 1];
                parts.erase(parts.begin() + static_cast<ptrdiff_t>(best_idx + 1));
            }
            for (const auto & p : parts) {
                if (const auto it = piece_to_id_.find(p); it != piece_to_id_.end()) {
                    out.push_back(it->second);
                } else if (specials_.unk >= 0) {
                    out.push_back(specials_.unk);
                }
            }
        }
        return out;
    }

    std::vector<int32_t> encode_greedy(std::string_view text) const {
        std::vector<int32_t> out;
        size_t i = 0;
        while (i < text.size()) {
            size_t best_len = 0;
            int32_t best_id = -1;
            for (size_t l = std::min(text.size() - i, size_t(32)); l > 0; --l) {
                std::string sub(text.substr(i, l));
                auto it = piece_to_id_.find(sub);
                if (it != piece_to_id_.end()) {
                    best_len = l;
                    best_id = it->second;
                    break;
                }
            }
            if (best_len > 0) {
                out.push_back(best_id);
                i += best_len;
            } else {
                if (specials_.unk >= 0) {
                    out.push_back(specials_.unk);
                }
                ++i;
            }
        }
        return out;
    }

    // GPT-2 byte-level BPE, HF-exact: pretokenize, map each piece's bytes
    // through bytes_to_unicode, then merge the lowest-ranked adjacent pair
    // (leftmost on ties) until no listed merge applies - per piece. Until
    // 2026-09-23 this ran BPE over the whole string with no pretokenizer, so
    // merges could cross word boundaries (e.g. "hello 123" diverged from HF).
    std::vector<int32_t> encode_bpe(std::string_view text) const {
        const auto & b2u = get_byte_to_unicode();
        std::vector<int32_t> ids;
        for (const std::string & word : pretokenize(text)) {
            std::vector<std::string> pieces;
            pieces.reserve(word.size());
            for (const unsigned char c : word) {
                pieces.push_back(b2u.at(c));
            }

            while (pieces.size() >= 2) {
                int min_rank = std::numeric_limits<int>::max();
                size_t best_idx = 0;
                for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                    const std::string pair = pieces[i] + k_merge_sep + pieces[i + 1];
                    const auto it = merge_ranks_.find(pair);
                    if (it != merge_ranks_.end() && it->second < min_rank) {
                        min_rank = it->second;
                        best_idx = i;
                    }
                }
                if (min_rank == std::numeric_limits<int>::max()) {
                    break;
                }
                pieces[best_idx] += pieces[best_idx + 1];
                pieces.erase(pieces.begin() + static_cast<ptrdiff_t>(best_idx + 1));
            }

            for (const auto & p : pieces) {
                if (const auto it = piece_to_id_.find(p); it != piece_to_id_.end()) {
                    ids.push_back(it->second);
                } else if (specials_.unk >= 0) {
                    ids.push_back(specials_.unk);
                }
            }
        }
        return ids;
    }

    friend class GgufTokenizerBuilder;
};

// Builder to parse GGUF context and instantiate Tokenizer
class GgufTokenizerImpl final : public GenericTokenizerImpl {
public:
    static TokenizerPtr build(const gguf_context * ctx) {
        if (ctx == nullptr) {
            return nullptr;
        }

        auto tok = std::make_shared<GgufTokenizerImpl>();

        // Model type
        int64_t model_key = gguf_find_key(ctx, "tokenizer.ggml.model");
        if (model_key >= 0 && gguf_get_kv_type(ctx, model_key) == GGUF_TYPE_STRING) {
            std::string model_name = gguf_get_val_str(ctx, model_key);
            if (model_name == "gpt2") {
                tok->model_ = TokenizerModel::ByteLevelBpe;
                tok->pre_   = Pretokenizer::Gpt2;
            } else if (model_name == "bpe") {
                tok->model_ = TokenizerModel::Bpe;
            } else {
                tok->model_ = TokenizerModel::Unigram;
            }
        }

        // Pretokenizer. Only the splits this hub implements are honoured;
        // anything else keeps the model's default (GPT-2 for byte-level BPE).
        int64_t pre_key = gguf_find_key(ctx, "tokenizer.ggml.pre");
        if (pre_key >= 0 && gguf_get_kv_type(ctx, pre_key) == GGUF_TYPE_STRING) {
            const std::string pre_name = gguf_get_val_str(ctx, pre_key);
            if (pre_name == "qwen2") {
                tok->pre_ = Pretokenizer::Qwen2;
            } else if (pre_name == "gpt2") {
                tok->pre_ = Pretokenizer::Gpt2;
            }
        }

        // Tokens
        int64_t tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
        if (tokens_key >= 0 && gguf_get_kv_type(ctx, tokens_key) == GGUF_TYPE_ARRAY) {
            size_t n = gguf_get_arr_n(ctx, tokens_key);
            tok->tokens_.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const char * str = gguf_get_arr_str(ctx, tokens_key, i);
                tok->tokens_[i] = str ? str : "";
                tok->piece_to_id_[tok->tokens_[i]] = static_cast<int32_t>(i);
            }
        }

        if (tok->model_ == TokenizerModel::Bpe) {
            tok->sp_pieces_ = vocab_uses_sp_markers(tok->tokens_);
        }

        // Scores
        int64_t scores_key = gguf_find_key(ctx, "tokenizer.ggml.scores");
        if (scores_key >= 0 && gguf_get_kv_type(ctx, scores_key) == GGUF_TYPE_ARRAY) {
            size_t n = gguf_get_arr_n(ctx, scores_key);
            const float * data = static_cast<const float *>(gguf_get_arr_data(ctx, scores_key));
            if (data) {
                tok->scores_.assign(data, data + n);
            }
        }

        // Token types
        int64_t types_key = gguf_find_key(ctx, "tokenizer.ggml.token_type");
        if (types_key >= 0 && gguf_get_kv_type(ctx, types_key) == GGUF_TYPE_ARRAY) {
            size_t n = gguf_get_arr_n(ctx, types_key);
            const int32_t * data = static_cast<const int32_t *>(gguf_get_arr_data(ctx, types_key));
            if (data) {
                tok->token_type_.assign(data, data + n);
            }
        }

        // Merges
        int64_t merges_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
        if (merges_key >= 0 && gguf_get_kv_type(ctx, merges_key) == GGUF_TYPE_ARRAY) {
            size_t n = gguf_get_arr_n(ctx, merges_key);
            for (size_t i = 0; i < n; ++i) {
                const char * m = gguf_get_arr_str(ctx, merges_key, i);
                if (m) {
                    std::string merge_str(m);
                    size_t sp = merge_str.find(' ');
                    if (sp != std::string::npos) {
                        std::string key = merge_str.substr(0, sp) + k_merge_sep + merge_str.substr(sp + 1);
                        tok->merge_ranks_[key] = static_cast<int32_t>(i);
                    }
                }
            }
        }

        // Special tokens
        auto read_special = [&](const char * key) -> int {
            int64_t k = gguf_find_key(ctx, key);
            if (k >= 0) {
                if (gguf_get_kv_type(ctx, k) == GGUF_TYPE_UINT32) {
                    return static_cast<int>(gguf_get_val_u32(ctx, k));
                }
                if (gguf_get_kv_type(ctx, k) == GGUF_TYPE_INT32) {
                    return gguf_get_val_i32(ctx, k);
                }
            }
            return -1;
        };

        tok->specials_.unk   = read_special("tokenizer.ggml.unknown_token_id");
        tok->specials_.bos   = read_special("tokenizer.ggml.bos_token_id");
        tok->specials_.eos   = read_special("tokenizer.ggml.eos_token_id");
        tok->specials_.pad   = read_special("tokenizer.ggml.padding_token_id");
        tok->specials_.blank = read_special("tokenizer.ggml.blank_token_id");

        return tok;
    }
};

class RawTokensTokenizerImpl final : public GenericTokenizerImpl {
    // A bare vocabulary has no model tag, but the two families it can come
    // from mark spaces differently: SentencePiece pieces use U+2581 ("\xE2\x96\x81"),
    // GPT-2 byte-level pieces use U+0120 ("\xC4\xA0"). Until 2026-09-23 every
    // non-raw-bytes vocab was treated as byte-level, so SentencePiece vocabs -
    // including every load_tokenizer_from_sentencepiece() result - decoded with
    // literal U+2581 characters instead of spaces.
    static TokenizerModel infer_model(const std::vector<std::string> & tokens) {
        return vocab_uses_sp_markers(tokens) ? TokenizerModel::Unigram : TokenizerModel::ByteLevelBpe;
    }

public:
    RawTokensTokenizerImpl(std::vector<std::string> tokens, const SpecialTokens & specials, bool raw_bytes) {
        tokens_   = std::move(tokens);
        specials_ = specials;
        model_    = raw_bytes ? TokenizerModel::TiktokenRawBytes : infer_model(tokens_);
        pre_      = Pretokenizer::Gpt2;

        for (size_t i = 0; i < tokens_.size(); ++i) {
            piece_to_id_[tokens_[i]] = static_cast<int32_t>(i);
        }
    }
};

TokenizerPtr load_tokenizer_from_gguf(const gguf_context * ctx) {
    return GgufTokenizerImpl::build(ctx);
}

TokenizerPtr load_tokenizer_from_sentencepiece(const std::string & model_path) {
    auto sp = engine::tokenizers::load_sentencepiece_tokenizer(model_path);
    if (!sp) {
        return nullptr;
    }
    // Convert into GenericTokenizerImpl or wrap SentencePieceTokenizer
    auto tok = std::make_shared<GenericTokenizerImpl>();
    auto pieces = engine::tokenizers::load_sentencepiece_model(model_path);
    std::vector<std::string> tokens(pieces.size());
    for (size_t i = 0; i < pieces.size(); ++i) {
        tokens[i] = pieces[i].text;
    }
    return load_tokenizer_from_tokens(std::move(tokens));
}

TokenizerPtr load_tokenizer_from_hf_json(const std::string & json_path) {
    auto hf = engine::tokenizers::load_huggingface_tokenizer_json(json_path);
    if (!hf) {
        return nullptr;
    }
    return load_tokenizer_from_tokens(hf->id_to_token());
}

TokenizerPtr load_tokenizer_from_tokens(std::vector<std::string> tokens,
                                         const SpecialTokens &    specials,
                                         bool                     raw_bytes) {
    return std::make_shared<RawTokensTokenizerImpl>(std::move(tokens), specials, raw_bytes);
}

}  // namespace engine::text
