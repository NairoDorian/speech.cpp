// granite_speech tokenizer: the GGUF "gpt2" byte-level BPE the transcribe.cpp
// granite arch encodes its chat prompt with.
//
// Ported from src/runtime/transcribe-tokenizer.cpp (Tokenizer::load /
// Tokenizer::encode / decode_gpt2_bytes) and src/runtime/transcribe-unicode.cpp
// (granite_split_offsets / pretokenize_granite). Not the engine TokenizerHub:
// the hub has no "granite" pretokenizer (it falls back to the GPT-2 regex),
// and the granite split is load-bearing - convert-granite.py writes
// tokenizer.ggml.pre = "granite" for 4.0-1b / 4.1-2b, whose BPE has a "?\n"
// merge (id 5380) that HF never applies because its regex engine emits "?"
// and "\n" as separate pretokens. Getting that wrong changes the prompt ids
// right after "format?" and with them the transcript. The -plus variant
// writes "gpt2", served by llama.cpp's HF-verified GPT-2 split.
//
// Unicode categories and UTF-8 / byte-level helpers come from the vendored
// llama.cpp tokenizer (external/llama_tokenizer/unicode.h) - the same tables
// the arch's transcribe-unicode-data.cpp was copied from.

#include "engine/models/granite_speech/model.h"

#include "engine/framework/assets/gguf_metadata.h"

#include "unicode.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::granite_speech {
namespace {

// U+001F joins the two sides of a merge-rank key; never part of a
// byte-level token (arch transcribe-tokenizer.cpp k_merge_sep).
constexpr char kMergeSep = '\x1F';
constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

// HF ByteLevel GPT-2 split. llama.cpp's unicode_regex_split recognises this
// exact string and takes its hand-written GPT-2 path (same as the engine
// TokenizerHub's k_gpt2_split_regex and the arch's gpt2_split_offsets).
constexpr const char * kGpt2SplitRegex =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";

uint32_t tolower_ascii(uint32_t c) {
    return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

bool has_category(const unicode_cpt_flags & f) {
    return f.category_flag() != 0;
}

// Verbatim port of transcribe-unicode.cpp:granite_split_offsets. Same as the
// Qwen2 split except the symbol-run alternative does NOT swallow a trailing
// [\r\n]* (the HF tokenizers crate's reading of the granite regex).
std::vector<size_t> granite_split_offsets(const std::vector<uint32_t> & cpts) {
    std::vector<size_t> out;
    const size_t begin = 0;
    const size_t end = cpts.size();

    auto get_cpt = [&](size_t p) -> uint32_t {
        return (begin <= p && p < end) ? cpts[p] : kOutOfRange;
    };
    auto get_flags = [&](size_t p) -> unicode_cpt_flags {
        return (begin <= p && p < end) ? unicode_cpt_flags_from_cpt(cpts[p]) : unicode_cpt_flags{};
    };

    size_t prev = begin;
    auto push_upto = [&](size_t p) {
        if (p > prev) {
            out.push_back(p);
            prev = p;
        }
    };

    size_t pos = begin;
    while (pos < end) {
        const uint32_t cpt = get_cpt(pos);
        const unicode_cpt_flags flags = get_flags(pos);

        // Contractions (case-insensitive).
        if (cpt == '\'' && pos + 1 < end) {
            const uint32_t c1 = tolower_ascii(get_cpt(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += 2;
                push_upto(pos);
                continue;
            }
            if (pos + 2 < end) {
                const uint32_t c2 = tolower_ascii(get_cpt(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos += 3;
                    push_upto(pos);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]? \p{L}+
        if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
            const bool self_is_letter = flags.is_letter;
            const bool next_is_letter = get_flags(pos + 1).is_letter;
            if (self_is_letter || next_is_letter) {
                pos++;
                while (get_flags(pos).is_letter) {
                    pos++;
                }
                push_upto(pos);
                continue;
            }
        }

        // \p{N}{1,3}
        if (flags.is_number) {
            size_t n = 0;
            while (n < 3 && get_flags(pos + n).is_number) {
                ++n;
            }
            pos += n;
            push_upto(pos);
            continue;
        }

        // " "? [^\s\p{L}\p{N}]+ (no trailing [\r\n]*: the granite divergence).
        {
            const unicode_cpt_flags f2 = (cpt == ' ') ? get_flags(pos + 1) : flags;
            if (!(f2.is_whitespace || f2.is_letter || f2.is_number) && has_category(flags)) {
                pos += (cpt == ' ' ? 1 : 0);
                while (true) {
                    const unicode_cpt_flags fx = get_flags(pos);
                    if (fx.is_whitespace || fx.is_letter || fx.is_number || !has_category(fx)) {
                        break;
                    }
                    pos++;
                }
                push_upto(pos);
                continue;
            }
        }

        // \s*[\r\n]+ | \s+(?!\S) | \s+
        size_t ws_count = 0;
        size_t last_after_nl = 0;
        while (get_flags(pos + ws_count).is_whitespace) {
            const uint32_t cw = get_cpt(pos + ws_count);
            if (cw == '\r' || cw == '\n') {
                last_after_nl = pos + ws_count + 1;
            }
            ws_count++;
        }
        if (last_after_nl > 0) {
            pos = last_after_nl;
            push_upto(pos);
            continue;
        }
        if (ws_count > 1 && get_cpt(pos + ws_count) != kOutOfRange) {
            pos += ws_count - 1;
            push_upto(pos);
            continue;
        }
        if (ws_count > 0) {
            pos += ws_count;
            push_upto(pos);
            continue;
        }

        // Fallback: one codepoint as its own pretoken.
        pos++;
        push_upto(pos);
    }
    return out;
}

// arch transcribe-tokenizer.cpp Symbol / Bigram / BigramGreater.
struct Symbol {
    const char * text = nullptr;
    size_t n = 0;
    int prev = -1;
    int next = -1;
};

struct Bigram {
    int left = -1;
    int right = -1;
    std::string text;
    int rank = 0;
    size_t size = 0;
};

struct BigramGreater {
    bool operator()(const Bigram & a, const Bigram & b) const {
        if (a.rank != b.rank) {
            return a.rank > b.rank;
        }
        return a.left > b.left;
    }
};

}  // namespace

void GraniteSpeechTokenizer::load(const std::filesystem::path & gguf_path) {
    const auto meta = assets::GgufMetadata::open(gguf_path);
    const auto model = meta.find_string("tokenizer.ggml.model").value_or("");
    if (model != "gpt2") {
        throw std::runtime_error("Granite Speech GGUF tokenizer.ggml.model must be \"gpt2\", got \"" + model + "\"");
    }
    pre_ = meta.find_string("tokenizer.ggml.pre").value_or("");
    if (pre_ != "granite" && pre_ != "gpt2") {
        // The arch falls back to the Qwen2 split for other flavours; no
        // granite checkpoint ships one, so refuse rather than guess.
        throw std::runtime_error("Granite Speech GGUF has unsupported tokenizer.ggml.pre \"" + pre_ +
                                 "\" (expected \"granite\" or \"gpt2\")");
    }

    auto tokens = meta.find_string_array("tokenizer.ggml.tokens");
    if (!tokens.has_value() || tokens->empty()) {
        throw std::runtime_error("Granite Speech GGUF has no tokenizer.ggml.tokens");
    }
    tokens_ = std::move(*tokens);
    piece_to_id_.clear();
    piece_to_id_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); ++i) {
        piece_to_id_.emplace(tokens_[i], static_cast<int32_t>(i));  // first wins (arch emplace)
    }

    const auto merges = meta.find_string_array("tokenizer.ggml.merges");
    if (!merges.has_value() || merges->empty()) {
        throw std::runtime_error("Granite Speech GGUF has no tokenizer.ggml.merges; the prompt cannot be encoded");
    }
    merge_rank_.clear();
    merge_rank_.reserve(merges->size() * 2);
    for (size_t i = 0; i < merges->size(); ++i) {
        const std::string & line = (*merges)[i];
        // Scan from 1 so a merge whose left side is " " parses (llama.cpp).
        const size_t sp = line.find(' ', 1);
        if (sp == std::string::npos) {
            throw std::runtime_error("Granite Speech GGUF merge " + std::to_string(i) + " has no separator: " + line);
        }
        std::string key;
        key.reserve(line.size());
        key.append(line, 0, sp);
        key.push_back(kMergeSep);
        key.append(line, sp + 1, std::string::npos);
        merge_rank_.emplace(std::move(key), static_cast<int32_t>(i));
    }

    unicode_to_byte_.clear();
    for (int b = 0; b < 256; ++b) {
        unicode_to_byte_.emplace(unicode_byte_to_utf8(static_cast<uint8_t>(b)), static_cast<uint8_t>(b));
    }

    const auto eos = meta.find_int("tokenizer.ggml.eos_token_id");
    if (!eos.has_value() || *eos < 0 || *eos >= static_cast<int64_t>(tokens_.size())) {
        throw std::runtime_error("Granite Speech GGUF has no valid tokenizer.ggml.eos_token_id");
    }
    eos_id_ = static_cast<int32_t>(*eos);
}

int32_t GraniteSpeechTokenizer::find(const std::string & piece) const {
    const auto it = piece_to_id_.find(piece);
    return it == piece_to_id_.end() ? -1 : it->second;
}

std::vector<std::string> GraniteSpeechTokenizer::pretokenize(const std::string & text) const {
    if (pre_ == "gpt2") {
        // byte_encode=true: every piece comes back through bytes_to_unicode.
        return unicode_regex_split(text, {kGpt2SplitRegex}, /*byte_encode=*/true);
    }
    // arch transcribe-unicode.cpp:pretokenize_granite.
    std::vector<std::string> out;
    const auto cpts = unicode_cpts_from_utf8(text);
    const auto ends = granite_split_offsets(cpts);
    out.reserve(ends.size());
    size_t prev = 0;
    for (size_t e : ends) {
        std::string encoded;
        encoded.reserve((e - prev) * 2);
        for (size_t i = prev; i < e; ++i) {
            for (const char c : unicode_cpt_to_utf8(cpts[i])) {
                encoded += unicode_byte_to_utf8(static_cast<uint8_t>(c));
            }
        }
        out.emplace_back(std::move(encoded));
        prev = e;
    }
    return out;
}

// arch Tokenizer::encode, "gpt2" branch: per pretoken, seed one symbol per
// UTF-8 codepoint of the byte-encoded word, merge the lowest-ranked bigram
// (leftmost on ties) until none applies, then map pieces to ids with the
// single-codepoint fallback.
std::vector<int32_t> GraniteSpeechTokenizer::encode(const std::string & text) const {
    std::vector<int32_t> out_ids;
    if (text.empty()) {
        return out_ids;
    }
    const auto words = pretokenize(text);
    out_ids.reserve(words.size() * 2);

    std::vector<Symbol> symbols;
    symbols.reserve(64);
    std::priority_queue<Bigram, std::vector<Bigram>, BigramGreater> queue;

    auto rank_of = [&](const std::string & left, const std::string & right) -> int {
        std::string key;
        key.reserve(left.size() + 1 + right.size());
        key.append(left);
        key.push_back(kMergeSep);
        key.append(right);
        const auto it = merge_rank_.find(key);
        return it == merge_rank_.end() ? -1 : it->second;
    };
    auto push_bigram = [&](int l, int r) {
        if (l < 0 || r < 0) {
            return;
        }
        const std::string left(symbols[static_cast<size_t>(l)].text, symbols[static_cast<size_t>(l)].n);
        const std::string right(symbols[static_cast<size_t>(r)].text, symbols[static_cast<size_t>(r)].n);
        const int rank = rank_of(left, right);
        if (rank < 0) {
            return;
        }
        Bigram bg;
        bg.left = l;
        bg.right = r;
        bg.text = left + right;
        bg.rank = rank;
        bg.size = left.size() + right.size();
        queue.push(std::move(bg));
    };

    for (const auto & word : words) {
        while (!queue.empty()) {
            queue.pop();
        }
        symbols.clear();
        int index = 0;
        size_t offset = 0;
        while (offset < word.size()) {
            const size_t clen = std::min(word.size() - offset, unicode_len_utf8(word[offset]));
            Symbol s;
            s.text = word.data() + offset;
            s.n = clen;
            s.prev = index - 1;
            s.next = (offset + clen == word.size()) ? -1 : index + 1;
            symbols.push_back(s);
            offset += clen;
            ++index;
        }
        for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
            push_bigram(i - 1, i);
        }
        while (!queue.empty()) {
            Bigram bg = queue.top();
            queue.pop();
            Symbol & left = symbols[static_cast<size_t>(bg.left)];
            Symbol & right = symbols[static_cast<size_t>(bg.right)];
            if (left.n == 0 || right.n == 0) {
                continue;
            }
            // Stale entry: a side was merged into something else since the push.
            if (left.n + right.n != bg.size || std::memcmp(left.text, bg.text.data(), left.n) != 0 ||
                std::memcmp(right.text, bg.text.data() + left.n, right.n) != 0) {
                continue;
            }
            left.n += right.n;
            right.n = 0;
            left.next = right.next;
            if (right.next >= 0) {
                symbols[static_cast<size_t>(right.next)].prev = bg.left;
            }
            push_bigram(left.prev, bg.left);
            push_bigram(bg.left, left.next);
        }
        if (symbols.empty()) {
            continue;
        }
        for (int i = 0; i != -1; i = symbols[static_cast<size_t>(i)].next) {
            const Symbol & s = symbols[static_cast<size_t>(i)];
            if (s.n == 0) {
                continue;
            }
            const std::string piece(s.text, s.n);
            if (const auto it = piece_to_id_.find(piece); it != piece_to_id_.end()) {
                out_ids.push_back(it->second);
                continue;
            }
            // Not a whole-vocab token: one id per byte-level codepoint.
            for (size_t j = 0; j < piece.size();) {
                const size_t clen = std::min(piece.size() - j, unicode_len_utf8(piece[j]));
                const std::string sub(piece, j, clen);
                const auto it2 = piece_to_id_.find(sub);
                if (it2 == piece_to_id_.end()) {
                    throw std::runtime_error("Granite Speech tokenizer: byte-fallback piece not in vocab");
                }
                out_ids.push_back(it2->second);
                j += clen;
            }
        }
    }
    return out_ids;
}

// arch transcribe-tokenizer.cpp:decode_gpt2_bytes: byte-level glyphs map
// back to their byte; anything else (a special token's literal) passes
// through as raw UTF-8.
std::string GraniteSpeechTokenizer::decode(const int32_t * ids, size_t count) const {
    std::string out;
    if (ids == nullptr || count == 0) {
        return out;
    }
    out.reserve(count * 2);
    std::string one_cpt;
    one_cpt.reserve(4);
    for (size_t i = 0; i < count; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
            continue;
        }
        const std::string & p = tokens_[static_cast<size_t>(id)];
        size_t j = 0;
        while (j < p.size()) {
            const size_t w = std::min(p.size() - j, unicode_len_utf8(p[j]));
            one_cpt.assign(p.data() + j, w);
            if (const auto it = unicode_to_byte_.find(one_cpt); it != unicode_to_byte_.end()) {
                out.push_back(static_cast<char>(it->second));
            } else {
                out.append(p, j, w);
            }
            j += w;
        }
    }
    return out;
}

}  // namespace engine::models::granite_speech
