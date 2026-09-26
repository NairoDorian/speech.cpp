// engine/models/voxtral/tokenizer.cpp - the arch's byte-level BPE tokenizer
// for the Voxtral GGUF vocabulary (see VoxtralTokenizer in assets.h).
//
// Ported from src/runtime/transcribe-tokenizer.cpp (Tokenizer::load for a
// "gpt2" model, Tokenizer::decode's Gpt2ByteUnicode mode, Tokenizer::encode's
// merge loop) and src/runtime/transcribe-unicode.cpp (qwen2_split_offsets,
// pretokenize_qwen2). Code-point categories come from the llama.cpp unicode
// tables vendored in external/llama_tokenizer (the same tables the engine
// TokenizerHub uses), which is the only place a Unicode-table difference to
// the arch could arise; the split and merge logic is the arch's verbatim.

#include "engine/models/voxtral/assets.h"

#include "unicode.h"  // external/llama_tokenizer

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::voxtral {

namespace {

constexpr char kMergeSep = '\x1F';
constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

// GPT-2 bytes_to_unicode tables (byte -> UTF-8 glyph, glyph -> byte), built
// from llama.cpp's unicode_byte_to_utf8 (same mapping as the arch's
// transcribe::unicode::byte_to_unicode / unicode_to_byte).
struct ByteMaps {
  std::string byte_to_glyph[256];
  std::unordered_map<std::string, uint8_t> glyph_to_byte;

  ByteMaps() {
    glyph_to_byte.reserve(512);
    for (int b = 0; b < 256; ++b) {
      byte_to_glyph[b] = unicode_byte_to_utf8(static_cast<uint8_t>(b));
      glyph_to_byte.emplace(byte_to_glyph[b], static_cast<uint8_t>(b));
    }
  }
};

const ByteMaps &byte_maps() {
  static const ByteMaps maps;
  return maps;
}

uint32_t tolower_ascii(uint32_t cpt) {
  if (cpt >= 'A' && cpt <= 'Z') {
    return cpt + ('a' - 'A');
  }
  return cpt;
}

// transcribe-unicode.cpp: qwen2_split_offsets, verbatim over llama.cpp's
// code-point flags. Returns pretoken END offsets (in code points).
std::vector<size_t> qwen2_split_offsets(const std::vector<uint32_t> &cpts) {
  std::vector<size_t> out;
  const size_t begin = 0;
  const size_t end = cpts.size();

  auto get_cpt = [&](size_t p) -> uint32_t { return (begin <= p && p < end) ? cpts[p] : kOutOfRange; };
  auto get_flags = [&](size_t p) -> unicode_cpt_flags {
    return (begin <= p && p < end) ? unicode_cpt_flags_from_cpt(cpts[p]) : unicode_cpt_flags{};
  };
  // CptFlags::has_category(): any category bit set (out-of-range reads none).
  auto has_category = [](const unicode_cpt_flags &f) {
    return (f.as_uint() & unicode_cpt_flags::MASK_CATEGORIES) != 0;
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

    // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
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

    // \p{N}   (Qwen2 is single-digit-per-pretoken)
    if (flags.is_number) {
      pos++;
      push_upto(pos);
      continue;
    }

    // " "? [^\s\p{L}\p{N}]+ [\r\n]*
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
        uint32_t cn = get_cpt(pos);
        while (cn == '\r' || cn == '\n') {
          pos++;
          cn = get_cpt(pos);
        }
        push_upto(pos);
        continue;
      }
    }

    // Whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
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

    // No match: one code point as its own pretoken (as the arch / llama.cpp).
    pos++;
    push_upto(pos);
  }
  return out;
}

// pretokenize_qwen2: split, then byte-encode every UTF-8 byte of each pretoken.
std::vector<std::string> pretokenize_qwen2(std::string_view text) {
  std::vector<std::string> out;
  if (text.empty()) {
    return out;
  }
  const auto cpts = unicode_cpts_from_utf8(std::string(text));
  const auto ends = qwen2_split_offsets(cpts);
  const auto &maps = byte_maps();
  out.reserve(ends.size());
  size_t prev = 0;
  for (size_t e : ends) {
    std::string encoded;
    encoded.reserve((e - prev) * 2);
    for (size_t i = prev; i < e; ++i) {
      const std::string u = unicode_cpt_to_utf8(cpts[i]);
      for (char c : u) {
        encoded += maps.byte_to_glyph[static_cast<uint8_t>(c)];
      }
    }
    out.emplace_back(std::move(encoded));
    prev = e;
  }
  return out;
}

// The arch's merge-loop workspace (transcribe-tokenizer.cpp Symbol / Bigram).
struct Symbol {
  const char *text = nullptr;
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
  bool operator()(const Bigram &a, const Bigram &b) const {
    // Lower rank wins; on ties, lower `left` index wins.
    if (a.rank != b.rank) {
      return a.rank > b.rank;
    }
    return a.left > b.left;
  }
};

}  // namespace

VoxtralTokenizer::VoxtralTokenizer(std::vector<std::string> tokens,
                                   const std::vector<std::string> &merges, std::string pre,
                                   int32_t bos_id, int32_t eos_id)
    : tokens_(std::move(tokens)), pre_(std::move(pre)), bos_id_(bos_id), eos_id_(eos_id) {
  // First occurrence wins (the arch's piece_to_id_.emplace order).
  piece_to_id_.reserve(tokens_.size() * 2);
  for (size_t i = 0; i < tokens_.size(); ++i) {
    piece_to_id_.emplace(tokens_[i], static_cast<int32_t>(i));
  }
  merge_rank_.reserve(merges.size() * 2);
  for (size_t i = 0; i < merges.size(); ++i) {
    const std::string &line = merges[i];
    // "left<space>right"; scan from position 1 so a left side that is a
    // single space still parses (llama.cpp's bpe_ranks loader, as the arch).
    const size_t sp = line.find(' ', 1);
    if (sp == std::string::npos) {
      throw std::runtime_error("voxtral: tokenizer merge " + std::to_string(i) +
                               " has no space separator: \"" + line + "\"");
    }
    std::string key;
    key.reserve(line.size());
    key.append(line, 0, sp);
    key.push_back(kMergeSep);
    key.append(line, sp + 1, std::string::npos);
    merge_rank_.emplace(std::move(key), static_cast<int32_t>(i));
  }
}

int32_t VoxtralTokenizer::find(std::string_view piece) const {
  const auto it = piece_to_id_.find(std::string(piece));
  return it != piece_to_id_.end() ? it->second : -1;
}

std::string VoxtralTokenizer::decode(const std::vector<int32_t> &ids) const {
  // decode_gpt2_bytes: invert the byte-unicode map per code point; a code
  // point that is not a byte glyph passes through as raw UTF-8.
  const auto &maps = byte_maps();
  std::string out;
  out.reserve(ids.size() * 2);
  std::string one_cpt;
  one_cpt.reserve(4);
  for (const int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
      continue;
    }
    const std::string &p = tokens_[static_cast<size_t>(id)];
    size_t j = 0;
    while (j < p.size()) {
      const size_t w = std::min(p.size() - j, unicode_len_utf8(p[j]));
      one_cpt.assign(p.data() + j, w);
      const auto it = maps.glyph_to_byte.find(one_cpt);
      if (it != maps.glyph_to_byte.end()) {
        out.push_back(static_cast<char>(it->second));
      } else {
        out.append(p, j, w);
      }
      j += w;
    }
  }
  return out;
}

std::vector<int32_t> VoxtralTokenizer::encode(std::string_view text) const {
  std::vector<int32_t> out_ids;
  if (merge_rank_.empty()) {
    throw std::runtime_error(
        "voxtral: tokenizer loaded without tokenizer.ggml.merges; encode unavailable");
  }
  if (text.empty()) {
    return out_ids;
  }

  // Pretokenizer: the arch dispatches gpt2 / granite / (everything else ->
  // qwen2). A Voxtral GGUF says "tekken", which lands on qwen2; "gpt2" or
  // "granite" never reach this family (the converter writes "tekken").
  if (pre_ == "gpt2" || pre_ == "granite") {
    throw std::runtime_error("voxtral: unsupported tokenizer.ggml.pre \"" + pre_ +
                             "\" (the Voxtral vocabulary is tekken)");
  }
  const std::vector<std::string> words = pretokenize_qwen2(text);

  out_ids.reserve(words.size() * 2);
  std::vector<Symbol> symbols;
  symbols.reserve(64);
  std::priority_queue<Bigram, std::vector<Bigram>, BigramGreater> queue;

  auto rank_of = [&](const std::string &left, const std::string &right) -> int {
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

  for (const auto &word : words) {
    while (!queue.empty()) {
      queue.pop();
    }
    symbols.clear();

    // Seed: one symbol per UTF-8 code point of the byte-encoded word.
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
      Symbol &left = symbols[static_cast<size_t>(bg.left)];
      Symbol &right = symbols[static_cast<size_t>(bg.right)];
      if (left.n == 0 || right.n == 0) {
        continue;
      }
      // Stale-entry checks (a side merged away since the push).
      if (left.n + right.n != bg.size) {
        continue;
      }
      if (std::memcmp(left.text, bg.text.data(), left.n) != 0) {
        continue;
      }
      if (std::memcmp(right.text, bg.text.data() + left.n, right.n) != 0) {
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

    for (int i = 0; i != -1; i = symbols[static_cast<size_t>(i)].next) {
      const Symbol &s = symbols[static_cast<size_t>(i)];
      if (s.n == 0) {
        continue;
      }
      const std::string piece(s.text, s.n);
      const auto it = piece_to_id_.find(piece);
      if (it != piece_to_id_.end()) {
        out_ids.push_back(it->second);
        continue;
      }
      // BPE did not land on a whole-vocab token: one id per code point.
      for (size_t j = 0; j < piece.size();) {
        const size_t clen = std::min(piece.size() - j, unicode_len_utf8(piece[j]));
        const std::string sub(piece, j, clen);
        const auto it2 = piece_to_id_.find(sub);
        if (it2 == piece_to_id_.end()) {
          throw std::runtime_error("voxtral: tokenizer byte-fallback piece not in vocab (\"" + sub +
                                   "\")");
        }
        out_ids.push_back(it2->second);
        j += clen;
      }
    }
  }
  return out_ids;
}

}  // namespace engine::models::voxtral
