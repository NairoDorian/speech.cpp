// engine/models/medasr/decoding.cpp - greedy CTC decode and token rows for the
// native engine MedASR package (see decoding.h).

#include "engine/models/medasr/decoding.h"

#include <cstddef>

namespace engine::models::medasr {

CtcGreedyResult ctc_greedy_decode(const float *logits, int vocab, int n_frames,
                                  int blank_id) {
  CtcGreedyResult out;
  if (logits == nullptr || vocab <= 0 || n_frames <= 0) {
    return out;
  }
  int prev = -1;
  for (int t = 0; t < n_frames; ++t) {
    const float *row = logits + static_cast<size_t>(t) * static_cast<size_t>(vocab);
    int best = 0;
    float best_v = row[0];
    for (int v = 1; v < vocab; ++v) {
      if (row[v] > best_v) {
        best_v = row[v];
        best = v;
      }
    }
    if (best == blank_id) {
      prev = -1;
      continue;
    }
    if (best == prev) {
      continue;
    }
    prev = best;
    // skip_special_tokens=True: <s> = 1, </s> = 2, <unk> = 3 (LASR's fixed
    // layout; the arch hard-codes these ids too).
    if (best == 1 || best == 2 || best == 3) {
      continue;
    }
    out.tokens.push_back(best);
    out.frames.push_back(t);
  }
  return out;
}

std::vector<CtcToken> make_ctc_tokens(const CtcGreedyResult &decoded) {
  std::vector<CtcToken> tokens;
  tokens.reserve(decoded.tokens.size());
  for (size_t i = 0; i < decoded.tokens.size(); ++i) {
    CtcToken token;
    token.id = decoded.tokens[i];
    token.t0_ms = static_cast<int64_t>(decoded.frames[i]) * kCtcFrameMs;
    token.t1_ms = token.t0_ms + kCtcFrameMs;
    tokens.push_back(token);
  }
  return tokens;
}

} // namespace engine::models::medasr
