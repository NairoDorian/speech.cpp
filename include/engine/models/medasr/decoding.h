#pragma once

// engine/models/medasr/decoding.h - host-side CTC decode for the native engine
// MedASR package. Pure functions (no model, no ggml), so the decode rules are
// testable in isolation.
//
// Ported from src/runtime/arch/medasr/model.cpp (decode_ctc_greedy and the
// token-row construction in run() / decode_one_utterance()). Text joining is
// TokenizerHub's (SentencePiece piece join with <0xNN> byte fallback for the
// "bpe" vocab the MedASR converter writes).

#include <cstdint>
#include <vector>

namespace engine::models::medasr {

// Milliseconds per CTC frame the arch stamps on every token. The arch
// hard-codes 40 (16 kHz / hop 160 = 100 fps, 4x subsampling = 25 fps) rather
// than deriving it; kept verbatim so token timings match it exactly.
constexpr int64_t kCtcFrameMs = 40;

struct CtcGreedyResult {
  std::vector<int32_t> tokens; // kept ids, in order
  std::vector<int32_t> frames; // encoder frame index of each kept id
};

// Greedy CTC: per-frame argmax (first maximum wins), collapse repeats, drop the
// blank, and drop the LASR special ids 1 (<s>), 2 (</s>), 3 (<unk>) - the
// reference's batch_decode(skip_special_tokens=True). A blank resets the
// repeat tracker; a dropped special id does not (it still counts as `prev`).
// `logits` is frame-major: frame t is `vocab` contiguous floats at t * vocab.
CtcGreedyResult ctc_greedy_decode(const float *logits, int vocab, int n_frames,
                                  int blank_id);

// One CTC token row with the arch's timing: [frame * 40, frame * 40 + 40) ms.
// The arch publishes no probability (p = 0) and no word grouping
// (word_index = -1); its row text is the raw vocabulary piece.
struct CtcToken {
  int32_t id = 0;
  int64_t t0_ms = 0;
  int64_t t1_ms = 0;
};

std::vector<CtcToken> make_ctc_tokens(const CtcGreedyResult &decoded);

} // namespace engine::models::medasr
