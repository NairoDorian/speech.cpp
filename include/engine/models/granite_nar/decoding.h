#pragma once

// engine/models/granite_nar/decoding.h - host-side pieces of the Granite
// Speech NAR forward: pure functions (no model, no ggml), so each rule is
// testable in isolation. Every function names the arch function it ports.

#include <cstdint>
#include <vector>

namespace engine::models::granite_nar {

// arch encoder.cpp compute_mel_encoder_input (the stacking half): drop a
// trailing odd frame, then stack frame pairs into rows of 2 * n_mels
// ([mel(2t) | mel(2t+1)]). `mel_major` is row-major [n_mels][n_frames] (the
// frontend's layout). Returns T_enc (0 when fewer than two frames); `out` is
// frame-major [T_enc][2 * n_mels], the byte layout of the ggml
// [input_dim, T_enc] encoder input.
int stack_mel_frames(const std::vector<float> &mel_major, int n_mels, int n_frames,
                     std::vector<float> &out);

// transcribe.cpp 585b98f7 encoder.cpp precompute_pos_rows: one rel_pos_emb
// row index per relative offset for the Shaw SKEW path. Slot i (i in
// [0, 2*ctx - 1)) carries clamp(ctx - 1 - i, +/- ctx) + max_pos_emb, i.e. the
// offset for d = key - query + ctx - 1 (the layout rel_shift rotates).
std::vector<int32_t> precompute_pos_rows(int context_size, int max_pos_emb);

// Pre-585b98f7 encoder.cpp precompute_attention_dists (the DIRECT path, kept
// for the `granite_nar.shaw_bias=direct` diagnostic): [ctx * ctx] indices,
// value at (c, r) = clamp(c - r, +/- ctx) + max_pos_emb.
std::vector<int32_t> precompute_attention_dists(int context_size, int max_pos_emb);

// arch encoder.cpp precompute_last_block_mask: [ctx * ctx] additive mask,
// -inf on key columns >= remainder (all zero when remainder <= 0 or >= ctx).
std::vector<float> precompute_last_block_mask(int context_size, int t_enc_remainder);

// The [ctx, ctx, n_blocks] per-block mask the arch's run() uploads: zero
// everywhere except the last block's plane (the pad-key columns).
std::vector<float> build_block_mask(int context_size, int n_blocks, int last_block_rem);

// transcribe.cpp HEAD (b174a427) model.cpp compute_bpe_ctc_initial_hypothesis,
// host half: the posterior-weighted mean of the FINAL encoder layer over one
// pooling window (weights = non_blank / sum over the window). Returns false
// (and leaves `dst` untouched) for an all-blank window (sum <= 1e-9), which
// the decode then treats as blank. `enc_cat` is host-row-major
// [T_enc][cat_h]; the final layer's state sits at channel `final_offset`.
bool pool_window_hidden(const float *enc_cat, int64_t cat_h, int64_t final_offset, int hidden,
                        const float *non_blank, int t0, int t1, float *dst);

// Greedy BPE-CTC collapse over the per-window argmaxes (same function):
// drop blank_id, collapse repeats (a blank resets the tracker), and map the
// channel to an LLM id (blank_id == 0: synthetic blank channel, id = argmax - 1;
// otherwise channels ARE LLM ids). `prev` carries across chunks.
void collapse_bpe_ctc(const int32_t *window_argmax, const uint8_t *valid, int n, int blank_id,
                      int &prev, std::vector<int32_t> &out_ids);

// arch decoder.cpp add_insertion_slots: eos between and around every token,
// length max(2n + 1, 8), hypothesis tokens at the odd positions.
void add_insertion_slots(const std::vector<int32_t> &hyp_ids, int32_t eos_id,
                         std::vector<int32_t> &out);

// arch decoder.cpp argmax_collapse_drop_eos: per-row argmax (first maximum
// wins), drop eos (it still becomes `prev`), collapse consecutive repeats.
// `text_logits` is row-major [n_text][vocab].
void argmax_collapse_drop_eos(const std::vector<float> &text_logits, int vocab, int n_text,
                              int32_t eos_id, std::vector<int32_t> &out_ids);

} // namespace engine::models::granite_nar
