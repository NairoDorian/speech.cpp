// engine/models/granite_nar/decoding.cpp - host-side pieces of the Granite
// Speech NAR forward (see decoding.h). Ported line for line from
// src/runtime/arch/granite_nar/{encoder,decoder,model}.cpp and, where the
// parent moved on, from transcribe.cpp HEAD (585b98f7 precompute_pos_rows,
// b174a427 bounded BPE-CTC).

#include "engine/models/granite_nar/decoding.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

namespace engine::models::granite_nar {

int stack_mel_frames(const std::vector<float> &mel_major, int n_mels, int n_frames,
                     std::vector<float> &out) {
  out.clear();
  if (n_mels <= 0 || n_frames <= 0) {
    return 0;
  }
  // The row stride is the frontend's emitted frame count (the arch derives it
  // as raw.size() / n_mels); the stacking reads the first even count of it.
  const int stride = static_cast<int>(mel_major.size() / static_cast<size_t>(n_mels));
  int frames = n_frames;
  if ((frames % 2) == 1) {
    --frames;
  }
  const int t_enc = frames / 2;
  if (t_enc <= 0 || stride < frames) {
    return 0;
  }
  const int input_dim = 2 * n_mels;
  out.assign(static_cast<size_t>(t_enc) * input_dim, 0.0f);
  for (int t = 0; t < t_enc; ++t) {
    float *dst = out.data() + static_cast<size_t>(t) * input_dim;
    for (int m = 0; m < n_mels; ++m) {
      dst[m] = mel_major[static_cast<size_t>(m) * stride + 2 * t];
      dst[m + n_mels] = mel_major[static_cast<size_t>(m) * stride + 2 * t + 1];
    }
  }
  return t_enc;
}

std::vector<int32_t> precompute_pos_rows(int context_size, int max_pos_emb) {
  const int n = 2 * context_size - 1;
  std::vector<int32_t> rows(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    int d = context_size - 1 - i;
    if (d < -context_size) {
      d = -context_size;
    }
    if (d > context_size) {
      d = context_size;
    }
    rows[static_cast<size_t>(i)] = static_cast<int32_t>(d + max_pos_emb);
  }
  return rows;
}

std::vector<int32_t> precompute_attention_dists(int context_size, int max_pos_emb) {
  std::vector<int32_t> dists(static_cast<size_t>(context_size) * context_size);
  for (int c = 0; c < context_size; ++c) {
    for (int r = 0; r < context_size; ++r) {
      int d = c - r;
      if (d < -context_size) {
        d = -context_size;
      }
      if (d > context_size) {
        d = context_size;
      }
      dists[static_cast<size_t>(c) * context_size + r] = static_cast<int32_t>(d + max_pos_emb);
    }
  }
  return dists;
}

std::vector<float> precompute_last_block_mask(int context_size, int t_enc_remainder) {
  std::vector<float> mask(static_cast<size_t>(context_size) * context_size, 0.0f);
  if (t_enc_remainder <= 0 || t_enc_remainder >= context_size) {
    return mask;
  }
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (int q = 0; q < context_size; ++q) {
    for (int k = t_enc_remainder; k < context_size; ++k) {
      mask[static_cast<size_t>(q) * context_size + k] = neg_inf;
    }
  }
  return mask;
}

std::vector<float> build_block_mask(int context_size, int n_blocks, int last_block_rem) {
  // arch model.cpp run(): "Upload encoder inputs" block.
  const size_t plane = static_cast<size_t>(context_size) * context_size;
  std::vector<float> mask(plane * static_cast<size_t>(n_blocks), 0.0f);
  if (n_blocks > 0 && last_block_rem > 0 && last_block_rem < context_size) {
    const std::vector<float> last = precompute_last_block_mask(context_size, last_block_rem);
    std::memcpy(mask.data() + plane * static_cast<size_t>(n_blocks - 1), last.data(),
                plane * sizeof(float));
  }
  return mask;
}

bool pool_window_hidden(const float *enc_cat, int64_t cat_h, int64_t final_offset, int hidden,
                        const float *non_blank, int t0, int t1, float *dst) {
  float total = 0.0f;
  for (int t = t0; t < t1; ++t) {
    total += non_blank[t];
  }
  if (total <= 1e-9f) {
    return false;
  }
  for (int t = t0; t < t1; ++t) {
    const float wt = non_blank[t] / total;
    const float *src = enc_cat + static_cast<size_t>(t) * static_cast<size_t>(cat_h) +
                       static_cast<size_t>(final_offset);
    for (int h = 0; h < hidden; ++h) {
      dst[h] += wt * src[h];
    }
  }
  return true;
}

void collapse_bpe_ctc(const int32_t *window_argmax, const uint8_t *valid, int n, int blank_id,
                      int &prev, std::vector<int32_t> &out_ids) {
  for (int w = 0; w < n; ++w) {
    const int argmax = valid[w] ? window_argmax[w] : blank_id;
    if (argmax != blank_id && argmax != prev) {
      const int shift = (blank_id == 0) ? 1 : 0;
      out_ids.push_back(argmax - shift);
    }
    prev = argmax;
  }
}

void add_insertion_slots(const std::vector<int32_t> &hyp_ids, int32_t eos_id,
                         std::vector<int32_t> &out) {
  const int n = static_cast<int>(hyp_ids.size());
  const int total_len = std::max(2 * n + 1, 8);
  out.assign(static_cast<size_t>(total_len), eos_id);
  for (int i = 0; i < n; ++i) {
    out[static_cast<size_t>(2 * i + 1)] = hyp_ids[static_cast<size_t>(i)];
  }
}

void argmax_collapse_drop_eos(const std::vector<float> &text_logits, int vocab, int n_text,
                              int32_t eos_id, std::vector<int32_t> &out_ids) {
  out_ids.clear();
  int32_t prev = -1;
  for (int i = 0; i < n_text; ++i) {
    const float *row = text_logits.data() + static_cast<size_t>(i) * vocab;
    int32_t best = 0;
    float best_v = row[0];
    for (int v = 1; v < vocab; ++v) {
      if (row[v] > best_v) {
        best_v = row[v];
        best = v;
      }
    }
    if (best == eos_id) {
      prev = best;
      continue;
    }
    if (best == prev) {
      continue;
    }
    out_ids.push_back(best);
    prev = best;
  }
}

} // namespace engine::models::granite_nar
