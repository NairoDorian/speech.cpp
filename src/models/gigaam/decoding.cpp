// engine/models/gigaam/decoding.cpp - GigaAM RNN-T / CTC greedy decode on the
// host.
//
// Ported from src/runtime/arch/gigaam/decoder.cpp. Only the non-BLAS branches
// are carried over: speech.cpp never defines TRANSCRIBE_HAS_BLAS, so they are
// the arch's live code, and keeping their exact f32 accumulation order is what
// makes the transcripts identical. The only additions are the RunControl poll
// hook at decode-step boundaries and exceptions in place of status codes.

#include "engine/models/gigaam/decoding.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::gigaam {

namespace {

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y = W @ x + b, W row-major [out, in]; b may be null.
void matvec_add(const float *W, const float *b, const float *x, int out, int in_dim, float *y) {
  for (int i = 0; i < out; ++i) {
    float acc = (b != nullptr) ? b[i] : 0.0f;
    const float *row = W + static_cast<size_t>(i) * in_dim;
    for (int j = 0; j < in_dim; ++j) {
      acc += row[j] * x[j];
    }
    y[i] = acc;
  }
}

// One LSTM step (batch 1), PyTorch gate order (i, f, g, o).
void lstm_step(const float *x, const float *h_prev, const float *c_prev, const float *Wx,
               const float *Wh, const float *b, int H, float *h_out, float *c_out,
               float *scratch_gates) {
  for (int i = 0; i < 4 * H; ++i) {
    float acc = b[i];
    const float *wx_row = Wx + static_cast<size_t>(i) * H;
    const float *wh_row = Wh + static_cast<size_t>(i) * H;
    for (int j = 0; j < H; ++j) {
      acc += wx_row[j] * x[j] + wh_row[j] * h_prev[j];
    }
    scratch_gates[i] = acc;
  }

  const float *gi = scratch_gates;
  const float *gf = scratch_gates + H;
  const float *gg = scratch_gates + 2 * H;
  const float *go = scratch_gates + 3 * H;
  for (int j = 0; j < H; ++j) {
    const float i_t = sigmoidf(gi[j]);
    const float f_t = sigmoidf(gf[j]);
    const float g_t = std::tanh(gg[j]);
    const float o_t = sigmoidf(go[j]);
    const float c_new = f_t * c_prev[j] + i_t * g_t;
    c_out[j] = c_new;
    h_out[j] = o_t * std::tanh(c_new);
  }
}

// One joint pass off a precomputed encoder projection; returns the argmax
// (log_softmax is argmax-invariant, so the reference's is skipped).
int joint_argmax(const GigaamHostDecoder &h, const float *enc_proj, const float *g_t, int pred_d,
                 int joint_h, int n_classes, std::vector<float> &scratch_join,
                 std::vector<float> &scratch_logits) {
  scratch_join.assign(joint_h, 0.0f);
  scratch_logits.assign(n_classes, 0.0f);

  matvec_add(h.joint_pred_w.data(), h.joint_pred_b.data(), g_t, joint_h, pred_d,
             scratch_join.data());
  for (int i = 0; i < joint_h; ++i) {
    const float v = scratch_join[i] + enc_proj[i];
    scratch_join[i] = v > 0.0f ? v : 0.0f;
  }
  matvec_add(h.joint_out_w.data(), h.joint_out_b.data(), scratch_join.data(), n_classes, joint_h,
             scratch_logits.data());

  int best_idx = 0;
  float best_val = scratch_logits[0];
  for (int i = 1; i < n_classes; ++i) {
    if (scratch_logits[i] > best_val) {
      best_val = scratch_logits[i];
      best_idx = i;
    }
  }
  return best_idx;
}

// Score consecutive frames while the predictor state is fixed (a blank does
// not update it, so the scores stay valid through the first non-blank).
void joint_argmax_span(const GigaamHostDecoder &h, const float *enc_proj, const float *g_t,
                       int n_frames, int pred_d, int joint_h, int n_classes,
                       std::vector<float> &pred_proj, std::vector<float> &join,
                       std::vector<float> &logits, std::vector<int> &out_tokens) {
  pred_proj.assign(joint_h, 0.0f);
  join.resize(static_cast<size_t>(n_frames) * joint_h);
  logits.resize(static_cast<size_t>(n_frames) * n_classes);
  out_tokens.resize(n_frames);

  matvec_add(h.joint_pred_w.data(), h.joint_pred_b.data(), g_t, joint_h, pred_d,
             pred_proj.data());
  for (int t = 0; t < n_frames; ++t) {
    const float *enc_row = enc_proj + static_cast<size_t>(t) * joint_h;
    float *join_row = join.data() + static_cast<size_t>(t) * joint_h;
    for (int j = 0; j < joint_h; ++j) {
      const float value = enc_row[j] + pred_proj[j];
      join_row[j] = value > 0.0f ? value : 0.0f;
    }
  }

  for (int t = 0; t < n_frames; ++t) {
    matvec_add(h.joint_out_w.data(), nullptr, join.data() + static_cast<size_t>(t) * joint_h,
               n_classes, joint_h, logits.data() + static_cast<size_t>(t) * n_classes);
  }

  // The output bias is added during the argmax, not written to every row.
  for (int t = 0; t < n_frames; ++t) {
    const float *row = logits.data() + static_cast<size_t>(t) * n_classes;
    int best_idx = 0;
    float best_val = row[0] + h.joint_out_b[0];
    for (int i = 1; i < n_classes; ++i) {
      const float value = row[i] + h.joint_out_b[i];
      if (value > best_val) {
        best_val = value;
        best_idx = i;
      }
    }
    out_tokens[t] = best_idx;
  }
}

} // namespace

void decode_rnnt_greedy(const GigaamHostDecoder &host, const GigaamHParams &hp,
                        const float *encoded, int T_enc, int max_symbols_per_step,
                        std::vector<int32_t> &out_tokens, std::vector<int> &out_frames,
                        const GigaamDecodePoll &poll) {
  out_tokens.clear();
  out_frames.clear();

  const int H = hp.pred_hidden;
  const int enc_d = hp.enc_d_model;
  const int joint_h = hp.joint_hidden;
  const int n_class = hp.joint_n_classes;
  const int blank_id = n_class - 1;

  if (encoded == nullptr || T_enc <= 0) {
    return;
  }
  if (host.lstm_Wx.empty() || host.joint_out_b.size() != static_cast<size_t>(n_class) ||
      host.joint_enc_w.size() != static_cast<size_t>(joint_h) * enc_d) {
    throw std::runtime_error("gigaam: RNN-T host head does not match the hparams");
  }

  // Encoder projection for every frame, hoisted out of the loop (the arch's
  // non-BLAS path: bias-free matvec, then the bias in a second pass).
  std::vector<float> enc_proj_all(static_cast<size_t>(T_enc) * static_cast<size_t>(joint_h));
  for (int t = 0; t < T_enc; ++t) {
    const float *frame = encoded + static_cast<size_t>(t) * enc_d;
    float *proj = enc_proj_all.data() + static_cast<size_t>(t) * joint_h;
    matvec_add(host.joint_enc_w.data(), nullptr, frame, joint_h, enc_d, proj);
  }
  for (int t = 0; t < T_enc; ++t) {
    float *proj = enc_proj_all.data() + static_cast<size_t>(t) * joint_h;
    for (int j = 0; j < joint_h; ++j) {
      proj[j] += host.joint_enc_b[j];
    }
  }

  std::vector<float> h_cur(H, 0.0f);
  std::vector<float> c_cur(H, 0.0f);
  std::vector<float> h_next(H, 0.0f);
  std::vector<float> c_next(H, 0.0f);
  std::vector<float> gates(4 * H, 0.0f);
  std::vector<float> pred_in(H, 0.0f);
  std::vector<float> jh(joint_h, 0.0f);
  std::vector<float> logits(n_class, 0.0f);
  std::vector<float> span_pred_proj;
  std::vector<float> span_join;
  std::vector<float> span_logits;
  std::vector<int> span_tokens;

  // First step: zero input + zero state (reference predict(None, None)).
  bool fresh = true;
  // On a blank the predictor inputs are unchanged, so h_next is reused.
  bool predictor_dirty = true;

  constexpr int lookahead = 32;
  int t = 0;
  while (t < T_enc) {
    if (poll) {
      poll(t, T_enc);
    }
    if (predictor_dirty) {
      if (fresh) {
        std::fill(pred_in.begin(), pred_in.end(), 0.0f);
      } else {
        const int prev = out_tokens.back();
        std::memcpy(pred_in.data(), host.pred_embed.data() + static_cast<size_t>(prev) * H,
                    H * sizeof(float));
      }
      lstm_step(pred_in.data(), h_cur.data(), c_cur.data(), host.lstm_Wx[0].data(),
                host.lstm_Wh[0].data(), host.lstm_b[0].data(), H, h_next.data(), c_next.data(),
                gates.data());
      predictor_dirty = false;
    }

    const int span = std::min(lookahead, T_enc - t);
    joint_argmax_span(host, enc_proj_all.data() + static_cast<size_t>(t) * joint_h,
                      h_next.data(), span, H, joint_h, n_class, span_pred_proj, span_join,
                      span_logits, span_tokens);

    int offset = 0;
    while (offset < span && span_tokens[offset] == blank_id) {
      ++offset;
    }
    if (offset == span) {
      t += span;
      continue;
    }

    // Commit the first token and the candidate predictor state at its frame.
    t += offset;
    int tok = span_tokens[offset];
    out_tokens.push_back(tok);
    out_frames.push_back(t);
    h_cur = h_next;
    c_cur = c_next;
    fresh = false;
    predictor_dirty = true;

    // More symbols on this frame until blank or max_symbols_per_step.
    int sym_count = 1;
    while (max_symbols_per_step <= 0 || sym_count < max_symbols_per_step) {
      const int prev = out_tokens.back();
      std::memcpy(pred_in.data(), host.pred_embed.data() + static_cast<size_t>(prev) * H,
                  H * sizeof(float));
      lstm_step(pred_in.data(), h_cur.data(), c_cur.data(), host.lstm_Wx[0].data(),
                host.lstm_Wh[0].data(), host.lstm_b[0].data(), H, h_next.data(), c_next.data(),
                gates.data());
      predictor_dirty = false;
      const float *enc_p = enc_proj_all.data() + static_cast<size_t>(t) * joint_h;
      tok = joint_argmax(host, enc_p, h_next.data(), H, joint_h, n_class, jh, logits);
      if (tok == blank_id) {
        // Advances time without committing the candidate state; the next
        // frame reuses h_next.
        break;
      }
      out_tokens.push_back(tok);
      out_frames.push_back(t);
      h_cur = h_next;
      c_cur = c_next;
      predictor_dirty = true;
      ++sym_count;
    }
    ++t;
  }
}

void decode_ctc_greedy(const GigaamHostDecoder &host, const GigaamHParams &hp,
                       const float *encoded, int T_enc, std::vector<int32_t> &out_tokens,
                       std::vector<int> &out_frames, const GigaamDecodePoll &poll) {
  out_tokens.clear();
  out_frames.clear();

  const int d_model = hp.enc_d_model;
  const int n_classes = hp.head_n_classes;
  const int blank_id = n_classes - 1;

  if (encoded == nullptr || T_enc <= 0 || d_model <= 0 || n_classes <= 0) {
    throw std::invalid_argument("gigaam: CTC decode needs a non-empty encoder output");
  }
  if (host.ctc_w.size() != static_cast<size_t>(n_classes) * d_model ||
      host.ctc_b.size() != static_cast<size_t>(n_classes)) {
    throw std::runtime_error("gigaam: CTC host head shape mismatch (ctc_w=" +
                             std::to_string(host.ctc_w.size()) + " expected=" +
                             std::to_string(n_classes * d_model) + ", ctc_b=" +
                             std::to_string(host.ctc_b.size()) + " expected=" +
                             std::to_string(n_classes) + ")");
  }

  // logits[T_enc, n_classes] row-major.
  std::vector<float> logits(static_cast<size_t>(T_enc) * static_cast<size_t>(n_classes), 0.0f);
  for (int t = 0; t < T_enc; ++t) {
    if (poll && (t % 256) == 0) {
      poll(t, T_enc);
    }
    const float *frame = encoded + static_cast<size_t>(t) * d_model;
    float *row = logits.data() + static_cast<size_t>(t) * n_classes;
    matvec_add(host.ctc_w.data(), host.ctc_b.data(), frame, n_classes, d_model, row);
  }

  // Per-frame log_softmax in place (kept: it can merge near-equal logits into
  // exact ties, which changes the argmax the same way it does in the arch).
  for (int t = 0; t < T_enc; ++t) {
    float *row = logits.data() + static_cast<size_t>(t) * n_classes;
    float max_v = row[0];
    for (int c = 1; c < n_classes; ++c) {
      if (row[c] > max_v) {
        max_v = row[c];
      }
    }
    double sum = 0.0;
    for (int c = 0; c < n_classes; ++c) {
      sum += std::exp(static_cast<double>(row[c] - max_v));
    }
    const float log_sum = static_cast<float>(std::log(sum)) + max_v;
    for (int c = 0; c < n_classes; ++c) {
      row[c] -= log_sum;
    }
  }

  // Greedy collapse: drop runs of the same label, then drop blanks.
  int prev_label = -1;
  for (int t = 0; t < T_enc; ++t) {
    const float *row = logits.data() + static_cast<size_t>(t) * n_classes;
    int best = 0;
    float best_v = row[0];
    for (int c = 1; c < n_classes; ++c) {
      if (row[c] > best_v) {
        best_v = row[c];
        best = c;
      }
    }
    if (best == prev_label) {
      continue;
    }
    prev_label = best;
    if (best == blank_id) {
      continue;
    }
    out_tokens.push_back(best);
    out_frames.push_back(t);
  }
  if (poll) {
    poll(T_enc, T_enc);
  }
}

} // namespace engine::models::gigaam
