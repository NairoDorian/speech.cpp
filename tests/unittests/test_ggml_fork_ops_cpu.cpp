// Numerical parity tests for the audio.cpp fork-only ggml ops that gained CPU
// compute kernels during the speech.cpp merge (plan V6, Phase 0.L / Phase 2):
//
//   GGML_OP_SAGE_ATTN2     -> ggml_compute_forward_sage_attn2
//   GGML_OP_CONVROT_LINEAR -> ggml_compute_forward_convrot_linear
//   GGML_OP_MUL_MAT_PACK4  -> ggml_compute_forward_mul_mat (fallthrough)
//
// Before this, all three were CUDA-only: on CPU they fell through the compute
// dispatch default and aborted. Each op is checked against an independent
// in-test reference so the kernels are validated as *math*, not just as symbols
// that link.
//
// It also covers ggml_flash_attn_ext_with_bias_mask, the fourth fork-only op
// that needed converging (patch 0002). That one is not a kernel but a wrapper
// that folds a per-head additive bias into the F16 mask — and the per-head mask
// is the interesting part: only the CPU backend implements it. See the
// "does NOT collapse to bias slice" checks below.
//
// The CONVROT_LINEAR check is the load-bearing one: the CUDA kernel fuses a
// QuaRot-style orthonormal rotation into its activation quantization, and the
// weights ship pre-rotated. A CPU kernel that skips the rotation links, runs,
// and returns confident garbage. The test pins the rotation by exploiting its
// orthogonality — rotating the float weight rows with the same transform must
// reproduce the un-rotated reference matmul.
//
// Every case runs on the CPU backend and, when the build has one, on a GPU
// device as well; the two are then compared against each other. That CPU<->GPU
// delta is the per-tensor tolerance the plan's golden-manifest methodology
// wants, and it is where the two implementations' quantization choices show up:
// the CPU kernels are deliberately references (no INT8/FP8 activation
// quantization), so they should be *closer* to the exact answer than CUDA is.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    if (ok) {
        std::printf("  ok   %s\n", what.c_str());
    } else {
        std::printf("  FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

void check_le(float got, float limit, const std::string & what) {
    if (got <= limit) {
        std::printf("  ok   %s (%.3e <= %.2e)\n", what.c_str(), got, limit);
    } else {
        std::printf("  FAIL %s (%.3e > %.2e)\n", what.c_str(), got, limit);
        ++g_failures;
    }
}

// Formats one statistic, showing its bound only when it has one. An unbounded
// statistic is still printed: a number that is only reported today is what
// tells the next reader whether a later backend drifted.
std::string term(const char * name, float got, float limit) {
    char buf[128];
    if (std::isinf(limit)) {
        std::snprintf(buf, sizeof(buf), "%s %.3e", name, got);
    } else {
        std::snprintf(buf, sizeof(buf), "%s %.3e %s %.2e", name, got,
                      got <= limit ? "<=" : ">", limit);
    }
    return buf;
}

struct ErrStats {
    float max_abs = 0.0f;
    // rms(got - want) / rms(want). Scale-free and, unlike per-element relative
    // error, not dominated by outputs that happen to land near zero — this is
    // the per-tensor tolerance shape the plan's golden-manifest methodology uses.
    float rms_rel = 0.0f;
};

ErrStats compare(const std::vector<float> & got, const std::vector<float> & want) {
    ErrStats st;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float abs_err = std::fabs(got[i] - want[i]);
        st.max_abs = std::fmax(st.max_abs, abs_err);
        sq_err += (double) abs_err * abs_err;
        sq_ref += (double) want[i] * want[i];
    }
    st.rms_rel = sq_ref > 0.0 ? (float) std::sqrt(sq_err / sq_ref) : (float) std::sqrt(sq_err);
    return st;
}

// Checks both statistics at once, and always prints both. Pass INFINITY for a
// bound that does not apply to this comparison — the number is still reported.
void check_stats(const ErrStats & st, float abs_tol, float rms_tol, const std::string & what) {
    const bool ok = st.max_abs <= abs_tol && st.rms_rel <= rms_tol;
    std::printf("  %s %s (%s, %s)\n", ok ? "ok  " : "FAIL", what.c_str(),
                term("max abs", st.max_abs, abs_tol).c_str(),
                term("rms", st.rms_rel, rms_tol).c_str());
    if (!ok) {
        ++g_failures;
    }
}

// --------------------------------------------------------------------------
// Reference: the radix-4 rotation that ggml-cuda/convrot-linear.cu fuses into
// its activation quantizer. Written independently of the CPU kernel (straight
// from the CUDA source) so agreement is evidence, not a tautology.
// --------------------------------------------------------------------------

float h4_row_dot(int d, float x0, float x1, float x2, float x3) {
    switch (d) {
        case 0:  return  x0 + x1 + x2 - x3;
        case 1:  return  x0 + x1 - x2 + x3;
        case 2:  return  x0 - x1 + x2 + x3;
        default: return -x0 + x1 + x2 + x3;
    }
}

// Applies R to one contiguous group of `group` (256) values, in place.
void rotate_group(float * v, int group) {
    std::vector<float> cur(v, v + group);
    std::vector<float> next((size_t) group);
    for (int stage = 0; stage < 4; ++stage) {
        const int stride = stage == 0 ? 1 : stage == 1 ? 4 : stage == 2 ? 16 : 64;
        for (int lane = 0; lane < group; ++lane) {
            const int d          = (lane / stride) & 3;
            const int local_base = lane - d * stride;
            next[(size_t) lane] = 0.5f * h4_row_dot(
                d,
                cur[(size_t) local_base],
                cur[(size_t) (local_base + stride)],
                cur[(size_t) (local_base + 2 * stride)],
                cur[(size_t) (local_base + 3 * stride)]);
        }
        cur.swap(next);
    }
    std::memcpy(v, cur.data(), (size_t) group * sizeof(float));
}

void rotate_row(std::vector<float> & row, int group) {
    for (size_t g = 0; g + (size_t) group <= row.size(); g += (size_t) group) {
        rotate_group(row.data() + g, group);
    }
}

// --------------------------------------------------------------------------
// Backend-agnostic graph runner
// --------------------------------------------------------------------------

struct Graph {
    ggml_context *        ctx    = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int                   n_threads = 1;

    Graph(ggml_backend_t b, int threads, size_t n_tensors = 64)
        : backend(b), n_threads(threads) {
        ggml_init_params p{};
        // no_alloc: tensor data lives in a backend buffer, not the ggml context.
        p.mem_size   = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        ctx = ggml_init(p);
    }
    ~Graph() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    // Call once, after every tensor is created and before writing any data.
    void allocate() {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    void set(ggml_tensor * t, const void * data, size_t bytes) {
        ggml_backend_tensor_set(t, data, 0, bytes);
    }

    void get(const ggml_tensor * t, void * data, size_t bytes) const {
        ggml_backend_tensor_get(t, data, 0, bytes);
    }

    void compute(ggml_tensor * out) {
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }
};

// --------------------------------------------------------------------------
// 1. SAGE_ATTN2 — F16 scaled dot-product attention with GQA + optional causal
// --------------------------------------------------------------------------

const int64_t kD = 64;   // head_dim (kernel supports 64 or 128)
const int64_t kN = 128;  // queries (CUDA tiles at CTA_Q=128)
const int64_t kK = 128;  // keys    (CUDA tiles at CTA_K=64)
const int64_t kHQ = 4;   // query heads
const int64_t kHK = 2;   // kv heads (GQA group of 2)
const int64_t kNB = 2;   // batch

// Deterministic inputs shared by every backend so results are comparable.
struct SageInputs {
    std::vector<uint16_t> q, k, v;   // ggml_fp16_t bit patterns
    std::vector<float>    qf, kf, vf; // the same values read back through F16
};

SageInputs make_sage_inputs() {
    SageInputs in;
    std::mt19937 rng(1234u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill = [&](std::vector<uint16_t> & bits, std::vector<float> & shadow, int64_t n) {
        bits.resize((size_t) n);
        shadow.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(dist(rng));
            std::memcpy(&bits[(size_t) i], &h, sizeof(uint16_t));
            // Read back through F16 so the reference sees exactly what the
            // kernel sees — otherwise F16 rounding shows up as "error".
            shadow[(size_t) i] = ggml_fp16_to_fp32(h);
        }
    };
    fill(in.q, in.qf, kD * kN * kHQ * kNB);
    fill(in.k, in.kf, kD * kK * kHK * kNB);
    fill(in.v, in.vf, kD * kK * kHK * kNB);
    return in;
}

std::vector<float> sage_attn2_reference(const SageInputs & in, bool causal, float scale) {
    std::vector<float> want((size_t) (kD * kHQ * kN * kNB), 0.0f);
    std::vector<float> logits((size_t) kK);
    for (int64_t b = 0; b < kNB; ++b) {
        for (int64_t h = 0; h < kHQ; ++h) {
            const int64_t hk = h * kHK / kHQ;
            for (int64_t qi = 0; qi < kN; ++qi) {
                float max_logit = -INFINITY;
                for (int64_t ki = 0; ki < kK; ++ki) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < kD; ++d) {
                        dot += in.qf[(size_t) (((b * kHQ + h) * kN + qi) * kD + d)]
                             * in.kf[(size_t) (((b * kHK + hk) * kK + ki) * kD + d)];
                    }
                    float logit = dot * scale;
                    if (causal && ki > qi) {
                        logit = -INFINITY;
                    }
                    logits[(size_t) ki] = logit;
                    max_logit = std::fmax(max_logit, logit);
                }
                float sum = 0.0f;
                for (int64_t ki = 0; ki < kK; ++ki) {
                    logits[(size_t) ki] = std::exp(logits[(size_t) ki] - max_logit);
                    sum += logits[(size_t) ki];
                }
                for (int64_t d = 0; d < kD; ++d) {
                    float acc = 0.0f;
                    for (int64_t ki = 0; ki < kK; ++ki) {
                        acc += (logits[(size_t) ki] / sum)
                             * in.vf[(size_t) (((b * kHK + hk) * kK + ki) * kD + d)];
                    }
                    // dst layout: [D, HQ, N, NB] -> d + h*D + qi*HQ*D + b*N*HQ*D
                    want[(size_t) (((b * kN + qi) * kHQ + h) * kD + d)] = acc;
                }
            }
        }
    }
    return want;
}

// Returns the op output, or an empty vector if the backend does not support it.
std::vector<float> run_sage_attn2(ggml_backend_t backend, int n_threads,
                                  const SageInputs & in, bool causal, float scale) {
    Graph g(backend, n_threads);

    ggml_tensor * q = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, kD, kN, kHQ, kNB);
    ggml_tensor * k = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, kD, kK, kHK, kNB);
    ggml_tensor * v = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, kD, kK, kHK, kNB);
    ggml_tensor * out = ggml_sage_attn2(g.ctx, q, k, v, scale, causal);

    if (!ggml_backend_supports_op(backend, out)) {
        return {};
    }
    g.allocate();
    g.set(q, in.q.data(), in.q.size() * sizeof(uint16_t));
    g.set(k, in.k.data(), in.k.size() * sizeof(uint16_t));
    g.set(v, in.v.data(), in.v.size() * sizeof(uint16_t));
    g.compute(out);

    check(out->ne[0] == kD && out->ne[1] == kHQ && out->ne[2] == kN && out->ne[3] == kNB,
          "sage_attn2 output shape is [D, HQ, N, NB]");

    std::vector<uint16_t> raw((size_t) ggml_nelements(out));
    g.get(out, raw.data(), raw.size() * sizeof(uint16_t));
    std::vector<float> got(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        ggml_fp16_t h;
        std::memcpy(&h, &raw[i], sizeof(uint16_t));
        got[i] = ggml_fp16_to_fp32(h);
    }
    return got;
}

// --------------------------------------------------------------------------
// 2. CONVROT_LINEAR — rotated INT8 linear
// --------------------------------------------------------------------------

const int    kGroup  = 256;
const int64_t kCk    = 512;   // in_features (2 rotation groups)
const int64_t kCn    = 256;   // out_features
const int64_t kTokens = 5;

struct ConvrotInputs {
    std::vector<int8_t> w_i8;
    std::vector<float>  w_scale;
    std::vector<float>  bias;
    std::vector<float>  x;                 // [k, tokens] row-major by token
    std::vector<float>  want_dequant;      // <dequantized rotated row, rotated x>
    std::vector<float>  want_float;        // the un-rotated float matmul
};

ConvrotInputs make_convrot_inputs(bool with_bias) {
    std::mt19937 rng(4321u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ConvrotInputs in;
    std::vector<std::vector<float>> w_float((size_t) kCn, std::vector<float>((size_t) kCk));
    for (auto & row : w_float) {
        for (auto & value : row) {
            value = dist(rng);
        }
    }
    in.bias.assign((size_t) kCn, 0.0f);
    for (auto & value : in.bias) {
        value = with_bias ? dist(rng) : 0.0f;
    }
    std::vector<std::vector<float>> x((size_t) kTokens, std::vector<float>((size_t) kCk));
    for (auto & row : x) {
        for (auto & value : row) {
            value = dist(rng);
        }
    }

    // Ship the weights the way the exporter does: rotate each row by R, then
    // quantize to INT8 with a per-row scale. Because R is orthonormal,
    //     <R w_j, R x> == <w_j, x>
    // so the op must reproduce the *un-rotated* float matmul. If the kernel
    // skipped the activation rotation this identity would not hold.
    in.w_i8.assign((size_t) (kCn * kCk), 0);
    in.w_scale.assign((size_t) kCn, 0.0f);
    std::vector<std::vector<float>> w_rot_dequant((size_t) kCn, std::vector<float>((size_t) kCk));
    for (int64_t j = 0; j < kCn; ++j) {
        std::vector<float> row = w_float[(size_t) j];
        rotate_row(row, kGroup);
        float max_abs = 0.0f;
        for (float value : row) {
            max_abs = std::fmax(max_abs, std::fabs(value));
        }
        const float s = std::fmax(max_abs / 127.0f, 1e-8f);
        in.w_scale[(size_t) j] = s;
        for (int64_t i = 0; i < kCk; ++i) {
            int q = (int) std::lrint(row[(size_t) i] / s);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            in.w_i8[(size_t) (j * kCk + i)] = (int8_t) q;
            w_rot_dequant[(size_t) j][(size_t) i] = (float) q * s;
        }
    }

    in.x.assign((size_t) (kTokens * kCk), 0.0f);
    for (int64_t t = 0; t < kTokens; ++t) {
        std::memcpy(in.x.data() + t * kCk, x[(size_t) t].data(), (size_t) kCk * sizeof(float));
    }

    in.want_dequant.assign((size_t) (kCn * kTokens), 0.0f);
    in.want_float.assign((size_t) (kCn * kTokens), 0.0f);
    for (int64_t t = 0; t < kTokens; ++t) {
        std::vector<float> rx = x[(size_t) t];
        rotate_row(rx, kGroup);
        for (int64_t j = 0; j < kCn; ++j) {
            float acc_q = 0.0f;
            float acc_f = 0.0f;
            for (int64_t i = 0; i < kCk; ++i) {
                acc_q += w_rot_dequant[(size_t) j][(size_t) i] * rx[(size_t) i];
                acc_f += w_float[(size_t) j][(size_t) i] * x[(size_t) t][(size_t) i];
            }
            in.want_dequant[(size_t) (t * kCn + j)] = acc_q + in.bias[(size_t) j];
            in.want_float[(size_t) (t * kCn + j)]   = acc_f + in.bias[(size_t) j];
        }
    }
    return in;
}

std::vector<float> run_convrot_linear(ggml_backend_t backend, int n_threads,
                                      const ConvrotInputs & in, bool with_bias) {
    Graph g(backend, n_threads);

    ggml_tensor * t_w     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I8,  kCk, kCn);
    ggml_tensor * t_x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kCk, kTokens);
    ggml_tensor * t_scale = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, kCn);
    ggml_tensor * t_bias  = with_bias ? ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, kCn) : nullptr;
    ggml_tensor * out = ggml_convrot_linear(g.ctx, t_w, t_x, t_scale, t_bias, kGroup);

    if (!ggml_backend_supports_op(backend, out)) {
        return {};
    }
    g.allocate();
    g.set(t_w, in.w_i8.data(), in.w_i8.size() * sizeof(int8_t));
    g.set(t_x, in.x.data(), in.x.size() * sizeof(float));
    g.set(t_scale, in.w_scale.data(), in.w_scale.size() * sizeof(float));
    if (t_bias) {
        g.set(t_bias, in.bias.data(), in.bias.size() * sizeof(float));
    }
    g.compute(out);

    check(out->ne[0] == kCn && out->ne[1] == kTokens, "convrot_linear output shape is [n, tokens]");

    std::vector<float> got((size_t) ggml_nelements(out));
    g.get(out, got.data(), got.size() * sizeof(float));
    return got;
}

// Guard rail: prove the rotation is load-bearing, so this suite can never be
// satisfied by a kernel that silently drops it.
void test_convrot_rotation_is_required() {
    std::mt19937 rng(99u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w((size_t) kGroup);
    std::vector<float> x((size_t) kGroup);
    for (int i = 0; i < kGroup; ++i) {
        w[(size_t) i] = dist(rng);
        x[(size_t) i] = dist(rng);
    }

    std::vector<float> rw = w;
    std::vector<float> rx = x;
    rotate_row(rw, kGroup);
    rotate_row(rx, kGroup);

    float plain = 0.0f, both_rotated = 0.0f, only_weight_rotated = 0.0f;
    for (int i = 0; i < kGroup; ++i) {
        plain               += w[(size_t) i]  * x[(size_t) i];
        both_rotated        += rw[(size_t) i] * rx[(size_t) i];
        only_weight_rotated += rw[(size_t) i] * x[(size_t) i];
    }

    check_le(std::fabs(both_rotated - plain), 1e-3f,
             "rotation is orthonormal: <Rw, Rx> == <w, x>");
    check(std::fabs(only_weight_rotated - plain) > 0.1f,
          "rotation is load-bearing: <Rw, x> != <w, x> (a kernel skipping it is wrong)");
}

// --------------------------------------------------------------------------
// 3. FLASH_ATTN_EXT_WITH_BIAS_MASK — the fork-only wrapper (patch 0002)
//
// It folds a dense additive bias (relative-position scores) into the F16 mask
// ggml_flash_attn_ext expects: mask = F16(scale * bias), so the effective
// logits are scale*(QK + bias). The bias carries one slice per query head, so
// the mask has ne[2] == n_head, and that is the part backends disagree on: the
// CPU kernel implements it, ggml's CUDA kernels do not, and CUDA's
// ggml_cuda_get_best_fattn_kernel therefore refuses such masks outright. This
// case runs wherever the backend claims support and is skipped where it does
// not, so it documents that split rather than asserting one answer.
//
// Both a per-head and a shared (ne[2] == 1) mask are covered. The shared case
// is the control: if a backend gets that one right and the per-head one wrong,
// the fault is the mask's head indexing and nothing else.
// --------------------------------------------------------------------------

const int64_t kFN  = 64;  // queries        (a padded KV length keeps every CUDA
const int64_t kFK  = 64;  // keys            fast path eligible, so nothing is
                          //                 skipped for an incidental reason)
const int64_t kFHQ = 4;   // query heads
const int64_t kFHK = 2;   // kv heads       (gqa_ratio == 2)
const int64_t kFNB = 2;   // batch

struct BiasMaskInputs {
    std::vector<float>    q;
    std::vector<uint16_t> k, v;    // ggml_fp16_t bit patterns
    std::vector<float>    kf, vf;  // the same values read back through F16
    std::vector<float>    bias;
    int64_t               bias_heads = kFHQ;
};

BiasMaskInputs make_bias_mask_inputs(int64_t bias_heads) {
    BiasMaskInputs in;
    in.bias_heads = bias_heads;
    std::mt19937 rng(4242u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    in.q.resize((size_t) (kD * kFN * kFHQ * kFNB));
    for (auto & value : in.q) {
        value = dist(rng);
    }
    auto fill_f16 = [&](std::vector<uint16_t> & bits, std::vector<float> & shadow, int64_t n) {
        bits.resize((size_t) n);
        shadow.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(dist(rng));
            std::memcpy(&bits[(size_t) i], &h, sizeof(uint16_t));
            shadow[(size_t) i] = ggml_fp16_to_fp32(h);
        }
    };
    fill_f16(in.k, in.kf, kD * kFK * kFHK * kFNB);
    fill_f16(in.v, in.vf, kD * kFK * kFHK * kFNB);

    // Per-head bias values are pulled far apart across heads so that serving one
    // head another head's slice cannot pass as rounding noise.
    in.bias.resize((size_t) (kFK * kFN * bias_heads * kFNB));
    for (int64_t b = 0; b < kFNB; ++b) {
        for (int64_t h = 0; h < bias_heads; ++h) {
            const float head_offset = 4.0f * (float) h;
            for (int64_t n = 0; n < kFN; ++n) {
                for (int64_t kk = 0; kk < kFK; ++kk) {
                    in.bias[(size_t) (((b * bias_heads + h) * kFN + n) * kFK + kk)] =
                        head_offset + dist(rng);
                }
            }
        }
    }
    return in;
}

// force_bias_head >= 0 makes every query head read that one bias slice. It is a
// diagnostic: it distinguishes "the kernel indexes the per-head mask wrongly"
// from "the kernel is merely imprecise", because a backend that ignores nb32
// matches this variant exactly rather than the real reference.
std::vector<float> bias_mask_reference(const BiasMaskInputs & in, float scale,
                                       int64_t force_bias_head = -1) {
    std::vector<float> want((size_t) (kD * kFHQ * kFN * kFNB), 0.0f);
    std::vector<float> logits((size_t) kFK);
    const int64_t gqa_ratio = kFHQ / kFHK;
    for (int64_t b = 0; b < kFNB; ++b) {
        for (int64_t h = 0; h < kFHQ; ++h) {
            const int64_t hk = h / gqa_ratio;
            const int64_t hb = force_bias_head >= 0 ? force_bias_head : h % in.bias_heads;
            for (int64_t qi = 0; qi < kFN; ++qi) {
                float max_logit = -INFINITY;
                for (int64_t ki = 0; ki < kFK; ++ki) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < kD; ++d) {
                        dot += in.q[(size_t) (((b * kFHQ + h) * kFN + qi) * kD + d)]
                             * in.kf[(size_t) (((b * kFHK + hk) * kFK + ki) * kD + d)];
                    }
                    const float bias = in.bias[(size_t) (((b * in.bias_heads + hb) * kFN + qi) * kFK + ki)];
                    // The wrapper scales the bias and then stores the mask as
                    // F16, so the reference has to see the same rounding —
                    // otherwise the mask's own quantization reads as kernel error.
                    const float mask = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale * bias));
                    logits[(size_t) ki] = scale * dot + mask;
                    max_logit = std::fmax(max_logit, logits[(size_t) ki]);
                }
                float sum = 0.0f;
                for (int64_t ki = 0; ki < kFK; ++ki) {
                    logits[(size_t) ki] = std::exp(logits[(size_t) ki] - max_logit);
                    sum += logits[(size_t) ki];
                }
                for (int64_t d = 0; d < kD; ++d) {
                    float acc = 0.0f;
                    for (int64_t ki = 0; ki < kFK; ++ki) {
                        acc += (logits[(size_t) ki] / sum)
                             * in.vf[(size_t) (((b * kFHK + hk) * kFK + ki) * kD + d)];
                    }
                    want[(size_t) (((b * kFN + qi) * kFHQ + h) * kD + d)] = acc;
                }
            }
        }
    }
    return want;
}

std::vector<float> run_bias_mask(ggml_backend_t backend, int n_threads,
                                 const BiasMaskInputs & in, float scale) {
    Graph g(backend, n_threads);
    ggml_tensor * q    = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, kD,  kFN, kFHQ, kFNB);
    ggml_tensor * k    = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, kD,  kFK, kFHK, kFNB);
    ggml_tensor * v    = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, kD,  kFK, kFHK, kFNB);
    ggml_tensor * bias = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, kFK, kFN, in.bias_heads, kFNB);

    ggml_tensor * out = ggml_flash_attn_ext_with_bias_mask(
        g.ctx, q, k, v, bias, /*mask =*/ nullptr, scale, /*max_bias =*/ 0.0f, /*logit_softcap =*/ 0.0f);

    // The wrapper builds scale/cast/cont nodes ahead of the op, so probing
    // support has to cover the whole chain, not just the leaf.
    if (!ggml_backend_supports_op(backend, out) ||
        !ggml_backend_supports_op(backend, out->src[3])) {
        return {};
    }
    g.allocate();
    g.set(q,    in.q.data(),    in.q.size()    * sizeof(float));
    g.set(k,    in.k.data(),    in.k.size()    * sizeof(uint16_t));
    g.set(v,    in.v.data(),    in.v.size()    * sizeof(uint16_t));
    g.set(bias, in.bias.data(), in.bias.size() * sizeof(float));
    g.compute(out);

    check(out->ne[0] == kD && out->ne[1] == kFHQ && out->ne[2] == kFN && out->ne[3] == kFNB,
          "flash_attn_ext_with_bias_mask output shape is [D, HQ, N, NB]");

    std::vector<float> got((size_t) ggml_nelements(out));
    g.get(out, got.data(), got.size() * sizeof(float));
    return got;
}

// --------------------------------------------------------------------------
// 4. MUL_MAT_PACK4 — must match plain MUL_MAT
// --------------------------------------------------------------------------

const int64_t kMk = 64;
const int64_t kMn = 32;   // must be divisible by 4
const int64_t kMm = 7;

struct MulMatInputs {
    std::vector<float> a, b;
};

MulMatInputs make_mul_mat_inputs() {
    MulMatInputs in;
    std::mt19937 rng(777u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    in.a.resize((size_t) (kMk * kMn));
    in.b.resize((size_t) (kMk * kMm));
    for (auto & value : in.a) { value = dist(rng); }
    for (auto & value : in.b) { value = dist(rng); }
    return in;
}

std::vector<float> run_mul_mat(ggml_backend_t backend, int n_threads,
                               const MulMatInputs & in, bool packed) {
    Graph g(backend, n_threads);
    ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kMk, kMn);
    ggml_tensor * b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kMk, kMm);
    ggml_tensor * out = packed ? ggml_mul_mat_pack4(g.ctx, a, b) : ggml_mul_mat(g.ctx, a, b);

    if (!ggml_backend_supports_op(backend, out)) {
        return {};
    }
    g.allocate();
    g.set(a, in.a.data(), in.a.size() * sizeof(float));
    g.set(b, in.b.data(), in.b.size() * sizeof(float));
    g.compute(out);

    std::vector<float> got((size_t) ggml_nelements(out));
    g.get(out, got.data(), got.size() * sizeof(float));
    return got;
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------

struct BackendRun {
    std::string name;
    std::vector<float> sage_noncausal, sage_causal;
    std::vector<float> convrot_nobias, convrot_bias;
    std::vector<float> bias_mask_per_head, bias_mask_shared;
    std::vector<float> pack4, plain_mul_mat;
};

BackendRun run_all(ggml_backend_t backend, const std::string & name, int n_threads,
                   const SageInputs & sage_in,
                   const ConvrotInputs & convrot_nobias_in,
                   const ConvrotInputs & convrot_bias_in,
                   const BiasMaskInputs & bias_per_head_in,
                   const BiasMaskInputs & bias_shared_in,
                   const MulMatInputs & mul_in,
                   float scale) {
    BackendRun out;
    out.name = name;
    std::printf("\n[%s, %d thread(s)]\n", name.c_str(), n_threads);

    out.sage_noncausal     = run_sage_attn2(backend, n_threads, sage_in, false, scale);
    out.sage_causal        = run_sage_attn2(backend, n_threads, sage_in, true,  scale);
    out.convrot_nobias     = run_convrot_linear(backend, n_threads, convrot_nobias_in, false);
    out.convrot_bias       = run_convrot_linear(backend, n_threads, convrot_bias_in,   true);
    out.bias_mask_per_head = run_bias_mask(backend, n_threads, bias_per_head_in, scale);
    out.bias_mask_shared   = run_bias_mask(backend, n_threads, bias_shared_in,   scale);
    out.pack4              = run_mul_mat(backend, n_threads, mul_in, true);
    out.plain_mul_mat      = run_mul_mat(backend, n_threads, mul_in, false);
    return out;
}

// --------------------------------------------------------------------------
// Tolerances are a property of what a kernel computes, not of which backend
// runs it. The CPU kernels here are references: activations stay in F32 and
// only the stored INT8 weight is dequantized. The CUDA kernels quantize
// activations as part of the op — sage_attn2 runs INT8 Q·Kᵀ with an FP8 PV
// accumulation (`ggml-cuda/sage-attn2/qattn/qk_int_sv_f8_cuda_sm89.cuh`) and
// convrot_linear quantizes each token by max_abs/127
// (`ggml-cuda/convrot-linear.cu:112`). Holding a quantizing kernel to a
// reference kernel's max-abs bound measures the quantizer, not a defect, so the
// two classes carry separate numbers.
//
// The normalized-RMS bound is the load-bearing one for the quantizing class,
// and for convrot_linear it is *identical* across both classes: INT8 activation
// rounding is zero-mean noise that averages out under RMS (CPU 0.71%, CUDA
// 1.01%, bound 5%), whereas a kernel that is actually *wrong* — a convrot that
// drops the rotation — lands at rms_rel ≈ 1.4. That single bound is therefore
// what certifies the rotation on every backend. sage_attn2 is the exception:
// its INT8 Q·Kᵀ + FP8 PV error does not average away (~3.7% RMS), so it gets a
// per-class RMS bound as well.
// --------------------------------------------------------------------------
struct Tolerances {
    const char * kind;
    float sage_abs;
    float sage_rms;
    float convrot_abs;   // vs the dequantized-weight reference
    float convrot_rms;   // vs the float matmul — this is what pins the rotation
    float bias_mask_abs; // vs the per-head softmax reference
    float bias_mask_rms;
};

// dst is F16 for sage_attn2, so ~2e-3 absolute is the representation floor;
// convrot's RMS floor is the INT8 *weight* quantization (~0.8% at these shapes).
constexpr Tolerances kExactActivationTol = {
    "exact-activation", 4e-3f, 1e-2f, 2e-3f, 0.05f, 4e-3f, 1e-2f,
};

// Measured on sm_89 (RTX 4070, CUDA 13.3) with ~2x headroom: sage_attn2 max-abs
// 9.6e-3 / 5.6e-2 and ~3.7e-2 RMS, convrot max-abs 2.5e-1 / 2.7e-1. Needing to
// loosen these for a new backend is a finding, not a merge conflict — it means
// that backend's quantizer is worse than CUDA's and its consumers should know.
// flash_attn_ext_with_bias_mask is not activation-quantized on either side, but
// the CUDA kernels compute it in F16, so its bound is a precision floor rather
// than a quantizer's. It stays far tighter than what a wrong per-head mask slice
// produces: bias values are spread 4.0 apart across heads, so serving the wrong
// slice moves the softmax by whole heads, not by a rounding step.
constexpr Tolerances kQuantizedActivationTol = {
    "quantized-activation", 0.12f, 0.08f, 0.60f, 0.05f, 6e-3f, 2e-2f,
};

void verify(const BackendRun & run, const Tolerances & tol, const SageInputs & sage_in,
            const ConvrotInputs & convrot_nobias_in, const ConvrotInputs & convrot_bias_in,
            const BiasMaskInputs & bias_per_head_in, const BiasMaskInputs & bias_shared_in,
            float scale) {
    std::printf("\n[%s: vs reference, %s tolerances]\n", run.name.c_str(), tol.kind);

    if (!run.sage_noncausal.empty()) {
        const auto want = sage_attn2_reference(sage_in, false, scale);
        check_stats(compare(run.sage_noncausal, want), tol.sage_abs, tol.sage_rms,
                    "sage_attn2 non-causal vs reference");
    }
    if (!run.sage_causal.empty()) {
        const auto want = sage_attn2_reference(sage_in, true, scale);
        check_stats(compare(run.sage_causal, want), tol.sage_abs, tol.sage_rms,
                    "sage_attn2 causal vs reference");
    }
    struct ConvrotCase { const std::vector<float> * got; const ConvrotInputs * in; const char * tag; };
    const ConvrotCase convrot_cases[] = {
        { &run.convrot_nobias, &convrot_nobias_in, "convrot_linear no bias" },
        { &run.convrot_bias,   &convrot_bias_in,   "convrot_linear with bias" },
    };
    for (const auto & c : convrot_cases) {
        if (c.got->empty()) {
            continue;
        }
        // Against an exact-activation reference: for the CPU kernel this is
        // tight, for a quantizing kernel it is the activation step size.
        check_stats(compare(*c.got, c.in->want_dequant), tol.convrot_abs, INFINITY,
                    std::string(c.tag) + " vs dequantized reference");
        // The orthogonality identity: the op reproduces the un-rotated float
        // matmul to within INT8 weight-quantization error. A kernel that dropped
        // the activation rotation lands ~140% off — two orders of magnitude
        // outside this bound — so the check cannot pass by accident. Same bound
        // on every backend; this is the one that certifies the rotation.
        check_stats(compare(*c.got, c.in->want_float), INFINITY, tol.convrot_rms,
                    std::string(c.tag) + " vs float matmul");
    }
    struct BiasCase { const std::vector<float> * got; const BiasMaskInputs * in; const char * tag; };
    const BiasCase bias_cases[] = {
        { &run.bias_mask_per_head, &bias_per_head_in, "flash_attn_ext_with_bias_mask per-head" },
        { &run.bias_mask_shared,   &bias_shared_in,   "flash_attn_ext_with_bias_mask shared"   },
    };
    for (const auto & c : bias_cases) {
        if (c.got->empty()) {
            std::printf("  skip %s (not supported on this backend)\n", c.tag);
            continue;
        }
        check_stats(compare(*c.got, bias_mask_reference(*c.in, scale)),
                    tol.bias_mask_abs, tol.bias_mask_rms, c.tag);
        // A backend that accepts a per-head mask and then reads one slice for
        // every head produces a *plausible* answer, so the check above is not
        // enough on its own: pin it from the other side too. This is not
        // hypothetical — ggml's CUDA kernels do exactly this, which is why
        // ggml_cuda_get_best_fattn_kernel refuses ne[2] != 1 outright and why
        // the relative-attention module keeps this lowering on CPU
        // (src/framework/modules/attention/common_relative_attention.cpp).
        if (c.in->bias_heads > 1) {
            for (int64_t forced = 0; forced < c.in->bias_heads; ++forced) {
                const float rms = compare(*c.got, bias_mask_reference(*c.in, scale, forced)).rms_rel;
                char what[160];
                std::snprintf(what, sizeof(what),
                              "%s does NOT collapse to bias slice %lld", c.tag, (long long) forced);
                check(rms > 1e-2f, what);
            }
        }
    }
    if (!run.pack4.empty() && !run.plain_mul_mat.empty()) {
        check(run.pack4.size() == run.plain_mul_mat.size(),
              "mul_mat_pack4 and mul_mat agree on shape");
        check_le(compare(run.pack4, run.plain_mul_mat).max_abs, 1e-5f,
                 "mul_mat_pack4 == mul_mat");
    }
}

// CPU<->GPU agreement. Reported with generous bounds and full numbers: the two
// implementations make different quantization choices on purpose (the CPU
// kernels are references and skip the INT8/FP8 activation quantization CUDA
// does), so the interesting output is the measured delta, not a tight pass.
void cross_check(const BackendRun & a, const BackendRun & b) {
    std::printf("\n[%s vs %s]\n", a.name.c_str(), b.name.c_str());
    struct Pair { const std::vector<float> * x; const std::vector<float> * y; const char * tag; float tol; };
    // The mul_mat bound is not an F32 precision claim: ggml-cuda enables
    // CUBLAS_TF32_TENSOR_OP_MATH on every cuBLAS handle
    // (`ggml-cuda/common.cuh:1502`), so an F32 GEMM there runs on TF32 tensor
    // cores — a 10-bit mantissa, i.e. ~1e-3 relative. That is a property of F32
    // mul_mat on CUDA and nothing to do with the pack4 wrapper, which is why
    // plain mul_mat is cross-checked alongside it and the two deltas are then
    // required to be equal.
    const Pair pairs[] = {
        { &a.sage_noncausal, &b.sage_noncausal, "sage_attn2 non-causal",     0.10f },
        { &a.sage_causal,    &b.sage_causal,    "sage_attn2 causal",         0.10f },
        { &a.convrot_nobias, &b.convrot_nobias, "convrot_linear no bias",    0.10f },
        { &a.convrot_bias,   &b.convrot_bias,   "convrot_linear with bias",  0.10f },
        { &a.bias_mask_per_head, &b.bias_mask_per_head, "flash bias mask per-head", 2e-2f },
        { &a.bias_mask_shared,   &b.bias_mask_shared,   "flash bias mask shared",   2e-2f },
        { &a.pack4,          &b.pack4,          "mul_mat_pack4",             5e-3f },
        { &a.plain_mul_mat,  &b.plain_mul_mat,  "mul_mat (F32 baseline)",    5e-3f },
    };
    for (const auto & p : pairs) {
        if (p.x->empty() || p.y->empty() || p.x->size() != p.y->size()) {
            std::printf("  skip %s (not run on both backends)\n", p.tag);
            continue;
        }
        check_le(compare(*p.y, *p.x).rms_rel, p.tol,
                 std::string(p.tag) + " normalized RMS error");
    }

    // pack4 must not cost anything of its own: whatever the F32 GEMM path
    // differs by across these two backends, pack4 must differ by exactly that.
    // This is what turns the loose bound above into a real statement about the
    // wrapper — a pack4 that dispatched somewhere else would break the equality
    // even while staying under 5e-3.
    if (!a.pack4.empty() && !b.pack4.empty() &&
        !a.plain_mul_mat.empty() && !b.plain_mul_mat.empty()) {
        const float pack4_delta = compare(b.pack4, a.pack4).rms_rel;
        const float plain_delta = compare(b.plain_mul_mat, a.plain_mul_mat).rms_rel;
        check_le(std::fabs(pack4_delta - plain_delta), 1e-9f,
                 "mul_mat_pack4 cross-backend delta == plain mul_mat cross-backend delta");
    }
}

}  // namespace

int main() {
    std::printf("ggml fork-op kernels\n");

    std::printf("\n[rotation properties]\n");
    test_convrot_rotation_is_required();

    const float scale = 1.0f / std::sqrt((float) kD);
    const SageInputs    sage_in    = make_sage_inputs();
    const ConvrotInputs convrot_nb = make_convrot_inputs(false);
    const ConvrotInputs convrot_b  = make_convrot_inputs(true);
    const BiasMaskInputs bias_ph  = make_bias_mask_inputs(kFHQ);
    const BiasMaskInputs bias_sh  = make_bias_mask_inputs(1);
    const MulMatInputs  mul_in     = make_mul_mat_inputs();

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        std::printf("could not initialize the CPU backend\n");
        return 1;
    }

    // Threading is a CPU-only concern, but it is the cheapest way to catch a
    // kernel whose work split or per-thread workspace is wrong.
    BackendRun cpu_run;
    for (int n_threads : {1, 4}) {
        BackendRun run = run_all(cpu, "cpu", n_threads, sage_in, convrot_nb, convrot_b, bias_ph, bias_sh, mul_in, scale);
        verify(run, kExactActivationTol, sage_in, convrot_nb, convrot_b, bias_ph, bias_sh, scale);
        if (n_threads == 1) {
            cpu_run = run;
        } else {
            cross_check(cpu_run, run);
        }
    }

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev == nullptr) {
        std::printf("\nno GPU device in this build; skipping the CPU<->GPU parity pass\n");
    } else {
        ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
        if (gpu == nullptr) {
            std::printf("\nGPU device present but failed to initialize; skipping parity pass\n");
        } else {
            const std::string gpu_name = ggml_backend_dev_name(gpu_dev);
            BackendRun gpu_run = run_all(gpu, gpu_name, 1, sage_in, convrot_nb, convrot_b, bias_ph, bias_sh, mul_in, scale);
            // The CPU kernels are the exact-activation references; every GPU
            // implementation of these three ops that exists today quantizes
            // activations inside the op. If a future backend ships an
            // exact-activation GPU kernel, it should be held to the reference
            // tolerances instead — and it will pass them.
            verify(gpu_run, kQuantizedActivationTol, sage_in, convrot_nb, convrot_b, bias_ph, bias_sh, scale);
            cross_check(cpu_run, gpu_run);
            ggml_backend_free(gpu);
        }
    }

    ggml_backend_free(cpu);

    if (g_failures == 0) {
        std::printf("\nAll checks passed.\n");
        return 0;
    }
    std::printf("\n%d check(s) FAILED.\n", g_failures);
    return 1;
}
