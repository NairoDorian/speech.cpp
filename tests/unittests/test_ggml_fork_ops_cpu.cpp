// Numerical parity tests for the audio.cpp fork-only ggml ops that gained CPU
// compute kernels during the speech.cpp merge (plan V6, Phase 0.L / Phase 2):
//
//   GGML_OP_SAGE_ATTN2     -> ggml_compute_forward_sage_attn2
//   GGML_OP_CONVROT_LINEAR -> ggml_compute_forward_convrot_linear
//   GGML_OP_MUL_MAT_PACK4  -> ggml_compute_forward_mul_mat (fallthrough)
//
// Before this, all three were CUDA-only: on CPU they fell through the compute
// dispatch default and aborted. Each op is checked against an independent
// in-test reference so the kernels are validated as *math*, not just as
// symbols that link.
//
// The CONVROT_LINEAR check is the load-bearing one: the CUDA kernel fuses a
// QuaRot-style orthonormal rotation into its activation quantization, and the
// weights ship pre-rotated. A CPU kernel that skips the rotation links, runs,
// and returns confident garbage. The test pins the rotation by exploiting its
// orthogonality — rotating the float weight rows with the same transform must
// reproduce the un-rotated reference matmul.

#include "ggml.h"
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

void check_close(float got, float want, float tol, const std::string & what) {
    const float err = std::fabs(got - want);
    if (err <= tol) {
        std::printf("  ok   %s (|%.6f - %.6f| = %.2e <= %.2e)\n", what.c_str(), got, want, err, tol);
    } else {
        std::printf("  FAIL %s (|%.6f - %.6f| = %.2e > %.2e)\n", what.c_str(), got, want, err, tol);
        ++g_failures;
    }
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
// Small ggml graph runner
// --------------------------------------------------------------------------

struct Graph {
    ggml_context * ctx = nullptr;

    explicit Graph(size_t mem_mb) {
        ggml_init_params p{};
        p.mem_size   = mem_mb * 1024 * 1024;
        p.mem_buffer = nullptr;
        p.no_alloc   = false;
        ctx = ggml_init(p);
    }
    ~Graph() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    void run(ggml_tensor * out, int n_threads) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    }
};

// --------------------------------------------------------------------------
// 1. SAGE_ATTN2 — F16 scaled dot-product attention with GQA + optional causal
// --------------------------------------------------------------------------

void test_sage_attn2(bool causal, int n_threads) {
    const int64_t D  = 64;   // head_dim (kernel supports 64 or 128)
    const int64_t N  = 12;   // queries
    const int64_t K  = 20;   // keys
    const int64_t HQ = 4;    // query heads
    const int64_t HK = 2;    // kv heads (GQA group of 2)
    const int64_t NB = 2;    // batch

    Graph g(64);

    ggml_tensor * q = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, D, N, HQ, NB);
    ggml_tensor * k = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, D, K, HK, NB);
    ggml_tensor * v = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, D, K, HK, NB);

    std::mt19937 rng(1234u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto fill = [&](ggml_tensor * t, std::vector<float> & shadow) {
        const int64_t n = ggml_nelements(t);
        shadow.resize((size_t) n);
        auto * data = (ggml_fp16_t *) t->data;
        for (int64_t i = 0; i < n; ++i) {
            const float value = dist(rng);
            data[i] = ggml_fp32_to_fp16(value);
            // Read back through F16 so the reference sees exactly what the
            // kernel sees — otherwise the F16 rounding shows up as "error".
            shadow[(size_t) i] = ggml_fp16_to_fp32(data[i]);
        }
    };

    std::vector<float> qh, kh, vh;
    fill(q, qh);
    fill(k, kh);
    fill(v, vh);

    const float scale = 1.0f / std::sqrt((float) D);

    ggml_tensor * out = ggml_sage_attn2(g.ctx, q, k, v, scale, causal);
    g.run(out, n_threads);

    check(out->ne[0] == D && out->ne[1] == HQ && out->ne[2] == N && out->ne[3] == NB,
          "sage_attn2 output shape is [D, HQ, N, NB]");

    // Independent reference in F32.
    std::vector<float> want((size_t) ggml_nelements(out), 0.0f);
    std::vector<float> logits((size_t) K);
    for (int64_t b = 0; b < NB; ++b) {
        for (int64_t h = 0; h < HQ; ++h) {
            const int64_t hk = h * HK / HQ;
            for (int64_t qi = 0; qi < N; ++qi) {
                float max_logit = -INFINITY;
                for (int64_t ki = 0; ki < K; ++ki) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        const float qv = qh[(size_t) (((b * HQ + h) * N + qi) * D + d)];
                        const float kv = kh[(size_t) (((b * HK + hk) * K + ki) * D + d)];
                        dot += qv * kv;
                    }
                    float logit = dot * scale;
                    if (causal && ki > qi) {
                        logit = -INFINITY;
                    }
                    logits[(size_t) ki] = logit;
                    max_logit = std::fmax(max_logit, logit);
                }
                float sum = 0.0f;
                for (int64_t ki = 0; ki < K; ++ki) {
                    logits[(size_t) ki] = std::exp(logits[(size_t) ki] - max_logit);
                    sum += logits[(size_t) ki];
                }
                for (int64_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int64_t ki = 0; ki < K; ++ki) {
                        const float vv = vh[(size_t) (((b * HK + hk) * K + ki) * D + d)];
                        acc += (logits[(size_t) ki] / sum) * vv;
                    }
                    // dst layout: [D, HQ, N, NB] -> d + h*D + qi*HQ*D + b*N*HQ*D
                    want[(size_t) (((b * N + qi) * HQ + h) * D + d)] = acc;
                }
            }
        }
    }

    std::vector<float> got((size_t) ggml_nelements(out));
    const auto * out_data = (const ggml_fp16_t *) out->data;
    for (size_t i = 0; i < got.size(); ++i) {
        got[i] = ggml_fp16_to_fp32(out_data[i]);
    }

    const ErrStats st = compare(got, want);
    // dst is F16, so ~1e-3 absolute is the representation floor.
    const std::string tag = causal ? "sage_attn2 causal" : "sage_attn2 non-causal";
    check_close(st.max_abs, 0.0f, 2e-3f, tag + " max abs error (nth=" + std::to_string(n_threads) + ")");
}

// --------------------------------------------------------------------------
// 2. CONVROT_LINEAR — rotated INT8 linear
// --------------------------------------------------------------------------

void test_convrot_linear(bool with_bias, int n_threads) {
    const int group = 256;
    const int64_t k = 512;   // in_features (2 rotation groups)
    const int64_t n = 96;    // out_features
    const int64_t tokens = 5;

    std::mt19937 rng(4321u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Ground-truth float layer.
    std::vector<std::vector<float>> w_float((size_t) n, std::vector<float>((size_t) k));
    for (auto & row : w_float) {
        for (auto & value : row) {
            value = dist(rng);
        }
    }
    std::vector<float> bias_vec((size_t) n);
    for (auto & value : bias_vec) {
        value = with_bias ? dist(rng) : 0.0f;
    }
    std::vector<std::vector<float>> x((size_t) tokens, std::vector<float>((size_t) k));
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
    std::vector<int8_t> w_i8((size_t) (n * k));
    std::vector<float>  w_scale((size_t) n);
    std::vector<std::vector<float>> w_rot_dequant((size_t) n, std::vector<float>((size_t) k));
    for (int64_t j = 0; j < n; ++j) {
        std::vector<float> row = w_float[(size_t) j];
        rotate_row(row, group);
        float max_abs = 0.0f;
        for (float value : row) {
            max_abs = std::fmax(max_abs, std::fabs(value));
        }
        const float s = std::fmax(max_abs / 127.0f, 1e-8f);
        w_scale[(size_t) j] = s;
        for (int64_t i = 0; i < k; ++i) {
            int q = (int) std::lrint(row[(size_t) i] / s);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            w_i8[(size_t) (j * k + i)] = (int8_t) q;
            w_rot_dequant[(size_t) j][(size_t) i] = (float) q * s;
        }
    }

    Graph g(64);
    ggml_tensor * t_w     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I8,  k, n);
    ggml_tensor * t_x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, k, tokens);
    ggml_tensor * t_scale = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n);
    ggml_tensor * t_bias  = with_bias ? ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n) : nullptr;

    std::memcpy(t_w->data, w_i8.data(), w_i8.size() * sizeof(int8_t));
    std::memcpy(t_scale->data, w_scale.data(), w_scale.size() * sizeof(float));
    if (t_bias) {
        std::memcpy(t_bias->data, bias_vec.data(), bias_vec.size() * sizeof(float));
    }
    for (int64_t t = 0; t < tokens; ++t) {
        std::memcpy((char *) t_x->data + (size_t) t * t_x->nb[1],
                    x[(size_t) t].data(), (size_t) k * sizeof(float));
    }

    ggml_tensor * out = ggml_convrot_linear(g.ctx, t_w, t_x, t_scale, t_bias, group);
    g.run(out, n_threads);

    check(out->ne[0] == n && out->ne[1] == tokens, "convrot_linear output shape is [n, tokens]");

    // Reference: <dequantized rotated weight row, rotated activation row>.
    // This is what the op is *defined* to compute; it differs from the float
    // ground truth only by the INT8 weight quantization error.
    std::vector<float> want_exact((size_t) (n * tokens));
    std::vector<float> want_float((size_t) (n * tokens));
    for (int64_t t = 0; t < tokens; ++t) {
        std::vector<float> rx = x[(size_t) t];
        rotate_row(rx, group);
        for (int64_t j = 0; j < n; ++j) {
            float acc_q = 0.0f;
            float acc_f = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                acc_q += w_rot_dequant[(size_t) j][(size_t) i] * rx[(size_t) i];
                acc_f += w_float[(size_t) j][(size_t) i] * x[(size_t) t][(size_t) i];
            }
            want_exact[(size_t) (t * n + j)] = acc_q + bias_vec[(size_t) j];
            want_float[(size_t) (t * n + j)] = acc_f + bias_vec[(size_t) j];
        }
    }

    std::vector<float> got((size_t) (n * tokens));
    for (int64_t t = 0; t < tokens; ++t) {
        std::memcpy(got.data() + (size_t) (t * n),
                    (const char *) out->data + (size_t) t * out->nb[1],
                    (size_t) n * sizeof(float));
    }

    const std::string tag = std::string("convrot_linear ") + (with_bias ? "with bias" : "no bias");
    const ErrStats vs_exact = compare(got, want_exact);
    check_close(vs_exact.max_abs, 0.0f, 2e-3f,
                tag + " vs dequantized reference, max abs error (nth=" + std::to_string(n_threads) + ")");

    // The orthogonality identity: the op reproduces the un-rotated float matmul
    // to within INT8 weight-quantization error (~0.7% RMS here). A kernel that
    // dropped the activation rotation lands ~140% off — two orders of magnitude
    // outside this bound — so the check cannot pass by accident.
    const ErrStats vs_float = compare(got, want_float);
    check_close(vs_float.rms_rel, 0.0f, 0.05f, tag + " vs float matmul, normalized RMS error");
}

// Guard rail: prove the rotation is load-bearing, so this suite can never be
// satisfied by a kernel that silently drops it.
void test_convrot_rotation_is_required() {
    const int group = 256;
    std::mt19937 rng(99u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w((size_t) group);
    std::vector<float> x((size_t) group);
    for (int i = 0; i < group; ++i) {
        w[(size_t) i] = dist(rng);
        x[(size_t) i] = dist(rng);
    }

    std::vector<float> rw = w;
    std::vector<float> rx = x;
    rotate_row(rw, group);
    rotate_row(rx, group);

    float plain = 0.0f, both_rotated = 0.0f, only_weight_rotated = 0.0f;
    for (int i = 0; i < group; ++i) {
        plain               += w[(size_t) i]  * x[(size_t) i];
        both_rotated        += rw[(size_t) i] * rx[(size_t) i];
        only_weight_rotated += rw[(size_t) i] * x[(size_t) i];
    }

    check_close(both_rotated, plain, 1e-3f, "rotation is orthonormal: <Rw, Rx> == <w, x>");
    check(std::fabs(only_weight_rotated - plain) > 0.1f,
          "rotation is load-bearing: <Rw, x> != <w, x> (a kernel skipping it is wrong)");
}

// --------------------------------------------------------------------------
// 3. MUL_MAT_PACK4 — must match plain MUL_MAT on CPU
// --------------------------------------------------------------------------

void test_mul_mat_pack4(int n_threads) {
    const int64_t k = 64;
    const int64_t n = 32;   // must be divisible by 4
    const int64_t m = 7;

    std::mt19937 rng(777u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Graph g(64);
    ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, k, m);
    for (int64_t i = 0; i < ggml_nelements(a); ++i) {
        ((float *) a->data)[i] = dist(rng);
    }
    for (int64_t i = 0; i < ggml_nelements(b); ++i) {
        ((float *) b->data)[i] = dist(rng);
    }

    ggml_tensor * packed = ggml_mul_mat_pack4(g.ctx, a, b);
    g.run(packed, n_threads);
    std::vector<float> got((size_t) ggml_nelements(packed));
    std::memcpy(got.data(), packed->data, got.size() * sizeof(float));

    Graph g2(64);
    ggml_tensor * a2 = ggml_new_tensor_2d(g2.ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * b2 = ggml_new_tensor_2d(g2.ctx, GGML_TYPE_F32, k, m);
    std::memcpy(a2->data, a->data, (size_t) ggml_nbytes(a));
    std::memcpy(b2->data, b->data, (size_t) ggml_nbytes(b));
    ggml_tensor * plain = ggml_mul_mat(g2.ctx, a2, b2);
    g2.run(plain, n_threads);
    std::vector<float> want((size_t) ggml_nelements(plain));
    std::memcpy(want.data(), plain->data, want.size() * sizeof(float));

    check(got.size() == want.size(), "mul_mat_pack4 and mul_mat agree on shape");
    const ErrStats st = compare(got, want);
    check_close(st.max_abs, 0.0f, 1e-5f,
                "mul_mat_pack4 == mul_mat (nth=" + std::to_string(n_threads) + ")");
}

} // namespace

int main() {
    std::printf("ggml fork-op CPU kernels\n");

    std::printf("\n[rotation properties]\n");
    test_convrot_rotation_is_required();

    for (int n_threads : {1, 4}) {
        std::printf("\n[sage_attn2, %d thread(s)]\n", n_threads);
        test_sage_attn2(/*causal=*/false, n_threads);
        test_sage_attn2(/*causal=*/true, n_threads);

        std::printf("\n[convrot_linear, %d thread(s)]\n", n_threads);
        test_convrot_linear(/*with_bias=*/false, n_threads);
        test_convrot_linear(/*with_bias=*/true, n_threads);

        std::printf("\n[mul_mat_pack4, %d thread(s)]\n", n_threads);
        test_mul_mat_pack4(n_threads);
    }

    if (g_failures == 0) {
        std::printf("\nAll checks passed.\n");
        return 0;
    }
    std::printf("\n%d check(s) FAILED.\n", g_failures);
    return 1;
}
