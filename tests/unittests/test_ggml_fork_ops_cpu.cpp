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
// 3. MUL_MAT_PACK4 — must match plain MUL_MAT
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
    std::vector<float> pack4, plain_mul_mat;
};

BackendRun run_all(ggml_backend_t backend, const std::string & name, int n_threads,
                   const SageInputs & sage_in,
                   const ConvrotInputs & convrot_nobias_in,
                   const ConvrotInputs & convrot_bias_in,
                   const MulMatInputs & mul_in,
                   float scale) {
    BackendRun out;
    out.name = name;
    std::printf("\n[%s, %d thread(s)]\n", name.c_str(), n_threads);

    out.sage_noncausal = run_sage_attn2(backend, n_threads, sage_in, false, scale);
    out.sage_causal    = run_sage_attn2(backend, n_threads, sage_in, true,  scale);
    out.convrot_nobias = run_convrot_linear(backend, n_threads, convrot_nobias_in, false);
    out.convrot_bias   = run_convrot_linear(backend, n_threads, convrot_bias_in,   true);
    out.pack4          = run_mul_mat(backend, n_threads, mul_in, true);
    out.plain_mul_mat  = run_mul_mat(backend, n_threads, mul_in, false);
    return out;
}

// dst is F16 for sage_attn2, so ~2e-3 absolute is the representation floor.
constexpr float kSageAbsTol   = 4e-3f;
constexpr float kConvrotAbsTol = 2e-3f;
// INT8 weight quantization floor for the convrot shapes above.
constexpr float kConvrotRmsTol = 0.05f;

void verify(const BackendRun & run, const SageInputs & sage_in,
            const ConvrotInputs & convrot_nobias_in, const ConvrotInputs & convrot_bias_in,
            float scale) {
    std::printf("\n[%s: vs reference]\n", run.name.c_str());

    if (!run.sage_noncausal.empty()) {
        const auto want = sage_attn2_reference(sage_in, false, scale);
        check_le(compare(run.sage_noncausal, want).max_abs, kSageAbsTol,
                 "sage_attn2 non-causal max abs error");
    }
    if (!run.sage_causal.empty()) {
        const auto want = sage_attn2_reference(sage_in, true, scale);
        check_le(compare(run.sage_causal, want).max_abs, kSageAbsTol,
                 "sage_attn2 causal max abs error");
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
        check_le(compare(*c.got, c.in->want_dequant).max_abs, kConvrotAbsTol,
                 std::string(c.tag) + " vs dequantized reference, max abs error");
        // The orthogonality identity: the op reproduces the un-rotated float
        // matmul to within INT8 weight-quantization error. A kernel that dropped
        // the activation rotation lands ~140% off — two orders of magnitude
        // outside this bound — so the check cannot pass by accident.
        check_le(compare(*c.got, c.in->want_float).rms_rel, kConvrotRmsTol,
                 std::string(c.tag) + " vs float matmul, normalized RMS error");
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
    const Pair pairs[] = {
        { &a.sage_noncausal, &b.sage_noncausal, "sage_attn2 non-causal",   0.10f },
        { &a.sage_causal,    &b.sage_causal,    "sage_attn2 causal",       0.10f },
        { &a.convrot_nobias, &b.convrot_nobias, "convrot_linear no bias",  0.10f },
        { &a.convrot_bias,   &b.convrot_bias,   "convrot_linear with bias",0.10f },
        { &a.pack4,          &b.pack4,          "mul_mat_pack4",           1e-4f },
    };
    for (const auto & p : pairs) {
        if (p.x->empty() || p.y->empty() || p.x->size() != p.y->size()) {
            std::printf("  skip %s (not run on both backends)\n", p.tag);
            continue;
        }
        check_le(compare(*p.y, *p.x).rms_rel, p.tol,
                 std::string(p.tag) + " normalized RMS error");
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
        BackendRun run = run_all(cpu, "cpu", n_threads, sage_in, convrot_nb, convrot_b, mul_in, scale);
        verify(run, sage_in, convrot_nb, convrot_b, scale);
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
            BackendRun gpu_run = run_all(gpu, gpu_name, 1, sage_in, convrot_nb, convrot_b, mul_in, scale);
            verify(gpu_run, sage_in, convrot_nb, convrot_b, scale);
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
