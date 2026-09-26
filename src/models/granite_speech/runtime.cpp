// granite_speech runtime: Conformer encoder -> Q-Former projector -> Granite-4
// LM greedy decode, one utterance per call.
//
// Ported op-for-op from the transcribe.cpp arch:
//   src/runtime/arch/granite/encoder.cpp   build_encoder_graph, granite_macaron,
//                                          granite_conv_module,
//                                          precompute_attention_dists,
//                                          precompute_last_block_mask
//   src/runtime/granite_conformer/shaw_attn.cpp -> the shared
//                                          engine::modules::build_shaw_block_attention
//                                          (the arch itself routes there, so both
//                                          sides run the same Shaw sub-graph)
//   src/runtime/conformer/conformer.cpp    macaron_ff_residual, fused_batch_norm
//   src/runtime/arch/granite/projector.cpp build_projector_graph, qformer_layer,
//                                          qformer_attn
//   src/runtime/arch/granite/decoder.cpp   build_prefill_graph / block_prefill,
//                                          build_step_graph / block_step (the
//                                          flash-attention path the arch runs by
//                                          default: decoder_use_flash = true)
//   src/runtime/arch/granite/model.cpp     run(): input gate, decode budget, KV
//                                          sizing, greedy loop, truncation
//
// Deliberate structural differences (numerically neutral on CPU):
//   - Encoder and projector are one graph; the arch copies the encoder output
//     to the host and uploads it into a separate projector graph (an exact F32
//     round trip).
//   - Graphs run on the session's single backend through a ggml_gallocr
//     instead of the arch's ggml_backend_sched with a CPU fallback.
//   - The step graph uses the separate gate / up matrices with
//     silu(gate) * up; the arch packs them and calls ggml_swiglu. Per output
//     element both are the same dot product followed by the same silu kernel
//     and one multiply, so CPU results are bit-identical without doubling the
//     FFN weight memory.
//
// Cached-graph rule: the encoder and prefill graphs are built, allocated and
// computed once per call. The step graph is reused for every decode step of a
// call, and ALL of its inputs (token id, RoPE position, KV row, mask) are
// re-uploaded before every compute, so ggml-alloc reusing an input's memory
// in place cannot leak state between steps. Weights and the KV cache live in
// their own backend buffers, outside the allocator.

#include "engine/models/granite_speech/model.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/errors.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::granite_speech {
namespace {

constexpr int kGenReserve = 256;              // arch model.cpp k_gen_reserve
constexpr int kKvBucket = 256;                // arch run() kKvBucket
constexpr int kTranscriptTokensPerSec = 12;   // transcribe-decode-budget.h
constexpr float kEncoderLayerNormEps = 1e-5f; // nn.LayerNorm default (arch encoder.cpp)
constexpr uint16_t kF16Zero = 0x0000;
constexpr uint16_t kF16NegInf = 0xFC00;
constexpr size_t kEncoderGraphNodes = 32768;
constexpr size_t kPrefillGraphNodes = 16384;
constexpr size_t kStepGraphNodes = 8192;

// transcribe-decode-budget.h:predict_transcript_tokens.
int predict_transcript_tokens(int audio_tokens, double ms_per_audio_token) {
    if (audio_tokens <= 0) {
        return 0;
    }
    if (!(ms_per_audio_token > 0.0)) {
        return audio_tokens;
    }
    const double seconds = static_cast<double>(audio_tokens) * ms_per_audio_token / 1000.0;
    const double predicted = seconds * kTranscriptTokensPerSec;
    if (predicted <= 0.0) {
        return 0;
    }
    constexpr double kIntMax = 2147483647.0;
    return predicted >= kIntMax ? 2147483647 : static_cast<int>(predicted);
}

// transcribe-decode-budget.h:pick_decode_budget.
int pick_decode_budget(int predicted, int floor_tokens, int t_prompt, int ceiling) {
    int budget = std::max(floor_tokens, predicted);
    const int room = ceiling - t_prompt;
    if (budget > room) {
        budget = room;
    }
    return budget > 0 ? budget : 0;
}

// One ggml graph: a no_alloc metadata context, the graph, and a gallocr on
// the session backend.
class GraphArena {
public:
    GraphArena(core::ExecutionContext & execution, size_t nodes, const char * label)
        : execution_(execution), label_(label) {
        const size_t mem = ggml_tensor_overhead() * nodes * 2 + ggml_graph_overhead_custom(nodes, false);
        ctx_ = ggml_init({mem, nullptr, true});
        if (ctx_ == nullptr) {
            throw std::runtime_error(std::string(label_) + ": graph context allocation failed");
        }
        graph_ = ggml_new_graph_custom(ctx_, nodes, false);
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (graph_ == nullptr || allocator_ == nullptr) {
            release();
            throw std::runtime_error(std::string(label_) + ": graph allocation failed");
        }
    }
    GraphArena(const GraphArena &) = delete;
    GraphArena & operator=(const GraphArena &) = delete;
    ~GraphArena() { release(); }

    ggml_context * ctx() const noexcept { return ctx_; }
    ggml_cgraph * graph() const noexcept { return graph_; }

    void allocate() {
        core::validate_backend_graph_supported(execution_.backend(), graph_, label_);
        if (!ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw runtime::CapacityError(std::string(label_) + ": compute buffer allocation failed (out of memory)");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    void compute() {
        if (core::compute_graph(execution_, graph_, plan_, label_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(label_) + ": graph compute failed");
        }
    }

private:
    void release() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_, false);
        }
        if (allocator_ != nullptr) {
            ggml_gallocr_free(allocator_);
            allocator_ = nullptr;
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        graph_ = nullptr;
    }

    core::ExecutionContext & execution_;
    const char * label_;
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    core::HostGraphPlan plan_;
};

ggml_tensor * input_tensor(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

// y = LN(x) * gamma + beta (arch encoder.cpp / projector.cpp layer_norm).
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, const modules::NormWeights & w, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w.weight->tensor);
    if (w.bias.has_value()) {
        y = ggml_add(ctx, y, w.bias->tensor);
    }
    return y;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const modules::LinearWeights & w) {
    ggml_tensor * y = ggml_mul_mat(ctx, w.weight.tensor, x);
    if (w.bias.has_value()) {
        y = ggml_add(ctx, y, w.bias->tensor);
    }
    return y;
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, const core::TensorValue & weight, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight.tensor);
}

// conformer.cpp:macaron_ff_residual: x + 0.5 * Down(SiLU(Up(LN(x)))).
ggml_tensor * macaron(ggml_context * ctx, ggml_tensor * x, const modules::NormWeights & norm,
                      const modules::LinearWeights & up, const modules::LinearWeights & down) {
    ggml_tensor * y = layer_norm(ctx, x, norm, kEncoderLayerNormEps);
    y = linear(ctx, y, up);
    y = ggml_silu(ctx, y);
    y = linear(ctx, y, down);
    y = ggml_scale(ctx, y, 0.5f);
    return ggml_add(ctx, x, y);
}

// encoder.cpp:granite_conv_module (direct depthwise branch, the arch default):
// LN -> pointwise1 (d -> 2*inner) -> GLU -> depthwise k -> fused BN -> SiLU ->
// pointwise2 (inner -> d), on x = [d, T].
ggml_tensor * conv_module(ggml_context * ctx, ggml_tensor * x, const GraniteSpeechEncoderBlock & b,
                          int conv_kernel, int64_t inner_dim) {
    const int64_t T = x->ne[1];
    x = layer_norm(ctx, x, b.norm_conv, kEncoderLayerNormEps);
    x = linear(ctx, x, b.conv_pointwise1);  // [2*inner, T]

    // GLU over ne[0]: first half * sigmoid(second half).
    ggml_tensor * gate = ggml_view_2d(ctx, x, inner_dim, T, x->nb[1], 0);
    ggml_tensor * value = ggml_view_2d(ctx, x, inner_dim, T, x->nb[1], inner_dim * ggml_element_size(x));
    x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));  // [inner, T]

    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [T, inner]
    const int padding = (conv_kernel - 1) / 2;
    ggml_tensor * knl = ggml_reshape_4d(ctx, b.conv_depthwise.tensor, conv_kernel, 1, 1, inner_dim);
    ggml_tensor * d4 = ggml_reshape_4d(ctx, x, x->ne[0], 1, inner_dim, 1);
    x = ggml_conv_2d_dw_direct(ctx, knl, d4, 1, 1, padding, 0, 1, 1);
    x = ggml_reshape_3d(ctx, x, x->ne[0], inner_dim, 1);  // [T, inner, 1]

    // conformer.cpp:fused_batch_norm: [C] -> [1, C, 1, 1] broadcast over T.
    ggml_tensor * scale_4d = ggml_reshape_4d(ctx, b.conv_bn_scale.tensor, 1, inner_dim, 1, 1);
    ggml_tensor * bias_4d = ggml_reshape_4d(ctx, b.conv_bn_bias.tensor, 1, inner_dim, 1, 1);
    x = ggml_mul(ctx, x, scale_4d);
    x = ggml_add(ctx, x, bias_4d);

    x = ggml_silu(ctx, x);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [inner, T, 1]
    return linear(ctx, x, b.conv_pointwise2);              // [d, T, 1]
}

// projector.cpp:qformer_attn. q/k/v: [hidden, seq, batch]; per-batch SDPA,
// head_dim = hidden / n_heads, no mask (the padded window frames are zeros
// that the reference attends to as well).
ggml_tensor * qformer_attention(ggml_context * ctx, ggml_tensor * q_in, ggml_tensor * k_in, ggml_tensor * v_in,
                                int n_heads, int head_dim) {
    const int64_t q_seq = q_in->ne[1];
    const int64_t k_seq = k_in->ne[1];
    const int64_t batch = q_in->ne[2];
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const auto split_heads = [&](ggml_tensor * t, int64_t seq) {
        ggml_tensor * r = ggml_reshape_4d(ctx, t, head_dim, n_heads, seq, batch);
        return ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));  // [hd, seq, heads, batch]
    };
    ggml_tensor * q = split_heads(q_in, q_seq);
    ggml_tensor * k = split_heads(k_in, k_seq);
    ggml_tensor * v = split_heads(v_in, k_seq);
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    kq = ggml_scale(ctx, kq, scale);
    ggml_tensor * attn = ggml_soft_max(ctx, kq);
    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor * out = ggml_mul_mat(ctx, v_t, attn);             // [hd, q_seq, heads, batch]
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));      // [hd, heads, q_seq, batch]
    return ggml_reshape_3d(ctx, out, static_cast<int64_t>(head_dim) * n_heads, q_seq, batch);
}

// projector.cpp:qformer_layer: post-LN self-attn -> cross-attn -> GELU(erf) FFN.
ggml_tensor * qformer_layer(ggml_context * ctx, ggml_tensor * query, ggml_tensor * enc_window,
                            const GraniteSpeechProjectorBlock & b, int n_heads, int head_dim, float eps) {
    ggml_tensor * sa = qformer_attention(ctx, linear(ctx, query, b.self_q), linear(ctx, query, b.self_k),
                                         linear(ctx, query, b.self_v), n_heads, head_dim);
    sa = linear(ctx, sa, b.self_out);
    sa = ggml_add(ctx, sa, query);
    sa = layer_norm(ctx, sa, b.norm_self, eps);

    ggml_tensor * ca = qformer_attention(ctx, linear(ctx, sa, b.cross_q), linear(ctx, enc_window, b.cross_k),
                                         linear(ctx, enc_window, b.cross_v), n_heads, head_dim);
    ca = linear(ctx, ca, b.cross_out);
    ca = ggml_add(ctx, ca, sa);
    ca = layer_norm(ctx, ca, b.norm_cross, eps);

    ggml_tensor * mid = linear(ctx, ca, b.ffn_up);
    mid = ggml_gelu_erf(ctx, mid);  // PyTorch's default exact-erf GELU
    ggml_tensor * ffn = linear(ctx, mid, b.ffn_down);
    ffn = ggml_add(ctx, ffn, ca);
    return layer_norm(ctx, ffn, b.norm_ffn, eps);
}

struct DecoderParams {
    int64_t n_heads = 0;
    int64_t n_kv_heads = 0;
    int64_t head_dim = 0;
    int max_position = 0;
    float rms_eps = 0.0f;
    float rope_theta = 0.0f;
    float attn_scale = 0.0f;    // attention_multiplier replaces 1/sqrt(head_dim)
    float residual_mul = 0.0f;  // every residual add is x + residual_multiplier * f(x)
};

DecoderParams decoder_params(const GraniteSpeechHParams & hp) {
    DecoderParams p;
    p.n_heads = hp.dec_n_heads;
    p.n_kv_heads = hp.dec_n_kv_heads;
    p.head_dim = hp.dec_head_dim;
    p.max_position = hp.dec_max_position_embeddings;
    p.rms_eps = hp.dec_rms_norm_eps;
    p.rope_theta = hp.dec_rope_theta;
    p.attn_scale = hp.dec_attention_multiplier;
    p.residual_mul = hp.dec_residual_multiplier;
    return p;
}

ggml_tensor * rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions, const DecoderParams & p) {
    // NeoX RoPE, no scaling, same constants as the arch.
    return ggml_rope_ext(ctx, x, positions, nullptr, static_cast<int>(p.head_dim), GGML_ROPE_TYPE_NEOX,
                         p.max_position, p.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

// decoder.cpp:block_prefill (flash branch) + the shared MLP tail.
ggml_tensor * mlp_residual(ggml_context * ctx, ggml_tensor * x, const GraniteSpeechDecoderBlock & b,
                           const DecoderParams & p) {
    ggml_tensor * ff_norm = rms_norm(ctx, x, b.norm_ffn, p.rms_eps);
    ggml_tensor * gate = ggml_mul_mat(ctx, b.gate.tensor, ff_norm);
    ggml_tensor * up = ggml_mul_mat(ctx, b.up.tensor, ff_norm);
    ggml_tensor * ff = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ff = ggml_mul_mat(ctx, b.down.tensor, ff);
    ff = ggml_scale(ctx, ff, p.residual_mul);
    return ggml_add(ctx, x, ff);
}

}  // namespace

// Self-attention KV cache in the arch causal_lm::KvCache layout: one 1-D
// tensor per K / V, [layer][position][kv_head][head_dim], zero-initialised.
struct GraniteSpeechRuntime::KvCache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    int n_ctx = 0;
    ggml_type type = GGML_TYPE_F16;

    ~KvCache() { release(); }

    void release() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        k = v = nullptr;
        n_ctx = 0;
    }

    void ensure(ggml_backend_t backend, int want_n_ctx, ggml_type want_type, const GraniteSpeechHParams & hp) {
        if (buffer == nullptr || n_ctx != want_n_ctx || type != want_type) {
            release();
            ctx = ggml_init({2 * ggml_tensor_overhead() + 256, nullptr, true});
            if (ctx == nullptr) {
                throw std::runtime_error("Granite Speech KV context allocation failed");
            }
            const int64_t elems = static_cast<int64_t>(hp.dec_n_kv_heads) * hp.dec_head_dim * want_n_ctx *
                                  hp.dec_n_layers;
            k = ggml_new_tensor_1d(ctx, want_type, elems);
            v = ggml_new_tensor_1d(ctx, want_type, elems);
            ggml_set_name(k, "granite_speech.kv_k");
            ggml_set_name(v, "granite_speech.kv_v");
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (buffer == nullptr) {
                release();
                throw runtime::CapacityError("Granite Speech KV cache allocation failed (n_ctx=" +
                                             std::to_string(want_n_ctx) + ") - out of memory; lower n_ctx or "
                                             "shorten the audio");
            }
            n_ctx = want_n_ctx;
            type = want_type;
        }
        // Fresh zeros every run, as the arch's per-run kv_init.
        ggml_backend_buffer_clear(buffer, 0);
    }
};

GraniteSpeechRuntime::GraniteSpeechRuntime(
    const GraniteSpeechAssets & assets, const GraniteSpeechWeights & weights, core::ExecutionContext & execution)
    : assets_(assets), weights_(weights), execution_(execution), kv_(std::make_unique<KvCache>()) {}

GraniteSpeechRuntime::~GraniteSpeechRuntime() = default;

// Encoder + projector (arch encoder.cpp:build_encoder_graph +
// projector.cpp:build_projector_graph, model.cpp:run input uploads). Returns
// the audio embeddings [dec_hidden, n_audio_tokens] (column-major per token).
std::vector<float> GraniteSpeechRuntime::encode_audio(const GraniteSpeechFeatures & features,
                                                      int32_t & n_audio_tokens) {
    const auto & hp = assets_.hparams;
    const auto & w = weights_;
    const int T_enc = features.t_enc;
    const int ctx_size = hp.enc_context_size;
    const int n_blocks = (T_enc + ctx_size - 1) / ctx_size;
    const int T_pad = n_blocks * ctx_size;
    const int last_block_rem = T_enc - (n_blocks - 1) * ctx_size;
    const int64_t d_model = hp.enc_hidden;
    const int64_t inner_dim = static_cast<int64_t>(hp.enc_hidden) * hp.enc_conv_expansion;

    GraphArena arena(execution_, kEncoderGraphNodes, "Granite Speech encoder");
    ggml_context * ctx = arena.ctx();
    core::ModuleBuildContext mctx{};
    mctx.ggml = ctx;
    mctx.backend_type = execution_.backend_type();
    mctx.module_instance_name = "granite_speech.encoder";

    ggml_tensor * mel_in = input_tensor(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.enc_input_dim, T_enc), "enc.mel_in");
    ggml_tensor * dists = input_tensor(
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(ctx_size) * ctx_size), "enc.attention_dists");
    ggml_tensor * block_mask = input_tensor(
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ctx_size, ctx_size, n_blocks), "enc.last_block_mask");
    ggml_tensor * zero_pad = nullptr;
    if (T_pad > T_enc) {
        zero_pad = input_tensor(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_pad - T_enc), "enc.zero_pad");
    }

    modules::ShawAttentionConfig shaw;
    shaw.num_heads = hp.enc_n_heads;
    shaw.head_dim = hp.enc_head_dim;
    shaw.context_size = ctx_size;
    shaw.num_blocks = n_blocks;
    shaw.sequence_length = T_enc;
    shaw.layer_norm_eps = kEncoderLayerNormEps;
    const auto dists_value = core::wrap_tensor(
        dists, core::TensorShape::from_dims({static_cast<int64_t>(ctx_size) * ctx_size}), GGML_TYPE_I32);
    const auto mask_value = core::wrap_tensor(
        block_mask, core::TensorShape::from_dims({n_blocks, ctx_size, ctx_size}), GGML_TYPE_F32);
    std::optional<core::TensorValue> zero_pad_value;
    if (zero_pad != nullptr) {
        zero_pad_value = core::wrap_tensor(zero_pad, core::TensorShape::from_dims({1, T_pad - T_enc, d_model}));
    }

    ggml_tensor * x = linear(ctx, mel_in, w.enc_input_linear);  // [d, T_enc]
    const int n_layers = hp.enc_n_layers;
    const int bypass_after = n_layers / 2;  // 1-indexed boundary
    std::vector<ggml_tensor *> exported;
    for (int i = 0; i < n_layers; ++i) {
        const auto & b = w.encoder[static_cast<size_t>(i)];
        x = macaron(ctx, x, b.norm_ff1, b.ff1_up, b.ff1_down);

        // Block-local Shaw attention at T_enc (pads to T_pad internally and
        // slices back before to_out, so the conv never sees pad rows).
        const auto x_value = core::wrap_tensor(x, core::TensorShape::from_dims({1, T_enc, d_model}));
        const auto attn = modules::build_shaw_block_attention(
            mctx, x_value, zero_pad_value, dists_value, mask_value, b.attention, shaw);
        x = ggml_add(ctx, x, attn.tensor);

        x = ggml_add(ctx, x, conv_module(ctx, x, b, hp.enc_conv_kernel_size, inner_dim));
        x = macaron(ctx, x, b.norm_ff2, b.ff2_up, b.ff2_down);
        x = layer_norm(ctx, x, b.norm_post, kEncoderLayerNormEps);

        // cat_hidden_layers (-plus): HF captures the 1-indexed layer output
        // BEFORE the CTC bypass fires.
        for (const int32_t k : hp.enc_cat_hidden_layers) {
            if (i + 1 == k) {
                exported.push_back(x);
                break;
            }
        }
        // Self-conditioned CTC bypass after layer num_layers // 2.
        if (i + 1 == bypass_after) {
            ggml_tensor * ctc = linear(ctx, x, w.enc_ctc_proj);
            ctc = ggml_soft_max(ctx, ctc);
            x = ggml_add(ctx, x, linear(ctx, ctc, w.enc_ctc_bypass));
        }
    }
    // HF: torch.cat([*exported, hidden], dim=-1): captures first, channel axis.
    for (auto it = exported.rbegin(); it != exported.rend(); ++it) {
        x = ggml_concat(ctx, *it, x, 0);
    }

    // ---- Projector ----
    const int window = hp.window_size;
    const int num_queries = hp.prj_num_queries;
    const int n_windows = (T_enc + window - 1) / window;
    const int t_enc_pad = n_windows * window - T_enc;
    const int64_t enc_hidden = hp.prj_encoder_hidden_size;
    const int prj_hidden = hp.prj_hidden;
    const int prj_heads = hp.prj_n_heads;
    const int prj_head_dim = prj_hidden / prj_heads;

    ggml_tensor * enc_pad = nullptr;
    ggml_tensor * enc_full = x;
    if (t_enc_pad > 0) {
        enc_pad = input_tensor(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc_hidden, t_enc_pad), "proj.enc_pad");
        enc_full = ggml_concat(ctx, x, enc_pad, 1);
    }
    // Frame i -> (window i / W, slot i % W): the reference's .view(bsz*nblocks, W, dim).
    ggml_tensor * enc_window = ggml_reshape_3d(ctx, enc_full, enc_hidden, window, n_windows);

    // Learned queries (stored at the checkpoint dtype, BF16 on granite) -> F32,
    // then the Q-Former INPUT layernorm, then broadcast over windows.
    ggml_tensor * query = ggml_cast(ctx, w.proj_query.tensor, GGML_TYPE_F32);
    query = layer_norm(ctx, query, w.proj_qformer_norm, hp.prj_layer_norm_eps);
    query = ggml_repeat_4d(ctx, query, prj_hidden, num_queries, n_windows, 1);
    for (int i = 0; i < hp.prj_n_layers; ++i) {
        query = qformer_layer(ctx, query, enc_window, w.projector[static_cast<size_t>(i)], prj_heads,
                              prj_head_dim, hp.prj_layer_norm_eps);
    }
    // flat index = window * num_queries + query (the reference's .view order).
    ggml_tensor * tokens = ggml_reshape_2d(ctx, query, prj_hidden, static_cast<int64_t>(num_queries) * n_windows);
    ggml_tensor * out = linear(ctx, tokens, w.proj_linear);  // [dec_hidden, n_audio]
    ggml_set_output(out);
    ggml_build_forward_expand(arena.graph(), out);
    arena.allocate();

    // ---- Inputs ----
    ggml_backend_tensor_set(mel_in, features.values.data(), 0, features.values.size() * sizeof(float));
    {
        // encoder.cpp:precompute_attention_dists: dists[c * C + r] =
        // clamp(c - r, -C, C) + max_pos_emb.
        std::vector<int32_t> d(static_cast<size_t>(ctx_size) * ctx_size);
        for (int c = 0; c < ctx_size; ++c) {
            for (int r = 0; r < ctx_size; ++r) {
                const int v = std::clamp(c - r, -ctx_size, ctx_size);
                d[static_cast<size_t>(c) * ctx_size + r] = v + hp.enc_max_pos_emb;
            }
        }
        ggml_backend_tensor_set(dists, d.data(), 0, d.size() * sizeof(int32_t));
    }
    {
        // encoder.cpp:precompute_last_block_mask: -inf on the pad KEY columns
        // of the last block only (masking pad query rows would NaN them).
        const size_t plane = static_cast<size_t>(ctx_size) * ctx_size;
        std::vector<float> m(plane * static_cast<size_t>(n_blocks), 0.0f);
        if (last_block_rem > 0 && last_block_rem < ctx_size) {
            float * last = m.data() + plane * static_cast<size_t>(n_blocks - 1);
            for (int q = 0; q < ctx_size; ++q) {
                for (int k = last_block_rem; k < ctx_size; ++k) {
                    last[static_cast<size_t>(q) * ctx_size + k] = -std::numeric_limits<float>::infinity();
                }
            }
        }
        ggml_backend_tensor_set(block_mask, m.data(), 0, m.size() * sizeof(float));
    }
    for (ggml_tensor * zeros : {zero_pad, enc_pad}) {
        if (zeros != nullptr) {
            const std::vector<float> z(static_cast<size_t>(ggml_nelements(zeros)), 0.0f);
            ggml_backend_tensor_set(zeros, z.data(), 0, z.size() * sizeof(float));
        }
    }

    arena.compute();
    n_audio_tokens = static_cast<int32_t>(out->ne[1]);
    std::vector<float> audio(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, audio.data(), 0, audio.size() * sizeof(float));
    return audio;
}

GraniteSpeechDecodeResult GraniteSpeechRuntime::transcribe(
    const std::vector<float> & pcm_16k,
    const std::vector<int32_t> & prefix_ids,
    const std::vector<int32_t> & suffix_ids,
    const GraniteSpeechDecodeLimits & limits,
    const std::function<void(int64_t, int64_t)> & poll) {
    const auto & hp = assets_.hparams;
    const auto & w = weights_;
    const auto p = decoder_params(hp);
    if (poll) {
        poll(0, 1);
    }

    // ---- Frontend + encoder + projector ----
    const auto features = compute_granite_speech_features(pcm_16k, assets_.frontend, execution_.config().threads);
    GraniteSpeechDecodeResult result;
    const std::vector<float> audio = encode_audio(features, result.n_audio_tokens);
    const int n_audio = result.n_audio_tokens;
    if (poll) {
        poll(0, 1);
    }

    // ---- Prompt + input gate (arch run()) ----
    const int prefix_len = static_cast<int>(prefix_ids.size());
    const int suffix_len = static_cast<int>(suffix_ids.size());
    const int T_prompt = prefix_len + n_audio + suffix_len;
    int ceiling = hp.dec_max_position_embeddings;
    if (limits.n_ctx > 0 && limits.n_ctx < ceiling) {
        ceiling = limits.n_ctx;
    }
    if (T_prompt + kGenReserve > ceiling) {
        throw runtime::InputTooLong(
            "Granite Speech input too long: " + std::to_string(n_audio) + " audio + " +
            std::to_string(prefix_len + suffix_len) + " prompt tokens leave no room for output within the " +
            std::to_string(ceiling) + "-token context (need " + std::to_string(T_prompt + kGenReserve) +
            "); shorten or split the audio");
    }
    int gen_budget = pick_decode_budget(predict_transcript_tokens(n_audio, assets_.ms_per_audio_token), kGenReserve,
                                        T_prompt, ceiling);
    if (limits.max_tokens > 0) {
        // Caller cap (engine extension; the arch has no per-run knob).
        gen_budget = static_cast<int>(std::min<int64_t>(limits.max_tokens, ceiling - T_prompt));
    }
    const int needed_raw = std::min(T_prompt + gen_budget, ceiling);
    const int needed_n_ctx = ((needed_raw + kKvBucket - 1) / kKvBucket) * kKvBucket;
    kv_->ensure(execution_.backend(), needed_n_ctx, limits.kv_type, hp);
    const int n_ctx = kv_->n_ctx;
    const int64_t kv_dim = p.n_kv_heads * p.head_dim;
    const int64_t q_dim = p.n_heads * p.head_dim;
    const size_t k_elem = ggml_element_size(kv_->k);
    const size_t v_elem = ggml_element_size(kv_->v);
    const int64_t hidden = hp.dec_hidden;
    const int64_t vocab = hp.dec_vocab_size;
    const float inv_logits = 1.0f / hp.dec_logits_scaling;

    // ---- Prefill (decoder.cpp:build_prefill_graph) ----
    std::vector<float> last_logits(static_cast<size_t>(vocab));
    {
        GraphArena arena(execution_, kPrefillGraphNodes, "Granite Speech prefill");
        ggml_context * ctx = arena.ctx();
        ggml_cgraph * gf = arena.graph();
        ggml_tensor * ids_in = input_tensor(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt), "dec.input_ids");
        ggml_tensor * audio_in = input_tensor(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_audio), "dec.audio");
        ggml_tensor * pos_in = input_tensor(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt), "dec.positions");
        ggml_tensor * mask_in = input_tensor(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T_prompt, T_prompt), "dec.mask");

        // Audio injection by 3-way concat of prefix embeddings | audio | suffix.
        ggml_tensor * emb = ggml_get_rows(ctx, w.token_embedding.tensor, ids_in);
        const size_t emb_elem = ggml_element_size(emb);
        ggml_tensor * x = audio_in;
        if (prefix_len > 0) {
            ggml_tensor * pre = ggml_cont(ctx, ggml_view_2d(ctx, emb, hidden, prefix_len, emb_elem * hidden, 0));
            x = ggml_concat(ctx, pre, x, 1);
        }
        if (suffix_len > 0) {
            ggml_tensor * suf = ggml_cont(ctx, ggml_view_2d(ctx, emb, hidden, suffix_len, emb_elem * hidden,
                emb_elem * hidden * static_cast<size_t>(prefix_len + n_audio)));
            x = ggml_concat(ctx, x, suf, 1);
        }
        x = ggml_scale(ctx, x, hp.dec_embedding_multiplier);  // after the injection

        for (int il = 0; il < hp.dec_n_layers; ++il) {
            const auto & b = w.decoder[static_cast<size_t>(il)];
            ggml_tensor * xn = rms_norm(ctx, x, b.norm_attn, p.rms_eps);
            ggml_tensor * Q = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, b.q.tensor, xn), p.head_dim, p.n_heads, T_prompt, 1);
            ggml_tensor * K = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, b.k.tensor, xn), p.head_dim, p.n_kv_heads, T_prompt, 1);
            ggml_tensor * V = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, b.v.tensor, xn), p.head_dim, p.n_kv_heads, T_prompt, 1);
            Q = rope(ctx, Q, pos_in, p);
            K = rope(ctx, K, pos_in, p);

            // KV write, expanded NOW so it precedes the attention reads.
            const size_t layer_off = static_cast<size_t>(il) * static_cast<size_t>(n_ctx) * static_cast<size_t>(kv_dim);
            const size_t n_elem = static_cast<size_t>(T_prompt) * static_cast<size_t>(kv_dim);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, K, ggml_view_1d(ctx, kv_->k, n_elem, k_elem * layer_off)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, V, ggml_view_1d(ctx, kv_->v, n_elem, v_elem * layer_off)));
            ggml_tensor * K_att = ggml_view_3d(ctx, kv_->k, p.head_dim, T_prompt, p.n_kv_heads, k_elem * kv_dim,
                                               k_elem * p.head_dim, k_elem * layer_off);
            ggml_tensor * V_att = ggml_view_3d(ctx, kv_->v, p.head_dim, T_prompt, p.n_kv_heads, v_elem * kv_dim,
                                               v_elem * p.head_dim, v_elem * layer_off);
            ggml_tensor * Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            ggml_tensor * o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask_in, p.attn_scale, 0.0f, 0.0f);
            o = ggml_reshape_2d(ctx, o, q_dim, T_prompt);
            o = ggml_mul_mat(ctx, b.o.tensor, o);
            o = ggml_scale(ctx, o, p.residual_mul);
            x = ggml_add(ctx, x, o);
            x = mlp_residual(ctx, x, b, p);
        }
        x = rms_norm(ctx, x, w.output_norm, p.rms_eps);
        ggml_tensor * last_x = ggml_cont(ctx, ggml_view_2d(ctx, x, hidden, 1, ggml_element_size(x) * hidden,
            ggml_element_size(x) * hidden * static_cast<size_t>(T_prompt - 1)));
        ggml_tensor * logits = ggml_mul_mat(ctx, w.output.tensor, last_x);
        logits = ggml_reshape_1d(ctx, logits, vocab);
        logits = ggml_scale(ctx, logits, inv_logits);  // logits / logits_scaling
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        arena.allocate();

        // Audio positions carry id 0 (HF masks audio_token_id to 0 before the
        // lookup; those rows are replaced by the concat anyway).
        std::vector<int32_t> ids;
        ids.reserve(static_cast<size_t>(T_prompt));
        ids.insert(ids.end(), prefix_ids.begin(), prefix_ids.end());
        ids.insert(ids.end(), static_cast<size_t>(n_audio), 0);
        ids.insert(ids.end(), suffix_ids.begin(), suffix_ids.end());
        ggml_backend_tensor_set(ids_in, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(audio_in, audio.data(), 0, audio.size() * sizeof(float));
        std::vector<int32_t> positions(static_cast<size_t>(T_prompt));
        for (int i = 0; i < T_prompt; ++i) {
            positions[static_cast<size_t>(i)] = i;
        }
        ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
        std::vector<uint16_t> mask(static_cast<size_t>(T_prompt) * T_prompt, kF16Zero);
        for (int q = 0; q < T_prompt; ++q) {
            for (int k = q + 1; k < T_prompt; ++k) {
                mask[static_cast<size_t>(q) * T_prompt + k] = kF16NegInf;
            }
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(uint16_t));
        arena.compute();
        ggml_backend_tensor_get(logits, last_logits.data(), 0, last_logits.size() * sizeof(float));
    }

    // Host argmax, first maximum wins (arch argmax_logits).
    int32_t next_id = 0;
    {
        float best = last_logits[0];
        for (int32_t i = 1; i < static_cast<int32_t>(last_logits.size()); ++i) {
            if (last_logits[static_cast<size_t>(i)] > best) {
                best = last_logits[static_cast<size_t>(i)];
                next_id = i;
            }
        }
    }

    // ---- Greedy step loop (decoder.cpp:build_step_graph + model.cpp:run) ----
    const int max_n_kv = n_ctx;
    const int max_steps = std::max(0, std::min({gen_budget, max_n_kv - T_prompt, ceiling - T_prompt}));
    const int32_t eos_id = assets_.tokenizer.eos_id();
    if (max_steps > 0 && next_id != eos_id) {
        GraphArena arena(execution_, kStepGraphNodes, "Granite Speech decode step");
        ggml_context * ctx = arena.ctx();
        ggml_cgraph * gf = arena.graph();
        ggml_tensor * id_in = input_tensor(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), "step.input_id");
        ggml_tensor * pos_in = input_tensor(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), "step.position");
        ggml_tensor * row_in = input_tensor(ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1), "step.kv_idx");
        ggml_tensor * mask_in = input_tensor(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, max_n_kv, 1), "step.mask");

        ggml_tensor * x = ggml_get_rows(ctx, w.token_embedding.tensor, id_in);  // [hidden, 1]
        x = ggml_scale(ctx, x, hp.dec_embedding_multiplier);
        for (int il = 0; il < hp.dec_n_layers; ++il) {
            const auto & b = w.decoder[static_cast<size_t>(il)];
            ggml_tensor * xn = rms_norm(ctx, x, b.norm_attn, p.rms_eps);
            ggml_tensor * Q = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, b.q.tensor, xn), p.head_dim, p.n_heads, 1, 1);
            ggml_tensor * K = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, b.k.tensor, xn), p.head_dim, p.n_kv_heads, 1, 1);
            ggml_tensor * V = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, b.v.tensor, xn), p.head_dim, p.n_kv_heads, 1, 1);
            Q = rope(ctx, Q, pos_in, p);
            K = rope(ctx, K, pos_in, p);

            const size_t off_k = k_elem * static_cast<size_t>(il) * static_cast<size_t>(n_ctx) * static_cast<size_t>(kv_dim);
            const size_t off_v = v_elem * static_cast<size_t>(il) * static_cast<size_t>(n_ctx) * static_cast<size_t>(kv_dim);
            ggml_tensor * k_layer = ggml_view_2d(ctx, kv_->k, kv_dim, n_ctx, k_elem * kv_dim, off_k);
            ggml_tensor * v_layer = ggml_view_2d(ctx, kv_->v, kv_dim, n_ctx, v_elem * kv_dim, off_v);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_layer, ggml_reshape_2d(ctx, K, kv_dim, 1), row_in));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_layer, ggml_reshape_2d(ctx, V, kv_dim, 1), row_in));
            ggml_tensor * K_att = ggml_view_3d(ctx, kv_->k, p.head_dim, max_n_kv, p.n_kv_heads, k_elem * kv_dim,
                                               k_elem * p.head_dim, off_k);
            ggml_tensor * V_att = ggml_view_3d(ctx, kv_->v, p.head_dim, max_n_kv, p.n_kv_heads, v_elem * kv_dim,
                                               v_elem * p.head_dim, off_v);
            ggml_tensor * Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            ggml_tensor * o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask_in, p.attn_scale, 0.0f, 0.0f);
            o = ggml_reshape_2d(ctx, o, q_dim, 1);
            o = ggml_mul_mat(ctx, b.o.tensor, o);
            o = ggml_scale(ctx, o, p.residual_mul);
            x = ggml_add(ctx, x, o);
            x = mlp_residual(ctx, x, b, p);
        }
        x = rms_norm(ctx, x, w.output_norm, p.rms_eps);
        ggml_tensor * logits = ggml_mul_mat(ctx, w.output.tensor, x);
        logits = ggml_reshape_1d(ctx, logits, vocab);
        logits = ggml_scale(ctx, logits, inv_logits);
        ggml_tensor * amax = ggml_argmax(ctx, logits);
        ggml_set_output(amax);
        ggml_build_forward_expand(gf, amax);
        arena.allocate();

        std::vector<uint16_t> step_mask(static_cast<size_t>(max_n_kv), kF16NegInf);
        result.tokens.reserve(static_cast<size_t>(max_steps));
        for (int step = 0; step < max_steps; ++step) {
            if (next_id == eos_id) {
                break;
            }
            if (poll) {
                poll(step, max_steps);
            }
            result.tokens.push_back(next_id);
            const int32_t pos = T_prompt + step;  // RoPE position == KV row
            const int64_t row = pos;
            std::fill(step_mask.begin(), step_mask.end(), kF16NegInf);
            std::fill(step_mask.begin(), step_mask.begin() + pos + 1, kF16Zero);
            // Every input, every step (cached-graph rule).
            ggml_backend_tensor_set(id_in, &next_id, 0, sizeof(int32_t));
            ggml_backend_tensor_set(pos_in, &pos, 0, sizeof(int32_t));
            ggml_backend_tensor_set(row_in, &row, 0, sizeof(int64_t));
            ggml_backend_tensor_set(mask_in, step_mask.data(), 0, step_mask.size() * sizeof(uint16_t));
            arena.compute();
            int32_t amax_id = 0;
            ggml_backend_tensor_get(amax, &amax_id, 0, sizeof(int32_t));
            next_id = amax_id;
        }
    }
    // Stopped at the budget / context before EOS: keep the partial transcript
    // and say so (arch run(): transcribe_was_truncated).
    result.truncated = next_id != eos_id;
    if (poll) {
        poll(max_steps, max_steps);
    }
    return result;
}

}  // namespace engine::models::granite_speech
