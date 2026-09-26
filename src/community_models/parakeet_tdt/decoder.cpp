#include "engine/community_models/parakeet_tdt/decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

using Clock = std::chrono::steady_clock;
struct GgmlDeleter { void operator()(ggml_context* c) const noexcept { if (c) ggml_free(c); } };

// Frames between two progress / cancellation polls of the greedy loops
// (25 encoder frames = 2 s of audio at the 80 ms frame rate).
constexpr int64_t kProgressFrames = 25;

int32_t argmax_vocab(const std::vector<float>& v, int64_t vocab_sz) {
    return static_cast<int32_t>(std::distance(v.begin(),
        std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(vocab_sz))));
}

int32_t argmax_dur(const std::vector<float>& v, int64_t vocab_sz, int64_t n_dur) {
    auto s = v.begin() + static_cast<std::ptrdiff_t>(vocab_sz);
    return static_cast<int32_t>(std::distance(s, std::max_element(s, s + static_cast<std::ptrdiff_t>(n_dur))));
}

// arch/parakeet/decoder.cpp argmax_range: ties go to the first occurrence.
int32_t argmax_range(const float * data, int64_t n) {
    int64_t best_i = 0;
    float best_v = data[0];
    for (int64_t i = 1; i < n; ++i) {
        if (data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    return static_cast<int32_t>(best_i);
}

// arch/parakeet/decoder.cpp token_confidence: 1 - entropy(softmax) / log(n).
float token_confidence(const float * token_logits, int64_t n, std::vector<float> & scratch) {
    float max_logit = token_logits[0];
    for (int64_t i = 1; i < n; ++i) {
        max_logit = std::max(max_logit, token_logits[i]);
    }
    scratch.resize(static_cast<size_t>(n));
    double sum_exp = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float e = std::exp(token_logits[i] - max_logit);
        scratch[static_cast<size_t>(i)] = e;
        sum_exp += static_cast<double>(e);
    }
    const float inv_sum = static_cast<float>(1.0 / sum_exp);
    double entropy = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float p = scratch[static_cast<size_t>(i)] * inv_sum;
        entropy -= static_cast<double>(p) * std::log(static_cast<double>(p) + 1e-10);
    }
    const double max_entropy = std::log(static_cast<double>(n));
    if (max_entropy <= 0.0) {
        return 1.0f;
    }
    return static_cast<float>(1.0 - entropy / max_entropy);
}

// One small cached graph and its allocator.
struct OwnedGraph {
    std::unique_ptr<ggml_context, GgmlDeleter> ggml;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    engine::core::HostGraphPlan plan;
    ~OwnedGraph() { if (alloc) ggml_gallocr_free(alloc); }
};

ggml_context * init_small_context(OwnedGraph & g, size_t tensors) {
    ggml_init_params p{ggml_tensor_overhead() * tensors + ggml_graph_overhead(), nullptr, true};
    g.ggml.reset(ggml_init(p));
    if (!g.ggml) {
        throw std::runtime_error("Parakeet decoder: ggml_init failed");
    }
    return g.ggml.get();
}

void finalize_graph(
    OwnedGraph & g,
    engine::core::ExecutionContext & ec,
    std::initializer_list<ggml_tensor *> outputs,
    const char * label) {
    g.graph = ggml_new_graph(g.ggml.get());
    for (ggml_tensor * t : outputs) {
        ggml_set_output(t);
        ggml_build_forward_expand(g.graph, t);
    }
    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ec.backend()));
    if (!g.alloc || !ggml_gallocr_alloc_graph(g.alloc, g.graph)) {
        throw std::runtime_error(std::string(label) + " graph alloc failed");
    }
    engine::core::validate_backend_graph_supported(ec.backend(), g.graph, label);
    engine::core::prepare_host_graph_plan(ec, g.graph, g.plan);
}

ggml_tensor * joint_activation(ggml_context * ctx, ggml_tensor * x, ParakeetJointActivation act) {
    switch (act) {
        case ParakeetJointActivation::Relu: return ggml_relu(ctx, x);
        case ParakeetJointActivation::Sigmoid: return ggml_sigmoid(ctx, x);
        case ParakeetJointActivation::Tanh: return ggml_tanh(ctx, x);
    }
    return ggml_relu(ctx, x);
}

}  // namespace

struct ParakeetDecoderRuntime::StepGraph {
    std::unique_ptr<ggml_context, GgmlDeleter> ggml;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    engine::core::HostGraphPlan plan;
    engine::core::TensorValue token_id, enc_frame;
    std::vector<engine::core::TensorValue> h_in, c_in, h_out, c_out;
    engine::core::TensorValue pred_cache, logits;
    ~StepGraph() { if (alloc) ggml_gallocr_free(alloc); }
};

struct ParakeetDecoderRuntime::JointGraph {
    std::unique_ptr<ggml_context, GgmlDeleter> ggml;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    engine::core::HostGraphPlan plan;
    engine::core::TensorValue enc_frame, pred_cache, logits;
    ~JointGraph() { if (alloc) ggml_gallocr_free(alloc); }
};

// TranscribeGguf decoder graphs: the arch's host decoder rebuilt on the
// session backend, op for op (arch/parakeet/decoder.cpp build_pred_graph,
// build_joint_graph, build_joint_graph_batch). Every input is uploaded before
// every compute, so the cached graphs are safe to reuse.
struct ParakeetDecoderRuntime::TranscribeGraphs {
    // Predictor: x [H] (host embedding row, zeros for the start state),
    // per-layer (h, c) in -> (h, c) out.
    OwnedGraph pred;
    engine::core::TensorValue pred_x;
    std::vector<engine::core::TensorValue> pred_h_in, pred_c_in;
    std::vector<ggml_tensor *> pred_h_out, pred_c_out;
    // Joint over W consecutive frames: pred_in [H] and the precomputed
    // encoder projections [W, J] -> logits [W, N]. W = 1 (serial) and, for
    // RNN-T, the arch's default W = 16 window.
    struct Joint {
        OwnedGraph graph;
        int64_t width = 1;
        engine::core::TensorValue pred_in, enc_in;
        ggml_tensor * logits = nullptr;
    };
    Joint serial;
    Joint window;
};

ParakeetDecoderRuntime::ParakeetDecoderRuntime(
    std::shared_ptr<const ParakeetTDTAssets> a, std::shared_ptr<const ParakeetWeights> w,
    engine::core::ExecutionContext& ec, size_t arena)
    : assets_(std::move(a)), weights_(std::move(w)), execution_context_(&ec), graph_arena_bytes_(arena) {
    if (!assets_ || !weights_) throw std::runtime_error("decoder requires assets/weights");
}
ParakeetDecoderRuntime::~ParakeetDecoderRuntime() = default;
void ParakeetDecoderRuntime::prepare() {
    if (assets_->transcribe_layout()) {
        ensure_transcribe_graphs();
        return;
    }
    ensure_step_graph();
    ensure_joint_graph();
}

void ParakeetDecoderRuntime::ensure_step_graph() {
    if (step_graph_) return;
    const auto t0 = Clock::now();
    const auto& cfg = assets_->config; const auto& dw = weights_->decoder;
    auto g = std::make_unique<StepGraph>();
    ggml_init_params p{graph_arena_bytes_, nullptr, true};
    g->ggml.reset(ggml_init(p));
    engine::core::ModuleBuildContext ctx{g->ggml.get(), "parakeet.decoder", execution_context_->backend_type()};

    g->token_id = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1}));
    g->enc_frame = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.encoder.hidden_size}));
    for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
        g->h_in.push_back(engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.decoder_hidden_size})));
        g->c_in.push_back(engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.decoder_hidden_size})));
    }
    ggml_set_input(g->token_id.tensor); ggml_set_input(g->enc_frame.tensor);
    for (int64_t l = 0; l < cfg.decoder_layers; ++l) { ggml_set_input(g->h_in[static_cast<size_t>(l)].tensor); ggml_set_input(g->c_in[static_cast<size_t>(l)].tensor); }

    auto h = engine::modules::EmbeddingModule({cfg.vocab_size, cfg.decoder_hidden_size}).build(ctx, g->token_id, dw.embedding);
    h = engine::core::reshape_tensor(ctx, h, engine::core::TensorShape::from_dims({1, cfg.decoder_hidden_size}));
    for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
        auto out = engine::modules::LSTMCellModule({cfg.decoder_hidden_size, cfg.decoder_hidden_size})
                       .build(ctx, h, g->h_in[static_cast<size_t>(l)], g->c_in[static_cast<size_t>(l)], dw.lstm_layers[static_cast<size_t>(l)]);
        g->h_out.push_back(out.hidden); g->c_out.push_back(out.cell); h = g->h_out.back();
    }
    auto pred = engine::modules::LinearModule({cfg.decoder_hidden_size, cfg.decoder_hidden_size, true}).build(ctx, h, dw.decoder_projector);
    g->pred_cache = pred;
    auto proj_enc = engine::modules::LinearModule({cfg.encoder.hidden_size, cfg.decoder_hidden_size, true}).build(ctx, g->enc_frame, dw.joint_enc);
    auto joint = engine::modules::AddModule().build(ctx, proj_enc, g->pred_cache);
    joint = engine::modules::ReluModule().build(ctx, joint);
    g->logits = engine::modules::LinearModule({cfg.decoder_hidden_size, static_cast<int64_t>(cfg.vocab_size + cfg.durations.size()), true}).build(ctx, joint, dw.joint_head);

    ggml_set_output(g->logits.tensor); ggml_set_output(g->pred_cache.tensor);
    for (size_t l = 0; l < g->h_out.size(); ++l) { ggml_set_output(g->h_out[l].tensor); ggml_set_output(g->c_out[l].tensor); }

    g->graph = ggml_new_graph(g->ggml.get());
    ggml_build_forward_expand(g->graph, g->logits.tensor); ggml_build_forward_expand(g->graph, g->pred_cache.tensor);
    for (size_t l = 0; l < g->h_out.size(); ++l) { ggml_build_forward_expand(g->graph, g->h_out[l].tensor); ggml_build_forward_expand(g->graph, g->c_out[l].tensor); }

    g->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (!g->alloc || !ggml_gallocr_alloc_graph(g->alloc, g->graph)) throw std::runtime_error("step graph alloc failed");
    engine::core::validate_backend_graph_supported(execution_context_->backend(), g->graph, "Parakeet decoder");
    engine::core::prepare_host_graph_plan(*execution_context_, g->graph, g->plan);

    size_t vt = static_cast<size_t>(cfg.vocab_size + cfg.durations.size());
    logits_scratch_.assign(vt, 0.f);
    hidden_scratch_.assign(static_cast<size_t>(cfg.decoder_layers * cfg.decoder_hidden_size), 0.f);
    cell_scratch_.assign(static_cast<size_t>(cfg.decoder_layers * cfg.decoder_hidden_size), 0.f);
    decoder_cache_scratch_.assign(static_cast<size_t>(cfg.decoder_hidden_size), 0.f);
    hidden_read_scratch_.assign(static_cast<size_t>(cfg.decoder_hidden_size), 0.f);
    cell_read_scratch_.assign(static_cast<size_t>(cfg.decoder_hidden_size), 0.f);
    step_graph_ = std::move(g);
    debug::timing_log_scalar("parakeet.decoder.graph_build_ms", engine::debug::elapsed_ms(t0, Clock::now()));
}

void ParakeetDecoderRuntime::ensure_joint_graph() {
    if (joint_graph_) return;
    const auto t0 = Clock::now(); const auto& cfg = assets_->config;
    auto g = std::make_unique<JointGraph>();
    ggml_init_params p{graph_arena_bytes_, nullptr, true};
    g->ggml.reset(ggml_init(p));
    engine::core::ModuleBuildContext ctx{g->ggml.get(), "parakeet.decoder_joint", execution_context_->backend_type()};
    g->enc_frame = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.encoder.hidden_size}));
    g->pred_cache = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.decoder_hidden_size}));
    ggml_set_input(g->enc_frame.tensor); ggml_set_input(g->pred_cache.tensor);
    auto proj_enc = engine::modules::LinearModule({cfg.encoder.hidden_size, cfg.decoder_hidden_size, true}).build(ctx, g->enc_frame, weights_->decoder.joint_enc);
    auto joint = engine::modules::AddModule().build(ctx, proj_enc, g->pred_cache);
    joint = engine::modules::ReluModule().build(ctx, joint);
    g->logits = engine::modules::LinearModule({cfg.decoder_hidden_size, static_cast<int64_t>(cfg.vocab_size + cfg.durations.size()), true}).build(ctx, joint, weights_->decoder.joint_head);
    ggml_set_output(g->logits.tensor);
    g->graph = ggml_new_graph(g->ggml.get()); ggml_build_forward_expand(g->graph, g->logits.tensor);
    g->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (!g->alloc || !ggml_gallocr_alloc_graph(g->alloc, g->graph)) throw std::runtime_error("joint graph alloc failed");
    engine::core::validate_backend_graph_supported(execution_context_->backend(), g->graph, "Parakeet decoder joint");
    engine::core::prepare_host_graph_plan(*execution_context_, g->graph, g->plan);
    joint_graph_ = std::move(g);
    debug::timing_log_scalar("parakeet.decoder.joint_graph_build_ms", engine::debug::elapsed_ms(t0, Clock::now()));
}

void ParakeetDecoderRuntime::ensure_transcribe_graphs() {
    if (transcribe_graphs_) return;
    const auto & cfg = assets_->config;
    auto g = std::make_unique<TranscribeGraphs>();
    if (cfg.head_kind == ParakeetHeadKind::Ctc) {
        transcribe_graphs_ = std::move(g);  // CTC projects per call (project_frames)
        return;
    }
    const auto t0 = Clock::now();
    const auto & dw = weights_->decoder;
    const int64_t H = cfg.decoder_hidden_size;
    const int64_t J = cfg.joint_hidden_size;
    const auto backend_type = execution_context_->backend_type();

    // arch decoder.cpp build_pred_graph:
    //   gates = (Wx @ x + Wh @ h) + b; c' = f*c + i*g; h' = o*tanh(c'), [i,f,g,o].
    {
        auto & pg = g->pred;
        ggml_context * gctx = init_small_context(pg, static_cast<size_t>(32 * cfg.decoder_layers + 16));
        engine::core::ModuleBuildContext ctx{gctx, "parakeet.transcribe.pred", backend_type};
        g->pred_x = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, H}));
        ggml_set_input(g->pred_x.tensor);
        ggml_tensor * in = g->pred_x.tensor;
        std::vector<ggml_tensor *> outputs;
        for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
            const auto & cell = dw.lstm_layers[static_cast<size_t>(l)];
            auto h_in = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, H}));
            auto c_in = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, H}));
            ggml_set_input(h_in.tensor);
            ggml_set_input(c_in.tensor);
            g->pred_h_in.push_back(h_in);
            g->pred_c_in.push_back(c_in);
            ggml_tensor * gates = ggml_add(gctx,
                ggml_add(gctx, ggml_mul_mat(gctx, cell.weight_ih.tensor, in), ggml_mul_mat(gctx, cell.weight_hh.tensor, h_in.tensor)),
                cell.bias_ih.tensor);
            gates = ggml_reshape_1d(gctx, gates, 4 * H);
            auto part = [&](int64_t k) {
                return ggml_view_1d(gctx, gates, H, static_cast<size_t>(k * H) * ggml_element_size(gates));
            };
            ggml_tensor * i_ = ggml_sigmoid(gctx, part(0));
            ggml_tensor * f_ = ggml_sigmoid(gctx, part(1));
            ggml_tensor * gg = ggml_tanh(gctx, part(2));
            ggml_tensor * o_ = ggml_sigmoid(gctx, part(3));
            ggml_tensor * c_prev = ggml_reshape_1d(gctx, c_in.tensor, H);
            ggml_tensor * c_new = ggml_add(gctx, ggml_mul(gctx, f_, c_prev), ggml_mul(gctx, i_, gg));
            ggml_tensor * h_new = ggml_mul(gctx, o_, ggml_tanh(gctx, c_new));
            g->pred_h_out.push_back(h_new);
            g->pred_c_out.push_back(c_new);
            outputs.push_back(h_new);
            outputs.push_back(c_new);
            in = h_new;
        }
        pg.graph = ggml_new_graph(gctx);
        for (ggml_tensor * t : outputs) {
            ggml_set_output(t);
            ggml_build_forward_expand(pg.graph, t);
        }
        pg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
        if (!pg.alloc || !ggml_gallocr_alloc_graph(pg.alloc, pg.graph)) {
            throw std::runtime_error("Parakeet predictor graph alloc failed");
        }
        engine::core::validate_backend_graph_supported(execution_context_->backend(), pg.graph, "Parakeet predictor");
        engine::core::prepare_host_graph_plan(*execution_context_, pg.graph, pg.plan);
    }

    // arch decoder.cpp build_joint_graph / build_joint_graph_batch:
    //   pred_proj = pred_w @ pred_in + pred_b; summed = enc_in + pred_proj;
    //   logits = out_w @ act(summed) + out_b.
    auto build_joint = [&](TranscribeGraphs::Joint & joint, int64_t width) {
        joint.width = width;
        ggml_context * gctx = init_small_context(joint.graph, 24);
        engine::core::ModuleBuildContext ctx{gctx, "parakeet.transcribe.joint", backend_type};
        joint.pred_in = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, H}));
        joint.enc_in = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({width, J}));
        ggml_set_input(joint.pred_in.tensor);
        ggml_set_input(joint.enc_in.tensor);
        ggml_tensor * pred_proj = ggml_add(gctx,
            ggml_mul_mat(gctx, dw.decoder_projector.weight.tensor, joint.pred_in.tensor), dw.decoder_projector.bias->tensor);
        ggml_tensor * summed = ggml_add(gctx, joint.enc_in.tensor, pred_proj);
        ggml_tensor * activated = joint_activation(gctx, summed, cfg.joint_activation);
        joint.logits = ggml_add(gctx, ggml_mul_mat(gctx, dw.joint_head.weight.tensor, activated), dw.joint_head.bias->tensor);
        finalize_graph(joint.graph, *execution_context_, {joint.logits}, "Parakeet joint");
    };
    build_joint(g->serial, 1);
    if (cfg.head_kind == ParakeetHeadKind::Rnnt) {
        // arch decoder.cpp decode_rnnt_greedy: resolve_joint_batch_w(16).
        build_joint(g->window, 16);
    }
    transcribe_graphs_ = std::move(g);
    debug::timing_log_scalar("parakeet.decoder.transcribe_graph_build_ms", engine::debug::elapsed_ms(t0, Clock::now()));
}

int32_t ParakeetDecoderRuntime::run_joint_step(const float* enc, int32_t* out_dur_id) {
    auto& g = *joint_graph_;
    const auto& cfg = assets_->config;
    engine::core::write_tensor_f32(g.enc_frame, enc, static_cast<size_t>(cfg.encoder.hidden_size));
    engine::core::write_tensor_f32(g.pred_cache, decoder_cache_scratch_);
    if (engine::core::compute_graph(*execution_context_, g.graph, g.plan, "Parakeet joint") != GGML_STATUS_SUCCESS)
        throw std::runtime_error("joint compute failed");
    engine::core::read_tensor_f32_into(g.logits.tensor, logits_scratch_);
    if (out_dur_id) *out_dur_id = argmax_dur(logits_scratch_, assets_->config.vocab_size, static_cast<int64_t>(assets_->config.durations.size()));
    return argmax_vocab(logits_scratch_, assets_->config.vocab_size);
}

int32_t ParakeetDecoderRuntime::run_step(int32_t tok, const float* enc, bool pred_valid, int32_t* out_dur_id) {
    auto& g = *step_graph_;
    const auto& cfg = assets_->config;
    const int32_t blank = static_cast<int32_t>(cfg.blank_token_id);
    if (!pred_valid || tok != blank) {
        engine::core::write_tensor_i32(g.token_id, &tok, 1);
        engine::core::write_tensor_f32(g.enc_frame, enc, static_cast<size_t>(cfg.encoder.hidden_size));
        for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
            size_t off = static_cast<size_t>(l * cfg.decoder_hidden_size);
            engine::core::write_tensor_f32(g.h_in[static_cast<size_t>(l)], hidden_scratch_.data() + off, static_cast<size_t>(cfg.decoder_hidden_size));
            engine::core::write_tensor_f32(g.c_in[static_cast<size_t>(l)], cell_scratch_.data() + off, static_cast<size_t>(cfg.decoder_hidden_size));
        }
        if (engine::core::compute_graph(*execution_context_, g.graph, g.plan, "Parakeet step") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("step compute failed");
        engine::core::read_tensor_f32_into(g.logits.tensor, logits_scratch_);
        engine::core::read_tensor_f32_into(g.pred_cache.tensor, decoder_cache_scratch_);
        for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
            size_t off = static_cast<size_t>(l * cfg.decoder_hidden_size);
            engine::core::read_tensor_f32_into(g.h_out[static_cast<size_t>(l)].tensor, hidden_read_scratch_);
            std::copy(hidden_read_scratch_.begin(), hidden_read_scratch_.end(), hidden_scratch_.begin() + static_cast<std::ptrdiff_t>(off));
            engine::core::read_tensor_f32_into(g.c_out[static_cast<size_t>(l)].tensor, cell_read_scratch_);
            std::copy(cell_read_scratch_.begin(), cell_read_scratch_.end(), cell_scratch_.begin() + static_cast<std::ptrdiff_t>(off));
        }
    } else {
        return run_joint_step(enc, out_dur_id);
    }
    if (out_dur_id) *out_dur_id = argmax_dur(logits_scratch_, assets_->config.vocab_size, static_cast<int64_t>(assets_->config.durations.size()));
    return argmax_vocab(logits_scratch_, assets_->config.vocab_size);
}

std::string ParakeetDecoderRuntime::decode_text(const std::vector<int32_t>& ids, bool keep_tags) const {
    std::vector<int32_t> f; f.reserve(ids.size());
    for (auto id : ids) {
        if (id == static_cast<int32_t>(assets_->config.blank_token_id) || id == static_cast<int32_t>(assets_->config.pad_token_id)) continue;
        if (!keep_tags && id >= 0 && id < static_cast<int32_t>(assets_->special_token_ids.size()) && assets_->special_token_ids[static_cast<size_t>(id)]) continue;
        f.push_back(id);
    }
    return assets_->tokenizer->decode_ids(f);
}

std::vector<runtime::WordTimestamp> ParakeetDecoderRuntime::build_word_timestamps(
    const std::vector<int32_t>& ids,
    const std::vector<int32_t>& frame_indices,
    const std::vector<int32_t>& durs,
    int64_t audio_end_frame) const {
    std::vector<runtime::WordTimestamp> out;
    const int64_t spf = assets_->config.frontend.hop_length * assets_->config.encoder.subsampling_factor;
    const size_t count = std::min({ids.size(), frame_indices.size(), durs.size()});
    constexpr const char* kSentencePieceSpace = "\xE2\x96\x81";

    std::string current_word;
    int64_t current_start_frame = 0;
    int64_t current_natural_end_frame = 0;

    auto flush_word = [&](int64_t boundary_frame) {
        if (current_word.empty()) return;
        const int64_t end_frame = std::clamp(
            boundary_frame,
            current_start_frame,
            audio_end_frame);
        runtime::WordTimestamp ts;
        ts.span.start_sample = current_start_frame * spf;
        ts.span.end_sample = end_frame * spf;
        ts.word = std::move(current_word);
        ts.confidence = 0.f;
        out.push_back(std::move(ts));
        current_word.clear();
    };

    for (size_t i = 0; i < count; ++i) {
        const int32_t tid = ids[i];
        if (tid == static_cast<int32_t>(assets_->config.blank_token_id) || tid == static_cast<int32_t>(assets_->config.pad_token_id)) continue;
        if (tid >= 0 && tid < static_cast<int32_t>(assets_->special_token_ids.size()) && assets_->special_token_ids[static_cast<size_t>(tid)]) continue;
        if (tid < 0 || tid >= static_cast<int32_t>(assets_->tokenizer->id_to_token().size())) continue;

        std::string piece = assets_->tokenizer->id_to_token()[static_cast<size_t>(tid)];
        const bool starts_word = piece.rfind(kSentencePieceSpace, 0) == 0;
        if (starts_word) {
            piece.erase(0, 3);
        }
        if (piece.empty()) continue;

        const int64_t token_start = std::clamp<int64_t>(
            frame_indices[i],
            0,
            audio_end_frame);
        const int64_t token_end = std::clamp<int64_t>(
            frame_indices[i] + std::max<int32_t>(durs[i], 1),
            token_start,
            audio_end_frame);

        if (starts_word && !current_word.empty()) {
            // A word ends at the emission boundary of the next word. This
            // keeps adjacent spans non-overlapping even when a token predicts
            // a longer duration than the next observed boundary.
            flush_word(token_start);
        }
        if (current_word.empty()) {
            current_start_frame = token_start;
            current_natural_end_frame = token_end;
        } else {
            current_natural_end_frame = std::max(current_natural_end_frame, token_end);
        }
        current_word += piece;
    }
    flush_word(std::max(current_natural_end_frame, current_start_frame));
    return out;
}

void ParakeetDecoderRuntime::reset_state() {
    const auto& cfg = assets_->config;
    hidden_scratch_.assign(static_cast<size_t>(cfg.decoder_layers * cfg.decoder_hidden_size), 0.f);
    cell_scratch_.assign(static_cast<size_t>(cfg.decoder_layers * cfg.decoder_hidden_size), 0.f);
    decoder_cache_scratch_.assign(static_cast<size_t>(cfg.decoder_hidden_size), 0.f);
    pending_input_token_ = static_cast<int32_t>(cfg.blank_token_id);
    predictor_cache_valid_ = false;
    transcribe_h_.assign(static_cast<size_t>(cfg.decoder_layers),
                         std::vector<float>(static_cast<size_t>(cfg.decoder_hidden_size), 0.f));
    transcribe_c_ = transcribe_h_;
    transcribe_last_token_ = -1;
    transcribe_prev_ctc_label_ = -1;
    state_initialized_ = true;
}

ParakeetDecodedText ParakeetDecoderRuntime::decode_incremental(
    const ParakeetEncodedAudio& enc,
    const ParakeetDecodeOptions& opts,
    int64_t frame_offset) {
    if (enc.valid_frames <= 0 || enc.hidden_size != assets_->config.encoder.hidden_size)
        throw std::runtime_error("decoder requires encoded frames");
    if (!state_initialized_) {
        throw std::runtime_error("incremental decoder state must be reset before decoding");
    }
    if (frame_offset < 0) {
        throw std::runtime_error("incremental decoder frame offset must be non-negative");
    }
    if (assets_->transcribe_layout()) {
        return decode_incremental_transcribe(enc, opts, frame_offset);
    }
    const auto t0 = Clock::now(); ensure_step_graph();
    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    const auto& cfg = assets_->config;
    const int64_t max_tok = opts.max_tokens > 0
        ? opts.max_tokens
        : (enc.valid_frames * cfg.max_symbols_per_step);

    ParakeetDecodedText out;
    out.token_ids.reserve(static_cast<size_t>(std::min(max_tok, int64_t{4096})));
    out.token_frame_indices.reserve(out.token_ids.capacity());
    out.durations.reserve(out.token_ids.capacity());

    const int32_t blank = static_cast<int32_t>(cfg.blank_token_id);

    // TDT decode loop with duration-based frame skipping
    int64_t fi = 0;
    int64_t last_label_frame = -1;
    int64_t labels_at_current_frame = 0;
    int64_t reported_frame = 0;
    while (fi < enc.valid_frames && static_cast<int64_t>(out.token_ids.size()) < max_tok) {
        if (opts.progress && fi >= reported_frame + kProgressFrames) {
            reported_frame = fi;
            opts.progress(fi, enc.valid_frames);  // cancellation point
        }
        const float* f = enc.values.data() + static_cast<std::ptrdiff_t>(fi * enc.hidden_size);
        int32_t dur_id = 0;
        const int32_t tok = run_step(
            pending_input_token_,
            f,
            predictor_cache_valid_,
            &dur_id);
        predictor_cache_valid_ = true;
        pending_input_token_ = tok;

        int32_t duration = cfg.durations.at(static_cast<size_t>(dur_id));
        if (tok == blank) {
            // A zero-duration blank must still make progress.
            fi += duration == 0 ? 1 : duration;
            continue;
        }

        out.token_ids.push_back(tok);
        out.token_frame_indices.push_back(static_cast<int32_t>(frame_offset + fi));
        out.durations.push_back(duration);

        if (fi == last_label_frame) {
            ++labels_at_current_frame;
        } else {
            last_label_frame = fi;
            labels_at_current_frame = 1;
        }

        fi += duration;
        if (labels_at_current_frame >= cfg.max_symbols_per_step && fi == last_label_frame) {
            ++fi;
        }
    }
    const bool truncated = fi < enc.valid_frames;
    if (opts.progress) {
        opts.progress(std::min(fi, enc.valid_frames), enc.valid_frames);
    }
    out = format_tokens(
        std::move(out.token_ids),
        std::move(out.token_frame_indices),
        std::move(out.durations),
        opts,
        frame_offset + enc.valid_frames);
    out.truncated = truncated;
    debug::timing_log_scalar("parakeet.decoder_ms", engine::debug::elapsed_ms(t0, Clock::now()));
    return out;
}

// ---------------------------------------------------------------------------
// TranscribeGguf greedy decoding (arch/parakeet/decoder.cpp), with the arch's
// hard error on the iteration cap turned into result.truncated.
// ---------------------------------------------------------------------------

std::vector<float> ParakeetDecoderRuntime::project_frames(
    const ParakeetEncodedAudio & enc,
    const engine::modules::LinearWeights & projection,
    int64_t out_features,
    const char * label) const {
    // arch decoder.cpp precompute_enc_proj_ggml (and the CTC head): one
    // [T, d] x [d, out] GEMM per utterance, built and freed per call.
    const int64_t T = enc.valid_frames;
    const int64_t d = enc.hidden_size;
    OwnedGraph g;
    ggml_context * gctx = init_small_context(g, 8);
    engine::core::ModuleBuildContext ctx{gctx, "parakeet.transcribe.project", execution_context_->backend_type()};
    auto input = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({T, d}));
    ggml_set_input(input.tensor);
    ggml_tensor * out = ggml_add(gctx, ggml_mul_mat(gctx, projection.weight.tensor, input.tensor), projection.bias->tensor);
    finalize_graph(g, *execution_context_, {out}, label);
    engine::core::write_tensor_f32(input, enc.values.data(), static_cast<size_t>(T * d));
    if (engine::core::compute_graph(*execution_context_, g.graph, g.plan, label) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(label) + " compute failed");
    }
    std::vector<float> values;
    engine::core::read_tensor_f32_into(out, values);
    if (static_cast<int64_t>(values.size()) != T * out_features) {
        throw std::runtime_error(std::string(label) + " produced an unexpected shape");
    }
    return values;
}

ParakeetDecodedText ParakeetDecoderRuntime::decode_incremental_transcribe(
    const ParakeetEncodedAudio & enc,
    const ParakeetDecodeOptions & opts,
    int64_t frame_offset) {
    const auto t0 = Clock::now();
    ensure_transcribe_graphs();
    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    ParakeetDecodedText out;
    if (assets_->config.head_kind == ParakeetHeadKind::Ctc) {
        decode_ctc_transcribe(enc, opts, frame_offset, out);
    } else {
        decode_transducer_transcribe(enc, opts, frame_offset, out);
    }
    const bool truncated = out.truncated;
    out = format_tokens(
        std::move(out.token_ids),
        std::move(out.token_frame_indices),
        std::move(out.durations),
        opts,
        frame_offset + enc.valid_frames,
        std::move(out.token_probabilities));
    out.truncated = truncated;
    debug::timing_log_scalar("parakeet.decoder_ms", engine::debug::elapsed_ms(t0, Clock::now()));
    return out;
}

void ParakeetDecoderRuntime::decode_transducer_transcribe(
    const ParakeetEncodedAudio & enc,
    const ParakeetDecodeOptions & opts,
    int64_t frame_offset,
    ParakeetDecodedText & out) {
    const auto & cfg = assets_->config;
    auto & graphs = *transcribe_graphs_;
    const bool tdt = cfg.head_kind == ParakeetHeadKind::Tdt;
    const int64_t T = enc.valid_frames;
    const int64_t H = cfg.decoder_hidden_size;
    const int64_t J = cfg.joint_hidden_size;
    const int64_t L = cfg.decoder_layers;
    const int64_t n_token_cls = cfg.vocab_size;
    const int64_t n_dur = static_cast<int64_t>(cfg.durations.size());
    const int64_t joint_n = n_token_cls + n_dur;
    const int32_t blank = static_cast<int32_t>(cfg.blank_token_id);
    const int64_t max_symbols = cfg.max_symbols_per_step;
    const auto & embed = weights_->decoder.embedding_host;

    // TDT decodes serially (arch default W = 1); RNN-T through the 16-frame
    // joint window (identical decisions; see build_joint_graph_batch).
    auto & joint = (!tdt && graphs.window.width > 1) ? graphs.window : graphs.serial;
    const int64_t W = joint.width;

    std::vector<float> enc_proj = project_frames(enc, weights_->decoder.joint_enc, J, "Parakeet encoder projection");
    if (W > 1) {
        // Zero-pad W-1 frames so a window near the end stays in bounds; the
        // padded columns are never consumed.
        enc_proj.resize(static_cast<size_t>((T + W - 1) * J), 0.0f);
    }

    if (static_cast<int64_t>(transcribe_h_.size()) != L) {
        transcribe_h_.assign(static_cast<size_t>(L), std::vector<float>(static_cast<size_t>(H), 0.f));
        transcribe_c_ = transcribe_h_;
        transcribe_last_token_ = -1;
    }
    std::vector<std::vector<float>> next_h = transcribe_h_;
    std::vector<std::vector<float>> next_c = transcribe_c_;
    std::vector<float> x(static_cast<size_t>(H), 0.0f);
    std::vector<float> logits_w;
    std::vector<float> scratch_probs;

    auto run_predictor = [&]() {
        if (transcribe_last_token_ < 0) {
            std::fill(x.begin(), x.end(), 0.0f);
        } else {
            std::copy_n(embed.begin() + static_cast<std::ptrdiff_t>(transcribe_last_token_) * H, H, x.begin());
        }
        engine::core::write_tensor_f32(graphs.pred_x, x);
        for (int64_t l = 0; l < L; ++l) {
            engine::core::write_tensor_f32(graphs.pred_h_in[static_cast<size_t>(l)], transcribe_h_[static_cast<size_t>(l)]);
            engine::core::write_tensor_f32(graphs.pred_c_in[static_cast<size_t>(l)], transcribe_c_[static_cast<size_t>(l)]);
        }
        if (engine::core::compute_graph(*execution_context_, graphs.pred.graph, graphs.pred.plan, "Parakeet predictor") !=
            GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Parakeet predictor compute failed");
        }
        for (int64_t l = 0; l < L; ++l) {
            engine::core::read_tensor_f32_into(graphs.pred_h_out[static_cast<size_t>(l)], next_h[static_cast<size_t>(l)]);
            engine::core::read_tensor_f32_into(graphs.pred_c_out[static_cast<size_t>(l)], next_c[static_cast<size_t>(l)]);
        }
    };

    int64_t step = 0;
    int64_t new_symbols = 0;
    const int64_t max_iters = 16 * T + 1024;
    int64_t iter = 0;
    bool predictor_dirty = true;
    int64_t win_base = 0;
    bool win_valid = false;
    int64_t reported = 0;
    bool token_budget_hit = false;

    while (step < T && iter < max_iters) {
        ++iter;
        if (opts.progress && step >= reported + kProgressFrames) {
            reported = step;
            opts.progress(step, T);  // cancellation point
        }
        if (opts.max_tokens > 0 && static_cast<int64_t>(out.token_ids.size()) >= opts.max_tokens) {
            token_budget_hit = true;
            break;
        }

        if (predictor_dirty) {
            run_predictor();
            predictor_dirty = false;
        }
        const std::vector<float> & decoder_out = next_h.back();

        if (!win_valid || step < win_base || step >= win_base + W) {
            engine::core::write_tensor_f32(joint.pred_in, decoder_out);
            engine::core::write_tensor_f32(joint.enc_in, enc_proj.data() + static_cast<size_t>(step * J),
                static_cast<size_t>(W * J));
            if (engine::core::compute_graph(*execution_context_, joint.graph.graph, joint.graph.plan, "Parakeet joint") !=
                GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Parakeet joint compute failed");
            }
            engine::core::read_tensor_f32_into(joint.logits, logits_w);
            win_base = step;
            win_valid = true;
        }
        const float * frame_logits = logits_w.data() + static_cast<size_t>((step - win_base) * joint_n);
        const int32_t pred_token = argmax_range(frame_logits, n_token_cls);

        if (tdt) {
            // arch decoder.cpp decode_tdt_greedy
            const int32_t decision = argmax_range(frame_logits + n_token_cls, n_dur);
            const int32_t duration = cfg.durations[static_cast<size_t>(decision)];
            const bool is_blank = pred_token == blank;
            if (!is_blank) {
                out.token_ids.push_back(pred_token);
                out.token_frame_indices.push_back(static_cast<int32_t>(frame_offset + step));
                out.durations.push_back(duration);
                out.token_probabilities.push_back(token_confidence(frame_logits, n_token_cls, scratch_probs));
                transcribe_last_token_ = pred_token;
                std::swap(transcribe_h_, next_h);  // commit
                std::swap(transcribe_c_, next_c);
                predictor_dirty = true;
                win_valid = false;
            }
            step += duration;
            new_symbols += 1;
            if (duration != 0) {
                new_symbols = 0;
            } else if (max_symbols > 0 && new_symbols >= max_symbols) {
                step += 1;
                new_symbols = 0;
            } else if (is_blank && max_symbols > 0) {
                // A zero-duration blank repeats identically until max_symbols
                // forces the advance; fast-forward like the arch does.
                const int64_t skip = max_symbols - new_symbols;
                if (skip > 0 && iter + skip < max_iters) {
                    iter += skip;
                    step += 1;
                    new_symbols = 0;
                }
            }
        } else {
            // arch decoder.cpp decode_rnnt_greedy
            if (pred_token == blank) {
                step += 1;
                new_symbols = 0;
            } else {
                out.token_ids.push_back(pred_token);
                out.token_frame_indices.push_back(static_cast<int32_t>(frame_offset + step));
                out.durations.push_back(1);
                out.token_probabilities.push_back(token_confidence(frame_logits, n_token_cls, scratch_probs));
                transcribe_last_token_ = pred_token;
                std::swap(transcribe_h_, next_h);
                std::swap(transcribe_c_, next_c);
                predictor_dirty = true;
                win_valid = false;
                new_symbols += 1;
                if (max_symbols > 0 && new_symbols >= max_symbols) {
                    step += 1;
                    new_symbols = 0;
                }
            }
        }
    }
    // The arch fails the run at the iteration cap; the engine keeps what it
    // decoded and reports it truncated.
    out.truncated = step < T && (token_budget_hit || iter >= max_iters);
    if (opts.progress) {
        opts.progress(std::min(step, T), T);
    }
}

void ParakeetDecoderRuntime::decode_ctc_transcribe(
    const ParakeetEncodedAudio & enc,
    const ParakeetDecodeOptions & opts,
    int64_t frame_offset,
    ParakeetDecodedText & out) {
    // arch decoder.cpp decode_ctc_greedy: logits = W @ enc + b, per-frame
    // log_softmax (double sum), argmax, collapse repeats then drop blanks.
    const auto & cfg = assets_->config;
    if (!weights_->decoder.ctc_head.has_value()) {
        throw std::runtime_error("Parakeet CTC decode requires the CTC head");
    }
    const int64_t T = enc.valid_frames;
    const int64_t n = cfg.vocab_size;
    const int32_t blank = static_cast<int32_t>(cfg.blank_token_id);
    auto logits = project_frames(enc, *weights_->decoder.ctc_head, n, "Parakeet CTC head");
    if (opts.progress) {
        opts.progress(0, T);  // cancellation point after the projection
    }
    int64_t reported = 0;
    for (int64_t t = 0; t < T; ++t) {
        if (opts.progress && t >= reported + kProgressFrames) {
            reported = t;
            opts.progress(t, T);
        }
        float * row = logits.data() + static_cast<size_t>(t * n);
        float max_v = row[0];
        for (int64_t c = 1; c < n; ++c) {
            max_v = std::max(max_v, row[c]);
        }
        double sum = 0.0;
        for (int64_t c = 0; c < n; ++c) {
            sum += std::exp(static_cast<double>(row[c] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int64_t c = 0; c < n; ++c) {
            row[c] -= log_sum;
        }
        const int32_t label = argmax_range(row, n);
        if (label == transcribe_prev_ctc_label_) {
            continue;
        }
        transcribe_prev_ctc_label_ = label;
        if (label == blank) {
            continue;
        }
        if (opts.max_tokens > 0 && static_cast<int64_t>(out.token_ids.size()) >= opts.max_tokens) {
            out.truncated = true;
            break;
        }
        out.token_ids.push_back(label);
        out.token_frame_indices.push_back(static_cast<int32_t>(frame_offset + t));
        out.durations.push_back(1);
        out.token_probabilities.push_back(std::exp(row[label]));
    }
    if (opts.progress) {
        opts.progress(T, T);
    }
}

void ParakeetDecoderRuntime::format_transcribe(ParakeetDecodedText & out, bool keep_language_tags) const {
    // arch/parakeet/model.cpp build_result_from_raw_tokens.
    const auto & assets = *assets_;
    const int64_t spf = assets.config.frontend.hop_length * assets.config.encoder.subsampling_factor;
    static constexpr char kMarker[] = "\xE2\x96\x81";
    const size_t count = std::min(out.token_ids.size(), out.token_frame_indices.size());

    std::vector<int32_t> kept_ids;
    kept_ids.reserve(count);
    std::vector<size_t> kept_index;
    kept_index.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const int32_t id = out.token_ids[i];
        const bool special = id >= 0 && static_cast<size_t>(id) < assets.special_token_ids.size() &&
            assets.special_token_ids[static_cast<size_t>(id)] != 0;
        if (special && !keep_language_tags) {
            continue;
        }
        kept_ids.push_back(id);
        kept_index.push_back(i);
    }

    out.token_timestamps.clear();
    out.word_timestamps.clear();
    out.segment_span.reset();
    out.raw_text.reset();
    out.text.clear();
    if (kept_ids.empty()) {
        return;
    }

    std::vector<int32_t> word_ids;
    auto close_word = [&](size_t first_token, size_t last_token) {
        runtime::WordTimestamp word;
        word.span.start_sample = out.token_timestamps[first_token].span.start_sample;
        word.span.end_sample = out.token_timestamps[last_token].span.end_sample;
        std::string text = decode_transcribe_pieces(assets, word_ids.data(), word_ids.size());
        if (!text.empty() && text.front() == ' ') {
            text.erase(text.begin());
        }
        word.word = std::move(text);
        out.word_timestamps.push_back(std::move(word));
        word_ids.clear();
    };

    size_t word_first = 0;
    for (size_t k = 0; k < kept_ids.size(); ++k) {
        const size_t i = kept_index[k];
        const int32_t id = kept_ids[k];
        const std::string & piece = (id >= 0 && static_cast<size_t>(id) < assets.vocab_pieces.size())
            ? assets.vocab_pieces[static_cast<size_t>(id)] : std::string();
        const bool starts_word = k == 0 || (piece.size() >= 3 && std::memcmp(piece.data(), kMarker, 3) == 0);
        if (starts_word && k > 0) {
            close_word(word_first, k - 1);
        }
        if (starts_word) {
            word_first = k;
        }
        runtime::TokenTimestamp row;
        const int64_t frame = out.token_frame_indices[i];
        const int64_t duration = i < out.durations.size() ? out.durations[i] : 1;
        row.span.start_sample = frame * spf;
        // duration 0 is a zero-width "point in time" token, as in the arch.
        row.span.end_sample = (frame + duration) * spf;
        row.id = id;
        row.text = decode_transcribe_pieces(assets, &id, 1);
        row.probability = i < out.token_probabilities.size() ? out.token_probabilities[i] : 0.0f;
        row.word_index = static_cast<int32_t>(out.word_timestamps.size());
        out.token_timestamps.push_back(std::move(row));
        word_ids.push_back(id);
    }
    close_word(word_first, kept_ids.size() - 1);

    // model.cpp normalize_transcript_whitespace: collapse space runs, trim.
    const std::string full = decode_transcribe_pieces(assets, kept_ids.data(), kept_ids.size());
    std::string norm;
    norm.reserve(full.size());
    bool prev_space = false;
    for (const char ch : full) {
        const bool is_space = ch == ' ';
        if (is_space && prev_space) {
            continue;
        }
        norm.push_back(ch);
        prev_space = is_space;
    }
    const size_t first = norm.find_first_not_of(' ');
    const size_t last = norm.find_last_not_of(' ');
    out.text = first == std::string::npos ? std::string() : norm.substr(first, last - first + 1);
    out.raw_text = decode_transcribe_pieces(assets, out.token_ids.data(), count);
    out.segment_span = runtime::TimeSpan{out.token_timestamps.front().span.start_sample,
                                         out.token_timestamps.back().span.end_sample};
}

ParakeetDecodedText ParakeetDecoderRuntime::format_tokens(
    std::vector<int32_t> token_ids,
    std::vector<int32_t> token_frame_indices,
    std::vector<int32_t> durations,
    const ParakeetDecodeOptions& opts,
    int64_t audio_end_frame,
    std::vector<float> token_probabilities) const {
    ParakeetDecodedText out;
    out.token_ids = std::move(token_ids);
    out.token_frame_indices = std::move(token_frame_indices);
    out.durations = std::move(durations);
    out.token_probabilities = std::move(token_probabilities);
    if (assets_->transcribe_layout()) {
        format_transcribe(out, opts.keep_language_tags);
        return out;
    }
    out.text = decode_text(out.token_ids, opts.keep_language_tags);
    out.word_timestamps = build_word_timestamps(
        out.token_ids,
        out.token_frame_indices,
        out.durations,
        audio_end_frame);
    return out;
}

ParakeetDecodedText ParakeetDecoderRuntime::decode(
    const ParakeetEncodedAudio& enc,
    const ParakeetDecodeOptions& opts) {
    reset_state();
    return decode_incremental(enc, opts);
}

}  // namespace engine::community_models::parakeet_tdt
