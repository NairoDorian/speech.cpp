// Chunked diarization scheduler for Sortformer. Exact host ports of NeMo's
// SortformerModules (sortformer_modules.py): streaming_update (sync branch),
// _get_silence_profile, and the exactness-critical _compress_spkcache stack
// (_get_log_pred_scores -> _disable_low_scores -> scores_boost_latest ->
// _boost_topk_scores x2 -> silence pad -> _get_topk_indices -> gather).
//
// Carried over from transcribe.cpp's src/arch/sortformer/stream.cpp, where the
// compression internals were verified bit-exact against NeMo through a neutral
// arbiter (docs/porting/families/sortformer.md, residual #1). Batch size is
// always 1 here, so every per-batch loop in the reference collapses and
// spk_perm is always None (inference).

#include "engine/models/sortformer_diar/streaming.h"

#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace engine::models::sortformer_diar {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

const char * env_get(const char * name) {
    const char * value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

bool env_i64(const char * name, int64_t & out) {
    const char * value = env_get(name);
    if (value == nullptr) {
        return false;
    }
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value) {
        throw std::runtime_error(std::string("Sortformer diar: ") + name + " is not an integer: " + value);
    }
    out = static_cast<int64_t>(parsed);
    return true;
}

// Mirrors scripts/diar/run_reference_sortformer_nemo.py PRESETS in
// transcribe.cpp plus the "small" diagnostic; lc stays at the model default.
const std::vector<SortformerStreamPreset> & presets() {
    static const std::vector<SortformerStreamPreset> table = {
        {"very_high_latency", 340, 40, 40, 300, 188},
        {"high_latency", 124, 1, 124, 124, 188},
        {"low_latency", 6, 7, 188, 144, 188},
        {"small", 20, 1, 10, 20, 24},
    };
    return table;
}

void apply_preset(SortformerStreamParams & params, const SortformerStreamPreset & preset) {
    params.chunk_len = preset.chunk_len;
    params.chunk_right_context = preset.chunk_right_context;
    params.fifo_len = preset.fifo_len;
    params.spkcache_update_period = preset.spkcache_update_period;
    params.spkcache_len = preset.spkcache_len;
}

std::optional<std::string> find_option(
    const std::unordered_map<std::string, std::string> & options,
    const char * key) {
    const std::string prefixed = std::string("sortformer_diar.") + key;
    return runtime::find_option(options, {std::string_view(prefixed), std::string_view(key)});
}

bool apply_i64_option(
    const std::unordered_map<std::string, std::string> & options,
    const char * key,
    int64_t & out) {
    const std::string prefixed = std::string("sortformer_diar.") + key;
    const auto value = runtime::parse_i64_option(options, {std::string_view(prefixed), std::string_view(key)});
    if (!value.has_value()) {
        return false;
    }
    out = *value;
    return true;
}

void validate_params(const SortformerStreamParams & params, int64_t n_spk) {
    if (params.chunk_len <= 0) {
        throw std::runtime_error("Sortformer diar stream_chunk_len must be positive");
    }
    if (params.chunk_left_context < 0 || params.chunk_right_context < 0) {
        throw std::runtime_error("Sortformer diar stream_left_context / stream_right_context must be non-negative");
    }
    if (params.fifo_len < 0) {
        throw std::runtime_error("Sortformer diar stream_fifo_len must be non-negative");
    }
    if (params.spkcache_update_period <= 0) {
        throw std::runtime_error("Sortformer diar stream_update_period must be positive");
    }
    if (n_spk <= 0) {
        throw std::runtime_error("Sortformer diar requires a positive speaker count");
    }
    // _compress_spkcache reserves spkcache_sil_frames_per_spk silence rows per
    // speaker and needs at least one real row per speaker on top of that.
    const int64_t per_spk = params.spkcache_len / n_spk - params.scoring.spkcache_sil_frames_per_spk;
    if (params.spkcache_len <= 0 || per_spk < 1) {
        throw std::runtime_error(
            "Sortformer diar stream_spkcache_len must leave at least one frame per speaker after the silence reserve");
    }
}

// ---- _get_silence_profile (sortformer_modules.py:636-667) ----

void get_silence_profile(
    SortformerStreamState & state,
    float sil_threshold,
    const float * pop_embs,
    const float * pop_preds,
    int64_t n,
    int64_t n_spk,
    int64_t emb_dim) {
    if (n <= 0) {
        return;
    }
    // is_sil[i] = (sum_s preds[i,s]) < sil_threshold ; sil_count = sum is_sil
    std::vector<uint8_t> is_sil(static_cast<size_t>(n));
    int64_t sil_count = 0;
    for (int64_t i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int64_t c = 0; c < n_spk; ++c) {
            s += pop_preds[static_cast<size_t>(i * n_spk + c)];
        }
        is_sil[static_cast<size_t>(i)] = (s < sil_threshold) ? 1 : 0;
        sil_count += is_sil[static_cast<size_t>(i)];
    }
    if (sil_count == 0) {
        return;  // has_new_sil false -> unchanged
    }
    const int64_t old_n = state.n_sil_frames;
    const int64_t new_n = old_n + sil_count;
    const float denom = static_cast<float>(std::max<int64_t>(new_n, 1));
    for (int64_t d = 0; d < emb_dim; ++d) {
        float sil_sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            if (is_sil[static_cast<size_t>(i)]) {
                sil_sum += pop_embs[static_cast<size_t>(i * emb_dim + d)];
            }
        }
        const float old_sum = state.mean_sil_emb[static_cast<size_t>(d)] * static_cast<float>(old_n);
        state.mean_sil_emb[static_cast<size_t>(d)] = (old_sum + sil_sum) / denom;
    }
    state.n_sil_frames = new_n;
}

// _get_log_pred_scores (sortformer_modules.py:669-686). scores[i,s] =
// log(clamp(p,thr)) - log(clamp(1-p,thr)) + sum_s log(clamp(1-p,thr)) - log(0.5).
std::vector<float> get_log_pred_scores(const float * preds, int64_t n, int64_t n_spk, float thr) {
    std::vector<float> scores(static_cast<size_t>(n * n_spk));
    const float log_half = std::log(0.5f);
    for (int64_t i = 0; i < n; ++i) {
        float l1sum = 0.0f;
        for (int64_t s = 0; s < n_spk; ++s) {
            const float p = preds[static_cast<size_t>(i * n_spk + s)];
            l1sum += std::log(std::max(1.0f - p, thr));
        }
        for (int64_t s = 0; s < n_spk; ++s) {
            const float p = preds[static_cast<size_t>(i * n_spk + s)];
            const float lp = std::log(std::max(p, thr));
            const float l1 = std::log(std::max(1.0f - p, thr));
            scores[static_cast<size_t>(i * n_spk + s)] = lp - l1 + l1sum - log_half;
        }
    }
    return scores;
}

// _disable_low_scores (sortformer_modules.py:782-808).
void disable_low_scores(const float * preds, std::vector<float> & scores, int64_t n, int64_t n_spk, int64_t min_pos) {
    // Non-speech (preds <= 0.5) -> -inf.
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t s = 0; s < n_spk; ++s) {
            const size_t idx = static_cast<size_t>(i * n_spk + s);
            if (!(preds[idx] > 0.5f)) {
                scores[idx] = kNegInf;
            }
        }
    }
    // Per speaker: if it has >= min_pos positive-scored frames, disable its
    // non-positive (but speech) frames.
    for (int64_t s = 0; s < n_spk; ++s) {
        int64_t pos_count = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (scores[static_cast<size_t>(i * n_spk + s)] > 0.0f) {
                ++pos_count;
            }
        }
        if (pos_count >= min_pos) {
            for (int64_t i = 0; i < n; ++i) {
                const size_t idx = static_cast<size_t>(i * n_spk + s);
                const bool is_speech = preds[idx] > 0.5f;
                const bool is_pos = scores[idx] > 0.0f;
                if (is_speech && !is_pos) {
                    scores[idx] = kNegInf;
                }
            }
        }
    }
}

// _boost_topk_scores (sortformer_modules.py:611-634). For each speaker column,
// the k highest-scored frames get scores -= scale*log(0.5) (log(0.5)<0 =>
// boosts). Tie-break matches ATen CPU topk: value desc, frame index asc.
void boost_topk_scores(std::vector<float> & scores, int64_t n, int64_t n_spk, int64_t k, float scale) {
    if (k <= 0) {
        return;
    }
    const float delta = scale * std::log(0.5f);  // negative
    std::vector<int64_t> order(static_cast<size_t>(n));
    for (int64_t s = 0; s < n_spk; ++s) {
        std::iota(order.begin(), order.end(), int64_t{0});
        const int64_t kk = std::min(k, n);
        std::partial_sort(order.begin(), order.begin() + kk, order.end(), [&](int64_t a, int64_t b) {
            const float va = scores[static_cast<size_t>(a * n_spk + s)];
            const float vb = scores[static_cast<size_t>(b * n_spk + s)];
            if (va != vb) {
                return va > vb;  // descending value
            }
            return a < b;  // tie -> lower index first
        });
        for (int64_t j = 0; j < kk; ++j) {
            scores[static_cast<size_t>(order[static_cast<size_t>(j)] * n_spk + s)] -= delta;
        }
    }
}

}  // namespace

const SortformerStreamPreset * find_sortformer_stream_preset(std::string_view name) noexcept {
    for (const auto & preset : presets()) {
        if (name == preset.name) {
            return &preset;
        }
    }
    return nullptr;
}

const std::vector<SortformerStreamPreset> & sortformer_stream_presets() {
    return presets();
}

SortformerRunPlan resolve_sortformer_run_plan(
    const SortformerModelConfig & config,
    const std::unordered_map<std::string, std::string> & options) {
    SortformerRunPlan plan;
    plan.params.scoring = config.scoring;
    if (config.streaming.present) {
        plan.params.chunk_len = config.streaming.chunk_len;
        plan.params.chunk_left_context = config.streaming.chunk_left_context;
        plan.params.chunk_right_context = config.streaming.chunk_right_context;
        plan.params.fifo_len = config.streaming.fifo_len;
        plan.params.spkcache_len = config.streaming.spkcache_len;
        plan.params.spkcache_update_period = config.streaming.spkcache_update_period;
    }
    plan.mode = config.streaming.present ? SortformerRunMode::Chunked : SortformerRunMode::WholeWindow;
    plan.source = "default";

    if (const auto preset = find_option(options, "stream_preset")) {
        if (*preset == "default") {
            // keep the shipped decision
        } else if (*preset == "offline") {
            plan.mode = SortformerRunMode::WholeWindow;
            plan.source = "offline";
        } else if (const auto * named = find_sortformer_stream_preset(*preset)) {
            apply_preset(plan.params, *named);
            plan.mode = SortformerRunMode::Chunked;
            plan.source = *preset;
        } else {
            std::string menu = "default, offline";
            for (const auto & entry : presets()) {
                menu += ", ";
                menu += entry.name;
            }
            throw std::runtime_error("Sortformer diar stream_preset '" + *preset + "' is not one of: " + menu);
        }
    }

    bool any_field = false;
    any_field |= apply_i64_option(options, "stream_chunk_len", plan.params.chunk_len);
    any_field |= apply_i64_option(options, "stream_left_context", plan.params.chunk_left_context);
    any_field |= apply_i64_option(options, "stream_right_context", plan.params.chunk_right_context);
    any_field |= apply_i64_option(options, "stream_fifo_len", plan.params.fifo_len);
    any_field |= apply_i64_option(options, "stream_spkcache_len", plan.params.spkcache_len);
    any_field |= apply_i64_option(options, "stream_update_period", plan.params.spkcache_update_period);
    if (any_field) {
        if (plan.mode == SortformerRunMode::WholeWindow && plan.source == "offline") {
            throw std::runtime_error("Sortformer diar stream_* geometry options conflict with stream_preset=offline");
        }
        plan.mode = SortformerRunMode::Chunked;
        plan.source += "+fields";
    }

    // Validation hooks (the DER harness and validate.py drive these); they
    // outrank the options so a measurement run cannot be silently redirected.
    if (const char * name = env_get("TRANSCRIBE_SORTFORMER_STREAM_PRESET")) {
        const auto * named = find_sortformer_stream_preset(name);
        if (named == nullptr) {
            throw std::runtime_error(std::string("Sortformer diar: TRANSCRIBE_SORTFORMER_STREAM_PRESET names no preset: ") + name);
        }
        apply_preset(plan.params, *named);
        plan.mode = SortformerRunMode::Chunked;
        plan.source += std::string("+env:") + name;
    }
    bool any_env = false;
    any_env |= env_i64("TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN", plan.params.chunk_len);
    any_env |= env_i64("TRANSCRIBE_SORTFORMER_STREAM_FIFO_LEN", plan.params.fifo_len);
    any_env |= env_i64("TRANSCRIBE_SORTFORMER_STREAM_SPKCACHE_LEN", plan.params.spkcache_len);
    any_env |= env_i64("TRANSCRIBE_SORTFORMER_STREAM_UPDATE_PERIOD", plan.params.spkcache_update_period);
    any_env |= env_i64("TRANSCRIBE_SORTFORMER_STREAM_RC", plan.params.chunk_right_context);
    any_env |= env_i64("TRANSCRIBE_SORTFORMER_STREAM_LC", plan.params.chunk_left_context);
    if (any_env) {
        plan.mode = SortformerRunMode::Chunked;
        plan.source += "+env-fields";
    }

    if (plan.mode == SortformerRunMode::Chunked) {
        validate_params(plan.params, config.modules.num_speakers);
    }
    return plan;
}

void SortformerStreamState::reset(int64_t emb_dim) {
    spkcache.clear();
    spkcache_preds.clear();
    spkcache_n = 0;
    spkcache_preds_init = false;
    fifo.clear();
    fifo_preds.clear();
    fifo_n = 0;
    mean_sil_emb.assign(static_cast<size_t>(emb_dim), 0.0f);
    n_sil_frames = 0;
    total_preds.clear();
    total_n = 0;
    compress_count = 0;
}

// ---- _compress_spkcache (sortformer_modules.py:838-896) ----

void sortformer_compress_spkcache(
    SortformerStreamState & state,
    const SortformerStreamParams & params,
    int64_t n_spk,
    int64_t emb_dim) {
    const int64_t N = state.spkcache_n;      // n_frames (> spkcache_len)
    const int64_t L = params.spkcache_len;   // target frames
    const int64_t sil = params.scoring.spkcache_sil_frames_per_spk;

    const int64_t per_spk = L / n_spk - sil;
    const int64_t strong = static_cast<int64_t>(std::floor(static_cast<double>(per_spk) * params.scoring.strong_boost_rate));
    const int64_t weak = static_cast<int64_t>(std::floor(static_cast<double>(per_spk) * params.scoring.weak_boost_rate));
    const int64_t min_pos = static_cast<int64_t>(std::floor(static_cast<double>(per_spk) * params.scoring.min_pos_scores_rate));

    const float * preds = state.spkcache_preds.data();  // [N, n_spk]

    // 1. log-pred scores, disable low scores.
    std::vector<float> scores = get_log_pred_scores(preds, N, n_spk, params.scoring.pred_score_threshold);
    disable_low_scores(preds, scores, N, n_spk, min_pos);

    // 2. boost latest frames (rows >= spkcache_len). += on -inf stays -inf.
    if (params.scoring.scores_boost_latest > 0.0f) {
        for (int64_t i = L; i < N; ++i) {
            for (int64_t s = 0; s < n_spk; ++s) {
                scores[static_cast<size_t>(i * n_spk + s)] += params.scoring.scores_boost_latest;
            }
        }
    }

    // 3. strong then weak top-k boosting.
    boost_topk_scores(scores, N, n_spk, strong, /*scale=*/2.0f);
    boost_topk_scores(scores, N, n_spk, weak, /*scale=*/1.0f);

    // 4. append `sil` rows of +inf per speaker; n_frames = N + sil.
    const int64_t n_frames = N + sil;
    const int64_t n_frames_no_sil = N;
    std::vector<float> scores_ext(static_cast<size_t>(n_frames * n_spk));
    std::copy(scores.begin(), scores.end(), scores_ext.begin());
    for (int64_t i = N; i < n_frames; ++i) {
        for (int64_t s = 0; s < n_spk; ++s) {
            scores_ext[static_cast<size_t>(i * n_spk + s)] = kPosInf;
        }
    }

    // 5. _get_topk_indices: flatten permute(0,2,1) -> flat[s*n_frames + i];
    // top-L largest (value desc, flat-idx asc), -inf picks -> max_index, sort
    // ascending, derive frame idx + is_disabled.
    const int64_t M = n_spk * n_frames;
    std::vector<int64_t> flat(static_cast<size_t>(M));
    std::iota(flat.begin(), flat.end(), int64_t{0});
    auto flat_val = [&](int64_t f) -> float {
        const int64_t s = f / n_frames;
        const int64_t i = f % n_frames;
        return scores_ext[static_cast<size_t>(i * n_spk + s)];
    };
    const int64_t kk = std::min<int64_t>(L, M);
    std::partial_sort(flat.begin(), flat.begin() + kk, flat.end(), [&](int64_t a, int64_t b) {
        const float va = flat_val(a);
        const float vb = flat_val(b);
        if (va != vb) {
            return va > vb;
        }
        return a < b;
    });
    std::vector<int64_t> picks(static_cast<size_t>(L));
    for (int64_t j = 0; j < L; ++j) {
        if (j < kk) {
            const int64_t f = flat[static_cast<size_t>(j)];
            picks[static_cast<size_t>(j)] = (flat_val(f) == kNegInf) ? params.max_index : f;
        } else {
            picks[static_cast<size_t>(j)] = params.max_index;  // fewer than L finite picks
        }
    }
    std::sort(picks.begin(), picks.end());

    std::vector<int64_t> frame_idx(static_cast<size_t>(L));
    std::vector<char> is_disabled(static_cast<size_t>(L));
    for (int64_t j = 0; j < L; ++j) {
        const int64_t idx = picks[static_cast<size_t>(j)];
        bool disabled = (idx == params.max_index);
        int64_t f = disabled ? 0 : (idx % n_frames);
        if (!disabled && f >= n_frames_no_sil) {
            disabled = true;  // came from a +inf silence pad row
        }
        if (disabled) {
            f = 0;
        }
        frame_idx[static_cast<size_t>(j)] = f;
        is_disabled[static_cast<size_t>(j)] = disabled ? 1 : 0;
    }

    // 6. gather embeddings + preds; disabled -> mean_sil_emb / 0.
    std::vector<float> new_emb(static_cast<size_t>(L * emb_dim));
    std::vector<float> new_preds(static_cast<size_t>(L * n_spk));
    for (int64_t j = 0; j < L; ++j) {
        const int64_t f = frame_idx[static_cast<size_t>(j)];
        if (is_disabled[static_cast<size_t>(j)]) {
            std::copy(state.mean_sil_emb.begin(), state.mean_sil_emb.end(),
                      new_emb.begin() + static_cast<std::ptrdiff_t>(j * emb_dim));
            // preds row already zero-initialized.
        } else {
            std::copy(state.spkcache.begin() + static_cast<std::ptrdiff_t>(f * emb_dim),
                      state.spkcache.begin() + static_cast<std::ptrdiff_t>((f + 1) * emb_dim),
                      new_emb.begin() + static_cast<std::ptrdiff_t>(j * emb_dim));
            std::copy(state.spkcache_preds.begin() + static_cast<std::ptrdiff_t>(f * n_spk),
                      state.spkcache_preds.begin() + static_cast<std::ptrdiff_t>((f + 1) * n_spk),
                      new_preds.begin() + static_cast<std::ptrdiff_t>(j * n_spk));
        }
    }
    ++state.compress_count;

    state.spkcache = std::move(new_emb);
    state.spkcache_preds = std::move(new_preds);
    state.spkcache_n = L;
}

// ---- streaming_update (sync branch, sortformer_modules.py:526-609) ----

void sortformer_streaming_update(
    SortformerStreamState & state,
    const SortformerStreamParams & params,
    int64_t n_spk,
    int64_t emb_dim,
    const std::vector<float> & chunk_embs,
    int64_t T_diar,
    const std::vector<float> & preds,
    int64_t lc,
    int64_t rc) {
    const int64_t S = state.spkcache_n;  // spkcache_len (local, pre-update)
    const int64_t F = state.fifo_n;      // fifo_len (local, pre-update)
    const int64_t C = T_diar - lc - rc;  // chunk_len (middle frames)
    if (C < 0) {
        throw std::runtime_error("Sortformer diar chunk window shorter than its context");
    }
    if (static_cast<int64_t>(chunk_embs.size()) != T_diar * emb_dim ||
        static_cast<int64_t>(preds.size()) != (S + F + T_diar) * n_spk) {
        throw std::runtime_error("Sortformer diar streaming update received inconsistent buffers");
    }

    // fifo_preds = preds[S : S+F]  (rebuilt from the concat preds each call).
    state.fifo_preds.assign(preds.begin() + static_cast<std::ptrdiff_t>(S * n_spk),
                            preds.begin() + static_cast<std::ptrdiff_t>((S + F) * n_spk));
    // chunk_preds = preds[S+F+lc : S+F+lc+C].
    std::vector<float> chunk_preds(preds.begin() + static_cast<std::ptrdiff_t>((S + F + lc) * n_spk),
                                   preds.begin() + static_cast<std::ptrdiff_t>((S + F + lc + C) * n_spk));

    // Append the middle chunk (embs + preds) to the FIFO.
    state.fifo.insert(state.fifo.end(), chunk_embs.begin() + static_cast<std::ptrdiff_t>(lc * emb_dim),
                      chunk_embs.begin() + static_cast<std::ptrdiff_t>((lc + C) * emb_dim));
    state.fifo_preds.insert(state.fifo_preds.end(), chunk_preds.begin(), chunk_preds.end());
    state.fifo_n = F + C;

    if (F + C > params.fifo_len) {
        int64_t pop = params.spkcache_update_period;
        pop = std::max(pop, C - params.fifo_len + F);
        pop = std::min(pop, F + C);

        std::vector<float> pop_embs(state.fifo.begin(), state.fifo.begin() + static_cast<std::ptrdiff_t>(pop * emb_dim));
        std::vector<float> pop_preds(state.fifo_preds.begin(), state.fifo_preds.begin() + static_cast<std::ptrdiff_t>(pop * n_spk));

        get_silence_profile(state, params.scoring.sil_threshold, pop_embs.data(), pop_preds.data(), pop, n_spk, emb_dim);

        // fifo = fifo[pop:].
        state.fifo.erase(state.fifo.begin(), state.fifo.begin() + static_cast<std::ptrdiff_t>(pop * emb_dim));
        state.fifo_preds.erase(state.fifo_preds.begin(), state.fifo_preds.begin() + static_cast<std::ptrdiff_t>(pop * n_spk));
        state.fifo_n = (F + C) - pop;

        // Append pop-out to speaker cache.
        state.spkcache.insert(state.spkcache.end(), pop_embs.begin(), pop_embs.end());
        state.spkcache_n = S + pop;
        if (state.spkcache_preds_init) {
            state.spkcache_preds.insert(state.spkcache_preds.end(), pop_preds.begin(), pop_preds.end());
        }

        if (state.spkcache_n > params.spkcache_len) {
            if (!state.spkcache_preds_init) {
                // First compress: seed spkcache_preds = preds[:S] ++ pop_preds.
                state.spkcache_preds.assign(preds.begin(), preds.begin() + static_cast<std::ptrdiff_t>(S * n_spk));
                state.spkcache_preds.insert(state.spkcache_preds.end(), pop_preds.begin(), pop_preds.end());
                state.spkcache_preds_init = true;
            }
            sortformer_compress_spkcache(state, params, n_spk, emb_dim);
        }
    }

    // Accumulate the chunk's predictions.
    state.total_preds.insert(state.total_preds.end(), chunk_preds.begin(), chunk_preds.end());
    state.total_n += C;
}

std::vector<SortformerChunkWindow> plan_sortformer_chunks(
    int64_t feature_frames,
    int64_t subsampling_factor,
    const SortformerStreamParams & params) {
    std::vector<SortformerChunkWindow> windows;
    if (feature_frames <= 0 || subsampling_factor <= 0 || params.chunk_len <= 0) {
        return windows;
    }
    const int64_t sub = subsampling_factor;
    int64_t stt = 0;
    int64_t end = 0;
    while (end < feature_frames) {
        const int64_t left_offset = std::min(params.chunk_left_context * sub, stt);
        end = std::min(stt + params.chunk_len * sub, feature_frames);
        const int64_t right_offset = std::min(params.chunk_right_context * sub, feature_frames - end);
        SortformerChunkWindow window;
        window.win_lo = stt - left_offset;
        window.win_hi = end + right_offset;
        // Diar-frame context: left_offset is a multiple of sub -> round is
        // exact; right_offset may be short on the final chunk -> ceil.
        window.lc = (left_offset + sub / 2) / sub;
        window.rc = (right_offset + sub - 1) / sub;
        windows.push_back(window);
        stt = end;
    }
    return windows;
}

SortformerChunkedResult run_sortformer_chunked(
    const SortformerFeatureBatch & features,
    int64_t n_mels,
    int64_t subsampling_factor,
    int64_t n_spk,
    int64_t emb_dim,
    const SortformerStreamParams & params,
    const SortformerPreEncodeFn & pre_encode,
    const SortformerInferFn & infer,
    const SortformerChunkProgressFn & progress) {
    const int64_t feat_len = features.valid_frames;
    SortformerChunkedResult result;
    if (feat_len <= 0) {
        return result;
    }
    if (static_cast<int64_t>(features.time_major.size()) < feat_len * n_mels) {
        throw std::runtime_error("Sortformer diar feature buffer is shorter than its valid frame count");
    }
    const auto windows = plan_sortformer_chunks(feat_len, subsampling_factor, params);
    const int64_t n_chunks = static_cast<int64_t>(windows.size());

    SortformerStreamState state;
    state.reset(emb_dim);
    std::vector<float> chunk_embs;
    std::vector<float> concat;
    std::vector<float> preds;

    for (int64_t chunk_index = 0; chunk_index < n_chunks; ++chunk_index) {
        if (progress) {
            progress(chunk_index, n_chunks);
        }
        const auto & window = windows[static_cast<size_t>(chunk_index)];
        const int64_t M = window.win_hi - window.win_lo;

        // ---- graph A: pre_encode over the mel window ----
        int64_t T_diar = 0;
        pre_encode(features.time_major.data() + static_cast<std::ptrdiff_t>(window.win_lo * n_mels), M, chunk_embs, T_diar);
        if (T_diar <= 0 || static_cast<int64_t>(chunk_embs.size()) < T_diar * emb_dim) {
            throw std::runtime_error("Sortformer diar pre-encode produced no embeddings for a chunk");
        }
        chunk_embs.resize(static_cast<size_t>(T_diar * emb_dim));

        // ---- host concat [spkcache | fifo | chunk_embs] ----
        const int64_t S = state.spkcache_n;
        const int64_t F = state.fifo_n;
        const int64_t T_concat = S + F + T_diar;
        concat.resize(static_cast<size_t>(T_concat * emb_dim));
        std::copy(state.spkcache.begin(), state.spkcache.begin() + static_cast<std::ptrdiff_t>(S * emb_dim), concat.begin());
        std::copy(state.fifo.begin(), state.fifo.begin() + static_cast<std::ptrdiff_t>(F * emb_dim),
                  concat.begin() + static_cast<std::ptrdiff_t>(S * emb_dim));
        std::copy(chunk_embs.begin(), chunk_embs.end(), concat.begin() + static_cast<std::ptrdiff_t>((S + F) * emb_dim));

        // ---- graph B: encoder + transformer + head over the concat ----
        infer(concat, T_concat, preds);
        if (static_cast<int64_t>(preds.size()) < T_concat * n_spk) {
            throw std::runtime_error("Sortformer diar body produced too few probabilities for a chunk");
        }
        preds.resize(static_cast<size_t>(T_concat * n_spk));

        // ---- host streaming update (FIFO + AOSC compress) ----
        sortformer_streaming_update(state, params, n_spk, emb_dim, chunk_embs, T_diar, preds, window.lc, window.rc);
    }

    // Trim padding tail: NeMo total_preds[:, :ceil(feat_len/sub)].
    const int64_t n_frames = (feat_len + subsampling_factor - 1) / subsampling_factor;
    result.frames = std::min(state.total_n, n_frames);
    result.probabilities.assign(state.total_preds.begin(),
                                state.total_preds.begin() + static_cast<std::ptrdiff_t>(result.frames * n_spk));
    return result;
}

}  // namespace engine::models::sortformer_diar
