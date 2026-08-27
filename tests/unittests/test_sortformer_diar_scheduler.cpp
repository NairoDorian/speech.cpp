// tests/unittests/test_sortformer_diar_scheduler.cpp - host-side contract of
// the chunked Sortformer scheduler (Phase 10.5, family 3, step 2). No model:
// the stem and body are stand-ins, so what is under test is the bookkeeping
// that decides which frames the model sees and which predictions survive -
// chunk planning, FIFO / speaker-cache updates, cache compression, and the
// operating-point resolution with its precedence rules.

#include "engine/models/sortformer_diar/streaming.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace sf = engine::models::sortformer_diar;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    if (value[0] == '\0') {
        ::unsetenv(name);
    } else {
        ::setenv(name, value, 1);
    }
#endif
}

sf::SortformerModelConfig streaming_config() {
    sf::SortformerModelConfig config;
    config.modules.num_speakers = 4;
    config.fc_encoder.hidden_size = 8;
    config.fc_encoder.subsampling_factor = 8;
    config.streaming.present = true;
    config.streaming.chunk_len = 188;
    config.streaming.chunk_left_context = 1;
    config.streaming.chunk_right_context = 1;
    config.streaming.fifo_len = 0;
    config.streaming.spkcache_len = 188;
    config.streaming.spkcache_update_period = 188;
    return config;
}

// Subsampled frame count of a window of M mel frames through three 3x3
// stride-2 stages with padding 1 (what the stem yields).
int64_t subsampled(int64_t m) {
    for (int stage = 0; stage < 3; ++stage) {
        m = (m + 2 - 3) / 2 + 1;
    }
    return m;
}

void test_chunk_planning() {
    sf::SortformerStreamParams params;
    // Shipped geometry: a 1250-frame clip is one chunk with no context.
    auto windows = sf::plan_sortformer_chunks(1250, 8, params);
    CHECK(windows.size() == 1);
    CHECK(windows[0].win_lo == 0 && windows[0].win_hi == 1250);
    CHECK(windows[0].lc == 0 && windows[0].rc == 0);

    // The `small` diagnostic: 20-frame chunks, 1 frame of context each side.
    const auto * small = sf::find_sortformer_stream_preset("small");
    CHECK(small != nullptr);
    params.chunk_len = small->chunk_len;
    params.chunk_right_context = small->chunk_right_context;
    windows = sf::plan_sortformer_chunks(1250, 8, params);
    // ceil(1250 / 160) = 8 chunks.
    CHECK(windows.size() == 8);
    CHECK(windows[0].win_lo == 0 && windows[0].win_hi == 168 && windows[0].lc == 0 && windows[0].rc == 1);
    CHECK(windows[1].win_lo == 152 && windows[1].win_hi == 328 && windows[1].lc == 1 && windows[1].rc == 1);
    // Last chunk: 1120..1250 plus 8 frames of left context, no room on the right.
    CHECK(windows[7].win_lo == 1112 && windows[7].win_hi == 1250 && windows[7].lc == 1 && windows[7].rc == 0);
    // Windows tile the sequence: every new frame appears in exactly one chunk.
    int64_t covered = 0;
    for (const auto & w : windows) {
        covered += (w.win_hi - w.win_lo) - std::min<int64_t>(8, w.win_lo) * (w.lc > 0 ? 1 : 0) - (w.rc > 0 ? 8 : 0);
    }
    CHECK(covered == 1250);
    CHECK(sf::plan_sortformer_chunks(0, 8, params).empty());
}

void test_streaming_update_and_compression() {
    // Two synthetic speakers: frame f is speaker 0 when (f / 5) is even,
    // speaker 1 otherwise; embeddings encode the frame index so the cache can
    // be inspected after compression.
    const int64_t n_spk = 4;
    const int64_t D = 4;
    sf::SortformerStreamParams params;
    params.chunk_len = 20;
    params.chunk_left_context = 0;
    params.chunk_right_context = 0;
    params.fifo_len = 10;
    params.spkcache_len = 24;
    params.spkcache_update_period = 20;

    sf::SortformerStreamState state;
    state.reset(D);
    int64_t frame_counter = 0;
    auto make_chunk = [&](int64_t T, std::vector<float> & embs, std::vector<float> & chunk_preds) {
        embs.assign(static_cast<size_t>(T * D), 0.0f);
        chunk_preds.assign(static_cast<size_t>(T * n_spk), 0.0f);
        for (int64_t t = 0; t < T; ++t) {
            const int64_t f = frame_counter++;
            embs[static_cast<size_t>(t * D)] = static_cast<float>(f);
            const int64_t speaker = ((f / 5) % 2 == 0) ? 0 : 1;
            chunk_preds[static_cast<size_t>(t * n_spk + speaker)] = 0.9f;
        }
    };

    int64_t total = 0;
    for (int chunk = 0; chunk < 6; ++chunk) {
        std::vector<float> embs, chunk_preds;
        make_chunk(params.chunk_len, embs, chunk_preds);
        const int64_t S = state.spkcache_n;
        const int64_t F = state.fifo_n;
        // The body sees [spkcache | fifo | chunk]; stand in with the cached
        // preds followed by the chunk's.
        std::vector<float> preds;
        preds.insert(preds.end(), state.spkcache_preds.begin(), state.spkcache_preds.end());
        preds.resize(static_cast<size_t>(S * n_spk), 0.0f);
        preds.insert(preds.end(), state.fifo_preds.begin(), state.fifo_preds.end());
        preds.resize(static_cast<size_t>((S + F) * n_spk), 0.0f);
        preds.insert(preds.end(), chunk_preds.begin(), chunk_preds.end());
        sf::sortformer_streaming_update(state, params, n_spk, D, embs, params.chunk_len, preds, 0, 0);
        total += params.chunk_len;
        CHECK(state.total_n == total);
        CHECK(state.fifo_n <= params.fifo_len + params.chunk_len);
        CHECK(state.spkcache_n <= params.spkcache_len);
        CHECK(static_cast<int64_t>(state.spkcache.size()) == state.spkcache_n * D);
        CHECK(static_cast<int64_t>(state.fifo.size()) == state.fifo_n * D);
        CHECK(static_cast<int64_t>(state.total_preds.size()) == state.total_n * n_spk);
    }
    // 120 frames went through; the cache overflowed and was compressed.
    CHECK(state.compress_count >= 1);
    CHECK(state.spkcache_n == params.spkcache_len);
    CHECK(state.spkcache_preds_init);
    // Every retained cache row is either a real frame (embedding = its frame
    // index, prediction active) or the silence profile (no silence was fed,
    // so the profile is all zeros and its prediction row is zero).
    int64_t real_rows = 0;
    for (int64_t j = 0; j < state.spkcache_n; ++j) {
        const float frame = state.spkcache[static_cast<size_t>(j * D)];
        float active = 0.0f;
        for (int64_t s = 0; s < n_spk; ++s) {
            active += state.spkcache_preds[static_cast<size_t>(j * n_spk + s)];
        }
        if (active > 0.0f) {
            ++real_rows;
            CHECK(frame == std::floor(frame) && frame >= 0.0f && frame < static_cast<float>(total));
            const int64_t f = static_cast<int64_t>(frame);
            const int64_t speaker = ((f / 5) % 2 == 0) ? 0 : 1;
            CHECK(state.spkcache_preds[static_cast<size_t>(j * n_spk + speaker)] == 0.9f);
        } else {
            CHECK(frame == 0.0f);
        }
    }
    // Both speakers keep rows in the cache (the compressor reserves per-speaker
    // capacity: spkcache_len / n_spk - sil_frames = 3 rows each).
    int64_t rows_by_speaker[2] = {0, 0};
    for (int64_t j = 0; j < state.spkcache_n; ++j) {
        for (int64_t s = 0; s < 2; ++s) {
            if (state.spkcache_preds[static_cast<size_t>(j * n_spk + s)] > 0.5f) {
                ++rows_by_speaker[s];
            }
        }
    }
    CHECK(rows_by_speaker[0] >= 1 && rows_by_speaker[1] >= 1);
    CHECK(real_rows >= 2);
}

void test_run_plan_resolution() {
    set_env("TRANSCRIBE_SORTFORMER_STREAM_PRESET", "");
    set_env("TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN", "");
    const auto config = streaming_config();

    // Shipped geometry -> chunked by default with the shipped values.
    auto plan = sf::resolve_sortformer_run_plan(config, {});
    CHECK(plan.mode == sf::SortformerRunMode::Chunked);
    CHECK(plan.params.chunk_len == 188 && plan.params.fifo_len == 0 && plan.params.spkcache_len == 188);
    CHECK(plan.source == "default");

    // No shipped geometry -> whole-window by default, chunked on request.
    auto offline_config = config;
    offline_config.streaming.present = false;
    plan = sf::resolve_sortformer_run_plan(offline_config, {});
    CHECK(plan.mode == sf::SortformerRunMode::WholeWindow);
    plan = sf::resolve_sortformer_run_plan(offline_config, {{"stream_preset", "high_latency"}});
    CHECK(plan.mode == sf::SortformerRunMode::Chunked);
    CHECK(plan.params.chunk_len == 124 && plan.params.fifo_len == 124);

    // Explicit whole-window on a streaming package.
    plan = sf::resolve_sortformer_run_plan(config, {{"sortformer_diar.stream_preset", "offline"}});
    CHECK(plan.mode == sf::SortformerRunMode::WholeWindow);
    CHECK(plan.source == "offline");

    // Named presets: the publisher's table.
    plan = sf::resolve_sortformer_run_plan(config, {{"stream_preset", "very_high_latency"}});
    CHECK(plan.params.chunk_len == 340 && plan.params.chunk_right_context == 40 && plan.params.fifo_len == 40 &&
          plan.params.spkcache_update_period == 300 && plan.params.spkcache_len == 188);
    plan = sf::resolve_sortformer_run_plan(config, {{"stream_preset", "low_latency"}});
    CHECK(plan.params.chunk_len == 6 && plan.params.chunk_right_context == 7 && plan.params.fifo_len == 188 &&
          plan.params.spkcache_update_period == 144);

    // Per-field options ride on top of the preset and force the chunked path.
    plan = sf::resolve_sortformer_run_plan(config, {{"stream_preset", "high_latency"}, {"stream_chunk_len", "50"}});
    CHECK(plan.params.chunk_len == 50 && plan.params.fifo_len == 124);
    CHECK(plan.source == "high_latency+fields");

    // Rejections.
    bool threw = false;
    try {
        (void)sf::resolve_sortformer_run_plan(config, {{"stream_preset", "bogus"}});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)sf::resolve_sortformer_run_plan(config, {{"stream_spkcache_len", "4"}});  // no room per speaker
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)sf::resolve_sortformer_run_plan(config, {{"stream_preset", "offline"}, {"stream_chunk_len", "10"}});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);

    // Validation hooks outrank the options.
    set_env("TRANSCRIBE_SORTFORMER_STREAM_PRESET", "small");
    plan = sf::resolve_sortformer_run_plan(config, {{"stream_preset", "offline"}});
    CHECK(plan.mode == sf::SortformerRunMode::Chunked);
    CHECK(plan.params.chunk_len == 20 && plan.params.spkcache_len == 24);
    set_env("TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN", "33");
    plan = sf::resolve_sortformer_run_plan(config, {});
    CHECK(plan.params.chunk_len == 33);
    set_env("TRANSCRIBE_SORTFORMER_STREAM_PRESET", "");
    set_env("TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN", "");
}

void test_chunked_driver() {
    // A stand-in model: the stem returns one embedding row per subsampled
    // frame carrying the absolute mel frame it came from; the body marks every
    // row "speaker 0". The driver must then produce exactly ceil(feat_len /
    // sub) rows, in order, regardless of the chunk geometry.
    const int64_t n_mels = 2;
    const int64_t sub = 8;
    const int64_t n_spk = 4;
    const int64_t D = 1;
    const int64_t feat_len = 1250;
    sf::SortformerFeatureBatch features;
    features.valid_frames = feat_len;
    features.frames = feat_len;
    features.time_major.resize(static_cast<size_t>(feat_len * n_mels));
    for (int64_t t = 0; t < feat_len; ++t) {
        features.time_major[static_cast<size_t>(t * n_mels)] = static_cast<float>(t);
    }
    int64_t chunks_seen = 0;
    const sf::SortformerPreEncodeFn pre_encode = [&](const float * mel_window, int64_t window_frames,
                                                     std::vector<float> & embeddings, int64_t & T_diar) {
        T_diar = subsampled(window_frames);
        embeddings.assign(static_cast<size_t>(T_diar * D), 0.0f);
        for (int64_t t = 0; t < T_diar; ++t) {
            embeddings[static_cast<size_t>(t)] = mel_window[0] + static_cast<float>(t * sub);
        }
    };
    const sf::SortformerInferFn infer = [&](const std::vector<float> & concat, int64_t T_concat, std::vector<float> & preds) {
        preds.assign(static_cast<size_t>(T_concat * n_spk), 0.0f);
        for (int64_t t = 0; t < T_concat; ++t) {
            preds[static_cast<size_t>(t * n_spk)] = 0.5f + concat[static_cast<size_t>(t)] / 100000.0f;
        }
    };
    const sf::SortformerChunkProgressFn progress = [&](int64_t index, int64_t count) {
        CHECK(index == chunks_seen);
        CHECK(count > index);
        ++chunks_seen;
    };

    for (const char * preset_name : {"very_high_latency", "high_latency", "low_latency", "small"}) {
        const auto * preset = sf::find_sortformer_stream_preset(preset_name);
        CHECK(preset != nullptr);
        sf::SortformerStreamParams params;
        params.chunk_len = preset->chunk_len;
        params.chunk_right_context = preset->chunk_right_context;
        params.fifo_len = preset->fifo_len;
        params.spkcache_update_period = preset->spkcache_update_period;
        params.spkcache_len = preset->spkcache_len;
        chunks_seen = 0;
        const auto result = sf::run_sortformer_chunked(features, n_mels, sub, n_spk, D, params, pre_encode, infer, progress);
        CHECK(result.frames == (feat_len + sub - 1) / sub);
        CHECK(static_cast<int64_t>(result.probabilities.size()) == result.frames * n_spk);
        CHECK(chunks_seen == static_cast<int64_t>(sf::plan_sortformer_chunks(feat_len, sub, params).size()));
        // Rows are the chunk's middle frames in order: probability encodes the
        // absolute frame, so it must be strictly increasing.
        bool increasing = true;
        for (int64_t t = 1; t < result.frames; ++t) {
            if (result.probabilities[static_cast<size_t>(t * n_spk)] <= result.probabilities[static_cast<size_t>((t - 1) * n_spk)]) {
                increasing = false;
            }
        }
        CHECK(increasing);
        // The first row is frame 0 and the rows advance by one subsampled step.
        CHECK(std::fabs(result.probabilities[0] - 0.5f) < 1.0e-6f);
        CHECK(std::fabs(result.probabilities[static_cast<size_t>(n_spk)] - (0.5f + static_cast<float>(sub) / 100000.0f)) < 1.0e-6f);
    }
}

}  // namespace

int main() {
    test_chunk_planning();
    test_streaming_update_and_compression();
    test_run_plan_resolution();
    test_chunked_driver();
    if (g_failures != 0) {
        std::fprintf(stderr, "sortformer_diar_scheduler_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("sortformer_diar_scheduler_test: OK\n");
    return 0;
}
