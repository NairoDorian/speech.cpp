#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/community_models/parakeet_tdt/encoder.h"
#include "engine/community_models/parakeet_tdt/weights.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct ParakeetDecodeOptions {
    int64_t max_tokens = 0;
    bool keep_language_tags = false;
    // Invoked with (encoder frames consumed, frames in this call) as the greedy
    // loop advances. The session routes it to emit_progress(), which throws
    // runtime::ProgressCanceled on a cancel request - that is the decode
    // loop's cancellation point.
    std::function<void(int64_t, int64_t)> progress;
};

struct ParakeetDecodedText {
    std::string text;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> token_frame_indices;
    std::vector<int32_t> durations;
    // TranscribeGguf: per-token confidence as the arch computes it (TDT/RNNT
    // entropy confidence, CTC exp(log-prob)); empty for the audio.cpp layout.
    std::vector<float> token_probabilities;
    std::vector<runtime::WordTimestamp> word_timestamps;
    // TranscribeGguf result rows (arch/parakeet/model.cpp
    // build_result_from_raw_tokens): one row per kept token, the single
    // segment's span, and the unfiltered decode.
    std::vector<runtime::TokenTimestamp> token_timestamps;
    std::optional<runtime::TimeSpan> segment_span;
    std::optional<std::string> raw_text;
    // The token budget (max_tokens) or the greedy loop's iteration cap stopped
    // decoding before the last encoder frame.
    bool truncated = false;
};

class ParakeetDecoderRuntime {
public:
    ParakeetDecoderRuntime(
        std::shared_ptr<const ParakeetTDTAssets> assets,
        std::shared_ptr<const ParakeetWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes);
    ~ParakeetDecoderRuntime();

    void prepare();

    ParakeetDecodedText decode(
        const ParakeetEncodedAudio & encoded,
        const ParakeetDecodeOptions & options);

    void reset_state();
    ParakeetDecodedText decode_incremental(
        const ParakeetEncodedAudio & encoded,
        const ParakeetDecodeOptions & options,
        int64_t frame_offset = 0);
    ParakeetDecodedText format_tokens(
        std::vector<int32_t> token_ids,
        std::vector<int32_t> token_frame_indices,
        std::vector<int32_t> durations,
        const ParakeetDecodeOptions & options,
        int64_t audio_end_frame,
        std::vector<float> token_probabilities = {}) const;

    int32_t run_step(int32_t input_token, const float * encoder_frame, bool decoder_cache_valid, int32_t * out_dur_id = nullptr);
    int32_t run_joint_step(const float * encoder_frame, int32_t * out_dur_id = nullptr);

private:
    struct StepGraph;
    struct JointGraph;
    struct TranscribeGraphs;

    std::shared_ptr<const ParakeetTDTAssets> assets_;
    std::shared_ptr<const ParakeetWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<StepGraph> step_graph_;
    std::unique_ptr<JointGraph> joint_graph_;
    std::unique_ptr<TranscribeGraphs> transcribe_graphs_;
    std::vector<float> logits_scratch_;
    std::vector<float> hidden_scratch_;
    std::vector<float> cell_scratch_;
    std::vector<float> decoder_cache_scratch_;
    std::vector<float> hidden_read_scratch_;
    std::vector<float> cell_read_scratch_;
    int32_t pending_input_token_ = 0;
    bool predictor_cache_valid_ = false;
    bool state_initialized_ = false;

    // TranscribeGguf greedy state carried across decode_incremental() calls
    // (long-form / buffered streaming); the arch starts every utterance from
    // it zeroed (decoder.cpp LstmState::reset, last_token = -1).
    std::vector<std::vector<float>> transcribe_h_;
    std::vector<std::vector<float>> transcribe_c_;
    int32_t transcribe_last_token_ = -1;
    int32_t transcribe_prev_ctc_label_ = -1;

    void ensure_step_graph();
    void ensure_joint_graph();
    void ensure_transcribe_graphs();
    ParakeetDecodedText decode_incremental_transcribe(
        const ParakeetEncodedAudio & encoded,
        const ParakeetDecodeOptions & options,
        int64_t frame_offset);
    void decode_transducer_transcribe(
        const ParakeetEncodedAudio & encoded,
        const ParakeetDecodeOptions & options,
        int64_t frame_offset,
        ParakeetDecodedText & out);
    void decode_ctc_transcribe(
        const ParakeetEncodedAudio & encoded,
        const ParakeetDecodeOptions & options,
        int64_t frame_offset,
        ParakeetDecodedText & out);
    std::vector<float> project_frames(
        const ParakeetEncodedAudio & encoded,
        const engine::modules::LinearWeights & projection,
        int64_t out_features,
        const char * label) const;
    void format_transcribe(ParakeetDecodedText & out, bool keep_language_tags) const;
    std::string decode_text(const std::vector<int32_t> & token_ids, bool keep_language_tags) const;
    std::vector<runtime::WordTimestamp> build_word_timestamps(
        const std::vector<int32_t> & token_ids,
        const std::vector<int32_t> & token_frame_indices,
        const std::vector<int32_t> & durations,
        int64_t audio_end_frame) const;
};

}  // namespace engine::community_models::parakeet_tdt
