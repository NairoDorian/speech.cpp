#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/stream_chunker.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::runtime {

enum class StreamLifecycleState {
    Idle = 0,
    Active = 1,
    Finished = 2,
    Failed = 3,
};

enum class StreamCommitPolicy {
    Auto = 0,
    OnFinalize = 1,
    StablePrefix = 2,
};

/**
 * StreamingSessionBase provides the unified 4-state streaming lifecycle,
 * monotonic revision counter, and commitment policy engine for all streaming
 * model sessions.
 *
 * It guarantees:
 *   1. The 4-state lifecycle (IDLE -> ACTIVE -> FINISHED | FAILED) is managed
 *      strictly by the base class.
 *   2. Revision numbers increase monotonically whenever observable text or events change.
 *   3. Pre-clear validation guarantees that a malformed request does not mutate
 *      prior snapshots or corrupt session state.
 *   4. Append-only committed text semantics: once text is committed, it is never
 *      rewritten by finalize or subsequent feeds.
 */
class StreamingSessionBase : public RuntimeSessionBase, public virtual IStreamingVoiceTaskSession {
public:
    explicit StreamingSessionBase(const SessionOptions & options);
    ~StreamingSessionBase() override = default;

    // --- State & Inspection Accessors ---
    StreamLifecycleState stream_state() const noexcept { return state_; }
    uint64_t stream_revision() const noexcept { return revision_; }
    StreamCommitPolicy commit_policy() const noexcept { return commit_policy_; }
    uint32_t stable_prefix_agreement_n() const noexcept { return stable_prefix_agreement_n_; }

    const std::string & full_text() const noexcept { return full_text_; }
    const std::string & committed_text() const noexcept { return committed_text_; }
    const std::string & tentative_text() const noexcept { return tentative_text_; }

    void set_commit_policy(StreamCommitPolicy policy, uint32_t agreement_n = 3) noexcept;

    const StreamChunker & chunker() const noexcept { return chunker_; }
    StreamChunker & chunker() noexcept { return chunker_; }

    // --- IStreamingVoiceTaskSession overrides (Base-owned lifecycle) ---
    void start_stream(const TaskRequest & request) override;
    StreamEvent process_audio_chunk(const AudioChunk & chunk) override;
    TaskResult finalize() override;
    TaskResult finish_stream() override;
    void reset() override;

    void set_stream_event_sink(StreamEventCallback sink) override;
    std::optional<StreamEvent> next_stream_event() override;

    /** Feed arbitrary PCM buffer; breaks into AudioChunks using StreamChunker. */
    void feed_pcm(const float * pcm, size_t n);

protected:
    // --- Model-Specific Virtual Hooks ---
    virtual void on_start_stream(const TaskRequest & request) { (void)request; }
    virtual StreamEvent on_process_audio_chunk(const AudioChunk & chunk) = 0;
    virtual TaskResult on_finalize() = 0;
    virtual void on_reset() {}

    // --- Pre-clear Validation Hooks (Pure / Const) ---
    virtual bool validate_stream(const TaskRequest & request) const { (void)request; return true; }
    virtual bool validate_chunk(const AudioChunk & chunk) const { (void)chunk; return true; }

    /** Helper to update text and evaluate commitment policy. */
    void update_text(const std::string & new_full_text, bool is_finalize = false);

    /** Helper to record and push an event to the sink or queue. */
    void record_event(const StreamEvent & event);

private:
    void apply_commitment_policy(bool is_finalize);
    size_t compute_stable_prefix_candidate() const;
    void mark_failed();

    StreamLifecycleState state_ = StreamLifecycleState::Idle;
    uint64_t revision_ = 0;
    StreamCommitPolicy commit_policy_ = StreamCommitPolicy::Auto;
    uint32_t stable_prefix_agreement_n_ = 3;

    std::string full_text_;
    std::string committed_text_;
    std::string tentative_text_;
    size_t raw_tentative_start_bytes_ = 0;

    std::deque<std::string> hypothesis_history_;
    std::deque<StreamEvent> pending_events_;
    StreamEventCallback event_sink_;

    StreamChunker chunker_;
};

}  // namespace engine::runtime
