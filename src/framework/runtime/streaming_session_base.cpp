#include "engine/framework/runtime/streaming_session_base.h"

#include <algorithm>
#include <stdexcept>

namespace engine::runtime {

namespace {

// Ensure we do not split UTF-8 continuation bytes [0x80, 0xBF].
static size_t utf8_floor_boundary(const std::string & text, size_t pos) {
    if (pos >= text.size()) {
        return text.size();
    }
    while (pos > 0 && (static_cast<uint8_t>(text[pos]) & 0xC0) == 0x80) {
        --pos;
    }
    return pos;
}

static size_t common_prefix_bytes(const std::string & a, const std::string & b) {
    const size_t limit = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < limit && a[i] == b[i]) {
        ++i;
    }
    return i;
}

}  // namespace

StreamingSessionBase::StreamingSessionBase(const SessionOptions & options)
    : RuntimeSessionBase(options) {}

void StreamingSessionBase::set_commit_policy(StreamCommitPolicy policy, uint32_t agreement_n) noexcept {
    commit_policy_ = policy;
    stable_prefix_agreement_n_ = (agreement_n > 0) ? agreement_n : 3;
}

void StreamingSessionBase::set_stream_event_sink(StreamEventCallback sink) {
    event_sink_ = std::move(sink);
}

std::optional<StreamEvent> StreamingSessionBase::next_stream_event() {
    if (pending_events_.empty()) {
        return std::nullopt;
    }
    StreamEvent ev = std::move(pending_events_.front());
    pending_events_.pop_front();
    return ev;
}

void StreamingSessionBase::mark_failed() {
    state_ = StreamLifecycleState::Failed;
    revision_++;
}

void StreamingSessionBase::start_stream(const TaskRequest & request) {
    if (state_ == StreamLifecycleState::Active) {
        throw std::runtime_error("start_stream() called while stream is already Active; finalize or reset first");
    }

    if (!validate_stream(request)) {
        throw std::invalid_argument("stream validation failed for request");
    }

    // Reset runtime state
    full_text_.clear();
    committed_text_.clear();
    tentative_text_.clear();
    raw_tentative_start_bytes_ = 0;
    hypothesis_history_.clear();
    pending_events_.clear();

    const auto policy = streaming_policy();
    chunker_.reset(policy.preferred_audio_chunk_samples);

    run_control().reset_abort();
    state_ = StreamLifecycleState::Active;
    revision_++;

    try {
        on_start_stream(request);
    } catch (...) {
        mark_failed();
        throw;
    }
}

StreamEvent StreamingSessionBase::process_audio_chunk(const AudioChunk & chunk) {
    if (state_ != StreamLifecycleState::Active) {
        throw std::runtime_error("process_audio_chunk() called while stream is not in Active state");
    }

    if (!validate_chunk(chunk)) {
        throw std::invalid_argument("chunk validation failed");
    }

    if (run_control().poll_abort()) {
        mark_failed();
        throw ProgressCanceled("Stream aborted by user request");
    }

    StreamEvent ev;
    try {
        ev = on_process_audio_chunk(chunk);
    } catch (...) {
        mark_failed();
        throw;
    }

    if (ev.partial_text.has_value() && !ev.partial_text->text.empty()) {
        update_text(ev.partial_text->text, /*is_finalize=*/false);
    }

    record_event(ev);
    return ev;
}

void StreamingSessionBase::feed_pcm(const float * pcm, size_t n) {
    if (state_ != StreamLifecycleState::Active) {
        throw std::runtime_error("feed_pcm() called while stream is not Active");
    }
    chunker_.feed(pcm, n, [this](const AudioChunk & chunk) {
        process_audio_chunk(chunk);
        return !run_control().poll_abort();
    });
}

TaskResult StreamingSessionBase::finalize() {
    if (state_ != StreamLifecycleState::Active) {
        throw std::runtime_error("finalize() called while stream is not in Active state");
    }

    // Flush any remaining buffered samples in chunker
    chunker_.flush([this](const AudioChunk & chunk) {
        process_audio_chunk(chunk);
        return true;
    });

    TaskResult result;
    try {
        result = on_finalize();
    } catch (...) {
        mark_failed();
        throw;
    }

    if (result.text_output.has_value()) {
        update_text(result.text_output->text, /*is_finalize=*/true);
    } else {
        apply_commitment_policy(/*is_finalize=*/true);
    }

    state_ = StreamLifecycleState::Finished;
    revision_++;
    return result;
}

TaskResult StreamingSessionBase::finish_stream() {
    return finalize();
}

void StreamingSessionBase::reset() {
    try {
        on_reset();
    } catch (...) {
        // Suppress on reset
    }

    full_text_.clear();
    committed_text_.clear();
    tentative_text_.clear();
    raw_tentative_start_bytes_ = 0;
    hypothesis_history_.clear();
    pending_events_.clear();

    chunker_.reset(streaming_policy().preferred_audio_chunk_samples);
    run_control().reset_abort();

    state_ = StreamLifecycleState::Idle;
    revision_++;
}

void StreamingSessionBase::update_text(const std::string & new_full_text, bool is_finalize) {
    const std::string prev_committed = committed_text_;
    const std::string prev_tentative = tentative_text_;

    full_text_ = new_full_text;
    apply_commitment_policy(is_finalize);

    if (committed_text_ != prev_committed || tentative_text_ != prev_tentative) {
        revision_++;
    }
}

void StreamingSessionBase::record_event(const StreamEvent & event) {
    if (event_sink_) {
        event_sink_(event);
    }
    pending_events_.push_back(event);
    revision_++;
}

size_t StreamingSessionBase::compute_stable_prefix_candidate() const {
    if (hypothesis_history_.empty()) {
        return 0;
    }
    if (stable_prefix_agreement_n_ <= 1) {
        return full_text_.size();
    }
    if (hypothesis_history_.size() < stable_prefix_agreement_n_) {
        return 0;
    }

    size_t prefix_len = hypothesis_history_.front().size();
    for (const auto & hyp : hypothesis_history_) {
        prefix_len = std::min(prefix_len, common_prefix_bytes(hypothesis_history_.front(), hyp));
    }
    return utf8_floor_boundary(full_text_, prefix_len);
}

void StreamingSessionBase::apply_commitment_policy(bool is_finalize) {
    if (is_finalize) {
        if (commit_policy_ == StreamCommitPolicy::OnFinalize || committed_text_.empty()) {
            committed_text_ = full_text_;
            raw_tentative_start_bytes_ = full_text_.size();
        } else if (full_text_.size() >= committed_text_.size() &&
                   full_text_.compare(0, committed_text_.size(), committed_text_) == 0) {
            committed_text_ = full_text_;
            raw_tentative_start_bytes_ = full_text_.size();
        }
        tentative_text_.clear();
        return;
    }

    if (full_text_.empty()) {
        tentative_text_.clear();
        return;
    }

    size_t candidate_bytes = 0;
    switch (commit_policy_) {
        case StreamCommitPolicy::OnFinalize:
            candidate_bytes = 0;
            break;
        case StreamCommitPolicy::StablePrefix: {
            hypothesis_history_.push_back(full_text_);
            while (hypothesis_history_.size() > stable_prefix_agreement_n_) {
                hypothesis_history_.pop_front();
            }
            candidate_bytes = compute_stable_prefix_candidate();
            break;
        }
        case StreamCommitPolicy::Auto:
        default:
            // Auto commits everything that is verified or stable
            candidate_bytes = full_text_.size();
            break;
    }

    const size_t old_boundary = utf8_floor_boundary(
        full_text_, std::min(raw_tentative_start_bytes_, full_text_.size()));
    const size_t candidate = utf8_floor_boundary(full_text_, std::min(candidate_bytes, full_text_.size()));

    if (candidate > old_boundary && old_boundary == committed_text_.size() &&
        full_text_.compare(0, old_boundary, committed_text_) == 0) {
        committed_text_.append(full_text_.data() + old_boundary, candidate - old_boundary);
        raw_tentative_start_bytes_ = candidate;
    }

    const size_t tentative_start = utf8_floor_boundary(
        full_text_, std::min(raw_tentative_start_bytes_, full_text_.size()));
    if (tentative_start < full_text_.size()) {
        tentative_text_.assign(full_text_.data() + tentative_start, full_text_.size() - tentative_start);
    } else {
        tentative_text_.clear();
    }
}

}  // namespace engine::runtime
