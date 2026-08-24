#include "engine/framework/runtime/streaming_session_base.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace engine::runtime;

class MockStreamingSession : public StreamingSessionBase {
public:
    explicit MockStreamingSession(const SessionOptions & options = {})
        : StreamingSessionBase(options) {}

    std::string family() const override { return "mock_streaming"; }
    VoiceTaskKind task_kind() const override { return VoiceTaskKind::Asr; }
    RunMode run_mode() const override { return RunMode::Streaming; }
    void prepare(const SessionPreparationRequest &) override { mark_prepared(); }

    StreamingPolicy streaming_policy() const override {
        StreamingPolicy policy;
        policy.input = StreamingInputKind::AudioChunks;
        policy.output = StreamingOutputKind::FinalResult;
        policy.preferred_audio_chunk_samples = 512;
        policy.preferred_audio_chunk_seconds = 512.0 / 16000.0;
        return policy;
    }

    void set_next_chunk_text(std::string text) {
        next_chunk_text_ = std::move(text);
    }

    void set_fail_validation(bool val) {
        fail_validation_ = val;
    }

protected:
    bool validate_stream(const TaskRequest &) const override {
        return !fail_validation_;
    }

    StreamEvent on_process_audio_chunk(const AudioChunk &) override {
        StreamEvent ev;
        if (!next_chunk_text_.empty()) {
            Transcript tr;
            tr.text = next_chunk_text_;
            ev.partial_text = tr;
        }
        return ev;
    }

    TaskResult on_finalize() override {
        TaskResult res;
        Transcript tr;
        tr.text = full_text();
        res.text_output = tr;
        return res;
    }

private:
    std::string next_chunk_text_;
    bool fail_validation_ = false;
};

static void test_lifecycle_and_monotonic_revision() {
    MockStreamingSession session;
    session.prepare({});
    assert(session.stream_state() == StreamLifecycleState::Idle);
    uint64_t rev0 = session.stream_revision();

    // Start stream
    TaskRequest req;
    session.start_stream(req);
    assert(session.stream_state() == StreamLifecycleState::Active);
    uint64_t rev1 = session.stream_revision();
    assert(rev1 > rev0);

    // Double start_stream throws
    bool threw = false;
    try {
        session.start_stream(req);
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);

    // Process chunk
    session.set_next_chunk_text("hello world");
    AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.samples.resize(512, 0.0f);
    session.process_audio_chunk(chunk);
    uint64_t rev2 = session.stream_revision();
    assert(rev2 > rev1);

    // Finalize
    TaskResult res = session.finalize();
    assert(session.stream_state() == StreamLifecycleState::Finished);
    assert(res.text_output.has_value());
    assert(res.text_output->text == "hello world");
    uint64_t rev3 = session.stream_revision();
    assert(rev3 > rev2);

    // Reset back to Idle
    session.reset();
    assert(session.stream_state() == StreamLifecycleState::Idle);
    assert(session.full_text().empty());
    assert(session.committed_text().empty());
    uint64_t rev4 = session.stream_revision();
    assert(rev4 > rev3);

    std::cout << "[PASS] test_lifecycle_and_monotonic_revision" << std::endl;
}

static void test_commitment_policies() {
    // 1. Auto policy
    {
        MockStreamingSession session;
        session.prepare({});
        session.set_commit_policy(StreamCommitPolicy::Auto);
        session.start_stream({});

        session.set_next_chunk_text("alpha beta");
        AudioChunk chunk;
        chunk.sample_rate = 16000;
        chunk.samples.resize(512, 0.0f);
        session.process_audio_chunk(chunk);

        assert(session.full_text() == "alpha beta");
        assert(session.committed_text() == "alpha beta");
        assert(session.tentative_text().empty());
    }

    // 2. OnFinalize policy
    {
        MockStreamingSession session;
        session.prepare({});
        session.set_commit_policy(StreamCommitPolicy::OnFinalize);
        session.start_stream({});

        session.set_next_chunk_text("alpha beta");
        AudioChunk chunk;
        chunk.sample_rate = 16000;
        chunk.samples.resize(512, 0.0f);
        session.process_audio_chunk(chunk);

        assert(session.full_text() == "alpha beta");
        assert(session.committed_text().empty());
        assert(session.tentative_text() == "alpha beta");

        session.finalize();
        assert(session.committed_text() == "alpha beta");
        assert(session.tentative_text().empty());
    }

    // 3. StablePrefix policy (N=3 agreement)
    {
        MockStreamingSession session;
        session.prepare({});
        session.set_commit_policy(StreamCommitPolicy::StablePrefix, 3);
        session.start_stream({});

        AudioChunk chunk;
        chunk.sample_rate = 16000;
        chunk.samples.resize(512, 0.0f);

        // Feed step 1: "The quick brown fox"
        session.set_next_chunk_text("The quick brown fox");
        session.process_audio_chunk(chunk);
        assert(session.committed_text().empty());

        // Feed step 2: "The quick brown dog"
        session.set_next_chunk_text("The quick brown dog");
        session.process_audio_chunk(chunk);
        assert(session.committed_text().empty());

        // Feed step 3: "The quick brown cat" -> 3 hyp agree on "The quick brown "
        session.set_next_chunk_text("The quick brown cat");
        session.process_audio_chunk(chunk);
        assert(session.committed_text() == "The quick brown ");
        assert(session.tentative_text() == "cat");

        session.finalize();
        assert(session.committed_text() == "The quick brown cat");
        assert(session.tentative_text().empty());
    }

    std::cout << "[PASS] test_commitment_policies" << std::endl;
}

static void test_pre_clear_validation_failure() {
    MockStreamingSession session;
    session.prepare({});
    session.set_fail_validation(true);

    TaskRequest req;
    bool threw = false;
    try {
        session.start_stream(req);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    assert(session.stream_state() == StreamLifecycleState::Idle);

    std::cout << "[PASS] test_pre_clear_validation_failure" << std::endl;
}

int main() {
    test_lifecycle_and_monotonic_revision();
    test_commitment_policies();
    test_pre_clear_validation_failure();
    std::cout << "All StreamingSessionBase unit tests passed successfully." << std::endl;
    return 0;
}
