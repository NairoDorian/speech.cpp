#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::text {
enum class TextChunkMode;
}

namespace engine::runtime {

enum class VoiceTaskKind {
    Vad,
    Asr,
    Diarization,
    SourceSeparation,
    AudioGeneration,
    Tts,
    VoiceCloning,
    VoiceConversion,
    SpeechToSpeech,
    Alignment,
    VoiceDesign,
    SpeakerRecognition,
    Svc,
    Midi,
};

enum class RunMode {
    Offline,
    Streaming,
};

enum class GraphCapacityMode {
    Fixed,
    Tiered,
    Grow,
    Double,
    Unsupported,
};

struct TaskSpec {
    VoiceTaskKind task = VoiceTaskKind::Vad;
    RunMode mode = RunMode::Offline;
};

struct SessionOptions {
    engine::core::BackendConfig backend = {};
    std::unordered_map<std::string, std::string> options;
};

struct AudioBuffer {
    int sample_rate = 0;
    int channels = 1;
    std::vector<float> samples;
};

struct NamedAudioBuffer {
    std::string id;
    AudioBuffer audio;
    std::unordered_map<std::string, std::string> meta;
};

struct AudioChunk {
    int sample_rate = 0;
    int channels = 1;
    int64_t start_sample = 0;
    std::vector<float> samples;
};

struct Transcript {
    std::string text;
    std::string language;
};

struct TimeSpan {
    int64_t start_sample = 0;
    int64_t end_sample = 0;
};

struct SpeechSegment {
    TimeSpan span;
    float confidence = 0.0f;
    std::string text;
};

struct SpeakerTurn {
    TimeSpan span;
    std::string speaker_id;
    float confidence = 0.0f;
    std::string text;
};

struct WordTimestamp {
    TimeSpan span;
    std::string word;
    float confidence = 0.0f;
};

struct VoiceReference {
    std::optional<AudioBuffer> audio = std::nullopt;
    std::optional<std::string> cached_voice_id = std::nullopt;
};

struct StyleCondition {
    std::optional<std::string> language = std::nullopt;
    std::optional<std::string> emotion = std::nullopt;
    std::optional<float> speaking_rate = std::nullopt;
    std::optional<float> pitch_shift = std::nullopt;
    std::optional<float> energy_scale = std::nullopt;
    std::unordered_map<std::string, std::string> tags;
};

struct VoiceCondition {
    std::optional<VoiceReference> speaker = std::nullopt;
    std::optional<StyleCondition> style = std::nullopt;
};

enum class ArtifactKind {
    SpeakerEmbedding,
    StyleEmbedding,
    PromptEmbedding,
    AcousticTokens,
    Midi,
    TranscriptAlignment,
    DiarizationState,
    VadState,
    Custom,
};

struct VoiceArtifact {
    ArtifactKind kind = ArtifactKind::Custom;
    std::string id;
    std::vector<std::byte> payload;
    std::unordered_map<std::string, std::string> meta;
};

std::vector<std::byte> bytes_from_string(std::string_view value);
VoiceArtifact make_voice_artifact(
    ArtifactKind kind,
    std::string id,
    std::vector<std::byte> payload,
    std::unordered_map<std::string, std::string> meta = {});
VoiceArtifact make_text_artifact(
    ArtifactKind kind,
    std::string id,
    std::string_view payload,
    std::unordered_map<std::string, std::string> meta = {});

struct TaskRequest {
    std::optional<Transcript> text_input = std::nullopt;
    std::optional<AudioBuffer> audio_input = std::nullopt;
    std::optional<VoiceCondition> voice = std::nullopt;
    std::vector<VoiceArtifact> input_artifacts;
    std::unordered_map<std::string, std::string> options;
};

struct AudioPreparationContract {
    int sample_rate = 0;
    int channels = 1;
    int64_t max_input_samples = 0;
};

struct SessionPreparationRequest {
    std::optional<AudioPreparationContract> audio = std::nullopt;
    std::optional<Transcript> text = std::nullopt;
    std::optional<VoiceCondition> voice = std::nullopt;
    std::unordered_map<std::string, std::string> options;
};

struct VoiceActivityEvent {
    enum class Kind {
        SpeechStart,
        SpeechEnd,
        SpeechSegment,
    };

    Kind kind = Kind::SpeechSegment;
    int64_t sample = 0;
    float probability = 0.0f;
    std::optional<SpeechSegment> segment = std::nullopt;
};

struct DecodeTelemetry {
    int64_t t0_ms = 0;
    int64_t t1_ms = 0;
    float temperature_used = 0.0f;
    float compression_ratio = 0.0f;
    float avg_logprob = 0.0f;
    float no_speech_prob = 0.0f;
    bool no_speech_triggered = false;
    int32_t n_fallbacks = 0;
};

struct TaskResult {
    std::optional<AudioBuffer> audio_output = std::nullopt;
    std::vector<NamedAudioBuffer> named_audio_outputs;
    std::optional<Transcript> text_output = std::nullopt;
    std::vector<SpeechSegment> speech_segments;
    std::vector<SpeakerTurn> speaker_turns;
    std::vector<WordTimestamp> word_timestamps;
    std::vector<DecodeTelemetry> decode_telemetry;
    std::optional<VoiceArtifact> artifact_output = std::nullopt;
    std::vector<VoiceArtifact> output_artifacts;
};

struct StreamEvent {
    std::vector<VoiceActivityEvent> voice_activity;
    std::optional<Transcript> partial_text = std::nullopt;
    std::optional<AudioBuffer> audio_output = std::nullopt;
    std::vector<NamedAudioBuffer> named_audio_outputs;
    std::vector<SpeakerTurn> speaker_turns;
    std::vector<WordTimestamp> word_timestamps;
    std::vector<VoiceArtifact> output_artifacts;
    bool is_final = false;
};

// Progress reporting for offline run(). A session invokes the callback at chunk
// boundaries (for chunked TTS/ASR) or once at start/end for single-shot models.
// Returning false from the callback requests cancellation; run() then throws
// ProgressCanceled. The callback is always invoked synchronously on the same
// thread that called run().
struct ProgressInfo {
    float progress = 0.0f;          // [0.0, 1.0]
    std::string stage;              // model family name, e.g. "qwen3_tts"
    int64_t completed_units = 0;    // chunks completed so far
    int64_t total_units = 0;        // total chunks (1 for single-shot models)
};

using ProgressCallback = std::function<bool(const ProgressInfo &)>;

// Thrown by run() when a progress callback returns false (cancellation).
struct ProgressCanceled : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class StreamingInputKind {
    None,
    AudioChunks,
};

enum class StreamingOutputKind {
    FinalResult,
    PullEvents,
};

struct StreamingPolicy {
    StreamingInputKind input = StreamingInputKind::None;
    StreamingOutputKind output = StreamingOutputKind::FinalResult;
    int64_t preferred_audio_chunk_samples = 0;
    double preferred_audio_chunk_seconds = 0.0;
};

using StreamEventCallback = std::function<void(const StreamEvent &)>;

class IVoiceTaskSession {
public:
    virtual ~IVoiceTaskSession() = default;

    virtual std::string family() const = 0;
    virtual VoiceTaskKind task_kind() const = 0;
    virtual RunMode run_mode() const = 0;
    virtual void prepare(const SessionPreparationRequest & request) = 0;

    // Install a progress observer invoked from inside run() / streaming runs.
    // Default no-op; RuntimeSessionBase overrides to store the callback so all
    // concrete sessions inherit the plumbing. Call before run()/start_stream().
    virtual void set_progress_callback(ProgressCallback cb) {
        (void)cb;
    }

    // Request cooperative abortion of the in-flight run or stream.
    virtual void request_abort() {}

    // Check whether abortion has been requested.
    virtual bool is_aborted() const {
        return false;
    }
};

class IOfflineVoiceTaskSession : public virtual IVoiceTaskSession {
public:
    ~IOfflineVoiceTaskSession() override = default;

    virtual TaskResult run(const TaskRequest & request) = 0;

    // Batched offline execution (Phase 4). Default implementation runs
    // serially over requests; model sessions can override with a native
    // shared batch graph for high-throughput multi-utterance decoding.
    virtual std::vector<TaskResult> run_batch(const std::vector<TaskRequest> & requests) {
        std::vector<TaskResult> results;
        results.reserve(requests.size());
        for (const auto & req : requests) {
            results.push_back(run(req));
        }
        return results;
    }
};

class IStreamingVoiceTaskSession : public virtual IVoiceTaskSession {
public:
    ~IStreamingVoiceTaskSession() override = default;

    virtual StreamingPolicy streaming_policy() const {
        StreamingPolicy policy;
        policy.input = StreamingInputKind::AudioChunks;
        policy.output = StreamingOutputKind::FinalResult;
        policy.preferred_audio_chunk_samples = 512;
        return policy;
    }
    virtual void start_stream(const TaskRequest & request) {
        (void)request;
        reset();
    }
    virtual std::optional<StreamEvent> next_stream_event() {
        return std::nullopt;
    }
    virtual void set_stream_event_sink(StreamEventCallback sink) {
        (void)sink;
    }
    virtual TaskResult finish_stream() {
        return finalize();
    }
    virtual void reset() = 0;
    virtual StreamEvent process_audio_chunk(const AudioChunk & chunk) = 0;
    virtual TaskResult finalize() = 0;
};

SessionPreparationRequest build_preparation_request(const AudioBuffer & audio);
SessionPreparationRequest build_preparation_request(const TaskRequest & request);
std::vector<TaskRequest> chunk_text_request(const TaskRequest & request, int64_t codepoint_budget);
std::vector<TaskRequest> chunk_text_request(
    const TaskRequest & request,
    int64_t codepoint_budget,
    engine::text::TextChunkMode mode);
void append_audio_buffer(AudioBuffer & dst, const AudioBuffer & src);

class IGraphCapacityAdapter {
public:
    virtual ~IGraphCapacityAdapter() = default;

    virtual int64_t base_capacity() const = 0;
    virtual int64_t fixed_capacity() const = 0;
    virtual int64_t canonical_capacity_for_request(int64_t request_size) const = 0;
    virtual std::vector<int64_t> prepared_capacities() const = 0;
    virtual void prepare_capacity(int64_t capacity) = 0;
};

using GraphCapacityCanonicalFn = std::function<int64_t(int64_t)>;
using GraphCapacityPreparedFn = std::function<std::vector<int64_t>()>;
using GraphCapacityPrepareFn = std::function<void(int64_t)>;

class MappedGraphCapacityAdapter final : public IGraphCapacityAdapter {
public:
    MappedGraphCapacityAdapter(
        int64_t base_capacity,
        int64_t fixed_capacity,
        GraphCapacityCanonicalFn canonical_capacity_fn,
        GraphCapacityPreparedFn prepared_capacities_fn,
        GraphCapacityPrepareFn prepare_capacity_fn);

    int64_t base_capacity() const override;
    int64_t fixed_capacity() const override;
    int64_t canonical_capacity_for_request(int64_t request_size) const override;
    std::vector<int64_t> prepared_capacities() const override;
    void prepare_capacity(int64_t capacity) override;

private:
    int64_t base_capacity_ = 0;
    int64_t fixed_capacity_ = 0;
    GraphCapacityCanonicalFn canonical_capacity_fn_;
    GraphCapacityPreparedFn prepared_capacities_fn_;
    GraphCapacityPrepareFn prepare_capacity_fn_;
};

class DiscreteGraphCapacityAdapter final : public IGraphCapacityAdapter {
public:
    DiscreteGraphCapacityAdapter(
        std::vector<int64_t> capacities,
        GraphCapacityPreparedFn prepared_capacities_fn,
        GraphCapacityPrepareFn prepare_capacity_fn);

    int64_t base_capacity() const override;
    int64_t fixed_capacity() const override;
    int64_t canonical_capacity_for_request(int64_t request_size) const override;
    std::vector<int64_t> prepared_capacities() const override;
    void prepare_capacity(int64_t capacity) override;

private:
    std::vector<int64_t> capacities_;
    GraphCapacityPreparedFn prepared_capacities_fn_;
    GraphCapacityPrepareFn prepare_capacity_fn_;
};

class GraphCapacityController {
public:
    GraphCapacityController() = default;
    explicit GraphCapacityController(GraphCapacityMode mode);

    GraphCapacityMode mode() const noexcept;
    int64_t ensure_prepared(IGraphCapacityAdapter & adapter, int64_t request_size = 0) const;
    int64_t select_capacity_for_run(const IGraphCapacityAdapter & adapter, int64_t request_size) const;

private:
    GraphCapacityMode mode_ = GraphCapacityMode::Unsupported;
};

const char * to_string(VoiceTaskKind task) noexcept;
const char * to_string(RunMode mode) noexcept;
const char * to_string(GraphCapacityMode mode) noexcept;
VoiceTaskKind parse_voice_task_kind(const std::string & value);
RunMode parse_run_mode(const std::string & value);
GraphCapacityMode parse_graph_capacity_mode(const std::string & value);
GraphCapacityMode resolve_graph_capacity_mode(
    const SessionOptions & options,
    GraphCapacityMode default_mode);
GraphCapacityMode resolve_graph_capacity_mode(
    const SessionOptions & options,
    GraphCapacityMode default_mode,
    std::initializer_list<const char *> keys);

}  // namespace engine::runtime
