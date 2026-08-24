#pragma once

#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace engine::runtime {

/**
 * StreamChunker manages PCM sample buffering and aligns arbitrary incoming
 * audio buffers to the fixed-size, contiguous AudioChunk requirements of
 * streaming sessions.
 *
 * It guarantees:
 *   1. Chunks dispatched to the session have contiguous start_sample cursors.
 *   2. Exactly preferred_audio_chunk_samples are emitted per step.
 *   3. Undispatched samples remain buffered across feed() calls.
 *   4. flush() emits trailing audio with zero-padding up to the required chunk
 *      boundary so no audio is lost at stream finalization.
 */
class StreamChunker {
public:
    explicit StreamChunker(int64_t chunk_samples = 0, int sample_rate = 16000);
    ~StreamChunker() = default;

    /** Reset state and set chunk granularity. */
    void reset(int64_t chunk_samples = 0, int sample_rate = 16000);

    int64_t input_samples() const noexcept { return input_samples_; }
    int64_t committed_samples() const noexcept { return committed_samples_; }
    int64_t buffered_samples() const noexcept { return input_samples_ - committed_samples_; }
    int64_t chunk_samples() const noexcept { return chunk_samples_; }
    int sample_rate() const noexcept { return sample_rate_; }

    /**
     * Append `n` samples from `pcm` and invoke `emit(chunk)` for each full chunk.
     * Emit is a callable taking `const AudioChunk &` and returning `bool` (false to abort).
     */
    template <typename EmitFn>
    void feed(const float * pcm, size_t n, const EmitFn & emit) {
        if (pcm != nullptr && n > 0) {
            pending_.insert(pending_.end(), pcm, pcm + n);
            input_samples_ += static_cast<int64_t>(n);
        }
        if (chunk_samples_ <= 0) {
            if (!pending_.empty()) {
                AudioChunk chunk = make_chunk(pending_, committed_samples_);
                if (emit(chunk)) {
                    committed_samples_ += static_cast<int64_t>(pending_.size());
                    pending_.clear();
                }
            }
            return;
        }

        const size_t chunk_sz = static_cast<size_t>(chunk_samples_);
        size_t offset = 0;
        while (pending_.size() - offset >= chunk_sz) {
            std::vector<float> block(pending_.begin() + static_cast<std::ptrdiff_t>(offset),
                                     pending_.begin() + static_cast<std::ptrdiff_t>(offset + chunk_sz));
            AudioChunk chunk = make_chunk(block, committed_samples_);
            if (!emit(chunk)) {
                break;
            }
            committed_samples_ += static_cast<int64_t>(chunk_sz);
            offset += chunk_sz;
        }
        if (offset > 0) {
            pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(offset));
        }
    }

    /**
     * Flush remaining buffered samples.
     * If chunk_samples_ > 0, the chunk is zero-padded up to the chunk size.
     */
    template <typename EmitFn>
    void flush(const EmitFn & emit) {
        if (pending_.empty()) {
            return;
        }
        const int64_t real_samples = static_cast<int64_t>(pending_.size());
        if (chunk_samples_ > 0 && pending_.size() < static_cast<size_t>(chunk_samples_)) {
            pending_.resize(static_cast<size_t>(chunk_samples_), 0.0f);
        }
        AudioChunk chunk = make_chunk(pending_, committed_samples_);
        (void)emit(chunk);
        committed_samples_ += real_samples;
        pending_.clear();
    }

private:
    AudioChunk make_chunk(const std::vector<float> & samples, int64_t start_sample) const;

    int64_t chunk_samples_ = 0;
    int sample_rate_ = 16000;
    int64_t input_samples_ = 0;
    int64_t committed_samples_ = 0;
    std::vector<float> pending_;
};

}  // namespace engine::runtime
