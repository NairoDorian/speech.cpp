#include "engine/framework/runtime/stream_chunker.h"

namespace engine::runtime {

StreamChunker::StreamChunker(int64_t chunk_samples, int sample_rate)
    : chunk_samples_(chunk_samples > 0 ? chunk_samples : 0),
      sample_rate_(sample_rate > 0 ? sample_rate : 16000) {}

void StreamChunker::reset(int64_t chunk_samples, int sample_rate) {
    chunk_samples_ = chunk_samples > 0 ? chunk_samples : 0;
    sample_rate_ = sample_rate > 0 ? sample_rate : 16000;
    input_samples_ = 0;
    committed_samples_ = 0;
    pending_.clear();
}

AudioChunk StreamChunker::make_chunk(const std::vector<float> & samples, int64_t start_sample) const {
    AudioChunk chunk;
    chunk.sample_rate = sample_rate_;
    chunk.channels = 1;
    chunk.start_sample = start_sample;
    chunk.samples = samples;
    return chunk;
}

}  // namespace engine::runtime
