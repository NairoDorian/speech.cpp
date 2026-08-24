#include "engine/framework/runtime/stream_chunker.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace engine::runtime;

static void test_chunk_reblocking() {
    StreamChunker chunker(512, 16000);
    assert(chunker.chunk_samples() == 512);
    assert(chunker.sample_rate() == 16000);

    std::vector<float> feed1(300, 1.0f);
    std::vector<AudioChunk> emitted;

    chunker.feed(feed1.data(), feed1.size(), [&emitted](const AudioChunk & chunk) {
        emitted.push_back(chunk);
        return true;
    });

    // 300 < 512, no chunk emitted yet
    assert(emitted.empty());
    assert(chunker.input_samples() == 300);
    assert(chunker.committed_samples() == 0);
    assert(chunker.buffered_samples() == 300);

    // Feed 300 more -> total 600 -> 1 chunk emitted (512 samples), 88 remain buffered
    std::vector<float> feed2(300, 2.0f);
    chunker.feed(feed2.data(), feed2.size(), [&emitted](const AudioChunk & chunk) {
        emitted.push_back(chunk);
        return true;
    });

    assert(emitted.size() == 1);
    assert(emitted[0].start_sample == 0);
    assert(emitted[0].samples.size() == 512);
    assert(chunker.input_samples() == 600);
    assert(chunker.committed_samples() == 512);
    assert(chunker.buffered_samples() == 88);

    // Flush trailing 88 samples -> emitted chunk must be zero-padded to 512, start_sample = 512
    chunker.flush([&emitted](const AudioChunk & chunk) {
        emitted.push_back(chunk);
        return true;
    });

    assert(emitted.size() == 2);
    assert(emitted[1].start_sample == 512);
    assert(emitted[1].samples.size() == 512);
    assert(chunker.committed_samples() == 600);  // Only real 600 samples committed
    assert(chunker.buffered_samples() == 0);

    std::cout << "[PASS] test_chunk_reblocking" << std::endl;
}

static void test_unconstrained_chunker() {
    StreamChunker chunker(0, 16000);
    std::vector<float> feed1(100, 1.0f);
    std::vector<AudioChunk> emitted;

    chunker.feed(feed1.data(), feed1.size(), [&emitted](const AudioChunk & chunk) {
        emitted.push_back(chunk);
        return true;
    });

    assert(emitted.size() == 1);
    assert(emitted[0].start_sample == 0);
    assert(emitted[0].samples.size() == 100);
    assert(chunker.committed_samples() == 100);

    std::cout << "[PASS] test_unconstrained_chunker" << std::endl;
}

int main() {
    test_chunk_reblocking();
    test_unconstrained_chunker();
    std::cout << "All StreamChunker unit tests passed successfully." << std::endl;
    return 0;
}
