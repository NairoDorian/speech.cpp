#pragma once

// engine/framework/asr/sampling.h - token selection and decode-quality metrics
// shared by autoregressive ASR decoders (Phase 11a / W2b).
//
// These are the HF-generate() conventions the transcribe.cpp Whisper arch
// matched, lifted out of it verbatim so the engine package and any later
// family (the temperature-fallback recipe is not Whisper-specific) compute the
// same numbers. The arch now calls these too, so there is one implementation.
//
//   argmax_with_logprob   greedy pick + its log-softmax, in two passes
//   token_logprob         log_softmax(logits * rescale)[token], HF rescaling
//   sample_logits         multinomial draw at temperature T (T <= 0 = argmax)
//   token_compression_ratio
//                         HF _retrieve_compression_ratio: deflate ratio of the
//                         little-endian packed token ids (repetition detector)
//
// -INFINITY logits (suppressed / masked tokens) carry zero probability mass
// everywhere. All functions are pure apart from the RNG state they advance.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace engine::asr {

// Greedy argmax (first maximum wins, like argmax_logits in decode_driver.h)
// fused with the log-softmax of the winner. When every logit is -inf the
// result is id 0 with logprob -inf. `out_logprob` may be null.
int argmax_with_logprob(const float * logits, int vocab_size, float * out_logprob);

// log_softmax(logits * r)[token_id] with r = temperature if temperature > 0
// else 1 - the scaling HF's _retrieve_avg_logprobs applies (it multiplies by
// T rather than dividing, which is HF's choice and is what the fallback
// thresholds were tuned against). Out-of-range ids and all -inf rows give -inf.
float token_logprob(const float * logits, int vocab_size, int token_id, float temperature);

// Draws a token from softmax(logits / temperature). temperature <= 0 is the
// deterministic argmax. `scratch` is reused across calls to keep the per-step
// double[vocab] allocation off the decode hot path. The draw consumes exactly
// one uniform_real_distribution<double> sample from `rng`, so a seeded decode
// is reproducible across runs and across this function's callers.
int sample_logits(const float * logits, int vocab_size, float temperature, std::mt19937 & rng,
                  std::vector<double> & scratch);

// HF _retrieve_compression_ratio (generation_whisper.py): pack each id as
// little-endian with floor(log2(vocab_size) / 8) + 1 bytes, deflate at level 6,
// return raw_bytes / compressed_bytes. A ratio above ~2.4 means the decoder is
// repeating itself. Uses vendored miniz rather than Python's zlib, so the ratio
// can differ from HF by a few compressed bytes - accepted: it only gates a
// coarse threshold. Empty input (or a compressor failure) returns 0, which
// passes any threshold.
float token_compression_ratio(const int32_t * ids, size_t count, int64_t vocab_size);

}  // namespace engine::asr
