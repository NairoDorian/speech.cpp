// engine/framework/asr/sampling.cpp - see sampling.h.
//
// Ported verbatim from src/runtime/arch/whisper/model.cpp (sample_from_logits,
// sample_argmax_and_logprob, logprob_of_token_hf,
// compute_compression_ratio_hf; the arch was deleted in B16c and this is now
// the only copy); keep the arithmetic (double accumulation, max subtraction,
// strict > comparisons) unchanged - seeded decodes depend on it bit for bit.

#include "engine/framework/asr/sampling.h"

#include "miniz.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace engine::asr {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

}  // namespace

int argmax_with_logprob(const float * logits, int vocab_size, float * out_logprob) {
    int best_id = 0;
    float best_l = logits[0];
    float max_l = std::isfinite(best_l) ? best_l : kNegInf;
    for (int i = 1; i < vocab_size; ++i) {
        const float l = logits[i];
        if (l > best_l) {
            best_l = l;
            best_id = i;
        }
        if (std::isfinite(l) && l > max_l) {
            max_l = l;
        }
    }

    if (out_logprob != nullptr) {
        if (!std::isfinite(max_l)) {
            *out_logprob = kNegInf;
        } else {
            double sum_exp = 0.0;
            for (int i = 0; i < vocab_size; ++i) {
                const float l = logits[i];
                if (std::isfinite(l)) {
                    sum_exp += std::exp(static_cast<double>(l - max_l));
                }
            }
            if (sum_exp <= 0.0) {
                *out_logprob = kNegInf;
            } else {
                const float log_z = max_l + static_cast<float>(std::log(sum_exp));
                *out_logprob = best_l - log_z;
            }
        }
    }
    return best_id;
}

float token_logprob(const float * logits, int vocab_size, int token_id, float temperature) {
    if (token_id < 0 || token_id >= vocab_size) {
        return kNegInf;
    }
    const float rescale = (temperature > 0.0f) ? temperature : 1.0f;
    float max_l = kNegInf;
    for (int i = 0; i < vocab_size; ++i) {
        const float l = logits[i] * rescale;
        if (std::isfinite(l) && l > max_l) {
            max_l = l;
        }
    }
    if (!std::isfinite(max_l)) {
        return kNegInf;
    }
    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        const float l = logits[i] * rescale;
        if (std::isfinite(l)) {
            sum_exp += std::exp(static_cast<double>(l - max_l));
        }
    }
    if (sum_exp <= 0.0) {
        return kNegInf;
    }
    const float log_z = max_l + static_cast<float>(std::log(sum_exp));
    return logits[token_id] * rescale - log_z;
}

int sample_logits(const float * logits, int vocab_size, float temperature, std::mt19937 & rng,
                  std::vector<double> & scratch) {
    if (temperature <= 0.0f) {
        int best_id = 0;
        float best = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > best) {
                best = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    // Stabilise: subtract the max finite logit before exponentiating.
    float max_l = kNegInf;
    for (int i = 0; i < vocab_size; ++i) {
        const float l = logits[i];
        if (std::isfinite(l) && l > max_l) {
            max_l = l;
        }
    }
    if (!std::isfinite(max_l)) {
        return 0;  // every token masked; the caller's EOS handling ends the step loop
    }

    if (scratch.size() < static_cast<size_t>(vocab_size)) {
        scratch.resize(static_cast<size_t>(vocab_size));
    }
    std::fill_n(scratch.begin(), vocab_size, 0.0);

    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        const float l = logits[i];
        if (std::isfinite(l)) {
            const double p = std::exp(static_cast<double>((l - max_l) / temperature));
            scratch[static_cast<size_t>(i)] = p;
            sum += p;
        }
    }
    if (sum <= 0.0) {
        int best_id = 0;
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > logits[best_id]) {
                best_id = i;
            }
        }
        return best_id;
    }

    std::uniform_real_distribution<double> u(0.0, sum);
    const double r = u(rng);
    double acc = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        acc += scratch[static_cast<size_t>(i)];
        if (r < acc) {
            return i;
        }
    }
    return vocab_size - 1;  // floating-point edge: r landed at the very top
}

float token_compression_ratio(const int32_t * ids, size_t count, int64_t vocab_size) {
    if (ids == nullptr || count == 0) {
        return 0.0f;
    }
    int bytes_per_token = 1;
    if (vocab_size > 1) {
        const double lg2 = std::log2(static_cast<double>(vocab_size));
        bytes_per_token = static_cast<int>(std::floor(lg2 / 8.0)) + 1;
    }

    std::vector<uint8_t> raw(count * static_cast<size_t>(bytes_per_token));
    for (size_t i = 0; i < count; ++i) {
        const uint64_t v = static_cast<uint64_t>(static_cast<uint32_t>(ids[i]));
        for (int b = 0; b < bytes_per_token; ++b) {
            raw[i * static_cast<size_t>(bytes_per_token) + static_cast<size_t>(b)] =
                static_cast<uint8_t>((v >> (8 * b)) & 0xFF);
        }
    }

    mz_ulong dest_len = mz_compressBound(static_cast<mz_ulong>(raw.size()));
    std::vector<unsigned char> compressed(dest_len);
    const int rc = mz_compress(compressed.data(), &dest_len, raw.data(), static_cast<mz_ulong>(raw.size()));
    if (rc != MZ_OK || dest_len == 0) {
        return 0.0f;
    }
    return static_cast<float>(raw.size()) / static_cast<float>(dest_len);
}

}  // namespace engine::asr
