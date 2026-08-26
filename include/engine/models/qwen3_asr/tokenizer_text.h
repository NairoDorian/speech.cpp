#pragma once

#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_asr/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::qwen3_asr {

class Qwen3ASRTextTokenizer {
public:
    struct Impl;

    explicit Qwen3ASRTextTokenizer(std::shared_ptr<const Qwen3ASRAssets> assets);

    Qwen3ASRPrompt build_prompt(
        const std::string & context,
        const std::string & language,
        int64_t audio_feature_tokens) const;

    Qwen3ASRPrompt build_raw_audio_prompt(
        const std::string & text,
        int64_t audio_feature_tokens) const;

    // Raw byte-level BPE encode of `text` (special tokens split out), i.e. the
    // exact ids build_prompt() feeds the thinker. Exposed for parity gates.
    std::vector<int32_t> encode(const std::string & text) const;

    std::string decode(const std::vector<int32_t> & token_ids) const;

private:
    std::shared_ptr<const Qwen3ASRAssets> assets_;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::qwen3_asr
