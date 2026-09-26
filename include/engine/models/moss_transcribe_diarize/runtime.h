#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/text/tokenizer_hub.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_transcribe_diarize {

// Hyperparameters the runtime reads, from config.json / processor_config.json /
// generation_config.json (audio.cpp package) or from stt.moss.* KVs
// (transcribe.cpp GGUF).
struct TranscribeConfig {
    int64_t vocab_size = 0;
    int64_t max_context = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    float audio_tokens_per_second = 12.5f;
    int64_t time_marker_every_seconds = 0;
    bool enable_time_marker = false;
    // generation_config max_new_tokens; 0 = the retired arch's adaptive budget
    // (max(256, 2 * audio tokens + 128)), used for the transcribe.cpp GGUF.
    int64_t default_max_tokens = 0;
};

// Opens either layout:
//   - the audio.cpp package (model spec resource bundle: HF json configs, the
//     Qwen2 BPE files, and a GGUF whose tensors carry the audio.cpp names), or
//   - the transcribe.cpp GGUF the retired `moss` arch read
//     (general.architecture == "moss", stt.moss.* KVs, tokenizer.ggml.* table,
//     huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf), whose
//     tensors are exposed under the audio.cpp names through a rename view.
struct TranscribeAssets {
    assets::ResourceBundle resources;  // empty for the transcribe.cpp GGUF
    std::shared_ptr<const assets::TensorSource> source;
    TranscribeConfig config;
    // audio.cpp package: the HF tokenizer files.
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> bpe;
    // transcribe.cpp GGUF: the tokenizer.ggml.* table, the ids HF encodes
    // atomically (its added tokens, longest first), and the converter's baked
    // default-prompt halves and digit ids (stt.moss.prompt_*_tokens / digit_tokens).
    text::TokenizerPtr gguf_tokenizer;
    std::vector<std::pair<std::string, int32_t>> added_tokens;
    std::vector<int32_t> prompt_prefix;
    std::vector<int32_t> prompt_suffix;
    std::vector<int32_t> digit_tokens;
};

std::shared_ptr<const TranscribeAssets> load_transcribe_assets(const std::filesystem::path & path);
bool looks_like_transcribe_moss_gguf(const std::filesystem::path & path);

// Called at every encoder chunk and decode step; may throw (RunControl cancellation).
using TranscribePoll = std::function<void(const char * stage, int64_t completed, int64_t total)>;

class TranscribeRuntime {
public:
    TranscribeRuntime(const TranscribeAssets & assets, core::ExecutionContext & execution);
    ~TranscribeRuntime();
    // max_tokens: nullopt = the layout default (TranscribeConfig::default_max_tokens).
    std::string transcribe(const runtime::AudioBuffer & audio, const std::string & instruction, std::optional<int64_t> max_tokens,
                           const TranscribePoll & poll = {});
    void start(const runtime::AudioBuffer & audio, const std::string & instruction, std::optional<int64_t> max_tokens,
               const TranscribePoll & poll = {});
    // Next decoded text delta; nullopt at EOS or when the token budget (clamped
    // to the decoder context) runs out, in which case truncated() is true
    // and the text generated so far stands as the partial result.
    std::optional<std::string> next_text();
    bool truncated() const;
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_transcribe_diarize
