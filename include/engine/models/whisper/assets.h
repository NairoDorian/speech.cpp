#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/text/tokenizer_hub.h"
#include "engine/models/whisper/graphs_internal.h"

#include "ggml.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::whisper {

// The two ways Whisper is distributed.
//
//   TranscribeGguf  parent transcribe.cpp's GGUF (`general.architecture =
//                   "whisper"`, `stt.whisper.*` / `stt.frontend.*` KVs, a
//                   GPT-2 tokenizer holding every special and timestamp token,
//                   the mel filterbank and window as tensors). Published for
//                   every variant under huggingface.co/handy-computer/.
//   LegacyBin       whisper.cpp's monolithic `.bin` (ggml magic, 11 int32
//                   hparams, filterbank, raw-bytes vocab, tensors). No
//                   generation config: specials, suppression lists and the
//                   language table are synthesized.
enum class WhisperWeightLayout { TranscribeGguf, LegacyBin };

// Host-side model resources shared across sessions (Phase 11 W2a/W2b).
struct WhisperAssets {
  WhisperHParams hparams;
  WhisperWeightLayout layout = WhisperWeightLayout::LegacyBin;
  std::string variant; // "tiny.en", "whisper-base", ...
  std::filesystem::path model_path;

  // Weights, for both layouts, through the shared TensorSource (the .bin one
  // streams each payload from disk). Tensor names differ per layout; ask
  // through tensor_name() with the legacy whisper.cpp name.
  std::shared_ptr<const assets::TensorSource> source;

  // Encode + decode, HF-exact for both layouts (TokenizerHub: GPT-2 byte-level
  // BPE for the GGUF, tiktoken rank BPE over the raw-bytes vocab for the .bin).
  text::TokenizerPtr tokenizer;

  // Languages in the order the arch iterated them for detection (GGUF:
  // general.languages; .bin: whisper.cpp's table), with their <|xx|> ids.
  // English-only models keep {"en"} and no ids: their prompt has no slot.
  std::vector<std::string> language_codes;
  std::vector<int32_t> language_token_ids;

  // Frontend: row-major [n_mels, n_fft / 2 + 1]; the window is shipped by the
  // GGUF and computed (periodic Hann) for the .bin, which carries none.
  std::vector<float> mel_filterbank;
  std::vector<float> window;

  // Raw vocabulary pieces of a .bin (empty for the GGUF); kept for tests.
  std::vector<std::string> vocab_tokens;

  const WhisperHParams &config() const noexcept { return hparams; }

  // The source tensor name for a legacy whisper.cpp name
  // ("encoder.blocks.3.attn.query.weight"): itself for a .bin, the
  // transcribe.cpp canonical name ("enc.blocks.3.attn.q.weight") for a GGUF.
  std::string tensor_name(std::string_view legacy_name) const;

  // Id of a special-token literal ("<|en|>", "<|0.00|>", "<|startofprev|>"),
  // or nullopt. Used to reject special tokens typed into a prompt, the way
  // HF's get_prompt_ids does.
  std::optional<int32_t> special_token_id(std::string_view literal) const;

  // Text <-> ids. decode() drops nothing: callers pass text ids only.
  std::vector<int32_t> encode(std::string_view text) const;
  std::string decode(const std::vector<int32_t> &ids) const;
};

// Loads either layout (dispatch on the file's magic). Throws
// std::runtime_error on anything that is not a well-formed Whisper model.
std::shared_ptr<const WhisperAssets>
load_whisper_assets(const std::filesystem::path &model_path);

// Cheap sniffs used by the loader's can_load().
bool looks_like_whisper_bin(const std::filesystem::path &path);
bool looks_like_whisper_gguf(const std::filesystem::path &path);

// The generation-time suppression list the legacy format omits. English-only
// and multilingual `.bin` files carry different tokenizers, so their
// non-speech ids differ; special-token ids additionally shift with the
// multilingual language count.
std::vector<int32_t> synthesize_bin_suppress_tokens(bool is_multilingual,
                                                    int n_vocab);

} // namespace engine::models::whisper
