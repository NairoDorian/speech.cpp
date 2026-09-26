#pragma once

// engine/models/gigaam/assets.h - host-side resources of the native engine
// GigaAM package, loaded from the transcribe.cpp GGUF layout
// (general.architecture = "gigaam"; published per variant under
// huggingface.co/handy-computer/gigaam-v3-{e2e-rnnt,e2e-ctc,rnnt,ctc}-gguf).
//
// Every KV and tensor name is the one src/runtime/arch/gigaam/weights.cpp
// (read_gigaam_hparams / build_gigaam_weights) reads, validated the same way:
// same required keys, same invariants, same dtype allowlists and shapes.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/text/tokenizer_hub.h"
#include "engine/models/gigaam/decoding.h"
#include "engine/models/gigaam/graphs_internal.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::gigaam {

// How token ids become text - the arch transcribe::Tokenizer's decode modes
// for the `tokenizer.ggml.model` values it accepts.
enum class GigaamTextDecode {
  SentencePiece, // "bpe" / "unigram": U+2581 -> ' ', <0xHH> byte fallback (e2e variants)
  Charwise,      // "char": pieces are literal characters, concatenated (non-e2e)
  ByteLevel,     // "gpt2": GPT-2 byte-level (accepted by the arch; not published)
};

struct GigaamAssets {
  GigaamHParams hparams;
  std::string variant; // stt.variant, e.g. "gigaam-v3-e2e-rnnt"
  std::filesystem::path model_path;

  // Encoder weights, streamed into each session's BackendWeightStore.
  std::shared_ptr<const assets::TensorSource> source;

  // Vocabulary (tokenizer.ggml.*). TokenizerHub holds the pieces; the text
  // decode is done here because the hub decodes a "bpe" GGUF as GPT-2
  // byte-level and has no <0xHH> byte fallback, while GigaAM's "bpe" is
  // SentencePiece (see decode()).
  text::TokenizerPtr vocab;
  std::string tokenizer_model; // tokenizer.ggml.model
  GigaamTextDecode text_decode = GigaamTextDecode::SentencePiece;
  // Charwise vocabularies only: the arch's transcribe_tokenize encoder (a
  // raw-bytes vocabulary). Null for SentencePiece, where the arch has none.
  text::TokenizerPtr text_encoder;

  // general.languages (["ru"]); empty when the file declares none.
  std::vector<std::string> languages;
  bool lang_detect = false; // stt.capability.lang_detect

  // Frontend buffers baked by the converter: row-major [n_mels, n_fft/2+1]
  // HTK filterbank and the [win_length] periodic Hann window.
  std::vector<float> mel_filterbank;
  std::vector<float> window;

  // Predictor + joint (RNN-T) or CTC head, f32 on the host.
  GigaamHostDecoder host_decoder;

  const GigaamHParams &config() const noexcept { return hparams; }

  // Token ids -> text exactly as the arch's Tokenizer::decode does
  // (out-of-range ids skipped). No trimming here.
  std::string decode(const std::vector<int32_t> &ids) const;
  // Raw vocabulary piece ("" when out of range) - the arch's token row text.
  std::string piece(int32_t id) const;
};

// Loads and validates a transcribe.cpp GigaAM GGUF. Throws std::runtime_error
// (message names the file and the key / tensor) on anything the arch would
// have rejected, plus the frontend geometry the MelExtractor configuration
// requires (n_fft == win_length, n_fft / 2 a multiple of hop).
std::shared_ptr<const GigaamAssets> load_gigaam_assets(const std::filesystem::path &model_path);

// Cheap sniff for the loader's can_load(): GGUF magic and
// general.architecture == "gigaam".
bool looks_like_gigaam_gguf(const std::filesystem::path &path);

} // namespace engine::models::gigaam
