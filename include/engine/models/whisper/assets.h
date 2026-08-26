#pragma once

#include "engine/models/whisper/graphs_internal.h"

#include "ggml.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::whisper {

// One tensor in the legacy whisper.cpp `.bin`, located but not read. Payload
// bytes are streamed per tensor at weight-upload time so a large-v3 `.bin`
// (~1.5 GB) is never held twice in host memory.
struct WhisperBinTensor {
  std::string name;             // legacy whisper.cpp name
  ggml_type type = GGML_TYPE_F32;
  int32_t n_dims = 0;
  int64_t ne[4] = {1, 1, 1, 1}; // ggml order, as stored in the file
  uint64_t offset = 0;          // from start of file
  uint64_t nbytes = 0;          // == ggml_nbytes(type, ne)
};

// Host-side model resources shared across sessions.
//
// W2a loads the legacy whisper.cpp `.bin` rather than a GGUF, because
// model_specs/whisper.json is catalog-only: its 16 packages point at
// Whisper-*-GGUF paths that do not exist in audio-cpp/audio.cpp-gguf, so no
// GGUF is obtainable for this family. The `.bin` is the canonical
// distribution (ggerganov/whisper.cpp) and is what the pinned gate model uses.
struct WhisperAssets {
  WhisperHParams hparams;
  std::string variant; // "tiny.en", "base", ...
  std::filesystem::path model_path;

  // Decode-only vocabulary: whisper's `.bin` stores raw byte strings, so
  // detokenization is byte concatenation over the generated ids.
  std::vector<std::string> vocab_tokens;

  // Mel filterbank shipped inside the `.bin` (n_mel x n_fft_bins, row-major).
  int32_t n_mel_filters = 0;
  int32_t n_fft_filters = 0;
  std::vector<float> mel_filterbank;

  std::vector<WhisperBinTensor> tensors;

  const WhisperHParams &config() const noexcept { return hparams; }
};

// Parse and validate a legacy whisper.cpp `.bin`. Throws std::runtime_error
// on a wrong magic, a non-Whisper geometry, or any structural inconsistency.
std::shared_ptr<const WhisperAssets>
load_whisper_assets(const std::filesystem::path &model_path);

// True when the file at `path` starts with the whisper.cpp `ggml` magic.
// Cheap sniff used by the loader's can_load().
bool looks_like_whisper_bin(const std::filesystem::path &path);

// The generation-time suppression list the legacy format omits. English-only
// and multilingual `.bin` files carry different tokenizers, so their
// non-speech ids differ; special-token ids additionally shift with the
// multilingual language count.
std::vector<int32_t> synthesize_bin_suppress_tokens(bool is_multilingual,
                                                    int n_vocab);

} // namespace engine::models::whisper
