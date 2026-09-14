#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::assets {

// Opens a legacy whisper.cpp `.bin` checkpoint (the ggml container format
// with magic 0x67676d6c) as a tensor source. Only files whose hparams pass
// the Whisper geometry gate (n_mels ∈ {80, 128}, n_vocab ∈
// {51864, 51865, 51866}) are accepted; other ggml-magic files (e.g.
// Silero VAD) are rejected so the caller surfaces UNSUPPORTED_ARCH rather
// than a parse error.
//
// File layout:
//   magic (uint32)
//   hparams (11 × int32)
//   mel  (int32 n_mel, int32 n_fft, float[n_mel*n_fft])
//   vocab (int32 count, then per-token: uint32 len + bytes)
//   tensors (repeat: int32 n_dims, int32 name_len, int32 ttype,
//            n_dims × int32 dims, name_len bytes name, payload)
std::shared_ptr<const TensorSource> open_whisper_bin_tensor_source(const std::filesystem::path & path);
// Peeks at the first 4 bytes to see if the file has the ggml magic
// (0x67676d6c) that the whisper .bin format uses.  Does NOT read the
// whole file; used by open_tensor_source to route .bin files.
bool looks_like_ggml_whisper_bin(const std::filesystem::path & path);

}  // namespace engine::assets
