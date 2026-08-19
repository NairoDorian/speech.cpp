#pragma once

// Shared weight-source resolution for the MOSS-Audio-Tokenizer-v2 parity
// harnesses (codec_dequant_parity / codec_encode_parity / codec_decode_parity).
//
// These harnesses used to pass a HuggingFace snapshot DIRECTORY straight into
// the MossAudioTokenizer* constructors. Those now take an
// engine::assets::TensorSource, and engine::assets::open_tensor_source() opens
// a weights FILE (.safetensors / .safetensors.index.json / .gguf), so the
// harnesses stopped compiling and took `-DENGINE_BUILD_TESTS=ON` down with
// them. Accept either a file or a directory here and resolve the latter.

#include "engine/framework/assets/tensor_source.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace moss_parity {

// Resolve `path` to a tensor source. A file is opened directly; a directory is
// searched for a sharded index first, then a single-file checkpoint.
inline std::shared_ptr<const engine::assets::TensorSource> open_codec_weights(
    const std::filesystem::path & path) {
    namespace fs = std::filesystem;

    if (!fs::exists(path)) {
        throw std::runtime_error("codec weights path does not exist: " + path.string());
    }
    if (!fs::is_directory(path)) {
        return engine::assets::open_tensor_source(path);
    }

    std::vector<fs::path> indexes;
    std::vector<fs::path> singles;
    for (const auto & entry : fs::directory_iterator(path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.size() > 21 && name.rfind(".safetensors.index.json") != std::string::npos) {
            indexes.push_back(entry.path());
        } else {
            const std::string ext = entry.path().extension().string();
            if (ext == ".safetensors" || ext == ".gguf") {
                singles.push_back(entry.path());
            }
        }
    }
    // Deterministic pick regardless of directory iteration order.
    std::sort(indexes.begin(), indexes.end());
    std::sort(singles.begin(), singles.end());

    if (!indexes.empty()) {
        return engine::assets::open_tensor_source(indexes.front());
    }
    if (!singles.empty()) {
        return engine::assets::open_tensor_source(singles.front());
    }
    throw std::runtime_error(
        "no .safetensors / .gguf weights found in codec directory: " + path.string());
}

}  // namespace moss_parity
