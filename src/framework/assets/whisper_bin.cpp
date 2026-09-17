// whisper_bin.cpp - TensorSource for legacy whisper.cpp `.bin` checkpoints.
//
// Implements the TensorSource interface by parsing the ggml container format
// used by whisper.cpp: a 0x67676d6c magic, 11 int32 hparams, mel filterbank,
// token vocabulary, and a contiguous tensor manifest.  The file is read into
// memory once on construction; individual tensor payloads are served on
// demand from the in-memory buffer.
//
// This consolidates the parallel parser in `src/runtime/transcribe-bin-loader.cpp`
// (arch-level, C ABI) and the ported copy in `src/models/whisper/assets.cpp`
// (engine-level) into a single `TensorSource` implementation.

#include "engine/framework/assets/whisper_bin.h"

#include <ggml.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::assets {

namespace {

constexpr const char * kTag = "whisper-bin";

// ggml magic used by whisper.cpp .bin checkpoints (little-endian "ggml").
constexpr uint32_t k_whisper_bin_magic = 0x67676d6c;

struct TensorEntry {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    // PyTorch/row-major shape (slowest-varying dim first).  The .bin
    // format stores ggml ne-order (fastest first), so we reverse.
    std::vector<int64_t> shape;
    // Byte offset into the file's mmap/buffer where the payload starts.
    size_t byte_offset = 0;
    // Number of bytes occupied by the tensor payload.
    size_t byte_size = 0;
};

class WhisperBinTensorSource final : public TensorSource {
public:
    explicit WhisperBinTensorSource(std::filesystem::path path)
        : path_(std::move(path)) {
        read_and_parse();
    }

    // TensorSource interface ------------------------------------------------

    const std::filesystem::path & source_path() const noexcept override {
        return path_;
    }

    bool has_tensor(std::string_view name) const noexcept override {
        return index_.find(std::string(name)) != index_.end();
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        auto it = index_.find(std::string(name));
        if (it == index_.end()) {
            throw std::runtime_error(
                "tensor not found in whisper .bin: " + std::string(name));
        }
        return TensorMetadata{it->second.name, ggml_type_name(it->second.type), it->second.shape};
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(index_.size());
        for (const auto & kv : index_) {
            out.push_back(TensorMetadata{kv.second.name, ggml_type_name(kv.second.type), kv.second.shape});
        }
        return out;
    }

    void release_storage() const override {
        data_.clear();
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        auto it = index_.find(std::string(name));
        if (it == index_.end()) {
            throw std::runtime_error(
                "tensor not found in whisper .bin: " + std::string(name));
        }
        const auto & entry = it->second;
        const size_t offset = entry.byte_offset;
        const size_t size = entry.byte_size;
        if (offset + size > data_.size()) {
            throw std::runtime_error(
                "tensor data out of bounds in whisper .bin: " + std::string(name));
        }
        std::vector<std::byte> raw(size);
        std::memcpy(raw.data(), data_.data() + offset, size);
        return RawTensorData{
            TensorMetadata{entry.name, ggml_type_name(entry.type), entry.shape},
            std::move(raw),
        };
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        auto it = index_.find(std::string(name));
        if (it == index_.end()) {
            throw std::runtime_error(
                "tensor not found in whisper .bin: " + std::string(name));
        }
        const auto & entry = it->second;
        if (expected_shape.has_value() && entry.shape != *expected_shape) {
            throw std::runtime_error(
                "whisper .bin tensor \"" + std::string(name) + "\" shape mismatch");
        }
        const int64_t n_elem = [&] {
            int64_t n = 1;
            for (int64_t d : entry.shape) n *= d;
            return n;
        }();
        std::vector<float> out(n_elem);
        const auto * data = data_.data() + entry.byte_offset;
        switch (entry.type) {
            case GGML_TYPE_F32:
                std::memcpy(out.data(), data, n_elem * sizeof(float));
                break;
            case GGML_TYPE_F16:
                ggml_fp16_to_fp32_row(
                    reinterpret_cast<const ggml_fp16_t *>(data),
                    out.data(), static_cast<int64_t>(n_elem));
                break;
            case GGML_TYPE_BF16:
                ggml_bf16_to_fp32_row(
                    reinterpret_cast<const ggml_bf16_t *>(data),
                    out.data(), static_cast<int64_t>(n_elem));
                break;
            default: {
                const ggml_type_traits * traits = ggml_get_type_traits(entry.type);
                if (traits == nullptr || traits->to_float == nullptr) {
                    throw std::runtime_error("tensor type is not readable as F32: " + std::string(name));
                }
                traits->to_float(data, out.data(), static_cast<int64_t>(n_elem));
                break;
            }
        }
        return out;
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        throw std::runtime_error(
            "whisper .bin has no scalar tensors: " + std::string(name));
    }

private:
    std::filesystem::path path_;
    // File contents, read on construction.  Released after release_storage().
    mutable std::vector<uint8_t> data_;
    std::map<std::string, TensorEntry> index_;

    // --- parsing -----------------------------------------------------------

    static bool is_known_ggml_type(int32_t ttype) {
        switch (ttype) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q8_K:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
                return true;
            default:
                return false;
        }
    }

    static uint64_t tensor_nbytes(ggml_type type, const int64_t ne[4]) {
        const int64_t n_elem = ne[0] * (ne[1] > 0 ? ne[1] : 1) *
                               (ne[2] > 0 ? ne[2] : 1) *
                               (ne[3] > 0 ? ne[3] : 1);
        const int64_t blck = ggml_blck_size(type);
        if (blck <= 0) return 0;
        return static_cast<uint64_t>(n_elem / blck) * static_cast<uint64_t>(ggml_type_size(type));
    }

    void read_and_parse() {
        // Read entire file into memory.
        std::ifstream fin(path_, std::ios::binary | std::ios::ate);
        if (!fin) {
            throw std::runtime_error(std::string(kTag) + ": failed to open " + path_.string());
        }
        const size_t file_size = static_cast<size_t>(fin.tellg());
        data_.resize(file_size);
        fin.seekg(0);
        fin.read(reinterpret_cast<char *>(data_.data()), static_cast<std::streamsize>(file_size));
        if (!fin) {
            throw std::runtime_error(std::string(kTag) + ": failed to read " + path_.string());
        }

        // --- magic ---
        if (file_size < sizeof(uint32_t)) {
            throw std::runtime_error(std::string(kTag) + ": file too small for magic");
        }
        uint32_t magic = 0;
        std::memcpy(&magic, data_.data(), sizeof(magic));
        if (magic != k_whisper_bin_magic) {
            throw std::runtime_error(std::string(kTag) + ": bad magic (not a whisper .bin)");
        }

        // --- hparams (11 × int32), starting at offset 4 ---
        const int32_t * hp = reinterpret_cast<const int32_t *>(data_.data() + 4);
        // hp[0]=n_vocab, hp[1]=n_audio_ctx, hp[2]=n_audio_state, hp[3]=n_audio_head,
        // hp[4]=n_audio_layer, hp[5]=n_text_ctx, hp[6]=n_text_state, hp[7]=n_text_head,
        // hp[8]=n_text_layer, hp[9]=n_mels, hp[10]=ftype

        // Validate Whisper geometry (same gate as parse_whisper_bin).
        const int32_t n_mels = hp[9];
        const int32_t n_vocab = hp[0];
        if (n_mels != 80 && n_mels != 128) {
            throw std::runtime_error(std::string(kTag) + ": hparams n_mels != 80/128");
        }
        if (n_vocab != 51864 && n_vocab != 51865 && n_vocab != 51866) {
            throw std::runtime_error(std::string(kTag) + ": hparams n_vocab not a known Whisper value");
        }

        size_t pos = 4 + 11 * sizeof(int32_t);

        // --- mel filterbank ---
        const int32_t n_mel_filters = *reinterpret_cast<const int32_t *>(data_.data() + pos);
        pos += sizeof(int32_t);
        const int32_t n_fft_filters = *reinterpret_cast<const int32_t *>(data_.data() + pos);
        pos += sizeof(int32_t);
        pos += static_cast<size_t>(n_mel_filters) * static_cast<size_t>(n_fft_filters) * sizeof(float);

        // --- vocab ---
        int32_t n_vocab_in_file = *reinterpret_cast<const int32_t *>(data_.data() + pos);
        pos += sizeof(int32_t);
        for (int32_t i = 0; i < n_vocab_in_file; ++i) {
            const uint32_t len = *reinterpret_cast<const uint32_t *>(data_.data() + pos);
            pos += sizeof(uint32_t);
            pos += len;
        }

        // --- tensor manifest ---
        while (pos + sizeof(int32_t) <= file_size) {
            // Peek for clean EOF.
            int32_t n_dims = *reinterpret_cast<const int32_t *>(data_.data() + pos);
            if (pos + sizeof(int32_t) == file_size) {
                break;  // clean EOF
            }
            if (pos + sizeof(int32_t) > file_size || n_dims < 1 || n_dims > 4) {
                throw std::runtime_error(std::string(kTag) + ": truncated or invalid tensor header");
            }
            pos += sizeof(int32_t);

            int32_t name_len = *reinterpret_cast<const int32_t *>(data_.data() + pos);
            pos += sizeof(int32_t);
            int32_t ttype = *reinterpret_cast<const int32_t *>(data_.data() + pos);
            pos += sizeof(int32_t);

            if (name_len <= 0 || name_len > 256) {
                throw std::runtime_error(std::string(kTag) + ": invalid tensor name length");
            }
            if (!is_known_ggml_type(ttype)) {
                throw std::runtime_error(std::string(kTag) + ": unknown tensor type");
            }

            int64_t ne[4] = {1, 1, 1, 1};
            for (int32_t i = 0; i < n_dims; ++i) {
                ne[i] = *reinterpret_cast<const int32_t *>(data_.data() + pos);
                pos += sizeof(int32_t);
            }

            std::string name(reinterpret_cast<const char *>(data_.data() + pos), name_len);
            pos += name_len;

            const ggml_type type = static_cast<ggml_type>(ttype);
            const uint64_t nbytes = tensor_nbytes(type, ne);
            if (nbytes == 0) {
                throw std::runtime_error(std::string(kTag) + ": tensor has zero nbytes");
            }

            // Reverse ggml ne-order (fastest first) to row-major (slowest first).
            std::vector<int64_t> shape;
            shape.reserve(n_dims);
            for (int32_t i = n_dims - 1; i >= 0; --i) {
                shape.push_back(ne[i]);
            }

            TensorEntry entry;
            entry.name = std::move(name);
            entry.type = type;
            entry.shape = std::move(shape);
            entry.byte_offset = pos;
            entry.byte_size = static_cast<size_t>(nbytes);

            index_.try_emplace(entry.name, std::move(entry));
            pos += nbytes;
        }

        if (index_.empty()) {
            throw std::runtime_error(std::string(kTag) + ": file declares no tensors");
        }
    }
};

}  // namespace

std::shared_ptr<const TensorSource> open_whisper_bin_tensor_source(const std::filesystem::path & path) {
    return std::make_shared<WhisperBinTensorSource>(path);
}

bool looks_like_ggml_whisper_bin(const std::filesystem::path & path) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) return false;
    const auto file_size = fin.tellg();
    if (file_size < 4) return false;
    fin.seekg(0);
    uint32_t magic = 0;
    fin.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (!fin || magic != k_whisper_bin_magic) return false;

    // Validate Whisper geometry in the hparams.
    char hp_buf[4 * 11];
    fin.read(hp_buf, 11 * sizeof(int32_t));
    if (!fin || static_cast<int32_t>(fin.gcount()) != 11 * sizeof(int32_t)) return false;
    const int32_t n_mels = *reinterpret_cast<const int32_t *>(hp_buf + 9 * sizeof(int32_t));
    const int32_t n_vocab = *reinterpret_cast<const int32_t *>(hp_buf);
    if (n_mels != 80 && n_mels != 128) return false;
    return n_vocab == 51864 || n_vocab == 51865 || n_vocab == 51866;
}

}  // namespace engine::assets
