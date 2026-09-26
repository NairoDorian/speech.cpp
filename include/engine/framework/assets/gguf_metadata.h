#pragma once

// engine/framework/assets/gguf_metadata.h - typed, read-only access to a GGUF
// file's key/value metadata.
//
// Families that read their hyperparameters from GGUF KVs (the transcribe.cpp
// `stt.<family>.*` layout, the audio.cpp `audiocpp.*` layout) each grew a
// private set of read_u32_kv / read_string_kv helpers. This is the shared one:
// it opens the file with no_alloc (no tensor data is read; tensors go through
// TensorSource), and it applies one policy everywhere:
//
//   - find_*()    : absent key -> std::nullopt; present with the wrong type ->
//                   throws (a mistyped key is a converter bug, never a default).
//   - require_*() : like find_*(), but an absent key throws too.
//
// Integer readers accept any GGUF integer width and reject values that do not
// fit the requested type; float readers accept f32 and f64; bool readers accept
// bool and (as older converters wrote) integer 0/1. Error messages name the file
// and the key.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct gguf_context;
struct ggml_context;

namespace engine::assets {

class GgufMetadata {
public:
    // Throws std::runtime_error when the file cannot be opened as GGUF.
    static GgufMetadata open(const std::filesystem::path & path);

    GgufMetadata(GgufMetadata && other) noexcept;
    GgufMetadata & operator=(GgufMetadata && other) noexcept;
    GgufMetadata(const GgufMetadata &) = delete;
    GgufMetadata & operator=(const GgufMetadata &) = delete;
    ~GgufMetadata();

    [[nodiscard]] const gguf_context * context() const noexcept { return gguf_; }
    [[nodiscard]] const std::filesystem::path & path() const noexcept { return path_; }
    [[nodiscard]] bool has(std::string_view key) const;

    [[nodiscard]] std::optional<int64_t> find_int(std::string_view key) const;
    [[nodiscard]] std::optional<int32_t> find_i32(std::string_view key) const;
    [[nodiscard]] std::optional<float> find_float(std::string_view key) const;
    [[nodiscard]] std::optional<bool> find_bool(std::string_view key) const;
    [[nodiscard]] std::optional<std::string> find_string(std::string_view key) const;
    [[nodiscard]] std::optional<std::vector<int32_t>> find_i32_array(std::string_view key) const;
    [[nodiscard]] std::optional<std::vector<std::string>> find_string_array(std::string_view key) const;
    // FLOAT32 (or FLOAT64, narrowed) arrays, e.g. tokenizer.ggml.scores.
    [[nodiscard]] std::optional<std::vector<float>> find_f32_array(std::string_view key) const;

    [[nodiscard]] int32_t require_i32(std::string_view key) const;
    [[nodiscard]] float require_float(std::string_view key) const;
    [[nodiscard]] bool require_bool(std::string_view key) const;
    [[nodiscard]] std::string require_string(std::string_view key) const;

private:
    GgufMetadata(std::filesystem::path path, gguf_context * gguf, ggml_context * meta) noexcept;
    [[noreturn]] void fail_type(std::string_view key, const char * expected) const;
    [[noreturn]] void fail_missing(std::string_view key) const;
    [[nodiscard]] int64_t key_id(std::string_view key) const;

    std::filesystem::path path_;
    gguf_context * gguf_ = nullptr;
    ggml_context * meta_ = nullptr;
};

}  // namespace engine::assets
