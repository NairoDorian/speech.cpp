// engine/framework/assets/gguf_metadata.cpp - see gguf_metadata.h.

#include "engine/framework/assets/gguf_metadata.h"

#include <ggml.h>
#include <gguf.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::assets {

namespace {

bool is_integer_type(gguf_type type) {
    switch (type) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
        return true;
    default:
        return false;
    }
}

// Reads a scalar integer KV of any width. UINT64 values above INT64_MAX are
// reported as out of range by the caller's narrowing check (they wrap negative
// here, which no metadata key legitimately needs).
int64_t read_integer(const gguf_context * gguf, int64_t id) {
    switch (gguf_get_kv_type(gguf, id)) {
    case GGUF_TYPE_UINT8:  return gguf_get_val_u8(gguf, id);
    case GGUF_TYPE_INT8:   return gguf_get_val_i8(gguf, id);
    case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf, id);
    case GGUF_TYPE_INT16:  return gguf_get_val_i16(gguf, id);
    case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf, id);
    case GGUF_TYPE_INT32:  return gguf_get_val_i32(gguf, id);
    case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(gguf, id));
    case GGUF_TYPE_INT64:  return gguf_get_val_i64(gguf, id);
    default:               return 0;  // unreachable: callers check is_integer_type
    }
}

}  // namespace

GgufMetadata GgufMetadata::open(const std::filesystem::path & path) {
    ggml_context * meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta;
    gguf_context * gguf = gguf_init_from_file(path.string().c_str(), params);
    if (gguf == nullptr) {
        if (meta != nullptr) {
            ggml_free(meta);
        }
        throw std::runtime_error("failed to open GGUF metadata: " + path.string());
    }
    return GgufMetadata(path, gguf, meta);
}

GgufMetadata::GgufMetadata(std::filesystem::path path, gguf_context * gguf, ggml_context * meta) noexcept
    : path_(std::move(path)), gguf_(gguf), meta_(meta) {}

GgufMetadata::GgufMetadata(GgufMetadata && other) noexcept
    : path_(std::move(other.path_)), gguf_(std::exchange(other.gguf_, nullptr)),
      meta_(std::exchange(other.meta_, nullptr)) {}

GgufMetadata & GgufMetadata::operator=(GgufMetadata && other) noexcept {
    if (this != &other) {
        this->~GgufMetadata();
        path_ = std::move(other.path_);
        gguf_ = std::exchange(other.gguf_, nullptr);
        meta_ = std::exchange(other.meta_, nullptr);
    }
    return *this;
}

GgufMetadata::~GgufMetadata() {
    if (gguf_ != nullptr) {
        gguf_free(gguf_);
        gguf_ = nullptr;
    }
    if (meta_ != nullptr) {
        ggml_free(meta_);
        meta_ = nullptr;
    }
}

int64_t GgufMetadata::key_id(std::string_view key) const {
    return gguf_find_key(gguf_, std::string(key).c_str());
}

bool GgufMetadata::has(std::string_view key) const { return key_id(key) >= 0; }

void GgufMetadata::fail_type(std::string_view key, const char * expected) const {
    throw std::runtime_error(path_.filename().string() + ": GGUF metadata key " + std::string(key) +
                             " has the wrong type (expected " + expected + ")");
}

void GgufMetadata::fail_missing(std::string_view key) const {
    throw std::runtime_error(path_.filename().string() + ": missing required GGUF metadata key " +
                             std::string(key));
}

std::optional<int64_t> GgufMetadata::find_int(std::string_view key) const {
    const int64_t id = key_id(key);
    if (id < 0) {
        return std::nullopt;
    }
    if (!is_integer_type(gguf_get_kv_type(gguf_, id))) {
        fail_type(key, "integer");
    }
    return read_integer(gguf_, id);
}

std::optional<int32_t> GgufMetadata::find_i32(std::string_view key) const {
    const auto value = find_int(key);
    if (!value.has_value()) {
        return std::nullopt;
    }
    if (*value < std::numeric_limits<int32_t>::min() || *value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(path_.filename().string() + ": GGUF metadata key " + std::string(key) +
                                 " does not fit in int32");
    }
    return static_cast<int32_t>(*value);
}

std::optional<float> GgufMetadata::find_float(std::string_view key) const {
    const int64_t id = key_id(key);
    if (id < 0) {
        return std::nullopt;
    }
    switch (gguf_get_kv_type(gguf_, id)) {
    case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf_, id);
    case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(gguf_, id));
    default:                fail_type(key, "float32/float64");
    }
}

std::optional<bool> GgufMetadata::find_bool(std::string_view key) const {
    const int64_t id = key_id(key);
    if (id < 0) {
        return std::nullopt;
    }
    const gguf_type type = gguf_get_kv_type(gguf_, id);
    if (type == GGUF_TYPE_BOOL) {
        return gguf_get_val_bool(gguf_, id);
    }
    if (is_integer_type(type)) {
        return read_integer(gguf_, id) != 0;
    }
    fail_type(key, "bool");
}

std::optional<std::string> GgufMetadata::find_string(std::string_view key) const {
    const int64_t id = key_id(key);
    if (id < 0) {
        return std::nullopt;
    }
    if (gguf_get_kv_type(gguf_, id) != GGUF_TYPE_STRING) {
        fail_type(key, "string");
    }
    return std::string(gguf_get_val_str(gguf_, id));
}

std::optional<std::vector<int32_t>> GgufMetadata::find_i32_array(std::string_view key) const {
    const int64_t id = key_id(key);
    if (id < 0) {
        return std::nullopt;
    }
    if (gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY) {
        fail_type(key, "int32 array");
    }
    const gguf_type elem = gguf_get_arr_type(gguf_, id);
    const size_t n = gguf_get_arr_n(gguf_, id);
    std::vector<int32_t> out(n);
    const void * data = n > 0 ? gguf_get_arr_data(gguf_, id) : nullptr;
    switch (elem) {
    case GGUF_TYPE_INT32:
        for (size_t i = 0; i < n; ++i) {
            out[i] = static_cast<const int32_t *>(data)[i];
        }
        break;
    case GGUF_TYPE_UINT32:
        for (size_t i = 0; i < n; ++i) {
            const uint32_t v = static_cast<const uint32_t *>(data)[i];
            if (v > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
                fail_type(key, "int32 array (element out of range)");
            }
            out[i] = static_cast<int32_t>(v);
        }
        break;
    default:
        fail_type(key, "int32/uint32 array");
    }
    return out;
}

std::optional<std::vector<std::string>> GgufMetadata::find_string_array(std::string_view key) const {
    const int64_t id = key_id(key);
    if (id < 0) {
        return std::nullopt;
    }
    if (gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf_, id) != GGUF_TYPE_STRING) {
        fail_type(key, "string array");
    }
    const size_t n = gguf_get_arr_n(gguf_, id);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(gguf_, id, i));
    }
    return out;
}

int32_t GgufMetadata::require_i32(std::string_view key) const {
    if (auto v = find_i32(key)) {
        return *v;
    }
    fail_missing(key);
}

float GgufMetadata::require_float(std::string_view key) const {
    if (auto v = find_float(key)) {
        return *v;
    }
    fail_missing(key);
}

bool GgufMetadata::require_bool(std::string_view key) const {
    if (auto v = find_bool(key)) {
        return *v;
    }
    fail_missing(key);
}

std::string GgufMetadata::require_string(std::string_view key) const {
    if (auto v = find_string(key)) {
        return *v;
    }
    fail_missing(key);
}

}  // namespace engine::assets
