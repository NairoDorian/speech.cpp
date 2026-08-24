#pragma once

// SharedWeightRegistry — process-wide sharing of GPU weight buffers.
//
// Without sharing, every session uploads its own copy of a model's weights
// (measured: +2.7 GB VRAM per miotts session). With sharing, the device
// memory for the weights is allocated and filled once; later sessions bind
// their weight tensors to the same buffer by offset and never copy a byte,
// dropping the per-session VRAM delta to just the graph arenas.
//
// Lifecycle: reference counted. Every bound BackendWeightStore holds a
// shared_ptr<SharedWeightEntry>; when the last store dies (model unload) the
// entry dies and the backend buffer is freed, so VRAM really returns to the
// driver. The registry itself only stores weak_ptr and never extends a
// buffer's lifetime.
//
// Safety: a reuse is allowed only when BOTH the share key (model path +
// backend + device, set by ScopedWeightShareKey) AND the tensor fingerprint
// (per-tensor name/dims/type sequence, which also pins the buffer layout)
// match. A fingerprint mismatch (different model, or a different storage-type
// configuration for the same model) makes acquire() return null and the
// caller falls back to an independent allocation — sharing is an optimization
// that must never change results.
//
// Thread safety: acquire() serializes lookup + create under one mutex; the
// first (possibly slow) upload runs inside the lock, later concurrent
// callers just bind.

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::core {

// One weight tensor's identity (fingerprint part) and its byte offset inside
// the shared buffer (assigned by the provider after allocation).
struct SharedWeightTensorMeta {
    std::string name;
    std::vector<int64_t> dims;
    ggml_type type = GGML_TYPE_F32;
    size_t offset = 0;
};

// Owns one shared weight buffer. The provider (first session) allocates and
// fills it and records the offsets; consumers only read. The buffer is freed
// when the last holder of the shared_ptr goes away.
class SharedWeightEntry {
public:
    SharedWeightEntry(
        ggml_backend_buffer_t buffer,
        std::string fingerprint,
        std::vector<SharedWeightTensorMeta> metas)
        : buffer_(buffer),
          base_(buffer != nullptr ? ggml_backend_buffer_get_base(buffer) : nullptr),
          fingerprint_(std::move(fingerprint)),
          metas_(std::move(metas)) {
        if (buffer_ == nullptr || base_ == nullptr) {
            throw std::runtime_error("SharedWeightEntry requires a valid backend buffer");
        }
    }

    ~SharedWeightEntry() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    SharedWeightEntry(const SharedWeightEntry &) = delete;
    SharedWeightEntry & operator=(const SharedWeightEntry &) = delete;

    ggml_backend_buffer_t buffer() const noexcept { return buffer_; }
    const void * base() const noexcept { return base_; }
    const std::string & fingerprint() const noexcept { return fingerprint_; }
    const std::vector<SharedWeightTensorMeta> & metas() const noexcept { return metas_; }

private:
    ggml_backend_buffer_t buffer_ = nullptr;
    const void * base_ = nullptr;
    std::string fingerprint_;
    std::vector<SharedWeightTensorMeta> metas_;
};

inline std::vector<int64_t> tensor_dims_for_fingerprint(const ggml_tensor * tensor) {
    std::vector<int64_t> dims;
    const int ndims = ggml_n_dims(tensor);
    dims.reserve(static_cast<size_t>(ndims));
    for (int i = 0; i < ndims; ++i) {
        dims.push_back(tensor->ne[i]);
    }
    return dims;
}

// Deterministic serialization of the tensor identity sequence. Two weight
// sets share a buffer only when their fingerprints are byte-identical, which
// also guarantees identical allocation order inside the buffer.
inline std::string shared_weight_fingerprint(const std::vector<SharedWeightTensorMeta> & metas) {
    std::string fp;
    fp.reserve(metas.size() * 64);
    for (const auto & meta : metas) {
        fp += meta.name;
        fp += ':';
        for (size_t i = 0; i < meta.dims.size(); ++i) {
            if (i != 0) {
                fp += 'x';
            }
            fp += std::to_string(meta.dims[i]);
        }
        fp += ':';
        fp += std::to_string(static_cast<int>(meta.type));
        fp += '|';
    }
    return fp;
}

// Registry of live shared weight buffers, keyed by share_key. Stores weak
// pointers only; entries live as long as their users.
class SharedWeightRegistry {
public:
    static SharedWeightRegistry & instance() {
        static SharedWeightRegistry registry;
        return registry;
    }

    size_t hit_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return hit_count_;
    }
    size_t miss_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return miss_count_;
    }
    size_t conflict_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return conflict_count_;
    }
    void reset_counters() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        hit_count_ = 0;
        miss_count_ = 0;
        conflict_count_ = 0;
    }

    // Look up (key, fingerprint). On hit returns the entry with created=false.
    // On miss calls create() inside the lock, registers the entry and returns
    // it with created=true. On a fingerprint conflict for the same key
    // (different model / different storage config) returns {nullptr, false} so
    // the caller can fall back to an independent allocation instead of
    // reusing — or worse, overwriting — a foreign buffer.
    std::pair<std::shared_ptr<SharedWeightEntry>, bool> acquire(
        const std::string & key,
        const std::string & fingerprint,
        const std::function<std::shared_ptr<SharedWeightEntry>()> & create) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(key);
        if (it != entries_.end()) {
            auto entry = it->second.lock();
            if (!entry) {
                entries_.erase(it);  // stale weak_ptr; treat as miss
            } else if (entry->fingerprint() == fingerprint) {
                ++hit_count_;
                return {std::move(entry), false};
            } else {
                ++conflict_count_;
                return {nullptr, false};  // fingerprint conflict
            }
        }
        auto entry = create();
        entries_[key] = entry;
        ++miss_count_;
        return {std::move(entry), true};
    }

private:
    SharedWeightRegistry() = default;
    SharedWeightRegistry(const SharedWeightRegistry &) = delete;
    SharedWeightRegistry & operator=(const SharedWeightRegistry &) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::weak_ptr<SharedWeightEntry>> entries_;
    size_t hit_count_ = 0;
    size_t miss_count_ = 0;
    size_t conflict_count_ = 0;
};

// Current thread's share key; empty means sharing is disabled on this thread.
inline std::string & current_weight_share_key() {
    static thread_local std::string key;
    return key;
}

// Thread-local share key scope. A caller that loads a model and immediately
// creates a session (server / CLI / workflow / CAPI) wraps both calls in one
// of these; every BackendWeightStore constructed in between picks up the key
// (plus its own backend/device discriminator) and joins the sharing.
// Nested scopes save and restore the previous key. The default empty key
// disables sharing, so stores constructed outside a scope behave exactly as
// before.
class ScopedWeightShareKey {
public:
    explicit ScopedWeightShareKey(std::string key) : saved_(current_weight_share_key()) {
        current_weight_share_key() = std::move(key);
    }

    ~ScopedWeightShareKey() {
        current_weight_share_key() = std::move(saved_);
    }

    ScopedWeightShareKey(const ScopedWeightShareKey &) = delete;
    ScopedWeightShareKey & operator=(const ScopedWeightShareKey &) = delete;

private:
    std::string saved_;
};

}  // namespace engine::core
