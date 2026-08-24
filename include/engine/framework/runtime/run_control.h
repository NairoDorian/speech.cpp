#pragma once

#include "engine/framework/runtime/session.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace engine::runtime {

/**
 * RunControl unifies progress reporting and abort/cancellation control across
 * offline and streaming sessions.
 *
 * It provides:
 *   1. A lockless, thread-safe abort flag checked via poll_abort().
 *   2. Progress emission via emit_progress(), which throws ProgressCanceled
 *      if aborted or if the installed ProgressCallback returns false.
 *   3. Safe thread-safe abort requests from any external worker or thread.
 */
class RunControl {
public:
    RunControl() = default;
    explicit RunControl(ProgressCallback callback) : callback_(std::move(callback)) {}

    /** Check if an abort has been requested (lockless, cheap for inner loops). */
    bool poll_abort() const noexcept {
        return aborted_.load(std::memory_order_relaxed);
    }

    /** Request cancellation of the current execution (thread-safe from any thread). */
    void request_abort() noexcept {
        aborted_.store(true, std::memory_order_relaxed);
    }

    /** Clear any pending abort request (used when resetting or starting clean runs). */
    void reset_abort() noexcept {
        aborted_.store(false, std::memory_order_relaxed);
    }

    /** Install or replace the progress observer callback. */
    void set_progress_callback(ProgressCallback callback) {
        callback_ = std::move(callback);
    }

    /** Check if a progress callback is installed. */
    bool has_progress_callback() const noexcept {
        return static_cast<bool>(callback_);
    }

    /**
     * Emit a progress event.
     * Throws ProgressCanceled if poll_abort() is true or if the callback returns false.
     */
    void emit_progress(const ProgressInfo & info) const {
        if (poll_abort()) {
            throw ProgressCanceled("Operation aborted by user request");
        }
        if (callback_) {
            if (!callback_(info)) {
                throw ProgressCanceled("Operation canceled by progress callback");
            }
        }
    }

    /** Helper to construct and emit progress with stage and unit counts. */
    void emit_progress(const char * stage, int64_t completed_units, int64_t total_units) const {
        ProgressInfo info;
        info.stage = stage ? stage : "";
        info.completed_units = completed_units;
        info.total_units = total_units;
        info.progress = (total_units > 0)
            ? (static_cast<float>(completed_units) / static_cast<float>(total_units))
            : 0.0f;
        emit_progress(info);
    }

private:
    std::atomic<bool> aborted_{false};
    ProgressCallback callback_;
};

}  // namespace engine::runtime
