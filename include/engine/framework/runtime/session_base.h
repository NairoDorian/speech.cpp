#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/artifacts.h"
#include "engine/framework/runtime/cache.h"
#include "engine/framework/runtime/graph_executor.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/workspace.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace engine::runtime {

// Virtually inherits IVoiceTaskSession so the concrete model classes, which
// also derive from IOfflineVoiceTaskSession / IStreamingVoiceTaskSession (both
// virtually deriving from IVoiceTaskSession), get a single shared
// IVoiceTaskSession subobject. This lets set_progress_callback below override
// the virtual declared on IVoiceTaskSession unambiguously through the diamond.
class RuntimeSessionBase : public virtual IVoiceTaskSession {
public:
    explicit RuntimeSessionBase(const SessionOptions & options);
    ~RuntimeSessionBase() override = default;

    // Overrides IVoiceTaskSession's no-op default. Concrete sessions inherit
    // this single implementation; emit_progress() below reads the stored member.
    void set_progress_callback(ProgressCallback cb) override;

protected:
    engine::core::ExecutionContext & execution_context() noexcept;
    const engine::core::ExecutionContext & execution_context() const noexcept;
    ArtifactStore & artifacts() noexcept;
    const ArtifactStore & artifacts() const noexcept;
    RuntimeCache & cache() noexcept;
    const RuntimeCache & cache() const noexcept;
    RuntimeWorkspace & workspace() noexcept;
    const RuntimeWorkspace & workspace() const noexcept;
    GraphExecutor & graph_executor() noexcept;
    const GraphExecutor & graph_executor() const noexcept;
    void mark_prepared() noexcept;
    void require_prepared(std::string_view operation) const;
    void trace(engine::debug::LogLevel level, std::string_view category, std::string_view message) const;
    engine::debug::ScopeTimer profile(engine::debug::LogLevel level, std::string_view category, std::string_view name) const;
    const SessionOptions & options() const noexcept;

    // Emit a progress update to the installed callback (if any). Throws
    // ProgressCanceled when the callback returns false, so callers in run()
    // loops can simply call this at chunk boundaries — a cancel request
    // unwinds run() automatically. No-op when no callback is set.
    void emit_progress(const char * stage, int64_t completed_units, int64_t total_units);

private:
    SessionOptions options_;
    engine::core::ExecutionContext execution_context_;
    ArtifactStore artifacts_;
    RuntimeCache cache_;
    RuntimeWorkspace workspace_;
    GraphExecutor graph_executor_;
    bool prepared_ = false;
    ProgressCallback progress_callback_;
};

}  // namespace engine::runtime
