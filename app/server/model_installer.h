#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace minitts::server {

// Runs model package work outside the request thread. Normal package downloads,
// inventory, version checks, cancellation, cleanup, and uninstall are native C++.
// Requests with explicit converter inputs retain the deprecated Python helper
// until those checkpoint-conversion workflows migrate to native tools.
class ModelInstaller {
public:
    ModelInstaller(std::filesystem::path repository_root, std::filesystem::path models_root);

    // Cancels any active job (writes its cancellation marker) and joins every
    // worker thread before returning, so no helper subprocess outlives the
    // installer. Detached workers used to race process exit — on Windows a
    // worker terminated inside CreateProcess leaves a permanently suspended
    // child cmd.exe pinning the inherited stdio handles (observed as a CTest
    // hang) — and raced the job directory cleanup against their redirections.
    ~ModelInstaller();

    ModelInstaller(const ModelInstaller &) = delete;
    ModelInstaller & operator=(const ModelInstaller &) = delete;

    std::string start(
        const std::string & package_id,
        const std::string & source_file,
        const std::string & output_file,
        const std::string & source_directory,
        const std::string & variant,
        bool overwrite);
    std::string status(const std::string & package_id = {}) const;
    std::string package_sizes();
    std::string stop(const std::string & package_id);
    std::string clean_partial(const std::string & package_id);
    std::string remove(const std::string & package_id);
    bool has_active_jobs() const;

private:
    struct State;

    // Runs body on a tracked worker thread; finished workers are reaped on
    // every launch so the tracking list stays small on long-lived servers.
    void launch_worker(std::function<void()> body);

    std::shared_ptr<State> state_;
};

}  // namespace minitts::server
