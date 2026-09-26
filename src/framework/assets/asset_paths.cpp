#include "engine/framework/assets/asset_paths.h"

#include <cstdlib>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace engine::assets {

namespace {

bool exists_quietly(const std::filesystem::path & path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::exists(path, ec);
}

// Directory of the module (DLL / shared object / executable) this function
// was linked into, not of the host process: a C-ABI host's executable can
// live anywhere, while the engine library sits with its bundled assets.
std::filesystem::path engine_module_directory() {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&engine_module_directory), &module)) {
        return {};
    }
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        return {};
    }
    return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&engine_module_directory), &info) != 0 && info.dli_fname != nullptr) {
        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(info.dli_fname, ec);
        return (ec ? std::filesystem::path(info.dli_fname) : canonical).parent_path();
    }
    return {};
#endif
}

}  // namespace

std::filesystem::path resolve_bundled_asset(const std::filesystem::path & path) {
    if (path.empty() || exists_quietly(path) || path.is_absolute()) {
        return path;
    }
    std::vector<std::filesystem::path> roots;
    for (const char * variable : {"SPEECHCPP_ASSET_ROOT", "AUDIOCPP_ASSET_ROOT"}) {
        if (const char * value = std::getenv(variable); value != nullptr && value[0] != '\0') {
            roots.emplace_back(value);
        }
    }
    std::filesystem::path module_dir = engine_module_directory();
    for (int level = 0; level < 4 && !module_dir.empty(); ++level) {
        roots.push_back(module_dir);
        const auto parent = module_dir.parent_path();
        if (parent == module_dir) {
            break;
        }
        module_dir = parent;
    }
#ifdef SPEECHCPP_SOURCE_ROOT
    roots.emplace_back(SPEECHCPP_SOURCE_ROOT);
#endif
    for (const auto & root : roots) {
        const auto candidate = root / path;
        if (exists_quietly(candidate)) {
            return candidate;
        }
    }
    return path;
}

}  // namespace engine::assets
