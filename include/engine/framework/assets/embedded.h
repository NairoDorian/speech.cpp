#pragma once

// Accessor for assets embedded into the binary when AUDIOCPP_EMBED_VAD_ASSETS=ON.
// When that option is OFF, has_embedded_asset() always returns false and the
// other functions return nullptr, so callers can compile-gate the embedded path.

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace engine::assets::embedded {

// True if an embedded asset with the given id (e.g. "silero_vad", "marblenet_vad")
// is available in this build.
bool has_embedded_asset(std::string_view id);

// Return a pointer to the embedded asset bytes and its size, or nullptr if no
// such embedded asset exists. The pointer is valid for the program's lifetime
// (it points at static .rodata).
const std::byte * embedded_asset_data(std::string_view id, std::size_t * out_size);

// Materialize the named embedded asset to a file under a per-process cache
// directory and return the path. `dest_filename` controls the on-disk name
// (e.g. "marblenet_vad_config.json") so consumers that look up files by name
// still work. The file is created once and reused for the process lifetime.
// Returns an empty path if no such embedded asset exists.
const std::filesystem::path embedded_asset_file(
    std::string_view id, std::string_view dest_filename);

// Resolve where an internally-created VAD session should load its weights
// from. ASR families (sense_asr, qwen3_asr, vibevoice_asr, ...) chunk long
// audio with an internal silero_vad session, and load_silero_vad_model only
// consults the embedded weights when model_path is EMPTY — a hardcoded disk
// path silently bypasses them. Policy:
//   - an explicit user-configured path always wins;
//   - otherwise prefer the baked-in weights (empty path => embedded branch)
//     when this build embeds them;
//   - otherwise fall back to the on-disk default.
// `embedded_available` is passed in (typically has_embedded_asset("silero_vad"))
// so the policy stays unit-testable without a specific build configuration.
inline std::filesystem::path prefer_embedded_vad_model_path(
    std::optional<std::string> explicit_path,
    std::filesystem::path disk_default,
    bool embedded_available)
{
    if (explicit_path.has_value()) {
        return std::filesystem::path(*explicit_path);
    }
    if (embedded_available) {
        return {};
    }
    return disk_default;
}

}  // namespace engine::assets::embedded
