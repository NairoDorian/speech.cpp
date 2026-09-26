#pragma once

#include <filesystem>

namespace engine::assets {

// Resolve a path to a bundled asset (for example the default Silero VAD,
// "assets/framework/models/silero_vad") without depending on the process's
// working directory. Returns the first candidate that exists:
//
//   1. `path` as given (absolute, or relative to the working directory - the
//      historical behaviour, so explicit paths never change meaning);
//   2. $SPEECHCPP_ASSET_ROOT / path (also accepts $AUDIOCPP_ASSET_ROOT);
//   3. the directory of the module that contains the engine (the shared
//      library or executable) and up to three of its parents / path - so
//      build-*/bin/<exe> finds <repo>/assets and an install keeps assets
//      beside the binaries;
//   4. the source tree recorded at build time (SPEECHCPP_SOURCE_ROOT) / path,
//      for developer builds run from anywhere.
//
// When none exists, `path` is returned unchanged so the caller's error names
// the path the user can fix.
std::filesystem::path resolve_bundled_asset(const std::filesystem::path & path);

}  // namespace engine::assets
