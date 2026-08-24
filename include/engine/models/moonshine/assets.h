#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/text/tokenizer_hub.h"

#include "engine/models/moonshine/graphs_internal.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace engine::models::moonshine {

// Host-side model resources shared across sessions: hparams parsed from GGUF
// metadata, the hub tokenizer, and the (lazy) tensor source. Backend weight
// upload happens per session via core::BackendWeightStore.
struct MoonshineAssets {
  MoonshineHParams hparams;
  text::TokenizerPtr tokenizer;
  std::shared_ptr<const assets::TensorSource> source;
  std::string variant;

  const MoonshineHParams &config() const noexcept { return hparams; }
};

// Loads and validates a Moonshine GGUF package (metadata + tensor catalog).
// Throws std::runtime_error on any structural mismatch.
std::shared_ptr<const MoonshineAssets>
load_moonshine_assets(const std::filesystem::path &model_path);

} // namespace engine::models::moonshine
