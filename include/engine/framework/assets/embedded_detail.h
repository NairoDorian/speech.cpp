#pragma once

// Private header shared between embedded.cpp and the generated
// embedded_*_assets.cpp files (produced by tools/embed_asset.py). Defines the
// asset-entry struct and the per-prefix table accessor in ONE place so the
// struct layout cannot drift between the hand-written and generated TUs.
//
// NOT part of the public API — do not include from outside the embedded-asset
// implementation.

#include <cstddef>

namespace engine::assets::embedded {

struct EmbeddedAssetEntry {
    const char * id;
    const unsigned char * data;
    unsigned long long size;
};

// Each generated TU (one per AUDIOCPP_EMBED_* option) defines one of these,
// named by prefix (e.g. vad_embedded_asset_table, aud_embedded_asset_table).
// Returns a pointer to a static table and writes its entry count to *count.

}  // namespace engine::assets::embedded
