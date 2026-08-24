#pragma once

#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace engine::runtime {

struct FamilyEntry {
    std::string_view canonical_id;
    const std::string_view * aliases_data = nullptr;
    size_t aliases_count = 0;
    const std::string_view * gguf_archs_data = nullptr;
    size_t gguf_archs_count = 0;
    std::string_view spec_file;
    VoiceTaskKind primary_task = VoiceTaskKind::Tts;
};

// Look up a family entry by canonical id, alias, or GGUF arch name.
// Returns nullptr if not found.
const FamilyEntry * resolve_family(std::string_view any_spelling);

// Returns all registered families in the static catalog.
const FamilyEntry * all_registered_families_data();
size_t all_registered_families_count();

}  // namespace engine::runtime
