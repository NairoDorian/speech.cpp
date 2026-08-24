// test_family_registry.cpp - unit test for FamilyRegistry v1 in reporting mode (Fusion Roadmap §7.8).

#include "engine/framework/runtime/family_registry.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <string_view>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using namespace engine::runtime;

void test_resolve_canonical_and_aliases() {
    // Canonical lookups
    const FamilyEntry * entry1 = resolve_family("qwen3_asr");
    CHECK(entry1 != nullptr);
    if (entry1) {
        CHECK(entry1->canonical_id == "qwen3_asr");
        CHECK(entry1->primary_task == VoiceTaskKind::Asr);
    }

    // Alias lookup
    const FamilyEntry * entry2 = resolve_family("qwen-asr");
    CHECK(entry2 == entry1);

    // GGUF arch lookup
    const FamilyEntry * entry3 = resolve_family("qwen3_asr");
    CHECK(entry3 == entry1);

    // Parakeet lookup
    const FamilyEntry * entry4 = resolve_family("parakeet");
    CHECK(entry4 != nullptr);
    if (entry4) {
        CHECK(entry4->canonical_id == "parakeet_tdt");
    }

    // Voxtral lookups
    const FamilyEntry * entry_vr = resolve_family("voxtral_realtime");
    CHECK(entry_vr != nullptr);

    const FamilyEntry * entry_vo = resolve_family("voxtral");
    CHECK(entry_vo != nullptr);

    // Unknown lookup
    CHECK(resolve_family("nonexistent_unknown_family_xyz") == nullptr);
    CHECK(resolve_family("") == nullptr);
}

void test_report_inventory() {
    const FamilyEntry * families = all_registered_families_data();
    const size_t count = all_registered_families_count();
    std::cout << "family_registry_unit: reporting mode (Phase 7)\n";
    std::cout << "  registered families: " << count << "\n";

    std::set<std::string_view> seen_canonicals;
    std::set<std::string_view> seen_aliases;

    for (size_t i = 0; i < count; ++i) {
        const auto & f = families[i];
        if (seen_canonicals.count(f.canonical_id) > 0) {
            std::cout << "  [REPORT] duplicate canonical id: " << f.canonical_id << "\n";
        }
        seen_canonicals.insert(f.canonical_id);

        for (size_t j = 0; j < f.aliases_count; ++j) {
            const auto & a = f.aliases_data[j];
            if (seen_aliases.count(a) > 0) {
                std::cout << "  [REPORT] overlapping alias: " << a << " for " << f.canonical_id << "\n";
            }
            seen_aliases.insert(a);
        }
    }
}

}  // namespace

int main() {
    test_resolve_canonical_and_aliases();
    test_report_inventory();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_family_registry: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_family_registry: ALL PASSED (reporting mode)\n");
    return 0;
}
