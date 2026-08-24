#include "engine/framework/core/shared_weight_registry.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using namespace engine::core;

void test_scoped_weight_share_key() {
    CHECK(current_weight_share_key().empty());

    {
        ScopedWeightShareKey scope1("model_a");
        CHECK(current_weight_share_key() == "model_a");

        {
            ScopedWeightShareKey scope2("model_b");
            CHECK(current_weight_share_key() == "model_b");
        }

        CHECK(current_weight_share_key() == "model_a");
    }

    CHECK(current_weight_share_key().empty());
}

void test_shared_weight_fingerprint() {
    std::vector<SharedWeightTensorMeta> metas1 = {
        {"encoder.layer.0.weight", {512, 512}, GGML_TYPE_F32, 0},
        {"encoder.layer.0.bias", {512}, GGML_TYPE_F32, 512 * 512 * 4},
    };
    std::string fp1 = shared_weight_fingerprint(metas1);
    CHECK(!fp1.empty());

    std::vector<SharedWeightTensorMeta> metas2 = metas1;
    std::string fp2 = shared_weight_fingerprint(metas2);
    CHECK(fp1 == fp2);

    std::vector<SharedWeightTensorMeta> metas3 = {
        {"encoder.layer.0.weight", {512, 256}, GGML_TYPE_F32, 0},
    };
    std::string fp3 = shared_weight_fingerprint(metas3);
    CHECK(fp1 != fp3);
}

void test_registry_hit_miss_and_cleanup() {
    auto & registry = SharedWeightRegistry::instance();
    registry.reset_counters();
    CHECK(registry.hit_count() == 0);
    CHECK(registry.miss_count() == 0);
    CHECK(registry.conflict_count() == 0);

    const std::string key = "test_model_share_key_v1";
    const std::string fp = "test_fingerprint_abc123";

    // Allocate a buffer on CPU backend to simulate a real SharedWeightEntry
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    CHECK(cpu_backend != nullptr);

    // Context to create a buffer
    ggml_init_params p{1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    (void)t;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    CHECK(buf != nullptr);

    std::shared_ptr<SharedWeightEntry> entry1;
    {
        std::vector<SharedWeightTensorMeta> metas = {{"t", {256}, GGML_TYPE_F32, 0}};
        auto [acquired, created] = registry.acquire(key, fp, [&]() {
            return std::make_shared<SharedWeightEntry>(buf, fp, metas);
        });
        CHECK(created == true);
        CHECK(acquired != nullptr);
        CHECK(registry.miss_count() == 1);
        CHECK(registry.hit_count() == 0);

        entry1 = acquired;

        // Second acquire on the same key and fingerprint: HIT
        auto [acquired2, created2] = registry.acquire(key, fp, [&]() {
            // Should not be called
            return nullptr;
        });
        CHECK(created2 == false);
        CHECK(acquired2 == entry1);
        CHECK(registry.hit_count() == 1);
        CHECK(registry.miss_count() == 1);

        // Third acquire on same key with DIFFERENT fingerprint: CONFLICT
        const std::string fp_different = "different_fp_xyz789";
        auto [acquired_conf, created_conf] = registry.acquire(key, fp_different, [&]() {
            return nullptr;
        });
        CHECK(created_conf == false);
        CHECK(acquired_conf == nullptr);
        CHECK(registry.conflict_count() == 1);
    }

    // Release entry1: the weak_ptr in registry should become stale
    entry1.reset();

    // After reset, acquire on same key should be treated as a MISS (not stale hit)
    // Create new buffer
    ggml_context * ctx2 = ggml_init(p);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 256);
    (void)t2;
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, cpu_backend);
    std::vector<SharedWeightTensorMeta> metas2 = {{"t", {256}, GGML_TYPE_F32, 0}};
    auto [acquired3, created3] = registry.acquire(key, fp, [&]() {
        return std::make_shared<SharedWeightEntry>(buf2, fp, metas2);
    });
    CHECK(created3 == true);
    CHECK(acquired3 != nullptr);
    CHECK(registry.miss_count() == 2);

    acquired3.reset();
    ggml_free(ctx);
    ggml_free(ctx2);
    ggml_backend_free(cpu_backend);
}

}  // namespace

int main() {
    test_scoped_weight_share_key();
    test_shared_weight_fingerprint();
    test_registry_hit_miss_and_cleanup();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_shared_weight_vram: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_shared_weight_vram: ALL PASSED\n");
    return 0;
}
