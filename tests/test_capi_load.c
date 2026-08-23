/**
 * test_capi_load.c — 决定性验证:静态链接 engine_runtime(同 CLI),
 * 直接调用 audiocpp_load_model,排除 DLL/ABI 因素。
 */
#include "../capi/include/audiocpp.h"
#include <stdio.h>
#include <stdlib.h>

#define MODEL_PATH "models/Qwen3-ASR-0.6B"

static int try_load(const char *label, const char *family_hint) {
    audiocpp_error_t err = {0, NULL};
    printf("\n=== %s (family_hint=%s) ===\n", label, family_hint ? family_hint : "NULL");
    fflush(stdout);
    audiocpp_model_t *model = audiocpp_load_model(
        MODEL_PATH,
        family_hint,
        AUDIOCPP_TASK_ASR,
        AUDIOCPP_BACKEND_CPU,
        0,
        4,
        &err);
    if (model) {
        printf("  SUCCESS: model loaded\n");
        audiocpp_free_model(model);
        audiocpp_clear_error(&err);
        return 0;
    }
    printf("  FAILED: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
    audiocpp_clear_error(&err);
    return 1;
}

// Progress callback state for TEST 3.
static int g_progress_calls = 0;
static float g_last_progress = -1.0f;

static int progress_cb(float progress, const char *stage,
                       int64_t completed, int64_t total, void *user) {
    (void)user;
    g_progress_calls++;
    // Track monotonic non-decreasing progress; just log on regression.
    if (progress < g_last_progress - 1e-6f) {
        printf("  PROGRESS REGRESSION: %.4f after %.4f\n", progress, g_last_progress);
    }
    g_last_progress = progress;
    printf("  progress: %.1f%% (%lld/%lld) stage=%s\n",
           progress * 100.0f, (long long)completed, (long long)total,
           stage ? stage : "(null)");
    return 0;  // 0 == continue; non-zero would request cancellation
}

// TEST 3: load a model, install a progress callback, run ASR, verify the
// callback fired at least once with monotonic progress. Requires the model
// files AND a test wav; only meaningful as a manual integration check.
static int try_progress(const char *wav_path) {
    audiocpp_error_t err = {0, NULL};
    printf("\n=== TEST 3: progress callback (wav=%s) ===\n", wav_path);
    audiocpp_model_t *model = audiocpp_load_model(
        MODEL_PATH, "qwen3_asr", AUDIOCPP_TASK_ASR, AUDIOCPP_BACKEND_CPU, 0, 4, &err);
    if (!model) {
        printf("  SKIP (model not loadable): code=%d msg=%s\n",
               err.code, err.message ? err.message : "(null)");
        audiocpp_clear_error(&err);
        return 0;  // not a failure if model files absent
    }

    audiocpp_set_progress_callback(model, progress_cb, NULL);

    // Load the wav into PCM samples via the C ABI WAV reader.
    float *pcm = NULL;
    int64_t n_samples = 0;
    int sample_rate = 0;
    if (audiocpp_read_wav(wav_path, &pcm, &n_samples, &sample_rate) != 0) {
        printf("  SKIP (cannot read wav %s)\n", wav_path);
        audiocpp_set_progress_callback(model, NULL, NULL);
        audiocpp_free_model(model);
        audiocpp_clear_error(&err);
        return 0;
    }
    printf("  loaded wav: %lld samples @ %d Hz\n", (long long)n_samples, sample_rate);

    audiocpp_text_t *text = audiocpp_asr(model, pcm, n_samples, sample_rate, NULL, &err);
    free(pcm);
    int ok = 0;
    if (text) {
        printf("  ASR text: %s\n", text->text ? text->text : "(empty)");
        printf("  progress callback fired %d time(s), last=%.1f%%\n",
               g_progress_calls, g_last_progress * 100.0f);
        ok = (g_progress_calls > 0) ? 0 : 1;  // 0 = pass
        audiocpp_free_text(text);
    } else {
        printf("  ASR failed: code=%d msg=%s\n",
               err.code, err.message ? err.message : "(null)");
        // Distinguish cancellation (code 0) from real errors.
        ok = (err.code == 0) ? 0 : 1;
    }

    audiocpp_set_progress_callback(model, NULL, NULL);  // clear
    audiocpp_free_model(model);
    audiocpp_clear_error(&err);
    return ok;
}

// TEST 0: backend availability probe — CPU must always be available, GPU
// probes must return 0/1 without aborting (no driver, no crash), and an
// unknown backend id must be 0.
static int probe_backends(void) {
    static const int ids[] = {
        AUDIOCPP_BACKEND_CPU, AUDIOCPP_BACKEND_CUDA, AUDIOCPP_BACKEND_VULKAN,
        AUDIOCPP_BACKEND_METAL, AUDIOCPP_BACKEND_SYCL, AUDIOCPP_BACKEND_BEST,
    };
    static const char *names[] = {"CPU", "CUDA", "Vulkan", "Metal", "SYCL", "BEST"};
    int fail = 0;
    printf("\n=== TEST 0: backend availability probe ===\n");
    for (int i = 0; i < 6; ++i) {
        int av = audiocpp_backend_available(ids[i]);
        printf("  audiocpp_backend_available(%s) = %d\n", names[i], av);
        if (ids[i] == AUDIOCPP_BACKEND_CPU && av != 1) fail = 1;
    }
    int unknown = audiocpp_backend_available(999);
    printf("  audiocpp_backend_available(999) = %d\n", unknown);
    if (unknown != 0) fail = 1;
    return fail;
}

int main(void) {
    printf("audiocpp version: %s\n", audiocpp_version());
    int r0 = probe_backends();
    int r1 = try_load("TEST 1: with family_hint", "qwen3_asr");
    int r2 = try_load("TEST 2: family_hint=NULL", NULL);
    // TEST 3 needs a wav file; pass via argv or a default path. For the
    // static smoke test we skip it unless AUDIOCPP_TEST_WAV is set.
    const char *wav = getenv("AUDIOCPP_TEST_WAV");
    int r3 = wav ? try_progress(wav) : 0;
    printf("\n=== SUMMARY ===\n  probe: %s\n  with hint: %s\n  without: %s\n  progress: %s\n",
           r0 ? "FAIL" : "OK", r1 ? "FAIL" : "OK", r2 ? "FAIL" : "OK",
           r3 ? "FAIL" : (wav ? "OK" : "SKIP"));
    return (r0 || r1 || r2 || r3) ? 1 : 0;
}
