/* capi_tts_test.c — CAPI TTS comparison driver.
 *
 * Loads a model through the audiocpp C ABI, runs one TTS synthesis, prints
 * timing + PCM statistics, writes a 16-bit WAV (via audiocpp_write_wav) and a
 * raw f32 PCM dump (<out>.f32) for offline analysis.
 *
 * Usage:
 *   capi_tts_test.exe --model <path> [--family <hint>] --text <text> --out <wav>
 *                     [--backend <0=cpu|1=cuda>] [--threads <n>]
 *                     [--task <AUDIOCPP_TASK_*>] [--options <json>]
 *
 * All output lines are ASCII. Exit code 0 = success.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "audiocpp.h"

typedef struct {
    const char *model_path;
    const char *family_hint;
    const char *text;
    const char *out_wav;
    const char *options_json;
    const char *load_options_json;  /* session options for load_model_ex */
    int backend;
    int n_threads;
    int task;
} Args;

static const char *arg_value(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return NULL;
}

static int arg_flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

static double now_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static void print_stats(const float *samples, int64_t n) {
    double sum = 0.0, sum_sq = 0.0;
    float vmin = 1e30f, vmax = -1e30f;
    int64_t nan_count = 0, inf_count = 0, clip_count = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = samples[i];
        if (isnan(v)) { nan_count++; continue; }
        if (isinf(v)) { inf_count++; continue; }
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        sum += v;
        sum_sq += (double)v * (double)v;
        if (v > 1.0f || v < -1.0f) clip_count++;
    }
    int64_t valid = n - nan_count - inf_count;
    double mean = valid > 0 ? sum / (double)valid : 0.0;
    double rms = valid > 0 ? sqrt(sum_sq / (double)valid) : 0.0;
    printf("pcm_min=%.8f\n", valid > 0 ? (double)vmin : 0.0);
    printf("pcm_max=%.8f\n", valid > 0 ? (double)vmax : 0.0);
    printf("pcm_mean=%.8f\n", mean);
    printf("pcm_rms=%.8f\n", rms);
    printf("pcm_nan=%lld\n", (long long)nan_count);
    printf("pcm_inf=%lld\n", (long long)inf_count);
    printf("pcm_clip=%lld\n", (long long)clip_count);
}

int main(int argc, char **argv) {
    Args a = {0};
    a.backend = AUDIOCPP_BACKEND_CUDA;
    a.n_threads = 8;
    a.task = AUDIOCPP_TASK_TTS;
    a.options_json = "{}";

    a.model_path = arg_value(argc, argv, "--model");
    a.family_hint = arg_value(argc, argv, "--family");
    a.text = arg_value(argc, argv, "--text");
    a.out_wav = arg_value(argc, argv, "--out");
    {
        const char *v;
        if ((v = arg_value(argc, argv, "--backend")) != NULL) a.backend = atoi(v);
        if ((v = arg_value(argc, argv, "--threads")) != NULL) a.n_threads = atoi(v);
        if ((v = arg_value(argc, argv, "--task")) != NULL) a.task = atoi(v);
        if ((v = arg_value(argc, argv, "--options")) != NULL) a.options_json = v;
    }
    /* --options-file: read the options JSON from a file (immune to shell
     * quoting mangling through ssh/PowerShell/cmd chains). */
    {
        const char *opt_file = arg_value(argc, argv, "--options-file");
        if (opt_file) {
            static char file_buf[16384];
            FILE *f = fopen(opt_file, "rb");
            if (!f) {
                fprintf(stderr, "cannot open --options-file: %s\n", opt_file);
                return 2;
            }
            size_t n = fread(file_buf, 1, sizeof(file_buf) - 1, f);
            fclose(f);
            file_buf[n] = '\0';
            a.options_json = file_buf;
        }
    }
    /* --load-options-file: session options for audiocpp_load_model_ex. */
    {
        const char *opt_file = arg_value(argc, argv, "--load-options-file");
        if (opt_file) {
            static char file_buf2[16384];
            FILE *f = fopen(opt_file, "rb");
            if (!f) {
                fprintf(stderr, "cannot open --load-options-file: %s\n", opt_file);
                return 2;
            }
            size_t n = fread(file_buf2, 1, sizeof(file_buf2) - 1, f);
            fclose(f);
            file_buf2[n] = '\0';
            a.load_options_json = file_buf2;
        }
    }

    if (!a.model_path || !a.text || !a.out_wav) {
        fprintf(stderr, "usage: capi_tts_test.exe --model <path> [--family <hint>] --text <text> --out <wav> [--backend n] [--threads n] [--task n] [--options json]\n");
        return 2;
    }

    printf("capi_version=%s\n", audiocpp_version());
    printf("capi_build_id=%s\n", audiocpp_build_id());
    printf("backend=%d threads=%d task=%d\n", a.backend, a.n_threads, a.task);
    printf("model=%s\n", a.model_path);
    printf("options=%s\n", a.options_json ? a.options_json : "(null)");
    if (a.family_hint) printf("family_hint=%s\n", a.family_hint);

    audiocpp_error_t err = {0};
    double t0 = now_ms();
    audiocpp_model_t *model = a.load_options_json
        ? audiocpp_load_model_ex(
              a.model_path, a.family_hint, a.task, a.backend, 0, a.n_threads,
              a.load_options_json, &err)
        : audiocpp_load_model(
              a.model_path, a.family_hint, a.task, a.backend, 0, a.n_threads, &err);
    double load_ms = now_ms() - t0;
    if (!model) {
        printf("load_failed=1\n");
        printf("error_code=%d\n", err.code);
        printf("error_message=%s\n", err.message ? err.message : "(null)");
        audiocpp_free_string(err.message);
        return 1;
    }
    printf("load_ms=%.1f\n", load_ms);

    audiocpp_model_info_t info = {0};
    if (audiocpp_model_info(model, &info) == 0) {
        printf("model_family=%s\n", info.family ? info.family : "");
        printf("model_variant=%s\n", info.variant ? info.variant : "");
        audiocpp_free_model_info(&info);
    }

    t0 = now_ms();
    audiocpp_audio_t *audio = audiocpp_tts(model, a.text, a.options_json, &err);
    double synth_ms = now_ms() - t0;
    if (!audio) {
        printf("tts_failed=1\n");
        printf("error_code=%d\n", err.code);
        printf("error_message=%s\n", err.message ? err.message : "(null)");
        audiocpp_free_string(err.message);
        audiocpp_free_model(model);
        return 1;
    }
    printf("tts_failed=0\n");
    printf("synth_ms=%.1f\n", synth_ms);
    printf("n_samples=%lld\n", (long long)audio->n_samples);
    printf("sample_rate=%d\n", audio->sample_rate);
    printf("channels=%d\n", audio->channels);
    if (audio->sample_rate > 0) {
        printf("duration_s=%.3f\n", (double)audio->n_samples / (double)audio->sample_rate);
    }
    print_stats(audio->samples, audio->n_samples);

    /* Raw f32 dump for offline analysis */
    char raw_path[4096];
    snprintf(raw_path, sizeof(raw_path), "%s.f32", a.out_wav);
    FILE *raw = fopen(raw_path, "wb");
    if (raw) {
        fwrite(audio->samples, sizeof(float), (size_t)audio->n_samples, raw);
        fclose(raw);
        printf("raw_dump=%s\n", raw_path);
    }

    t0 = now_ms();
    int rc = audiocpp_write_wav_ex(
        a.out_wav, audio->samples, audio->n_samples, audio->sample_rate, audio->channels);
    printf("write_wav_ms=%.1f\n", now_ms() - t0);
    printf("write_wav_rc=%d\n", rc);
    printf("wav_path=%s\n", a.out_wav);

    audiocpp_free_audio(audio);
    audiocpp_free_model(model);
    printf("DONE\n");
    return rc == 0 ? 0 : 1;
}
