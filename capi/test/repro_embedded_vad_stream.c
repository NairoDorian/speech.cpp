// Reproducer for the embedded-VAD stream_start bug:
//   load_model_ex(NULL, "silero_vad", ...) creates an offline session that
//   uploads the shared embedded weights and release_storage()s them; a later
//   stream_start() creates a second session which re-reads the source and
//   previously failed with "failed to open binary file: <embedded>".
// Build with AUDIOCPP_EMBED_VAD_ASSETS=ON; run with audiocpp.dll on PATH.
// Expect: model != NULL and stream != NULL.
#include <stdint.h>
#include <stdio.h>
#include <windows.h>

typedef struct {
    int code;
    char *message;
} audiocpp_error_t;

typedef void *audiocpp_model_t;
typedef void *audiocpp_stream_t;

int main(void) {
    HMODULE lib = LoadLibraryA("audiocpp.dll");
    if (!lib) {
        printf("FAIL: failed to load audiocpp.dll: %lu\n", GetLastError());
        return 1;
    }

    typedef audiocpp_model_t *(*load_model_ex_fn)(
        const char *, const char *, int, int, int, int, const char *, audiocpp_error_t *);
    typedef audiocpp_stream_t *(*stream_start_fn)(
        const audiocpp_model_t *, int, const char *, int64_t, audiocpp_error_t *);
    typedef void (*free_model_fn)(audiocpp_model_t *);
    typedef void (*free_stream_fn)(audiocpp_stream_t *);
    typedef void (*free_string_fn)(char *);

    const load_model_ex_fn load_model_ex =
        (load_model_ex_fn)GetProcAddress(lib, "audiocpp_load_model_ex");
    const stream_start_fn stream_start =
        (stream_start_fn)GetProcAddress(lib, "audiocpp_stream_start");
    const free_model_fn free_model = (free_model_fn)GetProcAddress(lib, "audiocpp_free_model");
    const free_stream_fn free_stream = (free_stream_fn)GetProcAddress(lib, "audiocpp_stream_free");
    const free_string_fn free_string = (free_string_fn)GetProcAddress(lib, "audiocpp_free_string");
    if (!load_model_ex || !stream_start || !free_model || !free_stream || !free_string) {
        printf("FAIL: required export missing\n");
        return 1;
    }

    // Empty path + silero_vad hint => embedded weights.
    audiocpp_error_t err = {0, NULL};
    audiocpp_model_t *model = load_model_ex(
        "", "silero_vad", 0 /* AUDIOCPP_TASK_VAD */, 0 /* AUDIOCPP_BACKEND_CPU */,
        0, 2, NULL, &err);
    if (err.message) {
        printf("load_model_ex err: %s\n", err.message);
        free_string(err.message);
    }
    if (!model) {
        printf("FAIL: load_model_ex returned NULL\n");
        return 1;
    }
    printf("load_model_ex OK (embedded silero_vad)\n");

    // Second session from the same loaded model => must re-upload weights.
    audiocpp_error_t serr = {0, NULL};
    audiocpp_stream_t *stream = stream_start(model, 0 /* VAD */, NULL, 0, &serr);
    if (serr.message) {
        printf("stream_start err: %s\n", serr.message);
        free_string(serr.message);
    }
    if (!stream) {
        printf("FAIL: stream_start returned NULL\n");
        return 1;
    }
    printf("stream_start OK\n");

    free_stream(stream);
    free_model(model);
    FreeLibrary(lib);
    printf("PASS\n");
    return 0;
}
