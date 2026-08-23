#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <windows.h>

typedef const char* (*version_fn)(void);

typedef struct { int code; char* message; } err_t;
typedef void* (*load_fn)(const char*, const char*, int, int, int, int, err_t*);
typedef void (*free_model_fn)(void*);
typedef void* (*asr_fn)(void*, const float*, long long, int, const char*, err_t*);
typedef void (*free_text_fn)(void*);

typedef struct { float* samples; long long n; int sr; } audio_t;
typedef struct { char* text; char* lang; } text_t;

int main() {
    HMODULE lib = LoadLibraryA("audiocpp.dll");
    if (!lib) { printf("Failed to load dll\n"); return 1; }
    printf("dll loaded\n");

    version_fn ver = (version_fn)GetProcAddress(lib, "audiocpp_version");
    printf("version: %s\n", ver());

    load_fn load = (load_fn)GetProcAddress(lib, "audiocpp_load_model");
    free_model_fn free_model = (free_model_fn)GetProcAddress(lib, "audiocpp_free_model");
    asr_fn asr = (asr_fn)GetProcAddress(lib, "audiocpp_asr");
    free_text_fn free_text = (free_text_fn)GetProcAddress(lib, "audiocpp_free_text");

    err_t err = {0, NULL};
    printf("loading GGUF model...\n");
    void* model = load(
        "C:/Users/56579/ZCodeProject/sound/models/audiocpp/qwen3-asr-0.6b-q8_0.gguf",
        "qwen3_asr", 1, 0, 0, 0, &err);
    if (!model) {
        printf("load FAILED: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        return 1;
    }
    printf("model loaded OK: %p\n", model);

    free_model(model);
    printf("model freed\n");
    FreeLibrary(lib);
    printf("done\n");
    return 0;
}
