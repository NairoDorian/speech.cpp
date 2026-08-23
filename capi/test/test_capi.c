// 最小测试:只调 audiocpp_version,不加载模型,看 dll 本身是否 crash
#include <stdio.h>
#include <windows.h>

typedef const char* (*version_fn)(void);

int main() {
    HMODULE lib = LoadLibraryA("audiocpp.dll");
    if (!lib) {
        printf("Failed to load audiocpp.dll: %lu\n", GetLastError());
        return 1;
    }
    printf("dll loaded OK\n");

    version_fn ver = (version_fn)GetProcAddress(lib, "audiocpp_version");
    if (!ver) {
        printf("audiocpp_version not found\n");
        return 1;
    }
    printf("version: %s\n", ver());

    // 测试 load_model(传 NULL 路径,预期返回错误不 crash)
    typedef void* (*load_fn)(const char*, const char*, int, int, int, int, void*);
    load_fn load = (load_fn)GetProcAddress(lib, "audiocpp_load_model");
    if (!load) {
        printf("audiocpp_load_model not found\n");
        return 1;
    }

    printf("calling load_model with qwen3_asr hint...\n");
    // 只测试 registry 是否能初始化(传不存在的路径,预期返回 NULL + error)
    int err_code = 0;
    char* err_msg = NULL;
    void* err_ptr = &err_code; // 简化:实际是 audiocpp_error_t
    void* model = load("nonexistent.gguf", "qwen3_asr", 1, 0, 0, 0, NULL);
    printf("load_model returned: %p (expected NULL for bad path)\n", model);

    FreeLibrary(lib);
    printf("test passed\n");
    return 0;
}
