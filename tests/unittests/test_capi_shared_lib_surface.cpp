// Shared-library symbol-surface test for libaudiocpp.
//
// Loads the BUILT audiocpp shared library (path injected via
// AUDIOCPP_TEST_LIB_PATH = $<TARGET_FILE:audiocpp>) and checks the dynamic
// symbol table contract that downstream FFI consumers (e.g. sound-server's
// Rust bindings) rely on:
//
//   1. Every public AUDIOCPP_API function declared in capi/include/audiocpp.h
//      resolves. A merge that drops an export (visibility change, .def /
//      version-script drift, rename) fails here instead of at the consumer.
//   2. Vendored cJSON and libyaml symbols do NOT leak. The fork hides them
//      (CJSON_HIDE_SYMBOLS + hidden visibility presets); a leaked cJSON_*
//      export collides with any host process that links its own cJSON.
//
// The export list below mirrors audiocpp.h exactly — when adding a public
// function, extend the header AND this list (and the .def whitelist in
// CMakeLists.txt).

#include "test_assert.h"

#include <cstdio>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

const char *const kRequiredExports[] = {
    "audiocpp_align",
    "audiocpp_artifact_create",
    "audiocpp_artifact_free",
    "audiocpp_artifact_set_meta",
    "audiocpp_asr",
    "audiocpp_audio_transform",
    "audiocpp_audio_transform_with_voice_ref",
    "audiocpp_backend_available",
    "audiocpp_build_id",
    "audiocpp_clear_error",
    "audiocpp_denoise",
    "audiocpp_device_count",
    "audiocpp_device_info",
    "audiocpp_diar",
    "audiocpp_free_align",
    "audiocpp_free_artifacts",
    "audiocpp_free_audio",
    "audiocpp_free_audio_batch",
    "audiocpp_free_capabilities",
    "audiocpp_free_diar",
    "audiocpp_free_gen_result",
    "audiocpp_free_model",
    "audiocpp_free_model_info",
    "audiocpp_free_stems",
    "audiocpp_free_stream_event",
    "audiocpp_free_string",
    "audiocpp_free_text",
    "audiocpp_free_vad",
    "audiocpp_generate",
    "audiocpp_list_devices",
    "audiocpp_load_model",
    "audiocpp_load_model_ex",
    "audiocpp_midi",
    "audiocpp_midi_from_wav",
    "audiocpp_model_capabilities",
    "audiocpp_model_info",
    "audiocpp_read_wav",
    "audiocpp_set_progress_callback",
    "audiocpp_stream_finish",
    "audiocpp_stream_free",
    "audiocpp_stream_pull",
    "audiocpp_stream_push",
    "audiocpp_stream_start",
    "audiocpp_super_resolve",
    "audiocpp_transform_stems",
    "audiocpp_tts",
    "audiocpp_tts_batch",
    "audiocpp_tts_with_voice_ref",
    "audiocpp_vad",
    "audiocpp_vad_energy",
    "audiocpp_version",
    "audiocpp_write_wav",
    "audiocpp_write_wav_ex",
};

// Vendored symbols that must never be visible from libaudiocpp.
const char *const kForbiddenExports[] = {
    "cJSON_Parse",
    "cJSON_Print",
    "cJSON_Delete",
    "yaml_parser_initialize",
    "yaml_parser_delete",
};

void *load_library(const char *path) {
#ifdef _WIN32
    return static_cast<void *>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void *resolve_symbol(void *lib, const char *name) {
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

const char *load_error() {
#ifdef _WIN32
    static char buf[256];
    const DWORD rc = GetLastError();
    snprintf(buf, sizeof(buf), "LoadLibrary failed (GetLastError=%lu)", static_cast<unsigned long>(rc));
    return buf;
#else
    const char *err = dlerror();
    return err != nullptr ? err : "dlopen failed";
#endif
}

}  // namespace

int main() {
    engine::test::require(
        AUDIOCPP_TEST_LIB_PATH != nullptr && AUDIOCPP_TEST_LIB_PATH[0] != '\0',
        "AUDIOCPP_TEST_LIB_PATH must be defined at build time");

    void *lib = load_library(AUDIOCPP_TEST_LIB_PATH);
    engine::test::require(lib != nullptr,
        std::string("failed to load built library '") + AUDIOCPP_TEST_LIB_PATH +
            "': " + load_error());
    std::printf("loaded: %s\n", AUDIOCPP_TEST_LIB_PATH);

    for (const char *name : kRequiredExports) {
        engine::test::require(
            resolve_symbol(lib, name) != nullptr,
            std::string("required export missing: ") + name);
    }
    std::printf("required exports: %zu resolved\n",
                sizeof(kRequiredExports) / sizeof(kRequiredExports[0]));

    for (const char *name : kForbiddenExports) {
        engine::test::require(
            resolve_symbol(lib, name) == nullptr,
            std::string("vendored symbol leaked into the shared library: ") + name);
    }
    std::printf("vendored cJSON/libyaml symbols: none leaked\n");

    std::printf("capi_shared_lib_surface_test: all cases passed\n");
    return 0;
}
