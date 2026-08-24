#include "audiocpp.h"

#include <cassert>
#include <iostream>
#include <vector>

static void test_device_functions_containment() {
    int count = audiocpp_device_count();
    assert(count >= 0);

    audiocpp_device_info_t info{};
    int res = audiocpp_device_info(-1, &info);
    assert(res == -1);

    res = audiocpp_device_info(999999, &info);
    assert(res == -1);

    res = audiocpp_device_info(0, nullptr);
    assert(res == -1);

    int avail = audiocpp_backend_available(AUDIOCPP_BACKEND_CPU);
    assert(avail == 1);

    avail = audiocpp_backend_available(99999);
    assert(avail == 0);

    std::cout << "[PASS] test_device_functions_containment" << std::endl;
}

static void test_model_info_null_containment() {
    audiocpp_model_info_t info{};
    int res = audiocpp_model_info(nullptr, &info);
    assert(res == -1);
    assert(info.family == nullptr);

    audiocpp_model_capabilities_t caps{};
    res = audiocpp_model_capabilities(nullptr, &caps);
    assert(res == -1);
    assert(caps.n_supported_tasks == 0);

    // Free with null / zeroed info
    audiocpp_free_model_info(&info);
    audiocpp_free_capabilities(&caps);
    audiocpp_free_model(nullptr);

    std::cout << "[PASS] test_model_info_null_containment" << std::endl;
}

static void test_wav_io_containment() {
    float * samples = nullptr;
    int64_t n_samples = 0;
    int sample_rate = 0;

    int res = audiocpp_read_wav(nullptr, &samples, &n_samples, &sample_rate);
    assert(res == -1);

    res = audiocpp_read_wav("non_existent_file_xyz_12345.wav", &samples, &n_samples, &sample_rate);
    assert(res == -1);
    assert(samples == nullptr);

    res = audiocpp_write_wav(nullptr, nullptr, 0, 16000);
    assert(res == -1);

    res = audiocpp_write_wav_ex("/invalid_path_xyz/out.wav", nullptr, -1, 16000, 1);
    assert(res == -1);

    std::cout << "[PASS] test_wav_io_containment" << std::endl;
}

static void test_artifact_containment() {
    audiocpp_artifact_t * art = audiocpp_artifact_create(0, nullptr, nullptr, 0);
    assert(art == nullptr);

    art = audiocpp_artifact_create(AUDIOCPP_ARTIFACT_CUSTOM, "test_id", nullptr, 0);
    assert(art != nullptr);

    int res = audiocpp_artifact_set_meta(nullptr, "key", "val");
    assert(res == -1);

    res = audiocpp_artifact_set_meta(art, "key1", "val1");
    assert(res == 0);
    assert(art->n_meta == 1);

    audiocpp_artifact_free(art);
    audiocpp_artifact_free(nullptr);
    audiocpp_free_artifacts(nullptr);

    std::cout << "[PASS] test_artifact_containment" << std::endl;
}

static void test_memory_free_containment() {
    audiocpp_free_audio(nullptr);
    audiocpp_free_text(nullptr);
    audiocpp_free_text_batch(nullptr);
    audiocpp_free_diar(nullptr);
    audiocpp_free_vad(nullptr);
    audiocpp_free_string(nullptr);
    audiocpp_clear_error(nullptr);
    audiocpp_free_stream_event(nullptr);
    audiocpp_stream_free(nullptr);
    audiocpp_free_audio_batch(nullptr);
    audiocpp_free_align(nullptr);
    audiocpp_free_stems(nullptr);
    audiocpp_free_gen_result(nullptr);

    audiocpp_error_t err{};
    audiocpp_clear_error(&err);

    std::cout << "[PASS] test_memory_free_containment" << std::endl;
}

int main() {
    test_device_functions_containment();
    test_model_info_null_containment();
    test_wav_io_containment();
    test_artifact_containment();
    test_memory_free_containment();
    std::cout << "All C API exception containment tests passed successfully." << std::endl;
    return 0;
}
