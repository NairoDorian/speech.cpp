// Unit tests for the C ABI session-options plumbing:
//   - audiocpp::detail::collect_option_fields / parse_session_options_json
//     (used by audiocpp_load_model_ex, declared in audiocpp_internal.h)
//   - pack_audio_output(AudioBuffer) channel preservation (the fix for
//     stereo model output being returned as mono)
//   - audiocpp_write_wav_ex stereo WAV round-trip
//   - audiocpp_load_model_ex strict-JSON behavior
//
// This TU compiles audiocpp_capi.cpp directly (like capi_option_number_test)
// so the file-local helpers are reachable. Stays pure ASCII because the C-API
// source carries CJK comments and this target sets /utf-8 only on the source.

#include "audiocpp.h"

#include "audiocpp_internal.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/session.h"

#include "cJSON.h"

#include "test_assert.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

namespace {

std::filesystem::path temp_wav_path(const char * tag) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("capi_session_options_" + std::string(tag) + "_" +
            std::to_string(counter++) + ".wav");
}

// --- collect_option_fields -------------------------------------------------

void test_collect_option_fields_rendering() {
    cJSON * root = cJSON_Parse(
        "{\"string_key\":\"hello\",\"int_key\":2,\"float_key\":0.7,"
        "\"bool_key\":true,\"big_key\":9007199254740993}");
    engine::test::require(root != nullptr, "parse options object");

    std::unordered_map<std::string, std::string> options;
    audiocpp::detail::collect_option_fields(root, options);
    cJSON_Delete(root);

    engine::test::require_eq(options.at("string_key"), std::string("hello"), "string passthrough");
    // Integers render without decimals so the stoll option path accepts them.
    engine::test::require_eq(options.at("int_key"), std::string("2"), "integer rendering");
    engine::test::require_eq(options.at("float_key"), std::string("0.7"), "float rendering");
    engine::test::require_eq(options.at("bool_key"), std::string("true"), "bool rendering");
    // Beyond 2^53: scientific notation, never silently rounded.
    engine::test::require(options.at("big_key").find('e') != std::string::npos ||
                              options.at("big_key").find('E') != std::string::npos,
                          "big integer must use scientific notation");
}

// --- parse_session_options_json -------------------------------------------

void test_parse_session_options_json_empty_is_defaults() {
    std::unordered_map<std::string, std::string> options{{"pre", "existing"}};
    engine::test::require(audiocpp::detail::parse_session_options_json(nullptr, options),
                          "NULL json is OK");
    engine::test::require(audiocpp::detail::parse_session_options_json("", options),
                          "empty json is OK");
    engine::test::require_eq(options.size(), static_cast<size_t>(1), "empty json adds nothing");
}

void test_parse_session_options_json_rejects_malformed() {
    std::unordered_map<std::string, std::string> options;
    engine::test::require(!audiocpp::detail::parse_session_options_json("{not json", options),
                          "malformed json rejected");
    engine::test::require(options.empty(), "malformed json adds nothing");
}

void test_parse_session_options_json_rejects_non_object() {
    std::unordered_map<std::string, std::string> options;
    engine::test::require(!audiocpp::detail::parse_session_options_json("[1,2,3]", options),
                          "array root rejected");
    engine::test::require(!audiocpp::detail::parse_session_options_json("\"str\"", options),
                          "string root rejected");
}

void test_parse_session_options_json_round_trip() {
    std::unordered_map<std::string, std::string> options;
    engine::test::require(audiocpp::detail::parse_session_options_json(
                              "{\"miotts.codec_model_path\":\"D:/codec.gguf\","
                              "\"qwen3_tts.perf_mode\":\"flash_attention\","
                              "\"moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb\":512}",
                              options),
                          "valid session options parsed");
    engine::test::require_eq(options.at("miotts.codec_model_path"), std::string("D:/codec.gguf"),
                             "codec path passthrough");
    engine::test::require_eq(options.at("qwen3_tts.perf_mode"), std::string("flash_attention"),
                             "perf mode passthrough");
    engine::test::require_eq(options.at("moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb"),
                             std::string("512"), "integer arena size");
}

// --- pack_audio_output channel preservation --------------------------------

void test_pack_audio_output_preserves_channels() {
    const std::vector<float> stereo{0.5f, -0.5f, 0.25f, -0.25f};
    engine::runtime::AudioBuffer buf;
    buf.sample_rate = 48000;
    buf.channels = 2;
    buf.samples = stereo;

    audiocpp_audio_t * out = audiocpp::detail::pack_audio_output(buf);
    engine::test::require(out != nullptr, "packed audio");
    engine::test::require_eq(out->channels, 2, "stereo channels preserved");
    engine::test::require_eq(out->sample_rate, 48000, "sample rate preserved");
    engine::test::require_eq(out->n_samples, static_cast<int64_t>(stereo.size()),
                             "n_samples is total interleaved count");
    for (size_t i = 0; i < stereo.size(); ++i) {
        engine::test::require_eq(static_cast<double>(out->samples[i]),
                                 static_cast<double>(stereo[i]), "sample value");
    }
    audiocpp_free_audio(out);
}

void test_pack_audio_output_guards_zero_channels() {
    engine::runtime::AudioBuffer buf;
    buf.sample_rate = 24000;
    buf.channels = 0;  // defensive: some producers may leave it unset
    buf.samples = {0.1f, 0.2f};
    audiocpp_audio_t * out = audiocpp::detail::pack_audio_output(buf);
    engine::test::require(out != nullptr, "packed audio with channels=0");
    engine::test::require_eq(out->channels, 1, "channels=0 normalized to mono");
    audiocpp_free_audio(out);
}

// --- audiocpp_write_wav_ex stereo round-trip --------------------------------

void test_write_wav_ex_stereo_round_trip() {
    const std::vector<float> stereo{0.5f, -0.5f, 0.25f, -0.25f, 0.1f, -0.1f};
    const std::filesystem::path path = temp_wav_path("stereo");
    const int rc = audiocpp_write_wav_ex(
        path.string().c_str(), stereo.data(),
        static_cast<int64_t>(stereo.size()), 48000, 2);
    engine::test::require_eq(rc, 0, "stereo wav written");

    const auto wav = engine::audio::read_wav_f32(path);
    std::filesystem::remove(path);
    engine::test::require_eq(wav.channels, 2, "wav header channels=2");
    engine::test::require_eq(wav.sample_rate, 48000, "wav header rate");
    engine::test::require_eq(wav.samples.size(), stereo.size(), "wav frame count");
    // f32 -> s16 -> f32 quantization tolerance.
    constexpr double kQuant = 1.0 / 32768.0 + 1e-6;
    for (size_t i = 0; i < stereo.size(); ++i) {
        engine::test::require(std::fabs(static_cast<double>(wav.samples[i]) - stereo[i]) < kQuant,
                              "sample survives s16 round trip");
    }
}

void test_write_wav_ex_validation() {
    const float samples[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    const std::filesystem::path path = temp_wav_path("bad");
    engine::test::require_eq(audiocpp_write_wav_ex(path.string().c_str(), samples, 4, 48000, 0), -1,
                             "channels=0 rejected");
    engine::test::require_eq(audiocpp_write_wav_ex(path.string().c_str(), samples, 4, 48000, 3), -1,
                             "non-divisible interleaved count rejected");
    engine::test::require_eq(audiocpp_write_wav_ex(nullptr, samples, 4, 48000, 1), -1,
                             "null path rejected");
    engine::test::require_eq(audiocpp_write_wav_ex(path.string().c_str(), nullptr, 4, 48000, 1), -1,
                             "null samples rejected");
    std::filesystem::remove(path);
}

// --- audiocpp_load_model_ex strict-JSON behavior ----------------------------

void test_load_model_ex_rejects_malformed_options() {
    audiocpp_error_t err = {};
    audiocpp_model_t * model = audiocpp_load_model_ex(
        "nonexistent.gguf", "qwen3_tts", AUDIOCPP_TASK_TTS,
        AUDIOCPP_BACKEND_CPU, 0, 1, "{not json", &err);
    engine::test::require(model == nullptr, "load fails on malformed options");
    engine::test::require(err.message != nullptr, "error message present");
    engine::test::require(std::string(err.message).find("session options JSON") != std::string::npos,
                          "error names the session options JSON");
    audiocpp_free_string(err.message);
}

void test_load_model_ex_valid_options_falls_through_to_path_check() {
    audiocpp_error_t err = {};
    audiocpp_model_t * model = audiocpp_load_model_ex(
        "nonexistent.gguf", "qwen3_tts", AUDIOCPP_TASK_TTS,
        AUDIOCPP_BACKEND_CPU, 0, 1, "{\"qwen3_tts.perf_mode\":\"flash_attention\"}", &err);
    engine::test::require(model == nullptr, "load fails on missing model path");
    engine::test::require(err.message != nullptr, "error message present");
    engine::test::require(std::string(err.message).find("session options JSON") == std::string::npos,
                          "valid options do not trip the JSON error");
    audiocpp_free_string(err.message);
}

void test_load_model_wrapper_still_works() {
    audiocpp_error_t err = {};
    audiocpp_model_t * model = audiocpp_load_model(
        "nonexistent.gguf", "qwen3_tts", AUDIOCPP_TASK_TTS,
        AUDIOCPP_BACKEND_CPU, 0, 1, &err);
    engine::test::require(model == nullptr, "wrapper load fails on missing model path");
    engine::test::require(err.message != nullptr, "wrapper error message present");
    audiocpp_free_string(err.message);
}

}  // namespace

int main() {
    test_collect_option_fields_rendering();
    test_parse_session_options_json_empty_is_defaults();
    test_parse_session_options_json_rejects_malformed();
    test_parse_session_options_json_rejects_non_object();
    test_parse_session_options_json_round_trip();
    test_pack_audio_output_preserves_channels();
    test_pack_audio_output_guards_zero_channels();
    test_write_wav_ex_stereo_round_trip();
    test_write_wav_ex_validation();
    test_load_model_ex_rejects_malformed_options();
    test_load_model_ex_valid_options_falls_through_to_path_check();
    test_load_model_wrapper_still_works();
    std::cout << "capi_session_options_test passed\n";
    return 0;
}
