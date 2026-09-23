// test_adapter_run_params.cpp - the C ABI adapter's translation of tri-state
// run knobs (transcribe_run_params::pnc / itn / diarize) into engine request
// options.
//
// Regression (found 2026-09-23 while triaging transcribe.cpp 9eed7f09): the
// adapter wrote the raw enum number into bool options, so through the C ABI
// DEFAULT (0) parsed as false, OFF (1) as true, and ON (2) made
// parse_bool_option throw - ITN was inverted for sense_asr / fun_asr_nano and
// PNC for canary_asr / cohere_asr. The mapping must be: OFF -> false, ON ->
// true, DEFAULT -> unset (the family's own spec default applies).
//
// Also covers the Whisper run extension (W2b.2, the C-ABI takeover of Whisper
// by the engine package): the adapter validates transcribe_whisper_run_ext
// before the previous result is cleared and forwards every field under the
// option names model_specs/whisper.json declares, with floats (including the
// +/-INF _DISABLED sentinels) round-tripping exactly through the engine's
// parser.

#include "transcribe-arch-adapter.h"
#include "transcribe.h"
#include "transcribe/whisper.h"

#include "engine/framework/runtime/options.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <unordered_map>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// The value the adapter emits for `mode`, run through the same parser the
// engine families use. Returns -1 when the adapter leaves the option unset,
// -2 when the parser rejects it.
int parsed(int mode) {
    const char * value = transcribe::adapter_tristate_bool_option(mode);
    if (value == nullptr) {
        return -1;
    }
    try {
        return engine::runtime::parse_bool_option(value, "tristate") ? 1 : 0;
    } catch (const std::exception &) {
        return -2;
    }
}

using Options = std::unordered_map<std::string, std::string>;

std::string opt(const Options & o, const char * key) {
    const auto it = o.find(key);
    return it == o.end() ? std::string("<unset>") : it->second;
}

float opt_float(const Options & o, const char * key) {
    return engine::runtime::parse_float_option(o, {key}).value_or(-12345.0f);
}

transcribe_run_params run_params_with(transcribe_whisper_run_ext * ext) {
    transcribe_run_params params;
    transcribe_run_params_init(&params);
    params.family = ext != nullptr ? &ext->ext : nullptr;
    return params;
}

void test_whisper_run_ext() {
    using transcribe::adapter_family_run_ext_options;
    using transcribe::adapter_validate_family_run_ext;

    // No extension: nothing to validate, nothing forwarded (engine defaults).
    {
        const transcribe_run_params params = run_params_with(nullptr);
        CHECK(adapter_validate_family_run_ext("whisper", &params) == TRANSCRIBE_OK);
        CHECK(adapter_family_run_ext_options("whisper", &params).empty());
    }

    // The init defaults forward the recipe defaults verbatim.
    {
        transcribe_whisper_run_ext ext;
        transcribe_whisper_run_ext_init(&ext);
        const transcribe_run_params params = run_params_with(&ext);
        CHECK(adapter_validate_family_run_ext("whisper", &params) == TRANSCRIBE_OK);
        const Options o = adapter_family_run_ext_options("whisper", &params);
        CHECK(opt(o, "prompt_condition") == "first_segment");
        CHECK(opt(o, "condition_on_prev_tokens") == "false");
        CHECK(opt(o, "max_prev_context_tokens") == "223");
        CHECK(opt_float(o, "temperature") == 0.0f);
        CHECK(opt_float(o, "temperature_inc") == 0.2f);
        CHECK(opt_float(o, "compression_ratio_thold") == 2.4f);
        CHECK(opt_float(o, "logprob_thold") == -1.0f);
        CHECK(opt_float(o, "no_speech_thold") == 0.6f);
        CHECK(opt(o, "seed") == "0");
        CHECK(opt_float(o, "max_initial_timestamp") == 1.0f);
        CHECK(opt(o, "initial_prompt") == "<unset>");
        CHECK(opt(o, "prompt_tokens") == "<unset>");
    }

    // Disabled thresholds, a full-range seed, prompt precedence.
    {
        transcribe_whisper_run_ext ext;
        transcribe_whisper_run_ext_init(&ext);
        const int32_t ids[] = {50363, 440, 1};
        ext.initial_prompt = "ignored because prompt_tokens wins";
        ext.prompt_tokens = ids;
        ext.n_prompt_tokens = 3;
        ext.prompt_condition = TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS;
        ext.condition_on_prev_tokens = true;
        ext.temperature = 0.1f;  // not exactly representable: must round-trip bit-exact
        ext.compression_ratio_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED;
        ext.logprob_thold = TRANSCRIBE_WHISPER_LOGPROB_DISABLED;
        ext.no_speech_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED;
        ext.seed = 4000000000u;  // above INT32_MAX
        ext.max_initial_timestamp = -1.0f;  // negative disables the cap
        const transcribe_run_params params = run_params_with(&ext);
        CHECK(adapter_validate_family_run_ext("whisper", &params) == TRANSCRIBE_OK);
        const Options o = adapter_family_run_ext_options("whisper", &params);
        CHECK(opt(o, "prompt_tokens") == "50363,440,1");
        CHECK(opt(o, "initial_prompt") == "<unset>");
        CHECK(opt(o, "prompt_condition") == "all_segments");
        CHECK(opt(o, "condition_on_prev_tokens") == "true");
        CHECK(opt_float(o, "temperature") == 0.1f);
        CHECK(std::isinf(opt_float(o, "compression_ratio_thold")) && opt_float(o, "compression_ratio_thold") > 0);
        CHECK(std::isinf(opt_float(o, "logprob_thold")) && opt_float(o, "logprob_thold") < 0);
        CHECK(std::isinf(opt_float(o, "no_speech_thold")));
        CHECK(engine::runtime::parse_u32_option(o, {"seed"}).value_or(0) == 4000000000u);
        CHECK(opt_float(o, "max_initial_timestamp") == -1.0f);
    }

    // initial_prompt forwards when no prompt_tokens are given.
    {
        transcribe_whisper_run_ext ext;
        transcribe_whisper_run_ext_init(&ext);
        ext.initial_prompt = "Glossary: Quilter, Sandalen.";
        const transcribe_run_params params = run_params_with(&ext);
        CHECK(opt(adapter_family_run_ext_options("whisper", &params), "initial_prompt") ==
              "Glossary: Quilter, Sandalen.");
    }

    // Rejections that need no model state, checked before the prior result is
    // cleared.
    {
        transcribe_whisper_run_ext ext;
        transcribe_whisper_run_ext_init(&ext);
        ext.prompt_condition = TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS;  // without condition_on_prev_tokens
        transcribe_run_params params = run_params_with(&ext);
        CHECK(adapter_validate_family_run_ext("whisper", &params) == TRANSCRIBE_ERR_INVALID_ARG);

        transcribe_whisper_run_ext_init(&ext);
        ext.prompt_condition = static_cast<transcribe_whisper_prompt_condition>(7);
        CHECK(adapter_validate_family_run_ext("whisper", &params) == TRANSCRIBE_ERR_INVALID_ARG);

        transcribe_whisper_run_ext_init(&ext);
        ext.temperature = -0.5f;
        CHECK(adapter_validate_family_run_ext("whisper", &params) == TRANSCRIBE_ERR_INVALID_ARG);

        transcribe_whisper_run_ext_init(&ext);
        ext.max_prev_context_tokens = -1;
        CHECK(adapter_validate_family_run_ext("whisper", &params) == TRANSCRIBE_ERR_INVALID_ARG);

        // A truncated extension (an older, smaller struct) is refused.
        transcribe_whisper_run_ext_init(&ext);
        ext.ext.size = sizeof(transcribe_ext);
        CHECK(adapter_validate_family_run_ext("whisper", &params) != TRANSCRIBE_OK);

        // The Whisper extension pointed at a family without a run surface.
        transcribe_whisper_run_ext_init(&ext);
        CHECK(adapter_validate_family_run_ext("sense_asr", &params) == TRANSCRIBE_ERR_INVALID_ARG);
    }
}

}  // namespace

int main() {
    // ITN (sense_asr / fun_asr_nano "enable_itn").
    CHECK(parsed(TRANSCRIBE_ITN_MODE_DEFAULT) == -1);
    CHECK(parsed(TRANSCRIBE_ITN_MODE_OFF) == 0);
    CHECK(parsed(TRANSCRIBE_ITN_MODE_ON) == 1);

    // PNC (canary_asr / cohere_asr "pnc", spec default true).
    CHECK(parsed(TRANSCRIBE_PNC_MODE_DEFAULT) == -1);
    CHECK(parsed(TRANSCRIBE_PNC_MODE_OFF) == 0);
    CHECK(parsed(TRANSCRIBE_PNC_MODE_ON) == 1);

    // DIARIZE.
    CHECK(parsed(TRANSCRIBE_DIARIZE_MODE_DEFAULT) == -1);
    CHECK(parsed(TRANSCRIBE_DIARIZE_MODE_OFF) == 0);
    CHECK(parsed(TRANSCRIBE_DIARIZE_MODE_ON) == 1);

    // A value outside the enum never becomes a bool the family would act on.
    CHECK(parsed(3) == -1);
    CHECK(parsed(-1) == -1);

    test_whisper_run_ext();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_adapter_run_params: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_adapter_run_params: ok\n");
    return 0;
}
