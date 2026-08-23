// Unit tests for audiocpp::detail::option_number_to_string — the C ABI
// helper that renders a JSON number into the string stored in the options map.
//
// The original bug: {"min_tokens": 2} was rendered as "2.000000", which the
// engine's stoll-based parse_int_value rejects ("must be an integer"). These
// tests lock down the fix and the 2^53 boundary (values beyond double's
// exact-integer range are emitted via the float serializer and rejected by
// stoll, never silently rounded). Mirrors test_cli_request_options.cpp.
//
// Note: this TU compiles audiocpp_capi.cpp directly (like the capi_test target)
// so the file-local helper is reachable. Stays pure ASCII because the C-API
// source carries CJK comments and this target sets /utf-8 only on the source.

#include "audiocpp_internal.h"

#include "engine/framework/runtime/options.h"

#include "test_assert.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

constexpr double kMaxExactJsonInteger = 9007199254740992.0;  // 2^53

struct NumberCase {
    const char * label;
    double input;
    const char * expected;  // nullptr => only assert it contains 'e'/'E'
};

const NumberCase kExactIntegerCases[] = {
    {"positive integer",            2.0,                       "2"},
    {"zero",                        0.0,                       "0"},
    {"negative integer",            -7.0,                      "-7"},
    {"large safe integer 1e15",     1000000000000000.0,        "1000000000000000"},
    {"1e15 plus one",               1000000000000001.0,        "1000000000000001"},
    {"negative large safe integer", -1000000000000000.0,       "-1000000000000000"},
    {"just below 2^53",             kMaxExactJsonInteger - 1,  "9007199254740991"},
};

const NumberCase kFloatCases[] = {
    {"plain fraction",   0.7,   "0.7"},
    {"multi-digit frac", 1.25,  "1.25"},
    {"negative float",   -0.5,  "-0.5"},
};

// Values at or beyond 2^53 cannot be represented exactly by a double: they must
// NOT be emitted as a plain integer (which would let stoll silently round a
// value the caller never sent). The helper routes them to the float serializer,
// so the output carries an exponent and parse_int_value rejects it.
const NumberCase kScientificCases[] = {
    {"exactly 2^53",     kMaxExactJsonInteger,     nullptr},
    {"2^53 plus one",    kMaxExactJsonInteger + 1, nullptr},
};

bool has_exponent(const std::string & s) {
    return s.find('e') != std::string::npos || s.find('E') != std::string::npos;
}

void test_exact_integers_render_without_decimals() {
    for (const auto & tc : kExactIntegerCases) {
        const std::string got = audiocpp::detail::option_number_to_string(tc.input);
        engine::test::require_eq(got, std::string(tc.expected), tc.label);
        // Regression guard: the old code produced "2.000000" here.
        engine::test::require(got.find('.') == std::string::npos,
                              std::string(tc.label) + " must have no decimal point");
    }
}

void test_floats_keep_their_fraction() {
    for (const auto & tc : kFloatCases) {
        const std::string got = audiocpp::detail::option_number_to_string(tc.input);
        engine::test::require_eq(got, std::string(tc.expected), tc.label);
    }
}

void test_beyond_exact_integer_range_uses_scientific_notation() {
    for (const auto & tc : kScientificCases) {
        const std::string got = audiocpp::detail::option_number_to_string(tc.input);
        engine::test::require(has_exponent(got),
                              std::string(tc.label) + " must use scientific notation (got: " + got + ")");
    }
}

// --- round-trip regression against the real option parser -------------------
// The whole point of the fix: a JSON integer round-trips through the engine's
// stoll parser. "2" (not "2.000000") must parse as 2 without throwing.

int parse_as_int(const std::string & value) {
    const std::unordered_map<std::string, std::string> options{{"min_tokens", value}};
    const auto parsed = engine::runtime::parse_int_option(options, {"min_tokens"});
    if (!parsed.has_value()) {
        throw std::runtime_error("option was not parsed");
    }
    return *parsed;
}

void require_int_rejected(const std::string & value) {
    try {
        (void) parse_as_int(value);
    } catch (const std::runtime_error &) {
        return;  // expected: stoll rejects the scientific-notation string
    }
    throw std::runtime_error("value unexpectedly accepted as int: " + value);
}

void test_integer_round_trips_through_option_parser() {
    // The headline regression: {"min_tokens": 2} -> "2" -> parse_int_option -> 2.
    const std::string two = audiocpp::detail::option_number_to_string(2.0);
    engine::test::require_eq(two, std::string("2"), "serialized min_tokens");
    engine::test::require_eq(parse_as_int(two), 2, "parsed min_tokens value");

    // A larger in-range integer round-trips too.
    const std::string steps = audiocpp::detail::option_number_to_string(50.0);
    engine::test::require_eq(parse_as_int(steps), 50, "parsed num_inference_steps");
}

void test_beyond_2_53_is_rejected_not_silently_rounded() {
    // 2^53 + 1 cannot be held exactly by a double; the helper must route it to
    // the float serializer so stoll rejects it — instead of casting to a
    // rounded long long (which the old +/-9.2e18 implementation did).
    const std::string beyond = audiocpp::detail::option_number_to_string(kMaxExactJsonInteger + 1);
    require_int_rejected(beyond);
}

void test_negative_integers_round_trip() {
    const std::string neg = audiocpp::detail::option_number_to_string(-7.0);
    engine::test::require_eq(neg, std::string("-7"), "serialized negative integer");
    engine::test::require_eq(parse_as_int(neg), -7, "parsed negative integer");
}

}  // namespace

int main() {
    test_exact_integers_render_without_decimals();
    test_floats_keep_their_fraction();
    test_beyond_exact_integer_range_uses_scientific_notation();
    test_integer_round_trips_through_option_parser();
    test_beyond_2_53_is_rejected_not_silently_rounded();
    test_negative_integers_round_trip();
    std::cout << "capi_option_number_test passed\n";
    return 0;
}
