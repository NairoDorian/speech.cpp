// test_abi_arch_engine_verdict.cpp - Phase 11b verdict harness: one GGUF, two
// implementations, one C ABI.
//
// Loads the same transcribe.cpp-layout GGUF twice through the public C ABI:
// once as the builtin transcribe.cpp arch, once through the engine family that
// reads the same layout (SPEECHCPP_ENGINE_ARCHS=all, engine_route_forced()).
// Every clip is then run on both sessions with identical parameters and the
// complete observable result is compared field by field: status, full / raw
// text, detected language, truncation, returned timestamp kind, and every
// segment / word / token / speaker row (timings, text, links, speaker ids).
// Capabilities and feature bits are compared too. A batched pass compares the
// per-utterance status and text.
//
// This is the measurement a family retirement is decided on (Phase 10.5 / 11b
// procedure: measure the C ABI first). It replaces a hand-written per-family
// arch-vs-engine test for the common case; family tests keep the typed-ext
// and family-specific assertions.
//
// Links ONLY the public C ABI (like abi_stream_hello): what a binding sees.
//
// usage: abi_arch_engine_verdict <model.gguf> <wav-or-dir>... [options]
//   --language <code>     run language (default: model default / auto)
//   --timestamps <kind>   none|auto|segment|word|token (default auto)
//   --threads <n>         session threads (default 0 = library default)
//   --max-clips <n>       limit the clip count (default all)
//   --text-only           compare text/status only (rows reported, not failed)
//   --no-batch            skip the batched pass
//   --p-tol <x>           token probability tolerance (default 1e-3)
//   --fresh-session       new session per clip on both sides (for an arch that
//                         cannot run twice on one session - voxtral frees its
//                         KV cache after a run but keeps its size)
//
// Exit: 0 = identical, 1 = differences (all printed), 2 = model/fixtures
// missing (skip).

#include "abi_test_wav.h"
#include "transcribe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Options {
    std::string model;
    std::vector<std::string> inputs;
    std::string language;
    transcribe_timestamp_kind timestamps = TRANSCRIBE_TIMESTAMPS_AUTO;
    int threads = 0;
    size_t max_clips = 0;
    bool text_only = false;
    bool batch = true;
    bool fresh_session = false;
    double p_tol = 1e-3;
};

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (value[0] == '\0') {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

std::string str_or_empty(const char * s) {
    return s != nullptr ? std::string(s) : std::string();
}

// ---------------------------------------------------------------------------
// Result snapshot through the C ABI
// ---------------------------------------------------------------------------

struct Segment {
    int64_t t0 = 0, t1 = 0;
    int first_word = 0, n_words = 0, first_token = 0, n_tokens = 0;
    int32_t speaker = 0;
    std::string text;
};
struct Word {
    int64_t t0 = 0, t1 = 0;
    int seg = 0, first_token = 0, n_tokens = 0;
    std::string text;
};
struct Token {
    int id = 0;
    float p = 0.0f;
    int64_t t0 = 0, t1 = 0;
    int seg = 0, word = 0;
    std::string text;
};
struct Speaker {
    int64_t t0 = 0, t1 = 0;
    int32_t id = 0;
    float p = 0.0f;
};

struct Snapshot {
    transcribe_status status = TRANSCRIBE_OK;
    std::string full_text, raw_text, language;
    bool truncated = false;
    int kind = 0;
    std::vector<Segment> segments;
    std::vector<Word> words;
    std::vector<Token> tokens;
    std::vector<Speaker> speakers;
    double seconds = 0.0;
};

Snapshot snapshot(const transcribe_session * s, transcribe_status status, double seconds) {
    Snapshot out;
    out.status = status;
    out.seconds = seconds;
    out.full_text = str_or_empty(transcribe_full_text(s));
    out.raw_text = str_or_empty(transcribe_raw_text(s));
    out.language = str_or_empty(transcribe_detected_language(s));
    out.truncated = transcribe_was_truncated(s);
    out.kind = static_cast<int>(transcribe_returned_timestamp_kind(s));
    for (int i = 0, n = transcribe_n_segments(s); i < n; ++i) {
        transcribe_segment seg;
        transcribe_segment_init(&seg);
        if (transcribe_get_segment(s, i, &seg) != TRANSCRIBE_OK) {
            continue;
        }
        out.segments.push_back({seg.t0_ms, seg.t1_ms, seg.first_word, seg.n_words, seg.first_token,
                                seg.n_tokens, seg.speaker_id, str_or_empty(seg.text)});
    }
    for (int i = 0, n = transcribe_n_words(s); i < n; ++i) {
        transcribe_word w;
        transcribe_word_init(&w);
        if (transcribe_get_word(s, i, &w) != TRANSCRIBE_OK) {
            continue;
        }
        out.words.push_back({w.t0_ms, w.t1_ms, w.seg_index, w.first_token, w.n_tokens, str_or_empty(w.text)});
    }
    for (int i = 0, n = transcribe_n_tokens(s); i < n; ++i) {
        transcribe_token t;
        transcribe_token_init(&t);
        if (transcribe_get_token(s, i, &t) != TRANSCRIBE_OK) {
            continue;
        }
        out.tokens.push_back({t.id, t.p, t.t0_ms, t.t1_ms, t.seg_index, t.word_index, str_or_empty(t.text)});
    }
    for (int i = 0, n = transcribe_n_speaker_segments(s); i < n; ++i) {
        transcribe_speaker_segment sp;
        transcribe_speaker_segment_init(&sp);
        if (transcribe_get_speaker_segment(s, i, &sp) != TRANSCRIBE_OK) {
            continue;
        }
        out.speakers.push_back({sp.t0_ms, sp.t1_ms, sp.speaker_id, sp.p});
    }
    return out;
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

struct Diff {
    std::vector<std::string> lines;
    template <typename... Args>
    void add(const char * fmt, Args... args) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), fmt, args...);
        lines.emplace_back(buf);
    }
    bool empty() const { return lines.empty(); }
};

bool same_p(float a, float b, double tol) {
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    return std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= tol;
}

// Returns the text-level differences in `text`, row-level ones in `rows`.
void compare(const Snapshot & a, const Snapshot & e, double p_tol, Diff & text, Diff & rows) {
    if (a.status != e.status) {
        text.add("status: arch %s, engine %s", transcribe_status_string(a.status),
                 transcribe_status_string(e.status));
    }
    if (a.full_text != e.full_text) {
        text.add("full_text:\n      arch   \"%s\"\n      engine \"%s\"", a.full_text.c_str(), e.full_text.c_str());
    }
    if (a.raw_text != e.raw_text) {
        text.add("raw_text:\n      arch   \"%s\"\n      engine \"%s\"", a.raw_text.c_str(), e.raw_text.c_str());
    }
    if (a.language != e.language) {
        text.add("detected_language: arch \"%s\", engine \"%s\"", a.language.c_str(), e.language.c_str());
    }
    if (a.truncated != e.truncated) {
        text.add("truncated: arch %d, engine %d", a.truncated ? 1 : 0, e.truncated ? 1 : 0);
    }
    if (a.kind != e.kind) {
        rows.add("returned timestamp kind: arch %d, engine %d", a.kind, e.kind);
    }
    if (a.segments.size() != e.segments.size()) {
        rows.add("segments: arch %zu, engine %zu", a.segments.size(), e.segments.size());
    }
    for (size_t i = 0; i < std::min(a.segments.size(), e.segments.size()); ++i) {
        const auto & x = a.segments[i];
        const auto & y = e.segments[i];
        if (x.t0 != y.t0 || x.t1 != y.t1 || x.text != y.text || x.speaker != y.speaker || x.n_words != y.n_words ||
            x.first_word != y.first_word || x.n_tokens != y.n_tokens || x.first_token != y.first_token) {
            rows.add("segment %zu: arch [%lld,%lld] spk %d w%d+%d t%d+%d \"%s\" | engine [%lld,%lld] spk %d w%d+%d "
                     "t%d+%d \"%s\"",
                     i, static_cast<long long>(x.t0), static_cast<long long>(x.t1), x.speaker, x.first_word,
                     x.n_words, x.first_token, x.n_tokens, x.text.c_str(), static_cast<long long>(y.t0),
                     static_cast<long long>(y.t1), y.speaker, y.first_word, y.n_words, y.first_token, y.n_tokens,
                     y.text.c_str());
            break;
        }
    }
    if (a.words.size() != e.words.size()) {
        rows.add("words: arch %zu, engine %zu", a.words.size(), e.words.size());
    }
    for (size_t i = 0; i < std::min(a.words.size(), e.words.size()); ++i) {
        const auto & x = a.words[i];
        const auto & y = e.words[i];
        if (x.t0 != y.t0 || x.t1 != y.t1 || x.text != y.text || x.seg != y.seg || x.n_tokens != y.n_tokens ||
            x.first_token != y.first_token) {
            rows.add("word %zu: arch [%lld,%lld] seg %d \"%s\" | engine [%lld,%lld] seg %d \"%s\"", i,
                     static_cast<long long>(x.t0), static_cast<long long>(x.t1), x.seg, x.text.c_str(),
                     static_cast<long long>(y.t0), static_cast<long long>(y.t1), y.seg, y.text.c_str());
            break;
        }
    }
    if (a.tokens.size() != e.tokens.size()) {
        rows.add("tokens: arch %zu, engine %zu", a.tokens.size(), e.tokens.size());
    }
    for (size_t i = 0; i < std::min(a.tokens.size(), e.tokens.size()); ++i) {
        const auto & x = a.tokens[i];
        const auto & y = e.tokens[i];
        if (x.id != y.id || x.t0 != y.t0 || x.t1 != y.t1 || x.text != y.text || x.seg != y.seg || x.word != y.word ||
            !same_p(x.p, y.p, p_tol)) {
            rows.add("token %zu: arch id %d p %.4f [%lld,%lld] s%d w%d \"%s\" | engine id %d p %.4f [%lld,%lld] s%d "
                     "w%d \"%s\"",
                     i, x.id, static_cast<double>(x.p), static_cast<long long>(x.t0), static_cast<long long>(x.t1),
                     x.seg, x.word, x.text.c_str(), y.id, static_cast<double>(y.p), static_cast<long long>(y.t0),
                     static_cast<long long>(y.t1), y.seg, y.word, y.text.c_str());
            break;
        }
    }
    if (a.speakers.size() != e.speakers.size()) {
        rows.add("speaker rows: arch %zu, engine %zu", a.speakers.size(), e.speakers.size());
    }
    for (size_t i = 0; i < std::min(a.speakers.size(), e.speakers.size()); ++i) {
        const auto & x = a.speakers[i];
        const auto & y = e.speakers[i];
        if (x.t0 != y.t0 || x.t1 != y.t1 || x.id != y.id) {
            rows.add("speaker row %zu: arch [%lld,%lld] spk %d | engine [%lld,%lld] spk %d", i,
                     static_cast<long long>(x.t0), static_cast<long long>(x.t1), x.id, static_cast<long long>(y.t0),
                     static_cast<long long>(y.t1), y.id);
            break;
        }
    }
}

void compare_caps(const transcribe_model * a, const transcribe_model * e, Diff & out) {
    transcribe_capabilities ca, ce;
    transcribe_capabilities_init(&ca);
    transcribe_capabilities_init(&ce);
    if (transcribe_model_get_capabilities(a, &ca) != TRANSCRIBE_OK ||
        transcribe_model_get_capabilities(e, &ce) != TRANSCRIBE_OK) {
        out.add("capabilities: query failed");
        return;
    }
    if (ca.native_sample_rate != ce.native_sample_rate) {
        out.add("caps.native_sample_rate: arch %d, engine %d", ca.native_sample_rate, ce.native_sample_rate);
    }
    if (ca.max_timestamp_kind != ce.max_timestamp_kind) {
        out.add("caps.max_timestamp_kind: arch %d, engine %d", static_cast<int>(ca.max_timestamp_kind),
                static_cast<int>(ce.max_timestamp_kind));
    }
    if (ca.max_audio_ms != ce.max_audio_ms) {
        out.add("caps.max_audio_ms: arch %lld, engine %lld", static_cast<long long>(ca.max_audio_ms),
                static_cast<long long>(ce.max_audio_ms));
    }
    const auto flag = [&](const char * name, bool x, bool y) {
        if (x != y) {
            out.add("caps.%s: arch %d, engine %d", name, x ? 1 : 0, y ? 1 : 0);
        }
    };
    flag("supports_language_detect", ca.supports_language_detect, ce.supports_language_detect);
    flag("supports_translate", ca.supports_translate, ce.supports_translate);
    flag("supports_streaming", ca.supports_streaming, ce.supports_streaming);
    flag("supports_spec_decode", ca.supports_spec_decode, ce.supports_spec_decode);
    std::vector<std::string> la, le;
    for (int i = 0; i < ca.n_languages; ++i) la.emplace_back(ca.languages[i]);
    for (int i = 0; i < ce.n_languages; ++i) le.emplace_back(ce.languages[i]);
    std::sort(la.begin(), la.end());
    std::sort(le.begin(), le.end());
    if (la != le) {
        out.add("caps.languages: arch %zu, engine %zu (sets differ)", la.size(), le.size());
    }
    static const struct {
        transcribe_feature f;
        const char * name;
    } kFeatures[] = {
        {TRANSCRIBE_FEATURE_INITIAL_PROMPT, "INITIAL_PROMPT"},
        {TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK, "TEMPERATURE_FALLBACK"},
        {TRANSCRIBE_FEATURE_LONG_FORM, "LONG_FORM"},
        {TRANSCRIBE_FEATURE_CANCELLATION, "CANCELLATION"},
        {TRANSCRIBE_FEATURE_PNC, "PNC"},
        {TRANSCRIBE_FEATURE_ITN, "ITN"},
        {TRANSCRIBE_FEATURE_DIARIZATION, "DIARIZATION"},
    };
    for (const auto & feature : kFeatures) {
        flag(feature.name, transcribe_model_supports(a, feature.f), transcribe_model_supports(e, feature.f));
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

bool parse_args(int argc, char ** argv, Options & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--language") {
            const char * v = next();
            if (!v) return false;
            o.language = v;
        } else if (a == "--timestamps") {
            const char * v = next();
            if (!v) return false;
            const std::string k = v;
            if (k == "none") o.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
            else if (k == "auto") o.timestamps = TRANSCRIBE_TIMESTAMPS_AUTO;
            else if (k == "segment") o.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
            else if (k == "word") o.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
            else if (k == "token") o.timestamps = TRANSCRIBE_TIMESTAMPS_TOKEN;
            else return false;
        } else if (a == "--threads") {
            const char * v = next();
            if (!v) return false;
            o.threads = std::atoi(v);
        } else if (a == "--max-clips") {
            const char * v = next();
            if (!v) return false;
            o.max_clips = static_cast<size_t>(std::atoll(v));
        } else if (a == "--p-tol") {
            const char * v = next();
            if (!v) return false;
            o.p_tol = std::atof(v);
        } else if (a == "--text-only") {
            o.text_only = true;
        } else if (a == "--no-batch") {
            o.batch = false;
        } else if (a == "--fresh-session") {
            o.fresh_session = true;
        } else if (o.model.empty()) {
            o.model = a;
        } else {
            o.inputs.push_back(a);
        }
    }
    return !o.model.empty() && !o.inputs.empty();
}

std::vector<std::string> collect_wavs(const std::vector<std::string> & inputs) {
    std::vector<std::string> out;
    for (const auto & in : inputs) {
        std::error_code ec;
        if (fs::is_directory(in, ec)) {
            std::vector<std::string> dir;
            for (const auto & entry : fs::directory_iterator(in, ec)) {
                if (entry.path().extension() == ".wav") {
                    dir.push_back(entry.path().string());
                }
            }
            std::sort(dir.begin(), dir.end());
            out.insert(out.end(), dir.begin(), dir.end());
        } else if (fs::exists(in, ec)) {
            out.push_back(in);
        }
    }
    return out;
}

struct Side {
    const char * label;
    transcribe_model * model = nullptr;
    transcribe_session * session = nullptr;
    double total_seconds = 0.0;

    ~Side() {
        if (session) transcribe_session_free(session);
        if (model) transcribe_model_free(model);
    }
};

bool open_session(Side & side, const Options & o) {
    if (side.session) {
        transcribe_session_free(side.session);
        side.session = nullptr;
    }
    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    sp.n_threads = o.threads;
    const transcribe_status ss = transcribe_session_init(side.model, &sp, &side.session);
    if (ss != TRANSCRIBE_OK) {
        std::fprintf(stderr, "%s: session init failed: %s\n", side.label, transcribe_status_string(ss));
        return false;
    }
    return true;
}

bool open_side(Side & side, const Options & o, const char * engine_archs) {
    set_env("SPEECHCPP_ENGINE_ARCHS", engine_archs);
    transcribe_model_load_params lp;
    transcribe_model_load_params_init(&lp);
    const transcribe_status st = transcribe_model_load_file(o.model.c_str(), &lp, &side.model);
    set_env("SPEECHCPP_ENGINE_ARCHS", "");
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "%s: model load failed: %s\n", side.label, transcribe_status_string(st));
        return false;
    }
    if (!open_session(side, o)) {
        return false;
    }
    std::printf("%s: loaded (arch string \"%s\")\n", side.label, str_or_empty(transcribe_model_arch_string(side.model)).c_str());
    return true;
}

transcribe_run_params run_params(const Options & o) {
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.timestamps = o.timestamps;
    if (!o.language.empty()) {
        rp.language = o.language.c_str();
    }
    return rp;
}

Snapshot run_one(Side & side, const std::vector<float> & pcm, const transcribe_run_params & rp) {
    const auto t0 = std::chrono::steady_clock::now();
    const transcribe_status st = transcribe_run(side.session, pcm.data(), static_cast<int>(pcm.size()), &rp);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    side.total_seconds += seconds;
    return snapshot(side.session, st, seconds);
}

}  // namespace

int main(int argc, char ** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <wav-or-dir>... [--language xx] [--timestamps kind] [--threads n] "
                     "[--max-clips n] [--text-only] [--no-batch] [--p-tol x]\n",
                     argv[0]);
        return 2;
    }
    std::error_code ec;
    if (!fs::exists(o.model, ec)) {
        std::printf("SKIP: model not found: %s\n", o.model.c_str());
        return 2;
    }
    std::vector<std::string> wavs = collect_wavs(o.inputs);
    if (o.max_clips > 0 && wavs.size() > o.max_clips) {
        wavs.resize(o.max_clips);
    }
    if (wavs.empty()) {
        std::printf("SKIP: no .wav fixtures\n");
        return 2;
    }

    Side arch{"arch"};
    Side engine{"engine"};
    if (!open_side(arch, o, "") || !open_side(engine, o, "all")) {
        return 1;
    }

    int failures = 0;
    Diff caps;
    compare_caps(arch.model, engine.model, caps);
    for (const auto & line : caps.lines) {
        std::printf("  CAPS  %s\n", line.c_str());
    }
    failures += static_cast<int>(caps.lines.size());

    const transcribe_run_params rp = run_params(o);
    std::vector<std::vector<float>> clips;
    double audio_seconds = 0.0;
    size_t text_identical = 0;
    for (const auto & path : wavs) {
        int rate = 0;
        std::vector<float> pcm;
        try {
            pcm = abi_test::read_wav_mono_f32(path, rate);
        } catch (const std::exception & ex) {
            std::printf("  skip %s: %s\n", path.c_str(), ex.what());
            continue;
        }
        if (rate != 16000) {
            std::printf("  skip %s: %d Hz (fixtures must be 16 kHz)\n", path.c_str(), rate);
            continue;
        }
        audio_seconds += static_cast<double>(pcm.size()) / 16000.0;
        if (o.fresh_session && (!open_session(arch, o) || !open_session(engine, o))) {
            return 1;
        }
        const Snapshot a = run_one(arch, pcm, rp);
        const Snapshot e = run_one(engine, pcm, rp);
        Diff text, rows;
        compare(a, e, o.p_tol, text, rows);
        const std::string name = fs::path(path).filename().string();
        if (text.empty()) {
            ++text_identical;
        }
        if (text.empty() && rows.empty()) {
            std::printf("  same  %s (arch %.2fs, engine %.2fs)\n", name.c_str(), a.seconds, e.seconds);
        } else {
            std::printf("  DIFF  %s (arch %.2fs, engine %.2fs)\n", name.c_str(), a.seconds, e.seconds);
            for (const auto & line : text.lines) std::printf("    %s\n", line.c_str());
            for (const auto & line : rows.lines) std::printf("    %s%s\n", o.text_only ? "(rows) " : "", line.c_str());
            failures += static_cast<int>(text.lines.size());
            if (!o.text_only) {
                failures += static_cast<int>(rows.lines.size());
            }
        }
        clips.push_back(std::move(pcm));
    }

    if (o.batch && clips.size() > 1 && o.fresh_session && (!open_session(arch, o) || !open_session(engine, o))) {
        return 1;
    }
    if (o.batch && clips.size() > 1) {
        std::vector<const float *> ptrs;
        std::vector<int> lens;
        for (const auto & c : clips) {
            ptrs.push_back(c.data());
            lens.push_back(static_cast<int>(c.size()));
        }
        const transcribe_status sa = transcribe_run_batch(arch.session, ptrs.data(), lens.data(),
                                                          static_cast<int>(clips.size()), &rp);
        const transcribe_status se = transcribe_run_batch(engine.session, ptrs.data(), lens.data(),
                                                          static_cast<int>(clips.size()), &rp);
        if (sa != se) {
            std::printf("  BATCH status: arch %s, engine %s\n", transcribe_status_string(sa), transcribe_status_string(se));
            ++failures;
        } else if (sa == TRANSCRIBE_OK) {
            const int n = std::min(transcribe_batch_n_results(arch.session), transcribe_batch_n_results(engine.session));
            if (transcribe_batch_n_results(arch.session) != transcribe_batch_n_results(engine.session)) {
                std::printf("  BATCH n_results: arch %d, engine %d\n", transcribe_batch_n_results(arch.session),
                            transcribe_batch_n_results(engine.session));
                ++failures;
            }
            int batch_same = 0;
            for (int i = 0; i < n; ++i) {
                const transcribe_status bsa = transcribe_batch_status(arch.session, i);
                const transcribe_status bse = transcribe_batch_status(engine.session, i);
                const std::string ta = str_or_empty(transcribe_batch_full_text(arch.session, i));
                const std::string te = str_or_empty(transcribe_batch_full_text(engine.session, i));
                if (bsa != bse || ta != te) {
                    std::printf("  BATCH[%d]: arch %s \"%s\" | engine %s \"%s\"\n", i, transcribe_status_string(bsa),
                                ta.c_str(), transcribe_status_string(bse), te.c_str());
                    ++failures;
                } else {
                    ++batch_same;
                }
            }
            std::printf("  batch: %d/%d utterances identical\n", batch_same, n);
        }
    }

    std::printf("\nverdict: %zu/%zu clips text-identical; %d difference(s)\n", text_identical, clips.size(), failures);
    if (audio_seconds > 0.0) {
        std::printf("RTF: arch %.3f, engine %.3f (%.1f s of audio, single-shot)\n", arch.total_seconds / audio_seconds,
                    engine.total_seconds / audio_seconds, audio_seconds);
    }
    return failures == 0 ? 0 : 1;
}
