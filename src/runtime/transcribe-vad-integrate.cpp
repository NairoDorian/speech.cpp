// transcribe-vad-integrate.cpp - VAD chunk loop + result merging.

#include "transcribe-vad-integrate.h"

#include "transcribe-arch.h"
#include "transcribe-log.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe-vad.h"

#include "engine/framework/audio/activity.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/runtime/model.h"
#include "engine/models/silero_vad/session.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace transcribe::vad {

chunk_baseline snapshot(const transcribe_session & s) {
    return chunk_baseline{ s.segments.size(), s.words.size(), s.tokens.size() };
}

// Offset the entries added since `base` by chunk.keep_span.start_ms, and
// fix up cross-references so word.seg_index / token.seg_index /
// segment.first_word / segment.first_token point at the GLOBAL arrays.
void offset_chunk_results(transcribe_session & s, const chunk_baseline & base, const chunk_plan & chunk) {
    const int64_t dt = chunk.keep_span.start_ms;

    // tokens
    for (size_t i = base.n_tokens; i < s.tokens.size(); ++i) {
        s.tokens[i].t0_ms += dt;
        s.tokens[i].t1_ms += dt;
        s.tokens[i].seg_index += static_cast<int>(base.n_segments);
    }
    // words
    for (size_t i = base.n_words; i < s.words.size(); ++i) {
        s.words[i].t0_ms += dt;
        s.words[i].t1_ms += dt;
        s.words[i].seg_index += static_cast<int>(base.n_segments);
        s.words[i].first_token += static_cast<int>(base.n_tokens);
    }
    // segments
    for (size_t i = base.n_segments; i < s.segments.size(); ++i) {
        s.segments[i].t0_ms += dt;
        s.segments[i].t1_ms += dt;
        s.segments[i].first_word += static_cast<int>(base.n_words);
        s.segments[i].first_token += static_cast<int>(base.n_tokens);
    }
}

void rollback_to(transcribe_session & s, const chunk_baseline & base) {
    if (s.segments.size() > base.n_segments) {
        s.segments.resize(base.n_segments);
    }
    if (s.words.size() > base.n_words) {
        s.words.resize(base.n_words);
    }
    if (s.tokens.size() > base.n_tokens) {
        s.tokens.resize(base.n_tokens);
    }
}

void rebuild_full_text(transcribe_session & s) {
    s.full_text.clear();
    for (size_t i = 0; i < s.segments.size(); ++i) {
        if (i > 0 && !s.full_text.empty() && s.full_text.back() != ' ') {
            s.full_text.push_back(' ');
        }
        s.full_text += s.segments[i].text;
    }
    s.raw_text = s.full_text;
}

std::vector<time_span> detect_speech(const float * pcm, int n_samples, int sample_rate, const struct transcribe_vad_params & vp) {
    std::vector<time_span> out;
    if (pcm == nullptr || n_samples <= 0 || sample_rate <= 0) {
        return out;
    }

    if (vp.mode == TRANSCRIBE_VAD_ENERGY) {
        engine::audio::QuietEnergyAudioChunkOptions opts;
        opts.chunk_samples = static_cast<int64_t>(30.0 * sample_rate);
        opts.boundary_context_samples = static_cast<int64_t>(2.0 * sample_rate);
        opts.min_energy_window_samples = static_cast<int64_t>(0.1 * sample_rate);
        if (opts.chunk_samples <= 0) opts.chunk_samples = sample_rate;

        std::vector<float> mono(pcm, pcm + static_cast<size_t>(n_samples));
        const auto spans = engine::audio::plan_quiet_energy_audio_chunks(mono, opts);
        out.reserve(spans.size());
        for (const auto & sp : spans) {
            time_span ts;
            ts.start_ms = static_cast<int64_t>(sp.start_sample * 1000.0 / sample_rate);
            ts.end_ms = static_cast<int64_t>(sp.end_sample * 1000.0 / sample_rate);
            ts.confidence = 1.0f;
            out.push_back(ts);
        }
        return out;
    }

    if (vp.mode == TRANSCRIBE_VAD_SILERO) {
        static std::mutex s_vad_mutex;
        static std::shared_ptr<engine::runtime::ILoadedVoiceModel> s_silero_model;

        std::lock_guard<std::mutex> lock(s_vad_mutex);
        if (!s_silero_model) {
            engine::runtime::ModelLoadRequest req;
            req.model_path = vp.weight_path ? std::string(vp.weight_path) : "assets/framework/models/silero_vad";
            req.family_hint = "silero_vad";
            try {
                auto loader = engine::models::silero_vad::make_silero_vad_loader();
                if (loader && loader->can_load(req)) {
                    s_silero_model = loader->load(req);
                }
            } catch (...) {
                // fall back
            }
            if (!s_silero_model) {
                try {
                    s_silero_model = engine::models::silero_vad::load_silero_vad_model(req);
                } catch (...) {
                    // degradation signaled by empty return
                }
            }
        }

        if (s_silero_model) {
            engine::runtime::TaskSpec task{engine::runtime::VoiceTaskKind::Vad, engine::runtime::RunMode::Offline};
            engine::runtime::SessionOptions session_opts;
            auto vad_session = s_silero_model->create_task_session(task, session_opts);
            if (vad_session) {
                engine::runtime::TaskRequest treq;
                treq.audio_input = engine::runtime::AudioBuffer{};
                treq.audio_input->sample_rate = sample_rate;
                treq.audio_input->channels = 1;
                treq.audio_input->samples.assign(pcm, pcm + n_samples);

                if (vp.silero_threshold > 0.0f) {
                    treq.options["threshold"] = std::to_string(vp.silero_threshold);
                }
                if (vp.silero_min_speech_ms > 0) {
                    treq.options["min_speech_duration_ms"] = std::to_string(vp.silero_min_speech_ms);
                }
                if (vp.silero_min_silence_ms > 0) {
                    treq.options["min_silence_duration_ms"] = std::to_string(vp.silero_min_silence_ms);
                }

                vad_session->prepare(engine::runtime::build_preparation_request(treq));
                auto * offline_sess = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(vad_session.get());
                if (offline_sess) {
                    auto result = offline_sess->run(treq);
                    out.reserve(result.speech_segments.size());
                    for (const auto & seg : result.speech_segments) {
                        time_span ts;
                        ts.start_ms = static_cast<int64_t>(seg.span.start_sample * 1000.0 / sample_rate);
                        ts.end_ms = static_cast<int64_t>(seg.span.end_sample * 1000.0 / sample_rate);
                        ts.confidence = seg.confidence;
                        out.push_back(ts);
                    }
                }
            }
        }
    }

    return out;
}

transcribe_status run_with_vad(struct transcribe_session *          session,
                               const float *                        pcm,
                               int                                  n_samples,
                               const struct transcribe_run_params * params,
                               bool &                               degraded) {
    degraded = false;

    const transcribe_vad_mode mode = effective_mode(params);
    if (mode == TRANSCRIBE_VAD_OFF) {
        degraded = true;  // caller asked for VAD but field says OFF; degrade safely
        return TRANSCRIBE_OK;
    }

    const auto & vp = params->vad;
    const int sample_rate = 16000;
    std::vector<time_span> speech;
    try {
        speech = detect_speech(pcm, n_samples, sample_rate, vp);
    } catch (const std::exception & e) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "VAD: detection failed (%s); falling back to full-buffer decode",
                            e.what());
        degraded = true;
        return TRANSCRIBE_OK;
    }

    if (speech.empty() && mode == TRANSCRIBE_VAD_SILERO) {
        // If silero returned empty due to model load absence on long audio, degrade safely
        const int64_t total_ms = static_cast<int64_t>(n_samples) * 1000 / sample_rate;
        if (total_ms > 3000) {
            degraded = true;
            return TRANSCRIBE_OK;
        }
    }

    // No speech at all -> empty result, success
    if (speech.empty()) {
        session->has_result = true;
        session->full_text.clear();
        session->raw_text.clear();
        return TRANSCRIBE_OK;
    }

    const int64_t total_ms = static_cast<int64_t>(n_samples) * 1000 / sample_rate;
    int64_t max_chunk = vp.max_chunk_ms;
    if (max_chunk <= 0) {
        transcribe_session_limits lim{};
        transcribe_session_limits_init(&lim);
        if (transcribe_session_get_limits(session, &lim) == TRANSCRIBE_OK && lim.effective_max_audio_ms > 0) {
            max_chunk = lim.effective_max_audio_ms;
        } else {
            max_chunk = 30000;
        }
    }
    const int64_t merge_gap = vp.merge_gap_ms != 0 ? vp.merge_gap_ms : 500;
    const int64_t padding   = vp.padding_ms >= 0 ? vp.padding_ms : 250;

    std::vector<chunk_plan> chunks = plan(speech, total_ms, max_chunk, merge_gap, padding);
    if (chunks.empty()) {
        session->has_result = true;
        session->full_text.clear();
        session->raw_text.clear();
        return TRANSCRIBE_OK;
    }

    const auto * arch = session->model->arch;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto & chunk = chunks[i];
        const int64_t start_samp = std::clamp<int64_t>(chunk.source_span.start_ms * sample_rate / 1000, 0, n_samples);
        const int64_t end_samp   = std::clamp<int64_t>(chunk.source_span.end_ms * sample_rate / 1000, 0, n_samples);
        const int     chunk_len  = static_cast<int>(end_samp - start_samp);
        if (chunk_len <= 0) {
            continue;
        }

        const float * chunk_pcm = pcm + start_samp;
        const chunk_baseline base = snapshot(*session);

        transcribe_status status = arch->run(session, chunk_pcm, chunk_len, params);
        if (status != TRANSCRIBE_OK) {
            rollback_to(*session, base);
            rebuild_full_text(*session);
            return status;
        }

        offset_chunk_results(*session, base, chunk);

        if (session->poll_abort()) {
            rebuild_full_text(*session);
            return TRANSCRIBE_ERR_ABORTED;
        }
    }

    rebuild_full_text(*session);
    session->has_result = true;
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::vad
