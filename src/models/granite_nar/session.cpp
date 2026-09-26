// engine/models/granite_nar/session.cpp - offline ASR session + loader for the
// native engine Granite Speech NAR package (port of src/runtime/arch/granite_nar).
//
// Loads parent transcribe.cpp's GGUF (general.architecture =
// "granite_speech_nar", huggingface.co/handy-computer/
// granite-speech-4.1-2b-nar-gguf). can_load() sniffs the file rather than
// resolving a model-spec bundle: those GGUFs are third-party and embed no
// sidecars.
//
// Capabilities are the arch's (capabilities.cpp apply_family_invariants plus
// read_capability_kv / read_languages_kv): ASR only, no timestamps, no
// translation / language detection unless the GGUF says so (it says no),
// offline only, cancellation, and the advisory max_audio_ms
// (granite_nar_max_audio_ms: (4096 - 256) / 3 windows * 15 frames * 2 mel
// frames * 10 ms = 384 s for the shipped checkpoint).
//
// Request options. The arch's run() read none of transcribe_run_params
// ("NLE has no language / task knobs"):
//   language    accepted and ignored (the C ABI validates it against
//               general.languages before dispatch)
//   task        transcribe (translate -> std::invalid_argument)
//   timestamps  accepted and ignored: one untimed transcript
// Session options:
//   n_ctx                   lowers the editor context ceiling (0 = model max)
//   granite_nar.shaw_bias   skew (default, transcribe.cpp 585b98f7) | direct
//                           (speech.cpp's pre-585b98f7 arch; diagnostic)

#include "engine/models/granite_nar/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/granite_nar/runtime.h"

#include <algorithm>
#include <cctype>
#include <new>
#include <stdexcept>
#include <utility>

namespace engine::models::granite_nar {

namespace {

constexpr const char *kFamily = "granite_nar";

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

GraniteNarRuntimeOptions runtime_options_from(const runtime::SessionOptions &options) {
  GraniteNarRuntimeOptions out;
  if (const auto n_ctx = runtime::parse_int_option(options.options, {"granite_nar.n_ctx", "n_ctx"})) {
    if (*n_ctx < 0) {
      throw std::invalid_argument("granite_nar.n_ctx must be >= 0");
    }
    out.n_ctx = *n_ctx;
  }
  if (const auto bias = runtime::find_option(options.options, {"granite_nar.shaw_bias", "shaw_bias"})) {
    const std::string v = lower(*bias);
    if (v.empty() || v == "skew") {
      out.shaw_bias = ShawBias::Skew;
    } else if (v == "direct") {
      out.shaw_bias = ShawBias::Direct;
    } else {
      throw std::invalid_argument("granite_nar.shaw_bias must be skew or direct");
    }
  }
  return out;
}

void check_request_options(const runtime::TaskRequest &request) {
  if (const auto task = runtime::find_option(request.options, {"task"})) {
    const std::string t = lower(*task);
    if (!t.empty() && t != "transcribe") {
      throw std::invalid_argument("granite_nar: task '" + *task +
                                  "' is not supported (transcription only)");
    }
  }
}

std::vector<float> pcm_mono(const runtime::AudioBuffer &audio, int32_t sample_rate) {
  if (audio.samples.empty()) {
    throw std::invalid_argument("granite_nar: empty audio input");
  }
  if (audio.channels == 1 && audio.sample_rate == sample_rate) {
    return audio.samples; // the C ABI's buffer, bit for bit
  }
  if (audio.sample_rate <= 0 || audio.channels <= 0) {
    throw std::invalid_argument("granite_nar: audio needs a positive sample rate and channel count");
  }
  return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
      audio.samples, audio.sample_rate, audio.channels, sample_rate);
}

runtime::TaskResult to_task_result(const GraniteNarTranscription &t) {
  runtime::TaskResult result;
  // No language detection; full_text == raw_text in the arch (no
  // post-processing), and there is no decode budget, so never truncated.
  result.text_output = runtime::Transcript{t.text, ""};
  result.raw_text = t.text;
  result.truncated = false;
  return result;
}

runtime::CapabilitySet base_capabilities() {
  runtime::CapabilitySet caps;
  caps.supported_tasks = {{runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
  caps.supports_timestamps = false; // TRANSCRIBE_TIMESTAMPS_NONE
  caps.supports_cancellation = true; // RunControl polled per stage / BPE-CTC chunk
  caps.supports_translate = false;
  caps.supports_language_detection = false;
  caps.supports_initial_prompt = false;
  caps.supports_temperature_fallback = false;
  caps.supports_long_form = false; // hard context cap, no chunker
  return caps;
}

class GraniteNarLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return kFamily; }

  std::vector<std::string> family_aliases() const override {
    // "granite_speech_nar" is the GGUF general.architecture / catalog family.
    return {"granite_speech_nar", "granite-nar", "granite_nar_asr"};
  }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet caps = base_capabilities();
    caps.languages = {"en", "fr", "de", "es", "pt"}; // the converter's general.languages
    return caps;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value()) {
      const auto aliases = family_aliases();
      if (*request.family_hint != family() &&
          std::find(aliases.begin(), aliases.end(), *request.family_hint) == aliases.end()) {
        return false;
      }
    }
    return looks_like_granite_nar_gguf(request.model_path);
  }

  runtime::ModelInspection inspect(const runtime::ModelLoadRequest &request) const override {
    runtime::ModelInspection inspection;
    inspection.model_root = request.model_path.parent_path();
    inspection.metadata.family = family();
    inspection.metadata.variant = "granite-speech-4.1-2b-nar";
    inspection.metadata.description =
        "IBM Granite Speech 4.1 2B NAR (conformer CTC + Q-Former + bidirectional Granite-4 "
        "editor), transcribe.cpp GGUF.";
    inspection.capabilities = advertised_capabilities();
    inspection.cli.request_options = {
        {"language", "en|fr|de|es|pt", "Accepted and ignored (the model does not condition on it).",
         false, "auto"},
    };
    inspection.cli.session_options = {
        {"n_ctx", "n", "Lower the editor context ceiling (0 = model maximum).", false, "0", "0"},
        {"granite_nar.shaw_bias", "skew|direct",
         "Shaw positional-bias form: skew (default) or the older direct lookup.", false, "skew"},
    };
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    auto model_assets = load_granite_nar_assets(request.model_path);
    runtime::ModelMetadata metadata;
    metadata.family = family();
    metadata.variant = model_assets->variant;
    metadata.description = "IBM Granite Speech NAR (non-autoregressive editor) speech recognition.";
    runtime::CapabilitySet caps = base_capabilities();
    caps.languages = model_assets->languages;
    caps.supports_translate = model_assets->supports_translate;
    caps.supports_language_detection = model_assets->supports_language_detect;
    caps.max_audio_ms = model_assets->max_audio_ms;
    return std::make_unique<GraniteNarLoadedModel>(std::move(metadata), std::move(caps),
                                                   std::move(model_assets));
  }
};

} // namespace

GraniteNarSession::GraniteNarSession(runtime::TaskSpec task, runtime::SessionOptions options,
                                     std::shared_ptr<const GraniteNarAssets> model_assets)
    : RuntimeSessionBase(std::move(options)), task_(std::move(task)),
      assets_(std::move(model_assets)) {
  if (task_.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("granite_nar only supports VoiceTaskKind::Asr");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("granite_nar supports offline mode only");
  }
  runtime_ = std::make_unique<GraniteNarRuntime>(assets_, execution_context(),
                                                 runtime_options_from(RuntimeSessionBase::options()));
}

GraniteNarSession::~GraniteNarSession() = default;

std::string GraniteNarSession::family() const { return kFamily; }

runtime::VoiceTaskKind GraniteNarSession::task_kind() const { return task_.task; }

runtime::RunMode GraniteNarSession::run_mode() const { return task_.mode; }

void GraniteNarSession::prepare(const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error("granite_nar prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::TaskResult GraniteNarSession::run(const runtime::TaskRequest &request) {
  require_prepared("granite_nar run()");
  if (!request.audio_input.has_value()) {
    throw std::invalid_argument("granite_nar: run() requires audio_input");
  }
  check_request_options(request);
  const std::vector<float> pcm = pcm_mono(*request.audio_input, runtime_->sample_rate());
  return to_task_result(runtime_->transcribe(pcm, run_control()));
}

std::vector<runtime::TaskResult>
GraniteNarSession::run_batch(const std::vector<runtime::TaskRequest> &requests) {
  require_prepared("granite_nar run_batch()");
  std::vector<runtime::TaskResult> results;
  results.reserve(requests.size());
  for (const auto &request : requests) {
    runtime::TaskResult result;
    try {
      result = run(request);
    } catch (const runtime::ProgressCanceled &) {
      throw;
    } catch (const runtime::InputTooLong &e) {
      result = runtime::TaskResult{};
      result.item_status = runtime::TaskItemStatus::InputTooLong;
      result.item_error = e.what();
    } catch (const std::invalid_argument &e) {
      result = runtime::TaskResult{};
      result.item_status = runtime::TaskItemStatus::InvalidArgument;
      result.item_error = e.what();
    } catch (const std::bad_alloc &e) {
      result = runtime::TaskResult{};
      result.item_status = runtime::TaskItemStatus::OutOfMemory;
      result.item_error = e.what();
    } catch (const std::exception &e) {
      result = runtime::TaskResult{};
      result.item_status = runtime::TaskItemStatus::Failed;
      result.item_error = e.what();
    }
    results.push_back(std::move(result));
  }
  return results;
}

GraniteNarLoadedModel::GraniteNarLoadedModel(runtime::ModelMetadata metadata,
                                             runtime::CapabilitySet capabilities,
                                             std::shared_ptr<const GraniteNarAssets> model_assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(model_assets)) {}

const runtime::ModelMetadata &GraniteNarLoadedModel::metadata() const noexcept { return metadata_; }

const runtime::CapabilitySet &GraniteNarLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
GraniteNarLoadedModel::create_task_session(const runtime::TaskSpec &task,
                                           const runtime::SessionOptions &options) const {
  return std::make_unique<GraniteNarSession>(task, options, assets_);
}

std::optional<std::vector<int32_t>> GraniteNarLoadedModel::tokenize(const std::string &text) const {
  return assets_->encode(text);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_granite_nar_loader() {
  return std::make_shared<GraniteNarLoader>();
}

} // namespace engine::models::granite_nar
