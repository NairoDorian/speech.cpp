// engine/models/gigaam/session.cpp - offline ASR session + loader for the
// native engine GigaAM package (port of src/runtime/arch/gigaam/).
//
// Loads the transcribe.cpp GGUF layout (general.architecture = "gigaam", the
// files published under huggingface.co/handy-computer/gigaam-v3-*-gguf).
// can_load() sniffs the file, like the whisper package, because those GGUFs
// carry no audio.cpp model-spec bundle.
//
// Request options (each also accepted with a "gigaam." prefix):
//   language    "" / "auto" / a code from general.languages ("ru"); others are
//               rejected (std::invalid_argument), as the C ABI dispatcher does
//   task        transcribe (translate is rejected: no translate head)
//   timestamps  none | auto | segment | word | token (or the C ABI's 0..4).
//               Validated, otherwise ignored: like the arch, every run
//               publishes its per-token rows (TaskResult::token_timestamps,
//               the arch's 40 ms frame timing, p = 0, no word) - the C ABI
//               then reports TRANSCRIBE_TIMESTAMPS_TOKEN, as the arch did.
// Result: text_output (one leading space trimmed), raw_text (the untrimmed
// decode, when it differs), token_timestamps; truncated is always false.
// Capabilities: TOKEN timestamps, max_audio_ms = 25000 (the arch's advisory
// soft window - longer audio is still decoded whole, with a warning),
// cancellation, languages from general.languages.
// Session options:
//   gigaam.weight_type  native (default: the GGUF's own types) | f32 | f16

#include "engine/models/gigaam/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/models/gigaam/runtime.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace engine::models::gigaam {

namespace {

constexpr const char *kFamily = "gigaam";

assets::TensorStorageType weight_type_from_options(const runtime::SessionOptions &options) {
  auto it = options.options.find("gigaam.weight_type");
  if (it == options.options.end()) {
    it = options.options.find("weight_type");
  }
  if (it == options.options.end()) {
    return assets::TensorStorageType::Native;
  }
  const auto storage_type = assets::parse_tensor_storage_type(it->second);
  switch (storage_type) {
  case assets::TensorStorageType::Native:
  case assets::TensorStorageType::F32:
  case assets::TensorStorageType::F16:
    return storage_type;
  default:
    throw std::runtime_error("gigaam.weight_type supports native, f32 and f16");
  }
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string strip(const std::string &s) {
  const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  size_t a = 0;
  size_t b = s.size();
  while (a < b && is_space(s[a])) {
    ++a;
  }
  while (b > a && is_space(s[b - 1])) {
    --b;
  }
  return s.substr(a, b - a);
}

// Validates a request against what the arch accepted. Nothing in it changes
// the decode: GigaAM has one greedy recipe and always emits token rows.
void validate_request(const GigaamAssets &model_assets,
                      const std::unordered_map<std::string, std::string> &options) {
  using runtime::find_option;

  const std::string language =
      lower(strip(find_option(options, {"gigaam.language", "language"}).value_or("")));
  if (!language.empty() && language != "auto" && !model_assets.languages.empty()) {
    const auto &codes = model_assets.languages;
    if (std::find(codes.begin(), codes.end(), language) == codes.end()) {
      std::string supported;
      for (const auto &code : codes) {
        supported += (supported.empty() ? "" : ", ") + code;
      }
      throw std::invalid_argument("gigaam: unsupported language '" + language +
                                  "' (this model transcribes: " + supported + ")");
    }
  }

  const std::string task =
      lower(strip(find_option(options, {"gigaam.task", "task"}).value_or("transcribe")));
  if (task == "translate") {
    throw std::invalid_argument("gigaam: translate is not supported (monolingual ASR, no "
                                "translate head)");
  }
  if (!task.empty() && task != "transcribe") {
    throw std::invalid_argument("gigaam: task must be transcribe");
  }

  // Any granularity up to TOKEN is accepted and answered with token rows, as
  // the arch answered every request (its max_timestamp_kind is TOKEN).
  const std::string ts =
      lower(strip(find_option(options, {"gigaam.timestamps", "timestamps"}).value_or("none")));
  static const char *const kAccepted[] = {"",        "none", "auto", "segment", "word",
                                          "token",   "0",    "1",    "2",       "3",
                                          "4"};
  if (std::find(std::begin(kAccepted), std::end(kAccepted), ts) == std::end(kAccepted)) {
    throw std::invalid_argument("gigaam: timestamps '" + ts +
                                "' is not one of none, auto, segment, word, token");
  }
}

runtime::TaskResult to_task_result(const GigaamTranscription &transcription, int64_t rate) {
  runtime::TaskResult result;
  result.text_output = runtime::Transcript{transcription.text, ""};
  if (transcription.raw_text != transcription.text) {
    result.raw_text = transcription.raw_text;
  }
  result.truncated = transcription.truncated;
  // The arch's token rows: id, raw piece, [t0, t1) of the emitting encoder
  // frame, no probability (0), not grouped into words.
  result.token_timestamps.reserve(transcription.tokens.size());
  for (const auto &token : transcription.tokens) {
    runtime::TokenTimestamp row;
    row.span.start_sample = token.t0_ms * rate / 1000;
    row.span.end_sample = token.t1_ms * rate / 1000;
    row.id = token.id;
    row.text = token.piece;
    row.probability = 0.0f;
    row.word_index = -1;
    result.token_timestamps.push_back(std::move(row));
  }
  return result;
}

class GigaamLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return kFamily; }

  // The family registry's aliases for "gigaam".
  std::vector<std::string> family_aliases() const override {
    return {"gigaam-asr", "gigaam_v2"};
  }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet result;
    result.supported_tasks = {{runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
    result.languages = {"ru"};
    // Per-token rows, the arch's TRANSCRIBE_TIMESTAMPS_TOKEN.
    result.supports_timestamps = true;
    result.timestamp_granularity = runtime::TimestampGranularity::Token;
    result.supports_translate = false;
    result.supports_language_detection = false;
    result.supports_cancellation = true; // RunControl polled per stage and decode span
    // The arch's advisory soft window (k_safe_audio_ms): trained on <= 25 s;
    // longer input is still decoded whole, with a warning.
    result.max_audio_ms = 25000;
    return result;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value()) {
      const auto &hint = *request.family_hint;
      if (hint != family() && hint != "gigaam-asr" && hint != "gigaam_v2") {
        return false;
      }
    }
    return looks_like_gigaam_gguf(request.model_path);
  }

  runtime::ModelInspection inspect(const runtime::ModelLoadRequest &request) const override {
    runtime::ModelInspection inspection;
    inspection.model_root = request.model_path.parent_path();
    inspection.metadata.family = family();
    inspection.metadata.variant = "gigaam";
    inspection.metadata.description =
        "GigaAM-v3 Conformer ASR (Russian; RNN-T or CTC head), transcribe.cpp GGUF.";
    inspection.capabilities = advertised_capabilities();
    inspection.cli.request_options = {
        {"language", "code|auto", "Spoken language (monolingual: ru).", false, "auto"},
        {"task", "transcribe", "Only transcribe; translate is rejected.", false, "transcribe"},
        {"timestamps", "none|auto|segment|word|token",
         "Validated only: every run returns per-token rows (40 ms encoder frames).", false,
         "none"},
    };
    inspection.cli.session_options = {
        {"gigaam.weight_type", "native|f32|f16", "Encoder weight storage preference.", false,
         "native"},
    };
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    auto model_assets = load_gigaam_assets(request.model_path);
    runtime::ModelMetadata metadata;
    metadata.family = family();
    metadata.variant = model_assets->variant;
    metadata.description = "GigaAM-v3 Conformer ASR.";
    runtime::CapabilitySet caps = advertised_capabilities();
    caps.languages = model_assets->languages;
    caps.supports_language_detection = model_assets->lang_detect;
    return std::make_unique<GigaamLoadedModel>(std::move(metadata), std::move(caps),
                                               std::move(model_assets));
  }
};

} // namespace

GigaamSession::GigaamSession(runtime::TaskSpec task, runtime::SessionOptions options,
                             std::shared_ptr<const GigaamAssets> model_assets)
    : RuntimeSessionBase(std::move(options)), task_(std::move(task)),
      assets_(std::move(model_assets)) {
  if (task_.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("GigaAM only supports VoiceTaskKind::Asr");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("GigaAM supports offline mode only");
  }
  runtime_ = std::make_unique<GigaamRuntime>(
      assets_, execution_context(), weight_type_from_options(RuntimeSessionBase::options()));
}

GigaamSession::~GigaamSession() = default;

std::string GigaamSession::family() const { return kFamily; }

runtime::VoiceTaskKind GigaamSession::task_kind() const { return task_.task; }

runtime::RunMode GigaamSession::run_mode() const { return task_.mode; }

void GigaamSession::prepare(const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error("GigaAM prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::TaskResult GigaamSession::run(const runtime::TaskRequest &request) {
  require_prepared("GigaAM run()");
  if (!request.audio_input.has_value()) {
    throw std::invalid_argument("GigaAM run() requires audio_input");
  }
  validate_request(*assets_, request.options);
  const GigaamTranscription transcription =
      runtime_->transcribe(*request.audio_input, run_control());
  return to_task_result(transcription, runtime_->sample_rate());
}

std::vector<runtime::TaskResult>
GigaamSession::run_batch(const std::vector<runtime::TaskRequest> &requests) {
  require_prepared("GigaAM run_batch()");
  std::vector<runtime::TaskResult> results;
  if (requests.empty()) {
    return results;
  }

  // Shared encoder dispatch when every request is usable (the arch's rule).
  std::vector<const runtime::AudioBuffer *> audios;
  audios.reserve(requests.size());
  bool usable = true;
  for (size_t i = 0; i < requests.size() && usable; ++i) {
    if (!requests[i].audio_input.has_value()) {
      usable = false;
      break;
    }
    try {
      validate_request(*assets_, requests[i].options);
    } catch (const std::invalid_argument &) {
      usable = false;
      break;
    }
    audios.push_back(&*requests[i].audio_input);
  }
  if (usable) {
    std::vector<GigaamTranscription> batch;
    bool batched = true;
    try {
      batch = runtime_->transcribe_batch(audios, run_control());
    } catch (const std::invalid_argument &) {
      batched = false; // a malformed utterance: per-utterance path below
    }
    if (batched) {
      results.reserve(batch.size());
      for (size_t i = 0; i < batch.size(); ++i) {
        results.push_back(to_task_result(batch[i], runtime_->sample_rate()));
      }
      return results;
    }
  }

  // Per-utterance fallback: each request keeps its own failure isolation.
  results.reserve(requests.size());
  for (const auto &request : requests) {
    try {
      results.push_back(run(request));
    } catch (const runtime::ProgressCanceled &) {
      throw;
    } catch (const std::exception &) {
      results.emplace_back();
    }
  }
  return results;
}

GigaamLoadedModel::GigaamLoadedModel(runtime::ModelMetadata metadata,
                                     runtime::CapabilitySet capabilities,
                                     std::shared_ptr<const GigaamAssets> model_assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(model_assets)) {}

const runtime::ModelMetadata &GigaamLoadedModel::metadata() const noexcept { return metadata_; }

const runtime::CapabilitySet &GigaamLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
GigaamLoadedModel::create_task_session(const runtime::TaskSpec &task,
                                       const runtime::SessionOptions &options) const {
  return std::make_unique<GigaamSession>(task, options, assets_);
}

std::optional<std::vector<int32_t>> GigaamLoadedModel::tokenize(const std::string &text) const {
  if (!assets_->text_encoder) {
    return std::nullopt;
  }
  return assets_->text_encoder->encode(text);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_gigaam_loader() {
  return std::make_shared<GigaamLoader>();
}

} // namespace engine::models::gigaam
