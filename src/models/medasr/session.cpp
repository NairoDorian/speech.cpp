// engine/models/medasr/session.cpp - offline ASR session + loader for the
// native engine MedASR (Google LASR-CTC) package.
//
// Loads the transcribe.cpp GGUF (general.architecture = "medasr", published
// under huggingface.co/handy-computer/medasr-gguf). can_load() sniffs the file
// rather than resolving a model-spec bundle, the Whisper loader's rule.
//
// Capabilities are the arch's (src/runtime/arch/medasr/capabilities.cpp):
// English, no language detection, no translation, offline only, cancellation.
// Results are the arch's: TOKEN rows (one per kept CTC frame, 40 ms each, raw
// piece text, p = 0, no word grouping), raw_text = the untrimmed decode, and
// the advisory max_audio_ms (400 s, the RoPE-trained window).
//
// Request options (validated; none changes the result, as in the arch):
//   language    "", "auto" or one of the GGUF's general.languages ("en")
//   task        transcribe (translate is refused: no translation head)
//   timestamps  none | auto | segment | word | token (or 0..4)
//   return_timestamps  bool (the engine CLI's switch)
// Session options:
//   medasr.weight_type / weight_type   native (default) | f32 | f16

#include "engine/models/medasr/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/models/medasr/decoding.h"
#include "engine/models/medasr/runtime.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::medasr {

namespace {

constexpr const char *kFamily = "medasr";

assets::TensorStorageType weight_type_from_options(const runtime::SessionOptions &options) {
  auto it = options.options.find("medasr.weight_type");
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
    throw std::invalid_argument("medasr.weight_type supports native, f32 and f16");
  }
}

std::string lower_strip(std::string s) {
  const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  size_t a = 0;
  size_t b = s.size();
  while (a < b && is_space(s[a])) {
    ++a;
  }
  while (b > a && is_space(s[b - 1])) {
    --b;
  }
  s = s.substr(a, b - a);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// Validates a request before any compute (so a bad request never costs an
// encode). Nothing in it changes the decode: like the arch, every run
// returns the full token rows whatever timestamp granularity was asked for.
void validate_request_options(const MedAsrAssets &assets,
                              const std::unordered_map<std::string, std::string> &options) {
  using runtime::find_option;

  const std::string language = lower_strip(find_option(options, {"language"}).value_or(""));
  if (!language.empty() && language != "auto" && !assets.language_codes.empty()) {
    const auto &codes = assets.language_codes;
    if (std::find(codes.begin(), codes.end(), language) == codes.end()) {
      std::string supported;
      for (const auto &code : codes) {
        supported += (supported.empty() ? "" : ", ") + code;
      }
      throw std::invalid_argument("medasr: language '" + language +
                                  "' is not supported (this model transcribes: " + supported + ")");
    }
  }

  const std::string task = lower_strip(find_option(options, {"task"}).value_or("transcribe"));
  if (task == "translate") {
    throw std::invalid_argument("medasr: translation is not supported (no translation head)");
  }
  if (!task.empty() && task != "transcribe") {
    throw std::invalid_argument("medasr: task must be transcribe");
  }

  const std::string ts = lower_strip(find_option(options, {"timestamps"}).value_or("none"));
  if (!(ts.empty() || ts == "none" || ts == "0" || ts == "auto" || ts == "1" ||
        ts == "segment" || ts == "2" || ts == "word" || ts == "3" || ts == "token" ||
        ts == "4")) {
    throw std::invalid_argument("medasr: timestamps must be none, auto, segment, word or token");
  }
  if (const auto v = find_option(options, {"return_timestamps"})) {
    (void)runtime::parse_bool_option(*v, "return_timestamps"); // validated, rows always returned
  }
}

// The arch's result snapshot: full_text = decode with one leading space
// trimmed, raw_text = the untrimmed decode, and one TOKEN row per kept CTC
// frame {id, raw vocabulary piece, p = 0, [frame*40, frame*40+40) ms,
// word_index = -1}. No word or segment rows (the arch publishes none).
runtime::TaskResult to_task_result(const MedAsrAssets &assets,
                                   const MedAsrTranscription &transcription) {
  runtime::TaskResult result;
  result.text_output = runtime::Transcript{transcription.text, ""};
  result.raw_text = transcription.raw_text;
  const int64_t rate = assets.hparams.fe_sample_rate;
  result.token_timestamps.reserve(transcription.tokens.size());
  for (const CtcToken &token : transcription.tokens) {
    runtime::TokenTimestamp row;
    row.span.start_sample = token.t0_ms * rate / 1000;
    row.span.end_sample = token.t1_ms * rate / 1000;
    row.id = token.id;
    row.text = assets.piece(token.id);
    row.probability = 0.0f;
    row.word_index = -1;
    result.token_timestamps.push_back(std::move(row));
  }
  // CTC consumes the whole input and has no token budget: never truncated.
  // (Past the RoPE-trained window the output may degrade; it is not cut.)
  result.truncated = false;
  return result;
}

class MedAsrLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return kFamily; }

  std::vector<std::string> family_aliases() const override {
    return {"medasr-asr", "medasr_clinical"};
  }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet result;
    result.supported_tasks = {{runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
    result.languages = {"en"};
    result.supports_timestamps = true; // one TOKEN row per kept CTC frame (40 ms)
    result.timestamp_granularity = runtime::TimestampGranularity::Token;
    result.supports_cancellation = true; // RunControl polled per stage / utterance
    result.supports_language_detection = false;
    result.supports_translate = false;
    return result;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value()) {
      const auto aliases = family_aliases();
      const std::string &hint = *request.family_hint;
      if (hint != family() && std::find(aliases.begin(), aliases.end(), hint) == aliases.end()) {
        return false;
      }
    }
    return looks_like_medasr_gguf(request.model_path);
  }

  runtime::ModelInspection inspect(const runtime::ModelLoadRequest &request) const override {
    runtime::ModelInspection inspection;
    inspection.model_root = request.model_path.parent_path();
    inspection.metadata.family = family();
    inspection.metadata.variant = kFamily;
    inspection.metadata.description =
        "MedASR (Google LASR-CTC) Conformer + RoPE CTC ASR, transcribe.cpp GGUF.";
    inspection.capabilities = advertised_capabilities();
    inspection.cli.request_options = {
        {"language", "en|auto", "Spoken language (English only).", false, "auto"},
        {"timestamps", "none|token", "Accepted for the C ABI; token rows (40 ms CTC grid) are "
                                     "always returned, as the arch did.", false, "none"},
    };
    inspection.cli.session_options = {
        {"medasr.weight_type", "native|f32|f16", "Shared model weight storage preference."},
    };
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    auto model_assets = load_medasr_assets(request.model_path);
    runtime::ModelMetadata metadata;
    metadata.family = family();
    metadata.variant = model_assets->variant;
    metadata.description = "MedASR (Google LASR-CTC) medical-dictation ASR.";
    runtime::CapabilitySet caps = advertised_capabilities();
    caps.languages = model_assets->language_codes; // general.languages (may be empty)
    // The arch's advisory soft window: enc_max_pos_emb * 40 ms = 400000 ms for
    // the shipped checkpoint (0 when the GGUF lacks the rate / RoPE range).
    caps.max_audio_ms = model_assets->max_audio_ms();
    return std::make_unique<MedAsrLoadedModel>(std::move(metadata), std::move(caps),
                                               std::move(model_assets));
  }
};

} // namespace

MedAsrSession::MedAsrSession(runtime::TaskSpec task, runtime::SessionOptions options,
                             std::shared_ptr<const MedAsrAssets> model_assets)
    : RuntimeSessionBase(std::move(options)), task_(std::move(task)),
      assets_(std::move(model_assets)) {
  if (task_.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("MedASR only supports VoiceTaskKind::Asr");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("MedASR supports offline mode only (no streaming)");
  }
  runtime_ = std::make_unique<MedAsrRuntime>(
      assets_, execution_context(), weight_type_from_options(RuntimeSessionBase::options()));
}

MedAsrSession::~MedAsrSession() = default;

std::string MedAsrSession::family() const { return kFamily; }

runtime::VoiceTaskKind MedAsrSession::task_kind() const { return task_.task; }

runtime::RunMode MedAsrSession::run_mode() const { return task_.mode; }

void MedAsrSession::prepare(const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error("MedASR prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::TaskResult MedAsrSession::run(const runtime::TaskRequest &request) {
  require_prepared("MedASR run()");
  if (!request.audio_input.has_value()) {
    throw std::invalid_argument("medasr: run() requires audio_input");
  }
  validate_request_options(*assets_, request.options);
  const std::vector<float> pcm = runtime_->prepare_pcm(*request.audio_input);
  const MedAsrTranscription transcription = runtime_->transcribe(pcm, run_control());
  return to_task_result(*assets_, transcription);
}

std::vector<runtime::TaskResult>
MedAsrSession::run_batch(const std::vector<runtime::TaskRequest> &requests) {
  require_prepared("MedASR run_batch()");
  if (requests.empty()) {
    return {};
  }

  // The arch's fast path needs every utterance well-formed; validate first
  // (options, audio, length) without any compute.
  std::vector<std::vector<float>> pcm;
  pcm.reserve(requests.size());
  bool all_ok = true;
  for (const auto &request : requests) {
    try {
      if (!request.audio_input.has_value()) {
        throw std::invalid_argument("medasr: run_batch() request without audio_input");
      }
      validate_request_options(*assets_, request.options);
      pcm.push_back(runtime_->prepare_pcm(*request.audio_input));
      runtime_->validate_length(pcm.back().size());
    } catch (const std::exception &) {
      all_ok = false;
      break;
    }
  }

  std::vector<runtime::TaskResult> results;
  results.reserve(requests.size());
  if (all_ok) {
    // One encoder dispatch over the whole batch; ProgressCanceled propagates.
    const auto transcriptions = runtime_->transcribe_batch(pcm, run_control());
    for (size_t i = 0; i < transcriptions.size(); ++i) {
      results.push_back(to_task_result(*assets_, transcriptions[i]));
    }
    return results;
  }

  // Per-utterance fallback (the arch's malformed-input path): each request
  // keeps its own failure; a failed slot carries an empty result.
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

MedAsrLoadedModel::MedAsrLoadedModel(runtime::ModelMetadata metadata,
                                     runtime::CapabilitySet capabilities,
                                     std::shared_ptr<const MedAsrAssets> model_assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(model_assets)) {}

const runtime::ModelMetadata &MedAsrLoadedModel::metadata() const noexcept { return metadata_; }

const runtime::CapabilitySet &MedAsrLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
MedAsrLoadedModel::create_task_session(const runtime::TaskSpec &task,
                                       const runtime::SessionOptions &options) const {
  return std::make_unique<MedAsrSession>(task, options, assets_);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_medasr_loader() {
  return std::make_shared<MedAsrLoader>();
}

} // namespace engine::models::medasr
