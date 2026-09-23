// engine/models/whisper/session.cpp - offline ASR session + loader for the
// native engine Whisper package (Phase 11 W2a, full recipe since W2b).
//
// Loads both distribution formats: parent transcribe.cpp's GGUF
// (general.architecture = "whisper", published under handy-computer/) and
// whisper.cpp's legacy .bin. can_load() sniffs the file instead of resolving a
// model-spec bundle, because model_specs/whisper.json names packages that
// audio-cpp/audio.cpp-gguf does not host.
//
// Request options (the transcribe_whisper_run_ext fields). Each whisper.* knob
// is also accepted without the prefix - the name model_specs/whisper.json
// declares and the C ABI adapter forwards:
//   language                  ISO code, "" / "auto" = detect (multilingual)
//   task                      transcribe | translate
//   timestamps                none | auto | segment (or the ABI's 0 / 1 / 2)
//   max_tokens                per-window generation cap (default 256)
//   whisper.initial_prompt    text, tokenized as HF get_prompt_ids
//   whisper.prompt_tokens     comma-separated ids (wins over initial_prompt)
//   whisper.prompt_condition  first_segment | all_segments
//   whisper.condition_on_prev_tokens, whisper.max_prev_context_tokens
//   whisper.temperature, whisper.temperature_inc, whisper.seed
//   whisper.compression_ratio_thold, whisper.logprob_thold,
//   whisper.no_speech_thold   ("inf" / "-inf" disable, as the ABI sentinels)
//   whisper.max_initial_timestamp (seconds)

#include "engine/models/whisper/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/models/whisper/runtime.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::whisper {

namespace {

assets::TensorStorageType
weight_type_from_options(const runtime::SessionOptions &options) {
  auto it = options.options.find("whisper.weight_type");
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
    throw std::runtime_error(
        "whisper.weight_type supports native, f32 and f16");
  }
}

// Session option whisper.kv_type: f32 (default) | f16 | auto (the arch /
// whisper.cpp policy: F16 for F16 / quantized decoders; sentinel COUNT).
std::optional<ggml_type> kv_type_from_options(const runtime::SessionOptions &options) {
  auto it = options.options.find("whisper.kv_type");
  if (it == options.options.end()) {
    it = options.options.find("kv_type");
  }
  if (it == options.options.end() || it->second.empty()) {
    return std::nullopt;
  }
  if (it->second == "auto") {
    return GGML_TYPE_COUNT;
  }
  if (it->second == "f32") {
    return GGML_TYPE_F32;
  }
  if (it->second == "f16") {
    return GGML_TYPE_F16;
  }
  throw std::runtime_error("whisper.kv_type must be auto, f32 or f16");
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string strip(const std::string &s) {
  const auto is_space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  };
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

// Tokenizes an initial prompt the way HF's get_prompt_ids does
// (" " + text.strip()), rejecting any special-token literal typed into it -
// the tokenizer would otherwise byte-encode "<|en|>" as ordinary text.
std::vector<int32_t> tokenize_prompt(const WhisperAssets &assets,
                                     const std::string &raw) {
  const std::string text = strip(raw);
  if (text.empty()) {
    return {};
  }
  for (size_t i = 0; i + 1 < text.size();) {
    if (text[i] == '<' && text[i + 1] == '|') {
      const size_t close = text.find("|>", i + 2);
      if (close != std::string::npos) {
        const std::string literal = text.substr(i, close + 2 - i);
        if (assets.special_token_id(literal).has_value()) {
          throw std::invalid_argument(
              "whisper.initial_prompt contains the special token " + literal);
        }
        i = close + 2;
        continue;
      }
    }
    ++i;
  }
  return assets.encode(" " + text);
}

std::vector<int32_t> parse_id_list(const std::string &text, std::string_view key) {
  std::vector<int32_t> ids;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    item = strip(item);
    if (item.empty()) {
      continue;
    }
    try {
      size_t used = 0;
      const long long v = std::stoll(item, &used);
      if (used != item.size() || v < 0 || v > 1'000'000) {
        throw std::invalid_argument("");
      }
      ids.push_back(static_cast<int32_t>(v));
    } catch (const std::exception &) {
      throw std::invalid_argument(std::string(key) + ": not a token id: " + item);
    }
  }
  return ids;
}

WhisperDecodeOptions decode_options_from_request(
    const WhisperAssets &assets,
    const std::unordered_map<std::string, std::string> &options) {
  using runtime::find_option;
  const auto &hp = assets.hparams;
  WhisperDecodeOptions out;
  out.language_candidates = assets.language_token_ids;

  // Language. English-only models take "en" (or nothing) and have no slot.
  const std::string language = lower(strip(find_option(options, {"language"}).value_or("")));
  if (!language.empty() && language != "auto") {
    if (!hp.is_multilingual) {
      if (language != "en") {
        throw std::invalid_argument("whisper: this English-only model cannot transcribe '" +
                                    language + "'");
      }
    } else {
      const auto &codes = assets.language_codes;
      const auto it = std::find(codes.begin(), codes.end(), language);
      if (it == codes.end()) {
        throw std::invalid_argument("whisper: unknown language '" + language + "'");
      }
      out.language_token = assets.language_token_ids[static_cast<size_t>(it - codes.begin())];
    }
  }

  const std::string task = lower(find_option(options, {"task"}).value_or("transcribe"));
  if (task != "transcribe" && task != "translate") {
    throw std::invalid_argument("whisper: task must be transcribe or translate");
  }
  out.translate = (task == "translate");
  // model_specs/whisper.json has always declared a boolean `translate`.
  if (const auto v = find_option(options, {"whisper.translate", "translate"})) {
    out.translate = out.translate || runtime::parse_bool_option(*v, "translate");
  }

  const std::string ts = lower(strip(find_option(options, {"timestamps"}).value_or("none")));
  if (ts == "none" || ts == "0" || ts.empty()) {
    out.timestamps = false;
  } else if (ts == "auto" || ts == "1" || ts == "segment" || ts == "2") {
    out.timestamps = true;
  } else {
    // Word / token timing needs cross-attention DTW, which neither the arch
    // nor this package implements; refuse rather than degrade silently.
    throw std::invalid_argument("whisper: timestamps '" + ts +
                                "' are not supported (segment only)");
  }

  if (const auto max_tokens = runtime::parse_int_option(options, {"max_tokens"})) {
    if (*max_tokens > 0) {
      out.max_new_tokens = *max_tokens;
    }
  }

  // Prompt: pre-tokenized ids win over text (the ABI's documented order).
  if (const auto ids = find_option(options, {"whisper.prompt_tokens", "prompt_tokens"})) {
    out.prompt_ids = parse_id_list(*ids, "whisper.prompt_tokens");
    if (!out.prompt_ids.empty() && out.prompt_ids.front() == hp.prev_sot_token_id) {
      throw std::invalid_argument(
          "whisper.prompt_tokens must not start with <|startofprev|>; it is prepended");
    }
  } else if (const auto text = find_option(options, {"whisper.initial_prompt", "initial_prompt"})) {
    out.prompt_ids = tokenize_prompt(assets, *text);
  }
  if (const auto mode = find_option(options, {"whisper.prompt_condition", "prompt_condition"})) {
    const std::string m = lower(strip(*mode));
    if (m == "all_segments" || m == "all" || m == "1") {
      out.prompt_all_segments = true;
    } else if (m != "first_segment" && m != "first" && m != "0") {
      throw std::invalid_argument("whisper.prompt_condition must be first_segment or all_segments");
    }
  }
  if (const auto v = find_option(options, {"whisper.condition_on_prev_tokens", "condition_on_prev_tokens"})) {
    out.condition_on_prev_tokens =
        runtime::parse_bool_option(*v, "whisper.condition_on_prev_tokens");
  }
  const auto max_prev = runtime::parse_int_option(options, {"whisper.max_prev_context_tokens", "max_prev_context_tokens"});
  out.max_prev_context_tokens = (max_prev.has_value() && *max_prev > 0)
                                    ? *max_prev
                                    : hp.dec_max_target_positions / 2 - 1;

  auto float_opt = [&](const char *prefixed, const char *plain, float &dst) {
    if (const auto v = runtime::parse_float_option(options, {prefixed, plain})) {
      dst = *v;
    }
  };
  float_opt("whisper.temperature", "temperature", out.temperature);
  float_opt("whisper.temperature_inc", "temperature_inc", out.temperature_inc);
  float_opt("whisper.compression_ratio_thold", "compression_ratio_thold", out.compression_ratio_thold);
  float_opt("whisper.logprob_thold", "logprob_thold", out.logprob_thold);
  float_opt("whisper.no_speech_thold", "no_speech_thold", out.no_speech_thold);
  float_opt("whisper.max_initial_timestamp", "max_initial_timestamp", out.max_initial_timestamp);
  if (!(out.temperature >= 0.0f) || !(out.temperature_inc >= 0.0f)) {
    throw std::invalid_argument("whisper: temperatures must be >= 0");
  }
  if (const auto seed = runtime::parse_u32_option(options, {"whisper.seed", "seed"})) {
    out.seed = *seed;
  }
  return out;
}

class WhisperLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "whisper"; }

  std::vector<std::string> family_aliases() const override {
    return {"whisper-offline"};
  }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet result;
    result.supported_tasks = {{runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
    result.languages = {"en"};
    result.supports_timestamps = true;   // segment timestamps (W2b)
    result.timestamp_granularity = runtime::TimestampGranularity::Segment;
    result.supports_cancellation = true; // RunControl polled per window and token
    result.supports_initial_prompt = true;
    result.supports_temperature_fallback = true;
    result.supports_long_form = true;
    // Per checkpoint (load()): translate / detection need a multilingual model.
    result.supports_language_detection = false;
    return result;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value() && *request.family_hint != family() &&
        *request.family_hint != "whisper-offline") {
      return false;
    }
    return looks_like_whisper_bin(request.model_path) ||
           looks_like_whisper_gguf(request.model_path);
  }

  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    runtime::ModelInspection inspection;
    inspection.model_root = request.model_path.parent_path();
    inspection.metadata.family = family();
    inspection.metadata.variant = "whisper";
    inspection.metadata.description =
        "Whisper encoder-decoder ASR (transcribe.cpp GGUF or whisper.cpp .bin).";
    inspection.capabilities = advertised_capabilities();
    inspection.cli.request_options = {
        {"language", "code|auto", "Spoken language; auto detects on multilingual models.", false, "auto"},
        {"task", "transcribe|translate", "Translate to English (multilingual models).", false, "transcribe"},
        {"timestamps", "none|segment", "Emit segment timestamps.", false, "none"},
        {"max_tokens", "n", "Per-window generation cap.", false, "256", "1"},
        {"whisper.initial_prompt", "text", "Context / vocabulary prompt."},
        {"whisper.temperature", "t", "First fallback tier.", false, "0"},
        {"whisper.temperature_inc", "dt", "Fallback step (0 disables fallback).", false, "0.2"},
        {"whisper.seed", "n", "Sampler seed for temperature > 0 (0 = random).", false, "0"},
    };
    inspection.cli.session_options = {
        {"whisper.weight_type", "native|f32|f16", "Shared model weight storage preference."},
        {"whisper.kv_type", "f32|f16|auto", "KV-cache dtype; auto = F16 for F16 / quantized weights."},
    };
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    auto model_assets = load_whisper_assets(request.model_path);
    runtime::ModelMetadata metadata;
    metadata.family = family();
    metadata.variant = model_assets->variant;
    metadata.description = "Whisper encoder-decoder ASR.";
    runtime::CapabilitySet caps = advertised_capabilities();
    caps.languages = model_assets->language_codes;
    caps.supports_language_detection = model_assets->hparams.is_multilingual;
    caps.supports_translate = model_assets->hparams.is_multilingual;
    return std::make_unique<WhisperLoadedModel>(std::move(metadata), std::move(caps),
                                                std::move(model_assets));
  }
};

} // namespace

WhisperSession::WhisperSession(runtime::TaskSpec task, runtime::SessionOptions options,
                               std::shared_ptr<const WhisperAssets> model_assets)
    : RuntimeSessionBase(std::move(options)), task_(std::move(task)),
      assets_(std::move(model_assets)),
      runtime_(std::make_unique<WhisperRuntime>(
          assets_, execution_context(), weight_type_from_options(RuntimeSessionBase::options()),
          kv_type_from_options(RuntimeSessionBase::options()))) {
  if (task_.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("Whisper only supports VoiceTaskKind::Asr");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("Whisper supports offline mode (any length) only");
  }
}

WhisperSession::~WhisperSession() = default;

std::string WhisperSession::family() const { return "whisper"; }

runtime::VoiceTaskKind WhisperSession::task_kind() const { return task_.task; }

runtime::RunMode WhisperSession::run_mode() const { return task_.mode; }

void WhisperSession::prepare(const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error("Whisper prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::TaskResult WhisperSession::run(const runtime::TaskRequest &request) {
  require_prepared("Whisper run()");
  if (!request.audio_input.has_value()) {
    throw std::runtime_error("Whisper run() requires audio_input");
  }

  const auto wall_start = std::chrono::steady_clock::now();
  const WhisperDecodeOptions options = decode_options_from_request(*assets_, request.options);
  const int64_t rate = runtime_->sample_rate();
  const auto to_task_result = [rate](const WhisperTranscription &transcription) {
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{transcription.text, transcription.language};
    for (const auto &seg : transcription.segments) {
      runtime::SpeechSegment out;
      out.span.start_sample = seg.t0_ms * rate / 1000;
      out.span.end_sample = seg.t1_ms * rate / 1000;
      out.text = seg.text;
      result.speech_segments.push_back(std::move(out));
    }
    result.decode_telemetry = transcription.traces;
    result.truncated = transcription.truncated;
    return result;
  };

  WhisperTranscription partial;
  WhisperTranscription transcription;
  try {
    transcription = runtime_->transcribe(*request.audio_input, run_control(), options, &partial);
  } catch (const runtime::ProgressCanceled &canceled) {
    // Hand the windows decoded before the abort to the caller (the C ABI's
    // ERR_ABORTED contract keeps them readable).
    runtime::ProgressCanceled with_partial(canceled.what());
    with_partial.partial = std::make_shared<const runtime::TaskResult>(to_task_result(partial));
    throw with_partial;
  }
  engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
  return to_task_result(transcription);
}

std::vector<runtime::TaskResult>
WhisperSession::run_batch(const std::vector<runtime::TaskRequest> &requests) {
  // Serial per utterance: each request keeps its own options and failure
  // isolation. (The arch's lockstep batched decode is a W2b follow-up.)
  std::vector<runtime::TaskResult> results;
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

WhisperLoadedModel::WhisperLoadedModel(runtime::ModelMetadata metadata,
                                       runtime::CapabilitySet capabilities,
                                       std::shared_ptr<const WhisperAssets> model_assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(model_assets)) {}

const runtime::ModelMetadata &WhisperLoadedModel::metadata() const noexcept { return metadata_; }

const runtime::CapabilitySet &WhisperLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
WhisperLoadedModel::create_task_session(const runtime::TaskSpec &task,
                                        const runtime::SessionOptions &options) const {
  return std::make_unique<WhisperSession>(task, options, assets_);
}

std::optional<std::vector<int32_t>>
WhisperLoadedModel::tokenize(const std::string &text) const {
  return assets_->encode(text);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_whisper_loader() {
  return std::make_shared<WhisperLoader>();
}

} // namespace engine::models::whisper
