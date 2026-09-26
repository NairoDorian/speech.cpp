// engine/models/voxtral/session.cpp - offline ASR session + loader for the
// native engine Voxtral (2507) package.
//
// Loads the transcribe.cpp GGUF (general.architecture = "voxtral", published
// under huggingface.co/handy-computer/Voxtral-{Mini-3B,Small-24B}-2507-gguf).
// There is no audio.cpp layout of this model, so can_load() sniffs the file
// (the `accepts_foreign_layout` rule of spec_backed_model.h) instead of
// resolving a model-spec resource bundle.
//
// Why a hand-written loader rather than SpecBackedVoiceModelConfig: the model
// spec cannot express supports_translate / supports_language_detection, and
// the arch advertised both (capabilities.cpp: translation via the instruct
// template; stt.capability.lang_detect). The loader therefore takes the
// schema-v1 contract from model_specs/voxtral.json (metadata, options,
// cancellation) and adds the GGUF's capability KVs on top, as the arch's
// load() did (read_capability_kv / read_languages_kv).

#include "engine/models/voxtral/model.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/voxtral/runtime.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::voxtral {

namespace {

constexpr const char *kFamily = "voxtral";

// family_registry.cpp: kAliasesVoxtral.
const std::vector<std::string> &family_alias_list() {
  static const std::vector<std::string> aliases = {"voxtral-offline", "voxtral_small_24b", "voxtral_mini_3b"};
  return aliases;
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

bool contains(const std::vector<std::string> &list, const std::string &value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}

std::string join(const std::vector<std::string> &list) {
  std::string out;
  for (const auto &item : list) {
    out += (out.empty() ? "" : ", ") + item;
  }
  return out;
}

// Session knobs (transcribe_session_params kv_type / n_ctx, as the adapter's
// build_session_options forwards them, or the spec's prefixed names).
VoxtralRuntimeOptions runtime_options(const runtime::SessionOptions &options) {
  VoxtralRuntimeOptions out;
  if (const auto kv = runtime::find_option(options.options, {"voxtral.kv_type", "kv_type"})) {
    const std::string value = strip(*kv);
    if (value.empty() || value == "f16" || value == "auto") {
      out.kv_type = GGML_TYPE_F16;  // the arch's AUTO / F16
    } else if (value == "f32") {
      out.kv_type = GGML_TYPE_F32;
    } else {
      throw std::invalid_argument("voxtral: kv_type must be f16, f32 or auto (got '" + value + "')");
    }
  }
  if (const auto n_ctx = runtime::parse_i64_option(options.options, {"voxtral.n_ctx", "n_ctx"})) {
    if (*n_ctx < 0 || *n_ctx > 0x7fffffff) {
      throw std::invalid_argument("voxtral: n_ctx must be >= 0");
    }
    out.n_ctx = static_cast<int>(*n_ctx);
  }
  // init_context: encoder + decoder flash on, then the global env overrides.
  apply_flash_env_overrides(out);
  return out;
}

// Validates a request before any compute (so a bad request never costs an
// encode) and maps it onto the arch's run() parameters. The language /
// target checks are the C ABI dispatcher's (transcribe.cpp run gates), which
// ran before the arch ever saw the request.
VoxtralRequest request_from_options(const VoxtralAssets &assets,
                                    const std::unordered_map<std::string, std::string> &options) {
  VoxtralRequest out;

  std::string language = strip(runtime::find_option(options, {"language"}).value_or(""));
  if (language == "auto") {
    language.clear();  // no hint: the model detects the language itself
  }
  if (!language.empty() && !assets.languages.empty() && !contains(assets.languages, language)) {
    throw std::invalid_argument("voxtral: language '" + language + "' is not supported (this model lists: " +
                                join(assets.languages) + ")");
  }
  out.language = language;

  const std::string task = strip(runtime::find_option(options, {"task"}).value_or("transcribe"));
  if (task == "translate") {
    if (!assets.supports_translate) {
      throw std::invalid_argument("voxtral: this model does not advertise translation");
    }
    out.translate = true;
  } else if (!task.empty() && task != "transcribe") {
    throw std::invalid_argument("voxtral: task must be transcribe or translate (got '" + task + "')");
  }

  if (out.translate) {
    const std::string target = strip(runtime::find_option(options, {"target_language"}).value_or(""));
    if (!target.empty() && !assets.translate_target_languages.empty() &&
        !contains(assets.translate_target_languages, target)) {
      throw std::invalid_argument("voxtral: translation target '" + target +
                                  "' is not supported (targets: " + join(assets.translate_target_languages) +
                                  ")");
    }
    out.target_language = target;
  }

  // TRANSCRIBE_TIMESTAMPS_NONE is the arch's max_timestamp_kind; AUTO means
  // "best available", which is none here.
  const std::string ts = strip(runtime::find_option(options, {"timestamps"}).value_or("none"));
  if (!(ts.empty() || ts == "none" || ts == "0" || ts == "auto" || ts == "1")) {
    throw std::invalid_argument("voxtral: the model produces no timestamps (timestamps must be none)");
  }
  return out;
}

runtime::CapabilitySet capabilities_for(const model_spec::ModelContract &contract, const VoxtralAssets *assets) {
  runtime::CapabilitySet caps = contract.capabilities;
  // capabilities.cpp: no timestamps of any kind; translation through the
  // instruct template (stt.capability.translate, default true); language
  // detection per stt.capability.lang_detect.
  caps.supports_timestamps = false;
  caps.supports_translate = assets != nullptr ? assets->supports_translate : true;
  caps.supports_language_detection = assets != nullptr ? assets->supports_language_detect : true;
  if (assets != nullptr && !assets->languages.empty()) {
    caps.languages = assets->languages;  // general.languages
  }
  // The arch set TRANSCRIBE_FEATURE_INITIAL_PROMPT, but its run() never read
  // a prompt; the engine does not advertise what it does not implement.
  caps.supports_initial_prompt = false;
  return caps;
}

class VoxtralLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return kFamily; }

  std::vector<std::string> family_aliases() const override { return family_alias_list(); }

  runtime::CapabilitySet advertised_capabilities() const override {
    return capabilities_for(*runtime::require_model_contract(kFamily), nullptr);
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value()) {
      const std::string &hint = *request.family_hint;
      if (hint != kFamily && !contains(family_alias_list(), hint)) {
        return false;
      }
    }
    try {
      return looks_like_voxtral_gguf(request.model_path);
    } catch (const std::exception &) {
      return false;
    }
  }

  runtime::ModelInspection inspect(const runtime::ModelLoadRequest &request) const override {
    const auto contract = runtime::require_model_contract(kFamily);
    runtime::ModelInspection inspection;
    inspection.model_root = request.model_path.parent_path();
    inspection.metadata = contract->metadata;
    inspection.capabilities = capabilities_for(*contract, nullptr);
    inspection.cli = contract->cli;
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest &request) const override {
    auto model_assets = load_voxtral_assets(request.model_path);
    auto contract = runtime::require_model_contract(kFamily);
    runtime::ModelMetadata metadata = contract->metadata;
    metadata.variant = model_assets->variant;
    runtime::CapabilitySet caps = capabilities_for(*contract, model_assets.get());
    return std::make_unique<VoxtralLoadedModel>(std::move(metadata), std::move(caps), std::move(model_assets),
                                                std::move(contract));
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

VoxtralSession::VoxtralSession(const runtime::TaskSpec &task, const runtime::SessionOptions &options,
                               std::shared_ptr<const VoxtralAssets> model_assets,
                               std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), assets_(std::move(model_assets)), contract_(std::move(contract)) {
  if (contract_ == nullptr || assets_ == nullptr) {
    throw std::invalid_argument("voxtral: session requires assets and a model contract");
  }
  runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Voxtral");
  if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("Voxtral requires an offline ASR session (no streaming)");
  }
  runtime_ = std::make_unique<VoxtralRuntime>(assets_, execution_context(), runtime_options(options));
}

VoxtralSession::~VoxtralSession() = default;

std::string VoxtralSession::family() const { return kFamily; }

runtime::VoiceTaskKind VoxtralSession::task_kind() const { return runtime::VoiceTaskKind::Asr; }

runtime::RunMode VoxtralSession::run_mode() const { return runtime::RunMode::Offline; }

void VoxtralSession::prepare(const runtime::SessionPreparationRequest &request) {
  runtime::validate_spec_backed_request_options(request.options, *contract_, "Voxtral");
  mark_prepared();
}

runtime::TaskResult VoxtralSession::run(const runtime::TaskRequest &request) {
  require_prepared("Voxtral run()");
  runtime::validate_spec_backed_request_options(request.options, *contract_, "Voxtral");
  if (!request.audio_input.has_value()) {
    throw std::invalid_argument("voxtral: run() requires audio_input");
  }
  const VoxtralRequest req = request_from_options(*assets_, request.options);
  const std::vector<float> pcm = runtime_->prepare_pcm(*request.audio_input);
  const VoxtralTranscription transcription = runtime_->transcribe(pcm, req, run_control());

  runtime::TaskResult result;
  result.text_output = runtime::Transcript{transcription.text, req.translate ? req.target_language
                                                                           : req.language};
  // Budget reached before EOS: the partial transcript is returned and flagged
  // (the arch's TRANSCRIBE_ERR_OUTPUT_TRUNCATED + transcribe_was_truncated()).
  result.truncated = transcription.truncated;
  return result;
}

// ---------------------------------------------------------------------------
// Loaded model + loader factory
// ---------------------------------------------------------------------------

VoxtralLoadedModel::VoxtralLoadedModel(runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
                                       std::shared_ptr<const VoxtralAssets> model_assets,
                                       std::shared_ptr<const model_spec::ModelContract> contract)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)), assets_(std::move(model_assets)),
      contract_(std::move(contract)) {}

const runtime::ModelMetadata &VoxtralLoadedModel::metadata() const noexcept { return metadata_; }

const runtime::CapabilitySet &VoxtralLoadedModel::capabilities() const noexcept { return capabilities_; }

std::unique_ptr<runtime::IVoiceTaskSession> VoxtralLoadedModel::create_task_session(
    const runtime::TaskSpec &task, const runtime::SessionOptions &options) const {
  return std::make_unique<VoxtralSession>(task, options, assets_, contract_);
}

std::optional<std::vector<int32_t>> VoxtralLoadedModel::tokenize(const std::string &text) const {
  try {
    return assets_->tokenizer.encode(text);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

std::shared_ptr<runtime::IVoiceModelLoader> make_voxtral_loader() { return std::make_shared<VoxtralLoader>(); }

}  // namespace engine::models::voxtral
