#pragma once

#include "engine/framework/runtime/session.h"

#include <filesystem>
#include <optional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::model_spec {
enum class ResourceKind;
}

namespace engine::runtime {

struct NamedAsset {
    std::string id;
    std::filesystem::path path;
};

struct TaskCapability {
    VoiceTaskKind task = VoiceTaskKind::Vad;
    std::vector<RunMode> modes;
};

// Finest timing an ASR family returns when supports_timestamps is set.
// Token: per-token rows (TaskResult::token_timestamps) as CTC / TDT
// families time them - the transcribe.cpp arches' TRANSCRIBE_TIMESTAMPS_TOKEN.
enum class TimestampGranularity { Segment, Word, Token };

struct CapabilitySet {
    std::vector<TaskCapability> supported_tasks;
    std::vector<std::string> languages;
    bool supports_speaker_reference = false;
    bool supports_style_condition = false;
    bool supports_timestamps = false;
    // Whisper decodes segment timestamps only (word timing would need
    // cross-attention DTW); the C ABI must then reject WORD requests.
    TimestampGranularity timestamp_granularity = TimestampGranularity::Word;

    // ASR decode features the C ABI adapter publishes as capability fields
    // and TRANSCRIBE_FEATURE_* bits (added for the Whisper takeover, W2b.2).
    bool supports_translate = false;
    // Spoken-language identification. Unset keeps the adapter's historical
    // inference (any language list means detection); an English-only Whisper
    // lists {"en"} but cannot detect, so it says false explicitly.
    std::optional<bool> supports_language_detection = std::nullopt;
    bool supports_initial_prompt = false;
    bool supports_temperature_fallback = false;
    bool supports_long_form = false;
    // The offline autoregressive decode honours the spec_k_drafts request
    // option (n-gram-lookup speculative decoding). Mirrors
    // transcribe_capabilities::supports_spec_decode across the C ABI adapter.
    bool supports_speculative_decode = false;
    // run()/run_batch() poll RunControl at stage and decode-step boundaries,
    // so request_abort() (or a declining progress callback) unwinds them
    // promptly. Mirrors TRANSCRIBE_FEATURE_CANCELLATION across the C ABI.
    bool supports_cancellation = false;
    // Longest input one run accepts, in milliseconds; 0 = unbounded (the
    // session chunks internally). Mirrors transcribe_capabilities::max_audio_ms,
    // which the C ABI adapter used to hard-code to 0 for every engine family.
    int64_t max_audio_ms = 0;
};

struct ModelMetadata {
    std::string family;
    std::string variant;
    std::string description;
    std::vector<std::string> config_candidates;
    std::vector<std::string> weight_candidates;
};

struct CliOptionInfo {
    std::string name;
    std::string value_name;
    std::string description;
    bool required = false;
    std::optional<std::string> default_value = std::nullopt;
    std::optional<std::string> min_value = std::nullopt;
    std::optional<std::string> max_value = std::nullopt;
};

struct ModelCliInterface {
    std::vector<CliOptionInfo> request_options;
    std::vector<CliOptionInfo> session_options;
    std::vector<CliOptionInfo> load_options;
};

struct ModelInspection {
    ModelMetadata metadata;
    CapabilitySet capabilities;
    ModelCliInterface cli;
    std::filesystem::path model_root;
    std::vector<NamedAsset> discovered_configs;
    std::vector<NamedAsset> discovered_weights;
};

struct ModelLoadRequest {
    std::filesystem::path model_path;
    std::optional<std::filesystem::path> model_spec_override = std::nullopt;
    std::optional<std::string> family_hint = std::nullopt;
    std::optional<std::string> config_id = std::nullopt;
    std::optional<std::string> weight_id = std::nullopt;
    std::unordered_map<std::string, std::string> options;
};

std::vector<NamedAsset> discover_named_assets(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::vector<NamedAsset> discover_named_assets_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    engine::model_spec::ResourceKind kind);

const NamedAsset * find_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id) noexcept;

const NamedAsset & require_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id,
    const std::string & role);

const NamedAsset * select_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::optional<std::string> & id,
    const std::string & role) noexcept(false);

class ILoadedVoiceModel {
public:
    virtual ~ILoadedVoiceModel() = default;

    virtual const ModelMetadata & metadata() const noexcept = 0;
    virtual const CapabilitySet & capabilities() const noexcept = 0;
    virtual std::unique_ptr<IVoiceTaskSession> create_task_session(
        const TaskSpec & task,
        const SessionOptions & options) const = 0;

    // Text -> the model's token ids (what transcribe_tokenize() returns through
    // the C ABI). Optional: nullopt when the family exposes no text tokenizer.
    virtual std::optional<std::vector<int32_t>> tokenize(const std::string & text) const {
        (void) text;
        return std::nullopt;
    }
};

struct LoaderAdvertisement {
    std::string family;
    CapabilitySet capabilities;
    std::string instructions_policy;
    std::vector<std::string> api_endpoints;
};

/** Map advertised capabilities to the HTTP surfaces they normally use. */
std::vector<std::string> default_api_endpoints_for_capabilities(const CapabilitySet & capabilities);

class IVoiceModelLoader {
public:
    virtual ~IVoiceModelLoader() = default;

    virtual std::string family() const = 0;
    /** Alternate family names accepted for family hints (e.g. "habibi" for "f5_tts"). */
    virtual std::vector<std::string> family_aliases() const { return {}; }
    virtual bool can_load(const ModelLoadRequest & request) const = 0;
    virtual ModelInspection inspect(const ModelLoadRequest & request) const = 0;
    virtual std::unique_ptr<ILoadedVoiceModel> load(const ModelLoadRequest & request) const = 0;

    /**
     * Path-free loader catalog for ``--list-loaders --json``.
     * Override ``advertised_capabilities`` (and policy when non-default) on each loader.
     */
    virtual CapabilitySet advertised_capabilities() const;
    virtual std::string advertised_instructions_policy() const;
    virtual std::vector<std::string> advertised_api_endpoints() const;
    LoaderAdvertisement advertise() const;
};

}  // namespace engine::runtime
