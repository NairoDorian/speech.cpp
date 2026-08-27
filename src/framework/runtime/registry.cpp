#include "engine/framework/runtime/registry.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/family_registry.h"

#include <gguf.h>
#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/models/marblenet_vad/session.h"
#include "engine/models/silero_vad/session.h"

#include "model_registry_includes.inc"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace engine::runtime {

namespace {

std::vector<std::string> split_csv(std::string value) {
    std::vector<std::string> items;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(',', start);
        std::string item = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) { return std::isspace(ch) == 0; }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) { return std::isspace(ch) == 0; }).base(), item.end());
        if (!item.empty()) {
            items.push_back(std::move(item));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return items;
}

// An audio.cpp GGUF names its own family in the `audiocpp.model_spec.family`
// KV (its `general.architecture` is the container tag "audiocpp", identical
// for every family). That name is authoritative and must beat can_load()
// probing: several loaders answer "yes" to any GGUF whose resource bundle
// resolves - Silero VAD among them - so without this, auto-detection returned
// whichever loader happened to be registered first. The symptom was a
// successful load followed by "missing tensor: stft_conv.weight" at run time,
// i.e. a Qwen3-ASR GGUF decoded as Silero VAD.
std::optional<std::string> embedded_gguf_family(const std::filesystem::path & path) {
    if (!engine::io::is_existing_file(path)) {
        return std::nullopt;
    }
    try {
        if (const auto spec = engine::assets::read_gguf_embedded_model_spec(path)) {
            if (!spec->family.empty()) {
                return spec->family;
            }
        }
    } catch (const std::exception &) {
        // Not a GGUF, or no readable metadata: fall back to can_load probing.
    }
    return std::nullopt;
}

// The general.architecture of a GGUF that is NOT an audio.cpp package (those
// name their family through the embedded spec above). Third-party
// conversions - NVIDIA's Sortformer GGUF ("sortformer"), transcribe.cpp's
// ("moonshine", "parakeet", ...) - carry the family only here, and the family
// registry maps every such arch name to exactly one canonical id (§5.5). Nullopt
// for non-GGUF files, unreadable metadata, or an audio.cpp package.
std::optional<std::string> foreign_gguf_architecture(const std::filesystem::path & path) {
    if (!engine::io::is_existing_file(path)) {
        return std::nullopt;
    }
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension != ".gguf") {
        return std::nullopt;
    }
    ggml_context * tensors = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &tensors;
    gguf_context * gguf = gguf_init_from_file(path.string().c_str(), params);
    if (gguf == nullptr) {
        if (tensors != nullptr) ggml_free(tensors);
        return std::nullopt;
    }
    std::optional<std::string> arch;
    const int64_t id = gguf_find_key(gguf, "general.architecture");
    if (id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING) {
        const std::string value = gguf_get_val_str(gguf, id);
        if (!value.empty() && value != "audiocpp") {
            arch = value;
        }
    }
    gguf_free(gguf);
    if (tensors != nullptr) ggml_free(tensors);
    return arch;
}

void log_model_load_trace(const ModelInspection & inspection, const ILoadedVoiceModel & model) {
    if (!engine::debug::trace_log_enabled()) {
        return;
    }
    const auto & metadata = model.metadata();
    engine::debug::trace_log_scalar("runtime.model.family", metadata.family);
    engine::debug::trace_log_scalar("runtime.model.variant", metadata.variant);
    engine::debug::trace_log_scalar("runtime.model.root", inspection.model_root.string());
    engine::debug::trace_log_scalar("runtime.model.discovered_config_count", inspection.discovered_configs.size());
    engine::debug::trace_log_scalar("runtime.model.discovered_weight_count", inspection.discovered_weights.size());
    engine::debug::trace_log_scalar("runtime.model.task_count", model.capabilities().supported_tasks.size());
    engine::debug::trace_log_scalar("runtime.model.language_count", model.capabilities().languages.size());
    engine::debug::trace_log_scalar("runtime.model.supports_speaker_reference", model.capabilities().supports_speaker_reference);
    engine::debug::trace_log_scalar("runtime.model.supports_style_condition", model.capabilities().supports_style_condition);
    engine::debug::trace_log_scalar("runtime.model.supports_timestamps", model.capabilities().supports_timestamps);
}

}  // namespace

void ModelRegistry::register_loader(std::shared_ptr<IVoiceModelLoader> loader) {
    if (loader == nullptr) {
        throw std::invalid_argument("model loader must not be null");
    }
    loaders_.push_back(std::move(loader));
}

bool ModelRegistry::empty() const noexcept {
    return loaders_.empty();
}

size_t ModelRegistry::size() const noexcept {
    return loaders_.size();
}

std::vector<std::string> ModelRegistry::families() const {
    std::vector<std::string> names;
    names.reserve(loaders_.size());
    for (const auto & loader : loaders_) {
        names.push_back(loader->family());
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

bool ModelRegistry::supports_family(const std::string & family) const noexcept {
    for (const auto & loader : loaders_) {
        if (loader->family() == family) {
            return true;
        }
        const auto aliases = loader->family_aliases();
        if (std::find(aliases.begin(), aliases.end(), family) != aliases.end()) {
            return true;
        }
    }
    return false;
}

std::vector<LoaderAdvertisement> ModelRegistry::advertise_loaders() const {
    std::vector<LoaderAdvertisement> out;
    out.reserve(loaders_.size());
    for (const auto & loader : loaders_) {
        if (loader == nullptr) {
            continue;
        }
        out.push_back(loader->advertise());
    }
    std::sort(out.begin(), out.end(), [](const LoaderAdvertisement & a, const LoaderAdvertisement & b) {
        return a.family < b.family;
    });
    return out;
}

ModelInspection ModelRegistry::inspect(const ModelLoadRequest & request) const {
    engine::model_spec::ScopedSpecOverride spec_override(request.model_spec_override, request.model_path);
    validate_request(request);
    const auto * loader = find_loader(request);
    if (loader == nullptr) {
        throw std::runtime_error("no registered model loader can inspect: " + request.model_path.string());
    }
    return loader->inspect(request);
}

ModelInspection ModelRegistry::inspect(const std::filesystem::path & model_path) const {
    ModelLoadRequest request;
    request.model_path = model_path;
    return inspect(request);
}

std::unique_ptr<ILoadedVoiceModel> ModelRegistry::load(const ModelLoadRequest & request) const {
    engine::model_spec::ScopedSpecOverride spec_override(request.model_spec_override, request.model_path);
    validate_request(request);
    const auto * loader = find_loader(request);
    if (loader == nullptr) {
        throw std::runtime_error("no registered model loader can load: " + request.model_path.string());
    }
    const auto inspection = engine::debug::trace_log_enabled()
        ? std::optional<ModelInspection>(loader->inspect(request))
        : std::nullopt;
    auto model = loader->load(request);
    if (inspection.has_value()) {
        log_model_load_trace(*inspection, *model);
    }
    return model;
}

std::unique_ptr<ILoadedVoiceModel> ModelRegistry::load(const std::filesystem::path & model_path) const {
    ModelLoadRequest request;
    request.model_path = model_path;
    return load(request);
}

void ModelRegistry::validate_request(const ModelLoadRequest & request) const {
    if (!engine::io::is_existing_file(request.model_path) && !engine::io::is_existing_directory(request.model_path)) {
        throw std::runtime_error("model path does not exist: " + request.model_path.string());
    }
    if (request.family_hint.has_value() && !supports_family(*request.family_hint)) {
        throw std::runtime_error("unsupported model family hint: " + *request.family_hint);
    }
    if (request.model_spec_override.has_value() &&
        !engine::io::is_existing_file(*request.model_spec_override) &&
        !engine::io::is_existing_directory(*request.model_spec_override)) {
        throw std::runtime_error("model spec override path does not exist: " + request.model_spec_override->string());
    }
}

const IVoiceModelLoader * ModelRegistry::find_loader(const ModelLoadRequest & request) const {
    const auto matches_family = [](const IVoiceModelLoader & loader, const std::string & name) {
        if (loader.family() == name) {
            return true;
        }
        const auto aliases = loader.family_aliases();
        return std::find(aliases.begin(), aliases.end(), name) != aliases.end();
    };

    if (request.family_hint.has_value()) {
        for (const auto & loader : loaders_) {
            if (matches_family(*loader, *request.family_hint)) {
                return loader.get();
            }
        }
        return nullptr;
    }

    // The file may name its own family; trust that over can_load() probing.
    if (const auto embedded = embedded_gguf_family(request.model_path)) {
        for (const auto & loader : loaders_) {
            if (matches_family(*loader, *embedded)) {
                return loader.get();
            }
        }
        // A GGUF that names a family this build does not carry is not a
        // candidate for some other loader's can_load() - say so instead of
        // silently decoding it as the wrong family.
        throw std::runtime_error(
            "model " + request.model_path.string() + " declares family '" + *embedded +
            "', which is not registered in this build");
    }

    // A GGUF written by another tool names its family only through
    // general.architecture. Resolve that through the family registry rather
    // than by probing: several loaders' can_load() accept any GGUF whose
    // resource bundle resolves, so probing decoded NVIDIA's Sortformer package
    // as Silero VAD ("missing tensor: stft_conv.weight") - the same
    // mis-detection the embedded-family check above closed for audio.cpp
    // packages (Phase 10.5, family 3).
    if (const auto arch = foreign_gguf_architecture(request.model_path)) {
        if (const FamilyEntry * entry = resolve_family(*arch)) {
            for (const auto & loader : loaders_) {
                if (matches_family(*loader, std::string(entry->canonical_id))) {
                    return loader.get();
                }
            }
            throw std::runtime_error(
                "model " + request.model_path.string() + " has general.architecture '" + *arch +
                "' (family '" + std::string(entry->canonical_id) + "'), which is not registered in this build");
        }
    }

    for (const auto & loader : loaders_) {
        if (loader->can_load(request)) {
            return loader.get();
        }
    }
    return nullptr;
}

RegistryConfig load_registry_config(const std::filesystem::path & path) {
    const auto config = engine::io::load_config_map(path);
    RegistryConfig registry_config;
    if (const auto it = config.find("families"); it != config.end()) {
        registry_config.enabled_families = split_csv(it->second);
    } else if (const auto it = config.find("loaders"); it != config.end()) {
        registry_config.enabled_families = split_csv(it->second);
    }
    return registry_config;
}

ModelRegistry make_registry_from_config(
    const RegistryConfig & config,
    const std::vector<std::shared_ptr<IVoiceModelLoader>> & available_loaders) {
    ModelRegistry registry;
    if (config.enabled_families.empty()) {
        return registry;
    }

    for (const auto & family : config.enabled_families) {
        auto it = std::find_if(
            available_loaders.begin(),
            available_loaders.end(),
            [&](const std::shared_ptr<IVoiceModelLoader> & loader) {
                return loader != nullptr && loader->family() == family;
            });
        if (it == available_loaders.end()) {
            throw std::runtime_error("registry config requested unknown loader family: " + family);
        }
        registry.register_loader(*it);
    }
    return registry;
}

ModelRegistry make_default_registry(const std::optional<std::filesystem::path> & config_path) {
    const std::vector<std::shared_ptr<IVoiceModelLoader>> available_loaders = {
        engine::models::silero_vad::make_silero_vad_loader(),
        engine::models::marblenet_vad::make_marblenet_vad_loader(),
#include "model_registry_loaders.inc"
    };
    if (!config_path.has_value()) {
        ModelRegistry registry;
        for (const auto & loader : available_loaders) {
            registry.register_loader(loader);
        }
        return registry;
    }
    if (!engine::io::is_existing_file(*config_path)) {
        throw std::runtime_error("registry config path does not exist: " + config_path->string());
    }
    const auto config = load_registry_config(*config_path);
    return make_registry_from_config(config, available_loaders);
}

}  // namespace engine::runtime
