#include "engine/framework/runtime/family_registry.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace engine::runtime {

namespace {

// Static alias and gguf_arch arrays
static constexpr std::string_view kAliasesSileroVad[] = {"silero-vad", "silero"};
static constexpr std::string_view kGgufArchsSileroVad[] = {"silero_vad"};

static constexpr std::string_view kAliasesMarblenetVad[] = {"marblenet-vad", "marblenet"};
static constexpr std::string_view kGgufArchsMarblenetVad[] = {"marblenet_vad"};

static constexpr std::string_view kAliasesCitrinetAsr[] = {"citrinet", "citrinet-asr"};
static constexpr std::string_view kGgufArchsCitrinetAsr[] = {"citrinet_asr", "citrinet"};

static constexpr std::string_view kAliasesNemotronAsr[] = {"nemotron", "nemotron-asr"};
static constexpr std::string_view kGgufArchsNemotronAsr[] = {"nemotron_asr"};

static constexpr std::string_view kAliasesQwen3Asr[] = {"qwen3-asr", "qwen-asr", "qwen3_asr_gguf"};
static constexpr std::string_view kGgufArchsQwen3Asr[] = {"qwen3_asr"};

static constexpr std::string_view kAliasesFunAsrNano[] = {"funasr-nano", "funasr_nano", "fun_asr"};
static constexpr std::string_view kGgufArchsFunAsrNano[] = {"fun_asr_nano", "funasr_nano"};

static constexpr std::string_view kAliasesSenseAsr[] = {"sensevoice", "sense-asr", "sense_voice"};
static constexpr std::string_view kGgufArchsSenseAsr[] = {"sense_asr", "sensevoice"};

static constexpr std::string_view kAliasesHviskeAsr[] = {"hviske", "hviske-asr"};
static constexpr std::string_view kGgufArchsHviskeAsr[] = {"hviske_asr"};

static constexpr std::string_view kAliasesHiggsAudioStt[] = {"higgs-stt", "higgs_stt", "higgs-audio-stt"};
static constexpr std::string_view kGgufArchsHiggsAudioStt[] = {"higgs_audio_stt"};

static constexpr std::string_view kAliasesVibevoiceAsr[] = {"vibevoice-asr", "vibevoice_asr_stt"};
static constexpr std::string_view kGgufArchsVibevoiceAsr[] = {"vibevoice_asr"};

static constexpr std::string_view kAliasesKrokoAsr[] = {"kroko", "kroko-asr"};
static constexpr std::string_view kGgufArchsKrokoAsr[] = {"kroko_asr"};

static constexpr std::string_view kAliasesParakeetTdt[] = {"parakeet", "parakeet-tdt", "parakeet_ctc"};
static constexpr std::string_view kGgufArchsParakeetTdt[] = {"parakeet_tdt", "parakeet"};

static constexpr std::string_view kAliasesVoxtralRealtime[] = {"voxtral-realtime", "voxtral_rt"};
static constexpr std::string_view kGgufArchsVoxtralRealtime[] = {"voxtral_realtime"};

static constexpr std::string_view kAliasesVoxtral[] = {"voxtral-offline", "voxtral_small_24b", "voxtral_mini_3b"};
static constexpr std::string_view kGgufArchsVoxtral[] = {"voxtral"};

static constexpr std::string_view kAliasesSortformerDiar[] = {"sortformer", "sortformer-diar"};
static constexpr std::string_view kGgufArchsSortformerDiar[] = {"sortformer_diar", "sortformer"};

static constexpr std::string_view kAliasesRoformer[] = {"roformer-sep", "bs_roformer"};
static constexpr std::string_view kGgufArchsRoformer[] = {"roformer"};

static constexpr std::string_view kAliasesMoss[] = {"moss-asr", "moss_diar", "moss_stt"};
static constexpr std::string_view kGgufArchsMoss[] = {"moss"};

static constexpr std::string_view kAliasesWhisper[] = {"whisper-asr", "openai_whisper"};
static constexpr std::string_view kGgufArchsWhisper[] = {"whisper"};

static constexpr std::string_view kAliasesMoonshine[] = {"moonshine-offline"};
static constexpr std::string_view kGgufArchsMoonshine[] = {"moonshine"};

static constexpr std::string_view kAliasesMoonshineStreaming[] = {"moonshine-stream", "moonshine-streaming"};
static constexpr std::string_view kGgufArchsMoonshineStreaming[] = {"moonshine_streaming"};

static constexpr std::string_view kAliasesCanary[] = {"canary-asr", "canary-1b"};
static constexpr std::string_view kGgufArchsCanary[] = {"canary"};

static constexpr std::string_view kAliasesCanaryQwen[] = {"canary-qwen", "canary_qwen_asr"};
static constexpr std::string_view kGgufArchsCanaryQwen[] = {"canary_qwen"};

static constexpr std::string_view kAliasesCohere[] = {"cohere-asr", "cohere_transcribe"};
static constexpr std::string_view kGgufArchsCohere[] = {"cohere"};

static constexpr std::string_view kAliasesGigaam[] = {"gigaam-asr", "gigaam_v2"};
static constexpr std::string_view kGgufArchsGigaam[] = {"gigaam"};

static constexpr std::string_view kAliasesGranite[] = {"granite-asr", "granite_speech"};
static constexpr std::string_view kGgufArchsGranite[] = {"granite"};

static constexpr std::string_view kAliasesGraniteNar[] = {"granite-nar", "granite_nar_asr"};
static constexpr std::string_view kGgufArchsGraniteNar[] = {"granite_nar"};

static constexpr std::string_view kAliasesMedasr[] = {"medasr-asr", "medasr_clinical"};
static constexpr std::string_view kGgufArchsMedasr[] = {"medasr"};

// TTS and Generation Models
static constexpr std::string_view kAliasesAceStep[] = {"ace-step"};
static constexpr std::string_view kAliasesChatterbox[] = {"chatterbox-tts"};
static constexpr std::string_view kAliasesConfucius4Tts[] = {"confucius4"};
static constexpr std::string_view kAliasesDemucs[] = {"demucs-sep"};
static constexpr std::string_view kAliasesDotsTts[] = {"dots"};
static constexpr std::string_view kAliasesDramabox[] = {"dramabox-tts"};
static constexpr std::string_view kAliasesFishAudio[] = {"fish-audio", "fish_tts"};
static constexpr std::string_view kAliasesHeartmula[] = {"heartmula-tts"};
static constexpr std::string_view kAliasesHiggsAudioTts[] = {"higgs-tts"};
static constexpr std::string_view kAliasesIndexTts2[] = {"index-tts", "index_tts"};
static constexpr std::string_view kAliasesIrodoriTts[] = {"irodori"};
static constexpr std::string_view kAliasesMagpieTts[] = {"magpie"};
static constexpr std::string_view kAliasesMeanvc2[] = {"meanvc", "mean-vc"};
static constexpr std::string_view kAliasesMiocodec[] = {"miocodec-codec"};
static constexpr std::string_view kAliasesMiotts[] = {"miotts-tts"};
static constexpr std::string_view kAliasesMuscriptor[] = {"muscriptor-midi"};
static constexpr std::string_view kAliasesNeutts[] = {"neutts-tts"};
static constexpr std::string_view kAliasesOmnivoice[] = {"omnivoice-s2s"};
static constexpr std::string_view kAliasesPersonaplex[] = {"personaplex-tts"};
static constexpr std::string_view kAliasesPocketTts[] = {"pocket"};
static constexpr std::string_view kAliasesQwen3ForcedAligner[] = {"qwen3-aligner"};
static constexpr std::string_view kAliasesQwen3Tts[] = {"qwen3-tts"};
static constexpr std::string_view kAliasesRvc[] = {"rvc-vc"};
static constexpr std::string_view kAliasesSeedVc[] = {"seed-vc"};
static constexpr std::string_view kAliasesStableAudio[] = {"stable-audio"};
static constexpr std::string_view kAliasesSupertonic[] = {"supertonic-tts"};
static constexpr std::string_view kAliasesVevo2[] = {"vevo"};
static constexpr std::string_view kAliasesVibevoice[] = {"vibevoice-tts"};
static constexpr std::string_view kAliasesVoxcpm2[] = {"voxcpm"};

// Community models
static constexpr std::string_view kAliasesF5Tts[] = {"f5-tts"};
static constexpr std::string_view kAliasesGlmTts[] = {"glm-tts"};
static constexpr std::string_view kAliasesInflectV2[] = {"inflect"};
static constexpr std::string_view kAliasesMinimaxH3[] = {"minimax-h3"};
static constexpr std::string_view kAliasesMinimaxMusic3[] = {"minimax-music"};
static constexpr std::string_view kAliasesMmsForcedAligner[] = {"mms-aligner"};
static constexpr std::string_view kAliasesMossVoicegen[] = {"moss-voicegen"};
static constexpr std::string_view kAliasesOutetts[] = {"outetts-tts"};
static constexpr std::string_view kAliasesVietneuTts[] = {"vietneu"};

#define ARR_DESC(arr) arr, (sizeof(arr)/sizeof((arr)[0]))
#define EMPTY_DESC nullptr, 0

static const FamilyEntry kStaticFamilies[] = {
    // VAD
    {"silero_vad", ARR_DESC(kAliasesSileroVad), ARR_DESC(kGgufArchsSileroVad), "model_specs/silero_vad.json", VoiceTaskKind::Vad},
    {"marblenet_vad", ARR_DESC(kAliasesMarblenetVad), ARR_DESC(kGgufArchsMarblenetVad), "model_specs/marblenet_vad.json", VoiceTaskKind::Vad},

    // ASR
    {"citrinet_asr", ARR_DESC(kAliasesCitrinetAsr), ARR_DESC(kGgufArchsCitrinetAsr), "model_specs/citrinet_asr.json", VoiceTaskKind::Asr},
    {"nemotron_asr", ARR_DESC(kAliasesNemotronAsr), ARR_DESC(kGgufArchsNemotronAsr), "model_specs/nemotron_asr.json", VoiceTaskKind::Asr},
    {"qwen3_asr", ARR_DESC(kAliasesQwen3Asr), ARR_DESC(kGgufArchsQwen3Asr), "model_specs/qwen3_asr.json", VoiceTaskKind::Asr},
    {"fun_asr_nano", ARR_DESC(kAliasesFunAsrNano), ARR_DESC(kGgufArchsFunAsrNano), "model_specs/fun_asr_nano.json", VoiceTaskKind::Asr},
    {"sense_asr", ARR_DESC(kAliasesSenseAsr), ARR_DESC(kGgufArchsSenseAsr), "model_specs/sense_asr.json", VoiceTaskKind::Asr},
    {"hviske_asr", ARR_DESC(kAliasesHviskeAsr), ARR_DESC(kGgufArchsHviskeAsr), "model_specs/hviske_asr.json", VoiceTaskKind::Asr},
    {"higgs_audio_stt", ARR_DESC(kAliasesHiggsAudioStt), ARR_DESC(kGgufArchsHiggsAudioStt), "model_specs/higgs_audio_stt.json", VoiceTaskKind::Asr},
    {"vibevoice_asr", ARR_DESC(kAliasesVibevoiceAsr), ARR_DESC(kGgufArchsVibevoiceAsr), "model_specs/vibevoice_asr.json", VoiceTaskKind::Asr},
    {"kroko_asr", ARR_DESC(kAliasesKrokoAsr), ARR_DESC(kGgufArchsKrokoAsr), "model_specs/kroko_asr.json", VoiceTaskKind::Asr},
    {"parakeet_tdt", ARR_DESC(kAliasesParakeetTdt), ARR_DESC(kGgufArchsParakeetTdt), "model_specs/parakeet_tdt.json", VoiceTaskKind::Asr},
    {"voxtral_realtime", ARR_DESC(kAliasesVoxtralRealtime), ARR_DESC(kGgufArchsVoxtralRealtime), "model_specs/voxtral_realtime.json", VoiceTaskKind::Asr},
    {"voxtral", ARR_DESC(kAliasesVoxtral), ARR_DESC(kGgufArchsVoxtral), "model_specs/voxtral.json", VoiceTaskKind::Asr},
    {"whisper", ARR_DESC(kAliasesWhisper), ARR_DESC(kGgufArchsWhisper), "model_specs/whisper.json", VoiceTaskKind::Asr},
    {"moonshine", ARR_DESC(kAliasesMoonshine), ARR_DESC(kGgufArchsMoonshine), "model_specs/moonshine.json", VoiceTaskKind::Asr},
    {"moonshine_streaming", ARR_DESC(kAliasesMoonshineStreaming), ARR_DESC(kGgufArchsMoonshineStreaming), "model_specs/moonshine_streaming.json", VoiceTaskKind::Asr},
    {"canary", ARR_DESC(kAliasesCanary), ARR_DESC(kGgufArchsCanary), "model_specs/canary.json", VoiceTaskKind::Asr},
    {"canary_qwen", ARR_DESC(kAliasesCanaryQwen), ARR_DESC(kGgufArchsCanaryQwen), "model_specs/canary_qwen.json", VoiceTaskKind::Asr},
    {"cohere", ARR_DESC(kAliasesCohere), ARR_DESC(kGgufArchsCohere), "model_specs/cohere.json", VoiceTaskKind::Asr},
    {"gigaam", ARR_DESC(kAliasesGigaam), ARR_DESC(kGgufArchsGigaam), "model_specs/gigaam.json", VoiceTaskKind::Asr},
    {"granite", ARR_DESC(kAliasesGranite), ARR_DESC(kGgufArchsGranite), "model_specs/granite.json", VoiceTaskKind::Asr},
    {"granite_nar", ARR_DESC(kAliasesGraniteNar), ARR_DESC(kGgufArchsGraniteNar), "model_specs/granite_nar.json", VoiceTaskKind::Asr},
    {"medasr", ARR_DESC(kAliasesMedasr), ARR_DESC(kGgufArchsMedasr), "model_specs/medasr.json", VoiceTaskKind::Asr},

    // Diarization & Separation
    {"sortformer_diar", ARR_DESC(kAliasesSortformerDiar), ARR_DESC(kGgufArchsSortformerDiar), "model_specs/sortformer_diar.json", VoiceTaskKind::Diarization},
    {"roformer", ARR_DESC(kAliasesRoformer), ARR_DESC(kGgufArchsRoformer), "model_specs/roformer.json", VoiceTaskKind::SourceSeparation},
    {"demucs", ARR_DESC(kAliasesDemucs), EMPTY_DESC, "model_specs/demucs.json", VoiceTaskKind::SourceSeparation},
    {"moss", ARR_DESC(kAliasesMoss), ARR_DESC(kGgufArchsMoss), "model_specs/moss.json", VoiceTaskKind::Asr},

    // TTS & Voice
    {"ace_step", ARR_DESC(kAliasesAceStep), EMPTY_DESC, "model_specs/ace_step.json", VoiceTaskKind::Tts},
    {"chatterbox", ARR_DESC(kAliasesChatterbox), EMPTY_DESC, "model_specs/chatterbox.json", VoiceTaskKind::Tts},
    {"confucius4_tts", ARR_DESC(kAliasesConfucius4Tts), EMPTY_DESC, "model_specs/confucius4_tts.json", VoiceTaskKind::Tts},
    {"dots_tts", ARR_DESC(kAliasesDotsTts), EMPTY_DESC, "model_specs/dots_tts.json", VoiceTaskKind::Tts},
    {"dramabox", ARR_DESC(kAliasesDramabox), EMPTY_DESC, "model_specs/dramabox.json", VoiceTaskKind::Tts},
    {"fish_audio", ARR_DESC(kAliasesFishAudio), EMPTY_DESC, "model_specs/fish_audio.json", VoiceTaskKind::Tts},
    {"heartmula", ARR_DESC(kAliasesHeartmula), EMPTY_DESC, "model_specs/heartmula.json", VoiceTaskKind::Tts},
    {"higgs_audio_tts", ARR_DESC(kAliasesHiggsAudioTts), EMPTY_DESC, "model_specs/higgs_audio_tts.json", VoiceTaskKind::Tts},
    {"index_tts2", ARR_DESC(kAliasesIndexTts2), EMPTY_DESC, "model_specs/index_tts2.json", VoiceTaskKind::Tts},
    {"irodori_tts", ARR_DESC(kAliasesIrodoriTts), EMPTY_DESC, "model_specs/irodori_tts.json", VoiceTaskKind::Tts},
    {"magpie_tts", ARR_DESC(kAliasesMagpieTts), EMPTY_DESC, "model_specs/magpie_tts.json", VoiceTaskKind::Tts},
    {"meanvc2", ARR_DESC(kAliasesMeanvc2), EMPTY_DESC, "model_specs/meanvc2.json", VoiceTaskKind::VoiceConversion},
    {"miocodec", ARR_DESC(kAliasesMiocodec), EMPTY_DESC, "model_specs/miocodec.json", VoiceTaskKind::AudioGeneration},
    {"miotts", ARR_DESC(kAliasesMiotts), EMPTY_DESC, "model_specs/miotts.json", VoiceTaskKind::Tts},
    {"muscriptor", ARR_DESC(kAliasesMuscriptor), EMPTY_DESC, "model_specs/muscriptor.json", VoiceTaskKind::AudioGeneration},
    {"neutts", ARR_DESC(kAliasesNeutts), EMPTY_DESC, "model_specs/neutts.json", VoiceTaskKind::Tts},
    {"omnivoice", ARR_DESC(kAliasesOmnivoice), EMPTY_DESC, "model_specs/omnivoice.json", VoiceTaskKind::SpeechToSpeech},
    {"personaplex", ARR_DESC(kAliasesPersonaplex), EMPTY_DESC, "model_specs/personaplex.json", VoiceTaskKind::Tts},
    {"pocket_tts", ARR_DESC(kAliasesPocketTts), EMPTY_DESC, "model_specs/pocket_tts.json", VoiceTaskKind::Tts},
    {"qwen3_forced_aligner", ARR_DESC(kAliasesQwen3ForcedAligner), EMPTY_DESC, "model_specs/qwen3_forced_aligner.json", VoiceTaskKind::Alignment},
    {"qwen3_tts", ARR_DESC(kAliasesQwen3Tts), EMPTY_DESC, "model_specs/qwen3_tts.json", VoiceTaskKind::Tts},
    {"rvc", ARR_DESC(kAliasesRvc), EMPTY_DESC, "model_specs/rvc.json", VoiceTaskKind::VoiceConversion},
    {"seed_vc", ARR_DESC(kAliasesSeedVc), EMPTY_DESC, "model_specs/seed_vc.json", VoiceTaskKind::VoiceConversion},
    {"stable_audio", ARR_DESC(kAliasesStableAudio), EMPTY_DESC, "model_specs/stable_audio.json", VoiceTaskKind::AudioGeneration},
    {"supertonic", ARR_DESC(kAliasesSupertonic), EMPTY_DESC, "model_specs/supertonic.json", VoiceTaskKind::Tts},
    {"vevo2", ARR_DESC(kAliasesVevo2), EMPTY_DESC, "model_specs/vevo2.json", VoiceTaskKind::VoiceConversion},
    {"vibevoice", ARR_DESC(kAliasesVibevoice), EMPTY_DESC, "model_specs/vibevoice.json", VoiceTaskKind::Tts},
    {"voxcpm2", ARR_DESC(kAliasesVoxcpm2), EMPTY_DESC, "model_specs/voxcpm2.json", VoiceTaskKind::Tts},

    // Community
    {"f5_tts", ARR_DESC(kAliasesF5Tts), EMPTY_DESC, "model_specs/f5_tts.json", VoiceTaskKind::Tts},
    {"glm_tts", ARR_DESC(kAliasesGlmTts), EMPTY_DESC, "model_specs/glm_tts.json", VoiceTaskKind::Tts},
    {"inflect_v2", ARR_DESC(kAliasesInflectV2), EMPTY_DESC, "model_specs/inflect_v2.json", VoiceTaskKind::Tts},
    {"minimax_h3", ARR_DESC(kAliasesMinimaxH3), EMPTY_DESC, "model_specs/minimax_h3.json", VoiceTaskKind::Tts},
    {"minimax_music3", ARR_DESC(kAliasesMinimaxMusic3), EMPTY_DESC, "model_specs/minimax_music3.json", VoiceTaskKind::AudioGeneration},
    {"mms_forced_aligner", ARR_DESC(kAliasesMmsForcedAligner), EMPTY_DESC, "model_specs/mms_forced_aligner.json", VoiceTaskKind::Alignment},
    {"moss_voicegen", ARR_DESC(kAliasesMossVoicegen), EMPTY_DESC, "model_specs/moss_voicegen.json", VoiceTaskKind::Tts},
    {"outetts", ARR_DESC(kAliasesOutetts), EMPTY_DESC, "model_specs/outetts.json", VoiceTaskKind::Tts},
    {"vietneu_tts", ARR_DESC(kAliasesVietneuTts), EMPTY_DESC, "model_specs/vietneu_tts.json", VoiceTaskKind::Tts},
};

static constexpr size_t kFamilyCount = sizeof(kStaticFamilies) / sizeof(kStaticFamilies[0]);

}  // namespace

const FamilyEntry * resolve_family(std::string_view any_spelling) {
    if (any_spelling.empty()) {
        return nullptr;
    }

    for (const auto & entry : kStaticFamilies) {
        if (entry.canonical_id == any_spelling) {
            return &entry;
        }
        for (size_t i = 0; i < entry.aliases_count; ++i) {
            if (entry.aliases_data[i] == any_spelling) {
                return &entry;
            }
        }
        for (size_t i = 0; i < entry.gguf_archs_count; ++i) {
            if (entry.gguf_archs_data[i] == any_spelling) {
                return &entry;
            }
        }
    }
    return nullptr;
}

const FamilyEntry * all_registered_families_data() {
    return kStaticFamilies;
}

size_t all_registered_families_count() {
    return kFamilyCount;
}

}  // namespace engine::runtime
