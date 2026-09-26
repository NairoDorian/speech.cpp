#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::parakeet_tdt {

// Which package layout the assets were read from. Both feed the same engine
// weights/graphs; the layout only decides the few places where the two
// reference runtimes differ (see the TranscribeGguf notes on each field).
//   AudioCpp       - the audio.cpp package (config.json + tokenizer.json +
//                    HF-named safetensors/GGUF tensors).
//   TranscribeGguf - the transcribe.cpp GGUF (general.architecture=parakeet,
//                    stt.parakeet.* KVs, NeMo-derived tensor names) that the
//                    retired src/runtime/arch/parakeet arch opened.
enum class ParakeetLayout { AudioCpp, TranscribeGguf };

// Decoder head (stt.parakeet.head_kind). The audio.cpp layout is TDT only.
enum class ParakeetHeadKind { Tdt, Rnnt, Ctc };

enum class ParakeetJointActivation { Relu, Sigmoid, Tanh };

struct ParakeetFrontendConfig {
    int64_t sample_rate = 16000;
    int64_t feature_size = 128;
    int64_t n_fft = 512;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    float preemphasis = 0.97f;
    float log_zero_guard = 5.9604644775390625e-8f;
    // transcribe.cpp MelFrontend frames the utterance as floor(n / hop) + 1
    // STFT frames and normalizes over ALL of them (transcribe-mel.h
    // n_frames_for); the audio.cpp/HF path counts floor(n / hop) valid frames
    // and zeroes the trailing one. TranscribeGguf sets this so the encoder sees
    // the arch's frame count.
    bool all_stft_frames_valid = false;
    // stt.frontend.normalize == "none" (per-feature normalization baked into
    // training). Only the TranscribeGguf layout can ask for it.
    bool normalize_per_feature = true;
};

struct ParakeetEncoderConfig {
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t layers = 24;
    int64_t heads = 8;
    int64_t conv_kernel = 9;
    int64_t subsampling_factor = 8;
    int64_t subsampling_channels = 256;
    int64_t subsampling_kernel = 3;
    int64_t subsampling_stride = 2;
    int64_t max_position_embeddings = 5000;
    // Linear/conv biases inside every conformer block (stt.parakeet.encoder.
    // use_bias; true on the 1.1B, CTC, RNNT and hybrid checkpoints).
    bool use_bias = false;
    // NeMo RelPositionalEncoding xscaling (x *= sqrt(d_model) after the
    // subsampling). The audio.cpp layout always scales (folded into the
    // subsampling projection); TranscribeGguf follows stt.parakeet.encoder.
    // xscaling exactly as the arch did (arch/parakeet/encoder.cpp).
    bool xscaling = true;
    // Local attention window (stt.parakeet.encoder.att_context_left/right,
    // Regular style). -1/-1 = full attention; parakeet-tdt_ctc-1.1b ships
    // [128, 128]. Applied as a band on the additive attention mask, which is
    // exactly the arch's -INF padding of matrix_bd (conformer.cpp
    // rel_pos_mhsa is_local branch).
    int64_t att_context_left = -1;
    int64_t att_context_right = -1;
    // Relative positional table computed as NeMo/the arch do it, in float
    // (model.cpp run_one_shot_inner: div_term = exp(2k * -ln(1e4)/d),
    // pe = sin/cos(pos * div_term)), rather than the audio.cpp long-double
    // rotation recurrence.
    bool nemo_f32_positional_encoding = false;
};

struct ParakeetConfig {
    std::string model_type;
    ParakeetHeadKind head_kind = ParakeetHeadKind::Tdt;
    int64_t vocab_size = 8193;  // token classes incl. the blank (last) class
    int64_t blank_token_id = 8192;
    int64_t pad_token_id = 2;   // -1 = none (TranscribeGguf)
    int64_t decoder_hidden_size = 640;
    int64_t decoder_layers = 2;
    int64_t joint_hidden_size = 640;
    ParakeetJointActivation joint_activation = ParakeetJointActivation::Relu;
    int64_t max_symbols_per_step = 10;
    std::vector<int32_t> durations = {0, 1, 2, 3, 4};  // empty for RNNT / CTC
    ParakeetFrontendConfig frontend;
    ParakeetEncoderConfig encoder;
};

// Per-file identity the transcribe.cpp GGUF carries (general.* / stt.*) and
// the C ABI publishes as capabilities. Empty for the audio.cpp layout, whose
// capabilities come from model_specs/parakeet_tdt.json.
struct ParakeetGgufIdentity {
    std::string variant;                 // stt.variant (e.g. "tdt-0.6b-v3")
    std::string display_name;            // general.name
    std::vector<std::string> languages;  // general.languages
    bool lang_detect = false;            // stt.capability.lang_detect
    bool streaming = false;              // stt.capability.streaming (buffered, unified-en)
};

struct ParakeetTDTAssets {
    ParakeetLayout layout = ParakeetLayout::AudioCpp;
    assets::ResourceBundle resources;
    ParakeetConfig config;
    std::shared_ptr<const assets::TensorSource> source;
    // AudioCpp layout: the HF tokenizer.json. Null for TranscribeGguf.
    std::shared_ptr<engine::tokenizers::HuggingFaceTokenizerJson> tokenizer;
    // TranscribeGguf layout: tokenizer.ggml.tokens / token_type, decoded with
    // the arch's SentencePiece rules (transcribe-tokenizer.cpp
    // decode_sentencepiece). Empty for AudioCpp.
    std::vector<std::string> vocab_pieces;
    std::vector<int32_t> vocab_types;
    // 1 = dropped from the text unless keep_language_tags. AudioCpp: the
    // tokenizer.json added_tokens marked special. TranscribeGguf: CONTROL-typed
    // pieces plus the <ll-RR> locale-tag shape (model.cpp is_strippable_special).
    std::vector<uint8_t> special_token_ids;
    ParakeetGgufIdentity identity;

    [[nodiscard]] bool transcribe_layout() const noexcept {
        return layout == ParakeetLayout::TranscribeGguf;
    }
};

std::shared_ptr<const ParakeetTDTAssets> load_parakeet_assets(const std::filesystem::path & model_path);

// True for a transcribe.cpp parakeet GGUF (magic + general.architecture ==
// "parakeet" + the stt.parakeet.* hparams). Such a file is loaded through the
// rename view; an unsupported variant inside it still sniffs true and then
// fails load with a message naming what the engine lacks.
bool looks_like_transcribe_parakeet_gguf(const std::filesystem::path & path);

// SentencePiece decode of raw piece ids exactly as the arch's tokenizer does
// it (U+2581 -> ' ', <0xHH> byte fallback, out-of-range ids skipped).
std::string decode_transcribe_pieces(const ParakeetTDTAssets & assets, const int32_t * ids, size_t count);

}  // namespace engine::community_models::parakeet_tdt
