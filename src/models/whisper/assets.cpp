// engine/models/whisper/assets.cpp - model resources for the native engine
// Whisper package, from either distribution format (see assets.h).
//
// GGUF (transcribe.cpp layout, Phase 11 W2b): hparams from the stt.whisper.* /
// stt.frontend.* / stt.capability.* KVs with the same validation the arch
// applied (the retired arch's src/runtime/arch/whisper/weights.cpp,
// read_whisper_hparams; deleted in B16c), the
// tokenizer from tokenizer.ggml.*, the filterbank and window from tensors,
// languages from general.languages.
//
// Legacy whisper.cpp .bin (W2a): the header, filterbank and raw-bytes vocab are
// parsed here and everything the format omits (specials, suppression lists,
// the language table) is synthesized as whisper.cpp does. Ported from
// src/runtime/{transcribe-bin-loader.cpp, arch/whisper/bin_load.cpp}, both
// deleted with the arch in B16c; this is now the only .bin parser.
//
// Weights for both go through the shared TensorSource; the .bin source
// streams each payload from disk, so a 3 GB large-v3 .bin is never held
// twice in host memory.

#include "engine/models/whisper/assets.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/assets/whisper_bin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::whisper {

namespace {

constexpr uint32_t kWhisperBinMagic = 0x67676d6cu;
constexpr uint32_t kGgufMagic = 0x46554747u; // "GGUF" little-endian

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("whisper: " + message);
}

// OpenAI Whisper's tokenizer-derived non-speech ids, excluding the six control
// tokens appended by synthesize_bin_suppress_tokens(). The legacy format
// stores the vocabulary but not generation_config, and its two tokenizer
// families assign different meanings to most ids above the shared punctuation
// prefix - hence two tables.
constexpr int32_t kEnglishNonSpeechTokens[] = {
    1,     2,     7,     8,     9,     10,    14,    25,    26,    27,
    28,    29,    31,    58,    59,    60,    61,    62,    63,    90,
    91,    92,    93,    357,   366,   438,   532,   685,   705,   796,
    930,   1058,  1220,  1267,  1279,  1303,  1343,  1377,  1391,  1635,
    1782,  1875,  2162,  2361,  2488,  3467,  4008,  4211,  4600,  4808,
    5299,  5855,  6329,  7203,  9609,  9959,  10563, 10786, 11420, 11709,
    11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203,
    19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283,
    40493, 40549, 47282, 49146,
};

constexpr int32_t kMultilingualNonSpeechTokens[] = {
    1,     2,     7,     8,     9,     10,    14,    25,    26,    27,
    28,    29,    31,    58,    59,    60,    61,    62,    63,    90,
    91,    92,    93,    359,   503,   522,   542,   873,   893,   902,
    918,   922,   931,   1350,  1853,  1982,  2460,  2627,  3246,  3253,
    3268,  3536,  3846,  3961,  4183,  4667,  6585,  6647,  7273,  9061,
    9383,  10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635,
    15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130,
    26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425,
    49870, 50254,
};

// whisper.cpp's language order: a .bin's language token i is sot + 1 + i.
constexpr const char *kWhisperLanguageCodes[] = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt",  "tr", "pl", "ca", "nl", "ar", "sv",  "it", "id",
    "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",  "da", "hu", "ta", "no", "th", "ur",  "hr", "bg",
    "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv",  "bn", "sr", "az", "sl", "kn", "et",  "mk", "br",
    "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",  "gl", "mr", "pa", "si", "km", "sn",  "yo", "so",
    "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi",  "lo", "uz", "fo", "ht", "ps", "tk",  "nn", "mt",
    "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
};

// Special token layout (mirrors whisper.cpp). Defaults are the .en /
// multilingual-base layout; for multilingual, eot/sot shift +1 and the
// downstream tokens shift by num_languages - 98.
struct WhisperSpecials {
  int eot = 50256;
  int sot = 50257;
  int translate = 50357;
  int transcribe = 50358;
  int solm = 50359;
  int prev = 50360;
  int nosp = 50361;
  int notimestamps = 50362; // <|notimestamps|>
  int beg = 50363;          // <|0.00|>, first timestamp token
};

WhisperSpecials compute_specials(int n_vocab) {
  WhisperSpecials s;
  if (n_vocab >= 51865) {
    s.eot += 1;
    s.sot += 1;
    const int num_languages = n_vocab - 51765 - 1;
    const int dt = num_languages - 98;
    s.translate += dt;
    s.transcribe += dt;
    s.solm += dt;
    s.prev += dt;
    s.nosp += dt;
    s.notimestamps += dt;
    s.beg += dt;
  }
  return s;
}

// Legacy whisper.cpp tensor name -> transcribe.cpp canonical GGUF name. The
// table the retired arch's bin_load.cpp used (k_static_renames /
// k_*_block_renames), so the two layouts provably name the same weights.
struct Rename {
  const char *legacy;
  const char *canonical;
};
constexpr Rename kStaticRenames[] = {
    {"encoder.positional_embedding", "enc.pos_emb.weight"},
    {"encoder.conv1.weight", "enc.conv.0.weight"},
    {"encoder.conv1.bias", "enc.conv.0.bias"},
    {"encoder.conv2.weight", "enc.conv.1.weight"},
    {"encoder.conv2.bias", "enc.conv.1.bias"},
    {"encoder.ln_post.weight", "enc.final_norm.weight"},
    {"encoder.ln_post.bias", "enc.final_norm.bias"},
    {"decoder.positional_embedding", "dec.pos_emb.weight"},
    {"decoder.token_embedding.weight", "dec.token_embd.weight"},
    {"decoder.ln.weight", "dec.final_norm.weight"},
    {"decoder.ln.bias", "dec.final_norm.bias"},
};
constexpr Rename kEncoderBlockRenames[] = {
    {"attn_ln.weight", "norm_attn.weight"},  {"attn_ln.bias", "norm_attn.bias"},
    {"attn.query.weight", "attn.q.weight"},  {"attn.query.bias", "attn.q.bias"},
    {"attn.key.weight", "attn.k.weight"},    {"attn.value.weight", "attn.v.weight"},
    {"attn.value.bias", "attn.v.bias"},      {"attn.out.weight", "attn.out.weight"},
    {"attn.out.bias", "attn.out.bias"},      {"mlp_ln.weight", "norm_ffn.weight"},
    {"mlp_ln.bias", "norm_ffn.bias"},        {"mlp.0.weight", "ffn.fc1.weight"},
    {"mlp.0.bias", "ffn.fc1.bias"},          {"mlp.2.weight", "ffn.fc2.weight"},
    {"mlp.2.bias", "ffn.fc2.bias"},
};
constexpr Rename kDecoderBlockRenames[] = {
    {"attn_ln.weight", "norm_self.weight"},
    {"attn_ln.bias", "norm_self.bias"},
    {"attn.query.weight", "self_attn.q.weight"},
    {"attn.query.bias", "self_attn.q.bias"},
    {"attn.key.weight", "self_attn.k.weight"},
    {"attn.value.weight", "self_attn.v.weight"},
    {"attn.value.bias", "self_attn.v.bias"},
    {"attn.out.weight", "self_attn.out.weight"},
    {"attn.out.bias", "self_attn.out.bias"},
    {"cross_attn_ln.weight", "norm_cross.weight"},
    {"cross_attn_ln.bias", "norm_cross.bias"},
    {"cross_attn.query.weight", "cross_attn.q.weight"},
    {"cross_attn.query.bias", "cross_attn.q.bias"},
    {"cross_attn.key.weight", "cross_attn.k.weight"},
    {"cross_attn.value.weight", "cross_attn.v.weight"},
    {"cross_attn.value.bias", "cross_attn.v.bias"},
    {"cross_attn.out.weight", "cross_attn.out.weight"},
    {"cross_attn.out.bias", "cross_attn.out.bias"},
    {"mlp_ln.weight", "norm_ffn.weight"},
    {"mlp_ln.bias", "norm_ffn.bias"},
    {"mlp.0.weight", "ffn.fc1.weight"},
    {"mlp.0.bias", "ffn.fc1.bias"},
    {"mlp.2.weight", "ffn.fc2.weight"},
    {"mlp.2.bias", "ffn.fc2.bias"},
};

std::string canonical_tensor_name(std::string_view legacy) {
  for (const auto &r : kStaticRenames) {
    if (legacy == r.legacy) {
      return r.canonical;
    }
  }
  auto block = [&](std::string_view prefix, const char *out_prefix,
                   const auto &table) -> std::string {
    if (legacy.substr(0, prefix.size()) != prefix) {
      return {};
    }
    const std::string_view rest = legacy.substr(prefix.size());
    const size_t dot = rest.find('.');
    if (dot == std::string_view::npos) {
      return {};
    }
    const std::string_view index = rest.substr(0, dot);
    const std::string_view suffix = rest.substr(dot + 1);
    for (const auto &r : table) {
      if (suffix == r.legacy) {
        return std::string(out_prefix) + std::string(index) + "." + r.canonical;
      }
    }
    return {};
  };
  if (auto name = block("encoder.blocks.", "enc.blocks.", kEncoderBlockRenames);
      !name.empty()) {
    return name;
  }
  if (auto name = block("decoder.blocks.", "dec.blocks.", kDecoderBlockRenames);
      !name.empty()) {
    return name;
  }
  fail("no GGUF name for legacy tensor " + std::string(legacy));
}

uint32_t read_magic(const std::filesystem::path &path) {
  std::ifstream f(path, std::ios::binary);
  uint32_t magic = 0;
  if (f) {
    f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  }
  return f ? magic : 0u;
}

std::vector<float> hann_periodic(int n) {
  std::vector<float> w(static_cast<size_t>(n));
  const double step = 2.0 * 3.14159265358979323846 / static_cast<double>(n);
  for (int k = 0; k < n; ++k) {
    w[static_cast<size_t>(k)] =
        static_cast<float>(0.5 * (1.0 - std::cos(step * static_cast<double>(k))));
  }
  return w;
}

// ---- legacy .bin ----------------------------------------------------------

template <typename T> T read_pod(std::ifstream &f, const char *what) {
  T value{};
  f.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!f) {
    fail(std::string("truncated .bin while reading ") + what);
  }
  return value;
}

int32_t read_i32(std::ifstream &f, const char *what) {
  return read_pod<int32_t>(f, what);
}

std::string detect_variant(int n_audio_layer, int n_audio_state,
                           bool is_multilingual) {
  const char *base = "unknown";
  // Geometry table from whisper.cpp: layers uniquely identify the size class.
  switch (n_audio_layer) {
  case 4:
    base = "tiny";
    break;
  case 6:
    base = "base";
    break;
  case 12:
    base = "small";
    break;
  case 24:
    base = "medium";
    break;
  case 32:
    base = (n_audio_state == 1280) ? "large" : "unknown";
    break;
  default:
    break;
  }
  std::string variant = std::string("whisper-") + base;
  if (!is_multilingual) {
    variant += ".en";
  }
  return variant;
}

void load_bin(WhisperAssets &out) {
  const auto &model_path = out.model_path;
  const uint64_t file_size =
      static_cast<uint64_t>(std::filesystem::file_size(model_path));
  std::ifstream f(model_path, std::ios::binary);
  if (!f) {
    fail("cannot open: " + model_path.string());
  }
  if (read_pod<uint32_t>(f, "magic") != kWhisperBinMagic) {
    fail("not a whisper.cpp .bin: " + model_path.string());
  }

  // ---- hparams: 11 int32 ----
  const int32_t n_vocab = read_i32(f, "n_vocab");
  const int32_t n_audio_ctx = read_i32(f, "n_audio_ctx");
  const int32_t n_audio_state = read_i32(f, "n_audio_state");
  const int32_t n_audio_head = read_i32(f, "n_audio_head");
  const int32_t n_audio_layer = read_i32(f, "n_audio_layer");
  const int32_t n_text_ctx = read_i32(f, "n_text_ctx");
  const int32_t n_text_state = read_i32(f, "n_text_state");
  const int32_t n_text_head = read_i32(f, "n_text_head");
  const int32_t n_text_layer = read_i32(f, "n_text_layer");
  const int32_t n_mels = read_i32(f, "n_mels");
  (void)read_i32(f, "ftype");

  // Reject non-Whisper `ggml`-magic files (e.g. Silero VAD) before we start
  // interpreting the rest of the file as Whisper geometry.
  if (n_vocab < 1000 || n_audio_state <= 0 || n_audio_head <= 0 ||
      n_audio_layer <= 0 || n_text_state <= 0 || n_text_head <= 0 ||
      n_text_layer <= 0 || (n_mels != 80 && n_mels != 128) ||
      n_audio_state % n_audio_head != 0 || n_text_state % n_text_head != 0) {
    fail("`ggml` magic but hparams are not Whisper-shaped (n_vocab=" +
         std::to_string(n_vocab) + " n_audio_state=" +
         std::to_string(n_audio_state) + " n_mels=" + std::to_string(n_mels) +
         ")");
  }

  const bool is_multilingual = n_vocab >= 51865;
  const WhisperSpecials sp = compute_specials(n_vocab);

  auto &hp = out.hparams;
  hp.enc_n_layers = n_audio_layer;
  hp.enc_d_model = n_audio_state;
  hp.enc_n_heads = n_audio_head;
  hp.enc_ffn_dim = 4 * n_audio_state;
  hp.enc_num_mel_bins = n_mels;
  hp.enc_max_source_positions = n_audio_ctx;
  hp.dec_n_layers = n_text_layer;
  hp.dec_d_model = n_text_state;
  hp.dec_n_heads = n_text_head;
  hp.dec_ffn_dim = 4 * n_text_state;
  hp.dec_max_target_positions = n_text_ctx;
  hp.dec_vocab_size = n_vocab;
  hp.dec_tie_word_embeddings = true;

  hp.decoder_start_token_id = sp.sot;
  hp.eot_token_id = sp.eot;
  hp.no_timestamps_token_id = sp.notimestamps;
  hp.transcribe_token_id = sp.transcribe;
  hp.translate_token_id = sp.translate;
  hp.prev_sot_token_id = sp.prev;
  hp.no_speech_token_id = sp.nosp;
  hp.first_language_token_id = sp.sot + 1;
  hp.n_languages = is_multilingual ? (n_vocab - 51765 - 1) : 0;
  hp.is_multilingual = is_multilingual;
  hp.suppress_tokens = synthesize_bin_suppress_tokens(is_multilingual, n_vocab);
  hp.begin_suppress_tokens = {220, sp.eot};

  // Frontend constants are fixed across whisper variants; no .bin field
  // carries them.
  hp.fe_num_mels = n_mels;

  out.variant = detect_variant(n_audio_layer, n_audio_state, is_multilingual);

  // ---- mel filters: n_mel rows of n_fft/2+1 ----
  const int32_t n_mel_filters = read_i32(f, "mel n_mel");
  const int32_t n_fft_filters = read_i32(f, "mel n_fft");
  if (n_mel_filters != n_mels || n_fft_filters != hp.fe_n_fft / 2 + 1) {
    fail("mel filterbank is " + std::to_string(n_mel_filters) + "x" +
         std::to_string(n_fft_filters) + ", expected " +
         std::to_string(n_mels) + "x" + std::to_string(hp.fe_n_fft / 2 + 1));
  }
  out.mel_filterbank.resize(static_cast<size_t>(n_mel_filters) *
                            static_cast<size_t>(n_fft_filters));
  f.read(reinterpret_cast<char *>(out.mel_filterbank.data()),
         static_cast<std::streamsize>(out.mel_filterbank.size() * sizeof(float)));
  if (!f) {
    fail("truncated .bin while reading the mel filterbank");
  }
  // The .bin carries no window; whisper.cpp computes the periodic Hann.
  out.window = hann_periodic(hp.fe_n_fft);

  // ---- vocab: raw UTF-8 pieces, id = merge rank (tiktoken) ----
  const int32_t n_vocab_file = read_i32(f, "vocab count");
  if (n_vocab_file <= 0 || n_vocab_file > n_vocab) {
    fail("vocab count " + std::to_string(n_vocab_file) +
         " inconsistent with n_vocab " + std::to_string(n_vocab));
  }
  out.vocab_tokens.resize(static_cast<size_t>(n_vocab_file));
  for (int32_t i = 0; i < n_vocab_file; ++i) {
    const int32_t len = read_i32(f, "vocab token length");
    if (len < 0 || static_cast<uint64_t>(len) > file_size) {
      fail("implausible vocab token length at id " + std::to_string(i));
    }
    std::string &piece = out.vocab_tokens[static_cast<size_t>(i)];
    piece.resize(static_cast<size_t>(len));
    if (len > 0) {
      f.read(piece.data(), len);
      if (!f) {
        fail("truncated .bin while reading vocab token " + std::to_string(i));
      }
    }
  }
  text::SpecialTokens specials;
  specials.eos = sp.eot;
  specials.bos = sp.eot;
  out.tokenizer =
      text::load_tokenizer_from_tokens(out.vocab_tokens, specials, /*raw_bytes=*/true);

  // ---- languages: sot + 1 + i in whisper.cpp's order ----
  if (is_multilingual) {
    const int n = std::min<int>(hp.n_languages,
                                static_cast<int>(std::size(kWhisperLanguageCodes)));
    for (int i = 0; i < n; ++i) {
      out.language_codes.emplace_back(kWhisperLanguageCodes[i]);
      out.language_token_ids.push_back(sp.sot + 1 + i);
    }
  } else {
    out.language_codes.emplace_back("en");
  }

  out.source = engine::assets::open_whisper_bin_tensor_source(model_path);
}

// ---- transcribe.cpp GGUF --------------------------------------------------

void load_gguf(WhisperAssets &out) {
  const auto meta = engine::assets::GgufMetadata::open(out.model_path);
  if (meta.find_string("general.architecture").value_or("") != "whisper") {
    fail("GGUF general.architecture is not \"whisper\": " +
         out.model_path.string());
  }

  auto &hp = out.hparams;
  hp.enc_n_layers = meta.require_i32("stt.whisper.encoder.n_layers");
  hp.enc_d_model = meta.require_i32("stt.whisper.encoder.d_model");
  hp.enc_n_heads = meta.require_i32("stt.whisper.encoder.n_heads");
  hp.enc_ffn_dim = meta.require_i32("stt.whisper.encoder.ffn_dim");
  hp.enc_num_mel_bins = meta.require_i32("stt.whisper.encoder.num_mel_bins");
  hp.enc_max_source_positions =
      meta.require_i32("stt.whisper.encoder.max_source_positions");
  hp.enc_activation = meta.require_string("stt.whisper.encoder.activation");
  hp.dec_n_layers = meta.require_i32("stt.whisper.decoder.n_layers");
  hp.dec_d_model = meta.require_i32("stt.whisper.decoder.d_model");
  hp.dec_n_heads = meta.require_i32("stt.whisper.decoder.n_heads");
  hp.dec_ffn_dim = meta.require_i32("stt.whisper.decoder.ffn_dim");
  hp.dec_max_target_positions =
      meta.require_i32("stt.whisper.decoder.max_target_positions");
  hp.dec_vocab_size = meta.require_i32("stt.whisper.decoder.vocab_size");
  hp.dec_activation = meta.require_string("stt.whisper.decoder.activation");
  hp.dec_tie_word_embeddings =
      meta.find_bool("stt.whisper.decoder.tie_word_embeddings").value_or(true);
  hp.dec_scale_embedding =
      meta.find_bool("stt.whisper.decoder.scale_embedding").value_or(false);

  hp.decoder_start_token_id = meta.require_i32("stt.whisper.decoder_start_token_id");
  hp.no_timestamps_token_id = meta.require_i32("stt.whisper.no_timestamps_token_id");
  hp.transcribe_token_id = meta.find_i32("stt.whisper.transcribe_token_id").value_or(-1);
  hp.translate_token_id = meta.find_i32("stt.whisper.translate_token_id").value_or(-1);
  hp.prev_sot_token_id = meta.find_i32("stt.whisper.prev_sot_token_id").value_or(-1);
  hp.no_speech_token_id = hp.no_timestamps_token_id - 1;
  hp.suppress_tokens = meta.find_i32_array("stt.whisper.suppress_tokens").value_or(
      std::vector<int32_t>{});
  hp.begin_suppress_tokens =
      meta.find_i32_array("stt.whisper.begin_suppress_tokens")
          .value_or(std::vector<int32_t>{});
  // HF's is_multilingual signal: English-only checkpoints disable detection.
  hp.is_multilingual = meta.find_bool("stt.capability.lang_detect").value_or(false);

  // Frontend contract. The engine MelExtractor implements exactly Whisper's
  // log-mel (periodic Hann, reflect-padded centred STFT, slaney filterbank,
  // log10 -> clamp(max - 8) -> (x + 4) / 4); refuse anything else rather than
  // produce a silently different spectrogram (the arch's rule).
  hp.fe_num_mels = meta.require_i32("stt.frontend.num_mels");
  hp.fe_sample_rate = meta.require_i32("stt.frontend.sample_rate");
  hp.fe_n_fft = meta.require_i32("stt.frontend.n_fft");
  hp.fe_win_length = meta.require_i32("stt.frontend.win_length");
  hp.fe_hop_length = meta.require_i32("stt.frontend.hop_length");
  hp.fe_chunk_length = meta.find_i32("stt.frontend.chunk_length").value_or(30);
  hp.fe_n_samples = meta.find_i32("stt.frontend.n_samples").value_or(480000);
  hp.fe_nb_max_frames = meta.find_i32("stt.frontend.nb_max_frames").value_or(3000);
  const std::string fe_type = meta.require_string("stt.frontend.type");
  const std::string fe_window = meta.require_string("stt.frontend.window");
  const std::string fe_normalize = meta.require_string("stt.frontend.normalize");
  const std::string fe_mel_norm = meta.find_string("stt.frontend.mel_norm").value_or("slaney");
  const std::string fe_pad_mode = meta.find_string("stt.frontend.pad_mode").value_or("reflect");
  const bool fe_center = meta.find_bool("stt.frontend.center").value_or(true);
  const float fe_dither = meta.find_float("stt.frontend.dither").value_or(0.0f);
  const float fe_pre_emphasis = meta.find_float("stt.frontend.pre_emphasis").value_or(0.0f);

  const auto require = [&](bool ok, const std::string &what) {
    if (!ok) {
      fail(out.model_path.filename().string() + ": unsupported " + what);
    }
  };
  require(hp.enc_n_layers > 0 && hp.enc_d_model > 0 && hp.enc_n_heads > 0 &&
              hp.enc_d_model % hp.enc_n_heads == 0 && hp.enc_ffn_dim > 0 &&
              hp.enc_max_source_positions > 0,
          "encoder geometry");
  require(hp.dec_n_layers > 0 && hp.dec_d_model > 0 && hp.dec_n_heads > 0 &&
              hp.dec_d_model % hp.dec_n_heads == 0 && hp.dec_ffn_dim > 0 &&
              hp.dec_max_target_positions > 0 && hp.dec_vocab_size > 0,
          "decoder geometry");
  require(hp.enc_d_model == hp.dec_d_model, "encoder/decoder width mismatch");
  require(hp.enc_activation == "gelu" && hp.dec_activation == "gelu",
          "activation (only gelu)");
  require(hp.dec_tie_word_embeddings, "untied lm_head");
  require(!hp.dec_scale_embedding, "scaled decoder embedding");
  require(fe_type == "mel", "frontend type " + fe_type);
  require(fe_window == "hann" || fe_window == "hann_periodic", "window " + fe_window);
  require(fe_normalize == "whisper_logmel", "normalization " + fe_normalize);
  require(fe_mel_norm == "slaney" && fe_pad_mode == "reflect" && fe_center,
          "STFT framing");
  require(fe_dither == 0.0f && fe_pre_emphasis == 0.0f, "dither / pre-emphasis");
  require(hp.fe_win_length == hp.fe_n_fft, "window length != n_fft");
  require(hp.fe_num_mels == hp.enc_num_mel_bins, "frontend / encoder mel count");

  // Tokenizer: the GGUF holds every special and timestamp piece, so ids are
  // looked up rather than computed.
  out.tokenizer = text::load_tokenizer_from_gguf(meta.context());
  if (!out.tokenizer || out.tokenizer->model() != text::TokenizerModel::ByteLevelBpe) {
    fail("GGUF tokenizer is not GPT-2 byte-level BPE");
  }
  hp.eot_token_id = out.tokenizer->specials().eos; // tokenizer.ggml.eos_token_id
  if (hp.prev_sot_token_id < 0) {
    hp.prev_sot_token_id = out.tokenizer->find("<|startofprev|>").value_or(-1);
  }
  require(hp.eot_token_id >= 0 && hp.decoder_start_token_id >= 0 &&
              hp.no_timestamps_token_id > 0 &&
              hp.no_timestamps_token_id < hp.dec_vocab_size,
          "special-token contract");

  // Languages: general.languages order (the arch's detection order).
  const auto languages =
      meta.find_string_array("general.languages").value_or(std::vector<std::string>{});
  for (const auto &code : languages) {
    if (code.empty()) {
      continue;
    }
    out.language_codes.push_back(code);
    if (hp.is_multilingual) {
      const auto id = out.tokenizer->find("<|" + code + "|>");
      if (!id.has_value()) {
        fail("language '" + code + "' has no <|" + code + "|> token");
      }
      out.language_token_ids.push_back(*id);
    }
  }
  if (out.language_codes.empty()) {
    out.language_codes.emplace_back("en");
  }
  if (!out.language_token_ids.empty()) {
    hp.first_language_token_id = out.language_token_ids.front();
    hp.n_languages = static_cast<int32_t>(out.language_token_ids.size());
  }

  out.variant = meta.find_string("stt.variant").value_or("whisper");

  out.source = engine::assets::open_tensor_source(out.model_path);
  const int64_t n_freq = hp.fe_n_fft / 2 + 1;
  out.mel_filterbank = out.source->require_f32(
      "frontend.mel_filterbank", std::vector<int64_t>{hp.fe_num_mels, n_freq});
  out.window = out.source->require_f32("frontend.window",
                                       std::vector<int64_t>{hp.fe_n_fft});
}

} // namespace

std::string WhisperAssets::tensor_name(std::string_view legacy_name) const {
  return layout == WhisperWeightLayout::TranscribeGguf
             ? canonical_tensor_name(legacy_name)
             : std::string(legacy_name);
}

std::optional<int32_t>
WhisperAssets::special_token_id(std::string_view literal) const {
  if (layout == WhisperWeightLayout::TranscribeGguf) {
    const auto id = tokenizer->find(literal);
    if (id.has_value() && *id >= hparams.eot_token_id) {
      return id;
    }
    return std::nullopt;
  }
  // .bin: the vocab stores only the base pieces; synthesize the literals
  // whisper.cpp / HF know (the arch's install_tokenizer did the same).
  const auto &hp = hparams;
  const std::pair<const char *, int32_t> fixed[] = {
      {"<|endoftext|>", hp.eot_token_id},
      {"<|startoftranscript|>", hp.decoder_start_token_id},
      {"<|startofprev|>", hp.prev_sot_token_id},
      {"<|nospeech|>", hp.no_speech_token_id},
      {"<|notimestamps|>", hp.no_timestamps_token_id},
      {"<|translate|>", hp.is_multilingual ? hp.translate_token_id : -1},
      {"<|transcribe|>", hp.is_multilingual ? hp.transcribe_token_id : -1},
  };
  for (const auto &[text, id] : fixed) {
    if (id >= 0 && literal == text) {
      return id;
    }
  }
  for (size_t i = 0; i < language_token_ids.size(); ++i) {
    if (literal == "<|" + language_codes[i] + "|>") {
      return language_token_ids[i];
    }
  }
  // <|0.00|> .. <|30.00|>
  if (literal.size() >= 6 && literal.substr(0, 2) == "<|" &&
      literal.substr(literal.size() - 2) == "|>") {
    const std::string body(literal.substr(2, literal.size() - 4));
    char *end = nullptr;
    const double seconds = std::strtod(body.c_str(), &end);
    if (end != body.c_str() && *end == '\0' && seconds >= 0.0 && seconds <= 30.0) {
      char canonical[16];
      std::snprintf(canonical, sizeof(canonical), "%.2f", seconds);
      if (body == canonical) {
        return hp.no_timestamps_token_id + 1 +
               static_cast<int32_t>(std::lround(seconds / 0.02));
      }
    }
  }
  return std::nullopt;
}

std::vector<int32_t> WhisperAssets::encode(std::string_view text) const {
  return tokenizer->encode(text);
}

std::string WhisperAssets::decode(const std::vector<int32_t> &ids) const {
  return tokenizer->decode(ids);
}

std::vector<int32_t> synthesize_bin_suppress_tokens(bool is_multilingual,
                                                    int n_vocab) {
  std::vector<int32_t> result;
  if (is_multilingual) {
    result.assign(std::begin(kMultilingualNonSpeechTokens),
                  std::end(kMultilingualNonSpeechTokens));
  } else {
    result.assign(std::begin(kEnglishNonSpeechTokens),
                  std::end(kEnglishNonSpeechTokens));
  }
  const WhisperSpecials sp = compute_specials(n_vocab);
  result.insert(result.end(), {sp.sot, sp.translate, sp.transcribe, sp.solm,
                               sp.prev, sp.nosp});
  return result;
}

bool looks_like_whisper_bin(const std::filesystem::path &path) {
  return read_magic(path) == kWhisperBinMagic;
}

bool looks_like_whisper_gguf(const std::filesystem::path &path) {
  if (read_magic(path) != kGgufMagic) {
    return false;
  }
  try {
    const auto meta = engine::assets::GgufMetadata::open(path);
    return meta.find_string("general.architecture").value_or("") == "whisper";
  } catch (const std::exception &) {
    return false;
  }
}

std::shared_ptr<const WhisperAssets>
load_whisper_assets(const std::filesystem::path &model_path) {
  if (!std::filesystem::exists(model_path)) {
    fail("model file not found: " + model_path.string());
  }
  auto out = std::make_shared<WhisperAssets>();
  out->model_path = model_path;
  const uint32_t magic = read_magic(model_path);
  if (magic == kGgufMagic) {
    out->layout = WhisperWeightLayout::TranscribeGguf;
    load_gguf(*out);
  } else if (magic == kWhisperBinMagic) {
    out->layout = WhisperWeightLayout::LegacyBin;
    load_bin(*out);
  } else {
    fail("neither a Whisper GGUF nor a whisper.cpp .bin: " + model_path.string());
  }
  return out;
}

} // namespace engine::models::whisper
