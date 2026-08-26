// engine/models/whisper/assets.cpp - legacy whisper.cpp `.bin` parser,
// hparam synthesis and decode vocabulary for the native engine Whisper
// package.
//
// Ported from src/runtime/{transcribe-bin-loader.cpp, arch/whisper/bin_load.cpp}
// so the engine package carries no src/runtime/ dependency.
//
// File layout (whisper.cpp's monolithic format - single magic, no version):
//
//   magic         uint32  0x67676d6c ("ggml" little-endian)
//   hparams       11 x int32 (n_vocab, n_audio_*, n_text_*, n_mels, ftype)
//   mel filters   int32 n_mel, int32 n_fft, then n_mel*n_fft float32
//   vocab         int32 count, then per token: int32 len + raw bytes
//   tensors (xN)  int32 n_dims, int32 name_len, int32 ttype,
//                 n_dims x int32 dims (ggml ne order, fastest-varying first),
//                 name_len bytes name, then ggml_nbytes bytes payload
//
// Parsing is metadata-only: tensor payloads are located, not read. The
// runtime streams each tensor's bytes at upload time, so a 1.5 GB large-v3
// `.bin` is never held twice in host memory.

#include "engine/models/whisper/assets.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace engine::models::whisper {

namespace {

constexpr uint32_t kWhisperBinMagic = 0x67676d6cu;

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

// ---- little-endian readers over an ifstream ----

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

// ggml type ids as written by whisper.cpp's convert script.
ggml_type ggml_type_from_bin(int32_t ttype) {
  switch (ttype) {
  case 0:
    return GGML_TYPE_F32;
  case 1:
    return GGML_TYPE_F16;
  case 2:
    return GGML_TYPE_Q4_0;
  case 3:
    return GGML_TYPE_Q4_1;
  case 6:
    return GGML_TYPE_Q5_0;
  case 7:
    return GGML_TYPE_Q5_1;
  case 8:
    return GGML_TYPE_Q8_0;
  case 9:
    return GGML_TYPE_Q8_1;
  case 10:
    return GGML_TYPE_Q2_K;
  case 11:
    return GGML_TYPE_Q3_K;
  case 12:
    return GGML_TYPE_Q4_K;
  case 13:
    return GGML_TYPE_Q5_K;
  case 14:
    return GGML_TYPE_Q6_K;
  default:
    fail("unsupported .bin tensor type id " + std::to_string(ttype));
  }
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
  std::string variant = base;
  if (!is_multilingual) {
    variant += ".en";
  }
  return variant;
}

} // namespace

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
  result.insert(result.end(),
                {sp.sot, sp.translate, sp.transcribe, sp.solm, sp.prev,
                 sp.nosp});
  return result;
}

bool looks_like_whisper_bin(const std::filesystem::path &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  uint32_t magic = 0;
  f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  return static_cast<bool>(f) && magic == kWhisperBinMagic;
}

std::shared_ptr<const WhisperAssets>
load_whisper_assets(const std::filesystem::path &model_path) {
  if (!std::filesystem::exists(model_path)) {
    fail("model file not found: " + model_path.string());
  }
  const uint64_t file_size =
      static_cast<uint64_t>(std::filesystem::file_size(model_path));

  std::ifstream f(model_path, std::ios::binary);
  if (!f) {
    fail("cannot open: " + model_path.string());
  }

  const uint32_t magic = read_pod<uint32_t>(f, "magic");
  if (magic != kWhisperBinMagic) {
    fail("not a whisper.cpp .bin (magic 0x" + std::to_string(magic) + ")");
  }

  auto out = std::make_shared<WhisperAssets>();
  out->model_path = model_path;

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
    fail("`ggml` magic but hparams are not Whisper-shaped "
         "(n_vocab=" +
         std::to_string(n_vocab) + " n_audio_state=" +
         std::to_string(n_audio_state) + " n_mels=" + std::to_string(n_mels) +
         ")");
  }

  const bool is_multilingual = n_vocab >= 51865;
  const WhisperSpecials sp = compute_specials(n_vocab);

  auto &hp = out->hparams;
  hp.enc_n_layers = n_audio_layer;
  hp.enc_d_model = n_audio_state;
  hp.enc_n_heads = n_audio_head;
  hp.enc_ffn_dim = 4 * n_audio_state;
  hp.enc_num_mel_bins = n_mels;
  hp.enc_max_source_positions = n_audio_ctx;
  hp.enc_activation = "gelu";

  hp.dec_n_layers = n_text_layer;
  hp.dec_d_model = n_text_state;
  hp.dec_n_heads = n_text_head;
  hp.dec_ffn_dim = 4 * n_text_state;
  hp.dec_max_target_positions = n_text_ctx;
  hp.dec_vocab_size = n_vocab;
  hp.dec_activation = "gelu";
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
  hp.fe_type = "log_mel";
  hp.fe_num_mels = n_mels;
  hp.fe_sample_rate = 16000;
  hp.fe_n_fft = 400;
  hp.fe_win_length = 400;
  hp.fe_hop_length = 160;
  hp.fe_chunk_length = 30;
  hp.fe_n_samples = 480000;
  hp.fe_nb_max_frames = 3000;

  out->variant = detect_variant(n_audio_layer, n_audio_state, is_multilingual);

  // ---- mel filters ----
  out->n_mel_filters = read_i32(f, "mel n_mel");
  out->n_fft_filters = read_i32(f, "mel n_fft");
  if (out->n_mel_filters <= 0 || out->n_fft_filters <= 0) {
    fail("invalid mel filter dimensions");
  }
  {
    const size_t count = static_cast<size_t>(out->n_mel_filters) *
                         static_cast<size_t>(out->n_fft_filters);
    out->mel_filterbank.resize(count);
    f.read(reinterpret_cast<char *>(out->mel_filterbank.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) {
      fail("truncated .bin while reading the mel filterbank");
    }
  }

  // ---- vocab ----
  {
    const int32_t n_vocab_file = read_i32(f, "vocab count");
    if (n_vocab_file <= 0 || n_vocab_file > n_vocab) {
      fail("vocab count " + std::to_string(n_vocab_file) +
           " inconsistent with n_vocab " + std::to_string(n_vocab));
    }
    out->vocab_tokens.resize(static_cast<size_t>(n_vocab));
    std::string buf;
    for (int32_t i = 0; i < n_vocab_file; ++i) {
      const int32_t len = read_i32(f, "vocab token length");
      if (len < 0 || static_cast<uint64_t>(len) > file_size) {
        fail("implausible vocab token length at id " + std::to_string(i));
      }
      buf.resize(static_cast<size_t>(len));
      if (len > 0) {
        f.read(buf.data(), len);
        if (!f) {
          fail("truncated .bin while reading vocab token " + std::to_string(i));
        }
      }
      out->vocab_tokens[static_cast<size_t>(i)] = buf;
    }
  }

  // ---- tensor manifest (metadata only; payloads located, not read) ----
  while (true) {
    int32_t n_dims = 0;
    f.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    if (!f) {
      break; // clean EOF: manifest complete
    }
    const int32_t name_len = read_i32(f, "tensor name length");
    const int32_t ttype = read_i32(f, "tensor type");

    if (n_dims < 1 || n_dims > 4) {
      fail("tensor n_dims out of range: " + std::to_string(n_dims));
    }
    if (name_len <= 0 || static_cast<uint64_t>(name_len) > file_size) {
      fail("implausible tensor name length " + std::to_string(name_len));
    }

    WhisperBinTensor entry;
    entry.n_dims = n_dims;
    entry.type = ggml_type_from_bin(ttype);
    for (int32_t d = 0; d < n_dims; ++d) {
      const int32_t dim = read_i32(f, "tensor dim");
      if (dim <= 0) {
        fail("non-positive tensor dim");
      }
      entry.ne[d] = dim; // file stores ggml ne order already
    }
    entry.name.resize(static_cast<size_t>(name_len));
    f.read(entry.name.data(), name_len);
    if (!f) {
      fail("truncated .bin while reading a tensor name");
    }

    // Compute the payload size the same way ggml would, then validate it
    // fits: a truncated file (like upstream's for-tests-*.bin fixtures)
    // fails here with a specific diagnostic rather than a later crash.
    const int64_t blck = ggml_blck_size(entry.type);
    if (blck <= 0 || entry.ne[0] % blck != 0) {
      fail("tensor " + entry.name + " ne[0]=" + std::to_string(entry.ne[0]) +
           " is not a multiple of the block size for its type");
    }
    uint64_t elems = 1;
    for (int d = 0; d < 4; ++d) {
      elems *= static_cast<uint64_t>(entry.ne[d]);
    }
    entry.nbytes = elems / static_cast<uint64_t>(blck) *
                   static_cast<uint64_t>(ggml_type_size(entry.type));

    entry.offset = static_cast<uint64_t>(f.tellg());
    if (entry.offset + entry.nbytes > file_size) {
      fail("truncated .bin: tensor " + entry.name + " needs " +
           std::to_string(entry.nbytes) + " bytes at offset " +
           std::to_string(entry.offset) + " but the file is only " +
           std::to_string(file_size) + " bytes");
    }
    f.seekg(static_cast<std::streamoff>(entry.nbytes), std::ios::cur);
    if (!f) {
      fail("seek past tensor payload failed for " + entry.name);
    }

    out->tensors.push_back(std::move(entry));
  }

  if (out->tensors.empty()) {
    fail("no tensors found in " + model_path.string());
  }

  return out;
}

} // namespace engine::models::whisper
