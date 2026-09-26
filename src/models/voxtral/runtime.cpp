// engine/models/voxtral/runtime.cpp - inference for the native engine Voxtral
// (2507, offline) package (see runtime.h).
//
// Ported from src/runtime/arch/voxtral/model.cpp: load() (GPU half: weight
// catalog, backend upload, pack_gate_up), init_context() (flash defaults, KV
// type), run() (30 s chunking, mel, per-chunk encoder, prompt, input-length
// gate, KV sizing, single-shot / chunked prefill, greedy step loop, EOS /
// truncation, decode + trim) and prefill_chunked().
//
// Differences from the arch, none numeric on the CPU:
//   - one backend + ggml_gallocr per graph (the engine convention) instead of
//     a multi-backend scheduler with CPU op fallback;
//   - the packed gate+up tensors are filled straight from the GGUF bytes into
//     their own buffer, so the separate gate/up tensors are never uploaded
//     (the arch kept both copies resident; same bytes, less memory);
//   - the KV cache is (re)allocated for every run and freed at its end. The
//     arch freed it at the end of run() too (cleanup_gpu) but kept its n_ctx,
//     so a second run() on the same session that needed no growth skipped the
//     re-allocation and failed to build its prefill graph; the engine always
//     re-allocates. The KV width keeps the arch's 4096-token floor
//     (init_context's initial allocation) so the step graph's attention
//     window (max_n_kv) is the arch's for every input;
//   - the encoder graph is built once per run and reused for every 30 s chunk
//     (its only input, the chunk mel, is uploaded before every compute);
//   - debug tensor dumps (TRANSCRIBE_DUMP_*) are not carried over.

#include "engine/models/voxtral/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/transformers/causal_lm_ops.h"
#include "engine/framework/runtime/errors.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::voxtral {

namespace {

constexpr const char *kTag = "voxtral";

// The arch init_context()'s initial KV allocation; run() only ever grew it.
constexpr int kInitialKvContext = 4096;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(std::string(kTag) + ": " + message);
}

std::string lname(const char *fmt, int i) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), fmt, i);
  return std::string(buf);
}

std::string shape_string(const std::vector<int64_t> &shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += (i != 0 ? ", " : "") + std::to_string(shape[i]);
  }
  return out + "]";
}

bool env_flag(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// transcribe-decode-budget.h: pick_decode_budget.
int pick_decode_budget(int predicted, int floor_tokens, int t_prompt, int ceiling) {
  int budget = std::max(floor_tokens, predicted);
  const int room = ceiling - t_prompt;
  if (budget > room) {
    budget = room;
  }
  return budget > 0 ? budget : 0;
}

// BCP-47 -> English language name for the synthesized translate instruction
// (arch model.cpp: k_lang_names / lang_name_for); falls back to the raw code.
const char *lang_name_for(const std::string &bcp47) {
  struct LangName {
    const char *code;
    const char *name;
  };
  static const LangName kNames[] = {
      {"en", "English"}, {"fr", "French"},     {"de", "German"}, {"es", "Spanish"},
      {"it", "Italian"}, {"pt", "Portuguese"}, {"nl", "Dutch"},  {"hi", "Hindi"},
  };
  if (bcp47.empty()) {
    return "English";
  }
  for (const auto &e : kNames) {
    if (bcp47 == e.code) {
      return e.name;
    }
  }
  return bcp47.c_str();
}

// A no_alloc ggml context + a graph allocator on the session backend. The
// context is re-created per graph; the allocator is kept across graphs of
// one stage (ggml_gallocr_alloc_graph re-reserves when the topology changes).
struct GraphRun {
  ggml_context *ctx = nullptr;
  ggml_gallocr_t gallocr = nullptr;

  GraphRun() = default;
  GraphRun(const GraphRun &) = delete;
  GraphRun &operator=(const GraphRun &) = delete;
  ~GraphRun() { free(); }

  void free_ctx() {
    if (ctx != nullptr) {
      ggml_free(ctx);
      ctx = nullptr;
    }
  }
  void free() {
    if (gallocr != nullptr) {
      ggml_gallocr_free(gallocr);
      gallocr = nullptr;
    }
    free_ctx();
  }
  // The arch's compute-context sizes (32 / 64 / 16 MB of tensor metadata).
  void init_ctx(size_t mem_bytes, const char *what) {
    free_ctx();
    ggml_init_params params{};
    params.mem_size = mem_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ctx = ggml_init(params);
    if (ctx == nullptr) {
      throw runtime::CapacityError(std::string("voxtral: ggml_init (") + what + ") failed - out of memory");
    }
  }
  void allocate(ggml_backend_t backend, ggml_cgraph *graph, const char *what) {
    if (gallocr == nullptr) {
      gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
      if (gallocr == nullptr) {
        throw runtime::CapacityError(std::string("voxtral: ggml_gallocr_new (") + what + ") failed");
      }
    }
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
      throw runtime::CapacityError(std::string("voxtral: ") + what +
                                   " graph allocation failed - out of memory. Lower the session "
                                   "n_ctx or shorten the audio.");
    }
  }
};

// Frees the KV cache when a run leaves (the arch's cleanup_gpu), on every
// path including exceptions and cancellation.
struct KvRelease {
  causal_lm::KvCache &kv;
  ~KvRelease() { kv.free(); }
};

}  // namespace

void apply_flash_env_overrides(VoxtralRuntimeOptions &options) {
  if (env_flag("TRANSCRIBE_NO_FLASH")) {
    options.encoder_use_flash = false;
    options.decoder_use_flash = false;
  }
  if (env_flag("TRANSCRIBE_FORCE_FLASH")) {
    options.encoder_use_flash = true;
    options.decoder_use_flash = true;
  }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

// A weight by its transcribe.cpp name, checked against the row-major shape
// (the reverse of the ggml ne the arch's GET_LIN / GET_CONV checked), kept in
// its GGUF storage type (the arch never converted weights).
ggml_tensor *VoxtralRuntime::load_matrix(const std::string &name, const std::vector<int64_t> &shape) {
  const auto &source = *assets_->source;
  if (!source.has_tensor(name)) {
    fail("missing tensor " + name + " in " + assets_->model_path.string());
  }
  const auto metadata = source.require_metadata(name);
  if (metadata.shape != shape) {
    fail("tensor " + name + " has shape " + shape_string(metadata.shape) + ", expected " + shape_string(shape));
  }
  return store_->load_tensor(source, name, assets::TensorStorageType::Native, shape).tensor;
}

// Norm scales, biases and the positional table (GET_F32 in the arch).
ggml_tensor *VoxtralRuntime::load_vector(const std::string &name, int64_t n) {
  const auto &source = *assets_->source;
  if (!source.has_tensor(name)) {
    fail("missing tensor " + name + " in " + assets_->model_path.string());
  }
  const std::vector<int64_t> shape{n};
  const auto metadata = source.require_metadata(name);
  if (metadata.shape != shape) {
    fail("tensor " + name + " has shape " + shape_string(metadata.shape) + ", expected " + shape_string(shape));
  }
  return store_->load_f32_tensor(source, name, shape).tensor;
}

// build_voxtral_weights, slot for slot (ggml ne reversed into row-major).
void VoxtralRuntime::load_weights() {
  const auto &hp = assets_->hparams;
  const auto &source = *assets_->source;
  const int64_t d = hp.enc_d_model;
  const int64_t n_mel = hp.enc_num_mel_bins;
  const int64_t ff = hp.enc_ffn_dim;
  const int64_t src_pos = hp.enc_max_source_positions;

  // Encoder stem: Conv1d kernels ggml [3, in, out] -> row-major {out, in, 3}.
  weights_.conv0_w = load_matrix("enc.conv.0.weight", {d, n_mel, 3});
  weights_.conv0_b = load_vector("enc.conv.0.bias", d);
  weights_.conv1_w = load_matrix("enc.conv.1.weight", {d, d, 3});
  weights_.conv1_b = load_vector("enc.conv.1.bias", d);
  // Fixed sinusoidal table ggml [d, src_pos] (F32).
  {
    const std::string name = "enc.pos_emb.weight";
    if (!source.has_tensor(name)) {
      fail("missing tensor " + name);
    }
    const std::vector<int64_t> shape{src_pos, d};
    if (source.require_metadata(name).shape != shape) {
      fail("tensor " + name + " has shape " + shape_string(source.require_metadata(name).shape) +
           ", expected " + shape_string(shape));
    }
    weights_.pos_emb_w = store_->load_f32_tensor(source, name, shape).tensor;
  }
  weights_.ln_post_w = load_vector("enc.ln_post.weight", d);
  weights_.ln_post_b = load_vector("enc.ln_post.bias", d);

  weights_.enc_blocks.assign(static_cast<size_t>(hp.enc_n_layers), EncBlockWeights{});
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &b = weights_.enc_blocks[static_cast<size_t>(i)];
    b.norm_attn_w = load_vector(lname("enc.blocks.%d.norm_attn.weight", i), d);
    b.norm_attn_b = load_vector(lname("enc.blocks.%d.norm_attn.bias", i), d);
    b.attn_q_w = load_matrix(lname("enc.blocks.%d.attn.q.weight", i), {d, d});
    b.attn_q_b = load_vector(lname("enc.blocks.%d.attn.q.bias", i), d);
    b.attn_k_w = load_matrix(lname("enc.blocks.%d.attn.k.weight", i), {d, d});  // no bias
    b.attn_v_w = load_matrix(lname("enc.blocks.%d.attn.v.weight", i), {d, d});
    b.attn_v_b = load_vector(lname("enc.blocks.%d.attn.v.bias", i), d);
    b.attn_out_w = load_matrix(lname("enc.blocks.%d.attn.out.weight", i), {d, d});
    b.attn_out_b = load_vector(lname("enc.blocks.%d.attn.out.bias", i), d);
    b.norm_ffn_w = load_vector(lname("enc.blocks.%d.norm_ffn.weight", i), d);
    b.norm_ffn_b = load_vector(lname("enc.blocks.%d.norm_ffn.bias", i), d);
    b.fc1_w = load_matrix(lname("enc.blocks.%d.ffn.fc1.weight", i), {ff, d});
    b.fc1_b = load_vector(lname("enc.blocks.%d.ffn.fc1.bias", i), ff);
    b.fc2_w = load_matrix(lname("enc.blocks.%d.ffn.fc2.weight", i), {d, ff});
    b.fc2_b = load_vector(lname("enc.blocks.%d.ffn.fc2.bias", i), d);
  }

  // Projector (no biases): ggml [proj_in, H] / [H, H].
  const int64_t h = hp.dec_hidden;
  weights_.proj_linear_1_w = load_matrix("proj.linear_1.weight", {h, static_cast<int64_t>(hp.proj_in)});
  weights_.proj_linear_2_w = load_matrix("proj.linear_2.weight", {h, h});

  // Text LM: embedding + UNTIED head, ggml [H, vocab].
  const int64_t vocab = hp.dec_vocab_size;
  weights_.token_embd_w = load_matrix("dec.token_embd.weight", {vocab, h});
  weights_.output_w = load_matrix("dec.output.weight", {vocab, h});

  const int64_t q_out = static_cast<int64_t>(hp.dec_n_heads) * hp.dec_head_dim;
  const int64_t kv_out = static_cast<int64_t>(hp.dec_n_kv_heads) * hp.dec_head_dim;
  const int64_t im = hp.dec_intermediate;
  weights_.dec_blocks.assign(static_cast<size_t>(hp.dec_n_layers), DecBlockWeights{});
  for (int i = 0; i < hp.dec_n_layers; ++i) {
    auto &b = weights_.dec_blocks[static_cast<size_t>(i)];
    b.norm_attn_w = load_vector(lname("dec.blocks.%d.norm_attn.weight", i), h);
    b.norm_ffn_w = load_vector(lname("dec.blocks.%d.norm_ffn.weight", i), h);
    b.attn_q_w = load_matrix(lname("dec.blocks.%d.attn.q.weight", i), {q_out, h});
    b.attn_k_w = load_matrix(lname("dec.blocks.%d.attn.k.weight", i), {kv_out, h});
    b.attn_v_w = load_matrix(lname("dec.blocks.%d.attn.v.weight", i), {kv_out, h});
    b.attn_o_w = load_matrix(lname("dec.blocks.%d.attn.o.weight", i), {h, q_out});
    b.ffn_down_w = load_matrix(lname("dec.blocks.%d.ffn.down.weight", i), {h, im});
    // ffn.gate / ffn.up are validated and packed in pack_gate_up().
  }
  weights_.output_norm_w = load_vector("dec.output_norm.weight", h);
}

// causal_lm::pack_gate_up: one [hidden, 2 * intermediate] tensor per block
// holding gate's bytes followed by up's bytes (concat along ggml dim 1 is a
// byte concat for row-wise quants), so the graph runs one mul_mat + swiglu.
void VoxtralRuntime::pack_gate_up() {
  const auto &hp = assets_->hparams;
  const auto &source = *assets_->source;
  const int64_t h = hp.dec_hidden;
  const int64_t im = hp.dec_intermediate;
  const size_t n_layers = static_cast<size_t>(hp.dec_n_layers);

  ggml_init_params params{};
  params.mem_size = n_layers * ggml_tensor_overhead() + 1024;
  params.mem_buffer = nullptr;
  params.no_alloc = true;
  packed_ctx_ = ggml_init(params);
  if (packed_ctx_ == nullptr) {
    fail("pack_gate_up: ggml_init failed");
  }

  const std::vector<int64_t> shape{im, h};
  for (size_t i = 0; i < n_layers; ++i) {
    const std::string gate = lname("dec.blocks.%d.ffn.gate.weight", static_cast<int>(i));
    const std::string up = lname("dec.blocks.%d.ffn.up.weight", static_cast<int>(i));
    for (const std::string *name : {&gate, &up}) {
      if (!source.has_tensor(*name)) {
        fail("missing tensor " + *name + " in " + assets_->model_path.string());
      }
      const auto metadata = source.require_metadata(*name);
      if (metadata.shape != shape) {
        fail("tensor " + *name + " has shape " + shape_string(metadata.shape) + ", expected " +
             shape_string(shape));
      }
    }
    const auto gate_meta = source.require_metadata(gate);
    const auto up_meta = source.require_metadata(up);
    if (gate_meta.dtype != up_meta.dtype) {
      fail("ffn gate/up dtype mismatch at layer " + std::to_string(i) + " (" + gate_meta.dtype + " vs " +
           up_meta.dtype + ")");
    }
    const ggml_type type = assets::ggml_type_for_tensor_dtype(gate_meta.dtype);
    ggml_tensor *t = ggml_new_tensor_2d(packed_ctx_, type, h, 2 * im);
    if (t == nullptr) {
      fail("pack_gate_up: new_tensor_2d failed at layer " + std::to_string(i));
    }
    weights_.dec_blocks[i].ffn_gate_up_w = t;
  }

  packed_buffer_ = ggml_backend_alloc_ctx_tensors(packed_ctx_, backend_);
  if (packed_buffer_ == nullptr) {
    throw runtime::CapacityError("voxtral: pack_gate_up backend buffer allocation failed - out of memory");
  }
  ggml_backend_buffer_set_usage(packed_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

  // Fill one block at a time (host peak = one gate + one up).
  for (size_t i = 0; i < n_layers; ++i) {
    const auto gate = source.require_tensor_data(lname("dec.blocks.%d.ffn.gate.weight", static_cast<int>(i)));
    const auto up = source.require_tensor_data(lname("dec.blocks.%d.ffn.up.weight", static_cast<int>(i)));
    ggml_tensor *packed = weights_.dec_blocks[i].ffn_gate_up_w;
    if (ggml_nbytes(packed) != gate.bytes.size() + up.bytes.size()) {
      fail("pack_gate_up size mismatch at layer " + std::to_string(i) + " (" + std::to_string(ggml_nbytes(packed)) +
           " vs " + std::to_string(gate.bytes.size()) + " + " + std::to_string(up.bytes.size()) + ")");
    }
    ggml_backend_tensor_set(packed, gate.bytes.data(), 0, gate.bytes.size());
    ggml_backend_tensor_set(packed, up.bytes.data(), gate.bytes.size(), up.bytes.size());
  }
}

VoxtralRuntime::VoxtralRuntime(std::shared_ptr<const VoxtralAssets> model_assets,
                               core::ExecutionContext &execution_context, VoxtralRuntimeOptions options)
    : assets_(std::move(model_assets)), execution_context_(execution_context), options_(options) {
  if (!assets_ || !assets_->source) {
    fail("runtime requires loaded assets");
  }
  backend_ = execution_context_.backend();
  if (backend_ == nullptr) {
    fail("execution backend is not initialized");
  }
  try {
    store_ = std::make_shared<core::BackendWeightStore>(backend_, execution_context_.backend_type(),
                                                        "voxtral.weights", 512ull * 1024ull * 1024ull);
    load_weights();
    store_->upload();
    pack_gate_up();
    assets_->source->release_storage();
  } catch (...) {
    if (packed_buffer_ != nullptr) {
      ggml_backend_buffer_free(packed_buffer_);
      packed_buffer_ = nullptr;
    }
    if (packed_ctx_ != nullptr) {
      ggml_free(packed_ctx_);
      packed_ctx_ = nullptr;
    }
    throw;
  }

  // The arch's MelConfig (model.cpp load()), field for field.
  const auto &hp = assets_->hparams;
  WhisperMelConfig cfg;
  cfg.sample_rate = hp.fe_sample_rate;
  cfg.num_mels = hp.fe_num_mels;
  cfg.n_fft = hp.fe_n_fft;
  cfg.win_length = hp.fe_win_length;
  cfg.hop_length = hp.fe_hop_length;
  cfg.pre_emphasis = hp.fe_pre_emphasis;
  cfg.f_min = hp.fe_f_min;
  cfg.f_max = hp.fe_f_max;
  cfg.pad_mode = hp.fe_pad_mode;
  cfg.window_type = hp.fe_window;   // "hann_periodic"
  cfg.normalize = hp.fe_normalize;  // "per_utterance"
  cfg.filterbank = assets_->mel_filterbank;
  cfg.window = assets_->window;
  mel_.emplace(cfg);
}

VoxtralRuntime::~VoxtralRuntime() {
  kv_cache_.free();
  if (packed_buffer_ != nullptr) {
    ggml_backend_buffer_free(packed_buffer_);
    packed_buffer_ = nullptr;
  }
  if (packed_ctx_ != nullptr) {
    ggml_free(packed_ctx_);
    packed_ctx_ = nullptr;
  }
}

int32_t VoxtralRuntime::sample_rate() const noexcept {
  return assets_ ? assets_->hparams.fe_sample_rate : 16000;
}

int VoxtralRuntime::n_threads() const {
  // The session's configured thread count (the C ABI adapter maps n_threads 0
  // to transcribe::default_n_threads()); never a hard-coded 1.
  return std::max(1, execution_context_.config().threads);
}

void VoxtralRuntime::compute(ggml_cgraph *graph, const char *what) {
  core::set_backend_threads(backend_, n_threads());
  if (core::compute_backend_graph(backend_, graph, nullptr, what) != GGML_STATUS_SUCCESS) {
    fail(std::string(what) + " compute failed");
  }
}

// ---------------------------------------------------------------------------
// Input + prompt
// ---------------------------------------------------------------------------

std::vector<float> VoxtralRuntime::prepare_pcm(const runtime::AudioBuffer &audio) const {
  if (audio.samples.empty()) {
    throw std::invalid_argument("voxtral: empty audio input");
  }
  if (audio.sample_rate <= 0 || audio.channels <= 0) {
    throw std::invalid_argument("voxtral: audio needs a positive sample rate and channel count");
  }
  if (audio.channels == 1 && audio.sample_rate == sample_rate()) {
    return audio.samples;  // the C ABI's input: already 16 kHz mono, untouched
  }
  return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(audio.samples, audio.sample_rate,
                                                                            audio.channels, sample_rate());
}

std::vector<int32_t> VoxtralRuntime::build_prompt(const VoxtralRequest &request, int n_audio, int &prefix_len,
                                                  int &suffix_len) const {
  const PromptSpecials &s = assets_->specials;
  const int32_t audio_id = assets_->hparams.audio_token_id;
  std::vector<int32_t> ids;
  ids.reserve(static_cast<size_t>(n_audio) + 32);
  ids.push_back(s.bos);
  ids.push_back(s.inst);
  ids.push_back(s.begin_audio);
  prefix_len = static_cast<int>(ids.size());  // 3
  ids.insert(ids.end(), static_cast<size_t>(n_audio), audio_id);
  if (request.translate) {
    // build_instruct_prompt: audio + BPE(instruction) + [/INST].
    const std::string instruction = std::string("Translate this to ") + lang_name_for(request.target_language) + ".";
    const auto instr_ids = assets_->tokenizer.encode(instruction);
    ids.insert(ids.end(), instr_ids.begin(), instr_ids.end());
    ids.push_back(s.end_inst);
  } else {
    // build_transcription_prompt: [/INST] (lang:<l>)? [TRANSCRIBE].
    ids.push_back(s.end_inst);
    if (!request.language.empty()) {
      const auto lang_ids = assets_->tokenizer.encode("lang:" + request.language);
      ids.insert(ids.end(), lang_ids.begin(), lang_ids.end());
    }
    ids.push_back(s.transcribe);
  }
  suffix_len = static_cast<int>(ids.size()) - prefix_len - n_audio;
  return ids;
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

std::vector<float> VoxtralRuntime::encode(const std::vector<float> &mel, int mel_n_frames, int n_chunks,
                                          const runtime::RunControl &control) {
  const auto &hp = assets_->hparams;
  const int n_mels = hp.enc_num_mel_bins;
  const int frames_per_chunk = hp.frames_per_chunk();
  const int audio_per_chunk = hp.audio_tokens_per_chunk();
  const int dec_h = hp.dec_hidden;

  std::vector<float> enc_host(static_cast<size_t>(dec_h) * static_cast<size_t>(n_chunks) *
                              static_cast<size_t>(audio_per_chunk));

  // One graph for every chunk: all chunks share the fixed 3000-frame shape.
  GraphRun run;
  run.init_ctx(32ull * 1024ull * 1024ull, "encoder");
  const EncoderBuild eb = build_encoder_graph(run.ctx, weights_, hp, frames_per_chunk, options_.encoder_use_flash);
  run.allocate(backend_, eb.graph, "encoder");
  if (ggml_nbytes(eb.out) != static_cast<size_t>(audio_per_chunk) * static_cast<size_t>(dec_h) * sizeof(float)) {
    fail("projector output size mismatch");
  }

  std::vector<float> chunk_mel(static_cast<size_t>(n_mels) * static_cast<size_t>(frames_per_chunk));
  for (int c = 0; c < n_chunks; ++c) {
    control.emit_progress("voxtral.encode", c, n_chunks);  // the arch's per-chunk abort poll
    // LAYOUT: the frontend is mel-major (m * n_frames + t); the encoder input
    // [n_mels, T] wants frame-major (t * n_mels + m). Same copy as the arch.
    for (int t = 0; t < frames_per_chunk; ++t) {
      const size_t tg = static_cast<size_t>(c) * static_cast<size_t>(frames_per_chunk) + static_cast<size_t>(t);
      for (int m = 0; m < n_mels; ++m) {
        chunk_mel[static_cast<size_t>(t) * n_mels + static_cast<size_t>(m)] =
            mel[static_cast<size_t>(m) * static_cast<size_t>(mel_n_frames) + tg];
      }
    }
    // Re-uploaded before EVERY compute: the graph is reused across chunks.
    ggml_backend_tensor_set(eb.mel_in, chunk_mel.data(), 0, chunk_mel.size() * sizeof(float));
    compute(eb.graph, "encoder");
    ggml_backend_tensor_get(eb.out,
                            enc_host.data() + static_cast<size_t>(c) * static_cast<size_t>(audio_per_chunk) * dec_h,
                            0, static_cast<size_t>(audio_per_chunk) * dec_h * sizeof(float));
  }
  return enc_host;
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

std::vector<float> VoxtralRuntime::prefill(const std::vector<int32_t> &prompt_ids, const std::vector<float> &enc_host,
                                           int prefix_len, int suffix_len, int n_audio,
                                           const runtime::RunControl &control) {
  const auto &hp = assets_->hparams;
  const int T_prompt = static_cast<int>(prompt_ids.size());
  const int hidden = hp.dec_hidden;
  const int vocab = hp.dec_vocab_size;
  const int chunk_size = engine::modules::transformers::prefill_chunk_size();
  std::vector<float> logits(static_cast<size_t>(vocab));

  if (T_prompt <= chunk_size) {
    // Single-shot prefill (the arch's short-prompt path).
    control.emit_progress("voxtral.prefill", 0, 1);
    GraphRun run;
    run.init_ctx(64ull * 1024ull * 1024ull, "prefill");
    const PrefillBuild pb = build_prefill_graph(run.ctx, weights_, hp, kv_cache_, T_prompt, n_audio, prefix_len,
                                                suffix_len, options_.decoder_use_flash);
    run.allocate(backend_, pb.graph, "prefill");

    ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(pb.enc_out_in, enc_host.data(), 0, enc_host.size() * sizeof(float));
    std::vector<int32_t> positions(static_cast<size_t>(T_prompt));
    for (int i = 0; i < T_prompt; ++i) {
      positions[static_cast<size_t>(i)] = i;
    }
    ggml_backend_tensor_set(pb.positions_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt) * static_cast<size_t>(T_prompt));
    engine::modules::transformers::fill_prefill_chunk_mask(mask.data(), T_prompt, T_prompt, /*n_past=*/0);
    ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    compute(pb.graph, "prefill");
    ggml_backend_tensor_get(pb.out, logits.data(), 0, logits.size() * sizeof(float));
    return logits;
  }

  // Chunked prefill (arch prefill_chunked): walk [prefix | audio | suffix] in
  // blocks against the growing KV. A single flash-attention call is capped at
  // 65535 query rows on Metal and the [T, T] mask is O(T^2), hence chunks.
  const int n_chunks = (T_prompt + chunk_size - 1) / chunk_size;
  const int aud_lo = prefix_len;
  const int aud_hi = prefix_len + n_audio;
  std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt) * static_cast<size_t>(std::min(chunk_size, T_prompt)));
  std::vector<int32_t> positions(static_cast<size_t>(chunk_size));
  std::vector<int64_t> kv_idx(static_cast<size_t>(chunk_size));

  GraphRun run;
  for (int c = 0; c < n_chunks; ++c) {
    control.emit_progress("voxtral.prefill", c, n_chunks);
    const int a = c * chunk_size;
    const int T_chunk = std::min(chunk_size, T_prompt - a);
    const int b = a + T_chunk;
    const int max_n_kv = b;
    const bool last = (c == n_chunks - 1);
    const int pre_n = std::max(0, std::min(b, aud_lo) - a);
    const int aud_n = std::max(0, std::min(b, aud_hi) - std::max(a, aud_lo));
    const int suf_n = T_chunk - pre_n - aud_n;

    run.init_ctx(64ull * 1024ull * 1024ull, "prefill chunk");
    const PrefillChunkBuild pb = build_prefill_chunk_graph(run.ctx, weights_, hp, kv_cache_, T_chunk, max_n_kv,
                                                           pre_n, aud_n, suf_n, options_.decoder_use_flash, last);
    run.allocate(backend_, pb.graph, "prefill chunk");

    if (pb.input_ids_in != nullptr) {  // null when the chunk is pure audio
      ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data() + a, 0,
                              static_cast<size_t>(T_chunk) * sizeof(int32_t));
    }
    if (aud_n > 0) {
      const size_t row0 = static_cast<size_t>(std::max(a, aud_lo) - aud_lo);
      ggml_backend_tensor_set(pb.enc_out_in, enc_host.data() + row0 * static_cast<size_t>(hidden), 0,
                              static_cast<size_t>(aud_n) * static_cast<size_t>(hidden) * sizeof(float));
    }
    for (int i = 0; i < T_chunk; ++i) {
      positions[static_cast<size_t>(i)] = a + i;
      kv_idx[static_cast<size_t>(i)] = a + i;
    }
    ggml_backend_tensor_set(pb.positions_in, positions.data(), 0, static_cast<size_t>(T_chunk) * sizeof(int32_t));
    ggml_backend_tensor_set(pb.kv_idx_in, kv_idx.data(), 0, static_cast<size_t>(T_chunk) * sizeof(int64_t));
    engine::modules::transformers::fill_prefill_chunk_mask(mask.data(), max_n_kv, T_chunk, /*n_past=*/a);
    ggml_backend_tensor_set(pb.mask_in, mask.data(), 0,
                            static_cast<size_t>(max_n_kv) * static_cast<size_t>(T_chunk) * sizeof(ggml_fp16_t));

    compute(pb.graph, "prefill chunk");
    if (last) {
      ggml_backend_tensor_get(pb.out, logits.data(), 0, logits.size() * sizeof(float));
    }
  }
  return logits;
}

// ---------------------------------------------------------------------------
// transcribe (the arch's run())
// ---------------------------------------------------------------------------

VoxtralTranscription VoxtralRuntime::transcribe(const std::vector<float> &pcm, const VoxtralRequest &request,
                                                const runtime::RunControl &control) {
  control.emit_progress("voxtral.mel", 0, 1);  // the arch polled abort at the top of run()
  if (pcm.empty()) {
    throw std::invalid_argument("voxtral: empty audio input");
  }
  const auto &hp = assets_->hparams;

  // ----- Chunking: pad PCM to a multiple of 30 s and mel it once -----
  const int samples_per_chunk = hp.samples_per_chunk();
  const int frames_per_chunk = hp.frames_per_chunk();
  const int audio_per_chunk = hp.audio_tokens_per_chunk();
  if (samples_per_chunk <= 0 || frames_per_chunk <= 0 || audio_per_chunk <= 0) {
    fail("invalid 30 s chunk geometry");
  }
  const size_t n_samples = pcm.size();
  const int n_chunks =
      std::max(1, static_cast<int>((n_samples + static_cast<size_t>(samples_per_chunk) - 1) /
                                   static_cast<size_t>(samples_per_chunk)));
  const size_t padded_len = static_cast<size_t>(n_chunks) * static_cast<size_t>(samples_per_chunk);
  std::vector<float> pcm_padded(padded_len, 0.0f);
  std::memcpy(pcm_padded.data(), pcm.data(), n_samples * sizeof(float));

  // Per-utterance normalization spans the whole padded signal (every chunk),
  // exactly as the arch computed one mel over pcm_padded.
  std::vector<float> mel;
  int mel_n_mels = 0;
  int mel_n_frames = 0;
  if (!mel_->compute(pcm_padded.data(), pcm_padded.size(), mel, mel_n_mels, mel_n_frames, n_threads())) {
    throw std::invalid_argument("voxtral: the mel frontend rejected the audio (" + std::to_string(n_samples) +
                                " samples)");
  }
  if (mel_n_mels != hp.enc_num_mel_bins) {
    fail("mel bins " + std::to_string(mel_n_mels) + " != " + std::to_string(hp.enc_num_mel_bins));
  }
  if (mel_n_frames < n_chunks * frames_per_chunk) {
    fail("mel frames " + std::to_string(mel_n_frames) + " < " + std::to_string(n_chunks) + "*" +
         std::to_string(frames_per_chunk));
  }

  // ----- Prompt construction (before the encoder so a bad request or an
  // over-length input costs no encode; the arch built it after) -----
  const int n_audio_total = n_chunks * audio_per_chunk;
  int prefix_len = 0;
  int suffix_len = 0;
  const std::vector<int32_t> prompt_ids = build_prompt(request, n_audio_total, prefix_len, suffix_len);
  const int T_prompt = static_cast<int>(prompt_ids.size());

  // ----- Input-length gate (docs/input-limits.md) -----
  const int model_max = voxtral_context_ceiling(options_.n_ctx, hp);
  if (T_prompt + kGenReserve > model_max) {
    throw runtime::InputTooLong(
        "voxtral: input too long - " + std::to_string(n_audio_total) + " audio tokens + " +
        std::to_string(T_prompt - n_audio_total) + " prompt exceed the " + std::to_string(model_max) +
        "-token context (need " + std::to_string(T_prompt + kGenReserve) +
        " incl. generation reserve). Shorten the audio (max " + std::to_string(assets_->max_audio_ms) +
        " ms) or split it into segments.");
  }
  const int max_new = pick_decode_budget(n_audio_total, kDecodeBudgetMin, T_prompt, model_max);

  // ----- Encoder + projector, per 30 s chunk -----
  const std::vector<float> enc_host = encode(mel, mel_n_frames, n_chunks, control);
  std::vector<float>().swap(mel);

  // ----- KV cache for this utterance -----
  const int want_ctx = engine::modules::transformers::pick_kv_cache_context(T_prompt + max_new, model_max);
  const int kv_ctx = std::max(kInitialKvContext, want_ctx);
  KvRelease release{kv_cache_};  // the arch's cleanup_gpu, on every exit path
  if (!causal_lm::kv_init(kv_cache_, backend_, kv_ctx, hp.dec_n_kv_heads, hp.dec_head_dim, hp.dec_n_layers,
                          options_.kv_type)) {
    throw runtime::CapacityError("voxtral: KV cache allocation failed (n_ctx=" + std::to_string(kv_ctx) +
                                 ") - out of memory. Lower the session n_ctx or shorten the audio.");
  }

  // ----- Prefill -> first token (host argmax, first max wins) -----
  const std::vector<float> logits = prefill(prompt_ids, enc_host, prefix_len, suffix_len, n_audio_total, control);
  int32_t next_tok = 0;
  {
    float best = logits[0];
    for (int32_t i = 1; i < static_cast<int32_t>(logits.size()); ++i) {
      if (logits[static_cast<size_t>(i)] > best) {
        best = logits[static_cast<size_t>(i)];
        next_tok = i;
      }
    }
  }
  std::vector<int32_t> generated;
  generated.reserve(static_cast<size_t>(max_new) + 1);
  generated.push_back(next_tok);

  // ----- Greedy step loop -----
  const int32_t eos_id = hp.eos_token_id;
  int cur_past = T_prompt;
  int max_n_kv = 1024;
  while (max_n_kv < T_prompt + max_new) {
    max_n_kv *= 2;
  }
  if (max_n_kv > kv_cache_.n_ctx) {
    max_n_kv = kv_cache_.n_ctx;
  }

  GraphRun step_run;
  step_run.init_ctx(16ull * 1024ull * 1024ull, "step");
  const StepBuild sb = build_step_graph(step_run.ctx, weights_, hp, kv_cache_, max_n_kv, options_.decoder_use_flash);
  step_run.allocate(backend_, sb.graph, "step");

  const ggml_fp16_t mask_zero = ggml_fp32_to_fp16(0.0f);
  const ggml_fp16_t mask_neg = ggml_fp32_to_fp16(-INFINITY);
  std::vector<ggml_fp16_t> step_mask(static_cast<size_t>(max_n_kv), mask_neg);
  while (next_tok != eos_id && static_cast<int>(generated.size()) < max_new && cur_past + 1 <= max_n_kv) {
    // The cancellation point and progress report of the decode loop.
    control.emit_progress("voxtral.decode", static_cast<int64_t>(generated.size()), max_new);

    // Every step-graph input is uploaded before every compute (the graph is
    // reused for the whole decode; see graphs.h).
    ggml_backend_tensor_set(sb.input_id_in, &next_tok, 0, sizeof(int32_t));
    const int32_t pos_val = cur_past;
    ggml_backend_tensor_set(sb.position_in, &pos_val, 0, sizeof(int32_t));
    const int64_t kv_idx_val = cur_past;
    ggml_backend_tensor_set(sb.kv_idx_in, &kv_idx_val, 0, sizeof(int64_t));
    if (cur_past == T_prompt) {
      std::fill(step_mask.begin(), step_mask.begin() + cur_past + 1, mask_zero);
    } else {
      step_mask[static_cast<size_t>(cur_past)] = mask_zero;
    }
    ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0, step_mask.size() * sizeof(ggml_fp16_t));

    compute(sb.graph, "step");
    int32_t argmax_tok = 0;
    ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
    next_tok = argmax_tok;
    generated.push_back(next_tok);
    cur_past += 1;
  }

  // Decode stopped at EOS (complete) or at the generation budget / KV width
  // (truncated): reported, never thrown (the arch's OUTPUT_TRUNCATED keeps the
  // partial transcript readable).
  VoxtralTranscription out;
  out.truncated = (next_tok != eos_id);
  if (out.truncated) {
    engine::debug::log_message(engine::debug::LogLevel::Warning, kTag,
                               "output truncated at " + std::to_string(generated.size()) +
                                   " tokens - decode reached the generation budget before end-of-stream; "
                                   "the transcript may be incomplete");
  }
  if (!generated.empty() && generated.back() == eos_id) {
    generated.pop_back();
  }

  std::string text = assets_->tokenizer.decode(generated);
  out.raw_text = text;
  // Trim leading/trailing whitespace (the assistant turn often starts with a
  // byte-level space token).
  size_t b = 0;
  size_t e = text.size();
  while (b < e && std::isspace(static_cast<unsigned char>(text[b]))) {
    ++b;
  }
  while (e > b && std::isspace(static_cast<unsigned char>(text[e - 1]))) {
    --e;
  }
  out.text = text.substr(b, e - b);
  out.tokens = std::move(generated);
  return out;
}

}  // namespace engine::models::voxtral
