// engine/models/moonshine_streaming/runtime.cpp - inference orchestration for
// the native engine Moonshine-Streaming package.
//
// Ported from src/runtime/arch/moonshine_streaming/model.cpp. Two paths share
// the same graph builders:
//
//   offline  encode whole clip -> adapter -> cross-KV project -> commit ->
//            greedy decode.
//   stream   per feed: encode a window with L/R context, adapter the emit
//            slice at ABSOLUTE frame offsets, project its cross-K/V, append to
//            host committed buffers; then (throttled) re-run the AR decoder
//            from BOS over the extended cross-KV.
//
// Per-feed slicing equals the one-shot result because the encoder is ergodic
// (no positional encoding) with per-layer sliding-window attention over a
// causal-conv frontend, the adapter pos_emb is an absolute-frame get_rows, and
// the cross-KV projection is per-frame linear.

#include "engine/models/moonshine_streaming/runtime.h"

#include "engine/models/moonshine_streaming/graphs_internal.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moonshine_streaming {

namespace {

constexpr const char *kTag = "moonshine_streaming";

// Default decode throttle when the caller does not ask for one. One
// cumulative right-context window (~12 frames x 20 ms) is about 4 updates/sec;
// smaller values push cost to 5x one-shot on tiny.
constexpr int32_t kDefaultMinDecodeIntervalMs = 240;

// Frontend conv-stack left-pad slack in ENCODER OUTPUT frames. Each causal
// conv is left-padded by k-1 input frames; we need enough PCM history that
// those pad slots hold real samples, otherwise a streamed chunk would differ
// from one-shot on the first conv output frame of every chunk.
//   conv1 left-pad: 4 embedder frames
//   conv2 left-pad: 4 conv1-output frames -> 8 embedder frames of stride
//                   history + 4 left-pad = 12
//   total: 16 embedder frames = 4 encoder output frames.
constexpr int kFrontendPadEncFrames = 4;

constexpr int64_t kNativeSampleRateHz = 16000;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(std::string(kTag) + ": " + message);
}

core::TensorValue load_slot(core::BackendWeightStore &store,
                            const assets::TensorSource &source,
                            const char *name,
                            assets::TensorStorageType storage_type,
                            std::initializer_list<int64_t> shape) {
  return store.load_tensor(source, name, storage_type, shape);
}

int cumulative_right_context(const MoonshineStreamingHParams &hp) {
  int total = 0;
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    const int R = hp.layer_right_window(i);
    if (R > 0) {
      total += (R - 1);
    }
  }
  return total;
}

int cumulative_left_context(const MoonshineStreamingHParams &hp) {
  int total = 0;
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    const int L = hp.layer_left_window(i);
    if (L > 1) {
      total += (L - 1);
    }
  }
  return total;
}

// One encoder output frame spans two stride-2 causal convs over an embedder
// running at frame_len samples/frame.
int samples_per_encoder_frame(const MoonshineStreamingHParams &hp) {
  return 4 * hp.enc_frame_len;
}

// Bound greedy decode by duration as well as the decoder position cap: some
// inputs never emit EOS. Upstream recommends about 6.5 tokens/sec.
int decode_generation_budget(const MoonshineStreamingHParams &hp, int T_enc) {
  if (hp.enc_frame_len <= 0 || T_enc <= 0) {
    return 0;
  }
  constexpr int64_t kBudgetNum = 13; // 6.5 tokens/sec numerator
  constexpr int64_t kBudgetDen = 2;
  constexpr int64_t kBudgetFloor = 24; // headroom for very short clips
  const int64_t audio_samples =
      static_cast<int64_t>(T_enc) * samples_per_encoder_frame(hp);
  return static_cast<int>(audio_samples * kBudgetNum /
                              (kBudgetDen * kNativeSampleRateHz) +
                          kBudgetFloor);
}

} // namespace

void MoonshineStreamingRuntime::GraphRun::free() {
  if (gallocr != nullptr) {
    ggml_gallocr_free(gallocr);
    gallocr = nullptr;
  }
  if (ctx != nullptr) {
    ggml_free(ctx);
    ctx = nullptr;
  }
  graph = nullptr;
}

MoonshineStreamingRuntime::MoonshineStreamingRuntime(
    std::shared_ptr<const MoonshineStreamingAssets> assets,
    core::ExecutionContext &execution_context,
    assets::TensorStorageType storage_type)
    : assets_(std::move(assets)), execution_context_(execution_context) {
  if (!assets_ || !assets_->source) {
    fail("runtime requires loaded assets");
  }
  backend_ = execution_context_.backend();
  if (backend_ == nullptr) {
    fail("execution backend is not initialized");
  }

  store_ = std::make_shared<core::BackendWeightStore>(
      backend_, execution_context_.backend_type(),
      "moonshine_streaming.weights", 256ull * 1024ull * 1024ull);

  auto &store = *store_;
  const auto &src = *assets_->source;
  const auto st = storage_type;
  const auto &hp = assets_->hparams;

  const int64_t enc_h = hp.enc_d_model;
  const int64_t enc_attn_h = hp.enc_attn_dim();
  const int64_t enc_ff = hp.enc_ffn_dim;
  const int64_t dec_h = hp.dec_d_model;
  const int64_t dec_ff = hp.dec_ffn_dim;
  const int64_t vocab = hp.dec_vocab_size;
  const int64_t frame_len = hp.enc_frame_len;
  const int64_t max_pos = hp.dec_max_position_embeddings;

  // ----- encoder embedder -----
  // Shapes are LOGICAL (PyTorch order); BackendWeightStore reverses into ggml
  // ne. Linear [out, in] -> ne [in, out]; Conv1d [out, in, K] -> ne [K, in,
  // out].
  weights_.embedder.comp_log_k =
      store.load_f32_tensor(src, "enc.embedder.comp.log_k", {1}).tensor;
  weights_.embedder.linear_w =
      load_slot(store, src, "enc.embedder.linear.weight", st,
                {enc_h, frame_len})
          .tensor;
  weights_.embedder.conv1_w =
      load_slot(store, src, "enc.embedder.conv1.weight", st,
                {2 * enc_h, enc_h, 5})
          .tensor;
  weights_.embedder.conv1_b =
      store.load_f32_tensor(src, "enc.embedder.conv1.bias", {2 * enc_h}).tensor;
  weights_.embedder.conv2_w =
      load_slot(store, src, "enc.embedder.conv2.weight", st,
                {enc_h, 2 * enc_h, 5})
          .tensor;
  weights_.embedder.conv2_b =
      store.load_f32_tensor(src, "enc.embedder.conv2.bias", {enc_h}).tensor;

  weights_.enc_top.final_norm_w =
      store.load_f32_tensor(src, "enc.final_norm.weight", {enc_h}).tensor;

  // ----- encoder blocks -----
  weights_.enc_blocks.resize(static_cast<size_t>(hp.enc_n_layers));
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &b = weights_.enc_blocks[static_cast<size_t>(i)];
    const std::string p = "enc.blocks." + std::to_string(i) + ".";

    b.norm_attn_w =
        store.load_f32_tensor(src, (p + "norm_attn.weight").c_str(), {enc_h})
            .tensor;
    b.attn_q_w = load_slot(store, src, (p + "attn.q.weight").c_str(), st,
                           {enc_attn_h, enc_h})
                     .tensor;
    b.attn_k_w = load_slot(store, src, (p + "attn.k.weight").c_str(), st,
                           {enc_attn_h, enc_h})
                     .tensor;
    b.attn_v_w = load_slot(store, src, (p + "attn.v.weight").c_str(), st,
                           {enc_attn_h, enc_h})
                     .tensor;
    b.attn_out_w = load_slot(store, src, (p + "attn.out.weight").c_str(), st,
                             {enc_h, enc_attn_h})
                       .tensor;

    b.norm_ffn_w =
        store.load_f32_tensor(src, (p + "norm_ffn.weight").c_str(), {enc_h})
            .tensor;
    b.ffn_fc1_w = load_slot(store, src, (p + "ffn.fc1.weight").c_str(), st,
                            {enc_ff, enc_h})
                      .tensor;
    b.ffn_fc1_b =
        store.load_f32_tensor(src, (p + "ffn.fc1.bias").c_str(), {enc_ff})
            .tensor;
    b.ffn_fc2_w = load_slot(store, src, (p + "ffn.fc2.weight").c_str(), st,
                            {enc_h, enc_ff})
                      .tensor;
    b.ffn_fc2_b =
        store.load_f32_tensor(src, (p + "ffn.fc2.bias").c_str(), {enc_h})
            .tensor;
  }

  // ----- adapter -----
  // pos_emb is an embedding table: ne [enc_h, max_pos] for get_rows.
  weights_.adapter.pos_emb_w =
      load_slot(store, src, "adapter.pos_emb.weight", st, {max_pos, enc_h})
          .tensor;
  if (hp.adapter_has_proj) {
    weights_.adapter.proj_w =
        load_slot(store, src, "adapter.proj.weight", st, {dec_h, enc_h}).tensor;
  }

  // ----- decoder top (UNTIED lm_head) -----
  weights_.dec_top.token_embd_w =
      load_slot(store, src, "dec.token_embd.weight", st, {vocab, dec_h}).tensor;
  weights_.dec_top.final_norm_w =
      store.load_f32_tensor(src, "dec.final_norm.weight", {dec_h}).tensor;
  weights_.dec_top.lm_head_w =
      load_slot(store, src, "dec.lm_head.weight", st, {vocab, dec_h}).tensor;

  // ----- decoder blocks -----
  weights_.dec_blocks.resize(static_cast<size_t>(hp.dec_n_layers));
  for (int i = 0; i < hp.dec_n_layers; ++i) {
    auto &b = weights_.dec_blocks[static_cast<size_t>(i)];
    const std::string p = "dec.blocks." + std::to_string(i) + ".";

    b.norm_self_w =
        store.load_f32_tensor(src, (p + "norm_self.weight").c_str(), {dec_h})
            .tensor;
    b.self_q_w = load_slot(store, src, (p + "self_attn.q.weight").c_str(), st,
                           {dec_h, dec_h})
                     .tensor;
    b.self_k_w = load_slot(store, src, (p + "self_attn.k.weight").c_str(), st,
                           {dec_h, dec_h})
                     .tensor;
    b.self_v_w = load_slot(store, src, (p + "self_attn.v.weight").c_str(), st,
                           {dec_h, dec_h})
                     .tensor;
    b.self_out_w = load_slot(store, src, (p + "self_attn.out.weight").c_str(),
                             st, {dec_h, dec_h})
                       .tensor;

    b.norm_cross_w =
        store.load_f32_tensor(src, (p + "norm_cross.weight").c_str(), {dec_h})
            .tensor;
    b.cross_q_w = load_slot(store, src, (p + "cross_attn.q.weight").c_str(), st,
                            {dec_h, dec_h})
                      .tensor;
    b.cross_k_w = load_slot(store, src, (p + "cross_attn.k.weight").c_str(), st,
                            {dec_h, dec_h})
                      .tensor;
    b.cross_v_w = load_slot(store, src, (p + "cross_attn.v.weight").c_str(), st,
                            {dec_h, dec_h})
                      .tensor;
    b.cross_out_w = load_slot(store, src, (p + "cross_attn.out.weight").c_str(),
                              st, {dec_h, dec_h})
                        .tensor;

    b.norm_ffn_w =
        store.load_f32_tensor(src, (p + "norm_ffn.weight").c_str(), {dec_h})
            .tensor;
    b.ffn_fc1_w = load_slot(store, src, (p + "ffn.fc1.weight").c_str(), st,
                            {2 * dec_ff, dec_h})
                      .tensor;
    b.ffn_fc1_b =
        store.load_f32_tensor(src, (p + "ffn.fc1.bias").c_str(), {2 * dec_ff})
            .tensor;
    b.ffn_fc2_w = load_slot(store, src, (p + "ffn.fc2.weight").c_str(), st,
                            {dec_h, dec_ff})
                      .tensor;
    b.ffn_fc2_b =
        store.load_f32_tensor(src, (p + "ffn.fc2.bias").c_str(), {dec_h})
            .tensor;
  }

  store.upload();
}

MoonshineStreamingRuntime::~MoonshineStreamingRuntime() {
  encoder_run_.free();
  adapter_run_.free();
  cross_kv_proj_run_.free();
  cross_kv_commit_run_.free();
  step_run_.free();
  kv_cache_.free();
}

int32_t MoonshineStreamingRuntime::sample_rate() const noexcept {
  return assets_ ? assets_->hparams.fe_sample_rate
                 : static_cast<int32_t>(kNativeSampleRateHz);
}

// ---------------------------------------------------------------------------
// Incremental pipeline steps
// ---------------------------------------------------------------------------

void MoonshineStreamingRuntime::encode_window_to_host(
    const float *pcm, int n_samples, std::vector<float> &out_enc_host,
    int &out_T_enc) {
  const auto &hp = assets_->hparams;
  if (pcm == nullptr || n_samples <= 0) {
    fail("encode_window: invalid arg");
  }

  ggml_init_params params{16ull * 1024ull * 1024ull, nullptr,
                          /*no_alloc=*/true};
  encoder_run_.free();
  encoder_run_.ctx = ggml_init(params);
  if (encoder_run_.ctx == nullptr) {
    fail("failed to init encoder compute context");
  }

  EncoderBuild eb = build_encoder_graph(encoder_run_.ctx, weights_, hp,
                                        n_samples, /*use_flash=*/true);
  if (eb.audio_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
    fail("encoder graph build failed (input too short or unaligned)");
  }
  const int T_enc = eb.T_enc;

  encoder_run_.gallocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (encoder_run_.gallocr == nullptr ||
      !ggml_gallocr_alloc_graph(encoder_run_.gallocr, eb.graph)) {
    fail("encoder graph allocation failed");
  }
  encoder_run_.graph = eb.graph;

  ggml_backend_tensor_set(eb.audio_in, pcm, 0,
                          static_cast<size_t>(n_samples) * sizeof(float));
  {
    std::vector<float> mask_buf(static_cast<size_t>(T_enc) *
                                static_cast<size_t>(T_enc));
    for (int i = 0; i < hp.enc_n_layers; ++i) {
      build_sliding_window_mask(T_enc, hp.layer_left_window(i),
                                hp.layer_right_window(i), mask_buf.data());
      ggml_backend_tensor_set(eb.per_layer_masks[static_cast<size_t>(i)],
                              mask_buf.data(), 0,
                              mask_buf.size() * sizeof(float));
    }
  }

  core::set_backend_threads(backend_,
                            std::max(1, execution_context_.config().threads));
  if (core::compute_backend_graph(backend_, eb.graph) != GGML_STATUS_SUCCESS) {
    fail("encoder compute failed");
  }
  ggml_backend_synchronize(backend_);

  out_enc_host.assign(static_cast<size_t>(hp.enc_d_model) *
                          static_cast<size_t>(T_enc),
                      0.0f);
  ggml_backend_tensor_get(eb.out, out_enc_host.data(), 0,
                          out_enc_host.size() * sizeof(float));
  out_T_enc = T_enc;
}

void MoonshineStreamingRuntime::apply_adapter_window(
    const float *enc_data, int n_frames, int abs_frame_offset,
    std::vector<float> &out_adapter) {
  const auto &hp = assets_->hparams;
  if (enc_data == nullptr || n_frames <= 0) {
    fail("adapter: invalid arg");
  }

  ggml_init_params params{8ull * 1024ull * 1024ull, nullptr, /*no_alloc=*/true};
  adapter_run_.free();
  adapter_run_.ctx = ggml_init(params);
  if (adapter_run_.ctx == nullptr) {
    fail("failed to init adapter compute context");
  }

  AdapterBuild ab =
      build_adapter_graph(adapter_run_.ctx, weights_, hp, n_frames);
  if (ab.out == nullptr || ab.graph == nullptr) {
    fail("adapter graph build failed");
  }

  adapter_run_.gallocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (adapter_run_.gallocr == nullptr ||
      !ggml_gallocr_alloc_graph(adapter_run_.gallocr, ab.graph)) {
    fail("adapter graph allocation failed");
  }
  adapter_run_.graph = ab.graph;

  ggml_backend_tensor_set(ab.encoder_out_in, enc_data, 0,
                          static_cast<size_t>(hp.enc_d_model) *
                              static_cast<size_t>(n_frames) * sizeof(float));

  // ABSOLUTE frame positions - this is what makes a window slice
  // concatenate to the one-shot result.
  std::vector<int32_t> pos_ids(static_cast<size_t>(n_frames));
  for (int i = 0; i < n_frames; ++i) {
    const int p = abs_frame_offset + i;
    if (p >= hp.dec_max_position_embeddings) {
      fail("adapter: absolute frame position " + std::to_string(p) +
           " exceeds pos_emb table (" +
           std::to_string(hp.dec_max_position_embeddings) + ")");
    }
    pos_ids[static_cast<size_t>(i)] = p;
  }
  ggml_backend_tensor_set(ab.pos_ids_in, pos_ids.data(), 0,
                          pos_ids.size() * sizeof(int32_t));

  core::set_backend_threads(backend_,
                            std::max(1, execution_context_.config().threads));
  if (core::compute_backend_graph(backend_, ab.graph) != GGML_STATUS_SUCCESS) {
    fail("adapter compute failed");
  }
  ggml_backend_synchronize(backend_);

  out_adapter.assign(static_cast<size_t>(hp.dec_d_model) *
                         static_cast<size_t>(n_frames),
                     0.0f);
  ggml_backend_tensor_get(ab.out, out_adapter.data(), 0,
                          out_adapter.size() * sizeof(float));
}

void MoonshineStreamingRuntime::project_cross_kv_window(
    const float *adapter_data, int n_frames,
    std::vector<std::vector<float>> &k_out,
    std::vector<std::vector<float>> &v_out) {
  const auto &hp = assets_->hparams;
  if (adapter_data == nullptr || n_frames <= 0) {
    fail("cross_kv_proj: invalid arg");
  }

  ggml_init_params params{16ull * 1024ull * 1024ull, nullptr,
                          /*no_alloc=*/true};
  cross_kv_proj_run_.free();
  cross_kv_proj_run_.ctx = ggml_init(params);
  if (cross_kv_proj_run_.ctx == nullptr) {
    fail("failed to init cross_kv projection context");
  }

  CrossKVProjectionBuild pb = build_cross_kv_projection_graph(
      cross_kv_proj_run_.ctx, weights_, hp, n_frames);
  if (pb.graph == nullptr || pb.encoder_out_in == nullptr) {
    fail("cross_kv projection graph build failed");
  }

  cross_kv_proj_run_.gallocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (cross_kv_proj_run_.gallocr == nullptr ||
      !ggml_gallocr_alloc_graph(cross_kv_proj_run_.gallocr, pb.graph)) {
    fail("cross_kv projection graph allocation failed");
  }
  cross_kv_proj_run_.graph = pb.graph;

  const size_t slice_floats =
      static_cast<size_t>(hp.dec_d_model) * static_cast<size_t>(n_frames);
  ggml_backend_tensor_set(pb.encoder_out_in, adapter_data, 0,
                          slice_floats * sizeof(float));

  core::set_backend_threads(backend_,
                            std::max(1, execution_context_.config().threads));
  if (core::compute_backend_graph(backend_, pb.graph) != GGML_STATUS_SUCCESS) {
    fail("cross_kv projection compute failed");
  }
  ggml_backend_synchronize(backend_);

  const size_t n_layers = static_cast<size_t>(hp.dec_n_layers);
  if (k_out.size() != n_layers || v_out.size() != n_layers) {
    k_out.assign(n_layers, std::vector<float>{});
    v_out.assign(n_layers, std::vector<float>{});
  }

  std::vector<float> scratch(slice_floats);
  for (size_t il = 0; il < n_layers; ++il) {
    ggml_backend_tensor_get(pb.per_layer_k[il], scratch.data(), 0,
                            slice_floats * sizeof(float));
    k_out[il].insert(k_out[il].end(), scratch.begin(), scratch.end());
    ggml_backend_tensor_get(pb.per_layer_v[il], scratch.data(), 0,
                            slice_floats * sizeof(float));
    v_out[il].insert(v_out[il].end(), scratch.begin(), scratch.end());
  }
}

void MoonshineStreamingRuntime::flush_stable_frames(int win_start, int win_end,
                                                    int emit_start,
                                                    int emit_end) {
  const auto &hp = assets_->hparams;
  const int spf = samples_per_enc_frame_;
  const int enc_h = hp.enc_d_model;
  const int dec_h = hp.dec_d_model;

  // Extend the slice leftward by frontend_pad frames so the conv stack's
  // left-pad holds real PCM (or implicit zero-pad at stream origin, which
  // matches one-shot).
  const int pad_left_frames = std::min(frontend_pad_frames_, win_start);
  const int slice_start_frame = win_start - pad_left_frames;
  const int slice_end_frame = win_end;

  const int64_t abs_pcm_start = static_cast<int64_t>(slice_start_frame) * spf;
  const int64_t abs_pcm_end = static_cast<int64_t>(slice_end_frame) * spf;
  if (abs_pcm_start < pcm_start_sample_) {
    fail("flush: slice underruns trimmed PCM");
  }
  const int64_t buf_pcm_start = abs_pcm_start - pcm_start_sample_;
  const int64_t buf_pcm_end = abs_pcm_end - pcm_start_sample_;
  if (buf_pcm_end > static_cast<int64_t>(pcm_buffer_.size())) {
    fail("flush: slice overruns buffered PCM");
  }
  const int n_samples = static_cast<int>(buf_pcm_end - buf_pcm_start);
  if (n_samples <= 0 || n_samples % hp.enc_frame_len != 0) {
    fail("flush: slice is empty or not frame-aligned");
  }

  std::vector<float> enc_host;
  int slice_T_enc = 0;
  encode_window_to_host(pcm_buffer_.data() + buf_pcm_start, n_samples, enc_host,
                        slice_T_enc);

  const int expected_T = slice_end_frame - slice_start_frame;
  if (slice_T_enc != expected_T) {
    fail("flush: slice T_enc mismatch (got " + std::to_string(slice_T_enc) +
         ", expected " + std::to_string(expected_T) + ")");
  }

  const int rel_emit_start = emit_start - slice_start_frame;
  const int rel_emit_end = emit_end - slice_start_frame;
  const int n_emit = rel_emit_end - rel_emit_start;
  if (n_emit <= 0) {
    fail("flush: empty emit slice");
  }

  const size_t emit_floats =
      static_cast<size_t>(enc_h) * static_cast<size_t>(n_emit);
  const size_t emit_off =
      static_cast<size_t>(enc_h) * static_cast<size_t>(rel_emit_start);
  std::vector<float> enc_emit_slice(emit_floats);
  std::memcpy(enc_emit_slice.data(), enc_host.data() + emit_off,
              emit_floats * sizeof(float));

  std::vector<float> adapter_slice;
  apply_adapter_window(enc_emit_slice.data(), n_emit,
                       /*abs_frame_offset=*/emit_start, adapter_slice);
  if (static_cast<int>(adapter_slice.size()) != dec_h * n_emit) {
    fail("flush: adapter slice size mismatch");
  }
  adapter_committed_.insert(adapter_committed_.end(), adapter_slice.begin(),
                            adapter_slice.end());

  project_cross_kv_window(adapter_slice.data(), n_emit, cross_k_committed_,
                          cross_v_committed_);

  T_emitted_ = emit_end;
}

// Drop PCM no future window can read. The leftmost frame the next feed (or
// finalize) can need is (T_emitted - L_total - frontend_pad). Quantize the
// trim to samples_per_enc_frame so pcm_start_sample_ stays frame-aligned.
void MoonshineStreamingRuntime::trim_pcm_buffer() {
  const int spf = samples_per_enc_frame_;
  const int64_t keep_from_frame =
      std::max<int64_t>(0, static_cast<int64_t>(T_emitted_) -
                               static_cast<int64_t>(L_total_frames_) -
                               static_cast<int64_t>(frontend_pad_frames_));
  const int64_t keep_from_sample = keep_from_frame * spf;
  if (keep_from_sample <= pcm_start_sample_) {
    return;
  }
  const int64_t drop = keep_from_sample - pcm_start_sample_;
  if (drop <= 0 || drop >= static_cast<int64_t>(pcm_buffer_.size())) {
    return;
  }
  pcm_buffer_.erase(pcm_buffer_.begin(),
                    pcm_buffer_.begin() + static_cast<size_t>(drop));
  pcm_start_sample_ = keep_from_sample;
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

void MoonshineStreamingRuntime::ensure_kv_cache_for_T(int T_enc) {
  const auto &hp = assets_->hparams;
  if (kv_cache_.buffer != nullptr && kv_cache_.T_enc != T_enc) {
    kv_cache_.free();
  }
  if (kv_cache_.buffer == nullptr) {
    const int n_ctx =
        hp.dec_max_position_embeddings > 0 ? hp.dec_max_position_embeddings : 512;
    if (!kv_cache_init(kv_cache_, backend_, n_ctx, T_enc, hp.dec_d_model,
                       hp.dec_n_layers, GGML_TYPE_F32)) {
      fail("KV cache allocation failed");
    }
  } else {
    kv_cache_.n = 0;
    kv_cache_.head = 0;
    kv_cache_.cross_populated = false;
  }
}

void MoonshineStreamingRuntime::commit_cross_kv_from_host(int T_enc) {
  const auto &hp = assets_->hparams;
  const size_t expect = static_cast<size_t>(hp.dec_d_model) *
                        static_cast<size_t>(T_enc);

  ggml_init_params params{16ull * 1024ull * 1024ull, nullptr,
                          /*no_alloc=*/true};
  cross_kv_commit_run_.free();
  cross_kv_commit_run_.ctx = ggml_init(params);
  if (cross_kv_commit_run_.ctx == nullptr) {
    fail("failed to init cross_kv commit context");
  }

  CrossKVCommitBuild cb = build_cross_kv_commit_graph(
      cross_kv_commit_run_.ctx, hp, kv_cache_, T_enc);
  if (cb.graph == nullptr) {
    fail("cross_kv commit graph build failed");
  }

  cross_kv_commit_run_.gallocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (cross_kv_commit_run_.gallocr == nullptr ||
      !ggml_gallocr_alloc_graph(cross_kv_commit_run_.gallocr, cb.graph)) {
    fail("cross_kv commit graph allocation failed");
  }
  cross_kv_commit_run_.graph = cb.graph;

  for (int il = 0; il < hp.dec_n_layers; ++il) {
    const auto &kbuf = cross_k_committed_[static_cast<size_t>(il)];
    const auto &vbuf = cross_v_committed_[static_cast<size_t>(il)];
    if (kbuf.size() < expect || vbuf.size() < expect) {
      fail("cross_kv commit: host buffer shorter than T_enc");
    }
    ggml_backend_tensor_set(cb.per_layer_k_in[static_cast<size_t>(il)],
                            kbuf.data(), 0, expect * sizeof(float));
    ggml_backend_tensor_set(cb.per_layer_v_in[static_cast<size_t>(il)],
                            vbuf.data(), 0, expect * sizeof(float));
  }

  core::set_backend_threads(backend_,
                            std::max(1, execution_context_.config().threads));
  if (core::compute_backend_graph(backend_, cb.graph) != GGML_STATUS_SUCCESS) {
    fail("cross_kv commit compute failed");
  }
  ggml_backend_synchronize(backend_);
  kv_cache_.cross_populated = true;
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

std::string
MoonshineStreamingRuntime::decode_token_ids(const std::vector<int32_t> &ids) const {
  if (ids.empty()) {
    return {};
  }
  std::string full = assets_->tokenizer->decode(ids.data(), ids.size());
  // SentencePiece-style byte-level BPE: the vocab pieces use U+2581 as the
  // word-boundary marker; render it as a space.
  const std::string boundary = "\xE2\x96\x81";
  std::size_t pos = 0;
  while ((pos = full.find(boundary, pos)) != std::string::npos) {
    full.replace(pos, boundary.size(), " ");
    pos += 1;
  }
  if (!full.empty() && full.front() == ' ') {
    full.erase(full.begin());
  }
  return full;
}

MoonshineStreamingTranscription MoonshineStreamingRuntime::decode_from_kv_cache(
    int T_enc, const runtime::RunControl &control, int32_t max_tokens) {
  const auto &hp = assets_->hparams;

  const int decoder_start = hp.decoder_start_token_id;
  const int eos = hp.eos_token_id;

  int max_pos = hp.dec_max_position_embeddings;
  const int budget = decode_generation_budget(hp, T_enc);
  if (budget > 0) {
    max_pos = std::min(max_pos, budget);
  }
  if (max_tokens > 0) {
    max_pos = std::min(max_pos, static_cast<int>(max_tokens));
  }

  std::vector<int32_t> generated_ids;
  int next_token = -1;
  int n_past = 0;

  auto run_step = [&](int token_id, int n_past_in) {
    ggml_init_params params{8ull * 1024ull * 1024ull, nullptr,
                            /*no_alloc=*/true};
    step_run_.free();
    step_run_.ctx = ggml_init(params);
    if (step_run_.ctx == nullptr) {
      fail("failed to init step compute context");
    }
    DecoderBuild db = build_decoder_graph_kv(
        step_run_.ctx, weights_, hp, kv_cache_, /*n_tokens=*/1, n_past_in,
        T_enc, /*skip_log_softmax=*/true, /*use_flash=*/true);
    if (db.out == nullptr || db.graph == nullptr) {
      fail("step graph build failed");
    }
    step_run_.gallocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (step_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(step_run_.gallocr, db.graph)) {
      fail("step graph allocation failed");
    }
    step_run_.graph = db.graph;

    const int32_t tid = static_cast<int32_t>(token_id);
    const int32_t pid = static_cast<int32_t>(n_past_in);
    ggml_backend_tensor_set(db.token_ids_in, &tid, 0, sizeof(int32_t));
    ggml_backend_tensor_set(db.pos_ids_in, &pid, 0, sizeof(int32_t));

    core::set_backend_threads(backend_,
                              std::max(1, execution_context_.config().threads));
    if (core::compute_backend_graph(backend_, db.graph) !=
        GGML_STATUS_SUCCESS) {
      fail("decoder compute failed");
    }
    ggml_backend_synchronize(backend_);

    int32_t argmax_id = 0;
    ggml_backend_tensor_get(db.argmax_out, &argmax_id, 0, sizeof(int32_t));

    kv_cache_.n = n_past_in + 1;
    kv_cache_.head = kv_cache_.n;
    return static_cast<int>(argmax_id);
  };

  next_token = run_step(decoder_start, 0);
  if (next_token != eos) {
    generated_ids.push_back(next_token);
  }
  n_past = kv_cache_.n;

  while (next_token != eos && n_past < max_pos) {
    control.emit_progress("decode", n_past, max_pos);
    next_token = run_step(next_token, n_past);
    if (next_token != eos) {
      generated_ids.push_back(next_token);
    }
    n_past = kv_cache_.n;
  }

  MoonshineStreamingTranscription result;
  result.truncated = (next_token != eos);
  result.text = decode_token_ids(generated_ids);
  stream_token_ids_ = std::move(generated_ids);
  return result;
}

bool MoonshineStreamingRuntime::decode_partial(
    const runtime::RunControl &control) {
  const int T_enc = T_emitted_;
  if (T_enc <= 0) {
    return false;
  }

  const std::string prev_text = stream_text_;

  ensure_kv_cache_for_T(T_enc);
  commit_cross_kv_from_host(T_enc);
  MoonshineStreamingTranscription partial =
      decode_from_kv_cache(T_enc, control, stream_max_tokens_);

  stream_text_ = std::move(partial.text);
  stream_truncated_ = partial.truncated;
  last_decoded_T_ = T_enc;
  return stream_text_ != prev_text;
}

// ---------------------------------------------------------------------------
// Streaming lifecycle
// ---------------------------------------------------------------------------

void MoonshineStreamingRuntime::stream_begin(int32_t min_decode_interval_ms,
                                             int32_t max_tokens) {
  const auto &hp = assets_->hparams;

  pcm_buffer_.clear();
  pcm_start_sample_ = 0;
  adapter_committed_.clear();
  cross_k_committed_.assign(static_cast<size_t>(hp.dec_n_layers),
                            std::vector<float>{});
  cross_v_committed_.assign(static_cast<size_t>(hp.dec_n_layers),
                            std::vector<float>{});
  T_emitted_ = 0;
  last_decoded_T_ = 0;
  stream_text_.clear();
  stream_token_ids_.clear();
  stream_truncated_ = false;
  stream_max_tokens_ = max_tokens;

  L_total_frames_ = cumulative_left_context(hp);
  R_total_frames_ = cumulative_right_context(hp);
  frontend_pad_frames_ = kFrontendPadEncFrames;
  samples_per_enc_frame_ = samples_per_encoder_frame(hp);

  const int32_t min_ms =
      min_decode_interval_ms < 0 ? kDefaultMinDecodeIntervalMs
                                 : min_decode_interval_ms;
  const int64_t spf =
      samples_per_enc_frame_ > 0 ? samples_per_enc_frame_ : 1;
  const int64_t min_decode_samples =
      static_cast<int64_t>(min_ms) * kNativeSampleRateHz;
  min_decode_frames_ = static_cast<int32_t>(
      (min_decode_samples + 1000LL * spf - 1) / (1000LL * spf));
  if (min_decode_frames_ < 1) {
    min_decode_frames_ = 1;
  }

  kv_cache_.n = 0;
  kv_cache_.head = 0;
  kv_cache_.cross_populated = false;
  streaming_ = true;
}

bool MoonshineStreamingRuntime::stream_feed(const float *pcm, size_t n_samples,
                                            const runtime::RunControl &control) {
  if (!streaming_) {
    fail("stream_feed before stream_begin");
  }
  if (pcm != nullptr && n_samples > 0) {
    pcm_buffer_.insert(pcm_buffer_.end(), pcm, pcm + n_samples);
  }

  const int spf = samples_per_enc_frame_;
  const int64_t total_pcm_abs =
      pcm_start_sample_ + static_cast<int64_t>(pcm_buffer_.size());
  const int available_frames = static_cast<int>(total_pcm_abs / spf);
  const int R = R_total_frames_;
  const int L = L_total_frames_;

  const int stable_frames = available_frames - R;
  bool changed = false;

  if (stable_frames > T_emitted_) {
    const int emit_start = T_emitted_;
    const int emit_end = stable_frames;
    const int win_start = std::max(0, emit_start - L);
    const int win_end = emit_end + R;
    flush_stable_frames(win_start, win_end, emit_start, emit_end);

    trim_pcm_buffer();

    // Per-feed partial decode: re-run the AR decoder from BOS over the
    // extended cross-KV. Skipped when nothing new committed, or when the
    // caller-requested throttle has not elapsed. Frames not decoded now are
    // still committed to the host buffers and get picked up by the next feed
    // that crosses the throttle, or by finalize regardless.
    const int frames_since_last_decode = T_emitted_ - last_decoded_T_;
    if (T_emitted_ > last_decoded_T_ &&
        frames_since_last_decode >= min_decode_frames_) {
      changed = decode_partial(control);
    }
  }
  return changed;
}

MoonshineStreamingTranscription
MoonshineStreamingRuntime::stream_finalize(const runtime::RunControl &control) {
  const auto &hp = assets_->hparams;

  MoonshineStreamingTranscription result;

  if (!streaming_) {
    return result;
  }

  // Empty stream: mirror transcribe() on zero-sample input.
  if (pcm_buffer_.empty() && pcm_start_sample_ == 0) {
    streaming_ = false;
    kv_cache_.free();
    return result;
  }

  // Right-pad the trailing buffer to a multiple of enc_frame_len so the
  // encoder reshape divides evenly. pcm_start_sample_ is already
  // frame-aligned, so a local pad keeps buffer-relative offsets clean.
  {
    const int orig = static_cast<int>(pcm_buffer_.size());
    const int rem = orig % hp.enc_frame_len;
    const int pad = (rem == 0) ? 0 : (hp.enc_frame_len - rem);
    if (pad > 0) {
      pcm_buffer_.resize(static_cast<size_t>(orig + pad), 0.0f);
    }
  }

  const int spf = samples_per_enc_frame_;
  const int64_t total_pcm_abs =
      pcm_start_sample_ + static_cast<int64_t>(pcm_buffer_.size());
  const int T_total = static_cast<int>(total_pcm_abs / spf);

  // Flush the tail: no right context is available at end of stream, so the
  // window ends exactly at emit_end.
  if (T_total > T_emitted_) {
    const int emit_start = T_emitted_;
    const int emit_end = T_total;
    const int win_start = std::max(0, emit_start - L_total_frames_);
    const int win_end = emit_end;
    flush_stable_frames(win_start, win_end, emit_start, emit_end);
  }

  const int T_enc = T_emitted_;
  if (T_enc <= 0) {
    streaming_ = false;
    kv_cache_.free();
    return result;
  }

  // Re-decode only when frames advanced since the last partial; otherwise the
  // last feed already produced the final transcript.
  if (T_enc > last_decoded_T_) {
    ensure_kv_cache_for_T(T_enc);
    commit_cross_kv_from_host(T_enc);
    MoonshineStreamingTranscription final_pass =
        decode_from_kv_cache(T_enc, control, stream_max_tokens_);
    stream_text_ = std::move(final_pass.text);
    stream_truncated_ = final_pass.truncated;
    last_decoded_T_ = T_enc;
  }

  result.text = stream_text_;
  result.truncated = stream_truncated_;

  streaming_ = false;
  // Free GPU buffers after each utterance to prevent accumulation across
  // repeated runs (matches the arch cleanup_gpu contract).
  kv_cache_.free();
  return result;
}

void MoonshineStreamingRuntime::stream_reset() {
  pcm_buffer_.clear();
  pcm_start_sample_ = 0;
  adapter_committed_.clear();
  cross_k_committed_.clear();
  cross_v_committed_.clear();
  T_emitted_ = 0;
  last_decoded_T_ = 0;
  stream_text_.clear();
  stream_token_ids_.clear();
  stream_truncated_ = false;
  streaming_ = false;
  kv_cache_.free();
}

// ---------------------------------------------------------------------------
// Offline one-shot
// ---------------------------------------------------------------------------

MoonshineStreamingTranscription MoonshineStreamingRuntime::transcribe(
    const runtime::AudioBuffer &audio, const runtime::RunControl &control,
    int32_t max_tokens) {
  const auto &hp = assets_->hparams;

  std::vector<float> pcm =
      engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
          audio.samples, audio.sample_rate, audio.channels, hp.fe_sample_rate);
  if (pcm.empty()) {
    fail("empty audio input");
  }
  // Right-pad to a whole number of embedder frames.
  {
    const int rem = static_cast<int>(pcm.size()) % hp.enc_frame_len;
    if (rem != 0) {
      pcm.resize(pcm.size() + static_cast<size_t>(hp.enc_frame_len - rem),
                 0.0f);
    }
  }

  control.emit_progress("encode", 0, 1);

  std::vector<float> enc_host;
  int T_enc = 0;
  encode_window_to_host(pcm.data(), static_cast<int>(pcm.size()), enc_host,
                        T_enc);
  if (T_enc <= 0) {
    fail("input too short for encoder");
  }

  std::vector<float> adapter_host;
  apply_adapter_window(enc_host.data(), T_enc, /*abs_frame_offset=*/0,
                       adapter_host);

  cross_k_committed_.assign(static_cast<size_t>(hp.dec_n_layers),
                            std::vector<float>{});
  cross_v_committed_.assign(static_cast<size_t>(hp.dec_n_layers),
                            std::vector<float>{});
  project_cross_kv_window(adapter_host.data(), T_enc, cross_k_committed_,
                          cross_v_committed_);

  T_emitted_ = T_enc;
  ensure_kv_cache_for_T(T_enc);
  commit_cross_kv_from_host(T_enc);

  control.emit_progress("encode", 1, 1);

  MoonshineStreamingTranscription result =
      decode_from_kv_cache(T_enc, control, max_tokens);

  // Offline runs are self-contained; release the cache like the arch path.
  kv_cache_.free();
  return result;
}

} // namespace engine::models::moonshine_streaming
