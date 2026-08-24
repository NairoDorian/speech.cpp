// engine/models/moonshine/runtime.cpp - inference orchestration for the
// native engine Moonshine package.
//
// Mirrors the arch run() flow: encoder graph -> host readback -> cross-KV
// precompute -> greedy single-token decode loop (dynamic per-step graphs,
// which is also the arch CPU path). Cancellation polls RunControl at every
// decode step via emit_progress.

#include "engine/models/moonshine/runtime.h"

#include "engine/models/moonshine/graphs_internal.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::moonshine {

namespace {

constexpr const char *kTag = "moonshine";

// Weight slot upload helper: creates the backend tensor with the expected
// ggml ne-order shape and queues a native-type copy from the source.
core::TensorValue load_slot(core::BackendWeightStore &store,
                            const assets::TensorSource &source,
                            const char *name,
                            assets::TensorStorageType storage_type,
                            std::initializer_list<int64_t> shape) {
  return store.load_tensor(source, name, storage_type, shape);
}

} // namespace

void MoonshineRuntime::GraphRun::free() {
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

MoonshineRuntime::MoonshineRuntime(
    std::shared_ptr<const MoonshineAssets> assets,
    core::ExecutionContext &execution_context,
    assets::TensorStorageType storage_type)
    : assets_(std::move(assets)), execution_context_(execution_context) {
  if (!assets_ || !assets_->source) {
    throw std::runtime_error(std::string(kTag) +
                             ": runtime requires loaded assets");
  }
  backend_ = execution_context_.backend();
  if (backend_ == nullptr) {
    throw std::runtime_error(std::string(kTag) +
                             ": execution backend is not initialized");
  }

  store_ = std::make_shared<core::BackendWeightStore>(
      backend_, execution_context_.backend_type(), "moonshine.weights",
      256ull * 1024ull * 1024ull);

  auto &store = *store_;
  const auto &src = *assets_->source;
  const auto st = storage_type;
  const auto &hp = assets_->hparams;

  const int64_t d_model = hp.enc_d_model;
  const int64_t enc_ff = hp.enc_ffn_dim;
  const int64_t dec_h = hp.dec_d_model;
  const int64_t dec_ff = hp.dec_ffn_dim;
  const int64_t vocab = hp.dec_vocab_size;
  const int64_t k0 = hp.conv_kernel_sizes[0];
  const int64_t k1 = hp.conv_kernel_sizes[1];
  const int64_t k2 = hp.conv_kernel_sizes[2];
  const int64_t c0 = hp.conv_channels[0];
  const int64_t c1 = hp.conv_channels[1];
  const int64_t c2 = hp.conv_channels[2];

  // Encoder conv stem (PyTorch [out, in, K] stored as ggml ne [K, in, out]).
  weights_.enc_stem.conv0_w =
      load_slot(store, src, "enc.conv.0.weight", st, {c0, 1, k0}).tensor;
  weights_.enc_stem.conv1_w =
      load_slot(store, src, "enc.conv.1.weight", st, {c1, c0, k1}).tensor;
  weights_.enc_stem.conv1_b =
      store.load_f32_tensor(src, "enc.conv.1.bias", {c1}).tensor;
  weights_.enc_stem.conv2_w =
      load_slot(store, src, "enc.conv.2.weight", st, {c2, c1, k2}).tensor;
  weights_.enc_stem.conv2_b =
      store.load_f32_tensor(src, "enc.conv.2.bias", {c2}).tensor;
  weights_.enc_stem.gn_w =
      store.load_f32_tensor(src, "enc.conv.norm.weight", {d_model}).tensor;
  weights_.enc_stem.gn_b =
      store.load_f32_tensor(src, "enc.conv.norm.bias", {d_model}).tensor;

  weights_.enc_top.final_norm_w =
      store.load_f32_tensor(src, "enc.final_norm.weight", {d_model}).tensor;

  weights_.enc_blocks.resize(static_cast<size_t>(hp.enc_n_layers));
  for (int i = 0; i < hp.enc_n_layers; ++i) {
    auto &b = weights_.enc_blocks[static_cast<size_t>(i)];
    b.norm_attn_w = store
                        .load_f32_tensor(src,
                                         ("enc.blocks." + std::to_string(i) +
                                          ".norm_attn.weight")
                                             .c_str(),
                                         {d_model})
                        .tensor;
    b.attn_q_w =
        load_slot(
            store, src,
            ("enc.blocks." + std::to_string(i) + ".attn.q.weight").c_str(), st,
            {d_model, d_model})
            .tensor;
    b.attn_k_w =
        load_slot(
            store, src,
            ("enc.blocks." + std::to_string(i) + ".attn.k.weight").c_str(), st,
            {d_model, d_model})
            .tensor;
    b.attn_v_w =
        load_slot(
            store, src,
            ("enc.blocks." + std::to_string(i) + ".attn.v.weight").c_str(), st,
            {d_model, d_model})
            .tensor;
    b.attn_out_w =
        load_slot(
            store, src,
            ("enc.blocks." + std::to_string(i) + ".attn.out.weight").c_str(),
            st, {d_model, d_model})
            .tensor;
    b.norm_ffn_w = store
                       .load_f32_tensor(src,
                                        ("enc.blocks." + std::to_string(i) +
                                         ".norm_ffn.weight")
                                            .c_str(),
                                        {d_model})
                       .tensor;
    b.ffn_fc1_w =
        load_slot(
            store, src,
            ("enc.blocks." + std::to_string(i) + ".ffn.fc1.weight").c_str(), st,
            {enc_ff, d_model})
            .tensor;
    b.ffn_fc1_b =
        store
            .load_f32_tensor(
                src,
                ("enc.blocks." + std::to_string(i) + ".ffn.fc1.bias").c_str(),
                {enc_ff})
            .tensor;
    b.ffn_fc2_w =
        load_slot(
            store, src,
            ("enc.blocks." + std::to_string(i) + ".ffn.fc2.weight").c_str(), st,
            {d_model, enc_ff})
            .tensor;
    b.ffn_fc2_b =
        store
            .load_f32_tensor(
                src,
                ("enc.blocks." + std::to_string(i) + ".ffn.fc2.bias").c_str(),
                {d_model})
            .tensor;
  }

  weights_.dec_top.token_embd_w =
      load_slot(store, src, "dec.token_embd.weight", st, {vocab, dec_h}).tensor;
  weights_.dec_top.final_norm_w =
      store.load_f32_tensor(src, "dec.final_norm.weight", {dec_h}).tensor;

  weights_.dec_blocks.resize(static_cast<size_t>(hp.dec_n_layers));
  for (int i = 0; i < hp.dec_n_layers; ++i) {
    auto &b = weights_.dec_blocks[static_cast<size_t>(i)];
    b.norm_self_w = store
                        .load_f32_tensor(src,
                                         ("dec.blocks." + std::to_string(i) +
                                          ".norm_self.weight")
                                             .c_str(),
                                         {dec_h})
                        .tensor;
    b.self_q_w =
        load_slot(
            store, src,
            ("dec.blocks." + std::to_string(i) + ".self_attn.q.weight").c_str(),
            st, {dec_h, dec_h})
            .tensor;
    b.self_k_w =
        load_slot(
            store, src,
            ("dec.blocks." + std::to_string(i) + ".self_attn.k.weight").c_str(),
            st, {dec_h, dec_h})
            .tensor;
    b.self_v_w =
        load_slot(
            store, src,
            ("dec.blocks." + std::to_string(i) + ".self_attn.v.weight").c_str(),
            st, {dec_h, dec_h})
            .tensor;
    b.self_out_w =
        load_slot(store, src,
                  ("dec.blocks." + std::to_string(i) + ".self_attn.out.weight")
                      .c_str(),
                  st, {dec_h, dec_h})
            .tensor;
    b.norm_cross_w = store
                         .load_f32_tensor(src,
                                          ("dec.blocks." + std::to_string(i) +
                                           ".norm_cross.weight")
                                              .c_str(),
                                          {dec_h})
                         .tensor;
    b.cross_q_w =
        load_slot(store, src,
                  ("dec.blocks." + std::to_string(i) + ".cross_attn.q.weight")
                      .c_str(),
                  st, {dec_h, dec_h})
            .tensor;
    b.cross_k_w =
        load_slot(store, src,
                  ("dec.blocks." + std::to_string(i) + ".cross_attn.k.weight")
                      .c_str(),
                  st, {dec_h, dec_h})
            .tensor;
    b.cross_v_w =
        load_slot(store, src,
                  ("dec.blocks." + std::to_string(i) + ".cross_attn.v.weight")
                      .c_str(),
                  st, {dec_h, dec_h})
            .tensor;
    b.cross_out_w =
        load_slot(store, src,
                  ("dec.blocks." + std::to_string(i) + ".cross_attn.out.weight")
                      .c_str(),
                  st, {dec_h, dec_h})
            .tensor;
    b.norm_ffn_w = store
                       .load_f32_tensor(src,
                                        ("dec.blocks." + std::to_string(i) +
                                         ".norm_ffn.weight")
                                            .c_str(),
                                        {dec_h})
                       .tensor;
    b.ffn_fc1_w =
        load_slot(
            store, src,
            ("dec.blocks." + std::to_string(i) + ".ffn.fc1.weight").c_str(), st,
            {2 * dec_ff, dec_h})
            .tensor;
    b.ffn_fc1_b =
        store
            .load_f32_tensor(
                src,
                ("dec.blocks." + std::to_string(i) + ".ffn.fc1.bias").c_str(),
                {2 * dec_ff})
            .tensor;
    b.ffn_fc2_w =
        load_slot(
            store, src,
            ("dec.blocks." + std::to_string(i) + ".ffn.fc2.weight").c_str(), st,
            {dec_h, dec_ff})
            .tensor;
    b.ffn_fc2_b =
        store
            .load_f32_tensor(
                src,
                ("dec.blocks." + std::to_string(i) + ".ffn.fc2.bias").c_str(),
                {dec_h})
            .tensor;
  }

  store.upload();
  assets_->source->release_storage();
}

MoonshineRuntime::~MoonshineRuntime() {
  kv_cache_.free();
  encoder_run_.free();
  cross_kv_run_.free();
  step_run_.free();
}

MoonshineTranscription
MoonshineRuntime::transcribe(const runtime::AudioBuffer &audio,
                             const runtime::RunControl &control,
                             int32_t max_tokens) {
  const auto &hp = assets_->hparams;

  std::vector<float> pcm =
      engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
          audio.samples, audio.sample_rate, audio.channels, hp.fe_sample_rate);
  const int n_samples = static_cast<int>(pcm.size());
  if (n_samples <= 0) {
    throw std::runtime_error(std::string(kTag) + ": empty audio input");
  }

  control.emit_progress("encode", 0, 1);

  // ----- encoder -----
  {
    ggml_init_params params{8ull * 1024ull * 1024ull, nullptr,
                            /*no_alloc=*/true};
    encoder_run_.free();
    encoder_run_.ctx = ggml_init(params);
    if (encoder_run_.ctx == nullptr) {
      throw std::runtime_error(std::string(kTag) +
                               ": failed to init encoder compute context");
    }
    EncoderBuild eb = build_encoder_graph(encoder_run_.ctx, weights_, hp,
                                          n_samples, /*use_flash=*/true);
    if (eb.audio_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
      throw std::runtime_error(std::string(kTag) +
                               ": input too short for encoder");
    }
    encoder_run_.gallocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (encoder_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(encoder_run_.gallocr, eb.graph)) {
      throw std::runtime_error(std::string(kTag) +
                               ": encoder graph allocation failed");
    }
    encoder_run_.graph = eb.graph;

    ggml_backend_tensor_set(eb.audio_in, pcm.data(), 0,
                            pcm.size() * sizeof(float));
    std::vector<int32_t> pos_ids(static_cast<size_t>(eb.T_enc));
    for (int i = 0; i < eb.T_enc; ++i) {
      pos_ids[static_cast<size_t>(i)] = i;
    }
    ggml_backend_tensor_set(eb.pos_ids_in, pos_ids.data(), 0,
                            pos_ids.size() * sizeof(int32_t));

    core::set_backend_threads(backend_,
                              std::max(1, execution_context_.config().threads));
    if (core::compute_backend_graph(backend_, eb.graph) !=
        GGML_STATUS_SUCCESS) {
      throw std::runtime_error(std::string(kTag) + ": encoder compute failed");
    }
    ggml_backend_synchronize(backend_);

    // Read the encoder output to host: the cross-KV graph lives in a
    // fresh compute context that cannot share tensor handles.
    enc_T_ = eb.T_enc;
    enc_host_.assign(static_cast<size_t>(hp.dec_d_model) *
                         static_cast<size_t>(eb.T_enc),
                     0.0f);
    ggml_backend_tensor_get(eb.out, enc_host_.data(), 0,
                            enc_host_.size() * sizeof(float));
  }

  // ----- KV cache -----
  {
    if (kv_cache_.buffer != nullptr && kv_cache_.T_enc != enc_T_) {
      kv_cache_.free();
    }
    if (kv_cache_.buffer == nullptr) {
      const int n_ctx = hp.dec_max_position_embeddings > 0
                            ? hp.dec_max_position_embeddings
                            : 512;
      if (!kv_cache_init(kv_cache_, backend_, n_ctx, enc_T_, hp.dec_d_model,
                         hp.dec_n_layers, GGML_TYPE_F32)) {
        throw std::runtime_error(std::string(kTag) +
                                 ": KV cache allocation failed");
      }
    } else {
      kv_cache_.n = 0;
      kv_cache_.head = 0;
      kv_cache_.cross_populated = false;
    }
  }

  // ----- cross-KV precompute -----
  {
    ggml_init_params params{4ull * 1024ull * 1024ull, nullptr,
                            /*no_alloc=*/true};
    cross_kv_run_.free();
    cross_kv_run_.ctx = ggml_init(params);
    if (cross_kv_run_.ctx == nullptr) {
      throw std::runtime_error(std::string(kTag) +
                               ": failed to init cross_kv compute context");
    }
    DecoderBuild db = build_cross_kv_graph(cross_kv_run_.ctx, weights_, hp,
                                           kv_cache_, enc_T_);
    if (db.graph == nullptr) {
      throw std::runtime_error(std::string(kTag) +
                               ": cross_kv graph build failed");
    }
    cross_kv_run_.gallocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (cross_kv_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(cross_kv_run_.gallocr, db.graph)) {
      throw std::runtime_error(std::string(kTag) +
                               ": cross_kv graph allocation failed");
    }
    cross_kv_run_.graph = db.graph;

    ggml_backend_tensor_set(db.encoder_out_in, enc_host_.data(), 0,
                            enc_host_.size() * sizeof(float));
    if (core::compute_backend_graph(backend_, db.graph) !=
        GGML_STATUS_SUCCESS) {
      throw std::runtime_error(std::string(kTag) + ": cross_kv compute failed");
    }
    ggml_backend_synchronize(backend_);
    kv_cache_.cross_populated = true;
  }

  control.emit_progress("encode", 1, 1);

  // ----- greedy decode loop -----
  const int decoder_start = hp.decoder_start_token_id;
  const int eos = hp.eos_token_id;
  int max_pos = hp.dec_max_position_embeddings;
  if (max_tokens > 0) {
    max_pos = std::min(max_pos, max_tokens);
  }

  std::vector<int32_t> generated_ids;
  int next_token = -1;
  int n_past = 0;

  // One decoder step. Returns the argmax token and advances the cache.
  auto run_step = [&](int token_id, int n_past_in) {
    ggml_init_params params{4ull * 1024ull * 1024ull, nullptr,
                            /*no_alloc=*/true};
    step_run_.free();
    step_run_.ctx = ggml_init(params);
    if (step_run_.ctx == nullptr) {
      throw std::runtime_error(std::string(kTag) +
                               ": failed to init step compute context");
    }
    DecoderBuild db =
        build_decoder_graph_kv(step_run_.ctx, weights_, hp, kv_cache_,
                               /*n_tokens=*/1, n_past_in, enc_T_,
                               /*skip_log_softmax=*/true, /*use_flash=*/true);
    if (db.out == nullptr || db.graph == nullptr) {
      throw std::runtime_error(std::string(kTag) + ": step graph build failed");
    }
    step_run_.gallocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (step_run_.gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(step_run_.gallocr, db.graph)) {
      throw std::runtime_error(std::string(kTag) +
                               ": step graph allocation failed");
    }
    step_run_.graph = db.graph;

    std::vector<int32_t> token_ids = {static_cast<int32_t>(token_id)};
    std::vector<int32_t> pos_ids = {static_cast<int32_t>(n_past_in)};
    ggml_backend_tensor_set(db.token_ids_in, token_ids.data(), 0,
                            sizeof(int32_t));
    ggml_backend_tensor_set(db.pos_ids_in, pos_ids.data(), 0, sizeof(int32_t));

    core::set_backend_threads(backend_,
                              std::max(1, execution_context_.config().threads));
    if (core::compute_backend_graph(backend_, db.graph) !=
        GGML_STATUS_SUCCESS) {
      throw std::runtime_error(std::string(kTag) + ": decoder compute failed");
    }
    ggml_backend_synchronize(backend_);

    int32_t argmax_id = 0;
    ggml_backend_tensor_get(db.argmax_out, &argmax_id, 0, sizeof(int32_t));

    kv_cache_.n = n_past_in + 1;
    kv_cache_.head = kv_cache_.n;
    return argmax_id;
  };

  // Prompt pass: single decoder_start token.
  next_token = run_step(decoder_start, 0);
  if (next_token != eos) {
    generated_ids.push_back(next_token);
  }
  n_past = kv_cache_.n;

  while (next_token != eos && n_past < max_pos) {
    // Progress + cancellation at every step (throws ProgressCanceled on
    // abort request or callback decline).
    control.emit_progress("decode", n_past, max_pos);
    next_token = run_step(next_token, n_past);
    if (next_token != eos) {
      generated_ids.push_back(next_token);
    }
    n_past = kv_cache_.n;
  }

  MoonshineTranscription result;
  result.truncated = (next_token != eos);
  if (!generated_ids.empty()) {
    std::string full =
        assets_->tokenizer->decode(generated_ids.data(), generated_ids.size());
    // SentencePiece-style byte-level BPE: the vocab pieces use ▁
    // (U+2581) as the word-boundary marker; render it as a space.
    const std::string boundary = "\xE2\x96\x81";
    std::size_t pos = 0;
    while ((pos = full.find(boundary, pos)) != std::string::npos) {
      full.replace(pos, boundary.size(), " ");
      pos += 1;
    }
    if (!full.empty() && full.front() == ' ') {
      full.erase(full.begin());
    }
    result.text = std::move(full);
  }
  return result;
}

} // namespace engine::models::moonshine
