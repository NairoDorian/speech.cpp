#pragma once

// engine/models/granite_nar/runtime.h - per-session inference state of the
// native engine Granite Speech NAR package: the backend weight store, the
// ported mel frontend, and the single-pass editor forward (the arch's run(),
// at transcribe.cpp HEAD's revision).

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/models/granite_nar/assets.h"
#include "engine/models/granite_nar/graphs.h"
#include "engine/models/granite_nar/mel_frontend.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::granite_nar {

struct GraniteNarRuntimeOptions {
  // Session context knob (transcribe_session_params.n_ctx): lowers, never
  // raises, the editor context ceiling. 0 = dec_max_pos_emb.
  int n_ctx = 0;
  // Shaw positional-bias form (graphs.h). Skew = the parent's current code.
  ShawBias shaw_bias = ShawBias::Skew;
};

struct GraniteNarTranscription {
  std::string text;          // tokenizer decode of the edited ids (the arch's full_text == raw_text)
  int n_audio_tokens = 0;
  int n_hypothesis_tokens = 0; // initial BPE-CTC hypothesis length
  int n_text_slots = 0;        // editor text positions (max(2n + 1, 8))
};

class GraniteNarRuntime {
public:
  GraniteNarRuntime(std::shared_ptr<const GraniteNarAssets> assets,
                    core::ExecutionContext &execution_context, GraniteNarRuntimeOptions options);
  ~GraniteNarRuntime();

  GraniteNarRuntime(const GraniteNarRuntime &) = delete;
  GraniteNarRuntime &operator=(const GraniteNarRuntime &) = delete;

  // One utterance (the arch's run()); `pcm` is mono at sample_rate(). Throws
  // std::invalid_argument where the arch returned INVALID_ARG,
  // runtime::InputTooLong (an invalid_argument; the C ABI adapter maps it to
  // TRANSCRIBE_ERR_INPUT_TOO_LONG) where it returned INPUT_TOO_LONG,
  // runtime::ProgressCanceled on abort (polled between stages and BPE-CTC
  // chunks), std::runtime_error on backend failures.
  GraniteNarTranscription transcribe(const std::vector<float> &pcm,
                                     const runtime::RunControl &control);

  int32_t sample_rate() const noexcept;

private:
  // A no_alloc graph context + its gallocr; freed after every stage so a
  // session does not pin peak activation memory between runs (the arch
  // released its scheduler after every run, 9aa6599f).
  struct GraphRun {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;

    GraphRun() = default;
    GraphRun(const GraphRun &) = delete;
    GraphRun &operator=(const GraphRun &) = delete;
    ~GraphRun() { free(); }
    void free();
    void init(const char *what);
    void allocate(ggml_backend_t backend, ggml_cgraph *graph, const char *what);
  };

  void load_weights();
  ggml_tensor *load(const std::string &name, std::initializer_list<int64_t> ne, int kind);
  void fuse_batch_norm();

  int n_threads() const;
  void compute(ggml_cgraph *graph, const char *what);

  // Encoder: cat_out (host-row-major [T_enc][cat_h]) and the per-frame
  // non-blank posterior of the self-conditioning CTC head.
  void encode(const std::vector<float> &mel_stacked, int T_enc, std::vector<float> &enc_cat,
              int64_t &cat_h, int64_t &final_offset, std::vector<float> &non_blank);
  // Initial hypothesis: bounded BPE-CTC projection + greedy collapse.
  std::vector<int32_t> bpe_hypothesis(const std::vector<float> &enc_cat, int64_t cat_h,
                                      int T_enc, int64_t final_offset,
                                      const std::vector<float> &non_blank,
                                      const runtime::RunControl &control);
  // Projector -> [llm_dim][n_audio] host rows, clipped to T_enc / downsample.
  std::vector<float> project(const std::vector<float> &enc_cat, int T_enc, int &n_audio);
  // Editor forward -> [n_text][vocab] host logits.
  std::vector<float> edit(const std::vector<float> &audio_rows, int n_audio,
                          const std::vector<int32_t> &text_ids, int &vocab);

  std::shared_ptr<const GraniteNarAssets> assets_;
  core::ExecutionContext &execution_context_;
  GraniteNarRuntimeOptions options_;
  ggml_backend_t backend_ = nullptr;
  bool conv_dw_direct_ = true;

  std::shared_ptr<core::BackendWeightStore> store_;
  GraniteNarWeights weights_{};
  std::optional<GraniteMelFrontend> mel_;
};

} // namespace engine::models::granite_nar
