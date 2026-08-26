#include "engine/framework/modules/optimizations/fast_kv_modules.h"

#include "../tensor_layout_utils.h"

#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kSetRowsInputs[] = {
    {"cache", core::PortKind::Activation, false},
    {"row", core::PortKind::Activation, false},
    {"row_index", core::PortKind::Activation, false},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kFastKVSetRowsSchema = {
    "FastKVSetRows",
    "tensor.optimization",
    kSetRowsInputs,
    3,
    kSingleOutput,
    1,
    "Appends one or more flattened KV rows into a cache tensor using ggml_set_rows.",
};

}  // namespace

FastKVSetRowsModule::FastKVSetRowsModule(FastKVSetRowsConfig config) : config_(config) {}

const core::ModuleSchema & FastKVSetRowsModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue FastKVSetRowsModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    const core::TensorValue & row,
    const core::TensorValue & row_index) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(cache, 4, 4, "cache");
    core::validate_rank_between(row, 4, 4, "row");
    const int64_t batch = cache.shape.dims[0];
    // rows > 1 is the multi-row single-sequence mode (Phase 10.5): `rows`
    // consecutive tokens of ONE sequence land at `rows` explicit slots - the
    // primitive a speculative verify pass is built on. rows == 1 keeps the
    // historical {1} / {batch} row_index contract byte-for-byte.
    const int64_t rows = row.shape.dims[1];
    if (row.shape.dims[0] != batch || rows <= 0 || row.shape.dims[2] != cache.shape.dims[2] ||
        row.shape.dims[3] != cache.shape.dims[3]) {
        throw std::runtime_error("FastKVSetRowsModule row must have shape {batch, rows, heads, dim} matching the cache");
    }
    const bool multi_row = rows > 1;
    if (multi_row && batch != 1) {
        throw std::runtime_error("FastKVSetRowsModule multi-row mode requires batch size 1");
    }
    if (row_index.shape.rank != 1 ||
        (multi_row ? row_index.shape.dims[0] != rows
                   : (row_index.shape.dims[0] != 1 && row_index.shape.dims[0] != batch))) {
        throw std::runtime_error(
            multi_row ? "FastKVSetRowsModule multi-row mode requires row_index of shape {rows}"
                      : "FastKVSetRowsModule row_index must have shape {1} or {batch}");
    }
    const bool optimized = config_.mode == FastKVSetRowsMode::BackendViewOptimized;
    if (((!optimized && cache.type != GGML_TYPE_F32) ||
         (optimized && cache.type != GGML_TYPE_F32 && cache.type != GGML_TYPE_F16 && cache.type != GGML_TYPE_BF16)) ||
        row.type != GGML_TYPE_F32) {
        throw std::runtime_error(
            optimized
                ? "FastKVSetRowsModule requires an f32/f16/bf16 cache and an f32 row tensor"
                : "FastKVSetRowsModule requires f32 cache and row tensors");
    }
    if (row_index.type != GGML_TYPE_I32 && row_index.type != GGML_TYPE_I64) {
        throw std::runtime_error("FastKVSetRowsModule requires i32 or i64 row_index tensor");
    }
    if (!core::has_backend_addressable_layout(cache.tensor)) {
        throw std::runtime_error("FastKVSetRowsModule requires a contiguous cache tensor");
    }

    const int64_t steps = cache.shape.dims[1];
    const int64_t row_elems = cache.shape.dims[2] * cache.shape.dims[3];
    if (multi_row) {
        auto flat_cache = core::reshape_tensor(ctx, cache, core::TensorShape::from_dims({steps, row_elems}));
        auto contiguous_row = tensor_layout::ensure_contiguous_layout_if_needed(ctx, row);
        auto flat_row = optimized
            ? core::wrap_tensor(
                  ggml_view_2d(
                      ctx.ggml,
                      contiguous_row.tensor,
                      row_elems,
                      rows,
                      contiguous_row.tensor->nb[2],
                      0),
                  core::TensorShape::from_dims({rows, row_elems}),
                  row.type)
            : core::reshape_tensor(ctx, contiguous_row, core::TensorShape::from_dims({rows, row_elems}));
        ggml_tensor * updated = ggml_set_rows(ctx.ggml, flat_cache.tensor, flat_row.tensor, row_index.tensor);
        if (optimized) {
            updated->src[2] = cache.tensor;
        }
        auto flat_updated = core::wrap_tensor(updated, flat_cache.shape, cache.type);
        return core::reshape_tensor(ctx, flat_updated, cache.shape);
    }
    if (row_index.shape.dims[0] == 1) {
        if (batch != 1) {
            throw std::runtime_error("FastKVSetRowsModule single row_index mode requires batch size 1");
        }
        auto flat_cache = core::reshape_tensor(ctx, cache, core::TensorShape::from_dims({steps, row_elems}));
        auto contiguous_row = tensor_layout::ensure_contiguous_layout_if_needed(ctx, row);
        auto flat_row = optimized
            ? core::wrap_tensor(
                  ggml_view_2d(
                      ctx.ggml,
                      contiguous_row.tensor,
                      row_elems,
                      1,
                      contiguous_row.tensor->nb[2],
                      0),
                  core::TensorShape::from_dims({1, row_elems}),
                  row.type)
            : core::reshape_tensor(ctx, contiguous_row, core::TensorShape::from_dims({1, row_elems}));
        ggml_tensor * updated = ggml_set_rows(ctx.ggml, flat_cache.tensor, flat_row.tensor, row_index.tensor);
        if (optimized) {
            updated->src[2] = cache.tensor;
        }
        auto flat_updated = core::wrap_tensor(updated, flat_cache.shape, cache.type);
        return core::reshape_tensor(ctx, flat_updated, cache.shape);
    }

    auto flat_cache = core::reshape_tensor(ctx, cache, core::TensorShape::from_dims({batch * steps, row_elems}));
    auto contiguous_row = tensor_layout::ensure_contiguous_layout_if_needed(ctx, row);
    auto flat_row = optimized
        ? core::wrap_tensor(
              ggml_view_2d(
                  ctx.ggml,
                  contiguous_row.tensor,
                  row_elems,
                  batch,
                  contiguous_row.tensor->nb[3],
                  0),
              core::TensorShape::from_dims({batch, row_elems}),
              row.type)
        : core::reshape_tensor(ctx, contiguous_row, core::TensorShape::from_dims({batch, row_elems}));
    ggml_tensor * updated = ggml_set_rows(ctx.ggml, flat_cache.tensor, flat_row.tensor, row_index.tensor);
    if (optimized) {
        updated->src[2] = cache.tensor;
    }
    auto flat_updated = core::wrap_tensor(updated, flat_cache.shape, cache.type);
    return core::reshape_tensor(ctx, flat_updated, cache.shape);
}

const core::ModuleSchema & FastKVSetRowsModule::static_schema() noexcept {
    return kFastKVSetRowsSchema;
}

}  // namespace engine::modules
