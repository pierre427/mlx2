#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "arena.h"
#include "arena_storage.h"
#include "mlx/backend/metal/device.h"
#include "mlx/device.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {
using ArenaPtr = std::shared_ptr<mlx2::paged_kv::Arena>;
constexpr const char* kCapsuleName = "mlx2.paged_kv.Arena.v1";

void destroy_arena(void* pointer) noexcept {
  delete static_cast<ArenaPtr*>(pointer);
}

ArenaPtr& arena_from(const nb::capsule& capsule) {
  return *static_cast<ArenaPtr*>(capsule.data(kCapsuleName));
}
} // namespace

NB_MODULE(_paged_kv_native, module) {
  module.doc() = "Experimental default-off paged KV byte arena.";
  module.def("arena_storage_capability", [] {
    auto* device = mx::metal::device(mx::Device::gpu).mtl_device();
    if (device == nullptr) throw std::invalid_argument("Metal device unavailable");
    nb::dict result;
    result["version"] = 1;
    result["layout"] = "contiguous_uint8_2d_large";
    result["large_plane_threshold_bytes"] = std::numeric_limits<int32_t>::max();
    result["large_plane_alignment_bytes"] = mlx2::paged_kv::kArenaLargePlaneRowBytes;
    result["max_plane_bytes"] = mlx2::paged_kv::arena_max_plane_bytes(device->maxBufferLength());
    result["exact_byte_allocation"] = true;
    return result;
  });
  module.def("create_arena", [](size_t bytes, const std::string& storage_dtype) {
    auto arena = mlx2::paged_kv::Arena::create(bytes, mlx2::paged_kv::storage_dtype_from_name(storage_dtype));
    return nb::capsule(new ArenaPtr(std::move(arena)), kCapsuleName, destroy_arena);
  }, "plane_bytes"_a, "storage_dtype"_a = "float16");
  module.def("storage_dtype", [](const nb::capsule& capsule) {
    return std::string(arena_from(capsule)->storage_dtype_name());
  }, "arena"_a);
  module.def("plane_bytes", [](const nb::capsule& capsule) {
    return arena_from(capsule)->plane_bytes();
  }, "arena"_a);
  module.def("plane_storage_shape", [](const nb::capsule& capsule) {
    return arena_from(capsule)->plane_storage_shape();
  }, "arena"_a);
  module.def("write", [](const nb::capsule& capsule, const mx::array& key,
                          const mx::array& value, size_t offset,
                          size_t byte_count, uint64_t epoch, mx::Stream stream,
                          bool inject_terminal_failure, bool permit_failure_probe) {
    if (byte_count == 0 || key.nbytes() != byte_count || value.nbytes() != byte_count) {
      throw std::invalid_argument("source bytes do not match declared write byte count");
    }
    if (inject_terminal_failure && !permit_failure_probe) {
      throw std::invalid_argument("terminal failure injection requires explicit probe enablement");
    }
    return arena_from(capsule)->write(
        key, value, offset, epoch, stream, inject_terminal_failure);
  }, "arena"_a, "key_bytes"_a, "value_bytes"_a, "offset"_a, "byte_count"_a,
     "epoch"_a, "stream"_a, "inject_terminal_failure"_a = false,
     "permit_failure_probe"_a = false);
  module.def("grouped_q1_write", [](const nb::capsule& capsule,
                                      const mx::array& keys,
                                      const mx::array& values,
                                      const std::vector<uint32_t>& pages,
                                      const std::vector<uint32_t>& slots,
                                      uint32_t kv_heads, uint32_t dim,
                                      uint64_t epoch, mx::Stream stream,
                                      bool permit_candidate) {
    if (!permit_candidate || pages.size() != 2 || slots.size() != 2)
      throw std::invalid_argument("grouped Q1 write requires two private lanes and permit");
    return arena_from(capsule)->grouped_q1_write(
        keys, values, {pages[0], pages[1]}, {slots[0], slots[1]},
        kv_heads, dim, epoch, stream);
  }, "arena"_a, "keys"_a, "values"_a, "pages"_a, "slots"_a,
     "kv_heads"_a, "dim"_a, "epoch"_a, "stream"_a,
     "permit_candidate"_a = false);
  module.def("grouped_q1_write_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->grouped_q1_write_count();
  }, "arena"_a);
  module.def("grouped_multirow_write", [](const nb::capsule& capsule,
      const mx::array& keys, const mx::array& values,
      const std::vector<uint32_t>& counts, const std::vector<uint32_t>& starts,
      const std::vector<uint32_t>& first_blocks, const std::vector<uint32_t>& table_begins,
      const std::vector<uint32_t>& page_ids, uint32_t kv_heads, uint32_t dim,
      uint64_t epoch, mx::Stream stream, bool permit_candidate) {
    if (!permit_candidate || counts.size() != 2 || starts.size() != 2 ||
        first_blocks.size() != 2 || table_begins.size() != 2)
      throw std::invalid_argument("grouped multirow write requires two lanes and permit");
    return arena_from(capsule)->grouped_multirow_write(
        keys, values, {counts[0], counts[1]}, {starts[0], starts[1]},
        {first_blocks[0], first_blocks[1]}, {table_begins[0], table_begins[1]},
        page_ids, kv_heads, dim, epoch, stream);
  }, "arena"_a, "keys"_a, "values"_a, "counts"_a, "starts"_a,
     "first_blocks"_a, "table_begins"_a, "page_ids"_a, "kv_heads"_a,
     "dim"_a, "epoch"_a, "stream"_a, "permit_candidate"_a = false);
  module.def("grouped_multirow_write_n20", [](const nb::capsule& capsule,
      const mx::array& keys, const mx::array& values,
      const std::vector<uint32_t>& counts, const std::vector<uint32_t>& starts,
      const std::vector<uint32_t>& first_blocks, const std::vector<uint32_t>& table_begins,
      const std::vector<uint32_t>& page_ids, uint32_t kv_heads, uint32_t dim,
      uint64_t epoch, mx::Stream stream, bool permit_candidate) {
    if (!permit_candidate) throw std::invalid_argument("grouped N20 write requires permit");
    return arena_from(capsule)->grouped_multirow_write_n20(keys,values,counts,starts,
        first_blocks,table_begins,page_ids,kv_heads,dim,epoch,stream);
  }, "arena"_a,"keys"_a,"values"_a,"counts"_a,"starts"_a,
     "first_blocks"_a,"table_begins"_a,"page_ids"_a,"kv_heads"_a,
     "dim"_a,"epoch"_a,"stream"_a,"permit_candidate"_a=false);
  module.def("grouped_n20_write_count",[](const nb::capsule& c){return arena_from(c)->grouped_n20_write_count();},"arena"_a);
  module.def("grouped_n20_row_count",[](const nb::capsule& c){return arena_from(c)->grouped_n20_row_count();},"arena"_a);
  module.def("grouped_multirow_write_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->grouped_multirow_write_count();
  }, "arena"_a);
  module.def("grouped_multirow_row_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->grouped_multirow_row_count();
  }, "arena"_a);
  module.def("write_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->write_dispatch_count();
  }, "arena"_a);
  module.def("copy_page", [](const nb::capsule& capsule, size_t source_offset,
                              size_t destination_offset, size_t byte_count,
                              uint64_t epoch, mx::Stream stream) {
    return arena_from(capsule)->copy_page(
        source_offset, destination_offset, byte_count, epoch, stream);
  }, "arena"_a, "source_offset"_a, "destination_offset"_a,
     "byte_count"_a, "epoch"_a, "stream"_a);
  module.def("validate_source_types", [](const mx::array& key, const mx::array& value,
                                         mx::Stream stream) {
    return key.dtype() == mx::uint8 && value.dtype() == mx::uint8 &&
        key.ndim() == 1 && value.ndim() == 1 &&
        key.flags().row_contiguous && value.flags().row_contiguous &&
        key.nbytes() != 0 && key.nbytes() == value.nbytes() &&
        stream.device.type == mx::Device::gpu;
  }, "key_bytes"_a, "value_bytes"_a, "stream"_a);
  module.def("diagnostic_read", [](const nb::capsule& capsule,
                                    const mx::array& dependency,
                                    size_t offset, size_t byte_count,
                                    mx::Stream stream, bool permit_diagnostic) {
    if (!permit_diagnostic) {
      throw std::invalid_argument("diagnostic paged KV read requires explicit enablement");
    }
    return arena_from(capsule)->diagnostic_read(dependency, offset, byte_count, stream);
  }, "arena"_a, "dependency"_a, "offset"_a, "byte_count"_a,
     "stream"_a, "permit_diagnostic"_a = false);
  module.def("gather_q1_fp16", [](const nb::capsule& capsule,
                                        const mx::array& query,
                                        const mx::array& dependency,
                                        const std::vector<std::vector<uint32_t>>& spans,
                                        const std::vector<uint32_t>& page_ids,
                                        uint32_t kv_heads, float scale,
                                        uint64_t epoch, mx::Stream stream,
                                        bool permit_candidate) {
    if (!permit_candidate) {
      throw std::invalid_argument("native paged attention requires explicit candidate enablement");
    }
    return arena_from(capsule)->gather_q1_fp16(
        query, dependency, spans, page_ids, kv_heads, scale, epoch, stream);
  }, "arena"_a, "query"_a, "dependency"_a, "spans"_a, "page_ids"_a,
     "kv_heads"_a, "scale"_a, "epoch"_a, "stream"_a,
     "permit_candidate"_a = false);
  module.def("attention_read_fp16", [](const nb::capsule& capsule,
                                        const mx::array& query,
                                        const mx::array& dependency,
                                        const std::vector<std::vector<uint32_t>>& spans,
                                        const std::vector<uint32_t>& page_ids,
                                        uint32_t kv_heads, float scale,
                                        uint64_t epoch, mx::Stream stream,
                                        bool permit_candidate) {
    if (!permit_candidate) {
      throw std::invalid_argument("native paged attention requires explicit candidate enablement");
    }
    return arena_from(capsule)->attention_read_fp16(
        query, dependency, spans, page_ids, kv_heads, scale, epoch, stream);
  }, "arena"_a, "query"_a, "dependency"_a, "spans"_a, "page_ids"_a,
     "kv_heads"_a, "scale"_a, "epoch"_a, "stream"_a,
     "permit_candidate"_a = false);
  // Generic names make the explicit arena storage contract visible while
  // preserving the historical FP16 entrypoint names and argument ABI.
  module.attr("attention_read") = module.attr("attention_read_fp16");
  module.attr("gather_q1") = module.attr("gather_q1_fp16");
  module.def("prefill_nax_capability",[]{nb::dict d;d["version"]=1;d["storage_dtype"]="bfloat16";d["head_dim"]=256;
    d["segmented_causal"]=true;d["threadgroup_bytes"]=256;d["threads_nax"]=256;d["threads_softmax"]=64;
    d["query_heads"]=24;d["kv_heads"]=4;d["min_query_count"]=9;d["max_query_count"]=129;d["max_causal_end"]=129;
    d["max_spans"]=2;d["origin_zero"]=true;d["window_zero"]=true;d["architecture"]="s";
    d["scratch_bound_bytes"]=3195072;d["scratch_bytes_per_query_row"]=24*129*2*2;
    d["physical_dispatches"]=3;d["qualified"]=false;d["selector"]="MLX2_PAGED_PREFILL_NAX_EXACT";return d;});
  module.def("prefill_nax_score_dispatch_count",[](const nb::capsule& c){return arena_from(c)->prefill_nax_score_dispatch_count();},"arena"_a);
  module.def("prefill_nax_softmax_dispatch_count",[](const nb::capsule& c){return arena_from(c)->prefill_nax_softmax_dispatch_count();},"arena"_a);
  module.def("prefill_nax_value_dispatch_count",[](const nb::capsule& c){return arena_from(c)->prefill_nax_value_dispatch_count();},"arena"_a);
  module.def("prefill_long_nax_capability",[]{nb::dict d;d["version"]=1;d["storage_dtype"]="bfloat16";
    d["head_dim"]=256;d["query_heads"]=24;d["kv_heads"]=4;d["max_spans"]=2;
    d["min_query_count"]=256;d["max_query_count"]=8192;d["origin_zero"]=true;
    d["max_causal_end"]=8192;
    d["window_zero"]=true;d["architecture"]="s";d["scratch_bytes"]=0;
    d["physical_dispatches"]=1;d["qualified"]=false;
    d["selector"]="MLX2_PAGED_PREFILL_NAX_LONG_FUSED";return d;});
  module.def("prefill_long_nax_dispatch_count",[](const nb::capsule& c){return arena_from(c)->prefill_long_nax_dispatch_count();},"arena"_a);
  module.def("packed_n20_capability",[]{nb::dict d;d["version"]=1;d["storage_dtype"]="bfloat16";
    d["head_dim"]=256;d["query_heads"]=24;d["kv_heads"]=4;
    d["max_spans"]=20;d["max_total_rows"]=20*8192;d["max_page_ids"]=20*128;
    d["min_prefill_count"]=256;d["max_prefill_count"]=8192;d["max_causal_end"]=8192;
    d["origin_zero"]=true;d["window_zero"]=true;d["architecture"]="s";
    d["prefill_score_scratch_bytes"]=0;d["max_q1_scratch_bytes"]=64*1024*1024;
    d["write_dispatches"]=1;d["prefill_read_dispatches"]=1;d["q1_read_dispatches"]=2;
    d["b1_survivor_scalar_dispatches"]=1;
    d["b1_stock_long_selector"]="MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON";
    d["b1_stock_long_dispatches"]=2;
    d["qualified"]=false;d["selector"]="MLX2_PAGED_PACKED_N20";return d;});
  module.def("prefill_long_n20_dispatch_count",[](const nb::capsule& c){return arena_from(c)->prefill_long_n20_dispatch_count();},"arena"_a);
  module.def("q1_stock_long_n20_partial_dispatch_count",[](const nb::capsule& c){return arena_from(c)->q1_stock_long_n20_partial_dispatch_count();},"arena"_a);
  module.def("q1_stock_long_n20_reduce_dispatch_count",[](const nb::capsule& c){return arena_from(c)->q1_stock_long_n20_reduce_dispatch_count();},"arena"_a);
  module.def("q1_stock_long_n20_singleton_partial_dispatch_count",[](const nb::capsule& c){return arena_from(c)->q1_stock_long_n20_singleton_partial_dispatch_count();},"arena"_a);
  module.def("q1_stock_long_n20_singleton_reduce_dispatch_count",[](const nb::capsule& c){return arena_from(c)->q1_stock_long_n20_singleton_reduce_dispatch_count();},"arena"_a);
  module.def("q1_scalar_dispatch_count",[](const nb::capsule& c){return arena_from(c)->q1_scalar_dispatch_count();},"arena"_a);
  module.def("diagnostic_n20_source_layout",[](const mx::array& source,bool permit_diagnostic){
    if(!permit_diagnostic)throw std::invalid_argument("N20 source layout diagnostic requires permit");
    nb::dict layout;
    layout["strides"]=std::vector<int64_t>(source.strides().begin(),source.strides().end());
    layout["byte_offset"]=source.offset();
    layout["backing_bytes"]=source.buffer_size();
    layout["row_contiguous"]=source.flags().row_contiguous;
    return layout;
  },"source"_a,"permit_diagnostic"_a=false);
  module.def("prefill_matrix_capability", [] {
    nb::dict result;
    result["version"] = 2;
    result["storage_dtype"] = "bfloat16";
    result["head_dim"] = 256;
    result["segmented_causal"] = true;
    result["requires_multiquery"] = true;
    result["supports_offsets"] = true;
    result["supports_window"] = true;
    result["max_spans"] = 8;
    result["max_causal_end"] = 8192;
    result["min_query_count"] = 9;
    result["max_query_count"] = 1023;
    result["arithmetic"] = "stock_short_bf16_scores_and_normalized_probabilities";
    result["passes"] = 2;
    result["query_tile"] = 16;
    result["key_tile"] = 16;
    result["threads"] = 64;
    result["threadgroup_bytes"] = 20736;
    result["global_scratch_bytes"] = 0;
    result["selector"] = "MLX2_PAGED_PREFILL_MATRIX";
    result["qualified"] = false;
    return result;
  });
  module.def("prefill_matrix_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->prefill_matrix_dispatch_count();
  }, "arena"_a);
  module.def("q1_gather_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_gather_dispatch_count();
  });
  module.def("q1_stripe_dispatch_count", [](const nb::capsule& capsule, uint32_t stripes) {
    return arena_from(capsule)->q1_stripe_dispatch_count(stripes);
  });
  module.def("q1_metadata_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_metadata_dispatch_count();
  }, "arena"_a);
  module.def("q1_split_partial_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_split_partial_dispatch_count();
  });
  module.def("q1_split_reduce_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_split_reduce_dispatch_count();
  });
  module.def("q1_stock_long_partial_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_stock_long_partial_dispatch_count();
  });
  module.def("q1_stock_long_reduce_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_stock_long_reduce_dispatch_count();
  });
  module.def("q1_stock_long_metadata_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_stock_long_metadata_dispatch_count();
  }, "arena"_a);
  module.def("q1_tile_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_tile_dispatch_count();
  }, "arena"_a);
  module.def("q1_stock_reduction_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_stock_reduction_dispatch_count();
  }, "arena"_a);
  module.def("q1_stock_singleton_dispatch_count", [](const nb::capsule& capsule) {
    return arena_from(capsule)->q1_stock_singleton_dispatch_count();
  }, "arena"_a);
  module.def("poll_completions", [](const nb::capsule& capsule) {
    std::vector<std::pair<uint64_t, bool>> result;
    for (const auto& event : arena_from(capsule)->poll_completions()) {
      result.emplace_back(event.epoch, event.succeeded);
    }
    return result;
  }, "arena"_a);
  module.def("wait_completions", [](const nb::capsule& capsule,
                                     double timeout_seconds) {
    auto owner = arena_from(capsule);
    std::vector<mlx2::paged_kv::Completion> events;
    {
      nb::gil_scoped_release release;
      events = owner->wait_completions(timeout_seconds);
    }
    std::vector<std::pair<uint64_t, bool>> result;
    for (const auto& event : events) {
      result.emplace_back(event.epoch, event.succeeded);
    }
    return result;
  }, "arena"_a, "timeout_seconds"_a);
  module.def("poll_read_completions", [](const nb::capsule& capsule) {
    std::vector<std::pair<uint64_t, bool>> result;
    for (const auto& event : arena_from(capsule)->poll_read_completions()) {
      result.emplace_back(event.epoch, event.succeeded);
    }
    return result;
  }, "arena"_a);
  module.def("wait_read_completions", [](const nb::capsule& capsule,
                                          double timeout_seconds) {
    // Retain the arena while the GIL is released. Metal's native completion
    // handler signals this queue without touching Python.
    auto owner = arena_from(capsule);
    std::vector<mlx2::paged_kv::Completion> events;
    {
      nb::gil_scoped_release release;
      events = owner->wait_read_completions(timeout_seconds);
    }
    std::vector<std::pair<uint64_t, bool>> result;
    for (const auto& event : events) {
      result.emplace_back(event.epoch, event.succeeded);
    }
    return result;
  }, "arena"_a, "timeout_seconds"_a);
}
