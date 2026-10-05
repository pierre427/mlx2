#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <array>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "storage_dtype.h"
#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx2::paged_kv {

struct Completion {
  uint64_t epoch;
  bool succeeded;
};

class Arena : public std::enable_shared_from_this<Arena> {
 public:
  static std::shared_ptr<Arena> create(size_t plane_bytes);
  static std::shared_ptr<Arena> create(size_t plane_bytes, StorageDtype storage_dtype);
  StorageDtype storage_kind() const { return storage_dtype_; }
  const char* storage_dtype_name() const { return mlx2::paged_kv::storage_dtype_name(storage_dtype_); }
  mlx::core::Dtype dtype() const {
    return storage_dtype_ == StorageDtype::Float16 ? mlx::core::float16 : mlx::core::bfloat16;
  }

  size_t plane_bytes() const { return plane_bytes_; }
  std::vector<int32_t> plane_storage_shape() const {
    return std::vector<int32_t>(keys_.shape().begin(), keys_.shape().end());
  }
  std::vector<Completion> poll_completions();
  std::vector<Completion> wait_completions(double timeout_seconds);
  std::vector<Completion> poll_read_completions();
  std::vector<Completion> wait_read_completions(double timeout_seconds);

  // Internal only. Caller owns generation validation and a host completion
  // lease before evaluating this graph node on the same explicit GPU stream.
  mlx::core::array write(
      const mlx::core::array& key_bytes,
      const mlx::core::array& value_bytes,
      size_t destination_offset,
      uint64_t epoch,
      mlx::core::Stream stream,
      bool inject_terminal_failure = false);

  // Copy an accepted partial page inside both arena planes. The host pins
  // source and destination generations until this command buffer terminates.
  mlx::core::array copy_page(
      size_t source_offset, size_t destination_offset, size_t byte_count,
      uint64_t epoch, mlx::core::Stream stream);

  // Diagnostic copy only. The caller retains the page generation through
  // synchronized evaluation; no writable arena view escapes.
  std::vector<mlx::core::array> diagnostic_read(
      const mlx::core::array& dependency,
      size_t source_offset,
      size_t byte_count,
      mlx::core::Stream stream);

  // Default-off dense fp16 attention over this arena's immutable page table.
  // Each span has [rows, query_start, kv_end, retained_start, first_block,
  // table_begin, table_count, window] in host-validated coordinates. A read
  // completion is separate from write/copy completions and is not a serving
  // route. The host pins every referenced generation through its callback.
  mlx::core::array attention_read_fp16(
      const mlx::core::array& query,
      const mlx::core::array& dependency,
      const std::vector<std::vector<uint32_t>>& spans,
      const std::vector<uint32_t>& page_ids,
      uint32_t kv_heads,
      float scale,
      uint64_t epoch,
      mlx::core::Stream stream);

  std::vector<mlx::core::array> gather_q1_fp16(
      const mlx::core::array& query, const mlx::core::array& dependency,
      const std::vector<std::vector<uint32_t>>& spans,
      const std::vector<uint32_t>& page_ids, uint32_t kv_heads,
      float scale, uint64_t epoch, mlx::core::Stream stream);
  void record_q1_gather_dispatch() { q1_gather_dispatches_.fetch_add(1); }
  uint64_t q1_gather_dispatch_count() const { return q1_gather_dispatches_.load(); }

  mlx::core::array grouped_q1_write(
      const mlx::core::array& keys, const mlx::core::array& values,
      std::array<uint32_t, 2> pages, std::array<uint32_t, 2> slots,
      uint32_t kv_heads, uint32_t dim, uint64_t epoch,
      mlx::core::Stream stream);
  mlx::core::array grouped_multirow_write(
      const mlx::core::array& keys, const mlx::core::array& values,
      std::array<uint32_t, 2> counts, std::array<uint32_t, 2> starts,
      std::array<uint32_t, 2> first_blocks, std::array<uint32_t, 2> table_begins,
      const std::vector<uint32_t>& page_ids, uint32_t kv_heads, uint32_t dim,
      uint64_t epoch, mlx::core::Stream stream);
  mlx::core::array grouped_multirow_write_n20(
      const mlx::core::array& keys, const mlx::core::array& values,
      const std::vector<uint32_t>& counts, const std::vector<uint32_t>& starts,
      const std::vector<uint32_t>& first_blocks,
      const std::vector<uint32_t>& table_begins,
      const std::vector<uint32_t>& page_ids, uint32_t kv_heads, uint32_t dim,
      uint64_t epoch, mlx::core::Stream stream);
  void record_grouped_multirow_write(uint32_t rows) {
    grouped_multirow_writes_.fetch_add(1);
    grouped_multirow_rows_.fetch_add(rows);
  }
  uint64_t grouped_multirow_write_count() const { return grouped_multirow_writes_.load(); }
  uint64_t grouped_multirow_row_count() const { return grouped_multirow_rows_.load(); }
  void record_grouped_n20_write(uint32_t rows) {
    grouped_n20_writes_.fetch_add(1);grouped_n20_rows_.fetch_add(rows);
  }
  uint64_t grouped_n20_write_count() const { return grouped_n20_writes_.load(); }
  uint64_t grouped_n20_row_count() const { return grouped_n20_rows_.load(); }
  void record_grouped_q1_write() { grouped_q1_writes_.fetch_add(1); }
  uint64_t grouped_q1_write_count() const { return grouped_q1_writes_.load(); }
  void record_write_dispatch() { write_dispatches_.fetch_add(1); }
  uint64_t write_dispatch_count() const { return write_dispatches_.load(); }

  // Counts selected tile dispatches after encoding, for research receipts.
  void record_prefill_nax_score_dispatch(){prefill_nax_score_.fetch_add(1);}
  uint64_t prefill_nax_score_dispatch_count() const{return prefill_nax_score_.load();}
  void record_prefill_nax_softmax_dispatch(){prefill_nax_softmax_.fetch_add(1);}
  uint64_t prefill_nax_softmax_dispatch_count() const{return prefill_nax_softmax_.load();}
  void record_prefill_nax_value_dispatch(){prefill_nax_value_.fetch_add(1);}
  uint64_t prefill_nax_value_dispatch_count() const{return prefill_nax_value_.load();}
  void record_prefill_long_nax_dispatch(){prefill_long_nax_.fetch_add(1);}
  uint64_t prefill_long_nax_dispatch_count() const{return prefill_long_nax_.load();}
  void record_prefill_long_n20_dispatch(){prefill_long_n20_.fetch_add(1);}
  uint64_t prefill_long_n20_dispatch_count() const{return prefill_long_n20_.load();}
  void record_prefill_matrix_dispatch() { prefill_matrix_dispatches_.fetch_add(1); }
  uint64_t prefill_matrix_dispatch_count() const { return prefill_matrix_dispatches_.load(); }
  void record_q1_tile_dispatch() { q1_tile_dispatches_.fetch_add(1); }
  void record_q1_scalar_dispatch() { q1_scalar_dispatches_.fetch_add(1); }
  uint64_t q1_scalar_dispatch_count() const { return q1_scalar_dispatches_.load(); }
  uint64_t q1_tile_dispatch_count() const { return q1_tile_dispatches_.load(); }
  void record_q1_stock_reduction_dispatch() { q1_stock_reduction_dispatches_.fetch_add(1); }
  uint64_t q1_stock_reduction_dispatch_count() const {
    return q1_stock_reduction_dispatches_.load();
  }

  void record_q1_stock_singleton_dispatch() { q1_stock_singleton_dispatches_.fetch_add(1); }
  uint64_t q1_stock_singleton_dispatch_count() const { return q1_stock_singleton_dispatches_.load(); }

  void record_q1_stripe_dispatch(uint32_t stripes) {
    if (stripes == 8) q1_stripes_8_.fetch_add(1);
    else if (stripes == 16) q1_stripes_16_.fetch_add(1);
    else if (stripes == 32) q1_stripes_32_.fetch_add(1);
  }
  uint64_t q1_stripe_dispatch_count(uint32_t stripes) const {
    if (stripes == 8) return q1_stripes_8_.load();
    if (stripes == 16) return q1_stripes_16_.load();
    if (stripes == 32) return q1_stripes_32_.load();
    throw std::invalid_argument("stripe dispatch counter requires 8, 16 or 32");
  }

  void record_q1_split_partial_dispatch() { q1_split_partial_dispatches_.fetch_add(1); }
  void record_q1_split_reduce_dispatch() { q1_split_reduce_dispatches_.fetch_add(1); }
  void record_q1_stock_long_partial_dispatch() { q1_stock_long_partial_dispatches_.fetch_add(1); }
  void record_q1_stock_long_reduce_dispatch() { q1_stock_long_reduce_dispatches_.fetch_add(1); }
  void record_q1_stock_long_metadata_dispatch() { q1_stock_long_metadata_dispatches_.fetch_add(1); }
  uint64_t q1_stock_long_metadata_dispatch_count() const { return q1_stock_long_metadata_dispatches_.load(); }
  uint64_t q1_stock_long_partial_dispatch_count() const { return q1_stock_long_partial_dispatches_.load(); }
  uint64_t q1_stock_long_reduce_dispatch_count() const { return q1_stock_long_reduce_dispatches_.load(); }
  void record_q1_stock_long_n20_partial_dispatch(){q1_stock_long_n20_partial_.fetch_add(1);}
  void record_q1_stock_long_n20_reduce_dispatch(){q1_stock_long_n20_reduce_.fetch_add(1);}
  void record_q1_stock_long_n20_singleton_partial_dispatch(){q1_stock_long_n20_singleton_partial_.fetch_add(1);}
  void record_q1_stock_long_n20_singleton_reduce_dispatch(){q1_stock_long_n20_singleton_reduce_.fetch_add(1);}
  uint64_t q1_stock_long_n20_partial_dispatch_count() const{return q1_stock_long_n20_partial_.load();}
  uint64_t q1_stock_long_n20_reduce_dispatch_count() const{return q1_stock_long_n20_reduce_.load();}
  uint64_t q1_stock_long_n20_singleton_partial_dispatch_count() const{return q1_stock_long_n20_singleton_partial_.load();}
  uint64_t q1_stock_long_n20_singleton_reduce_dispatch_count() const{return q1_stock_long_n20_singleton_reduce_.load();}
  uint64_t q1_split_partial_dispatch_count() const { return q1_split_partial_dispatches_.load(); }
  uint64_t q1_split_reduce_dispatch_count() const { return q1_split_reduce_dispatches_.load(); }

  // Counts actual encoded inline metadata dispatches, not graph construction.
  void record_q1_metadata_dispatch() { q1_metadata_dispatches_.fetch_add(1); }
  uint64_t q1_metadata_dispatch_count() const { return q1_metadata_dispatches_.load(); }

 private:
  Arena(size_t plane_bytes, mlx::core::array keys, mlx::core::array values, StorageDtype storage_dtype);
  void completed(uint64_t epoch, bool succeeded);
  void read_completed(uint64_t epoch, bool succeeded);

  size_t plane_bytes_;
  const StorageDtype storage_dtype_;
  std::atomic<uint64_t> q1_tile_dispatches_{0};
  std::atomic<uint64_t> q1_scalar_dispatches_{0};
  std::atomic<uint64_t> q1_stock_reduction_dispatches_{0};
  std::atomic<uint64_t> q1_stock_singleton_dispatches_{0};
  std::atomic<uint64_t> q1_split_partial_dispatches_{0}, q1_split_reduce_dispatches_{0};
  std::atomic<uint64_t> prefill_nax_score_{0};
  std::atomic<uint64_t> prefill_nax_softmax_{0};
  std::atomic<uint64_t> prefill_nax_value_{0};
  std::atomic<uint64_t> prefill_long_nax_{0};
  std::atomic<uint64_t> prefill_long_n20_{0};
  std::atomic<uint64_t> prefill_matrix_dispatches_{0};
  std::atomic<uint64_t> q1_stock_long_partial_dispatches_{0}, q1_stock_long_reduce_dispatches_{0};
  std::atomic<uint64_t> q1_stock_long_n20_partial_{0}, q1_stock_long_n20_reduce_{0};
  std::atomic<uint64_t> q1_stock_long_n20_singleton_partial_{0}, q1_stock_long_n20_singleton_reduce_{0};
  std::atomic<uint64_t> q1_stock_long_metadata_dispatches_{0};
  std::atomic<uint64_t> q1_gather_dispatches_{0};
  std::atomic<uint64_t> q1_metadata_dispatches_{0};
  std::atomic<uint64_t> q1_stripes_8_{0}, q1_stripes_16_{0}, q1_stripes_32_{0};
  std::atomic<uint64_t> grouped_q1_writes_{0};
  std::atomic<uint64_t> grouped_multirow_writes_{0}, grouped_multirow_rows_{0};
  std::atomic<uint64_t> grouped_n20_writes_{0}, grouped_n20_rows_{0};
  std::atomic<uint64_t> write_dispatches_{0};
  mlx::core::array keys_;
  mlx::core::array values_;
  std::mutex completions_mutex_;
  std::condition_variable read_completions_ready_;
  std::condition_variable completions_ready_;
  std::deque<Completion> completions_;
  std::deque<Completion> read_completions_;

  friend class WritePrimitive;
  friend class CopyPrimitive;
  friend class DiagnosticReadPrimitive;
  friend class AttentionReadPrimitive;
  friend class Q1GatherPrimitive;
  friend class GroupedQ1WritePrimitive;
  friend class GroupedMultirowWritePrimitive;
  friend class GroupedN20WritePrimitive;
};

} // namespace mlx2::paged_kv
