#include "arena.h"
#include "q1_metadata.h"
#include "q1_geometry.h"
#include "q1_stock_reduction.h"
#include "q1_splitkv.h"
#include "q1_stock_long.h"
#include "q1_stock_long_metadata.h"
#include "packed_multirow_write.h"
#include "packed_n20.h"
#include "packed_n20_layout.h"
#include "prefill_matrix.h"
#include "prefill_nax.h"
#include "prefill_long_nax.h"
#include "arena_storage.h"

#include <algorithm>
#include <array>
#include <limits>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/device.h"
#include "mlx/primitives.h"

namespace mlx2::paged_kv {
namespace mx = mlx::core;

namespace {
constexpr const char* kWriteSource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void mlx2_paged_kv_write(
    device const uchar* source_k [[buffer(0)]],
    device const uchar* source_v [[buffer(1)]],
    device uchar* destination_k [[buffer(2)]],
    device uchar* destination_v [[buffer(3)]],
    device uchar* dependency [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    destination_k[index] = source_k[index];
    destination_v[index] = source_v[index];
    if (index == 0) dependency[0] = 1;
  }
}
)metal";
constexpr const char* kGroupedQ1WriteSource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void mlx2_paged_kv_grouped_q1_write(
    device const half* source_k [[buffer(0)]],
    device const half* source_v [[buffer(1)]],
    device half* destination_k [[buffer(2)]],
    device half* destination_v [[buffer(3)]],
    device uchar* dependency [[buffer(4)]],
    constant uint2& pages [[buffer(5)]],
    constant uint2& slots [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& dim [[buffer(8)]],
    constant ulong* key_strides [[buffer(9)]],
    constant ulong* value_strides [[buffer(10)]],
    uint3 index [[thread_position_in_grid]]) {
  const uint channel = index.x;
  const uint head = index.y;
  const uint row = index.z;
  if (channel >= dim || head >= kv_heads || row >= 2) return;
  const size_t target = (((size_t)pages[row] * kv_heads + head) * 64 + slots[row]) * dim + channel;
  const size_t key_source = (size_t)row * key_strides[0] +
      (size_t)head * key_strides[1] + (size_t)channel * key_strides[2];
  const size_t value_source = (size_t)row * value_strides[0] +
      (size_t)head * value_strides[1] + (size_t)channel * value_strides[2];
  destination_k[target] = source_k[key_source];
  destination_v[target] = source_v[value_source];
  if (row == 0 && head == 0 && channel == 0) dependency[0] = 1;
}
)metal";
constexpr const char* kGroupedMultirowWriteSource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void mlx2_paged_kv_grouped_multirow_write(
    device const half* source_k [[buffer(0)]],
    device const half* source_v [[buffer(1)]],
    device half* destination_k [[buffer(2)]],
    device half* destination_v [[buffer(3)]],
    device uchar* dependency [[buffer(4)]],
    constant uint2& counts [[buffer(5)]],
    constant uint2& starts [[buffer(6)]],
    constant uint2& first_blocks [[buffer(7)]],
    constant uint2& table_begins [[buffer(8)]],
    constant uint* page_ids [[buffer(9)]],
    constant uint& kv_heads [[buffer(10)]],
    constant uint& dim [[buffer(11)]],
    uint3 index [[thread_position_in_grid]]) {
  const uint channel = index.x;
  const uint head = index.y;
  const uint row = index.z;
  if (channel >= dim || head >= kv_heads || row >= counts.x + counts.y) return;
  const uint lane = row < counts.x ? 0 : 1;
  const uint local = lane == 0 ? row : row - counts.x;
  const uint logical = starts[lane] + local;
  const uint block = logical / 64 - first_blocks[lane];
  const uint page = page_ids[table_begins[lane] + block];
  const size_t destination = (((size_t)page * kv_heads + head) * 64 + logical % 64) * dim + channel;
  const size_t source = ((size_t)row * kv_heads + head) * dim + channel;
  destination_k[destination] = source_k[source];
  destination_v[destination] = source_v[source];
  if (row == 0 && head == 0 && channel == 0) dependency[0] = 1;
}
)metal";
constexpr const char* kGroupedN20WriteSource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void mlx2_paged_kv_grouped_n20_write(
    device const half* source_k [[buffer(0)]],
    device const half* source_v [[buffer(1)]],
    device half* destination_k [[buffer(2)]],
    device half* destination_v [[buffer(3)]],
    device uchar* dependency [[buffer(4)]],
    constant uint* counts [[buffer(5)]],
    constant uint* starts [[buffer(6)]],
    constant uint* first_blocks [[buffer(7)]],
    constant uint* table_begins [[buffer(8)]],
    device const uint* page_ids [[buffer(9)]],
    constant uint* row_begins [[buffer(10)]],
    constant uint& kv_heads [[buffer(11)]],
    constant uint& dim [[buffer(12)]],
    uint3 index [[thread_position_in_grid]]) {
  const uint lane=index.z, local=index.y;
  const uint head=index.x/dim, channel=index.x%dim;
  if(local>=counts[lane] || head>=kv_heads)return;
  const uint row=row_begins[lane]+local;
  const uint logical=starts[lane]+local;
  const uint block=logical/64-first_blocks[lane];
  const uint page=page_ids[table_begins[lane]+block];
  const size_t destination=(((size_t)page*kv_heads+head)*64+logical%64)*dim+channel;
  const size_t source=((size_t)row*kv_heads+head)*dim+channel;
  destination_k[destination]=source_k[source];
  destination_v[destination]=source_v[source];
  if(row==0 && head==0 && channel==0)dependency[0]=1;
}
)metal";
constexpr const char* kReadSource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void mlx2_paged_kv_diagnostic_read(
    device const uchar* source_k [[buffer(0)]],
    device const uchar* source_v [[buffer(1)]],
    device const uchar* dependency [[buffer(2)]],
    device uchar* output_k [[buffer(3)]],
    device uchar* output_v [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    const bool ready = dependency[0] == 1;
    output_k[index] = ready ? source_k[index] : 0;
    output_v[index] = ready ? source_v[index] : 0;
  }
}
)metal";
constexpr const char* kCopySource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void mlx2_paged_kv_copy(
    device const uchar* source_k [[buffer(0)]],
    device const uchar* source_v [[buffer(1)]],
    device uchar* destination_k [[buffer(2)]],
    device uchar* destination_v [[buffer(3)]],
    device uchar* dependency [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    destination_k[index] = source_k[index];
    destination_v[index] = source_v[index];
    if (index == 0) dependency[0] = 1;
  }
}
)metal";
constexpr const char* kAttentionReadSource = R"metal(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
kernel void mlx2_paged_attention_read_fp16(
    device const half* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device const uchar* dependency [[buffer(3)]],
    device const uint* row_span [[buffer(4)]],
    device const uint* row_begin [[buffer(5)]],
    device const uint* query_start [[buffer(6)]],
    device const uint* retained_start [[buffer(7)]],
    device const uint* first_block [[buffer(8)]],
    device const uint* table_begin [[buffer(9)]],
    device const uint* window [[buffer(10)]],
    device const uint* page_ids [[buffer(11)]],
    device half* output [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    constant uint& query_heads [[buffer(14)]],
    constant uint& kv_heads [[buffer(15)]],
    constant uint& dim [[buffer(16)]],
    constant ulong& query_row_stride [[buffer(17)]],
    constant ulong& query_head_stride [[buffer(18)]],
    constant ulong& query_dim_stride [[buffer(19)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.y;
  const uint qh = group.z;
  if (dependency[0] != 1) {
    for (uint part = 0; part < dim / 32; ++part)
      output[((size_t)row * query_heads + qh) * dim + lane * (dim / 32) + part] = half(0);
    return;
  }
  const uint span = row_span[row];
  const uint upper = query_start[span] + row - row_begin[span] + 1;
  const uint lower = window[span] == 0 ? retained_start[span] :
      metal::max(retained_start[span], upper - metal::min(upper, window[span]));
  const uint kvh = qh / (query_heads / kv_heads);
  float q[8];
  float numerator[8] = {0};
  for (uint part = 0; part < dim / 32; ++part)
    q[part] = float(query[(size_t)row * query_row_stride +
                          (size_t)qh * query_head_stride +
                          (size_t)(lane * (dim / 32) + part) * query_dim_stride]);
  float maximum = -INFINITY;
  float denominator = 0.0f;
  for (uint chunk = lower / 512; chunk <= (upper - 1) / 512; ++chunk) {
    const uint begin = metal::max(lower, chunk * 512);
    const uint end = metal::min(upper, (chunk + 1) * 512);
    for (uint token = begin; token < end; ++token) {
      const uint page = page_ids[table_begin[span] + token / 64 - first_block[span]];
      const size_t base = (((size_t)page * kv_heads + kvh) * 64 + (token & 63)) * dim;
      float score = 0.0f;
      for (uint part = 0; part < dim / 32; ++part)
        score += q[part] * float(keys[base + lane * (dim / 32) + part]);
      score = simd_sum(score) * scale;
      const float next_max = metal::max(maximum, score);
      const float previous = maximum == -INFINITY ? 0.0f : metal::exp(maximum - next_max);
      const float current = metal::exp(score - next_max);
      maximum = next_max;
      denominator = denominator * previous + current;
      for (uint part = 0; part < dim / 32; ++part) {
        const uint d = lane * (dim / 32) + part;
        numerator[part] = numerator[part] * previous + current * float(values[base + d]);
      }
    }
  }
  for (uint part = 0; part < dim / 32; ++part) {
    const uint d = lane * (dim / 32) + part;
    output[((size_t)row * query_heads + qh) * dim + d] = half(numerator[part] / denominator);
  }
}
)metal";
// Research-only short Q1 path. Four SIMD groups independently visit token
// stripes, then combine online-softmax partials within one threadgroup.
// A separate kernel dispatch would dominate this ~64-token cell.
constexpr const char* kAttentionReadQ1TileSource = R"metal(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
kernel void mlx2_paged_attention_read_q1_simd_tile_fp16(
    device const half* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device const uchar* dependency [[buffer(3)]],
    device const uint* row_span [[buffer(4)]],
    device const uint* row_begin [[buffer(5)]],
    device const uint* query_start [[buffer(6)]],
    device const uint* retained_start [[buffer(7)]],
    device const uint* first_block [[buffer(8)]],
    device const uint* table_begin [[buffer(9)]],
    device const uint* window [[buffer(10)]],
    device const uint* page_ids [[buffer(11)]],
    device half* output [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    constant uint& query_heads [[buffer(14)]],
    constant uint& kv_heads [[buffer(15)]],
    constant uint& dim [[buffer(16)]],
    constant ulong& query_row_stride [[buffer(17)]],
    constant ulong& query_head_stride [[buffer(18)]],
    constant ulong& query_dim_stride [[buffer(19)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint stripe [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.y;
  const uint qh = group.z;
  if (dependency[0] != 1) {
    if (stripe == 0)
      for (uint part = 0; part < dim / 32; ++part)
        output[((size_t)row * query_heads + qh) * dim + lane * (dim / 32) + part] = half(0);
    return;
  }
  const uint span = row_span[row];
  const uint upper = query_start[span] + row - row_begin[span] + 1;
  const uint lower = window[span] == 0 ? retained_start[span] :
      metal::max(retained_start[span], upper - metal::min(upper, window[span]));
  const uint kvh = qh / (query_heads / kv_heads);
  float q[8];
  float numerator[8] = {0};
  for (uint part = 0; part < dim / 32; ++part)
    q[part] = float(query[(size_t)row * query_row_stride +
                          (size_t)qh * query_head_stride +
                          (size_t)(lane * (dim / 32) + part) * query_dim_stride]);
  float maximum = -INFINITY;
  float denominator = 0.0f;
  for (uint token = lower + stripe; token < upper; token += 4) {
    const uint page = page_ids[table_begin[span] + token / 64 - first_block[span]];
    const size_t base = (((size_t)page * kv_heads + kvh) * 64 + (token & 63)) * dim;
    float score = 0.0f;
    for (uint part = 0; part < dim / 32; ++part)
      score += q[part] * float(keys[base + lane * (dim / 32) + part]);
    score = simd_sum(score) * scale;
    const float next_max = metal::max(maximum, score);
    const float previous = maximum == -INFINITY ? 0.0f : metal::exp(maximum - next_max);
    const float current = metal::exp(score - next_max);
    maximum = next_max;
    denominator = denominator * previous + current;
    for (uint part = 0; part < dim / 32; ++part) {
      const uint d = lane * (dim / 32) + part;
      numerator[part] = numerator[part] * previous + current * float(values[base + d]);
    }
  }
  threadgroup float partial_maximum[4];
  threadgroup float partial_denominator[4];
  threadgroup float partial_numerator[4][256];
  if (lane == 0) {
    partial_maximum[stripe] = maximum;
    partial_denominator[stripe] = denominator;
  }
  for (uint part = 0; part < dim / 32; ++part)
    partial_numerator[stripe][lane * (dim / 32) + part] = numerator[part];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (stripe == 0) {
    float global_maximum = -INFINITY;
    for (uint tile = 0; tile < 4; ++tile)
      global_maximum = metal::max(global_maximum, partial_maximum[tile]);
    float total_denominator = 0.0f;
    for (uint tile = 0; tile < 4; ++tile)
      if (partial_denominator[tile] > 0.0f)
        total_denominator += partial_denominator[tile] *
            metal::exp(partial_maximum[tile] - global_maximum);
    for (uint part = 0; part < dim / 32; ++part) {
      const uint d = lane * (dim / 32) + part;
      float total_numerator = 0.0f;
      for (uint tile = 0; tile < 4; ++tile)
        if (partial_denominator[tile] > 0.0f)
          total_numerator += partial_numerator[tile][d] *
              metal::exp(partial_maximum[tile] - global_maximum);
      output[((size_t)row * query_heads + qh) * dim + d] =
          half(total_numerator / total_denominator);
    }
  }
}
)metal";
constexpr const char* kQ1GatherSource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void mlx2_paged_q1_gather_fp16(
    device const half* keys [[buffer(0)]],
    device const half* values [[buffer(1)]],
    device const uchar* dependency [[buffer(2)]],
    constant uint* lower [[buffer(3)]],
    constant uint* visible [[buffer(4)]],
    constant uint* first_block [[buffer(5)]],
    constant uint* table_begin [[buffer(6)]],
    constant uint* page_ids [[buffer(7)]],
    device half* dense_keys [[buffer(8)]],
    device half* dense_values [[buffer(9)]],
    constant uint& kv_heads [[buffer(10)]],
    constant uint& dim [[buffer(11)]],
    constant uint& count [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  const uint channel = index % dim;
  const uint token = (index / dim) % 128;
  const uint head = (index / (dim * 128)) % kv_heads;
  const uint row = index / (dim * 128 * kv_heads);
  half key = half(0), value = half(0);
  if (dependency[0] == 1 && token < visible[row]) {
    const uint position = lower[row] + token;
    const uint page = page_ids[table_begin[row] + position / 64 - first_block[row]];
    const size_t source = (((size_t)page * kv_heads + head) * 64 + (position & 63)) * dim + channel;
    key = keys[source];
    value = values[source];
  }
  dense_keys[index] = key;
  dense_values[index] = value;
}
)metal";
} // namespace

class WritePrimitive final : public mx::Primitive {
 public:
  WritePrimitive(std::shared_ptr<Arena> arena, size_t offset, uint64_t epoch,
                 mx::Stream stream, bool inject_terminal_failure)
      : Primitive(stream), arena_(std::move(arena)), offset_(offset),
        epoch_(epoch), inject_terminal_failure_(inject_terminal_failure) {}

  const char* name() const override { return "Mlx2PagedKVWrite"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("paged KV arena writes require a GPU stream");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    // Each descriptor shares the arena's Data owner. Never create a second
    // owning array from its raw allocator buffer.
    auto destination_k = arena_->keys_;
    auto destination_v = arena_->values_;
    outputs[0].set_data(mx::allocator::malloc(1));

    auto& device = mx::metal::device(stream().device);
    auto* library = device.get_library("mlx2_paged_kv_write_v1", [] { return std::string(kWriteSource); });
    auto* pipeline = device.get_kernel("mlx2_paged_kv_write", library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(destination_k, 2, static_cast<int64_t>(offset_));
    encoder.set_output_array(destination_v, 3, static_cast<int64_t>(offset_));
    encoder.set_output_array(outputs[0], 4);
    const auto count = static_cast<uint32_t>(inputs[0].nbytes());
    encoder.set_bytes(count, 5);
    // Register the terminal callback before dispatch. If encoding throws
    // later, the owner and lease remain pinned until the buffer terminates.
    auto owner = arena_;
    const auto epoch = epoch_;
    const auto inject_terminal_failure = inject_terminal_failure_;
    auto dispatched = std::make_shared<std::atomic<bool>>(false);
    encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer) {
      owner->completed(epoch, !inject_terminal_failure &&
                                dispatched->load(std::memory_order_acquire) &&
                                buffer->status() == MTL::CommandBufferStatusCompleted);
    });
    encoder.dispatch_threads(MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
    dispatched->store(true, std::memory_order_release);
    arena_->record_write_dispatch();
    // MLX's evaluator owns commit; committing here would rotate its buffer.
  }

 private:
  std::shared_ptr<Arena> arena_;
  size_t offset_;
  uint64_t epoch_;
  bool inject_terminal_failure_;
};

class GroupedQ1WritePrimitive final : public mx::Primitive {
 public:
  GroupedQ1WritePrimitive(std::shared_ptr<Arena> arena,
                          std::array<uint32_t, 2> pages,
                          std::array<uint32_t, 2> slots,
                          uint32_t kv_heads, uint32_t dim,
                          uint64_t epoch, mx::Stream stream)
      : Primitive(stream), arena_(std::move(arena)), pages_(pages), slots_(slots),
        kv_heads_(kv_heads), dim_(dim), epoch_(epoch) {}

  const char* name() const override { return "Mlx2PagedKVGroupedQ1Write"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("grouped paged KV writes require a GPU stream");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto checked_strides = [](const mx::array& source) {
      const auto& strides = source.strides();
      if (strides.size() != 3 || source.offset() < 0 ||
          strides[0] <= 0 || strides[1] <= 0 || strides[2] <= 0)
        throw std::invalid_argument("evaluated grouped Q1 source strides are invalid");
      std::array<uint64_t, 3> result{};
      uint64_t last = 0;
      for (size_t axis = 0; axis < 3; ++axis) {
        result[axis] = static_cast<uint64_t>(strides[axis]);
        const auto extent = static_cast<uint64_t>(source.shape(axis) - 1);
        if (extent != 0 && result[axis] > (UINT64_MAX - last) / extent)
          throw std::invalid_argument("grouped Q1 source stride overflows");
        last += extent * result[axis];
      }
      const auto offset = static_cast<uint64_t>(source.offset());
      const auto bytes = static_cast<uint64_t>(source.buffer_size());
      if (offset > bytes || last >= (bytes - offset) / sizeof(mx::float16_t))
        throw std::invalid_argument("grouped Q1 source exceeds storage");
      return result;
    };
    const auto key_strides = checked_strides(inputs[0]);
    const auto value_strides = checked_strides(inputs[1]);
    outputs[0].set_data(mx::allocator::malloc(1));
    auto& device = mx::metal::device(stream().device);
    const auto storage = arena_->storage_kind();
    auto* library = device.get_library("mlx2_paged_kv_grouped_q1_write_v1" + storage_library_suffix(storage),
        [storage] { return storage_specialized_source(kGroupedQ1WriteSource, storage, true, 4); });
    auto* pipeline = device.get_kernel("mlx2_paged_kv_grouped_q1_write", library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(arena_->keys_, 2);
    encoder.set_output_array(arena_->values_, 3);
    encoder.set_output_array(outputs[0], 4);
    encoder.set_bytes(pages_, 5);
    encoder.set_bytes(slots_, 6);
    encoder.set_bytes(kv_heads_, 7);
    encoder.set_bytes(dim_, 8);
    encoder.set_bytes(key_strides, 9);
    encoder.set_bytes(value_strides, 10);
    auto owner = arena_;
    const auto epoch = epoch_;
    auto dispatched = std::make_shared<std::atomic<bool>>(false);
    encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer) {
      owner->completed(epoch, dispatched->load(std::memory_order_acquire) &&
                                buffer->status() == MTL::CommandBufferStatusCompleted);
    });
    encoder.dispatch_threads(MTL::Size(dim_, kv_heads_, 2), MTL::Size(128, 1, 1));
    dispatched->store(true, std::memory_order_release);
    arena_->record_grouped_q1_write();
  }

 private:
  std::shared_ptr<Arena> arena_;
  std::array<uint32_t, 2> pages_;
  std::array<uint32_t, 2> slots_;
  uint32_t kv_heads_;
  uint32_t dim_;
  uint64_t epoch_;
};

class GroupedMultirowWritePrimitive final : public mx::Primitive {
 public:
  GroupedMultirowWritePrimitive(std::shared_ptr<Arena> arena,
      std::array<uint32_t, 2> counts, std::array<uint32_t, 2> starts,
      std::array<uint32_t, 2> first_blocks, std::array<uint32_t, 2> table_begins,
      std::vector<uint32_t> page_ids, uint32_t kv_heads, uint32_t dim,
      uint64_t epoch, mx::Stream stream)
      : Primitive(stream), arena_(std::move(arena)), counts_(counts), starts_(starts),
        first_blocks_(first_blocks), table_begins_(table_begins),
        page_ids_(std::move(page_ids)), kv_heads_(kv_heads), dim_(dim), epoch_(epoch) {}

  const char* name() const override { return "Mlx2PagedKVGroupedMultirowWrite"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("grouped multirow KV writes require a GPU stream");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto checked_contiguous = [this](const mx::array& source) {
      const auto& strides = source.strides();
      const uint64_t row_values = static_cast<uint64_t>(kv_heads_) * dim_;
      if (strides.size() != 3 || source.offset() < 0 ||
          static_cast<uint64_t>(source.offset()) % sizeof(mx::float16_t) != 0 ||
          strides[0] != static_cast<int64_t>(row_values) ||
          strides[1] != static_cast<int64_t>(dim_) || strides[2] != 1)
        throw std::invalid_argument("grouped multirow source must be contiguous token-major");
      const uint64_t offset = static_cast<uint64_t>(source.offset());
      const uint64_t bytes = static_cast<uint64_t>(source.buffer_size());
      const uint64_t values = static_cast<uint64_t>(counts_[0] + counts_[1]) * row_values;
      if (offset > bytes || values > (bytes - offset) / sizeof(mx::float16_t))
        throw std::invalid_argument("grouped multirow source exceeds backing bytes");
    };
    checked_contiguous(inputs[0]); checked_contiguous(inputs[1]);
    outputs[0].set_data(mx::allocator::malloc(1));
    auto& device = mx::metal::device(stream().device);
    const auto storage = arena_->storage_kind();
    auto* library = device.get_library("mlx2_paged_kv_grouped_multirow_write_v1" + storage_library_suffix(storage),
        [storage] { return storage_specialized_source(kGroupedMultirowWriteSource, storage, true, 4); });
    auto* pipeline = device.get_kernel("mlx2_paged_kv_grouped_multirow_write", library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(arena_->keys_, 2);
    encoder.set_output_array(arena_->values_, 3);
    encoder.set_output_array(outputs[0], 4);
    encoder.set_bytes(counts_, 5);
    encoder.set_bytes(starts_, 6);
    encoder.set_bytes(first_blocks_, 7);
    encoder.set_bytes(table_begins_, 8);
    encoder.set_bytes(page_ids_.data(), static_cast<int>(page_ids_.size()), 9);
    encoder.set_bytes(kv_heads_, 10);
    encoder.set_bytes(dim_, 11);
    auto owner = arena_;
    const auto epoch = epoch_;
    auto dispatched = std::make_shared<std::atomic<bool>>(false);
    encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer) {
      owner->completed(epoch, dispatched->load(std::memory_order_acquire) &&
                                buffer->status() == MTL::CommandBufferStatusCompleted);
    });
    encoder.dispatch_threads(MTL::Size(dim_, kv_heads_, counts_[0] + counts_[1]), MTL::Size(128, 1, 1));
    dispatched->store(true, std::memory_order_release);
    arena_->record_grouped_multirow_write(counts_[0] + counts_[1]);
  }

 private:
  std::shared_ptr<Arena> arena_;
  std::array<uint32_t, 2> counts_, starts_, first_blocks_, table_begins_;
  std::vector<uint32_t> page_ids_;
  uint32_t kv_heads_, dim_;
  uint64_t epoch_;
};

class GroupedN20WritePrimitive final : public mx::Primitive {
 public:
  GroupedN20WritePrimitive(std::shared_ptr<Arena> arena,
      std::vector<uint32_t> counts, std::vector<uint32_t> starts,
      std::vector<uint32_t> first_blocks, std::vector<uint32_t> table_begins,
      std::vector<uint32_t> row_begins, uint32_t rows, size_t page_count,
      uint32_t kv_heads, uint32_t dim, uint64_t epoch, mx::Stream stream)
      : Primitive(stream), arena_(std::move(arena)), counts_(std::move(counts)),
        starts_(std::move(starts)), first_blocks_(std::move(first_blocks)),
        table_begins_(std::move(table_begins)), row_begins_(std::move(row_begins)),
        rows_(rows), kv_heads_(kv_heads), dim_(dim), page_count_(page_count), epoch_(epoch) {}
  const char* name() const override { return "Mlx2PagedKVGroupedN20Write"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("grouped N20 KV writes require a GPU stream");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    if(inputs.size()!=3 || inputs[2].dtype()!=mx::uint32 || inputs[2].ndim()!=1 ||
       inputs[2].shape(0)!=static_cast<int>(page_count_) ||
       !inputs[2].flags().row_contiguous)
      throw std::invalid_argument("grouped N20 page metadata differs");
    for(const auto& source:{inputs[0],inputs[1]}) {
      const auto& strides=source.strides();
      validate_packed_n20_source_layout(rows_,kv_heads_,dim_,strides,
                                         source.offset(),source.buffer_size());
    }
    outputs[0].set_data(mx::allocator::malloc(1));
    auto& device=mx::metal::device(stream().device);
    const auto storage=arena_->storage_kind();
    auto* library=device.get_library("mlx2_paged_kv_grouped_n20_write_v1"+storage_library_suffix(storage),
        [storage]{return storage_specialized_source(kGroupedN20WriteSource,storage,true,4);});
    auto* pipeline=device.get_kernel("mlx2_paged_kv_grouped_n20_write",library);
    if(pipeline->maxTotalThreadsPerThreadgroup()<128 ||
       pipeline->staticThreadgroupMemoryLength()>device.mtl_device()->maxThreadgroupMemoryLength())
      throw std::invalid_argument("grouped N20 writer pipeline exceeds device bounds");
    auto& encoder=mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(inputs[0],0);encoder.set_input_array(inputs[1],1);
    encoder.set_output_array(arena_->keys_,2);encoder.set_output_array(arena_->values_,3);
    encoder.set_output_array(outputs[0],4);
    encoder.set_bytes(counts_.data(),static_cast<int>(counts_.size()),5);
    encoder.set_bytes(starts_.data(),static_cast<int>(starts_.size()),6);
    encoder.set_bytes(first_blocks_.data(),static_cast<int>(first_blocks_.size()),7);
    encoder.set_bytes(table_begins_.data(),static_cast<int>(table_begins_.size()),8);
    encoder.set_input_array(inputs[2],9);
    encoder.set_bytes(row_begins_.data(),static_cast<int>(row_begins_.size()),10);
    encoder.set_bytes(kv_heads_,11);encoder.set_bytes(dim_,12);
    auto owner=arena_;const auto epoch=epoch_;
    auto dispatched=std::make_shared<std::atomic<bool>>(false);
    encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer){
      owner->completed(epoch,dispatched->load(std::memory_order_acquire)&&
          buffer->status()==MTL::CommandBufferStatusCompleted);
    });
    const auto max_count=*std::max_element(counts_.begin(),counts_.end());
    encoder.dispatch_threads(MTL::Size(dim_*kv_heads_,max_count,counts_.size()),MTL::Size(128,1,1));
    dispatched->store(true,std::memory_order_release);
    arena_->record_grouped_n20_write(rows_);
  }
 private:
  std::shared_ptr<Arena> arena_;
  std::vector<uint32_t> counts_,starts_,first_blocks_,table_begins_,row_begins_;
  uint32_t rows_,kv_heads_,dim_;
  size_t page_count_;
  uint64_t epoch_;
};

class CopyPrimitive final : public mx::Primitive {
 public:
  CopyPrimitive(std::shared_ptr<Arena> arena, size_t source, size_t destination,
                size_t count, uint64_t epoch, mx::Stream stream)
      : Primitive(stream), arena_(std::move(arena)), source_(source),
        destination_(destination), count_(count), epoch_(epoch) {}

  const char* name() const override { return "Mlx2PagedKVCopy"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("paged KV arena copies require a GPU stream");
  }

  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(1));
    auto& device = mx::metal::device(stream().device);
    auto* library = device.get_library("mlx2_paged_kv_copy_v1", [] { return std::string(kCopySource); });
    auto* pipeline = device.get_kernel("mlx2_paged_kv_copy", library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(arena_->keys_, 0, static_cast<int64_t>(source_));
    encoder.set_input_array(arena_->values_, 1, static_cast<int64_t>(source_));
    encoder.set_output_array(arena_->keys_, 2, static_cast<int64_t>(destination_));
    encoder.set_output_array(arena_->values_, 3, static_cast<int64_t>(destination_));
    encoder.set_output_array(outputs[0], 4);
    encoder.set_bytes(static_cast<uint32_t>(count_), 5);
    auto owner = arena_;
    const auto epoch = epoch_;
    auto dispatched = std::make_shared<std::atomic<bool>>(false);
    encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer) {
      owner->completed(epoch, dispatched->load(std::memory_order_acquire) &&
                                buffer->status() == MTL::CommandBufferStatusCompleted);
    });
    encoder.dispatch_threads(MTL::Size(count_, 1, 1), MTL::Size(256, 1, 1));
    dispatched->store(true, std::memory_order_release);
  }

 private:
  std::shared_ptr<Arena> arena_;
  size_t source_;
  size_t destination_;
  size_t count_;
  uint64_t epoch_;
};

class DiagnosticReadPrimitive final : public mx::Primitive {
 public:
  DiagnosticReadPrimitive(std::shared_ptr<Arena> arena, size_t offset, size_t count,
                          mx::Stream stream)
      : Primitive(stream), arena_(std::move(arena)), offset_(offset), count_(count) {}

  const char* name() const override { return "Mlx2PagedKVDiagnosticRead"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("paged KV diagnostic reads require a GPU stream");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(count_));
    outputs[1].set_data(mx::allocator::malloc(count_));
    auto& device = mx::metal::device(stream().device);
    auto* library = device.get_library("mlx2_paged_kv_diagnostic_read_v1",
                                       [] { return std::string(kReadSource); });
    auto* pipeline = device.get_kernel("mlx2_paged_kv_diagnostic_read", library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(arena_->keys_, 0, static_cast<int64_t>(offset_));
    encoder.set_input_array(arena_->values_, 1, static_cast<int64_t>(offset_));
    encoder.set_input_array(inputs[0], 2);
    encoder.set_output_array(outputs[0], 3);
    encoder.set_output_array(outputs[1], 4);
    encoder.set_bytes(static_cast<uint32_t>(count_), 5);
    encoder.dispatch_threads(MTL::Size(count_, 1, 1), MTL::Size(256, 1, 1));
  }

 private:
  std::shared_ptr<Arena> arena_;
  size_t offset_;
  size_t count_;
};

class Q1GatherPrimitive final : public mx::Primitive {
 public:
  Q1GatherPrimitive(std::shared_ptr<Arena> arena, uint32_t kv_heads, uint32_t dim,
                    uint64_t epoch, std::array<uint32_t, 2> lower,
                    std::array<uint32_t, 2> visible,
                    std::array<uint32_t, 2> first_block,
                    std::array<uint32_t, 2> table_begin,
                    std::vector<uint32_t> pages, mx::Stream stream)
      : Primitive(stream), arena_(std::move(arena)), kv_heads_(kv_heads), dim_(dim),
        epoch_(epoch), lower_(lower), visible_(visible), first_block_(first_block),
        table_begin_(table_begin), pages_(std::move(pages)) {}
  const char* name() const override { return "Mlx2PagedQ1GatherFp16"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("paged Q1 gather requires a GPU stream");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    for (auto& output : outputs) output.set_data(mx::allocator::malloc(output.nbytes()));
    auto& device = mx::metal::device(stream().device);
    const auto storage = arena_->storage_kind();
    auto* library = device.get_library("mlx2_paged_q1_gather_fp16_v1" + storage_library_suffix(storage),
        [storage] { return storage_specialized_source(kQ1GatherSource, storage, true, 7); });
    auto* pipeline = device.get_kernel("mlx2_paged_q1_gather_fp16", library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(arena_->keys_, 0);
    encoder.set_input_array(arena_->values_, 1);
    encoder.set_input_array(inputs[0], 2);
    encoder.set_bytes(lower_, 3);
    encoder.set_bytes(visible_, 4);
    encoder.set_bytes(first_block_, 5);
    encoder.set_bytes(table_begin_, 6);
    encoder.set_bytes(pages_.data(), static_cast<int>(pages_.size()), 7);
    encoder.set_output_array(outputs[0], 8);
    encoder.set_output_array(outputs[1], 9);
    encoder.set_bytes(kv_heads_, 10);
    encoder.set_bytes(dim_, 11);
    const uint32_t count = 2 * kv_heads_ * 128 * dim_;
    encoder.set_bytes(count, 12);
    auto owner = arena_;
    const auto epoch = epoch_;
    auto dispatched = std::make_shared<std::atomic<bool>>(false);
    encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer) {
      owner->read_completed(epoch, dispatched->load(std::memory_order_acquire) &&
                                     buffer->status() == MTL::CommandBufferStatusCompleted);
    });
    encoder.dispatch_threads(MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
    dispatched->store(true, std::memory_order_release);
    arena_->record_q1_gather_dispatch();
  }
 private:
  std::shared_ptr<Arena> arena_;
  uint32_t kv_heads_, dim_;
  uint64_t epoch_;
  const std::array<uint32_t, 2> lower_, visible_, first_block_, table_begin_;
  const std::vector<uint32_t> pages_;
};

class AttentionReadPrimitive final : public mx::Primitive {
 public:
  AttentionReadPrimitive(std::shared_ptr<Arena> arena, float scale,
                         uint32_t query_heads, uint32_t kv_heads, uint32_t dim,
                         uint64_t epoch, bool q1_simd_tile, uint32_t q1_stripes,
                         bool q1_stock_reduction,
                         Q1SplitKVPlan split_plan, bool stock_long, bool n20, uint32_t dense_length,
                         std::vector<std::vector<uint32_t>> inline_metadata,
                         PrefillMatrixPlan prefill_plan, mx::Stream stream)
      : Primitive(stream), arena_(std::move(arena)), scale_(scale),
        query_heads_(query_heads), kv_heads_(kv_heads), dim_(dim), epoch_(epoch),
        q1_simd_tile_(q1_simd_tile), q1_stripes_(q1_stripes),
        q1_stock_reduction_(q1_stock_reduction), split_plan_(split_plan),
        stock_long_(stock_long), n20_(n20), dense_length_(dense_length),
        inline_metadata_(std::move(inline_metadata)), prefill_plan_(prefill_plan) {}

  const char* name() const override { return "Mlx2PagedAttentionReadFp16"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("paged attention read requires a GPU stream");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    const auto& query = inputs[0];
    // A lazy Q projection can claim row contiguity before evaluation and
    // become head-major after evaluation. Validate the storage actually bound
    // below; the Metal kernel uses these element strides, not logical shape.
    const auto& strides = query.strides();
    if (strides.size() != 3 || query.offset() < 0 ||
        strides[0] <= 0 || strides[1] <= 0 || strides[2] <= 0) {
      throw std::invalid_argument("evaluated paged attention query has invalid strides");
    }
    const auto storage_bytes = static_cast<uint64_t>(query.buffer_size());
    uint64_t last_index = 0;
    for (size_t axis = 0; axis < 3; ++axis) {
      const auto extent = static_cast<uint64_t>(query.shape(axis) - 1);
      const auto stride = static_cast<uint64_t>(strides[axis]);
      if (extent != 0 && stride > (UINT64_MAX - last_index) / extent) {
        throw std::invalid_argument("evaluated paged attention query stride overflows");
      }
      last_index += extent * stride;
    }
    const auto offset = static_cast<uint64_t>(query.offset());
    if (offset > storage_bytes ||
        last_index >= (storage_bytes - offset) / sizeof(mx::float16_t)) {
      throw std::invalid_argument("evaluated paged attention query exceeds storage");
    }
    const uint64_t query_row_stride = static_cast<uint64_t>(strides[0]);
    const uint64_t query_head_stride = static_cast<uint64_t>(strides[1]);
    const uint64_t query_dim_stride = static_cast<uint64_t>(strides[2]);
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    auto& device = mx::metal::device(stream().device);
    if (prefill_plan_.long_nax) {
      if (inputs.size()!=10 || !inline_metadata_.empty() || query_dim_stride!=1 ||
          device.get_architecture().empty() || device.get_architecture().back()!='s')
        throw std::invalid_argument("long fused NAX requires dynamic metadata, unit channels and architecture s");
      auto* library=device.get_library("mlx2_prefill_long_nax_bf16_d256_v1",[]{return prefill_long_nax_source();});
      auto* pipeline=device.get_kernel("mlx2_prefill_long_nax",library);
      if(pipeline->maxTotalThreadsPerThreadgroup()<256 ||
         pipeline->staticThreadgroupMemoryLength()>device.mtl_device()->maxThreadgroupMemoryLength())
        throw std::invalid_argument("long fused NAX exceeds pipeline resource bounds");
      auto& encoder=mx::metal::get_command_encoder(stream());
      encoder.set_compute_pipeline_state(pipeline);
      encoder.set_input_array(query,0);encoder.set_input_array(arena_->keys_,1);
      encoder.set_input_array(arena_->values_,2);encoder.set_input_array(inputs[1],3);
      for(int i=2;i<10;i++)encoder.set_input_array(inputs[i],i+2);
      encoder.set_output_array(outputs[0],12);encoder.set_bytes(scale_,13);
      encoder.set_bytes(query_heads_,14);encoder.set_bytes(kv_heads_,15);encoder.set_bytes(dim_,16);
      encoder.set_bytes(query_row_stride,17);encoder.set_bytes(query_head_stride,18);
      encoder.set_bytes(query_dim_stride,19);encoder.set_bytes(prefill_plan_.spans,20);
      const auto total_rows=static_cast<uint32_t>(query.shape(0));encoder.set_bytes(total_rows,21);
      auto owner=arena_;const auto epoch=epoch_;auto dispatched=std::make_shared<std::atomic<bool>>(false);
      encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer){
        owner->read_completed(epoch,dispatched->load(std::memory_order_acquire)&&
          buffer->status()==MTL::CommandBufferStatusCompleted);
      });
      encoder.dispatch_threadgroups(MTL::Size(prefill_plan_.max_tiles,query_heads_,prefill_plan_.spans),MTL::Size(256,1,1));
      dispatched->store(true,std::memory_order_release);
      if(prefill_plan_.long_n20)arena_->record_prefill_long_n20_dispatch();
      else arena_->record_prefill_long_nax_dispatch();
      arena_->record_prefill_matrix_dispatch();
      return;
    }
    if (prefill_plan_.exact_nax) {
      if(inputs.size()!=10 || !inline_metadata_.empty() || device.get_architecture().back()!='s')
        throw std::invalid_argument("prefill exact NAX requires dynamic metadata and pinned architecture s");
      if(prefill_plan_.scratch_bytes>3195072 || prefill_plan_.scratch_bytes>device.mtl_device()->maxBufferLength())
        throw std::invalid_argument("prefill exact NAX scratch bound exceeded");
      auto* library=device.get_library("mlx2_prefill_nax_exact_short_v1",[]{return prefill_nax_source();});
      auto* score=device.get_kernel("mlx2_prefill_nax_score",library);
      auto* softmax=device.get_kernel("mlx2_prefill_nax_softmax",library);
      auto* value=device.get_kernel("mlx2_prefill_nax_value",library);
      for(auto* pipe:{score,softmax,value})if(pipe->maxTotalThreadsPerThreadgroup()<(pipe==softmax?64u:256u)||
          pipe->staticThreadgroupMemoryLength()>device.mtl_device()->maxThreadgroupMemoryLength())
        throw std::invalid_argument("prefill exact NAX pipeline resource bound exceeded");
      const auto elements=static_cast<int32_t>(prefill_plan_.scratch_bytes/4);
      mx::array scores(mx::allocator::malloc(prefill_plan_.scratch_bytes/2),mx::Shape{elements},mx::bfloat16);
      mx::array probabilities(mx::allocator::malloc(prefill_plan_.scratch_bytes/2),mx::Shape{elements},mx::bfloat16);
      auto roots=std::make_shared<std::pair<mx::array,mx::array>>(scores,probabilities);
      auto& encoder=mx::metal::get_command_encoder(stream());
      encoder.add_temporary(scores);encoder.add_temporary(probabilities);
      auto owner=arena_;const auto epoch=epoch_;auto dispatched=std::make_shared<std::atomic<bool>>(false);
      encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer){
        (void)roots;owner->read_completed(epoch,dispatched->load(std::memory_order_acquire)&&buffer->status()==MTL::CommandBufferStatusCompleted);
      });
      auto bind=[&](MTL::ComputePipelineState* pipe){
        encoder.set_compute_pipeline_state(pipe);
        encoder.set_input_array(query,0);encoder.set_input_array(arena_->keys_,1);encoder.set_input_array(arena_->values_,2);
        encoder.set_input_array(inputs[1],3);for(int i=2;i<10;i++)encoder.set_input_array(inputs[i],i+2);
        encoder.set_output_array(outputs[0],12);encoder.set_bytes(scale_,13);encoder.set_bytes(query_heads_,14);
        encoder.set_bytes(kv_heads_,15);encoder.set_bytes(dim_,16);encoder.set_bytes(query_row_stride,17);
        encoder.set_bytes(query_head_stride,18);encoder.set_bytes(query_dim_stride,19);encoder.set_bytes(prefill_plan_.spans,20);
        const auto total_rows=static_cast<uint32_t>(query.shape(0));encoder.set_bytes(total_rows,21);
      };
      bind(score);encoder.set_output_array(scores,22);encoder.set_output_array(probabilities,23);
      encoder.dispatch_threadgroups(MTL::Size(prefill_plan_.max_tiles,query_heads_*2,prefill_plan_.spans),MTL::Size(256,1,1));
      arena_->record_prefill_nax_score_dispatch();
      bind(softmax);encoder.set_input_array(scores,22);encoder.set_output_array(probabilities,23);
      encoder.dispatch_threadgroups(MTL::Size(query.shape(0),query_heads_,1),MTL::Size(64,1,1));
      arena_->record_prefill_nax_softmax_dispatch();
      bind(value);encoder.set_input_array(scores,22);encoder.set_input_array(probabilities,23);
      encoder.dispatch_threadgroups(MTL::Size(prefill_plan_.max_tiles,query_heads_*2,prefill_plan_.spans),MTL::Size(256,1,1));
      arena_->record_prefill_nax_value_dispatch();
      dispatched->store(true,std::memory_order_release);arena_->record_prefill_matrix_dispatch();return;
    }
    if (prefill_plan_.spans != 0) {
      if (inputs.size() != 10 || !inline_metadata_.empty())
        throw std::invalid_argument("prefill matrix requires dynamic validated metadata");
      auto* library = device.get_library("mlx2_paged_prefill_matrix_bf16_d256_stock_short_v2", [] {
        return prefill_matrix_source();
      });
      auto* pipeline = device.get_kernel("mlx2_paged_prefill_matrix_bf16", library);
      if (pipeline->maxTotalThreadsPerThreadgroup() < 64 ||
          pipeline->staticThreadgroupMemoryLength() > device.mtl_device()->maxThreadgroupMemoryLength())
        throw std::invalid_argument("prefill matrix exceeds native pipeline resource bounds");
      auto& encoder = mx::metal::get_command_encoder(stream());
      encoder.set_compute_pipeline_state(pipeline);
      encoder.set_input_array(query, 0);
      encoder.set_input_array(arena_->keys_, 1);
      encoder.set_input_array(arena_->values_, 2);
      encoder.set_input_array(inputs[1], 3);
      for (int input = 2; input < 10; ++input)
        encoder.set_input_array(inputs[input], input + 2);
      encoder.set_output_array(outputs[0], 12);
      encoder.set_bytes(scale_, 13);
      encoder.set_bytes(query_heads_, 14);
      encoder.set_bytes(kv_heads_, 15);
      encoder.set_bytes(dim_, 16);
      encoder.set_bytes(query_row_stride, 17);
      encoder.set_bytes(query_head_stride, 18);
      encoder.set_bytes(query_dim_stride, 19);
      encoder.set_bytes(prefill_plan_.spans, 20);
      const auto total_rows = static_cast<uint32_t>(query.shape(0));
      encoder.set_bytes(total_rows, 21);
      auto owner = arena_;
      const auto epoch = epoch_;
      auto dispatched = std::make_shared<std::atomic<bool>>(false);
      encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer) {
        owner->read_completed(epoch, dispatched->load(std::memory_order_acquire) &&
                                       buffer->status() == MTL::CommandBufferStatusCompleted);
      });
      encoder.dispatch_threadgroups(MTL::Size(prefill_plan_.max_tiles, query_heads_, prefill_plan_.spans),
                                    MTL::Size(64, 1, 1));
      dispatched->store(true, std::memory_order_release);
      arena_->record_prefill_matrix_dispatch();
      return;
    }
    if (split_plan_.partition_tokens != 0) {
      if (split_plan_.scratch_bytes > device.mtl_device()->maxBufferLength())
        throw std::invalid_argument("Q1 split KV scratch exceeds device buffer limit");
      const auto dim = dim_, partition = split_plan_.partition_tokens;
      const auto storage = arena_->storage_kind();
      const bool inline_stock_long_metadata = stock_long_ && !inline_metadata_.empty();
      if (inline_stock_long_metadata && inline_metadata_.size() != 8)
        throw std::invalid_argument("stock-long inline metadata must bind eight fields");
      const std::string name = (stock_long_ ? "mlx2_paged_q1_stock_long_v1_d" :
          "mlx2_paged_q1_split_v1_d") + std::to_string(dim) +
          "_p" + std::to_string(partition) + storage_library_suffix(storage) +
          (inline_stock_long_metadata ? "_inline_metadata" : "");
      if (stock_long_ && device.get_architecture().back() != 's')
        throw std::invalid_argument("Q1 stock long requires installed MLX architecture s block policy");
      auto* library = device.get_library(name, [dim, partition, storage, stock_long = stock_long_,
                                                 inline_stock_long_metadata] {
        const auto source = stock_long ? q1_stock_long_source(dim) : q1_split_kv_source(dim, partition);
        const auto bound_source = inline_stock_long_metadata ? inline_metadata_source(source.c_str()) : source;
        return storage_specialized_source(bound_source.c_str(), storage, false, 6);
      });
      auto* partial = device.get_kernel(stock_long_ ? "mlx2_paged_q1_stock_long_partial_fp16" :
          "mlx2_paged_q1_split_partial_fp16", library);
      auto* reduce = device.get_kernel(stock_long_ ? "mlx2_paged_q1_stock_long_reduce_fp16" :
          "mlx2_paged_q1_split_reduce_fp16", library);
      for (auto* pipeline : {partial, reduce})
        if (pipeline->threadExecutionWidth() != 32 ||
            pipeline->maxTotalThreadsPerThreadgroup() <
                (stock_long_ && pipeline == reduce ? 1024u : stock_long_ ? 32u : 128u) ||
            pipeline->staticThreadgroupMemoryLength() > device.mtl_device()->maxThreadgroupMemoryLength())
          throw std::invalid_argument("Q1 split KV exceeds actual device threadgroup limits");
      mx::array scratch(mx::allocator::malloc(split_plan_.scratch_bytes),
          mx::Shape{static_cast<int32_t>(split_plan_.scratch_values)}, mx::float32);
      auto scratch_owner = std::make_shared<mx::array>(scratch);
      auto& encoder = mx::metal::get_command_encoder(stream());
      encoder.add_temporary(scratch);
      encoder.set_compute_pipeline_state(partial);
      encoder.set_input_array(inputs[0], 0);
      encoder.set_input_array(arena_->keys_, 1);
      encoder.set_input_array(arena_->values_, 2);
      encoder.set_input_array(inputs[1], 3);
      if (inline_stock_long_metadata) {
        for (size_t field = 0; field < inline_metadata_.size(); ++field) {
          const auto& items = inline_metadata_[field];
          encoder.set_bytes(items.data(), static_cast<int>(items.size()), static_cast<int>(field + 4));
        }
      } else {
        for (int input = 2; input < 10; ++input)
          encoder.set_input_array(inputs[input], input + 2);
      }
      encoder.set_output_array(scratch, 12);
      encoder.set_bytes(scale_, 13);
      encoder.set_bytes(query_heads_, 14);
      encoder.set_bytes(kv_heads_, 15);
      encoder.set_bytes(split_plan_.partitions, 16);
      encoder.set_bytes(query_row_stride, 17);
      encoder.set_bytes(query_head_stride, 18);
      encoder.set_bytes(query_dim_stride, 19);
      if (stock_long_) encoder.set_bytes(dense_length_, 20);
      auto owner = arena_;
      const auto epoch = epoch_;
      auto dispatched = std::make_shared<std::atomic<bool>>(false);
      // The callback pins scratch as well as arena/read epoch before either
      // stage dispatch. Success requires both stages encoded on this buffer.
      encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer) {
        (void)scratch_owner;
        owner->read_completed(epoch, dispatched->load(std::memory_order_acquire) &&
                                       buffer->status() == MTL::CommandBufferStatusCompleted);
      });
      const auto qrows=static_cast<uint32_t>(query.shape(0));
      encoder.dispatch_threadgroups(MTL::Size(split_plan_.partitions, n20_ ? qrows : 2, query_heads_),
                                    MTL::Size(stock_long_ ? 32 : 128, 1, 1));
      if (inline_stock_long_metadata) arena_->record_q1_stock_long_metadata_dispatch();
      if (n20_) {
        arena_->record_q1_stock_long_n20_partial_dispatch();
        if (qrows == 1) arena_->record_q1_stock_long_n20_singleton_partial_dispatch();
      }
      else if (stock_long_) arena_->record_q1_stock_long_partial_dispatch();
      else arena_->record_q1_split_partial_dispatch();
      encoder.set_compute_pipeline_state(reduce);
      // Registering scratch as input after its output dispatch lets MLX insert
      // the same resource barrier used by its other producer/consumer kernels.
      encoder.set_input_array(scratch, 0);
      encoder.set_output_array(outputs[0], 1);
      encoder.set_bytes(query_heads_, 2);
      encoder.set_bytes(split_plan_.partitions, 3);
      encoder.dispatch_threadgroups(MTL::Size(1, n20_ ? qrows : 2, query_heads_),
                                    MTL::Size(stock_long_ ? 1024 : 128, 1, 1));
      if (n20_) {
        arena_->record_q1_stock_long_n20_reduce_dispatch();
        if (qrows == 1) arena_->record_q1_stock_long_n20_singleton_reduce_dispatch();
      }
      else if (stock_long_) arena_->record_q1_stock_long_reduce_dispatch();
      else arena_->record_q1_split_reduce_dispatch();
      dispatched->store(true, std::memory_order_release);
      return;
    }
    const bool inline_metadata = !inline_metadata_.empty();
    const char* source = q1_simd_tile_ ? kAttentionReadQ1TileSource : kAttentionReadSource;
    std::string library_name = q1_simd_tile_ ?
        (inline_metadata ? "mlx2_paged_attention_q1_tile_inline_v1" :
                           "mlx2_paged_attention_read_q1_simd_tile_fp16_v1") :
        (inline_metadata ? "mlx2_paged_attention_inline_v1" :
                           "mlx2_paged_attention_read_fp16_v2");
    if (q1_stripes_ != 4)
      library_name += "_d" + std::to_string(dim_) + "_s" + std::to_string(q1_stripes_);
    if (q1_stock_reduction_) library_name += "_stock_reduce_v1";
    const auto storage = arena_->storage_kind();
    library_name += storage_library_suffix(storage);
    const auto stripes = q1_stripes_;
    const auto dim = dim_;
    const bool stock_reduction = q1_stock_reduction_;
    auto* library = device.get_library(library_name, [source, inline_metadata, stripes, dim, storage, stock_reduction] {
      auto text = stock_reduction ? stock_reduction_source(source) :
          (stripes == 4 ? std::string(source) : specialized_q1_source(source, dim, stripes));
      text = storage_specialized_source(text.c_str(), storage, false, 6);
      return inline_metadata ? inline_metadata_source(text.c_str()) : text;
    });
    auto* pipeline = device.get_kernel(q1_simd_tile_ ?
        "mlx2_paged_attention_read_q1_simd_tile_fp16" :
        "mlx2_paged_attention_read_fp16", library);
    if (q1_simd_tile_ &&
        (32 * q1_stripes_ > pipeline->maxTotalThreadsPerThreadgroup() ||
         pipeline->staticThreadgroupMemoryLength() > device.mtl_device()->maxThreadgroupMemoryLength()))
      throw std::invalid_argument("Q1 stripe geometry exceeds kernel threadgroup limits");
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(arena_->keys_, 1);
    encoder.set_input_array(arena_->values_, 2);
    encoder.set_input_array(inputs[1], 3); // Retain the write dependency in the graph.
    if (inline_metadata) {
      for (size_t field = 0; field < inline_metadata_.size(); ++field) {
        const auto& items = inline_metadata_[field];
        encoder.set_bytes(items.data(), static_cast<int>(items.size()),
                          static_cast<int>(field) + 4);
      }
    } else {
      for (int input = 2; input < 10; ++input)
        encoder.set_input_array(inputs[input], input + 2);
    }
    encoder.set_output_array(outputs[0], 12);
    encoder.set_bytes(scale_, 13);
    encoder.set_bytes(query_heads_, 14);
    encoder.set_bytes(kv_heads_, 15);
    encoder.set_bytes(dim_, 16);
    encoder.set_bytes(query_row_stride, 17);
    encoder.set_bytes(query_head_stride, 18);
    encoder.set_bytes(query_dim_stride, 19);
    auto owner = arena_;
    const auto epoch = epoch_;
    auto dispatched = std::make_shared<std::atomic<bool>>(false);
    encoder.get_command_buffer()->addCompletedHandler(^(MTL::CommandBuffer* buffer) {
      owner->read_completed(epoch, dispatched->load(std::memory_order_acquire) &&
                                     buffer->status() == MTL::CommandBufferStatusCompleted);
    });
    encoder.dispatch_threads(
        MTL::Size(q1_simd_tile_ ? 32 * q1_stripes_ : 32,
                  static_cast<size_t>(outputs[0].shape(0)), query_heads_),
        MTL::Size(q1_simd_tile_ ? 32 * q1_stripes_ : 32, 1, 1));
    dispatched->store(true, std::memory_order_release);
    if (!q1_simd_tile_ && outputs[0].shape(0) == 1)
      arena_->record_q1_scalar_dispatch();
    if (q1_simd_tile_) {
      arena_->record_q1_tile_dispatch();
      arena_->record_q1_stripe_dispatch(q1_stripes_);
      if (q1_stock_reduction_) {
        arena_->record_q1_stock_reduction_dispatch();
        if (outputs[0].shape(0) == 1) arena_->record_q1_stock_singleton_dispatch();
      }
    }
    if (inline_metadata)
      arena_->record_q1_metadata_dispatch();
  }

 private:
  std::shared_ptr<Arena> arena_;
  float scale_;
  uint32_t query_heads_;
  uint32_t kv_heads_;
  uint32_t dim_;
  uint64_t epoch_;
  bool q1_simd_tile_;
  uint32_t q1_stripes_;
  bool q1_stock_reduction_;
  Q1SplitKVPlan split_plan_;
  bool stock_long_;
  bool n20_;
  uint32_t dense_length_;
  const std::vector<std::vector<uint32_t>> inline_metadata_;
  PrefillMatrixPlan prefill_plan_;
};

Arena::Arena(size_t plane_bytes, mx::array keys, mx::array values, StorageDtype storage_dtype)
    : plane_bytes_(plane_bytes), storage_dtype_(storage_dtype), keys_(std::move(keys)), values_(std::move(values)) {}

std::shared_ptr<Arena> Arena::create(size_t plane_bytes) {
  return create(plane_bytes, StorageDtype::Float16);
}

std::shared_ptr<Arena> Arena::create(size_t plane_bytes, StorageDtype storage_dtype) {
  auto* metal_device = mx::metal::device(mx::Device::gpu).mtl_device();
  if (metal_device == nullptr)
    throw std::invalid_argument("Metal device unavailable for paged KV arena");
  const auto plan = plan_arena_storage(plane_bytes, metal_device->maxBufferLength());
  const auto shape = plan.multidimensional ? mx::Shape{plan.rows, plan.columns}
                                            : mx::Shape{plan.columns};
  mx::array keys(mx::allocator::malloc(plane_bytes), shape, mx::uint8);
  mx::array values(mx::allocator::malloc(plane_bytes), shape, mx::uint8);
  return std::shared_ptr<Arena>(new Arena(plane_bytes, std::move(keys), std::move(values), storage_dtype));
}

mx::array Arena::write(
    const mx::array& key_bytes,
    const mx::array& value_bytes,
    size_t destination_offset,
    uint64_t epoch,
    mx::Stream stream,
    bool inject_terminal_failure) {
  if (stream.device.type != mx::Device::gpu || key_bytes.dtype() != mx::uint8 ||
      value_bytes.dtype() != mx::uint8 || key_bytes.ndim() != 1 ||
      value_bytes.ndim() != 1 || key_bytes.nbytes() == 0 ||
      key_bytes.nbytes() != value_bytes.nbytes() ||
      !key_bytes.flags().row_contiguous || !value_bytes.flags().row_contiguous ||
      key_bytes.nbytes() > std::numeric_limits<uint32_t>::max() ||
      epoch == 0 ||
      destination_offset > plane_bytes_ || key_bytes.nbytes() > plane_bytes_ - destination_offset ||
      destination_offset > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument("invalid paged KV write span or stream");
  }
  auto primitive = std::make_shared<WritePrimitive>(
      shared_from_this(), destination_offset, epoch, stream, inject_terminal_failure);
  return mx::array(mx::Shape{1}, mx::uint8, std::move(primitive), {key_bytes, value_bytes});
}

mx::array Arena::grouped_q1_write(
    const mx::array& keys, const mx::array& values,
    std::array<uint32_t, 2> pages, std::array<uint32_t, 2> slots,
    uint32_t kv_heads, uint32_t dim, uint64_t epoch, mx::Stream stream) {
  const uint64_t page_bytes = static_cast<uint64_t>(kv_heads) * 64 * dim * 2;
  if (stream.device.type != mx::Device::gpu || epoch == 0 ||
      keys.dtype() != dtype() || values.dtype() != dtype() ||
      keys.ndim() != 3 || values.ndim() != 3 ||
      keys.shape(0) != 2 || values.shape(0) != 2 ||
      kv_heads == 0 || kv_heads > 32 ||
      (dim != 128 && dim != 256) ||
      keys.shape(1) != static_cast<int>(kv_heads) ||
      values.shape(1) != static_cast<int>(kv_heads) ||
      keys.shape(2) != static_cast<int>(dim) ||
      values.shape(2) != static_cast<int>(dim) ||
      page_bytes == 0 || plane_bytes_ % page_bytes != 0 ||
      pages[0] == pages[1] ||
      pages[0] >= plane_bytes_ / page_bytes ||
      pages[1] >= plane_bytes_ / page_bytes ||
      slots[0] >= 64 || slots[1] >= 64) {
    throw std::invalid_argument("invalid grouped Q1 native write geometry");
  }
  auto primitive = std::make_shared<GroupedQ1WritePrimitive>(
      shared_from_this(), pages, slots, kv_heads, dim, epoch, stream);
  return mx::array(mx::Shape{1}, mx::uint8, std::move(primitive), {keys, values});
}

mx::array Arena::grouped_multirow_write(
    const mx::array& keys, const mx::array& values,
    std::array<uint32_t, 2> counts, std::array<uint32_t, 2> starts,
    std::array<uint32_t, 2> first_blocks, std::array<uint32_t, 2> table_begins,
    const std::vector<uint32_t>& page_ids, uint32_t kv_heads, uint32_t dim,
    uint64_t epoch, mx::Stream stream) {
  if (!packed_multirow_write_selected(std::getenv("MLX2_PAGED_GROUPED_MULTIROW_WRITE")))
    throw std::invalid_argument("grouped multirow write requires explicit selector");
  const uint64_t page_bytes = static_cast<uint64_t>(kv_heads) * 64 * dim * 2;
  if (stream.device.type != mx::Device::gpu || epoch == 0 ||
      keys.dtype() != dtype() || values.dtype() != dtype() ||
      keys.ndim() != 3 || values.ndim() != 3 ||
      kv_heads == 0 || kv_heads > 32 || (dim != 128 && dim != 256) ||
      keys.shape(1) != static_cast<int>(kv_heads) ||
      values.shape(1) != static_cast<int>(kv_heads) ||
      keys.shape(2) != static_cast<int>(dim) ||
      values.shape(2) != static_cast<int>(dim) ||
      keys.shape(0) != values.shape(0) ||
      page_bytes == 0 || plane_bytes_ % page_bytes != 0)
    throw std::invalid_argument("invalid grouped multirow native write geometry");
  validate_packed_multirow_write(counts, starts, first_blocks, table_begins,
      page_ids, plane_bytes_ / page_bytes, static_cast<uint32_t>(keys.shape(0)));
  auto primitive = std::make_shared<GroupedMultirowWritePrimitive>(
      shared_from_this(), counts, starts, first_blocks, table_begins,
      page_ids, kv_heads, dim, epoch, stream);
  return mx::array(mx::Shape{1}, mx::uint8, std::move(primitive), {keys, values});
}

mx::array Arena::grouped_multirow_write_n20(
    const mx::array& keys, const mx::array& values,
    const std::vector<uint32_t>& counts, const std::vector<uint32_t>& starts,
    const std::vector<uint32_t>& first_blocks,
    const std::vector<uint32_t>& table_begins,
    const std::vector<uint32_t>& page_ids, uint32_t kv_heads, uint32_t dim,
    uint64_t epoch, mx::Stream stream) {
  if (!packed_n20_requested(std::getenv("MLX2_PAGED_PACKED_N20")))
    throw std::invalid_argument("grouped N20 write requires explicit selector");
  const uint64_t page_bytes=uint64_t(kv_heads)*64*dim*2;
  if(stream.device.type!=mx::Device::gpu || epoch==0 ||
     storage_kind()!=StorageDtype::BFloat16 ||
     keys.dtype()!=dtype() || values.dtype()!=dtype() ||
     keys.ndim()!=3 || values.ndim()!=3 || kv_heads!=4 || dim!=256 ||
     keys.shape(1)!=4 || values.shape(1)!=4 ||
     keys.shape(2)!=256 || values.shape(2)!=256 ||
     keys.shape(0)!=values.shape(0) || page_bytes==0 ||
     plane_bytes_%page_bytes!=0 || keys.shape(0)<=0 ||
     keys.shape(0)>20*8192)
    throw std::invalid_argument("invalid grouped N20 native write geometry");
  auto plan=validate_packed_n20_write(counts,starts,first_blocks,table_begins,
      page_ids,plane_bytes_/page_bytes,static_cast<uint32_t>(keys.shape(0)));
  mx::array pages(page_ids.begin(),mx::Shape{static_cast<int32_t>(page_ids.size())},mx::uint32);
  auto primitive=std::make_shared<GroupedN20WritePrimitive>(shared_from_this(),
      counts,starts,first_blocks,table_begins,std::move(plan.row_begin),plan.rows,
      page_ids.size(),kv_heads,dim,epoch,stream);
  return mx::array(mx::Shape{1},mx::uint8,std::move(primitive),{keys,values,pages});
}

mx::array Arena::copy_page(
    size_t source_offset, size_t destination_offset, size_t byte_count,
    uint64_t epoch, mx::Stream stream) {
  if (stream.device.type != mx::Device::gpu || byte_count == 0 || epoch == 0 ||
      byte_count > std::numeric_limits<uint32_t>::max() ||
      source_offset > plane_bytes_ || byte_count > plane_bytes_ - source_offset ||
      destination_offset > plane_bytes_ || byte_count > plane_bytes_ - destination_offset ||
      source_offset > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      destination_offset > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      (source_offset < destination_offset + byte_count &&
       destination_offset < source_offset + byte_count)) {
    throw std::invalid_argument("invalid paged KV copy span or stream");
  }
  auto primitive = std::make_shared<CopyPrimitive>(
      shared_from_this(), source_offset, destination_offset, byte_count, epoch, stream);
  return mx::array(mx::Shape{1}, mx::uint8, std::move(primitive), {});
}

std::vector<mx::array> Arena::diagnostic_read(
    const mx::array& dependency, size_t source_offset, size_t byte_count,
    mx::Stream stream) {
  if (stream.device.type != mx::Device::gpu || dependency.dtype() != mx::uint8 ||
      dependency.ndim() != 1 || dependency.nbytes() != 1 ||
      !dependency.flags().row_contiguous || byte_count == 0 ||
      byte_count > std::numeric_limits<uint32_t>::max() ||
      byte_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      source_offset > plane_bytes_ || byte_count > plane_bytes_ - source_offset ||
      source_offset > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument("invalid paged KV diagnostic read span or stream");
  }
  auto primitive = std::make_shared<DiagnosticReadPrimitive>(
      shared_from_this(), source_offset, byte_count, stream);
  auto shape = mx::Shape{static_cast<int32_t>(byte_count)};
  return mx::array::make_arrays({shape, shape}, {mx::uint8, mx::uint8},
                                primitive, {dependency});
}

mx::array Arena::attention_read_fp16(
    const mx::array& query, const mx::array& dependency,
    const std::vector<std::vector<uint32_t>>& spans,
    const std::vector<uint32_t>& page_ids, uint32_t kv_heads,
    float scale, uint64_t epoch, mx::Stream stream) {
  if (stream.device.type != mx::Device::gpu || epoch == 0 ||
      query.dtype() != dtype() || query.ndim() != 3 ||
      query.shape(0) <= 0 ||
      query.shape(0) > (packed_n20_requested(std::getenv("MLX2_PAGED_PACKED_N20")) ? 20*8192 : 65536) ||
      query.shape(1) <= 0 ||
      (query.shape(2) != 128 && query.shape(2) != 256) ||
      dependency.dtype() != mx::uint8 || dependency.ndim() != 1 ||
      dependency.nbytes() != 1 || !dependency.flags().row_contiguous ||
      !std::isfinite(scale) || scale <= 0 || kv_heads == 0 ||
      static_cast<uint32_t>(query.shape(1)) % kv_heads != 0 ||
      spans.empty() || spans.size() > static_cast<size_t>(query.shape(0)) ||
      page_ids.empty() || page_ids.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("invalid paged attention read geometry or stream");
  }
  const auto query_heads = static_cast<uint32_t>(query.shape(1));
  const auto dim = static_cast<uint32_t>(query.shape(2));
  const uint64_t page_bytes = static_cast<uint64_t>(kv_heads) * 64 * dim * 2;
  if (plane_bytes_ % page_bytes != 0 || plane_bytes_ / page_bytes > UINT32_MAX) {
    throw std::invalid_argument("paged attention arena geometry disagrees with query");
  }
  const auto capacity = static_cast<uint32_t>(plane_bytes_ / page_bytes);
  for (auto page : page_ids) {
    if (page >= capacity)
      throw std::invalid_argument("paged attention page ID exceeds arena capacity");
  }
  std::vector<uint32_t> row_span, row_begin, query_start, retained_start;
  std::vector<uint32_t> first_block, table_begin, window;
  row_span.reserve(static_cast<size_t>(query.shape(0)));
  uint64_t rows = 0;
  uint64_t pages = 0;
  bool short_q1 = true;
  bool q1_pair = query.shape(0) == 2 && spans.size() == 2;
  std::array<uint32_t, 2> visible_pair{};
  std::vector<uint32_t> visible_all;
  visible_all.reserve(spans.size());
  for (size_t i = 0; i < spans.size(); ++i) {
    const auto& span = spans[i];
    if (span.size() != 8)
      throw std::invalid_argument("paged attention span must have eight fields");
    const uint64_t count = span[0], start = span[1], end = span[2];
    const uint64_t retained = span[3], first = span[4], table = span[5];
    const uint64_t table_count = span[6];
    if (count == 0 || retained > start || start >= end || start + count != end ||
        first != retained / 64 || table != pages ||
        table_count != (end - 1) / 64 - first + 1 ||
        rows + count > static_cast<uint64_t>(query.shape(0)) ||
        pages + table_count > page_ids.size()) {
      throw std::invalid_argument("invalid paged attention span positions or table");
    }
    const uint64_t visible = span[7] == 0 ? end - retained :
        std::min<uint64_t>(end - retained, span[7]);
    short_q1 = short_q1 && count == 1 && visible >= 32 && visible <= 128;
    q1_pair = q1_pair && count == 1;
    if (i < 2) visible_pair[i] = static_cast<uint32_t>(visible);
    visible_all.push_back(static_cast<uint32_t>(visible));
    row_begin.push_back(static_cast<uint32_t>(rows));
    query_start.push_back(static_cast<uint32_t>(start));
    retained_start.push_back(static_cast<uint32_t>(retained));
    first_block.push_back(static_cast<uint32_t>(first));
    table_begin.push_back(static_cast<uint32_t>(table));
    window.push_back(span[7]);
    row_span.insert(row_span.end(), static_cast<size_t>(count), static_cast<uint32_t>(i));
    rows += count;
    pages += table_count;
  }
  if (rows != static_cast<uint64_t>(query.shape(0)) || pages != page_ids.size()) {
    throw std::invalid_argument("paged attention rows or pages are unused");
  }
  const bool matrix_requested = prefill_matrix_requested(std::getenv("MLX2_PAGED_PREFILL_MATRIX"));
  const bool multiquery = std::any_of(spans.begin(), spans.end(), [](const auto& span) { return span[0] > 1; });
  PrefillMatrixPlan prefill_plan{};
  const bool exact_nax_requested=prefill_matrix_requested(std::getenv("MLX2_PAGED_PREFILL_NAX_EXACT"));
  const bool long_nax_requested=prefill_matrix_requested(std::getenv("MLX2_PAGED_PREFILL_NAX_LONG_FUSED"));
  const bool n20_requested=packed_n20_requested(std::getenv("MLX2_PAGED_PACKED_N20"));
  if(exact_nax_requested&&!matrix_requested)throw std::invalid_argument("exact NAX requires explicit matrix selector");
  if(long_nax_requested && (!matrix_requested || exact_nax_requested))
    throw std::invalid_argument("long fused NAX requires matrix and excludes short exact NAX");
  if(n20_requested && (!matrix_requested || !long_nax_requested || exact_nax_requested))
    throw std::invalid_argument("packed N20 requires the explicit long fused matrix selector");
  if (matrix_requested && multiquery)
    prefill_plan = n20_requested?prefill_long_n20_plan(storage_kind()==StorageDtype::BFloat16,dim,query_heads,kv_heads,spans):
      long_nax_requested?prefill_long_nax_plan(storage_kind()==StorageDtype::BFloat16,dim,query_heads,kv_heads,spans):
      exact_nax_requested?prefill_nax_plan(storage_kind()==StorageDtype::BFloat16,dim,query_heads,kv_heads,spans):
      prefill_matrix_plan(storage_kind() == StorageDtype::BFloat16, dim, query_heads, kv_heads, spans);
  const uint32_t split_partition = q1_split_kv_partition(std::getenv("MLX2_PAGED_Q1_SPLIT_KV"));
  const bool stock_long_requested = q1_stock_long_requested(
      std::getenv("MLX2_PAGED_Q1_STOCK_LONG"));
  const uint32_t dense_length = n20_requested ?
      *std::max_element(visible_all.begin(),visible_all.end()) :
      std::max(visible_pair[0], visible_pair[1]);
  const bool stock_long = stock_long_requested && q1_pair && dense_length > 1024;
  const bool stock_n20_singleton_requested = q1_stock_long_n20_singleton_requested(
      std::getenv("MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON"));
  if (stock_n20_singleton_requested && (!n20_requested || !stock_long_requested))
    throw std::invalid_argument("N20 B1 stock-long requires N20 and stock-long selectors");
  const bool stock_n20 = n20_requested && !multiquery && rows >= 1 && rows <= 20 &&
      (rows >= 2 || stock_n20_singleton_requested) &&
      stock_long_requested && dense_length > 1024;
  if (stock_n20_singleton_requested && !multiquery && rows == 1 && !stock_n20)
    throw std::invalid_argument("N20 B1 stock-long selector has no admitted split geometry");
  const char* stock_blocks_override = std::getenv("MLX_SDPA_BLOCKS");
  if (stock_long && (split_partition != 0 || spans[0][3] != 0 || spans[1][3] != 0 ||
      spans[0][7] != 0 || spans[1][7] != 0 ||
      query_heads / kv_heads <= 4 ||
      (stock_blocks_override && std::string(stock_blocks_override) != "0")))
    throw std::invalid_argument("Q1 stock long requires unwindowed B2, no block/split overrides, and GQA above four");
  if (stock_n20 && (split_partition != 0 || query_heads / kv_heads <= 4 ||
      (stock_blocks_override && std::string(stock_blocks_override) != "0") ||
      std::any_of(spans.begin(),spans.end(),[](const auto& s){return s[3]!=0 || s[4]!=0 || s[7]!=0 || s[0]!=1;})))
    throw std::invalid_argument("Q1 stock N20 requires full unwindowed origin0 and no block/split override");
  Q1SplitKVPlan split_plan{};
  if (stock_n20) split_plan = q1_stock_long_n20_plan(dim, query_heads, visible_all);
  else if (stock_long) split_plan = q1_stock_long_plan(dim, query_heads, visible_pair);
  else if (split_partition != 0 && q1_pair && (visible_pair[0] > 128 || visible_pair[1] > 128))
    split_plan = q1_split_kv_plan(dim, query_heads, visible_pair, split_partition);
  // The long prefill selector persists across request steps. It is irrelevant
  // to Q1 arithmetic, but must not silently admit an unsupported Q1 shape.
  // B2 selects the separately proven stock-long split; a cancelled B2 peer
  // may continue through the existing bounded B1 scalar reader.
  const bool long_scalar_b1 = bounded_long_scalar_b1(
      stock_long_requested, storage_kind() == StorageDtype::BFloat16,
      dim, query_heads, kv_heads, static_cast<uint32_t>(rows), spans, split_partition);
  validate_long_nax_q1_handoff(long_nax_requested, multiquery,
      (stock_long || stock_n20) && split_plan.partition_tokens != 0, long_scalar_b1);
  if (n20_requested && !multiquery && !stock_n20 && !long_scalar_b1)
    throw std::invalid_argument("packed N20 has no admitted Q1 successor");
  auto as_array = [](const std::vector<uint32_t>& items) {
    return mx::array(items.begin(), mx::Shape{static_cast<int32_t>(items.size())}, mx::uint32);
  };
  const bool inline_q1_metadata = split_plan.partition_tokens == 0 && use_inline_q1_metadata(
      std::getenv("MLX2_PAGED_Q1_INLINE_METADATA"), short_q1,
      rows, spans.size(), page_ids.size());
  const char* long_inline_flag=std::getenv("MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA");
  if(n20_requested && long_inline_flag && std::string(long_inline_flag)!="0")
    throw std::invalid_argument("packed N20 requires dynamic long Q1 metadata");
  const bool inline_stock_long_metadata = use_inline_stock_long_metadata(
      long_inline_flag, stock_long && !n20_requested,
      split_plan.partition_tokens, rows, spans.size(),
      {row_span.size(), row_begin.size(), query_start.size(), retained_start.size(),
       first_block.size(), table_begin.size(), window.size(), page_ids.size()});
  std::vector<std::vector<uint32_t>> inline_metadata;
  std::vector<mx::array> inputs = {query, dependency};
  if (inline_q1_metadata || inline_stock_long_metadata) {
    inline_metadata.reserve(8);
    inline_metadata.emplace_back(std::move(row_span));
    inline_metadata.emplace_back(std::move(row_begin));
    inline_metadata.emplace_back(std::move(query_start));
    inline_metadata.emplace_back(std::move(retained_start));
    inline_metadata.emplace_back(std::move(first_block));
    inline_metadata.emplace_back(std::move(table_begin));
    inline_metadata.emplace_back(std::move(window));
    inline_metadata.emplace_back(page_ids);
  } else {
    inputs.insert(inputs.end(), {as_array(row_span), as_array(row_begin),
        as_array(query_start), as_array(retained_start), as_array(first_block),
        as_array(table_begin), as_array(window), as_array(page_ids)});
  }
  const char* tile_switch = std::getenv("MLX2_PAGED_Q1_SIMD_TILE");
  const bool q1_simd_tile = short_q1 && tile_switch != nullptr &&
      std::string(tile_switch) == "1";
  const bool stock_requested = use_q1_stock_reduction(
      std::getenv("MLX2_PAGED_Q1_STOCK_REDUCTION"));
  const uint32_t requested_stripes = q1_simd_tile ?
      q1_simd_stripes(std::getenv("MLX2_PAGED_Q1_SIMD_STRIPES")) : 4;
  const bool singleton_requested = use_q1_stock_singleton(
      std::getenv("MLX2_PAGED_Q1_STOCK_SINGLETON"));
  if (singleton_requested && !stock_requested)
    throw std::invalid_argument("Q1 stock singleton requires stock reduction selector");
  const bool short_singleton = rows == 1 && spans.size() == 1 && short_q1;
  const bool stock_reduction = q1_stock_reduction_active(
      stock_requested, q1_pair && short_q1, singleton_requested, short_singleton);
  if (stock_reduction) {
    // Do not index spans[1] for B1. Prefix/window validation applies to both arms.
    if (!q1_simd_tile || split_plan.partition_tokens != 0 ||
        spans[0][3] != 0 || spans[0][7] != 0 ||
        (q1_pair && (spans[1][3] != 0 || spans[1][7] != 0 ||
         !q1_stock_pair_padding_aligned(visible_pair[0], visible_pair[1]))))
      throw std::invalid_argument("Q1 stock reduction requires short unwindowed full-prefix B1 or aligned B2");
  }
  // Preflight the survivor geometry even while the pair selects 32 stripes.
  // A cancelled peer must not turn an admitted B2 route into an invalid B1.
  if (requested_stripes != 4 && !q1_geometry_supported(dim, requested_stripes))
    throw std::invalid_argument("Q1 survivor stripe geometry exceeds bounded threadgroup memory");
  const uint32_t q1_stripes = stock_reduction ? 32 : requested_stripes;
  auto primitive = std::make_shared<AttentionReadPrimitive>(
      shared_from_this(), scale, query_heads, kv_heads, dim, epoch, q1_simd_tile, q1_stripes,
      stock_reduction, split_plan, stock_long || stock_n20, n20_requested, dense_length,
      std::move(inline_metadata), prefill_plan, stream);
  return mx::array(query.shape(), dtype(), std::move(primitive), std::move(inputs));
}

std::vector<mx::array> Arena::gather_q1_fp16(
    const mx::array& query, const mx::array& dependency,
    const std::vector<std::vector<uint32_t>>& spans,
    const std::vector<uint32_t>& page_ids, uint32_t kv_heads,
    float scale, uint64_t epoch, mx::Stream stream) {
  if (stream.device.type != mx::Device::gpu || epoch == 0 ||
      query.dtype() != dtype() || query.ndim() != 3 ||
      query.shape(0) != 2 || query.shape(1) <= 0 ||
      (query.shape(2) != 128 && query.shape(2) != 256) ||
      dependency.dtype() != mx::uint8 || dependency.ndim() != 1 ||
      dependency.nbytes() != 1 || !dependency.flags().row_contiguous ||
      !std::isfinite(scale) || scale <= 0 || kv_heads == 0 || kv_heads > 32 ||
      static_cast<uint32_t>(query.shape(1)) % kv_heads != 0 ||
      spans.size() != 2 || page_ids.empty() || page_ids.size() > 6) {
    throw std::invalid_argument("invalid paged attention read geometry or stream");
  }
  const auto dim = static_cast<uint32_t>(query.shape(2));
  const uint64_t page_bytes = static_cast<uint64_t>(kv_heads) * 64 * dim * 2;
  if (plane_bytes_ % page_bytes != 0 || plane_bytes_ / page_bytes > UINT32_MAX) {
    throw std::invalid_argument("paged attention arena geometry disagrees with query");
  }
  const auto capacity = static_cast<uint32_t>(plane_bytes_ / page_bytes);
  for (auto page : page_ids) {
    if (page >= capacity)
      throw std::invalid_argument("paged attention page ID exceeds arena capacity");
  }
  uint64_t rows = 0;
  uint64_t pages = 0;
  bool short_q1 = true;
  for (size_t i = 0; i < spans.size(); ++i) {
    const auto& span = spans[i];
    if (span.size() != 8)
      throw std::invalid_argument("paged attention span must have eight fields");
    const uint64_t count = span[0], start = span[1], end = span[2];
    const uint64_t retained = span[3], first = span[4], table = span[5];
    const uint64_t table_count = span[6];
    if (count == 0 || retained > start || start >= end || start + count != end ||
        first != retained / 64 || table != pages ||
        table_count != (end - 1) / 64 - first + 1 ||
        rows + count > static_cast<uint64_t>(query.shape(0)) ||
        pages + table_count > page_ids.size()) {
      throw std::invalid_argument("invalid paged attention span positions or table");
    }
    const uint64_t visible = span[7] == 0 ? end - retained :
        std::min<uint64_t>(end - retained, span[7]);
    short_q1 = short_q1 && count == 1 && visible >= 32 && visible <= 128;
    rows += count;
    pages += table_count;
  }
  if (rows != static_cast<uint64_t>(query.shape(0)) || pages != page_ids.size()) {
    throw std::invalid_argument("paged attention rows or pages are unused");
  }
  if (!short_q1 || rows != 2 || spans.size() != 2 || kv_heads > 32 || page_ids.size() > 6)
    throw std::invalid_argument("paged Q1 gather requires bounded two-lane short reads");
  std::array<uint32_t, 2> lower{}, visible{}, first{}, table{};
  for (size_t row = 0; row < 2; ++row) {
    const auto& span = spans[row];
    const uint32_t end = span[2];
    lower[row] = span[7] == 0 ? span[3] : std::max(span[3], end - std::min(end, span[7]));
    visible[row] = end - lower[row];
    first[row] = span[4];
    table[row] = span[5];
  }
  auto primitive = std::make_shared<Q1GatherPrimitive>(shared_from_this(), kv_heads, dim,
      epoch, lower, visible, first, table, page_ids, stream);
  const mx::Shape shape{2, static_cast<int32_t>(kv_heads), 128, static_cast<int32_t>(dim)};
  return mx::array::make_arrays({shape, shape}, {dtype(), dtype()},
                                std::move(primitive), {dependency});
}

void Arena::completed(uint64_t epoch, bool succeeded) {
  {
    std::lock_guard lock(completions_mutex_);
    completions_.push_back({epoch, succeeded});
  }
  completions_ready_.notify_one();
}

void Arena::read_completed(uint64_t epoch, bool succeeded) {
  {
    std::lock_guard lock(completions_mutex_);
    read_completions_.push_back({epoch, succeeded});
  }
  read_completions_ready_.notify_one();
}

std::vector<Completion> Arena::poll_completions() {
  std::lock_guard lock(completions_mutex_);
  std::vector<Completion> result(completions_.begin(), completions_.end());
  completions_.clear();
  return result;
}

std::vector<Completion> Arena::wait_completions(double timeout_seconds) {
  if (!std::isfinite(timeout_seconds) || timeout_seconds < 0.0 || timeout_seconds > 120.0) {
    throw std::invalid_argument("invalid paged write completion wait");
  }
  std::unique_lock lock(completions_mutex_);
  completions_ready_.wait_for(
      lock, std::chrono::duration<double>(timeout_seconds),
      [this] { return !completions_.empty(); });
  std::vector<Completion> result(completions_.begin(), completions_.end());
  completions_.clear();
  return result;
}

std::vector<Completion> Arena::poll_read_completions() {
  std::lock_guard lock(completions_mutex_);
  std::vector<Completion> result(read_completions_.begin(), read_completions_.end());
  read_completions_.clear();
  return result;
}

std::vector<Completion> Arena::wait_read_completions(double timeout_seconds) {
  if (!std::isfinite(timeout_seconds) || timeout_seconds < 0.0 || timeout_seconds > 120.0) {
    throw std::invalid_argument("invalid paged read completion wait");
  }
  std::unique_lock lock(completions_mutex_);
  read_completions_ready_.wait_for(
      lock, std::chrono::duration<double>(timeout_seconds),
      [this] { return !read_completions_.empty(); });
  std::vector<Completion> result(read_completions_.begin(), read_completions_.end());
  read_completions_.clear();
  return result;
}

} // namespace mlx2::paged_kv
