"""Parent-indexed Flash-Next GDN with the served decode arithmetic.

Original composition of mlx2's existing fused decoder, not the dense Qwen
tree law. Each head threadgroup walks the tree once; recurrent predecessors
belong to that same group. Conv histories are gathered from input ancestry,
never from another threadgroup's writes. See provenance/flashnext-tree.json.
"""
from functools import lru_cache
import hashlib

import mlx.core as mx

from . import qwen4_fused_gdn as F
from .qwen_gdn_contracts import QWEN4_GDN_CONTRACT


_PREFIX = r"""
  constexpr size_t CD0 = 2 * HK * DK + HV * DV;
  constexpr size_t VD0 = HV * DV;
  constexpr size_t SD0 = HV * DV * DK;
  for (int node = 0; node < NODES; ++node) {
    const int parent = parents[node];
    const auto qn = qkv + node * CD0;
    const auto zn = z + node * VD0;
    const auto bn = b + node * HV;
    const auto an = a + node * HV;
    const auto cn = conv_histories + node * (K - 1) * CD0;
    const auto sn = parent < 0 ? recurrent_state : recurrent_states + parent * SD0;
    const auto on = output + node * VD0;
    const auto co = conv_states + node * (K - 1) * CD0;
    const auto so = recurrent_states + node * SD0;
    {
      const auto qkv = qn;
      const auto z = zn;
      const auto b = bn;
      const auto a = an;
      const auto conv_state = cn;
      const auto recurrent_state = sn;
      const auto output = on;
      const auto conv_state_out = co;
      const auto recurrent_state_out = so;
"""
_SUFFIX = r"""
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
"""

# Chain-parallel walk (opt-in): grid z is (chain, value head). Each
# threadgroup walks one root-to-leaf chain, so the serial depth is the chain
# length instead of the node count. Every node runs the same per-node body
# on the same inputs (its gathered conv history, its parent's state), so the
# outputs are bit-identical to the serial walk; nodes shared by several
# chains (the anchor) are written by each with identical values.
_CHAIN_PREFIX = r"""
  constexpr size_t CD0 = 2 * HK * DK + HV * DV;
  constexpr size_t VD0 = HV * DV;
  constexpr size_t SD0 = HV * DV * DK;
  const int chain = int(threadgroup_position_in_grid.z) / HV;
  for (int depth = 0; depth < MAXD; ++depth) {
    const int node = chains[chain * MAXD + depth];
    if (node < 0) break;
    const int parent = parents[node];
    const auto qn = qkv + node * CD0;
    const auto zn = z + node * VD0;
    const auto bn = b + node * HV;
    const auto an = a + node * HV;
    const auto cn = conv_histories + node * (K - 1) * CD0;
    const auto sn = parent < 0 ? recurrent_state : recurrent_states + parent * SD0;
    const auto on = output + node * VD0;
    const auto co = conv_states + node * (K - 1) * CD0;
    const auto so = recurrent_states + node * SD0;
    {
      const auto qkv = qn;
      const auto z = zn;
      const auto b = bn;
      const auto a = an;
      const auto conv_state = cn;
      const auto recurrent_state = sn;
      const auto output = on;
      const auto conv_state_out = co;
      const auto recurrent_state_out = so;
"""
_HEAD_SERIAL = "const uint hv = threadgroup_position_in_grid.z;"
_HEAD_CHAIN = "const uint hv = threadgroup_position_in_grid.z % HV;"


def ancestry(parents):
    from ..technologies.tree_verify import TreeTopology, legacy_topology

    return TreeTopology(parents, legacy_topology("flash_tree_ancestry")).paths


@lru_cache(maxsize=4)
def _chain_kernel(header, suffix):
    body = F._SOURCE
    if body.count(_HEAD_SERIAL) != 1:
        raise ValueError("served GDN decode body changed its head index; chain walk refused")
    source = _CHAIN_PREFIX + body.replace(_HEAD_SERIAL, _HEAD_CHAIN) + _SUFFIX
    digest = hashlib.sha256((header + source).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name="mlx2_qwen4_gdn_tree_chains_" + digest + suffix,
        input_names=["qkv", "z", "b", "a", "conv_histories", "conv_weight",
                     "A_log", "dt_bias", "recurrent_state", "norm_weight",
                     "norm_eps", "parents", "chains"],
        output_names=["output", "conv_states", "recurrent_states"],
        header=header, source=source, ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _kernel(header, suffix):
    source = _PREFIX + F._SOURCE + _SUFFIX
    digest = hashlib.sha256((header + source).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name="mlx2_qwen4_gdn_tree_" + digest + suffix,
        input_names=["qkv", "z", "b", "a", "conv_histories", "conv_weight",
                     "A_log", "dt_bias", "recurrent_state", "norm_weight",
                     "norm_eps", "parents"],
        output_names=["output", "conv_states", "recurrent_states"],
        header=header, source=source, ensure_row_contiguous=True,
    )


def tree(layer, qkv, z, b, a, cache, parents, *, chains=False):
    """Return all rows plus private conv/recurrent successors, one launch.

    ``chains=True`` walks each root-to-leaf chain in its own threadgroups
    (bit-identical outputs, serial depth = chain length)."""
    paths = ancestry(parents)
    if mx.default_device() != mx.gpu or not F.fused_gdn_runtime_supported():
        raise ValueError("Flash tree GDN requires Metal; no serial runtime fallback")
    if cache[1] is None or cache[1].dtype != mx.float32:
        raise ValueError("Flash tree GDN requires initialized fp32 recurrent state")
    admission = F.admit_qwen4_fused_gdn_decode(
        qkv=qkv[:, :1], z=z[:, :1], b=b[:, :1], a=a[:, :1],
        conv_state=cache[0], recurrent_state=cache[1], conv_weight=layer.conv1d.weight,
        A_log=layer.A_log, dt_bias=layer.dt_bias, norm_weight=layer.norm.weight,
        mask=None, spans=(), speculating=False, training=layer.training,
        sharded=layer.sharding_group is not None, num_key_heads=layer.num_k_heads,
        num_value_heads=layer.num_v_heads, key_head_dim=layer.head_k_dim,
        value_head_dim=layer.head_v_dim, conv_kernel=layer.conv_kernel_size,
        gate_activation=layer.norm.activation, contract=QWEN4_GDN_CONTRACT,
    )
    if not admission.accepted:
        raise ValueError("Flash tree GDN refused: " + admission.reason)
    ty = F.probe_qwen4_fused_gdn_decode(qkv.dtype, contract=QWEN4_GDN_CONTRACT)
    if ty is None:
        raise ValueError("Flash tree GDN served decode law declined")
    keep = layer.conv_kernel_size - 1
    rows = len(paths)
    # Incoming history excludes this node, and includes only its ancestors.
    indices = [list((tuple(range(keep)) + tuple(keep + j for j in path[:-1]))[-keep:])
               for path in paths]
    histories = mx.take(mx.concatenate([cache[0][0], qkv[0]], axis=0),
                        mx.array(indices, mx.int32), axis=0)
    header, suffix = F._served_decode_header()
    if chains:
        has_child = set(p for p in parents if p >= 0)
        leaves = [paths[row] for row in range(rows) if row not in has_child]
        maxd = max(len(path) for path in leaves)
        table = [list(path) + [-1] * (maxd - len(path)) for path in leaves]
        return tuple(_chain_kernel(header, suffix)(
            inputs=[qkv, z, b, a, histories, layer.conv1d.weight, layer.A_log,
                    layer.dt_bias, cache[1], layer.norm.weight, float(layer.norm.eps),
                    mx.array(parents, mx.int32), mx.array(table, mx.int32).reshape(-1)],
            template=[("T", qkv.dtype), ("HK", layer.num_k_heads), ("HV", layer.num_v_heads),
                      ("DK", layer.head_k_dim), ("DV", layer.head_v_dim),
                      ("K", layer.conv_kernel_size), ("TY", ty),
                      ("RATIO", layer.num_v_heads // layer.num_k_heads),
                      ("AGNES_NUMERICS", 0), ("NODES", rows), ("MAXD", maxd)],
            grid=(32, ty, layer.num_v_heads * len(leaves)), threadgroup=(32, ty, 1),
            output_shapes=[(1, rows, layer.num_v_heads * layer.head_v_dim),
                           (rows, keep, qkv.shape[-1]),
                           (rows, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim)],
            output_dtypes=[qkv.dtype, qkv.dtype, mx.float32],
        ))
    return tuple(_kernel(header, suffix)(
        inputs=[qkv, z, b, a, histories, layer.conv1d.weight, layer.A_log,
                layer.dt_bias, cache[1], layer.norm.weight, float(layer.norm.eps),
                mx.array(parents, mx.int32)],
        template=[("T", qkv.dtype), ("HK", layer.num_k_heads), ("HV", layer.num_v_heads),
                  ("DK", layer.head_k_dim), ("DV", layer.head_v_dim),
                  ("K", layer.conv_kernel_size), ("TY", ty),
                  ("RATIO", layer.num_v_heads // layer.num_k_heads),
                  ("AGNES_NUMERICS", 0), ("NODES", rows)],
        grid=(32, ty, layer.num_v_heads), threadgroup=(32, ty, 1),
        output_shapes=[(1, rows, layer.num_v_heads * layer.head_v_dim),
                       (rows, keep, qkv.shape[-1]),
                       (rows, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim)],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32],
    ))
