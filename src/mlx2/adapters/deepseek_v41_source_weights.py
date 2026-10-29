# SPDX-License-Identifier: MIT
"""Official FP4 experts and FP8 dense weights, preserving their original codes.

See provenance/deepseek-v41-source.json. Arithmetic is an unqualified MLX
reference profile; byte preservation does not imply CUDA reduction parity.
"""
from collections import OrderedDict
from contextlib import nullcontext
from functools import lru_cache
import hashlib
from pathlib import Path
import math
import re
import threading
import time
from types import SimpleNamespace

import numpy as np

from .deepseek_v41_mlx_weights import NativeExpertPlan, NativeExpertStore, NativeV41Weights, _quarantine_upload, _refuse_quarantined_uploads
from .deepseek_v41_source_artifact import checked_file, read_tensors
from .deepseek_v41_source_math import POLICIES
from ..runtime.io_ownership import ReadPool, storage_owner

REVISION = "mlx2.v41-official-mxfp4-fp8-act32-bf16.v1"
PROJECTIONS = {"gate": "w1", "up": "w3", "down": "w2"}

# Only these reviewed implementations can back the opt-in raw PP Q_b path.
Q_B_PREFILL_SOURCE_PINS = {
    'source_fp8_projection_tiles.py': '12e7617134e2dee24be28efd02d1117dee2dfacefa96c476ab916599b80a3c2b',
    'source_prefill_legacy_tiles.py': '004b4f1a8980661dce95f646329fc64c28b8aa7512ca864790b7622b4fa4bb16',
    'source_dense.py': '11fe30e8f0a58f966fba3fb6c555d88de30f6cd6190a34e1f809137029b35f65',
    'source_dense_legacy.py': '564e7e17bb8c1641f5a26ec1329968de7fad895ae83840793b226909bb023a0d',
}


@lru_cache(maxsize=1)
def _verify_q_b_prefill_source():
    from ..runtime.gguf import source_fp8_projection_tiles as tiles
    from ..runtime.gguf.source_prefill_legacy_tiles import _installed
    root = Path(tiles.__file__).resolve().parent
    for name, expected in Q_B_PREFILL_SOURCE_PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source Q_b prefill implementation changed; re-audit required')
    _installed()  # Pins the installed MLX binary and all transitive headers.



NAX_PREFILL_SOURCE_PINS = {
    'source_prefill_nax_profile.py': 'f01a8095b15aa804bdd0bed88578139599a7ded2a988a09839935ca1bb44779b',
    'source_prefill_nax.py': '29c1f7ec6f21228d180d42b02111e3385a8ee41b18da073dcc15e6bf680a3416',
    'source_prefill_legacy_tiles.py': '004b4f1a8980661dce95f646329fc64c28b8aa7512ca864790b7622b4fa4bb16',
}


@lru_cache(maxsize=1)
def _verify_nax_prefill_source():
    from ..runtime.gguf import source_prefill_nax_profile as native
    root=Path(native.__file__).resolve().parent
    for name,expected in NAX_PREFILL_SOURCE_PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Source native NAX prefill implementation changed; re-audit required')
    native.verify_sources()


def expert_names(layer, expert):
    return tuple(f"layers.{layer}.ffn.experts.{expert}.{part}.{field}"
                 for part in PROJECTIONS.values() for field in ("weight", "scale"))


def weight_plan(artifact, budget):
    c, tensors = artifact["args"], artifact["tensors"]
    largest = 0
    for layer in range(c.num_hidden_layers):
        for expert in range(c.n_routed_experts):
            names = expert_names(layer, expert)
            for part in PROJECTIONS.values():
                n = f"layers.{layer}.ffn.experts.{expert}.{part}."
                shape = (c.hidden_size, c.moe_intermediate_size) if part == "w2" else (c.moe_intermediate_size, c.hidden_size)
                w, s = tensors[n+"weight"], tensors[n+"scale"]
                if (w["dtype"] != "I8" or w["shape"] != (shape[0], shape[1]//2)
                        or s["dtype"] != "F8_E8M0" or s["shape"] != (shape[0], shape[1]//32)):
                    raise ValueError("official FP4 expert geometry mismatch")
            largest = max(largest, sum(tensors[n]["size"] for n in names))
    if type(budget) is not int or budget < largest:
        raise ValueError("official expert cache must hold a complete expert")
    dense = artifact["inventory"]["storage_bytes"]["dense"]
    # Expansion is limited to FP8 block-scale replication, BF16 grouped wo_a,
    # and the reference's FP32 head. No expert is dequantized or requantized.
    prepared = 0
    for alias, source in artifact["aliases"].items():
        t = tensors[source]
        if alias.endswith("attn_output_a.weight"):
            prepared += 2*math.prod(t["shape"])
        elif t["dtype"] == "F8_E4M3":
            prepared += math.prod(t["shape"])//32
        elif alias == "output.weight":
            prepared += 4*math.prod(t["shape"])
    staging = max(tensors[n]["size"] for n in artifact["aliases"].values()) + (16 << 20)
    active = largest*c.n_routed_experts
    return dict(dense_payload_bytes=dense, dense_prepared_bound_bytes=prepared,
                dense_staging_bound_bytes=staging, expert_cache_bytes=budget,
                minimum_expert_cache_bytes=largest, expert_active_bound_bytes=active,
                expert_staging_bound_bytes=largest+(16 << 20),
                expert_table_bytes=artifact["inventory"]["storage_bytes"]["expert"],
                manifest_binding_sha256=artifact["identity"]["fingerprint"], metadata_bound_bytes=256 << 20,
                total_reservation_bytes=dense+prepared+staging+budget+active+largest+(16 << 20))


class SourceExpertStore(NativeExpertStore):
    def __init__(self, artifact, bounds, *, read_workers=2):
        _refuse_quarantined_uploads()
        self.artifact, self.root, self.bounds = artifact, artifact["root"], bounds
        c = artifact["args"]
        self.tables = {f"blk.{i}": tuple((i, e) for e in range(c.n_routed_experts)) for i in range(c.num_hidden_layers)}
        self.files = {key: {"size": bounds["minimum_expert_cache_bytes"]} for keys in self.tables.values() for key in keys}
        self.cache_budget_bytes = self.maximum_cache_bytes = bounds["expert_cache_bytes"]
        self.minimum_cache_bytes = bounds["minimum_expert_cache_bytes"]
        self._cache, self._leased = OrderedDict(), {}
        self._pending_uploads, self._pending_host = {}, {}
        self._cache_bytes = 0
        self.ple_tables, self._ple_bound = {}, False
        self._active = self._closed = self._parked = False
        self._owner = self._error = self._synchronize = None
        self._lock = threading.RLock()
        self.stats = dict(cache_hits=0, cache_misses=0, bytes_read=0, evictions=0,
                          uploaded_experts=0, upload_batches=0, demand_wait_ns=0,
                          lookups=0, windows=0, parks=0, activations=0)
        self.binding = bounds["manifest_binding_sha256"]
        self.plan = NativeExpertPlan(layers=c.num_hidden_layers, experts_per_layer=c.n_routed_experts,
            expert_bytes=self.minimum_cache_bytes, table_bytes=bounds["expert_table_bytes"],
            capacity_experts=min(c.n_routed_experts, self.cache_budget_bytes//(c.num_hidden_layers*self.minimum_cache_bytes)),
            ceiling_bytes=self.cache_budget_bytes, projections=tuple(PROJECTIONS), load_phase="pre_materialization",
            top_k=c.num_experts_per_tok, transient_bound_bytes=bounds["expert_active_bound_bytes"]+bounds["expert_staging_bound_bytes"],
            staging_window=1, manifest_sha256=self.binding, table_kinds=("official_E2M1_E8M0_group32",), page_cache="bypass")
        self.pool = ReadPool(min(8, read_workers), "v41-source-experts", timeout=60, max_pending=1,
                             owner=storage_owner("v41-source-experts", self.root))

    def _adaptive_victim(self):
        # Packed bank views share whole backing allocations. A logical expert
        # eviction cannot reclaim one sibling while that backing is pinned.
        protected=set(self._leased)|set(self._pending_uploads)|set(getattr(self,'_resident_bank_pinned_keys',()))
        return next((key for key in self._cache if key not in protected),None)

    def _adaptive_evict_key(self,key):
        if key in getattr(self,'_resident_bank_pinned_keys',()):
            raise ValueError('Cannot individually evict an attached Source bank allocation')
        return super()._adaptive_evict_key(key)

    def acquire(self, prefix, expert):
        import mlx.core as mx
        with self._lock:
            self._ready()
            if not self._active or prefix not in self.tables or type(expert) is not int or not 0 <= expert < len(self.tables[prefix]):
                raise ValueError("official expert acquisition outside admitted window")
            key = self.tables[prefix][expert]
            names = expert_names(*key)
            for name in names:
                shard = self.artifact["tensors"][name]["file"]
                checked_file(self.root, shard, self.artifact["seals"][shard])
            if key in self._leased:
                return self._leased[key]
            if self.adaptive_residency_enabled:
                self.adaptive_residency_update(incoming_bytes=0 if key in self._cache else self.minimum_cache_bytes)
            if (sum(sum(v.nbytes for v in item.values()) for item in self._leased.values())
                    + self.minimum_cache_bytes > self.bounds["expert_active_bound_bytes"]):
                raise ValueError("official expert leases exceed one-layer bound")
            if key in self._cache:
                arrays = self._cache[key]
                self._cache.move_to_end(key)
                self.stats["cache_hits"] += 1
            else:
                started = time.perf_counter_ns()
                try:
                    raw = self.pool.call(read_tensors, self.artifact, names)
                finally:
                    self.stats["demand_wait_ns"] += time.perf_counter_ns()-started
                self._pending_host[key] = raw
                arrays = self._pending_uploads[key] = {}
                for kind, part in PROJECTIONS.items():
                    name = f"layers.{key[0]}.ffn.experts.{key[1]}.{part}."
                    arrays[kind+".qweight"] = mx.array(raw[name+"weight"].view("<u4"))
                    arrays[kind+".scales"] = mx.array(raw[name+"scale"])
                mx.eval(arrays)
                self._pending_host.pop(key)
                size = sum(v.nbytes for v in arrays.values())
                if size != self.minimum_cache_bytes or size > self.cache_budget_bytes:
                    raise ValueError("official expert upload exceeded admitted bytes")
                self.stats["cache_misses"] += 1
                self.stats["bytes_read"] += size
                self.stats["uploaded_experts"] += 1
                self.stats["upload_batches"] += 1
                self._cache[key] = arrays
                self._cache_bytes += size
            self._leased[key] = arrays
            self._pending_uploads.pop(key, None)
            if self.adaptive_residency_enabled:
                self.adaptive_residency_update()
            else:
                while self._cache_bytes > self.cache_budget_bytes:
                    victim=self._adaptive_victim()
                    if victim is None:
                        raise MemoryError('Source expert cache maximum exhausted by active leases')
                    self._adaptive_evict_key(victim)
            return arrays

    def apc_revision(self):
        return REVISION

    def receipt(self):
        value = super().receipt()
        value.update(mechanism_revision=REVISION, numeric_profile="official_mxfp4_fp8",
                     source_read_byte_scope="six exact tensor extents per expert; original packed payload and scales",
                     host_dequantization=False, expert_requantization=False)
        return value


class SourceV41Weights(NativeV41Weights):
    compute_policy = REVISION
    dense_prefill_policy = shared_policy = dense_epilogue_policy = dense_pair_policy = "reference"
    expert_math_policy = "reference"

    def __init__(self, artifact, *, expert_cache_bytes, expert_read_workers=2, expert_policy=None, admitted_bounds=None,
                 expert_math_policy="reference", dense_qat_policy="reference",
                 bf16_loads_policy="reference", completion_policy="reference", prefill_batch_policy="reference",
                 prefill_wide_dtype="reference", prefill_wide_math="reference", attention_math_policy="reference",
                 qakv_policy="reference", q_b_prefill_policy="reference", fp8_pp_policy="reference", woa_pp_policy="reference",
                 ffn_input_policy="reference", prefill_down_policy="reference", native_mx_policy="reference",
                 source_frame_reuse_policy="reference", submission_policy="reference",
                 source_row_packet_policy="reference", resident_gather_policy="reference", resident_prefill_policy="reference"):
        if expert_math_policy not in POLICIES:
            raise ValueError("invalid Source expert math policy")
        self.expert_math_policy = expert_math_policy
        if resident_gather_policy not in ('reference','resident_bank_v1'):
            raise ValueError('Invalid Source resident gather policy')
        if resident_gather_policy!='reference' and (expert_math_policy!='routed_fused_fp32' or ffn_input_policy!='reference'):
            raise ValueError('Source resident bank requires routed_fused_fp32 and reference FFN input')
        if resident_prefill_policy not in ('reference','expert_major_v1','expert_major_wide_v1'):
            raise ValueError('Invalid Source resident prefill policy')
        if resident_prefill_policy!='reference' and resident_gather_policy!='resident_bank_v1':
            raise ValueError('Source resident prefill requires resident_bank_v1')
        self.source_resident_prefill_policy=resident_prefill_policy
        self.source_resident_gather_policy=resident_gather_policy
        self.source_resident_bank_session=None
        self._resident_dense_coefficient_views={}
        if type(submission_policy) is not str or submission_policy not in ('reference','shared_first_v1'):
            raise ValueError('invalid Source submission policy')
        self.source_submission_policy = submission_policy
        if submission_policy=='shared_first_v1' and ffn_input_policy!='reference':
            raise ValueError('Source shared-first requires reference FFN input; routed/shared reuse consumes routed first')
        if type(source_row_packet_policy) is not str or source_row_packet_policy not in ('reference','row_packet_v1'):
            raise ValueError('invalid Source row packet policy')
        if source_row_packet_policy != 'reference' and source_frame_reuse_policy != 'prefill_v1':
            raise ValueError('Source row packet requires authoritative prefill_v1 frame reuse')
        self.source_row_packet_policy=source_row_packet_policy
        self._source_frame_serial=0
        self._source_current_frame=None
        if type(source_frame_reuse_policy) is not str or source_frame_reuse_policy not in ('reference','prefill_v1'):
            raise ValueError('invalid Source frame reuse policy')
        if source_frame_reuse_policy!='reference' and (expert_policy is None
                or expert_math_policy!='routed_fused_fp32' or prefill_batch_policy not in ('gate_up','gate_up_down')):
            raise ValueError('Source frame reuse requires routed batched prefill and LFU')
        self.source_frame_reuse_policy=source_frame_reuse_policy
        from .deepseek_v41_source_native_mx import validate_policy as validate_native_mx
        validate_native_mx(native_mx_policy,expert_math=expert_math_policy,prefill_batch=prefill_batch_policy,
            wide_math=prefill_wide_math,wide_dtype=prefill_wide_dtype,prefill_down=prefill_down_policy,ffn_input=ffn_input_policy)
        if native_mx_policy!='reference':
            from .deepseek_v41_source_native_mx import configuration_supported
            if expert_policy is None or not configuration_supported(artifact.get('args')):
                raise ValueError('Source native MX requires canonical B1 Source and bounded packed LFU')
        from .deepseek_v41_source_prefill_down import validate_policy
        validate_policy(prefill_down_policy, expert_math=expert_math_policy, prefill_batch=prefill_batch_policy)
        if prefill_down_policy != 'reference':
            from .deepseek_v41_source_prefill_down import configuration_supported
            if (expert_policy is None or expert_policy.get('mode') != 'hot_original_q1'
                    or not configuration_supported(artifact.get('args'))):
                raise ValueError('Source Q1 prefill DOWN requires canonical BF16 Source hot-original/Q1 profile')
        for label, value, choices in (
            ('FFN input', ffn_input_policy, ('reference','shared_routed_qat_reuse')),
            ('WoA PP', woa_pp_policy, ('reference','native_bf16_grouped_pp_v1')),
            ('FP8 PP', fp8_pp_policy, ('reference','native_bf16_qb_wob_v1')),
            ('Q_b prefill', q_b_prefill_policy, ('reference','tile64x32_raw')),
            ('Qa/KV', qakv_policy, ('reference','qat_reuse')),
            ('attention math', attention_math_policy, ('reference','all_retained_broadcast','omlx_nax_bf16_online_v1',
                'source_short_decode_joint_f32_v1','source_short_decode_grouped_f32_v1')),
            ('dense QAT', dense_qat_policy, ('reference','fused_fp32')),
            ('BF16 loads', bf16_loads_policy, ('reference','woa_f32','direct_f32')),
            ('completion', completion_policy, ('reference','bounded_2')),
            ('prefill batch', prefill_batch_policy, ('reference','gate_up','gate_up_down')),
            ('prefill wide dtype', prefill_wide_dtype, ('reference','bf16_p1')),
            ('prefill wide math', prefill_wide_math, ('reference','legacy_tiles','legacy_tiles_16x64','legacy_tiles_fused','legacy_tiles_fused_16x64','nax_k16_ordered_f32_v1')),
        ):
            if type(value) is not str or value not in choices:
                raise ValueError('invalid Source '+label+' policy')
        if qakv_policy != 'reference' and dense_qat_policy != 'fused_fp32':
            raise ValueError('Source Qa/KV reuse requires fused_fp32 Source dense QAT')
        if fp8_pp_policy!='reference' and q_b_prefill_policy!='reference':
            raise ValueError('Source native FP8 PP cannot combine with Q_b prefill tile')
        if ffn_input_policy!='reference':
            from .deepseek_v41_source_ffn_input import configuration_supported
            if dense_qat_policy!='fused_fp32' or expert_math_policy!='routed_fused_fp32':
                raise ValueError('Source FFN input reuse requires fused dense QAT and routed_fused_fp32')
            if not configuration_supported(artifact.get('args')):
                raise ValueError('Source FFN input reuse requires canonical BF16 Source configuration')
        if completion_policy != 'reference' and expert_policy is None:
            raise ValueError('bounded Source completion requires a byte-bounded LFU provider')
        if prefill_batch_policy!='reference' and (expert_policy is None or expert_math_policy!='routed_fused_fp32'):
            raise ValueError('Source prefill batching requires routed math and a byte-bounded LFU provider')
        if prefill_wide_math!='reference':
            if expert_policy is None or expert_math_policy!='routed_fused_fp32' or prefill_batch_policy not in ('gate_up','gate_up_down'):
                raise ValueError('Source wide tiles require routed batched prefill and LFU')
            if prefill_wide_dtype!='reference':
                raise ValueError('Source wide tiles cannot combine with bf16_p1')
        self.source_dense_qat_policy = dense_qat_policy
        self.source_qakv_policy = qakv_policy
        self.source_q_b_prefill_policy = q_b_prefill_policy
        self.source_fp8_pp_policy = fp8_pp_policy
        self.source_woa_pp_policy = woa_pp_policy
        self.source_ffn_input_policy = ffn_input_policy
        self.source_prefill_down_policy = prefill_down_policy
        self.source_native_mx_policy = native_mx_policy
        self._source_native_mx_attestations = None
        if woa_pp_policy!='reference':
            from .deepseek_v41_source_woa_pp import configuration_supported,implementation_identity
            if not configuration_supported(artifact.get('args')):
                raise ValueError('Source native WoA PP requires original BF16 producer geometry')
            # Model geometry binds this instance's numerical program even when
            # constructed directly, outside the adapter's profile wrapper.
            self.compute_policy=REVISION+':source_woa_pp='+woa_pp_policy+':'+implementation_identity()
        if ffn_input_policy!='reference':
            from .deepseek_v41_source_ffn_input import implementation_identity
            self.compute_policy=self.compute_policy+':source_ffn_input='+ffn_input_policy+':'+implementation_identity()
        if prefill_down_policy != 'reference':
            from .deepseek_v41_source_prefill_down import implementation_identity,tiny_implementation_identity
            identity=tiny_implementation_identity()if prefill_down_policy=='q1_tiny_batch_v1'else implementation_identity()
            self.compute_policy=self.compute_policy+':source_prefill_down='+prefill_down_policy+':'+identity
        if native_mx_policy!='reference':
            from .deepseek_v41_source_native_mx import implementation_identity
            self.compute_policy=self.compute_policy+':source_native_mx='+native_mx_policy+':'+implementation_identity()
        if source_frame_reuse_policy!='reference':
            self.compute_policy=self.compute_policy+':source_frame_reuse='+source_frame_reuse_policy+':route_plan_v1'
        if submission_policy != 'reference':
            from .deepseek_v41_source_submission import implementation_identity
            self.compute_policy=self.compute_policy+':source_submission='+submission_policy+':'+implementation_identity()
        if source_row_packet_policy != 'reference':
            from .deepseek_v41_source_row_packet import implementation_identity
            self.compute_policy=self.compute_policy+':source_row_packet='+source_row_packet_policy+':'+implementation_identity()
        self.dense_pair_policy = 'fused' if qakv_policy == 'qat_reuse' else 'reference'
        self.source_attention_math_policy = attention_math_policy
        self.source_bf16_loads_policy = bf16_loads_policy
        self.source_completion_policy = completion_policy
        self.source_prefill_batch_policy = prefill_batch_policy
        self.source_prefill_wide_dtype = prefill_wide_dtype
        self.source_prefill_wide_math = prefill_wide_math
        self._source_completion = None
        self.artifact, self.root, self.metadata = artifact, artifact["root"], {}
        self.bounds = weight_plan(artifact, expert_cache_bytes)
        if admitted_bounds is not None:
            self.bounds = admitted_bounds
        self.tensors = {}
        for alias, name in artifact["aliases"].items():
            row = artifact["tensors"][name]
            self.tensors[alias] = SimpleNamespace(name=alias, numpy_shape=row["shape"], shape=row["shape"][::-1],
                ggml_type=8 if row["dtype"] == "F8_E4M3" and not alias.endswith("attn_output_a.weight") else 0,
                storage={"kind": "source", "key": alias})
        self._dense, self._quant, self._packed = {}, {}, {}
        self._pending_dense, self._pending_dense_host = {}, None
        self._loaded = self._parked = self._closed = False
        self._error = None
        self.scheduled_expert_calls, self.scheduled_dense_calls, self.scheduled_grouped_calls = {}, {}, {}
        self.scheduled_expert_buffer_operations = self.expert_buffer_counters = {}
        self._read_pool = ReadPool(1, "v41-source-dense", timeout=60, max_pending=1,
                                  owner=storage_owner("v41-source-dense", self.root))
        self.expert_stream = None
        try:
            if expert_policy is None:
                self.expert_stream = SourceExpertStore(artifact, self.bounds, read_workers=expert_read_workers)
            else:
                from .deepseek_v41_source_lfu import LFUSourceExpertStore
                self.expert_stream = LFUSourceExpertStore(artifact, self.bounds, expert_policy, read_workers=expert_read_workers,
                    packed_buffers=expert_math_policy in ('epilogue_fused_fp32','routed_fused_fp32'))
            if native_mx_policy!='reference':
                self.enable_source_native_mx_attestation()
        except BaseException as error:
            if self.expert_stream is not None:
                try:self.expert_stream.close()
                except BaseException as cleanup:error.add_note('Source expert cleanup: '+type(cleanup).__name__)
            try:self._read_pool.close()
            except BaseException as cleanup:error.add_note('Source dense reader cleanup: '+type(cleanup).__name__)
            raise

    def enable_source_native_mx_attestation(self):
        """Explicit native/shadow opt-in before loads; never inspect cached GPU weights."""
        from .deepseek_v41_source_native_mx import Attestations,configuration_supported
        stream=self.expert_stream
        if (stream is None or not getattr(stream,'packed_buffers',False)
                or not configuration_supported(self.artifact.get('args'))):
            raise ValueError('Source native MX attestation requires canonical packed LFU Source')
        registry=getattr(self,'_source_native_mx_attestations',None)
        if registry is None:
            if getattr(stream,'_packed_original_loaded',None) is not None:
                raise ValueError('Source original-load observer already owned')
            if any(stream.kind(key)=='original' for key in stream._cache):
                raise ValueError('Source native MX attestation must be enabled before original experts are cached; no GPU readback fallback')
            registry=Attestations(self.artifact,stream)
            self._source_native_mx_attestations=registry
            stream._packed_original_loaded=registry.loaded
        elif getattr(getattr(stream,'_packed_original_loaded',None),'__self__',None) is not registry:
            raise ValueError('Source native MX original-load observer changed')
        return registry.receipt()

    def _source_native_mx_bound(self,prefix,expert,data):
        registry=getattr(self,'_source_native_mx_attestations',None)
        return None if registry is None else registry.get(prefix,expert,data)

    def _load_dense(self):
        import mlx.core as mx
        try:
            for alias, name in self.artifact["aliases"].items():
                row = self.artifact["tensors"][name]
                scale = name.removesuffix("weight")+"scale"
                names = (name, scale) if row["dtype"] == "F8_E4M3" else (name,)
                raw = self._read_pool.call(read_tensors, self.artifact, names)
                self._pending_dense_host = raw
                if row["dtype"] == "F8_E4M3":
                    if self.artifact["tensors"][scale]["shape"] != ((row["shape"][0]+31)//32, row["shape"][1]//32):
                        raise ValueError("official FP8 block-scale geometry mismatch")
                    q = mx.array(raw[name].view("<u4"))
                    scales = mx.repeat(mx.array(raw[scale]), 32, axis=0)[:row["shape"][0]]
                    self._pending_dense.update({alias+".q": q, alias+".s": scales})
                    if alias.endswith("attn_output_a.weight"):
                        # Official convert.py expands this grouped projection to BF16.
                        value = mx.dequantize(q, scales=scales, mode="mxfp8", dtype=mx.bfloat16)
                        self._pending_dense[alias] = value
                        mx.eval(self._pending_dense)
                        self._pending_dense.pop(alias+".q")
                        self._pending_dense.pop(alias+".s")
                    else:
                        self._quant[alias] = (q, scales)
                else:
                    value = mx.array(raw[name])
                    if row["dtype"] == "BF16":
                        value = value.view(mx.bfloat16)
                    if alias == "output.weight" and self.source_bf16_loads_policy in ('reference','woa_f32'):
                        value = value.astype(mx.float32)
                    self._pending_dense[alias] = value
                mx.eval(self._pending_dense)
                self._dense.update(self._pending_dense)
                self._pending_dense.clear()
                self._pending_dense_host = None
            self._loaded = True
        except BaseException as error:
            self._error = str(error)
            try:
                mx.synchronize()
            except BaseException:
                _quarantine_upload(self)
            raise

    def verify_source(self):
        for name, seal in self.artifact["seals"].items():
            checked_file(self.root, name, seal)
        return True

    def array(self, name):
        self._ensure_dense()
        return self._dense[name]

    def linear(self, name):
        self._ensure_dense()
        import mlx.core as mx
        def apply(x):
            self._ready()
            if name in self._quant:
                q, scales = self._quant[name]
                prepared = self._dense_input(x)
                if getattr(self,'source_fp8_pp_policy','reference')=='native_bf16_qb_wob_v1':
                    from .deepseek_v41_source_fp8_pp import maybe_project
                    native=maybe_project(name,x,prepared,q,scales,counters=self.scheduled_dense_calls)
                    if native is not None:return native
                candidate = self._q_b_prefill_project(name, x, prepared, q, scales)
                if candidate is not None:
                    return candidate
                self.scheduled_dense_calls["official_mxfp8_act32"] = self.scheduled_dense_calls.get("official_mxfp8_act32", 0)+1
                return mx.quantized_matmul(prepared, q, scales=scales, mode="mxfp8")
            if name == 'output.weight' and getattr(self,'source_bf16_loads_policy','reference') == 'direct_f32':
                from ..runtime.gguf.source_dense import maybe_project
                value=maybe_project(x.astype(mx.float32),self.array(name),'head') if x.ndim==3 and x.shape[1]==1 else None
                if value is not None:
                    key='source_bf16_head_direct_f32'
                    self.scheduled_dense_calls[key]=self.scheduled_dense_calls.get(key,0)+1
                    return value
                key='source_bf16_head_fallback'
                self.scheduled_dense_calls[key]=self.scheduled_dense_calls.get(key,0)+1
            return x.astype(mx.float32) @ self.array(name).astype(mx.float32).T
        return apply

    def _q_b_prefill_project(self, name, x, prepared, q, scales):
        """Raw F32 only; the ordinary model remains the BF16 boundary owner."""
        if getattr(self, 'source_q_b_prefill_policy', 'reference') != 'tile64x32_raw':
            return None
        match = re.fullmatch(r'blk\.(0|[1-9][0-9]?)\.attn_q_b\.weight', name)
        if match is None or int(match[1]) >= 40:
            return None
        import mlx.core as mx
        if (not isinstance(x,mx.array) or x.dtype != mx.float32 or x.shape != (1,99,1280)
                or not isinstance(q,mx.array) or q.dtype != mx.uint32 or q.shape != (32768,320)
                or not isinstance(scales,mx.array) or scales.dtype != mx.uint8 or scales.shape != (32768,40)):
            key = 'source_q_b_prefill_fallback_shape'
            self.scheduled_dense_calls[key] = self.scheduled_dense_calls.get(key,0)+1
            return None
        from ..runtime.gguf.source_dense_legacy import _device_supported
        if not _device_supported():
            key = 'source_q_b_prefill_fallback_device'
            self.scheduled_dense_calls[key] = self.scheduled_dense_calls.get(key,0)+1
            return None
        _verify_q_b_prefill_source()
        from ..runtime.gguf.source_fp8_projection_tiles import project
        # Shape/reduction law is pinned; neither input nor output is BF16 here.
        result = project(prepared,q,scales,'q_b',tile=(64,32),fused_boundary=False)
        key = 'source_q_b_prefill_tile64x32_raw'
        self.scheduled_dense_calls[key] = self.scheduled_dense_calls.get(key,0)+1
        return result

    def _dense_input(self,x):
        import mlx.core as mx
        from ..runtime.models.deepseek_v41_quant import quantize, fused_quantize
        fused=getattr(self,'source_dense_qat_policy','reference') == 'fused_fp32'
        key='source_dense_qat_fused' if fused else 'source_dense_qat_reference'
        self.scheduled_dense_calls[key]=self.scheduled_dense_calls.get(key,0)+1
        return (fused_quantize(x.astype(mx.float32),'fp8') if fused else quantize(x,'fp8')).astype(mx.float32)

    def ffn_input_token(self,x,prefix):
        if getattr(self,'source_ffn_input_policy','reference')=='reference':return None
        from .deepseek_v41_source_ffn_input import create
        return create(self,x,prefix)

    def gate_up(self,x,gate_name,up_name,*,ffn_input_token=None):
        if ffn_input_token is not None:
            from .deepseek_v41_source_ffn_input import FfnInputToken
            if not isinstance(ffn_input_token,FfnInputToken):
                raise ValueError('Source FFN input requires an explicit frame token')
            prepared=ffn_input_token.shared(self,x,gate_name,up_name)
        else:
            prepared=None
        if (getattr(self,'source_dense_qat_policy','reference') != 'fused_fp32'
                or gate_name not in self._quant or up_name not in self._quant):
            if ffn_input_token is not None:
                raise ValueError('Source FFN input shared projection capability changed')
            return super().gate_up(x,gate_name,up_name)
        import mlx.core as mx
        self._ensure_dense()
        if prepared is None:prepared=self._dense_input(x)
        outputs=[]
        for name in (gate_name,up_name):
            q,scales=self._quant[name]
            self.scheduled_dense_calls['official_mxfp8_act32']=self.scheduled_dense_calls.get('official_mxfp8_act32',0)+1
            outputs.append(mx.quantized_matmul(prepared,q,scales=scales,mode='mxfp8'))
        key='source_dense_gate_up_qat_reused'
        self.scheduled_dense_calls[key]=self.scheduled_dense_calls.get(key,0)+1
        return tuple(outputs)

    def can_fuse_dense(self, x, *names):
        """Only original Qa/local-KV QAT reuse; no bank or GEMM fusion."""
        if (getattr(self,'source_qakv_policy','reference') != 'qat_reuse'
                or getattr(self,'source_dense_qat_policy','reference') != 'fused_fp32'
                or len(names) != 2 or not all(type(n) is str for n in names)):
            return False
        match = re.fullmatch(r'blk\.(0|[1-9][0-9]?)\.attn_q_a\.weight',names[0])
        if match is None or int(match[1]) >= 40 or names[1] != f'blk.{match[1]}.attn_kv.weight':
            return False
        import mlx.core as mx
        if (not isinstance(x,mx.array) or x.dtype != mx.float32 or x.ndim != 3
                or x.shape[0] != 1 or not 1 <= x.shape[1] <= 1024 or x.shape[2] != 5120):
            return False
        self._ensure_dense()
        for name,outputs in zip(names,(1280,512)):
            params = self._quant.get(name)
            if not isinstance(params,tuple) or len(params) != 2:
                return False
            q,scales = params
            if (not isinstance(q,mx.array) or not isinstance(scales,mx.array)
                    or q.dtype != mx.uint32 or q.shape != (outputs,1280)
                    or scales.dtype != mx.uint8 or scales.shape != (outputs,160)):
                return False
        return True

    def paired_project(self, x, a, b, *, rounded=True):
        """Prepare Qa/KV QAT once; stock products and BF16 stores stay separate."""
        if type(rounded) is not bool or not self.can_fuse_dense(x,a,b):
            raise ValueError('unsupported Source Qa/KV QAT-reuse pair')
        import mlx.core as mx
        self._ready()
        prepared = self._dense_input(x)
        outputs = []
        for name in (a,b):
            q,scales = self._quant[name]
            key = 'official_mxfp8_act32'
            self.scheduled_dense_calls[key] = self.scheduled_dense_calls.get(key,0)+1
            raw = mx.quantized_matmul(prepared,q,scales=scales,mode='mxfp8')
            outputs.append(raw.astype(mx.bfloat16).astype(mx.float32) if rounded else raw.astype(mx.float32))
        key = 'source_qakv_qat_reuse_pairs'
        self.scheduled_dense_calls[key] = self.scheduled_dense_calls.get(key,0)+1
        key = 'source_qakv_input_qat_calls_avoided'
        self.scheduled_dense_calls[key] = self.scheduled_dense_calls.get(key,0)+1
        return tuple(outputs)

    def recommended_prefill_rows(self, maximum, *, context_tokens=0):
        """Live-memory chunk advice; cache maxima are never RAM reservations."""
        if type(maximum) is not int or maximum < 1 or type(context_tokens) is not int or context_tokens < 0:
            raise ValueError('Source prefill advice requires positive rows and nonnegative context')
        maximum=min(maximum,1024)
        stream=getattr(self,'expert_stream',None)
        if stream is None or not getattr(stream,'adaptive_residency_enabled',False):
            return maximum
        from .deepseek_v41_source_memory import SourceMemorySnapshot,SourceMemoryNeed,plan_source_memory,ORIGINAL_EXPERT_BYTES
        try:
            callback=getattr(stream,'_adaptive_available',None)
            if callback is not None:
                available=callback();physical=None
            else:
                from ..runtime.os_memory import host_memory_snapshot
                host=host_memory_snapshot()
                available=None if host is None else host.available_bytes
                physical=None if host is None else host.physical_bytes
            if type(available)is not int or available<0:available=None
            resident=max(0,int(getattr(stream,'_cache_bytes',0)))
            pinned=min(resident,sum(int(getattr(value,'allocation_nbytes',0)) or
                sum(int(getattr(v,'nbytes',0))for v in value.values())
                for value in getattr(stream,'_leased',{}).values()))
            args=self.artifact['args']
            dim=getattr(args,'head_dim',512)
            # Incremental pool/ring growth only, never a whole-context reserve.
            growth=maximum*dim*4*max(1,len(getattr(args,'kv_source_layer_ids',(2,8,14,20))))
            snapshot=SourceMemorySnapshot(physical,available,resident,pinned)
            need=SourceMemoryNeed(cache_ceiling_bytes=int(getattr(stream,'maximum_cache_bytes',75_000_000_000)),
                expert_payload_bytes=(ORIGINAL_EXPERT_BYTES,)*getattr(args,'num_experts_per_tok',6),
                expert_staging_bytes=int(getattr(stream,'_adaptive_staging',ORIGINAL_EXPERT_BYTES)),
                kv_growth_bytes=growth,phase='prefill',requested_prefill_rows=maximum,
                route_working_set_known=False)
            advice=plan_source_memory(snapshot,need)
            self.source_memory_advice=advice
            return 1 if available is None or advice['pressure_attempt'] else max(1,min(maximum,advice['prefill_chunk_rows']))
        except Exception:
            return 1

    def park(self):
        session=getattr(self,'source_resident_bank_session',None)
        if session is not None:session.close()
        result=super().park()
        getattr(self,'_resident_dense_coefficient_views',{}).clear()
        return result

    def close(self):
        # An uncertain bank epoch must retain the entire provider. Never let
        # native teardown clear cache/accounting before the bank cohort retires.
        session=getattr(self,'source_resident_bank_session',None)
        if session is not None:session.close()
        result=super().close()
        getattr(self,'_resident_dense_coefficient_views',{}).clear()
        return result

    def prepare_resident_coefficient_views(self,*,ownercheck):
        """Prepare immutable WoA alias capabilities once, outside token work."""
        if not callable(ownercheck)or ownercheck()is False:raise ValueError('Source dense view preparation requires ownership')
        session=getattr(self,'source_resident_bank_session',None)
        completion=getattr(self,'_source_completion',None)
        if (session is None or session.active or session.replaying or session.poisoned
                or getattr(completion,'active',False)or getattr(self.expert_stream,'_active',False)):
            raise ValueError('Source dense coefficient views require an idle prepared bank session')
        if self.source_bf16_loads_policy not in ('woa_f32','direct_f32'):
            raise ValueError('Source dense coefficient views require an admitted BF16 WoA consumer')
        self._ensure_dense()
        from ..runtime.gguf.source_dense import prepare_coefficient_view
        views=getattr(self,'_resident_dense_coefficient_views',None)
        if views is None:self._resident_dense_coefficient_views=views={}
        for name,weight in self._dense.items():
            if not name.endswith('attn_output_a.weight'):continue
            if name in views:views[name].operand(weight,'wo_a')
            else:views[name]=prepare_coefficient_view(weight,'wo_a',ownercheck=ownercheck)
        if not views:raise ValueError('Source dense coefficient views found no full WoA weights')
        return dict(prepared_views=len(views),kind='wo_a',coefficient_precision='unchanged BF16')

    def prepare_resident_bank(self,keys,*,ownercheck,cancelled=None,generation=1,
                              budget_bytes=75_000_000_000,max_bank_bytes=None,prefill=False):
        """Build/attach a closed bank at an explicitly owned idle boundary."""
        if type(prefill)is not bool:raise ValueError('Source bank prefill selection must be boolean')
        if prefill:
            if getattr(self,'source_resident_prefill_policy','reference')not in ('expert_major_v1','expert_major_wide_v1'):
                raise ValueError('Source PP bank preparation requires explicit expert-major policy')
            from .deepseek_v41_source_prefill_session import prepare_prefill_session
            return prepare_prefill_session(self,keys,ownercheck=ownercheck,cancelled=cancelled,
                generation=generation,budget_bytes=budget_bytes,
                max_bank_bytes=(3<<30)if max_bank_bytes is None else max_bank_bytes)
        from .deepseek_v41_source_bank import build_bank
        from .deepseek_v41_source_bank_session import PreparedBankSession
        import mlx.core as mx
        if self.source_resident_gather_policy!='resident_bank_v1':
            raise ValueError('Source bank preparation requires resident_bank_v1 policy')
        if self.source_resident_bank_session is not None:
            raise ValueError('Source bank session already attached')
        bank=build_bank(mx,self.expert_stream,keys,ownercheck=ownercheck,generation=generation,
                        budget_bytes=budget_bytes,max_bank_bytes=(4<<30)if max_bank_bytes is None else max_bank_bytes)
        session=PreparedBankSession(self,bank,ownercheck=ownercheck,cancelled=cancelled)
        self.source_resident_bank_session=session
        return session

    def execution_scope_for_rows(self, rows):
        """Select completion ownership before any route can acquire experts.

        Unknown routes use the largest original-format payload and include one
        other pending epoch. This only selects ownership, never token shapes.
        """
        if type(rows) is not int or not 1 <= rows <= 1024:
            raise ValueError('Source completion advice requires rows1..1024')
        if getattr(getattr(self,'expert_stream',None),'_resident_bank_preparing',None):
            raise ValueError('Source forward cannot enter during resident bank preparation')
        session=getattr(self,'source_resident_bank_session',None)
        if session is not None and session.active:
            return nullcontext()
        selected_rows=(rows==1 or getattr(self,'source_resident_prefill_policy','reference')!='reference')
        if selected_rows and getattr(self,'source_resident_gather_policy','reference')!='reference' and not (session is not None and session.replaying):
            raise ValueError('Source resident_bank_v1 execution requires an explicitly prepared bank session')
        stream=getattr(self,'expert_stream',None)
        if not getattr(stream,'adaptive_residency_enabled',False):
            return self._ordinary_execution_policy()
        completion=getattr(self,'_source_completion',None)
        if completion is not None and completion.active:
            raise ValueError('Cannot replace active completion ownership')
        from .deepseek_v41_source_memory import SourceMemorySnapshot,SourceMemoryNeed,plan_source_memory,ORIGINAL_EXPERT_BYTES
        forced=getattr(self,'_source_adaptive_force_ordinary',False)
        self._source_adaptive_force_ordinary=False
        bounded=False
        try:
            current=stream.adaptive_residency_update()
            available=current.get('measured_available_bytes') if current.get('availability_known') else None
            resident=int(getattr(stream,'_cache_bytes',0))
            pinned=min(resident,len(getattr(stream,'_leased',{}))*ORIGINAL_EXPERT_BYTES)
            count=min(384,rows*self.artifact['args'].num_experts_per_tok)
            # bounded_2 may keep a preceding layer's complete route alive.
            need=SourceMemoryNeed(cache_ceiling_bytes=int(stream.maximum_cache_bytes),
                expert_payload_bytes=(ORIGINAL_EXPERT_BYTES,)*count,
                expert_staging_bytes=int(getattr(stream,'_adaptive_staging',ORIGINAL_EXPERT_BYTES)),
                kv_growth_bytes=count*ORIGINAL_EXPERT_BYTES,
                phase='prefill' if rows>1 else 'decode',requested_prefill_rows=rows,
                route_working_set_known=True)
            advice=plan_source_memory(SourceMemorySnapshot(None,available,resident,pinned),need)
            bounded=(not forced and available is not None
                     and advice['expert_strategy']!='sequential_expert'
                     and (rows==1 or advice['prefill_chunk_rows']>=rows)
                     and 2*count*ORIGINAL_EXPERT_BYTES+need.expert_staging_bytes<=stream.maximum_cache_bytes)
            self.source_completion_memory_advice=dict(advice,worst_unique_experts=count,
                preceding_epoch_payload_bytes=count*ORIGINAL_EXPERT_BYTES,
                selected_completion='bounded_2' if bounded else 'reference')
        except Exception:
            bounded=False
        if bounded:
            key='source_adaptive_bounded_completion_executions'
            self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
            return self._ordinary_execution_policy()
        key='source_adaptive_ordinary_completion_executions'
        self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
        return nullcontext()

    def execution_scope(self):
        # Older callers without imminent row geometry stay conservative.
        if getattr(getattr(self,'expert_stream',None),'adaptive_residency_enabled',False):
            return self.execution_scope_for_rows(1024)
        return self._ordinary_execution_policy()

    def _ordinary_execution_policy(self):
        if getattr(self,'source_completion_policy','reference') == 'reference':
            return nullcontext()
        if self._source_completion is None:
            from .deepseek_v41_source_completion import BoundedSourceCompletion
            self._source_completion=BoundedSourceCompletion(self.expert_stream,maximum=2)
        return self._source_completion.execution()

    def dense_window(self):
        session=getattr(self,'source_resident_bank_session',None)
        if session is not None and session.active:return nullcontext()
        completion=getattr(self,'_source_completion',None)
        if completion is not None and completion.active:
            return completion.scope()
        return super().dense_window()

    def evaluate_dense_window(self,*values):
        session=getattr(self,'source_resident_bank_session',None)
        if session is not None and session.active:return session.retain(*values)
        completion=getattr(self,'_source_completion',None)
        if completion is not None and completion.active:
            return completion.evaluate(*values)
        return super().evaluate_dense_window(*values)

    def grouped(self, x, name, *, groups=8):
        import mlx.core as mx
        self._ensure_dense()
        weight = self.array(name)
        if getattr(self,'source_woa_pp_policy','reference')=='native_bf16_grouped_pp_v1':
            from .deepseek_v41_source_woa_pp import maybe_project
            value=maybe_project(name,x,weight,groups=groups,counters=self.scheduled_grouped_calls)
            if value is not None:return value
        if getattr(self,'source_bf16_loads_policy','reference') in ('woa_f32','direct_f32'):
            from ..runtime.gguf.source_dense import maybe_project
            session=getattr(self,'source_resident_bank_session',None)
            view_selected=(session is not None and session.active
                and getattr(session,'operand_policy','full_operands_v1')=='coefficient_views_v1')
            if x.ndim==4 and x.shape[1]==1 and view_selected:
                cap=getattr(self,'_resident_dense_coefficient_views',{}).get(name)
                if cap is None:raise ValueError('Source WoA coefficient alias was not prepared before decode')
                value=maybe_project(x.astype(mx.float32),weight,'wo_a',operand_policy='coefficient_views_v1',coefficient_view=cap)
                key='source_bf16_grouped_coefficient_views'
                self.scheduled_grouped_calls[key]=self.scheduled_grouped_calls.get(key,0)+1
            else:value=maybe_project(x.astype(mx.float32),weight,'wo_a') if x.ndim==4 and x.shape[1]==1 else None
            legacy=False
            # The exact legacy MMA candidate helps the measured small-row
            # shapes; the 99-row prefill probe regressed versus stock MLX.
            if x.ndim==4 and 2<=x.shape[1]<=7:
                from ..runtime.gguf import source_dense_legacy
                if source_dense_legacy.supports(x,weight):
                    value=source_dense_legacy.project(x,weight)
                    legacy=True
            if value is not None:
                key='source_bf16_grouped_legacy_f32' if legacy else 'source_bf16_grouped_direct_f32'
                self.scheduled_grouped_calls[key]=self.scheduled_grouped_calls.get(key,0)+1
                return value
            key='source_bf16_grouped_fallback'
            self.scheduled_grouped_calls[key]=self.scheduled_grouped_calls.get(key,0)+1
        self.scheduled_grouped_calls["official_bf16_grouped_f32"] = self.scheduled_grouped_calls.get("official_bf16_grouped_f32", 0)+1
        return mx.einsum("btgi,goi->btgo", x.astype(mx.float32), weight.astype(mx.float32).reshape(groups, weight.shape[0]//groups, weight.shape[1]))

    def begin_source_frame(self, start, rows):
        """CPU-only forward identity; no routes/activations persist between frames."""
        if type(start) is not int or start < 0 or type(rows) is not int or not 1 <= rows <= 1024:
            raise ValueError('Source frame requires a valid absolute start and live rows')
        if getattr(self,'_source_current_frame',None) is not None:
            raise ValueError('Source frame is already active; nested forwards are unsupported')
        self._source_frame_serial=getattr(self,'_source_frame_serial',0)+1
        self._source_current_frame=(self._source_frame_serial,start,rows)

    def end_source_frame(self):
        self._source_current_frame=None

    def _adaptive_expert_plan(self, values, prefix):
        """Choose a complete current route window or one completed expert.

        This never subdivides token rows or changes an expert's local M. Soft
        targets can evict unleased cache entries; active roots remain owned.
        """
        stream=self.expert_stream
        if not getattr(stream,'adaptive_residency_enabled',False):return None
        collector_active=getattr(stream,'_completion_collector',None)is not None
        from .deepseek_v41_source_memory import SourceMemorySnapshot,SourceMemoryNeed,plan_source_memory,ORIGINAL_EXPERT_BYTES,Q1_EXPERT_BYTES
        current=stream.adaptive_residency_update()
        available=current.get('measured_available_bytes')if current.get('availability_known')else None
        layer=int(prefix.split('.')[1]);expert_ids=tuple(int(e)for e in np.unique(values))
        def payload(key):
            return Q1_EXPERT_BYTES if getattr(stream,'kind',lambda k:'original')(key)=='q1'else ORIGINAL_EXPERT_BYTES
        sizes=tuple(payload((layer,e))for e in expert_ids)
        resident=int(getattr(stream,'_cache_bytes',0))
        pinned=min(resident,sum(payload(key)for key in getattr(stream,'_leased',{})))
        need=SourceMemoryNeed(cache_ceiling_bytes=int(stream.maximum_cache_bytes),
            expert_payload_bytes=sizes,expert_staging_bytes=int(getattr(stream,'_adaptive_staging',ORIGINAL_EXPERT_BYTES)),
            phase='prefill'if values.shape[0]>1 else'decode',requested_prefill_rows=values.shape[0],
            route_working_set_known=True)
        advice=plan_source_memory(SourceMemorySnapshot(None,available,resident,pinned),need)
        bulk=(advice['expert_strategy']!='sequential_expert'
              and (values.shape[0]==1 or advice['prefill_chunk_rows']>=values.shape[0]))
        updated=stream.set_adaptive_cache_target(advice['target_cache_bytes'],
            working_set_experts=len(sizes)if bulk else 1)
        bulk=bulk and sum(sizes)<=updated['effective_payload_target_bytes']
        if not bulk and collector_active:
            # A live epoch owns immutable expert generations. Never retire one
            # early through a key-based collector; select ordinary next forward.
            self._source_adaptive_force_ordinary=True
            counts=self.scheduled_expert_calls
            key='source_adaptive_pressure_deferred_to_next_forward'
            counts[key]=counts.get(key,0)+1
            bulk=True
            advice=dict(advice,completion_pressure_deferred=True)
        if not bulk and updated.get('imminent_expert_window_bytes',0)>stream.minimum_cache_bytes:
            stream.set_adaptive_cache_target(advice['target_cache_bytes'],working_set_experts=1)
        advice=dict(advice,expert_strategy=('batched_experts' if collector_active and advice.get('completion_pressure_deferred') else advice['expert_strategy'])if bulk else'sequential_expert',
                    actual_unique_experts=len(sizes),actual_route_payload_bytes=sum(sizes),
                    effective_payload_target_bytes=updated['effective_payload_target_bytes'],
                    local_expert_M_unchanged=True)
        self.source_expert_memory_advice=advice
        counts=self.scheduled_expert_calls
        key='source_adaptive_bulk_windows'if bulk else'source_adaptive_sequential_windows'
        counts[key]=counts.get(key,0)+1
        return advice

    def moe(self, x, ids, scores, prefix, *, ticket=None,ffn_input_token=None):
        import mlx.core as mx
        from ..runtime.models.deepseek_v4 import limited_swiglu
        from ..runtime.models.deepseek_v41_quant import quantize
        self._ready()
        if getattr(getattr(self,'expert_stream',None),'_resident_bank_preparing',None):
            raise ValueError('Source MoE cannot enter during resident bank preparation')
        session=getattr(self,'source_resident_bank_session',None)
        if session is not None and session.active:
            rows=x.shape[0]if len(x.shape)==2 else 0
            if (ticket is not None or ffn_input_token is not None or not 1<=rows<=getattr(session,'max_token_rows',1)
                    or x.shape!=(rows,self.artifact['args'].hidden_size)
                    or ids.shape!=(rows,self.artifact['args'].num_experts_per_tok) or scores.shape!=ids.shape):
                raise ValueError('Resident bank Source route geometry exceeds its prepared session')
            from ..runtime.models.deepseek_v41_quant import fused_quantize
            prepared=fused_quantize(x.astype(mx.float32),'fp8').astype(x.dtype).astype(mx.float32)
            key='official_source_resident_bank_gpu_routes'
            self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
            return session.moe(prepared,ids.reshape(-1),scores.reshape(-1).astype(mx.float32),prefix)
        if ticket is not None:
            raise ValueError("official source profile has synchronous expert reads")
        if ffn_input_token is not None:
            from .deepseek_v41_source_ffn_input import FfnInputToken
            if not isinstance(ffn_input_token,FfnInputToken):
                raise ValueError('Source FFN input requires an explicit frame token')
            ffn_input_token.validate_routed_request(self,x,prefix)
        values = np.asarray(ids)
        self.source_route_host_materializations=getattr(self,'source_route_host_materializations',0)+1
        if (values.dtype != np.uint32 or values.ndim != 2 or not 0 < values.shape[0] <= 1024
                or values.shape != scores.shape or x.shape != (values.shape[0], self.artifact["args"].hidden_size)
                or values.shape[1] != self.artifact["args"].num_experts_per_tok):
            raise ValueError("official expert route geometry mismatch")
        memory_plan=self._adaptive_expert_plan(values,prefix)
        sequential=memory_plan is not None and memory_plan['expert_strategy']=='sequential_expert'
        route_plan=None
        if (values.shape[0]>1 and getattr(self,'source_prefill_batch_policy','reference')!='reference'
                and getattr(self,'source_frame_reuse_policy','reference')=='prefill_v1'):
            from .deepseek_v41_source_prefill import plan_jobs
            layer=int(prefix.split('.')[1])
            route_plan=plan_jobs(values,lambda e:self.expert_stream.kind((layer,e)),reuse=True)
            values=route_plan.values
            flat=values.reshape(-1)
            order=route_plan.order
            sorted_ids=None # Selected batched path consumes the authoritative plan.
            for key in ('official_source_frame_route_plan_materialized','official_source_frame_route_order_reused'):
                self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
            if hasattr(self.expert_stream,'observe_routes'):
                self.expert_stream.observe_routes(prefix,values,histogram=(route_plan.expert_ids,route_plan.counts))
                key='official_source_frame_route_histogram_reused'
                self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
        else:
            flat = values.reshape(-1)
            if hasattr(self.expert_stream, 'observe_routes'):
                self.expert_stream.observe_routes(prefix, values)
            order = np.argsort(flat, kind="stable")
            sorted_ids = flat[order]
        fused_policy=self.expert_math_policy in ('epilogue_fused_fp32','routed_fused_fp32')
        fuse_down=(self.expert_math_policy=='routed_fused_fp32' and values.shape[0]==1
                   and len(np.unique(flat))==values.shape[1] and not sequential)
        if sequential and sorted_ids is None:
            sorted_ids=flat[order]
        owned = not self.expert_stream._active
        with self.expert_stream.window() if owned else nullcontext():
            parts = []
            down_inputs,down_experts,down_ids=[],[],[]
            rounded = lambda v: v.astype(mx.bfloat16).astype(mx.float32)
            # QAT reduces only within each row's 32-element blocks. Reuse it
            # before routing instead of repeating it for each selected expert.
            # This explicit profile preserves projection and reduction order.
            # See provenance/deepseek-v41-source-math-probe.json.
            all_selected = None
            if self.expert_math_policy != "reference":
                from ..runtime.models.deepseek_v41_quant import fused_quantize
                quantize = fused_quantize
                prepare=lambda:quantize(x.astype(mx.float32), "fp8").astype(x.dtype).astype(mx.float32)
                all_selected=(prepare() if ffn_input_token is None else
                              ffn_input_token.routed(self,x,prefix,prepare))
                if fused_policy:
                    key='official_source_input_qat'
                    self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
            if values.shape[0]>1 and getattr(self,'source_prefill_batch_policy','reference')!='reference' and not sequential:
                if route_plan is None:
                    output=self._prefill_batched_gate_up(all_selected,values,scores,prefix,order)
                else:
                    output=self._prefill_batched_gate_up(all_selected,values,scores,prefix,order,route_plan=route_plan)
                if owned:mx.eval(output)
                return output
            if fuse_down and getattr(self.expert_stream,'packed_buffers',False):
                from ..runtime.gguf import source_gate_up_fusion as gu
                from ..runtime.gguf.source_middle_fusion import exp_spelling
                from ..runtime.gguf.source_down_fusion import down_combine
                from ..runtime.gguf.source_mxfp4_pair import _device_supported
                expert_ids=[int(e) for e in np.unique(sorted_ids)]
                experts=[self.expert_stream.acquire(prefix,e) for e in expert_ids]
                route_slots=mx.array(order,mx.uint32)
                if gu.batch_supports(all_selected,experts,scores.reshape(-1),route_slots) and _device_supported():
                    mid=gu.batch_middle(all_selected,experts,scores.reshape(-1),route_slots,exp_spelling=exp_spelling())
                    output=down_combine([mid[i:i+1] for i in range(len(experts))],experts,expert_ids)
                    counts={'official_source_gate_up_middle_batch_fused':1,
                            'official_source_down_combine_fused':1,
                            'official_source_mapped_gate_up':len(experts),
                            'official_source_input_gather_bytes_avoided':values.size*x.shape[-1]*4,
                            'official_mxfp4_act32_routed_fused_fp32':1}
                    for key,amount in counts.items():
                        self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+amount
                    self.expert_stream.stats['lookups']+=1
                    if owned:mx.eval(output)
                    return output
            for expert in np.unique(sorted_ids):
                locations = order[sorted_ids == expert]
                rows = mx.array(locations//values.shape[1], mx.uint32)
                slots = mx.array(locations, mx.uint32)
                data = self.expert_stream.acquire(prefix, int(expert))
                def project(activation, kind):
                    if kind+'.biases' in data:
                        from ..runtime.gguf.source_q1 import matmul
                        return matmul(activation, data[kind+'.qweight'], data[kind+'.scales'], data[kind+'.biases'])
                    bf16=(values.shape[0]>1 and getattr(self,'source_prefill_wide_dtype','reference')=='bf16_p1'
                          and activation.shape[0]>=(33 if kind=='down' else 97))
                    if bf16:
                        key='official_source_prefill_bf16_p1_'+('down' if kind=='down' else 'gate_up')
                        self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
                        return mx.quantized_matmul(activation.astype(mx.bfloat16),data[kind+'.qweight'],
                            scales=data[kind+'.scales'],mode='mxfp4').astype(mx.float32)
                    return mx.quantized_matmul(activation, data[kind+".qweight"],
                                               scales=data[kind+".scales"], mode="mxfp4")
                if fused_policy:
                    from ..runtime.gguf.source_middle_fusion import middle,exp_spelling
                    mapped=False
                    from ..runtime.gguf import source_gate_up_fusion as gu
                    from ..runtime.gguf.source_mxfp4_pair import _device_supported
                    score_rows=scores.reshape(-1)[slots] if self.expert_math_policy=='routed_fused_fp32' else None
                    use_gu=(self.expert_math_policy=='routed_fused_fp32'
                            and gu.supports(all_selected,rows,score_rows,data) and _device_supported())
                    if use_gu:
                        down_input=gu.middle(all_selected,rows,score_rows,data,exp_spelling=exp_spelling())
                        mapped=True
                        key='official_source_gate_up_middle_fused'
                        self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
                    elif 'gate.biases' in data:
                        from ..runtime.gguf.source_q1 import paired_matmul_mapped
                        gate,up=paired_matmul_mapped(all_selected,rows,
                            *(data[kind+field] for kind in ('gate','up')
                              for field in ('.qweight','.scales','.biases')))
                        mapped=True
                    else:
                        from ..runtime.gguf.source_mxfp4_pair import paired_matmul_mapped, supports_mapped
                        args=tuple(data[kind+field] for kind in ('gate','up') for field in ('.qweight','.scales'))
                        if supports_mapped(all_selected,rows,*args):
                            gate,up=paired_matmul_mapped(all_selected,rows,*args)
                            mapped=True
                        else:
                            selected=all_selected[rows]
                            gate,up=project(selected,'gate'),project(selected,'up')
                    if mapped:
                        key='official_source_mapped_gate_up'
                        self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
                        key='official_source_input_gather_bytes_avoided'
                        self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+len(locations)*x.shape[-1]*4
                    if not use_gu:
                        down_input=middle(gate,up,scores.reshape(-1),slots)
                        key='official_source_fused_middle'
                        self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
                    if fuse_down:
                        down_inputs.append(down_input);down_experts.append(data);down_ids.append(int(expert))
                    else:
                        parts.append(rounded(project(down_input,'down')))
                        if sequential:
                            self.expert_stream.release_completed_experts(((int(prefix.split('.')[1]),int(expert)),),roots=(parts[-1],))
                            key='source_adaptive_completed_expert_parts'
                            self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
                            del data
                    continue
                # Share the exact reference QAT node across gate and up for
                # either expert format. See source-activation-reuse provenance.
                selected = (quantize(x[rows], "fp8").astype(mx.float32)
                            if all_selected is None else all_selected[rows])
                if self.expert_math_policy in ("paired_q1_fused_fp32", "paired_fused_fp32") and "gate.biases" in data:
                    from ..runtime.gguf.source_q1 import paired_matmul
                    gate, up = paired_matmul(selected, *(data[kind+field] for kind in ("gate", "up")
                        for field in (".qweight", ".scales", ".biases")))
                    gate, up = rounded(gate), rounded(up)
                    pair_key = "official_q1_paired_gate_up"
                    self.scheduled_expert_calls[pair_key] = self.scheduled_expert_calls.get(pair_key, 0)+1
                elif self.expert_math_policy == "paired_fused_fp32":
                    from ..runtime.gguf.source_mxfp4_pair import paired_matmul, supports
                    pair_args = tuple(data[kind+field] for kind in ("gate", "up")
                                      for field in (".qweight", ".scales"))
                    if supports(selected, *pair_args):
                        gate, up = paired_matmul(selected, *pair_args, verified_m5_vector_band=True)
                        gate, up = rounded(gate), rounded(up)
                        pair_key = "official_mxfp4_paired_gate_up"
                        self.scheduled_expert_calls[pair_key] = self.scheduled_expert_calls.get(pair_key, 0)+1
                    else:
                        gate, up = rounded(project(selected, "gate")), rounded(project(selected, "up"))
                else:
                    gate, up = rounded(project(selected, "gate")), rounded(project(selected, "up"))
                middle = rounded(limited_swiglu(gate, up)*scores.reshape(-1)[slots, None])
                parts.append(rounded(project(quantize(middle, "fp8").astype(mx.float32), "down")))
                if sequential:
                    self.expert_stream.release_completed_experts(((int(prefix.split('.')[1]),int(expert)),),roots=(parts[-1],))
                    key='source_adaptive_completed_expert_parts'
                    self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
                    del data
            if fuse_down:
                from ..runtime.gguf.source_down_fusion import down_combine
                output=down_combine(down_inputs,down_experts,down_ids)
                key='official_source_down_combine_fused'
                self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
            elif self.expert_math_policy in ("ordered_fused_fp32", "paired_q1_fused_fp32", "paired_fused_fp32", "epilogue_fused_fp32", "routed_fused_fp32"):
                from .deepseek_v41_source_math import ordered_combine
                output = ordered_combine(parts, values, order)
            else:
                sorted_output = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
                restored = sorted_output[mx.array(np.argsort(order), mx.uint32)].reshape(*values.shape, x.shape[-1])
                output = mx.zeros(x.shape, mx.float32)
                # The official reference accumulates routed experts in expert-ID order.
                ranking = mx.array(np.argsort(values, axis=1, kind="stable"), mx.uint32)
                for slot in range(values.shape[1]):
                    output = output + mx.take_along_axis(restored, ranking[:, slot:slot+1, None], axis=1)[:, 0]
            policy = ("official_mxfp4_act32_hoisted_fused" if all_selected is not None
                      else "official_mxfp4_act32_per_expert")
            if self.expert_math_policy in ("ordered_fused_fp32", "paired_q1_fused_fp32", "paired_fused_fp32", "epilogue_fused_fp32", "routed_fused_fp32"):
                policy = "official_mxfp4_act32_"+self.expert_math_policy
            self.scheduled_expert_calls[policy] = self.scheduled_expert_calls.get(policy, 0)+1
            self.expert_stream.stats["lookups"] += 1
            if owned:
                mx.eval(output)
            return output

    def _prefill_batched_gate_up(self,x,values,scores,prefix,order,*,route_plan=None):
        """Batch independent expert jobs; retain ordinary per-expert down math.

        The caller already holds the ordinary provider window. No expert row
        count is split to enter a different stock reduction band.
        Provenance: provenance/deepseek-v41-source-prefill-batching.json.
        """
        import mlx.core as mx
        from .deepseek_v41_source_prefill import Plan,plan_jobs
        from .deepseek_v41_source_math import ordered_combine
        from ..runtime.gguf import source_gate_up_prefill as gu
        from ..runtime.gguf.source_middle_fusion import middle,exp_spelling
        from ..runtime.gguf.source_mxfp4_pair import _device_supported
        from ..runtime.gguf.source_q1 import matmul
        stream=self.expert_stream
        if not getattr(stream,'packed_buffers',False) or not _device_supported():
            raise ValueError('Source prefill batch requires admitted packed objects and supported device')
        if route_plan is None:
            layer=int(prefix.split('.')[1])
            kinds={int(e):stream.kind((layer,int(e))) for e in np.unique(values)}
            plan=plan_jobs(values,kinds)
        else:
            if (getattr(self,'source_frame_reuse_policy','reference')!='prefill_v1'
                    or not isinstance(route_plan,Plan) or route_plan.values is not values
                    or route_plan.order is not order or route_plan.combine_offsets is None):
                raise ValueError('Source prefill route plan does not own this frame')
            plan=route_plan
        device_packet=None
        if getattr(self,'source_row_packet_policy','reference') == 'row_packet_v1':
            if route_plan is None:
                raise ValueError('Source row packet requires the current authoritative route plan')
            frame=getattr(self,'_source_current_frame',None)
            if frame is None or frame[2] != x.shape[0]:
                raise ValueError('Source row packet is outside its model forward frame')
            from .deepseek_v41_source_row_packet import prepare_row_packet
            packet=prepare_row_packet(plan,frame_key='source-forward-'+str(frame[0]),
                generation=frame[0],layer=int(prefix.split('.')[1]),position_start=frame[1])
            device_packet=packet.upload(mx)
            counters=self.scheduled_expert_calls
            for key,amount in (('source_row_packet_uploads',1),('source_row_packet_bytes',packet.backing.nbytes)):
                counters[key]=counters.get(key,0)+int(amount)
        def validate_packet(*, contents=False):
            if device_packet is not None:
                if getattr(self,'_source_current_frame',None) is not frame:
                    raise ValueError('Source row packet stale model forward frame')
                packet.validate(plan,frame_key='source-forward-'+str(frame[0]),
                    generation=frame[0],layer=int(prefix.split('.')[1]),contents=contents)
        experts=[stream.acquire(prefix,job.expert_id) for job in plan.jobs]
        if device_packet is not None:validate_packet(contents=True)
        middles=[None]*len(plan.jobs)
        batch_down=getattr(self,'source_prefill_batch_policy','reference')=='gate_up_down'
        middle_batches=[]
        tiny_owned={};tiny_down_batches=0
        tiny_selected=getattr(self,'source_prefill_down_policy','reference')=='q1_tiny_batch_v1'
        if tiny_selected:
            from .deepseek_v41_source_prefill_down import tiny_prefix,maybe_tiny_project
        flat_scores=scores.reshape(-1)
        for batch_index,batch in enumerate(plan.batches):
            if device_packet is not None:validate_packet()
            row_map,locations=(device_packet.batch(batch_index) if device_packet is not None
                               else (mx.array(batch.rows),mx.array(batch.locations)))
            packed=gu.pp_middle(x,[experts[j] for j in batch.job_indices],row_map,
                                flat_scores[locations],batch.row_counts,exp_spelling=exp_spelling(),
                                **({'row_packet':device_packet,'batch_index':batch_index} if device_packet is not None else {}))
            if batch_down:middle_batches.append((batch,packed))
            else:
                tiny_jobs=0;tiny_output=None
                if tiny_selected:
                    tiny_jobs,tiny_rows=tiny_prefix(batch,prefill_rows=x.shape[0])
                    if tiny_jobs:
                        activation=packed if tiny_rows==packed.shape[0]else packed[:tiny_rows]
                        tiny_output=maybe_tiny_project(activation,[experts[j]for j in batch.job_indices[:tiny_jobs]],
                            batch.row_counts[:tiny_jobs],policy='q1_tiny_batch_v1',expert_math=self.expert_math_policy,
                            prefill_batch=self.source_prefill_batch_policy,prefill_rows=x.shape[0],
                            counters=self.scheduled_expert_calls)
                        if tiny_output is not None:tiny_down_batches+=1
                for i,j in enumerate(batch.job_indices):
                    if tiny_output is not None and i<tiny_jobs:
                        tiny_owned[j]=tiny_output[batch.offsets[i]:batch.offsets[i+1]]
                    else:
                        middles[j]=packed[batch.offsets[i]:batch.offsets[i]+batch.row_counts[i]]
        # Wide whole-expert tile candidate is an independent explicit policy.
        # Keep each expert's stock M/K partitions and its separately rounded
        # gate/up boundary; only the batch's contiguous address bases change.
        # Provenance: deepseek-v41-source-prefill-wide-integration.json.
        wide_plan=None
        gu_fallback=plan.fallback_job_indices
        wide_policy=getattr(self,'source_prefill_wide_math','reference')
        if wide_policy!='reference':
            from .source_prefill_wide_plan import plan_wide_jobs
            from ..runtime.gguf.source_prefill_legacy_tiles import batch_project
            wide_plan=plan_wide_jobs(plan.jobs)
            completed=set(wide_plan.eligible_job_indices)
            if not completed.issubset(plan.fallback_job_indices):
                raise RuntimeError('Source wide tiles selected a non-fallback job')
            for batch in wide_plan.batches:
                row_map=mx.array(batch.rows)
                locations=mx.array(batch.locations)
                gathered=x[row_map]
                data=[experts[j] for j in batch.job_indices]
                if wide_policy=='nax_k16_ordered_f32_v1':
                    from ..runtime.gguf import source_prefill_nax_profile as native
                    _verify_nax_prefill_source()
                    gate=native.batch_project(gathered,data,batch.row_counts,projection='gate',numerical_profile=native.PROFILE)
                    up=native.batch_project(gathered,data,batch.row_counts,projection='up',numerical_profile=native.PROFILE)
                    packed=middle(gate,up,flat_scores,locations)
                elif wide_policy=='legacy_tiles':
                    gate=batch_project(gathered,data,batch.row_counts,projection='gate')
                    up=batch_project(gathered,data,batch.row_counts,projection='up')
                    packed=middle(gate,up,flat_scores,locations)
                elif wide_policy=='legacy_tiles_16x64':
                    gate=batch_project(gathered,data,batch.row_counts,projection='gate',tile=(16,64))
                    up=batch_project(gathered,data,batch.row_counts,projection='up',tile=(16,64))
                    packed=middle(gate,up,flat_scores,locations)
                else:
                    from ..runtime.gguf.source_prefill_legacy_tiles import batch_middle
                    tile=(16,64) if wide_policy=='legacy_tiles_fused_16x64' else (32,32)
                    packed=batch_middle(gathered,data,batch.row_counts,flat_scores,locations,
                                        exp_spelling=exp_spelling(),tile=tile)
                for i,j in enumerate(batch.job_indices):
                    middles[j]=packed[batch.row_offsets[i]:batch.row_offsets[i+1]]
            gu_fallback=tuple(j for j in plan.fallback_job_indices if j not in completed)
        def project(v,data,name):
            if name+'.biases' in data:
                return matmul(v,data[name+'.qweight'],data[name+'.scales'],data[name+'.biases'])
            if (getattr(self,'source_prefill_wide_dtype','reference')=='bf16_p1'
                    and v.shape[0]>=(33 if name=='down' else 97)):
                key='official_source_prefill_bf16_p1_'+('down' if name=='down' else 'gate_up')
                self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
                return mx.quantized_matmul(v.astype(mx.bfloat16),data[name+'.qweight'],
                    scales=data[name+'.scales'],mode='mxfp4').astype(mx.float32)
            return mx.quantized_matmul(v,data[name+'.qweight'],scales=data[name+'.scales'],mode='mxfp4')
        for j in gu_fallback:
            if device_packet is not None:validate_packet()
            job,data=plan.jobs[j],experts[j]
            job_rows,job_locations=(device_packet.job(j) if device_packet is not None
                                    else (mx.array(job.rows),mx.array(job.locations)))
            selected=x[job_rows]
            pair=None
            if getattr(self,'source_native_mx_policy','reference')!='reference':
                from .deepseek_v41_source_native_mx import maybe_gate_up
                pair=maybe_gate_up(self,selected,data,prefix,job.expert_id,prefill_rows=x.shape[0])
            gate,up=(project(selected,data,'gate'),project(selected,data,'up')) if pair is None else pair
            middles[j]=middle(gate,up,flat_scores,job_locations)
        parts=[];part_experts=[]
        owner_down_jobs=0
        native_down_jobs=0
        down_policy=getattr(self,'source_prefill_down_policy','reference')
        down_maybe_project=None
        if down_policy=='q1_m1_owner_store':
            from .deepseek_v41_source_prefill_down import maybe_project as down_maybe_project
        def owned_project_down(activation,data,expert_id):
            nonlocal owner_down_jobs,native_down_jobs
            if getattr(self,'source_native_mx_policy','reference')!='reference':
                from .deepseek_v41_source_native_mx import maybe_down
                native=maybe_down(self,activation,data,prefix,expert_id,prefill_rows=x.shape[0])
                if native is not None:
                    native_down_jobs+=1
                    return native.astype(mx.bfloat16).astype(mx.float32)
            if down_maybe_project is not None:
                owned=down_maybe_project(activation,data,policy=down_policy,expert_math=self.expert_math_policy,
                    prefill_batch=self.source_prefill_batch_policy,prefill_rows=x.shape[0],
                    counters=self.scheduled_expert_calls)
                if owned is not None:
                    owner_down_jobs+=1
                    return owned
            return project(activation,data,'down').astype(mx.bfloat16).astype(mx.float32)
        if batch_down:
            from ..runtime.gguf.source_down_prefill import pp_down
            from ..runtime.gguf.source_prefill_combine import combine_mapped
            for batch,activation in middle_batches:
                parts.append(pp_down(activation,[experts[j] for j in batch.job_indices],batch.row_counts))
                part_experts.append([(plan.jobs[j].expert_id,plan.jobs[j].count) for j in batch.job_indices])
            for j in plan.fallback_job_indices:
                if middles[j] is None:raise RuntimeError('Source prefill batch omitted a fallback job')
                parts.append(project(middles[j],experts[j],'down').astype(mx.bfloat16).astype(mx.float32))
                part_experts.append([(plan.jobs[j].expert_id,plan.jobs[j].count)])
            output=combine_mapped(parts,values,order,part_experts)
        else:
            for j,(job,data,activation) in enumerate(zip(plan.jobs,experts,middles)):
                if device_packet is not None:validate_packet()
                if j in tiny_owned:
                    parts.append(tiny_owned[j])
                    continue
                if activation is None:raise RuntimeError('Source prefill batch omitted an expert job')
                parts.append(owned_project_down(activation,data,job.expert_id))
            if route_plan is None:
                output=ordered_combine(parts,values,order)
            else:
                if device_packet is not None:validate_packet(contents=True)
                output=ordered_combine(parts,values,order,offsets=plan.combine_offsets,
                    **({'row_packet':device_packet} if device_packet is not None else {}))
                key='official_source_frame_combine_offsets_reused'
                self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+1
        counts={'official_source_prefill_gu_batches':len(plan.batches),
                'official_source_prefill_batched_jobs':plan.stats['batched_jobs'],
                'official_source_prefill_fallback_jobs':len(gu_fallback),
                'official_source_prefill_ordinary_down_jobs':len(plan.fallback_job_indices) if batch_down else len(parts)-owner_down_jobs-native_down_jobs-len(tiny_owned),
                'official_source_prefill_down_batches':len(middle_batches)+tiny_down_batches,
                'official_source_prefill_assembly_parts':len(parts),
                'official_source_mapped_gate_up':plan.stats['batched_jobs'],
                'official_source_input_gather_bytes_avoided':sum(sum(b.row_counts) for b in plan.batches)*x.shape[-1]*4,
                'official_mxfp4_act32_routed_fused_fp32':1}
        if wide_plan is not None:
            stats=wide_plan.stats
            counts.update(official_source_prefill_wide_jobs=stats['eligible_jobs'],
                          official_source_prefill_wide_gu_batches=stats['one_projection_batches'],
                          official_source_prefill_wide_projection_batches=stats['gate_plus_up_partial_dispatches'],
                          official_source_prefill_wide_projection_reductions=(0 if '_fused' in wide_policy else stats['gate_plus_up_partial_dispatches']),
                          official_source_prefill_wide_gather_logical_bytes=stats['gathered_input_bytes_summed_over_batches'],
                          official_source_prefill_wide_partial_logical_bytes=2*stats['one_projection_partial_bytes_summed_over_batches'])
            if wide_policy=='nax_k16_ordered_f32_v1':
                counts['official_source_prefill_nax_k16_jobs']=stats['eligible_jobs']
                counts['official_source_prefill_nax_k16_projection_batches']=stats['gate_plus_up_partial_dispatches']
            if wide_policy!='legacy_tiles':
                counts['official_source_prefill_wide_16x64_batches']=(stats['one_projection_batches'] if wide_policy.endswith('_16x64') else 0)
                counts['official_source_prefill_wide_fused_middle_batches']=(stats['one_projection_batches'] if '_fused' in wide_policy else 0)
        for key,amount in counts.items():
            self.scheduled_expert_calls[key]=self.scheduled_expert_calls.get(key,0)+amount
        stream.stats['lookups']+=1
        return output

    def dense_receipt(self):
        value = super().dense_receipt()
        value.update(mode="official_fp8_bf16_resident", numeric_profile=REVISION, original_source_required=True,
                     weight_requantization=False, activation_quantization="FP8/E8M0/group32", grouped_wo_a="source FP8 expanded to BF16 per official convert.py")
        value.update(source_dense_qat=getattr(self,'source_dense_qat_policy','reference'),
                     source_route_host_materializations=getattr(self,'source_route_host_materializations',0),
                     source_submission=getattr(self,'source_submission_policy','reference'),
                     source_row_packet=getattr(self,'source_row_packet_policy','reference'),
                     source_frame_reuse=getattr(self,'source_frame_reuse_policy','reference'),
                     source_native_mx=getattr(self,'source_native_mx_policy','reference'),
                     source_qakv=getattr(self,'source_qakv_policy','reference'),
                     source_q_b_prefill=getattr(self,'source_q_b_prefill_policy','reference'),
                     source_fp8_pp=getattr(self,'source_fp8_pp_policy','reference'),
                     source_woa_pp=getattr(self,'source_woa_pp_policy','reference'),
                     source_ffn_input=getattr(self,'source_ffn_input_policy','reference'),
                     source_prefill_down=getattr(self,'source_prefill_down_policy','reference'),
                     source_q_b_prefill_contract='raw F32 PP99 Q_b64x32; existing model owns BF16 boundary',
                     source_attention_math=getattr(self,'source_attention_math_policy','reference'),
                     source_bf16_loads=getattr(self,'source_bf16_loads_policy','reference'),
                     source_completion=getattr(self,'source_completion_policy','reference'),
                     source_prefill_batch=getattr(self,'source_prefill_batch_policy','reference'),
                     source_prefill_wide_dtype=getattr(self,'source_prefill_wide_dtype','reference'),
                     source_prefill_wide_math=getattr(self,'source_prefill_wide_math','reference'),
                     source_routed_dot_contract=('original MXFP4 GU M13..96 native BF16 NAX K16/F32; ascending F32 +0 partitions; existing BF16/scored FP8 middle; distinct unqualified numerical profile'
                         if getattr(self,'source_prefill_wide_math','reference')=='nax_k16_ordered_f32_v1' else 'existing selected Source routed numerical law'))
        if getattr(self,'source_attention_math_policy','reference') in ('source_short_decode_joint_f32_v1','source_short_decode_grouped_f32_v1'):
            from ..runtime.models.deepseek_v41_short_attention import implementation_identity
            value['source_attention_implementation_sha256']=implementation_identity()
            value['source_attention_numerical_contract']='explicit short-band F32 reduction; Source owner BF16; unqualified'
        if getattr(self,'source_attention_math_policy','reference') == 'omlx_nax_bf16_online_v1':
            from ..runtime.models.deepseek_v41_source_attention import implementation_identity
            value['source_attention_implementation_sha256'] = implementation_identity()
            value['source_attention_numerical_contract'] = 'PP-only BF16 NAX QK, online F32 softmax and FP16 hi+lo PV; existing owner BF16; distinct unqualified numerical profile'
        if getattr(self,'source_fp8_pp_policy','reference')=='native_bf16_qb_wob_v1':
            from .deepseek_v41_source_fp8_pp import implementation_identity
            value['source_fp8_pp_implementation_sha256']=implementation_identity()
            value['source_fp8_pp_numerical_contract']='B1 PP99 Q_b/WoB original MXFP8 codes, prepared QAT cast BF16, stock native BF16 result cast F32 then existing owner BF16; distinct unqualified numerical profile; raw F32 accumulator unexposed'
        if getattr(self,'source_woa_pp_policy','reference')=='native_bf16_grouped_pp_v1':
            from .deepseek_v41_source_woa_pp import implementation_identity
            value['source_woa_pp_implementation_sha256']=implementation_identity()
            value['source_woa_pp_numerical_contract']='B1 M99 G8 K4096 N1024 original resident BF16 coefficients; Source BF16-owned input cast BF16; stock grouped BF16 result cast F32 then existing owner/QAT/WoB. Distinct unqualified profile; raw F32 accumulator unexposed.'
        if getattr(self,'source_ffn_input_policy','reference')=='shared_routed_qat_reuse':
            from .deepseek_v41_source_ffn_input import implementation_identity
            value['source_ffn_input_implementation_sha256']=implementation_identity()
            value['source_ffn_input_contract']='B1/R1 or R99/H5120 F32 Source BF16-owned producer; explicit frame-owned FP8 group32 operand; unchanged routed quantizer reused by shared GU; unsupported geometry uses ordinary path; router/BF16/projection/score/down/HC unchanged; no global tensor cache'
        if getattr(self,'source_prefill_down_policy','reference')!='reference':
            from .deepseek_v41_source_prefill_down import implementation_identity,tiny_implementation_identity,TINY_CONTRACT,TINY_COUNTER
            tiny=getattr(self,'source_prefill_down_policy','reference')=='q1_tiny_batch_v1'
            value['source_prefill_down_implementation_sha256']=tiny_implementation_identity()if tiny else implementation_identity()
            value['source_prefill_down_contract']=(TINY_CONTRACT
                if tiny else 'PP-only Q1 M1: unchanged F32 dot and SIMD reduction, BF16 owner fused into final F32 store; decode/original/other-M fallback unchanged; unqualified')
            if tiny:
                value['source_prefill_down_counters']={k:v for k,v in self.scheduled_expert_calls.items()if k.startswith(TINY_COUNTER)}
        if getattr(self,'source_native_mx_policy','reference')!='reference':
            from .deepseek_v41_source_native_mx import implementation_identity
            value['source_native_mx_implementation_sha256']=implementation_identity()
            value['source_native_mx_contract']='B1 globalPP2..99 original localM16..99; explicit contiguous HALF activation consumption with rounding allowed; original E2M1/E8M0 coefficients attested once on CPU; unchanged scored/QAT middle and BF16 owners; Q1/small-M/decode reference; unqualified'
        registry=getattr(self,'_source_native_mx_attestations',None)
        if registry is not None:
            value['source_native_mx_attestations']=registry.receipt()
        if getattr(self,'_source_completion',None) is not None:
            value['source_completion_receipt']=self._source_completion.receipt()
        return value
