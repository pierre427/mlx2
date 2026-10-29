"""Physical FP32 DeepSeek V4 cache and separately charged prefill workspace.

FP8/E2M1 activation simulation is a numerical law, not a packed storage
claim: current local, pooled and pending state arrays remain float32. Shared
KV is one tensor, not an independent K/V pair. Global host/driver reserves
and the controller's existing forward allowance are not reduced here.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math


def _integer(name, value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} integer")
    return value


@dataclass(frozen=True)
class DeepSeekV4CacheBudget:
    compress_ratios: tuple[int, ...]
    hidden_size: int = 4096
    head_dim: int = 512
    index_head_dim: int = 128
    attention_heads: int = 64
    index_heads: int = 64
    sliding_window: int = 128
    hc_mult: int = 4
    q_lora_rank: int = 1024
    o_groups: int = 8
    o_lora_rank: int = 1024
    experts_per_token: int = 6
    routed_experts: int = 256
    expert_hidden: int = 2048
    shared_experts: int = 1
    item_bytes: int = 4
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    ledger_bytes_per_layer: int = 4096
    attention_workspace_bytes: int = 16 * 1024 * 1024
    indexer_workspace_bytes: int = 16 * 1024 * 1024
    # Existing controller allowance, kept separate from all cache geometry.
    transient_gib_per_lane: float = 1.76

    def __post_init__(self):
        if not self.compress_ratios or any(type(r) is not int or r not in (0,4,128) for r in self.compress_ratios):
            raise ValueError("unsupported DeepSeek V4 compression geometry")
        if self.item_bytes != 4:
            raise ValueError("DeepSeek V4 current cache storage is physical FP32")
        for name,value in asdict(self).items():
            if isinstance(value, int):
                _integer(name,value,positive=name != 'transcript_bytes_per_token')
        if not math.isfinite(self.transient_gib_per_lane) or self.transient_gib_per_lane < 0:
            raise ValueError("invalid separate forward workspace allowance")

    @classmethod
    def from_config(cls, config):
        def get(name, default=None):
            return config.get(name,default) if isinstance(config,dict) else getattr(config,name,default)
        count=get('num_hidden_layers')
        ratios=tuple(get('compress_ratios',()))
        if type(count) is not int or count != len(ratios):
            raise ValueError("DeepSeek V4 budget requires exact per-layer compression geometry")
        return cls(ratios, hidden_size=get('hidden_size',4096), head_dim=get('head_dim',512),
            index_head_dim=get('index_head_dim',128), attention_heads=get('num_attention_heads',64),
            index_heads=get('index_n_heads',64), sliding_window=get('sliding_window',128),
            hc_mult=get('hc_mult',4), q_lora_rank=get('q_lora_rank',1024),
            o_groups=get('o_groups',8), o_lora_rank=get('o_lora_rank',1024),
            experts_per_token=get('num_experts_per_tok',6), routed_experts=get('n_routed_experts',256),
            expert_hidden=get('moe_intermediate_size',2048), shared_experts=get('n_shared_experts',1))

    @property
    def layers(self):
        return len(self.compress_ratios)

    @property
    def local_live_bytes(self):
        return self.layers * self.sliding_window * self.head_dim * self.item_bytes

    @property
    def compression_fixed_bytes(self):
        total = 0
        for ratio in self.compress_ratios:
            if not ratio:
                continue
            for dim in (self.head_dim, self.index_head_dim) if ratio == 4 else (self.head_dim,):
                # KV plus gate pending window; ratio4 projections have 2*D.
                width=dim*(2 if ratio==4 else 1)
                total += 2 * (ratio-1) * width * self.item_bytes
                # Prior completed window's first half, needed across calls.
                if ratio == 4:
                    total += 2 * ratio * dim * self.item_bytes
        return total

    @property
    def resident_fixed_bytes(self):
        return self.local_live_bytes + self.compression_fixed_bytes + self.layers*self.ledger_bytes_per_layer

    @property
    def fixed_bytes(self):
        """Fixed running state plus a rollback copy; variable pools added separately."""
        return 2 * self.resident_fixed_bytes

    def _pool_capacity(self, context_tokens, ratio):
        count=context_tokens//ratio
        return math.ceil(count/self.allocation_step)*self.allocation_step + self.allocation_step

    def pooled_bytes(self, context_tokens):
        _integer('context_tokens',context_tokens)
        return sum(self._pool_capacity(context_tokens,r) *
                   (self.head_dim + (self.index_head_dim if r==4 else 0))*self.item_bytes
                   for r in self.compress_ratios if r)

    def snapshot_bytes(self, context_tokens):
        """Full array snapshot/restore copy; pools grow with prefix length."""
        _integer('context_tokens',context_tokens)
        return self.resident_fixed_bytes + self.pooled_bytes(context_tokens)

    def project_resident(self, context_tokens):
        _integer('context_tokens',context_tokens)
        return self.resident_fixed_bytes + self.pooled_bytes(context_tokens) + context_tokens*self.transcript_bytes_per_token

    def project(self, context_tokens):
        return self.project_resident(context_tokens) + self.snapshot_bytes(context_tokens)

    def project_pool(self, context_tokens, *, resident_lanes, running_lanes):
        _integer('context_tokens',context_tokens)
        _integer('resident_lanes',resident_lanes); _integer('running_lanes',running_lanes)
        if running_lanes > resident_lanes:
            raise ValueError("running_lanes cannot exceed resident_lanes")
        return resident_lanes*self.project_resident(context_tokens) + running_lanes*self.snapshot_bytes(context_tokens)

    @property
    def prefill_activation_elements_per_row(self):
        # Conservative graph-live activation bound: HC/residual, Q/KV/output,
        # selected experts and shared MLP, index QAT's 8-level intermediates.
        return (4*self.hc_mult*self.hidden_size + 3*self.attention_heads*self.head_dim
                + 2*self.q_lora_rank + 2*self.o_groups*self.o_lora_rank
                + 6*self.experts_per_token*self.expert_hidden
                + 4*self.shared_experts*self.expert_hidden
                + 8*self.index_heads*self.index_head_dim
                + 16*self.head_dim + 2*self.routed_experts)

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        _integer('context_tokens',context_tokens); _integer('chunk_rows',chunk_rows)
        if not chunk_rows:
            return 0
        # Appending immutable arrays can hold old/new pool copies simultaneously.
        copies=self.snapshot_bytes(context_tokens)
        # Every layer can retain incoming rows in addition to its128-row cache.
        bands=self.layers*chunk_rows*self.head_dim*self.item_bytes
        activations=chunk_rows*self.prefill_activation_elements_per_row*self.item_bytes
        # Current exact indexer masks are contiguous-causal bool matrices.
        max_pools=max((context_tokens//r for r in self.compress_ratios if r),default=0)
        masks=chunk_rows*max_pools
        return copies+bands+activations+masks+self.attention_workspace_bytes+self.indexer_workspace_bytes

    def as_dict(self):
        return {"schema":"deepseek-v4-fp32-cache-geometry-v1", **asdict(self),
                "layers":self.layers, "local_live_bytes":self.local_live_bytes,
                "compression_fixed_bytes":self.compression_fixed_bytes,
                "resident_fixed_bytes":self.resident_fixed_bytes, "fixed_bytes":self.fixed_bytes,
                "physical_storage":"float32-shared-kv-and-pools",
                "activation_simulation":"E4M3FN-KV/E2M1-indexer-does-not-pack-storage",
                "snapshot_scope":"one-full-context-array-copy-per-running-lane",
                "warm_copy":"observed-complete-cache-plus-projected-growth",
                "workspace":"prefill-copy-and-activation-bound-separate; controller-reserves-unchanged",
                "bound":"FP32-local+pending+previous+pooled+running-snapshot"}
