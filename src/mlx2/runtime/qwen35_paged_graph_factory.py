"""Explicit hybrid B2 bootstrap, bounded charges and terminal retirement.

Ordinary prefill imports completed FA KV and a generation-zero GDN boundary.
This factory makes no performance or qualification claim.
"""
from __future__ import annotations
from threading import RLock

from mlx2.runtime.native_admission_retirement import register_reaper

_LIMIT = 12 << 30
_LOCK = RLock()
_CHARGED = 0
_ORPHANS = []


class HybridReservation:
    def __init__(self, session, size):
        if type(size) is not int or size < 0 or size > session.transient_bound:
            raise MemoryError('hybrid reservation exceeds precharged bound')
        self.bytes = size
        self.session = session
        self.roots = ()
        self.completed = False
        self.phase_boundaries = []
    def evaluated_phase(self, event, *, writer, orphaned_reads, callback=None):
        """A research quantum may stop only after materialization/terminals.

        This is a host quiescence receipt, never a public cache checkpoint.
        Callback failures keep the complete reservation reachable for abort.
        """
        if (self.completed or self.session.closed or type(event) is not dict or
                event.get('materialized') is not True or
                event.get('native_terminals_drained') is not True or
                event.get('public_state_published') is not False or
                writer.pending_epochs or writer.ledger.pending_count or orphaned_reads or
                (callback is not None and not callable(callback))):
            raise RuntimeError('packed phase boundary lacks evaluated terminal proof')
        self.phase_boundaries.append(dict(event))
        if callback is not None: callback(dict(event))

    def release(self):
        self.completed = True
        self.roots = ()
    def retain_failure_roots(self, roots):
        self.roots = roots
        self.session.failure_reservations.append(self)


class HybridServingResources:
    """Charge before allocation; retain through the last terminal owner release."""
    def __init__(self, charge, transient_bound, cap, *, research_limit_bytes=12<<30):
        global _CHARGED
        if (type(research_limit_bytes) is not int or research_limit_bytes not in (12<<30,40<<30) or
                type(charge) is not int or type(cap) is not int or
                not 0 < charge <= cap <= research_limit_bytes):
            raise MemoryError('hybrid aggregate memory charge exceeds profile cap')
        with _LOCK:
            if _CHARGED + charge > research_limit_bytes:
                raise MemoryError('hybrid global native admission budget exhausted')
            _CHARGED += charge
        self.charge = charge
        self.transient_bound = transient_bound
        self.failure_reservations = []
        self.retirement_failures = []
        self.owners = []
        self.unattached = []
        self.candidate = None
        self.closed = False
    def reserve(self, size):
        if self.closed: raise RuntimeError('hybrid reservation after retirement')
        return HybridReservation(self, size)
    def reap(self):
        if self.closed: return True
        candidate = self.candidate
        unattached_arena=getattr(self,'_unattached_arena',None)
        if candidate is None and unattached_arena is not None and not unattached_arena._closed:
            unattached_arena.close_after_terminal()
        if candidate is not None:
            from .paged_native_retirement import reap_native_request_owner
            writer = candidate.backend.writer
            for owner in self.owners:
                reap_native_request_owner(owner, writer, candidate.backend)
            candidate.backend.drain_failed_read_events()
            if writer.pending_epochs: writer.poll_completions()
            if writer.pending_epochs or writer.ledger.pending_count or candidate.backend._orphaned_reads:
                return False
            if writer.poisoned: writer.teardown_failed_arena()
            for layer in self.unattached:
                layer.close()
                writer.pool.retire(writer.ledger.completed_epoch)
            if (any(not owner.fully_retired for owner in self.owners) or
                    writer.pending_epochs or writer.ledger.pending_count or
                    candidate.backend._orphaned_reads or writer.pool.allocated_count):
                return False
            if candidate._failure_roots:
                candidate.release_failure_roots_after_teardown()
            writer.backend.close_after_terminal()
            candidate._bootstrap_failure_roots.clear()
            self.failure_reservations.clear()
        global _CHARGED
        with _LOCK:
            _CHARGED -= self.charge
        self.failure_reservations.clear()
        self.unattached.clear()
        self.closed = True
        return True
    def abort(self):
        if self.closed:return
        # Root before a close/reap that may fail or await a late callback.
        if not self.closed and self not in _ORPHANS: _ORPHANS.append(self)
        try:
            for owner in self.owners: owner.close()
            self.reap()
        except BaseException as error:
            self.retirement_failures.append({"stage":"abort",
                "error":f"{type(error).__name__}: {error}"})
            raise
        finally:
            if self.closed and self in _ORPHANS: _ORPHANS.remove(self)


def reap_hybrid_admission_orphans():
    for resources in tuple(_ORPHANS):
        try:
            if resources.reap(): _ORPHANS.remove(resources)
        except Exception:
            pass


register_reaper(f"{__name__}:{id(_ORPHANS)}", reap_hybrid_admission_orphans)


def create_shared_hybrid_graph_pack(adapter, requests, *, profile,
                                     permit_candidate=False, cancelled=lambda: False):
    """requests=(uid, revision, exact prompt IDs, maximum), two cold lanes."""
    if not permit_candidate: raise RuntimeError('hybrid native serving disabled')
    if (type(requests) is not tuple or len(requests) != 2 or
            len({item[0] for item in requests}) != 2 or
            any(type(item) is not tuple or len(item) != 4 or
                type(item[0]) is not int or item[0] < 0 or
                not isinstance(item[1], str) or not item[1] or
                type(item[2]) is not tuple or not item[2] or
                any(type(token) is not int or token < 0 for token in item[2]) or
                type(item[3]) is not int or item[3] < 1 for item in requests)):
        raise ValueError('two exact cold hybrid requests required')
    if any(item[1] != adapter.identity['fingerprint'] for item in requests):
        raise ValueError('hybrid request revision differs from loaded artifact')
    if cancelled(): raise ValueError('hybrid cohort cancelled before allocation')
    from .paged_hybrid_research_profile import validate_hybrid_stock_pair, require_hybrid_native_capabilities
    stock_reduction = validate_hybrid_stock_pair(profile, tuple(len(item[2]) for item in requests))
    from ..adapters.qwen35_paged_candidate import Qwen35PagedCandidate, clone_recurrent_caches
    from .paged_attention_plan import PAGE_SIZE
    from .paged_kv_pool import PagedKVPool
    from .paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from .paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
    from .paged_gdn_checkpoint import GDNBoundaryCheckpoint
    from .paged_native_atomic_owner import NativeAtomicRequestOwner
    from .qwen3_paged_native_backend import NativeQwen3PagedBackend
    # Geometry/dtype validation does not allocate or cast model weights.
    probe = Qwen35PagedCandidate(adapter.model, None,
        q1_stripes=profile['q1_simd_stripes'], q1_split_partition=profile['q1_split_partition'])
    dtype = probe.native_dtype_preflight()
    if dtype != profile['storage_dtype']: raise ValueError('loaded hybrid dtype differs from profile')
    kv = TokenKVProfile(probe.args.num_key_value_heads, 256, dtype)
    depth = probe.native_layer_count
    if any(len(tokens) + maximum - 1 > probe.max_visible_tokens for _, _, tokens, maximum in requests):
        raise ValueError('hybrid request exceeds complete decode context bound')
    capacity = sum(depth * ((len(tokens) + maximum + PAGE_SIZE - 1) // PAGE_SIZE + 2)
                   for _, _, tokens, maximum in requests)
    arena_bytes = 2 * capacity * kv.page_bytes
    # Retain ordinary staging plus three boundary sizes for public/private/retired
    # GDN roots. Three full staging bounds conservatively cover these as well.
    staging = sum(3 * probe.bootstrap_staging_bytes(len(tokens)) for _, _, tokens, _ in requests)
    scratch = probe.forward_scratch_bytes(tuple(len(tokens) + maximum - 2 for _, _, tokens, maximum in requests))
    transient = max(staging, scratch)
    reap_hybrid_admission_orphans()
    resources = HybridServingResources(arena_bytes + staging + scratch, transient,
                                       profile['memory_budget_bytes'])
    bootstrap = []
    try:
        import _paged_kv_native as native_extension
        require_hybrid_native_capabilities(profile, native_extension)
        import mlx.core as mx
        pool = PagedKVPool(capacity)
        native = NativeWriteBackend(capacity * kv.page_bytes, mx.default_stream(mx.gpu),
                                     permit_candidate=True, storage_dtype=dtype)
        writer = PagedKVWriteOwner(pool, native, page_bytes=kv.page_bytes, permit_candidate=True)
        backend = NativeQwen3PagedBackend(writer, permit_candidate=True, profile_host=True)
        candidate = Qwen35PagedCandidate(adapter.model, backend,
            q1_stripes=profile['q1_simd_stripes'], q1_split_partition=profile['q1_split_partition'])
        resources.candidate = candidate
        candidate._research_staged_graph = True
        candidate._serving_b2 = True
        candidate._serving_stock_reduction = stock_reduction
        candidate._serving_stock_singleton = profile.get('stock_singleton', False)
        candidate.serving_route = 'native_hybrid_paged_b2'
        candidate.reserve_serving_scratch = resources.reserve
        candidate.reap_serving_resources = resources.reap
        candidate._serving_resources = resources
        candidate._serving_prompt_ids_by_uid = {uid: tokens for uid, _, tokens, _ in requests}
        for uid, revision, tokens, _ in requests:
            if cancelled(): raise ValueError('hybrid cohort cancelled during bootstrap')
            layers = tuple(PagedKVTokenOwner(writer, kv, permit_candidate=True) for _ in range(depth))
            resources.unattached.extend(layers)
            boot = candidate.bootstrap_ordinary(tokens, layers,
                reserve_staging=resources.reserve, permit_candidate=True)
            if (writer.pending_epochs or writer.ledger.pending_count or
                    any(layer.offset != len(tokens) for layer in layers)):
                raise RuntimeError('hybrid bootstrap terminal state incomplete')
            boundary = GDNBoundaryCheckpoint(revision, uid, len(tokens), 0, boot.recurrent_caches)
            owner = NativeAtomicRequestOwner(revision, layers, {'gdn': (boundary,)},
                supported_planes=('kv', 'gdn'), enabled=True, checkpoint_planes=('gdn',),
                recurrent_clone=clone_recurrent_caches, lane_id=uid, reuse_private_tail=True)
            resources.owners.append(owner)
            for layer in layers: resources.unattached.remove(layer)
            bootstrap.append(boot)
        if cancelled(): raise ValueError('hybrid cohort cancelled after bootstrap')
        candidate._hybrid_bootstrap_receipts = tuple(boot.receipt for boot in bootstrap)
        return tuple(resources.owners), candidate, tuple(bootstrap)
    except BaseException:
        try: resources.abort()
        except BaseException:
            if resources not in _ORPHANS: _ORPHANS.append(resources)
        raise
