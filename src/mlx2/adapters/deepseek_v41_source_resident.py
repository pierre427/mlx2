# SPDX-License-Identifier: MIT
"""Bounded Source resident-resource pilot and exact forward transaction.

This module performs no MLX work on import. The pilot binds every reachable
packed backing explicitly; it does not expose raw addresses or advertise an
arbitrary-residency serving backend. Provenance: deepseek-v41-source-resident.json.
"""
from dataclasses import dataclass
from types import MappingProxyType

from .deepseek_v41_source_buffers import PackedExpert, ExpertLayout

PILOT_MAX_EXPERTS = 15
PROFILE = 'source_resident_explicit_resource_pilot_v1'


class ResidentCapabilityError(ValueError):
    pass


class ResidentCapacityError(RuntimeError):
    pass


def _key(key):
    if (type(key) is not tuple or len(key) != 2
            or any(type(v) is not int for v in key)
            or not 0 <= key[0] < 40 or not 0 <= key[1] < 384):
        raise ValueError('Source resident key must be (layer0..39, expert0..383)')
    return key


@dataclass(frozen=True)
class ResidentBinding:
    key: tuple
    expert: PackedExpert
    serial: int

    @property
    def nbytes(self):
        return self.expert.layout.nbytes


class ResidentEpoch:
    """All reachable resources and independent roots live through completion.

    The injected ``wait`` must complete every submitted root. A failed wait
    retains the epoch and pins, allowing an explicit drain retry. Retiring an
    epoch is not authority to publish an incomplete model/cache frontier.
    """
    def __init__(self, owner, generation, bindings):
        self._owner = owner
        self.generation = generation
        self._bindings = bindings
        self._roots = {}
        self._closed = False

    def _ready(self):
        if self._closed:
            raise RuntimeError('Source resident epoch is retired')

    @property
    def bindings(self):
        self._ready()
        return self._bindings

    def note_roots(self, *roots):
        self._ready()
        if not roots or any(root is None for root in roots):
            raise ValueError('Resident completion requires nonempty independent roots')
        self._roots.update((id(root), root) for root in roots)

    def retire(self, wait):
        self._ready()
        if not self._roots or not callable(wait):
            raise ValueError('Resident completion requires roots and a wait callback')
        wait(tuple(self._roots.values()))
        # Failed waits never reach this release. Every table member is pinned,
        # including experts which the current router ultimately did not use.
        for binding in self._bindings:
            self._owner._pins[binding.serial] -= 1
            if not self._owner._pins[binding.serial]:
                del self._owner._pins[binding.serial]
        self._roots.clear()
        self._bindings = ()
        self._closed = True
        self._owner._active_epochs -= 1


class ResidentResourceSet:
    """CPU coordinator for a bounded explicit-resource component pilot.

    Fixed formats and a hard byte cap include pending host staging. This is
    deliberately not another LFU/cache: an eventual provider integration must
    obtain the existing provider's leases and charge the same allocations.
    It is currently unwired, so it cannot enlarge a serving cache.
    """
    def __init__(self, formats, *, budget_bytes, max_experts=PILOT_MAX_EXPERTS, max_epochs=2):
        if (type(budget_bytes) is not int or budget_bytes <= 0
                or type(max_experts) is not int or not 1 <= max_experts <= PILOT_MAX_EXPERTS
                or type(max_epochs) is not int or not 1 <= max_epochs <= 2):
            raise ValueError('Invalid bounded resident pilot capacity')
        clean = dict(formats)
        for key, kind in clean.items():
            _key(key)
            if kind not in ('original', 'q1'):
                raise ValueError('Resident fixed format must be original or q1')
        self.formats = MappingProxyType(clean)
        self.budget_bytes, self.max_experts = budget_bytes, max_experts
        self.generation = 0
        self._serial = 0
        self._resident = {}
        self._pending = {}
        self._pins = {}
        self._active_epochs, self._max_epochs = 0, max_epochs

    @property
    def live_bytes(self):
        return (sum(b.nbytes for b in self._resident.values())
                + sum(b.nbytes + staging for b, _, _, staging in self._pending.values()))

    def stage(self, key, expert, *, host_owner, staging_bytes, wait_upload):
        """Own upload source/destination before any wait or publication.

        An upload failure leaves both owners quarantined in pending. No EOF or
        interrupted upload can become a resident entry through this interface.
        ``staging_bytes`` is the complete host allocation, not just its payload.
        """
        key = _key(key)
        if (not isinstance(expert, PackedExpert) or not isinstance(expert.layout, ExpertLayout)
                or self.formats.get(key) != expert.layout.kind):
            raise ValueError('Resident expert does not match fixed precision map')
        if (host_owner is None or type(staging_bytes) is not int
                or staging_bytes < expert.layout.nbytes or not callable(wait_upload)):
            raise ValueError('Resident upload needs owned, fully charged host staging')
        if key in self._resident or key in self._pending:
            raise ValueError('Resident key already installed or pending')
        if len(self._resident) + len(self._pending) >= self.max_experts:
            raise ResidentCapacityError('Resident pilot resource-binding bound exceeded')
        if self.live_bytes + expert.layout.nbytes + staging_bytes > self.budget_bytes:
            raise ResidentCapacityError('Resident pilot budget includes pending staging')
        self._serial += 1
        binding = ResidentBinding(key, expert, self._serial)
        self._pending[key] = (binding, host_owner, wait_upload, staging_bytes)
        return binding.serial

    def publish(self, key):
        key = _key(key)
        if key not in self._pending:
            raise ValueError('Resident upload is not pending')
        binding, host_owner, wait_upload, staging_bytes = self._pending[key]
        wait_upload(binding.expert)
        # Upload completion is the ownership-transfer boundary. Exceptions keep
        # pending host and backing owners and all charged bytes unchanged.
        self._resident[key] = binding
        del self._pending[key]
        self.generation += 1
        return self.generation

    def epoch(self, *, layer, expected_generation=None):
        if type(layer) is not int or not 0 <= layer < 40:
            raise ValueError('Resident pilot layer out of range')
        if expected_generation is not None and (type(expected_generation) is not int
                                                 or expected_generation != self.generation):
            raise ValueError('Stale resident generation')
        if self._active_epochs >= self._max_epochs:
            raise ResidentCapacityError('Resident pending epoch bound reached; drain or yield')
        bindings = tuple(sorted((b for b in self._resident.values() if b.key[0] == layer),
                                key=lambda b: b.key))
        if not bindings:
            raise ValueError('Resident pilot layer has no admitted resources')
        for binding in bindings:
            self._pins[binding.serial] = self._pins.get(binding.serial, 0) + 1
        self._active_epochs += 1
        return ResidentEpoch(self, self.generation, bindings)

    def evict(self, key):
        key = _key(key)
        binding = self._resident.get(key)
        if binding is None:
            raise ValueError('Resident key is absent')
        if self._pins.get(binding.serial):
            raise ResidentCapacityError('Resident resource is pinned through completion')
        del self._resident[key]
        self.generation += 1
        return binding.nbytes

    def evict_for(self, required_free_bytes):
        """Finite preflight: either enough unpinned victims exist, or no mutation.

        This pilot guarantees progress/refusal and never spins on pinned slots.
        Serving LFU choice belongs to the existing provider, not this helper.
        """
        if type(required_free_bytes) is not int or required_free_bytes < 0:
            raise ValueError('Resident required bytes must be nonnegative')
        victims = []
        total = 0
        for key, binding in sorted(self._resident.items()):
            if total >= required_free_bytes:
                break
            if not self._pins.get(binding.serial):
                victims.append(key)
                total += binding.nbytes
        if total < required_free_bytes:
            raise ResidentCapacityError('Insufficient unpinned resources; yield or ordinary replay')
        for key in victims:
            self.evict(key)
        return tuple(victims)

    def receipt(self):
        return dict(profile=PROFILE, generation=self.generation,
                    resident=len(self._resident), pending_uploads=len(self._pending),
                    pinned_resources=len(self._pins), live_bytes=self.live_bytes,
                    active_epochs=self._active_epochs, maximum_epochs=self._max_epochs,
                    budget_bytes=self.budget_bytes, implemented='CPU coordinator and uncompiled probe',
                    selected=False, observed_used=False, qualified=False,
                    scope='bounded component pilot; no arbitrary-residency serving backend')


def resident_capabilities():
    """Import-safe diagnostics; presence never means an observed GPU dispatch."""
    return dict(profile=PROFILE, component_pilot_implemented=True,
                component_pilot_gpu_observed=False, serving_backend_available=False,
                maximum_explicit_resources=PILOT_MAX_EXPERTS,
                resource_model='ordinary declared MLX input arrays; no raw addresses',
                missing_serving_requirements=('maintained dynamic-resource MLX operation',
                                              'exact revision-bound cache transaction integration'),
                selected=False, qualified=False)


def require_serving_backend():
    """Fail closed rather than treating the explicit-resource pilot as serving."""
    raise ResidentCapabilityError(
        'Source GPU-resident serving unavailable: no maintained dynamic-resource '
        'MLX operation and exact cache transaction integration; bounded explicit-resource pilot only')


class ForwardTransaction:
    """Private next-state execution; publish only a completed successful frontier.

    ``fork`` must return a private exact state; ``validate_fork`` checks its
    mutable-storage isolation without device readback. Publication gets another
    isolated fork, so a caller holding its working object cannot mutate the
    published state afterward. ``commit(owner, candidate, expected_revision)``
    must compare-and-adopt atomically or leave the owner unchanged. These are
    injected backend contracts, not proof that the cache adapter provides them.
    """
    def __init__(self, published, *, fork, validate_fork, revision, commit, discard, max_replays=1):
        if (any(not callable(f) for f in (fork, validate_fork, revision, commit, discard))
                or type(max_replays) is not int or not 0 <= max_replays <= 2):
            raise ValueError('Invalid exact transaction hooks/retry bound')
        self.published = published
        self._fork, self._commit, self._discard = fork, commit, discard
        self._validate_fork, self._revision = validate_fork, revision
        self.max_replays = max_replays
        self.replays = 0
        self.working = None
        self._candidate = None
        self._base_revision = None
        self._working_valid = False
        self._candidate_valid = False
        self._quarantined = []
        self.closed = False
        self.last_status = None

    def _current_revision(self):
        value = self._revision(self.published)
        def immutable(v):
            return type(v) in (int, str, bytes) or (type(v) is tuple and all(immutable(x) for x in v))
        if not immutable(value):
            raise ValueError('Authoritative frontier revision must be an immutable value')
        return value

    def begin(self):
        if self.closed or self.working is not None:
            raise RuntimeError('Forward transaction already active or closed')
        self._base_revision = self._current_revision()
        working = self._fork(self.published)
        # Retain even a rejected fork: constructing it may have encoded copies.
        self.working = working
        if (working is None or working is self.published
                or self._validate_fork(self.published, working) is not True):
            raise ValueError('Forward transaction requires private exact state')
        if self._current_revision() != self._base_revision:
            raise ValueError('Authoritative frontier changed while forking')
        self._working_valid = True
        return working

    def finish(self, *, status, wait, missing=()):
        if self.closed or self.working is None:
            raise RuntimeError('Forward transaction has no active state')
        if status not in ('ok', 'miss', 'stale', 'cancelled', 'error') or not callable(wait):
            raise ValueError('Invalid forward status/completion callback')
        # A failed wait retains working state; cancellation is not completion.
        wait(self.working)
        self.last_status = status
        if status == 'ok':
            if missing:
                raise ValueError('Successful forward cannot contain missing experts')
            if not self._working_valid or self._current_revision() != self._base_revision:
                raise ValueError('Authoritative frontier revision changed or fork is invalid')
            if self._candidate is None:
                self._candidate = self._fork(self.working)
            candidate = self._candidate
            if (candidate is None or candidate is self.working or candidate is self.published
                    or self._validate_fork(self.working, candidate) is not True
                    or self._validate_fork(self.published, candidate) is not True):
                raise ValueError('Publication requires independently owned exact state')
            self._candidate_valid = True
            # The publication fork can itself encode copies. Failure must retain
            # both private frontiers until a successful drain/cancel retry.
            wait(candidate)
            if self._current_revision() != self._base_revision:
                raise ValueError('Authoritative frontier changed before publication')
            result = self._commit(self.published, candidate, self._base_revision)
            self.working = None
            self._candidate = None
            self.closed = True
            return result
        if self._candidate is not None:
            wait(self._candidate)
            if self._candidate_valid:
                self._discard(self._candidate)
            else:
                # A rejected alias may share authoritative storage. An ordinary
                # pool-discard hook must never poison/recycle that storage.
                self._quarantined.append(self._candidate)
            self._candidate = None
            self._candidate_valid = False
        if self._working_valid:
            self._discard(self.working)
        else:
            self._quarantined.append(self.working)
        self.working = None
        self._working_valid = False
        if status in ('miss', 'stale') and not self._quarantined and self.replays < self.max_replays:
            self.replays += 1
            return 'ordinary_replay_required'
        self.closed = True
        return 'refused_without_publication'

    def receipt(self):
        return dict(closed=self.closed, working_retained=self.working is not None,
                    candidate_retained=self._candidate is not None,
                    quarantined_invalid_forks=len(self._quarantined),
                    replays=self.replays, maximum_replays=self.max_replays,
                    last_status=self.last_status, cache_integration='unimplemented')
