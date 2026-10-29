# SPDX-License-Identifier: MIT
"""Bounded non-owning expert predictions attached to APCv2 checkpoints.

These values never contain coefficients, pointers, executable hooks or leases.
They are advisory input to an adapter/resource planner, not authority for KV
validity or a reason to reject lazy loading when a hint is absent.
"""
from dataclasses import dataclass, replace
import hashlib
import json

MAX_HINT_BYTES = 64 << 10
MAX_HINT_EXPERTS = 4096
BINDING_FIELDS = ('checkpoint', 'source_revision', 'quantization', 'layout', 'resource_abi')


def prefix_token_digest(tokens):
    values=tuple(tokens)
    if any(type(value)is not int or value<0 for value in values):
        raise ValueError('Expert hint token IDs must be nonnegative integers')
    return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExpertResidencyHints:
    """An observation or an advisory inventory, never a residency guarantee.

    resident_snapshot/advisory describes weights loaded at publication time,
    potentially including unused/rejected-branch experts and excluding evicted
    historical experts. truncated marks a non-exhaustive inventory; neither
    inventory form claims the committed path used the listed experts.
    """
    checkpoint: str
    source_revision: str
    quantization: str
    layout: str
    resource_abi: str
    snapshot_revision: str
    token_digest: str
    covered_tokens: int
    layer_expert_counts: tuple[int, ...]
    experts: tuple[tuple[int, int], ...]
    phase: str = 'committed'
    coverage: str = 'exact'
    truncated: bool = False

    def __post_init__(self):
        for name in (*BINDING_FIELDS, 'snapshot_revision'):
            value=getattr(self,name)
            if type(value)is not str or not 1<=len(value)<=512:
                raise ValueError('Expert hint identity must be bounded nonempty text: '+name)
        if (type(self.token_digest)is not str or len(self.token_digest)!=64
                or any(c not in '0123456789abcdef'for c in self.token_digest)):
            raise ValueError('Expert hints require an exact prefix token digest')
        if type(self.covered_tokens)is not int or not 0<=self.covered_tokens<1<<31:
            raise ValueError('Invalid expert hint checkpoint frontier')
        counts=self.layer_expert_counts
        if (type(counts)is not tuple or not 1<=len(counts)<=4096
                or any(type(v)is not int or not 1<=v<=1<<20 for v in counts)):
            raise ValueError('Expert hints require bounded per-layer expert geometry')
        if type(self.experts)is not tuple or len(self.experts)>MAX_HINT_EXPERTS:
            raise ValueError('Expert hint set exceeds finite metadata bound')
        for key in self.experts:
            if (type(key)is not tuple or len(key)!=2 or any(type(v)is not int for v in key)
                    or not 0<=key[0]<len(counts) or not 0<=key[1]<counts[key[0]]):
                raise ValueError('Expert hint ID is outside declared layer geometry')
        if tuple(sorted(set(self.experts)))!=self.experts:
            raise ValueError('Expert hint IDs must be sorted and unique')
        if self.phase not in ('committed','prefill','speculative','resident_snapshot'):
            raise ValueError('Unknown expert hint observation phase')
        if self.coverage not in ('exact','superset','advisory'):
            raise ValueError('Unknown expert hint coverage')
        if (self.phase=='resident_snapshot')!=(self.coverage=='advisory'):
            raise ValueError('Resident inventory requires advisory coverage; path observations cannot claim advisory inventory')
        if type(self.truncated)is not bool or(self.truncated and self.phase!='resident_snapshot'):
            raise ValueError('Only resident inventories may declare boolean truncation')
        if self.nbytes>MAX_HINT_BYTES:
            raise ValueError('Expert hint encoding exceeds finite metadata byte bound')

    @property
    def binding(self):
        return tuple((name,getattr(self,name))for name in BINDING_FIELDS)

    def to_manifest(self):
        return dict(schema='mlx2.expert-residency-hints.v1',
            **{name:getattr(self,name)for name in (*BINDING_FIELDS,'snapshot_revision','token_digest','covered_tokens','phase','coverage','truncated')},
            layer_expert_counts=list(self.layer_expert_counts), experts=[list(key)for key in self.experts])

    @property
    def nbytes(self):
        return len(json.dumps(self.to_manifest(),sort_keys=True,separators=(',',':')).encode())

    def for_restore(self,prefix_tokens,expected_binding=None):
        """Decline stale/speculative hints; retain truthful observation scope.

        covered_tokens/token_digest/snapshot_revision continue to identify the
        ORIGINAL observation. A returned superset never claims an exact set
        for a shorter restored checkpoint. Resident snapshots always remain
        advisory inventories, never historical path supersets. Their observed
        original frontier, snapshot identity and truncation mark stay intact.
        Longer continuations need new hints.
        """
        if type(prefix_tokens)is not int or not 0<=prefix_tokens<=self.covered_tokens:
            return None
        if self.phase=='speculative':return None
        if expected_binding is not None:
            try:expected=tuple(sorted(dict(expected_binding).items()))
            except (TypeError,ValueError):return None
            if expected!=tuple(sorted(self.binding)):return None
        return (replace(self,coverage='superset')
            if prefix_tokens<self.covered_tokens and self.phase!='resident_snapshot' else self)

    def for_store(self,tokens):
        values=tuple(tokens)
        if (self.phase=='speculative' or len(values)!=self.covered_tokens
                or prefix_token_digest(values)!=self.token_digest):return None
        return self

    @classmethod
    def from_manifest(cls,value):
        """Bad optional metadata never invalidates an otherwise valid KV hit."""
        try:
            if type(value)is not dict or value.get('schema')!='mlx2.expert-residency-hints.v1':return None
            if len(json.dumps(value,separators=(',',':')).encode())>MAX_HINT_BYTES:return None
            fields={name:value[name]for name in (*BINDING_FIELDS,'snapshot_revision','token_digest','covered_tokens','phase','coverage')}
            fields['truncated']=value.get('truncated',False)
            fields['layer_expert_counts']=tuple(value['layer_expert_counts'])
            fields['experts']=tuple(tuple(key)for key in value['experts'])
            return cls(**fields)
        except (KeyError,TypeError,ValueError,OverflowError):return None


def stored_expert_hints(hints,tokens):
    if type(hints)is not ExpertResidencyHints:return None
    return hints.for_store(tokens)
