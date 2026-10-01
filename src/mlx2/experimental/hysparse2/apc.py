"""Exact endpoint bridge to the existing APCv2 engine; no serving registration.

The rolling self KV, cross KV, boundary and n-gram history travel together as
one checkpointed ArraysCache. Interior trimming is never approximated.
"""

import json

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.semantic_capsules import canonical_json


class EndpointAPC:
    def __init__(self, model, engine, *, checkpoint_revision, tokenizer_fingerprint):
        if not checkpoint_revision or not tokenizer_fingerprint:
            raise ValueError("exact checkpoint and tokenizer revisions required")
        self.model, self.engine = model, engine
        self.revision, self.tokenizer = checkpoint_revision, tokenizer_fingerprint
        self._parameter_epoch = model._parameter_epoch

    def key(self):
        if self._parameter_epoch is not self.model._parameter_epoch:
            raise ValueError("parameter update requires a new exact checkpoint binding")
        identity = self.model.new_cache().apcv2_identity
        return APCKey(
            model=self.model.config.model_type,
            revision=self.revision,
            adapter=("hysparse2-research", self.model.adapter_revision),
            tokenizer_fingerprint=self.tokenizer,
            cache_layout_fingerprint=identity["cache_layout_fingerprint"],
            semantic_fingerprint=(
                *identity["semantic_fingerprint"],
                identity.get("capsule_read_fingerprint"),
            ),
        )

    def _validate_endpoint(self, cache):
        self.model.validate_cache_state(cache)

    def publish(self, tokens, cache):
        tokens = list(tokens)
        if (
            self.model.training
            or cache.owner is not self.model._cache_owner
            or cache.batch != 1
        ):
            raise ValueError(
                "publication requires idle eval state owned by this model, B1"
            )
        if (
            cache.length < 1
            or len(tokens) != cache.length
            or any(
                type(t) is not int or not 0 <= t < self.model.config.vocab_size
                for t in tokens
            )
        ):
            raise ValueError("prompt tokens must cover the exact cache endpoint")
        identity = self.model.new_cache().apcv2_identity
        if canonical_json(cache.apcv2_identity) != canonical_json(identity):
            raise ValueError("cache revision differs before publication")
        self._validate_endpoint(cache)
        arrays = []

        def add(value):
            if value is None:
                return None
            index = len(arrays) + 1
            arrays.append(value)
            return index

        groups = {}
        for group in ("self_kv", "cross_kv"):
            groups[group] = {
                str(layer): [[add(k), add(v), offset] for k, v, offset in blocks]
                for layer, blocks in getattr(cache, group).items()
            }
        header = {
            "schema": "mlx2.hysparse2-apc-endpoint.v1",
            "length": cache.length,
            "identity": identity,
            "checkpoint_revision": self.revision,
            "tokenizer": self.tokenizer,
            "groups": groups,
            "boundary": add(cache.boundary),
            "ple_history": add(cache.ple_history),
            "self_layer_calls": cache.self_layer_calls,
            "cross_layer_calls": cache.cross_layer_calls,
        }
        if header["boundary"] is None:
            raise ValueError("cache boundary is missing")
        header["arrays"] = [[list(a.shape), str(a.dtype)] for a in arrays]
        payload = canonical_json(header)
        leaf = ArraysCache(len(arrays) + 1)
        leaf.cache = [mx.array(list(payload), dtype=mx.uint8)[None, :], *arrays]
        leaf.lengths = mx.array([cache.length], dtype=mx.int32)
        leaf._host_lengths = (leaf.lengths, [cache.length])
        leaf.state_checkpoint([cache.length], force=True)
        if leaf.snap_trim_position(cache.length) != cache.length:
            raise ValueError("exact endpoint checkpoint recording is disabled")
        mx.eval(leaf.state)
        capability = self.engine.store(self.key(), tokens, [leaf])
        if capability.stored is not True:
            raise ValueError(f"APCv2 refused endpoint: {capability.reason}")
        return capability

    def restore(self, tokens):
        """Return exact state and lookup receipt; close receipt.cache after use."""
        if self.model.training:
            raise ValueError("restore requires model.eval()")
        result = self.engine.lookup(self.key(), tokens)
        if not result.hit:
            return None, result
        try:
            return self._restore_hit(result), result
        except BaseException:
            close = getattr(result.cache, "close", None)
            if callable(close):
                close()
            raise

    def _restore_hit(self, result):
        if (
            not result.cache
            or len(result.cache) != 1
            or not isinstance(result.cache[0], ArraysCache)
        ):
            raise ValueError("unexpected endpoint cache topology")
        arrays = result.cache[0].cache
        if (
            not arrays
            or arrays[0].dtype != mx.uint8
            or arrays[0].ndim != 2
            or arrays[0].shape[0] != 1
            or arrays[0].size > 1 << 20
        ):
            raise ValueError("invalid endpoint metadata")
        header = json.loads(bytes(arrays[0][0].tolist()))
        expected = self.model.new_cache().apcv2_identity
        if (
            header.get("schema") != "mlx2.hysparse2-apc-endpoint.v1"
            or header["length"] != result.cached_tokens
            or header["checkpoint_revision"] != self.revision
            or header["tokenizer"] != self.tokenizer
            or canonical_json(header["identity"]) != canonical_json(expected)
        ):
            raise ValueError("endpoint or revision differs")
        if [[list(a.shape), str(a.dtype)] for a in arrays[1:]] != header["arrays"]:
            raise ValueError("endpoint tensor layout differs")
        cache = self.model.new_cache()
        cache.length = header["length"]
        referenced = set()

        def get(index):
            if index is None:
                return None
            if type(index) is not int or not 1 <= index < len(arrays):
                raise ValueError("invalid endpoint array index")
            if index in referenced:
                raise ValueError("endpoint state reuses an array index")
            referenced.add(index)
            return arrays[index]

        for group in ("self_kv", "cross_kv"):
            setattr(
                cache,
                group,
                {
                    int(layer): [(get(k), get(v), offset) for k, v, offset in blocks]
                    for layer, blocks in header["groups"][group].items()
                },
            )
        cache.boundary, cache.ple_history = (
            get(header["boundary"]),
            get(header["ple_history"]),
        )
        cache.self_layer_calls, cache.cross_layer_calls = (
            header["self_layer_calls"],
            header["cross_layer_calls"],
        )
        self._validate_endpoint(cache)
        if referenced != set(range(1, len(arrays))):
            raise ValueError("endpoint state contains unreferenced arrays")
        return cache
