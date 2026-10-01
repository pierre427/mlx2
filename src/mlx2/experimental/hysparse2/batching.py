"""Research request cohorts with exact geometry and request-private state.

Ragged prompts use length cohorts rather than masked padding. Decode cohorts
also bind segment geometry and revisions. This is not a serving scheduler.
"""

import threading

import mlx.core as mx

from mlx2.runtime.semantic_capsules import canonical_json


class ResearchBatcher:
    def __init__(self, model, *, max_lanes=4):
        if type(max_lanes) is not int or not 1 <= max_lanes <= 32:
            raise ValueError("max_lanes must be from 1 to 32")
        self.model, self.max_lanes = model, max_lanes
        self._lock = threading.RLock()

    def _tokens(self, rows):
        rows = [tuple(row) for row in rows]
        if not rows or any(
            not row
            or any(
                type(t) is not int or not 0 <= t < self.model.config.vocab_size
                for t in row
            )
            for row in rows
        ):
            raise ValueError("nonempty valid token rows required")
        if self.model.training:
            raise ValueError("request batching requires model.eval()")
        return rows

    def _signature(self, cache):
        expected = self.model.new_cache().apcv2_identity
        if (
            cache.owner is not self.model._cache_owner
            or cache.batch != 1
            or canonical_json(cache.apcv2_identity) != canonical_json(expected)
        ):
            raise ValueError("request state owner, batch or revision differs")

        self.model.validate_cache_state(cache)

        def shape(array):
            return None if array is None else (tuple(array.shape[1:]), str(array.dtype))

        groups = tuple(
            (
                name,
                tuple(
                    (
                        layer,
                        tuple((shape(k), shape(v), offset) for k, v, offset in blocks),
                    )
                    for layer, blocks in sorted(getattr(cache, name).items())
                ),
            )
            for name in ("self_kv", "cross_kv")
        )
        return (
            cache.length,
            shape(cache.boundary),
            shape(cache.ple_history),
            groups,
            cache.self_layer_calls,
            cache.cross_layer_calls,
        )

    def _cohorts(self, signatures):
        groups = {}
        for row, signature in enumerate(signatures):
            groups.setdefault(signature, []).append(row)
        return [
            rows[i : i + self.max_lanes]
            for rows in groups.values()
            for i in range(0, len(rows), self.max_lanes)
        ]

    def _merge(self, caches):
        # Signatures are checked before any model call. Preserve segment edges:
        # changing coarse block boundaries could change candidate selection.
        merged = self.model.new_cache(len(caches))
        first = caches[0]
        merged.length = first.length
        merged.self_layer_calls, merged.cross_layer_calls = (
            first.self_layer_calls,
            first.cross_layer_calls,
        )
        for name in ("self_kv", "cross_kv"):
            setattr(
                merged,
                name,
                {
                    layer: [
                        (
                            mx.concatenate(
                                [getattr(c, name)[layer][i][0] for c in caches], axis=0
                            ),
                            mx.concatenate(
                                [getattr(c, name)[layer][i][1] for c in caches], axis=0
                            ),
                            offset,
                        )
                        for i, (_, _, offset) in enumerate(blocks)
                    ]
                    for layer, blocks in getattr(first, name).items()
                },
            )
        for name in ("boundary", "ple_history"):
            setattr(
                merged,
                name,
                None
                if getattr(first, name) is None
                else mx.concatenate([getattr(c, name) for c in caches], axis=0),
            )
        return merged

    def _split(self, cache):
        result = []
        for row in range(cache.batch):
            single = self.model.new_cache()
            single.length = cache.length
            single.self_layer_calls, single.cross_layer_calls = (
                cache.self_layer_calls,
                cache.cross_layer_calls,
            )

            def take(array, row=row):
                return None if array is None else mx.array(array[row : row + 1])

            for name in ("self_kv", "cross_kv"):
                setattr(
                    single,
                    name,
                    {
                        layer: [(take(k), take(v), offset) for k, v, offset in blocks]
                        for layer, blocks in getattr(cache, name).items()
                    },
                )
            single.boundary, single.ple_history = (
                take(cache.boundary),
                take(cache.ple_history),
            )
            mx.eval(single.arrays(), single.boundary, single.ple_history)
            result.append(single)
        return result

    def prefill(self, rows):
        with self._lock:
            rows = self._tokens(rows)
            if max(map(len, rows)) > self.model.config.max_context:
                raise ValueError("prompt exceeds context limit")
            cohorts = self._cohorts([len(row) for row in rows])
            logits, caches = [None] * len(rows), [None] * len(rows)
            for indices in cohorts:
                value, cache = self.model.prefill(mx.array([rows[i] for i in indices]))
                split = self._split(cache)
                for lane, i in enumerate(indices):
                    logits[i], caches[i] = mx.array(value[lane : lane + 1]), split[lane]
            mx.eval(logits)
            return logits, caches, self._receipt(cohorts, rows)

    def decode(self, rows, caches):
        with self._lock:
            rows = self._tokens(rows)
            if (
                len(rows) != len(caches)
                or any(len(row) != 1 for row in rows)
                or len({id(c) for c in caches}) != len(caches)
            ):
                raise ValueError(
                    "one unique request cache and one token per row required"
                )
            signatures = [self._signature(c) for c in caches]
            if any(c.length + 1 > self.model.config.max_context for c in caches):
                raise ValueError("decode exceeds context limit")
            cohorts = self._cohorts(signatures)
            logits, next_caches = [None] * len(rows), [None] * len(rows)
            for indices in cohorts:
                cache = self._merge([caches[i] for i in indices])
                value = self.model.decode(mx.array([rows[i] for i in indices]), cache)
                split = self._split(cache)
                for lane, i in enumerate(indices):
                    logits[i], next_caches[i] = (
                        mx.array(value[lane : lane + 1]),
                        split[lane],
                    )
            mx.eval(logits)
            return logits, next_caches, self._receipt(cohorts, rows)

    def _receipt(self, cohorts, rows):
        return {
            "schema": "mlx2.hysparse2-request-cohorts.v1",
            "lanes": len(rows),
            "model_calls": len(cohorts),
            "cohorts": cohorts,
            "padding_tokens": 0,
            "input_tokens": sum(map(len, rows)),
            "serving_route_qualified": False,
            "mixed_memory_batching": False,
        }
