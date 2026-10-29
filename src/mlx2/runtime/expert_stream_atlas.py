# SPDX-License-Identifier: Apache-2.0
"""Explicit predictive hints, separate from the collect-only expert atlas.

Learn token-history and adjacent-layer co-activation, never select target
experts. Bounded, source/recipe-bound records describe only exact byte reads.
Colibri coupling is a design reference; this implementation and format are new.
"""

import hashlib
import json
import os
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

REVISION = "mlx2.expert-stream-hints.v1"


class ExpertStreamAtlas:
    def __init__(self, binding, *, layers, experts, top_k, path=None, contexts=256):
        if (not isinstance(binding, str) or not binding
                or any(type(value) is not int for value in (layers, experts, top_k, contexts))
                or not 1 <= layers <= 256 or not 1 <= experts <= 4096
                or not 1 <= top_k <= experts or not 1 <= contexts <= 4096):
            raise ValueError("invalid expert atlas identity/geometry")
        self.binding, self.layers, self.experts, self.top_k = binding, layers, experts, top_k
        self.contexts, self.path = contexts, Path(path) if path else None
        self.tokens = [OrderedDict() for _ in range(layers)]
        self.couples = [dict() for _ in range(layers)]
        self._last = [OrderedDict() for _ in range(layers)]
        self.stats = dict(observations=0, token_predictions=0, coupling_predictions=0, loaded=0, ignored=0, save_failed=0)
        self.load()

    @staticmethod
    def keys(tokens, tail):
        history = list(map(int, tail))
        result = []
        for token in tokens:
            history.append(int(token))
            raw = np.asarray(history[-4:], dtype="<i8").tobytes()
            result.append(hashlib.blake2s(raw, digest_size=12).hexdigest())
        return tuple(result)

    def predict(self, layer, keys, *, previous=None, limit=6, excluded=(), diagnostics=None):
        """Rank optional reads, excluding owned experts before applying the limit.

        Diagnostics contain counts and only the bounded excluded candidate IDs;
        they do not retain token keys or change the authoritative router output.
        Omitting exclusions preserves the original prediction ordering.
        """
        keys, previous = self._arguments(layer, keys, previous)
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("invalid atlas prediction limit")
        if (not isinstance(excluded, (tuple, list, set, frozenset))
                or len(excluded) > self.experts
                or any(type(i) is not int or not 0 <= i < self.experts for i in excluded)):
            raise ValueError("invalid atlas prediction exclusions")
        if diagnostics is not None and not isinstance(diagnostics, dict):
            raise ValueError("invalid atlas prediction diagnostics")
        excluded = set(excluded)
        scores = Counter()
        for key in keys:
            record = self.tokens[layer].get(key)
            if record:
                scores.update({i: n * 4 for i, n in record.items()})
        if scores:
            self.stats["token_predictions"] += 1
        if previous is not None and layer:
            for expert in np.unique(previous):
                scores.update(self.couples[layer].get(int(expert), {}))
            self.stats["coupling_predictions"] += 1
        skipped = tuple(sorted(excluded.intersection(scores)))
        eligible = {i: n for i, n in scores.items() if i not in excluded}
        result = tuple(i for i, _ in sorted(eligible.items(), key=lambda p: (-p[1], p[0]))[:limit])
        if diagnostics is not None:
            diagnostics.update(candidates_total=len(scores), eligible_total=len(eligible),
                               excluded_experts=skipped, selected_total=len(result),
                               truncated_total=max(0, len(eligible)-len(result)))
        return result

    def observe(self, layer, keys, routes, *, previous=None):
        keys, previous = self._arguments(layer, keys, previous)
        routes = np.asarray(routes)
        if routes.shape != (len(keys), self.top_k) or routes.dtype.kind not in "iu" or np.any(routes >= self.experts) or np.any(routes < 0):
            raise ValueError("atlas observation geometry mismatch")
        if previous is not None and previous.shape != routes.shape:
            raise ValueError("atlas previous-route geometry mismatch")
        for row, key in enumerate(keys):
            signature = (tuple(sorted(map(int,routes[row]))),
                         tuple(sorted(map(int,previous[row]))) if previous is not None else ())
            if self._last[layer].get(key) == signature:
                continue
            self._last[layer][key] = signature
            self._last[layer].move_to_end(key)
            while len(self._last[layer]) > self.contexts:
                self._last[layer].popitem(last=False)
            counts = self.tokens[layer].setdefault(key, Counter())
            counts.update(map(int, routes[row]))
            self.tokens[layer].move_to_end(key)
            if max(counts.values()) >= 65536:
                for i in tuple(counts):
                    counts[i] = max(1, counts[i] // 2)
            # A token/history may have context-dependent routes. Bound its
            # retained alternatives; the router remains the sole authority.
            self.tokens[layer][key] = Counter(dict(counts.most_common(self.top_k*4)))
            while len(self.tokens[layer]) > self.contexts:
                self.tokens[layer].popitem(last=False)
            if previous is not None and layer:
                for source in previous[row]:
                    pair = self.couples[layer].setdefault(int(source), Counter())
                    pair.update(map(int, routes[row]))
                    if max(pair.values()) >= 65536:
                        for i in tuple(pair):
                            pair[i] = max(1, pair[i] // 2)
                    self.couples[layer][int(source)] = Counter(dict(pair.most_common(self.top_k*4)))
        self.stats["observations"] += len(keys)

    def _arguments(self, layer, keys, previous):
        if type(layer) is not int or not 0 <= layer < self.layers:
            raise ValueError("atlas layer outside admitted geometry")
        if not isinstance(keys, (tuple, list)) or len(keys) > 1024 or any(
                not isinstance(key, str) or len(key) != 24
                or any(c not in "0123456789abcdef" for c in key) for key in keys):
            raise ValueError("atlas keys outside admitted geometry")
        if previous is not None:
            previous = np.asarray(previous)
            if (previous.ndim != 2 or not 1 <= previous.shape[0] <= 1024
                    or previous.shape[1] != self.top_k or previous.dtype.kind not in "iu"
                    or np.any(previous < 0) or np.any(previous >= self.experts)):
                raise ValueError("atlas previous-route geometry mismatch")
        return keys, previous

    def manifest(self):
        return dict(revision=REVISION, binding=self.binding, layers=self.layers, experts=self.experts,
                    top_k=self.top_k, contexts=self.contexts)

    def load(self):
        if not self.path:
            return
        try:
            if not self.path.exists():
                return
            if self.path.is_symlink() or self.path.stat().st_size > 32 << 20:
                raise ValueError("unsafe atlas file")
            data = json.loads(self.path.read_text())
            if data["manifest"] != self.manifest():
                raise ValueError("foreign atlas")
            tokens, couples = [], []
            for kind in ("tokens", "couples"):
                result = []
                if len(data[kind]) != self.layers:
                    raise ValueError("atlas layer count")
                for rows in data[kind]:
                    bound = self.contexts if kind == "tokens" else self.experts
                    if len(rows) > bound:
                        raise ValueError("atlas row bound")
                    table = OrderedDict()
                    for key, values in rows:
                        if kind == "tokens":
                            if not isinstance(key, str) or len(key) != 24 or any(c not in "0123456789abcdef" for c in key):
                                raise ValueError("atlas token key")
                        elif type(key) is not int or not 0 <= key < self.experts:
                            raise ValueError("atlas expert key")
                        if len(values) > self.top_k*4 or key in table:
                            raise ValueError("atlas record bound")
                        count = Counter()
                        for expert, value in values:
                            if type(expert) is not int or not 0 <= expert < self.experts or type(value) is not int or not 1 <= value <= 65536 or expert in count:
                                raise ValueError("atlas counter")
                            count[expert] = value
                        table[key] = count
                    result.append(table)
                if kind == "tokens": tokens = result
                else: couples = result
            self.tokens, self.couples = tokens, couples
            self.stats["loaded"] = 1
        except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
            self.stats["ignored"] += 1

    def save(self):
        if not self.path:
            return False
        data = {"manifest": self.manifest(), "tokens": [], "couples": []}
        for kind in ("tokens", "couples"):
            data[kind] = [[(key, list(count.items())) for key, count in table.items()] for table in getattr(self, kind)]
        temporary = self.path.with_name(self.path.name+f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("x") as target:
                json.dump(data, target, separators=(",", ":"))
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
            return True
        except OSError:
            # Hints are optional. Persistence failure must not prevent exact
            # demand reads or turn successful model teardown into a failure.
            self.stats["save_failed"] += 1
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
