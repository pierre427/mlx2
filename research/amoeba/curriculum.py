"""The English-free curriculum for H0 (plan section 2.2).

Synthetic memory tasks over random token IDs, so the recurrence learns recall,
copying, induction and nesting but no English distribution. Every sequence is
T tokens with a loss mask on the positions whose target is determined by the
task. Token IDs come from a fixed random pool of the donor vocabulary; the loss
is computed over that pool only.

* mqar       key-value pairs, then queries of earlier keys (multi-query associative recall)
* copy       a random span, a separator, then the span again
* induction  a random stream with planted repeats: after a repeated token, predict what followed it before
* dyck       balanced brackets over 8 bracket-type pairs; at closers the target type is set by the stack
"""
from __future__ import annotations

import numpy as np

TASKS = ("mqar", "copy", "induction", "dyck")


class Curriculum:
    def __init__(self, vocab_size=151936, pool=8192, T=256, seed=0, reserved=151643):
        self.rng = np.random.default_rng(seed)
        ids = self.rng.choice(reserved, size=pool + 32, replace=False)   # no special tokens
        self.pool = np.sort(ids[:pool]).astype(np.int32)
        self.markers = ids[pool:pool + 32].astype(np.int32)             # task and separator tokens
        self.T = T
        self.index = {int(t): i for i, t in enumerate(self.pool)}

    def _rand(self, n):
        return self.pool[self.rng.integers(0, len(self.pool), n)]

    def mqar(self):
        T, x, m = self.T, [self.markers[0]], [0]
        n = int(self.rng.integers(8, 40))
        keys = self.rng.choice(self.pool, size=n, replace=False); vals = self._rand(n)
        for k, v in zip(keys, vals):
            x += [k, v]; m += [0, 0]
        x.append(self.markers[1]); m.append(0)
        while len(x) < T:
            j = int(self.rng.integers(0, n))
            x += [keys[j], vals[j]]; m += [0, 1]
        return np.array(x[:T]), np.array(m[:T])

    def copy(self):
        T = self.T
        L = int(self.rng.integers(16, (T - 3) // 2))
        span = self._rand(L)
        x = [self.markers[2]] + list(span) + [self.markers[3]] + list(span)
        m = [0] * (L + 2) + [1] * L
        x += list(self._rand(T - len(x))); m += [0] * (T - len(m))
        return np.array(x[:T]), np.array(m[:T])

    def induction(self):
        T = self.T
        x = list(self._rand(T)); x[0] = self.markers[4]
        m = [0] * T
        for _ in range(int(self.rng.integers(4, 16))):
            i = int(self.rng.integers(1, T // 2)); j = int(self.rng.integers(i + 2, T - 1))
            x[j] = x[i]; x[j + 1] = x[i + 1]; m[j + 1] = 1
        return np.array(x), np.array(m)

    def dyck(self):
        T = self.T
        opens, closes = self.markers[8:16], self.markers[16:24]
        x, m, stack = [self.markers[5]], [0], []
        while len(x) < T:
            if stack and (len(stack) >= 8 or self.rng.random() < 0.45 or len(x) + len(stack) >= T):
                t = stack.pop(); x.append(closes[t]); m.append(1)
            else:
                t = int(self.rng.integers(0, 8)); stack.append(t); x.append(opens[t]); m.append(0)
        return np.array(x[:T]), np.array(m[:T])

    def batch(self, B, tasks=TASKS):
        xs, ms, kinds = [], [], []
        for _ in range(B):
            k = tasks[int(self.rng.integers(0, len(tasks)))]
            x, m = getattr(self, k)()
            xs.append(x); ms.append(m); kinds.append(k)
        x = np.stack(xs).astype(np.int32); m = np.stack(ms).astype(np.float32)
        return x, m, kinds

    def vocab(self):
        """Loss vocabulary: the pool plus markers (targets are always in it)."""
        return np.concatenate([self.pool, self.markers]).astype(np.int32)
