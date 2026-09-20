# Campaign harness

`run_queue.sh` / `gates.py` — the queue runner and gate evaluator. Gates are
evaluated, not printed: a job that cannot evaluate its gate reports NOT
EVALUATED and exits non-zero rather than passing.

`run_queue_v2.sh` — adds the GPU sharing protocol three sessions used on this
machine on 2026-09-20, and a per-spec `LEASE_CEILING` so a job cannot hold the
GPU past its promised window.

## GPU sharing protocol

Ordering is by arrival, not by poll rate, so a session that polls politely is
not punished.

- Waiter file `/Users/Shared/mlxuag/gpu.lock.waiters/<session>.<label>.<pid>.<unix-ts>`.
  Timestamp last, so `${f##*.}` parses it even when the label contains dots;
  pid second-to-last.
- Register BEFORE polling. Do not take the lock while any non-stale waiter has
  an earlier timestamp. Acquire first, THEN remove your waiter; also remove it
  on every other exit from the wait loop, through one cleanup path.
- Three independent staleness signals: a pid failing `kill -0` is stale
  immediately; an mtime older than 300s is stale regardless of liveness (so
  re-touch your waiter on every poll while you are genuinely waiting); a
  filename timestamp older than 90 minutes is stale.
- Cap deference to any single earlier waiter at ~20 minutes, then override with
  a loud log naming the entry bypassed. The heartbeat should make this
  unreachable; it exists for the case nobody thought of.
- Release between specs with a yield gap, and reap only your own descendants
  when sweeping stray listeners.

Why the heartbeat: liveness proves a process exists, not that it still wants
the GPU. A live pid that had stopped waiting blocked all three sessions on the
protocol's first day.
