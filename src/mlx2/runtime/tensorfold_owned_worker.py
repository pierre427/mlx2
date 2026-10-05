"""Isolated, default-off TensorFold Qwen3.8 B1 execution proof.

The native TensorFold installer replaces process-wide MLX callables.  This
module is therefore an executable worker, never an in-process serving import.
Its JSON line protocol carries token IDs and receipts; model caches stay here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

EXPECTED_REVISION = "71377a5373ed7b394f1b480ba2a6a3986b03af1c"
EXPECTED_MLX_LM_REVISION = "1104ced19ed98800bdaf4ebcdca14bbdeb597c23"


def source_identity(root: str | Path) -> dict:
    """Pin the checkout content before importing any TensorFold code."""

    root = Path(root).expanduser().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != EXPECTED_REVISION:
        raise ValueError(f"TensorFold source revision {head} is not {EXPECTED_REVISION}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True
    )
    if dirty:
        raise ValueError("TensorFold source checkout must be clean")
    files = (
        "src/tensorfold/families/qwen3_5/__init__.py",
        "src/tensorfold/families/qwen3_5/family.py",
        "src/tensorfold/kernels/qwen/dense/v1/lane_tree.py",
        "src/tensorfold/kernels/qwen/dense/v1/lane_multi.py",
        "src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py",
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return {"revision": head, "source_digest": digest.hexdigest(), "root": str(root)}


def mlx_lm_identity(root: str | Path) -> dict:
    """Bind the separate mlx_lm loader/model source used only by the worker."""

    root = Path(root).expanduser().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != EXPECTED_MLX_LM_REVISION:
        raise ValueError(f"mlx_lm source revision {head} is not {EXPECTED_MLX_LM_REVISION}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True
    )
    if dirty:
        raise ValueError("mlx_lm source checkout must be clean")
    digest = hashlib.sha256()
    for name in ("mlx_lm/__init__.py", "mlx_lm/utils.py", "mlx_lm/models/qwen3_5.py",
                 "mlx_lm/models/qwen3_next.py"):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return {"revision": head, "source_digest": digest.hexdigest(), "root": str(root)}


def prepare_sources(source: str, mlx_lm_source: str) -> dict:
    """Bind and import the exact full native loader path, without model IO."""

    source_info, mlx_lm_info = source_identity(source), mlx_lm_identity(mlx_lm_source)
    sys.path.insert(0, mlx_lm_info["root"])
    sys.path.insert(0, str(Path(source_info["root"]) / "src"))
    from tensorfold.families.qwen3_5 import load as tensorfold_load
    from mlx_lm import load as mlx_lm_load
    import mlx_lm
    import tensorfold

    if not Path(mlx_lm.__file__).resolve().is_relative_to(Path(mlx_lm_info["root"])):
        raise RuntimeError("mlx_lm imported from outside the pinned worker source")
    if not Path(tensorfold.__file__).resolve().is_relative_to(Path(source_info["root"])):
        raise RuntimeError("TensorFold imported from outside the pinned worker source")
    if not callable(tensorfold_load) or not callable(mlx_lm_load):
        raise RuntimeError("native Qwen3.8 loader path is unavailable")
    return {"tensorfold": source_info, "mlx_lm": mlx_lm_info}


def artifact_identity(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    index_path = path / "model.safetensors.index.json"
    metadata = ["config.json"] + ([index_path.name] if index_path.is_file() else [])
    for name in metadata:
        payload = (path / name).read_bytes()
        digest.update(name.encode() + b"\0" + payload)
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("artifact requires a nonempty weight index")
        names = sorted(set(weight_map.values()))
    else:
        names = sorted(item.name for item in path.glob("*.safetensors"))
    if not names:
        raise ValueError("artifact has no safetensors weights")
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("artifact weight index contains an invalid shard path")
        digest.update(name.encode() + b"\0")
        with (path / name).open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return {"path": str(path), "content_digest": digest.hexdigest()}


def profile_identity(source: dict, target: dict, drafter: dict, mlx_lm: dict) -> str:
    """Separate this numerical/cache law from mlx2 ordinary and other pins."""

    payload = {
        "schema": "mlx2.tensorfold-owned-qwen38-b1.v1",
        "source_revision": source["revision"],
        "source_digest": source["source_digest"],
        "mlx_lm_revision": mlx_lm["revision"],
        "mlx_lm_digest": mlx_lm["source_digest"],
        "target_content": target["content_digest"],
        "drafter_content": drafter["content_digest"],
        "kernel_route": "native-tensorfold-lane-v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_tree_rows(tokens, parents, pending, exact_width):
    """Reject invalid parent order before a stateful native forward begins."""

    if (not isinstance(tokens, list) or not isinstance(parents, list)
            or not 1 <= len(tokens) <= exact_width or len(tokens) != len(parents)
            or tokens[0] != pending or parents[0] != -1):
        raise ValueError("tree rows must start with the pending root within exact width")
    for index, (token, parent) in enumerate(zip(tokens, parents)):
        if type(token) is not int or not 0 <= token < 248320:
            raise ValueError("tree contains an invalid Qwen3.8 token ID")
        if index and (type(parent) is not int or not 0 <= parent < index):
            raise ValueError("tree parents must precede their child")


def logical_state(item):
    """Read only the committed KV prefix; GDN state is wholly committed."""

    if (getattr(item, "keys", None) is not None
            and callable(getattr(item, "keys_and_values", None))):
        return item.keys_and_values()
    return getattr(item, "state", None)


def array_leaves(value, path=()):
    """Flatten nested cache planes without treating metadata as tensors."""

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from array_leaves(item, (*path, index))
    elif hasattr(value, "shape"):
        yield path, value


def logical_cache_exact(mx, left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if getattr(a, "keys", 0) is None and getattr(b, "keys", 0) is None:
            continue  # TensorFold DraftSlot is not target KV/GDN state.
        if getattr(a, "offset", None) != getattr(b, "offset", None):
            return False
        leaves_a, leaves_b = list(array_leaves(logical_state(a))), list(array_leaves(logical_state(b)))
        if len(leaves_a) != len(leaves_b):
            return False
        for (path_a, x), (path_b, y) in zip(leaves_a, leaves_b):
            if path_a != path_b or x.shape != y.shape or not bool(mx.array_equal(x, y).item()):
                return False
    return True


def set_live_round_policy(engine, streams):
    """Select one TensorFold cache law at every round boundary.

    A concurrent round uses the native shared ``hidden_rows`` path with one
    ordinary row per stream. Queued B1 head proposals are disposable: they
    have not changed target caches. The next singleton round may draft again.
    """

    active = [stream for stream in streams.values() if not stream.finished]
    if engine.pipelined or engine._inflight:
        # A queued token already advanced the target cache. Discarding it at
        # a width transition would break the per-stream cache law. This
        # Qwen3.8 profile requires TensorFold's non-pipelined family route.
        raise RuntimeError("live width policy requires no in-flight target forward")
    if len(active) > 8:
        raise ValueError("TensorFold-owned live profile is capped at eight streams")
    tree = len(active) == 1 and not getattr(active[0], "force_ordinary", False)
    for stream in active:
        stream.drafts = tree
        if not tree:
            engine._next.pop(stream.stream_id, None)
    if tree:
        return "b1_tree_eligible"
    return "b1_serial_ordinary" if len(active) == 1 else "b2plus_shared_ordinary"


def replay_lane_serial(family, prompt, token_count, engine_cls, stream_cls):
    """Use the *same native prompt prefill* as live LaneEngine admission."""

    reference = engine_cls(family, max_rows=1, max_draft=0)
    stream = stream_cls("serial-audit", list(prompt), token_count + 1,
                        sampling=None, drafts=False, retain=False)
    reference.add_stream(stream)
    while len(stream.emitted) < token_count:
        reference.step()
    if len(stream.emitted) != token_count or stream.finished:
        raise RuntimeError("native serial audit did not reach an active cache boundary")
    cache = next((cache for current, cache in reference._live if current is stream), None)
    if cache is None:
        raise RuntimeError("native serial audit lost its cache")
    return stream, cache


def logical_cache_differences(mx, left, right, *, limit=4):
    """Give the first visible cache leaf mismatch without inspecting KV slack."""

    differences = []
    if len(left) != len(right):
        differences.append({"kind": "cache_length", "left": len(left), "right": len(right)})
    for layer, (a, b) in enumerate(zip(left, right)):
        if getattr(a, "keys", 0) is None and getattr(b, "keys", 0) is None:
            continue
        if getattr(a, "offset", None) != getattr(b, "offset", None):
            differences.append({"layer": layer, "kind": "offset",
                                "serial": getattr(a, "offset", None),
                                "live": getattr(b, "offset", None)})
        parts_a = list(array_leaves(logical_state(a)))
        parts_b = list(array_leaves(logical_state(b)))
        if len(parts_a) != len(parts_b):
            differences.append({"layer": layer, "kind": "structure",
                                "serial_paths": [list(p) for p, _ in parts_a],
                                "live_paths": [list(p) for p, _ in parts_b]})
        for (pa, x), (pb, y) in zip(parts_a, parts_b):
            if pa != pb or x.shape != y.shape or not bool(mx.array_equal(x, y).item()):
                detail = {"layer": layer, "kind": "array", "cache_type": type(a).__name__,
                          "serial_path": list(pa), "live_path": list(pb),
                          "serial_shape": list(x.shape), "live_shape": list(y.shape)}
                if pa == pb and x.shape == y.shape:
                    detail["first_flat"] = int(mx.argmax((x != y).reshape(-1)).item())
                    detail["max_abs"] = float(mx.max(mx.abs(
                        x.astype(mx.float32) - y.astype(mx.float32))).item())
                differences.append(detail)
                if len(differences) >= limit:
                    return differences
    return differences[:limit]


class TensorfoldOwnedBackend:
    """One TensorFold model and independent single-request cache handles."""

    def __init__(self, source: str, target: str, drafter: str, mlx_lm_source: str):
        sources = prepare_sources(source, mlx_lm_source)
        source_info, mlx_lm_info = sources["tensorfold"], sources["mlx_lm"]
        target_info, draft_info = artifact_identity(target), artifact_identity(drafter)
        self.identity = profile_identity(source_info, target_info, draft_info, mlx_lm_info)
        from tensorfold.families.qwen3_5 import load

        # The installer mutates global MLX functions, confined to this worker.
        with contextlib.redirect_stdout(sys.stderr):
            self.family, _ = load(Path(target), lane_kernels="auto", drafter=drafter)
        self.sessions: dict[str, dict] = {}
        self.live_streams: dict[str, object] = {}
        self.live_engine = None
        self.next_id = 0
        self.mx = __import__("mlx.core", fromlist=["core"])

    def open(self, prompt: list[int]) -> dict:
        if any(not stream.finished for stream in self.live_streams.values()):
            raise RuntimeError("diagnostic sessions cannot enter an active live round")
        if not prompt or any(type(t) is not int or t < 0 for t in prompt):
            raise ValueError("prompt must contain nonnegative integer token IDs")
        cache = self.family.make_cache()
        if len(prompt) > 1:
            hidden = self.family.prefill(self.mx.array([prompt[:-1]]), cache)
            self.mx.eval(hidden)
        self.next_id += 1
        sid = str(self.next_id)
        self.sessions[sid] = {"cache": cache, "pending": prompt[-1], "position": len(prompt) - 1}
        return {"session": sid, "cache_layout": f"qwen38-tensorfold-owned:{self.identity}"}

    def ping(self) -> dict:
        return {"identity": self.identity, "exact_width": int(self.family.exact_width),
                "cache_layout": f"qwen38-tensorfold-owned:{self.identity}"}

    def live_open(self, prompt: list[int], max_new_tokens: int, route: str = "auto",
                  eos_ids: list[int] | None = None) -> dict:
        """Admit one greedy token-only request into the isolated native engine."""

        if self.sessions:
            raise RuntimeError("live rounds cannot share the family with diagnostic sessions")
        if (not isinstance(prompt, list) or not prompt
                or any(type(t) is not int or not 0 <= t < 248320 for t in prompt)):
            raise ValueError("live prompt requires Qwen3.8 token IDs")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
            raise ValueError("live max_new_tokens must be in [1, 256]")
        if route not in {"auto", "serial"}:
            raise ValueError("live route must be auto or serial")
        if (eos_ids is not None and (not isinstance(eos_ids, list)
                or any(type(token) is not int or not 0 <= token < 248320
                       for token in eos_ids))):
            raise ValueError("live EOS IDs must be Qwen3.8 token IDs")
        if sum(not stream.finished for stream in self.live_streams.values()) >= 8:
            raise ValueError("TensorFold-owned live profile is capped at eight streams")
        from tensorfold.engine.lane_engine import LaneEngine, LaneStream

        if self.live_engine is None:
            self.live_engine = LaneEngine(self.family, max_rows=128, max_draft=15)
            if self.live_engine.pipelined:
                self.live_engine = None
                raise RuntimeError("native Qwen3.8 live profile requires non-pipelined rounds")
        self.next_id += 1
        sid = f"live-{self.next_id}"
        stream = LaneStream(sid, prompt, max_new_tokens,
                            eos_ids=frozenset(eos_ids or ()),
                            sampling=None, drafts=True, retain=False)
        stream.force_ordinary = route == "serial"
        stream.drafts = not stream.force_ordinary
        prefill_start_ns = time.monotonic_ns()
        self.live_engine.add_stream(stream)
        prefill_end_ns = time.monotonic_ns()
        self.live_streams[sid] = stream
        return {"session": sid, "tokens": list(stream.emitted), "finished": stream.finished,
                "prefill_start_ns": prefill_start_ns, "prefill_end_ns": prefill_end_ns,
                "first_token_ns": prefill_end_ns if stream.emitted else None,
                "prompt_tokens": len(prompt), "cached_prompt_tokens": int(stream.cached_tokens),
                "route_requested": route,
                "finish_reason": stream.finish_reason, "cache_layout": f"qwen38-tensorfold-owned:{self.identity}",
                "cache_policy": "disabled", "qualified": False,
                "apcv2_lookup": False, "apcv2_store": False}

    def live_step(self) -> dict:
        if self.live_engine is None:
            raise RuntimeError("no TensorFold-owned live stream has been opened")
        mode = set_live_round_policy(self.live_engine, self.live_streams)
        if not any(not stream.finished for stream in self.live_streams.values()):
            return {"mode": "idle", "tokens": {}, "rounds": 0, "drafted": 0, "accepted": 0,
                    "cache_policy": "disabled", "qualified": False}
        before = (self.live_engine.drafted, self.live_engine.accepted,
                  len(self.live_engine.round_stats))
        per_stream_before = {sid: (stream.drafted, stream.accepted)
                             for sid, stream in self.live_streams.items() if not stream.finished}
        landed = self.live_engine.step()
        token_ready_ns = time.monotonic_ns()
        per_stream = {sid: {"drafted": self.live_streams[sid].drafted - counts[0],
                            "accepted": self.live_streams[sid].accepted - counts[1]}
                      for sid, counts in per_stream_before.items()}
        return {"mode": mode, "tokens": landed, "token_ready_ns": token_ready_ns,
                "active_width": len(per_stream_before),
                "finished": {sid: stream.finished for sid, stream in self.live_streams.items()},
                "finish_reasons": {sid: stream.finish_reason for sid, stream in self.live_streams.items()},
                "rounds": len(self.live_engine.round_stats) - before[2],
                "drafted": self.live_engine.drafted - before[0],
                "accepted": self.live_engine.accepted - before[1],
                "per_stream": per_stream,
                "cache_policy": "disabled", "qualified": False,
                "apcv2_lookup": False, "apcv2_store": False}

    def live_close(self, session: str) -> dict:
        stream = self.live_streams.pop(session)
        if not stream.finished:
            self.live_engine.discard_stream(stream)
        else:
            self.live_engine.finished_caches.pop(session, None)
            if stream in self.live_engine.streams:
                self.live_engine.streams.remove(stream)
        return {"closed": session}

    def live_compare_serial(self, session: str) -> dict:
        """Research oracle: replay emitted IDs in a separate native serial cache."""

        from tensorfold.engine.lane_engine import LaneEngine, LaneStream

        stream = self.live_streams[session]
        if stream.finished:
            raise ValueError("live serial audit requires an active stream")
        live_cache = next((cache for row, cache in self.live_engine._live if row is stream), None)
        if live_cache is None:
            raise RuntimeError("active native stream has no worker cache")
        serial_stream, serial_cache = replay_lane_serial(
            self.family, stream.prompt_ids, len(stream.emitted), LaneEngine, LaneStream)
        token_exact = serial_stream.emitted == stream.emitted
        cache_exact = logical_cache_exact(self.mx, serial_cache, live_cache)
        first_differences = logical_cache_differences(self.mx, serial_cache, live_cache)
        pending = stream.emitted[-1]
        serial_next = self.family.head(self.family.hidden(
            self.mx.array([[pending]]), LaneEngine.copy_single_cache(serial_cache)))
        live_next = self.family.head(self.family.hidden(
            self.mx.array([[pending]]), LaneEngine.copy_single_cache(live_cache)))
        next_logits_exact = bool(self.mx.array_equal(serial_next, live_next).item())
        return {"session": session, "tokens": len(stream.emitted),
                "serial_tokens_exact": token_exact, "cache_exact": cache_exact,
                "first_differences": first_differences,
                "next_logits_exact": next_logits_exact,
                "exact": token_exact and cache_exact and next_logits_exact}

    def serial(self, session: str) -> dict:
        row = self.sessions[session]
        hidden = self.family.hidden(self.mx.array([[row["pending"]]]), row["cache"])
        logits = self.family.head(hidden)
        token = int(self.mx.argmax(logits[0, 0]).item())
        self.mx.eval(logits)
        row["pending"] = token
        row["position"] += 1
        return {"token": token, "position": row["position"]}

    def tree(self, session: str, tokens: list[int], parents: list[int]) -> dict:
        row = self.sessions[session]
        validate_tree_rows(tokens, parents, row["pending"], self.family.exact_width)
        from tensorfold.kernels.qwen.dense.v1.lane_tree import accept_path

        hidden = self.family.hidden(self.mx.array([tokens]), row["cache"], parents)
        logits = self.family.head(hidden)
        picks = [int(t) for t in self.mx.argmax(logits[0], axis=-1).tolist()]
        path = accept_path(tokens, parents, picks)
        self.family.keep_rows(row["cache"], len(tokens), path)
        # Every accepted row was computed and committed; the last prediction
        # is the next pending root, exactly as one-row serial decoding.
        emitted = [picks[index] for index in path]
        row["pending"] = emitted[-1]
        row["position"] += len(path)
        return {"tokens": emitted, "accepted_rows": path, "position": row["position"]}

    def compare_round(self, session: str, tokens: list[int], parents: list[int]) -> dict:
        """Independent serial/tree copies from one native cache boundary."""

        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.family_common import cache_arrays
        from tensorfold.kernels.qwen.dense.v1.lane_tree import accept_path

        original = self.sessions[session]
        validate_tree_rows(tokens, parents, original["pending"], self.family.exact_width)
        serial_cache = LaneEngine.copy_single_cache(original["cache"])
        tree_cache = LaneEngine.copy_single_cache(original["cache"])
        hidden = self.family.hidden(self.mx.array([tokens]), tree_cache, parents)
        tree_logits = self.family.head(hidden)
        picks = [int(t) for t in self.mx.argmax(tree_logits[0], axis=-1).tolist()]
        path = accept_path(tokens, parents, picks)
        self.family.keep_rows(tree_cache, len(tokens), path)
        logit_exact = []
        for index in path:
            step_hidden = self.family.hidden(self.mx.array([[tokens[index]]]), serial_cache)
            step_logits = self.family.head(step_hidden)
            logit_exact.append(bool(self.mx.array_equal(step_logits[0, 0], tree_logits[0, index]).item()))
        serial_arrays, tree_arrays = cache_arrays(serial_cache), cache_arrays(tree_cache)
        differences = []
        raw_differences = []
        array_mismatches = 0
        raw_backing_mismatches = 0
        offset_mismatches = 0
        for layer, (serial_item, tree_item) in enumerate(zip(serial_cache, tree_cache)):
            if getattr(serial_item, "keys", 0) is None and getattr(tree_item, "keys", 0) is None:
                continue
            serial_offset = getattr(serial_item, "offset", None)
            tree_offset = getattr(tree_item, "offset", None)
            if serial_offset != tree_offset:
                offset_mismatches += 1
                if len(differences) < 24:
                    differences.append({"layer": layer, "kind": "offset",
                                        "serial": serial_offset, "tree": tree_offset})
            serial_raw, tree_raw = getattr(serial_item, "state", None), getattr(tree_item, "state", None)
            # KVCache.state exposes its 256-row allocation, including rows
            # beyond offset. tree_forward wrote rejected proposals there;
            # serial steps left zeros. Only keys_and_values() is readable
            # committed state, as used by the ordinary attention forward.
            serial_state, tree_state = logical_state(serial_item), logical_state(tree_item)
            if getattr(serial_item, "keys", None) is not None:
                for (plane, a), (tree_plane, b) in zip(array_leaves(serial_raw), array_leaves(tree_raw)):
                    if a.shape != b.shape or not bool(self.mx.array_equal(a, b).item()):
                        raw_backing_mismatches += 1
                        if len(raw_differences) < 4:
                            detail = {"layer": layer, "plane": list(plane),
                                      "tree_plane": list(tree_plane),
                                      "cache_type": type(serial_item).__name__,
                                      "serial_shape": list(a.shape), "tree_shape": list(b.shape),
                                      "serial_offset": serial_offset, "tree_offset": tree_offset}
                            if a.shape == b.shape:
                                flat = int(self.mx.argmax((a != b).reshape(-1)).item())
                                coords = []
                                for size in reversed(a.shape):
                                    coords.append(flat % int(size))
                                    flat //= int(size)
                                detail["first_index"] = list(reversed(coords))
                                detail["max_abs"] = float(self.mx.max(self.mx.abs(
                                    a.astype(self.mx.float32) - b.astype(self.mx.float32))).item())
                            raw_differences.append(detail)
            serial_parts, tree_parts = list(array_leaves(serial_state)), list(array_leaves(tree_state))
            if len(serial_parts) != len(tree_parts):
                array_mismatches += 1
                if len(differences) < 24:
                    differences.append({"layer": layer, "kind": "structure",
                                        "cache_type": type(serial_item).__name__,
                                        "serial_paths": [list(p) for p, _ in serial_parts],
                                        "tree_paths": [list(p) for p, _ in tree_parts]})
            for (plane, a), (tree_plane, b) in zip(serial_parts, tree_parts):
                same_shape = plane == tree_plane and a.shape == b.shape
                same = same_shape and bool(self.mx.array_equal(a, b).item())
                if same:
                    continue
                array_mismatches += 1
                if len(differences) < 24:
                    detail = {"layer": layer, "plane": list(plane),
                              "tree_plane": list(tree_plane),
                              "cache_type": type(serial_item).__name__,
                              "serial_shape": list(a.shape) if hasattr(a, "shape") else None,
                              "tree_shape": list(b.shape) if hasattr(b, "shape") else None,
                              "serial_offset": serial_offset, "tree_offset": tree_offset}
                    if same_shape:
                        detail["max_abs"] = float(self.mx.max(self.mx.abs(
                            a.astype(self.mx.float32) - b.astype(self.mx.float32))).item())
                        detail["first_flat"] = int(self.mx.argmax(
                            (a != b).reshape(-1)).item())
                    differences.append(detail)
        state_exact = (len(serial_arrays) == len(tree_arrays)
                       and len(serial_cache) == len(tree_cache)
                       and array_mismatches == 0 and offset_mismatches == 0
                       and logical_cache_exact(self.mx, serial_cache, tree_cache))
        continuation = None
        if state_exact and all(logit_exact):
            next_token = picks[path[-1]]
            serial_next = self.family.head(self.family.hidden(
                self.mx.array([[next_token]]), serial_cache))
            tree_next = self.family.head(self.family.hidden(
                self.mx.array([[next_token]]), tree_cache))
            continuation = {
                "input_token": next_token,
                "logits_exact": bool(self.mx.array_equal(serial_next, tree_next).item()),
                "visible_cache_exact": logical_cache_exact(self.mx, serial_cache, tree_cache),
            }
        return {"accepted_rows": path, "logit_rows_exact": logit_exact,
                "cache_arrays": len(serial_arrays), "cache_exact": state_exact,
                "array_mismatches": array_mismatches,
                "raw_backing_mismatches": raw_backing_mismatches,
                "first_raw_backing_differences": raw_differences,
                "offset_mismatches": offset_mismatches,
                "first_differences": differences,
                "next_serial_step": continuation,
                "exact": all(logit_exact) and state_exact and continuation is not None
                         and continuation["logits_exact"] and continuation["visible_cache_exact"]}

    def close(self, session: str) -> dict:
        del self.sessions[session]
        return {"closed": session}


class FakeBackend:
    """CPU protocol fixture only; cannot load model weights."""

    def __init__(self):
        self.identity = "fixture"
        self.sessions = {}
        self.live_streams = {}

    def open(self, prompt):
        sid = str(len(self.sessions) + 1)
        self.sessions[sid] = int(prompt[-1])
        return {"session": sid, "cache_layout": "qwen38-tensorfold-owned:fixture"}

    def ping(self):
        return {"identity": self.identity, "exact_width": 16,
                "cache_layout": "qwen38-tensorfold-owned:fixture"}

    def serial(self, session):
        self.sessions[session] += 1
        return {"token": self.sessions[session]}

    def tree(self, session, tokens, parents):
        self.sessions[session] += 1
        return {"tokens": [self.sessions[session]], "accepted_rows": [0]}

    def compare_round(self, session, tokens, parents):
        return {"accepted_rows": [0], "logit_rows_exact": [True], "cache_arrays": 1,
                "cache_exact": True, "exact": True}

    def close(self, session):
        del self.sessions[session]
        return {"closed": session}

    def live_open(self, prompt, max_new_tokens, route="auto", eos_ids=None):
        if not prompt or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
            raise ValueError("invalid live admission")
        if route not in {"auto", "serial"}:
            raise ValueError("invalid live route")
        prefill_start_ns = time.monotonic_ns()
        sid = f"live-{len(self.live_streams) + 1}"
        self.live_streams[sid] = {"token": prompt[-1], "remaining": max_new_tokens}
        prefill_end_ns = time.monotonic_ns()
        return {"session": sid, "tokens": [], "finished": False, "route_requested": route,
                "prefill_start_ns": prefill_start_ns,
                "prefill_end_ns": max(prefill_end_ns, prefill_start_ns + 1),
                "first_token_ns": None, "prompt_tokens": len(prompt),
                "cached_prompt_tokens": 0,
                "cache_layout": "qwen38-tensorfold-owned:fixture",
                "cache_policy": "disabled", "qualified": False,
                "apcv2_lookup": False, "apcv2_store": False}

    def live_step(self):
        mode = "b1_tree_eligible" if len(self.live_streams) == 1 else "b2plus_shared_ordinary"
        landed = {}
        for sid, row in self.live_streams.items():
            row["token"] += 1
            row["remaining"] -= 1
            landed[sid] = [row["token"]]
        return {"mode": mode, "tokens": landed, "token_ready_ns": time.monotonic_ns(),
                "cache_policy": "disabled", "qualified": False,
                "apcv2_lookup": False, "apcv2_store": False}

    def live_close(self, session):
        del self.live_streams[session]
        return {"closed": session}

    def live_compare_serial(self, session):
        if session not in self.live_streams:
            raise KeyError(session)
        return {"session": session, "tokens": 1, "serial_tokens_exact": True,
                "cache_exact": True, "next_logits_exact": True, "exact": True}


def serve(backend) -> None:
    handlers = {"ping": backend.ping, "open": backend.open, "serial": backend.serial,
                "tree": backend.tree, "compare_round": backend.compare_round, "close": backend.close,
                "live_open": backend.live_open, "live_step": backend.live_step,
                "live_close": backend.live_close,
                "live_compare_serial": backend.live_compare_serial}
    protocol_out = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            name = request["method"]
            if name not in handlers or not isinstance(request.get("params"), dict):
                raise ValueError("unknown worker method or invalid parameters")
            with contextlib.redirect_stdout(sys.stderr):
                result = handlers[name](**request["params"])
            response = {"ok": True, "result": result}
        except Exception as exc:  # isolated request boundary; no tensor state crosses IPC
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        protocol_out.write(json.dumps(response, separators=(",", ":")) + "\n")
        protocol_out.flush()


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--drafter")
    parser.add_argument("--mlx-lm-source")
    parser.add_argument("--fixture", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.import_smoke:
        if not args.source or not args.mlx_lm_source:
            parser.error("import smoke requires both source checkouts")
        print(json.dumps(prepare_sources(args.source, args.mlx_lm_source), sort_keys=True))
        return
    if args.fixture:
        backend = FakeBackend()
    else:
        if not all((args.source, args.target, args.drafter, args.mlx_lm_source)):
            parser.error("source, target, drafter and mlx_lm source are required")
        backend = TensorfoldOwnedBackend(args.source, args.target, args.drafter,
                                         args.mlx_lm_source)
    serve(backend)


if __name__ == "__main__":
    main()
