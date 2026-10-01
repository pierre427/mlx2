#!/usr/bin/env python3
"""Flash-Next: is greedy MTP-on output byte-identical to MTP-off output?

GPU-only (loads the real model once).  Refuses to run without
``--i-own-the-gpu``; ``--dry-run`` prints the plan.

Arms, all in one process on one loaded model, each on fresh caches:

* ``off``       ordinary one-token decode (the reference)
* ``mtp``       native self-MTP at the policy depth (3-row windows by default)
                with the route's copy drafts (9..17-row windows on copies)
* ``oracle``    ``mtp`` with the shape-stable per-token forward switches
                (``MLX_QWEN4_SHAPE_STABLE_SHORT_FORWARD`` +
                ``MLX_QWEN4_GDN_SHAPE_STABLE_PROJECTIONS``) flipped on
* ``rowexact``  ``mtp`` with the row-exact verify route (policy
                ``row_exact_verify``) switched on

For every prompt the report gives the first token where an arm's output
differs from ``off`` and, for every verify row whose input prefix still equals
the reference (rows past a rejected draft are excluded), whether its logits
are bit-identical to the one-token logits at the same position.  Rows that
differ are attributed to the first component whose output bits differ, in
forward order (per layer: ple, attn_hc, mixer, mlp_hc, mlp, out; then the
final HC mixer and the LM head), like omlx's ``mtp-window-check``.

Component outputs are compared by a per-row digest (blake2b of the bf16 bits),
so the check costs one host transfer per captured tensor; do not quote the
tok/s of a checked run.  ``--no-components`` compares logits only.

  scratchpad/gpuq.sh l5-check PYTHONPATH=src .venv/bin/python \\
      scripts/check_mtp_row_exact.py --i-own-the-gpu \\
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --out run.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_MODULE = '''class AdaptiveLookback:
    """Miss-driven lookback with rejection backoff inside a scheduler cap."""

    def __init__(self, ladder=(256, 1024, 4096, 16384), *, misses=4, rejects=2):
        self.ladder = tuple(int(value) for value in ladder)
        self.index = 0
        self.cap = self.ladder[-1]
        self.misses_to_widen = int(misses)
        self.rejects_to_narrow = int(rejects)
        self._misses = self._rejects = 0
        self.widen_events = self.narrow_events = 0

    @property
    def current(self):
        return min(self.ladder[self.index], self.cap)

    def observe(self, proposed, accepted):
        if not proposed:
            self._rejects = 0
            self._misses += 1
            if self._misses >= self.misses_to_widen:
                if self.index < len(self.ladder) - 1:
                    self.index += 1
                    self.widen_events += 1
                self._misses = 0
        elif not accepted:
            self._misses = 0
            self._rejects += 1
            if self._rejects >= self.rejects_to_narrow:
                if self.index:
                    self.index -= 1
                    self.narrow_events += 1
                self._rejects = 0
        else:
            self._misses = self._rejects = 0
'''

PROMPTS = {
    "prose": "Explain how a database transaction works, in numbered sections.",
    "code": "Write a Python function that parses an ISO 8601 duration string "
    "such as P3DT4H12M into total seconds, with a docstring and three doctests.",
    "copy": f"Here is a Python class:\n\n```python\n{_MODULE}```\n\nReturn the whole "
    "class unchanged except rename `_misses` to `_miss_count` everywhere. "
    "Output only the code.",
    "reason": "A train leaves at 09:40 and travels 212 km at 84 km/h, then waits "
    "11 minutes and travels 95 km at 76 km/h. When does it arrive? Show the "
    "arithmetic step by step.",
}

COMPONENTS = ("ple", "attn_hc", "mixer", "mlp_hc", "mlp", "out")
# Captured per-row digests are skipped for forwards wider than this (prefill).
CAPTURE_MAX_ROWS = 32


def _digest_rows(array):
    """One digest per row of a [1, R, D] (or [R, D]) array's raw bits."""
    import mlx.core as mx
    import numpy as np

    rows = array.reshape(-1, array.shape[-1])
    if rows.dtype in (mx.bfloat16, mx.float16):
        raw = np.array(rows.view(mx.uint16))
    else:
        raw = np.array(rows)
    return [hashlib.blake2b(row.tobytes(), digest_size=12).hexdigest() for row in raw]


class Recorder:
    """Per trunk forward: input ids, per-component row digests, logits rows."""

    def __init__(self, *, components: bool):
        self.components = components
        self.active = False
        self.forwards = []
        self._depth = 0
        self._current = None
        self._in_mtp = 0

    # -- trunk --------------------------------------------------------------
    def begin_trunk(self, inputs):
        self._depth += 1
        if self._depth > 1 or not self.active:
            return
        ids = [int(v) for v in inputs.reshape(-1).tolist()]
        self._current = {
            "ids": ids,
            "capture": self.components and len(ids) <= CAPTURE_MAX_ROWS,
            "pending": [],
            "digests": {},
            "logits": None,
        }

    def end_trunk(self, output):
        self._depth -= 1
        if self._depth or self._current is None:
            return
        current = self._current
        if current["capture"]:
            mixed = output[0] if isinstance(output, tuple) else output
            current["pending"].append(("final", mixed))
            self._flush(current)
        self.forwards.append(current)

    def capture(self, key, array):
        current = self._current
        if current is None or not current["capture"] or self._in_mtp:
            return
        current["pending"].append((key, array))

    def _flush(self, current):
        import mlx.core as mx

        pending, current["pending"] = current["pending"], []
        if not pending:
            return
        mx.eval([array for _, array in pending])
        for key, array in pending:
            current["digests"].setdefault(key, []).extend(_digest_rows(array))

    def attach_logits(self, logits):
        """Attach LM-head rows to the most recent trunk forward."""
        import mlx.core as mx
        import numpy as np

        if not self.active or self._in_mtp or not self.forwards:
            return
        current = self.forwards[-1]
        if current["logits"] is not None:
            return
        rows = logits.reshape(-1, logits.shape[-1])
        if rows.shape[0] > CAPTURE_MAX_ROWS:
            rows = rows[-1:]
        mx.eval(rows)
        current["logits"] = np.array(rows.view(mx.uint16)) if rows.dtype == mx.bfloat16 else np.array(rows)

    def close_trunk_logits(self):
        self._current = None


def install_hooks(model, recorder):
    """Wrap the trunk, decoder layers, LM head and MTP step (this process only)."""
    from mlx2.runtime.models import qwen4_exp as Q

    text_model_cls = Q.Qwen4ExpTextModel
    original_trunk = text_model_cls.__call__

    def trunk(self, inputs, cache=None, input_embeddings=None, return_hyper=False):
        if self is not model.language_model.model:
            return original_trunk(self, inputs, cache, input_embeddings, return_hyper)
        recorder.begin_trunk(inputs)
        try:
            out = original_trunk(self, inputs, cache, input_embeddings, return_hyper)
        finally:
            pass
        recorder.end_trunk(out)
        return out

    text_model_cls.__call__ = trunk
    trunk_layers = {id(layer): index for index, layer in enumerate(model.language_model.model.layers)}

    def layer_call(self, x, input_ids, mask=None, cache=None, ssm_mask=None):
        index = trunk_layers.get(id(self))
        cap = (lambda name, value: recorder.capture((index, name), value)) if index is not None else (lambda *_: None)
        if self.ple is not None:
            delta = self.ple(x, input_ids, cache, ssm_mask)
            cap("ple", delta)
            x = x + delta
        (mixed, residual, inject) = self.attn_hyper_connection(x)
        cap("attn_hc", mixed)
        if self.is_linear:
            branch = self.linear_attn(mixed, ssm_mask, cache)
        else:
            branch = self.self_attn(mixed, mask, cache)
        cap("mixer", branch)
        x = Q._apply_inject(residual, branch, inject)
        (mixed, residual, inject) = self.mlp_hyper_connection(x)
        cap("mlp_hc", mixed)
        branch = self.mlp(mixed)
        cap("mlp", branch)
        out = Q._apply_inject(residual, branch, inject)
        cap("out", out)
        return out

    Q.DecoderLayer.__call__ = layer_call

    original_text_call = Q.TextModel.__call__

    def text_call(self, inputs, cache=None, input_embeddings=None):
        logits = original_text_call(self, inputs, cache, input_embeddings)
        recorder.attach_logits(logits)
        return logits

    Q.TextModel.__call__ = text_call
    original_logits = Q.Model.logits

    def logits_call(self, hidden):
        logits = original_logits(self, hidden)
        recorder.attach_logits(logits)
        return logits

    Q.Model.logits = logits_call
    original_mtp_step = Q.Model._mtp_step

    def mtp_step(self, *args, **kwargs):
        recorder._in_mtp += 1
        try:
            return original_mtp_step(self, *args, **kwargs)
        finally:
            recorder._in_mtp -= 1

    Q.Model._mtp_step = mtp_step


def align(forwards, sequence):
    """Assign absolute positions to recorded trunk rows.

    Before a forward the committed trunk state covers ``sequence[:pos]``; the
    forward's rows start at ``pos`` and the leading rows equal to the
    committed sequence are the ones kept.  Returns [(forward, start, kept)].
    """
    placed = []
    pos = 0
    anomalies = []
    for forward in forwards:
        ids = forward["ids"]
        start = pos
        if sequence[start : start + 1] != ids[:1]:
            # A replayed or re-forwarded row: find the latest earlier match.
            found = None
            for candidate in range(start - 1, max(-1, start - 64), -1):
                if sequence[candidate : candidate + len(ids[:1])] == ids[:1]:
                    found = candidate
                    break
            anomalies.append({"expected": start, "found": found, "rows": len(ids)})
            if found is None:
                continue
            start = found
        kept = 0
        while kept < len(ids) and start + kept < len(sequence) and ids[kept] == sequence[start + kept]:
            kept += 1
        placed.append((forward, start, kept))
        pos = start + kept
    return placed, anomalies


def row_table(placed):
    """position -> (digests per component key, logits row) for kept rows."""
    table = {}
    for forward, start, kept in placed:
        logits = forward["logits"]
        rows = len(forward["ids"])
        lrows = 0 if logits is None else logits.shape[0]
        for row in range(kept):
            entry = {"window": rows, "row": row, "digests": {}}
            for key, digests in forward["digests"].items():
                if row < len(digests):
                    entry["digests"][key] = digests[row]
            logit_row = row - (rows - lrows)
            if logits is not None and 0 <= logit_row < lrows:
                entry["logits"] = logits[logit_row]
            table[start + row] = entry
    return table


def _logit_stats(a, b):
    import numpy as np

    if a.dtype == np.uint16:
        af = (a.astype(np.uint32) << 16).view(np.float32)
        bf = (b.astype(np.uint32) << 16).view(np.float32)
    else:
        af, bf = a.astype(np.float32), b.astype(np.float32)
    return float(np.max(np.abs(af - bf)))


def _component_order(num_layers):
    order = []
    for layer in range(num_layers):
        order += [(layer, name) for name in COMPONENTS]
    return order + ["final"]


def compare(reference, candidate, prompt_len, num_layers):
    """Per-row equality of candidate verify rows against one-token decode."""
    first_div = None
    ref_seq, cand_seq = reference["sequence"], candidate["sequence"]
    for index in range(prompt_len, min(len(ref_seq), len(cand_seq))):
        if ref_seq[index] != cand_seq[index]:
            first_div = index
            break
    limit = first_div if first_div is not None else min(len(ref_seq), len(cand_seq))
    ref_rows, cand_rows = reference["rows"], candidate["rows"]
    order = _component_order(num_layers)
    by_width = {}
    first_component = {}
    worst = 0.0
    compared = equal = 0
    examples = []
    for position in range(prompt_len - 1, limit - 1):
        cand = cand_rows.get(position)
        ref = ref_rows.get(position)
        if cand is None or ref is None or "logits" not in cand or "logits" not in ref:
            continue
        compared += 1
        same = bool((cand["logits"] == ref["logits"]).all())
        width = cand["window"]
        bucket = by_width.setdefault(width, [0, 0])
        bucket[1] += 1
        if same:
            equal += 1
            bucket[0] += 1
            continue
        delta = _logit_stats(cand["logits"], ref["logits"])
        worst = max(worst, delta)
        culprit = "lm_head"
        for key in order:
            k = key if isinstance(key, str) else key
            a = cand["digests"].get(k)
            b = ref["digests"].get(k)
            if a is None or b is None:
                continue
            if a != b:
                culprit = f"L{key[0]}.{key[1]}" if isinstance(key, tuple) else key
                break
        name = culprit.split(".", 1)[-1] if culprit.startswith("L") else culprit
        first_component[name] = first_component.get(name, 0) + 1
        if len(examples) < 12:
            examples.append({"position": position, "window": width, "row": cand["row"],
                             "first_differing": culprit, "max_abs_dlogit": delta})
    return {
        "first_divergent_token": None if first_div is None else first_div - prompt_len,
        "identical_output": first_div is None and len(ref_seq) == len(cand_seq),
        "rows_compared": compared,
        "rows_bit_equal": equal,
        "rows_by_window": {str(k): {"equal": v[0], "compared": v[1]} for k, v in sorted(by_width.items())},
        "first_differing_component": first_component,
        "max_abs_dlogit": worst,
        "examples": examples,
    }


def _numeric_leaves(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_numeric_leaves(item, f"{prefix}{key}."))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix[:-1]] = value
    return out


def _diagnostics(adapter):
    try:
        diag = adapter.diagnostics()
    except Exception as exc:  # noqa: BLE001 - informational
        return {"error": repr(exc)}
    keep = {k: diag.get(k) for k in ("fused_gdn", "moe", "segmented_mtp", "round_levers")}
    return _numeric_leaves(keep)


def run_arm(adapter, recorder, prompt_ids, *, arm, num_draft, max_tokens, prefill_step, copy_policy):
    import mlx.core as mx
    from mlx2.runtime import generate as G
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    kwargs = dict(completion_batch_size=1, prefill_batch_size=1, prefill_step_size=prefill_step)
    mtp = arm != "off"
    if mtp:
        kwargs["self_mtp"] = adapter.policy.batch_config(max_lanes=1, prefill_step=prefill_step)
        kwargs["self_mtp"]["num_draft"] = num_draft
        if copy_policy is not None:
            kwargs["copy_draft"] = copy_policy
    stats = {}
    kwargs["scheduler_stats"] = stats
    gen = G.BatchGenerator(model, **kwargs)
    diag_before = _diagnostics(adapter)
    recorder.forwards = []
    recorder.active = True
    tokens = []
    t0 = time.perf_counter()
    try:
        insert = dict(max_tokens=[max_tokens], lane_rngs=[LaneRNG(1)])
        if mtp:
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}]
        (uid,) = gen.insert([list(prompt_ids)], **insert)
        done = False
        while not done:
            _p, responses = gen.next()
            for response in responses:
                tokens.append(int(response.token))
                if response.finish_reason:
                    done = True
    finally:
        recorder.active = False
        gen.close()
    elapsed = time.perf_counter() - t0
    diag_after = _diagnostics(adapter)
    diag_delta = {
        key: value - diag_before.get(key, 0)
        for key, value in diag_after.items()
        if isinstance(value, (int, float)) and value != diag_before.get(key, 0)
    }
    sequence = list(prompt_ids) + tokens
    placed, anomalies = align(recorder.forwards, sequence)
    rows = row_table(placed)
    windows = {}
    for forward, _start, kept in placed:
        if len(forward["ids"]) <= CAPTURE_MAX_ROWS and forward["ids"]:
            key = str(len(forward["ids"]))
            windows[key] = windows.get(key, 0) + 1
    recorder.forwards = []
    mx.clear_cache()
    return {
        "arm": arm,
        "tokens": tokens,
        "sequence": sequence,
        "rows": rows,
        "windows": windows,
        "alignment_anomalies": anomalies[:8],
        "alignment_anomaly_count": len(anomalies),
        "elapsed_s": elapsed,
        "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float)) and v},
        "diagnostics_delta": diag_delta,
    }


def set_arm(adapter, arm):
    """Flip the in-process switches an arm needs; return a restore callable."""
    from mlx2.runtime.models import qwen4_exp as Q

    saved = (Q._SHAPE_STABLE_SHORT_FORWARD, Q._GDN_SHAPE_STABLE_PROJECTIONS)
    row_exact = getattr(adapter, "row_exact_verify", None)
    if arm == "oracle":
        Q._SHAPE_STABLE_SHORT_FORWARD = True
        Q._GDN_SHAPE_STABLE_PROJECTIONS = True
    if arm == "rowexact":
        if row_exact is None:
            from mlx2.runtime.models.qwen4_row_exact import install

            row_exact = adapter.row_exact_verify = install(adapter.model)
        row_exact.enable(True)

    def restore():
        Q._SHAPE_STABLE_SHORT_FORWARD, Q._GDN_SHAPE_STABLE_PROJECTIONS = saved
        if row_exact is not None:
            row_exact.enable(False)

    return restore


def _device_info(mx):
    try:
        info = mx.metal.device_info()
        return {k: info[k] for k in ("architecture", "device_name") if k in info}
    except Exception:  # noqa: BLE001 - informational
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(Path("~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()))
    parser.add_argument("--arms", default="off,mtp,oracle")
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-draft", type=int, default=None)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--no-copy", action="store_true", help="MTP arms without copy drafts")
    parser.add_argument("--no-components", action="store_true")
    parser.add_argument("--execution-policy", default=None, help="JSON FlashNextPolicy mapping")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    names = [p for p in args.prompts.split(",") if p]
    if "off" not in arms:
        parser.error("the off arm is the reference and must be included")
    if args.dry_run:
        print(json.dumps({"arms": arms, "prompts": names, "max_tokens": args.max_tokens}))
        return 0
    if not args.i_own_the_gpu:
        parser.error("Metal run: pass --i-own-the-gpu under the GPU lock")

    import mlx.core as mx
    from mlx2.adapters.flash_next import FlashNextAdapter

    mx.set_cache_limit(4 << 30)
    policy = json.loads(args.execution_policy) if args.execution_policy else None
    t0 = time.perf_counter()
    adapter = FlashNextAdapter(args.model, execution_policy=policy)
    load_s = time.perf_counter() - t0
    num_draft = args.num_draft or adapter.policy.num_draft
    copy_policy = None
    if not args.no_copy:
        copy_policy = adapter.default_route_execution_policy["native_mtp"]["self_mtp_copy_draft"]
    recorder = Recorder(components=not args.no_components)
    install_hooks(adapter.model, recorder)
    num_layers = len(adapter.model.language_model.model.layers)
    report = {
        "schema": "mlx2.check-mtp-row-exact.v1",
        "model": args.model,
        "mlx": mx.__version__,
        "device": _device_info(mx),
        "num_draft": num_draft,
        "copy_policy": copy_policy,
        "max_tokens": args.max_tokens,
        "prefill_step": args.prefill_step,
        "load_s": load_s,
        "policy": adapter.policy.as_dict(),
        "prompts": {},
    }
    for name in names:
        request = {"messages": [{"role": "user", "content": PROMPTS[name]}]}
        prompt_ids = list(adapter.prompt_tokens(request))
        results = {}
        for arm in arms:
            restore = set_arm(adapter, arm)
            try:
                results[arm] = run_arm(
                    adapter, recorder, prompt_ids, arm=arm, num_draft=num_draft,
                    max_tokens=args.max_tokens, prefill_step=args.prefill_step,
                    copy_policy=copy_policy,
                )
            finally:
                restore()
            if arm == "rowexact":
                results[arm]["route"] = adapter.row_exact_verify.status()
            print(name, arm, f"{len(results[arm]['tokens'])} tokens",
                  f"{results[arm]['elapsed_s']:.1f}s", results[arm]["windows"], flush=True)
        reference = results["off"]
        entry = {"prompt_tokens": len(prompt_ids), "arms": {}}
        for arm, result in results.items():
            summary = {
                "tokens": len(result["tokens"]),
                "tokens_sha": hashlib.sha256(json.dumps(result["tokens"]).encode()).hexdigest()[:16],
                "windows": result["windows"],
                "alignment_anomalies": result["alignment_anomaly_count"],
                "elapsed_s": result["elapsed_s"],
                "stats": result["stats"],
                "diagnostics_delta": result["diagnostics_delta"],
            }
            if "route" in result:
                summary["route"] = result["route"]
            if arm != "off":
                summary.update(compare(reference, result, len(prompt_ids), num_layers))
            entry["arms"][arm] = summary
            print(name, arm, json.dumps({k: summary.get(k) for k in (
                "identical_output", "first_divergent_token", "rows_compared",
                "rows_bit_equal", "first_differing_component", "max_abs_dlogit")}), flush=True)
        report["prompts"][name] = entry
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    totals = {}
    for arm in arms:
        if arm == "off":
            continue
        cells = [report["prompts"][n]["arms"][arm] for n in names]
        totals[arm] = {
            "prompts_identical": sum(bool(c["identical_output"]) for c in cells),
            "prompts": len(cells),
            "rows_bit_equal": sum(c["rows_bit_equal"] for c in cells),
            "rows_compared": sum(c["rows_compared"] for c in cells),
        }
    report["totals"] = totals
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print("TOTALS", json.dumps(totals), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
