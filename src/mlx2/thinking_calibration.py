"""Artifact-bound calibration of a thinking "commit direction" (alpha steering).

A commit direction is a property of one exact artifact: weights, quantization,
tokenizer and chat template.  Steering an artifact with another artifact's
direction is a same-norm *wrong* vector, and the calibration campaign measured
wrong vectors as worse than no steering at all.  So:

* every direction is stored with the ``artifact_identity`` it was measured on
  and is only ever loaded for that identity;
* when steering is wanted and no bound direction exists, the server tries to
  calibrate one itself at startup (natural traces -> positional direction ->
  held-out validation with a random control);
* if that does not pass its gates, steering stays off.  An operator who
  explicitly asked for steering gets a startup error instead of an unsteered
  server that looks steered (fail closed).

Everything here is model-neutral.  It needs an adapter that renders chat
prompts and declares a single thinking-close token, and a model that exposes
``model.model.residual_taps``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = "mlx2.commit-direction.v1"
COMMIT_TAIL = 24
REFLECT_FRACTION = 0.45
# Fail-closed gates for an automatic calibration.
MIN_USABLE_TRACES = 10
MIN_CONSISTENCY = 0.30
VALIDATION_ALPHA = 0.2

CALIBRATION_PROMPTS = [
    ("What is 37 + 58? Reply with the number only.", "95"),
    ("What is 14 times 12? Reply with the number only.", "168"),
    ("What is 1000 minus 377? Reply with the number only.", "623"),
    ("How many days are there in a leap year? Reply with the number only.", "366"),
    ("What is the capital of Japan? One word.", "tokyo"),
    ("What is the capital of Canada? One word.", "ottawa"),
    ("What is the chemical symbol for gold? Reply with the symbol only.", "au"),
    ("How many sides does a hexagon have? Reply with the number only.", "6"),
    ("What is the next prime after 31? Reply with the number only.", "37"),
    ("What is 2 to the power of 10? Reply with the number only.", "1024"),
    ("What is the greatest common divisor of 84 and 36? Reply with the number only.", "12"),
    ("A shirt costs 40 dollars and is discounted by 25 percent. What is the new price in dollars? Reply with the number only.", "30"),
    ("If a car travels 150 km in 2.5 hours, what is its average speed in km/h? Reply with the number only.", "60"),
    ("What is the least common multiple of 6 and 8? Reply with the number only.", "24"),
    ("How many minutes are there in 3.5 hours? Reply with the number only.", "210"),
    ("Translate the English word 'water' to Spanish. Reply with the word only.", "agua"),
    ("Which is larger, 0.7 or 0.65? Reply with the number only.", "0.7"),
    ("What is the square root of 196? Reply with the number only.", "14"),
    ("How many vowels are in the word 'education'? Reply with the number only.", "5"),
    ("What is 15 percent of 240? Reply with the number only.", "36"),
    ("Explain in three sentences what loop unrolling does in a compiler.", None),
    ("Explain in three sentences why the sky appears blue.", None),
    ("Write a Python function named double that returns twice its argument. Code only.", "def double"),
    ("List the first five square numbers, comma-separated.", "1, 4, 9, 16, 25"),
]
VALIDATION_PROMPTS = [  # disjoint from the calibration prompts
    ("What is 62 + 130? Reply with the number only.", "192"),
    ("What is 44 + 88? Reply with the number only.", "132"),
    ("Is Paris a coastal city? Answer yes or no.", "no"),
    ("What is the capital of Hungary? One word.", "budapest"),
    ("How many prime numbers are there below 60? Reply with the number only.", "17"),
    ("What is 7 to the power of 5, modulo 13? Reply with the number only.", "11"),
    ("Two trains start 300 km apart and head toward each other at 70 km/h and 80 km/h. After how many hours do they meet? Reply with the number only.", "2"),
    ("Sort these words alphabetically and reply with them comma-separated: pear, apple, mango, fig, banana, cherry.", "apple, banana, cherry, fig, mango, pear"),
]


# --------------------------------------------------------------------------- identity
def artifact_identity(model_path) -> str:
    """Content identity of an artifact for binding a calibration to it.

    Hashes the files that define behaviour (config, tokenizer, chat template,
    weight index) and, for every weight shard, its size plus the first 4 MiB and
    last 1 MiB of bytes: cheap, host-independent (no mtimes or paths), and
    different for any other quantization or fine-tune.
    """
    path = Path(model_path)
    digest = hashlib.sha256(SCHEMA.encode())
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json", "model.safetensors.index.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for shard in sorted(path.glob("*.safetensors")):
        size = shard.stat().st_size
        digest.update(f"{shard.name}:{size}".encode())
        with shard.open("rb") as handle:
            digest.update(handle.read(4 << 20))
            if size > (5 << 20):
                handle.seek(-(1 << 20), 2)
                digest.update(handle.read(1 << 20))
    return digest.hexdigest()


def calibration_cache_dir(cache_dir=None) -> Path:
    root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "mlx2"
    return root / "commit-directions"


def load_bound_direction(identity, candidates, *, hidden_size, num_layers, layer=None):
    """The first stored direction bound to ``identity``, or None.

    ``candidates`` are ``.npz`` paths, each with a sibling ``.json`` carrying
    ``artifact_identity`` and the chosen ``layer``.  A file measured on another
    artifact is ignored, however well its shape fits.
    """
    import mlx.core as mx
    import numpy as np

    for npz in candidates:
        npz = Path(npz)
        meta_path = npz.with_suffix(".json")
        if not npz.is_file() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("schema") != SCHEMA or meta.get("artifact_identity") != identity:
                continue
            chosen = int(layer if layer is not None else meta["layer"])
            data = np.load(npz)
            unit, rms = data[f"v_{chosen}"], float(data[f"rms_{chosen}"])
            if unit.shape != (int(hidden_size),) or not 0 <= chosen < int(num_layers):
                continue
            if not np.isfinite(unit).all() or abs(float(np.linalg.norm(unit)) - 1.0) > 1e-3:
                continue
            return {"layer": chosen, "vector": mx.array((rms * unit).astype(np.float32)),
                    "source": f"{npz.name}:L{chosen}", "artifact_identity": identity,
                    "origin": meta.get("origin", "shipped")}
        except Exception:  # noqa: BLE001 - an unreadable candidate is not a calibration
            continue
    return None


# --------------------------------------------------------------------------- measurement
def rep4(ids):
    if len(ids) < 8:
        return 0.0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return round(1.0 - len(set(grams)) / len(grams), 3)


def _eos_ids(tokenizer):
    eos = getattr(tokenizer, "eos_token_ids", None) or getattr(tokenizer, "eos_token_id", None)
    return set(eos if isinstance(eos, (list, tuple, set)) else [eos])


def generate(adapter, prompt, *, close_id, eos_ids, max_think, max_answer=260, steer=None):
    """Greedy B=1 generation; ``steer(step)`` returns ``(layer, vector)`` while thinking is open."""
    import mlx.core as mx

    model = adapter.model
    taps = model.model.residual_taps
    ids = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": prompt}], "enable_thinking": True}))
    cache = model.make_cache()
    for start in range(0, len(ids) - 1, 512):
        model(mx.array([ids[start:min(start + 512, len(ids) - 1)]]), cache=cache)
        mx.eval([c.state for c in cache])
    token, generated, close_step = ids[-1], [], -1
    while True:
        taps.steer = steer(len(generated)) if (steer is not None and close_step < 0) else None
        try:
            logits = model(mx.array([[token]]), cache=cache)
        finally:
            taps.steer = None
        token = int(mx.argmax(logits[0, -1]).item())
        if token in eos_ids:
            break
        generated.append(token)
        if token == close_id and close_step < 0:
            close_step = len(generated) - 1
        if close_step < 0 and len(generated) >= max_think:
            break
        if close_step >= 0 and len(generated) - close_step > max_answer:
            break
    answer = adapter.tokenizer.decode(generated[close_step + 1:]) if close_step >= 0 else ""
    for marker in ("<|START_TEXT|>", "<|END_TEXT|>", "<|START_RESPONSE|>", "<|END_RESPONSE|>"):
        answer = answer.replace(marker, "")
    return {"token_ids": generated, "close_step": close_step, "answer": answer.strip(),
            "rep4": rep4(generated[: close_step if close_step >= 0 else None])}


def is_correct(row, expected):
    if not row["answer"]:
        return False
    if expected is None:
        return len(row["answer"].split()) >= 20
    return expected.lower().replace(" ", "") in row["answer"].lower().replace(" ", "")


def collect_traces(adapter, prompts, *, close_id, eos_ids, max_think=1500, progress=None):
    traces = []
    for index, (prompt, expected) in enumerate(prompts):
        row = generate(adapter, prompt, close_id=close_id, eos_ids=eos_ids, max_think=max_think)
        row.update(index=index, prompt=prompt, correct=is_correct(row, expected))
        traces.append(row)
        if progress:
            progress(row)
    return traces


def extract_directions(adapter, traces, layers, *, commit_tail=COMMIT_TAIL, reflect_frac=REFLECT_FRACTION, progress=None):
    """Positional commit-minus-reflect direction per layer from closed, correct traces."""
    import mlx.core as mx
    import numpy as np

    taps = adapter.model.model.residual_taps
    sums = {L: {"reflect": 0.0, "commit": 0.0} for L in layers}
    counts = {"reflect": 0, "commit": 0}
    rms, dirs, used = {L: [] for L in layers}, {L: [] for L in layers}, 0
    for tr in traces:
        ct = tr["close_step"]
        if ct <= 40 or not tr["correct"]:
            continue
        warm = max(4, int(0.08 * ct))
        p_com = list(range(max(0, ct - commit_tail), ct))
        p_ref = [p for p in range(warm, int(reflect_frac * ct)) if p not in set(p_com)]
        if not p_ref:
            continue
        pids = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": tr["prompt"]}], "enable_thinking": True}))
        seq, off = pids + tr["token_ids"], len(pids)
        cache = adapter.model.make_cache()
        acc = {L: [] for L in layers}
        for start in range(0, len(seq), 512):
            taps.capture = {L: None for L in layers}
            try:
                adapter.model(mx.array([seq[start:start + 512]]), cache=cache)
                captured = taps.capture
            finally:
                taps.capture = None
            mx.eval(*captured.values())
            for L in layers:
                acc[L].append(np.array(captured[L][0].astype(mx.float32)))
        # The residual at position i predicts token i+1; generated token j sits at off+j.
        for L in layers:
            H = np.concatenate(acc[L], axis=0)
            hr, hc = H[[off + p - 1 for p in p_ref]], H[[off + p - 1 for p in p_com]]
            sums[L]["reflect"] = sums[L]["reflect"] + hr.sum(0)
            sums[L]["commit"] = sums[L]["commit"] + hc.sum(0)
            rms[L].append(float(np.linalg.norm(H[off:], axis=1).mean()))
            d = hc.mean(0) - hr.mean(0)
            dirs[L].append(d / (np.linalg.norm(d) + 1e-8))
        counts["reflect"] += len(p_ref)
        counts["commit"] += len(p_com)
        used += 1
        if progress:
            progress(tr)
    arrays, report = {}, {"traces_used": used, "counts": counts, "layers": {}}
    if used < 2:
        return arrays, report
    for L in layers:
        v = sums[L]["commit"] / counts["commit"] - sums[L]["reflect"] / counts["reflect"]
        D = np.stack(dirs[L])
        gram = D @ D.T
        consistency = float((gram.sum() - np.trace(gram)) / (len(D) * (len(D) - 1)))
        arrays[f"v_{L}"] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
        arrays[f"rms_{L}"] = np.array(float(np.mean(rms[L])))
        report["layers"][str(L)] = {"raw_norm": float(np.linalg.norm(v)), "rms": float(np.mean(rms[L])),
                                    "relative_norm": float(np.linalg.norm(v) / np.mean(rms[L])),
                                    "consistency": consistency}
    return arrays, report


def candidate_layers(num_layers):
    """Every fourth layer across the 25%–90% depth band the lab found commit bands in."""
    low, high = int(num_layers * 0.25), int(num_layers * 0.9)
    return [L for L in range(low - low % 4, high + 1, 4) if 0 < L < num_layers]


def choose_layer(report, num_layers):
    """Most consistent layer in the 50%–75% depth band (where every lab model's band sat)."""
    band = {int(L): row for L, row in report["layers"].items() if 0.5 * num_layers <= int(L) <= 0.75 * num_layers}
    pool = band or {int(L): row for L, row in report["layers"].items()}
    return max(pool, key=lambda L: pool[L]["consistency"]) if pool else None


def validate(adapter, arrays, layer, *, close_id, eos_ids, prompts=VALIDATION_PROMPTS, alpha=VALIDATION_ALPHA,
             max_think=1500, seed=20260918):
    """Held-out check: calibrated steering vs off vs a same-norm random direction."""
    import mlx.core as mx
    import numpy as np

    scale, unit = float(arrays[f"rms_{layer}"]), arrays[f"v_{layer}"]
    random_unit = np.random.default_rng(seed).standard_normal(unit.shape)
    random_unit = (random_unit / np.linalg.norm(random_unit)).astype(np.float32)
    arms = {"off": None, "calibrated": mx.array(alpha * scale * unit)[None, None, :],
            "random": mx.array(alpha * scale * random_unit)[None, None, :]}
    out = {}
    for name, vector in arms.items():
        steer = (lambda _step, v=vector: (layer, v)) if vector is not None else None
        rows = [generate(adapter, p, close_id=close_id, eos_ids=eos_ids, max_think=max_think, steer=steer) for p, _ in prompts]
        out[name] = {"correct": sum(is_correct(r, e) for r, (_, e) in zip(rows, prompts)),
                     "closed": sum(r["close_step"] >= 0 for r in rows),
                     "think_tokens": sum(r["close_step"] if r["close_step"] >= 0 else len(r["token_ids"]) for r in rows)}
    return out


def gates(report, layer, validation):
    """Every reason an automatic calibration must NOT be trusted (empty = pass)."""
    failures = []
    if report.get("traces_used", 0) < MIN_USABLE_TRACES:
        failures.append(f"only {report.get('traces_used', 0)} usable closed traces (need {MIN_USABLE_TRACES})")
    consistency = (report.get("layers", {}).get(str(layer)) or {}).get("consistency", -1.0)
    if layer is None or consistency < MIN_CONSISTENCY:
        failures.append(f"cross-trace consistency {consistency:+.3f} below {MIN_CONSISTENCY}")
    if validation:
        off, cal, rnd = validation["off"], validation["calibrated"], validation["random"]
        if cal["correct"] < off["correct"]:
            failures.append(f"steering lost accuracy on held-out prompts ({cal['correct']} < {off['correct']})")
        if cal["closed"] < off["closed"]:
            failures.append("steering closed fewer held-out traces than no steering")
        if cal["think_tokens"] > off["think_tokens"]:
            failures.append("steering did not shorten held-out reasoning")
        if rnd["think_tokens"] <= cal["think_tokens"] and rnd["correct"] >= cal["correct"]:
            failures.append("a same-norm random direction did as well: the effect is not directional")
    return failures


def supports_calibration(adapter):
    close = getattr(adapter, "thinking_close_token_ids", None)
    ids = close() if callable(close) else None
    taps = getattr(getattr(getattr(adapter, "model", None), "model", None), "residual_taps", None)
    return bool(ids) and len(ids) == 1 and taps is not None and callable(getattr(adapter, "prompt_tokens", None))


def auto_calibrate(adapter, identity, out_dir, *, progress=None, max_think=1500):
    """Calibrate, validate and store a direction for this artifact.

    Returns ``(npz_path | None, report)``.  Nothing is written unless every gate
    passes, so a failed attempt can never be picked up as a calibration later.
    """
    import numpy as np

    started = time.time()
    report = {"schema": SCHEMA, "artifact_identity": identity, "origin": "auto", "status": "failed"}
    if not supports_calibration(adapter):
        report["failures"] = ["model exposes no residual taps or no single thinking-close token"]
        return None, report
    close_id = adapter.thinking_close_token_ids()[0]
    eos_ids = _eos_ids(adapter.tokenizer)
    num_layers = int(adapter.model.args.num_hidden_layers)
    traces = collect_traces(adapter, CALIBRATION_PROMPTS, close_id=close_id, eos_ids=eos_ids,
                            max_think=max_think, progress=progress)
    arrays, extracted = extract_directions(adapter, traces, candidate_layers(num_layers))
    report.update(extracted)
    layer = choose_layer(extracted, num_layers) if arrays else None
    report["layer"] = layer
    validation = (validate(adapter, arrays, layer, close_id=close_id, eos_ids=eos_ids, max_think=max_think)
                  if layer is not None else None)
    report["validation"] = validation
    report["failures"] = gates(extracted, layer, validation)
    report["seconds"] = round(time.time() - started, 1)
    if report["failures"]:
        # Remember the verdict (never a direction) so the next start does not
        # spend a minute rediscovering it.  Delete the file to try again.
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{identity}.rejected.json").write_text(json.dumps(report, indent=1))
        return None, report
    report["status"] = "calibrated"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{identity}.npz"
    np.savez(npz, **arrays)
    npz.with_suffix(".json").write_text(json.dumps(report, indent=1))
    return npz, report


def resolve_direction(adapter, model_path, *, cache_dir=None, shipped=(), preferred_layer=None, allow_auto=True):
    """``(direction | None, status)`` for an artifact, calibrating it if allowed.

    Order: a shipped asset bound to this identity, then a previously stored
    automatic calibration, then (optionally) a fresh automatic calibration.
    """
    identity = artifact_identity(model_path)
    hidden, layers = int(adapter.model.args.hidden_size), int(adapter.model.args.num_hidden_layers)
    store = calibration_cache_dir(cache_dir)
    status = {"artifact_identity": identity, "state": "uncalibrated"}
    direction = load_bound_direction(identity, shipped, hidden_size=hidden, num_layers=layers, layer=preferred_layer)
    if direction is None:
        direction = load_bound_direction(identity, [store / f"{identity}.npz"], hidden_size=hidden, num_layers=layers)
    rejected = store / f"{identity}.rejected.json"
    if direction is None and allow_auto and rejected.is_file():
        try:
            previous = json.loads(rejected.read_text())
        except Exception:  # noqa: BLE001
            previous = {}
        status["auto_calibration"] = {**{k: previous.get(k) for k in ("status", "layer", "failures", "validation", "traces_used", "seconds")},
                                      "cached_verdict": str(rejected)}
        allow_auto = False
    if direction is None and allow_auto:
        log.warning("no commit direction is calibrated for this artifact; calibrating now")
        npz, report = auto_calibrate(adapter, identity, store)
        status["auto_calibration"] = {k: report.get(k) for k in ("status", "layer", "failures", "validation", "traces_used", "seconds")}
        if npz is not None:
            direction = load_bound_direction(identity, [npz], hidden_size=hidden, num_layers=layers)
    if direction is not None:
        status.update(state="calibrated", layer=direction["layer"], source=direction["source"], origin=direction["origin"])
    return direction, status
