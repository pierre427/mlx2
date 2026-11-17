# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
"""Owned GPU-window checks; small kernels first, then real tokenizer processors."""

import argparse, faulthandler, json, os, subprocess, time
from pathlib import Path

os.environ["MLX_METAL_FAST_SYNCH"] = "0"
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
stack = (args.out / "watchdog.txt").open("x")
faulthandler.enable(file=stack)
faulthandler.dump_traceback_later(90, file=stack, exit=True)
import mlx.core as mx
from mlx2.runtime.generation_stream import bind_generation_thread_stream

bind_generation_thread_stream()
from mlx2.runtime import amoeba_calc_resident as R


def save(name, value):
    p = args.out / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(p)


cases = [
    ("3+4", "7"),
    ("0.1+0.2", "0.3"),
    ("123456789123*987654321987", "121932631355968601347401"),
    (r"\dfrac{\frac{3}{4}+\frac{1}{8}}{\frac{7}{2}}", "0.25"),
    (r"\sum_{i=1}^{10}{i^2}", "385"),
    (r"\prod_{i=1}^{5}{i+1}", "720"),
    (r"\binom{20}{3}", "1140"),
    ("2^-3", "0.125"),
    ("(-2)^2", "4"),
    ("How many r's are in strawberry?", "3"),
    ("What is the sum of [0.1,0.2]?", "0.3"),
    ("How many items are in [1,2,3]?", "3"),
    ("1/0", None),
    ("1/3", None),
    ("sin(4)+3", None),
    (r"\sqrt{4}+3", None),
    ("unsupported 2+3", None),
    ("1 2+3", None),
    ("2^65", None),
    ("2026-11-17", None),
    ("2@1", None),
    ("3!!", None),
    (r"\FRAC{2}{3}+1", None),
]
rows = []
for text, want in cases:
    start = time.monotonic()
    data = mx.array(list(text.encode()), dtype=mx.uint8)
    p, n = R.parse_device(data, mx.array([len(text.encode())], dtype=mx.int32))
    print("parsing", text, flush=True)
    mx.eval(p, n)
    print("evaluating", int(p[0].item()), flush=True)
    result = R.evaluate_device(p, n).tolist()
    got = None
    if result[0] == 0:
        got = "".join(str(x) for x in result[4 : 4 + result[3]][::-1])
        scale = result[2]
        if scale:
            got = got.rjust(scale + 1, "0")
            got = got[:-scale] + "." + got[-scale:]
        if result[1] < 0:
            got = "-" + got
    row = dict(
        text=text,
        want=want,
        got=got,
        error=result[0],
        ok=got == want,
        seconds=time.monotonic() - start,
    )
    rows.append(row)
    save("arithmetic.json", rows)
    print(row, flush=True)
    if got != want:
        raise AssertionError(row)
    faulthandler.dump_traceback_later(45, file=stack, exit=True)

from transformers import AutoTokenizer
from types import SimpleNamespace
from mlx2.adapters.registry import inspect_model

models = [
    Path.home() / "mlx-models/Qwen3.8-27B-BF16-q8-mlx2",
    Path.home() / "mlx-models/gemma-4-26B-A4B-it-MLX-8bit",
]
processor_rows = []
for model in models:
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    checkpoint = model.name + "@" + inspect_model(model).artifact_fingerprint[:8]
    faulthandler.dump_traceback_later(45, file=stack, exit=True)
    print("Precomputing byte spellings for", checkpoint, flush=True)
    R.vocabulary_tables(tok)
    print("Vocabulary ready for", checkpoint, flush=True)
    for question, generated, want in [
        ("Calculate (19+8)*7.", "", "189"),
        ("Calculate 1/0.", "", None),
        ("Calculate sin(2)+3.", "", None),
        ("How many r's are in strawberry?", "", "3"),
        ("Explain your calculation.", "3+4= ", "7"),
        ("Explain your calculation.", r"\(3+4=\)", "7"),
        (
            "Explain your calculation.",
            r"\begin{array}{r}"
            + "\n"
            + r"109 \\"
            + "\n"
            + r"\times 71 \\"
            + "\n"
            + r"\hline",
            "7739",
        ),
        ("Explain your calculation.", "sin(2) + 3=", None),
        ("Quote this text exactly: 3+4=", "3+4=", None),
    ]:
        tiles = SimpleNamespace(forward_ids={"calc": 1}, stash_shape={"calc": (1, 1)})
        processor = R.ResidentCalculatorProcessor(
            tiles, "calc", tok, prompt_length=1, request_text=question
        )
        ids = [0]
        written = []
        identity = True
        if generated:
            processor.bind_row(0)
            logits = mx.arange(len(tok), dtype=mx.float32)[None]
            out = processor(mx.array(ids, dtype=mx.int32), logits)
            identity = bool(mx.array_equal(out, logits).item())
            ids.extend(tok.encode(generated, add_special_tokens=False))
        for _ in range(40):
            processor.bind_row(0)
            logits = mx.zeros((1, len(tok)), dtype=mx.float32)
            out = processor(mx.array(ids, dtype=mx.int32), logits)
            mx.eval(out)
            target = int(processor.state[10].item())
            if target < 0:
                if want is None:
                    identity = identity and bool(mx.array_equal(out, logits).item())
                break
            ids.append(target)
            written.append(target)
            if target == tok.eos_token_id:
                break
        got = (
            tok.decode([x for x in written if x != tok.eos_token_id])
            if written
            else None
        )
        row = dict(
            checkpoint=checkpoint,
            question=question,
            generated=generated,
            want=want,
            got=got,
            gate_open=int(processor.state[5].item()),
            unchanged_logits=identity,
            ok=got == want and identity,
        )
        processor_rows.append(row)
        save("processors.json", processor_rows)
        print(row, flush=True)
        if not row["ok"] or (want is None and row["gate_open"]):
            raise AssertionError(row)
        faulthandler.dump_traceback_later(45, file=stack, exit=True)
save(
    "summary.json",
    dict(
        checkpoint="synthetic kernel and exact local tokenizer checks, no model weights loaded",
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        arithmetic=len(rows),
        processors=len(processor_rows),
        ok=True,
    ),
)
faulthandler.cancel_dump_traceback_later()
stack.close()
