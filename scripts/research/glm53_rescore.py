"""GLM-5.3: teacher-force a recorded sampled answer and score it under one policy.

For each ``--case RUN_DIR:FROM_WORD:TOKENS`` (an ask-model evidence dir with
answer.json), rebuilds the token stream (the server's chat render of the brief
with thinking on, then ``reasoning + "</think>" + answer`` re-tokenized),
prefills it up to the token where answer word FROM_WORD starts, then feeds the
recorded tokens one decode step at a time (the decode kernels the served run
used) for TOKENS steps. Per position it keeps the recorded token's logprob and
rank, the row entropy, and the top-5 (token, logprob) window, raw and at the
served temperature (0.7). Running it once per policy on the same case compares
the two policies' predictions on identical history at every position of a
collapse. Research only. Re-tokenized text may differ from the sampled ids at
a few boundaries; both policies score the same ids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_ordinary_profile import CHECKPOINT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="RUN_DIR:FROM_WORD:TOKENS")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx

    from glm53_decode_replay import load

    adapter, policy = load(args.policy)
    model, tok = adapter.model, adapter.tokenizer
    prompt_text = adapter.render_prompt({"messages": [{"role": "user", "content": args.brief.read_text()}],
                                         "enable_thinking": True})
    result = {"schema": "mlx2.glm53-rescore.v1", "checkpoint": CHECKPOINT, "source_revision": args.source_revision,
              "policy": policy, "temperature": args.temperature, "cases": []}
    for spec in args.case:
        run, from_word, steps = spec.rsplit(":", 2)
        from_word, steps = int(from_word), int(steps)
        answer = json.loads((Path(run) / "answer.json").read_text())
        reasoning, text = answer.get("reasoning") or "", answer.get("answer") or ""
        head = prompt_text + reasoning + "</think>"
        ids = tok.encode(head + text, add_special_tokens=False)
        # Character offset of answer word FROM_WORD, then its token position.
        words, pos = 0, 0
        while words < from_word and pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            while pos < len(text) and not text[pos].isspace():
                pos += 1
            words += 1
        start = len(tok.encode(head + text[:pos], add_special_tokens=False))
        steps = min(steps, len(ids) - start - 1)
        cache = model.make_cache()
        t0 = time.time()
        logits = None
        for begin in range(0, start, 2048):
            piece = mx.array([ids[begin:min(start, begin + 2048)]])
            with model._prompt_scope(piece.shape[1]):
                logits = model(piece, cache, logits_to_keep=1)
            mx.eval(logits, [c.state for c in cache])
        prefill_s = time.time() - t0
        rows = []
        t1 = time.time()
        for k in range(steps):
            target = ids[start + k]
            row = logits[0, -1].astype(mx.float32)
            lp = row - mx.logsumexp(row)
            lpt = row / args.temperature
            lpt = lpt - mx.logsumexp(lpt)
            order = mx.argsort(-lp)[:5]
            ent = -(mx.exp(lp) * lp).sum()
            rank = (lp > lp[target]).sum()
            mx.eval(lp, lpt, order, ent, rank)
            rows.append({"pos": start + k, "token": target, "logprob": float(lp[target].item()),
                         "logprob_t": float(lpt[target].item()), "rank": int(rank.item()),
                         "entropy": float(ent.item()),
                         "top5": [[int(i), round(float(lp[i].item()), 4)] for i in order.tolist()]})
            logits = model(mx.array([[target]]), cache)
        case = {"run": run, "from_word": from_word, "start_token": start, "steps": steps,
                "total_tokens": len(ids), "prefill_s": prefill_s, "decode_s": time.time() - t1, "rows": rows}
        result["cases"].append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "rows"}), flush=True)
        del cache
        mx.clear_cache()
    (args.out / "rescore.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
