#!/usr/bin/env python3
"""Teacher-force greedy A/B divergences through the full target head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from qualify_mtp_draft_vocab import PROMPTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models.cache import make_prompt_cache

    result = json.loads(args.result.read_text())
    adapter = resolve_adapter(args.model, mtp=True)(
        args.model, execution_policy={"mtp_draft_vocab": True}
    )
    model = adapter.model
    mx.eval(model.parameters())
    prompts = [
        list(
            adapter.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )
        )
        for prompt in PROMPTS
    ]
    analyses = []
    try:
        for parity in result["parity"]:
            if parity["equal"]:
                continue
            full = next(
                row
                for row in result["results"]
                if row["rep"] == parity["rep"]
                and row["width"] == parity["width"]
                and row["arm"] == "full"
            )
            reduced = next(
                row
                for row in result["results"]
                if row["rep"] == parity["rep"]
                and row["width"] == parity["width"]
                and row["arm"] == "reduced"
            )
            for lane, (full_tokens, reduced_tokens) in enumerate(
                zip(full["tokens"], reduced["tokens"])
            ):
                position = next(
                    (
                        index
                        for index, pair in enumerate(zip(full_tokens, reduced_tokens))
                        if pair[0] != pair[1]
                    ),
                    None,
                )
                if position is None:
                    continue
                context = prompts[lane] + full_tokens[:position]
                cache = make_prompt_cache(model)
                values = mx.array(context, dtype=mx.uint32)[None]
                for start in range(0, values.shape[1] - 1, 4096):
                    stop = min(start + 4096, values.shape[1] - 1)
                    model(values[:, start:stop], cache=cache)
                logits = model(values[:, -1:], cache=cache)[0, -1].astype(mx.float32)
                mx.eval(logits)
                top_ids = mx.argpartition(logits, kth=-5)[-5:]
                top_ids = top_ids[mx.argsort(logits[top_ids])[::-1]]
                mx.eval(top_ids)
                top_ids = [int(value) for value in top_ids.tolist()]
                full_id = int(full_tokens[position])
                reduced_id = int(reduced_tokens[position])
                maximum = float(logits[top_ids[0]].item())
                analyses.append(
                    {
                        **parity,
                        "lane": lane,
                        "position": position,
                        "full_token": full_id,
                        "reduced_token": reduced_id,
                        "argmax": top_ids[0],
                        "full_margin": maximum - float(logits[full_id].item()),
                        "reduced_margin": maximum - float(logits[reduced_id].item()),
                        "top5": [
                            {"token": token_id, "logit": float(logits[token_id].item())}
                            for token_id in top_ids
                        ],
                    }
                )
                del cache
                mx.clear_cache()
    finally:
        adapter.close()
    output = {
        "schema": "mlx2.mtp-draft-vocab-divergence-analysis.v1",
        "source": result["source"],
        "artifact_identity": result["artifact_identity"],
        "analyses": analyses,
        "all_near_tie": all(
            min(row["full_margin"], row["reduced_margin"]) <= 0.125
            for row in analyses
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
