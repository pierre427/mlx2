#!/usr/bin/env python3
"""Teacher-forced DFlash acceptance on a Kimi target (the drafter card's metric).

In-process component probe (no serving): the target's own greedy
continuation of each prompt is generated with the ordinary one-token
forward; one forward then captures every layer's post-layer residual. For
block starts along the continuation the drafter sees the committed context
features of positions before the start and the start token as its anchor;
acceptance length is 1 + the consecutive argmax matches with the target's
continuation (block 16), exactly the card's offline definition. Variants
shift the tapped layers by -1/+1 to test the tap convention. No thermal
holds; sanity numbers. Requires the caller's CPG lease and paired locks.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFT = "~/Models/Kimi-Linear-48B-A3B-DFlash-240k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--continuation", type=int, default=160)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing an existing evidence file")
    sys.path.insert(0, str(ROOT/"scripts"))
    sys.path.insert(0, str(ROOT/"scripts"/"research"))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    import mlx.core as mx
    from kimi_linear_speculative import prompts

    from mlx2.adapters.kimi_dflash import inspect_drafter, load_drafter
    from mlx2.adapters.kimi_linear import KimiLinearAdapter, checkpoint_name

    adapter = KimiLinearAdapter(str(args.model), execution_policy=json.loads(args.policy))
    model = adapter.model
    record = inspect_drafter(DRAFT, args.model, target_config=adapter.config)
    draft = load_drafter(record, model)
    layers = list(draft.config.target_layer_ids)
    variants = {"config": layers, "minus1": [i-1 for i in layers], "plus1": [i+1 for i in layers]}
    receipt = {"checkpoint": checkpoint_name(args.model)+":"+adapter.identity["fingerprint"],
               "draft_checkpoint": record["checkpoint"]+":"+record["fingerprint"],
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
               "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                            text=True, cwd=ROOT).strip()),
               "policy": adapter.policy, "ownership": lease, "qualified": False,
               "kind": "teacher-forced offline DFlash acceptance (card metric), in-process component probe",
               "block": int(draft.config.block_size), "stride": args.stride, "cases": []}
    eos = set(adapter.tokenizer.eos_token_ids)
    for case, text, _check in prompts():
        tokens = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": text}]}))
        prompt_len = len(tokens)
        cache = model.make_cache()
        logits = model(mx.array([tokens]), cache)
        for _ in range(args.continuation):
            token = int(mx.argmax(logits[0, -1]).item())
            tokens.append(token)
            if token in eos:
                break
            logits = model(mx.array([[token]]), cache)
        del cache
        _, taps = model.forward_with_taps(mx.array([tokens]), model.make_cache(), list(range(27)))
        mx.eval(taps)
        width = int(adapter.config["hidden_size"])
        row = {"case": case, "prompt_tokens": prompt_len, "continuation_tokens": len(tokens)-prompt_len}
        block = int(draft.config.block_size)
        for name, ids in variants.items():
            columns = mx.concatenate([taps[..., i*width:(i+1)*width] for i in ids], axis=-1)
            lengths = []
            for start in range(prompt_len, len(tokens)-block, args.stride):
                logits = draft._block_logits([tokens[start]], columns[:, :start], draft.make_cache(), block-1)
                guess = [int(t) for t in mx.argmax(logits[0], axis=-1).tolist()]
                accepted = 0
                for predicted, actual in zip(guess, tokens[start+1:start+block]):
                    if predicted != actual:
                        break
                    accepted += 1
                lengths.append(accepted+1)
            row[name] = {"blocks": len(lengths), "acceptance_length": sum(lengths)/max(1, len(lengths)),
                         "lengths": lengths}
        receipt["cases"].append(row)
        print(json.dumps({"case": case, **{n: round(row[n]["acceptance_length"], 2) for n in variants}}), flush=True)
    for name in variants:
        all_lengths = [x for row in receipt["cases"] for x in row[name]["lengths"]]
        receipt[name+"_mean"] = sum(all_lengths)/max(1, len(all_lengths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({n: round(receipt[n+"_mean"], 2) for n in variants}))
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
