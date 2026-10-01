#!/usr/bin/env python3
"""Prepare a local, revision-bound Qwen3 parallel-drafter serving policy.

Only metadata and bytes are inspected. This does not load a model, run a GPU,
qualify a route, or produce measured verification costs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx2.adapters.standard_decoder import _json, inspect_artifact


def prepare(
    target,
    draft,
    *,
    num_draft=None,
    xpress_passes=None,
    adaptive=None,
    attention_windows=None,
):
    target, draft = (
        Path(target).expanduser().resolve(),
        Path(draft).expanduser().resolve(),
    )
    target_record = inspect_artifact(target, expected="qwen3")
    architecture = _json(draft / "config.json").get("architectures")
    if architecture == ["Qwen3XPressModel"]:
        from mlx2.adapters.xpress import content_revision, inspect_drafter
    elif architecture == ["LiLiCorrDraftModel"]:
        from mlx2.adapters.lilicorr import content_revision, inspect_drafter

        if xpress_passes is not None:
            raise ValueError("XPress passes require an XPress checkpoint")
    else:
        raise ValueError("Unsupported companion architecture")
    record = inspect_drafter(draft, target)
    from mlx2.runtime.drafters.attention_windows import validate_attention_windows

    windows = validate_attention_windows(
        attention_windows, record["args"].num_hidden_layers
    )
    count = record["args"].block_size - 1 if num_draft is None else num_draft
    if type(count) is not int or not 1 <= count < record["args"].block_size:
        raise ValueError("num_draft must fit trained block")
    policy = {
        "draft_model": str(draft),
        "num_draft": count,
        "draft_revision": content_revision(record),
        "target_fingerprint": target_record["identity"]["fingerprint"],
    }
    if xpress_passes is not None:
        if (
            type(xpress_passes) is not int
            or not 1 <= xpress_passes <= record["args"].block_size
        ):
            raise ValueError("XPress passes must fit trained block")
        policy["xpress_num_passes"] = xpress_passes
    if windows is not None:
        policy["draft_attention_windows"] = list(windows)
    if adaptive is not None:
        from mlx2.runtime.acceptance_estimator import AdaptiveVerificationPolicy

        AdaptiveVerificationPolicy.from_value(adaptive, count)
        policy["adaptive_verification"] = adaptive
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--num-draft", type=int)
    parser.add_argument("--xpress-passes", type=int)
    parser.add_argument(
        "--attention-windows", help="JSON list per draft layer, e.g. [512,null,1024]"
    )
    parser.add_argument(
        "--adaptive-policy",
        type=Path,
        help="JSON with measured cost inputs and adaptive settings",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    adaptive = _json(args.adaptive_policy) if args.adaptive_policy else None
    windows = json.loads(args.attention_windows) if args.attention_windows else None
    policy = prepare(
        args.target,
        args.draft,
        num_draft=args.num_draft,
        xpress_passes=args.xpress_passes,
        adaptive=adaptive,
        attention_windows=windows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=2) + "\n")
    print(
        json.dumps(
            {
                "policy": str(args.out.resolve()),
                "model_loaded": False,
                "qualified": False,
                "num_draft": policy["num_draft"],
            }
        )
    )


if __name__ == "__main__":
    main()
