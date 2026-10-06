#!/usr/bin/env python3
"""Read-only Qwen4 artifact census for upstream viability experiments.

This does not load model tensors or change an artifact.  It answers the cheap
questions that should precede a Metal or conversion experiment:

* are hyper-connection projections actually group-size 32;
* is an embedded PLE table named ``shard_N`` (current mlx2 layout) or
  ``shards.N`` (newer mlx-vlm/oMLX layout);
* are all three affine parts present for every embedded shard; and
* does the JSON configuration advertise a native MTP head.

Strata #619 concerns GGUF metadata rather than MLX JSON.  ``trunk_blocks`` is
kept as a small, testable helper for an extracted GGUF metadata mapping; the
CLI deliberately reports that a GGUF-only declaration is not a load feature
of mlx2.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_PLE_SHARD = re.compile(
    r"^(?P<prefix>.+\.ple\.ple_embedding\.ngram_embedding)"
    r"(?P<spelling>\.shard_|\.shards\.)(?P<index>\d+)\."
    r"(?P<part>weight|scales|biases)$"
)
_HC_PARTS = (
    "input_mix_weight_down",
    "input_mix_weight_up",
    "block_inject_weight",
)


def trunk_blocks(metadata: dict) -> dict:
    """Interpret Strata #619's Qwen4Exp GGUF block-count contract."""
    blocks = metadata.get("qwen4exp.block_count")
    nextn = metadata.get("qwen4exp.nextn_predict_layers", 0)
    if type(blocks) is not int or blocks < 0:
        raise ValueError("qwen4exp.block_count must be a non-negative integer")
    if type(nextn) is not int or nextn < 0:
        raise ValueError("qwen4exp.nextn_predict_layers must be a non-negative integer")
    if nextn > blocks:
        raise ValueError("qwen4exp.nextn_predict_layers exceeds block_count")
    return {
        "declared_blocks": blocks,
        "nextn_blocks": nextn,
        "trunk_blocks": blocks - nextn,
    }


def _quantization(config: dict) -> dict:
    quant = config.get("quantization", {})
    return quant if isinstance(quant, dict) else {}


def _quant_spec(config: dict, prefix: str) -> dict:
    quant = _quantization(config)
    candidates = (
        prefix,
        prefix.removeprefix("model."),
        prefix.removeprefix("language_model."),
        prefix.replace("model.language_model.", "language_model.", 1),
    )
    override = next(
        (quant[name] for name in candidates if isinstance(quant.get(name), dict)),
        {},
    )
    return {
        "bits": override.get("bits", quant.get("bits")),
        "group_size": override.get("group_size", quant.get("group_size")),
        "mode": override.get("mode", quant.get("mode", "affine")),
    }


def inspect_mapping(config: dict, weight_map: dict[str, str]) -> dict:
    embedded: dict[tuple[str, str], dict[int, set[str]]] = {}
    table_scales = []
    hc = []
    for name in sorted(weight_map):
        match = _PLE_SHARD.match(name)
        if match:
            key = (match.group("prefix"), match.group("spelling"))
            embedded.setdefault(key, {}).setdefault(
                int(match.group("index")), set()
            ).add(match.group("part"))
        if name.endswith(".ngram_embedding.weight_scale"):
            table_scales.append(name)
        if name.endswith(".weight") and any(
            name.endswith(f".{part}.weight") for part in _HC_PARTS
        ):
            prefix = name.removesuffix(".weight")
            hc.append({"path": prefix, **_quant_spec(config, prefix)})

    tables = []
    all_parts = {"weight", "scales", "biases"}
    for (prefix, spelling), shards in sorted(embedded.items()):
        indices = sorted(shards)
        missing = {
            str(index): sorted(all_parts - shards[index])
            for index in indices
            if shards[index] != all_parts
        }
        contiguous = indices == list(range(len(indices)))
        current_native = spelling == ".shard_"
        tables.append(
            {
                "prefix": prefix,
                "spelling": "shard_N" if current_native else "shards.N",
                "shards": len(indices),
                "contiguous_zero_based": contiguous,
                "missing_parts": missing,
                "current_mlx2_native_name": current_native,
                "tensorfold_330_name": not current_native,
                "compatible_without_conversion": current_native
                and contiguous
                and not missing,
            }
        )

    text = config.get("text_config", config)
    native_mtp = text.get("mtp_num_hidden_layers", 0)
    nextn = text.get(
        "num_nextn_predict_layers", config.get("num_nextn_predict_layers", 0)
    )
    hc32 = [item for item in hc if item["group_size"] == 32]
    formats = Counter(
        f"q{item['bits']}/g{item['group_size']}/{item['mode']}" for item in hc
    )
    return {
        "model_type": config.get("model_type"),
        "hc": {
            "projection_entries": len(hc),
            "formats": dict(sorted(formats.items())),
            "group_size_32_entries": len(hc32),
            "group_size_32_paths": [item["path"] for item in hc32],
            "synthetic_gs32_probe_worth_running": bool(hc32),
            "note": (
                "No local gs32 HC projection was found; a synthetic kernel probe is still possible, "
                "but a model-level A/B is artifact-conditional."
                if not hc32
                else "At least one gs32 HC projection is declared; verify every projection at load time."
            ),
        },
        "ple": {
            "embedded_tables": tables,
            "weight_scale_tensors": table_scales,
            "requires_weight_scale_value_check": bool(table_scales),
            "current_mlx2_external_sidecar_supported": True,
            "native_shards_dot_n_supported": False,
        },
        "mtp": {
            "mlx_json_mtp_num_hidden_layers": native_mtp,
            "mlx_json_num_nextn_predict_layers": nextn,
            "strata_619_gguf_metadata_applicable": False,
        },
    }


def inspect_artifact(path: Path) -> dict:
    root = path.expanduser().resolve()
    config = json.loads((root / "config.json").read_text())
    index_path = root / "model.safetensors.index.json"
    weight_map = {}
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    return {"artifact": str(root), **inspect_mapping(config, weight_map)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {"schema": "mlx2.qwen4-upstream-artifact-probe.v1", "artifacts": []}
    for artifact in args.artifacts:
        try:
            report["artifacts"].append(inspect_artifact(artifact))
        # Report every artifact; do not make partial support look viable.
        except Exception as exc:  # noqa: BLE001
            report["artifacts"].append(
                {
                    "artifact": str(artifact.expanduser().resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return int(any("error" in item for item in report["artifacts"]))


if __name__ == "__main__":
    raise SystemExit(main())
