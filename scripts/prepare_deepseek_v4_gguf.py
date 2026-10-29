#!/usr/bin/env python3
"""Prepare verified native GGUF adapter metadata; no tensors or GPU imported."""
from __future__ import annotations

import argparse
import json

from mlx2.adapters.deepseek_v4_gguf_artifact import prepare_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-reference", required=True)
    args = parser.parse_args()
    artifact = prepare_artifact(args.model_path, args.tokenizer_reference)
    print(json.dumps({"identity": artifact["identity"],
                      "inventory": artifact["inventory"],
                      "qualification": artifact["qualification"]}, indent=2))


if __name__ == "__main__":
    main()
