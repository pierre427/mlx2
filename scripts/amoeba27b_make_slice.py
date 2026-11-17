#!/usr/bin/env python3
"""Write a reduced-layer slice of the 27B q8 build (first --layers decoder layers, embeddings, final
norm, lm_head; no MTP, no vision) for CPU smoke tests of the amoeba trainers on the real architecture
(Gated DeltaNet + full attention). A code-test artifact only: never a result."""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "mlx-models/Qwen3.8-27B-BF16-q8-mlx2"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((src / "config.json").read_text())
    t = cfg["text_config"]
    t["num_hidden_layers"] = a.layers
    t["layer_types"] = t["layer_types"][:a.layers]
    t["mtp_num_hidden_layers"] = 0
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    def keep(k):
        if k.startswith(("mtp.", "model.visual")):
            return False
        if k.startswith("model.language_model.layers."):
            return int(k.split(".")[3]) < a.layers
        return True
    keys = [k for k in idx if keep(k)]
    tensors = {}
    for f in sorted({idx[k] for k in keys}):
        arrs = mx.load(str(src / f))
        for k in keys:
            if idx[k] == f:
                tensors[k] = arrs[k]
        mx.eval(list(tensors.values()))
    mx.save_safetensors(str(out / "model.safetensors"), tensors, metadata={"format": "mlx"})
    (out / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"):
        if (src / n).exists():
            shutil.copy2(src / n, out / n)
    print(f"wrote {out}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
