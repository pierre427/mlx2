"""CPU-only SHA verification, topology/tokenizer admission and integrity seal."""

import argparse
import json
from pathlib import Path

from mlx2.adapters.glm53_gguf_artifact import prepare_artifact

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument(
        "--native-read-view",
        action="store_true",
        help="APFS clone with a native packed safetensors read header",
    )
    args = parser.parse_args()
    seal = prepare_artifact(args.model_path)
    if args.native_read_view:
        from mlx2.adapters.glm53_gguf_artifact import FILENAME
        from mlx2.runtime.gguf.native_read_view import prepare_read_view

        root = Path(args.model_path).resolve()
        seal["native_read_view"] = str(
            prepare_read_view(
                root / FILENAME,
                root / "mlx2-native-packed.safetensors",
                source_sha256=seal["weight_sha256"],
            )
        )
    print(json.dumps(seal, indent=2))
