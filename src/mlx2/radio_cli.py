"""Encode an image with a standalone C-RADIO checkpoint."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--output", required=True, help="output NPZ (summary and features)"
    )
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args(argv)
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    from .adapters.radio import RadioImageAdapter

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    adapter = RadioImageAdapter(args.model)
    with Image.open(args.image) as image:
        pixels = adapter.preprocess(image, size=args.size)
    result = adapter.encode(pixels)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez(
            stream,
            summary=np.asarray(result.summary),
            features=np.asarray(result.features),
        )
    receipt = {**adapter.receipt, "device": args.device, "output": str(output)}
    output.with_suffix(output.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
