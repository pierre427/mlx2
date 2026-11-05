"""Check native runtime form values against the actual advertised CPU schemas."""
import json
from pathlib import Path
import subprocess
import tempfile

from mlx2.experimental.weight_space.live import TOOLS


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mlx2-runtime-arguments-") as raw:
        folder = Path(raw)
        schema = folder / "tools.json"
        schema.write_text(json.dumps(TOOLS))
        binary = folder / "checks"
        subprocess.run(["swiftc", "-swift-version", "5", "-parse-as-library",
                        str(root / "native/macos/RuntimeArgumentValue.swift"),
                        str(root / "native/macos/RuntimeOperationPresentation.swift"),
                        str(root / "tests/macos_runtime_arguments.swift"), "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(schema)], check=True, timeout=10)


if __name__ == "__main__":
    main()
