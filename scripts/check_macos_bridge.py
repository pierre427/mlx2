"""Exercise native bridge timeout and shutdown using a disposable synthetic child."""
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mlx2-bridge-check-") as raw:
        folder = Path(raw)
        binary = folder / "bridge-check"
        child = folder / "child.py"
        child.write_text('''import json,sys,time
mode = sys.argv[1]
for index, line in enumerate(sys.stdin):
    request = json.loads(line)
    if mode == "delayed":
        if index: raise RuntimeError("command written after transport failure")
        time.sleep(.5)
        print(json.dumps({"id": request["id"], "ok": True, "result": "x" * 1048576}), flush=True)
    else:
        invalid = request["command"] == "invalid"
        print(json.dumps({"id": request["id"], "ok": not invalid,
            "error": "Synthetic refusal", "result": {"value": request["command"]}}), flush=True)
''')
        subprocess.run(["swiftc", "-swift-version", "5", "-parse-as-library",
                        str(root / "native/macos/Bridge.swift"), str(root / "tests/macos_bridge.swift"),
                        "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(child), sys.executable], check=True, timeout=15)


if __name__ == "__main__":
    main()
