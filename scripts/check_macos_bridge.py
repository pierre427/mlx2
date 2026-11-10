"""Exercise native bridge multiplexing, timeout and shutdown using a disposable synthetic child."""
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
        child.write_text('''import json,sys,threading,time
mode = sys.argv[1]
lock = threading.Lock()
def reply(request):
    command = request["command"]
    if command == "slow":
        time.sleep(.5)
        result = "x" * 1048576
    else:
        if command == "wait":
            time.sleep(.4)
        result = {"value": command, "deadline": request.get("bridge_deadline")}
    with lock:
        print(json.dumps({"id": request["id"], "ok": command != "invalid",
            "error": "Synthetic refusal", "result": result}), flush=True)
threads = []
for index, line in enumerate(sys.stdin):
    request = json.loads(line)
    if mode == "stall" and index == 0:
        time.sleep(3)  # stop reading stdin, as when the companion's lane is busy
    if mode == "garbled":
        print("not json", flush=True)
        continue
    # Like the companion, answer concurrently: replies may arrive out of order.
    thread = threading.Thread(target=reply, args=(request,))
    thread.start()
    threads.append(thread)
for thread in threads:
    thread.join()
''')
        subprocess.run(["swiftc", "-swift-version", "5", "-parse-as-library",
                        str(root / "native/macos/Bridge.swift"), str(root / "tests/macos_bridge.swift"),
                        "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(child), sys.executable], check=True, timeout=15)


if __name__ == "__main__":
    main()
