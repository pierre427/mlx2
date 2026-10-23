"""Exercise the actual native chat client against a synthetic loopback SSE server.

Does not contact a real MLX2 endpoint, load a checkpoint or use the GPU.
The embedded agent-app metadata prevents transient Dock/recent-app entries.
"""

import base64
import json
import plistlib
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            assert self.headers.get("Authorization") is None
            assert self.headers.get("Cookie") is None
            if self.path == "/oversized-image":
                self.send_response(200)
                self.send_header("Content-Length", str(16 * 1024 * 1024 + 1))
                self.end_headers()
                try:
                    self.wfile.write(b"x" * 1024)
                    self.wfile.flush()
                    time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == "/image-redirect":
                self.send_response(302)
                self.send_header("Location", "/fixture.png")
                self.end_headers()
                return
            assert self.path == "/fixture.png"
            payload = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            prompt = body["messages"][-1]["content"].split("\n\n[Message time:", 1)[0]
            assert self.path == "/v1/chat/completions"
            if self.headers.get("Authorization") != "Bearer synthetic-secret":
                self.send_response(401)
                self.end_headers()
                return
            if prompt == "redirect":
                self.send_response(307)
                self.send_header("Location", "/must-not-follow")
                self.end_headers()
                return
            if prompt == "bad request":
                payload = json.dumps({"error": {"message": "tools must contain 1 to 64 function definitions"}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for delta in [
                    {"reasoning_content": "Considering"},
                    {"content": "Hello "},
                    {"content": "🌍"},
                ]:
                    chunk = json.dumps(
                        {"choices": [{"delta": delta}]}, ensure_ascii=False
                    )
                    self.wfile.write(f"data: {chunk}\n\n".encode())
                    self.wfile.flush()
                    if prompt == "slow":
                        time.sleep(0.8)
                if prompt != "broken":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mlx2-native-chat-") as raw:
        binary = Path(raw) / "chat-checks"
        info = Path(raw) / "Info.plist"
        info.write_bytes(plistlib.dumps({"LSUIElement": True, "CFBundleIdentifier": "local.mlx2.chat-checks"}))
        sources = sorted((root / "native/macos").glob("*.swift"))
        sources = [str(p) for p in sources if p.name != "MLX2App.swift"]
        subprocess.run(
            [
                "swiftc",
                "-swift-version",
                "5",
                "-parse-as-library",
                "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT",
                "-Xlinker", "__info_plist", "-Xlinker", str(info),
                *sources,
                str(root / "tests/macos_chat.swift"),
                "-o",
                str(binary),
            ],
            check=True,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            subprocess.run(
                [str(binary), f"http://127.0.0.1:{server.server_port}"],
                check=True,
                timeout=30,
            )
        finally:
            server.shutdown()
            server.server_close()
        assert len(requests[1]["messages"]) == 3
        assert requests[1]["messages"][1] == {
            "role": "assistant",
            "content": "Hello 🌍",
            "reasoning_content": "Considering",
        }
        assert len(requests[3]["messages"]) == 1
        assert requests[3]["messages"][0]["role"] == "user"
        content = requests[3]["messages"][0]["content"]
        assert content.startswith("after cancel\n\n[Message time: ")
        assert content.endswith("Use this timestamp only when current date, time, or elapsed time matters to the request.]")
        assert all(r["stream"] and r["max_tokens"] == 4096 for r in requests)
        print(
            "Synthetic HTTP verified: auth, history, isolation, cancel, redirects, truncated stream, server rejection details"
        )


if __name__ == "__main__":
    main()
