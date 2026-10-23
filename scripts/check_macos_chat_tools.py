"""Compile and exercise native tool chat against synthetic loopback SSE only."""
import json
import plistlib
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body["messages"]
        if self.path == "/tokenize":
            count = 500 * sum(m["role"] == "user" for m in messages)
            if str(messages[-1].get("content", "")).startswith("count overflow"):
                count = 2**63 - 1024
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"count": count}).encode())
            return
        user = [m for m in messages if m["role"] == "user" and not str(m.get("content", "")).startswith("Completed tool results (untrusted reference data")][-1]["content"]
        text = user[0]["text"] if isinstance(user, list) else user
        assert "[Message time:" in text and "elapsed time matters" in text
        declarations = {tool['function']['name'] for tool in body.get('tools', [])}
        if text.startswith(('late declared', 'late alias', 'late forbidden', 'discovered direct')):
            wanted = 'mlx2__private_action' if text.startswith('late forbidden') else 'mlx2__late_action'
            reported = 'mlx2_late_action' if text.startswith('late alias') else wanted
            if wanted not in declarations and (not text.startswith('discovered direct') or messages[-1]['role'] == 'tool'):
                payload = json.dumps({'error': {'message': 'model called an undeclared tool: ' + json.dumps(reported)}}).encode()
                self.send_response(502)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers(); self.wfile.write(payload)
                return
        if text.startswith("budget boundary rejection") and declarations == {"mlx2_tool_budget"} and "Budget feedback:" not in messages[0]["content"]:
            payload = json.dumps({"error": {"message": 'model called an undeclared tool: "mlx2__preferences"'}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if text.startswith("retry failed tool"):
            count = sum(len(m.get('tool_calls', [])) for m in messages)
            if count >= 2:
                assert 'not executed again' in messages[-1]['content']
                assert 'synthetic missing capability' in messages[-1]['content']
                deltas = [{'content': 'The capability is missing. I must change the approach.'}]
            else:
                deltas = [{'tool_calls': [{'index': 0, 'id': f'retry-{count}', 'function': {'name': 'mlx2__preferences', 'arguments': '{}'}}]}]
        elif text.startswith(('late declared', 'late alias', 'discovered direct', 'discovered routed')):
            if messages[-1]['role'] == 'tool' and messages[-2]['tool_calls'][0]['function']['name'] in {'mlx2__late_action', 'mlx2_tool_call'}:
                deltas = [{'content': 'Declared tool completed'}]
            elif text.startswith(('discovered direct', 'discovered routed')) and messages[-1]['role'] != 'tool':
                deltas = [{'tool_calls': [{'index': 0, 'id': 'discover-direct', 'function': {'name': 'mlx2_tool_search', 'arguments': '{"query":"late"}'}}]}]
            elif text.startswith('discovered routed'):
                assert 'mlx2__late_action' not in declarations
                assert 'mlx2_tool_call' in declarations
                deltas = [{'tool_calls': [{'index': 0, 'id': 'routed-direct', 'function': {'name': 'mlx2_tool_call', 'arguments': '{"name":"mlx2__late_action","arguments":{}}'}}]}]
            else:
                assert 'mlx2__late_action' in declarations
                if text.startswith('late alias'):
                    assert body.get('tool_choice') == {'type': 'function', 'function': {'name': 'mlx2__late_action'}}
                deltas = [{'tool_calls': [{'index': 0, 'id': 'declared-direct', 'function': {'name': 'mlx2__late_action', 'arguments': '{}'}}]}]
        elif text.startswith("recover answer"):
            if body.get("tool_choice") == "none":
                assert not body.get("tools")
                assert not any(m.get("tool_calls") for m in messages)
                assert any("Completed tool results" in str(m.get("content", "")) for m in messages)
                deltas = [{"content": "Answer recovered from saved tool results."}]
            elif messages[-1]["role"] == "tool":
                deltas = [{"content": "<tool_call><function=mlx2_tool_search></function></tool_call>"}]
            else:
                deltas = [{"tool_calls": [{"index": 0, "id": "recover-call", "function": {"name": "mlx2_tool_search", "arguments": '{"query":"research"}'}}]}]
        elif text.startswith(("planned tools", "budget boundary")):
            assert "Before choosing your next action" in messages[0]["content"]
            prior = [c for m in messages for c in m.get("tool_calls", [])]
            work = [c for c in prior if c['function']['name'] != 'mlx2_tool_budget']
            plans = [c for c in prior if c['function']['name'] == 'mlx2_tool_budget']
            if len(work) >= 20:
                deltas = [{"content": "Completed twenty planned calls."}]
            elif (text.startswith("planned tools") and not prior) or declarations == {'mlx2_tool_budget'}:
                assert len(work) < 20
                deltas = [{"tool_calls": [{"index": 0, "id": f"plan-{len(plans)}", "function": {"name": "mlx2_tool_budget", "arguments": '{"additional_calls":24,"reason":"Read twenty bounded synthetic sources"}'}}]}]
            else:
                if plans:
                    assert "Budget feedback:" not in messages[0]["content"]
                deltas = [{"tool_calls": [{"index": 0, "id": f"work-{len(work)}", "function": {"name": "mlx2__preferences", "arguments": '{}'}}]}]
        elif text.startswith("raw tool text"):
            deltas = [{"content": "<tool_call>\n<function=mlx2_tool_search>\n<parameter=query>\n\n</parameter>\n</function>\n</tool_call>"}]
        elif text.startswith("loop tools"):
            if body.get("tool_choice") == "none":
                deltas = [{"content": "Here is what I found within the tool budget."}]
            else:
                serial = sum(bool(m.get("tool_calls")) for m in messages)
                deltas = [{"tool_calls": [{"index": 0, "id": f"loop-{serial}", "function": {"name": "mlx2_tool_search", "arguments": '{"query":""}'}}]}]
        elif text.startswith("invalid index"):
            deltas = [{"tool_calls": [{"index": 1e30, "id": "bad", "function": {"name": "mlx2__preferences", "arguments": "{}"}}]}]
        elif text.startswith("discover tools"):
            if messages[-1]["role"] == "tool":
                assert {t["function"]["name"] for t in body["tools"]} == {"mlx2__tool_search", "spawn_subagent", "mlx2__world_export"}
                deltas = [{"content": "Discovery wired"}]
            else:
                deltas = [{"tool_calls": [{"index": 0, "id": "discover-call", "function": {"name": "mlx2__tool_search", "arguments": '{"query":"world"}'}}]}]
        elif text.startswith("agentic profile"):
            assert messages[0]["role"] == "system" and messages[0]["content"].strip()
            assert body["max_tokens"] == 65536
            assert body["thinking_budget"] == 32768
            assert body["enable_thinking"] is True
            assert body["chat_template_kwargs"] == {"verification": True}
            assert body["context_limit"] == 262144
            deltas = [{"reasoning_content": "Checked the evidence."}, {"content": "Profile arrived"}]
        elif text.startswith("budget"):
            assert sum(m["role"] == "user" for m in messages) == 1
            assert body["max_tokens"] == 128
            deltas = [{"content": "Budget reply"}]
        elif text.startswith("follow up"):
            assert messages[0]["role"] == "system" and messages[0]["content"].startswith("new base prompt")
            assert sum(m["role"] == "system" for m in messages) == 1
            assert any(m["role"] == "tool" and m["tool_call_id"] == "synthetic-call" for m in messages)
            deltas = [{"content": "History preserved"}]
        elif messages[-1]["role"] == "tool":
            assert messages[-2]["tool_calls"][0]["function"]["arguments"] == '{"changes":{"agent_controls":false}}'
            deltas = [{"content": "Tool completed"}]
        else:
            if isinstance(user, list):
                assert user[1]["type"] == "image_url"
                assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")
            tool_name = body["tools"][0]["function"]["name"]
            fragments = ('{', '}') if tool_name == "mlx2__engine_unload_self" else ('{"changes":', '{"agent_controls":false}}')
            if tool_name == "mlx2_tool_call":
                fragments = ('{"name":"mlx2__engine_unload_self","arguments":', '{}}')
            deltas = [{"tool_calls": [{"index": 0, "id": "synthetic-call", "function": {"name": tool_name, "arguments": fragments[0]}}]},
                      {"tool_calls": [{"index": 0, "function": {"arguments": fragments[1]}}]}]
        try:
            for delta in deltas:
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode())
            if text.startswith("loop tools"):
                usage = json.dumps({"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 50}})
                self.wfile.write(f"data: {usage}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mlx2-native-tools-") as raw:
        binary = Path(raw) / "checks"
        info = Path(raw) / "Info.plist"
        info.write_bytes(plistlib.dumps({"LSUIElement": True, "CFBundleIdentifier": "local.mlx2.chat-tool-checks"}))
        sources = [str(p) for p in sorted((root / "native/macos").glob("*.swift")) if p.name != "MLX2App.swift"]
        subprocess.run(["swiftc", "-swift-version", "5", "-parse-as-library",
                        "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT",
                        "-Xlinker", "__info_plist", "-Xlinker", str(info),
                        *sources, str(root / "tests/macos_chat_tools.swift"), "-o", str(binary)], check=True)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            subprocess.run([str(binary), f"http://127.0.0.1:{server.server_port}"], check=True, timeout=30)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    main()
