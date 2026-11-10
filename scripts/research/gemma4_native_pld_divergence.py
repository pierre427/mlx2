"""One-position logprob capture (completion token 77) for the gemma4 native PLD-vs-ordinary prose divergence; run under the GPU queue with an output dir."""
import json, secrets, sys, tempfile, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_rows_attention_ab as h
import gemma4_native_pld_check as p
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
model = Path.home() / "mlx-models/gemma-4-26B-A4B-it-MLX-8bit"
work = Path(tempfile.mkdtemp(prefix="gemma4-native-pld-")); tok = work / "t"; tok.write_text(secrets.token_hex(8)); tok.chmod(0o600)
res = {}
for i, (name, route) in enumerate((("ordinary", ["--ordinary"]), ("default", []))):
    proc, base = p._serve(model, route, 8495 + i, tok, work)
    try:
        body = {"model": h.MODEL_ID[0], "messages": [{"role": "user", "content": p._prompts()["prose"]}],
                "max_tokens": 256, "temperature": 0, "enable_thinking": False,
                "logprobs": True, "top_logprobs": 5, "logprobs_start": 77, "logprobs_end": 77}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
        res[name] = json.loads(urllib.request.urlopen(req, timeout=600).read())
    finally:
        h._stop(proc)
(out / "divergence.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, json.dumps(v["choices"][0]["logprobs"])[:1500], v["choices"][0]["message"]["content"][300:360])
