import json, time, urllib.request, urllib.error, sys
def post(body):
    req = urllib.request.Request("http://127.0.0.1:8297/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")
schema = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}, "landmark": {"type": "string"}}, "required": ["city", "country"], "additionalProperties": False}}}
cases = [("sampled json_object default", {"response_format": {"type":"json_object"}, "temperature": 0.7, "seed": 7, "max_tokens": 160}),
         ("sampled schema default", {"response_format": schema, "temperature": 0.7, "seed": 3}),
         ("sampled long string", {"response_format": {"type":"json_object"}, "temperature": 0.7, "seed": 9, "max_tokens": 200}),
         ("greedy json_object", {"response_format": {"type":"json_object"}, "temperature": 0, "max_tokens": 160}),
         ("top_k only", {"response_format": schema, "temperature": 0.7, "top_p": 1.0, "top_k": 40, "seed": 3}),
         ("pure temp", {"response_format": schema, "temperature": 0.7, "top_p": 1.0, "top_k": 0, "seed": 3})]
for label, extra in cases:
    t=time.time()
    prompt = "Describe Paris and its most famous landmark as JSON." if "long" not in label else "Return a JSON object with a single key 'essay' whose value is a 100-word paragraph about Paris."
    code, body = post({"messages":[{"role":"user","content":prompt}], "max_tokens": extra.pop("max_tokens", 80), "enable_thinking": False, **extra})
    c = body.get("choices",[{}])[0].get("message",{}).get("content","")
    try: ok = isinstance(json.loads(c), dict)
    except Exception: ok = False
    n = body.get('usage',{}).get('completion_tokens'); so = body.get("mlx2",{}).get("request_controls",{}).get("structured_output") or {}
    print(f"{label:28s} code={code} valid={ok} finish={body.get('choices',[{}])[0].get('finish_reason')} {round(time.time()-t,1)}s tokens={n} per_tok={round((time.time()-t)/max(n or 1,1),3)}s bound={round(so.get('tail_mass_bound',-1),4)} scans={so.get('parallel_scans')} :: {c[:36]!r} {body.get('error',{}).get('message','')[:50]}", flush=True)
