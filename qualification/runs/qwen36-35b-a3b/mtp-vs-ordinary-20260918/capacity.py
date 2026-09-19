import json, threading, time, urllib.request, urllib.error, sys
BASE="http://127.0.0.1:8297"
def post(body, timeout=3600):
    req = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}); t=time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read()), time.time()-t
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}"), time.time()-t
def status():
    with urllib.request.urlopen(BASE+"/v1/status", timeout=10) as r: return json.loads(r.read())
def chat(c, **k): return {"messages":[{"role":"user","content":c}], "enable_thinking": False, **k}
results = {}
for width in [int(w) for w in sys.argv[1:]] or [4, 8, 13, 14, 16, 20]:
    time.sleep(3)
    base = status(); base_active = base["metal_active_bytes"]; base_headroom = base["headroom_bytes"]
    samples = []
    stop = threading.Event()
    def poll():
        while not stop.is_set():
            try:
                s = status(); samples.append((s["metal_active_bytes"], s["process_physical_footprint_bytes"], s["headroom_bytes"], s["inflight"], s.get("admission",{}).get("stage")))
            except Exception: pass
            time.sleep(0.25)
    th = threading.Thread(target=poll, daemon=True); th.start()
    out = [None]*width
    def run(i): out[i] = post(chat(f"Write a detailed essay about topic number {i}: the history of bridges.", max_tokens=200, temperature=0.7, seed=1000+i))
    ths = [threading.Thread(target=run, args=(i,)) for i in range(width)]
    t0=time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall=time.time()-t0
    stop.set(); th.join()
    codes = [o[0] for o in out]
    ok = [o for o in out if o[0]==200]
    widths = sorted({w for o in ok for w in ((o[1].get("mlx2",{}).get("mtp") or {}).get("observed_compute_widths") or [])})
    peak_active = max((s[0] for s in samples), default=base_active)
    min_headroom = min((s[2] for s in samples), default=base_headroom)
    max_inflight = max((s[3] for s in samples), default=0)
    stages = sorted({s[4] for s in samples if s[4]})
    toks = sum(o[1]["usage"]["completion_tokens"] for o in ok)
    admission = status().get("admission",{})
    results[width] = dict(codes={c: codes.count(c) for c in set(codes)}, observed_widths=widths, wall_s=round(wall,1), agg_tok_s=round(toks/wall,1),
        metal_active_delta_gib=round((peak_active-base_active)/2**30,2), per_lane_gib=round((peak_active-base_active)/2**30/max(len(ok),1),3),
        min_headroom_gib=round(min_headroom/2**30,1), max_inflight=max_inflight, stages=stages, admission=admission,
        errors=sorted({(o[1].get("error") or {}).get("message","")[:80] for o in out if o[0]!=200}))
    print(json.dumps({width: results[width]}, default=str)[:900], flush=True)
json.dump(results, open(sys.argv[0].replace("capacity.py","capacity-results.json"),"w"), indent=2, default=str)
