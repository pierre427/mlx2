import json,glob,sys
s=json.load(open("status.json")); print(s["phase"], s.get("updated"))
for n in s["order"]:
    e=s["stages"].get(n)
    if not e: continue
    line=f"{n:22s} {e['state']:8s} {e.get('reason','')}"
    try:
        d=json.load(open(f"{n}-sanity.json"))
        bad={t:f"{v['correct']}/{v['n']}" for t,v in d["by_task"].items() if v["correct"]<v["n"]}
        tps=sorted(r["tokens_per_second"] for r in d["rounds_detail"])[len(d["rounds_detail"])//2]
        line+=f" correct={d['graded_correct']}/{d['requests']} http_err={d['http_errors']} issues={d['issue_totals']} peak_w={d['peak_observed_width']} med_tok/s={tps} miss={bad}"
    except FileNotFoundError: pass
    print(line)
