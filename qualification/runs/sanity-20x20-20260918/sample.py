import json,sys
d=json.load(open(sys.argv[1]+"-sanity.json")); rnd=int(sys.argv[2]) if len(sys.argv)>2 else 7
for r in d["records"]:
    if r["round"]==rnd:
        c=r["content"].replace("\n","⏎")
        tc=[(t["function"]["name"],t["function"]["arguments"]) for t in r["tool_calls"]]
        print(f"[{r['task']:18s}] {'OK ' if r['correct'] else 'BAD'} {str(r['finish']):6s} w={r['mlx2'].get('ordinary_compute_width')} tok={r['usage'].get('completion_tokens')} | {c[:230]}{' TOOLS='+str(tc) if tc else ''}{' | REASONING='+r['reasoning'][:60] if r['reasoning'] else ''}")
if len(sys.argv)>3:
    o=json.load(open(sys.argv[3]+"-sanity.json")); same=tot=0
    for a,b in zip(d["records"],o["records"]):
        if a["task"] in("sampled_prose","sampled_topk","two_samples"): continue
        tot+=1; same+=a["content"]==b["content"]
    print(f"greedy outputs identical vs {sys.argv[3]}: {same}/{tot}")
