import json, time, urllib.request, urllib.error, sys
BASE="http://127.0.0.1:8297"; results={}
def post(body):
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time()
    try:
        with urllib.request.urlopen(req,timeout=900) as r: return r.status,json.loads(r.read()),time.time()-t
    except urllib.error.HTTPError as e: return e.code,json.loads(e.read() or b"{}"),time.time()-t
def status(): return json.loads(urllib.request.urlopen(BASE+"/v1/status",timeout=10).read())
def chat(c,**k): return {"messages":[{"role":"user","content":c}],"temperature":0,"enable_thinking":False,**k}
def rec(name,ok,d):
    results[name]={"ok":bool(ok),**d}; print(("PASS " if ok else "FAIL ")+name, json.dumps(d,ensure_ascii=False)[:420],flush=True)
def so(b): return (b.get("mlx2",{}).get("request_controls",{}) or {}).get("structured_output")
def msg(b): return b.get("choices",[{}])[0].get("message",{})
c0=status()["counts"]
# 1 raw grammar that REQUIRES a byte-fallback-only character
code,b,el=post(chat("Output the marker.",grammar="=\U0010ffff=",max_tokens=16))
rec("grammar_requires_U+10FFFF",code==200 and msg(b).get("content")=="=\U0010ffff=",{"code":code,"content":ascii(msg(b).get("content")),"tokens":b.get("usage",{}).get("completion_tokens"),"receipt":so(b),"error":b.get("error",{}).get("message"),"s":round(el,2)})
# 2 emoji class, exactly three
code,b,el=post(chat("Reply with three happy faces.",grammar="[\U0001f600-\U0001f64f]{3}",max_tokens=24))
c=msg(b).get("content") or ""
rec("grammar_emoji_class",code==200 and len(c)==3 and all(0x1f600<=ord(x)<=0x1f64f for x in c),{"code":code,"content":c,"receipt":so(b),"error":b.get("error",{}).get("message"),"s":round(el,2)})
# 3 greedy json_object with an astral character in a free string
ask="Return a JSON object with key \"symbol\" whose value is exactly the alchemical symbol for vinegar (U+1F70A, 🜊) followed by the character U+10FFFF, and key \"name\" set to \"vinegar\"."
code,b,el=post(chat(ask,response_format={"type":"json_object"},max_tokens=80))
try: p=json.loads(msg(b).get("content") or "")
except Exception: p=None
rec("json_object_astral_greedy",code==200 and isinstance(p,dict) and any(ord(ch)>0xFFFF for ch in json.dumps(p,ensure_ascii=False)),{"code":code,"content":msg(b).get("content"),"receipt":so(b),"finish":b.get("choices",[{}])[0].get("finish_reason"),"error":b.get("error",{}).get("message"),"s":round(el,2)})
# 4 sampled
ok=0; outs=[]
for seed in range(4):
    code,b,el=post({**chat("Return a JSON object {\"mood\": <one emoji>, \"rare\": <the alchemical symbol 🜲>}.",response_format={"type":"json_object"},max_tokens=60),"temperature":0.8,"top_p":0.95,"seed":seed})
    try: p=json.loads(msg(b).get("content") or ""); good=code==200 and isinstance(p,dict)
    except Exception: good=False
    ok+=good; outs.append((code,msg(b).get("content"),(so(b) or {}).get("tail_mass_bound")))
rec("json_object_astral_sampled",ok==4,{"ok":ok,"outs":outs})
# 5 thinking + structured (deferral) with an astral character
code,b,el=post({**chat("Think briefly, then return a JSON object with key \"glyph\" set to 🜲.",response_format={"type":"json_object"},max_tokens=700),"enable_thinking":True})
try: p=json.loads(msg(b).get("content") or "")
except Exception: p=None
rec("thinking_deferral_astral",code==200 and isinstance(p,dict) and (so(b) or {}).get("deferred") is True,{"code":code,"content":msg(b).get("content"),"reasoning_chars":len(msg(b).get("reasoning_content") or ""),"receipt":so(b),"finish":b.get("choices",[{}])[0].get("finish_reason"),"error":b.get("error",{}).get("message"),"s":round(el,2)})
# 6 strict schema with optional props, ascii control
schema={"type":"json_schema","json_schema":{"name":"x","strict":True,"schema":{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city"],"additionalProperties":False}}}
code,b,el=post(chat("Describe Paris as JSON.",response_format=schema,max_tokens=60))
try: p=json.loads(msg(b).get("content") or "")
except Exception: p=None
rec("schema_control",code==200 and isinstance(p,dict) and "city" in p,{"code":code,"content":msg(b).get("content"),"receipt":so(b),"tok_per_s":round(b.get("usage",{}).get("completion_tokens",0)/max(el,1e-6),1)})
st=status()
rec("server_alive_final",st["healthy"] and st["inflight"]==0,{"structured_output":st.get("structured_output"),"failures_delta":st["counts"].get("structured_output_failures",0)-c0.get("structured_output_failures",0),"completed_delta":st["counts"].get("completed",0)-c0.get("completed",0)})
json.dump(results,open(sys.argv[1],"w"),indent=2,ensure_ascii=False)
print("SUMMARY",sum(r["ok"] for r in results.values()),"/",len(results))
