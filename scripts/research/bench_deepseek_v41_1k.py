"""One 1K serving cell, three repetitions, explicit10-second gaps."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import hashlib
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

from mlx2.desktop_campaign_worker import atomic
from mlx2.runtime.models.deepseek_v41_encoding import encode_messages
from deepseek_v41_context_ladder import http_json, stream


def prepare(model, output):
    from tokenizers import Tokenizer
    from mlx2.adapters.deepseek_v41_artifact import REFERENCE_FILE
    tokenizer=Tokenizer.from_file(str(Path(model)/REFERENCE_FILE))
    sentence="The archive describes stable records, exact model state, verified identifiers, and revision-bound prefix reuse.\n"
    prefix="Background reference notes:\n"
    suffix="\nIgnore the background notes. Count from one to two hundred using numerals, one number per line. Start with 1 immediately. Continue counting without an introduction or explanation."
    def build(repeats,padding=0):
        request={"messages":[{"role":"user","content":prefix+sentence*repeats+" x"*padding+suffix}],"enable_thinking":False}
        rendered=encode_messages(request["messages"],thinking_mode="chat")
        return request,rendered,tokenizer.encode(rendered,add_special_tokens=False).ids
    low,high=0,960
    while low<high:
        middle=(low+high+1)//2
        if len(build(middle)[2])<=960:low=middle
        else:high=middle-1
    for padding in range(128):
        request,rendered,tokens=build(low,padding)
        if len(tokens)==960:break
    else:raise ValueError("could not create exact960-token prompt")
    result=dict(context_budget=1024,prompt_tokens=960,output_tokens=64,request=request,
                rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                token_ids_sha256=hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest())
    atomic(output/"request-1024.json",result)
    return {k:v for k,v in result.items() if k!="request"}


def metrics(base):
    with urllib.request.urlopen(base+"/metrics",timeout=15) as reply:raw=reply.read().decode()
    values={}
    for line in raw.splitlines():
        if not line or line.startswith("#"):continue
        key,value,*_=line.split();values[key]=float(value)
    return raw,values


def summary(values):
    return dict(mean=statistics.mean(values),median=statistics.median(values),
                min=min(values),max=max(values),sample_stddev=statistics.stdev(values) if len(values)>1 else 0)


def run(spec,model_id,*,guard,environment,phase,admin_token):
    out=Path(spec["out"]);base="http://127.0.0.1:8285"
    request=json.loads((out/"request-1024.json").read_text())
    body={**request["request"],"model":model_id,"temperature":0,"max_tokens":64,"min_tokens":64,
          "stream":True,"return_token_trace":True,"logprobs":False}
    report=dict(schema="mlx2.v41-single-cell-perf.v1",checkpoint=spec["checkpoint"],
                source_revision=spec["source_revision"],source_snapshot_sha256=spec["source_snapshot_sha256"],
                context_budget=1024,prompt_tokens=960,output_tokens=64,repetitions=3,delay_seconds=10,
                thermal_controlled=False,qualification=False,performance_claim=False,runs=[],
                scope="same prompt; APCv2 cleared each repetition; model/expert/PLE caches retained; resident dense; ordinary B1;64 outputs via EOS mask; no thermal gate",
                timing_definitions={"prefill_seconds":"execution-lane attachment to first generated token; includes host/runtime and first-token selection, not pure GPU time",
                    "decode_seconds":"sum of63 server inter-token intervals, first through last output token",
                    "decode_receipt_seconds":"request elapsed minus TTFT, includes terminal work",
                    "delay":"previous HTTP completion to next request dispatch; setup performed within10-second gap"})
    previous_end=None
    for repeat in range(3):
        guard();directory=out/f"repeat-{repeat+1}";directory.mkdir(exist_ok=False)
        cleared=http_json(base,"/v1/admin/apc/clear",{"reason":f"V41-1K-repeat-{repeat+1}"},timeout=60,admin_token=admin_token)
        before=http_json(base,"/v1/status")
        raw,before_metrics=metrics(base);(directory/"metrics-before.prom").write_text(raw)
        if previous_end is not None:
            while (remaining:=previous_end+10-time.monotonic())>0:
                guard();time.sleep(min(1,remaining))
        guard();thermal_before=environment();started=time.monotonic();start_wall=time.time()
        phase("cell_repeat_started",repeat=repeat+1,gap_seconds=None if previous_end is None else started-previous_end)
        response=stream(base,body,directory/"response.sse",timeout=1200)
        ended=time.monotonic();end_wall=time.time()
        after=http_json(base,"/v1/status");batching=http_json(base,"/v1/status/batching")
        raw,after_metrics=metrics(base);(directory/"metrics-after.prom").write_text(raw)
        delta={key:value-before_metrics.get(key,0) for key,value in after_metrics.items()}
        receipt=response["receipt"]
        prefill=delta["mlx2_request_prefill_time_seconds_sum"]
        decode=delta["mlx2_inter_token_latency_seconds_sum"]
        intervals=delta["mlx2_inter_token_latency_seconds_count"]
        if (receipt["prompt_tokens"]!=960 or receipt["cached_tokens"]!=0 or receipt["completion_tokens"]!=64
                or receipt["route"]!="ordinary" or set(response["token_trace"]["execution_widths"])!={1}
                or delta["mlx2_request_prefill_time_seconds_count"]!=1 or intervals!=63 or min(prefill,decode)<=0):
            atomic(directory/"failed-response.json",dict(response=response,metrics_delta=delta))
            raise RuntimeError("single1K cell timing/shape/cache contract failed")
        buckets=[]
        for key,value in delta.items():
            match=re.fullmatch(r'mlx2_inter_token_latency_seconds_bucket\{le="([^"]+)"\}',key)
            if match:buckets.append((float(match[1]),value))
        quantiles={}
        for name,q in [("p50",.5),("p95",.95),("p99",.99)]:
            upper=next((bound for bound,count in sorted(buckets) if count>=intervals*q),None)
            quantiles[name+"_histogram_upper_ms"]=None if upper is None or upper==float("inf") else upper*1000
        counts={key:value-before.get("counts",{}).get(key,0) for key,value in after.get("counts",{}).items()
                if key.startswith("v41_expert_")}
        row=dict(repeat=repeat+1,started_at=start_wall,ended_at=end_wall,
                 gap_seconds=None if previous_end is None else started-previous_end,
                 prefill_seconds=prefill,prefill_tokens_per_second=960/prefill,
                 decode_seconds=decode,decode_tokens_per_second=63/decode,inter_token_mean_ms=1000*decode/63,
                 TTFT_seconds=response["TTFT_seconds"],request_seconds=response["elapsed_seconds"],
                 HTTP_seconds=response["HTTP_seconds"],receipt_decode_tokens_per_second=response["observed_decode_tokens_per_second"],
                 queue_seconds=delta.get("mlx2_request_queue_time_seconds_sum"),
                 memory={key:after.get(key) for key in ["metal_active_bytes","metal_peak_bytes","process_physical_footprint_bytes"]},
                 expert_counter_delta=counts,thermal_before=thermal_before,thermal_after=environment(),
                 before=before,after=after,batching=batching,response=response,APC_clear=cleared,**quantiles)
        atomic(directory/"result.json",row);report["runs"].append(row);atomic(out/"cell-1024.json",report)
        phase("cell_repeat_complete",repeat=repeat+1,prefill_tps=row["prefill_tokens_per_second"],decode_tps=row["decode_tokens_per_second"],TTFT_seconds=row["TTFT_seconds"],request_seconds=row["request_seconds"])
        previous_end=ended
    fields=["prefill_seconds","prefill_tokens_per_second","decode_seconds","decode_tokens_per_second","inter_token_mean_ms","TTFT_seconds","request_seconds","HTTP_seconds","receipt_decode_tokens_per_second"]
    report["averages"]={key:summary([row[key] for row in report["runs"]]) for key in fields}
    report["aggregate_prefill_tps"]=2880/sum(row["prefill_seconds"] for row in report["runs"])
    report["aggregate_decode_tps"]=189/sum(row["decode_seconds"] for row in report["runs"])
    report["same_generated_tokens"]=all(row["response"]["token_ids"]==report["runs"][0]["response"]["token_ids"] for row in report["runs"])
    report["passed"]=True;atomic(out/"cell-1024.json",report)
    return report
