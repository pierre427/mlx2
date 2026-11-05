#!/usr/bin/env python3
"""Capture Kimi real-weight BF16 projection/gather prefill dispatches.

A fresh process bounds captured resources to one q projection and one expert
bank. This tests installed MLX dispatch, not a substituted quantized kernel.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpg-lock",type=Path,required=True)
    parser.add_argument("--inspection-seconds",type=int,default=0)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise RuntimeError("nonempty evidence directory")
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from qualification_gpu_ownership import require_qualification_lease
    owner=json.loads((args.cpg_lock/"owner.json").read_text())
    lease=require_qualification_lease(task_id=owner["cpg_task"],generation=owner["cpg_generation"],cpg_owner_lock=args.cpg_lock)
    import mlx.core as mx

    from mlx2.adapters.kimi_linear import REPO, REVISION
    mx.set_default_device(mx.gpu)
    index=json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    qkey="model.layers.0.self_attn.q_proj.weight"
    names=[f"model.layers.1.block_sparse_moe.experts.{i}.w1.weight" for i in range(180)]
    selected=set(names+[qkey])
    weights={}
    for name in sorted({index[k] for k in selected}):
        weights.update({k:v for k,v in mx.load(args.model/name).items() if k in selected})
    q=weights.pop(qkey)
    bank=mx.stack([weights.pop(k) for k in names])
    x=mx.ones((128,2304),mx.bfloat16)
    # The production SwitchGLU path sorts one assignment per token/expert.
    # 128 rows x top-8 = 1024 sorted assignments meets the native RHS floor
    # for 180 experts. Restrict the tokens to eight chosen experts as a valid
    # routing case, retaining the complete original 180-expert weight bank.
    ids=mx.repeat(mx.arange(8,dtype=mx.uint32),128)
    rows=mx.broadcast_to(x,(8,128,2304)).reshape(1024,1,2304)
    mx.eval(q,bank,x,ids,rows)
    trace=args.output/"kimi-bf16-prefill.gputrace"
    mx.synchronize()
    mx.metal.start_capture(str(trace))
    try:
        y=x@q.T
        z=mx.gather_mm(rows,bank.swapaxes(-1,-2),rhs_indices=ids,sorted_indices=True)
        mx.eval(y,z)
        mx.synchronize()
    finally:mx.metal.stop_capture()
    receipt={"checkpoint":REPO+"@"+REVISION,
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "ownership":lease,"device":mx.device_info(),"capture":str(trace),
        "profiles":{"dense":{"input":[128,2304],"weight":[4096,2304]},
                    "moe":{"assignments":1024,"experts":180,"weight":[180,1024,2304]}},
        "dtype":"BF16","qualified":False,"full_model_request":False,
        "scope":"real checkpoint matrices; valid synthetic hidden rows/routing; native kernel dispatch evidence"}
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2,default=str)+"\n")
    print("CAPTURE_READY",trace,flush=True)
    # A short explicit inspection window lets Xcode replay the same trace
    # while the caller still holds both locks. No unbounded ownership wait.
    deadline=time.monotonic()+min(120,max(0,args.inspection_seconds))
    while time.monotonic()<deadline and not (args.output/"inspection-complete").exists():
        time.sleep(1)
    return 0


if __name__=="__main__":raise SystemExit(main())
