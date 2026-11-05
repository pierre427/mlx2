#!/usr/bin/env python3
"""One bounded Kimi loopback unit: cold/warm APCv2, default context, metrics.

Requires the caller's CPG and paired locks. This is functional evidence,
without thermal holds or quoted performance statistics. Always stops server.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpg-lock",type=Path,required=True)
    parser.add_argument("--port",type=int,default=18291)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise RuntimeError("nonempty evidence directory")
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from qualification_gpu_ownership import require_qualification_lease
    owner=json.loads((args.cpg_lock/"owner.json").read_text())
    lease=require_qualification_lease(task_id=owner["cpg_task"],generation=owner["cpg_generation"],cpg_owner_lock=args.cpg_lock)
    from mlx2.adapters.kimi_linear import REPO, REVISION
    receipt={"checkpoint":REPO+"@"+REVISION,
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "ownership":lease,"qualified":False,"rows":[],"kind":"loopback functional smoke",
        "thermal_before":subprocess.check_output(["pmset","-g","therm"],text=True).strip(),
        "swap_before":subprocess.check_output(["sysctl","-n","vm.swapusage"],text=True).strip()}
    root=f"http://127.0.0.1:{args.port}"

    def get(path):
        with urlopen(root+path,timeout=30) as response:return json.load(response)

    command=[sys.executable,"-m","mlx2.server","--model",str(args.model),"--host","127.0.0.1",
             "--port",str(args.port),"--max-lanes","2","--max-inflight","2","--prefill-step","128",
             "--cache-bytes",str(1<<30),"--cache-dir",str(args.output/"cache")]
    log=(args.output/"server.log").open("w")
    server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt["server_pid"]=server.pid
    try:
        deadline=time.monotonic()+180
        while True:
            if server.poll() is not None:raise RuntimeError("server exited before readiness")
            try:
                status=get("/v1/status")
                if status.get("ready") or status.get("state")=="ready" or status.get("status")=="ready":break
            except (URLError,TimeoutError):pass
            if time.monotonic()>deadline:raise RuntimeError("server readiness timed out")
            time.sleep(.5)
        receipt["startup_status"]=status
        prompt="Archive notes: "+"Inventory checked. "*100+"\nThe shipment identifier is ZX-3187. Reply with the shipment identifier only."
        payload={"model":args.model.name,"messages":[{"role":"user","content":prompt}],
                 "temperature":0,"max_tokens":32,"stream":False}
        for arm in ("cold","warm"):
            request=Request(root+"/v1/chat/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
            with urlopen(request,timeout=180) as response:result=json.load(response)
            content=result["choices"][0]["message"].get("content","")
            receipt["rows"].append({"arm":arm,"result":result,"oracle_passed":"ZX-3187" in content})
        # Worker diagnostics publish on a one-second cadence. An immediate
        # post-response scrape can legitimately contain the startup zeros.
        publication_deadline=time.monotonic()+10
        while True:
            final=get("/v1/status")
            execution=final.get("execution",{})
            if sum(execution.get("kda",{}).values()) and sum(execution.get("mla",{}).values()):break
            if time.monotonic()>publication_deadline:raise RuntimeError("KDA/MLA counters did not publish")
            time.sleep(.2)
        receipt["final_status"]=final
        with urlopen(root+"/metrics",timeout=30) as response:metrics=response.read().decode()
        (args.output/"metrics.prom").write_text(metrics)
        receipt["metrics"]= {name:name in metrics for name in ("mlx2_kda_events_total","mlx2_mla_events_total")}
        warm_cached=receipt["rows"][1]["result"]["usage"]["prompt_tokens_details"]["cached_tokens"]
        receipt["passed"]=(all(row["oracle_passed"] for row in receipt["rows"])
            and all(receipt["metrics"].values()) and warm_cached>0 and final.get("max_context")==1048576)
        print("HTTP cold/warm",receipt["passed"],flush=True)
    except BaseException as error:
        receipt["passed"]=False
        receipt["error"]=repr(error)
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL)
                server.wait(timeout=5)
        log.close()
        receipt["server_exit_code"]=server.returncode
        receipt["server_stopped"]=server.poll() is not None
        receipt["swap_after"]=subprocess.check_output(["sysctl","-n","vm.swapusage"],text=True).strip()
        receipt["thermal_after"]=subprocess.check_output(["pmset","-g","therm"],text=True).strip()
        tmp=args.output/"receipt.tmp"
        tmp.write_text(json.dumps(receipt,indent=2,default=str)+"\n")
        os.replace(tmp,args.output/"receipt.json")
    return 0 if receipt["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
