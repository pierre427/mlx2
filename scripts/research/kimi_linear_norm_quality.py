#!/usr/bin/env python3
"""Paired Kimi answer-quality probe: FLA FP32 vs Qwen BF16 norm rounding.

All other arithmetic, weights, prompts and greedy sampling stay fixed. This
is bounded functional evidence, not a general quality or performance benchmark.
Requires a live CPG lease and matching paired locks. No thermal admission hold.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def tasks():
    items = [
        ("multiply","What is 37 * 48? Reply with the integer only.","1776"),
        ("arithmetic","Compute 144 + 289 - 73. Reply with the integer only.","360"),
        ("gcd","What is the greatest common divisor of 84 and 126? Integer only.","42"),
        ("prime_no","Is 123 prime? Reply YES or NO only.","NO"),
        ("prime_yes","Is 127 prime? Reply YES or NO only.","YES"),
        ("work_rate","Seven workers finish a job in five days. At the same individual rate, how many days do five workers need? Integer only.","7"),
        ("fraction","Add 3/4 and 5/6. Reply with the reduced improper fraction only.","19/12"),
        ("capital","What is the capital of Canada? City name only.","Ottawa"),
        ("symbol","What is the chemical symbol for gold? Symbol only.","Au"),
        ("sort","Sort these integers in ascending order: 9, -2, 4, 0. Reply with comma-separated integers only.","-2,0,4,9"),
        ("identifier","The access code is ZX-3187. Repeat the access code exactly, and nothing else.","ZX-3187"),
        ("json","Reply with only a JSON object containing keys sum and product for the numbers 8 and 9.",{"sum":17,"product":72}),
        ("python_even","Write one Python expression using n that is true exactly when integer n is even. No explanation or code fences.","python_even"),
        ("python_square","Write one Python expression using x that computes x squared. No explanation or code fences.","python_square"),
    ]
    filler = " ".join(f"Archive row {i}: Routine inventory checked; no access code appears in this row." for i in range(240))
    items += [
        ("retrieve_start","The shipment identifier is LM-8246.\n"+filler+"\nWhat is the shipment identifier? Reply with that identifier only.","LM-8246"),
        ("retrieve_middle",filler[:len(filler)//2]+"\nThe verification key is RT-5902.\n"+filler[len(filler)//2:]+"\nRepeat the verification key exactly, nothing else.","RT-5902"),
    ]
    return items


def oracle(text,expected):
    text = text.strip()
    for dash in "‐‑‒–—−": text = text.replace(dash,"-")
    if isinstance(expected,dict):
        try: return json.loads(text) == expected
        except ValueError: return False
    if expected in {"python_even","python_square"}:
        expression = text.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        try:
            tree = ast.parse(expression,mode="eval")
            allowed = (ast.Expression,ast.BinOp,ast.UnaryOp,ast.Compare,ast.Name,ast.Load,ast.Constant,
                       ast.Mod,ast.Mult,ast.Pow,ast.Add,ast.Sub,ast.USub,ast.Eq,ast.NotEq,ast.Div,ast.FloorDiv)
            if sum(1 for _ in ast.walk(tree)) > 40 or any(not isinstance(n,allowed) for n in ast.walk(tree)):
                return False
            if any(isinstance(n,ast.Constant) and (type(n.value) not in {int,float} or abs(n.value)>10)
                   for n in ast.walk(tree)): return False
            code = compile(tree,"<bounded-expression-oracle>","eval")
            variable = "n" if expected == "python_even" else "x"
            return all(eval(code,{"__builtins__":{}},{variable:n}) == (n%2==0 if variable=="n" else n*n)
                       for n in range(-6,7))
        except (SyntaxError,NameError,TypeError,ValueError,ZeroDivisionError): return False
    compact = re.sub(r"\s+","",text).strip(".。")
    return compact == expected


def answer_oracle(text,expected):
    """Answer correctness independent of the separately scored output format."""
    if oracle(text,expected): return True
    if not isinstance(expected,str) or expected.startswith("python_"): return False
    normalized=text.replace("**","").strip()
    if re.fullmatch(r"-?\d+(?:/\d+)?",expected):
        numbers=re.findall(r"-?\d+(?:/\d+)?",normalized)
        return bool(numbers) and numbers[-1]==expected
    return re.search(r"(?<!\w)"+re.escape(expected)+r"(?!\w)",normalized[-80:]) is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpg-lock",type=Path,required=True)
    parser.add_argument("--max-tokens",type=int,default=96)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise RuntimeError("nonempty evidence directory")
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"],generation=owner["cpg_generation"],cpg_owner_lock=args.cpg_lock)

    import mlx.core as mx

    from mlx2.adapters.kimi_linear import (
        REPO,
        REVISION,
        KimiLinearAdapter,
        parse_policy,
    )

    mx.set_default_device(mx.gpu)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    source = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    receipt = {"checkpoint":REPO+"@"+REVISION,"source_revision":source,"ownership":lease,
        "kind":"paired deterministic norm-rounding answer-quality probe","qualified":False,
        "general_quality_claim":False,"rows":[],"max_tokens":args.max_tokens,
        "device":mx.device_info(),"sampling":{"temperature":0},"prefill_chunk":128,
        "thermal_before":subprocess.check_output(["pmset","-g","therm"],text=True).strip(),
        "swap_before":subprocess.check_output(["sysctl","-n","vm.swapusage"],text=True).strip()}

    def atomic():
        tmp = args.output/"receipt.tmp"
        tmp.write_text(json.dumps(receipt,indent=2,default=str)+"\n")
        os.replace(tmp,args.output/"receipt.json")

    adapter = None
    try:
        adapter = KimiLinearAdapter(args.model)
        receipt["artifact_fingerprint"] = adapter.identity["fingerprint"]
        receipt["loaded_bytes"] = mx.get_active_memory()

        def generate(prompt,policy,*,capture=None,force_tokens=None):
            adapter.model.configure(parse_policy(policy))
            cache = adapter.model.make_cache()
            for start in range(0,len(prompt),128):
                logits = adapter.model(mx.array([prompt[start:start+128]],mx.int32),cache=cache)
                mx.eval(logits,cache)
            output=[]
            window=None
            for pos in range(args.max_tokens):
                chosen=int(mx.argmax(logits[0,-1]).item())
                if capture is not None and pos == capture["position"]:
                    raw = logits[0,-1].astype(mx.float32)
                    lp = raw-mx.logsumexp(raw)
                    top = mx.argsort(-lp)[:8].tolist()
                    ids = list(dict.fromkeys(top+capture["tokens"]))
                    window={"position":pos,"greedy":chosen,"top8":top,
                            "logprobs":dict(zip(map(str,ids),lp[mx.array(ids)].tolist()))}
                    break
                if force_tokens is not None:
                    chosen=force_tokens[pos]
                output.append(chosen)
                if chosen in adapter.tokenizer.eos_token_ids: break
                logits=adapter.model(mx.array([[chosen]],mx.int32),cache=cache)
            mx.eval(cache)
            dtype={"kv":str(cache[3].keys.dtype),"recurrent":str(cache[0][3].dtype),
                   "conv":str(cache[0][0].dtype),"logits":str(logits.dtype)}
            return {"tokens":output,"text":adapter.tokenizer.decode(output),"window":window,"dtypes":dtype}

        smoke_prompt=adapter.prompt_tokens({"messages":[{"role":"user","content":"What is 2 + 2? Reply with the integer only."}]})
        smoke=generate(smoke_prompt,{"mla":"expanded","kda":"reference"})
        receipt["ordinary_reference_smoke"]=smoke
        receipt["ordinary_reference_smoke_passed"]=oracle(smoke["text"].replace("<|im_end|>",""),"4")
        atomic()
        if not receipt["ordinary_reference_smoke_passed"]: raise RuntimeError("ordinary reference smoke failed")

        # Explicit debug capture: Metal can include all live resources, so
        # do not capture by default in a full-model quality run.
        try:
            if os.environ.get("MLX2_KIMI_CAPTURE") != "1":
                raise RuntimeError("capture not requested; existing capture retained separately")
            mx.synchronize()
            mx.metal.start_capture(str(args.output/"kimi-q-prefill.gputrace"))
            try:
                q = adapter.model.layers[0].attention.q_proj
                probe = mx.ones((1,128,2304),mx.bfloat16)
                mx.eval(q(probe))
                mx.synchronize()
            finally: mx.metal.stop_capture()
            receipt["nax_capture"]="kimi-q-prefill.gputrace"
        except RuntimeError as error:
            receipt["nax_capture_error"]=str(error)
        atomic()

        for index,(name,text,expected) in enumerate(tasks()):
            prompt=adapter.prompt_tokens({"messages":[{"role":"user","content":text}]})
            row={"task":name,"prompt_tokens":len(prompt),"prompt":text,"expected":expected,"arms":{}}
            for norm in (("fla_fp32","qwen_bf16") if index%2==0 else ("qwen_bf16","fla_fp32")):
                result=generate(prompt,{"norm_rounding":norm})
                # EOS remains in raw output; remove only that known marker at
                # the oracle boundary, never identifier characters or case.
                answer=result["text"].replace("<|im_end|>","")
                result["format_passed"]=oracle(answer,expected)
                result["truncated"]=not bool(result["tokens"] and result["tokens"][-1] in adapter.tokenizer.eos_token_ids)
                result["passed"]=not result["truncated"] and answer_oracle(answer,expected)
                row["arms"][norm]=result
            a,b=(row["arms"][n] for n in ("fla_fp32","qwen_bf16"))
            div=next((i for i,(x,y) in enumerate(zip(a["tokens"],b["tokens"])) if x!=y),None)
            row["exact_token_parity"]=a["tokens"]==b["tokens"]
            if div is not None:
                capture={"position":div,"tokens":[a["tokens"][div],b["tokens"][div]]}
                windows={}
                for norm in ("fla_fp32","qwen_bf16"):
                    windows[norm]=generate(prompt,{"norm_rounding":norm},capture=capture,
                                            force_tokens=row["arms"][norm]["tokens"])["window"]
                row["first_divergence"]=windows
                ta,tb=capture["tokens"]
                wa,wb=windows["fla_fp32"],windows["qwen_bf16"]
                replay_exact=wa["greedy"]==ta and wb["greedy"]==tb
                margin_a=wa["logprobs"][str(ta)]-wa["logprobs"][str(tb)]
                margin_b=wb["logprobs"][str(tb)]-wb["logprobs"][str(ta)]
                row["numerical_classification"]=("near_tie_equivalent" if replay_exact and a["passed"] and b["passed"]
                    and 0<=margin_a<=.5 and 0<=margin_b<=.5 else "unclassified")
                row["cross_selected_margins"]=[margin_a,margin_b]
            receipt["rows"].append(row)
            receipt["scores"]={n:sum(r["arms"][n]["passed"] for r in receipt["rows"]) for n in ("fla_fp32","qwen_bf16")}
            receipt["format_scores"]={n:sum(r["arms"][n]["format_passed"] for r in receipt["rows"]) for n in ("fla_fp32","qwen_bf16")}
            atomic()
            print(name,receipt["scores"],"exact",row["exact_token_parity"],flush=True)
        receipt["diagnostics"]=adapter.diagnostics()
        receipt["completed"]=True
    except BaseException as error:
        receipt["error"]=repr(error)
        raise
    finally:
        if adapter is not None: adapter.close()
        receipt["swap_after"]=subprocess.check_output(["sysctl","-n","vm.swapusage"],text=True).strip()
        receipt["thermal_after"]=subprocess.check_output(["pmset","-g","therm"],text=True).strip()
        atomic()
    return 0


if __name__=="__main__": raise SystemExit(main())
