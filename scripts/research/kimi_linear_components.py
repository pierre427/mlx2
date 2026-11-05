#!/usr/bin/env python3
"""Bounded real-weight KDA/MLA probes; caller owns CPG and paired locks.

This is a functional component probe, with no thermal hold and no throughput
claim. It does not qualify the full checkpoint or its 1M context capacity.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpg-lock",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty component evidence directory")
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    ownership = require_qualification_lease(task_id=owner["cpg_task"],
        generation=owner["cpg_generation"],cpg_owner_lock=args.cpg_lock)

    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.kimi_linear import REPO, REVISION
    from mlx2.runtime.models.cache import ArraysCache, KVCache
    from mlx2.runtime.models.kimi_linear import (
        KimiDeltaAttention,
        ModelArgs,
        MultiLatentAttention,
    )

    mx.set_default_device(mx.gpu)
    config = json.loads((args.model/"config.json").read_text())
    index = json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    a = ModelArgs.from_dict(config)
    mx.random.seed(118)
    receipt = {"checkpoint":REPO+"@"+REVISION,
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_dirty":bool(subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip()),
        "ownership":ownership,"device":mx.device_info(),"kind":"real-weight component functional probe",
        "qualified":False,"full_checkpoint_decoded":False,"context_1m_tested":False,"rows":[]}

    def load_layer(module,layer):
        prefix = f"model.layers.{layer}.self_attn."
        files = {index[k] for k in index if k.startswith(prefix)}
        all_weights = {}
        for name in files:
            all_weights.update({k:v for k,v in mx.load(str(args.model/name)).items() if k.startswith(prefix)})
        weights = {}
        for key,value in all_weights.items():
            key = key[len(prefix):]
            if "conv1d.weight" in key:
                key = key.replace(".weight",".conv.weight")
                value = value.moveaxis(2,1)
            if key == "A_log": value = value.reshape(-1)
            weights[key] = value
        module.load_weights(list(weights.items()),strict=True)
        module.eval()
        mx.eval(module.parameters())

    def compare(name,ref,got,*,limit):
        mx.eval(ref,got)
        r,g = np.array(ref.astype(mx.float32)),np.array(got.astype(mx.float32))
        error = float(np.linalg.norm(g-r)/max(np.linalg.norm(r),1e-20))
        row = {"name":name,"shape":list(r.shape),"max_abs":float(np.max(np.abs(g-r))),
               "relative_l2":error,"limit":limit,"finite":bool(np.isfinite(g).all()),
               "passed":bool(np.isfinite(g).all() and error <= limit),
               "exact":bool(np.array_equal(r,g))}
        receipt["rows"].append(row)

    before = subprocess.check_output(["/usr/sbin/sysctl","-n","vm.swapusage"],text=True).strip()
    receipt["thermal_before"] = subprocess.check_output(["/usr/bin/pmset","-g","therm"],text=True).strip()
    try:
        kda = KimiDeltaAttention(a)
        load_layer(kda,0)
        kda.reference = True
        x = mx.random.normal((1,35,config["hidden_size"])).astype(mx.bfloat16)
        ref_cache = ArraysCache(size=4)
        ref = kda(x,cache=ref_cache)
        mx.eval(ref,ref_cache.state)
        kda.reference = False
        kernel_cache = ArraysCache(size=4)
        compare("kda_vector_output",ref,kda(x,cache=kernel_cache),limit=.003)
        compare("kda_vector_state",ref_cache[3],kernel_cache[3],limit=1e-5)
        # Grouping and norm fusion get separate arms for attribution.
        kda.prepare_projections()
        kda.fused_projections = True
        compare("kda_grouped_and_merged_conv",ref,kda(x,cache=ArraysCache(size=4)),limit=.01)
        kda.fused_projections = False
        kda.fused_norm = True
        compare("kda_fla_fused_norm",ref,kda(x,cache=ArraysCache(size=4)),limit=.01)
        kda.fused_norm = False
        kda.chunk32 = True
        compare("kda_chunk32",ref,kda(x,cache=ArraysCache(size=4)),limit=.01)
        receipt["kda_counters"] = dict(kda.calls)

        mla = MultiLatentAttention(a)
        load_layer(mla,3)
        x = mx.random.normal((1,17,config["hidden_size"])).astype(mx.bfloat16)
        mask = mx.arange(17)[:,None] >= mx.arange(17)[None,:]
        ca,cb = KVCache(),KVCache()
        ref = mla(x,mask=mask,cache=ca)
        mx.eval(ref)
        mla.policy = "latent_tiled"
        mla.key_tile = 7
        compare("mla_latent_tiled_prefill",ref,mla(x,mask=mask,cache=cb),limit=.03)
        mla.policy = "expanded"
        ref = mla(x[:,:1],cache=ca)
        mx.eval(ref)
        mla.policy = "latent_tiled"
        compare("mla_latent_tiled_decode",ref,mla(x[:,:1],cache=cb),limit=.03)
        # The fused split kernel must cover the shared NoPE contribution too.
        mla.policy = "latent_fused"
        cc = KVCache()
        mla(x,mask=mask,cache=cc)
        compare("mla_fused_decode",ref,mla(x[:,:1],cache=cc),limit=.03)
        receipt["mla_counters"] = dict(mla.calls)
        receipt["passed"] = all(r["passed"] for r in receipt["rows"])
    except BaseException as error:
        receipt["passed"] = False
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["swap_before"] = before
        receipt["swap_after"] = subprocess.check_output(["/usr/sbin/sysctl","-n","vm.swapusage"],text=True).strip()
        receipt["thermal_after"] = subprocess.check_output(["/usr/bin/pmset","-g","therm"],text=True).strip()
        path = args.output/"receipt.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt,indent=2,default=str)+"\n")
        os.replace(tmp,path)
    print(json.dumps(receipt["rows"],indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
