"""Owned actual-artifact dense paging and exact park/activate probes."""

import gc
import os
import statistics
import time
import urllib.error

import numpy as np

from mlx2.desktop_campaign_worker import atomic, check_ownership
from analyze_deepseek_v41_10token import REQUEST
from profile_deepseek_v41 import arrays_equal, arrays_numpy


def state_values(cache):
    # Placement policy intentionally namespaces APCv2. Compare every other
    # frontier/topology field in addition to raw array dtype/shape/bytes.
    return [(arrays_numpy(s.state), s.meta_state[:1]+s.meta_state[2:]) for s in cache]


def compare_dense_reads(spec, adapter, out, phase):
    """Paired actual-page host I/O, outside the model's throughput timings."""
    from mlx2.runtime.immutable_weight_pages import read_page
    pages = adapter.weights.dense_pages
    if pages is None or not hasattr(os, "preadv"):
        raise RuntimeError("paired dense read probe requires positional destination reads")
    # np.array_equal may allocate a page-sized boolean comparison in addition
    # to the two input pages. The model admission already reserves5GiB workspace.
    comparison_bound=3*max(p.source_bytes for p in pages.pages.values())+pages.chunk_bytes
    if comparison_bound > (5 << 30):
        raise MemoryError("dense read oracle exceeds its admitted comparison workspace")

    def copied(fd, offset, size, chunk_bytes):
        result=np.empty(size,np.uint8);cursor=0
        while cursor<size:
            raw=os.pread(fd,min(chunk_bytes,size-cursor),offset+cursor)
            if not raw:raise OSError("short reference page read")
            result[cursor:cursor+len(raw)]=np.frombuffer(raw,np.uint8)
            cursor+=len(raw)
        return result

    def timed(reader,*args):
        cpu=time.thread_time();value=reader(*args)
        return value,time.thread_time()-cpu

    report=dict(passed=False,checkpoint=spec["checkpoint"],source_revision=spec["source_revision"],
                source_snapshot_sha256=spec["source_snapshot_sha256"],pairs=[],
                pages=len(pages.pages),bytes_per_arm=sum(p.source_bytes for p in pages.pages.values()),
                extra_host_copy_bytes_per_arm=sum(p.source_bytes for p in pages.pages.values()),
                comparison_workspace_bound_bytes=comparison_bound,
                scope="three alternating same-page pairs; actual artifact; warm OS cache not flushed; two input pages plus bounded boolean-comparison scratch; no GPU timings or controlled performance claim")
    for repeat in range(3):
        check_ownership(spec);pages.verify()
        order=("copied","direct") if repeat%2==0 else ("direct","copied")
        wall=dict(copied=0.0,direct=0.0);cpu=dict(copied=0.0,direct=0.0)
        for index,page in enumerate(pages.pages.values()):
            if index%40==0:check_ownership(spec)
            values={}
            for label in order:
                start=time.perf_counter()
                values[label],used=pages.pool.call(timed,copied if label=="copied" else read_page,
                    pages.fd,page.offset,page.source_bytes,pages.chunk_bytes)
                wall[label]+=time.perf_counter()-start;cpu[label]+=used
            if not np.array_equal(values["copied"],values["direct"]):
                raise RuntimeError("direct positional dense page bytes changed")
            values.clear()
        pages.verify()
        row=dict(repeat=repeat,order=order,wall_seconds=wall,worker_cpu_seconds=cpu,all_page_bytes_equal=True)
        report["pairs"].append(row);atomic(out/"dense-read-pairs.json",report)
        phase("dense_read_pair",**row)
    report["median_wall_seconds"]={label:statistics.median(row["wall_seconds"][label] for row in report["pairs"]) for label in ("copied","direct")}
    report["passed"]=True;atomic(out/"dense-read-pairs.json",report)
    return report


def compare_dense_capacities(spec, mx, out, phase, streaming):
    from mlx2.adapters.deepseek_v41 import DeepSeekV41Adapter
    report=dict(passed=False,checkpoint=spec["checkpoint"],source_revision=spec["source_revision"],
                source_snapshot_sha256=spec["source_snapshot_sha256"],arms=[],
                scope="same exact native math; explicit1GiB/9GiB dense capacity; two warm samples; uncontrolled timing")
    oracle=None
    for cache_mib in (1024,9216):
        adapter=model=value=cache=None
        try:
            check_ownership(spec)
            policy={k:v for k,v in spec["policy"].items() if k.startswith("deepseek_")}
            policy["deepseek_dense_cache_mib"]=cache_mib
            adapter=DeepSeekV41Adapter(spec["model"],weight_streaming=streaming,execution_policy=policy)
            model=adapter.model;adapter.model_residency.bind_execution_synchronizer(mx.synchronize)
            pages=adapter.weights.dense_pages
            ids=mx.array([list(adapter.prompt_tokens(REQUEST))],mx.uint32)
            arm=dict(cache_mib=cache_mib,forwards=[])
            for repeat in range(3):
                before=dict(pages.stats);experts=adapter.weight_stream.counters();cache=model.make_cache()
                start=time.perf_counter();value=model.prefill_forward(ids,cache);mx.eval(value,[s.state for s in cache])
                elapsed=time.perf_counter()-start;actual=(np.array(value),state_values(cache))
                if oracle is None:oracle=actual
                if not arrays_equal(oracle,actual):raise RuntimeError("dense capacity changed logits/state bytes")
                counters={k:v-before[k] for k,v in pages.stats.items()}
                if cache_mib==9216 and repeat and (counters["read_bytes"] or counters["misses"]):
                    raise RuntimeError("9GiB dense cache did not retain actual full working set")
                arm["forwards"].append(dict(repeat=repeat,seconds=elapsed,strict_bytes_equal=True,
                    dense_counters=counters,expert_counters={k:v-experts[k] for k,v in adapter.weight_stream.counters().items()},
                    active_bytes=mx.get_active_memory()))
            arm["dense_receipt"]=adapter.weights.dense_receipt();report["arms"].append(arm)
            atomic(out/"dense-capacity-pairs.json",report)
            phase("dense_capacity_passed",cache_mib=cache_mib,warm_seconds=[x["seconds"] for x in arm["forwards"][1:]],
                  warm_read_bytes=[x["dense_counters"]["read_bytes"] for x in arm["forwards"][1:]])
        finally:
            model=value=cache=None
            if adapter is not None:adapter.close()
            adapter=None;gc.collect();mx.clear_cache()
    report["passed"]=True;atomic(out/"dense-capacity-pairs.json",report)
    return report


def direct(spec, mx, out, phase, streaming):
    from mlx2.adapters.deepseek_v41 import DeepSeekV41Adapter
    from mlx2.adapters.deepseek_v41_external import DeepSeekV41VerifyRows

    report = {"passed": False, "scope": "same native math; resident vs tensor-paged exact outputs, every verify commit prefix, bounded owned run", "arms": []}
    reference = None
    base = {k:v for k,v in spec["policy"].items() if k.startswith("deepseek_")}
    for storage in ("resident", "paged"):
        policy = dict(base, deepseek_dense_storage=storage)
        if storage == "resident": policy.pop("deepseek_dense_cache_mib", None)
        adapter = value = cache = snapshots = tx = model = None
        try:
            check_ownership(spec)
            start = time.perf_counter()
            adapter = DeepSeekV41Adapter(spec["model"], weight_streaming=streaming, execution_policy=policy)
            model = adapter.model
            adapter.model_residency.bind_execution_synchronizer(mx.synchronize)
            row = {"storage": storage, "load_seconds": time.perf_counter()-start,
                   "loaded_active_bytes": mx.get_active_memory(), "dense_after_load": adapter.weights.dense_receipt(),
                   "forward": [], "verification": []}
            ids = list(adapter.prompt_tokens(REQUEST)); tokens = mx.array([ids], mx.uint32)
            actual = {"forward": [], "verification": []}
            for arm in ("cold", "warm"):
                cache = model.make_cache(); before = adapter.weights.dense_receipt()
                start=time.perf_counter(); value=model.prefill_forward(tokens, cache); mx.eval(value,[s.state for s in cache])
                elapsed=time.perf_counter()-start
                actual["forward"].append((np.array(value),state_values(cache)))
                row["forward"].append(dict(arm=arm, seconds=elapsed, dense_before=before, dense_after=adapter.weights.dense_receipt(), active_bytes=mx.get_active_memory()))
                phase("paging_forward", storage=storage, arm=arm, seconds=elapsed)
            if storage == "paged":
                coordinator=adapter.weight_stream.elastic
                floor=sum(item.minimum_bytes for item in coordinator.resources.values())
                before=mx.get_active_memory();start=time.perf_counter()
                loan=coordinator.set_claim("paging_probe",coordinator.total_bytes-floor)
                elapsed=time.perf_counter()-start;after=mx.get_active_memory()
                state=model.make_cache();probe=model.prefill_forward(tokens,state);mx.eval(probe,[s.state for s in state])
                loan_exact=arrays_equal(actual["forward"][0],(np.array(probe),state_values(state)))
                if not loan_exact:raise RuntimeError("joint dense/expert/PLE reclaim changed output bytes")
                coordinator.set_claim("paging_probe",0)
                row["joint_reclamation"]=dict(claim=loan,seconds=elapsed,before_active_bytes=before,
                                               after_active_bytes=after,strict_bytes_equal=loan_exact,
                                               restored=coordinator.receipt())
                state=probe=None
            snapshots=[s.snapshot() for s in cache];mx.eval([s.arrays for s in snapshots])
            proposal=mx.array([ids[:6]], mx.uint32)
            for accepted in range(7):
                check_ownership(spec)
                for state,snapshot in zip(cache,snapshots):state.restore(snapshot)
                tx=DeepSeekV41VerifyRows(model,[cache],()).begin([6])
                start=time.perf_counter();value,_=tx.forward(proposal,taps=False);mx.eval(value,[s.state for s in cache])
                tx.commit([accepted]);mx.eval([s.state for s in cache]);elapsed=time.perf_counter()-start
                actual["verification"].append((np.array(value),state_values(cache)))
                row["verification"].append(dict(accepted=accepted,seconds=elapsed))
                tx=None
            if reference is None:
                reference=actual
            exact=all(arrays_equal(reference[key],actual[key]) for key in reference)
            row["strict_bytes_equal"]=exact
            if not exact:raise RuntimeError("dense paging changed native logits/state bytes")
            # Retain exact committed state while parking all model weight payloads.
            for state,snapshot in zip(cache,snapshots):state.restore(snapshot)
            selected=int(mx.argmax(value[0,-1]).item())
            control=model(mx.array([[selected]],mx.uint32),cache);mx.eval(control,[s.state for s in cache])
            control=(np.array(control),state_values(cache))
            for state,snapshot in zip(cache,snapshots):state.restore(snapshot)
            value=None;gc.collect();mx.clear_cache()
            before=mx.get_active_memory();identity=id(model)
            start=time.perf_counter();park=adapter.model_residency.park();elapsed=time.perf_counter()-start
            after=mx.get_active_memory()
            if adapter.model_residency.resident_bytes()!=0:raise RuntimeError("parked model retained payload")
            if before-after < (1 << 30):raise RuntimeError("whole-model park did not release at least 1 GiB of measured active memory")
            try:
                model(tokens,cache)
            except (ValueError,RuntimeError):
                refused=True
            else:
                raise RuntimeError("parked model accepted forward")
            admission=adapter.model_residency.admit()
            start=time.perf_counter();activation=adapter.model_residency.activate();activate_seconds=time.perf_counter()-start
            value=model(mx.array([[selected]],mx.uint32),cache);mx.eval(value,[s.state for s in cache])
            resume_exact=arrays_equal(control,(np.array(value),state_values(cache)))
            if not resume_exact or id(adapter.model)!=identity:raise RuntimeError("model park/activation changed reference continuation")
            row["parking"]=dict(before_active_bytes=before,after_active_bytes=after,seconds=elapsed,
                                activation_seconds=activate_seconds,park=park,admission=admission,activate=activation,
                                parked_forward_refused=refused,strict_bytes_equal=resume_exact,model_object_preserved=True)
            report["arms"].append(row);atomic(out/"paging-direct.json",report)
            phase("paging_arm_passed", storage=storage, released_bytes=before-after, exact=exact)
        finally:
            if tx is not None:
                tx.abort()
            value = cache = snapshots = tx = model = None
            if adapter is not None:adapter.close()
            adapter=None;gc.collect();mx.clear_cache()
    report["passed"]=True;atomic(out/"paging-direct.json",report)
    return report


def http(spec, model_id, out, phase, admin_token, reference):
    from deepseek_v41_context_ladder import http_json, stream

    base="http://127.0.0.1:8285"
    body={**REQUEST,"model":model_id,"max_tokens":10,"temperature":0,"stream":True,"return_token_trace":True}
    report={"passed":False,"cycles":[],"scope":"same process/model/profile; APCv2 disk suspend followed by exact resume; CPG remains owned for this bounded unit"}

    def admin(path, body=None):
        return http_json(base,path,body,timeout=15,admin_token=admin_token)

    def wait_state(target):
        deadline=time.monotonic()+180
        while True:
            check_ownership(spec)
            status=admin("/v1/admin/state")
            if status["state"]==target:return status
            if status["state"] in ("failed","error","park_failed","quiesce_failed") or time.monotonic()>deadline:
                raise RuntimeError("model residency transition failed: "+str(status))
            time.sleep(.25)

    for cycle in range(2):
        check_ownership(spec)
        before=admin("/v1/admin/state")
        started=time.perf_counter();requested=admin("/v1/admin/residency/park",{"drain_timeout_seconds":120})
        parked=wait_state("parked");park_seconds=time.perf_counter()-started
        try:
            http_json(base,"/v1/chat/completions",{**body,"stream":False},timeout=10)
        except urllib.error.HTTPError as error:
            refused=error.code
        else:
            raise RuntimeError("parked HTTP engine accepted generation")
        if refused not in (409,429,503):raise RuntimeError("unexpected parked-generation refusal")
        started=time.perf_counter();activate_requested=admin("/v1/admin/residency/activate",{})
        activated=wait_state("serving");activate_seconds=time.perf_counter()-started
        response=stream(base,body,out/f"paging-resume-{cycle}.sse",timeout=600)
        exact=response["token_ids"]==reference["token_ids"] and response["receipt"]["cached_tokens"]==9
        if not exact:raise RuntimeError("APCv2 model paging changed replay tokens or lost exact prefix")
        row=dict(cycle=cycle,before=before,park_requested=requested,parked=parked,park_seconds=park_seconds,
                 generation_refused_status=refused,activate_requested=activate_requested,activated=activated,
                 activate_seconds=activate_seconds,response=response,exact_tokens_and_prefix=True)
        report["cycles"].append(row);atomic(out/"paging-http.json",report)
        phase("paging_http_cycle_passed",cycle=cycle,park_seconds=park_seconds,activate_seconds=activate_seconds)
    report["passed"]=True;atomic(out/"paging-http.json",report)
    return report
