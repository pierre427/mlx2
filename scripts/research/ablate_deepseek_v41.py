"""Owned one-factor V4.1 diagnostics; uncontrolled timing, exact numerical gates."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import gc
import statistics
import time
from contextlib import contextmanager

import numpy as np

from mlx2.desktop_campaign_worker import atomic
from mlx2.runtime.models.deepseek_v41_engram import EngramRows
from mlx2.runtime.expert_residency import AdaptiveExpertResidency, AdaptiveResidencySettings
from mlx2.runtime.expert_stream_atlas import ExpertStreamAtlas
from analyze_deepseek_v41_10token import REQUEST
from profile_deepseek_v41 import arrays_equal, arrays_numpy


class DiagnosticPLE:
    """Bounded comparison readers; registered runtime owners never change.

    These direct, quiescent diagnostics do not publish cache state or invoke
    whole-model parking while an auxiliary map is selected. The reserve is
    separate from the runtime provider's immutable accounting.
    """

    reserve_bytes = 32 << 20
    max_rows = 32 * 24  # The widest diagnostic arm is 32 tokens, 24 hashes each.

    def __init__(self, adapter, mx):
        self.model, self.stream, self.mx = adapter.model, adapter.weights.expert_stream, mx
        self.original = self.model.engram_tables
        bound = self.stream.ple_tables
        if (adapter.engram_tables is not self.original or bound.keys() != self.original.keys()
                or any(bound[key] is not table for key, table in self.original.items())):
            raise ValueError("diagnostics require the original registered PLE providers")
        # Covers simultaneous pending reads, decode temporaries and returned
        # arrays, plus Python/LUT metadata. Both format caches share the rest.
        self.transient_bound_bytes = len(self.original) * self.max_rows * (16 << 10) + (1 << 20)
        available = self.reserve_bytes - self.transient_bound_bytes
        if available < 0:
            raise ValueError("diagnostic PLE geometry exceeds the 32 MiB reserve")
        self.cache_per_table = available // max(1, 2 * len(self.original)) // 1024 * 1024
        self.auxiliary = {}

    def select(self, cache_format, *, reset=False):
        if cache_format not in ("fp32", "bf16_lut"):
            raise ValueError("unknown diagnostic PLE cache format")
        if cache_format not in self.auxiliary:
            tables = self.auxiliary[cache_format] = {}
            for layer, table in self.original.items():
                tables[layer] = EngramRows(table.path, offset=table.offset, rows=table.rows,
                                          expected_stat=table.expected_stat, cache_format=cache_format,
                                          cache_bytes=self.cache_per_table, max_rows=self.max_rows,
                                          workers=table.workers, timeout=table.timeout)
        self.model.engram_tables = self.auxiliary[cache_format]
        if reset:
            for table in self.model.engram_tables.values():
                table.park()
                table.activate()

    def restore(self):
        self.model.engram_tables = self.original

    def receipt(self):
        tables = [table for group in self.auxiliary.values() for table in group.values()]
        return dict(reserve_bytes=self.reserve_bytes, transient_bound_bytes=self.transient_bound_bytes,
                    cache_budget_bytes=sum(table.resume_reservation_bytes() for table in tables),
                    cached_bytes=sum(table.diagnostics()["cached_bytes"] for table in tables),
                    max_lookup_rows=self.max_rows, provider_count=len(tables),
                    runtime_binding_unchanged=all(self.stream.ple_tables[k] is v for k,v in self.original.items()))

    def close(self):
        self.restore()
        failure = None
        try:
            self.mx.synchronize()
        except BaseException as error:
            failure = error
        # These are independent host readers, not dense GPU page leases.
        # EngramRows.close retains in-flight I/O ownership in its ReadPool.
        for tables in self.auxiliary.values():
            for table in tables.values():
                try:
                    table.close()
                except BaseException as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure


@contextmanager
def diagnostic_reclaim(stream, tables, mx):
    """Borrow only unclaimed room and restore the original coordinator."""
    if (tables.keys() != stream.ple_tables.keys()
            or any(stream.ple_tables[key] is not table for key, table in tables.items())):
        raise ValueError("diagnostic reclaim requires the original registered PLE providers")
    original, synchronize = stream.elastic, stream._execution_synchronize
    initial_budgets = None
    try:
        if original is None:
            stream.enable_elastic(tables)
            initial_budgets = {key:r.budget_bytes for key,r in stream.elastic.resources.items()}
            stream.bind_execution_synchronizer(mx.synchronize)
        coordinator = stream.elastic
        floor = sum(r.minimum_bytes for r in coordinator.resources.values())
        room = coordinator.total_bytes - floor - sum(coordinator.claims.values())
        with coordinator.borrow("ablation_workspace", room):
            yield coordinator.receipt()
    finally:
        if original is None and stream.elastic is not None:
            if stream.elastic.error is not None:
                raise RuntimeError("diagnostic reclamation failed; keep the poisoned coordinator attached")
            # A temporary coordinator may have started below its maxima.
            # Restore the actual original budgets before detaching it.
            mx.synchronize()
            for key, budget in (initial_budgets or {}).items():
                resource = stream.elastic.resources[key]
                if resource.budget_bytes != budget:
                    resource.resize(budget)
            stream.elastic = None
            stream._execution_synchronize = synchronize


def compare_final(adapter, mx, out, phase):
    """Alternate the pre-campaign optimized path and final explicit candidate.

    Both use the same loaded artifact and hot expert working set. Fresh state
    on every invocation prevents prefix reuse from masquerading as compute.
    """
    from analyze_deepseek_v41_10token import CPUWallProfiler, instrument

    model, w = adapter.model, adapter.weights
    stream = w.expert_stream
    baseline = dict(expert_small_policy="reference", expert_reuse_policy="one", expert_iq1_policy="scalar",
                    expert_down_policy="serial", expert_binding_policy="twelve", expert_io_policy="blocking",
                    dense_epilogue_policy="reference", dense_pair_policy="reference", expert_groups_policy="six",
                    expert_simd_policy="one", shared_policy="reference")
    candidate = {name: getattr(w, name) for name in baseline}
    model_names = ("pointwise_policy", "attention_epilogue_policy", "hc_epilogue_policy", "verify_rows_policy")
    model_candidate = {name: getattr(model, name) for name in model_names}
    atlas, adaptive, coalesced = stream.atlas, stream.adaptive, stream.coalesced
    ple = DiagnosticPLE(adapter, mx)
    whole = mx.array([list(adapter.prompt_tokens(REQUEST))], mx.uint32)
    result = {"passed": False, "scope": "six alternating hot full10/first-decode pairs; fresh KV each arm; same loaded artifact; thermally uncontrolled",
              "baseline_weights": baseline, "candidate_weights": candidate, "pairs": [], "profiles": {}}
    oracle = None

    def configure(label):
        for future in stream._hints.values():
            try: future.result(timeout=60)
            except Exception: pass
        stream._discard_hints()
        for name, value in (candidate if label == "candidate" else baseline).items():
            setattr(w, name, value)
        for name in model_names:
            setattr(model, name, model_candidate[name] if label == "candidate" else ("serial" if name == "verify_rows_policy" else "reference"))
        stream.atlas = atlas if label == "candidate" else None
        stream.adaptive = adaptive if label == "candidate" else {}
        stream.coalesced = coalesced if label == "candidate" else False
        if label == "candidate": ple.restore()
        else: ple.select("fp32")

    def forward():
        state = model.make_cache()
        before = stream.counters()
        start = time.perf_counter(); value = model.prefill_forward(whole, state); mx.eval(value, [s.state for s in state])
        elapsed = time.perf_counter()-start
        arrays = (np.array(value), [arrays_numpy(s.state) for s in state])
        next_token = int(mx.argmax(value[0,-1]).item())
        start = time.perf_counter(); value = model(mx.array([[next_token]], mx.uint32), state); mx.eval(value, [s.state for s in state])
        decode = time.perf_counter()-start
        arrays = (arrays, np.array(value), [arrays_numpy(s.state) for s in state])
        return dict(prefill_seconds=elapsed, decode_seconds=decode,
                    counters={k:v-before[k] for k,v in stream.counters().items()}), arrays

    try:
        for _ in range(2):
            for label in ("baseline", "candidate"):
                configure(label); _, arrays = forward()
                if oracle is None: oracle = arrays
                elif not arrays_equal(oracle, arrays): raise RuntimeError("final warm-up arithmetic/state mismatch")
        for repeat in range(6):
            row = {"repeat": repeat}
            for label in (("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")):
                configure(label); measurements, arrays = forward()
                measurements["strict_bytes_equal"] = arrays_equal(oracle, arrays)
                if not measurements["strict_bytes_equal"]: raise RuntimeError("final paired arithmetic/state mismatch")
                row[label] = measurements
            result["pairs"].append(row); atomic(out/"combined-comparison.json", result)
            phase("final_pair", repeat=repeat, baseline=row["baseline"]["prefill_seconds"], candidate=row["candidate"]["prefill_seconds"])
        for label in ("baseline", "candidate"):
            configure(label); p=CPUWallProfiler(mx); state=model.make_cache()
            with instrument(model,p) as reads:
                value=p.call("forward_other",label+"_prefill10",model.prefill_forward,(whole,state),{})
            exact=arrays_equal(oracle[0],(np.array(value),[arrays_numpy(s.state) for s in state]))
            if not exact: raise RuntimeError("final instrumented arithmetic/state mismatch")
            result["profiles"][label]={"strict_bytes_equal":exact,"selected_stack_copy_bytes":p.copy_bytes,"read_tasks":reads,**p.summary()}
        result["median_seconds"]={label:{key:statistics.median(row[label][key] for row in result["pairs"])
                                         for key in ("prefill_seconds","decode_seconds")} for label in ("baseline","candidate")}
        result["passed"]=True
        result["diagnostic_ple"] = ple.receipt()
        atomic(out/"combined-comparison.json",result)
        return result
    finally:
        try: configure("candidate")
        finally: ple.close()


def run(adapter, mx, out, phase):
    model, w = adapter.model, adapter.weights
    stream = w.expert_stream
    original = {name:getattr(w,name) for name in ("expert_small_policy","expert_reuse_policy","expert_iq1_policy",
                "expert_down_policy","expert_binding_policy","expert_io_policy","dense_epilogue_policy","dense_pair_policy",
                "expert_groups_policy","expert_simd_policy","shared_policy")}
    saved = dict(atlas=stream.atlas,adaptive=stream.adaptive,coalesced=stream.coalesced,capacity=stream.capacity,
                 pointwise=model.pointwise_policy,verify=model.verify_rows_policy,
                 attention_epilogue=model.attention_epilogue_policy,hc_epilogue=model.hc_epilogue_policy)
    ple = DiagnosticPLE(adapter, mx)
    base = dict(expert_small_policy="reference",expert_reuse_policy="one",expert_iq1_policy="scalar",
                expert_down_policy="serial",expert_binding_policy="twelve",expert_io_policy="blocking",
                dense_epilogue_policy="reference",dense_pair_policy="reference",expert_groups_policy="six",expert_simd_policy="one",shared_policy="reference")
    ids=list(adapter.prompt_tokens(REQUEST));whole=mx.array([ids],mx.uint32)
    result={"prompt_ids":ids,"arms":[],"passed":False,"scope":"one factor per arm; three warm repetitions; thermally uncontrolled; no controlled performance claim"}
    baseline=baseline_decode=None

    def drain_hints():
        for future in stream._hints.values():
            try:future.result(timeout=60)
            except Exception:pass
        stream._discard_hints()

    def configure(changes):
        drain_hints()
        for name,value in {**base,**{k:v for k,v in changes.items() if k in base}}.items():setattr(w,name,value)
        stream.coalesced=changes.get("coalesced",False)
        stream.adaptive=({p:AdaptiveExpertResidency(stream.plan.experts_per_layer,AdaptiveResidencySettings(enabled=True)) for p in stream.tables}
                         if changes.get("adaptive",False) else {})
        stream.atlas=(ExpertStreamAtlas(saved["atlas"].binding,layers=40,experts=384,top_k=6)
                      if changes.get("atlas",False) else None)
        stream.capacity=changes.get("capacity",saved["capacity"])
        model.pointwise_policy=changes.get("pointwise","reference")
        model.attention_epilogue_policy=changes.get("attention_epilogue","reference")
        model.hc_epilogue_policy=changes.get("hc_epilogue","reference")
        model.verify_rows_policy="serial"
        ple.select(changes.get("engram","fp32"), reset=True)

    def forward(tokens, *, cold=False):
        if cold:
            drain_hints()
            for cache in stream.caches.values():cache.clear()
            gc.collect();mx.clear_cache()
        cache=model.make_cache();before=stream.counters()
        start=time.perf_counter();value=model.prefill_forward(tokens,cache);mx.eval(value,[s.state for s in cache]);elapsed=time.perf_counter()-start
        return elapsed,value,cache,{k:v-before[k] for k,v in stream.counters().items()}

    arms=[("baseline",{}),
          ("direct_short_rows",dict(expert_small_policy="row_exact")),
          ("reuse_two",dict(expert_small_policy="row_exact",expert_reuse_policy="two")),
          ("reuse_four",dict(expert_small_policy="row_exact",expert_reuse_policy="four")),
          ("groups_ten",dict(expert_small_policy="row_exact",expert_reuse_policy="four",expert_groups_policy="ten")),
          ("simds_four",dict(expert_small_policy="row_exact",expert_reuse_policy="four",expert_simd_policy="four")),
          ("groups_ten_simds_four",dict(expert_small_policy="row_exact",expert_reuse_policy="four",expert_groups_policy="ten",expert_simd_policy="four")),
          ("iq1_broadcast",dict(expert_small_policy="row_exact",expert_reuse_policy="two",expert_iq1_policy="broadcast")),
          ("parallel_down",dict(expert_down_policy="parallel")),
          ("read_overlap",dict(expert_io_policy="overlap")),
          ("coalesced_reads",dict(coalesced=True)),
          ("compact_engram",dict(engram="bf16_lut")),
          ("dense_epilogue",dict(dense_epilogue_policy="fused")),
          ("paired_projections",dict(dense_pair_policy="fused")),
          ("pointwise",dict(pointwise="fused")),
          ("shared_fusion",dict(shared_policy="fused")),
          ("attention_epilogue",dict(attention_epilogue="fused")),
          ("hc_epilogue",dict(hc_epilogue="fused")),
          ("lru_pressure",dict(capacity=24)),
          ("adaptive_pressure",dict(capacity=24,adaptive=True)),
          ("atlas",dict(atlas=True))]
    try:
        for label,changes in arms:
            configure(changes)
            cold,value,state,counters=forward(whole,cold=True)
            actual=(np.array(value),[arrays_numpy(s.state) for s in state])
            if baseline is None:baseline=actual
            parity=arrays_equal(baseline,actual)
            timings=[];warm_counts=[]
            for _ in range(3):
                elapsed,value,state,counts=forward(whole)
                timings.append(elapsed);warm_counts.append(counts)
                parity &= arrays_equal(baseline,(np.array(value),[arrays_numpy(s.state) for s in state]))
            snapshots=[s.snapshot() for s in state];mx.eval([s.arrays for s in snapshots])
            next_token=int(mx.argmax(value[0,-1]).item())
            decode=[];decode_ref=None
            for _ in range(3):
                for s,snapshot in zip(state,snapshots):s.restore(snapshot)
                start=time.perf_counter();v=model(mx.array([[next_token]],mx.uint32),state);mx.eval(v,[s.state for s in state]);decode.append(time.perf_counter()-start)
                current=(np.array(v),[arrays_numpy(s.state) for s in state])
                if decode_ref is None:decode_ref=current
                else:parity &= arrays_equal(decode_ref,current)
            if baseline_decode is None:baseline_decode=decode_ref
            parity &= arrays_equal(baseline_decode,decode_ref)
            arm=dict(name=label,changes=changes,cold_seconds=cold,cold_counters=counters,
                     warm_seconds=timings,warm_median_seconds=statistics.median(timings),warm_counters=warm_counts,
                     decode_seconds=decode,decode_median_seconds=statistics.median(decode),strict_bytes_equal=parity,
                     expert_routes=dict(w.scheduled_expert_calls),engram={str(k):v.diagnostics() for k,v in model.engram_tables.items()})
            if label=="atlas":
                # A replay with private GPU rows emptied establishes prediction
                # use separately from an ordinary resident-cache warm hit.
                elapsed,v,s,counts=forward(whole,cold=True)
                arm["trained_replay"]={"seconds":elapsed,"counters":counts,"strict_bytes_equal":arrays_equal(baseline,(np.array(v),[arrays_numpy(c.state) for c in s]))}
                held=list(adapter.prompt_tokens({"messages":[{"role":"user","content":"Count from ten to twenty."}],"enable_thinking":False}))
                elapsed,v,s,counts=forward(mx.array([held],mx.uint32),cold=True)
                expected=(np.array(v),[arrays_numpy(c.state) for c in s]);stream.atlas=None
                control,v,s,control_counts=forward(mx.array([held],mx.uint32),cold=True)
                arm["held_out"]={"tokens":held,"hint_seconds":elapsed,"demand_seconds":control,"counters":counts,"demand_counters":control_counts,
                                  "strict_bytes_equal":arrays_equal(expected,(np.array(v),[arrays_numpy(c.state) for c in s]))}
                parity &= arm["trained_replay"]["strict_bytes_equal"] and arm["held_out"]["strict_bytes_equal"]
            result["arms"].append(arm);atomic(out/"ablations.json",result)
            phase("ablation",name_=label,cold_seconds=cold,warm_seconds=arm["warm_median_seconds"],decode_seconds=arm["decode_median_seconds"],strict_bytes_equal=parity)
            if not parity:raise RuntimeError("ablation changed ordinary arithmetic/state bytes: "+label)
            if label=="baseline":
                from analyze_deepseek_v41_10token import CPUWallProfiler,instrument
                p=CPUWallProfiler(mx);fresh=model.make_cache()
                with instrument(model,p) as reads:
                    value=p.call("forward_other","baseline_prefill10",model.prefill_forward,(whole,fresh),{})
                result["instrumented_baseline"]={"selected_stack_copy_bytes":p.copy_bytes,"read_tasks":reads,**p.summary()}
                atomic(out/"baseline-instrumented.json",result["instrumented_baseline"])
        # Compare the same 32-row NAX law with six versus twelve bindings.
        configure({})
        wide=mx.array([(ids*4)[:32]],mx.uint32);reference=None
        for group in ("twelve","six"):
            w.expert_binding_policy=group
            forward(wide)
            elapsed,value,state,counts=forward(wide)
            arrays=(np.array(value),[arrays_numpy(s.state) for s in state])
            if reference is None:reference=arrays
            exact=arrays_equal(reference,arrays)
            result["arms"].append(dict(name="binding_"+group,seconds=elapsed,strict_bytes_equal=exact,counters=counts))
            if not exact:raise RuntimeError("NAX binding group changed bytes")
        # Actual six-row verification with all seven accepted-prefix frontiers.
        configure(dict(expert_small_policy="row_exact",expert_reuse_policy="two"))
        from mlx2.adapters.deepseek_v41_external import DeepSeekV41VerifyRows
        _,_,state,_=forward(whole);snapshots=[s.snapshot() for s in state]
        proposal=mx.array([ids[:6]],mx.uint32)
        verification=[]
        for accepted in range(7):
            expected=None
            for mode in ("serial","layerwise"):
                for s,snapshot in zip(state,snapshots):s.restore(snapshot)
                model.verify_rows_policy=mode
                transaction=DeepSeekV41VerifyRows(model,[state],()).begin([6])
                start=time.perf_counter();value,_=transaction.forward(proposal,taps=False);mx.eval(value,[s.state for s in state]);elapsed=time.perf_counter()-start
                transaction.commit([accepted]);mx.eval([s.state for s in state])
                arrays=(np.array(value),[arrays_numpy(s.state) for s in state])
                if expected is None:expected=arrays
                exact=arrays_equal(expected,arrays)
                verification.append(dict(mode=mode,accepted=accepted,seconds=elapsed,strict_bytes_equal=exact))
                if not exact:raise RuntimeError(f"layerwise verify differs at accepted={accepted}")
        result["verification"]=verification
        # Exercise physical reclamation under an explicit workspace claim.
        # Retained state survives independently; release is measured, not
        # inferred from a smaller logical cache budget.
        configure(dict(expert_small_policy="row_exact",expert_reuse_policy="four",engram="bf16_lut"))
        ple.restore()
        _,value,state,_=forward(whole)
        reference=(np.array(value),[arrays_numpy(s.state) for s in state])
        before=dict(active=mx.get_active_memory(),live=stream.live_bytes(),reserved=stream.reserved_bytes(),
                    diagnostic_ple=ple.receipt())
        start=time.perf_counter()
        with diagnostic_reclaim(stream, ple.original, mx) as receipt:
            elapsed=time.perf_counter()-start
            after=dict(active=mx.get_active_memory(),live=stream.live_bytes(),reserved=stream.reserved_bytes(),
                       diagnostic_ple=ple.receipt())
            _,value,state,counts=forward(whole)
            exact=arrays_equal(reference,(np.array(value),[arrays_numpy(s.state) for s in state]))
            result["residency"]={"before":before,"after":after,"claim":receipt,"seconds":elapsed,"strict_bytes_equal":exact,"counters":counts}
        if not exact:raise RuntimeError("residency reclaim changed logits/state")
        result["passed"]=True
    finally:
        drain_hints()
        for name,value in original.items():setattr(w,name,value)
        stream.atlas,stream.adaptive,stream.coalesced,stream.capacity=saved["atlas"],saved["adaptive"],saved["coalesced"],saved["capacity"]
        model.pointwise_policy,model.verify_rows_policy=saved["pointwise"],saved["verify"]
        model.attention_epilogue_policy,model.hc_epilogue_policy=saved["attention_epilogue"],saved["hc_epilogue"]
        result["diagnostic_ple"] = ple.receipt()
        ple.close()
        atomic(out/"ablations.json",result)
    return result
