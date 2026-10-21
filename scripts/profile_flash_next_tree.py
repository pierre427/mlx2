"""Bounded synchronized attribution, never a serving/performance verdict.

Imported only by the owned artifact runner after model construction. Hooks
are temporary and removed in finally; no serving policy or numerical law is
changed. Fences intentionally inhibit fusion, so category times describe this
diagnostic, not an additive decomposition of uninstrumented serving latency.
"""
from collections import defaultdict
from contextlib import contextmanager
import time


class StageTrace:
    def __init__(self, mx, settle, *, clock=time.perf_counter):
        self.mx, self.settle, self.clock = mx, settle, clock
        self.rows = defaultdict(lambda: {"calls": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0})
        self.stack = []
        self.input_wait = 0.0

    def wrapped(self, function, category):
        def execute(*args, **kwargs):
            # Never flatten Module/dict arguments: that would materialize all
            # weights repeatedly. Inputs already have their committed state;
            # only direct activation operands need this dependency fence.
            inputs = [value for value in args if isinstance(value, self.mx.array)]
            start = self.clock()
            self.settle(inputs)
            if not self.stack:
                self.input_wait += self.clock() - start
            frame = [category, 0.0]
            self.stack.append(frame)
            start = self.clock()
            try:
                result = function(*args, **kwargs)
                self.settle(result)
                return result
            finally:
                elapsed = self.clock() - start
                self.stack.pop()
                row = self.rows[category]
                row["calls"] += 1
                row["inclusive_seconds"] += elapsed
                row["exclusive_seconds"] += max(0.0, elapsed - frame[1])
                if self.stack:
                    self.stack[-1][1] += elapsed
        return execute

    @contextmanager
    def hooks(self, entries):
        previous = []
        try:
            for owner, name, category in entries:
                own = name in vars(owner)
                function = getattr(owner, name)
                previous.append((owner, name, own, function))
                setattr(owner, name, self.wrapped(function, category))
            yield
        finally:
            for owner, name, own, function in reversed(previous):
                if own:
                    setattr(owner, name, function)
                else:
                    delattr(owner, name)

    def receipt(self, elapsed):
        total = sum(row["exclusive_seconds"] for row in self.rows.values())
        return {"wall_seconds": elapsed, "top_level_dependency_wait_seconds": self.input_wait,
                "unattributed_seconds": elapsed - total - self.input_wait,
                "stages": dict(self.rows)}


def trace_target(adapter, pattern, settle, equal, safety):
    import mlx.core as mx
    from mlx2.adapters import flash_next_tree as T
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
    from mlx2.runtime.models import qwen4_exp as Q, qwen4_tree_gdn as G
    from mlx2.runtime.models import qwen4_fused_gdn as F, qwen4_hc_decode as HC

    model = adapter.model
    prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content":
        "Explain how a database index speeds up queries. Use one short paragraph."}], "enable_thinking": False}))
    layers = (len(model.language_model.model.layers) - 1,)
    base = model.make_cache()
    features = model.prefill_body(mx.array([prompt]), base, layers)
    logits = model(mx.array([prompt]), cache=model.make_cache())
    settle(logits, features)
    anchor = int(mx.argmax(logits[0, -1]).item())
    draft = NativeMTPDraftModel(model, num_draft=max(pattern), revision=adapter.identity["fingerprint"])
    slate = draft.draft_packed_paths([anchor], features, draft.make_cache(), pattern,
                                   context_tokens=[prompt[1:] + [anchor]])[0]
    tokens, parents = [anchor], [-1]
    for path in slate:
        parent = 0
        for token in path.tokens:
            parents.append(parent)
            parent = len(tokens)
            tokens.append(token)
    if len(tokens) != 16:
        raise RuntimeError("attribution requires the full16-row tile")
    provider = T.FlashNextTargetTreeExecutor(model)

    def materialize(result):
        logits, features, transaction = result
        settle(logits, features)
        # Include every possible private successor, not just target logits.
        for path in transaction.paths:
            cache = transaction.prepared_cache(path)
            settle([(item.state, item.meta_state) for item in cache])
        return result

    # One untimed warmup; no profile wrappers or request-time canaries.
    warm = materialize(provider.forward(model, tokens, parents, base, layers))
    warm[2].abort()
    ordinary_cache = [T._clone(item) for item in base]
    ordinary_reference = model.forward_with_taps(mx.array([[anchor]]), ordinary_cache, layers)
    settle(ordinary_reference)
    safety()
    start = time.perf_counter()
    untraced = materialize(provider.forward(model, tokens, parents, base, layers))
    baseline_seconds = time.perf_counter() - start
    entries = [(T, "_project", "projections"),
               (T.TreeProjectionLinear, "__call__", "projections"),
               (T, "_ple", "ple"), (Q.PLELayer, "__call__", "ple"),
               (T, "_moe", "routing_and_experts"),
               (type(model.language_model.model.layers[0].mlp), "__call__", "routing_and_experts"),
               (HC, "hc_decode_launch", "hc"),
               (G, "tree", "gdn_core"),
               (F, "qwen4_fused_gdn_decode", "gdn_core"),
               (T.TreeAttention, "__call__", "qsa"),
               (Q, "_apply_inject", "residual_inject")]
    if hasattr(Q, "qwen4_fused_gdn_decode"):
        entries.append((Q, "qwen4_fused_gdn_decode", "gdn_core"))
    output = {"scope": "Synchronized diagnostic only; fences inhibit fusion and alter scheduling. Not serving timing or causal percentages of the earlier HTTP result.",
              "pattern": list(pattern), "rows": len(tokens), "metrics": untraced[2].metrics,
              "untraced_target_all_successors_seconds": baseline_seconds}
    try:
        for arm in ("ordinary_one_row", "packed_sixteen_rows"):
            safety()
            tracer = StageTrace(mx, settle)
            with tracer.hooks(entries):
                start = time.perf_counter()
                if arm == "ordinary_one_row":
                    cache = [T._clone(item) for item in base]
                    result = model.forward_with_taps(mx.array([[anchor]]), cache, layers)
                    settle(result, [(item.state, item.meta_state) for item in cache])
                else:
                    result = materialize(provider.forward(model, tokens, parents, base, layers))
                elapsed = time.perf_counter() - start
            output[arm] = tracer.receipt(elapsed)
            reference = ordinary_reference if arm == "ordinary_one_row" else untraced
            if not equal(result[0], reference[0]) or not equal(result[1], reference[1]):
                raise RuntimeError("stage instrumentation changed output arithmetic")
            output[arm]["outputs_exact_without_instrumentation"] = True
            if arm != "ordinary_one_row":
                result[2].abort()
        safety()
        return output
    finally:
        if not untraced[2].closed:
            untraced[2].abort()
