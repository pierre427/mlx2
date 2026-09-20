"""rm15 step 2b: inside layer 0's GatedDeltaNet, WHICH tensor first differs?

Records the recurrence's inputs (q,k,v,a,b at the last prefill position) and
its output state for each prefill chunk size, and diffs them against the
unchunked arm. If the INPUTS already differ the cause is upstream of the
recurrence (projection / conv). If only the OUTPUT differs, the recurrence
itself is chunk-sensitive.
"""
import argparse, json
import mlx.core as mx

mx.set_default_device(mx.cpu)

REC = {}


def build(dtype="float32", seed=0):
    mx.random.seed(seed)
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    text = dict(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        mtp_num_hidden_layers=0, num_experts=0, rms_norm_eps=1e-6,
        max_position_embeddings=65536,
        rope_parameters={"type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25},
    )
    m = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    if dtype != "float32":
        m.set_dtype(getattr(mx, dtype))
    mx.eval(m.parameters())
    return m


def patch():
    from mlx2.runtime.models import qwen3_5 as Q
    orig_gdu = Q.gated_delta_update
    orig_norm = Q.normalize_gdn_qk

    def gdu(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, **kw):
        out, st = orig_gdu(q, k, v, a, b, A_log, dt_bias, state, mask, **kw)
        r = REC.setdefault("gdn", {})
        r["in_q_last"] = q[:, -1]
        r["in_k_last"] = k[:, -1]
        r["in_v_last"] = v[:, -1]
        r["in_a_last"] = a[:, -1]
        r["in_b_last"] = b[:, -1]
        r["in_state"] = state if state is not None else mx.zeros((1,))
        r["out_state"] = st
        r["out_y_last"] = out[:, -1]
        r["mask_is_none"] = mx.array([1.0 if mask is None else 0.0])
        return out, st

    def nrm(q, k):
        qq, kk = orig_norm(q, k)
        r = REC.setdefault("gdn", {})
        r["postnorm_q_last"] = qq[:, -1]
        r["postnorm_k_last"] = kk[:, -1]
        return qq, kk

    Q.gated_delta_update = gdu
    Q.normalize_gdn_qk = nrm


def prefill(model, stream, ctx, step, layer0_only=True):
    from mlx2.runtime.models.cache import make_prompt_cache
    cache = list(make_prompt_cache(model))
    snap = None
    for start in range(0, ctx, step):
        REC.clear()
        mx.eval(model(stream[:, start:min(start + step, ctx)], cache=cache))
        snap = {k: v for k, v in REC.get("gdn", {}).items()}
    # snap holds the LAST gdn layer of the LAST chunk; capture layer-0 instead
    return cache, snap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--steps", default="512,256,128,64,32,16,8")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()

    model = build()
    # keep ONLY the requested gdn layer recording: record the first gdn call
    from mlx2.runtime.models import qwen3_5 as Q
    patch()
    orig_gdu = Q.gated_delta_update

    calls = {"n": 0}

    def first_only(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return orig_gdu(*args, **kw)
        # bypass recording for later layers
        REC_backup = dict(REC.get("gdn", {}))
        out = orig_gdu(*args, **kw)
        REC["gdn"] = REC_backup
        return out

    Q.gated_delta_update = first_only

    mx.random.seed(1234)
    ids = mx.random.randint(0, 128, (1, a.context + 8)).astype(mx.uint32)
    steps = [int(x) for x in a.steps.split(",")]

    snaps = {}
    for s in steps:
        calls["n"] = 0

        def prefill_arm():
            from mlx2.runtime.models.cache import make_prompt_cache
            cache = list(make_prompt_cache(model))
            snap = None
            for start in range(0, a.context, s):
                REC.clear()
                calls["n"] = 0
                mx.eval(model(ids[:, start:min(start + s, a.context)], cache=cache))
                snap = {k: v for k, v in REC.get("gdn", {}).items()}
            return snap

        snaps[s] = prefill_arm()

    ref = snaps[a.context] if a.context in snaps else snaps[steps[0]]
    rep = {"context": a.context, "layer": "first gdn layer", "device": "cpu",
           "reference": "unchunked", "arms": {}}
    for s in steps:
        if snaps[s] is ref:
            continue
        rows = {}
        for k in ref:
            x, y = snaps[s].get(k), ref[k]
            if x is None or x.shape != y.shape:
                rows[k] = "missing/shape"
                continue
            rows[k] = float(mx.max(mx.abs(x.astype(mx.float32) -
                                          y.astype(mx.float32))).item())
        rep["arms"][str(s)] = rows
    txt = json.dumps(rep, indent=2)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
