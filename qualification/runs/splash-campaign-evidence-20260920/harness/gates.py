#!/usr/bin/env python3
"""Spec-gate evaluator for the splash GPU queue.

Written 2026-09-20 as part of the runner repair.  The previous runner PRINTED
each spec's expected values and returned 0 regardless -- job 11 was marked "ok"
in 15s with five of its ten assertions wrong.  This module is the half that was
missing: it reads the artifacts a job produced and COMPARES them to the gates
its spec states.

Exit codes
  0  every gate this module evaluated passed
  1  at least one gate failed
  3  UNTESTABLE -- the spec does not state a machine-checkable gate, or the
     artifacts needed to evaluate it are absent.  Deliberately NOT 0: a gate
     that could not be evaluated is not a gate that passed.

Rule: never report PASS for a gate that was not actually evaluated.
"""

import json
import os
import re
import sys

FAILED = []
PASSED = []
SKIPPED = []


def ok(label, detail=""):
    PASSED.append(label)
    print(f"  spec PASS  {label}" + (f": {detail}" if detail else ""))


def bad(label, detail=""):
    FAILED.append(label)
    print(f"  spec FAIL  {label}" + (f": {detail}" if detail else ""))


def skip(label, why):
    SKIPPED.append(label)
    print(f"  spec ????  {label}: NOT EVALUATED ({why})")


def eq(label, actual, expected):
    (ok if actual == expected else bad)(label, f"got {actual!r} want {expected!r}")


def load(path, label=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        if label:
            bad(label, f"cannot read {path}: {exc}")
        return None


def text(path):
    try:
        with open(path) as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return ""


def content_of(msg):
    """Assistant text from an OpenAI-shaped chat completion."""
    try:
        return msg["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------- job 11 ----
def _assistant_message_gate(label, body):
    """Spec 11 wants 'a non-empty assistant message'.

    North-Mini-Code is a reasoning model and the spec's own request sets
    max_tokens=16, so the whole budget can be spent inside the thinking block:
    content == "", reasoning_content populated, finish_reason == "length".
    That is a real miss against the gate as written, but the cause is the
    spec's token budget meeting a reasoning model, not the HTTP transport
    policy under test -- so say which of the two it is rather than reporting a
    bare failure.
    """
    if body is None:
        return
    msg = content_of(body)
    if msg:
        ok(f"{label} body has a non-empty assistant message", repr(msg[:60]))
        return
    try:
        choice = body["choices"][0]
        reasoning = choice["message"].get("reasoning_content") or ""
        finish = choice.get("finish_reason")
    except Exception:  # noqa: BLE001
        reasoning, finish = "", None
    if reasoning and finish == "length":
        bad(f"{label} body has a non-empty assistant message",
            f"content is EMPTY but reasoning_content is {len(reasoning)} chars and "
            f"finish_reason=='length': the model spent the spec's whole max_tokens=16 "
            f"budget inside the thinking block. This is a SPEC/model-budget artifact, "
            f"not an HTTP-security failure -- the transport gates (A1..B5) all passed. "
            f"Re-check with a larger max_tokens before reading anything into it.")
    else:
        bad(f"{label} body has a non-empty assistant message", json.dumps(body)[:200])


def gates_11(out):
    """Spec 11 'Accept / reject': the HTTP codes are gated in the runner
    itself (gate_eq per code).  Here we check the body-level gates."""
    _assistant_message_gate("A1", load(os.path.join(out, "a1.body"),
                                      "A1 body is a chat completion"))
    _assistant_message_gate("B2", load(os.path.join(out, "b2.body"),
                                      "B2 body is a chat completion"))
    skip("A1 wall time vs main within 2 ms",
         "spec marks it optional and no main-tree baseline is captured by this queue")


# --------------------------------------------------------------- job 09 ----
def gates_09(out):
    a = load(os.path.join(out, "a.json"), "run A completion readable")
    b = load(os.path.join(out, "b.json"), "run B completion readable")
    if a and b:
        ta, tb = content_of(a), content_of(b)
        if ta is None or tb is None:
            bad("greedy text A == B", "a response was not a chat completion")
        else:
            eq("greedy text A == B", ta == tb, True)
            if ta != tb:
                print(f"       A: {ta[:160]!r}")
                print(f"       B: {tb[:160]!r}")

    sa = load(os.path.join(out, "a.status.json"), "run A /v1/status readable")
    sb = load(os.path.join(out, "b.status.json"), "run B /v1/status readable")
    ma = text(os.path.join(out, "a.metrics.txt"))
    mb = text(os.path.join(out, "b.metrics.txt"))

    if sa is not None:
        eq("A: no host_memory_signals in settings",
           "host_memory_signals" in sa.get("settings", {}), False)
        keys = [k for k in sa if k.startswith("host_memory")]
        eq("A: no host_memory_* keys in /v1/status", keys, [])
    if ma:
        eq("A: no mlx2_host_memory_* in /metrics",
           [l for l in ma.splitlines() if l.startswith("mlx2_host_memory")], [])

    if sb is not None:
        got = sb.get("settings", {}).get("host_memory_signals")
        eq("B: settings echo the policy", got,
           {"enabled": True, "fall_after_seconds": 5.0})
        level = sb.get("host_memory_pressure_level")
        if level in (0, 1, 2):
            ok("B: host_memory_pressure_level in {0,1,2}", str(level))
            kern = text(os.path.join(out, "kern.pressure")).strip()
            mapping = {"1": 0, "2": 1, "4": 2}
            if kern in mapping:
                if mapping[kern] == level:
                    ok("B: level matches mapped sysctl", f"sysctl={kern} -> {level}")
                else:
                    # the spec allows a recent kernel transition
                    bad("B: level matches mapped sysctl",
                        f"sysctl={kern} maps to {mapping[kern]} but status says {level} "
                        "(spec tolerates only a level change within the last 5s)")
            else:
                skip("B: level matches mapped sysctl", f"unmapped sysctl value {kern!r}")
        else:
            bad("B: host_memory_pressure_level in {0,1,2}", repr(level))
        avail = sb.get("host_memory_available_bytes")
        memsize = text(os.path.join(out, "kern.memsize")).strip()
        try:
            memsize = int(memsize)
        except ValueError:
            memsize = None
        if isinstance(avail, int) and avail > 0 and (memsize is None or avail <= memsize):
            ok("B: 0 < available_bytes <= hw.memsize", f"{avail} <= {memsize}")
        else:
            bad("B: 0 < available_bytes <= hw.memsize", f"avail={avail} memsize={memsize}")
        headroom_b = sb.get("headroom_bytes", 0)
        eq("B: headroom_bytes non-zero (spec: reject only if B headroom is 0)",
           bool(headroom_b), True)
    if mb:
        for metric in ("mlx2_host_memory_pressure_level",
                       "mlx2_host_memory_available_bytes"):
            eq(f"B: {metric} exported",
               any(l.startswith(metric) for l in mb.splitlines()), True)
    if sa is not None and sb is not None:
        print(f"       [informational] headroom A={sa.get('headroom_bytes',0)/2**30:.2f} GiB "
              f"B={sb.get('headroom_bytes',0)/2**30:.2f} GiB")


# --------------------------------------------------------------- job 14 ----
def gates_14(out):
    # gate 1: key file mode 0600 and 64 bytes
    st = text(os.path.join(out, "keystat.txt")).split()
    if len(st) == 2:
        mode, size = st
        eq("g1 key mode 0600", mode, "-rw-------")
        eq("g1 key size 64 bytes", size, "64")
    else:
        bad("g1 reasoning-signing.key stat", f"unreadable: {st!r}")
    alog = text(os.path.join(out, "a.log"))
    eq("g1 a.log announces the persisted key",
       "reasoning signing key persisted at" in alog, True)

    # gate 2: a.signing and b.signing both ephemeral False with the same key_id
    def signing(name):
        raw = text(os.path.join(out, name)).strip()
        try:  # the server prints a python dict repr
            return json.loads(raw.replace("'", '"').replace("False", "false").replace("True", "true"))
        except Exception:  # noqa: BLE001
            return None
    sa, sb, sc = signing("a.signing"), signing("b.signing"), signing("c.signing")
    if sa and sb:
        eq("g2 A ephemeral False", sa.get("ephemeral"), False)
        eq("g2 B ephemeral False", sb.get("ephemeral"), False)
        eq("g2 key_id identical across restart",
           sa.get("key_id") == sb.get("key_id"), True)
        print(f"       A key_id={sa.get('key_id')}  B key_id={sb.get('key_id')}")
    else:
        bad("g2 signing status readable", f"a={sa} b={sb}")

    # gate 3: a.msg.json carries a signed thinking block
    a = load(os.path.join(out, "a.msg.json"), "g3 run A message readable")
    if a is not None:
        blocks = a.get("content") if isinstance(a.get("content"), list) else []
        signed = [b for b in blocks if b.get("type") == "thinking" and b.get("signature")]
        if signed:
            ok("g3 A has a signed thinking block", signed[0]["signature"][:46] + "...")
        else:
            bad("g3 A has a signed thinking block", json.dumps(a)[:220])

    # SPEC DEFECT: the spec's own turn 2 sets budget_tokens == max_tokens == 64,
    # which the API rejects before any signature is verified.
    spec_code = text(os.path.join(out, "b.spec.code")).strip()
    spec_msg = load(os.path.join(out, "b.spec.msg.json"))
    if spec_code:
        if spec_code == "200":
            ok("spec-literal turn 2 accepted", "the budget==max defect is not live on this tree")
        else:
            detail = ""
            if isinstance(spec_msg, dict):
                detail = str(spec_msg.get("error", {}).get("message", ""))[:120]
            print(f"  spec NOTE  SPEC DEFECT: turn 2 as 14-agent-launcher.md writes it "
                  f"returns HTTP {spec_code} ({detail}). budget_tokens == max_tokens == 64 "
                  f"is rejected by parameter validation, so gate 4 cannot be evaluated "
                  f"from the spec-literal body. Evaluated from turn2fix.json instead.")

    # gate 4: replay accepted after restart, rejected counter 0
    bcode = text(os.path.join(out, "b.code")).strip()
    eq("g4 run B replay returns 200", bcode, "200")
    b = load(os.path.join(out, "b.msg.json"))
    if isinstance(b, dict):
        eq("g4 run B is a normal message", b.get("type"), "message")
    bmet = text(os.path.join(out, "b.metrics"))
    b_rej = rejected_count(bmet)
    eq("g4 rejected counter is 0 after a valid replay", b_rej, 0)

    # The tamper arm: this is what tells us gate 4 is not vacuous.
    tcode = text(os.path.join(out, "b.tamper.code")).strip()
    t_rej = rejected_count(text(os.path.join(out, "b.tamper.metrics")))
    print(f"       tamper arm: HTTP {tcode or '<no artifact>'}, rejected counter {t_rej}")
    if not tcode:
        # An absent artifact is NOT a pass -- that is precisely the bug this
        # whole repair exists to remove.
        skip("TAMPER CONTROL: a corrupted signature is detected",
             "no b.tamper.code artifact; the tamper arm did not run")
    elif tcode != "200" or (t_rej is not None and b_rej is not None and t_rej > b_rej):
        ok("TAMPER CONTROL: a corrupted signature is detected",
           f"HTTP {tcode}, rejected {b_rej}->{t_rej}")
    else:
        bad("TAMPER CONTROL: a corrupted signature is detected",
            f"a TAMPERED signature was accepted (HTTP {tcode}) and the rejected "
            f"counter did not move ({b_rej}->{t_rej}). Inbound signatures are not "
            f"being verified, which makes gate 4's pass vacuous.")

    # gate 5: control arm must be able to tell the two behaviours apart
    if sc:
        eq("g5 C ephemeral True", sc.get("ephemeral"), True)
        if sa:
            eq("g5 C key_id differs from A", sc.get("key_id") != sa.get("key_id"), True)
    c_rej = rejected_count(text(os.path.join(out, "c.metrics")))
    ccode = text(os.path.join(out, "c.code")).strip()
    print(f"       control arm C: HTTP {ccode or '<no artifact>'}, rejected counter {c_rej}")
    if not os.path.exists(os.path.join(out, "c.metrics")):
        skip("g5 rejected counter >= 1 under an ephemeral key",
             "no c.metrics artifact; run C did not complete")
    elif c_rej is not None and c_rej >= 1:
        ok("g5 rejected counter >= 1 under an ephemeral key", str(c_rej))
    else:
        bad("g5 rejected counter >= 1 under an ephemeral key",
            f"counter={c_rej}, replay under a DIFFERENT key returned HTTP {ccode}. "
            "The spec says: reject if gate 5 fails, because the smoke cannot tell "
            "the two behaviours apart.")

    # gate 6: launcher discovery
    disc = text(os.path.join(out, "discover.txt"))
    if disc.strip():
        eq("g6 discovery reports 32768", "32768" in disc, True)
        served = text(os.path.join(out, "served_id.txt")).strip()
        if served:
            eq("g6 discovery reports the served model id", served.split("=")[-1] in disc, True)
        else:
            skip("g6 discovery reports the served model id", "served id not recorded")
    else:
        bad("g6 launcher discovery produced output", "discover.txt empty")


def rejected_count(metrics_text):
    """mlx2_runtime_events_total{...event="rejected"} N -> N (absent counts as 0)."""
    if not metrics_text.strip():
        return 0
    for line in metrics_text.splitlines():
        if "reasoning_signature" in line and 'event="rejected"' in line:
            m = re.search(r"\s([0-9.]+)\s*$", line.strip())
            if m:
                return int(float(m.group(1)))
    return 0


# --------------------------------------------------------------- job 13 ----
def gates_13(out):
    """Spec 13: every smoke run prints "failures": [] and exits 0, both models,
    cold and warm; cold updates >= 2 on at least one model; warm cached > 0."""
    found = False
    cold_updates_ok = False
    warm_cached_ok = False
    for name in sorted(os.listdir(out)) if os.path.isdir(out) else []:
        if not name.endswith(".json"):
            continue
        data = load(os.path.join(out, name))
        if not isinstance(data, dict):
            continue
        found = True
        fails = data.get("failures")
        if fails == []:
            ok(f"{name}: failures == []")
        else:
            bad(f"{name}: failures == []", json.dumps(fails)[:200])
        updates = data.get("updates")
        if not name.endswith(".warm.json") and isinstance(updates, int) and updates >= 2:
            cold_updates_ok = True
        tail = json.dumps(data.get("progress_tail", data))
        if name.endswith(".warm.json") and re.search(r'"cached":\s*[1-9]', tail):
            warm_cached_ok = True
    if not found:
        bad("13 smoke artifacts present", f"no smoke JSON under {out}")
        return
    eq("cold updates >= 2 on at least one model", cold_updates_ok, True)
    eq("warm run shows cached > 0", warm_cached_ok, True)
    skip("cold wall/TTFT within 3% of main",
         "spec asks for a separate run against /private/tmp/mlx2-splash-base; "
         "this queue does not run the main-tree baseline")


# --------------------------------------------------------------- job 12 ----
def gates_12(out):
    """Spec 12 correctness gates.

    bench_tool_grammar.py records one ROW per (scenario, repeat) with an `ok`
    flag, and a `scenarios` summary dict.  An earlier version of this
    evaluator read `scenarios` and reported a pass while three rows per arm
    were failing -- the same "looked at the wrong field and called it green"
    mistake this whole repair exists to remove.  Gate on the rows.
    """
    arms = {}
    for name in sorted(os.listdir(out)) if os.path.isdir(out) else []:
        # bench results are "<tag>-A.json"/"<tag>-B.json"; policy-A/B.json are
        # the policy inputs this job writes, not results.
        if re.match(r".*-(A|B)\.json$", name) and not name.startswith("policy-"):
            arms[name] = load(os.path.join(out, name))
    if not arms:
        bad("12 bench artifacts present", f"no per-arm JSON under {out}")
        return
    for name, data in sorted(arms.items()):
        if not isinstance(data, dict):
            bad(f"{name} readable", "not a JSON object")
            continue
        rows = data.get("rows") or []
        if not rows:
            bad(f"{name} has bench rows", "no 'rows' array in the bench JSON")
            continue
        http5xx = [r for r in rows if isinstance(r.get("status"), int)
                   and r["status"] >= 500]
        eq(f"{name} g1 no HTTP 5xx",
           sorted({r.get("scenario") for r in http5xx}), [])
        failing = [r for r in rows if r.get("ok") is False]
        if failing:
            by_scenario = {}
            for r in failing:
                key = r.get("scenario", "?")
                err = r.get("error")
                by_scenario.setdefault(key, set()).add(
                    str(err)[:110] if err else f"ok=False status={r.get('status')} "
                                               f"calls={r.get('calls')}")
            detail = "; ".join(f"{k} x{sum(1 for r in failing if r.get('scenario') == k)} "
                               f"[{' | '.join(sorted(v))}]"
                               for k, v in sorted(by_scenario.items()))
            bad(f"{name} g1/g2 every scenario row ok", detail)
        else:
            ok(f"{name} g1/g2 every scenario row ok", f"{len(rows)} rows")
        for counter in ("structured_output_failures", "dead_end"):
            val = data.get(counter)
            if val is None:
                skip(f"{name} g1 {counter} == 0", "counter absent from bench JSON")
            else:
                eq(f"{name} g1 {counter} == 0", val, 0)
    # Gate 3 is a comparison BETWEEN arms: B must be >= A on auto_no_tool.
    a = next((v for k, v in arms.items() if k.endswith("-A.json")), None)
    b = next((v for k, v in arms.items() if k.endswith("-B.json")), None)
    if isinstance(a, dict) and isinstance(b, dict):
        def clean(d):
            return sum(1 for r in (d.get("rows") or [])
                       if r.get("scenario") == "auto_no_tool" and r.get("ok"))
        ca, cb = clean(a), clean(b)
        if cb >= ca:
            ok("g3 auto_no_tool: B >= A", f"A={ca} B={cb} clean rows")
        else:
            bad("g3 auto_no_tool: B >= A", f"A={ca} B={cb} clean rows")
    skip("g4 answer_or_call_* schema conformance",
         "requires the per-row response bodies; bench JSON records only ok/calls")
    skip("g5 streamed Responses SSE output_index ordering",
         "requires parsing responses-B.sse against the final response.completed "
         "order; not implemented -- evaluate by hand from the .sse artifact")
    skip("g6 thinking models keep reasoning_content",
         "only the flashnext arm ran (north/muse are --full); not evaluated")


# ------------------------------------------------- generic bench-arm jobs ----
def arm_gates(out, job, on_key, counters=(), text_equality=True):
    """Shared shape for 06/07/08/10: four arms (mtp/ord x off/on), each with a
    smoke response, a status JSON and a bench JSON."""
    arms = ["mtp-off", "mtp-on", "ord-off", "ord-on"]
    present = [a for a in arms if os.path.exists(os.path.join(out, f"{a}.status.json"))]
    if not present:
        bad(f"{job} arm artifacts present", f"no *.status.json under {out}")
        return
    for arm in present:
        st = load(os.path.join(out, f"{arm}.status.json"))
        settings = (st or {}).get("settings", {})
        if arm.endswith("-on"):
            eq(f"{arm} g1 settings carry {on_key}", on_key in settings, True)
        else:
            eq(f"{arm} g1 settings do NOT carry {on_key}", on_key in settings, False)
        smoke = load(os.path.join(out, f"{arm}.smoke.json"))
        # AUTHORITATIVE per-arm route assertion.  The smoke RECEIPT carries a
        # single unambiguous mlx2.route plus route_selection_source; /v1/status
        # exposes the value nested in several places and is only advisory.
        # segmented_self_mtp / continuous_batched_self_mtp are VARIANTS of the
        # native self-MTP route, not different routes.
        if isinstance(smoke, dict):
            rec = smoke.get("mlx2") or {}
            got = rec.get("route")
            src = rec.get("route_selection_source")
            want = "native_mtp" if arm.startswith("mtp-") else "ordinary"
            if got is None:
                skip(f"{arm} route assertion", "receipt carries no mlx2.route")
            elif want == "native_mtp":
                eq(f"{arm} route is native self-MTP (got {got!r}, source {src!r})",
                   ("native_mtp" in got) or ("self_mtp" in got), True)
            else:
                eq(f"{arm} route is ordinary (got {got!r}, source {src!r})",
                   got == "ordinary", True)
        if isinstance(smoke, dict):
            txt = content_of(smoke)
            fr = None
            try:
                fr = smoke["choices"][0]["finish_reason"]
            except Exception:  # noqa: BLE001
                pass
            eq(f"{arm} g1 smoke non-empty", bool(txt), True)
            eq(f"{arm} g1 finish_reason in stop/length", fr in ("stop", "length"), True)
        else:
            bad(f"{arm} g1 smoke response readable", "missing or unparseable")
        slog = os.path.join(out, f"{arm}.server.log")
        if os.path.exists(slog):
            eq(f"{arm} g2 no tracebacks", text(slog).count("Traceback"), 0)
        for counter in counters:
            val = (st or {}).get(counter)
            if val is None:
                skip(f"{arm} {counter} == 0", "counter absent from /v1/status")
            else:
                eq(f"{arm} {counter} == 0", val, 0)
    missing = [a for a in arms if a not in present]
    if missing:
        bad(f"{job} all four arms ran", f"missing: {missing}")
    if text_equality:
        skip(f"{job} greedy text equality on == off per route",
             "requires per-prompt text from the bench JSONs; the bench schema is not "
             "pinned here -- compare by hand from the *.bench.json artifacts")
    skip(f"{job} perf accept/reject thresholds",
         "spec thresholds are ratios over bench JSON fields; reported, not auto-gated")


# --------------------------------------------------------------- others ----
def gates_01(out):
    """Spec 01 gates A-F.  bench_gpu_accept_count.py reports per-width
    `all_prefixes_bit_exact` / `bitwise_mismatch_prefixes` plus a top-level
    `pass`, not a flat mismatch counter."""
    ab = load(os.path.join(out, "kernel_ab.json"))
    if ab is None:
        bad("gate A/B kernel_ab.json present", "missing")
    else:
        eq("gate A/B kernel_ab top-level pass", ab.get("pass"), True)
        eq("gate A/B cpu reference pass", ab.get("reference_pass"), True)
        for section in ("generic_masked_ab", "reconstruct_kernel_ab"):
            rows = ab.get(section) or []
            if not rows:
                skip(f"gate A/B {section}", "section absent")
                continue
            offenders = [r.get("width") for r in rows
                         if r.get("bitwise_mismatch_prefixes")]
            eq(f"gate A/B {section}: no bitwise mismatch at any width",
               offenders, [])
        cpu = (ab.get("cpu_reference") or {}).get("results") or []
        notexact = [r.get("width") for r in cpu
                    if r.get("all_prefixes_bit_exact") is not True]
        if cpu:
            eq("gate A/B cpu_reference bit-exact at every width", notexact, [])
    for name, label in (("replay_template.json", "gate C template arm"),
                        ("replay_dynamic.json", "gate C/D dynamic accept")):
        data = load(os.path.join(out, name))
        if data is None:
            bad(label, f"{name} missing")
            continue
        # qualify_qwen4_gdn_replay_model.py reports `status`/`qualified` plus a
        # final_stats block whose *_fallbacks counters are the exactness gate.
        status = data.get("status")
        GOOD = {"ok", "qualified", "qualified_selected"}
        if status is not None:
            if status in GOOD:
                ok(f"{label} status", status)
            else:
                bad(f"{label} status", f"got {status!r}, want one of {sorted(GOOD)}")
        if "qualified" in data:
            eq(f"{label} qualified", data.get("qualified"), True)
        stats = data.get("final_stats") or {}
        if stats:
            for counter in ("fallbacks", "verify_fallbacks", "replay_fallbacks",
                            "catchup_fallbacks"):
                if counter in stats:
                    eq(f"{label} final_stats.{counter} == 0", stats[counter], 0)
        if status is None and "qualified" not in data and not stats:
            skip(label, f"{name} exposes no pass flag, mismatch counter or final_stats")
    # GATE E: the old runner only PRINTED "compare greedy-on vs greedy-off".
    for i in (1, 2, 3):
        on = load(os.path.join(out, f"greedy-on-p{i}.json"))
        off = load(os.path.join(out, f"greedy-off-p{i}.json"))
        if on is None or off is None:
            skip(f"gate E prompt {i}", "one of the arms produced no output")
            continue
        ta, tb = content_of(on), content_of(off)
        if ta is None or tb is None:
            bad(f"gate E prompt {i}", "response was not a chat completion")
        elif ta == tb:
            ok(f"gate E prompt {i}: policy on == off", f"{len(ta)} chars")
        else:
            bad(f"gate E prompt {i}: policy on == off",
                f"on={ta[:80]!r} off={tb[:80]!r}")
    skip("gate F / perf numbers", "spec asks for them in $OUT/README.md; reported, not auto-gated")


def gates_04(queue_dir):
    res = load(os.path.join(queue_dir, "04-result.json"))
    if res is None:
        bad("04 in-process A/B result present", "04-result.json missing")
    else:
        for key in ("tokens_identical_greedy", "tokens_identical_sampled", "tokens_identical"):
            if key in res:
                eq(f"04 {key}", res[key], True)
        if not any(k.startswith("tokens_identical") for k in res):
            skip("04 tokens_identical", "bench JSON exposes no tokens_identical field")
    qual = load(os.path.join(queue_dir, "04-qualify.json"))
    if qual is None:
        bad("04 qualify_serving result present", "04-qualify.json missing")
    else:
        feats = json.dumps(qual)
        for feat in ("external_draft", "segmented_transaction"):
            eq(f"04 qualify requires {feat}", feat in feats, True)
    skip("04 perf (snapshot_alloc_bytes / freeze_us_median / tok_per_s)",
         "spec marks these informational; reported, not auto-gated")


def gates_02(out):
    """Spec 02: the bench writes {"micro": [...], "e2e": [...]} -- the gates are
    per ROW, not top-level scalars."""
    for path, label in (("/tmp/pair-select-k4.json", "k=4"),
                        ("/tmp/pair-select-k7.json", "k=7")):
        data = load(path)
        if not isinstance(data, dict):
            bad(f"{label} bench JSON present", f"{path} missing")
            continue
        micro = data.get("micro") or []
        if not micro:
            bad(f"{label} micro rows present", "no 'micro' array")
        for row in micro:
            w = row.get("width")
            eq(f"{label} micro w{w} token_or_rng_mismatches == 0",
               row.get("token_or_rng_mismatches"), 0)
            q = row.get("max_abs_q_diff")
            if q is None:
                skip(f"{label} micro w{w} max_abs_q_diff <= 1e-3", "field absent")
            else:
                (ok if q <= 1e-3 else bad)(
                    f"{label} micro w{w} max_abs_q_diff <= 1e-3", f"{q:.3g}")
            hm, bm = row.get("host_ms_median"), row.get("batched_ms_median")
            if hm is not None and bm is not None:
                (ok if bm <= hm else bad)(
                    f"{label} PERF micro w{w} batched <= host",
                    f"batched {bm:.2f} ms vs host {hm:.2f} ms")
        for row in data.get("e2e") or []:
            w, t = row.get("width"), row.get("temp")
            tag = f"{label} e2e w{w} T{t}"
            if t == 0 or t == 0.0:
                for flag in ("identical_tokens", "identical_receipts",
                             "identical_rng_draws"):
                    eq(f"{tag} {flag}", row.get(flag), True)
            else:
                eq(f"{tag} identical_rng_draws", row.get("identical_rng_draws"), True)
                if row.get("identical_tokens") is not True:
                    acc_h = row.get("host_acceptance")
                    acc_b = row.get("batched_acceptance")
                    print(f"       {tag}: identical_tokens False; acceptance "
                          f"host={acc_h} batched={acc_b} (spec allows this only if "
                          f"acceptance matches within 2 pts and micro gates hold)")
            sp = row.get("speedup")
            if sp is None:
                continue
            if isinstance(w, int) and w >= 4:
                (ok if sp >= 1.03 else bad)(
                    f"{tag} PERF speedup >= 1.03", f"{sp:.4f}")
            if sp < 0.97:
                bad(f"{tag} PERF no >3% regression", f"speedup {sp:.4f}")
    # serving smoke
    codes = {n: text(os.path.join(out, f"{n}.code")).strip().split()[-1:] or [""]
             for n in ("poem", "math", "tool")}
    if not any(c[0] for c in codes.values()):
        bad("02 serving smoke ran",
            "no HTTP codes recorded. SPEC DEFECT: 02-dflash-pair-select.md step 2 "
            "starts the server without --qualification-mode, which this tree now "
            "requires ('server.py: error: provide --qualification or explicitly run "
            "--qualification-mode'), so the smoke cannot run as written.")
        return
    for n, c in codes.items():
        eq(f"serving smoke {n} -> 200", c[0], "200")
    metrics = text(os.path.join(out, "metrics.txt"))
    m = re.search(r"external_pairwise_selection_groups\D+([0-9.]+)", metrics)
    if m:
        (ok if float(m.group(1)) > 0 else bad)(
            "external_pairwise_selection_groups > 0", m.group(1))
    else:
        bad("external_pairwise_selection_groups > 0", "counter absent from /metrics")


def gates_05(out):
    print("  spec NOTE  05-gpu-topk-accept.md has NO '## Accept / reject' section of its "
          "own; the gates below are read out of the prose at the end of sections 1 and 2.")
    served = {}
    for arm in ("off", "on"):
        st = load(os.path.join(out, f"05-status-{arm}.json"))
        if st is None:
            bad(f"05 status-{arm} present", "missing")
            continue
        served[arm] = st
        blob = json.dumps(st)
        if arm == "off":
            eq("OFF: no gpu_acceptance* keys anywhere in status",
               "gpu_acceptance" in blob, False)
        else:
            eq("ON: gpu_acceptance present in status", "gpu_acceptance" in blob, True)
            for counter in ("gpu_acceptance_fallback_greedy",
                            "gpu_acceptance_fallback_logprobs"):
                m = re.search(rf'"{counter}":\s*([0-9]+)', blob)
                if m:
                    (ok if int(m.group(1)) >= 1 else bad)(f"ON: {counter} >= 1", m.group(1))
                else:
                    bad(f"ON: {counter} >= 1", "counter absent from status")
    skip("05 micro decision_mismatches == 0",
         "the micro arm gates itself (bench_gpu_topk_accept.py exits 1); its rc is "
         "gated by the runner")
    skip("05 perf ON tok/s >= OFF tok/s * 1.05",
         "requires the serve bench JSONs; reported, not auto-gated")


def gates_03(out):
    sweep = load(os.path.join(out, "sweep.json"))
    if sweep is None:
        bad("03 sweep.json present", "missing")
    confirm = load(os.path.join(out, "confirm.json"))
    if confirm is None:
        bad("03 confirm.json present", "missing")
    print("  spec NOTE  spec 03 step 1b contains the literal placeholder `<best-from-1a>`, "
          "which is both a shell redirect and an unparseable K. The runner substitutes a "
          "fixed finalist set (QUEUE_03_CONFIRM_K, default 4,7,15), so K* from 1a is "
          "confirmed only if it happens to be in that set.")
    smoke = load(os.path.join(out, "serve-k15.json"))
    if isinstance(smoke, dict):
        eq("03 g2 K=15 serving smoke non-empty", bool(content_of(smoke)), True)
    else:
        bad("03 g2 K=15 serving smoke", "serve-k15.json missing or unparseable")
    skip("03 g1 greedy_exact_rows per K vs K=4",
         "requires the sweep JSON's per-K row counts; schema not pinned here")
    skip("03 g3 peak_memory_gib K=15/B=4 within 2 GiB of K=4/B=4",
         "requires the sweep JSON's memory fields; schema not pinned here")
    skip("03 g4 K* selection rule", "perf decision, reported not auto-gated")



def gates_08_checks(out):
    """Spec 08 gate 3: the *-on arms require bench.json.checks.accept == true,
    which in turn requires the four text-match flags, preemptions == replays
    == 3, reference_has_no_preemption and apcv2_store_failures == 0."""
    for arm in ("ord-on", "mtp-on"):
        path = os.path.join(out, f"{arm}.bench.json")
        data = load(path)
        if not isinstance(data, dict):
            bad(f"{arm} g3 bench.json readable", f"missing {path}")
            continue
        checks = data.get("checks", {})
        if not checks:
            bad(f"{arm} g3 bench checks present", "no 'checks' object")
            continue
        eq(f"{arm} g3 checks.accept", checks.get("accept"), True)
        for flag in ("decode_replay_text_matches", "prefill_replay_text_matches",
                     "concurrent_victim_text_matches", "concurrent_peer_text_matches",
                     "reference_has_no_preemption"):
            eq(f"{arm} g3 {flag}", checks.get(flag), True)
        eq(f"{arm} g3 preemptions == 3", checks.get("preemptions"), 3)
        eq(f"{arm} g3 replays == 3", checks.get("replays"), 3)
        eq(f"{arm} g3 store_failures == 0", checks.get("store_failures"), 0)


# ---------------- re-derived gates for the admission-dependent specs -------
# Derivations recorded in THRESHOLD-DERIVATION-06-07-08-10.txt, written before
# these ran.  Bars expressed as on/off ratios at the same lane count did not
# move; bars that assumed an absolute lane count did.

def _lifetime(status, key, default=None):
    blob = json.dumps(status or {})
    m = re.search(rf'"{key}"\s*:\s*([0-9]+)', blob)
    return int(m.group(1)) if m else default


def gates_06_junction(out):
    """06b: catch capture -> publish -> EVICTED-before-use.

    FIRST VERSION OF THIS GATE WAS WRONG and is worth recording.  It keyed on
    `junction_hits`, which reads 0 in this tree even when junctions are
    demonstrably resumed -- so it reported EVICT-BEFORE-USE on both on-arms
    while `cached_tokens` showed the feature plainly working.  The derivation
    required 06b and 06c to agree; they did not, and checking resolved it.
    The reliable signal is the per-turn `checkpoint_role`, which is the
    observable CONSEQUENCE (this turn resumed from a junction) rather than a
    counter that claims it happened.  `junction_hits` is now advisory, and its
    disagreement is reported as the counter defect it is.
    """
    def turns_of(arm):
        d = load(os.path.join(out, f"{arm}.bench.json"))
        if not isinstance(d, dict):
            return []
        return [t for t in (d.get("turns") or d.get("results") or [])
                if isinstance(t, dict)]

    for arm in ("mtp-on", "ord-on"):
        st = load(os.path.join(out, f"{arm}.status.json"))
        published = _lifetime(st, "apc_junction_checkpoints_published")
        captured = _lifetime(st, "apc_junction_checkpoints_captured")
        hits = _lifetime(st, "junction_hits")
        turns = turns_of(arm)
        roles = [t.get("checkpoint_role") or t.get("cache_checkpoint_role") for t in turns]
        used = sum(1 for r in roles if r == "junction")
        print(f"       {arm}: captured={captured} published={published} "
              f"junction_hits={hits} roles={roles}")
        if captured is None or published is None:
            bad(f"{arm} 06a junction counters present", "absent from /v1/status")
        else:
            eq(f"{arm} 06a captured >= 1", captured >= 1, True)
            eq(f"{arm} 06a published >= 1", published >= 1, True)
        # 06b: the branching turns (2..4, i.e. index 2..4 here) must actually
        # RESUME from a junction.  This is what eviction-before-use would break.
        if not turns:
            bad(f"{arm} 06b junction actually used", "no bench turns recorded")
        elif used >= 3:
            ok(f"{arm} 06b junction RESUMED on {used} turns (no evict-before-use)",
               f"roles={roles}")
        else:
            bad(f"{arm} 06b EVICT-BEFORE-USE: published {published} but only "
                f"{used} turn(s) resumed from a junction", f"roles={roles}")
        # 06c: independent consequence -- cached_tokens equal across branching
        # turns and above turn 1.  Must agree with 06b.
        ct = [t.get("cached_tokens") for t in turns]
        if len(ct) >= 5 and all(isinstance(v, int) for v in ct[2:5]):
            equal = len(set(ct[2:5])) == 1
            above = ct[2] > (ct[0] or 0)
            eq(f"{arm} 06c cached_tokens equal across branching turns", equal, True)
            eq(f"{arm} 06c cached_tokens above turn 1", above, True)
            if equal and above and used < 3:
                bad(f"{arm} 06 INCONCLUSIVE",
                    "06b and 06c disagree: cached_tokens says the junction is "
                    "working but the role/counter says it is not")
        else:
            skip(f"{arm} 06c cached_tokens shape", f"unexpected turn data: {ct}")
        # junction_hits demoted to advisory, and its disagreement reported.
        if hits is not None and used >= 3 and hits == 0:
            print(f"  spec NOTE  COUNTER DEFECT: {arm} resumed from a junction on "
                  f"{used} turns (checkpoint_role) but junction_hits reads 0. The "
                  f"counter is not wired to the resume path; do not gate on it.")
        degraded = _lifetime(st, "apc_junction_checkpoints_degraded")
        if degraded is None:
            skip(f"{arm} 06d degraded == 0", "counter absent")
        else:
            eq(f"{arm} 06d apc_junction_checkpoints_degraded == 0", degraded, 0)
    # off-arms must NOT resume from a junction
    for arm in ("mtp-off", "ord-off"):
        roles = [t.get("checkpoint_role") or t.get("cache_checkpoint_role")
                 for t in turns_of(arm)]
        eq(f"{arm} 06 off-arm never resumes from a junction",
           [r for r in roles if r == "junction"], [])


def gates_07_rolling(out):
    """07a: the +1 GiB memory bar was written for ~1 live rolling snapshot.
    At 4 seated lanes there can be up to 4, so the bar scales per lane."""
    LANES = 4
    def peak(arm):
        st = load(os.path.join(out, f"{arm}.status.json"))
        return _lifetime(st, "process_physical_footprint_bytes"), st
    for route in ("mtp", "ord"):
        on_v, on_st = peak(f"{route}-on")
        off_v, _ = peak(f"{route}-off")
        if on_v is None or off_v is None:
            skip(f"{route} 07a memory within +1 GiB x lanes",
                 "process_physical_footprint_bytes absent from /v1/status")
        else:
            delta = (on_v - off_v) / 2**30
            budget = 1.0 * LANES
            per_lane = delta / LANES
            (ok if delta <= budget else bad)(
                f"{route} 07a peak delta <= 1 GiB x {LANES} lanes",
                f"{delta:+.2f} GiB total, {per_lane:+.2f} GiB/lane, budget {budget:.1f} GiB")
        # 07d watched, NOT gated: no defensible bar exists for these yet.
        for counter in ("apc_rolling_checkpoints_skipped_pressure",
                        "apc_rolling_checkpoints_degraded"):
            v = _lifetime(on_st, counter)
            print(f"       [watched, not gated] {route}-on {counter} = {v}")


def gates_08_replay(out):
    """Item 08 RESTATED gate (requeue, branch claude/splash-08-onmain @82a3b4e).

    The two cross-position token-identity gates were REMOVED, not relaxed:
    tonight established that shape is unsatisfiable at batch width (8/8
    token-identical at width 1, 1/8 at width 16; logprob margins differ
    between two lanes running the same prompt in the same batch).  They are
    replaced by four properties that are invariants of the MECHANISM rather
    than of batch composition, so a failure here is a real defect and must be
    reported as one rather than restated away.

    Replay exactness is still asserted, on decode_replay and prefill_replay,
    compared within one process at one width -- which is where it is
    satisfiable and where it means what it says.
    """
    for arm in ("ord-on", "mtp-on"):
        data = load(os.path.join(out, f"{arm}.bench.json"))
        if not isinstance(data, dict):
            bad(f"{arm} 08 bench.json readable", "missing"); continue
        c = data.get("checks", {})
        if not c:
            bad(f"{arm} 08 bench checks present", "no 'checks'"); continue

        # exactness, same process, one width -- unchanged
        for flag in ("decode_replay_text_matches", "prefill_replay_text_matches"):
            eq(f"{arm} 08 {flag}", c.get(flag), True)
        # the four width-safe replacements
        for flag in ("concurrent_peer_not_preempted",
                     "concurrent_victim_completed", "concurrent_peer_completed"):
            eq(f"{arm} 08 {flag} [restated, width-safe]", c.get(flag), True)
        # Independent predicate, kept as a second opinion even though the
        # bench's own check was repaired at cb016c2 to route both gates
        # through `bool(replays) and fault_unfired is None`.  Reporting
        # AGREEMENT between the two is worth more than either alone.
        def really_preempted(scen):
            rec = ((data.get("arms") or {}).get(scen) or {}).get("preemption") or {}
            if not isinstance(rec, dict):
                return False, None
            return (bool(rec.get("replays")) and rec.get("fault_unfired") is None,
                    rec.get("fault_unfired"))

        vic_real, vic_reason = really_preempted("concurrent_victim")
        bench_says = c.get("concurrent_victim_preempted")
        eq(f"{arm} 08 independent predicate agrees with the bench's own check",
           vic_real, bench_says)
        if vic_real:
            ok(f"{arm} 08 concurrent_victim_preempted [restated, width-safe]",
               "receipt shows replays>0 and no fault_unfired")
        else:
            bad(f"{arm} 08 concurrent_victim_preempted [restated, width-safe]",
                f"fault_unfired={vic_reason!r} -- the fault was DECLINED, not fired")
        # decode_replay exactness is only meaningful if that fault fired.
        dec_real, dec_reason = really_preempted("decode_replay")
        if dec_real:
            ok(f"{arm} 08 decode_replay actually fired -- its text match is a REAL "
               f"assertion", "replays>0, no fault_unfired")
        else:
            skip(f"{arm} 08 decode_replay_text_matches IS VACUOUS",
                 f"decode_replay was not preempted (fault_unfired={dec_reason!r}); "
                 f"'replayed text matches reference' is trivially true because no "
                 f"replay occurred")
        st = load(os.path.join(out, f"{arm}.status.json"))
        # ---- instrumentation 1: TRIGGER PROVENANCE ----
        # This is the first run in which item 9's real P4 monitor can fire at
        # all (it replaced a NORMAL stub), so a _pressure preemption inside a
        # fault-scripted arm is a DIFFERENT event and would otherwise hide
        # inside the total.
        fault = _lifetime(st, "memory_preemptions_fault")
        pressure = _lifetime(st, "memory_preemptions_pressure")
        stall = _lifetime(st, "memory_preemptions_stall")
        total = _lifetime(st, "memory_preemptions")
        print(f"       {arm} TRIGGER PROVENANCE: total={total} fault={fault} "
              f"pressure={pressure} stall={stall}")
        if fault is None or total is None:
            bad(f"{arm} 08 trigger provenance available",
                "memory_preemptions_fault / _total absent from /v1/status")
        else:
            eq(f"{arm} 08 fault_preemptions == preemptions (scripted arm)", fault, total)
            if pressure:
                bad(f"{arm} 08 no PRESSURE-triggered preemption in a fault-scripted arm",
                    f"memory_preemptions_pressure={pressure}: item 9's live monitor "
                    f"fired inside a scripted arm; this is a different event from the "
                    f"injected fault and must not be counted as one")
            else:
                ok(f"{arm} 08 no pressure-triggered preemption in a scripted arm",
                   f"pressure={pressure}")
            if stall:
                bad(f"{arm} 08 no STALL-triggered preemption in a scripted arm",
                    f"memory_preemptions_stall={stall}")
            else:
                ok(f"{arm} 08 no stall-triggered preemption in a scripted arm",
                   f"stall={stall}")

        # ---- fault eligibility: NAME the block reason ----
        # 33a9f00: `preemption_block` exempts a lane only while it is in
        # prefill, so after_tokens:0 fires and any N>0 reaches the check in
        # decode and is declined by a non-None decode_replay_block.  The
        # refusal is correct; the SILENCE was the shipped defect.  It is now
        # carried on the receipt as preemption.fault_unfired, distinguishing
        # an eligibility decline from `never_reached`.
        for scen in ("decode_replay", "concurrent_victim", "prefill_replay"):
            arm_data = (data.get("arms") or {}).get(scen) or {}
            pre = arm_data.get("preemption") or {}
            unfired = pre.get("fault_unfired") if isinstance(pre, dict) else None
            fired = bool(pre) and not unfired
            print(f"       {arm} {scen}: preempted={fired} fault_unfired={unfired!r}")
        declines = _lifetime(st, "memory_preemption_fault_unfired")
        print(f"       {arm} memory_preemption_fault_unfired counter = {declines}")

        # ---- instrumentation 2: OBSERVED WIDTH / LANE COUNT ----
        max_lanes = _lifetime(st, "max_lanes")
        smoke = load(os.path.join(out, f"{arm}.smoke.json"))
        width = None
        if isinstance(smoke, dict):
            rec = smoke.get("mlx2") or {}
            width = rec.get("ordinary_compute_width")
            if width is None:
                width = ((rec.get("mtp") or {}).get("observed_compute_widths"))
        print(f"       {arm} WIDTH: configured max_lanes={max_lanes}, "
              f"observed compute width (smoke receipt)={width}")
        if max_lanes is None:
            skip(f"{arm} 08 lane ceiling recorded", "max_lanes absent from /v1/status")
        else:
            ok(f"{arm} 08 lane ceiling recorded", f"max_lanes={max_lanes}")
        skip(f"{arm} 08 peak observed batch width / active lanes at fault time",
             "NOT AVAILABLE in this tree: /v1/status's scheduler block exposes no "
             "target_max_width, peak_observed_batch_width or active-lane field, and "
             "bench_suspend_replay.py does not retain the receipt's compute width "
             "per arm. Recorded from the smoke receipt instead; adding a peak-width "
             "counter to the scheduler block would close this")


def gates_10_srpt(out):
    """10b/10d. 10a (the refusal) is gated in the runner, which is the only
    place that can observe a server refusing to start."""
    for arm in ("mtp-on", "ord-on", "mtp-off", "ord-off"):
        st = load(os.path.join(out, f"{arm}.status.json"))
        if st is None:
            bad(f"{arm} 10b status present", "missing -- job 10 captured no /v1/status before the harness fix")
            continue
        blob = json.dumps(st)
        keys = ["prefill_scheduling_bypasses", "prefill_scheduling_bypass_forced",
                "prefill_scheduling_one_slice_clamps"]
        present = [k for k in keys if f'"{k}"' in blob]
        if arm.endswith("-on"):
            eq(f"{arm} 10b scheduler counters exposed", sorted(present), sorted(keys))
            forced = _lifetime(st, "prefill_scheduling_bypass_forced")
            print(f"       {arm}: bypass_forced={forced} "
                  f"(window = --max-lanes = 4 = max_bypass+1, the minimum legal config)")
        else:
            eq(f"{arm} 10b off-arm exposes NO scheduler counters", present, [])
    mtp_on = load(os.path.join(out, "mtp-on.status.json"))
    if mtp_on is not None:
        v = _lifetime(mtp_on, "mtp_short_prefill_interleaved")
        if v is None:
            skip("10d mtp_short_prefill_interleaved > 0", "counter absent from status")
        else:
            (ok if v > 0 else bad)("10d mtp_short_prefill_interleaved > 0 (SRPT engaged)", str(v))


HANDLERS = {
    "11": gates_11, "09": gates_09, "14": gates_14, "13": gates_13, "12": gates_12,
    "01": gates_01, "04": gates_04, "02": gates_02, "05": gates_05, "03": gates_03,
    # The four admission-dependent specs run the shared arm checks PLUS their
    # re-derived gates (THRESHOLD-DERIVATION-06-07-08-10.txt).
    "06": lambda o: (arm_gates(o, "06", "apc_junction_checkpoints",
                               ("apcv2_store_failures",
                                "apc_junction_checkpoints_skipped_publish_failed")),
                     gates_06_junction(o)),
    "07": lambda o: (arm_gates(o, "07", "apc_rolling_checkpoints",
                               ("apcv2_store_failures",
                                "apc_rolling_checkpoints_skipped_publish_failed")),
                     gates_07_rolling(o)),
    "08": lambda o: (arm_gates(o, "08", "memory_preemption", ("apcv2_store_failures",)),
                     gates_08_replay(o)),
    "10": lambda o: (arm_gates(o, "10", "prefill_scheduling", ()),
                     gates_10_srpt(o)),
}


def main(argv):
    if len(argv) < 3:
        print("usage: gates.py <job_id> <artifact_dir>")
        return 2
    job, out = argv[1], argv[2]
    handler = HANDLERS.get(job)
    if handler is None:
        print(f"  no spec-gate handler for job {job}")
        return 3
    print(f"  --- spec gates for job {job} over {out} ---")
    handler(out)
    print(f"  --- {len(PASSED)} passed, {len(FAILED)} failed, "
          f"{len(SKIPPED)} NOT EVALUATED ---")
    if FAILED:
        return 1
    if not PASSED and SKIPPED:
        print("  SPEC GATES: UNTESTABLE -- nothing machine-checkable was evaluated.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
