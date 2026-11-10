#!/usr/bin/env python3
"""Quality/speed ablation ladder for DS4.1 Source precision tiers (Pierre 2026-11-08).

Runs a fixed screen of held-out prompts on an exact baseline arm, then on each ladder arm
(one server per arm, greedy, top-2 logprobs), and compares every arm to the baseline:

  * token identity and the first divergent position;
  * at that position, each arm's selected-vs-alternate margin and the top-2 sets, classified
    per AGENTS.md: identical | near_tie_equivalent | diverged_pass | diverged_fail | failed;
  * a functional oracle per prompt (translation keywords, YAML fields, GSM8K reference answer);
  * speed: decode tok/s, expert misses, bytes read and demand wait (status counter deltas).

Perplexity is deliberately not a metric. The server cannot score fixed text, so damage is read
from greedy outputs plus the margin at the first divergence.

usage:
  ablation_ladder.py LADDER.json OUT_DIR [--model PATH] [--port 8296] [--dry-run]
LADDER.json: {"base": {...policy...}, "arms": [{"name": "exact", "policy": {}},
              {"name": "l30_39_q2", "policy": {...merged over base...}}, ...],
              "sweeps": [{"name": "l13_26_q4_$N", "values": [256, 128, 64],
                          "template": {...policy with "$N" anywhere, e.g. an expert count...}}],
              "prompts": ["translate-6", ...] (optional), "max_tokens": 192 (optional),
              "warmup": 1, "repeat": 3 (optional)}
Warm mode (Pierre 2026-11-09: "a pre heated expert cache in memory, with decode only from in
memory experts"): with "warmup" W > 0 each arm server first runs every prompt W times unmeasured,
then "repeat" R measured passes. APCv2 is cleared before every pass (the server gets an admin
token in the arm directory), so each pass prefills and decodes the full prompt again. A measured
pass is `resident` only when it read zero expert bytes (original, Q2, Q1 or EXL3) and missed no
expert; the summary reports resident passes and quotes speed only from those. A non-resident
pass is kept in evidence but excluded, so a number never mixes disk streaming into it.
Sweeps expand after the explicit arms, one arm per value, so an expert count per tier per layer
span can be stepped down (Pierre: "ablate the specific number of experts at which quant"). A
string that is exactly "$N" becomes the value itself (int stays int); "$N" inside a longer
string is substituted as text.
The first arm is the baseline. Each arm directory is written atomically; a completed arm
(done.json) is skipped on rerun, and a partial arm directory is refused.
"""
import argparse, copy, json, os, re, signal, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

ED_PROMPTS = "/Volumes/T7/kimi-k3.1-pro/expert-draft-linear-20261107"
GSM8K = Path(__file__).resolve().parents[3] / "benchmarks" / "gsm8k" / "test.jsonl"
DEFAULT_MODEL = str(Path.home() / "mlx-models/DeepSeek-V4.1-Flash-1017-Source")
DEFAULT_PROMPTS = ["translate-6", "translate-7", "extract-6", "extract-7", "gsm8k-6", "gsm8k-7"]
NEAR_TIE_NAT = 0.5
NUMBER = re.compile(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")

# ------------------------------------------------------------------ oracles


def _norm_num(text):
    if text is None:
        return None
    m = NUMBER.search(text.replace("$", ""))
    if not m:
        return None
    v = m.group(0).replace(",", "").rstrip(".")
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def gsm8k_answer(text):
    if "####" in text:
        found = _norm_num(text.split("####", 1)[1])
        if found is not None:
            return found
    for marker in ("\\boxed{", "Answer:", "answer is"):
        if marker in text:
            found = _norm_num(text.rsplit(marker, 1)[1])
            if found is not None:
                return found
    nums = NUMBER.findall(text.replace("$", ""))
    return _norm_num(nums[-1]) if nums else None


def _keywords(text, groups):
    t = text.lower()
    return all(any(k in t for k in group) for group in groups)


def _yaml_maple(text):
    body = text.split("```")[1] if text.count("```") >= 2 else text
    body = re.sub(r"^\s*ya?ml\s*\n", "", body)
    try:
        import yaml
        d = yaml.safe_load(body)
    except Exception:
        return False
    if not isinstance(d, dict):
        return False
    get = lambda *ks: next((d[k] for k in d if str(k).lower().replace("_", " ") in ks), None)
    admins = get("admins")
    return (str(get("name")).strip() == "Maple Server" and str(get("port")) == "8443"
            and str(get("retries")) == "5" and get("tls") in (True, "enabled", "true", "yes")
            and isinstance(admins, list) and sorted(map(str, admins)) == ["alice", "bob"])


def oracles(gsm8k_rows=None):
    """prompt id -> (description, fn(reply) -> bool). GSM8K answers come from the vendored test set."""
    table = {
        "translate-6": ("English, keeps Munich/fifty minutes/signal", lambda r: _keywords(
            r, [["munich"], ["fifty minutes", "50 minutes"], ["signal"], ["late", "delay"]])),
        "translate-7": ("Portuguese, keeps library/hours/exam", lambda r: _keywords(
            r, [["biblioteca"], ["horário", "horario"], ["exame", "prova"]])),
        "extract-6": ("summary keeps solar/battery/gas", lambda r: _keywords(
            r, [["solar"], ["batter", "storage"], ["gas"]])),
        "extract-7": ("valid YAML with all fields", _yaml_maple),
    }
    for pid, question in (gsm8k_rows or {}).items():
        ref = question["reference"]
        table[pid] = (f"GSM8K answer {ref}", lambda r, ref=ref: gsm8k_answer(r) == ref)
    return table


def load_prompts(ids):
    sys.path.insert(0, ED_PROMPTS)
    from ed_prompts import CATS
    prompts, gsm = {}, {}
    test = [json.loads(l) for l in GSM8K.read_text().splitlines() if l.strip()] if GSM8K.exists() else []
    for pid in ids:
        cat, idx = pid.rsplit("-", 1)
        text = CATS[cat][int(idx)]
        prompts[pid] = text
        if cat == "gsm8k":
            head = re.sub(r"\s+", " ", text)[:60]
            row = next((r for r in test if re.sub(r"\s+", " ", r["question"]).startswith(head)), None)
            if row is None:
                raise SystemExit(f"{pid}: no GSM8K reference answer found in {GSM8K}")
            gsm[pid] = {"reference": _norm_num(row["answer"].split("####")[-1])}
    return prompts, gsm

# --------------------------------------------------------------- comparison


def tokens(response):
    content = ((response.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
    out = []
    for t in content:
        top = [(a["token"], a["logprob"]) for a in t.get("top_logprobs") or []]
        out.append({"token": t["token"], "logprob": t["logprob"], "top": top})
    return out


def _margin(entry):
    top = entry["top"]
    return None if len(top) < 2 else top[0][1] - top[1][1]


def compare(base, arm, base_pass, arm_pass):
    """Classify one arm's reply against the baseline reply (both token lists with top-2)."""
    n = min(len(base), len(arm))
    first = next((i for i in range(n) if base[i]["token"] != arm[i]["token"]), None)
    if first is None and len(base) == len(arm):
        return {"class": "identical" if arm_pass else "failed", "first_divergence": None,
                "shared_prefix": len(base)}
    if first is None:
        first = n                     # one reply is a strict prefix of the other
    row = {"first_divergence": first, "shared_prefix": first}
    if first >= n:
        row["class"] = "diverged_pass" if (base_pass and arm_pass) else "diverged_fail"
        row["note"] = "length differs after a shared prefix"
        return row
    b, a = base[first], arm[first]
    mb, ma = _margin(b), _margin(a)
    row.update(base_token=b["token"], arm_token=a["token"], base_margin=mb, arm_margin=ma,
               base_top=[t for t, _ in b["top"][:2]], arm_top=[t for t, _ in a["top"][:2]])
    same_top2 = set(row["base_top"]) == set(row["arm_top"]) and len(row["base_top"]) == 2
    near = (mb is not None and ma is not None and mb <= NEAR_TIE_NAT and ma <= NEAR_TIE_NAT and same_top2)
    if near and base_pass and arm_pass:
        row["class"] = "near_tie_equivalent"
    elif base_pass and arm_pass:
        row["class"] = "diverged_pass"
    else:
        row["class"] = "diverged_fail"
    return row


def counters(status):
    c = (((status or {}).get("execution") or {}).get("expert_stream") or {}).get("counters") or {}
    return {k: c.get(k, 0) for k in ("v41_expert_cache_misses", "v41_expert_cache_hits",
                                    "v41_expert_bytes_read", "v41_expert_demand_wait_ns",
                                    "v41_expert_exl3_bytes_read")}


def resident(experts):
    """Zero expert misses and zero expert bytes read from any bank during the pass."""
    return (experts.get("v41_expert_cache_misses", 0) == 0 and experts.get("v41_expert_bytes_read", 0) == 0
            and experts.get("v41_expert_exl3_bytes_read", 0) == 0)


def delta(after, before):
    return {k: after[k] - before.get(k, 0) for k in after}

# ------------------------------------------------------------------ serving


def merged(base, policy):
    out = copy.deepcopy(base)
    for k, v in policy.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _substitute(obj, value):
    if isinstance(obj, dict):
        return {k: _substitute(v, value) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, value) for v in obj]
    if obj == "$N":
        return value
    if isinstance(obj, str) and "$N" in obj:
        return obj.replace("$N", str(value))
    return obj


def expand(spec):
    """Explicit arms first, then one arm per sweep value."""
    arms = list(spec.get("arms", []))
    for sweep in spec.get("sweeps", []):
        for value in sweep["values"]:
            arms.append({"name": sweep["name"].replace("$N", str(value)),
                         "policy": _substitute(sweep["template"], value)})
    if not arms:
        raise SystemExit("ladder has no arms")
    return arms


def _get(url, timeout=30):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def _write(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def run_arm(name, policy, d, prompts, max_tokens, model_path, port, warmup=0, repeat=1):
    d.mkdir(parents=True)
    pol = d / "policy.json"; _write(pol, policy)
    cmd = [sys.executable, "-m", "mlx2.server", "--model", model_path, "--port", str(port), "--ordinary",
           "--execution-policy", str(pol), "--max-context", "16384"]
    token = None
    if warmup > 0:
        import secrets
        token = secrets.token_hex(16); tf = d / "admin-token"
        fd = os.open(tf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token + "\n")
        cmd += ["--admin-token-file", str(tf)]
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    _write(d / "source.json", {"arm": name, "mlx2_commit": rev, "tracked_dirty": dirty, "command": cmd,
                               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    log = open(d / "server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")})
    base = f"http://127.0.0.1:{port}"; replies = {}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{name}: server exited {proc.returncode} during load")
            try:
                _get(base + "/health", 5); break
            except Exception:
                time.sleep(5)
            if time.time() - t0 > 3600:
                raise RuntimeError(f"{name}: load timeout")
        model = _get(base + "/v1/models")["data"][0]["id"]
        before = _get(base + "/v1/status"); _write(d / "status-before.json", before)
        def one(text):
            if token:
                creq = urllib.request.Request(base + "/v1/admin/apc/clear", b'{"reason": "ladder warm pass"}',
                                              {"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
                try:
                    urllib.request.urlopen(creq, timeout=120).read()
                except urllib.error.HTTPError as error:
                    if error.code != 409:      # 409: this route keeps no prefix cache to reuse
                        raise
            s0 = counters(_get(base + "/v1/status"))
            body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                    "temperature": 0, "logprobs": True, "top_logprobs": 2}
            req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t = time.time(); resp = json.loads(urllib.request.urlopen(req, timeout=3600).read()); el = time.time() - t
            s1 = counters(_get(base + "/v1/status"))
            m = resp.get("mlx2") or {}
            ex = delta(s1, s0)
            return {"reply": resp["choices"][0]["message"].get("content") or "",
                    "finish": resp["choices"][0].get("finish_reason"), "elapsed_s": round(el, 2),
                    "ttft_s": m.get("ttft_seconds"), "completion_tokens": m.get("completion_tokens"),
                    "decode_tok_s": round((m.get("completion_tokens", 1) - 1) / max(1e-9, m.get("elapsed_seconds", 0) - m.get("ttft_seconds", 0)), 3),
                    "experts": ex, "resident": resident(ex), "tokens": tokens(resp),
                    "route": {k: v for k, v in resp.items() if k != "choices"}}

        warm = {}
        for w in range(warmup):
            for pid, text in prompts.items():
                r = one(text)
                warm.setdefault(pid, []).append({k: r[k] for k in ("elapsed_s", "ttft_s", "decode_tok_s", "experts", "resident")})
                _write(d / "warmup.json", warm)
        for pid, text in prompts.items():
            passes = [one(text) for _ in range(max(1, repeat))]
            last = passes[-1]
            if warmup > 0:
                # Speed is quoted only from passes that read nothing from disk.
                res = [p for p in passes if p["resident"]]
                last = dict(last)
                last["passes"] = [{k: p[k] for k in ("elapsed_s", "ttft_s", "decode_tok_s", "experts", "resident")} for p in passes]
                last["resident_passes"] = [len(res), len(passes)]
                last["tokens_stable"] = all(p["tokens"] == passes[0]["tokens"] for p in passes)
                if res:
                    dec = sorted(p["decode_tok_s"] for p in res); tt = sorted(p["ttft_s"] or 0 for p in res)
                    last["decode_tok_s"] = dec[len(dec) // 2]; last["ttft_s"] = tt[len(tt) // 2]
                else:
                    last["decode_tok_s"] = None
                last["experts"] = {k: sum(p["experts"][k] for p in passes) for k in passes[0]["experts"]}
            replies[pid] = last
            _write(d / "replies.json", replies)
        _write(d / "status-after.json", _get(base + "/v1/status"))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM); proc.wait(90)
        except Exception:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except Exception: pass
    _write(d / "done.json", {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return replies


def summarize(out, arm_names, oracle_table):
    base_name = arm_names[0]
    base = json.loads((out / base_name / "replies.json").read_text())
    rows = []
    for name in arm_names:
        f = out / name / "replies.json"
        if not (out / name / "done.json").exists():
            continue
        arm = json.loads(f.read_text())
        per, misses, wait, toks, dec, ttft, res_ok, res_all = {}, 0, 0, 0, [], [], 0, 0
        for pid, r in arm.items():
            desc, fn = oracle_table.get(pid, ("no oracle", lambda _r: True))
            ok, bok = bool(fn(r["reply"])), bool(fn(base[pid]["reply"]))
            truncated = base[pid].get("finish") == "length" and not bok
            if truncated:
                # The baseline itself ran out of tokens before answering: the oracle cannot
                # judge this prompt, so compare tokens and margins only.
                ok = bok = True
                desc += " (n/a: baseline hit max_tokens)"
            c = compare(base[pid]["tokens"], r["tokens"], bok, ok)
            # Behaviour drift the content oracles miss (2026-11-08: keep-top-1 kept facts but
            # stopped stopping): the baseline stopped and the arm ran to max_tokens, or the arm
            # is more than twice as long.
            bt, at = base[pid].get("completion_tokens") or 0, r.get("completion_tokens") or 0
            runaway = (base[pid].get("finish") == "stop" and (r.get("finish") == "length" or at > 2 * bt + 8))
            c.update(oracle=desc, oracle_pass=None if truncated else ok, decode_tok_s=r["decode_tok_s"],
                     ttft_s=r.get("ttft_s"), resident_passes=r.get("resident_passes"), tokens_stable=r.get("tokens_stable"),
                     runaway=bool(runaway), tokens_vs_base=[at, bt])
            per[pid] = c
            misses += r["experts"]["v41_expert_cache_misses"]; wait += r["experts"]["v41_expert_demand_wait_ns"]
            toks += r.get("completion_tokens") or 0
            if r["decode_tok_s"] is not None:
                dec.append(r["decode_tok_s"])
            if r.get("resident_passes"):
                res_ok += r["resident_passes"][0]; res_all += r["resident_passes"][1]
            if r.get("ttft_s") is not None:
                ttft.append(r["ttft_s"])
        classes = [v["class"] for v in per.values()]
        rows.append({"arm": name, "per_prompt": per,
                     "identical": classes.count("identical"), "near_tie": classes.count("near_tie_equivalent"),
                     "diverged_pass": classes.count("diverged_pass"), "diverged_fail": classes.count("diverged_fail"),
                     "failed": classes.count("failed"), "runaway": sum(1 for v in per.values() if v.get("runaway")), "oracle_pass": sum(1 for v in per.values() if v["oracle_pass"]),
                     "prompts": sum(1 for v in per.values() if v["oracle_pass"] is not None), "expert_misses": misses, "demand_wait_s": round(wait / 1e9, 1),
                     "completion_tokens": toks, "median_decode_tok_s": sorted(dec)[len(dec) // 2] if dec else None,
                     "median_ttft_s": sorted(ttft)[len(ttft) // 2] if ttft else None,
                     "resident_passes": [res_ok, res_all] if res_all else None})
    _write(out / "summary.json", rows)
    warm = any(r.get("resident_passes") for r in rows)
    head = "| arm | identical | near-tie | diverged (pass) | diverged (FAIL) | runaway | oracle pass | misses | wait s | decode tok/s |"
    if warm:
        head += " TTFT s | resident passes |"
    lines = [head, "|" + "---|" * (12 if warm else 10)]
    for r in rows:
        line = (f"| {r['arm']} | {r['identical']} | {r['near_tie']} | {r['diverged_pass']} | {r['diverged_fail'] + r['failed']} | {r['runaway']} "
                f"| {r['oracle_pass']}/{r['prompts']} | {r['expert_misses']} | {r['demand_wait_s']} | {r['median_decode_tok_s']} |")
        if warm:
            rp = r.get("resident_passes") or [0, 0]
            line += f" {r.get('median_ttft_s')} | {rp[0]}/{rp[1]} |"
        lines.append(line)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--port", type=int, default=8296)
    ap.add_argument("--dry-run", action="store_true", help="validate every arm policy on CPU and exit")
    a = ap.parse_args()
    spec = json.loads(a.ladder.read_text())
    base, arms = spec.get("base", {}), expand(spec)
    names = [x["name"] for x in arms]
    if len(set(names)) != len(names) or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", n) for n in names):
        raise SystemExit("arm names must be unique and filesystem-safe")
    prompts, gsm = load_prompts(spec.get("prompts", DEFAULT_PROMPTS))
    table = oracles(gsm)
    policies = {x["name"]: merged(base, x.get("policy", {})) for x in arms}
    sys.path.insert(0, str(Path.cwd() / "src"))
    from mlx2.adapters.deepseek_v41_source import source_policy
    for n, p in policies.items():
        source_policy(copy.deepcopy(p))             # fail closed before any GPU time
        print(f"ok {n}: {json.dumps(p, sort_keys=True)[:200]}")
    if a.dry_run:
        return
    a.out.mkdir(parents=True, exist_ok=True)
    _write(a.out / "ladder.json", {"spec": spec, "prompts": prompts, "gsm8k_references": gsm,
                                   "oracles": {k: v[0] for k, v in table.items()}})
    for n in names:
        d = a.out / n
        if (d / "done.json").exists():
            print(f"skip {n} (done)"); continue
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"refusing partial arm directory {d}; move it aside to retry")
        run_arm(n, policies[n], d, prompts, spec.get("max_tokens", 192), a.model, a.port,
                warmup=int(spec.get("warmup", 0)), repeat=int(spec.get("repeat", 1)))
        summarize(a.out, names, table)
    summarize(a.out, names, table)


if __name__ == "__main__":
    main()
