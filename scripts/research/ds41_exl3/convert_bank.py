#!/usr/bin/env python3
"""Convert a targeted DS4.1 expert set to EXL3 (Pierre 2026-11-09: "Hot set, 3 bpw").

Selection: the top-N experts per layer by routing count in the train-split usage map
(usage-train-split.json, the same map the reduced tiers use). Calibration and the
per-expert check are those of convert_experts.py (Hessians from rows that selected the
expert, mixed with a 4096-row layer sample below 2048 rows; 64 held-out evaluation rows;
decode of the SAVED file through mlx2's reference EXL3 decoder). Resumable: an expert whose
file and result row already exist is skipped. Writes manifest.json with per-file sha256.

usage: convert_bank.py [--top 50 | --experts FILE.json] [--words 48] [--layers 0-39] [--out DIR] [--inflight 2]

--experts: {"<layer>": [expert, ...]} (e.g. from capture_experts.py); converted in that order,
most-used first per layer, overriding --top.
"""
import argparse, hashlib, json, math, os, socket, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_experts as C  # noqa: E402  (sets paths, CPU default device)
import exl3_encode as E  # noqa: E402
from viterbi_metal import MX_LOCK, v2_supported  # noqa: E402

USAGE = C.CAP / "usage-train-split.json"


def prepare_by_id(art, ids, rng, layer, expert, rank, uses):
    X = np.load(C.CAP / "x_by_layer" / f"L{layer:02d}.npy", mmap_mode="r")
    sel = np.where((ids[:, layer] == expert).any(-1))[0]
    ev, cal = sel[:C.EVAL_ROWS], sel[C.EVAL_ROWS:]
    if len(sel) < 2 * C.EVAL_ROWS:
        # Rare expert (few or no selecting rows in the trace): split what there is, and take the
        # evaluation rows from the layer sample when it has none (eval_source records which).
        ev, cal = sel[: len(sel) // 2], sel[len(sel) // 2:]
    ev_source = "selected"
    if len(ev) == 0:
        ev = np.sort(rng.choice(len(X), size=C.EVAL_ROWS, replace=False)); ev_source = "layer_sample"
    raw = C.read_tensors(art, C.expert_names(layer, expert))
    W = {k: C.decode_mxfp4(raw[f"layers.{layer}.ffn.experts.{expert}.{p}.weight"],
                           raw[f"layers.{layer}.ffn.experts.{expert}.{p}.scale"]) for k, p in C.PROJECTIONS.items()}
    with MX_LOCK:
        xe = C.fp8(X[ev].astype(np.float32))
        if len(cal):
            xc = C.fp8(X[cal].astype(np.float32)); mid = C.swiglu_mid(xc, W["gate"], W["up"])
    calib = {"selected_rows": int(len(cal)), "mixed_with_layer_sample": False, "eval_source": ev_source}
    if len(cal):
        h_in, h_mid = C.gram(xc), C.gram(mid)
    if len(cal) < C.MIN_CALIB:
        pool = np.setdiff1d(np.arange(len(X)), ev)
        pool = np.sort(rng.choice(pool, size=min(C.ALL_SAMPLE, len(pool)), replace=False))
        with MX_LOCK:
            xa = C.fp8(X[pool].astype(np.float32))
            mida = C.swiglu_mid(xa, W["gate"], W["up"])
        if len(cal):
            h_in, h_mid = 0.5 * h_in + 0.5 * C.gram(xa), 0.5 * h_mid + 0.5 * C.gram(mida)
        else:
            h_in, h_mid = C.gram(xa), C.gram(mida)
        calib.update(mixed_with_layer_sample=True, layer_sample_rows=int(len(pool)))
    with MX_LOCK:
        ref = C.expert_out(xe, W["gate"], W["up"], W["down"])
    return {"layer": layer, "expert": expert, "pick": f"rank{rank}", "uses": uses, "eval_rows": int(len(ev)),
            "calib": calib, "W": W, "h_in": h_in, "h_mid": h_mid, "xe": xe, "ref": ref}


def checked(row):
    """Sweep 2026-11-09 EXL-1: a row is admitted only when its recorded check passed (the saved file
    decodes to the encoder's weight, and the whole-expert error is finite)."""
    err = row.get("expert_rel_err")
    return (row.get("decode_matches_encoder") is True and type(err) in (int, float) and math.isfinite(err))


def done_rows(out, words, rejected=None):
    """Checked rows at this bit rate from every results*.json (one file per converting process),
    keyed (layer, expert, words) (sweep 2026-11-09 EXL-2). Rows whose check failed are not done
    (EXL-1); their keys are appended to ``rejected`` when given. Each row carries ``_results``,
    the results file it came from."""
    rows = {}
    for f in sorted(out.glob("results*.json")):
        for r in json.loads(f.read_text()):
            if r.get("words") != words or not (out / r["file"]).exists():
                continue
            key = (r["layer"], r["expert"], words)
            if not checked(r):
                if rejected is not None:
                    rejected.append(key)
                continue
            rows[key] = dict(r, _results=f.name)
    return rows


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:   # sweep 2026-11-09 EXL-7: a live pid owned by another user
        return True
    return True


def _claim_name(layer, expert, words):
    # Keyed by bit rate like done_rows (review 2026-11-09): a w32 claim never blocks a w48 run.
    return f"{layer}.{expert}.w{words}"


def _take_lock(lock, me):
    """O_EXCL takeover lock holding "host pid". A lock left by a dead local process (killed
    between taking and removing it) is reclaimed: it is moved aside atomically, and only the mover
    that finds the dead holder's content removed it; the O_EXCL retry then admits one taker."""
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, me.encode()); os.close(fd)
            return True
        except FileExistsError:
            pass
        try:
            held = lock.read_text()
            host, pid = held.split(); pid = int(pid)
        except (ValueError, FileNotFoundError):
            return False
        if host != socket.gethostname() or pid == os.getpid() or _alive(pid):
            return False
        aside = lock.with_name(f".{lock.name}.{os.getpid()}.{threading.get_ident()}")
        try:
            os.rename(lock, aside)
        except FileNotFoundError:
            continue
        if aside.read_text() != held:   # moved a live lock taken meanwhile: put it back
            try:
                os.link(aside, lock)
            except FileExistsError:
                pass
            aside.unlink(missing_ok=True)
            return False
        aside.unlink(missing_ok=True)
    return False


def claim(out, layer, expert, skipped=None, *, words):
    """Atomic per-expert claim so several processes (M5 local, M3 remote) never encode the same one.
    A claim left by a dead local process is taken over; a claim from another host is never taken
    over and is appended to ``skipped`` (reported at the end of the run)."""
    d = out / "claims"; d.mkdir(exist_ok=True)
    path = d / _claim_name(layer, expert, words)
    me = f"{socket.gethostname()} {os.getpid()}"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, me.encode()); os.close(fd)
        return True
    except FileExistsError:
        pass
    try:
        stale = path.read_text()
        host, pid = stale.split(); pid = int(pid)
    except (ValueError, FileNotFoundError):
        return False
    if host != socket.gethostname():
        if skipped is not None:
            skipped.append((layer, expert, stale))
        return False
    if pid == os.getpid() or _alive(pid):
        return False
    # Takeover (sweep 2026-11-09 EXL-7): the claim file never disappears, so no O_EXCL creator can
    # slip in; a takeover lock admits one taker, which re-reads the claim and swaps it atomically.
    lock = d / f"{path.name}.takeover"
    if not _take_lock(lock, me):
        if skipped is not None:
            skipped.append((layer, expert, "takeover lock held"))
        return False
    try:
        try:
            if path.read_text() != stale:
                return False
        except FileNotFoundError:
            return False
        tmp = d / f".{path.name}.{os.getpid()}.{threading.get_ident()}"
        tmp.write_text(me); os.replace(tmp, path)
        return True
    finally:
        lock.unlink(missing_ok=True)


def unconverted(out, plan, words, skipped=()):
    """Plan entries with no checked row at the end of a run, printed with their claim holders."""
    done = done_rows(out, words)
    left = [p for p in plan if (p[0], p[1], words) not in done]
    if left:
        holders = {(l, e): why for l, e, why in skipped}
        print(f"{len(left)} plan entries not converted:", flush=True)
        for p in left:
            c = out / "claims" / _claim_name(p[0], p[1], words)
            why = holders.get((p[0], p[1])) or (c.read_text() if c.exists() else "unclaimed")
            print(f"  layer {p[0]} expert {p[1]}: {why}", flush=True)
    return left


def write_manifest(out, words):
    rejected = []
    rows = done_rows(out, words, rejected)
    if rejected:
        print("manifest: rejected rows whose check failed:", sorted(set(rejected)), flush=True)
    files = sorted({out / r["file"] for r in rows.values()})
    # Sweep 2026-11-09 EXL-3: the selection is what the bank holds, from the rows themselves (rows
    # written before per-row selection was recorded are named by their results file).
    picks = {}
    for r in rows.values():
        s = r.get("selection") or f"unrecorded selection ({r['_results']})"
        picks[s] = picks.get(s, 0) + 1
    manifest = {"source": "DeepSeek-V4.1-Flash-1017-Source@c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb",
                "format": "exl3 (upstream exllamav3 tensor layout, mcg codebook)", "bits_per_weight": words / 16,
                "selection": "; ".join(f"{s}: {n} rows" for s, n in sorted(picks.items())),
                "files": {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    tmp = out / "manifest.json.partial"; tmp.write_text(json.dumps(manifest, indent=1)); os.replace(tmp, out / "manifest.json")
    print("manifest", manifest["fingerprint"], len(files), "files", flush=True)


CONNECT_S, ENCODE_S = 30, 600


def remote_encode(addr, ctx, words, probe, lean=False, inflight=1):
    from exl3_worker import load_token, recv_msg, send_auth, send_msg
    W = ctx["W"]
    msg = {"w0": np.ascontiguousarray(W["gate"].T, np.float32), "w1": np.ascontiguousarray(W["up"].T, np.float32),
           "w2": np.ascontiguousarray(W["down"].T, np.float32), "h_in": ctx["h_in"], "h_mid": ctx["h_mid"],
           "meta": np.array(json.dumps({"words": words, "seed": ctx["layer"] * 1000 + ctx["expert"], "probe": probe,
                                        "return_weight": not lean}))}
    host, port = addr.split(":")
    # Sweep 2026-11-09 EXL-8: the connect/send timeout is separate from the reply wait, which covers
    # this request queueing behind up to inflight-1 others on the worker's GPU lock plus its own encode.
    with socket.create_connection((host, int(port)), timeout=CONNECT_S) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(ENCODE_S)
        send_auth(sock, load_token()); send_msg(sock, msg)
        sock.settimeout(ENCODE_S * max(1, inflight))
        reply = recv_msg(sock)
    if reply is None or "meta" not in reply:
        raise ConnectionError(f"exl3 worker {addr} closed the connection without a reply (wrong token?)")
    meta = json.loads(str(reply["meta"]))
    if not meta.get("ok"):
        raise RuntimeError(f"remote encode failed: {meta.get('error')}")
    if lean:
        # The worker checked decode == encoder itself; the local check then decodes the saved file
        # against this same decode (a file round-trip only), so the worker's result is the check.
        from mlx2.runtime.models.exl3_reference import effective_weight
        m = meta.get("remote_decode_match")
        if not (isinstance(m, list) and len(m) == 3 and all(v is True for v in m)):   # sweep 2026-11-09 EXL-4
            raise RuntimeError(f"remote decode check failed or missing: {m!r}")
        weights = [effective_weight(np.asarray(reply[f"trellis{i}"]).view(np.uint16), reply[f"suh{i}"], reply[f"svh{i}"], "mcg")
                   for i in range(3)]
    else:
        weights = [reply[f"weight{i}"] for i in range(3)]
    enc = [{"trellis": reply[f"trellis{i}"], "suh": reply[f"suh{i}"], "svh": reply[f"svh{i}"],
            "weight": weights[i], "stats": meta["stats"][i]} for i in range(3)]
    if lean:
        enc[0]["stats"]["remote_decode_match"] = meta["remote_decode_match"]
    enc[0]["stats"]["remote"] = {k: meta[k] for k in ("host", "wait_s", "encode_s")}
    return enc


def lean_row(row, enc):
    """--remote --lean (sweep 2026-11-09 EXL-4): the local compare decodes the saved file against the
    client's own decode of the reply, so it only proves the file round-trip; decode_matches_encoder
    is the worker's decode-vs-encoder check, and both are recorded."""
    m = enc[0]["stats"].get("remote_decode_match")
    row["saved_file_matches_reply"] = row["decode_matches_encoder"]
    row["remote_decode_match"] = m
    row["decode_matches_encoder"] = bool(row["decode_matches_encoder"] and isinstance(m, list) and len(m) == 3
                                         and all(v is True for v in m))
    row["decode_check"] = "worker (lean)"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--experts", help="explicit {layer: [experts]} JSON; overrides --top")
    ap.add_argument("--words", type=int, default=48)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--probe", type=int, default=32)
    ap.add_argument("--remote", help="host:port of an exl3_worker.py; this process preps locally and encodes there "
                         "(sends the shared token from MLX2_EXL3_WORKER_TOKEN or MLX2_EXL3_WORKER_TOKEN_FILE)")
    ap.add_argument("--reverse", action="store_true", help="walk the plan from its end (pairs with a forward process)")
    ap.add_argument("--name", default="", help="results file suffix for this process (results<name>.json)")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--lean", action="store_true", help="remote: worker checks its decode; no weight shipped back (slow links)")
    ap.add_argument("--out", default=str(Path.home() / "mlx-models/DeepSeek-V4.1-Flash-1017-Source-Expert-EXL3-3bpw-hot50"))
    a = ap.parse_args()
    if not v2_supported(a.words):
        raise SystemExit("streaming v2 kernel does not support this rate")
    lo, hi = map(int, a.layers.split("-"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(json.loads(USAGE.read_text())["counts"])
    if a.experts:
        want = {int(k): [int(e) for e in v] for k, v in json.loads(Path(a.experts).read_text()).items()}
        plan = [(l, e, r, int(counts[l, e])) for l in sorted(want) if lo <= l <= hi
                for r, e in enumerate(sorted(set(want[l]), key=lambda e: -counts[l, e]))]
        selection = f"explicit set {a.experts}, layers {lo}-{hi}"
    else:
        plan = [(l, int(e), r, int(counts[l, e])) for l in range(lo, hi + 1)
                for r, e in enumerate(np.argsort(-counts[l], kind="stable")[:a.top])]
        selection = f"top {a.top} per layer by usage-train-split counts, layers {lo}-{hi}"
    if a.manifest_only:
        write_manifest(out, a.words); return
    if a.reverse:
        plan = plan[::-1]
    res_path = out / f"results{a.name}.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = set(done_rows(out, a.words))
    todo = [p for p in plan if (p[0], p[1], a.words) not in done]
    skipped = []
    print(f"{len(plan)} planned, {len(done)} done anywhere, {len(todo)} to encode"
          + (f" via {a.remote}" if a.remote else ""), flush=True)
    art = C.inspect_artifact(C.MODEL)
    ids = np.load(C.TRACE)["ids"]
    args = argparse.Namespace(backend="metal" if not a.remote else f"metal@{a.remote}", probe=a.probe, no_ldlq=False)
    lock = threading.Lock()

    def emit(row):
        row["selection"] = selection   # sweep 2026-11-09 EXL-3: per-row provenance for the manifest
        with lock:
            results.append(row)
            print(json.dumps({k: row[k] for k in ("layer", "expert", "pick", "expert_rel_err", "encode_s")}), flush=True)
            tmp = res_path.with_suffix(".partial"); tmp.write_text(json.dumps(results, indent=1)); os.replace(tmp, res_path)

    def take():
        """Next unclaimed, unfinished plan entry (evaluated lazily, so the two processes interleave)."""
        for p in todo:
            if claim(out, p[0], p[1], skipped, words=a.words):
                yield p

    def rng_for(layer, expert):
        return np.random.default_rng(layer * 1000 + expert)

    if a.remote:
        def one(p):
            ctx = prepare_by_id(art, ids, rng_for(p[0], p[1]), *p)
            t0 = time.time(); enc = remote_encode(a.remote, ctx, a.words, a.probe, lean=a.lean, inflight=a.inflight)
            emit(lean_row(C.save_and_check(ctx, a.words, enc, out, args, time.time() - t0), enc) if a.lean
                 else C.save_and_check(ctx, a.words, enc, out, args, time.time() - t0))
        gen, glock, errors = take(), threading.Lock(), []

        def worker():
            # Each thread claims its next expert only when it is free, so claims stay a few ahead at most.
            while not errors:
                with glock:
                    p = next(gen, None)
                if p is None:
                    return
                try:
                    one(p)
                except Exception as exc:  # keep the claim's file unwritten; a rerun redoes it
                    errors.append(exc)
        threads = [threading.Thread(target=worker) for _ in range(a.inflight)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
    else:
        ctxs, post, futs = [], ThreadPoolExecutor(1), []

        def loader(layer, expert, rank, uses):
            def load():
                ctx = prepare_by_id(art, ids, rng_for(layer, expert), layer, expert, rank, uses)
                ctxs.append(ctx); W = ctx["W"]
                return [(W["gate"].T, ctx["h_in"]), (W["up"].T, ctx["h_in"]), (W["down"].T, ctx["h_mid"])], \
                    layer * 1000 + expert, a.words
            return load

        t_prev = time.time()
        for n, enc in enumerate(E.encode_experts((loader(*p) for p in take()), a.words, inflight=a.inflight, probe=a.probe)):
            now = time.time(); ctx = ctxs[n]; ctxs[n] = None
            futs.append(post.submit(C.save_and_check, ctx, a.words, enc, out, args, now - t_prev)); t_prev = now
            while futs and futs[0].done():
                emit(futs.pop(0).result())
        for f in futs:
            emit(f.result())
        post.shutdown()
    unconverted(out, plan, a.words, skipped)
    write_manifest(out, a.words)


if __name__ == "__main__":
    main()
