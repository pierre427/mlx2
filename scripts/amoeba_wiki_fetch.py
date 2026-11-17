#!/usr/bin/env python3
"""Fetch English Wikipedia article leads per taxonomy node from category membership (MediaWiki API).

Input: docs/research/amoeba-topics/sources/wikipedia-categories.json (node -> categories). For each node:
members (namespace 0) of each category and of its direct subcategories, up to --per-node articles; then the
plain-text lead (TextExtracts, exintro) and the current revision id of each article, 20 per request.
Writes OUT/<node>.jsonl rows {node, title, pageid, revid, category, text} and OUT/provenance.json. Licence:
CC BY-SA 4.0 (text), attribution = article title + revision URL. Approved by Pierre in chat, 2026-11-16.
Polite: one request at a time, maxlag=5, a generic User-Agent, retries with backoff.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "mlx2-research-probe-builder/1.0 (offline research corpus; low rate)"
PACE = 2.0                      # seconds between calls


FAILED = []


def call(params, tries=8):
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2, "maxlag": 5})
    last = None
    for k in range(tries):
        try:
            rq = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
            with urllib.request.urlopen(rq, timeout=60) as r:
                d = json.loads(r.read())
            if "error" in d and d["error"].get("code") == "maxlag":
                raise RuntimeError("maxlag")
            time.sleep(PACE)
            return d
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            wait = int(e.headers.get("Retry-After") or 0) or min(300, 15 * 2 ** k)
            print(f"  {last}; waiting {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            last = repr(e)
            time.sleep(min(120, 5 * 2 ** k))
    FAILED.append({"params": params, "error": last})
    print(f"  gave up on {params.get('gcmtitle') or params.get('cmtitle')}: {last}", flush=True)
    return {}


def leads(cat, limit):
    """Article leads (namespace 0) in one category: generator=categorymembers + extracts, 20 per call."""
    rows, cont = [], {}
    while len(rows) < limit:
        d = call({"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{cat}", "gcmnamespace": 0,
                  "gcmlimit": 20, "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": 20, **cont})
        for p in d.get("query", {}).get("pages", []):
            t = (p.get("extract") or "").strip()
            if len(t) >= 200 and not p["title"].startswith(("List of", "Outline of", "Index of", "Glossary of")):
                rows.append({"title": p["title"], "pageid": p["pageid"], "revid": p.get("lastrevid"), "category": cat, "text": t})
        if "continue" not in d:
            break
        cont = {k: v for k, v in d["continue"].items()}
    return rows


def subcats(cat, limit=30):
    d = call({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}", "cmnamespace": 14, "cmlimit": limit})
    return [m["title"].split(":", 1)[1] for m in d.get("query", {}).get("categorymembers", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(Path(__file__).resolve().parents[1] /
                                         "docs/research/amoeba-topics/sources/wikipedia-categories.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-node", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmap = json.loads(Path(args.map).read_text())["map"]
    prov = {"source": "en.wikipedia.org MediaWiki API (categorymembers + extracts exintro + info lastrevid)",
            "license": "CC BY-SA 4.0 (Wikipedia text); attribution: title and revision URL per row",
            "approved": "Pierre 2026-11-16 in chat ('All three (Recommended)')", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "nodes": {}}
    for node, cats in cmap.items():
        f = out / f"{node}.jsonl"
        if f.exists():
            continue
        rows, seen = [], set()

        def take(rs):
            for r in rs:
                if r["title"] not in seen:
                    seen.add(r["title"])
                    rows.append({"node": node, **r})
        for c in cats:
            take(leads(c, args.per_node))
        for c in cats:                              # subcategories only while the node is short
            if len(rows) >= args.per_node:
                break
            for sub in subcats(c):
                if len(rows) >= args.per_node:
                    break
                take(leads(sub, 60))
        rows = rows[:args.per_node]
        titles = rows
        tmp = f.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.rename(f)
        prov["nodes"][node] = {"categories": cats, "titles": len(titles), "rows": len(rows)}
        print(f"{time.strftime('%H:%M:%S')} {node}: {len(titles)} titles, {len(rows)} leads", flush=True)
    prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["failed_requests"] = FAILED
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
