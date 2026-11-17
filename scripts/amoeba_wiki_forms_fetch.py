#!/usr/bin/env python3
"""Fetch natural text for form nodes from two Wikimedia sites (CC BY-SA 4.0), approved by Pierre in chat 2026-11-16.

  wikibooks  en.wikibooks.org chapter pages (titles with '/'), first section -> form.textbook
  simple     simple.wikipedia.org article leads -> form.audience.children (written for children and learners)

Titles come from list=allpages in alphabetical slices (deterministic, spread over the alphabet), then
prop=extracts (exintro, plain text) + info (lastrevid), 20 per call. Same pacing and Retry-After handling
as amoeba_wiki_fetch.py. Writes OUT/<job>.jsonl {node, site, title, pageid, revid, text} and provenance.json.
"""
from __future__ import annotations

import argparse
import json
import string
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))      # our own scripts dir (python -I drops it)
import amoeba_wiki_fetch as W  # noqa: E402

JOBS = {"wikibooks": ("https://en.wikibooks.org/w/api.php", "form.textbook", True),
        "simple": ("https://simple.wikipedia.org/w/api.php", "form.audience.children", False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-job", type=int, default=320)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prov = {"license": "CC BY-SA 4.0; attribution: site, title and revision per row",
            "approved": "Pierre 2026-11-16 in chat ('Yes, both (Recommended)')", "jobs": {}}
    for job, (api, node, chapters) in JOBS.items():
        f = out / f"{job}.jsonl"
        if f.exists():
            continue
        W.API = api
        rows, per_letter = [], max(4, args.per_job // 20)
        for letter in string.ascii_uppercase:
            d = W.call({"action": "query", "list": "allpages", "apnamespace": 0, "apfilterredir": "nonredirects",
                        "apfrom": letter, "aplimit": 500})
            titles = [p["title"] for p in d.get("query", {}).get("allpages", []) if (("/" in p["title"]) == chapters)]
            titles = titles[::max(1, len(titles) // (2 * per_letter))][:2 * per_letter]
            got = 0
            for i in range(0, len(titles), 20):
                d = W.call({"action": "query", "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": 20,
                            "titles": "|".join(titles[i:i + 20])})
                for p in d.get("query", {}).get("pages", []):
                    t = (p.get("extract") or "").strip()
                    if len(t) >= 300 and got < per_letter:
                        rows.append({"node": node, "site": api.split("/")[2], "title": p["title"], "pageid": p["pageid"],
                                     "revid": p.get("lastrevid"), "text": t})
                        got += 1
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        prov["jobs"][job] = {"api": api, "node": node, "rows": len(rows)}
        print(f"{time.strftime('%H:%M:%S')} {job}: {len(rows)} rows", flush=True)
    prov["failed_requests"] = W.FAILED
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
