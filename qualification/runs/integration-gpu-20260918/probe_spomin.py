"""Live Spomin compaction: applied, coherent, never published, batches with exact lanes."""
import uuid
from probe_common import *

policy = status()["settings"]["spomin_live_surgery"]
capacity = policy["capacity_tokens"]
counts0 = dict(status()["counts"])


def long_prompt():
    nonce = uuid.uuid4().hex
    head = f"Session {nonce}. Remember: the vault code is ALPHA-7319.\n"
    rows = int(capacity * 0.8 / 17)
    filler = "".join(f"Log entry {i}: sensor {i % 97} reported a nominal reading of {i * 7 % 1000} units.\n" for i in range(rows))
    tail = "Final note: the meeting city is Lisbon.\nQuestion: which city is the meeting in? Answer with the single word."
    return head + filler + tail


def surgery(body):
    return body.get("mlx2", {}).get("spomin_live_surgery") or {}

request = chat(long_prompt(), max_tokens=96)
runs = [post(request) for _ in range(2)]
receipts = [surgery(b) for _c, b, _e in runs]
record("long_prompt_compacted",
       all(c == 200 for c, _b, _e in runs)
       and all(r.get("status") == "applied" and r.get("retained_tokens", 0) < r.get("source_tokens", 0) for r in receipts)
       and all("lisbon" in text(b).lower() for _c, b, _e in runs),
       {"codes": [c for c, _b, _e in runs], "receipts": receipts, "answers": [text(b) for _c, b, _e in runs],
        "prompt_tokens": [b.get("usage", {}).get("prompt_tokens") for _c, b, _e in runs],
        "seconds": [round(e, 1) for _c, _b, e in runs]})
record("compacted_state_never_published",
       all(b.get("mlx2", {}).get("cached_tokens") == 0 for _c, b, _e in runs)
       and status()["counts"].get("apcv2_store_skipped_approximate", 0) - counts0.get("apcv2_store_skipped_approximate", 0) >= 2,
       {"cached_tokens": [b.get("mlx2", {}).get("cached_tokens") for _c, b, _e in runs],
        "skipped": status()["counts"].get("apcv2_store_skipped_approximate", 0)})

code, body, _ = post(chat(long_prompt().replace("which city is the meeting in", "what is the vault code"), max_tokens=16))
record("protected_prefix_recall", code == 200 and surgery(body).get("status") == "applied",
       {"answer": text(body), "recalled": "7319" in text(body), "receipt": surgery(body).get("status")})

short = chat("Reply with exactly: SHORT_OK", max_tokens=8)
first, second = post(short), post(short)
record("short_prompt_stays_exact",
       first[0] == second[0] == 200 and surgery(first[1]).get("reason") == "below_pressure"
       and second[1].get("mlx2", {}).get("cached_tokens", 0) > 0 and "SHORT_OK" in text(second[1]),
       {"receipt": surgery(first[1]), "warm_cached_tokens": second[1].get("mlx2", {}).get("cached_tokens"), "answer": text(second[1])})

mixed = parallel([chat(long_prompt(), max_tokens=96), chat("Count from one to ten in words.", max_tokens=40),
                  chat("Name three primary colors.", max_tokens=24)])
record("compacted_lane_batches_with_exact_lanes",
       all(c == 200 for c, _b, _e in mixed) and surgery(mixed[0][1]).get("status") == "applied"
       and "lisbon" in text(mixed[0][1]).lower() and "ten" in text(mixed[1][1]).lower(),
       {"codes": [c for c, _b, _e in mixed], "answers": [text(b)[:60] for _c, b, _e in mixed],
        "widths": [b.get("mlx2", {}).get("ordinary_compute_width") for _c, b, _e in mixed],
        "receipt": surgery(mixed[0][1]).get("status"), "errors": [b.get("error") for _c, b, _e in mixed]})
snapshot = status().get("spomin_live_surgery", {})
record("manager_status", snapshot.get("enabled") and snapshot.get("counts", {}).get("applied", 0) >= 4 and snapshot.get("active_epochs") == 0,
       {"counts": snapshot.get("counts"), "active_epochs": snapshot.get("active_epochs")})
finish()
