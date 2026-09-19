"""Spomin: the exact pre-surgery boundary is published; the compacted state is not."""
import uuid
from probe_common import *

policy = status()["settings"]["spomin_live_surgery"]
capacity = policy["capacity_tokens"]
counts0 = dict(status()["counts"])
nonce = uuid.uuid4().hex
rows = int(capacity * 0.8 / 17)
filler = "".join(f"Log entry {i}: sensor {i % 97} reported a nominal reading of {i * 7 % 1000} units.\n" for i in range(rows))
prompt_text = f"Session {nonce}. Remember: the vault code is ALPHA-7319.\n" + filler + "Final note: the meeting city is Lisbon.\nQuestion: which city is the meeting in? Answer with the single word."
request = chat(prompt_text, max_tokens=96)


def surgery(body):
    return body.get("mlx2", {}).get("spomin_live_surgery") or {}

runs = [post(request) for _ in range(3)]
receipts = [surgery(b) for _c, b, _e in runs]
cached = [b.get("mlx2", {}).get("cached_tokens") for _c, b, _e in runs]
record("every_run_compacted_and_correct",
       all(c == 200 for c, _b, _e in runs) and all(r.get("status") == "applied" for r in receipts)
       and all("lisbon" in text(b).lower() for _c, b, _e in runs),
       {"answers": [text(b)[:40] for _c, b, _e in runs], "receipts": [(r.get("source_tokens"), r.get("retained_tokens")) for r in receipts]})
record("repeat_warm_hits_the_exact_boundary",
       cached[0] < 256 and all(c > receipts[0]["retained_tokens"] and c >= receipts[0]["source_tokens"] - 2 for c in cached[1:]),
       {"cached_tokens": cached, "source_tokens": receipts[0].get("source_tokens"), "retained_tokens": receipts[0].get("retained_tokens"),
        "seconds": [round(e, 2) for _c, _b, e in runs], "ttft": [b.get("mlx2", {}).get("ttft_seconds") for _c, b, _e in runs]})
record("identical_answers_cold_and_warm", len({text(b) for _c, b, _e in runs}) == 1, {"answers": [text(b) for _c, b, _e in runs]})
skipped = status()["counts"].get("apcv2_store_skipped_approximate", 0) - counts0.get("apcv2_store_skipped_approximate", 0)
record("compacted_end_state_still_not_published", skipped == 3, {"skipped": skipped})
# A different question changes the tail, and APCv2 reuses whole stored boundaries (sliding-window rings
# cannot be rewound to an interior position), so this is a cold, correct, compacted request: informational.
other = post(chat(prompt_text.replace("which city is the meeting in", "what is the vault code"), max_tokens=96))
record("different_question_same_context", other[0] == 200 and "7319" in text(other[1]) and surgery(other[1]).get("status") == "applied",
       {"answer": text(other[1])[:60], "cached_tokens": other[1]["mlx2"]["cached_tokens"]})
finish()
