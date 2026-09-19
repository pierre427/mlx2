"""Approximate KV quantization: applied, coherent, isolated from the exact prefix cache."""
import uuid
from probe_common import *

policy = status()["settings"]["approximate_kv"]
st0 = status()


def receipt(body):
    return body.get("mlx2", {}).get("approximate_kv") or {}

code, body, _ = post(chat("What is the capital of France? Answer with one word."))
record("short_answer", code == 200 and receipt(body).get("status") == "applied" and "paris" in text(body).lower(),
       {"code": code, "answer": text(body), "receipt": receipt(body), "error": body.get("error")})

rows = "".join(f"Ledger line {i}: account {i * 13 % 500} moved {i * 17 % 900} credits.\n" for i in range(260))
needle = f"Session {uuid.uuid4().hex}.\n" + rows[: len(rows) // 2] + "Important: the launch password is ORCHID-5521.\n" + rows[len(rows) // 2:] + "What is the launch password? Answer with the password only."
runs = [post(chat(needle, max_tokens=16)) for _ in range(2)]
record("needle_recall_through_quantized_kv",
       all(c == 200 and "5521" in text(b) and receipt(b).get("status") == "applied" for c, b, _e in runs)
       and text(runs[0][1]) == text(runs[1][1]),
       {"answers": [text(b) for _c, b, _e in runs], "prompt_tokens": runs[0][1].get("usage", {}).get("prompt_tokens"),
        "receipt": receipt(runs[0][1])})
st1 = status()
record("approximate_state_never_published",
       all(b.get("mlx2", {}).get("cached_tokens") == 0 for _c, b, _e in runs)
       and st1["counts"].get("apcv2_store_skipped_approximate", 0) > st0["counts"].get("apcv2_store_skipped_approximate", 0)
       and st1["apcv2"].get("nbytes", 0) == st0["apcv2"].get("nbytes", 0),
       {"cached_tokens": [b.get("mlx2", {}).get("cached_tokens") for _c, b, _e in runs],
        "skipped": st1["counts"].get("apcv2_store_skipped_approximate"), "apc_nbytes": [st0["apcv2"].get("nbytes"), st1["apcv2"].get("nbytes")]})
mixed = parallel([chat("Count from one to ten in words.", max_tokens=40), chat("Name three primary colors.", max_tokens=24),
                  chat("Spell the word 'quantize' letter by letter.", max_tokens=32)])
record("quantized_lanes_batch_together",
       all(c == 200 and receipt(b).get("status") == "applied" for c, b, _e in mixed) and "ten" in text(mixed[0][1]).lower(),
       {"codes": [c for c, _b, _e in mixed], "answers": [text(b)[:60] for _c, b, _e in mixed],
        "widths": [b.get("mlx2", {}).get("ordinary_compute_width") for _c, b, _e in mixed], "errors": [b.get("error") for _c, b, _e in mixed]})
block = status().get("approximate_kv", {})
record("status_block", block.get("applied", 0) >= 6, {"approximate_kv": block, "operation": policy.get("operation")})
finish()
