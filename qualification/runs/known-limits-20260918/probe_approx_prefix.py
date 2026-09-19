"""Approximate KV over a warm EXACT prefix: the branch is quantized privately, the entry stays exact.

Raw completions, so the long prompt is a true token extension of the short one
(a chat template would close the short turn and break the shared prefix).
"""
import json, time, uuid, urllib.request, urllib.error
from probe_common import BASE, record, finish, status

policy = status()["settings"]["approximate_kv"]
start = policy["start_tokens"]


def complete(prompt, max_tokens=12):
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps({"prompt": prompt, "temperature": 0, "max_tokens": max_tokens}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def out(body):
    return (body.get("choices", [{}])[0].get("text") or "").strip()


def receipt(body):
    return body.get("mlx2", {}).get("approximate_kv") or {}

nonce = uuid.uuid4().hex
rows = [f"Ledger line {i}: account {i * 13 % 500} moved {i * 17 % 900} credits.\n" for i in range(400)]
head = f"Session {nonce}.\nImportant: the launch password is ORCHID-5521.\n" + "".join(rows[:70])
short_p = head + "Ledger"                                   # ends mid-pattern: the long prompt extends it token for token
long_p = head + "".join(rows[70:260]) + "Reminder, the launch password is ORCHID-"

a1, a2 = complete(short_p), complete(short_p)
record("short_prompt_is_an_exact_lane_and_is_published",
       a1[0] == a2[0] == 200 and receipt(a1[1]).get("status") == "declined" and a2[1]["mlx2"]["cached_tokens"] > 0
       and a1[1]["usage"]["prompt_tokens"] < start and out(a1[1]) == out(a2[1]),
       {"prompt_tokens": a1[1]["usage"]["prompt_tokens"], "start_tokens": start, "reason": receipt(a1[1]).get("reason"),
        "warm_cached": a2[1]["mlx2"]["cached_tokens"], "texts": [out(a1[1])[:40], out(a2[1])[:40]], "error": a1[1].get("error")})
before = status()
b1 = complete(long_p)
after = status()
record("long_prompt_quantizes_a_warm_exact_prefix",
       b1[0] == 200 and receipt(b1[1]).get("status") == "applied" and b1[1]["mlx2"]["cached_tokens"] > 1000 and "5521" in out(b1[1])
       and after["approximate_kv"].get("requantized_prefix_hits", 0) > before["approximate_kv"].get("requantized_prefix_hits", 0),
       {"prompt_tokens": b1[1]["usage"]["prompt_tokens"], "cached_tokens": b1[1]["mlx2"]["cached_tokens"], "text": out(b1[1])[:40],
        "status": receipt(b1[1]).get("status"), "requantized_prefix_hits": after["approximate_kv"].get("requantized_prefix_hits"), "error": b1[1].get("error")})
a3 = complete(short_p)
record("exact_entry_is_untouched_by_the_quantized_branch",
       a3[0] == 200 and out(a3[1]) == out(a1[1]) and a3[1]["mlx2"]["cached_tokens"] == a2[1]["mlx2"]["cached_tokens"] and receipt(a3[1]).get("status") == "declined",
       {"text": out(a3[1])[:40], "cached_tokens": a3[1]["mlx2"]["cached_tokens"]})
b2 = complete(long_p)
record("long_prompt_repeat_is_stable_and_never_cached_as_exact",
       b2[0] == 200 and out(b2[1]) == out(b1[1]) and receipt(b2[1]).get("status") == "applied"
       and b2[1]["mlx2"]["cached_tokens"] == b1[1]["mlx2"]["cached_tokens"],
       {"text": out(b2[1])[:40], "cached_tokens": b2[1]["mlx2"]["cached_tokens"]})
leases = []
for _ in range(10):  # the lease is released at request teardown, just after the response is sent
    final = status()
    leases.append(final["apcv2"]["cow"]["active_leases"])
    if leases[-1] == 0:
        break
    time.sleep(1)
record("counters", final["counts"].get("apcv2_store_skipped_approximate", 0) > 0 and final["apcv2"]["cow"]["active_leases"] == 0,
       {"lease_readings": leases, "skipped": final["counts"].get("apcv2_store_skipped_approximate"), "applied": final["approximate_kv"].get("applied"),
        "requantized_prefix_hits": final["approximate_kv"].get("requantized_prefix_hits")})
finish()
