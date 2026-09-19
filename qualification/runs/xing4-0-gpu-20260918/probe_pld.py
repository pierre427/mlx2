"""Prompt lookup with/without rotating replay: deterministic outputs for cross-server comparison."""
from probe_common import *

passage = ("The harbor master logged each vessel: the Meridian carried tin, the Oriole carried salt, "
           "the Kestrel carried cedar, and the Lantern carried glass. ") * 6
prompts = [
    "Repeat the following passage exactly, then stop.\n\n" + passage,
    "Here is a Python function:\n\ndef area(width, height):\n    result = width * height\n    return result\n\nRewrite it three times, renaming it area_one, area_two and area_three, changing nothing else.",
    "List the numbers from 1 to 40 separated by commas.",
]
outputs = []
for index, prompt_text in enumerate(prompts):
    code, body, elapsed = post(chat(prompt_text, max_tokens=220))
    outputs.append({"code": code, "content": text(body), "finish": body.get("choices", [{}])[0].get("finish_reason"),
                    "tokens": body.get("usage", {}).get("completion_tokens"), "seconds": round(elapsed, 2)})
record("requests_complete", all(o["code"] == 200 and o["content"] for o in outputs), {"outputs": [{**o, "content": o["content"][:80]} for o in outputs]})
mixed = parallel([chat(p, max_tokens=120) for p in prompts])
record("concurrent_lanes", all(c == 200 for c, _b, _e in mixed), {"codes": [c for c, _b, _e in mixed], "errors": [b.get("error") for _c, b, _e in mixed]})
sched = status().get("scheduler", {})
pld = {k: v for k, v in sched.items() if k.startswith("pld_")}
replay_on = bool((status()["settings"].get("prompt_lookup") or {}).get("rotating_replay"))
record("mechanism_counters",
       pld.get("pld_retrieval_cycles", 0) > 0 and pld.get("pld_rollbacks", 0) > 0
       and ((pld.get("pld_rotating_replay_rounds", 0) > 0 and pld.get("pld_rotating_replay_rebuilds", 0) == 0) if replay_on
            else pld.get("pld_rotating_replay_rounds", 0) == 0),
       {"rotating_replay": replay_on, "pld": pld})
results["_outputs"] = {"ok": True, "contents": [o["content"] for o in outputs]}
finish()
