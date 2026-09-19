import json
import statistics
import sys

G = sys.argv[1]
arms = {}
for arm in ("base", "fix"):
    d = json.load(open(f"{G}/bench-35b-{arm}.json"))
    prop = acc = 0
    per_req = []
    outputs = {}
    for row in d["rows"]:
        for req in row["requests"]:
            st = req["receipt"]["mtp"]["stats"]
            prop += st["draft_proposed"]
            acc += st["draft_accepted"]
            per_req.append(st["draft_acceptance"])
            outputs.setdefault(row["width"], []).append(req["output_sha256"])
    arms[arm] = {
        "pooled_acceptance": acc / prop,
        "median_request_acceptance": statistics.median(per_req),
        "tok_s": {w: round(v["median_aggregate_tokens_per_second"], 1) for w, v in d["summary"].items()},
        "outputs": outputs,
        "warmup": {k: v["output_sha256"] for k, v in d["warmup"].items()},
    }
same_warmup = arms["base"]["warmup"] == arms["fix"]["warmup"]
same_outputs = all(
    sorted(arms["base"]["outputs"][w]) == sorted(arms["fix"]["outputs"][w])
    for w in arms["base"]["outputs"]
)
for arm, a in arms.items():
    print(arm, "pooled_acceptance=%.3f" % a["pooled_acceptance"],
          "median_req=%.3f" % a["median_request_acceptance"], "tok/s", a["tok_s"])
for w in arms["base"]["tok_s"]:
    print(f"width {w}: speedup x{arms['fix']['tok_s'][w] / arms['base']['tok_s'][w]:.2f}")
print("greedy warmup outputs identical across arms:", same_warmup)
print("all benchmark outputs identical across arms:", same_outputs)
