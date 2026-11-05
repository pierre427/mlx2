import json, random, sys
sys.path.insert(0, "src")
from mlx2.adapters.ripple_output import RippleOutputParser
from mlx2.runtime.tool_parsers.hermes_json import parse_tool_call
from tests.test_ripple_tools_cpu import TOOLS

def mk(stops=(), tools=True, parallel=True, tolerant=True):
    return RippleOutputParser(chat=True, thinking=False, tools=TOOLS if tools else None,
        parse_tool=parse_tool_call, stops=stops, parallel_tool_calls=parallel, tolerant_tool_markers=tolerant)

def run(p, chunks, fr):
    ev = []
    try:
        for c in chunks[:-1]:
            if p.stopped: break
            ev += p.push(c)
        if not p.stopped:
            ev += p.finish(chunks[-1], fr)
    except ValueError as e:
        return ("ERR", str(e)[:40])
    r = "".join(e.get("reasoning_content","") for e in ev)
    c = "".join(e.get("content","") for e in ev)
    t = [(x["function"]["name"], x["function"]["arguments"], x["index"]) for e in ev for x in e.get("tool_calls",())]
    return r, c, t, p.stopped, p.stop_sequence, p.truncated_tool_call, p.tool_call_constraint_truncations

def call(n="get_weather", city="Paris"):
    return "\n<tool_call>\n" + json.dumps({"name": n, "arguments": {"city": city}}) + "\n</tool_call>"

texts = [
 "We need X.\n</think>\n\nAnswer.",
 "We need X." + call() + call(city="Lyon"),
 "The user asks.\n</think>\n\nOk." + call() + call(city="Lyon"),
 "The user asks.\n</think>\n\n" + call() + "\n" + call(city="Lyon") + "\n",
 "We need <tool_call> thinking</think>\nans",
 "We need STOP x</think>ans",
 "We need x</think>ans STOP more",
 "We need x\n</think>\n\nans <tool_call>\n{\"name\": \"get_wea",
 "We need x\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"P",
 "We need x <tool_call>\n{\"name\": \"get_wea",
 "We",
 "We need",
 "<thi",
 "<think>\nWe need\n</think>\n\nans",
 "The user\n",
 "We need x </thi",
 "We need x </think",
 "We need a```\n<tool_call>\n```\n</think>\n\nans",
 "\nWe need x\n</think>\nans",
 "We need x\n</think>\n\n\n",
 "We need x\n</think>",
 "We need x\n</think>\n</think>ans",
 "We need x\n</think>\n\n<think>y</think>z",
]
stopsets = [(), ("STOP",), ("\n\n",), ("</think>",), ("ans",), (" </think>\n",), ("x\n</think>\n\nans <",)]
random.seed(1)
bad = 0
for text in texts:
  for stops in stopsets:
    for tools in (True, False):
      for parallel in (True, False):
        for tolerant in (True, False):
          for fr in ("stop", "length"):
            ref = run(mk(stops, tools, parallel, tolerant), [text], fr)
            chunkings = [list(text)+[""], [text, ""], ["", text]]
            for _ in range(30):
                cuts = sorted(random.sample(range(len(text)+1), k=min(len(text)+1, random.randint(1,6))))
                parts=[]; prev=0
                for c in cuts: parts.append(text[prev:c]); prev=c
                parts.append(text[prev:])
                chunkings.append(parts)
            for ch in chunkings:
                got = run(mk(stops, tools, parallel, tolerant), ch, fr)
                if got != ref:
                    bad += 1
                    if bad <= 25:
                        print("MISMATCH", repr(text), stops, tools, parallel, tolerant, fr, ch, "\n  ref", ref, "\n  got", got)
                    break
print("mismatches", bad)
