"""North: structured output with thinking on by default (grammar deferred, answer envelope)."""
import json as _json
from probe_common import *

SCHEMA = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": {
    "type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}, "coastal": {"type": "boolean"}},
    "required": ["city", "population", "coastal"], "additionalProperties": False}}}


def ask(content, **extra):
    return {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 900, **extra}


def parsed(body):
    try:
        return _json.loads(text(body))
    except Exception:
        return None


def so(body):
    return (body.get("mlx2", {}).get("request_controls", {}) or {}).get("structured_output") or {}

cities = ["Paris", "Budapest", "Lisbon", "Ottawa", "Helsinki", "Havana"]
truth = {"Paris": False, "Budapest": False, "Lisbon": True, "Ottawa": False, "Helsinki": True, "Havana": True}
runs = parallel([ask(f"Describe the city {c} as JSON.", response_format=SCHEMA) for c in cities])
docs = [parsed(b) for _c, b, _e in runs]
record("thinking_default_schema_valid",
       all(c == 200 for c, _b, _e in runs) and all(isinstance(d, dict) and isinstance(d.get("population"), int) and isinstance(d.get("coastal"), bool) for d in docs)
       and all(so(b).get("deferred") is True and b["mlx2"]["request_controls"]["thinking"] is True for _c, b, _e in runs),
       {"docs": docs, "deferred_tokens": [so(b).get("deferred_tokens") for _c, b, _e in runs], "errors": [b.get("error") for _c, b, _e in runs],
        "finish": [b.get("choices", [{}])[0].get("finish_reason") for _c, b, _e in runs]})
record("reasoning_and_answer_are_separated",
       all((b["choices"][0]["message"].get("reasoning_content") or "").strip() for _c, b, _e in runs)
       and not any(m in text(b) for _c, b, _e in runs for m in ("<|", "|>")),
       {"reasoning_chars": [len(b["choices"][0]["message"].get("reasoning_content") or "") for _c, b, _e in runs], "content": [text(b)[:80] for _c, b, _e in runs]})
correct = sum(bool(d) and d.get("coastal") == truth[c] for c, d in zip(cities, docs))
padded = sum("      " in text(b) for _c, b, _e in runs)
record("quality_with_thinking", correct >= 5, {"coastal_correct": f"{correct}/6", "padded_documents": padded,
                                               "populations": [d.get("population") if d else None for d in docs]})
off = parallel([ask(f"Describe the city {c} as JSON.", response_format=SCHEMA, enable_thinking=False, max_tokens=120) for c in cities[:3]])
record("thinking_off_still_enforced", all(c == 200 and isinstance(parsed(b), dict) and so(b).get("deferred") is False for c, b, _e in off),
       {"content": [text(b)[:90] for _c, b, _e in off], "padded": sum("      " in text(b) for _c, b, _e in off)})
obj = post(ask("Return a JSON object with keys country and capital for Kenya.", response_format={"type": "json_object"}))
record("json_object_thinking_default", obj[0] == 200 and isinstance(parsed(obj[1]), dict) and "nairobi" in text(obj[1]).lower(), {"content": text(obj[1])[:120], "receipt": so(obj[1])})
grammar = post(ask("Is water wet? Answer yes or no.", grammar="(?:yes|no)"))
record("raw_grammar_thinking_default", grammar[0] == 200 and text(grammar[1]) in ("yes", "no"), {"content": text(grammar[1]), "receipt": so(grammar[1]), "error": grammar[1].get("error")})
finish()
