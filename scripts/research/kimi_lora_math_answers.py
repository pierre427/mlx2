#!/usr/bin/env python3
"""Conservative, CPU-only companion scoring for frozen Kimi integer-math cases.

This does not replace the original task grader. An unextracted answer is an
abstention, not proof of mathematical error. Greater extraction/JSON coverage
alone must not be described as a reasoning gain. No model libraries are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


VERSION = "kimi-integer-answer-v3"
INTEGER = r"[+-]?[0-9]{1,512}"
BOX_PATTERN = (r"\\boxed\{\s*(?:\\text\{\s*(?:final\s+answer|answer)\s*:\s*\}\s*)?"
               + "(" + INTEGER + ")" + r"\s*\}")
NEGATIVE_CUE = re.compile(
    r"\b(?:wrong|incorrect|rejected|example|question|prompt|copied|hypothetical|maybe|perhaps|either|not\s+the\s+answer)\b",
    re.IGNORECASE,
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _normalize(text):
    return " ".join(text.split()).casefold()


ATX_INTEGER = re.compile(
    r"[ ]{0,3}#{1,6}[ \t]+(?:the[ \t]+)?(?:final[ \t]+answer|answer)"
    r"(?:[ \t]*:[ \t]*|[ \t]+is[ \t]+|[ \t]+)(" + INTEGER + r")\.?"
    r"(?:[ \t]+#+)?[ \t]*$", re.I,
)
HEADING_CONTEXT_CUE = re.compile(
    r"\b(?:quoted?|suggested|possible|candidate|alternative|uncertain|unsure|ambiguous|"
    r"illustration|(?:could|might|may)(?:\s+(?:instead|possibly|also))?\s+be|"
    r"cannot\s+(?:determine|decide)|do\s+not\s+know|don't\s+know|not\s+sure)\b", re.I,
)
HEADING_FRAME_CUE = re.compile(
    r"\b(?:example|quoted?|suggested|sample|hypothetical|illustration|template)\b", re.I,
)


def _terminal_atx_integer(raw_text, question_texts):
    """Narrow v3 addition; None delegates to the unchanged v2 grammar.

    Only a terminal, complete, same-line ATX integer assertion is added.
    Conservative context refusals may miss valid answers; they never infer a
    number from arbitrary reasoning. This is syntax coverage, not a proof check.
    """
    text = raw_text.rstrip()
    for marker in ("<|im_end|>", "[EOS]"):
        if text.endswith(marker):
            text = text[:-len(marker)].rstrip()
    lines = text.splitlines()
    if not lines:
        return None
    match = ATX_INTEGER.fullmatch(lines[-1])
    if match is None:
        return None
    reject = lambda reason: {"value": None, "method": None, "reason": reason}
    prefix = lines[:-1]
    # Markdown fences can have either delimiter, >=3 characters, and a longer
    # closer. Merely counting backticks misses unclosed tilde/language fences.
    fence = None
    for line in prefix:
        boundary = re.match(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$", line)
        if boundary is None:
            continue
        run, tail = boundary.groups()
        if fence is None:
            if run[0] != "`" or "`" not in tail:
                fence = (run[0], len(run))
        elif run[0] == fence[0] and len(run) >= fence[1] and not tail.strip():
            fence = None
    if fence is not None:
        return reject("heading_inside_unclosed_fence")
    prior = "\n".join(prefix)
    if (prior.count("<!--") != prior.count("-->") or
            prior.count("{") != prior.count("}") or prior.count("[") != prior.count("]")):
        return reject("heading_inside_unclosed_container")
    single_quote_open = False
    for line in prefix:
        for index, char in enumerate(line):
            if char != "'":
                continue
            if (0 < index < len(line) - 1 and line[index - 1].isalnum()
                    and line[index + 1].isalnum()):
                continue  # An ordinary apostrophe inside a word.
            if single_quote_open:
                single_quote_open = False
            elif not line[:index].strip():
                single_quote_open = True
    if (prior.count('"') % 2 or single_quote_open
            or prior.count("“") != prior.count("”") or prior.count("‘") > prior.count("’")):
        return reject("heading_inside_unclosed_quotation")
    html_stack = []
    for tag in re.finditer(r"<(\/?)(blockquote|pre|code)\b[^>]*>", prior, re.I):
        closing, name = tag.group(1), tag.group(2).lower()
        if tag.group(0).endswith("/>"):
            continue
        if closing:
            if not html_stack or html_stack.pop() != name:
                return reject("heading_inside_invalid_html_context")
        else:
            html_stack.append(name)
    if html_stack:
        return reject("heading_inside_unclosed_html_context")
    # A quotation/example can span paragraphs. Keep this new syntax narrow
    # rather than silently treating its last heading as our own assertion.
    framed = False
    for line in prefix:
        # An explicit fresh own-answer introduction can end earlier discussion
        # after its quotes/fences have closed. The terminal heading alone cannot
        # end an attributed response: it might still be part of that response.
        if re.fullmatch(r"\s*(?:#{1,6}\s+)?(?:my|our)\s+(?:own\s+)?(?:answer|solution|calculation)"
                        r"\s*(?:follows\s*:|:)\s*", line, re.I):
            framed = False
        elif (HEADING_FRAME_CUE.search(line)
              or re.search(r"\b(?:model|assistant|user|speaker|source|author|they|he|she)"
                           r"\s+(?:said|wrote|returned|stated)\s*:\s*$", line, re.I)
              or re.search(r"\bnot\s+(?:my|our|the)\s+(?:answer|response)\b", line, re.I)
              or (line.rstrip().endswith(":") and NEGATIVE_CUE.search(line)
                  and re.search(r"\b(?:answer|response|output)\b", line, re.I))):
            framed = True
    if framed:
        return reject("quoted_or_illustrative_heading_frame")
    # Bound the introduction to its last paragraph and nearest section header;
    # unrelated earlier discussion of a wrong approach need not veto an answer.
    paragraph = []
    for line in reversed(prefix):
        if not line.strip():
            if paragraph:
                break
            continue
        paragraph.append(line.strip())
    sections = [line.strip() for line in prefix if re.match(r"^[ ]{0,3}#{1,6}[ \t]+", line)]
    context = paragraph + sections[-1:]
    if any(NEGATIVE_CUE.search(line) or HEADING_CONTEXT_CUE.search(line)
           or line.startswith((">", '"', "'", "`")) for line in context):
        return reject("unasserted_or_ambiguous_heading_context")
    # Refuse a copied heading anywhere in a supplied question, not only its
    # final line. Normalized substring matching is deliberately conservative.
    heading = _normalize(lines[-1])
    for question in question_texts:
        if isinstance(question, str) and heading in _normalize(question):
            return reject("question_heading_echo")
    value = int(match.group(1))
    for line in prefix:
        # Distinct explicit answer claims make the terminal heading ambiguous.
        # Reuse the established grammar on each complete line; all such lines
        # are shorter than this response, so recursive heading handling ends.
        previous = extract_numeric_answer(line, question_texts)
        if previous["value"] is not None and previous["value"] != value:
            return reject("conflicting_explicit_answer_claims")
    return {"value": value, "method": "terminal_atx_heading_integer", "reason": "extracted"}


def extract_numeric_answer(text: str, question_texts=()):
    """Extract only an explicitly final integer; never search arbitrary numbers."""
    if not isinstance(text, str):
        raise ValueError("assistant response must be text")
    # Materialize once: the new context/echo checks and the unchanged grammar
    # must see the same questions even when a caller supplied an iterator.
    question_texts = tuple(question_texts)
    heading = _terminal_atx_integer(text, question_texts)
    if heading is not None:
        return heading
    text = text.strip()
    for marker in ("<|im_end|>", "[EOS]"):
        if text.endswith(marker):
            text = text[:-len(marker)].rstrip()
    abstain = lambda reason: {"value": None, "method": None, "reason": reason}
    found = lambda value, method: {"value": int(value), "method": method, "reason": "extracted"}
    if not text:
        return abstain("empty_response")
    last_line = text.splitlines()[-1].strip()
    for question in question_texts:
        if not isinstance(question, str) or not question.strip():
            continue
        if _normalize(text) == _normalize(question):
            return abstain("question_echo")
        if _normalize(last_line) == _normalize(question.strip().splitlines()[-1]):
            return abstain("question_final_line_echo")

    # Only a terminal object (or terminal JSON fence) is eligible. Extra JSON
    # fields do not invalidate an integer answer here; original format grading
    # is preserved separately. Nested objects and duplicate keys are rejected.
    json_text = text
    fence = re.search(r"(?:^|\n)```(?:json)?\s*\n(\{[\s\S]*\})\s*\n```\s*$", text, re.I)
    if fence:
        json_text = text[:fence.start()] + "\n" + fence.group(1)
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for match in re.finditer(r"(?:^|\n)[ \t]*(?=\{)", json_text):
        start = match.end()
        try:
            value, end = decoder.raw_decode(json_text, start)
        except (ValueError, json.JSONDecodeError):
            continue
        if json_text[end:].strip():
            continue
        prefix = json_text[:start]
        # A closing inner object of an unfinished outer object is not a final
        # JSON response. This conservative balance check may abstain on prose
        # containing an unmatched literal brace, which is safer than guessing.
        if prefix.count("{") != prefix.count("}") or prefix.count("[") != prefix.count("]"):
            return abstain("unbalanced_json_prefix")
        preceding = prefix.strip().splitlines()
        if preceding and NEGATIVE_CUE.search(preceding[-1]):
            return abstain("unasserted_or_ambiguous_json_context")
        if isinstance(value, dict) and type(value.get("answer")) is int:
            return found(value["answer"], "terminal_json_integer")
        return abstain("terminal_json_answer_not_integer")

    # v2 accepts a terminal display containing only one integer box, including
    # a benign text label inside that box. This is a syntax rule, independent
    # of any case ID, reference value, or observed candidate output.
    display = re.search(r"(?P<open>\\\[|\$\$)\s*(?P<box>" + BOX_PATTERN
                        + r")\s*(?P<close>\\\]|\$\$)\.?\s*$", text, re.I)
    if display:
        if (display.group("open"), display.group("close")) not in ((r"\[", r"\]"), ("$$", "$$")):
            return abstain("mismatched_math_delimiters")
        preceding = text[:display.start()].strip().splitlines()
        if preceding and (NEGATIVE_CUE.search(preceding[-1]) or preceding[-1].lstrip().startswith(">")):
            return abstain("unasserted_or_ambiguous_box_context")
        last_line = display.group(0).strip()
    elif (text.count(r"\[") != text.count(r"\]") or text.count("$$") % 2):
        return abstain("unfinished_math_display")

    # A quoted/example line is never an asserted final answer. Do not reject
    # unrelated earlier discussion of errors in an otherwise valid solution.
    if last_line.startswith((">", '"', "'", "`")) or NEGATIVE_CUE.search(last_line):
        return abstain("unasserted_or_ambiguous_final_line")
    boxes = list(re.finditer(BOX_PATTERN, last_line, re.I))
    if len(boxes) == 1:
        box = boxes[0]
        tail = last_line[box.end():].strip()
        # Closing display/inline math and sentence punctuation are harmless;
        # units, equations, alternatives, and prose after the integer are not.
        if re.fullmatch(r"(?:\\\]|\\\)|\$\$?|[.]){0,3}", tail):
            prefix = last_line[:box.start()]
            if not re.search(r"\\boxed|\b(?:and|or)\s*$", prefix, re.I):
                return found(box.group(1), "final_boxed_integer")
    label_line = last_line.replace("**", "")
    label = re.fullmatch(
        r"(?:the\s+)?(?:final\s+answer|answer)\s*(?::|is)\s*(" + INTEGER + r")\.?",
        label_line, re.I,
    )
    if label:
        return found(label.group(1), "labelled_final_integer")
    if re.fullmatch(INTEGER, text):
        return found(text, "sole_integer_response")
    return abstain("no_unambiguous_final_integer")


def read_verified_jsonl(path, expected_sha256):
    path = Path(path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or actual != expected_sha256:
        raise ValueError(f"input hash mismatch: {path}")
    rows, refs, offset = {}, {}, 0
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            raise ValueError(f"blank record at {path}:{number}")
        row = json.loads(line, object_pairs_hook=_unique_object)
        record_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(record_id, str) or not record_id or record_id in rows:
            raise ValueError(f"missing/duplicate record ID at {path}:{number}")
        rows[record_id] = row
        refs[record_id] = {"line": number, "offset": offset, "length": len(line),
                           "sha256": hashlib.sha256(line).hexdigest()}
        offset += len(line)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows, refs, {"path": str(path.resolve()), "sha256": actual, "records": len(rows)}


def _grade_arm(case, row):
    for field in ("domain", "split"):
        if row.get(field) != case.get(field):
            raise ValueError(f"{field} mismatch for {case['id']}")
    grade = row.get("grade", {})
    if any(type(grade.get(key)) is not bool for key in ("correct", "format_ok", "capped")):
        raise ValueError(f"missing boolean original grade for {case['id']}")
    turns = row.get("turns")
    if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], dict):
        raise ValueError(f"integer challenge requires one complete result turn: {case['id']}")
    if turns[0].get("finish_reason") not in ("stop", "length"):
        raise ValueError(f"unknown finish reason for {case['id']}")
    if grade["capped"] != (turns[0]["finish_reason"] == "length"):
        raise ValueError(f"inconsistent capped flag for {case['id']}")
    extracted = extract_numeric_answer(turns[0].get("text"),
                                      [m.get("content", "") for m in case["messages"] if m["role"] == "user"])
    answer = case["oracle"]["expected"]["answer"]
    return {"strict_task_correct": grade["correct"], "format_ok": grade["format_ok"],
            "capped": grade["capped"], "numeric_answer_correct": None if extracted["value"] is None else extracted["value"] == answer,
            "extraction": extracted}


def _summarize(rows):
    count = len(rows)
    extracted = sum(r["numeric_answer_correct"] is not None for r in rows)
    correct = sum(r["numeric_answer_correct"] is True for r in rows)
    methods = {}
    for row in rows:
        method = row["extraction"]["method"] or "abstain"
        methods[method] = methods.get(method, 0) + 1
    return {"cases": count, "strict_task_correct": sum(r["strict_task_correct"] for r in rows),
            "format_ok": sum(r["format_ok"] for r in rows), "capped": sum(r["capped"] for r in rows),
            "numeric_extracted": extracted, "numeric_correct": correct, "numeric_incorrect": extracted - correct,
            "numeric_abstentions": count - extracted, "numeric_correct_fraction_all_cases": correct / count,
            "numeric_accuracy_among_extracted": correct / extracted if extracted else None,
            "uncapped_numeric_correct": sum(r["numeric_answer_correct"] is True and not r["capped"] for r in rows),
            "extraction_methods": methods}


def paired_report(cases, base, adapter, *, allow_final=False):
    if not cases or set(cases) != set(base) or set(cases) != set(adapter):
        raise ValueError("case/base/adapter IDs must match exactly; no partial pairing")
    if not allow_final and any(c.get("split") == "final" for c in cases.values()):
        raise ValueError("final evaluation requires explicit --allow-final after candidate selection")
    details = []
    for record_id, case in cases.items():
        if any(row.get("id") != record_id for row in (case, base[record_id], adapter[record_id])):
            raise ValueError(f"record ID/key mismatch for {record_id}")
        if case.get("domain") != "heldout_math_integer":
            continue
        expected = case.get("oracle", {}).get("expected", {})
        if (case.get("mode") != "single_turn" or case.get("oracle", {}).get("kind") != "json_fields"
                or type(expected.get("answer")) is not int):
            raise ValueError(f"invalid integer-math oracle for {record_id}")
        details.append({"id": record_id, "split": case["split"], "expected": expected["answer"],
                        "base": _grade_arm(case, base[record_id]), "adapter": _grade_arm(case, adapter[record_id])})
    if not details:
        raise ValueError("no heldout_math_integer cases")
    common = [r for r in details if all(r[arm]["numeric_answer_correct"] is not None for arm in ("base", "adapter"))]
    transitions = {}
    for row in details:
        key = " -> ".join("abstain" if row[arm]["numeric_answer_correct"] is None else "correct" if row[arm]["numeric_answer_correct"] else "incorrect" for arm in ("base", "adapter"))
        transitions[key] = transitions.get(key, 0) + 1
    return {"schema_version": VERSION, "input_cases": len(cases), "math_cases": len(details),
            "nonmath_cases_not_scored": len(cases) - len(details),
            "interpretation": ["Original strict task correctness is retained, not recomputed.",
                "Abstention means no safely extractable integer, not a proven mathematical error.",
                "JSON or extraction coverage improvement alone is not evidence of reasoning improvement.",
                "Common-extracted comparisons are conditional on output coverage and may be selection-biased.",
                "Capped outputs are counted separately, including any complete integer they already emitted."],
            "base": _summarize([r["base"] for r in details]), "adapter": _summarize([r["adapter"] for r in details]),
            "paired": {"numeric_transitions": transitions, "both_extracted": len(common),
                "base_correct_when_both_extracted": sum(r["base"]["numeric_answer_correct"] for r in common),
                "adapter_correct_when_both_extracted": sum(r["adapter"]["numeric_answer_correct"] for r in common),
                "strict_task_correct_delta": sum(r["adapter"]["strict_task_correct"] - r["base"]["strict_task_correct"] for r in details)},
            "records": details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "base", "adapter"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-final", action="store_true")
    args = parser.parse_args(argv)
    inputs = {name: read_verified_jsonl(getattr(args, name), getattr(args, name + "_sha256"))
              for name in ("cases", "base", "adapter")}
    report = paired_report(*(inputs[name][0] for name in ("cases", "base", "adapter")), allow_final=args.allow_final)
    report["inputs"] = {name: value[2] for name, value in inputs.items()}
    report["scorer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for row in report["records"]:
        row["references"] = {name: value[1][row["id"]] for name, value in inputs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=args.output.name + ".partial-", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, args.output)  # Atomic and refuses to replace a frozen report.
    finally:
        os.unlink(temp)
    print(json.dumps({"report": str(args.output), "math_cases": report["math_cases"],
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
