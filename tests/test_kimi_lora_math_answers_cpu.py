"""CPU-only extraction/identity tests; no target-model outputs or model imports."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.kimi_lora_math_answers import (
    extract_numeric_answer, main, paired_report, read_verified_jsonl,
)


@pytest.mark.parametrize("text,expected,method", [
    ('{"answer":42}', 42, "terminal_json_integer"),
    ('Calculation gives 42.\n{"answer":42,"other":"ignored"}', 42, "terminal_json_integer"),
    ('Explanation.\n```json\n{"answer":-7}\n```', -7, "terminal_json_integer"),
    (r"Thus $\boxed{42}$.", 42, "final_boxed_integer"),
    (r"\[\boxed{-7}\]", -7, "final_boxed_integer"),
    ('Final answer: 42.', 42, "labelled_final_integer"),
    ('The final answer is -7', -7, "labelled_final_integer"),
    ('**Answer:** **42**', 42, "labelled_final_integer"),
    ('42', 42, "sole_integer_response"),
    ('  +42 [EOS]', 42, "sole_integer_response"),
    ('An incorrect intermediate approach gives 43.\nInstead compute 6*7.\nFinal answer: 42', 42, "labelled_final_integer"),
])
def test_extract_final_integer(text, expected, method):
    result = extract_numeric_answer(text)
    assert (result["value"], result["method"]) == (expected, method)


@pytest.mark.parametrize("text", [
    "", "There were 42 trials and 7 successes.", "42 cm", "42/3", "42.0", "42 or 43",
    'Final answer: 42 kg', 'Answer: 42 and 43', 'Answer: 42e3', 'Answer: 42.0',
    r'Thus \boxed{42/3}', r'Thus \boxed{42} cm', r'Thus \boxed{42} + 3',
    r'Thus \boxed{42} or \boxed{43}', r'Thus 41 and \boxed{42}',
    r'The wrong answer is \boxed{42}', r'Maybe \boxed{42}',
    '> Final answer: 42', '"Final answer: 42"',
    '{"answer":true}', '{"answer":"42"}', '{"answer":42.0}',
    '{"answer":42,"answer":43}', '{"answer":42', '{"result":{"answer":42}}',
    '{"answer":42}\nNow I will check it.', 'Example:\n{"answer":42}',
    '{"answer":42}\nThis is not my final answer.', 'Final answer: ' + '9' * 513,
    '{"unfinished":\n{"answer":42}', '[\n{"answer":42}',
    r'The prompt says \boxed{42}',
])
def test_abstains_on_ambiguous_noninteger_or_partial_answer(text):
    assert extract_numeric_answer(text)["value"] is None


def test_copied_question_not_an_answer():
    question = 'A worked example ends:\nFinal answer: 42'
    assert extract_numeric_answer(question, [question])["reason"] == "question_echo"
    assert extract_numeric_answer('Final answer: 42', [question])["value"] is None


def case(record_id="x", split="dev"):
    return {"id": record_id, "domain": "heldout_math_integer", "split": split,
            "mode": "single_turn", "messages": [{"role": "user", "content": "Compute six times seven."}],
            "oracle": {"kind": "json_fields", "expected": {"answer": 42}}}


def result(text, *, correct=False, capped=False, split="dev"):
    return {"id": "x", "domain": "heldout_math_integer", "split": split,
            "turns": [{"text": text, "finish_reason": "length" if capped else "stop"}],
            "grade": {"correct": correct, "format_ok": correct, "capped": capped}}


def test_numeric_correct_is_separate_from_original_task_and_format():
    report = paired_report({"x": case()}, {"x": result("Answer: 42")},
                           {"x": result('{"answer":42}', correct=True)})
    assert report["base"]["numeric_correct"] == report["adapter"]["numeric_correct"] == 1
    assert report["base"]["strict_task_correct"] == 0
    assert report["adapter"]["strict_task_correct"] == 1
    assert report["paired"]["numeric_transitions"] == {"correct -> correct": 1}


def test_abstention_is_not_incorrect_and_capped_separate():
    report = paired_report({"x": case()}, {"x": result("I tried 42 calculations.", capped=True)},
                           {"x": result('Final answer: 43', capped=True)})
    assert report["base"]["numeric_abstentions"] == 1
    assert report["base"]["numeric_incorrect"] == 0
    assert report["adapter"]["numeric_incorrect"] == 1
    assert report["paired"]["both_extracted"] == 0
    assert report["adapter"]["capped"] == 1


def test_fail_closed_on_missing_pair_or_final_or_invalid_grade():
    with pytest.raises(ValueError, match="IDs"):
        paired_report({"x": case()}, {}, {"x": result("42")})
    with pytest.raises(ValueError, match="final evaluation"):
        paired_report({"x": case(split="final")}, {"x": result("42", split="final")}, {"x": result("42", split="final")})
    bad = result("42"); bad["grade"]["correct"] = 1
    with pytest.raises(ValueError, match="boolean"):
        paired_report({"x": case()}, {"x": bad}, {"x": result("42")})
    bad = result("42"); bad["grade"]["capped"] = True
    with pytest.raises(ValueError, match="capped"):
        paired_report({"x": case()}, {"x": bad}, {"x": result("42")})


def test_hash_ids_and_byte_references(tmp_path):
    path = tmp_path / "cases.jsonl"
    raw = (json.dumps(case()) + "\n").encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    rows, refs, receipt = read_verified_jsonl(path, digest)
    assert rows["x"] == case()
    assert refs["x"] == {"line": 1, "offset": 0, "length": len(raw), "sha256": digest}
    assert receipt["sha256"] == digest
    with pytest.raises(ValueError, match="hash mismatch"):
        read_verified_jsonl(path, "0" * 64)
    path.write_bytes(raw + raw)
    with pytest.raises(ValueError, match="duplicate"):
        read_verified_jsonl(path, hashlib.sha256(raw + raw).hexdigest())


def test_cli_frozen_report_and_complete_identity(tmp_path):
    arguments = []
    for name, row in (("cases", case()), ("base", result("Answer: 42")), ("adapter", result('{"answer":42}', correct=True))):
        path = tmp_path / (name + ".jsonl")
        path.write_text(json.dumps(row) + "\n")
        arguments.extend(["--" + name, str(path), "--" + name + "-sha256", hashlib.sha256(path.read_bytes()).hexdigest()])
    output = tmp_path / "report.json"
    main(arguments + ["--output", str(output)])
    report = json.loads(output.read_text())
    assert report["records"][0]["references"]["base"]["line"] == 1
    assert len(report["scorer_sha256"]) == 64
    with pytest.raises(FileExistsError):
        main(arguments + ["--output", str(output)])


def test_source_case_fixture_has_exact_integer_oracle():
    # Small source-only fixture from audited challenge-v2; no model response is
    # loaded. Each admitted response format is generated from its reference.
    source = {"id": "challenge-bc0a75a81d04032a222d28803e5bf19d17906a88d554be559e8efbe91f88835a",
              "oracle": {"kind": "json_fields", "expected": {"answer": 14}}}
    value = source["oracle"]["expected"]["answer"]
    for text in (json.dumps({"answer": value}), rf"\boxed{{{value}}}", f"Final answer: {value}", str(value)):
        assert extract_numeric_answer(text)["value"] == value


@pytest.mark.parametrize("text,expected", [
    ("Final Answer:\n\\[\n\\boxed{7}\n\\]", 7),
    ("\\[\n\\boxed{\\text{answer: }7}\n\\]", 7),
    ("\\[\n\\boxed{\\text{Final answer: }-123}\n\\]", -123),
    ("$$\n\\boxed{+812}\n$$", 812),
    (r"Thus $\boxed{\text{answer: }-42}$.", -42),
    ("\\[\n\\boxed{\n 12345 \n}\n\\].", 12345),
    (r"\boxed{\text{ANSWER: }0}", 0),
])
def test_v2_general_terminal_latex_forms(text, expected):
    value = extract_numeric_answer(text)
    assert value["value"] == expected
    assert value["method"] == "final_boxed_integer"


@pytest.mark.parametrize("text", [
    "\\[\n\\boxed{42}\n\\]\nMore work follows.",
    "\\[\n\\boxed{42}\n$$",
    "Example:\n\\[\n\\boxed{42}\n\\]",
    "> A possible answer:\n\\[\n\\boxed{42}\n\\]",
    "\\[\n\\boxed{42} \\text{cm}\n\\]",
    "\\[\n\\boxed{\\text{wrong answer: }42}\n\\]",
    "\\[\n\\boxed{\\text{answer: }42/3}\n\\]",
    "\\[\n\\boxed{\\text{answer: }42.0}\n\\]",
    "\\[\n\\boxed{\\text{answer: }42kg}\n\\]",
    "\\[\n\\boxed{42}\n\\boxed{43}\n\\]",
    "\\[\n\\boxed{42} + 3\n\\]",
    "\\[\n\\boxed{\\text{answer: }42,43}\n\\]",
    "\\[\n\\boxed{\\text{answer: }42}\n",
])
def test_v2_latex_does_not_admit_ambiguous_partial_or_unit_answers(text):
    assert extract_numeric_answer(text)["value"] is None


@pytest.mark.parametrize('text,expected', [
    ('### Final answer 7', 7),
    ('# Answer 42', 42),
    ('## Final Answer: -192', -192),
    ('### The final answer is +42.', 42),
    ('###### answer:0 ######', 0),
    ('   ### Final answer -003', -3),
    ('### Final answer:\t7', 7),
    ('### FINAL ANSWER IS 314159', 314159),
    ('Calculation gives seven.\n\n### Final answer 7\n', 7),
    ('An incorrect intermediate approach gives 8.\n\nCompute 3+4 instead.\n### Final answer 7', 7),
    ('### Final answer 7 [EOS]', 7),
    ('### Final answer 7<|im_end|>', 7),
    ('### Final answer 42\n### Final answer 42', 42),
    ('Final answer: 42\n### Final answer 42', 42),
])
def test_v3_terminal_atx_heading(text, expected):
    result = extract_numeric_answer(text)
    assert result == {'value': expected, 'method': 'terminal_atx_heading_integer', 'reason': 'extracted'}


@pytest.mark.parametrize('text', [
    '####### Final answer 7', '###Final answer 7', '    ### Final answer 7',
    '\t### Final answer 7', '### Final answer', '### Final answer\n7',
    '### Final answer 42 kg', '### Final answer 42.0', '### Final answer 42e3',
    '### Final answer 42/3', '### Final answer 42 or 43', '### Final answer 42 and 43',
    '### Final answer 7#', '### Final answer 7 # trailing',
    '### Final answer 7\nNow I will check.', '### Final answer ' + '9'*513,
    '> ### Final answer 7', '"### Final answer 7"', "'### Final answer 7'",
    '```\n### Final answer 7', '```markdown\n### Final answer 7',
    '~~~markdown\n### Final answer 7', '````text\n```\n### Final answer 7',
    '~~~\n```\n### Final answer 7', '```\n### Final answer 7\n```',
    'Example:\n### Final answer 7', 'For example:\n\n### Final answer 7',
    'Quoted answer:\n### Final answer 7', 'The suggested response is:\n### Final answer 7',
    'A possible answer:\n### Final answer 7', 'Perhaps this is right:\n### Final answer 7',
    'I cannot determine the result.\n### Final answer 7',
    'I do not know.\n### Final answer 7', 'It might be:\n### Final answer 7',
    'It could be:\n### Final answer 7', 'This is not the answer:\n### Final answer 7',
    '> The response was:\n### Final answer 7', '"\n### Final answer 7',
    '## Example\n\nSome arithmetic here.\n### Final answer 7',
    'Example:\nUse this sample result:\n### Final answer 7',
    '<!--\n### Final answer 7', '{"unfinished":\n### Final answer 7',
    '[\n### Final answer 7', '\\[\n### Final answer 7',
    'Final answer: 8\n### Final answer 7', '### Final answer 8\n### Final answer 7',
    '{"answer":8}\n### Final answer 7', '\\boxed{8}\n### Final answer 7',
])
def test_v3_heading_rejects_nonassertions_conflicts_and_incomplete_context(text):
    assert extract_numeric_answer(text)['value'] is None


def test_v3_heading_echo_anywhere_in_question_and_iterator_questions():
    question = 'A quoted sample says ### Final answer 7, then asks you to explain it.'
    assert extract_numeric_answer('### Final answer 7', [question])['reason'] == 'question_heading_echo'
    assert extract_numeric_answer('### Final answer 7', iter([question]))['value'] is None
    assert extract_numeric_answer('### Final answer 7', ['Compute three plus four.'])['value'] == 7


def test_v3_heading_does_not_change_original_format_grade_or_cap():
    report = paired_report({'x': case()}, {'x': result('### Final answer 42')},
                           {'x': result('### Final answer 42', capped=True)})
    assert report['schema_version'] == 'kimi-integer-answer-v3'
    assert report['base']['numeric_correct'] == report['adapter']['numeric_correct'] == 1
    assert report['base']['format_ok'] == report['adapter']['format_ok'] == 0
    assert report['base']['strict_task_correct'] == report['adapter']['strict_task_correct'] == 0
    assert report['base']['uncapped_numeric_correct'] == 1
    assert report['adapter']['uncapped_numeric_correct'] == 0


@pytest.mark.parametrize('text', [
    'Example:\n\nSome arithmetic.\n\n### Final answer 7',
    'A quoted response follows.\n\nCalculation.\n\n### Final answer 7',
    'The suggested response follows.\n\nCalculation.\n\n### Final answer 7',
    'Use this template.\n\nCalculation.\n\n### Final answer 7',
])
def test_v3_illustrative_frame_cannot_escape_via_blank_lines(text):
    assert extract_numeric_answer(text)['value'] is None


@pytest.mark.parametrize('prefix', [
    '"\n\nA computation.\n\n', "'\n\nA computation.\n\n", '“\n\nA computation.\n\n',
    '<blockquote>\nA computation.\n\n', '<pre class="text">\nA computation.\n\n',
    '<code>\nA computation.\n\n', 'The previous model said:\n\nA computation.\n\n',
    'Here is the wrong response:\n\nA computation.\n\n',
    'This is not my answer:\n\nA computation.\n\n',
    'I am unsure.\n', 'This is an ambiguous answer:\n', 'The answer could instead be 8.\n',
])
def test_v3_peer_context_regressions(prefix):
    assert extract_numeric_answer(prefix + '### Final answer 7')['value'] is None


def test_v3_explicit_fresh_own_solution_can_end_prior_frame():
    text = 'The previous model said:\n\nSome computation.\n\nMy own solution:\n\nCompute three plus four.\n### Final answer 7'
    assert extract_numeric_answer(text)['value'] == 7


def test_v3_line_leading_single_quote_across_paragraphs():
    assert extract_numeric_answer("'An earlier response begins here.\n\nCalculation.\n\n### Final answer 7")['value'] is None
    assert extract_numeric_answer("'An earlier response ended.'\n\nMy calculation gives seven.\n### Final answer 7")['value'] == 7
    assert extract_numeric_answer("It's seven by addition.\n### Final answer 7")['value'] == 7
