"""Independent source-only adversarial coverage for the additive ATX grammar."""
import pytest
from scripts.research.kimi_lora_math_answers import extract_numeric_answer

@pytest.mark.parametrize('prefix', [
 '"\n\nCalculation.\n\n', "'\n\nCalculation.\n\n", '“\n\nCalculation.\n\n', '‘\n\nCalculation.\n\n',
 '<blockquote>\nCalculation.\n\n','<pre>\nCalculation.\n\n','<code>\nCalculation.\n\n',
 'The previous model said:\n\nCalculation.\n\n',
 'Here is the wrong response:\n\nCalculation.\n\n',
 'This is not my answer:\n\nCalculation.\n\n',
 'I am unsure.\n','This is an ambiguous answer.\n','It could instead be 8.\n',
])
def test_nonasserted_heading_is_not_answer(prefix):
 assert extract_numeric_answer(prefix+'### Final answer 7')['value'] is None

@pytest.mark.parametrize('heading',[
 '# Answer 0', '## Final answer: -99', '### The final answer is +12.',
 '#### Final answer 001', '##### answer:\t-0', '###### Final answer 234 ###',
])
def test_positive_assertions(heading):
 assert extract_numeric_answer(heading)['value'] is not None

@pytest.mark.parametrize('text',[
 '"A prior quotation ended."\n\nMy calculation gives seven.\n### Final answer 7',
 'The incorrect intermediate method gives eight.\n\nComputing3+4 gives seven.\n### Final answer 7',
 '```text\nA prior code block.\n```\n\nMy calculation gives seven.\n### Final answer 7',
])
def test_unrelated_closed_prior_discussion_does_not_veto(text):
 assert extract_numeric_answer(text)['value']==7

@pytest.mark.parametrize('text',[
 '### Final answer 7.5','### Final answer 7kg','### Final answer 7 %',
 '### Final answer 7/2','### Final answer -7+1','### Final answer 7\nLater work',
 '> ### Final answer 7','    ### Final answer 7','####### Final answer 7',
 '### Final answer 8\n### Final answer 7','{\"answer\":8}\n### Final answer 7',
])
def test_nonterminal_noninteger_quote_and_conflict(text):
 assert extract_numeric_answer(text)['value'] is None

def test_line_leading_single_quote_spans_paragraphs():
 assert extract_numeric_answer("'An earlier response begins here.\n\nCalculation.\n\n### Final answer 7")['value'] is None

def test_closed_single_quote_and_ordinary_apostrophe():
 text="'A prior comment ended.'\n\nI don't need another method. My calculation gives seven.\n### Final answer 7"
 assert extract_numeric_answer(text)['value']==7
