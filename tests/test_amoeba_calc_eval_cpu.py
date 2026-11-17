"""Independent numeric and extraction oracles for the calculator comparison."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('calc_eval', Path(__file__).parents[1]/'scripts/research/amoeba_calc_eval.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

@pytest.mark.parametrize('text,expected', [
    ('Some work: 2 + 3 = 5. #### 5<|im_end|>', 5),
    ('#### 123456789123456789123456', 123456789123456789123456),
    ('#### 1.234567891234567891e23', 123456789123456789100000),
    (r'#### 1.23 \times 10^{12}', 1230000000000),
    (r'#### \frac{7}{4}', Fraction(7,4)),
])
def test_numeric_answer(text,expected):
    assert E.numeric_answer(text)==expected


def test_extraction_decimal_oracle_preserves_literal_precision():
    assert E.expression_value('123456789123.123456789 * 2') == Fraction('246913578246.246913578')
    assert E.expression_value('__import__("os").system("true")') is None
    assert E.expression_value('3**999') is None


def test_final_answer_overrides_earlier_scientific_work():
    assert E.numeric_answer(r'Estimate 1.23 \times 10^{12}. Final answer: 5') == 5
    assert E.numeric_answer(r'Estimate \frac{7}{4}. Final answer: 5') == 5


def test_decorative_hashes_do_not_erase_final_answer():
    assert E.numeric_answer('#### Final Answer: 142\n\n#######<|im_end|>') == 142
    assert E.numeric_answer('#### Final Answer: #### 60') == 60


@pytest.mark.parametrize("text,expected", [
    (r"#### -\frac{7}{4}", Fraction(-7, 4)),
    ("#### −2.5", Fraction(-5, 2)),
    ("#### .75", Fraction(3, 4)),
    ("#### -1 / 2", Fraction(-1, 2)),
])
def test_final_answer_numeric_spelling(text, expected):
    assert E.numeric_answer(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("#### Final answer: **132**.00.<|im_end|>", Fraction(132)),
    ("#### **$132**.05", Fraction("132.05")),
    ("#### **−1,234**.50", Fraction("-1234.5")),
    ("#### **132**. 5 is the final answer.", Fraction(5)),
])
def test_answer_decimal_split_by_markdown(text, expected):
    assert E.numeric_answer(text) == expected
