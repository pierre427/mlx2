"""CPU-only functional coverage; checkpoint N/A, no model or GPU execution."""
import math
from dataclasses import dataclass
from fractions import Fraction

import pytest

from mlx2.runtime import amoeba_math_latex as latex


def exact(source):
    return latex.evaluate_exact(latex.parse_latex(source))


@pytest.mark.parametrize(("source", "expected"), [
    (r"$\frac{1}{2}+\dfrac{1}{3}+\tfrac{1}{6}$", Fraction(1)),
    (r"\[\left(2+3\right)\times 4\cdot 2\div 5\]", Fraction(8)),
    (r"\(\frac{\frac{1}{2}+1}{\frac{3}{4}}\)", Fraction(2)),
    (r"\left\{2+[3-1]\right\}^{3}", Fraction(64)),
    (r"-2^2+(-2)^2", Fraction(0)),
    (r"2^{-3}", Fraction(1, 8)),
    (r"2^{3^2}", Fraction(512)),
    (r"0^0", Fraction(1)),
    (r"2(3+4)+(2)(3)", Fraction(20)),
    (r".5+0.125", Fraction(5, 8)),
    (r"1,234+5,678.125", Fraction('6912.125')),
    (r"\displaystyle 2\,\times\!3", Fraction(6)),
    (r"5!", Fraction(120)),
    (r"0!", Fraction(1)),
    (r"\binom{10}{3}+\dbinom{4}{2}+\tbinom{3}{0}", Fraction(127)),
    (r"(-17)\bmod 5", Fraction(3)),
    (r"\sqrt{\frac{81}{16}}", Fraction(9, 4)),
    (r"\sqrt[3]{-125}", Fraction(-5)),
    (r"\sqrt[64]{1}", Fraction(1)),
    (r"\sqrt{0}", Fraction(0)),
    (r"\sum_{i=1}^{5}i^2", Fraction(55)),
    (r"\sum_{i=-2}^{2}{i^2+1}", Fraction(15)),
    (r"\prod_{k=1}^{5}{k}", Fraction(120)),
    (r"\sum_{i=3}^{1}{1/0}", Fraction(0)),
    (r"\prod_{i=3}^{1}{1/0}", Fraction(1)),
    (r"\sum_{i=1}^{3}{\sum_{j=1}^{2}{i+j}}", Fraction(21)),
    (r"\sum_{i=1}^{2}{\sum_{i=1}^{3}{i}+i}", Fraction(15)),
    (r"\sum_{i=1}^{3}i^2+10", Fraction(24)),
    (r"\sum_{i=1}^{3}{2i}", Fraction(12)),
])
def test_exact_capabilities(source, expected):
    assert exact(source) == expected


@pytest.mark.parametrize(("source", "reason"), [
    (r"1/0", "division_by_zero"),
    (r"0^{-1}", "division_by_zero"),
    (r"2^{1/2}", "exponent_not_small_integer"),
    (r"2^{65}", "exponent_not_small_integer"),
    (r"(-1)!", "factorial_domain"),
    (r"(1/2)!", "factorial_domain"),
    (r"101!", "factorial_bound"),
    (r"100!", "too_large"),
    (r"\binom{2}{3}", "binomial_domain"),
    (r"\binom{257}{1}", "binomial_bound"),
    (r"\binom{256}{128}", "too_large"),
    (r"5\bmod 0", "modulo_domain"),
    (r"5\bmod (-2)", "modulo_domain"),
    (r"(1/2)\bmod 2", "modulo_domain"),
    (r"\sqrt{-4}", "root_domain"),
    (r"\sqrt{2}", "non_perfect_root"),
    (r"\sqrt[0]{1}", "root_degree"),
    (r"\sqrt[65]{1}", "root_degree"),
    (r"\sqrt[1/2]{1}", "root_degree"),
    ("9" * 60 + "+1", "too_large"),
    (r"(10^{30})^2", "too_large"),
])
def test_preserves_domain_errors(source, reason):
    with pytest.raises(latex.MathError, match="^" + reason + "$"):
        exact(source)


@pytest.mark.parametrize("source", [
    r"\input{secret}", r"\newcommand{\a}{2}", r"\operatorname{sin}(2)",
    r"\sin(2)", r"\int_0^1 x dx", r"\lim_{x=0}x", r"\pi", r"\infty",
    r"\begin{matrix}1&2\\3&4\end{matrix}", r"1\pmod{3}", r"3!!",
    r"1\frac{1}{2}", r"-1\frac{1}{2}", r"2/3(4)", r"2 3", r"1\,234", r"1,23", r"1,2345", r"1234,567", r"1e3", r"2x",
    r"x+1", r"2=2", r"$2$+$3$", r"\[2\)", r"$2", r"2$", r"$$$2$$$",
    r"\left(2+3)", r"(2+3\right)", r"\left(2+3\right]", r"\left.2\right)",
    r"\frac12", r"\frac{1}", r"\frac{}{2}", r"\sqrt2", r"2^12", r"2^-2",
    r"2^3^2", r"1_2", r"\sum_{i=1+1}^{3}i", r"\sum_{i=1}^{129}i",
    r"\sum_{i=1}^{3}", r"\sum_{i=1}^{3}{j}", r"\sum_{i=1}^{i}i",
    r"__import__('os')", r"1%2", r"1 # comment", r"２+３", "", "()",
])
def test_fail_closed_for_foreign_ambiguous_or_incomplete_input(source):
    with pytest.raises(latex.MathError):
        latex.parse_latex(source)


@dataclass(frozen=True)
class FakeNumber:
    op: str
    args: tuple

    @classmethod
    def literal(cls, text):
        return cls("literal", (text,))


def interpret_ir(node):
    """Independent tiny CPU oracle for the emitted arithmetic, not production code."""
    if node.op == "literal":
        return Fraction(node.args[0])
    args = [interpret_ir(child) for child in node.args]
    if node.op == "neg":
        return -args[0]
    a, b = args
    return {"+": lambda: a+b, "-": lambda: a-b, "*": lambda: a*b,
            "/": lambda: a/b, "^": lambda: a**b.numerator}[node.op]()


@pytest.mark.parametrize("source", [
    r"\frac{1}{3}+\frac{2}{3}", r"2^{-3}", r"5!", r"0!",
    r"\binom{10}{3}", r"\binom{10}{10}", r"\binom{10}{0}",
    r"\sum_{i=-2}^{3}{i^2+1}", r"\prod_{i=1}^{5}{i}",
    r"\sum_{i=0}^{5}{i!}", r"\sum_{i=0}^{5}{\binom{5}{i}}",
    r"\sum_{i=3}^{1}{1/0}", r"\prod_{i=3}^{1}{1/0}",
    r"\sum_{i=1}^{3}{\sum_{j=1}^{2}{i+j}}",
    r"\sum_{i=1}^{2}{\sum_{i=1}^{3}{i}+i}",
])
def test_deferred_expansions_preserve_math(source):
    expr = latex.parse_latex(source)
    ir = latex.to_deferred(expr, FakeNumber, max_nodes=256)
    assert interpret_ir(ir) == latex.evaluate_exact(expr)


def test_parse_and_lower_never_call_reference_or_numeric_functions(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU answer computation occurred during parsing/lowering")
    monkeypatch.setattr(latex, "evaluate_exact", forbidden)
    monkeypatch.setattr(latex, "Fraction", forbidden)
    monkeypatch.setattr(latex, "comb", forbidden)
    monkeypatch.setattr(latex, "factorial", forbidden)
    for source in (r"\frac{1}{3}+\frac{2}{3}", r"5!", r"\binom{8}{3}", r"\sum_{i=1}^{5}i^2"):
        latex.to_deferred(latex.parse_latex(source), FakeNumber, max_nodes=256)
    # Parsing must retain errors for backend evaluation rather than hiding them.
    ir = latex.to_deferred(latex.parse_latex(r"\frac{1}{0}"), FakeNumber)
    assert ir.op == "/" and ir.args[1].args == ("0",)


@pytest.mark.parametrize(("source", "operation"), [
    (r"\sqrt{4}", "root"), (r"5\bmod 3", "mod"),
    (r"(2+3)!", "factorial"), (r"\binom{2+3}{2}", "binomial"),
])
def test_deferred_unsupported_operations_do_not_fall_back(source, operation):
    expr = latex.parse_latex(source)
    with pytest.raises(latex.MathError) as error:
        latex.to_deferred(expr, FakeNumber)
    assert error.value.reason == "unsupported_backend_operation:" + operation


def test_backend_can_explicitly_declare_extended_operations():
    ir = latex.to_deferred(latex.parse_latex(r"\sqrt{4}"), FakeNumber,
                           supported_ops=latex.CORE_OPS | {"root"})
    assert ir.op == "root" and ir.args[0].args == ("4",) and ir.args[1].args == ("2",)


def test_default_number_and_program_compatibility_without_gpu():
    from mlx2.runtime.amoeba_calc_gpu import Number, program
    ir = latex.to_deferred(latex.parse_latex(r"\frac{1}{3}+\frac{2}{3}"))
    assert isinstance(ir, Number)
    _, ops, _, _ = program({"op": "expr", "value": ir})
    assert ops == [0, 0, 4, 0, 0, 4, 1]


def test_bounds_apply_to_syntax_evaluation_and_expansion():
    for source in ("1" * 4097, "(" * 60 + "1" + ")" * 60, "+" * 60 + "1",
                   "+".join(["1"] * 300), "9" * 61, "0." + "0" * 60 + "1"):
        with pytest.raises(latex.MathError):
            latex.parse_latex(source)
    with pytest.raises(latex.MathError, match="deferred_instruction_bound"):
        latex.to_deferred(latex.parse_latex(r"\sum_{i=1}^{128}{i^2}"), FakeNumber)
    with pytest.raises(latex.MathError, match="evaluation_work_bound"):
        exact(r"\sum_{i=1}^{128}{\sum_{j=1}^{128}{i+j}}")
    with pytest.raises(latex.MathError, match="invalid_instruction_bound"):
        latex.to_deferred(latex.parse_latex("1"), FakeNumber, max_nodes=0)


def test_public_ast_is_validated_before_execution():
    for bad in (latex.MathExpr("evil"), latex.MathExpr("literal", text="1e999999"),
                latex.MathExpr("+", (latex.MathExpr("literal", text="1"),)),
                latex.MathExpr("variable", text="secret"), latex.MathExpr("literal", text=1),
                latex.MathExpr("+", []), latex.MathExpr([], ())):
        for operation in (latex.evaluate_exact, lambda e: latex.to_deferred(e, FakeNumber)):
            with pytest.raises(latex.MathError):
                operation(bad)


def test_factorial_binomial_and_perfect_root_properties():
    for n in range(15):
        assert exact(str(n) + "!") == math.factorial(n)
        for k in range(n + 1):
            source = rf"\binom{{{n}}}{{{k}}}"
            assert exact(source) == math.comb(n, k)
            assert interpret_ir(latex.to_deferred(latex.parse_latex(source), FakeNumber, max_nodes=256)) == math.comb(n, k)
    for degree in range(2, 9):
        for base in range(10):
            assert exact(rf"\sqrt[{degree}]{{{base**degree}}}") == base
            if degree % 2:
                assert exact(rf"\sqrt[{degree}]{{{-base**degree}}}") == -base


@pytest.mark.parametrize(("source", "canonical"), [
    (r"\[\left(\dfrac{1}{2}+\tfrac{3}{4}\right)\times 2\]", "(1/2+3/4)*2"),
    (r"\frac{\frac{1}{2}}{\frac{3}{4}}", "1/2/(3/4)"),
    (r"\frac{1}{\frac{2}{3}}", "1/(2/3)"),
    (r"1+(2+3)", "1+(2+3)"),
    (r"1-(2-3)", "1-(2-3)"),
    (r"2(3\times 4)", "2*(3*4)"),
    (r"-(2+3)^2", "-(2+3)^2"),
    (r"(-2)^{12}", "(-2)^{12}"),
    (r"2^{-3}", "2^{-3}"),
    (r"2^{3^2}", "2^{3^2}"),
    (r"(2^3)^2", "(2^3)^2"),
    (r"(3!)!", "(3!)!"),
    (r"(2^3)!", "(2^3)!"),
    (r"(-17)\bmod 5", r"-17\bmod 5"),
    (r"\sqrt[3]{-8}", r"\sqrt[3]{-8}"),
    (r"\sqrt{\frac{81}{16}}", r"\sqrt{81/16}"),
    (r"\binom{2+3}{2}", r"\binom{2+3}{2}"),
    (r"\sum_{i=-2}^{3}{i^2+1}", r"\sum_{i=-2}^{3}{i^2+1}"),
    (r"(\prod_{i=1}^{3}{i})*2", r"(\prod_{i=1}^{3}{i})*2"),
    (r"2/(\sum_{i=1}^{3}{i})", r"2/(\sum_{i=1}^{3}{i})"),
    (r"\sum_{i=1}^{2}{i\bmod i}", r"\sum_{i=1}^{2}{i\bmod i}"),
    (r"\sum_{i=1}^{2}{\sum_{i=1}^{3}{i}+i}", r"\sum_{i=1}^{2}{(\sum_{i=1}^{3}{i})+i}"),
])
def test_canonical_serialization_preserves_operation_tree(source, canonical):
    original = latex.parse_latex(source)
    assert latex.canonical_expression(original) == canonical
    assert latex.parse_latex(canonical) == original


def test_semantic_length_cap_follows_formatting_normalization():
    source = "1"
    for _ in range(8):
        source = r"\dfrac{1}{\left(" + source + r"\right)}"
    source = r"\[\displaystyle " + source + r"\]"
    assert len(source) > 160
    syntax = latex.parse_latex(source)
    normalized = latex.canonical_expression(syntax)
    assert len(normalized) < 160
    assert not any(macro in normalized for macro in (r"\dfrac", r"\left", r"\right", r"\displaystyle"))
    assert latex.parse_latex(normalized) == syntax
    assert exact(normalized) == exact(source)


def test_canonicalization_preserves_long_numbers_and_long_semantic_formula():
    number = "1234567890" * 6
    source = r"\left(" + number + "+" + number + "+" + number + r"\right)"
    canonical = latex.canonical_expression(latex.parse_latex(source))
    assert canonical == "+".join([number] * 3)
    assert len(canonical) > 160
    assert latex.parse_latex(canonical) == latex.parse_latex(source)


@pytest.mark.parametrize("source", [
    r"\displaystyle 1\unknown{2}", r"\left(1+2\right)\input{3}",
    r"1\fracx{1}{2}", r"\dfrac{1}{\left(2\mystery 3\right)}",
])
def test_unknown_macros_never_disappear_into_canonical_formula(source):
    with pytest.raises(latex.MathError, match="unsupported_latex_command"):
        latex.canonical_expression(latex.parse_latex(source))


def test_canonicalization_does_not_evaluate_or_expand_indices(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("canonicalization performed numerical evaluation")
    monkeypatch.setattr(latex, "evaluate_exact", forbidden)
    monkeypatch.setattr(latex, "Fraction", forbidden)
    monkeypatch.setattr(latex, "comb", forbidden)
    monkeypatch.setattr(latex, "factorial", forbidden)
    source = r"\sum_{i=-2}^{3}{\binom{5}{i}+i!+\frac{1}{0}}"
    syntax = latex.parse_latex(source)
    canonical = latex.canonical_expression(syntax)
    assert "i!" in canonical and r"\sum_{i=-2}^{3}" in canonical and "1/0" in canonical
    assert latex.parse_latex(canonical) == syntax
