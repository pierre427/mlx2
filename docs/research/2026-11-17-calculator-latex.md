# Bounded LaTeX arithmetic frontend

`src/mlx2/runtime/amoeba_math_latex.py` implements an exact-rational arithmetic
subset of LaTeX notation. It parses one complete formula into an immutable AST;
it does not execute TeX, expand user macros, extract prose, or decide when a
serving route should trigger. This is implemented CPU-tested frontend behavior,
not whole-model qualification. Integration now has separate GPU evidence and model accuracy profiles in `2026-11-17-calculator-results.md`; this frontend alone does not select a serving default.

## API and integration

```python
from mlx2.runtime.amoeba_math_latex import (
    MathError, parse_latex, evaluate_exact, to_deferred,
)

syntax = parse_latex(r"\frac{1}{3}+\frac{2}{3}")
reference = evaluate_exact(syntax)  # Explicit CPU reference: Fraction(1, 1).
ir = to_deferred(syntax, max_nodes=64)  # Number IR; does not evaluate answers.
expression = {"op": "expr", "value": ir}  # Existing GPU program input contract.
```

The caller first extracts a complete math span and removes any equality/result
cue. `parse_latex` accepts an optional single outer `$...$`, `$$...$$`,
`\(...\)`, or `\[...\]` wrapper. It rejects prose, equality signs, partial
formulas, and any remaining tokens. It returns `MathExpr(op, args, text)`;
`args` is a tuple of child nodes, and `text` is a literal's spelling or a bound
variable's one-letter name. All refusals raise `MathError`, whose `.reason`
is suitable for a route receipt. Do not catch an unsupported backend operation
and run the CPU reference on a selected GPU route.

`to_deferred(expr, number_type=None, *, supported_ops=CORE_OPS, max_nodes=64)`
constructs the existing `amoeba_calc_gpu.Number` by default. A custom type must
provide `.literal(str)` and a constructor `(op, tuple_of_child_nodes)`.
`CORE_OPS` is exactly `literal,+,-,*,/,^,neg`. Backend capabilities must be
explicit; advertising a new operation does not implement or qualify it.
The caller must align `max_nodes` with the backend instruction cap. It counts
all emitted literal and operation nodes, including repeated subexpressions.
There are no imports of MLX, GPU calls, or numerical answer computations in
parsing or lowering. In particular, parsing division by zero still produces a
division node; the execution backend owns that domain error.

## Formula-length normalization

`canonical_expression(expr) -> str` serializes a parsed AST into compact strict
arithmetic syntax. For a 160-character semantic formula cap, the caller should
extract a complete raw candidate, retain its raw bound/truncation provenance,
parse it with `parse_latex`, and then check
`len(canonical_expression(expr)) <= 160` before evaluation/lowering. Do not cap
the raw candidate at 160 first: `\dfrac`, `\left`, style commands, and nested
fraction formatting can inflate a small formula beyond that length. The frontend's
independent hard limits remain **4096 raw characters, 1024 tokens, 512 AST nodes,
and depth 48**. The serializer does not change those parsing admission limits.

Canonicalization removes formatting by serializing only known AST operations,
uses `/` for fractions and explicit `*` for implicit multiplication, and keeps
parentheses needed to preserve the exact operation tree. It preserves literal
digits, exponent scope, unary precedence, finite-loop index names/bounds, and
nested-loop scope. Right-hand addition/multiplication groups remain grouped,
because intermediate bounds are observable. Loop expressions used as operands
are parenthesized to prevent a following multiplication from entering the loop
body. No sums, factorials, roots, fractions, or binomials are evaluated or
expanded for this length check. Real formulas longer than 160 after serialization
remain longer; long numbers are never shortened or rounded. Unknown commands
fail strict parsing and cannot disappear into a valid canonical expression.

## Capability matrix

| Input / operation | Explicit CPU exact reference | Existing GPU core IR |
| --- | --- | --- |
| Integers and terminating decimal literals, `+ - * /` | Exact `Fraction` | Deferred literals/arithmetic |
| `\frac`, `\dfrac`, `\tfrac`, `\cfrac` | Exact division | Deferred division |
| `\times`, `\cdot`, `\div`; Unicode multiplication/division/minus | Supported | Deferred core operation |
| `{...}`, `(...)`, `[...]`; matched `\left...\right` | Grouping only | Grouping only |
| Integer powers, unary signs | Exact, exponent magnitude at most 64 | Deferred power; backend validates exponent |
| `n!`, `\binom{n}{k}`, style variants | Exact, bounded integer arguments | Literal integer arguments (including substituted loop indices) expand into multiplication/division; computed arguments refuse unless explicitly supported |
| `a\bmod m` | Integer `a`, positive integer `m`; Euclidean remainder | Refuses `unsupported_backend_operation:mod` |
| `\sqrt{x}`, `\sqrt[n]{x}` | Exact perfect rational roots; odd roots of negative values allowed | Refuses `unsupported_backend_operation:root`, even for a perfect literal root |
| `\sum_{i=a}^{b}{body}`, `\prod_{i=a}^{b}{body}` | Finite exact evaluation, bounded work | Literal bounds expand syntactically; arithmetic remains deferred |
| Elementary transcendental functions, irrational constants/roots | Unsupported; no approximation type is provided | Unsupported |
| Matrices, symbolic algebra, integration, limits, equations, proofs | Unsupported | Unsupported |

GPU support above describes compatibility with the arithmetic IR. Targeted GPU fraction/factorial/binomial/sum/product checks passed in the 478-case arithmetic comparison; the model-level route remains unqualified, with separate accuracy regressions under repair. The reference returns
exact fractions including `1/3`; the existing GPU display layer may refuse a
nonterminating decimal result. Rational intermediates remain deferred. GPU digit
and instruction bounds can also refuse a mathematically valid expression.

Literal factorial expansion creates factors, never the answer. Literal binomial
expansion uses the recurrence `v = v*(n-k+j)/j` with `k=min(k,n-k)`, avoiding
factorial-sized intermediates. The GPU still applies its own intermediate digit
bound: a binomial accepted by the CPU reference can exceed that bound during
its deferred multiplicative recurrence. Sums/products substitute index literals
and emit addition/multiplication. No host `Fraction`, `comb`, or `factorial`
operation is used by `to_deferred`; tests replace those functions and the CPU
reference with failures to enforce that boundary.

## Grammar and semantic limits

- The source is limited to 4096 characters, 1024 tokens, 512 AST nodes and depth
  48. Each numerator/denominator is limited to 60 digits; the CPU reference
  checks each rational intermediate. Powers have integer exponents in `[-64,64]`;
  `0^0` is 1, matching the calculator convention. A negative power of zero is a
  division-by-zero error.
- Fraction, binomial, and root radicand arguments require braces. Root degree
  goes in optional square brackets and must evaluate to an integer from 2 to
  64. A nonperfect rational root is `non_perfect_root`, not a rounded value.
- Unbraced superscripts contain a single digit or a bound variable. Write
  `2^{12}`, `2^{-3}`, or `2^{3^2}`. `2^12`, `2^-3`, and `2^3^2` refuse rather
  than guess their TeX scope. Unary minus binds below powers: `-2^2` is -4.
  Repeated factorial signs (double factorial notation) are unsupported.
- Factorials require nonnegative integers at most 100, subject to the digit
  bound. Binomials require integers `0 <= k <= n <= 256`, also subject to the
  digit bound. Generalized binomials are unsupported. Modulo requires integers
  and a strictly positive modulus.
- Sums/products require literal integer bounds in the order `_{i=a}^{b}`.
  Bounds cannot depend on outer loop variables or computed expressions.
  There are at most 128 terms per loop and 8192 total visited nodes during CPU
  evaluation. Descending ranges are empty (sum 0, product 1), and an empty
  body is not evaluated. Nested one-letter indices have lexical scope,
  including correct shadowing. An unbraced body consumes one multiplicative
  term: `\sum_{i=1}^{3}i^2+10` is 24; braces include an additive body.
- Implicit multiplication is accepted for grouping and bound variables, for
  example `2(3+4)` or `2i` within a sum. Adjacent numbers, mixed-number
  `1\frac{1}{2}` notation, and implicit multiplication after a slash in the
  same term refuse as ambiguous. Use explicit multiplication or grouping.
  Strict three-digit comma grouping is supported in the repaired frontend; malformed grouping refuses. Scientific notation remains unsupported.
- Only the documented math commands, common math spacing, and four math style
  commands are accepted. These formatting commands never alter semantics.
  `\left` and `\right` must pair around matching parentheses/brackets/braces;
  invisible/mismatched/bar delimiters are unsupported. Unknown macros,
  `\begin` environments, comments, free variables, relation operators,
  subscripts other than finite-loop limits, and arbitrary Unicode refuse.
- Public AST entry points validate structure, operators, bounds and arity before
  evaluation/lowering. Consumers should still use `parse_latex` to obtain
  canonical literal spellings and lexical variable scoping.

Representative refusal reasons include `division_by_zero`, `too_large`,
`exponent_not_small_integer`, `factorial_domain`, `binomial_domain`,
`modulo_domain`, `root_domain`, `root_degree`, `non_perfect_root`,
`term_bound`, `evaluation_work_bound`, `deferred_instruction_bound`, and
`unsupported_backend_operation:<op>`. Syntax errors also carry a reason,
but callers should not depend on the exact ordering of syntax diagnostics.

## Source and validation scope

Original project implementation; no upstream implementation was mined or copied.
Notation was checked against the user-supplied
[Wikibooks LaTeX/Mathematics reference](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
on 2026-11-17, particularly powers/grouping, fractions/binomials, roots,
modulo, and finite-operator notation. This is a deliberately smaller arithmetic
grammar, not support for the whole typesetting language.

CPU validation lives in `tests/runtime/test_amoeba_math_latex.py`, including
exact functional examples, domain errors, hostile/ambiguous syntax, expansion
parity with an independent arithmetic oracle, nested lexical scope, resource
bounds, root/binomial/factorial properties, and a no-CPU-evaluation assertion for
parsing/lowering. Model checkpoint: N/A (no model loaded). Source revision:
implementation commit reported with delivery; base `c8fbacef6` plus only this
module, test file, and documentation. Run from the isolated checkout with:

```sh
PYTHONPATH="$PWD/src" .venv/bin/python -m pytest tests/runtime/test_amoeba_math_latex.py
```

No model/GPU jobs or production services are part of these tests.

License: covered original calculator documentation is under [Pierre Lamy Calculator NC-ND 1.0](../../provenance/LICENSE-calculator.txt); [scope and earlier grants](../../provenance/LICENSE-calculator-scope.json).
