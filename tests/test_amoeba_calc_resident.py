"""Compile the *same* Metal parser/VM as C++: no MLX or GPU required."""

import ctypes
from fractions import Fraction
from pathlib import Path
import random
import subprocess

import pytest

from mlx2.runtime import amoeba_calc_gpu as G
from mlx2.runtime import amoeba_calc_resident as R


@pytest.fixture(scope="module")
def portable(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("resident-cpp")
    source = tmp / "oracle.cpp"
    lib = tmp / "oracle.dylib"
    source.write_text(
        """#include <cstdint>
using uchar=unsigned char; using uint=unsigned int;
#define thread
#define device
#define constant
#define RESIDENT_ARITHMETIC 1
"""
        + G._HEADER
        + R.header()
        + """
extern "C" int calculate(const uchar *s,int len,int mode,int *out) {
 int program[1025]={}; uchar normalized[4096]={};
 resident_parse(s,len,mode,program,normalized);
 resident_evaluate(program,normalized,out);
 return program[0];
}
extern "C" int match_piece(const uchar *answer, int size, const uchar *piece,
                           int length, int cursor, int leading, int ordinary) {
 int state[288]={}, offsets[32]={}, lengths[32]={}, classes[32]={};
 uchar pool[64]={};
 if (size < 1 || size >= 32 || length < 0 || size + length > 64) return -1;
 state[0]=size; state[1]=cursor;
 state[11]=leading ? RC_ALLOW_LEADING_SPACE : RC_MATCH_ANSWER_BYTES;
 for (int i=0;i<size;i++) {
  pool[i]=answer[i]; offsets[i]=i; lengths[i]=1; state[16+i]=i;
 }
 offsets[size]=size; lengths[size]=length;
 classes[size]=ordinary ? RC_ANSWER_PIECE : 0;
 for (int i=0;i<length;i++) pool[size+i]=piece[i];
 return rc_match_answer_piece(state,size,pool,offsets,lengths,classes,size+1);
}
"""
    )
    subprocess.run(
        [
            "clang++",
            "-std=c++17",
            "-O1",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(lib),
        ],
        check=True,
        capture_output=True,
    )
    dll = ctypes.CDLL(str(lib))
    fn = dll.calculate
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    fn.restype = ctypes.c_int
    match_piece = dll.match_piece
    match_piece.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    match_piece.restype = ctypes.c_int

    def run(text, mode=0):
        data = text.encode()
        out = (ctypes.c_int * 268)()
        n = fn(data, len(data), mode, out)
        if out[0]:
            return None, n, out[0]
        digits = "".join(str(x) for x in list(out)[4 : 4 + out[3]][::-1])
        scale = out[2]
        if scale:
            digits = digits.rjust(scale + 1, "0")
            digits = digits[:-scale] + "." + digits[-scale:]
        return ("-" if out[1] < 0 else "") + digits, n, 0

    def match(answer, piece, cursor=0, leading=True, ordinary=True):
        answer, piece = answer.encode(), piece.encode()
        return match_piece(answer, len(answer), piece, len(piece), cursor, leading, ordinary)

    run.match_answer_piece = match
    return run


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Calculate (19+8)*7.", "189"),
        ("12,345,678,901 * 9", "111111110109"),
        ("0.1+0.2", "0.3"),
        ("-.25 * 8", "-2"),
        ("2^-3", "0.125"),
        ("-2^2", "-4"),
        ("(-2)^2", "4"),
        ("2^3^2", "512"),
        ("0!", "1"),
        ("10!", "3628800"),
        (r"\(\dfrac{3}{8}+0.125\)", "0.5"),
        (r"\frac{\frac{3}{4}+\frac{1}{8}}{\frac{7}{2}}", "0.25"),
        (r"\left(\frac{8}{4}+3\right)^2", "25"),
        (r"\binom{20}{3}", "1140"),
        ("12 × 4 ÷ 3 − 1", "15"),
        ("What is 4 + 5?", "9"),
        ("1 / 3 * 3", "1"),
        (r"Determine the numerical result of \(\dfrac{7}{2}+1\).", "4.5"),
        ("Find the exact value of ((12-5)*7).", "49"),
        (r"Evaluate exactly: \(\frac{4}{2}\).", "2"),
        ("Work out the following without rounding: 4 / 8.", "0.5"),
        (r"\dfrac{" + r"\left( " * 20 + "3" + r" \right)" * 20 + r"}{4}", "0.75"),
        (r"\sum_{i=1}^{10}{i^2}", "385"),
        (r"\prod_{k=1}^{5}{k}", "120"),
        (r"\prod_{k=1}^{5}{k+1}", "720"),
        (r"\sum_{i=-3}^{3}{i}", "0"),
        (r"\sum_{i=1}^{4}{\frac{i}{4}}", "2.5"),
        ("What is 7 multiplied by 12?", "84"),
        ("What is 3 × 12?", "36"),
        ("What is 18 divided by 4?", "4.5"),
        (r"\begin{array}{r}109 \\ \times 71 \\ \hline", "7739"),
        ("How many r's are in strawberry?", "3"),
        ("Count 's' in 'Mississippi'.", "4"),
        ("Count 'r' in 'stRawbeRry' case-sensitive.", "1"),
        ("Count 'r' in 'stRawbeRry' ignore case.", "3"),
        ("How many items are in []?", "0"),
        ("How many items are in [4, 5, 4, 9]?", "4"),
        ("What is the sum of [0.1, 0.2, 6.5]?", "6.8"),
        (
            "What is the sum of [123456789123.12345678, 0.00000001]?",
            "123456789123.12345679",
        ),
    ],
)
def test_exact(portable, text, expected):
    answer, n, error = portable(text)
    assert answer == expected, (n, error)


@pytest.mark.parametrize(
    "text",
    [
        "sin(2)+3",
        r"\sqrt{4}+3",
        r"\unknown{2}+3",
        "x+2+3",
        "2+3;4+5",
        "Repeat exactly: 2+3",
        "Do not calculate 2+3",
        "2+3 garbage",
        "1 2+3",
        "(2+3",
        "2+3)",
        r"\left(2+3)",
        r"\frac{2}{3",
        "1,23+4",
        "1,234,56+2",
        "2(3+4)",
        "2**3",
        "2=3=",
        "1e6+2",
        "1/0",
        "1/3",
        "2^0.5",
        "2^65",
        "(-2)!",
        r"\binom{3}{4}",
        "9" * 61 + "*0",
        "1" + "+1" * 300,
        "$2+3",
        r"\(2+3\]",
        r"\timesi+3",
        "2%3",
        "12 metres + 3 metres",
        r"\sum_{i=1}^{2}{\sum_{j=1}^{2}{i+j}}",
        r"\sum_{i=1}^{129}{i}",
        r"1\quad2",
        r"1\,2",
        r"\sum_{i=1}^{3}{j}",
        "Count 'r' in 'strawberry' then multiply by two",
        "Sum [1,,2]",
        "Sum [1,2,]",
        "Sum [1,2+3]",
        "Count 'é' in 'café'",
        "How many items are in [1,cat]?",
        "2026-11-17",
        "555-1234",
        "12/25/2024",
        "7",
        "-7",
        "2@1",
        "3!!",
        "3! !",
        r"\FRAC{2}{3}+1",
        r"\frac{2}{3}\Times 3",
        r"\sum_{i=4}^{1}{\sqrt{i}}+2",
        r"\begin{array}{r}1 \\ \times x \\ \hline",
    ],
)
def test_refusal_never_yields_a_result(portable, text):
    answer, _, error = portable(text)
    assert answer is None and error


@pytest.mark.parametrize("text", [r"$3+4=$", r"\(3+4=\)", r"\[3+4=\]", "3+4= "])
def test_merged_delimiter_cues(portable, text):
    assert portable(text, 1)[0] == "7"


@pytest.mark.parametrize(
    "text,mode,expected",
    [
        ("- 90 - 40 =", 1, "50"),
        ("  -\t90 - 40 =", 1, "50"),
        (r"- \(90 - 40=\)", 1, "50"),
        ("- -90 - 40 =", 1, "-130"),
        ("-90 - 40 =", 1, "-130"),
        (r"$- 90 - 40=$", 1, "-130"),
        ("− 90 - 40 =", 1, "-130"),
        ("Calculate - 90 - 40.", 0, "-130"),
        ("- 90 =", 1, None),
    ],
)
def test_generated_bullet_and_unary_minus(portable, text, mode, expected):
    assert portable(text, mode)[0] == expected


@pytest.mark.parametrize(
    "answer,piece,cursor,leading,expected",
    [
        ("96", " ", 0, True, 0),
        ("96", " 96", 0, True, 2),
        ("96", "96", 0, True, 2),
        ("96", "\t9", 0, True, 1),
        ("96", "95", 0, True, -1),
        ("96", "960", 0, True, -1),
        ("96", "9 6", 0, True, -1),
        ("96", "96 years", 0, True, -1),
        ("96", " ", 0, False, -1),
        ("96", " 6", 1, False, -1),
        ("96", "6", 1, False, 1),
        ("-0.5", " -0.", 0, True, 3),
        ("-0.5", "5", 3, False, 1),
    ],
)
def test_answer_pieces_preserve_only_correct_bytes(portable, answer, piece, cursor, leading, expected):
    assert portable.match_answer_piece(answer, piece, cursor, leading) == expected


def test_answer_pieces_do_not_accept_special_tokens(portable):
    assert portable.match_answer_piece("96", "96", ordinary=False) == -1


def test_differential_exact_compositions(portable):
    rng = random.Random(171117)
    for _ in range(300):
        a, b, c, d = [rng.randrange(1, 100000) for _ in range(4)]
        # Exact rational cancellation tests intermediate nonterminating fractions.
        expr = f"(({a}/{b})+({c}/{d}))*({b}*{d})"
        expected = str(a * d + c * b)
        assert portable(expr)[0] == expected, expr


def test_no_literal_truncation_or_suffix_salvage(portable):
    assert portable("unsupported(" + (" " * 4000) + "2+3=)", 1)[0] is None
    assert portable("1" * 4097)[0] is None
