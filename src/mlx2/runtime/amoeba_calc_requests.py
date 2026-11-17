# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Strict complete literal requests for the calculator's optional request gate.

The original user expression is parsed once, before a model can round/copy its
operands. This is a deterministic expression/count route, not a word-problem
solver. It never receives benchmark answers or an evaluator's expression field.
"""
from __future__ import annotations
import re
from . import amoeba_counting as counting
from . import amoeba_math_latex as math

_INTRO=re.compile(r'(?is)^(?:(?:please\s+)?(?:calculate|compute|evaluate|simplify)(?:\s+exactly)?\s*:?[ \t]+|what is\s+|(?:find|give|determine)\s+the\s+(?:(?:exact|numerical)\s+)?(?:value|result)\s+of\s+|work out\s+(?:the following\s+)?(?:without rounding\s*)?:?\s*)(.+)$')
_VERBATIM_INTRO = re.compile(
    r"^\s*(?:please\s+)?(?:repeat|copy|quote|transcribe)\s+"
    r"(?:exactly|verbatim|this text|the following text|the exact text)\b",
    re.IGNORECASE,
)


def requests_verbatim_copy(text):
    """A standalone exact-copy instruction must not acquire an answer slot."""
    return isinstance(text, str) and _VERBATIM_INTRO.match(text) is not None


def request_text_from_payload(request):
    """Use only the current complete user turn; never drop tools or image parts.

    A tool/assistant continuation must not reuse an older user's calculation.
    The strict expression parser still decides whether this text is supported.
    """
    if not isinstance(request, dict):
        return None
    if 'messages' in request:
        messages = request['messages']
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
            return None
        latest = messages[-1]
        if latest.get('role') != 'user':
            return None
        content = latest.get('content')
        if isinstance(content, list):
            if not content or any(not isinstance(p, dict) or p.get('type') not in ('text', 'input_text')
                                  or not isinstance(p.get('text'), str) for p in content):
                return None
            content = '\n'.join(p['text'] for p in content)
    else:
        content = request.get('prompt')
    return content if isinstance(content, str) and len(content) <= math.MAX_SOURCE else None


def parse_request(text):
    if not isinstance(text,str) or len(text)>math.MAX_SOURCE:return None
    count=counting.parse_request(text)
    if count:return dict(kind='counting',spec=count)
    text=text.strip()
    m=_INTRO.fullmatch(text)
    if m:text=m[1].strip()
    elif not text.startswith(('\\(', '\\[', '$')):return None
    if text.endswith(('.', '?')):text=text[:-1].rstrip()
    if '\\' not in text and '$' not in text:
        # Plain arithmetic accepts multi-digit signed powers. LaTeX continues
        # to require braces; do not reinterpret malformed control sequences.
        text=re.sub(r'\^([+-]?[0-9]+)(?![0-9.])',r'^{\1}',text)
    try:
        tree=math.parse_latex(text)
        canonical=math.canonical_expression(tree)
        if len(canonical)>160:return None
        # Refuse a bare number/variable: the request must actually ask for work.
        def operation(n):return n.op not in ('literal','variable','neg') or any(operation(a) for a in n.args)
        if not operation(tree):return None
        return dict(kind='arithmetic',tree=tree,expression=text,normalized=canonical)
    except math.MathError:return None


def evaluate(request,backend='gpu'):
    if request['kind']=='counting':
        return counting.evaluate(request['spec'],backend)
    try:
        if backend=='cpu':
            from .amoeba_det_tiles import _decimal_string
            answer=_decimal_string(math.evaluate_exact(request['tree']))
            execution='cpu_reference'
            why='non_terminating_quotient' if answer is None else None
        elif backend in ('gpu','gemm'):
            from .amoeba_calc_gpu import evaluate as gpu_evaluate
            expr=dict(op='expr',value=math.to_deferred(request['tree'],max_nodes=256))
            answer,why=gpu_evaluate(expr,backend)
            execution=expr.get('arithmetic_execution')
        else:raise ValueError('unknown arithmetic backend')
    except math.MathError as error:return None,dict(refused=error.reason)
    if answer is None:return None,dict(refused=why)
    return answer,dict(operation='literal_expression',expr=request['expression'],normalized=request['normalized'],
                       arithmetic_backend=backend,arithmetic_execution=execution)
