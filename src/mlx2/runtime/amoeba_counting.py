# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Bounded literal character/list counting and exact explicit-list summation.

Original implementation. Request extraction is host-side; selected GPU execution
compares/reduces ASCII characters on device or uses the existing exact Metal sum.
No learned counting gate or general semantic extraction is claimed.
"""
from __future__ import annotations
import ast
import re
from fractions import Fraction

MAX_TEXT=16384
MAX_ITEMS=128
_QUOTED=r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')'''
_TARGET=rf'(?:{_QUOTED}|[A-Za-z])'
_SOURCE=rf'(?:{_QUOTED}|[A-Za-z]+)'
_MODE=r'(?:\s+(?P<mode>case[- ]sensitive|case[- ]insensitive|ignore case))?'
_CHAR_PATTERNS=[
 rf"how many\s+(?:(?:letters?|characters?)\s+)?(?P<target>{_TARGET})(?:'?s)?\s+(?:(?:are there|are|occur|appear)\s+)?in\s+(?:the\s+(?:word|text|string)\s+)?(?P<source>{_SOURCE})",
 rf"count\s+(?:the\s+)?(?:occurrences\s+of\s+)?(?P<target>{_TARGET})\s+in\s+(?:the\s+(?:word|text|string)\s+)?(?P<source>{_SOURCE})",
 rf"how many times does\s+(?:the\s+letter\s+)?(?P<target>{_TARGET})\s+(?:occur|appear)\s+in\s+(?:the\s+(?:word|text|string)\s+)?(?P<source>{_SOURCE})",
]
_NUM=re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)')


def _literal(s):
    value=ast.literal_eval(s) if s[:1] in ('\"',"'") else s
    if not isinstance(value,str):raise ValueError('counting_literal')
    return value


def parse_request(text):
    if not isinstance(text,str) or len(text)>MAX_TEXT: return None
    text=text.strip().rstrip('?. ')
    for pattern in _CHAR_PATTERNS:
        match=re.fullmatch(pattern+_MODE,text,re.I|re.S)
        if match:
            try:source,target=_literal(match['source']),_literal(match['target'])
            except (ValueError,SyntaxError):return None
            if len(target)!=1:return None
            return dict(operation='count_character',source=source,target=target,
                        case_sensitive=bool(match['mode'] and match['mode'].lower() in ('case-sensitive','case sensitive')))
    match=re.fullmatch(r'(?:how many (?:numbers|items|entries|terms)(?: are)? in (?:the list\s*)?|count (?:the )?(?:items|entries|terms) in\s*)\[(?P<body>[^\[\]]*)\]',text,re.I)
    kind='count_entries'
    if match is None:
        match=re.fullmatch(r'(?:what is the (?:sum|total) of\s*|(?:sum|total|add)\s+(?:(?:the following numbers|the numbers|these numbers|the list)\s*:?\s*)?)\[(?P<body>[^\[\]]*)\]',text,re.I)
        kind='sum_entries'
    if match is None:return None
    body=match['body'].strip();values=[x.strip() for x in body.split(',')] if body else []
    if len(values)>MAX_ITEMS or any(not _NUM.fullmatch(x) for x in values):return None
    if any(sum(c.isdigit() for c in x)>60 for x in values):return None
    return dict(operation=kind,source=body,values=values)


def evaluate(spec,backend='gpu'):
    if backend not in ('cpu','gpu','gemm'):raise ValueError('unknown counting backend')
    op=spec['operation'];info=dict(operation=op,arithmetic_backend=backend)
    if op=='sum_entries':
        if backend=='cpu':
            value=Fraction(0)
            for text in spec['values']:
                value+=Fraction(text)
                if max(len(str(abs(value.numerator))),len(str(value.denominator)))>60:return None,dict(refused='too_large')
            from .amoeba_det_tiles import _decimal_string
            answer=_decimal_string(value);info['arithmetic_execution']='cpu_reference'
        else:
            from .amoeba_calc_gpu import Number,evaluate as gpu_evaluate
            value=None
            for text in spec['values']:
                negative=text.startswith('-');text=text.lstrip('+-')
                if text.startswith('.'):text='0'+text
                if text.endswith('.'):text+='0'
                term=Number.literal(text);term=-term if negative else term
                value=term if value is None else value+term
            if value is None:value=Number.literal('0')
            expr=dict(op='expr',value=value);answer,why=gpu_evaluate(expr,backend)
            if answer is None:return None,dict(refused=why)
            info['arithmetic_execution']=expr['arithmetic_execution']
        return answer,info
    if backend=='cpu':
        if op=='count_entries':count=len(spec['values'])
        elif op=='count_character':
            target=spec['target'];source=spec['source']
            count=sum(c==target for c in source) if spec['case_sensitive'] else sum(c.casefold()==target.casefold() for c in source)
        else:raise ValueError('unknown counting operation')
        return str(count),dict(**info,arithmetic_execution='cpu_reference')
    # A GPU route refuses unsupported Unicode instead of silently counting on CPU.
    if not spec['source'].isascii() or (op=='count_character' and not spec['target'].isascii()):
        return None,dict(refused='gpu_count_ascii_only')
    import mlx.core as mx
    with mx.stream(mx.gpu):
        data=mx.array(list(spec['source'].encode('ascii')),dtype=mx.int32)
        if op=='count_entries':
            count=mx.sum((data==44).astype(mx.int32))+mx.minimum(mx.array(data.size,dtype=mx.int32),mx.array(1,dtype=mx.int32))
        elif op=='count_character':
            target=mx.array(ord(spec['target']),dtype=mx.int32)
            if not spec['case_sensitive']:
                data=mx.where((data>=65)&(data<=90),data+32,data)
                target=mx.where((target>=65)&(target<=90),target+32,target)
            count=mx.sum((data==target).astype(mx.int32))
        else:raise ValueError('unknown counting operation')
        answer=str(int(count.item()))
    return answer,dict(**info,arithmetic_execution='gpu_integer_reduction')
