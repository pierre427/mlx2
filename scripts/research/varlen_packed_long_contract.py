"""Pinned maintained prompt input and separate long physical-proof assertions."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROMPT_RECEIPT=ROOT/'qualification/runs/stock27b-wall-7df61ae6-20261004/receipt.json'
PROMPT_SHA='036a302ae05733aaed31fb648d383c5d7582e08f49ba031b5061b08008dc2903'
COUNTS=(6950,6929)


def prompt_ids(path=PROMPT_RECEIPT):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROMPT_SHA:raise ValueError('maintained long prompt receipt identity differs')
    rows=json.loads(raw)['prompts']
    ids=tuple(tuple(row['token_ids']) for row in rows)
    if tuple(map(len,ids))!=COUNTS or any(type(t) is not int or t<0 for lane in ids for t in lane):
        raise ValueError('maintained long prompt tokens differ')
    return ids


def exact_prompts(adapter):
    result=[]
    for ids in prompt_ids():
        text=adapter.tokenizer.decode(list(ids))
        if tuple(adapter.prompt_tokens({'prompt':text}))!=ids:
            raise ValueError('maintained whole long prompt HTTP text does not roundtrip exact IDs')
        result.append((text,ids))
    return tuple(result)


def physical_prefill(depth=16):
    return {'grouped_multirow_write_count':depth,'grouped_multirow_row_count':sum(COUNTS)*depth,
        'prefill_matrix_dispatch_count':depth,'prefill_long_nax_dispatch_count':depth}


def validate_q1(proof,width,depth=16):
    if (proof.get('stock_long_selected') is not True or proof.get('native_tile_dispatches')!=0 or
        proof.get('native_stock_reduction_dispatches')!=0 or proof.get('native_split_partial_dispatches')!=0 or
        proof.get('native_stock_long_partial_dispatches')!=(depth if width==2 else 0) or
        proof.get('native_stock_long_reduce_dispatches')!=(depth if width==2 else 0) or
        proof.get('grouped_q1_writes')!=(depth if width==2 else 0) or
        (width==1 and (proof.get('scalar_native_writes')!=proof.get('expected_scalar_write_spans') or
                       type(proof.get('scalar_native_writes')) is not int or proof['scalar_native_writes']<=0))):
        raise RuntimeError('long Q1 actual B2 stock-long/B1 scalar physical proof differs')


def validate_physical(snapshot,depth,*unused):
    if (snapshot.get('q1_stock_long_partial_dispatches')!=depth or snapshot.get('q1_stock_long_reduce_dispatches')!=depth or
        any(snapshot.get(k,0)!=0 for k in ('q1_stock_reduction_dispatches','q1_stock_singleton_dispatches',
            'q1_tile_dispatches','q1_split_partial_dispatches','q1_split_reduce_dispatches'))):
        raise RuntimeError('long HTTP physical continuation counters differ')
