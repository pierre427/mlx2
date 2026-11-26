# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Owned-window GPU counting and whole-request differential checks; no model."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['MLX_METAL_FAST_SYNCH']='0'
    from mlx2.runtime import amoeba_calc_requests as R,amoeba_counting as C
    rng=random.Random(20261118)
    questions=["How many r's are in strawberry?",'Count items in []','Sum []',
               'Sum ['+','.join(str(i) for i in range(128))+']',
               "Count 'r' in 'stRawbeRry' case-sensitive.",
               'Calculate ((123456789123*987654321987)+3)/4.',
               r'Calculate \frac{12,345}{8}+5!.',r'Compute \sum_{i=1}^{8}{(3\times i)^2}.']
    for _ in range(40):
        text=''.join(rng.choice('aAbrR ?') for _ in range(rng.randrange(0,2000)))
        questions.append('Count "r" in '+json.dumps(text)+' ignore case.')
        questions.append('Sum ['+', '.join(f'{rng.randrange(-99999,99999)}.{rng.randrange(10000):04d}' for _ in range(rng.randrange(1,30)))+']')
    rows=[]
    for q in questions:
        request=R.parse_request(q)
        if request is None:raise AssertionError(q)
        expected,_=R.evaluate(request,'cpu');actual,info=R.evaluate(request,'gpu')
        rows.append(dict(question=q,expected=expected,answer=actual,info=info,correct=actual==expected and actual is not None))
        if not rows[-1]['correct']:raise AssertionError(rows[-1])
    # A GPU route must refuse Unicode rather than silently run CPU counting.
    actual,info=C.evaluate(C.parse_request('Count "é" in "été"'),'gpu')
    assert actual is None and info['refused']=='gpu_count_ascii_only'
    import mlx.core as mx
    from transformers import AutoTokenizer
    from mlx2.runtime.amoeba_calc_trigger import GPUCueWindow
    snapshot=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    cues=[]
    for text,want in [("Count 'r' in 'strawberry' =",True),('Sum [0.1, 0.2] = ',True),('Ordinary prose about a forest.',False)]:
        window=GPUCueWindow(tok,allow_symbolic=True,allow_counting=True)
        ids=tok.encode(text,add_special_tokens=False)
        got=bool(window.update(mx.array(ids,dtype=mx.uint32),mx.array([[False]])).item())
        assert got==want
        cues.append(dict(text=text,trigger=got))
    (args.out/'cues.json').write_text(json.dumps(cues,indent=2))
    (args.out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    receipt=dict(checkpoint='N/A (synthetic numerical kernels; no model loaded)',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 source_files={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (R,C)},
                 correct=len(rows),total=len(rows),unicode_gpu_refusal=True,tokenizer='Qwen/Qwen3-0.6B@c1899de2',cue_checks=len(cues),qualification='unqualified')
    (args.out/'complete.json').write_text(json.dumps(receipt,indent=2));print(receipt)


if __name__=='__main__':main()
