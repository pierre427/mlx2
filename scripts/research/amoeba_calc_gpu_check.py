# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Exact GPU calculator differential checks; run only inside an owned GPU window."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse,json,random,subprocess,time
from pathlib import Path
from mlx2.runtime import amoeba_det_tiles as D

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
rng=random.Random(17112026)
cases=['999999999999999999999999999999 * 999999999999999999999999999999','25 * 1','1 * 25','100 * 25','63181 / 10','79416.21 / 25','63912 / 74','1 / 3 * 3','1 / 7 + 6 / 7','2^-64','8^-64','999999999999.123456 / 0.0004','109 * 71','123456789123 * 987654321987','0.000012345 * 99999.123456','(3^3 + 78) / 5','-7 * 8','7 / -8','2^-3','-2^2','(-2)^2','2^3^2','1 / 3','9 / 0','0^0','0^-1','2^65','2^0.5','0.1 + 0.2','999999999999 + 1','1000000000000 - 1','1.2500 / 0.5','0.0001 / 0.00002']
for i in range(200):
 a,b,c=rng.randint(2,99999),rng.randint(2,99),rng.randint(2,15)
 cases.append(rng.choice([f'({a} + {b}) * {c}',f'{a} * {b}',f'{a}.{b:02d} / {rng.choice([2,4,5,8,16,25])}',f'-{a} + {b} * {c}',f'{a} / {b}',f'{c}^{rng.randint(0,6)} + {a}']))
rows=[]
for backend in ['gpu','gemm']:
 for i,e in enumerate(cases):
  text='So '+e+' =';expected,ei=D.calculator_answer(text,parser=D.PARSER_V3,arithmetic_backend='cpu')
  t=time.perf_counter();got,gi=D.calculator_answer(text,parser=D.PARSER_V3,arithmetic_backend=backend);wall=time.perf_counter()-t
  ok=got==expected
  row=dict(backend=backend,expression=e,cpu=expected,gpu=got,cpu_info=ei,gpu_info=gi,ok=ok,wall_s=wall);rows.append(row)
  with (args.out/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if not ok:print('MISMATCH',row,flush=True)
 print(backend,'checked',len(cases),flush=True)
latex_cases=[r'\frac{1}{3}+\frac{2}{3}',r'5!',r'\binom{10}{3}',r'\sum_{i=1}^{5}i^2',r'\prod_{i=1}^{5}i', r'\left('+r'\displaystyle '*20+r'\dfrac{12}{\dfrac{9}{3}}\right)']
for backend in ['gpu','gemm']:
 for e in latex_cases:
  text='So '+e+' ='
  expected,ei=D.calculator_answer(text,parser=D.PARSER_V4,arithmetic_backend='cpu')
  got,gi=D.calculator_answer(text,parser=D.PARSER_V4,arithmetic_backend=backend)
  row=dict(backend=backend,expression=e,cpu=expected,gpu=got,cpu_info=ei,gpu_info=gi,ok=got==expected and expected is not None);rows.append(row)
  with (args.out/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if not row['ok']:print('MISMATCH',row,flush=True)
# Trigger check uses the exact checkpoint tokenizer but does not load model weights.
import mlx.core as mx
from transformers import AutoTokenizer
from mlx2.runtime.amoeba_calc_trigger import GPUCueWindow
snapshot=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
trigger_rows=[]
for text,opened,want in [('Ordinary prose about forests.',False,False),('So 12 * 34 =',True,True),('So 12 * 34 = ',False,False),('So 12 * 34 equals',True,True),('So 12 * 34 is equal to',True,True),('\\begin{array}{r}109 \\\\ \\times 71 \\\\ \\hline',False,True)]:
 window=GPUCueWindow(tok);ids=tok.encode(text,add_special_tokens=False)
 for n in range(1,len(ids)+1):
  flag=window.update(mx.array(ids[:n],dtype=mx.uint32),mx.array([[opened]]))
 got=bool(flag.item());trigger_rows.append(dict(text=text,got=got,want=want,ok=got==want))
# Exercise merged pieces directly, including tokens absent from this tokenizer's merges.
class Pieces:
 def __init__(self):self.pieces=['ordinary', '= ', '=$', r'=\)', r'=\]', r'x\hline ', '=answer']
 def __len__(self):return len(self.pieces)
 def decode(self,ids):return ''.join(self.pieces[i] for i in ids)
pieces=Pieces()
for i,piece in enumerate(pieces.pieces):
 window=GPUCueWindow(pieces)
 got=bool(window.update(mx.array([i],dtype=mx.uint32),mx.array([[True]])).item())
 want=i!=0
 trigger_rows.append(dict(text=piece,got=got,want=want,ok=got==want))
# Every real vocabulary piece containing either cue is represented in the GPU lookup.
flags=GPUCueWindow(tok).tables[-1].tolist()
for i in range(len(tok)):
 piece=tok.decode([i])
 if '=' in piece or r'\hline' in piece:
  assert flags[i] & (1 if '=' in piece else 2), (i,piece)
(args.out/'triggers.json').write_text(json.dumps(trigger_rows,indent=2))
if any(not r['ok'] for r in trigger_rows):raise RuntimeError('GPU cue trigger mismatch: '+str(trigger_rows))
summary=dict(checkpoint='not applicable: synthetic exact-arithmetic kernel check',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),n=len(rows),mismatches=sum(not r['ok'] for r in rows))
(args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if summary['mismatches']:raise SystemExit(1)
