# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Native MLX-LM batches with isolated calculator lanes and completion alarms.

Original research harness adapter over mlx-lm 0.31.3's public BatchGenerator.
Every callback is invoked in physical-row order on the same decode forward;
rebinding by forward ID survives lane compaction without trusting stale UID rows.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

from amoeba_calc_defenses import CompletionDefenses


def calculator_tile_id(processor):
    return processor.tile_id if hasattr(processor,"tile_id") else processor.tile.id


class RowBinder:
    def __init__(self, tiles):
        self.tiles=tiles;self.forward=None;self.row=0
    def bind(self, processor):
        tile_id=calculator_tile_id(processor)
        forward=self.tiles.forward_ids.get(tile_id)
        rows,positions=self.tiles.stash_shape[tile_id]
        if forward!=self.forward:self.forward=forward;self.row=0
        if positions!=1 or self.row>=rows:raise RuntimeError('invalid batched calculator forward binding')
        row=self.row;self.row+=1;processor.bind_row(row)
        return row,rows


def calculator_processor(tiles, tokenizer, prompt_length, request_text):
    from mlx2.runtime.amoeba_det_tiles import processors_for

    request = ({"messages": [{"role": "user", "content": request_text}]}
               if request_text is not None else None)
    processor, = processors_for(tiles, tokenizer, prompt_length=prompt_length, request=request)
    return processor


def answer_in_progress(processor):
    if processor is None:
        return False
    if hasattr(processor, "tile_id"):
        # Only read device state when a defense would otherwise interrupt an answer.
        length, position = processor.state[:2].tolist()
        return length > position
    return processor.plan is not None


def batch_stage(model,tok,tiles,prompts,limits,*,watch=False,calculator=True,synchronous_callbacks=False,request_texts=None):
    from mlx_lm.generate import BatchGenerator
    binder=RowBinder(tiles)
    states=[dict(token_ids=[],events=[],defense=CompletionDefenses() if watch else None,
                 widths=set(),rows=set(),checked=False,finish_reason=None) for _ in prompts]
    callbacks=[]
    request_texts=request_texts if request_texts is not None else [None]*len(prompts)
    if len(request_texts)!=len(prompts):raise ValueError('one request text per lane is required')
    for prompt,state,request in zip(prompts,states,request_texts):
        proc=calculator_processor(tiles,tok,len(prompt),request) if calculator else None
        state['processor']=proc
        def process(tokens,logits,state=state,prompt=prompt,proc=proc):
            if synchronous_callbacks:
                # Diagnostic recovery mode: finish the model forward before a
                # callback submits small calculator kernels and host flag reads.
                # This changes scheduling, not tokens, arithmetic or gate policy.
                import mlx.core as mx
                mx.eval(tokens,logits,tiles.stash[calculator_tile_id(proc)])
            if not state['checked']:
                # BatchGenerator supplies the COMPLETE prompt history (unlike generate_step).
                if tokens[:len(prompt)].tolist()!=prompt:raise RuntimeError('batch prompt history mismatch')
                state['checked']=True
            row,width=binder.bind(proc);state['rows'].add(row);state['widths'].add(width)
            return proc(tokens,logits)
        callbacks.append([process] if proc else [])
    generator=BatchGenerator(model,stop_tokens=[[t] for t in tok.eos_token_ids],
                             completion_batch_size=len(prompts),prefill_batch_size=len(prompts))
    try:
        uids=generator.insert(prompts,max_tokens=limits,logits_processors=callbacks)
        by_uid=dict(zip(uids,states))
        while responses:=generator.next_generated():
            remove=[]
            for response in responses:
                state=by_uid[response.uid];new=state['token_ids'];new.append(int(response.token))
                state['finish_reason']=response.finish_reason
                guard=state['defense']
                if response.finish_reason is None and guard:
                    text=tok.decode(new) if len(new)%16==0 else None
                    if guard.feed(response.token,text) and not answer_in_progress(state['processor']):
                        state['finish_reason']='defense';remove.append(response.uid)
            if remove:generator.remove(remove)
    finally:generator.close()
    for state in states:
        processor=state.pop('processor')
        state['events']=getattr(processor,'events',[])
        state['widths']=sorted(state['widths']);state['rows']=sorted(state['rows'])
    return states
