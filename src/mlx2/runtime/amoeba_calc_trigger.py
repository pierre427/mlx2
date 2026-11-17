# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""GPU-resident token window and conservative calculator cue/gate prefilter.

One small flag is read on the CPU to decide whether to run the text parser. This
reduces string work; it does not claim a synchronization-free generation loop.
"""
from functools import lru_cache
import threading

_TABLES={}
_LOCK=threading.Lock()


def _table(tokenizer):
    import numpy as np
    import mlx.core as mx
    key=(id(tokenizer),len(tokenizer))
    with _LOCK:
        if key not in _TABLES:
            pool=bytearray();offsets=[];lengths=[];cue_flags=[]
            for i in range(len(tokenizer)):
                offsets.append(len(pool))
                try:piece=tokenizer.decode([i]).encode('utf-8')
                except Exception:piece=b'\xff'  # unknown pieces force the full parser
                lengths.append(len(piece));pool.extend(piece)
                cue_flags.append(int(b"=" in piece) | (2 if b"\\hline" in piece else 0))
            with mx.stream(mx.gpu):
                values=(mx.array(np.frombuffer(pool,dtype=np.uint8)),mx.array(offsets,dtype=mx.uint32),mx.array(lengths,dtype=mx.uint32),mx.array(cue_flags,dtype=mx.uint32))
            if len(_TABLES)>=4:_TABLES.pop(next(iter(_TABLES)))
            _TABLES[key]=(tokenizer,values)
        return _TABLES[key][1]


_SOURCE=r'''
if(thread_position_in_grid.x!=0)return;
for(int i=0;i<48;i++)next_window[i]=meta[0]?((i==47)?last[0]:window[i+1]):window[i];
uchar reversed[160];int used=0;bool unknown=false;
for(int i=47;i>=48-meta[1] && used<160;i--){
 uint id=next_window[i];if(id>=offsets_shape[0]){unknown=true;break;}
 uint start=offsets[id],n=lengths[id];
 for(int j=int(n)-1;j>=0 && used<160;j--){uchar c=pool[start+uint(j)];reversed[used++]=c;if(c==255)unknown=true;}
}
int p=0;while(p<used && (reversed[p]==32 || reversed[p]==9 || reversed[p]==10 || reversed[p]==13))p++;
if(p<used && reversed[p]>=128)unknown=true; // possible Unicode whitespace at the cue boundary
bool eq=p<used && reversed[p]=='=';
bool equals=p+6<=used && reversed[p]=='s' && reversed[p+1]=='l' && reversed[p+2]=='a' && reversed[p+3]=='u' && reversed[p+4]=='q' && reversed[p+5]=='e';
bool equalto=p+11<=used && reversed[p]=='o' && reversed[p+1]=='t' && reversed[p+2]==' ' && reversed[p+3]=='l' && reversed[p+4]=='a' && reversed[p+5]=='u' && reversed[p+6]=='q' && reversed[p+7]=='e' && reversed[p+8]==' ' && reversed[p+9]=='s' && reversed[p+10]=='i';
bool latex=p+6<=used && reversed[p]=='e' && reversed[p+1]=='n' && reversed[p+2]=='i' && reversed[p+3]=='l' && reversed[p+4]=='h' && reversed[p+5]=='\\';
uint last_flags=last[0]<cue_flags_shape[0]?cue_flags[last[0]]:0;
bool symbolic=false;
if(meta[3]){for(int i=0;i<used;i++)if(reversed[i]=='\\' || reversed[i]=='$' || reversed[i]=='!')symbolic=true;}
if(meta[4]){for(int i=0;i<used;i++)if(reversed[i]=='[' || reversed[i]=='\'' || reversed[i]=='"')symbolic=true;}
trigger[0]=unknown || latex || (last_flags&2) || ((eq || equals || equalto || (last_flags&1)) && (gate[meta[2]] || symbolic));
'''


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='amoeba_calculator_cue_window',input_names=['window','last','pool','offsets','lengths','cue_flags','gate','meta'],output_names=['next_window','trigger'],source=_SOURCE)


class GPUCueWindow:
    def __init__(self,tokenizer,*,allow_symbolic=False,allow_counting=False):
        self.allow_symbolic=allow_symbolic
        self.allow_counting=allow_counting
        self.tables=_table(tokenizer)
        self.window=None
        self.count=0

    def update(self,tokens,gate,gate_index=0):
        import mlx.core as mx
        with mx.stream(mx.gpu):
            append=self.window is not None
            if not append:
                self.count=min(48,int(tokens.size))
                self.window=mx.concatenate([mx.zeros((48-self.count,),dtype=mx.uint32),tokens[-self.count:].astype(mx.uint32)])
            else:self.count=min(48,self.count+1)
            self.window,trigger=_kernel()(inputs=[self.window,tokens[-1:].astype(mx.uint32),*self.tables,gate,mx.array([int(append),self.count,gate_index,int(self.allow_symbolic),int(self.allow_counting)],dtype=mx.int32)],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(48,),(1,)],output_dtypes=[mx.uint32,mx.bool_])
            mx.async_eval(self.window,trigger)
            return trigger
