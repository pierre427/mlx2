"""B1 actual FA3/7/55 boundary diagnostic; original Apache-2.0 froma2efa889.

Reuse existing terminal raw-native-KV export and actual SDPA capture contract.
No model/runtime imports until execution; replays remain explicitly diagnostic.
"""
from __future__ import annotations
from contextlib import contextmanager
from varlen_hybrid_fa_boundary_probe import _export_logical_native_kv
LAYERS=(3,7,55)


class B1FABoundaryProbe:
    def __init__(self):
        self.candidate={};self.ordinary={};self._stock_sdpa=None;self._ordinary_calls=0
    def candidate_callback(self,event):
        layer=event.get('layer_index')
        if layer not in LAYERS:return
        required={'layer_index','fa_index','offsets','queries','keys','values','native_attention','hidden_dtype'}
        if (type(event) is not dict or set(event)!=required or layer in self.candidate or
                type(event['offsets']) not in (tuple,list) or len(event['offsets'])!=1):
            raise ValueError('B1 FA callback incomplete or duplicated')
        self.candidate[layer]=dict(event)
    @contextmanager
    def capture_ordinary(self,attention_module):
        original=attention_module.scaled_dot_product_attention
        if self._stock_sdpa is not None:raise RuntimeError('B1 FA capture already active')
        self._stock_sdpa=original;self._ordinary_calls=0
        def observe(queries,keys,values,*,cache,scale,mask,**kwargs):
            ordinal=self._ordinary_calls;self._ordinary_calls+=1;layer=4*ordinal+3
            output=original(queries,keys,values,cache=cache,scale=scale,mask=mask,**kwargs)
            if layer in LAYERS:
                self.ordinary[layer]={'queries':queries,'keys':keys,'values':values,'attention':output,
                    'mask':mask,'scale':scale,'offset':cache.offset,'padding':cache.left_padding}
            return output
        attention_module.scaled_dot_product_attention=observe
        try:yield
        finally:
            attention_module.scaled_dot_product_attention=original
            if self._ordinary_calls!=16 or set(self.ordinary)!=set(LAYERS):
                raise RuntimeError('actual B1 ordinary FA calls missing')
    def compare(self,branches,*,mx):
        from varlen_hybrid_b1_numeric_gate import tensor_metrics
        if (type(branches) is not tuple or len(branches)!=1 or
                set(self.candidate)!=set(LAYERS) or set(self.ordinary)!=set(LAYERS) or self._stock_sdpa is None):
            raise ValueError('B1 actual complete FA captures and branch required')
        details=[]
        for layer in LAYERS:
            candidate=self.candidate[layer];ordinary=self.ordinary[layer];ordinal=candidate['fa_index']
            if ordinal!=layer//4 or ordinal>=len(branches[0].layers):raise RuntimeError('B1 FA ordinal differs')
            keys,values=_export_logical_native_kv(branches[0].layers[ordinal],mx)
            offset=keys.shape[1];before=int(candidate['offsets'][0])
            logical_offset=tuple(int(value.item()) for value in ordinary['offset'])
            padding=tuple(int(value.item()) for value in ordinary['padding'])
            if offset!=before+1 or logical_offset!=(offset,) or padding!=(0,):
                raise RuntimeError('B1 survivor logical native/ordinary offsets differ')
            mask=ordinary['mask']
            if mask is not None:
                if mask.dtype!=mx.bool_ or tuple(mask.shape)!=(1,1,1,offset) or int(mx.sum(mask).item())!=offset:
                    raise RuntimeError('B1 ordinary mask is not complete unpadded prefix')
            query=candidate['queries'][:,:,None,:]
            native=candidate['native_attention']
            if native.ndim==3:native=native[:,:,None,:]
            replay=self._stock_sdpa(query,keys[None],values[None],cache=None,
                scale=ordinary['scale'],mask=mask)
            metrics={
                'query':tensor_metrics(query,ordinary['queries'],mx),
                'new_key':tensor_metrics(candidate['keys'],ordinary['keys'][:,:,-1,:],mx),
                'new_value':tensor_metrics(candidate['values'],ordinary['values'][:,:,-1,:],mx),
                'full_logical_key':tensor_metrics(keys,ordinary['keys'][0],mx),
                'full_logical_value':tensor_metrics(values,ordinary['values'][0],mx),
                'native_vs_actual_ordinary_attention':tensor_metrics(native,ordinary['attention'],mx),
                'native_vs_same_QKV_stock_replay':tensor_metrics(native,replay,mx),
                'stock_replay_vs_actual_ordinary_attention':tensor_metrics(replay,ordinary['attention'],mx)}
            inputs_exact=all(metrics[name]['exact'] for name in ('query','new_key','new_value','full_logical_key','full_logical_value'))
            details.append({'layer':layer,'pre_offset':before,'post_offset':offset,
                'mask_kind':'none' if mask is None else 'all_true','same_QKV_exact':inputs_exact,
                'same_QKV_native_vs_stock_difference':not metrics['native_vs_same_QKV_stock_replay']['exact'],
                'metrics':metrics})
        return {'schema':'mlx2.hybrid-b1-fa-boundary-probe.v1','actual_ordinary_calls':self._ordinary_calls,
            'layers':list(LAYERS),'details':details,'stock_replay_scope':'diagnostic same native QKV, actual ordinary mask/scale',
            'qualified':False}
