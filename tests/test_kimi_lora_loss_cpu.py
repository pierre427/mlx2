import importlib.util
from pathlib import Path
import mlx.core as mx
import pytest
mx.set_default_device(mx.cpu)
SPEC = importlib.util.spec_from_file_location('kimi_lora_loss', Path(__file__).parents[1] / 'scripts/research/kimi_lora_loss.py')
loss = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loss)


def row():
    return {'id':'fixture', 'split':'validation','evaluation_only':True,'never_train':True,
            'input_ids':[0,1,2,0,1], 'loss_mask':[0,0,1,0,1]}


class Toy:
    def __init__(self):
        self.states=[]
    def make_cache(self):
        cache=[]; self.states.append(cache); return cache
    def __call__(self, ids, cache):
        cache.extend(ids[0].tolist())
        return mx.stack([mx.stack([mx.array([0.,0.,0.]).at[(int(x)+1)%3].add(3) for x in ids[0]])])


def test_shift_mask_chunk_boundaries_and_fresh_state():
    model=Toy()
    a=loss.evaluate_loss(model,[row(),{**row(),'id':'second'}],chunk_size=2)
    b=loss.evaluate_loss(Toy(),[row()],chunk_size=20)
    assert a['assistant_tokens']==4
    assert a['token_mean_nll']==pytest.approx(b['token_mean_nll'],abs=1e-6)
    assert a['token_mean_nll']<.1
    assert model.states==[[0,1,2,0],[0,1,2,0]]
    assert model.states[0] is not model.states[1]


@pytest.mark.parametrize('change',[{'split':'train'},{'split':'test'},{'never_train':False},
    {'loss_mask':[0,0,0,0,0]},{'loss_mask':[0,0,float('nan'),0,1]},{'input_ids':[0,1,True,0,1]}])
def test_no_training_or_final_leakage(change):
    with pytest.raises(ValueError): loss.validate_loss_rows([{**row(),**change}])
