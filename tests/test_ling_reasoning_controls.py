"""Checkpoint-template reasoning wiring; no weight loading or GPU work."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from mlx2.adapters.ling3_flash import Ling3FlashAdapter, DESCRIPTOR
from mlx2.contracts import Capability


def adapter():
    a=object.__new__(Ling3FlashAdapter)
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            self.last=(copy.deepcopy(messages),kwargs)
            return [101,102] if kwargs['tokenize'] else 'rendered'
        def encode(self,text,**kwargs):return [7] if text=='</think>' else [9]
    a.tokenizer=Tokenizer();return a


@pytest.mark.parametrize('enabled',[True,False])
def test_prompt_and_parser_honor_same_explicit_control(enabled):
    a=adapter();r={'messages':[{'role':'user','content':'hi'}],'enable_thinking':enabled}
    assert a.prompt_tokens(r)==[101,102]
    assert a.tokenizer.last[1]['enable_thinking'] is enabled
    assert a.render_prompt(r)=='rendered'
    assert a.tokenizer.last[1]['enable_thinking'] is enabled
    assert a.output_parser(r).channel==('reasoning_content' if enabled else 'content')
    assert Capability.REASONING in DESCRIPTOR.capabilities
    assert a.reasoning_effort_semantics=='thinking_toggle'
    assert a.thinking_close_token_ids()==(7,)


def test_split_closing_marker_routes_channels_without_leaking_marker():
    p=adapter().output_parser({'messages':[],'enable_thinking':True})
    events=[]
    for chunk in ['work','</thi','nk>','answer']:events.extend(p.push(chunk))
    events.extend(p.finish("", "stop"))
    assert ''.join(e.get('reasoning_content','') for e in events)=='work'
    assert ''.join(e.get('content','') for e in events)=='answer'


def test_raw_completion_and_unavailable_tools_still_fail_closed():
    a=adapter();r={'prompt':'hi','enable_thinking':True}
    assert a.prompt_tokens(r)==[9] and a.render_prompt(r)=='hi'
    assert a.output_parser(r).channel=='content'
    with pytest.raises(ValueError,match='tools'):
        a.output_parser({'messages':[],'tools':[{'type':'function','function':{'name':'x'}}]})


def test_real_checkpoint_template_has_roundtrippable_on_off_controls():
    from transformers import AutoTokenizer
    p=Path('~/mlx-models/TensorFold-Ling-3.0-flash-oQ4')
    if not p.is_dir():pytest.skip('checkpoint absent')
    tokenizer=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
    from mlx2.runtime.tokenizer_utils import TokenizerWrapper, BPEStreamingDetokenizer
    a=object.__new__(Ling3FlashAdapter);a.tokenizer=TokenizerWrapper(tokenizer,detokenizer_class=BPEStreamingDetokenizer)
    for enabled,ending in [(True,'\n<think>'),(False,'\n<think></think>')]:
        r={'messages':[{'role':'user','content':'hi'}],'enable_thinking':enabled}
        rendered=a.render_prompt(r)
        assert rendered.endswith(ending)
        assert f'detailed thinking {"on" if enabled else "off"}' in rendered
        assert a.prompt_tokens(r)==tokenizer.encode(rendered,add_special_tokens=False)
    assert len(a.thinking_close_token_ids())==1
