"""Pure index-probe geometry and ownership refusal; no model/GPU evidence."""
import importlib.util
from pathlib import Path
import mlx.core as mx
import pytest

spec=importlib.util.spec_from_file_location('ds41_index_probe_cpu',Path(__file__).parents[1]/'scripts/research/ds41_ast_index_components.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def record(layer,node,count):return dict(layer=layer,node=node,count=count,heads=32,k=128)


def test_chronological_count_classes_do_not_pad_to_common_physical_width():
    rows=[record(20,0,1025),record(2,1,513),record(2,2,513),record(2,3,514),record(20,1,1026)]
    assert probe.groups(rows)==((2,513,(1,2)),(2,514,(3,)),(20,1025,(0,)),(20,1026,(4,)))


@pytest.mark.parametrize('rows',[[],[record(2,0,512)],[{**record(20,1,1025),'heads':64}],[record(8,1,513)]])
def test_probe_refuses_missing_or_foreign_index_geometry(rows):
    with pytest.raises(ValueError):probe.groups(rows)


def test_cpu_and_missing_ownership_refuse_before_model_access(tmp_path):
    mx.set_default_device(mx.cpu)
    with pytest.raises(ValueError,match='callback'):probe.run(object(),tmp_path,None)
    with pytest.raises(ValueError,match='parent-owned GPU'):probe.run(object(),tmp_path,lambda:True)
