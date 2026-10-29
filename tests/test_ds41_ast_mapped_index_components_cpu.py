"""Mapped-index probe ownership refusals before source/model access."""
from pathlib import Path
import importlib.util
import mlx.core as mx
import pytest

spec=importlib.util.spec_from_file_location('mapped_index_probe_cpu',Path(__file__).parents[1]/'scripts/research/ds41_ast_mapped_index_components.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_owned_gpu_required_before_fixture_or_model_access(tmp_path):
    mx.set_default_device(mx.cpu)
    with pytest.raises(ValueError,match='callback'):probe.run_replay(object(),tmp_path,None,'missing','missing')
    with pytest.raises(ValueError,match='parent-owned GPU'):probe.run_replay(object(),tmp_path,lambda:True,'missing','missing')
