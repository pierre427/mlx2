"""Pure NumPy oracle tests; no GPU or model qualification."""
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('segmented_probe',Path(__file__).parents[1]/'scripts/probe_nanbeige43_segmented_attention.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def synthetic_receipt(record_property):
    root=Path(__file__).parents[1]
    record_property('source_revision',subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    record_property('checkpoint','synthetic-segmented-fp64-oracle@sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record_property('working_probe_sha256',hashlib.sha256((root/'scripts/probe_nanbeige43_segmented_attention.py').read_bytes()).hexdigest())


def test_bfloat_rounding_uses_fp64_side_of_midpoint_and_ties_to_even():
    halfway=1+2**-8
    values=np.array([np.nextafter(halfway,-np.inf),halfway,np.nextafter(halfway,np.inf),-np.nextafter(halfway,np.inf)])
    np.testing.assert_array_equal(probe.round_reference(values,'bfloat16'),[1,1,1+2**-7,-(1+2**-7)])
    halfway_odd=1+3*2**-8
    np.testing.assert_array_equal(probe.round_reference([halfway_odd],'bfloat16'),[1+2**-6])


def test_oracle_gqa7_mapping_ragged_prefix_and_sibling_visibility():
    q=np.zeros((2,56,3,128));prefixes=[0,2]
    ancestors=np.array([[[1,0,0],[1,1,0],[1,0,1]]]*2,dtype=bool)
    keys=[np.zeros((1,8,p+3,128)) for p in prefixes];values=[]
    for p in prefixes:
        v=np.zeros((1,8,p+3,128))
        for h in range(8): v[0,h,:,0]=100*h+np.arange(p+3)
        values.append(v)
    result=probe.float64_attention_reference(q,keys,values,prefixes,ancestors,1)
    for b,p in enumerate(prefixes):
        for h in range(56):
            np.testing.assert_allclose(result[b,h,0,0],100*(h//7)+np.mean(list(range(p))+[p]),atol=1e-12,rtol=0)
            np.testing.assert_allclose(result[b,h,2,0],100*(h//7)+np.mean(list(range(p))+[p,p+2]),atol=1e-12,rtol=0)
    assert np.all(result[:,:,:,1:]==0)


def test_output_ulp_metrics_distinguish_native_and_fp32_accumulator_rounding():
    reference=np.array([1+2**-8+1e-10]);rounded=probe.round_reference(reference,'bfloat16')
    metrics=probe.numerical_metrics(rounded,np.array([1.]),reference,'bfloat16')
    assert metrics['candidate']['max_output_ulp_vs_rounded_reference']==0
    assert metrics['native']['max_output_ulp_vs_rounded_reference']==1
