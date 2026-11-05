"""Offline compilation exercises Metal 4 types without taking GPU ownership."""
from pathlib import Path
import shutil
import subprocess

import mlx.core as mx
import pytest

from mlx2.runtime.models import fp_verify_tensor as fp


@pytest.mark.parametrize('dtype',['half','bfloat'])
def test_offline_compile(dtype,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    signature=[f'const device {dtype}* q',f'const device {dtype}* k',f'const device {dtype}* v',
               'const device bool* mask','const device float* scale','device float* partial',
               'device float* maxima','device float* sums']
    args=',\n'.join(f'{value} [[buffer({i})]]' for i,value in enumerate(signature))
    source='#include "mlx/backend/metal/kernels/utils.h"\n'+fp.HEADER
    source+='\nkernel void fp_probe('+args+',\nuint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\nuint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {\n'
    source+=f'using T={dtype}; constexpr int B=1,HQ=16,HK=2,L=4,N=1024,G=8,SPLITS=2; constexpr bool MASKED=true,CAUSAL=false;\n'
    source+=fp.SOURCE+'\n}\n'
    path=tmp_path/'verify.metal';path.write_text(source)
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-I',str(Path(mx.__file__).parent/'include'),
                           '-c',str(path),'-o',str(tmp_path/'verify.air')],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
