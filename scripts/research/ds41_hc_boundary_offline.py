"""Compile bounded Source HC boundary Metal wrappers offline; never GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from mlx2.runtime.models.deepseek_v41_source_hc_boundary import compilation_source,COLLAPSE_POLICIES


def wrapper(stage,policy,diagnostics):
    c=compilation_source(stage,collapse_policy=policy,return_frontiers=diagnostics)
    args=[f'device const float* {n} [[buffer({i})]]'for i,n in enumerate(c['inputs'])]
    args += [f'device float* {n} [[buffer({i+len(c["inputs"])})]]'for i,n in enumerate(c['outputs'])]
    args += ['uint3 thread_position_in_grid [[thread_position_in_grid]]']
    return ('#include <metal_stdlib>\nusing namespace metal;\n#define DIM 5120\n#define ELEMENTS 15360\n#define BLOCK 32\n#define KIND 0\n'
            +c['header']+'\nkernel void boundary_fixture('+',\n'.join(args)+') {\n'+c['source']+'\n}\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path);a=parser.parse_args()
    if a.out.exists()and any(a.out.iterdir()):raise FileExistsError(a.out)
    a.out.mkdir(parents=True,exist_ok=True);records=[]
    cases=[('residual',p,d)for p in COLLAPSE_POLICIES for d in(False,True)]+[('norm_qat','stock_frontier',False)]
    for stage,policy,diagnostics in cases:
        name=f'{stage}-{policy}-{int(diagnostics)}';source=a.out/(name+'.metal');target=a.out/(name+'.air')
        source.write_text(wrapper(stage,policy,diagnostics))
        result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(source),'-o',str(target)],capture_output=True,text=True,timeout=45)
        records.append(dict(stage=stage,collapse_policy=policy,return_frontiers=diagnostics,
            source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            returncode=result.returncode,stderr=result.stderr,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest()if target.exists()else None))
    report=dict(scope='Offline Metal3.2 compilation only; no GPU execution/parity/timing claim',passed=all(r['returncode']==0 for r in records),cases=records)
    (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'],cases=len(records))))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
