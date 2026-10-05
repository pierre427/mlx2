"""Run directly: stdlib numeric oracles, C++ plans, offline Metal compilation."""
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]

def accumulate(scores,values):
    dim=len(values[0]) if values else 256
    maximum=-math.inf;den=0.;num=[0.]*dim
    for score,value in zip(scores,values):
        next_max=max(maximum,score)
        old=0. if den==0 else f32(math.exp(maximum-next_max))
        new=f32(math.exp(score-next_max))
        maximum=next_max;den=f32(f32(den*old)+new)
        num=[f32(f32(n*old)+f32(v*new)) for n,v in zip(num,value)]
    return maximum,den,num

def merge(parts):
    maximum=max(p[0] for p in parts)
    weights=[f32(math.exp(m-maximum)) if d>0 else 0. for m,d,n in parts]
    den=0.;num=[0.]*len(parts[0][2])
    for (m,d,n),w in zip(parts,weights):
        den=f32(den+f32(d*w))
        num=[f32(total+f32(v*w)) for total,v in zip(num,n)]
    return maximum,den,num

def merge_striped(parts):
    maximum=max(p[0] for p in parts)
    weights=[f32(math.exp(m-maximum)) if d>0 else 0. for m,d,n in parts]
    dim=len(parts[0][2]);stripe_nums=[[0.]*dim for _ in range(4)]
    den=0.
    for (m,d,n),w in zip(parts,weights):den=f32(den+f32(d*w))
    for stripe in range(4):
        for p in range(stripe,len(parts),4):
            stripe_nums[stripe]=[f32(a+f32(b*weights[p])) for a,b in zip(stripe_nums[stripe],parts[p][2])]
    numerator=[0.]*dim
    for stripe in stripe_nums:numerator=[f32(a+b) for a,b in zip(numerator,stripe)]
    return maximum,den,numerator

class SplitKVCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory(prefix='q1-splitkv-cpu-');cls.path=Path(cls.folder.name)
        cpp=cls.path/'probe.cpp'
        cpp.write_text(r'''
#include "q1_splitkv.h"
#include <iostream>
#include <cstdlib>
int main(int argc, char** argv) {
  using namespace mlx2::paged_kv;
  try {
    if (argc == 2) { std::cout << q1_split_kv_partition(argv[1]); return 0; }
    const auto dim=std::strtoul(argv[1],nullptr,10), part=std::strtoul(argv[2],nullptr,10);
    if (argc==3) { std::cout << q1_split_kv_source(dim,part); return 0; }
    const auto plan=q1_split_kv_plan(dim,std::strtoul(argv[3],nullptr,10),
        {static_cast<uint32_t>(std::strtoul(argv[4],nullptr,10)),static_cast<uint32_t>(std::strtoul(argv[5],nullptr,10))},part);
    std::cout << plan.partitions << ' ' << plan.scratch_values << ' ' << plan.scratch_bytes;
  } catch(const std::invalid_argument&) { return 2; }
}
''')
        cls.probe=cls.path/'probe'
        subprocess.run([shutil.which('c++') or 'c++','-std=c++20','-Wall','-Wextra','-Werror',
                        '-I',str(ROOT/'native/paged_kv'),str(cpp),'-o',str(cls.probe)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def invoke(self,*args):return subprocess.run([str(self.probe),*map(str,args)],text=True,capture_output=True)

    def test_flags_default_off_and_invalid(self):
        for flag,expected in [('0',0),('128',128),('256',256)]:self.assertEqual(self.invoke(flag).stdout,str(expected))
        for flag in ('','1','64','512','true'):self.assertEqual(self.invoke(flag).returncode,2)

    def test_boundary_partition_and_scratch_plans(self):
        for dim in (128,256):
            for size in (128,256):
                for visible in (1,63,64,65,127,128,129,255,256,257,4096,8191,8192):
                    p=math.ceil(visible/size)
                    result=self.invoke(dim,size,24,visible,1)
                    self.assertEqual(result.stdout,f'{p} {2*24*p*(dim+2)} {2*24*p*(dim+2)*4}')
        for args in ((64,128,24,1,1),(256,64,24,1,1),(256,128,0,1,1),(256,128,129,1,1),
                     (256,128,24,0,1),(256,128,24,8193,1)):
            self.assertEqual(self.invoke(*args).returncode,2)

    def test_offline_metal_variants(self):
        if subprocess.run(['xcrun','--find','metal'],capture_output=True).returncode:
            self.skipTest('offline Metal compiler unavailable')
        for dim in (128,256):
            for partition in (128,256):
                source=self.invoke(dim,partition)
                self.assertEqual(source.returncode,0,source.stderr)
                path=self.path/f'd{dim}-p{partition}.metal';path.write_text(source.stdout)
                result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(path),'-o',str(path.with_suffix('.air'))],text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_split_numeric_fp32_tail_extreme_and_empty(self):
        rng=random.Random(1004)
        for width in (63,64,65,127,128,129,257,4096,8192):
            dim=256
            scores=[f32(rng.uniform(-20,20)) for _ in range(width)]
            values=[[f32(rng.uniform(-1,1)) for _ in range(dim)] for _ in range(width)]
            # Independent stable dense double oracle for the partition/merge math.
            maximum=max(scores);weights=[math.exp(s-maximum) for s in scores];den=sum(weights)
            expected=[sum(w*v[d] for w,v in zip(weights,values))/den for d in range(dim)]
            for size in (128,256):
                parts=[]
                for begin in range(0,width,size):
                    stripes=[accumulate(scores[begin+s:min(width,begin+size):4],values[begin+s:min(width,begin+size):4]) for s in range(4)]
                    parts.append(merge(stripes))
                # A shorter B2 lane has fully empty scratch partitions.
                parts.append((-math.inf,0.,[0.]*dim))
                for merged in (merge(parts),merge_striped(parts)):
                    _,d,n=merged
                    self.assertLess(max(abs(x/d-y) for x,y in zip(n,expected)),2e-6)
        parts=[(-math.inf,0.,[0.]*256)]*64
        _,d,n=merge(parts);self.assertEqual(d,0.);self.assertEqual(n,[0.]*256)
        for scores in ([10000.,-10000.,9999.],[-10000.,-10001.,-9999.]):
            _,d,n=merge([accumulate([s],[[.5]*256]) for s in scores])
            self.assertAlmostEqual(n[0]/d,.5,places=6)

    def test_absolute_page_edges_window_and_u32_tail(self):
        for lower,visible,size in ((63,129,128),(65,4096,256),(4294967295-129,129,128)):
            upper=lower+visible
            seen=[]
            for p in range(math.ceil(visible/size)+1):
                begin=lower+min(upper-lower,p*size);end=begin+min(upper-begin,size)
                for stripe in range(4):seen += [begin+i for i in range(stripe,end-begin,4)]
            self.assertEqual(sorted(seen),list(range(lower,upper)))
            first=lower//64;pages=[1000+block for block in range(math.ceil((upper-first*64)/64))]
            self.assertEqual([(pages[t//64-first],t&63) for t in range(lower,upper)],
                             [(1000+t//64-first,t%64) for t in range(lower,upper)])

    def test_paged_dot_product_head_mapping_and_noncontiguous_query(self):
        rng=random.Random(256)
        dim=256;query_heads=4;kv_heads=2;scale=1/math.sqrt(dim)
        # Absolute offset crosses page and partition boundaries; physical page
        # IDs are permuted independently of logical positions.
        lower=63;visible=129;upper=lower+visible;first=lower//64
        pages=[3,1,4];store={}
        for page in pages:
            for head in range(kv_heads):
                for slot in range(64):
                    store[page,head,slot]=([f32(rng.uniform(-.2,.2)) for _ in range(dim)],
                                          [f32(rng.uniform(-1,1)) for _ in range(dim)])
        row_stride=dim*query_heads*3;head_stride=dim*2;dim_stride=2
        query=[0.]*(row_stride*2+head_stride*query_heads)
        for row in range(2):
            for head in range(query_heads):
                for d in range(dim):query[row*row_stride+head*head_stride+d*dim_stride]=f32(rng.uniform(-.2,.2))
        for row in range(2):
            for head in range(query_heads):
                kvh=head//(query_heads//kv_heads);q=[query[row*row_stride+head*head_stride+d*dim_stride] for d in range(dim)]
                scores=[];values=[]
                for token in range(lower,upper):
                    k,v=store[pages[token//64-first],kvh,token&63]
                    scores.append(f32(sum(a*b for a,b in zip(q,k))*scale));values.append(v)
                maximum=max(scores);weights=[math.exp(x-maximum) for x in scores];den=sum(weights)
                expected=[sum(w*v[d] for w,v in zip(weights,values))/den for d in range(dim)]
                for size in (128,256):
                    parts=[]
                    for begin in range(0,visible,size):
                        parts.append(merge([accumulate(scores[begin+s:min(visible,begin+size):4],values[begin+s:min(visible,begin+size):4]) for s in range(4)]))
                    _,d,n=merge_striped(parts)
                    self.assertLess(max(abs(x/d-y) for x,y in zip(n,expected)),2e-6)

    def test_source_lifetime_barrier_counters_and_old_kernel_preservation(self):
        source=(ROOT/'native/paged_kv/arena.cpp').read_text()
        split=source[source.index('    if (split_plan_.partition_tokens != 0)'):source.index('    const bool inline_metadata =')]
        self.assertIn('encoder.add_temporary(scratch)',split)
        self.assertIn('(void)scratch_owner',split)
        self.assertIn('pipeline->threadExecutionWidth() != 32',split)
        self.assertIn('staticThreadgroupMemoryLength()',split)
        self.assertLess(split.index('addCompletedHandler'),split.index('encoder.dispatch_threadgroups'))
        self.assertLess(split.index('encoder.dispatch_threadgroups'),split.index('record_q1_split_partial_dispatch'))
        self.assertLess(split.index('record_q1_split_partial_dispatch'),split.index('encoder.set_input_array(scratch, 0)'))
        self.assertLess(split.index('record_q1_split_reduce_dispatch'),split.index('dispatched->store(true'))
        self.assertIn('split_plan.partition_tokens == 0 && use_inline_q1_metadata',source)

if __name__=='__main__':unittest.main()
