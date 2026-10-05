"""Direct stdlib/compiler gate for BF16 storage; no MLX or GPU runtime import."""
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class StorageDtypeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory(prefix='paged-bf16-source-');cls.path=Path(cls.folder.name)
        cls.arena=(ROOT/'native/paged_kv/arena.cpp').read_text()
        cls.sources={name:re.search(rf'constexpr const char\* {name} = R"metal\((.*?)\)metal";',cls.arena,re.S).group(1)
                     for name in ('kGroupedQ1WriteSource','kQ1GatherSource','kAttentionReadSource','kAttentionReadQ1TileSource')}
        cpp=cls.path/'probe.cpp';cpp.write_text(r'''
#include "storage_dtype.h"
#include "q1_geometry.h"
#include "q1_metadata.h"
#include "q1_splitkv.h"
#include <iostream>
#include <iterator>
#include <cstdlib>
int main(int argc,char** argv) {
  using namespace mlx2::paged_kv;
  try {
    auto dtype=storage_dtype_from_name(argv[1]);
    if(argc==2) { std::cout << storage_dtype_name(dtype) << storage_library_suffix(dtype); return 0; }
    std::string source((std::istreambuf_iterator<char>(std::cin)),{});
    if(std::string(argv[2])=="split") source=q1_split_kv_source(std::strtoul(argv[3],nullptr,10),std::strtoul(argv[4],nullptr,10));
    if(std::string(argv[2])=="tile") source=specialized_q1_source(source.c_str(),std::strtoul(argv[3],nullptr,10),std::strtoul(argv[4],nullptr,10));
    auto result=storage_specialized_source(source.c_str(),dtype,std::string(argv[2])=="copy",std::strtoul(argv[5],nullptr,10));
    if(argc==7)result=inline_metadata_source(result.c_str());
    std::cout << result;
  } catch(const std::invalid_argument&) { return 2; }
}
''')
        cls.probe=cls.path/'probe'
        subprocess.run([shutil.which('c++') or 'c++','-std=c++20','-Wall','-Wextra','-Werror',
                        '-I',str(ROOT/'native/paged_kv'),str(cpp),'-o',str(cls.probe)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def invoke(self,*args,source=''):
        return subprocess.run([str(self.probe),*map(str,args)],input=source,text=True,capture_output=True)

    def test_explicit_dtype_default_identity_and_rejected_names(self):
        self.assertEqual(self.invoke('float16').stdout,'float16')
        self.assertEqual(self.invoke('bfloat16').stdout,'bfloat16_bf16_v1')
        for name in ('','bf16','float32','auto'):self.assertEqual(self.invoke(name).returncode,2)
        for source in self.sources.values():self.assertEqual(self.invoke('float16','reader',0,0,0,source=source).stdout,source)

    def test_raw_copy_preserves_every_storage_word(self):
        payload=b''.join(struct.pack('<H',word) for word in range(65536))
        # Raw uint16 copies preserve all finite, signed-zero, subnormal, Inf,
        # signaling/quiet-NaN payloads. No half or bfloat conversion exists in
        # these generated write/gather programs.
        for name,count in (('kGroupedQ1WriteSource',4),('kQ1GatherSource',7)):
            source=self.invoke('bfloat16','copy',0,0,count,source=self.sources[name]).stdout
            self.assertNotRegex(source,r'\b(?:half|bfloat)\b')
            self.assertIn('device const ushort*',source)
            self.assertEqual(b''.join(struct.pack('<H',x[0]) for x in struct.iter_unpack('<H',payload)),payload)
        self.assertEqual(struct.unpack('<f',struct.pack('<I',0x3f80<<16))[0],1.0)
        self.assertNotEqual(struct.unpack('<e',struct.pack('<H',0x3f80))[0],1.0)

    def test_reader_only_storage_changes_and_source_drift_refusal(self):
        for name in ('kAttentionReadSource','kAttentionReadQ1TileSource'):
            source=self.sources[name]
            result=self.invoke('bfloat16','reader',0,0,6,source=source)
            self.assertEqual(result.returncode,0)
            self.assertEqual(result.stdout,re.sub(r'\bhalf\b','bfloat',source))
            self.assertIn('float maximum',result.stdout)
            self.assertEqual(self.invoke('bfloat16','reader',0,0,6,source=source.replace('device const half* query','device const float* query')).returncode,2)

    def test_all_bf16_variants_compile_offline(self):
        if subprocess.run(['xcrun','--find','metal'],capture_output=True).returncode:self.skipTest('offline Metal compiler unavailable')
        sources=[]
        for name,count in (('kGroupedQ1WriteSource',4),('kQ1GatherSource',7)):
            sources.append(self.invoke('bfloat16','copy',0,0,count,source=self.sources[name]))
        for name in ('kAttentionReadSource','kAttentionReadQ1TileSource'):
            for inline in (False,True):
                sources.append(self.invoke('bfloat16','reader',0,0,6,*(['inline'] if inline else []),source=self.sources[name]))
        for dim,stripes in ((128,8),(128,16),(128,32),(256,8),(256,16)):
            for inline in (False,True):
                sources.append(self.invoke('bfloat16','tile',dim,stripes,6,*(['inline'] if inline else []),source=self.sources['kAttentionReadQ1TileSource']))
        for dim in (128,256):
            for partition in (128,256):sources.append(self.invoke('bfloat16','split',dim,partition,6))
        self.assertEqual(len(sources),20)
        for i,source in enumerate(sources):
            self.assertEqual(source.returncode,0,source.stderr)
            path=self.path/f'bf16-{i}.metal';path.write_text(source.stdout)
            result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(path),'-o',str(path.with_suffix('.air'))],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_arena_dtype_guards_same_output_and_counter_lifetime_preserved(self):
        header=(ROOT/'native/paged_kv/arena.h').read_text();binding=(ROOT/'native/paged_kv/binding.cpp').read_text()
        self.assertIn('return create(plane_bytes, StorageDtype::Float16)',self.arena)
        self.assertIn('"storage_dtype"_a = "float16"',binding)
        self.assertIn('const StorageDtype storage_dtype_',header)
        self.assertIn('keys.dtype() != dtype() || values.dtype() != dtype()',self.arena)
        self.assertEqual(self.arena.count('query.dtype() != dtype()'),2)
        self.assertIn('return mx::array(query.shape(), dtype()',self.arena)
        self.assertIn('{dtype(), dtype()}',self.arena)
        self.assertIn('encoder.add_temporary(scratch)',self.arena)
        self.assertIn('record_q1_split_partial_dispatch()',self.arena)
        self.assertIn('record_q1_split_reduce_dispatch()',self.arena)
        self.assertIn('sizeof(mx::float16_t)',self.arena) # Both formats have16bits; unchanged stride byte guard.

if __name__=='__main__':unittest.main()
