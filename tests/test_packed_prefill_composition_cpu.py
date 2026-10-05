"""CPU source/profile composition: owned Tree15 + packed routes stay explicit."""
import argparse,ast,importlib.abc,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('GPU import forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(ROOT/'src'))
from mlx2.runtime.paged_packed_prefill_serving_profile import make_profile,factory_profile
class Tests(unittest.TestCase):
    def test_server_merge_keeps_both_explicit_startup_validations(self):
        source=(ROOT/'src/mlx2/server.py').read_text();tree=ast.parse(source)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        text=ast.unparse(main)
        self.assertIn('packed_paths =',text);self.assertIn('owned_paths =',text)
        self.assertIn('packed-prefill requires profile, artifact manifest and MLX wheel',text)
        self.assertIn('TensorFold-owned live route requires its explicit flag and all three pinned paths',text)
        self.assertNotIn('<<<<<<<',source)
    def test_merged_packed_cli_options_remain_unselected_by_default(self):
        tree=ast.parse((ROOT/'src/mlx2/server.py').read_text())
        parser_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_parser')
        names={'--native-packed-prefill-profile','--native-packed-prefill-manifest','--native-packed-prefill-mlx-wheel'}
        nodes=[n for n in ast.walk(parser_node) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
               isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='add_argument' and
               n.value.args and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value in names]
        self.assertEqual(len(nodes),3)
        parser=argparse.ArgumentParser();exec(compile(ast.Module(body=nodes,type_ignores=[]),'merged CLI','exec'),{'parser':parser})
        options=parser.parse_args([])
        self.assertTrue(all(value is None for value in vars(options).values()))
    def test_optimized_factory_profile_preserves_explicit_block_one_default(self):
        p=make_profile({});self.assertEqual(p['required_environment']['MLX2_PAGED_PREFILL_EVAL_BLOCK_SIZE'],'1')
        self.assertNotIn('prefill_eval_block_size',factory_profile(p,(32,96)))
        self.assertFalse(p['qualified']);self.assertFalse(p['serving_default']);self.assertFalse(p['price_usable'])
if __name__=='__main__':unittest.main()
