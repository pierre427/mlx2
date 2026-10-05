"""Host-only pure admission tests for environments unable to import MLX.

Run directly with Python (unittest), outside pytest's MLX conftest. These
execute the actual structural functions on shape/dtype records, not tensor
math. --baseline executes the pre-change source and must fail 27B admission.
The tiny native model tests remain in test_qwen38_fused_gdn.py.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Optional
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = 'src/mlx2/runtime/models/qwen4_fused_gdn.py'
BASELINE = '--baseline' in sys.argv
if BASELINE:
    sys.argv.remove('--baseline')


BASELINE_COMMIT = '531a876d'


def baseline_available():
    """The pre-change commit exists only in the private history."""
    probe = subprocess.run(['git', 'cat-file', '-e', BASELINE_COMMIT + ':' + SOURCE_PATH],
                           cwd=ROOT, capture_output=True, check=False)
    return probe.returncode == 0


def source(baseline=False):
    if baseline:
        return subprocess.check_output(['git', 'show', BASELINE_COMMIT + ':' + SOURCE_PATH],
                                       cwd=ROOT, text=True)
    return (ROOT / SOURCE_PATH).read_text()


def pure_admission(text):
    tree = ast.parse(text)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id == '_HEADER' for n in node.targets
        ):
            break
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            if isinstance(node, ast.Assign) and any(
                isinstance(n, ast.Name) and n.id == 'logger' for n in node.targets
            ):
                continue
            nodes.append(node)
    scope = dict(__name__=__name__, dataclass=dataclass, Any=Any, Optional=Optional,
                 mx=SimpleNamespace(bfloat16='bf16', float32='fp32', float16='fp16'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), SOURCE_PATH, 'exec'), scope)
    return scope


@dataclass
class Operand:
    shape: tuple
    dtype: str = 'bf16'


def operands(rows=1, architecture='qwen38'):
    return dict(qkv=Operand((rows, 1, 10240)), z=Operand((rows, 1, 6144)),
        a=Operand((rows, 1, 48)), b=Operand((rows, 1, 48)),
        conv_state=Operand((rows, 3, 10240)), recurrent_state=Operand((rows, 48, 128, 128), 'fp32'),
        conv_weight=Operand((10240, 4, 1)), A_log=Operand((48,), 'fp32'),
        dt_bias=Operand((48,)), norm_weight=Operand((128,)),
        mask=None, spans=(), training=False, sharded=False, speculating=False,
        num_key_heads=16, num_value_heads=48, key_head_dim=128, value_head_dim=128,
        conv_kernel=4, gate_activation='swish' if architecture == 'qwen38' else 'sigmoid',
        architecture=architecture)


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = pure_admission(source(BASELINE))

    def admit(self, rows, **changes):
        kw = operands(rows)
        kw.update(changes)
        fn = 'admit_qwen4_fused_gdn_decode' if rows == 1 else 'admit_qwen4_fused_gdn_batch_decode'
        return self.ns[fn](**kw)

    def test_27b_geometry_reaches_admission_for_required_batches(self):
        for rows in (1, 2, 4, 8, 16):
            with self.subTest(rows=rows):
                self.assertTrue(self.admit(rows).accepted)

    def test_27b_admission_fails_closed_on_incompatible_operands(self):
        changes = [dict(training=True), dict(sharded=True), dict(speculating=True),
                   dict(num_value_heads=32), dict(gate_activation='sigmoid'),
                   dict(qkv=Operand((1, 1, 10240), 'fp32')),
                   dict(conv_state=Operand((1, 4, 10240))),
                   dict(recurrent_state=Operand((1, 48, 128, 128), 'bf16')),
                   dict(mask=object()), dict(spans=None)]
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(self.admit(1, **change).accepted)

    def test_flash_next_admission_and_metal_sources_are_unchanged(self):
        if not baseline_available():
            self.skipTest(f'private baseline commit {BASELINE_COMMIT} absent '
                          '(public mirror history does not carry it)')
        old = pure_admission(source(True))
        for rows in (1, 2, 4, 8, 16):
            name = 'admit_qwen4_fused_gdn_decode' if rows == 1 else 'admit_qwen4_fused_gdn_batch_decode'
            for change in ({}, {'training': True}, {'gate_activation': 'swish'},
                           {'num_value_heads': 32}, {'spans': None}):
                kw = operands(rows, 'qwen4')
                kw.update(change)
                a, b = self.ns[name](**kw), old[name](**kw)
                self.assertEqual((a.accepted, a.reason), (b.accepted, b.reason))
        def literals(text):
            return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(text).body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in ('_HEADER', '_SOURCE')}
        self.assertEqual(literals(source()), literals(source(True)))


if __name__ == '__main__':
    unittest.main()
