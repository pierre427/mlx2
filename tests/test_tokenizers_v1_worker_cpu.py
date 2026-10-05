"""Standalone unittest CPU gate; no mlx2 package initialization or MLX imports.

Run with baseline isolated Python and MLX2_TOKENIZERS_V1_TEST_MANIFEST pointing
at the locally generated candidate manifest. The test is intentionally opt-in.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'src/mlx2/runtime'
PACKAGE = '_mlx2_tokenizer_cpu'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(RUNTIME)]
sys.modules[PACKAGE] = package

def load(name):
    full = PACKAGE + '.' + name
    spec = importlib.util.spec_from_file_location(full, RUNTIME/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module

worker_module = load('tokenizers_v1_worker')
utils = load('tokenizer_utils')
integrity = load('tokenizer_integrity')
MANIFEST = os.environ.get('MLX2_TOKENIZERS_V1_TEST_MANIFEST')

@unittest.skipUnless(MANIFEST, 'explicit isolated v1 manifest required')
class WorkerCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.manifest = json.loads(Path(MANIFEST).read_text())
        cls.model = Path(cls.manifest['tokenizer']['path']).parent
        cls.raw = AutoTokenizer.from_pretrained(str(cls.model),local_files_only=True,trust_remote_code=False)
        cls.raw.chat_template = Path(cls.manifest['chat_template']['path']).read_text()
        integrity.repair_loaded_tokenizer(cls.raw,cls.model)

    def worker(self):
        worker = worker_module.TokenizersV1Worker(MANIFEST,model_path=self.model)
        self.addCleanup(worker.close)
        return worker

    def wrapper(self, selected=True):
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop('MLX2_TOKENIZERS_V1_MANIFEST',None)
            wrapper = utils.TokenizerWrapper(self.raw,detokenizer_class=utils.BPEStreamingDetokenizer)
        self.addCleanup(wrapper.close)
        if selected:
            wrapper.enable_v1_encode(MANIFEST)
        return wrapper

    def test_exact_canaries_and_special_tokens(self):
        worker = self.worker()
        for case in self.manifest['canaries']:
            self.assertEqual(worker.encode(case['text'],add_special_tokens=case['add_special_tokens']),case['ids'])
        self.assertTrue(worker.status()['observed_used'])
        self.assertGreater(worker.status()['worker_peak_rss_bytes'],0)

    def test_default_off_short_path_and_unsupported_forms(self):
        ordinary = self.wrapper(False)
        self.assertFalse(ordinary.tokenizer_v1_status()['selected'])
        selected = self.wrapper()
        self.assertFalse(selected.tokenizer_v1_status()['observed_used'])
        self.assertEqual(selected.encode('Hello',add_special_tokens=False),self.raw.encode('Hello',add_special_tokens=False))
        self.assertEqual(selected.encode('Hello',text_pair='World'),self.raw.encode('Hello',text_pair='World'))
        self.assertEqual(selected.tokenizer_v1_status()['counts']['successful_encodes'],0)

    def test_long_chat_template_and_existing_streaming_decoder(self):
        wrapper = self.wrapper()
        messages=[{'role':'user','content':('हिन्दी ไทย é 🤖 1234567890\n'*800)}]
        self.assertEqual(wrapper.apply_chat_template(messages,enable_thinking=False),
                         self.raw.apply_chat_template(messages,return_dict=False,enable_thinking=False))
        self.assertTrue(wrapper.tokenizer_v1_status()['observed_used'])
        tokens=wrapper.encode('é e\u0301 🤖 हिन्दी ไทย',add_special_tokens=False)
        decoder=wrapper.detokenizer
        for token in tokens:
            decoder.add_token(token)
        decoder.finalize()
        self.assertEqual(decoder.text,self.raw.decode(tokens))

    def test_no_fork_thread_serialization_and_explicit_restart(self):
        worker=self.worker()
        with patch.object(os,'fork',side_effect=AssertionError('fork forbidden')):
            worker.start()
            outputs=[]
            threads=[threading.Thread(target=lambda:outputs.append(worker.encode('Unicode 🤖',add_special_tokens=False))) for _ in range(8)]
            for thread in threads:thread.start()
            for thread in threads:thread.join()
            self.assertEqual(outputs,[self.raw.encode('Unicode 🤖',add_special_tokens=False)]*8)
            first=worker.status()['pid']; worker.restart()
            self.assertNotEqual(first,worker.status()['pid'])
        worker.close()
        self.assertIsNone(worker.status()['pid'])
        with self.assertRaises(RuntimeError):worker.encode('closed')

    def test_crash_is_fail_closed_then_next_request_respawns(self):
        worker=self.worker();worker.start();pid=worker.status()['pid']
        os.kill(pid,signal.SIGKILL)
        with self.assertRaises((RuntimeError,BrokenPipeError)):worker.encode('after crash')
        self.assertIsNone(worker.status()['pid'])
        self.assertEqual(worker.encode('recovered',add_special_tokens=False),self.raw.encode('recovered',add_special_tokens=False))
        self.assertEqual(worker.status()['counts']['spawns'],2)

    def test_timeout_reaps_worker(self):
        worker=self.worker();worker.start();pid=worker.status()['pid']
        worker.timeout=.05
        os.kill(pid,signal.SIGSTOP)
        with self.assertRaises(TimeoutError):worker.encode('timed out')
        self.assertIsNone(worker.status()['pid'])
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_identity_and_loaded_component_drift_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest=dict(self.manifest);manifest['extension']=dict(manifest['extension'],sha256='0'*64)
            path=Path(directory)/'manifest.json';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'extension identity'):
                worker_module.TokenizersV1Worker(path,model_path=self.model)
        wrapper=self.wrapper(False)
        original=self.raw.backend_tokenizer.pre_tokenizer
        try:
            from tokenizers.pre_tokenizers import ByteLevel
            self.raw.backend_tokenizer.pre_tokenizer=ByteLevel()
            with self.assertRaisesRegex(ValueError,'pre_tokenizer differs'):
                wrapper.enable_v1_encode(MANIFEST)
        finally:self.raw.backend_tokenizer.pre_tokenizer=original

    def test_environment_selects_explicit_manifest_and_model_mismatch_refuses(self):
        with patch.dict(os.environ,{'MLX2_TOKENIZERS_V1_MANIFEST':MANIFEST}):
            wrapper=utils.TokenizerWrapper(self.raw)
            self.addCleanup(wrapper.close)
            status=wrapper.tokenizer_v1_status()
            self.assertTrue(status['selected'])
            self.assertFalse(status['observed_used'])
            self.assertGreater(status['worker_peak_rss_bytes'],0)
        with self.assertRaisesRegex(ValueError,'model path mismatch'):
            worker_module.TokenizersV1Worker(MANIFEST,model_path='/tmp/unqualified-model')

    def test_startup_canary_refusal_reaps_child(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest=json.loads(json.dumps(self.manifest))
            manifest['canaries'][0]['ids']=[99999999]
            path=Path(directory)/'manifest.json';path.write_text(json.dumps(manifest))
            worker=worker_module.TokenizersV1Worker(path,model_path=self.model)
            self.addCleanup(worker.close)
            with self.assertRaises(RuntimeError):worker.start()
            self.assertIsNone(worker.status()['pid'])

    def test_batched_keyword_template_retains_reference_api(self):
        wrapper=self.wrapper()
        conversations=[[{'role':'user','content':'Hello'}],[{'role':'user','content':'World'}]]
        self.assertEqual(wrapper.apply_chat_template(conversation=conversations),
                         self.raw.apply_chat_template(conversation=conversations,return_dict=False,enable_thinking=wrapper.has_thinking))
        self.assertFalse(wrapper.tokenizer_v1_status()['observed_used'])

    def test_no_mlx_import(self):
        self.assertFalse(any(name=='mlx' or name.startswith('mlx.') for name in sys.modules))

if __name__=='__main__':
    unittest.main()
