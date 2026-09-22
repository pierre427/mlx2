import json
import os
from pathlib import Path
import tempfile
import unittest

from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory, Scope
from mlx2.runtime.semantic_capsules import CapsuleIntegrityError, CapsuleStore


class HyperDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.capsules = CapsuleStore(root / "capsules")
        self.directory = HyperDirectory(root / "directory", self.capsules)

    def tearDown(self):
        self.temporary.cleanup()

    def capsule(self, label, *, parents=()):
        return self.capsules.put(
            kind="semantic_delta" if parents else "semantic_base",
            data={"label": label},
            parents=parents,
            model_binding="model-fingerprint",
            tokenizer_binding="tokenizer-fingerprint",
            runtime_binding="runtime-fingerprint",
            provenance={"source": "unit-test"},
        )

    def test_capsules_are_content_addressed_immutable_and_private(self):
        first = self.capsule("forest")
        same = self.capsule("forest")
        second = self.capsule("leaves", parents=(first.digest,))
        self.assertEqual(first.digest, same.digest)
        self.assertNotEqual(first.digest, second.digest)
        path = self.capsules.objects / f"{first.digest}.json"
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.capsules.get(second.digest)["parents"], [first.digest])

    def test_corruption_is_quarantined_and_fails_closed(self):
        capsule = self.capsule("forest")
        path = self.capsules.objects / f"{capsule.digest}.json"
        value = json.loads(path.read_text())
        value["data"]["label"] = "tampered"
        path.write_text(json.dumps(value))
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(CapsuleIntegrityError, "quarantined"):
            self.capsules.get(capsule.digest)
        self.assertFalse(path.exists())
        self.assertEqual(len(list(self.capsules.quarantine.iterdir())), 1)

    def test_layers_override_and_fingerprint_binds_revisions(self):
        global_capsule = self.capsule("global")
        session_capsule = self.capsule("session", parents=(global_capsule.digest,))
        context = DirectoryContext(model="qwen", tenant="tenant-a", session="s1")
        self.directory.update(
            Scope.GLOBAL,
            context,
            expected_revision=0,
            handles={"memory": global_capsule.digest},
            policies={"retrieval_limit": 4},
        )
        before = self.directory.resolve(context)
        self.directory.update(
            Scope.SESSION,
            context,
            expected_revision=0,
            handles={"memory": session_capsule.digest},
            policies={"retrieval_limit": 8},
            relationships=[
                {"source": "memory", "type": "derived_from", "target": "global-memory"}
            ],
        )
        after = self.directory.resolve(context)
        self.assertNotEqual(before.fingerprint, after.fingerprint)
        self.assertEqual(after.handles["memory"], session_capsule.digest)
        self.assertEqual(after.policies["retrieval_limit"], 8)
        self.assertEqual(len(after.relationships), 1)

    def test_compare_and_swap_and_tenant_isolation(self):
        capsule = self.capsule("shared-payload")
        a = DirectoryContext(model="qwen", tenant="a", session="same")
        b = DirectoryContext(model="qwen", tenant="b", session="same")
        self.directory.update(
            Scope.SESSION, a, expected_revision=0, handles={"memory": capsule.digest}
        )
        with self.assertRaisesRegex(ValueError, "revision conflict"):
            self.directory.update(Scope.SESSION, a, expected_revision=0)
        self.assertIn("memory", self.directory.resolve(a).handles)
        self.assertNotIn("memory", self.directory.resolve(b).handles)
        self.assertTrue(self.directory.delete_session(a))
        self.assertNotIn("memory", self.directory.resolve(a).handles)

    def test_scope_names_with_separator_do_not_collide(self):
        a = DirectoryContext(model="m--t", tenant="u", session="s")
        b = DirectoryContext(model="m", tenant="t--u", session="s")
        capsule = self.capsule("a")
        self.directory.update(Scope.SESSION, a, expected_revision=0, handles={"memory": capsule.digest})
        self.assertNotEqual(self.directory._path(Scope.SESSION, a.key_for(Scope.SESSION)),
                            self.directory._path(Scope.SESSION, b.key_for(Scope.SESSION)))
        self.assertNotIn("memory", self.directory.resolve(b).handles)
        self.directory.update(Scope.SESSION, b, expected_revision=0)
        self.assertIn("memory", self.directory.resolve(a).handles)
        self.assertEqual(self.directory.resolve(b).layers[-1]["revision"], 1)

    def test_legacy_layer_is_read_and_migrated_on_update(self):
        context = DirectoryContext(model="m", tenant="t", session="s")
        key = context.key_for(Scope.SESSION)
        legacy = self.directory._legacy_path(Scope.SESSION, key)
        legacy.write_text(json.dumps({
            "schema": "mlx2-hyper-directory-v1", "scope": "session", "key": list(key),
            "revision": 1, "handles": {}, "policies": {}, "relationships": [],
        }))
        self.assertEqual(self.directory.resolve(context).layers[-1]["revision"], 1)
        self.directory.update(Scope.SESSION, context, expected_revision=1)
        self.assertEqual(self.directory.resolve(context).layers[-1]["revision"], 2)
        self.assertTrue(self.directory.delete_session(context))
        self.assertFalse(legacy.exists())


if __name__ == "__main__":
    unittest.main()
