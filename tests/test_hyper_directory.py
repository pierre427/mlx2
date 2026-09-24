import json
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest

from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory, Scope
from mlx2.runtime.semantic_capsules import CapsuleIntegrityError, CapsuleStore, canonical_json


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

    def test_delete_keeps_session_revisions_monotonic(self):
        capsule = self.capsule("session-payload")
        context = DirectoryContext(model="qwen", tenant="a", session="s")
        self.directory.update(
            Scope.SESSION, context, expected_revision=0, handles={"memory": capsule.digest}
        )
        self.assertTrue(self.directory.delete_session(context))
        resolved = self.directory.resolve(context)
        self.assertEqual(resolved.layers[-1]["revision"], 2)
        self.assertEqual(resolved.handles, {})
        # A writer that last saw the empty session at revision 0 is stale.
        with self.assertRaisesRegex(ValueError, "revision conflict"):
            self.directory.update(
                Scope.SESSION, context, expected_revision=0, handles={"memory": capsule.digest}
            )
        self.assertFalse(self.directory.delete_session(context))
        layer = self.directory.update(Scope.SESSION, context, expected_revision=2)
        self.assertEqual(layer["revision"], 3)
        self.assertNotIn("deleted", layer)

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

    def test_delete_removes_capsules_only_the_legacy_layer_still_names(self):
        # A session migrated from its legacy layer keeps that file until the
        # delete, and a derived capsule rebuilt after the migration leaves
        # the old one named only there: it holds the session's facts too.
        context = DirectoryContext(model="m", tenant="t", session="s")
        other = DirectoryContext(model="m", tenant="t", session="other")
        key = context.key_for(Scope.SESSION)
        shared = self.capsule("shared")
        base = self.capsule("secret v0", parents=(shared.digest,))
        neural = self.capsules.put(
            kind="neural_state",
            data={"label": "derived from v0"},
            parents=(base.digest,),
            provenance={"source": "unit-test"},
        )
        legacy = self.directory._legacy_path(Scope.SESSION, key)
        legacy.write_text(json.dumps({
            "schema": "mlx2-hyper-directory-v1", "scope": "session", "key": list(key),
            "revision": 1,
            "handles": {"semantic": base.digest, "neural": neural.digest},
            "policies": {}, "relationships": [],
        }))
        self.directory.update(Scope.SESSION, other, expected_revision=0,
                              handles={"memory": shared.digest})
        self.directory.update(Scope.SESSION, context, expected_revision=1)
        current = self.capsule("secret v1", parents=(base.digest,))
        rebuilt = self.capsules.put(
            kind="neural_state",
            data={"label": "derived from v1"},
            parents=(current.digest,),
            provenance={"source": "unit-test"},
        )
        self.directory.update(
            Scope.SESSION, context, expected_revision=2,
            handles={"semantic": current.digest, "neural": rebuilt.digest},
        )
        self.assertTrue(legacy.exists())

        self.assertTrue(self.directory.delete_session(context))

        self.assertFalse(legacy.exists())
        remaining = {path.stem for path in self.capsules.objects.glob("*.json")}
        self.assertEqual(remaining, {shared.digest})

    def test_delete_refuses_a_malformed_legacy_layer_beside_a_valid_one(self):
        # The canonical layer shadows the legacy file on every read, so a
        # corrupted or crafted legacy file is never validated by resolving.
        # Its handles must not reach the doomed closure: they could name any
        # capsule no layer references, such as one a commit has put but not
        # yet published.
        context = DirectoryContext(model="m", tenant="t", session="s")
        other = DirectoryContext(model="m", tenant="t", session="other")
        key = context.key_for(Scope.SESSION)
        own = self.capsule("own secret")
        shared = self.capsule("shared")
        victim = self.capsule("unpublished")
        self.directory.update(Scope.SESSION, other, expected_revision=0,
                              handles={"memory": shared.digest})
        self.directory.update(Scope.SESSION, context, expected_revision=0,
                              handles={"memory": own.digest, "base": shared.digest})
        legacy = self.directory._legacy_path(Scope.SESSION, key)
        valid = {
            "schema": "mlx2-hyper-directory-v1", "scope": "session", "key": list(key),
            "revision": 1, "handles": {"memory": victim.digest},
            "policies": {}, "relationships": [],
        }
        malformed = {
            "missing schema": {k: v for k, v in valid.items() if k != "schema"},
            "wrong schema": {**valid, "schema": "mlx2-hyper-directory-v0"},
            "string revision": {**valid, "revision": "1"},
            "negative revision": {**valid, "revision": -1},
            "handle name": {**valid, "handles": {"../memory": victim.digest}},
            "policies": {**valid, "policies": ["retrieval_limit"]},
            "relationships": {**valid, "relationships": [{"source": "memory"}]},
        }
        objects = {own.digest, shared.digest, victim.digest}
        for label, value in malformed.items():
            with self.subTest(label):
                legacy.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "invalid hyper directory layer"):
                    self.directory.delete_session(context)
                remaining = {path.stem for path in self.capsules.objects.glob("*.json")}
                self.assertEqual(remaining, objects)
                self.assertTrue(legacy.exists())
                # No tombstone either: the session still resolves as it was.
                resolved = self.directory.resolve(context)
                self.assertEqual(resolved.layers[-1]["revision"], 1)
                self.assertEqual(resolved.handles["memory"], own.digest)

        legacy.unlink()
        self.assertTrue(self.directory.delete_session(context))
        remaining = {path.stem for path in self.capsules.objects.glob("*.json")}
        self.assertEqual(remaining, {shared.digest, victim.digest})

    def test_composite_keys_do_not_alias_or_delete_other_sessions(self):
        a = DirectoryContext(model="qwen", tenant="a--b", session="c")
        b = DirectoryContext(model="qwen", tenant="a", session="b--c")
        self.directory.update(Scope.SESSION, a, expected_revision=0, policies={"owner": "a"})
        self.assertFalse(self.directory.delete_session(b))
        self.directory.update(Scope.SESSION, b, expected_revision=0, policies={"owner": "b"})
        self.assertEqual(self.directory.resolve(a).policies["owner"], "a")
        self.assertEqual(self.directory.resolve(b).policies["owner"], "b")
        self.directory.delete_session(b)
        self.assertEqual(self.directory.resolve(a).policies["owner"], "a")

    def test_maximum_length_scope_components_fit_filesystem(self):
        context = DirectoryContext(model="m" * 128, tenant="t" * 128, session="s" * 128)
        self.directory.update(Scope.SESSION, context, expected_revision=0, policies={"limit": 4})
        self.assertEqual(self.directory.resolve(context).policies["limit"], 4)
        self.assertTrue(self.directory.delete_session(context))

    def test_previous_hashed_layer_migrates_and_deletes_without_resurrection(self):
        context = DirectoryContext(model="qwen", tenant="a", session="s")
        key = context.key_for(Scope.SESSION)
        identity = hashlib.sha256(canonical_json(["session", *key])).hexdigest()
        previous = self.directory.root / f"session--{identity}.json"
        capsule = self.capsule("previous hash format")
        layer = self.directory._empty(Scope.SESSION, key)
        layer.update(revision=3, handles={"memory": capsule.digest})
        previous.write_text(json.dumps(layer))
        self.assertEqual(self.directory.resolve(context).handles["memory"], capsule.digest)
        self.directory.update(Scope.SESSION, context, expected_revision=3)
        self.assertTrue(self.directory.delete_session(context))
        self.assertFalse(previous.exists())
        resolved = self.directory.resolve(context)
        self.assertEqual(resolved.handles, {})
        self.assertEqual(resolved.layers[-1]["revision"], 5)
        self.assertFalse((self.capsules.objects / f"{capsule.digest}.json").exists())

    def test_nested_transaction_blocks_other_instances_until_publication(self):
        context = DirectoryContext(model="qwen", tenant="a", session="s")
        other = HyperDirectory(self.directory.root, self.capsules)
        started, finished = threading.Event(), threading.Event()
        results = []

        def delete():
            started.set()
            results.append(other.delete_session(context))
            finished.set()

        thread = threading.Thread(target=delete)
        try:
            with self.directory.transaction():
                capsule = self.capsule("pending publication")
                with self.directory.transaction():
                    self.directory.update(
                        Scope.SESSION, context, expected_revision=0,
                        handles={"memory": capsule.digest},
                    )
                    self.assertEqual(self.directory.resolve(context).handles["memory"], capsule.digest)
                thread.start()
                self.assertTrue(started.wait(5))
                self.assertFalse(finished.wait(0.1))
                self.assertTrue((self.capsules.objects / f"{capsule.digest}.json").exists())
        finally:
            if thread.ident is not None:
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [True])

    def test_legacy_layer_migrates_without_resurrection(self):
        context = DirectoryContext(model="qwen", tenant="a", session="s")
        legacy = self.directory.root / "session--qwen--a--s.json"
        value = self.directory._empty(Scope.SESSION, context.key_for(Scope.SESSION))
        value.update(revision=3, policies={"legacy": True})
        legacy.write_text(json.dumps(value))
        self.assertEqual(self.directory.resolve(context).policies, {"legacy": True})
        self.directory.update(Scope.SESSION, context, expected_revision=3, policies={"new": True})
        self.assertEqual(self.directory.resolve(context).layers[-1]["revision"], 4)
        self.assertTrue(self.directory.delete_session(context))
        self.assertEqual(self.directory.resolve(context).policies, {})

    def test_corrupt_non_object_and_non_utf8_capsules_are_quarantined(self):
        for payload in (b"null", b"[]", b"\xff"):
            with self.subTest(payload=payload):
                capsule = self.capsule(str(payload))
                path = self.capsules.objects / f"{capsule.digest}.json"
                path.write_bytes(payload)
                with self.assertRaisesRegex(CapsuleIntegrityError, "quarantined"):
                    self.capsules.get(capsule.digest)
                self.assertFalse(path.exists())

    def test_compare_and_swap_serializes_independent_directory_instances(self):
        context = DirectoryContext(model="qwen", tenant="a", session="s")
        other = HyperDirectory(self.directory.root, self.capsules)
        read = self.directory._read
        first_read, release, second_done = threading.Event(), threading.Event(), threading.Event()
        results = []

        def paused_read(scope, key):
            value = read(scope, key)
            first_read.set()
            if not release.wait(5):
                raise RuntimeError("test update was not released")
            return value

        def update(directory, completed=None):
            try:
                directory.update(Scope.SESSION, context, expected_revision=0)
                results.append("committed")
            except ValueError as error:
                results.append(str(error))
            finally:
                if completed is not None:
                    completed.set()

        self.directory._read = paused_read
        first = threading.Thread(target=update, args=(self.directory,))
        second = threading.Thread(target=update, args=(other, second_done))
        first.start()
        try:
            self.assertTrue(first_read.wait(5))
            second.start()
            premature_commit = second_done.wait(0.1)
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(premature_commit)
        self.assertEqual(results.count("committed"), 1)
        self.assertTrue(any("revision conflict" in result for result in results))


if __name__ == "__main__":
    unittest.main()
