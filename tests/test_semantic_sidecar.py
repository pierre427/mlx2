from pathlib import Path
import tempfile
import unittest

from mlx2.semantic_sidecar import SemanticServingMiddleware


class SemanticSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.sidecar = SemanticServingMiddleware.create(
            Path(self.temporary.name),
            model_binding="qwen9b",
            tokenizer_binding="qwen-tokenizer",
            runtime_binding="mlx2-runtime",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_automatic_post_delivery_memory_round_trip(self):
        first = {
            "messages": [
                {
                    "role": "user",
                    "content": "Remember that my favorite trail is Cedar Loop.",
                }
            ],
            "session_id": "walk",
        }
        prepared, state = self.sidecar.prepare(
            first, tenant_id="alice@example", authenticated_tenant=True
        )
        self.assertEqual(prepared["messages"], first["messages"])
        receipt = self.sidecar.receipt(state)
        self.assertEqual(receipt["commit_boundary"], "after-response-delivery")
        commit = self.sidecar.complete(state, "I'll remember that.")
        self.assertTrue(commit["committed"])

        followup = {
            "messages": [{"role": "user", "content": "What is my favorite trail?"}],
            "session_id": "walk",
        }
        prepared, state = self.sidecar.prepare(
            followup, tenant_id="alice@example", authenticated_tenant=True
        )
        self.assertEqual(prepared["messages"][0]["role"], "system")
        self.assertIn("my favorite trail has_property cedar loop", prepared["messages"][0]["content"])
        self.assertGreater(state.retrieved_concepts, 0)

    def test_no_durable_write_without_authenticated_tenant(self):
        body = {
            "messages": [{"role": "user", "content": "Remember that the code is amber."}],
            "session_id": "s",
        }
        _, state = self.sidecar.prepare(
            body, tenant_id="header-only", authenticated_tenant=False
        )
        result = self.sidecar.complete(state, "Okay.")
        self.assertEqual(result["reason"], "request-local-only")

    def test_directory_fingerprint_joins_request_scope_and_delete(self):
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "s",
        }
        prepared, _ = self.sidecar.prepare(
            body, tenant_id="alice", authenticated_tenant=True
        )
        self.assertRegex(prepared["_mlx2_semantic_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertFalse(self.sidecar.delete_session("alice", "s"))


if __name__ == "__main__":
    unittest.main()
