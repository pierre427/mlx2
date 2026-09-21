from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest

from mlx2.semantic_sidecar import SemanticServingMiddleware
from mlx2.server import score_choice_tokens_via_engine


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

    def test_same_model_classifier_can_defer_a_candidate(self):
        self.sidecar.classifier_token_ids = {
            "store": 10,
            "defer": 11,
            "reject": 12,
        }
        body = {
            "messages": [{"role": "user", "content": "Remember that the code is amber."}],
            "session_id": "classified",
        }
        _, state = self.sidecar.prepare(
            body, tenant_id="alice", authenticated_tenant=True
        )
        result = self.sidecar.complete(
            state,
            "Okay.",
            score_tokens=lambda _prompt, _tokens: {
                "store": 0.0,
                "defer": 5.0,
                "reject": -2.0,
            },
        )
        self.assertEqual(result["accepted_edges"], 0)
        self.assertEqual(result["deferred_proposals"], 1)

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

    def test_classifier_scores_travel_through_serving_job(self):
        events = queue.Queue()
        events.put(
            {
                "logprob": {
                    "id": 10,
                    "token": "store",
                    "bytes": list(b" store"),
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"id": 10, "logprob": -0.1},
                        {"id": 11, "logprob": -2.0},
                        {"id": 12, "logprob": -3.0},
                    ],
                }
            }
        )
        events.put({"finish_reason": "length", "receipt": {}})
        job = SimpleNamespace(
            events=events,
            cancelled=__import__("threading").Event(),
            prompt_tokens=5,
            completion_tokens=1,
        )

        class Engine:
            def submit(self, request, *, tenant_id):
                self.request, self.tenant_id = request, tenant_id
                return job

        engine = Engine()
        scores = score_choice_tokens_via_engine(
            engine,
            "classify",
            {"store": 10, "defer": 11, "reject": 12},
            tenant_id="alice",
        )
        self.assertEqual(scores, {"store": -0.1, "defer": -2.0, "reject": -3.0})
        self.assertEqual(engine.request["max_tokens"], 1)
        self.assertEqual(engine.request["top_logprobs"], 3)
        self.assertTrue(job.cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
