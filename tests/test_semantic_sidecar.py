from pathlib import Path
import queue
import tempfile
import threading
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
        prepared, next_state = self.sidecar.prepare(
            {
                "messages": [{"role": "user", "content": "What is the code?"}],
                "session_id": "classified",
            },
            tenant_id="alice",
            authenticated_tenant=True,
        )
        self.assertEqual(prepared["messages"][0]["role"], "user")
        self.assertEqual(next_state.retrieved_concepts, 0)

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

    def test_request_prepared_before_delete_cannot_commit_after_it(self):
        def body(text):
            return {"session_id": "s1", "messages": [{"role": "user", "content": text}]}

        _, in_flight = self.sidecar.prepare(
            body("Remember that my pin is 4242."),
            tenant_id="alice",
            authenticated_tenant=True,
        )
        _, other = self.sidecar.prepare(
            body("Remember that the colour is blue."),
            tenant_id="alice",
            authenticated_tenant=True,
        )
        self.assertTrue(self.sidecar.complete(other, "ok")["committed"])
        self.assertTrue(self.sidecar.delete_session("alice", "s1"))
        self.assertFalse(self.sidecar.delete_session("alice", "s1"))
        # The in-flight request was prepared at the pre-delete revision; the
        # delete must invalidate it rather than let it resurrect the session.
        stale = self.sidecar.complete(in_flight, "ok")
        self.assertFalse(stale["committed"])
        prepared, state = self.sidecar.prepare(
            body("Remember that my colour pin is 7."),
            tenant_id="alice",
            authenticated_tenant=True,
        )
        self.assertEqual(prepared["messages"][0]["role"], "user")
        self.assertEqual(state.retrieved_concepts, 0)
        fresh = self.sidecar.complete(state, "ok")
        self.assertTrue(fresh["committed"])
        self.assertGreater(fresh["revision"], state.directory_revision)
        context = self.sidecar._context("alice", "s1")
        recalled = self.sidecar.memory.retrieve(context, "my pin 4242 colour blue")
        labels = {concept["label"] for concept in recalled.concepts}
        self.assertNotIn("4242", labels)
        self.assertNotIn("blue", labels)
        self.assertIn("7", labels)

    def _remember(self, session, text):
        _, state = self.sidecar.prepare(
            {"session_id": session, "messages": [{"role": "user", "content": text}]},
            tenant_id="bob",
            authenticated_tenant=True,
        )
        return self.sidecar.complete(state, "ok")

    def _capsule_texts(self):
        objects = Path(self.temporary.name) / "capsules" / "objects"
        return [path.read_text() for path in objects.glob("*.json")]

    def test_delete_removes_session_capsules_but_keeps_shared_ones(self):
        # Both sessions commit the same first turn, so content addressing
        # gives them one shared base capsule.
        self.assertTrue(self._remember("s1", "Remember that the colour is blue.")["committed"])
        self.assertTrue(self._remember("s2", "Remember that the colour is blue.")["committed"])
        for index in range(3):
            self.assertTrue(
                self._remember("s1", f"Remember that secret{index} is value{index}.")[
                    "committed"
                ]
            )
        self.assertTrue(any("secret0" in text for text in self._capsule_texts()))
        self.assertTrue(self.sidecar.delete_session("bob", "s1"))
        texts = self._capsule_texts()
        self.assertEqual(sum("secret" in text or "value" in text for text in texts), 0)
        self.assertEqual(len(texts), 1)
        survivor = self.sidecar.memory.retrieve(
            self.sidecar._context("bob", "s2"), "the colour"
        )
        self.assertIn("blue", {concept["label"] for concept in survivor.concepts})
        self.assertTrue(self._remember("s2", "Remember that the shape is round.")["committed"])

    def test_delete_cannot_sweep_a_capsule_an_in_flight_commit_reuses(self):
        self.assertTrue(self._remember("s1", "Remember that the colour is blue.")["committed"])
        self.assertTrue(self._remember("s1", "Remember that the pin is 4242.")["committed"])
        capsules = self.sidecar.memory.capsules
        original_put = capsules.put
        deleted = []
        threads = []

        def put_then_race_a_delete(**kwargs):
            # s2's first commit reproduces s1's base capsule. Deleting s1
            # while s2 has put but not yet published that capsule must not
            # remove it out from under s2.
            identity = original_put(**kwargs)
            deleter = threading.Thread(
                target=lambda: deleted.append(self.sidecar.delete_session("bob", "s1"))
            )
            threads.append(deleter)
            deleter.start()
            deleter.join(0.2)
            return identity

        capsules.put = put_then_race_a_delete
        try:
            result = self._remember("s2", "Remember that the colour is blue.")
        finally:
            capsules.put = original_put
        for thread in threads:
            thread.join(5)
        self.assertTrue(result["committed"], result)
        self.assertEqual(deleted, [True])
        capsules.get(result["capsule"])
        texts = self._capsule_texts()
        self.assertEqual(sum("4242" in text for text in texts), 0)
        survivor = self.sidecar.memory.retrieve(
            self.sidecar._context("bob", "s2"), "the colour"
        )
        self.assertIn("blue", {concept["label"] for concept in survivor.concepts})

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
