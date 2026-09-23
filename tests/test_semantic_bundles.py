from pathlib import Path
import tempfile
import unittest

from mlx2.runtime.classifier_bundle import AdaptiveBundleSelector, ForcedChoiceClassifier
from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import SemanticMemory, SemanticProposal, concept_token


class SemanticBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        capsules = CapsuleStore(root / "capsules")
        directory = HyperDirectory(root / "directory", capsules)
        self.memory = SemanticMemory(
            capsules=capsules,
            directory=directory,
            model_binding="qwen9b-artifact",
            tokenizer_binding="qwen-tokenizer",
            runtime_binding="mlx2-runtime",
        )
        self.context = DirectoryContext(model="qwen9b", tenant="alice", session="walk")

    def tearDown(self):
        self.temporary.cleanup()

    def test_concept_tokens_are_stable_and_normalized(self):
        self.assertEqual(concept_token("Walking in a Forest"), concept_token("walking in a forest"))
        self.assertNotEqual(concept_token("walking in a forest"), concept_token("smelling leaves"))

    def test_post_delivery_gate_and_relation_aware_retrieval(self):
        # Unsupported relation is rejected before persistence.
        with self.assertRaises(ValueError):
            SemanticProposal("a", "contains", "b", 1.0, 1.0, "e")
        proposals = [
            SemanticProposal("standard model", "part_of", "particle physics", 0.98, 0.72, "turn-1"),
            SemanticProposal("quarks", "related_to", "standard model", 0.96, 0.50, "turn-1"),
            SemanticProposal("string theory", "related_to", "standard model", 0.61, 0.03, "turn-1"),
        ]
        receipt = self.memory.commit_after_delivery(
            self.context,
            proposals,
            response_delivered=True,
            authenticated_tenant=True,
        )
        self.assertEqual(receipt["accepted_edges"], 2)
        self.assertEqual(receipt["deferred_proposals"], 1)
        result = self.memory.retrieve(self.context, "How do quarks fit the standard model?", limit=5)
        labels = {item["label"] for item in result.concepts}
        self.assertTrue({"quarks", "standard model", "particle physics"} <= labels)
        self.assertIn("Verified semantic memory", result.preamble())

    def test_failed_or_unauthenticated_request_never_persists(self):
        proposal = SemanticProposal("forest", "has_property", "leaf smell", 0.99, 0.8, "turn")
        for delivered, authenticated, reason in (
            (False, True, "response-not-delivered"),
            (True, False, "request-local-only"),
        ):
            receipt = self.memory.commit_after_delivery(
                self.context,
                [proposal],
                response_delivered=delivered,
                authenticated_tenant=authenticated,
            )
            self.assertEqual(receipt["reason"], reason)
        self.assertFalse(self.memory.retrieve(self.context, "forest").concepts)

    def test_stale_request_revision_fails_closed(self):
        first = SemanticProposal("forest", "related_to", "leaves", 0.99, 0.8, "one")
        self.memory.commit_after_delivery(
            self.context, [first], response_delivered=True, authenticated_tenant=True
        )
        second = SemanticProposal("forest", "related_to", "walking", 0.99, 0.8, "two")
        with self.assertRaisesRegex(ValueError, "revision changed"):
            self.memory.commit_after_delivery(
                self.context,
                [second],
                response_delivered=True,
                authenticated_tenant=True,
                expected_revision=0,
            )

    def test_repeated_identical_evidence_does_not_publish_another_revision(self):
        proposal = SemanticProposal("forest", "related_to", "leaves", 0.99, 0.8, "turn-1")
        first = self.memory.commit_after_delivery(
            self.context, [proposal], response_delivered=True, authenticated_tenant=True
        )
        prepared = []
        second = self.memory.commit_after_delivery(
            self.context, [proposal], response_delivered=True, authenticated_tenant=True,
            expected_revision=first["revision"],
            prepare_derived_handles=lambda *_args: prepared.append(True) or {},
        )
        self.assertEqual(second["reason"], "unchanged")
        self.assertEqual(second["revision"], first["revision"])
        self.assertEqual(prepared, [])
        graph, digest, revision = self.memory.load(self.context)
        self.assertEqual(revision, first["revision"])
        self.assertEqual(digest, first["capsule"])
        self.assertEqual(len(graph["edges"]), 1)

    def test_one_hop_retrieval_does_not_chain_by_edge_order(self):
        chain = (("alpha", "beta"), ("beta", "gamma"), ("gamma", "delta"))
        for index, links in enumerate((chain, tuple(reversed(chain)))):
            context = DirectoryContext(model="qwen9b", tenant="alice", session=f"chain-{index}")
            self.memory.commit_after_delivery(
                context,
                [SemanticProposal(a, "related_to", b, 0.99, 0.8, "e") for a, b in links],
                response_delivered=True,
                authenticated_tenant=True,
            )
            labels = {item["label"] for item in self.memory.retrieve(context, "alpha").concepts}
            self.assertEqual(labels, {"alpha", "beta"})

    def test_request_context_uses_session_revision_for_commit(self):
        context = DirectoryContext(model="qwen9b", tenant="alice", session="walk", request="turn-1")
        first = self.memory.commit_after_delivery(
            context, [SemanticProposal("forest", "related_to", "leaves", 0.99, 0.8, "one")],
            response_delivered=True, authenticated_tenant=True,
        )
        self.assertEqual(first["revision"], 1)
        self.assertEqual(self.memory.load(context)[2], 1)
        second = self.memory.commit_after_delivery(
            context, [SemanticProposal("forest", "related_to", "walking", 0.99, 0.8, "two")],
            response_delivered=True, authenticated_tenant=True, expected_revision=1,
        )
        self.assertEqual(second["revision"], 2)


class ClassifierBundleTests(unittest.TestCase):
    def test_counterbalanced_calibrated_choice_and_abstention(self):
        prompts = []

        def confident(prompt, token_ids):
            prompts.append(prompt)
            self.assertEqual(token_ids, {"supports": 10, "contradicts": 11, "unknown": 12})
            return {"supports": 4.0, "contradicts": 0.0, "unknown": -1.0}

        classifier = ForcedChoiceClassifier(
            labels=("supports", "contradicts", "unknown"),
            label_token_ids={"supports": 10, "contradicts": 11, "unknown": 12},
            score_tokens=confident,
            confidence_threshold=0.90,
            margin_threshold=0.20,
        )
        result = classifier.classify("Claim A; evidence B")
        self.assertEqual(result.label, "supports")
        self.assertTrue(result.passes_commit_gate)
        self.assertEqual(len(prompts), 2)
        self.assertNotEqual(prompts[0], prompts[1])

        classifier.score_tokens = lambda _prompt, _tokens: {
            "supports": 1.0,
            "contradicts": 0.9,
            "unknown": 0.8,
        }
        result = classifier.classify("ambiguous")
        self.assertTrue(result.abstained)
        self.assertIsNone(result.label)

    def test_bundle_selector_tightens_gates_with_proposal_count(self):
        def score(prompt, _token_ids):
            proposal = prompt.split("Proposed bundle", 1)[1]
            if "matching forest bundle" in proposal:
                return {"store": 5.0, "defer": 0.0, "reject": -1.0}
            return {"store": -1.0, "defer": 0.0, "reject": 4.0}

        selector = AdaptiveBundleSelector(
            label_token_ids={"store": 1, "defer": 2, "reject": 3},
            score_tokens=score,
        )
        two = selector.select(
            "Which forest bundle applies?",
            ("unrelated city bundle", "matching forest bundle"),
        )
        eight = selector.select(
            "Which forest bundle applies?",
            tuple(["unrelated city bundle"] * 7 + ["matching forest bundle"]),
        )
        self.assertEqual(two.selected_index, 1)
        self.assertEqual(eight.selected_index, 7)
        self.assertGreater(eight.confidence_threshold, two.confidence_threshold)
        self.assertGreater(eight.margin_threshold, two.margin_threshold)

        rejected = selector.select(
            "Which forest bundle applies?",
            ("unrelated city bundle", "unrelated ocean bundle"),
        )
        self.assertTrue(rejected.abstained)
        self.assertIsNone(rejected.selected_index)

    def test_bundle_selector_combines_directory_prior_with_neural_reranker(self):
        selector = AdaptiveBundleSelector(
            label_token_ids={"relevant": 1, "unrelated": 2},
            score_tokens=lambda _prompt, _tokens: {
                "relevant": -0.8,
                "unrelated": 0.0,
            },
        )
        result = selector.select(
            "What aroma belongs to the cedar route?",
            ("cedar subject bundle", "unrelated subject bundle"),
            priors=(1.0, 0.0),
        )
        self.assertEqual(result.selected_index, 0)
        self.assertFalse(result.abstained)
        self.assertLess(result.candidate_relevance[0], selector.base_confidence)
        self.assertGreater(result.candidate_combined[0], result.confidence_threshold)
        self.assertEqual(result.candidate_priors, (1.0, 0.0))

        with self.assertRaisesRegex(ValueError, "match the proposal count"):
            selector.select("query", ("one", "two"), priors=(1.0,))


if __name__ == "__main__":
    unittest.main()
