"""CPU-only tests for derived varlen performance geometries."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))

import prepare_varlen_skew_inputs as P


class CharacterTokenizer:
    def apply_chat_template(self, messages, **_keywords):
        size = 100 + sum(len(message["content"]) for message in messages)
        return list(range(size))


class Tests(unittest.TestCase):
    def test_resize_is_text_bound_and_hits_requested_geometry(self):
        source = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "a" * 2000},
            {"role": "user", "content": "<authoritative_audit_note>A</authoritative_audit_note>"},
            {"role": "assistant", "content": "b" * 2000},
            {"role": "user", "content": "final"},
        ]
        messages, ids = P.resize_messages(CharacterTokenizer(), source, 1536)
        self.assertEqual(len(ids), 1536)
        self.assertEqual(messages[0], source[0])
        self.assertEqual(messages[-1], source[-1])
        self.assertIn("Authoritative source notes", messages[1]["content"])
        self.assertIn("<authoritative_audit_note>", messages[1]["content"])

    def test_derivation_rebinds_rows_and_manifest(self):
        rows = []
        for index in range(3):
            body = {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "x" * 8000},
                    {"role": "user", "content": "<authoritative_audit_note>A</authoritative_audit_note>"},
                    {"role": "assistant", "content": "y" * 8000},
                    {"role": "user", "content": "final"},
                ],
                "max_tokens": 192, "temperature": 0,
                "enable_thinking": False,
            }
            rows.append({
                "case_id": f"d.{index}", "domain": "d", "body": body,
                "body_sha256": "old", "prompt_token_ids": [1],
                "prompt_tokens": 1, "preparation_receipt": {},
            })
        inputs = {"domain_order": ["d"], "rows": rows,
                  "inputs_sha256": "old", "qualified": True,
                  "price_usable": True}
        value = P.derive(
            inputs, CharacterTokenizer(), (1536, 4096, 6982))
        self.assertEqual(
            [row["prompt_tokens"] for row in value["rows"]],
            [1536, 4096, 6982])
        self.assertNotEqual(value["inputs_sha256"], "old")
        self.assertFalse(value["qualified"])
        self.assertFalse(value["price_usable"])
        self.assertTrue(
            value["varlen_research_derivation"]["stock_quantized_math_required"])
        self.assertFalse(
            value["varlen_research_derivation"]["speculative_decoding_required"])


if __name__ == "__main__":
    unittest.main()
