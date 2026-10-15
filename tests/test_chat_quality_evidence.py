from scripts.chat_quality import quality_evidence_manifest


def test_prompt_side_rows_cannot_veto_generation_quality():
    manifest = quality_evidence_manifest(
        {
            "hadamard": {
                "tool_calls": {"top1": 0.8242},
                "tool_calls@assistant": {"top1": 0.99},
                "own_tool_chat": {"top1": 1.0},
                "own_summary": {"top1": 1.0},
            }
        }
    )
    assert manifest["diagnostic_only_cells"] == ["tool_calls"]
    assert manifest["promotion_gate_cells"] == [
        "own_summary",
        "own_tool_chat",
        "tool_calls@assistant",
    ]
