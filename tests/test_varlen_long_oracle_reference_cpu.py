"""Import-free guard for the long NAX oracle's production reference."""
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/varlen_prefill_long_nax_device_oracle.py"


def test_per_span_stock_is_gate_and_padded_stock_is_diagnostic():
    source = SCRIPT.read_text()
    assert "lane=mx.fast.scaled_dot_product_attention(" in source
    assert "mask='causal',force_fused=True" in source
    assert "reference=mx.concatenate(lane_refs);mx.eval(reference)" in source
    assert "padded_reference=mx.concatenate(" in source
    assert "raw_bit_equal=bool(mx.all(output.view(mx.uint16)==reference.view(mx.uint16)).item())" in source
    assert "padded_fused_raw_equal=bool(mx.all(output.view(mx.uint16)==padded_reference.view(mx.uint16)).item())" in source
    assert "if output.dtype!=mx.bfloat16 or not finite or not close or not receipt['raw_bit_equal']" in source


def test_binary_and_source_are_explicitly_pinned_before_device_import():
    source = SCRIPT.read_text()
    assert "BINARY=args.native_binary;BINARY_SHA=args.native_sha256" in source
    assert "actual!=expected_source" in source
    assert "hashlib.sha256(BINARY.read_bytes()).hexdigest()!=BINARY_SHA" in source
    assert source.index("actual!=expected_source") < source.index("import mlx.core as mx")
