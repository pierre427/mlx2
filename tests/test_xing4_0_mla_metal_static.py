"""CPU-only contract checks; the Metal path still needs a leased GPU gate."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/mlx2/runtime/models/xing4_0.py"
KERNEL = ROOT / "src/mlx2/runtime/models/xing4_0_mla_metal.py"
PROVENANCE = ROOT / "provenance/xing4-0-mla-metal.json"


def test_fused_mla_is_default_off_and_reference_path_remains():
    model = MODEL.read_text()
    ast.parse(model)
    assert "_FUSED_MLA = False" in model
    assert "MLX2_XING_FUSED_MLA" not in model
    assert "if _FUSED_MLA:" in model
    assert "use_absorbed_path(L, k_pe.shape[-2], self.absorbed_geometry)" in model
    assert "fused_mla_stats" in model
    # The refusal path runs before cache.update_and_fetch and therefore
    # cannot advance cache.offset for a rejected opt-in request.
    preflight = model.index("fused Xing MLA geometry/device/mask gate rejected before cache append")
    append = model.index("kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)")
    assert preflight < append
    assert "cache.trim(cache.offset - previous_offset)" in model


def test_kernel_contract_and_provenance_are_static_and_bounded():
    source = KERNEL.read_text()
    ast.parse(source)
    assert "_MAX_QUERY = 4" in source
    assert "_MAX_CONTEXT = 131072" in source
    assert "_MAX_SCRATCH_BYTES = 64 * 1024 * 1024" in source
    assert "STATS = {\"fused_calls\": 0, \"rejected_calls\": 0}" in source
    assert "\"fused Xing MLA requires GPU BF16/FP16" in source
    assert "part_m" in source and "part_l" in source and "part_o" in source
    provenance = json.loads(PROVENANCE.read_text())
    assert provenance["destination"] == "src/mlx2/runtime/models/xing4_0_mla_metal.py"
    assert len(provenance["source_revision"]) == 40
    assert provenance["license"] == "Apache-2.0"
