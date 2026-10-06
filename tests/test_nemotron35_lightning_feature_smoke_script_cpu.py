from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_nemotron35_lightning_feature_smoke.py"


def load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "run_nemotron35_lightning_feature_smoke", SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def valid_status(smoke):
    return {
        "artifact": smoke.EXPECTED_ARTIFACT,
        "profile": smoke.EXPECTED_PROFILE,
        "qualification": "candidate",
        "apcv2": {"layout_name": smoke.EXPECTED_LAYOUT},
        "implemented_capabilities": ["apc_v2", "mtp"],
        "capabilities": ["apc_v2", "tools"],
    }


def test_startup_binding_accepts_exact_ordinary_candidate():
    smoke = load_script()
    assert smoke.startup_binding_errors(valid_status(smoke)) == []


def test_startup_binding_fails_closed_for_each_identity_or_route_mismatch():
    smoke = load_script()
    baseline = valid_status(smoke)
    mutations = {
        "artifact": {**baseline, "artifact": "wrong"},
        "profile": {**baseline, "profile": "wrong"},
        "cache_layout": {**baseline, "apcv2": {"layout_name": "wrong"}},
        "qualification": {**baseline, "qualification": "qualified"},
        "mtp_implementation": {
            **baseline,
            "implemented_capabilities": ["apc_v2"],
        },
        "mtp_selected_on_ordinary_server": {
            **baseline,
            "capabilities": ["apc_v2", "mtp"],
        },
    }
    for expected, value in mutations.items():
        assert expected in smoke.startup_binding_errors(value)


def test_harness_keeps_native_mtp_out_of_ordinary_server_command():
    source = SCRIPT.read_text()
    server = source[source.index("server_command = ["):source.index("server = None")]
    assert '"--ordinary"' in server
    assert '"--native-mtp"' not in server
    assert '"--num-draft"' not in server
    assert "common.stop_server(server)" in source
    assert "common.validate_ownership(args.session, args.label)" in source


def test_common_feature_runner_can_emit_model_specific_schema_and_prompts():
    common_source = (ROOT / "scripts" / "run_north_feature_smoke.py").read_text()
    assert 'schema: str = "mlx2.north-feature-smoke.v1"' in common_source
    assert 'f"Reply with exactly {marker}"' in common_source
    assert 'f"Explain {subject} in numbered steps. nonce {nonce_prefix}-{index}"' in common_source
