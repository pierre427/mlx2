import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/research/dependency_contract_inventory.py"
spec = importlib.util.spec_from_file_location("dependency_contract_inventory", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "build", ["0.32.2.dev20260919+39400a0d4", "unknown-future-build"]
)
def test_inventory_never_turns_a_source_check_into_compatibility(build):
    catalog = json.loads(PATH.with_name("mlx_mechanism_contracts.json").read_text())
    report = module.inventory(ROOT, catalog, build)
    assert report["historical_build_list_membership"] == (
        build != "unknown-future-build"
    )
    assert not report["framework_imported"]
    assert not report["package_constraints_changed"]
    assert len(report["mechanisms"]) == 8
    for mechanism in report["mechanisms"].values():
        assert not mechanism["compatibility_established_by_this_report"]
        assert not mechanism["hardware_canaries_run"]
        assert mechanism["source_sha256"]
        assert all(
            not t["executed_by_inventory"] for t in mechanism["test_inventory"].values()
        )


def test_missing_or_escaping_paths_fail_closed(tmp_path):
    for path in ["", "../outside.py", "/etc/passwd"]:
        with pytest.raises(ValueError):
            module.checked_path(tmp_path, path)


def test_changed_build_declaration_is_not_executed(tmp_path):
    path = tmp_path / "src/mlx2/runtime/mlx_build.py"
    path.parent.mkdir(parents=True)
    path.write_text("VERIFIED_MLX_BUILDS = dangerous_loader()")
    with pytest.raises(ValueError, match="unrecognized"):
        module.known_builds(tmp_path)
