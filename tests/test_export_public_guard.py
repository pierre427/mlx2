"""Publication boundary tests: only temporary Git repositories, no network."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "export_public_guard", Path(__file__).parents[1] / "scripts/export_public.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def private_repo(tmp_path, monkeypatch):
    repo = tmp_path / "private"
    repo.mkdir()
    for name, content in {
        "src/mlx2/example.py": "VALUE = 1\n",
        "tests/test_example.py": "assert True\n",
        "qualification/private-run.json": '{"admin_token": "' + "a" * 64 + '"}',
        "weights.npz": "private weights",
        "src/mlx2/private.npz": "private asset",
        "README.md": "Public summary\n",
    }.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (repo / "src/mlx2/binary.py").write_bytes(b"private\0payload")
    (repo / "src/mlx2/link.py").symlink_to("../../qualification/private-run.json")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    exporter.run("git", "add", ".", cwd=repo)
    exporter.run("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "fixture", cwd=repo)
    monkeypatch.setattr(exporter, "environment_tokens", lambda _repo: ([], []))
    return repo


def test_no_default_whole_tree_export(private_repo, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="allowlist"):
        exporter.export(private_repo, "HEAD", out)
    assert not out.exists()


def test_exact_allowlist_excludes_tracked_private_payloads(private_repo, tmp_path):
    out = tmp_path / "out"
    exporter.export(private_repo, "HEAD", out, include=["src/mlx2/example.py", "README.md"])
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()) == [
        "README.md", "src/mlx2/example.py"
    ]
    assert exporter.scan(private_repo, out) == []


@pytest.mark.parametrize("name", ["qualification/private-run.json", "weights.npz",
    "src/mlx2/private.npz", "src", "src/mlx2/*.py", "../private.py", "/tmp/private.py",
    "src/mlx2/link.py", "src/mlx2/missing.py"])
def test_refuse_data_directories_globs_and_links(private_repo, tmp_path, name):
    with pytest.raises(ValueError):
        exporter.export(private_repo, "HEAD", tmp_path / "out", include=[name])


def test_export_reads_selected_commit_not_dirty_worktree(private_repo, tmp_path):
    (private_repo / "src/mlx2/example.py").write_text("PRIVATE_DIRTY = True\n")
    out = tmp_path / "out"
    exporter.export(private_repo, "HEAD", out, include=["src/mlx2/example.py"])
    assert (out / "src/mlx2/example.py").read_text() == "VALUE = 1\n"


def test_push_fails_before_any_external_command(monkeypatch, tmp_path):
    monkeypatch.setattr(exporter.subprocess, "run", lambda *a, **k: pytest.fail("unexpected command"))
    with pytest.raises(RuntimeError, match="disabled"):
        exporter.push(tmp_path, tmp_path, "unreachable", "sha", "main")


def test_output_must_be_empty(private_repo, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "private.json").write_text("private")
    with pytest.raises(ValueError, match="empty"):
        exporter.export(private_repo, "HEAD", out, include=["src/mlx2/example.py"])


def test_reject_binary_payload_disguised_as_code(private_repo, tmp_path):
    with pytest.raises(ValueError, match="binary payload"):
        exporter.export(private_repo, "HEAD", tmp_path / "out", include=["src/mlx2/binary.py"])


def test_mcp_code_is_never_a_publication_input():
    # Pierre, 2026-10-08: MCP servers stay private.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "export_public.py"
    spec = importlib.util.spec_from_file_location("export_public_mcp", path)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    for private in ("src/mlx2/example_mcp/server.py", "src/mlx2/example_mcp/__init__.py",
                    "tests/test_example_mcp_cpu.py", "scripts/run_mcp_server.py",
                    "src/mlx2/MCP/tool.py"):
        assert not ex.is_public_code_path(private), private
    for public in ("src/mlx2/server.py", "src/mlx2/compact.py", "pyproject.toml"):
        assert ex.is_public_code_path(public), public
    pyproject = (
        '[project.optional-dependencies]\n'
        'radio = ["mlx>=0.32"]\n'
        'example-mcp = ["fastmcp>=2.2,<3"]\n'
        '[project.scripts]\n'
        'mlx2-serve = "mlx2.server:main"\n'
        'mlx2-example-mcp = "mlx2.example_mcp.server:main"\n'
    )
    public_text = ex.public_text("pyproject.toml", pyproject)
    assert "mcp" not in public_text.lower()
    assert 'radio = ["mlx>=0.32"]' in public_text and "mlx2-serve" in public_text
    assert ex.public_text("src/mlx2/server.py", "x = 'mcp'\n") == "x = 'mcp'\n"


def test_only_reviewed_qualification_runfiles_and_policy_are_exportable():
    allowed = {
        "docs/QUALIFICATION.md",
        "qualification/runs/qualify-1010-correctness/dloop_qualification.py",
        "qualification/runs/qualify-1010-correctness/ladder.py",
        "qualification/runs/qualify-1010-correctness/performance_assessment.py",
        "qualification/runs/qualify-1010-correctness/qualification_verdict.py",
        "qualification/runs/qualify-1010-correctness/run_profile.py",
        "qualification/runs/qualify-1010-correctness/thermal_ladder.py",
        "qualification/runs/qualify-1010-correctness/thermal-policy.json",
        "qualification/runs/qualify-1010-correctness/test_dloop_qualification.py",
        "qualification/runs/qualify-1010-correctness/test_qualification_verdict.py",
        "qualification/runs/qualify-1010-correctness/test_run_profile.py",
        "qualification/runs/qualify-1010-correctness/test_thermal_ladder.py",
    }
    assert all(exporter.is_public_code_path(path) for path in allowed)
    for private in (
        "docs/audits/2026-10-10-rc2/qualification-plan.md",
        "qualification/runs/qualify-1010-correctness/README.md",
        "qualification/runs/qualify-1010-correctness/host-manifest.json",
        "qualification/runs/qualify-1010-correctness/thermal-policy-private.json",
        "qualification/runs/qualify-1010-correctness/events.log",
        "qualification/receipts/private.json",
        "qualification/runs/qualify-1010-correctness/../private.py",
    ):
        assert not exporter.is_public_code_path(private), private
