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


# --- withhold list and public push guard (2026-11-10: a snapshot exported with
# this script alone published media, Sushi probes and private-evidence tests,
# because their exclusions lived only in out-of-tree sync tooling) ---

def test_withhold_list_is_loaded_and_never_exported():
    assert exporter.WITHHELD is not None
    assert not exporter.is_public_code_path("scripts/public_export_withheld.txt")
    for name in ("src/mlx2/media/api.py", "scripts/media/qualify_music.py",
                 "src/mlx2/adapters/media/rife425.py", "scripts/sushi_probe/gpu.py",
                 "tests/test_ds41_streaming_ab_cpu.py"):
        assert exporter.is_private_path(name) and exporter.never_public(name), name
        assert not exporter.is_public_code_path(name), name
    # frozen public copies are not exported but may stay in the public tree
    assert exporter.withheld_kind("tests/test_media_qualification.py") == "frozen"
    assert exporter.is_private_path("tests/test_media_qualification.py")
    assert not exporter.never_public("tests/test_media_qualification.py")
    assert not exporter.is_private_path("src/mlx2/serving.py")


def test_withhold_list_rejects_malformed_lines():
    with pytest.raises(ValueError):
        exporter.parse_withheld("maybe src/mlx2/media/\n")
    with pytest.raises(ValueError):
        exporter.parse_withheld("never ../outside.py\n")
    parsed = exporter.parse_withheld("# c\nnever  a/\nfrozen b.py  # why\n")
    assert parsed["never"]["prefix"] == {"a/"} and parsed["frozen"]["exact"] == {"b.py"}


def test_export_fails_closed_without_the_withhold_list(private_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "WITHHELD", None)
    with pytest.raises(ValueError, match="withhold list"):
        exporter.export(private_repo, "HEAD", tmp_path / "out", include=["src/mlx2/example.py"])


def _public_clone(tmp_path):
    remote = tmp_path / "public.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    clone = tmp_path / "public"
    subprocess.run(["git", "clone", "-q", str(remote), str(clone)], check=True,
                   capture_output=True)
    ident = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]

    def commit(name, text="x = 1\n"):
        path = clone / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        subprocess.run(["git", "-C", str(clone), "add", "-f", name], check=True)
        subprocess.run(["git", "-C", str(clone), *ident, "commit", "-qm", name], check=True)
    return clone, commit


def test_pre_push_guard_refuses_never_public_paths_and_allows_code(tmp_path):
    clone, commit = _public_clone(tmp_path)
    exporter.install_push_guard(clone)
    hook = clone / ".git/hooks/pre-push"
    assert hook.is_file() and str(Path(exporter.__file__).resolve()) in hook.read_text()
    exclude = (clone / ".git/info/exclude").read_text()
    assert "/src/mlx2/media/" in exclude and "/scripts/sushi_probe/" in exclude
    commit("src/mlx2/serving.py")
    ok = subprocess.run(["git", "-C", str(clone), "push", "-q", "origin", "HEAD:main"],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    # info/exclude keeps an accidental export out of `git add -A`
    (clone / "src/mlx2/media").mkdir(parents=True)
    (clone / "src/mlx2/media/api.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
    assert "media" not in subprocess.run(["git", "-C", str(clone), "diff", "--cached", "--name-only"],
                                         capture_output=True, text=True, check=True).stdout
    # a forced add still cannot be pushed
    commit("src/mlx2/media/api.py")
    commit("src/mlx2/other.py")
    refused = subprocess.run(["git", "-C", str(clone), "push", "-q", "origin", "HEAD:main"],
                             capture_output=True, text=True)
    assert refused.returncode != 0
    assert "src/mlx2/media/api.py" in refused.stderr


def test_pre_push_guard_refuses_history_that_adds_then_removes(tmp_path):
    clone, commit = _public_clone(tmp_path)
    commit("README.md")
    subprocess.run(["git", "-C", str(clone), "push", "-q", "origin", "HEAD:main"], check=True)
    base = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    commit("scripts/sushi_probe/gpu.py")
    subprocess.run(["git", "-C", str(clone), "rm", "-q", "scripts/sushi_probe/gpu.py"], check=True)
    subprocess.run(["git", "-C", str(clone), "-c", "user.name=T", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "remove"], check=True)
    tip = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert exporter.check_tree(clone, tip) == []
    problems = exporter.check_push(clone, [("refs/heads/main", tip, "refs/heads/main", base)])
    assert any("adds scripts/sushi_probe/gpu.py" in p for p in problems)


def test_private_guard_blocks_pushes_to_the_public_repository(tmp_path):
    repo = tmp_path / "private"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "github",
                    "https://github.com/pierre427/mlx2.git"], check=True)
    exporter.install_private_guard(repo)
    exporter.install_private_guard(repo)  # idempotent
    hook = (repo / ".git/hooks/pre-push").read_text()
    assert hook.count("never push to the public repository") == 1
    assert "DISABLED" in exporter.run("git", "-C", str(repo), "remote", "get-url", "--push", "github")
    refuse = subprocess.run(["sh", str(repo / ".git/hooks/pre-push"), "x",
                             "https://github.com/pierre427/mlx2.git"], capture_output=True, text=True)
    assert refuse.returncode == 1 and "private mlx2 checkout" in refuse.stderr
    allow = subprocess.run(["sh", str(repo / ".git/hooks/pre-push"), "origin",
                            "ssh://git@forge.example/owner/mlx2.git"], capture_output=True, text=True)
    assert allow.returncode == 0
