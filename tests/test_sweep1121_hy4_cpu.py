"""Bug sweep 2026-11-21, lane fix-hy4 (HY4A-1/2/3, HY4M-1/2), CPU.

Tiny random checkpoints from the Hy4-flash test fixtures; no server, no GPU.
"""

import importlib.util
import json
import socket
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_hy4_assemble_cpu import VISION, _vision_tensors  # noqa: E402
from test_hy_v4_flash_cpu import _tiny_artifact, _write_artifact, checkpoint, tiny  # noqa: E402, F401
from test_hy_v4_flash_mlx_pack_cpu import pack_config, write_pack  # noqa: E402
from test_hy_v4_flash_vision_cpu import _vision_artifact, vtiny  # noqa: E402, F401

ROOT = Path(__file__).resolve().parents[1]
PROCESSOR = {"image_processor": {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                                 "min_pixels": 64, "max_pixels": 4096}}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# HY4A-1 ---------------------------------------------------------------------

@pytest.mark.parametrize("policy", [None, {"draft_model": "embedded_dfly", "num_draft": 3},
                                    {"draft_model": "prompt_lookup"}])
def test_checkpoint_label_names_the_artifact_on_every_route(tmp_path, tiny, policy):  # noqa: F811
    from mlx2.adapters import hy_v4_flash as A

    config, W, _, _ = tiny
    path = _tiny_artifact(tmp_path, config, W)
    fingerprint = A.inspect_artifact(path)["identity"]["fingerprint"]
    adapter = A.HYV4FlashAdapter(str(path), execution_policy=policy)
    try:
        assert adapter.diagnostics()["checkpoint"] == f"{path.name}@{fingerprint}"
        if policy and policy["draft_model"] == "embedded_dfly":
            # The bind did replace identity["fingerprint"] (else this test is vacuous).
            assert adapter.identity["fingerprint"] != fingerprint
            _, receipt = adapter.generate_dfly(list(range(3, 12)), max_tokens=4)
            assert receipt["checkpoint"] == f"{path.name}@{fingerprint}"
    finally:
        adapter.close()


@pytest.mark.parametrize("name", ["hy4_context_ladder", "smoke_hy4_vision_http"])
def test_http_scripts_take_the_checkpoint_from_execution_not_the_route_digest(name):
    label = _script(name).checkpoint_label
    status = {"artifact": "sha256:routedigest", "execution": {"checkpoint": "pack@sha256:artifactprint"}}
    assert label("pack", status) == "pack@sha256:artifactprint"
    assert label("pack", {"artifact": "sha256:plain"}) == "pack@sha256:plain"
    assert label("pack", {}) is None


# HY4A-2 ---------------------------------------------------------------------

def test_registry_descriptor_declares_vision_only_for_a_vision_artifact(tmp_path, tiny, vtiny):  # noqa: F811
    from mlx2.adapters.registry import inspect_model
    from mlx2.contracts import Capability

    config, W, VW, _ = vtiny
    (tmp_path / "v").mkdir()
    (tmp_path / "t").mkdir()
    vision = inspect_model(_vision_artifact(tmp_path / "v", config, W, VW))
    assert vision.artifact["has_vision"] and Capability.VISION in vision.descriptor.capabilities
    config, W, _, _ = tiny
    text = inspect_model(_tiny_artifact(tmp_path / "t", config, W))
    assert Capability.VISION not in text.descriptor.capabilities


# HY4A-3 ---------------------------------------------------------------------

class _Server:
    pid = 0
    returncode = None

    def poll(self):
        return None


def _run_ladder(tmp_path, monkeypatch, *, answer, cached, warm_ids=(5, 6), extra=()):
    import transformers

    ladder = _script("hy4_context_ladder")
    code = ladder.make_code(1024)
    calls = []

    def stream_one(base, body, *, timeout):
        arm = "warm" if calls else "cold"
        calls.append(arm)
        return {"text": answer.format(code=code), "token_ids": list(warm_ids) if arm == "warm" else [5, 6],
                "ttft_s": 0.5, "elapsed_s": 1.0, "decode_tok_s": 9.0, "finish_reason": "stop",
                "receipt": {"prompt_tokens": 1000, "cached_tokens": cached if arm == "warm" else 0}}

    status = {"status": "ok", "artifact": "sha256:routedigest",
              "execution": {"checkpoint": "m@sha256:artifactprint"}}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(ladder, "build_prompt", lambda tok, cell, depth, code, offset, long_answer=False: ("prompt text", cell))
    monkeypatch.setattr(ladder.subprocess, "Popen", lambda *a, **k: _Server())
    monkeypatch.setattr(ladder, "get_json", lambda base, path, timeout=10: status)
    monkeypatch.setattr(ladder, "post_json", lambda *a, **k: {"cleared": True})
    monkeypatch.setattr(ladder, "stream_one", stream_one)
    monkeypatch.setattr(ladder, "stop_group", lambda process: "stopped")
    monkeypatch.setattr(ladder, "swap_used_mb", lambda: 10.0)
    monkeypatch.setattr(ladder, "git", lambda *a: "0" * 40)
    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    model = tmp_path / "m"
    model.mkdir()
    out = tmp_path / "out"
    rc = ladder.main(["--model", str(model), "--out-dir", str(out), "--execution-policy", str(policy),
                      "--cells", "1024", "--port", str(port), *extra])
    return rc, json.loads((out / "ladder.json").read_text())


@pytest.mark.parametrize("arg", [
    "--port", "--port=8282", "--por=8282", "--allow-production-p", "--host", "--mo", "--max-lan=4",
    "--max-infl=4", "--max-con=65536", "--admin-token-f", "--execution-p", "--cache-dir", "--no-cache-d",
    "--ast", "--ordi", "--external-d", "--prompt-look", "--native-m", "--",
])
def test_ladder_refuses_owned_server_options_before_setup(tmp_path, arg, capsys):
    ladder = _script("hy4_context_ladder")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        ladder.main(["--model", "unused", "--execution-policy", "unused", "--out-dir", str(out),
                     f"--server-arg={arg}"])
    assert exc.value.code == 2 and "ladder-owned" in capsys.readouterr().err
    assert not out.exists()


def test_ladder_preserves_server_extensions(tmp_path, monkeypatch):
    rc, report = _run_ladder(tmp_path, monkeypatch, answer="{code}", cached=999,
                             extra=["--server-arg=--prefill-step", "--server-arg=1024",
                                    "--server-arg=--fabric=off"])
    assert rc == 0
    assert report["server_command"][-3:] == ["--prefill-step", "1024", "--fabric=off"]


def test_ladder_passes_when_oracles_pass_and_the_warm_arm_hits(tmp_path, monkeypatch):
    rc, report = _run_ladder(tmp_path, monkeypatch, answer="The code is {code}.", cached=999)
    assert rc == 0 and report["status"] == "completed"
    assert report["verdict"] == {"oracle_failures": [], "warm_misses": [], "cold_warm_mismatches": [],
                                 "warm_restore_expected": True}
    assert report["checkpoint"] == "m@sha256:artifactprint"


def test_ladder_fails_on_an_oracle_failure(tmp_path, monkeypatch):
    rc, report = _run_ladder(tmp_path, monkeypatch, answer="I could not find the vault code.", cached=999)
    assert rc == 1 and report["status"].startswith("failed")
    assert report["verdict"]["oracle_failures"] == ["1024:cold", "1024:warm"]


@pytest.mark.parametrize("cached", [0, None])
def test_ladder_fails_when_the_warm_arm_restored_nothing(tmp_path, monkeypatch, cached):
    rc, report = _run_ladder(tmp_path, monkeypatch, answer="{code}", cached=cached)
    assert rc == 1 and report["status"].startswith("failed")
    assert report["verdict"]["warm_misses"] == [1024]


def test_ladder_fails_on_a_cold_warm_token_mismatch(tmp_path, monkeypatch):
    # d52f055d9 (2026-11-25): width-one cold/warm replay of one route requires exact token parity.
    rc, report = _run_ladder(tmp_path, monkeypatch, answer="{code}", cached=999, warm_ids=(5, 7))
    assert rc == 1 and report["status"] == "failed: 1 cold warm mismatches"
    assert report["verdict"]["cold_warm_mismatches"] == [1024]


# HY4M-1 ---------------------------------------------------------------------

def test_converter_carries_the_image_processor_config(tmp_path, vtiny):  # noqa: F811
    from mlx2.adapters import hy_v4_flash as A

    config, W, VW, _ = vtiny
    (tmp_path / "src").mkdir()
    src = _vision_artifact(tmp_path / "src", config, W, VW)
    (src / "processor_config.json").write_text(json.dumps(PROCESSOR))
    out = tmp_path / "out"
    _script("convert_hy_v4_flash_mlx").convert(src, out, group_size=32, expert_bits=8, trunk_bits=8,
                                               resume=False)
    assert (out / "processor_config.json").read_bytes() == (src / "processor_config.json").read_bytes()
    adapter = A.HYV4FlashAdapter(str(out))
    try:
        assert adapter.image_pixel_bounds == (64, 4096)
        assert adapter.image_processor_settings["image_mean"] == (0.5, 0.5, 0.5)
        assert adapter.diagnostics()["vision"]["image_processor"] == {
            "source": "processor_config.json", "fallback": False}
    finally:
        adapter.close()


def test_vision_artifact_without_a_processor_config_is_a_labelled_fallback(tmp_path, tiny, vtiny):  # noqa: F811
    from mlx2.adapters import hy_v4_flash as A

    config, W, VW, _ = vtiny
    (tmp_path / "v").mkdir()
    (tmp_path / "t").mkdir()
    adapter = A.HYV4FlashAdapter(str(_vision_artifact(tmp_path / "v", config, W, VW)))
    try:  # still loads, on the class defaults, and says so
        assert adapter.diagnostics()["vision"]["image_processor"] == {
            "source": A.IMAGE_PROCESSOR_FALLBACK, "fallback": True}
    finally:
        adapter.close()
    config, W, _, _ = tiny
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path / "t", config, W)))
    try:  # no tower: nothing to label
        assert adapter.diagnostics()["vision"]["image_processor"] is None
    finally:
        adapter.close()


# HY4M-2 ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    config = pack_config()
    config.update(vision_config=VISION, image_token_id=90)
    W = checkpoint(config, seed=7, scale=0.2)
    original = _write_artifact(tmp_path_factory.mktemp("original"), config, {**W, **_vision_tensors()})
    (original / "processor_config.json").write_text(json.dumps(PROCESSOR))
    pack_dir, _, _ = write_pack(tmp_path_factory.mktemp("pack"), config, W)
    return original, pack_dir


def test_assembler_takes_a_missing_processor_config_from_the_original(parts, tmp_path):
    original, pack_dir = parts
    assert not (pack_dir / "processor_config.json").exists()
    out = tmp_path / "out"
    assert _script("assemble_hy4_mlx_pack").main([str(pack_dir), str(original), str(out)]) == 0
    assert (out / "processor_config.json").read_bytes() == (original / "processor_config.json").read_bytes()
    assembly = json.loads((out / "config.json").read_text())["mlx2_assembly"]
    assert assembly["processor_config_from_original"] == ["processor_config.json"]
    assert "processor_config.json" not in assembly["hardlinked_from_pack"]


def test_assembler_refuses_processor_configs_that_differ_and_links_equal_ones(parts, tmp_path):
    original, pack_dir = parts
    target = pack_dir / "processor_config.json"
    try:
        target.write_text(json.dumps({"image_processor": {"min_pixels": 256}}))
        with pytest.raises(SystemExit, match="disagree on processor_config.json"):
            _script("assemble_hy4_mlx_pack").main([str(pack_dir), str(original), str(tmp_path / "bad")])
        assert not (tmp_path / "bad").exists()
        target.write_bytes((original / "processor_config.json").read_bytes())
        out = tmp_path / "same"
        assert _script("assemble_hy4_mlx_pack").main([str(pack_dir), str(original), str(out)]) == 0
        assembly = json.loads((out / "config.json").read_text())["mlx2_assembly"]
        assert assembly["processor_config_from_original"] == []
        assert "processor_config.json" in assembly["hardlinked_from_pack"]
    finally:
        target.unlink(missing_ok=True)
