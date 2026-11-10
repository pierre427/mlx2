"""DSpine drafter on the lane draft kernel: policy, identity and loader (CPU).

The lane matmul itself needs a Metal backend; tests/test_dspine_lane_metal.py
runs the real install.  Here the backend is faked only where inspection
derives the law name, never to execute a lane kernel.
"""
import json

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters import dspine as dspine_adapter
from mlx2.adapters.dspine import VERSION_SCALARS
from mlx2.runtime.drafters import dflash2 as dflash2_runtime

from test_dspine_drafter_cpu import hybrid_pair


def _artifacts(tmp_path):
    from mlx.utils import tree_flatten
    from test_qwen38_dflash2_cpu import VOCAB as HV

    raw, target_model, draft = hybrid_pair()
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 32, "vocab_size": HV, "num_hidden_layers": 8}}))
    (target / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}))
    (target / "a.safetensors").write_bytes(b"target-weight-payload")
    path = tmp_path / "draft"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(raw))
    weights = {k: v for k, v in tree_flatten(draft.parameters())
               if not k.startswith(("embed_tokens.", "lm_head."))}
    weights.update({name: mx.array(1, dtype=mx.int64) for name in VERSION_SCALARS})
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    return target, path, target_model


def test_lane_kernel_is_accepted_for_dspine_and_bound_into_its_identity(tmp_path, monkeypatch):
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from mlx2.runtime import lane

    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    target, path, _ = _artifacts(tmp_path)
    policy = {"draft_model": str(path), "num_draft": 5}
    plain = inspect_external_policy(dict(policy), target)
    stock = inspect_external_policy({**policy, "draft_kernel": "stock"}, target)
    assert stock["fingerprint"] == plain["fingerprint"] and stock["draft_kernel"] is None
    lane_record = inspect_external_policy({**policy, "draft_kernel": "lane"}, target)
    assert lane_record["draft_kernel"] == "lane"
    q8 = {"bits": 8, "group_size": 64}
    q8_lane = inspect_external_policy({**policy, "draft_kernel": "lane", "draft_quantization": q8}, target)
    q4_lane = inspect_external_policy(
        {**policy, "draft_kernel": "lane", "draft_quantization": {"bits": 4, "group_size": 64}}, target)
    assert len({plain["fingerprint"], lane_record["fingerprint"], q8_lane["fingerprint"],
                q4_lane["fingerprint"]}) == 4


def test_fused_kernel_is_refused_for_dspine(tmp_path, monkeypatch):
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from mlx2.runtime import lane

    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    target, path, _ = _artifacts(tmp_path)
    with pytest.raises(ValueError, match='"stock" or "lane"'):
        inspect_external_policy({"draft_model": str(path), "num_draft": 5, "draft_kernel": "fused"}, target)


def test_lane_kernel_fails_closed_without_a_backend(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, path, _ = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="lane matmul backend"):
        inspect_external_policy({"draft_model": str(path), "num_draft": 5, "draft_kernel": "lane"}, target)


def test_loader_installs_the_lane_on_the_backbone_only_before_bind(tmp_path, monkeypatch):
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from mlx2.adapters.external_drafter_formats import drafter_format_for_record
    from mlx2.runtime import lane

    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    target, path, target_model = _artifacts(tmp_path)
    record = inspect_external_policy(
        {"draft_model": str(path), "num_draft": 5, "draft_kernel": "lane"}, target)
    calls = []

    def fake(model, kernel, *, drafter="dflash2"):
        assert getattr(model, "lm_head", None) is None  # before bind
        calls.append((kernel, drafter))
        return {"kernel": kernel, "law_id": "fake"}

    monkeypatch.setattr(dflash2_runtime, "configure_draft_kernel", fake)
    loaded = drafter_format_for_record(record).load_drafter(
        record, target_model, runtime_quantization=None, draft_kernel=record["draft_kernel"])
    assert calls == [("lane", "dspine")]
    assert loaded.receipt_settings["draft_kernel"] == {"kernel": "lane", "law_id": "fake"}
    with pytest.raises(ValueError, match='"stock" or "lane"'):
        dspine_adapter.load_drafter(record, target_model, draft_kernel="fused")


def test_dspine_law_names_exactly_the_bridge_projections_the_installer_skips(monkeypatch):
    import hashlib

    import mlx.nn as nn
    from mlx2.runtime import lane
    from mlx2.runtime.drafters.dflash2_kernel import DSPINE_LANE_SKIPPED, draft_kernel_law

    _raw, _target, draft = hybrid_pair()
    bridge = sorted(name for name, module in draft.named_modules()
                    if isinstance(module, (nn.Linear, nn.QuantizedLinear))
                    and name.startswith("selected_prefix_bridge"))
    assert tuple(bridge) == DSPINE_LANE_SKIPPED
    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    digest = hashlib.sha256("\n".join(bridge).encode()).hexdigest()[:12]
    law = draft_kernel_law("lane", drafter="dspine")
    # Same order as the lane installer: rows window, skip set, grouping.
    assert law == f"{lane.LAW_IDS['mpp']}+rows-le-128+skip[{digest}]+grouped+fused-context-v1"
    assert "blocktopk" not in law
    assert draft_kernel_law("lane") != law  # DFlash2 law unchanged by DSpine
    with pytest.raises(ValueError, match='"stock" or "lane"'):
        draft_kernel_law("fused", drafter="dspine")


def test_dspine_takes_the_lane_kernel_only():
    for kernel in ("fused",):
        with pytest.raises(ValueError, match="lane draft kernel only"):
            dflash2_runtime.configure_draft_kernel(object(), kernel, drafter="dspine")
