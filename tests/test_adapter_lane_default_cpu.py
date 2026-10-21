"""Adapter-declared lane defaults replace only the server's generic default.

model-matrix-20261021: a bare `mlx2.server --model <Nemotron 3.5 Bolt/Super
VL>` refused to load because the adapter serves B1 (B2/B4 only under exact
progressive verification) and the server default is 16 lanes.
"""
from types import SimpleNamespace

import pytest

from mlx2.server import DEFAULT_MAX_LANES, apply_adapter_lane_default, build_parser


def _resolution(adapter_type):
    return SimpleNamespace(adapter_type=adapter_type)


def test_nemotron35_super_and_bolt_default_to_one_lane():
    from mlx2.adapters.nemotron35_bolt import Nemotron35BoltAdapter
    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter

    for adapter in (Nemotron35SuperAdapter, Nemotron35BoltAdapter):
        args = build_parser().parse_args(["--model", "fixture"])
        assert args.max_lanes == DEFAULT_MAX_LANES
        assert apply_adapter_lane_default(args, _resolution(adapter)) == 1
        assert args.max_lanes == 1


def test_explicit_max_lanes_is_never_replaced():
    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter

    args = build_parser().parse_args(["--model", "fixture", "--max-lanes", "4"])
    assert apply_adapter_lane_default(args, _resolution(Nemotron35SuperAdapter)) == 4


def test_adapters_without_a_declaration_keep_the_server_default():
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

    args = build_parser().parse_args(["--model", "fixture"])
    assert apply_adapter_lane_default(args, _resolution(MuseGlimmerAdapter)) == DEFAULT_MAX_LANES


def test_invalid_declaration_fails_closed():
    class Bad:
        default_max_lanes = 0

    with pytest.raises(ValueError):
        apply_adapter_lane_default(build_parser().parse_args(["--model", "x"]), _resolution(Bad))
