"""Engram reader families on /metrics (adapter diagnostics -> Prometheus), CPU.

Uses the real Hy4-flash ``EngramTableRows`` diagnostics from the tiny MLX
pack fixture (IQ4_NL GGUF tables) and a DeepSeek-V4.1-shaped reader
diagnostics dict; the exposition must parse with the official parser.
"""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.prometheus import PrometheusBuilder, _add_execution, render_engine_metrics  # noqa: E402
from test_hy_v4_flash_cpu import TOKENS  # noqa: E402
from test_hy_v4_flash_mlx_pack_cpu import pack  # noqa: E402,F401  (module fixture)
from test_prometheus import FakeEngine  # noqa: E402


@pytest.fixture(scope="module")
def hy4_execution(pack):  # noqa: F811
    from mlx2.adapters import hy_v4_flash as A

    root, _, _ = pack
    model, _, readers = A.load_model(root, A.inspect_artifact(root), engram_table_format="iq4_nl")
    try:
        model(mx.array([TOKENS]))
        model(mx.array([TOKENS[:1]]))
        return {"engram": {i: r.diagnostics() for i, r in readers.items()},
                "engram_table_format": {**A.engram_format_record("iq4_nl"), "selected": True},
                "approximate_state": True}
    finally:
        for r in readers.values():
            r.close()


def test_hy4_engram_readers_reach_metrics(hy4_execution):
    builder = PrometheusBuilder()
    _add_execution(builder, hy4_execution)
    text = builder.render()
    layer = sorted(hy4_execution["engram"])[0]
    diag = hy4_execution["engram"][layer]
    assert diag["unique_rows"] > 0
    assert f'mlx2_engram_rows_total{{kind="unique",layer="{layer}"}} {diag["unique_rows"]}' in text
    assert f'mlx2_engram_read_bytes_total{{layer="{layer}"}} {diag["bytes_read"]}' in text
    assert f'mlx2_engram_lookups_total{{layer="{layer}"}} {diag["lookups"]}' in text
    assert 'mlx2_engram_reads_total{layer="%s",phase="decode"}' % layer in text
    assert 'mlx2_engram_table_info{approximate="true",format="iq4_nl",layer="%s"} 1' % layer in text
    assert ('mlx2_engram_table_format_info{approximate="true",default="false",'
            'format="iq4_nl",qualified="false"} 1') in text
    assert "mlx2_route_approximate_state 1" in text


def test_engram_families_parse_in_a_full_scrape(hy4_execution):
    parser = pytest.importorskip("prometheus_client.parser")
    engine = FakeEngine()
    engine.snapshot["execution"] = {**engine.snapshot.get("execution", {}), **hy4_execution}
    families = {f.name: f for f in parser.text_string_to_metric_families(render_engine_metrics(engine))}
    for name in ("mlx2_engram_lookups", "mlx2_engram_rows", "mlx2_engram_read_bytes",
                 "mlx2_engram_reads", "mlx2_engram_table_info", "mlx2_engram_table_format_info",
                 "mlx2_route_approximate_state"):
        assert name in families and families[name].samples, name
    assert families["mlx2_engram_rows"].type == "counter"
    assert not [f.name for f in families.values() if f.type == "unknown"]


def test_deepseek_shaped_readers_and_hostile_labels_stay_bounded():
    builder = PrometheusBuilder()
    _add_execution(builder, {
        "engram": {
            "3": {"lookups": 4, "requested_rows": 64, "unique_rows": 40, "cache_hits": 10,
                  "cache_misses": 30, "bytes_read": 7920, "table_bytes": 1 << 30,
                  "storage": "pread-e4m3-e8m0-row264"},
            "x" * 9: {"lookups": 1},
            "7": {"lookups": True, "storage": 'evil"}\nmlx2_fake 1'},
            "9": "not a mapping",
        },
        "approximate_state": "yes",
    })
    text = builder.render()
    assert 'mlx2_engram_rows_total{kind="cache_hit",layer="3"} 10' in text
    assert 'mlx2_engram_table_info{approximate="false",format="pread-e4m3-e8m0-row264",layer="3"} 1' in text
    assert 'mlx2_engram_table_info{approximate="false",format="other",layer="7"} 1' in text
    assert 'mlx2_engram_lookups_total{layer="7"}' not in text
    assert "mlx2_fake" not in text and "xxxxxxxxx" not in text
    assert "mlx2_route_approximate_state" not in text
    assert "mlx2_engram_reads_total" not in text


def test_no_engram_diagnostics_adds_no_engram_families():
    builder = PrometheusBuilder()
    _add_execution(builder, {})
    assert "mlx2_engram" not in builder.render()
