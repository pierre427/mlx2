"""Flash-Next accepts both model-supported config layouts without GPU I/O."""

import json

import pytest

from tests.test_qwen36_35b_port import _flash_next_stub_load


@pytest.mark.parametrize("nested", [False, True])
def test_flash_next_context_limit_uses_normalized_text_config(tmp_path, monkeypatch, nested):
    flash_next = _flash_next_stub_load(tmp_path, monkeypatch)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["max_position_embeddings"] = 8192
    if not nested:
        config.update(config.pop("text_config"))
    config_path.write_text(json.dumps(config))
    adapter = flash_next.FlashNextAdapter(str(tmp_path))
    try:
        assert adapter.max_context == 8192
    finally:
        adapter.close()
