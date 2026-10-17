"""Evidence writing of scripts/qualify_upstream_intake_live.py (sweep 2026-10-16)."""

import importlib.util
import json
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qualify_upstream_intake_live.py"


def _module():
    spec = importlib.util.spec_from_file_location("_intake_live", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concurrent_writers_publish_one_whole_payload_and_leave_no_temporaries(tmp_path):
    write = _module().write_new_evidence
    output = tmp_path / "evidence.json"
    start = threading.Barrier(8)
    outcomes = []

    def writer(index):
        start.wait()
        try:
            write(output, {"writer": index, "body": "x" * 100_000})
            outcomes.append(("won", index))
        except FileExistsError:
            outcomes.append(("refused", index))

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    winners = [index for (kind, index) in outcomes if kind == "won"]
    assert len(winners) == 1 and len(outcomes) == 8
    assert json.loads(output.read_text())["writer"] == winners[0]
    assert [path.name for path in tmp_path.iterdir()] == ["evidence.json"]
    with pytest.raises(FileExistsError):
        write(output, {"writer": "late"})
    assert json.loads(output.read_text())["writer"] == winners[0]
