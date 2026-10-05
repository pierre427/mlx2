"""Compile and execute the native packed-row geometry guard without MLX/GPU."""

from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_native_multirow_guard_rejects_aliases_and_bounds(tmp_path):
    source = tmp_path / "guard.cpp"
    source.write_text(r'''
#include "packed_multirow_write.h"
#include <cassert>
#include <functional>
#include <stdexcept>
using namespace mlx2::paged_kv;
static bool rejects(const std::function<void()>& f) {
  try { f(); } catch (const std::invalid_argument&) { return true; }
  return false;
}
int main() {
  assert(!packed_multirow_write_selected(nullptr));
  assert(packed_multirow_write_selected("1"));
  assert(rejects([] { packed_multirow_write_selected("true"); }));
  auto touched = validate_packed_multirow_write({63,129}, {0,0}, {0,0},
      {0,1}, {0,1,2,3}, 8, 192);
  assert((touched == std::vector<uint32_t>{0,1,2,3}));
  assert(rejects([] { validate_packed_multirow_write({63,129}, {0,0},
      {0,0}, {0,1}, {0,1,1,3}, 8, 192); }));
  assert(rejects([] { validate_packed_multirow_write({63,129}, {0,0},
      {0,0}, {0,1}, {0,1,2,1}, 8, 192); }));
  assert(rejects([] { validate_packed_multirow_write({63,129}, {0,0},
      {0,0}, {0,1}, {0,1,2,8}, 8, 192); }));
  assert(rejects([] { validate_packed_multirow_write({63,129}, {0,0},
      {0,0}, {0,1}, {0,1,2,3}, 8, 191); }));
  assert(rejects([] { validate_packed_multirow_write({63,129}, {0,0},
      {0,0}, {0,1}, {0,1,2,3}, 8, 193); }));
  assert(rejects([] { validate_packed_multirow_write({1,1}, {8191,8192},
      {0,0}, {0,1}, {0,1}, 8, 2); }));
}
''')
    binary = tmp_path / "guard"
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "native/paged_kv"), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_native_page_table_binding_uses_vector_payload_and_failure_receipt(tmp_path):
    source = (ROOT / "native/paged_kv/arena.cpp").read_text()
    assert "encoder.set_bytes(page_ids_.data(), static_cast<int>(page_ids_.size()), 9);" in source
    assert "encoder.set_bytes(page_ids_, 9);" not in source
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps({"source_commit": "invalid", "source_tree_sha256": "0",
                                    "native_path": __file__, "native_sha256": "0",
                                    "mlx_version": "0"}))
    receipt = tmp_path / "failure.json"
    result = subprocess.run([sys.executable,
                             str(ROOT / "scripts/research/varlen_packed_multirow_write_oracle.py"),
                             "--manifest", str(manifest), "--receipt", str(receipt),
                             "--preflight-only"], capture_output=True, text=True)
    assert result.returncode != 0
    recorded = json.loads(receipt.read_text())
    assert recorded["status"] == "failed" and recorded["gpu_executed"] is False
    assert recorded["error_type"] == "RuntimeError"
    assert recorded["unresolved_failure_roots"] == 0


def test_case_failure_retires_synchronized_read_lease_and_stage(monkeypatch):
    scripts = str(ROOT / "scripts/research")
    monkeypatch.syspath_prepend(scripts)
    import varlen_packed_multirow_write_oracle as oracle
    from mlx2.runtime import paged_kv_write, qwen3_paged_native_backend

    class FakeArena:
        closed = False

        def __init__(self, plane_bytes, _stream, **_kw):
            self.plane_bytes = plane_bytes

        def poll_completions(self):
            return []

        def close_after_terminal(self):
            self.closed = True

    monkeypatch.setattr(paged_kv_write, "NativeWriteBackend", FakeArena)
    monkeypatch.setattr(qwen3_paged_native_backend, "NativeWriteBackend", FakeArena)

    def fail_after_read_lease(_mx, _np, _counts, _dtype, _pages, _pool,
                              _stream, _arena, writer, _backend, owners, state):
        touched, *_ = owners[0].stage_packed_multirow(object(), object(), 1)
        lease = writer.ledger.prepare(touched)
        writer.ledger.submit(lease)
        state["read_lease"] = lease
        raise AssertionError("injected byte mismatch")

    monkeypatch.setattr(oracle, "_case_body", fail_after_read_lease)
    oracle.CASE_CLEANUP.clear()
    oracle.FAILURE_ROOTS.clear()
    fake_mx = SimpleNamespace(gpu=object(), default_stream=lambda _device: object(),
                              synchronize=lambda _stream: None)
    try:
        oracle._case(fake_mx, None, (32, 96), "float16")
    except AssertionError as exc:
        assert str(exc) == "injected byte mismatch"
    else:
        raise AssertionError("injected failure did not propagate")
    cleanup, = oracle.CASE_CLEANUP
    assert cleanup["synchronized"] and cleanup["read_lease_terminal"]
    assert cleanup["owners_closed"] and cleanup["arena_closed"]
    assert cleanup["pending_leases"] == cleanup["pending_epochs"] == 0
    assert cleanup["retained_pages"] == 0 and not oracle.FAILURE_ROOTS
