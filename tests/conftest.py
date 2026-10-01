"""Routine tests are CPU-only; explicit Metal oracle tests opt in themselves."""
import mlx.core as mx

# Apply before test-module imports create streams or tensor fixtures.
mx.set_default_device(mx.cpu)

# The structured-output scanner pool spawns processes; unit tests exercise
# it explicitly where they need it.
import os

os.environ.setdefault("MLX2_STRUCTURED_WORKERS", "0")

import pytest


@pytest.fixture(autouse=True)
def _restore_process_environment():
    """Adapters' configure_environment() writes os.environ directly.  Restore
    it after every test so one test's serving profile (e.g. Flash-Next's
    segmented self-MTP layout) cannot change a later module's behaviour
    (test_qualify_gdn_retirement after test_execution_policy, 2026-10-01)."""
    saved = dict(os.environ)
    yield
    if os.environ != saved:
        os.environ.clear()
        os.environ.update(saved)
