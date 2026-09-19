"""Routine tests are CPU-only; explicit Metal oracle tests opt in themselves."""
import mlx.core as mx

# Apply before test-module imports create streams or tensor fixtures.
mx.set_default_device(mx.cpu)

# The structured-output scanner pool spawns processes; unit tests exercise
# it explicitly where they need it.
import os

os.environ.setdefault("MLX2_STRUCTURED_WORKERS", "0")
