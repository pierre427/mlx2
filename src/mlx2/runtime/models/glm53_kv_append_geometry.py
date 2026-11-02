# SPDX-License-Identifier: MIT
"""Dependency-free geometry of glm53_kv_append (read by the memory planner).

``glm53_kv_append`` (which imports MLX) reads these at call time; the cache
budget prices them without importing MLX.
"""

KV_APPEND_POLICIES = ("reference", "rotate_v1")
# Rotating full buffers per MLA plane under rotate_v1 (bench: K=2 removes the
# per-token copy for a one-step-lookahead decode loop; K=3/4 measured equal).
RING_BUFFERS = 2
# Capacity reservation granularity of a rotate_v1 buffer, in rows.
BASE_BLOCK = 2048
