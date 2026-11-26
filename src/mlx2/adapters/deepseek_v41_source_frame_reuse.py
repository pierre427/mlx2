# SPDX-License-Identifier: MIT
"""Identity for opt-in, frame-owned Source prefill setup reuse."""
from functools import lru_cache
import hashlib
from pathlib import Path

POLICY='prefill_v1'

@lru_cache(maxsize=1)
def implementation_identity():
    root=Path(__file__).resolve().parents[1]
    paths=('adapters/deepseek_v41_source_frame_reuse.py',
           'adapters/deepseek_v41_source_prefill.py',
           'adapters/deepseek_v41_source_math.py',
           'adapters/deepseek_v41_source_weights.py',
           'adapters/deepseek_v41_source_lfu.py',
           'runtime/models/deepseek_v41.py',
           'runtime/technologies/sparse_selection.py', 'runtime/technologies/window_attention/__init__.py',
           'runtime/technologies/window_attention/vector_sources.py', 'runtime/technologies/window_attention/vector_launch.py',
           'runtime/technologies/dispatch.py',
           'runtime/models/deepseek_v41_fused.py',
           'runtime/models/deepseek_v41_frame_geometry.py',
           'runtime/technologies/hyper_connections.py')
    records='\n'.join(p+':'+hashlib.sha256((root/p).read_bytes()).hexdigest()for p in paths)
    return hashlib.sha256((POLICY+'\n'+records).encode()).hexdigest()
