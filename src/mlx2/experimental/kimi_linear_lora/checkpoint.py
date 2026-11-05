"""Immutable adapter-only Adam checkpoints; no base-model serialization.

Original project implementation. Resume is exact-bound to the saved run's
checkpoint, source, code, data, spec, optimizer law and MLX version.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import mlx.core as mx
from mlx import optimizers
from mlx.utils import tree_flatten

SCHEMA = 'mlx2.kimi_linear_lora.training_checkpoint.v1'
MAX_BYTES = 512 << 20
HASH = re.compile(r'[0-9a-f]{64}\Z')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def _binding(session, provenance):
    required = {'checkpoint', 'source_revision', 'code_sha256', 'data_sha256', 'spec_sha256'}
    if not isinstance(provenance, dict) or not required <= provenance.keys():
        raise ValueError('checkpoint provenance requires checkpoint/source/code/data/spec bindings')
    if provenance['checkpoint'] != session.base_checkpoint or provenance['source_revision'] != session.source_revision:
        raise ValueError('checkpoint provenance differs from active session identity')
    code = provenance['code_sha256']
    if (not isinstance(code, dict) or not code or any(not isinstance(k, str) or not k for k in code)
            or any(not isinstance(v, str) or not HASH.fullmatch(v)
                   for v in [*code.values(), provenance['data_sha256'], provenance['spec_sha256']])):
        raise ValueError('code/data/spec bindings require SHA256 hashes')
    # Canonical JSON also refuses nonfinite/nonserializable provenance.
    return json.loads(_json(provenance))


def _optimizer_config(optimizer):
    if type(optimizer) is not optimizers.Adam or optimizer._schedulers:
        raise ValueError('checkpoint supports stock Adam with constant learning rate')
    return {'type': 'mlx.optimizers.Adam', 'betas': list(optimizer.betas),
            'eps': optimizer.eps, 'bias_correction': optimizer.bias_correction,
            'learning_rate': float(optimizer.learning_rate.item())}


def _adapter_config(session):
    return {'keys': list(session.keys), 'rank': session.rank, 'scale': session.scale,
            'training_kda_policy': getattr(session, 'kda_policy', 'reference'),
            'training_mla_policy': getattr(session, 'mla_policy', 'expanded'),
            'projection_chunk_size': getattr(session, 'projection_chunk_size', None)}


def _historical_adapter_defaults(config):
    # Old manifests describe expanded MLA and dense head loss only. Preserve
    # their hashed bytes; defaults are applied solely to the comparison view.
    if not isinstance(config, dict):
        raise ValueError('checkpoint adapter configuration must be an object')
    result = {'training_mla_policy': 'expanded', 'projection_chunk_size': None, **config}
    mla = result['training_mla_policy']
    size = result['projection_chunk_size']
    if type(mla) is not str or mla not in {'expanded', 'padded_sdpa'}:
        raise ValueError('checkpoint training_mla_policy is invalid')
    if size is not None and (type(size) is not int or size < 1):
        raise ValueError('checkpoint projection_chunk_size must be a positive integer or null')
    return result


def _check_path(path, session):
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink():
        raise ValueError('checkpoint path must not be a symlink')
    path = raw.resolve()
    if any(path == base or path.is_relative_to(base) or base.is_relative_to(path)
           for base in session.base_paths):
        raise ValueError('checkpoint path overlaps a protected base path')
    return path


def _pack(value, tensors):
    if isinstance(value, mx.array):
        key = f'tensor_{len(tensors):06d}'
        tensors[key] = value
        return {'kind': 'array', 'key': key, 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {'kind': 'dict', 'items': {key: _pack(child, tensors) for key, child in value.items()}}
    if isinstance(value, (list, tuple)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list',
                'items': [_pack(child, tensors) for child in value]}
    if value is None or type(value) in (bool, int, str) or (type(value) is float and math.isfinite(value)):
        return {'kind': 'scalar', 'value': value}
    raise ValueError(f'unsupported checkpoint state type: {type(value).__name__}')


def _unpack(node, tensors, used):
    kind = node['kind']
    if kind == 'array':
        key = node['key']
        if key in used or key not in tensors:
            raise ValueError('checkpoint tensor coverage mismatch')
        value = tensors[key]
        if list(value.shape) != node['shape'] or str(value.dtype) != node['dtype']:
            raise ValueError('checkpoint tensor shape/dtype mismatch')
        used.add(key)
        return value
    if kind == 'dict':
        return {key: _unpack(child, tensors, used) for key, child in node['items'].items()}
    if kind in {'list', 'tuple'}:
        values = [_unpack(child, tensors, used) for child in node['items']]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'scalar':
        return node['value']
    raise ValueError('invalid checkpoint state tree')


def _validate_state(session, state):
    factors = session.weights()
    saved = state['adapter']
    if set(saved) != set(factors) or any(saved[k].shape != factors[k].shape or saved[k].dtype != factors[k].dtype for k in factors):
        raise ValueError('checkpoint adapter coverage/shape/dtype mismatch')
    expected = {'step', 'learning_rate'} | {f'{key}.{moment}' for key in factors for moment in ('m', 'v')}
    actual = dict(tree_flatten(state['optimizer']))
    if set(actual) != expected:
        raise ValueError('checkpoint Adam state must cover only the adapter factors')
    for key, factor in factors.items():
        for moment in ('m', 'v'):
            value = actual[f'{key}.{moment}']
            if value.shape != factor.shape or value.dtype != factor.dtype:
                raise ValueError('checkpoint Adam moment shape/dtype mismatch')
    for key in ('step', 'learning_rate'):
        if actual[key].shape != ():
            raise ValueError('checkpoint Adam step/rate must be scalar')
    if actual['step'].item() < 0 or actual['learning_rate'].item() <= 0:
        raise ValueError('invalid checkpoint Adam step/rate')
    rng = state['rng']
    if not isinstance(rng, list) or len(rng) != len(mx.random.state):
        raise ValueError('checkpoint RNG state mismatch')
    for saved_rng, current_rng in zip(rng, mx.random.state, strict=True):
        if saved_rng.shape != current_rng.shape or saved_rng.dtype != current_rng.dtype:
            raise ValueError('checkpoint RNG shape/dtype mismatch')


def save_checkpoint(path, session, optimizer, *, next_data_cursor, provenance):
    """Write a new durable checkpoint after an update; never replace a path."""
    if type(next_data_cursor) is not int or next_data_cursor < 0:
        raise ValueError('next_data_cursor must be a nonnegative integer')
    provenance = _binding(session, provenance)
    config = _optimizer_config(optimizer)
    path = _check_path(path, session)
    if path.exists():
        raise ValueError('checkpoint output already exists')
    if not optimizer._initialized:
        raise ValueError('checkpoint requires initialized Adam state')
    state = {'adapter': session.weights(), 'optimizer': optimizer.state,
             'rng': [mx.array(key) for key in mx.random.state]}
    _validate_state(session, state)
    tensors = {}
    tree = _pack(state, tensors)
    if sum(value.nbytes for value in tensors.values()) > MAX_BYTES:
        raise ValueError('adapter-only checkpoint exceeds size bound')
    mx.eval(tensors)
    if any(mx.issubdtype(value.dtype, mx.floating) and not mx.all(mx.isfinite(value)).item() for value in tensors.values()):
        raise ValueError('checkpoint contains nonfinite values')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{path.name}.tmp-', dir=path.parent))
    try:
        payload = temporary / 'state.safetensors'
        mx.save_safetensors(str(payload), tensors)
        manifest = {'schema': SCHEMA, 'mlx_version': version('mlx'), 'provenance': provenance,
                    'adapter': _adapter_config(session), 'optimizer': config,
                    'next_data_cursor': next_data_cursor, 'optimizer_step': int(optimizer.step.item()),
                    'state_tree': tree, 'state_sha256': _sha(payload), 'state_bytes': payload.stat().st_size}
        manifest['checkpoint_id'] = hashlib.sha256(_json(manifest).encode()).hexdigest()
        (temporary / 'checkpoint.json').write_text(_json(manifest) + '\n')
        for file in temporary.iterdir():
            with file.open('rb') as stream:
                os.fsync(stream.fileno())
        # Reserve exclusively: rename(directory) could replace an empty
        # destination created after an existence check. The manifest is the
        # commit marker and is published last. An interrupted publication
        # leaves an incomplete, immutable directory which restore refuses.
        os.mkdir(path, mode=0o700)
        os.replace(payload, path / 'state.safetensors')
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
            os.replace(temporary / 'checkpoint.json', path / 'checkpoint.json')
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def restore_checkpoint(path, session, optimizer, *, expected_provenance):
    """Validate every binding and state tensor before restoring factors/moments/RNG."""
    expected = _binding(session, expected_provenance)
    path = _check_path(path, session)
    metadata, payload = path / 'checkpoint.json', path / 'state.safetensors'
    if any(p.is_symlink() or not p.is_file() for p in (metadata, payload)):
        raise ValueError('checkpoint files are missing or symlinked')
    if metadata.stat().st_size > 16 << 20 or payload.stat().st_size > MAX_BYTES + (1 << 20):
        raise ValueError('checkpoint exceeds size bound')
    manifest = json.loads(metadata.read_text())
    identifier = manifest.pop('checkpoint_id', None)
    if identifier != hashlib.sha256(_json(manifest).encode()).hexdigest():
        raise ValueError('checkpoint manifest hash mismatch')
    manifest['checkpoint_id'] = identifier
    if (manifest['schema'] != SCHEMA or manifest['provenance'] != expected
            or manifest['mlx_version'] != version('mlx')
            or _historical_adapter_defaults(manifest['adapter']) != _adapter_config(session)
            or manifest['optimizer'] != _optimizer_config(optimizer)):
        raise ValueError('checkpoint provenance, adapter, optimizer or MLX binding mismatch')
    if manifest['state_bytes'] != payload.stat().st_size or manifest['state_sha256'] != _sha(payload):
        raise ValueError('checkpoint payload hash mismatch')
    if type(manifest['next_data_cursor']) is not int or manifest['next_data_cursor'] < 0:
        raise ValueError('invalid checkpoint data cursor')
    tensors = mx.load(str(payload))
    used = set()
    state = _unpack(manifest['state_tree'], tensors, used)
    if used != set(tensors):
        raise ValueError('checkpoint has unreferenced tensors')
    _validate_state(session, state)
    mx.eval(tensors)
    if (int(state['optimizer']['step'].item()) != manifest['optimizer_step']
            or any(mx.issubdtype(value.dtype, mx.floating) and not mx.all(mx.isfinite(value)).item() for value in tensors.values())):
        raise ValueError('checkpoint optimizer step mismatch or nonfinite values')
    # All checks above are read-only. The setters below have validated shapes.
    session.restore_weights(state['adapter'])
    optimizer.state = state['optimizer']
    # Current MLX exposes an indexable _RandomState, not a replaceable list.
    # Mutate its key arrays in place so the native generator sees the restore.
    for current, saved in zip(mx.random.state, state['rng'], strict=True):
        current[:] = saved
    mx.eval(list(mx.random.state))
    return manifest
