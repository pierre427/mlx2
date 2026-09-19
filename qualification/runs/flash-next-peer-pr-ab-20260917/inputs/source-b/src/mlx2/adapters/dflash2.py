"""Metadata binding and strict loader for external DFlash2 draft artifacts."""
from pathlib import Path
import hashlib
import json
import struct
from ..runtime.drafters.dflash2_config import DFlash2Config


_SAFETENSORS_HEADER_LIMIT = 64 << 20
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key in safetensors header: {key}")
        value[key] = item
    return value


def _decode_unique_json(raw, label):
    try:
        return json.loads(raw, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON metadata: {label}") from exc


def _expected_weight_shapes(args):
    """Raw checkpoint schema before ``DFlash2DraftModel.sanitize``."""
    hidden = args.hidden_size
    heads = args.num_attention_heads * args.head_dim
    kv_heads = args.num_key_value_heads * args.head_dim
    groups = hidden // args.conv_group_size
    shapes = {
        "fc.weight": [hidden, len(args.target_layer_ids) * hidden],
        "hidden_norm.weight": [hidden],
        "norm.weight": [hidden],
        "candidate_selector.hidden_projection.weight": [args.selector_rank, hidden],
        "candidate_selector.predecessor_codebook": [args.vocab_size, args.selector_rank],
        "candidate_selector.successor_codebook": [args.vocab_size, args.selector_rank],
    }
    for index in range(args.num_hidden_layers):
        prefix = f"layers.{index}."
        shapes.update(
            {
                prefix + "attention_conv.base_kernel": [2, args.conv_kernel_size, hidden],
                prefix + "attention_conv.kernel_projection.weight": [
                    2 * args.conv_kernel_size * groups,
                    hidden,
                ],
                prefix + "input_layernorm.weight": [hidden],
                prefix + "mlp.down_proj.weight": [hidden, args.intermediate_size],
                prefix + "mlp.gate_proj.weight": [args.intermediate_size, hidden],
                prefix + "mlp.up_proj.weight": [args.intermediate_size, hidden],
                prefix + "mlp_conv.base_kernel": [2, args.conv_kernel_size, hidden],
                prefix + "mlp_conv.kernel_projection.weight": [
                    2 * args.conv_kernel_size * groups,
                    hidden,
                ],
                prefix + "post_attention_layernorm.weight": [hidden],
                prefix + "self_attn.k_norm.weight": [args.head_dim],
                prefix + "self_attn.k_proj.weight": [kv_heads, hidden],
                prefix + "self_attn.o_proj.weight": [hidden, heads],
                prefix + "self_attn.q_norm.weight": [args.head_dim],
                prefix + "self_attn.q_proj.weight": [heads, hidden],
                prefix + "self_attn.v_proj.weight": [kv_heads, hidden],
            }
        )
    return shapes


def _read_safetensors_header(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Truncated safetensors file: {path.name}")
        length = struct.unpack("<Q", raw_length)[0]
        if not 0 < length <= min(_SAFETENSORS_HEADER_LIMIT, size - 8):
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        raw_header = stream.read(length)
    header = _decode_unique_json(raw_header, path.name)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {path.name}")
    return raw_header, header, size - 8 - length


def _validate_weight_headers(files, mapping, args, configured_dtype):
    expected = _expected_weight_shapes(args)
    observed = {}
    header_digests = []
    dtype = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}.get(
        str(configured_dtype).lower()
    )
    if dtype is None:
        raise ValueError(f"Unsupported DFlash2 checkpoint dtype: {configured_dtype!r}")
    for path in files:
        raw_header, header, payload_size = _read_safetensors_header(path)
        header_digests.append(hashlib.sha256(raw_header).hexdigest())
        ranges = []
        for name, record in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"Invalid tensor metadata for {name!r}")
            if name in observed:
                raise ValueError(f"Duplicate DFlash2 tensor across shards: {name}")
            shape = record["shape"]
            offsets = record["data_offsets"]
            if record["dtype"] != dtype or not isinstance(shape, list) or not all(
                type(value) is int and value >= 0 for value in shape
            ):
                raise ValueError(f"DFlash2 tensor dtype/shape mismatch: {name}")
            if not (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(type(value) is int for value in offsets)
                and 0 <= offsets[0] <= offsets[1] <= payload_size
            ):
                raise ValueError(f"Invalid DFlash2 tensor offsets: {name}")
            elements = 1
            for value in shape:
                elements *= value
            if offsets[1] - offsets[0] != elements * _DTYPE_BYTES[dtype]:
                raise ValueError(f"DFlash2 tensor byte size mismatch: {name}")
            ranges.append((offsets[0], offsets[1], name))
            observed[name] = shape
            if mapping is not None and mapping.get(name) != path.name:
                raise ValueError(f"DFlash2 weight index/shard mismatch: {name}")
        for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
            if previous[1] > current[0]:
                raise ValueError(
                    f"Overlapping DFlash2 tensor payloads: {previous[2]}, {current[2]}"
                )
    if mapping is not None and set(mapping) != set(observed):
        raise ValueError("DFlash2 weight index does not match shard headers")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(
            name for name in set(expected) & set(observed) if expected[name] != observed[name]
        )
        raise ValueError(
            "DFlash2 checkpoint schema mismatch "
            f"(missing={missing}, extra={extra}, wrong_shapes={wrong})"
        )
    return header_digests


def inspect_drafter(path, target):
    path = Path(path).expanduser().resolve()
    config = _decode_unique_json((path/'config.json').read_bytes(), 'draft config.json')
    if config.get('architectures') != ['DFlash2DraftModel']:
        raise ValueError('Expected DFlash2DraftModel artifact')
    args = DFlash2Config.from_dict(config)
    target_config = _decode_unique_json((Path(target)/'config.json').read_bytes(), 'target config.json')
    text = target_config.get('text_config', target_config)
    for name in ('hidden_size','vocab_size'):
        if text[name] != getattr(args,name): raise ValueError(f'DFlash2 target {name} mismatch')
    if text['num_hidden_layers'] != args.num_target_layers:
        raise ValueError('DFlash2 target layer count mismatch')
    digest=hashlib.sha256(); digest.update((path/'config.json').read_bytes())
    index=path/'model.safetensors.index.json'
    if index.exists():
        content=index.read_bytes();digest.update(content);mapping=_decode_unique_json(content, 'draft weight index').get('weight_map')
        if not isinstance(mapping,dict) or not mapping: raise ValueError('Invalid draft weight index')
        names=sorted(set(mapping.values()))
    else:
        mapping=None;names=['model.safetensors']
    files=[]
    paths=[]
    for name in names:
        file=(path/name).resolve()
        if not file.is_relative_to(path) or file.suffix!='.safetensors': raise ValueError('Invalid draft shard path')
        stat=file.stat();record=(name,stat.st_size,stat.st_mtime_ns);files.append(record);paths.append(file);digest.update(json.dumps(record).encode())
    header_digests = _validate_weight_headers(paths, mapping, args, config.get('dtype'))
    digest.update(json.dumps(header_digests).encode())
    return {'path':str(path),'fingerprint':digest.hexdigest(),'files':files,'header_sha256':header_digests,'config':config,'args':args}


def load_drafter(record,target_model):
    # Import/load only after both artifact roles and dimensions are checked.
    import mlx.core as mx
    import mlx.nn as nn
    from ..runtime.drafters.dflash2 import DFlash2DraftModel
    model=DFlash2DraftModel(record['args'])
    weights={}
    for name,_,_ in record['files']:weights.update(mx.load(str(Path(record['path'])/name)))
    weights=model.sanitize(weights)
    quant=record['config'].get('quantization') or record['config'].get('quantization_config')
    if quant:
        def predicate(name,module):
            override=quant.get(name,quant.get('model.'+name))
            return override if override is not None else hasattr(module,'to_quantized') and name+'.scales' in weights
        nn.quantize(model,group_size=quant['group_size'],bits=quant['bits'],mode=quant.get('mode','affine'),class_predicate=predicate)
    # Target embedding/head are absent from draft checkpoint and bound AFTER strict load.
    model.load_weights(list(weights.items()),strict=True)
    model.eval();mx.eval(model.parameters());weights.clear()
    return model.bind(target_model)
