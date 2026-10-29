# SPDX-License-Identifier: MIT
"""Source-owned V4.1 vision binding; see provenance/deepseek-v41-source-vision.json."""

import math
from dataclasses import dataclass

import numpy as np
from PIL import ImageOps

REVISION = "mlx2.v41.source-vision-reading-order-bf16.v1"
TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
_MARKER = "\ue000mlx2-v41-image-"


@dataclass(frozen=True)
class ImageInput:
    start: int
    patches: np.ndarray
    n_vit_h: int
    n_vit_w: int
    types: np.ndarray


def image_token_types(height, width):
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("image grid must have positive integer dimensions")
    return np.array([IMAGE_START] + ([IMAGE] * width + [IMAGE_NEW_LINE]) * height + [IMAGE_END], np.int32)


def image_grid(width, height, config):
    """Official resize law, independent of token position and compressor groups."""
    if min(width, height) <= 0:
        raise ValueError("image dimensions must be positive")
    p, r = config["patch_size"], config["downsample_ratio"]
    budget = config["max_image_tokens"]
    ratio_limit = config.get("max_wh_ratio")
    if ratio_limit is not None and width > height * ratio_limit:
        width = height * ratio_limit
    if width * height < config["min_pixels"]:
        scale = math.sqrt(config["min_pixels"] / (width * height))
        width, height = int(width * scale), int(height * scale)
    bh, bw = math.ceil(height / p) * p, math.ceil(width / p) * p
    def grid(h, w):
        return math.ceil((h // p) / r), math.ceil((w // p) / r)
    gh, gw = grid(bh, bw)
    if gh * (gw + 1) + 2 > budget:
        ratio = height / width
        wf = math.sqrt((budget - 2) / ratio + 0.25) - 0.5
        hf, cell = wf * ratio, p * r
        if wf < 1:
            bh, bw = (budget - 2) // 2 * cell, cell
        elif hf < 1:
            bh, bw = cell, (budget - 3) * cell
        else:
            scale = min(math.floor(wf) * cell / width, math.floor(hf) * cell / height)
            bh, bw = math.floor(height * scale / p) * p, math.floor(width * scale / p) * p
        gh, gw = grid(bh, bw)
    if min(bh, bw, gh, gw) <= 0 or gh * (gw + 1) + 2 > budget:
        raise ValueError("image cannot fit the configured token budget")
    return gh, gw, bh, bw


def prepare_image(image, config, *, start):
    gh, gw, height, width = image_grid(image.width, image.height, config)
    ratio_limit = config.get("max_wh_ratio")
    if ratio_limit is not None and image.width >= ratio_limit * image.height:
        image = image.resize((width, height))
    else:
        image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    p = config["patch_size"]
    nh, nw = height // p, width // p
    pixels = (np.asarray(image, np.float32).transpose(2, 0, 1) / 255 - 0.5) / 0.5
    patches = pixels.reshape(3, nh, p, nw, p).transpose(1, 3, 0, 2, 4).reshape(nh*nw, 3, p, p)
    return ImageInput(start, patches, nh, nw, image_token_types(gh, gw))


def expand_images(tokens, images, config, image_token_id):
    if tokens.count(image_token_id) != len(images):
        raise ValueError("V4.1 image placeholder count differs from images")
    output, prepared = [], []
    source = iter(images)
    for token in tokens:
        if token != image_token_id:
            output.append(token)
        else:
            image = prepare_image(next(source), config, start=len(output))
            prepared.append(image)
            output.extend([image_token_id] * len(image.types))
    return output, prepared


def vision_tensor_specs(config):
    v = config["vision_config"]
    d, hidden = v["hidden_size"], config["text_config"]["hidden_size"]
    p, mid, r = v["patch_size"], v["intermediate_size"], v["downsample_ratio"]
    specs = {"vision.patch_embed.proj.weight": (d, 3*p*p), "vision.patch_embed.proj.bias": (d,),
             "vision.norm.weight": (d,), "aligner.w1.weight": (hidden, d*r*r),
             "aligner.w1.bias": (hidden,), "aligner.w2.weight": (hidden, hidden),
             "aligner.w2.bias": (hidden,), "image_start": (hidden,), "image_end": (hidden,),
             "image_newline": (hidden,)}
    for i in range(v["num_hidden_layers"]):
        prefix = f"vision.blocks.{i}."
        specs.update({prefix + n: shape for n, shape in {
            "norm1.weight": (d,), "norm2.weight": (d,),
            "attn.wqkv.weight": (3*d, d), "attn.wqkv.bias": (3*d,),
            "attn.wo.weight": (d, d), "attn.wo.bias": (d,),
            "mlp.w1.weight": (2*mid, d), "mlp.w2.weight": (d, mid)}.items()})
    return specs


def vision_admission_plan(artifact):
    specs, tensors = vision_tensor_specs(artifact["config"]), artifact["tensors"]
    actual = {n for n in tensors if n.startswith(("vision.", "aligner.", "image_"))}
    if actual != set(specs):
        raise ValueError("V4.1 vision tensor inventory mismatch")
    for name, shape in specs.items():
        row = tensors[name]
        if row["dtype"] != "BF16" or tuple(row["shape"]) != shape:
            raise ValueError("V4.1 vision precision/shape mismatch: " + name)
    v = artifact["config"]["vision_config"]
    patches = v["max_image_tokens"] * v["downsample_ratio"]**2
    # Conservative dense-attention bound even when the selected SDPA is tiled.
    workspace = 8 * v["num_attention_heads"] * patches**2 + 32 * patches * v["hidden_size"]
    payload = sum(int(tensors[n]["size"]) for n in specs)
    staging = max(int(tensors[n]["size"]) for n in specs)
    return {"revision": REVISION, "payload_bytes": payload, "staging_bound_bytes": staging,
            "workspace_bound_bytes": workspace, "reservation_bytes": payload + staging + workspace,
            "tensor_count": len(specs), "qualification": "unqualified"}


class SourceVision:
    def __init__(self, artifact, target, *, admitted_bytes):
        from .deepseek_v41_mlx_weights import _refuse_quarantined_uploads
        _refuse_quarantined_uploads()
        self.plan = vision_admission_plan(artifact)
        if type(admitted_bytes) is not int or admitted_bytes < self.plan["reservation_bytes"]:
            raise MemoryError("Source vision was not reserved before target materialization")
        self.target, self.artifact = target, artifact
        self.components, self.sentinels = None, {}
        self.closed = False
        self._pending_host = None
        self._pending_arrays = []

    def load(self):
        if self.closed or self.components is not None:
            raise ValueError("Source vision owner cannot be loaded twice or after close")
        import mlx.core as mx

        from ..runtime.models.deepseek_v4_vision import VisionArgs, VisionComponents
        from .deepseek_v41_source_artifact import read_tensors
        v = self.artifact["config"]["vision_config"]
        args = VisionArgs(vision_n_layers=v["num_hidden_layers"], vision_dim=v["hidden_size"],
            vision_n_heads=v["num_attention_heads"], vision_inter_dim=v["intermediate_size"],
            vision_patch_size=v["patch_size"], vision_rope_theta=v["rope_theta"],
            vision_downsample_ratio=v["downsample_ratio"], dim=self.target.args.hidden_size)
        self.components = VisionComponents(args)
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        specs = vision_tensor_specs(self.artifact["config"])
        leaves = tree_flatten(self.components.leaf_modules(),
                              is_leaf=lambda node: isinstance(node, nn.Module))
        groups, mapped = [], set()
        for prefix, module in leaves:
            names = []
            for local, parameter in tree_flatten(module.parameters()):
                name = prefix + "." + local
                if name in mapped or name not in specs or tuple(parameter.shape) != specs[name]:
                    raise ValueError("V4.1 vision leaf parameter mapping mismatch: " + name)
                mapped.add(name)
                names.append((local, name))
            if not names:
                raise ValueError("V4.1 vision leaf has no mapped parameters: " + prefix)
            groups.append((module, names))
        expected = {name for name in specs if not name.startswith("image_")}
        if mapped != expected:
            raise ValueError("V4.1 vision leaf tensor inventory mismatch")
        # Each leaf receives its complete named parameter set in one strict
        # load. Uploads remain one host tensor at a time; evaluated arrays are
        # retained by this owner until the leaf has accepted all its weights.
        groups.extend((None, [(None, name)]) for name in specs if name.startswith("image_"))
        for module, names in groups:
            weights = []
            for local, name in names:
                data = read_tensors(self.artifact, [name])[name]
                self._pending_host = data
                if tuple(data.shape) != specs[name] or data.dtype != np.uint16:
                    raise ValueError("V4.1 vision uploaded tensor precision/shape mismatch: " + name)
                raw = mx.array(data)
                self._pending_arrays.append(raw)
                value = raw.view(mx.bfloat16)
                self._pending_arrays.append(value)
                mx.eval(value)
                if module is None:
                    self.sentinels[{"image_start": IMAGE_START, "image_end": IMAGE_END,
                                    "image_newline": IMAGE_NEW_LINE}[name]] = value
                else:
                    weights.append((local, value))
                self._pending_host = None
            if module is not None:
                module.load_weights(weights, strict=True)
            self._pending_arrays.clear()
        # Ensure no random initialized/unbound parameters survive a loader typo.
        expected = {n for n in vision_tensor_specs(self.artifact["config"]) if not n.startswith("image_")}
        actual = dict(tree_flatten(self.components.parameters()))
        if set(actual) != expected or any(a.dtype != mx.bfloat16 for a in actual.values()):
            raise ValueError("V4.1 vision loader left unmatched parameters")
        mx.eval(self.components.parameters(), self.sentinels)
        return self

    def __call__(self, tokens, cache, images):
        import mlx.core as mx
        if self.closed or self.components is None or tokens.shape[0] != 1 or any(c.offset for c in cache):
            raise ValueError("V4.1 image encoding requires an empty ordinary B1 frontier")
        ids = np.asarray(tokens[0])
        mask = np.zeros(len(ids), bool)
        h = self.target.round(self.target.weights.embedding(tokens.astype(mx.uint32)))
        image_id = self.artifact["config"]["image_token_id"]
        for item in images:
            first, last = item.start, item.start + len(item.types)
            ratio = self.artifact["config"]["vision_config"]["downsample_ratio"]
            expected = image_token_types(math.ceil(item.n_vit_h/ratio), math.ceil(item.n_vit_w/ratio))
            if (first < 0 or last > len(ids) or mask[first:last].any()
                    or not np.array_equal(item.types, expected) or np.any(ids[first:last] != image_id)):
                raise ValueError("V4.1 image span/token/geometry mismatch")
            features = self.components(mx.array(item.patches).astype(mx.bfloat16), item.n_vit_h, item.n_vit_w)
            span = mx.zeros((len(item.types), self.target.args.hidden_size), mx.bfloat16)
            for kind, vector in self.sentinels.items():
                span[mx.array(np.flatnonzero(item.types == kind))] = vector
            span[mx.array(np.flatnonzero(item.types == IMAGE))] = features
            h[0, first:last] = span.astype(h.dtype)
            # Admission reserves one tower workspace. Materialize this image's
            # merged embedding before another image can retain a tower graph.
            mx.eval(h)
            mask[first:last] = True
        if not images or not np.array_equal(mask, ids == image_id):
            raise ValueError("V4.1 image placeholders are not covered exactly once")
        return h, mask

    def close(self):
        if self.closed:
            return
        import mlx.core as mx

        from .deepseek_v41_mlx_weights import _quarantine_upload, _release_upload
        try:
            mx.synchronize()
        except BaseException:
            # Construction may have failed before its adapter was returned.
            # Keep incomplete uploads and the borrowed target strongly owned.
            _quarantine_upload(self)
            raise
        self.components, self.sentinels = None, {}
        self._pending_host = None
        self._pending_arrays.clear()
        self.closed = True
        _release_upload(self)


def prepare_request(adapter, request, *, file_loader=None):
    from ..multimodal import media_fingerprint, resolve_media
    from ..runtime.models.deepseek_v41_encoding import IMAGE_PLACEHOLDER
    from .mlx_vlm import _plain_messages, _source
    if getattr(adapter, "vision_weights", None) is None or adapter.draft_model is not None:
        raise ValueError("V4.1 images require the explicitly selected ordinary Source vision route")
    media, replacements = [], []
    for message in request["messages"]:
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for part in parts:
            if part["type"] == "text":
                if _MARKER in part.get("text", "") or IMAGE_PLACEHOLDER in part.get("text", ""):
                    raise ValueError("V4.1 text must not contain image control placeholders")
                continue
            if part["type"] not in ("image_url", "input_image"):
                raise ValueError("V4.1 supports images, not audio/video input")
            media.append(resolve_media(_source(part, "image"), kind="image", file_loader=file_loader))
            replacements.append(f"{_MARKER}{len(media)-1}\ue001")
    if not media or len(media) > 8:
        raise ValueError("V4.1 image requests require one through eight images")
    messages = _plain_messages(request["messages"], replacements)
    text = adapter.render_prompt({**request, "messages": messages})
    for marker in replacements:
        if text.count(marker) != 1:
            raise ValueError("V4.1 image markers were changed by prompt rendering")
        text = text.replace(marker, IMAGE_PLACEHOLDER)
    tokens = list(adapter.tokenizer.encode(text, add_special_tokens=False))
    tokens, images = expand_images(tokens, [m.value for m in media], adapter.config["vision_config"], adapter.config["image_token_id"])
    if len(tokens) > adapter.max_context or images[-1].start + len(images[-1].types) >= len(tokens):
        raise ValueError("V4.1 image prompt must fit context and end in a text generation prefix")
    return {**request, "messages": messages, "_mlx2_prompt_tokens": tokens,
        "_mlx2_prefill_inputs": {"image_inputs": images},
        "_mlx2_media_token_end": images[-1].start + len(images[-1].types),
        "_mlx2_media_fingerprint": media_fingerprint(media, policy={"processor": REVISION,
            "artifact": adapter.identity["fingerprint"], "vision_config": adapter.config["vision_config"]})}
