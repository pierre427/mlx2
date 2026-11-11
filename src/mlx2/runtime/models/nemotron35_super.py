"""Owned Super VL composition using the existing Nemotron-H and RADIO math.

Source: the pinned NVIDIA remote-code checkpoint, not the unrelated T7 RADIO
base weights. Provenance: provenance/nemotron35-super-vl.json.
"""

import mlx.core as mx
from mlx import nn

from ...adapters.radio_config import RadioConfig
from ..progressive_external_verify import PROGRESSIVE_TARGET_PROTOCOL_VERSION
from .nemotron_h import Model as LanguageModel
from .radio import RadioModel


def pixel_shuffle_v2(x):
    batch, height, width, channels = x.shape
    if height % 2 or width % 2:
        raise ValueError("Super VL pixel-shuffle grid must be even")
    x = x.reshape(batch, height, width // 2, channels * 2).transpose(0, 2, 1, 3)
    return x.reshape(batch, width // 2, height // 2, channels * 4).transpose(0, 2, 1, 3)


class SuperRadio(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.radio_model = RadioModel(config)
        self.radio_model.model.patch_generator.video_embedder = nn.Linear(
            6 * config.patch_size**2, config.hidden_size, bias=False
        )

    def __call__(self, normalized_pixels, *, video=False):
        x = normalized_pixels
        expected = 6 if video else 3
        if x.ndim != 4 or x.shape[1] != expected or x.shape[0] < 1:
            raise ValueError(
                "Super VL expects normalized NCHW RGB pixels or packed frame pairs"
            )
        cfg = self.radio_model.config
        height, width = x.shape[-2:]
        if (
            height % (2 * cfg.patch_size)
            or width % (2 * cfg.patch_size)
            or height * width // cfg.patch_size**2 > 13312
        ):
            raise ValueError("Super VL image grid exceeds the supported patch budget")
        generator = self.radio_model.model.patch_generator
        projection = generator.video_embedder if video else generator.embedder
        # The checkpoint composition externalizes InputConditioner. Pixels
        # have already been resized and normalized exactly once by the processor.
        x = projection(generator._im_to_patches(x).astype(projection.weight.dtype))
        x = generator.cls_token(
            x
            + generator._get_pos_embeddings(
                x.shape[0], (height // cfg.patch_size, width // cfg.patch_size)
            )
        )
        for block in self.radio_model.model.blocks:
            x = block(x)
        return x[:, cfg.num_skip :]


class VisionMLP(nn.Module):
    def __init__(self, inputs, intermediate, outputs):
        super().__init__()
        self.norm = nn.RMSNorm(inputs, eps=1e-5)
        self.linear1 = nn.Linear(inputs, intermediate, bias=False)
        self.linear2 = nn.Linear(intermediate, outputs, bias=False)

    def __call__(self, x):
        return self.linear2(nn.relu2(self.linear1(self.norm(x))))


class VisionProjector(nn.Module):
    def __init__(self, vision_width, intermediate, output_width):
        super().__init__()
        self.vision_final_layernorm = nn.LayerNorm(vision_width, eps=1e-6)
        self.mlp1 = VisionMLP(vision_width * 4, intermediate, output_width)

    def __call__(self, features, *, grid):
        batch = features.shape[0]
        x = self.vision_final_layernorm(features)
        x = pixel_shuffle_v2(x.reshape(batch, *grid, x.shape[-1]))
        return self.mlp1(x.reshape(batch, -1, x.shape[-1]))


class Model(LanguageModel):
    # B1 speculative forwards inherit ordinary S=1 recurrence/attention and
    # exact hybrid record/replay. The shared coordinator stages each tile on
    # private caches and publishes only the accepted boundary. This declares
    # that functional protocol, not a fused row kernel or qualification.
    progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION

    def __init__(self, args, *, radio_config=None, projector_hidden_size=20480):
        super().__init__(args)
        radio_config = radio_config or RadioConfig(
            1280, 32, 16, 5120, 16, 2048, 3, 7, (0, 1), (512, 512), False
        )
        self.vision_model = SuperRadio(radio_config)
        self.vision_projector = VisionProjector(
            radio_config.hidden_size, projector_hidden_size, args.hidden_size
        )

    def encode_images(self, normalized_pixels):
        cfg = self.vision_model.radio_model.config
        features = self.vision_model(normalized_pixels)
        return self.vision_projector(
            features,
            grid=tuple(n // cfg.patch_size for n in normalized_pixels.shape[-2:]),
        )

    def encode_video(self, normalized_frames):
        if normalized_frames.shape[0] % 2:
            normalized_frames = mx.concatenate(
                [normalized_frames, normalized_frames[-1:]]
            )
        count, _channels, height, width = normalized_frames.shape
        pixels = normalized_frames.reshape(count // 2, 6, height, width)
        cfg = self.vision_model.radio_model.config
        return self.vision_projector(
            self.vision_model(pixels, video=True),
            grid=(height // cfg.patch_size, width // cfg.patch_size),
        )

    def prefill_forward(
        self, inputs, cache=None, *, inputs_embeds=None, image_pixels=None,
        video_pixels=None, media_order=None,
    ):
        if image_pixels is not None or video_pixels is not None:
            if inputs_embeds is not None:
                raise ValueError("Super VL image prefill requires one unmixed row")
            inputs_embeds = self.image_embeddings(
                inputs, image_pixels, video_pixels=video_pixels, media_order=media_order
            )
        return self._checkpoint_logits(inputs, cache, inputs_embeds)

    def image_embeddings(self, inputs, image_pixels, *, video_pixels=None, media_order=None):
        """Token embeddings with each image's (and video's) projected features
        at its ``<image>`` (id 18) placeholders, in prompt order.

        ``media_order`` names each media item's kind in prompt order; absent,
        every item is an image.  A video contributes its tubelets' features in
        order (``encode_video``), as the source's ``masked_scatter`` does."""
        if inputs.shape[0] != 1:
            raise ValueError("Super VL image prefill requires one unmixed row")
        import numpy as np

        images = list(image_pixels or ())
        videos = list(video_pixels or ())
        order = list(media_order) if media_order is not None else ["image"] * len(images)
        if order.count("image") != len(images) or order.count("video") != len(videos) or (
            set(order) - {"image", "video"}
        ):
            raise ValueError("Super VL media order does not match its inputs")
        images, videos = iter(images), iter(videos)
        parts = []
        for kind in order:
            if kind == "image":
                parts.append(self.encode_images(mx.array(next(images))))
            else:
                encoded = self.encode_video(mx.array(next(videos)))
                parts.append(encoded.reshape(1, -1, encoded.shape[-1]))
        positions = np.flatnonzero(np.asarray(inputs[0]) == 18).tolist()
        features = mx.concatenate(parts, axis=1)
        if len(positions) != features.shape[1]:
            raise ValueError("Super VL media placeholders/features differ")
        embeddings = self.embed_tokens(inputs)
        embeddings[0, mx.array(positions)] = features[0].astype(embeddings.dtype)
        return embeddings

    def media_prefill_body(self, inputs, cache, capture_layers, *, image_pixels=None,
                           video_pixels=None, media_order=None):
        """External-draft media prefill: the ordinary route's image prefill
        (same embeddings, same full-chunk backbone) returning the drafter's
        target taps instead of logits.  The runtime runs the whole uncached
        prompt but its anchor as one call."""
        embeddings = self.image_embeddings(
            inputs, image_pixels, video_pixels=video_pixels, media_order=media_order
        )
        taps = []
        self.backbone(
            inputs, cache=cache, capture_layers=tuple(capture_layers),
            hidden_sink=taps, inputs_embeds=embeddings,
        )
        stats = self.__dict__.setdefault("_mlx2_media_stats", {"external_prefills": 0})
        stats["external_prefills"] += 1
        return mx.concatenate(taps, axis=-1)

    def sanitize(self, weights):
        output = {}
        for name, value in weights.items():
            if name.startswith("language_model."):
                name = name.removeprefix("language_model.")
            if name.startswith("mlp1."):
                # Vendor ``mlp1.N.weight``; MLX-VLM exports ``mlp1.layers.N``.
                parts = name.removeprefix("mlp1.").removeprefix("layers.").split(".")
                parts[0] = {"0": "norm", "1": "linear1", "3": "linear2"}[parts[0]]
                name = "vision_projector.mlp1." + ".".join(parts)
            if name.startswith("vision_final_layernorm."):
                # MLX-VLM exports the projector's leading LayerNorm top level.
                name = "vision_projector." + name
            if name == "vision_model.summary_idxs":
                name = "vision_model.radio_model.summary_idxs"
                value = value.astype(mx.int32)
            output[name] = value
        if (
            "vision_model.radio_model.summary_idxs" not in output
            and any(name.startswith("vision_model.") for name in output)
        ):
            # Not exported by MLX-VLM; it is configuration (summary_idxs
            # [0, 1], checked by the inspector), not a learned tensor.
            output["vision_model.radio_model.summary_idxs"] = (
                self.vision_model.radio_model.summary_idxs
            )
        return super().sanitize(output)
