# SPDX-License-Identifier: MIT
# Adapted from mlx-vlm; see provenance/muse-vision-serving.json and .NOTICE.
"""Muse image encoder composed with the native text/AST target contract."""
import mlx.core as mx
import mlx.nn as nn

from ...adapters.muse_vision_config import VisionConfig
from .muse_glimmer import Model as TextModel
from .muse_vision import VisionModel, gelu
from ..prefill_checkpoint_rows import select_checkpoint_rows


class VisionAdapter(nn.Module):
    def __init__(self, width, projector):
        super().__init__()
        self.fc1 = nn.Linear(width, projector, bias=False)
        self.fc2 = nn.Linear(projector, projector, bias=False)

    def __call__(self, x):
        return gelu(self.fc2(gelu(self.fc1(x))))


class Model(TextModel):
    def __init__(self, args, root_config):
        super().__init__(args)
        vision = VisionConfig.from_dict(root_config['vision_config'])
        merged_width = vision.hidden_size * vision.merge_size ** 2
        if root_config.get('out_hidden_size', merged_width) != merged_width or root_config.get('projector_hidden_act', 'gelu') != 'gelu':
            raise ValueError('Unsupported Muse vision projection geometry')
        self.vision_tower = VisionModel(vision)
        projector = int(root_config.get('projector_hidden_size', 4096))
        self.vision_adapter = VisionAdapter(merged_width, projector)
        self.vision_projection = nn.Linear(projector, args.hidden_size, bias=False)
        self.image_token_id = int(root_config['image_token_id'])
        self.vision_stats = {'encoder_calls': 0, 'images': 0, 'patches': 0,
                             'feature_tokens': 0, 'ordinary_prefills': 0, 'ast_prefills': 0}

    def image_embeddings(self, inputs, *, pixel_values, image_grid_thw):
        pixel_values = mx.array(pixel_values)
        if inputs.ndim != 2 or inputs.shape[0] != 1:
            raise ValueError('Muse image prefill requires an isolated B=1 prompt')
        grid = image_grid_thw.tolist() if hasattr(image_grid_thw, 'tolist') else image_grid_thw
        merge = self.vision_tower.config.merge_size
        if not grid or any(len(row) != 3 or row[0] != 1 or any(type(n) is not int or n <= 0 for n in row)
                           or row[1] % merge or row[2] % merge for row in grid):
            raise ValueError('Muse image grid must contain unmerged still-image patch dimensions')
        patches = sum(t * h * w for t, h, w in grid)
        expected = patches // (merge * merge)
        positions = [i for i, token in enumerate(inputs[0].tolist()) if token == self.image_token_id]
        if (pixel_values.ndim != 2 or pixel_values.shape !=
            (patches, self.vision_tower.config.patch_temporal * 3 * self.vision_tower.config.patch_size ** 2)
            or len(positions) != expected):
            raise ValueError('Muse image features, patch grids and placeholder tokens differ')
        # The checkpoint's patch embedder is BF16. The projection can be Q4;
        # packed integer weight dtypes must never become pixel compute dtypes.
        dtype = self.vision_tower.ln_pre.weight.dtype
        features = self.vision_tower(pixel_values.astype(dtype), mx.array(grid, mx.int32))
        features = self.vision_projection(self.vision_adapter(features))
        features = mx.fast.rms_norm(features, None, self.args.rms_norm_eps)
        embeddings = mx.fast.rms_norm(self.model.embed_tokens(inputs), None, self.args.rms_norm_eps)
        embeddings[0, mx.array(positions, mx.int32)] = features.astype(embeddings.dtype)
        self.vision_stats['encoder_calls'] += 1
        self.vision_stats['images'] += len(grid)
        self.vision_stats['patches'] += patches
        self.vision_stats['feature_tokens'] += expected
        return embeddings

    def prefill_forward(self, inputs, *, cache=None, **media):
        if not media:
            return self(inputs, cache=cache)
        embeddings = self.image_embeddings(inputs, **media)
        hidden = select_checkpoint_rows(self.model(inputs, cache=cache, inputs_embeds=embeddings))
        logits = self.model.embed_tokens.as_linear(hidden) if self.tie_word_embeddings else self.lm_head(hidden)
        self.vision_stats['ordinary_prefills'] += 1
        cap = self.args.final_logit_softcapping
        return mx.tanh(logits * self.args.output_multiplier / cap) * cap

    def media_prefill_body(self, inputs, cache, capture_layers, **media):
        embeddings = self.image_embeddings(inputs, **media)
        features = self.forward_with_taps(inputs, cache, capture_layers,
            inputs_embeds=embeddings, body_only=True)[1]
        self.vision_stats['ast_prefills'] += 1
        return features

    def sanitize(self, weights):
        vision = {}
        for key, value in weights.items():
            normalized = key.removeprefix('model.')
            if normalized.startswith(('vision_tower.', 'vision_adapter.', 'vision_projection.')):
                vision[normalized] = value
        return {**super().sanitize(weights), **vision}
