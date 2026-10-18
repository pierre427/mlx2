"""Text-only Nemotron 3.5 Bolt composition over the shared Nemotron-H math.

Bolt is a Puzzle-NAS pruning of the Super language stack; the per-layer
routed-expert counts and widths come from ``block_configs`` and are applied
by ``nemotron_h._moe_layer_args``.  The converted artifact keeps the RADIO
tensors, but this composition does not load them: images fail closed in the
adapter until a vision slice is brought up.  Provenance:
provenance/nemotron35-bolt.json.
"""

from ..prefill_checkpoint_rows import select_checkpoint_rows
from ..progressive_external_verify import PROGRESSIVE_TARGET_PROTOCOL_VERSION
from .nemotron_h import Model as LanguageModel

VISION_PREFIXES = ("vision_model.", "mlp1.")


class Model(LanguageModel):
    # Same B1 speculative protocol as Super: ordinary S=1 recurrence and
    # attention with exact hybrid record/replay; not a fused row kernel.
    progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION

    def prefill_forward(self, inputs, cache=None, *, inputs_embeds=None, image_pixels=None):
        if image_pixels is not None:
            raise ValueError("Nemotron 3.5 Bolt serving is text-only")
        hidden = self.backbone(inputs, cache=cache, inputs_embeds=inputs_embeds)
        return self.lm_head(select_checkpoint_rows(hidden))

    def sanitize(self, weights):
        output = {}
        for name, value in weights.items():
            if name.startswith(VISION_PREFIXES):
                continue
            output[name.removeprefix("language_model.")] = value
        return super().sanitize(output)
