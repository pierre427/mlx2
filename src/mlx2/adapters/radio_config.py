"""CPU-only configuration gate for the standalone C-RADIO image encoder."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RadioConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int
    max_resolution: int
    num_cls_tokens: int
    num_registers: int
    summary_idxs: tuple[int, ...]
    preferred_resolution: tuple[int, int]
    align_corners: bool = True

    @classmethod
    def from_dict(cls, config):
        args = config.get("args")
        if not isinstance(args, dict):
            raise TypeError("RADIO requires a published recipe config (args)")
        architectures = {
            "vit_base_patch16_224": (768, 12, 12),
            "vit_large_patch16_224": (1024, 24, 16),
            "vit_huge_patch16_224": (1280, 32, 16),
        }
        architecture = args.get("model")
        if architecture not in architectures:
            raise ValueError(f"unsupported RADIO architecture {architecture!r}")
        if args.get("model_norm") or args.get("normalize_patches"):
            raise ValueError("RADIO normalization variant is unsupported")
        if config.get("feature_normalizer_config") or config.get(
            "inter_feature_normalizer_config"
        ):
            raise ValueError("RADIO feature normalizers are unsupported")
        if (
            config.get("vitdet_window_size")
            or args.get("qk_norm")
            or args.get("layer_scale_init_value")
        ):
            raise ValueError(
                "RADIO windowed attention, QK norm or layer scale is unsupported"
            )
        if config.get("adaptor_names"):
            raise ValueError("teacher-specific RADIO adaptors are unsupported")
        teachers = args.get("teachers", [])
        per_teacher = args.get("cls_token_per_teacher", True)
        count = len({t["name"] for t in teachers}) if per_teacher and teachers else 1
        summary = tuple(i for i, t in enumerate(teachers) if t.get("use_summary", True))
        if not per_teacher or not teachers:
            summary = (0,)
        if not summary or any(i >= count for i in summary):
            raise ValueError("unsupported RADIO teacher summary layout")
        registers = args.get("cpe_num_registers")
        multiple = args.get("register_multiple")
        if not registers:
            registers = multiple - count % multiple if multiple else 0
        if not isinstance(registers, int) or registers < 0:
            raise ValueError("invalid RADIO register count")
        patch = config.get("patch_size", 16)
        maximum = args.get("cpe_max_size") or config.get("max_resolution", 2048)
        if (
            patch != 16
            or not isinstance(maximum, int)
            or maximum < patch
            or maximum % patch
        ):
            raise ValueError("unsupported RADIO patch geometry")
        preferred = tuple(config.get("preferred_resolution", [768, 768]))
        if len(preferred) != 2 or any(
            not isinstance(v, int) or v < patch or v > maximum or v % patch
            for v in preferred
        ):
            raise ValueError("invalid preferred RADIO resolution")
        width, depth, heads = architectures[architecture]
        return cls(
            width,
            depth,
            heads,
            width * 4,
            patch,
            maximum,
            count,
            registers,
            summary,
            preferred,
            not str(config.get("version", ""))
            .removeprefix("c-")
            .startswith("radio_v4"),
        )

    @property
    def num_skip(self):
        return self.num_cls_tokens + self.num_registers
