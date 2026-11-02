"""Decoder allowlist and identity for images that arrive from outside mlx2.

``Image.open`` tries every registered Pillow plugin until one accepts the
header, so a format check after opening still runs the TIFF, BMP, MIC, PDF,
EPS or Tar parser on hostile bytes.  Passing ``formats`` keeps those plugins
out of reach.  The Pillow 12.3 advisories fixed only in 13.0 are concentrated
in those decoders; GIF is excluded too (quadratic comment concatenation).
PNG, JPEG and WebP advisories remain until the Pillow 13 requalification.

Pillow's resampling output also changes between releases (12.3 -> 13.0
bicubic/Lanczos/bilinear differ by up to 6/255), so the installed version is
part of the serving runtime identity.
"""

from __future__ import annotations

import importlib.metadata

UNTRUSTED_IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")


def pillow_version() -> str:
    try:
        return importlib.metadata.version("pillow")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
