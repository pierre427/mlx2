"""Malformed EXIF is invalid client media, not a server fault.

Pillow's TIFF/EXIF parser signals a bad header with ``SyntaxError`` (its
internal "not my format" signal), which ``ImageOps.exif_transpose`` lets
escape.  ``decode_image`` only converted ``OSError``/``ValueError``, so a PNG
or WebP carrying a corrupt eXIf chunk reached the HTTP handler's catch-all and
was answered 500 "internal server error" instead of 400 (sweep 2026-10-08).
"""

import base64
from io import BytesIO

import pytest
from PIL import Image

from mlx2.multimodal import decode_image, resolve_media

BAD_EXIF = b"Exif\x00\x00MM\xf6*\x00\x00\x00\x08"


def _encoded(fmt, **save):
    payload = BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(payload, format=fmt, **save)
    return payload.getvalue()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("WEBP", "image/webp")])
def test_malformed_exif_is_rejected_as_invalid_media(fmt, mime):
    payload = _encoded(fmt, exif=BAD_EXIF)
    with pytest.raises(ValueError, match="failed to decode image"):
        decode_image(payload, mime)
    source = f"data:{mime};base64," + base64.b64encode(payload).decode()
    with pytest.raises(ValueError, match="failed to decode image"):
        resolve_media({"url": source}, kind="image")


def test_well_formed_exif_orientation_still_applies():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    payload = BytesIO()
    Image.new("RGB", (4, 6)).save(payload, format="PNG", exif=exif.tobytes())
    media = decode_image(payload.getvalue(), "image/png")
    assert media.metadata == {"width": 6, "height": 4}
