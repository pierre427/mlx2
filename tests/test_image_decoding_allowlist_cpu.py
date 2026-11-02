"""Untrusted image bytes reach only the PNG, JPEG and WebP Pillow decoders.

A format check after ``Image.open`` still runs whichever plugin claimed the
header; the Pillow 12.3 advisories fixed only in 13.0 sit mostly in the TIFF,
BMP, GIF, MIC, PDF and Tar decoders.  Every external entry point passes
``formats`` so those plugins are never tried, and the serving runtime identity
records the Pillow version because its resampling output changes between
releases.
"""

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from mlx2.conversion.calibration_corpus import image_geometry
from mlx2.desktop_world import Worlds
from mlx2.image_decoding import UNTRUSTED_IMAGE_FORMATS, pillow_version
from mlx2.multimodal import decode_image, resolve_media

REFUSED = ["TIFF", "BMP", "GIF", "PPM", "ICO"]


def _encoded(fmt, size=(32, 32)):
    stream = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(stream, format=fmt)
    return stream.getvalue()


def _opened_formats(monkeypatch):
    """Record which plugin openers Pillow tries, without changing them."""
    tried = []
    for fmt, (factory, accept) in list(Image.OPEN.items()):
        def traced(fp, filename=None, *, _fmt=fmt, _factory=factory):
            tried.append(_fmt)
            return _factory(fp, filename)
        monkeypatch.setitem(Image.OPEN, fmt, (traced, accept))
    return tried


@pytest.mark.parametrize("fmt", UNTRUSTED_IMAGE_FORMATS)
def test_allowed_formats_still_decode(fmt):
    media = decode_image(_encoded(fmt), f"image/{fmt.lower()}")
    assert media.metadata == {"width": 32, "height": 32}


@pytest.mark.parametrize("fmt", REFUSED)
def test_media_refuses_other_formats_without_running_their_decoder(fmt, monkeypatch):
    Image.init()
    payload = _encoded(fmt)
    tried = _opened_formats(monkeypatch)
    with pytest.raises(ValueError, match="failed to decode image"):
        decode_image(payload, "image/png")
    source = "data:image/png;base64," + base64.b64encode(payload).decode()
    with pytest.raises(ValueError, match="failed to decode image"):
        resolve_media({"url": source}, kind="image")
    assert set(tried) <= set(UNTRUSTED_IMAGE_FORMATS)


@pytest.mark.parametrize("fmt", ["TIFF", "BMP"])
def test_world_texture_refuses_other_formats(tmp_path, fmt):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    with pytest.raises(ValueError, match="valid bounded image"):
        worlds.store_image(_encoded(fmt))
    assert worlds.store_image(_encoded("PNG"))


@pytest.mark.parametrize("fmt", ["TIFF", "BMP"])
def test_calibration_corpus_refuses_other_formats(fmt):
    with pytest.raises(OSError):
        image_geometry(_encoded(fmt))
    assert image_geometry(_encoded("JPEG"))["format"] == "JPEG"


def test_runtime_identity_binds_the_pillow_version():
    from mlx2.serving import runtime_identity

    assert pillow_version() == Image.__version__
    assert runtime_identity()["pillow"] == Image.__version__
