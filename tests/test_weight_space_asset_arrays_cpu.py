"""Reject allocation-bomb NPY headers before np.load, not after allocation."""

import io
import zipfile

import numpy as np
import pytest

from mlx2.experimental.weight_space.asset_arrays import bounded_npz_arrays


def archive(name, data):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as z:
        z.writestr(name, data)
    return result.getvalue()


def test_numeric_arrays_roundtrip_including_fortran_and_boolean_masks():
    raw = io.BytesIO()
    values = np.asfortranarray(np.arange(12).reshape(3, 4))
    np.savez_compressed(raw, values=values, mask=np.array([True, False]))
    loaded = bounded_npz_arrays(raw.getvalue())
    np.testing.assert_array_equal(loaded["values"], values)
    np.testing.assert_array_equal(loaded["mask"], [True, False])


def test_huge_declared_shape_is_refused_before_numpy_load(monkeypatch):
    value = io.BytesIO()
    np.lib.format.write_array_header_1_0(value, {"descr": "<f8", "fortran_order": False,
                                              "shape": (1_000_000_000,)})
    value.write(b"12345678")
    raw = archive("payload.npy", value.getvalue())

    def forbidden(*args, **kwargs):
        raise AssertionError("np.load reached before header bounds")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="declared geometry"):
        bounded_npz_arrays(raw)


@pytest.mark.parametrize("array", [np.array([object()], dtype=object),
                                  np.array([(1,)], dtype=[("field", "i8")]),
                                  np.array(["text"]), np.array([1j])])
def test_object_structured_and_non_numeric_payloads_are_refused(array):
    value = io.BytesIO()
    np.save(value, array)
    with pytest.raises(ValueError, match="plain numeric"):
        bounded_npz_arrays(archive("payload.npy", value.getvalue()))


def test_truncated_payload_and_duplicate_or_non_npy_entries_are_refused():
    value = io.BytesIO()
    np.save(value, np.eye(4))
    with pytest.raises(ValueError, match="declared geometry"):
        bounded_npz_arrays(archive("payload.npy", value.getvalue()[:-1]))
    with pytest.raises(ValueError, match="NPY arrays"):
        bounded_npz_arrays(archive("readme.txt", b"hello"))
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as z:
        z.writestr("same.npy", value.getvalue())
        with pytest.warns(UserWarning):
            z.writestr("same.npy", value.getvalue())
    with pytest.raises(ValueError, match="repeats"):
        bounded_npz_arrays(result.getvalue())
