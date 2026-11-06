"""CPU contract checks for the same-boundary tree/chain discriminator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE = (Path(__file__).resolve().parents[1] / "qualification" / "runs" /
          "tree15-b1-discriminator-20261003" / "same_cache_chain.py")
SPEC = importlib.util.spec_from_file_location("same_cache_chain", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_accepted_record_rows_are_mapped_by_path_not_position():
    tree, chain = [], []
    for layer in range(64):
        if (layer + 1) % 4:
            fields = [np.arange(4, dtype=np.float32).reshape(1, 4, 1) for _ in range(6)]
            tree.append(("gdn", 2, np.ones((1, 1)), np.ones((1, 1)), 0, fields))
            chain.append(("gdn", 2, np.ones((1, 1)), np.ones((1, 1)), 0,
                          [part[:, [0, 2]] for part in fields]))
        else:
            plane = np.arange(4, dtype=np.float32).reshape(1, 1, 4, 1)
            tree.append(("kv", plane, plane))
            chain.append(("kv", plane[:, :, [0, 2]], plane[:, :, [0, 2]]))
    result = MODULE.compare_record_products(tree, chain, [0, 2])
    assert len(result) == 64
    assert MODULE.first_difference_layer(result, "products") is None
    chain[3] = ("kv", chain[3][1] + 1, chain[3][2])
    assert MODULE.first_difference_layer(MODULE.compare_record_products(tree, chain, [0, 2]),
                                         "products") == 3


def test_capture_geometry_and_earliest_layer():
    tree = np.zeros((1, 16, 64 * 2), dtype=np.float32)
    chain = np.zeros((1, 2, 64 * 2), dtype=np.float32)
    ordinary = np.zeros_like(chain)
    tree[0, 3, 9 * 2] = 1
    result = MODULE.compare_layer_captures(tree, chain, ordinary, [0, 3], 2)
    assert MODULE.first_difference_layer(result) == 9
    with pytest.raises(ValueError, match="geometry"):
        MODULE.compare_layer_captures(tree[:, :15], chain, ordinary, [0, 3], 2)


def test_normalized_input_row_mapping_and_projection_metrics():
    tree = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    chain = np.array([[[101.0], [102.0], [103.0], [104.0]]])
    path = [0, 1, 3, 5]
    same = MODULE.embed_accepted_rows(tree, chain, path, concatenate=np.concatenate)
    np.testing.assert_array_equal(same[:, path], chain)
    np.testing.assert_array_equal(same[:, [2, 4, 6]], tree[:, [2, 4, 6]])
    report = MODULE.compare_projection_rows(same[:, path], chain)
    assert report["bitwise_equal"] and report["first_mismatch"] is None
    assert report["nrms"] == 0
    changed = chain.copy()
    changed[0, 2, 0] += 0.03125
    report = MODULE.compare_projection_rows(changed, chain)
    assert report["first_mismatch"] == [0, 2, 0]
    assert report["max_abs_diff"] == 0.03125
    assert report["nrms"] > 0
    with pytest.raises(ValueError, match="geometry"):
        MODULE.embed_accepted_rows(tree, chain, [0, 1, 1, 5], concatenate=np.concatenate)
