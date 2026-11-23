"""aMoEba gate features on a host whose trunk forward is ``hidden`` and whose layers take hyper-connection streams
(Hy4-flash: the inner module is not callable; layer input is ``[B, T, hc_mult, D]``)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

mx.set_default_device(mx.cpu)

from mlx2.experimental.amoeba import child_probe as CP  # noqa: E402


class Layer(nn.Module):
    def __call__(self, streams, *args):
        return streams + 1.0


class Inner(nn.Module):              # holds the layers but has no forward (as HY4FlashModel's trunk)
    def __init__(self):
        super().__init__()
        self.layers = [Layer() for _ in range(3)]


class Host:
    def __init__(self, n):
        self.inner = Inner()
        self.layers = self.inner.layers
        self.n = n

    def hidden(self, ids):
        x = ids.astype(mx.float32)[..., None] * mx.arange(1, 5, dtype=mx.float32)   # [B, T, D=4]
        s = mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], self.n, x.shape[-1])) if self.n else x
        if self.n:
            s = s + mx.arange(self.n, dtype=mx.float32)[None, None, :, None]   # streams differ
        for lay in self.layers:
            s = lay(s, None)
        return s


def test_streams_are_averaged_and_hidden_is_the_trunk():
    host = Host(4)
    H = CP.layer_input(host, [[1, 2, 3]], 1)
    assert H.shape == (1, 3, 4)
    # layer 1 input = layer 0 output = base + stream offset + 1; mean of offsets 0..3 is 1.5
    want = np.array([[t * np.arange(1, 5) + 1.5 + 1 for t in (1, 2, 3)]], dtype=np.float32)
    np.testing.assert_allclose(H, want)
    assert type(host.layers[1]) is Layer          # capture class restored


def test_plain_residual_unchanged():
    H = CP.layer_input(Host(0), [[2, 5]], 2)
    want = np.array([[t * np.arange(1, 5) + 2 for t in (2, 5)]], dtype=np.float32)
    np.testing.assert_allclose(H, want)
