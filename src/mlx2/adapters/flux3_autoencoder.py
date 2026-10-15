"""FLUX 3 image VAE in MLX NHWC layout; see provenance/flux3-image-klein.json."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .flux3_transformer import latent_ids


def normalize(channels):
    return nn.GroupNorm(32, channels, eps=1e-6, pytorch_compatible=True)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x):
        return self.conv(mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2))


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def __call__(self, x):
        return self.conv(mx.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0))))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x):
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        return (self.nin_shortcut(x) if "nin_shortcut" in self else x) + h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = normalize(channels)
        for name in ("q", "k", "v", "proj_out"):
            setattr(self, name, nn.Conv2d(channels, channels, 1))

    def __call__(self, x):
        h = self.norm(x)
        b, height, width, channels = h.shape

        def flat(t):
            return t.reshape(b, 1, height * width, channels)

        h = mx.fast.scaled_dot_product_attention(
            flat(self.q(h)),
            flat(self.k(h)),
            flat(self.v(h)),
            scale=channels**-0.5,
        ).reshape(b, height, width, channels)
        return x + self.proj_out(h)


class Mid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def __call__(self, x):
        return self.block_2(self.attn_1(self.block_1(x)))


class Level(nn.Module):
    def __init__(self, in_channels, out_channels, count, resample=None):
        super().__init__()
        self.block = [
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(count)
        ]
        if resample == "up":
            self.upsample = Upsample(out_channels)
        elif resample == "down":
            self.downsample = Downsample(out_channels)

    def __call__(self, x):
        for block in self.block:
            x = block(x)
        if "upsample" in self:
            x = self.upsample(x)
        if "downsample" in self:
            x = self.downsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_conv = nn.Conv2d(64, 64, 1)
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)
        self.down = []
        channels = 128
        for i, out in enumerate((128, 256, 512, 512)):
            self.down.append(Level(channels, out, 2, "down" if i < 3 else None))
            channels = out
        self.mid = Mid(512)
        self.norm_out = normalize(512)
        self.conv_out = nn.Conv2d(512, 64, 3, padding=1)

    def __call__(self, x):
        h = self.conv_in(x)
        for level in self.down:
            h = level(h)
        return self.quant_conv(self.conv_out(nn.silu(self.norm_out(self.mid(h)))))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(32, 32, 1)
        self.conv_in = nn.Conv2d(32, 512, 3, padding=1)
        self.mid = Mid(512)
        self.up = []
        channels = 512
        for i in reversed(range(4)):
            out = (128, 256, 512, 512)[i]
            self.up.insert(0, Level(channels, out, 3, "up" if i else None))
            channels = out
        self.norm_out = normalize(128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def __call__(self, z):
        h = self.mid(self.conv_in(self.post_quant_conv(z)))
        for level in reversed(self.up):
            h = level(h)
        return self.conv_out(nn.silu(self.norm_out(h)))


class LatentStats(nn.Module):
    def __init__(self):
        super().__init__()
        self.running_mean = mx.zeros((128,))
        self.running_var = mx.ones((128,))
        self.num_batches_tracked = mx.array(0, dtype=mx.int64)


class Flux3AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bn = LatentStats()

    def encode(self, image):
        # image is HWC float32 in [-1, 1]. Patch order is C, patch_y, patch_x.
        mean, _ = mx.split(self.encoder(image[None]), 2, axis=-1)
        _, height, width, _ = mean.shape
        h, w = height // 2, width // 2
        z = (
            mean.reshape(1, h, 2, w, 2, 32)
            .transpose(0, 1, 3, 5, 2, 4)
            .reshape(h * w, 128)
        )
        z = (z - self.bn.running_mean) * mx.rsqrt(self.bn.running_var + 1e-4)
        return z, latent_ids(h, w)

    def decode(self, tokens, height, width):
        z = (
            tokens.astype(mx.float32) * mx.sqrt(self.bn.running_var + 1e-4)
            + self.bn.running_mean
        )
        z = z.reshape(1, height, width, 32, 2, 2).transpose(0, 1, 4, 2, 5, 3)
        z = z.reshape(1, 2 * height, 2 * width, 32)
        return self.decoder(z)[0]


def convert_vae_weights(weights):
    """Torch OIHW -> MLX OHWI; all VAE computations stay in float32."""
    return {
        name: (value.transpose(0, 2, 3, 1) if value.ndim == 4 else value).astype(
            mx.int64 if name.endswith("num_batches_tracked") else mx.float32
        )
        for name, value in weights.items()
    }
