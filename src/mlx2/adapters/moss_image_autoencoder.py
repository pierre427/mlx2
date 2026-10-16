"""Native MLX FLUX.2-small VAE used by MOSS-Image."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


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
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x):
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        return (self.conv_shortcut(x) if "conv_shortcut" in self else x) + h


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = normalize(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x):
        residual = x
        batch, height, width, channels = x.shape
        h = self.group_norm(x).reshape(batch, height * width, channels)
        query, key, value = self.to_q(h), self.to_k(h), self.to_v(h)
        h = mx.fast.scaled_dot_product_attention(
            query[:, None], key[:, None], value[:, None], scale=channels**-0.5
        )[:, 0]
        return residual + self.to_out[0](h).reshape(batch, height, width, channels)


class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resnets = [
            ResnetBlock(channels, channels),
            ResnetBlock(channels, channels),
        ]
        self.attentions = [Attention(channels)]

    def __call__(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, count, downsample):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(count)
        ]
        self.downsamplers = [Downsample(out_channels)] if downsample else []

    def __call__(self, x):
        for block in self.resnets:
            x = block(x)
        return self.downsamplers[0](x) if self.downsamplers else x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, count, upsample):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(count)
        ]
        self.upsamplers = [Upsample(out_channels)] if upsample else []

    def __call__(self, x):
        for block in self.resnets:
            x = block(x)
        return self.upsamplers[0](x) if self.upsamplers else x


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_in = nn.Conv2d(3, channels[0], 3, padding=1)
        self.down_blocks = []
        current = channels[0]
        for index, out in enumerate(channels):
            self.down_blocks.append(
                DownBlock(current, out, 2, index < len(channels) - 1)
            )
            current = out
        self.mid_block = MidBlock(current)
        self.conv_norm_out = normalize(current)
        self.conv_out = nn.Conv2d(current, 64, 3, padding=1)


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        current = channels[-1]
        self.conv_in = nn.Conv2d(32, current, 3, padding=1)
        self.mid_block = MidBlock(current)
        self.up_blocks = []
        for index, out in enumerate(reversed(channels)):
            self.up_blocks.append(UpBlock(current, out, 3, index < len(channels) - 1))
            current = out
        self.conv_norm_out = normalize(current)
        self.conv_out = nn.Conv2d(current, 3, 3, padding=1)

    def __call__(self, x):
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class LatentStats(nn.Module):
    def __init__(self):
        super().__init__()
        self.running_mean = mx.zeros((128,))
        self.running_var = mx.ones((128,))
        self.num_batches_tracked = mx.array(0, dtype=mx.int64)


class MOSSImageAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_channels = tuple(config["block_out_channels"])
        decoder_channels = tuple(config["decoder_block_out_channels"])
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.quant_conv = nn.Conv2d(64, 64, 1)
        self.post_quant_conv = nn.Conv2d(32, 32, 1)
        self.bn = LatentStats()
        self.eps = float(config.get("batch_norm_eps", 1e-4))

    def decode(self, latents):
        """Decode NHWC [1,H,W,128] patch-packed, normalized latents."""
        z = (
            latents.astype(mx.float32) * mx.sqrt(self.bn.running_var + self.eps)
            + self.bn.running_mean
        )
        batch, height, width, channels = z.shape
        if channels != 128:
            raise ValueError("MOSS-Image VAE expects 128 packed latent channels")
        z = z.reshape(batch, height, width, 32, 2, 2).transpose(0, 1, 4, 2, 5, 3)
        z = z.reshape(batch, height * 2, width * 2, 32)
        return self.decoder(self.post_quant_conv(z))


def convert_vae_weights(weights):
    """Convert PyTorch convolution OIHW tensors to MLX OHWI layout."""
    return {
        name: (value.transpose(0, 2, 3, 1) if value.ndim == 4 else value).astype(
            mx.int64 if name.endswith("num_batches_tracked") else mx.float32
        )
        for name, value in weights.items()
    }
