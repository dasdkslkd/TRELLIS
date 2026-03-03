"""
Two-Stage Sparse Structure VAE Models.

Stage 1: Reconstruct occupancy field (ch 0) + one-hot categories (ch 1-7) = 8 channels
Stage 2: Reconstruct sample point coordinates (ch 8-82) = 75 channels (25 points × 3D)

The encoder is shared (takes all 83 channels), and two separate decoders handle the two stages.
Stage 2 decoder is conditioned on stage 1 output for better reconstruction.
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.norm import GroupNorm32, ChannelLayerNorm32
from ..modules.spatial import pixel_shuffle_3d
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: Literal["conv", "avgpool"] = "conv"):
        assert mode in ["conv", "avgpool"]
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: Literal["conv", "nearest"] = "conv"):
        assert mode in ["conv", "nearest"]
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)
        elif mode == "nearest":
            assert in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class SparseStructureCondDecoder(nn.Module):
    """
    Conditioned Decoder for Stage 2 of the two-stage sparse structure VAE.

    Takes the latent code z and the stage 1 decoder output as condition.
    The stage 1 output is encoded by a small convolutional network to match
    the latent spatial dimensions, then concatenated with z before decoding.

    Args:
        out_channels (int): Output channels (75 = 25 sample points × 3D).
        latent_channels (int): Channels of the latent representation.
        cond_channels (int): Channels of the stage 1 output (condition input).
        cond_latent_channels (int): Intermediate channels for encoding the condition.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks (high-to-low resolution).
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (str): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16 for the torso.
    """

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        cond_channels: int = 8,
        cond_latent_channels: int = 32,
        num_res_blocks: int = 2,
        channels: List[int] = [512, 128, 32],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.cond_channels = cond_channels
        self.cond_latent_channels = cond_latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Condition encoder: downsample stage1 output to match latent spatial dims.
        # Number of downsample levels = len(channels) - 1 (matching the encoder's structure).
        num_downsample = len(channels) - 1
        cond_layers = [
            nn.Conv3d(cond_channels, cond_latent_channels, 3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(num_downsample):
            cond_layers.extend([
                nn.Conv3d(cond_latent_channels, cond_latent_channels, 2, stride=2),
                nn.SiLU(),
            ])
        self.cond_encoder = nn.Sequential(*cond_layers)

        # Main decoder: input is latent + encoded condition
        self.input_layer = nn.Conv3d(latent_channels + cond_latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0], norm_type=norm_type)
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch, norm_type=norm_type)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_channels, h, w, d] latent code from encoder.
            cond: [B, cond_channels, H, W, D] stage 1 decoder output (activated).

        Returns:
            [B, out_channels, H, W, D] stage 2 reconstruction.
        """
        # Encode condition to match latent spatial dims
        cond_feat = self.cond_encoder(cond.float())  # [B, cond_latent_ch, h, w, d]

        # Ensure spatial dims match (safety check)
        if cond_feat.shape[2:] != z.shape[2:]:
            cond_feat = F.adaptive_avg_pool3d(cond_feat, z.shape[2:])

        # Concatenate latent and condition features
        h = torch.cat([z, cond_feat], dim=1)
        h = self.input_layer(h)

        h = h.type(self.dtype)

        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)

        h = h.type(z.dtype)
        h = self.out_layer(h)
        return h
