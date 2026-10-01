"""Grouped-input SIC forecaster.

The raw sample layout is kept compatible with :mod:`dataset`: ``[B, 15, T, H, W]``.
Inputs are first encoded by physically meaningful groups and only then fused before
the existing 3-D Swin backbone.  This avoids treating unrelated raw variables as
interchangeable channels at the first learned projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_geo_TTT_5 import SwinTransformer


MODEL_ARCHITECTURE = "grouped-v1"

# Channel positions follow dataset.VAR_KEYS, with landmask in the final channel.
ICE_CHANNELS = (0, 1)                    # siconc, sithick
OCEAN_CHANNELS = (3, 4, 5, 6)            # uo0, uo10, vo0, vo10
ATMOSPHERE_CHANNELS = (2, 7, 8, 9, 10, 11, 12, 13)  # tas and seven zg levels
LANDMASK_CHANNEL = 14


class GroupEncoder(nn.Module):
    """Encode one physical variable group without mixing it with other groups."""

    def __init__(self, in_channels: int, feature_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, feature_channels, kernel_size=1),
            nn.GroupNorm(4, feature_channels),
            nn.GELU(),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(4, feature_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GroupedInputEncoder(nn.Module):
    """Separate ice, ocean-dynamics, and atmosphere/circulation encoders."""

    def __init__(self, output_channels: int = 15, feature_channels: int = 16):
        super().__init__()
        self.ice_encoder = GroupEncoder(len(ICE_CHANNELS), feature_channels)
        self.ocean_encoder = GroupEncoder(len(OCEAN_CHANNELS), feature_channels)
        self.atmosphere_encoder = GroupEncoder(len(ATMOSPHERE_CHANNELS), feature_channels)
        # The mask is intentionally injected only at fusion, as a static boundary condition.
        self.fusion = nn.Sequential(
            nn.Conv3d(feature_channels * 3 + 1, output_channels, kernel_size=1),
            nn.GroupNorm(3, output_channels),
            nn.GELU(),
            nn.Conv3d(output_channels, output_channels, kernel_size=1),
        )

    @staticmethod
    def _select(x: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return x[:, indices, :, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W] input, got {tuple(x.shape)}")
        if x.shape[1] != LANDMASK_CHANNEL + 1:
            raise ValueError(
                f"Grouped model expects {LANDMASK_CHANNEL + 1} channels in the dataset order, "
                f"got {x.shape[1]}."
            )

        ice = self.ice_encoder(self._select(x, ICE_CHANNELS))
        ocean = self.ocean_encoder(self._select(x, OCEAN_CHANNELS))
        atmosphere = self.atmosphere_encoder(self._select(x, ATMOSPHERE_CHANNELS))
        landmask = x[:, LANDMASK_CHANNEL:LANDMASK_CHANNEL + 1]
        return self.fusion(torch.cat((ice, ocean, atmosphere, landmask), dim=1))


class GroupedSwinTransformer(nn.Module):
    """Grouped encoders followed by the established shared 3-D Swin predictor."""

    def __init__(self, in_chans: int = 15, feature_channels: int = 16):
        super().__init__()
        if in_chans != LANDMASK_CHANNEL + 1:
            raise ValueError(
                f"GroupedSwinTransformer requires the 15-channel dataset layout, got {in_chans}."
            )
        self.input_encoder = GroupedInputEncoder(in_chans, feature_channels)
        self.backbone = SwinTransformer(
            in_chans=in_chans,
            patch_size=(1, 2, 2),
            stride=(1, 2, 2),
            embed_dim=108,
            depths=(2, 6, 4),
            num_heads=(3, 6, 12),
            window_sizes=[(3, 5, 5), (3, 5, 5), (3, 5, 5)],
            shift_size_tuples=[(1, 2, 2), (1, 2, 2), (1, 2, 2)],
            merge_sizes=[(2, 3, 3), (1, 1, 2)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.input_encoder(x))


def model(**kwargs) -> GroupedSwinTransformer:
    return GroupedSwinTransformer(
        in_chans=kwargs.pop("in_chans", LANDMASK_CHANNEL + 1),
        feature_channels=kwargs.pop("feature_channels", 16),
    )
