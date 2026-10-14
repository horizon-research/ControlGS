from __future__ import annotations

import torch

from xrgs_utils.color_utils import scale_srgb_in_linear, srgb_to_linear


R_COEFF = 228.5245898
G_COEFF = 205.6587915
B_COEFF = 567.6371534
POWER_EPS = 1e-12


def display_power(image: torch.Tensor) -> torch.Tensor:
    """OLED display power for an RGB image in sRGB-like space.

    Values are not clipped here; callers decide where the display pipeline
    applies its final `[0, 1]` clamp.
    """
    if image.shape[-3] < 3:
        raise ValueError(f"display_power expects at least 3 channels, got shape {tuple(image.shape)}")
    linear_image = srgb_to_linear(image[..., :3, :, :])
    power = (
        B_COEFF * linear_image[..., 2, :, :].abs().mean(dim=(-2, -1))
        + G_COEFF * linear_image[..., 1, :, :].abs().mean(dim=(-2, -1))
        + R_COEFF * linear_image[..., 0, :, :].abs().mean(dim=(-2, -1))
    ) / 1000.0
    return power


def target_power_from_saving(reference_image: torch.Tensor, saving_ratio: torch.Tensor | float) -> torch.Tensor:
    saving = torch.as_tensor(saving_ratio, dtype=reference_image.dtype, device=reference_image.device)
    return display_power(reference_image) * (1.0 - saving).clamp_min(0.0)


def linear_rgb_match_to_target_power(
    image: torch.Tensor,
    reference_image: torch.Tensor,
    saving_ratio: float,
) -> torch.Tensor:
    """Apply the audited direct display mapping in linear RGB space.

    The baseline-derived target is detached, but `current_power` keeps gradients.
    This prevents the plug-in from learning to over-brighten the pre-display
    render and rely on a post-render scale to dim it afterwards.
    """
    target = target_power_from_saving(reference_image.detach(), saving_ratio).detach()
    current = display_power(image)
    scale = target / current.clamp_min(POWER_EPS)
    return scale_srgb_in_linear(image, scale)
