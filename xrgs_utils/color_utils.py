from __future__ import annotations

import torch


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    safe = torch.clamp(image, min=0.04045)
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        torch.pow((safe + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    safe = torch.clamp(image, min=0.0031308)
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * torch.pow(safe, 1.0 / 2.4) - 0.055,
    )


def scale_srgb_in_linear(image: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    scale_tensor = torch.as_tensor(scale, dtype=image.dtype, device=image.device)
    while scale_tensor.ndim < image.ndim:
        scale_tensor = scale_tensor.unsqueeze(-1)
    return linear_to_srgb(srgb_to_linear(image) * scale_tensor)


def relative_luminance_rec709(image: torch.Tensor) -> torch.Tensor:
    """Rec.709/sRGB relative luminance Y from sRGB-like RGB tensors."""
    if image.shape[-3] < 3:
        raise ValueError(f"relative_luminance_rec709 expects at least 3 channels, got shape {tuple(image.shape)}")
    linear = srgb_to_linear(image[..., :3, :, :])
    return 0.2126 * linear[..., 0, :, :] + 0.7152 * linear[..., 1, :, :] + 0.0722 * linear[..., 2, :, :]


def luminance_max_match_scale(image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
    image_max = relative_luminance_rec709(image).amax(dim=(-2, -1))
    reference_max = relative_luminance_rec709(reference_image).amax(dim=(-2, -1))
    return (reference_max / image_max.clamp_min(1e-6)).detach()
