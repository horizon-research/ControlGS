from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from xrgs_utils.display_power import linear_rgb_match_to_target_power
from xrgs_utils.color_utils import scale_srgb_in_linear
from xrgs_utils.lens_utils import apply_lens_distortion, lens_pad_pixels_for_size, prepare_lens_gt


class PostProcessingResult(NamedTuple):
    render_raw: torch.Tensor
    lens_input: torch.Tensor
    gt_canvas: torch.Tensor
    distorted_native: torch.Tensor
    reference_distorted_native: torch.Tensor
    display_canvas: torch.Tensor
    reference_display_canvas: torch.Tensor


def apply_display_mapping(image: torch.Tensor, reference_image: torch.Tensor, saving: float) -> torch.Tensor:
    """Direct linear-RGB target-power matching against a no-condition distorted render."""
    return linear_rgb_match_to_target_power(image, reference_image, saving).clamp(0.0, 1.0)


def downsample_image(image: torch.Tensor, downscale: float, mode: str = "area") -> torch.Tensor:
    if downscale <= 1.000001:
        return image
    if mode != "area":
        raise ValueError(f"XRGS downsample protocol uses average-pooling area mode, got {mode}")
    height, width = image.shape[-2:]
    target_h = max(1, int(round(height / downscale)))
    target_w = max(1, int(round(width / downscale)))
    return F.interpolate(image.unsqueeze(0), size=(target_h, target_w), mode="area").squeeze(0)


def resample_to_display_resolution(
    image: torch.Tensor,
    downscale: float,
    downsample_mode: str = "area",
) -> torch.Tensor:
    """Apply low-resolution sampling while keeping the original display extent."""
    if downscale <= 1.000001:
        return image
    height, width = image.shape[-2:]
    low = downsample_image(image, downscale, downsample_mode)
    return F.interpolate(low.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)


def resize_to_reference_canvas(image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
    if image.shape[-2:] == reference_image.shape[-2:]:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=reference_image.shape[-2:],
        mode="nearest",
    ).squeeze(0)


def display_canvas_pair(
    image: torch.Tensor,
    gt_image: torch.Tensor,
    downscale: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the physical display canvas used by SS, display power, and lens target.

    The rendered/reference image uses area downsample plus nearest upsample.
    GT stays at native resolution by default, but can follow the same protocol
    via args.resample_gt_display_canvas=True.
    """
    downsample_mode = getattr(args, "ss_downsample_mode", "area")
    resample_gt = bool(getattr(args, "resample_gt_display_canvas", False))
    image_canvas = resample_to_display_resolution(image, downscale, downsample_mode)

    if resample_gt:
        gt_canvas = resample_to_display_resolution(gt_image, downscale, downsample_mode)
    else:
        gt_canvas = gt_image
    gt_canvas = gt_canvas.clamp(0.0, 1.0)
    return image_canvas, gt_canvas


def PostProcessing(
    image: torch.Tensor,
    reference_image: torch.Tensor,
    gt_image: torch.Tensor,
    downscale: float,
    saving: float,
    args,
    distort_mlp: torch.nn.Module | None = None,
    reference_distort_mlp: torch.nn.Module | None = None,
    display_scale: torch.Tensor | float | None = None,
    pad_value: float = 0.0,
) -> PostProcessingResult:
    """Apply the proposed display post-processing order.

    Order:
    1. lens distortion at native render/reference resolution
    2. display-canvas downsample followed by nearest upsample
    3. display mapping against the processed reference image, or a precomputed
       global display scale when display_scale is provided

    This wrapper is the shared protocol boundary used by unified training,
    evaluation, and inference rendering.
    """
    downsample_mode = getattr(args, "ss_downsample_mode", "area")
    resample_gt = bool(getattr(args, "resample_gt_display_canvas", False))
    if reference_distort_mlp is None:
        reference_distort_mlp = distort_mlp

    distorted_native = apply_lens_distortion(image, distort_mlp=distort_mlp, pad_value=pad_value)
    reference_distorted_native = apply_lens_distortion(
        reference_image,
        distort_mlp=reference_distort_mlp,
        pad_value=pad_value,
    )
    display_canvas = resample_to_display_resolution(distorted_native, downscale, downsample_mode)
    reference_display_canvas = resample_to_display_resolution(reference_distorted_native, downscale, downsample_mode)
    # Low-resolution no-SS renders already carry the target display sampling.
    # Resize them to the reference/simulation canvas without applying a second
    # display box filter.
    if distorted_native.shape[-2:] != reference_distorted_native.shape[-2:]:
        display_canvas = resize_to_reference_canvas(display_canvas, reference_display_canvas)
    if display_scale is None:
        lens_input = apply_display_mapping(display_canvas, reference_display_canvas, saving)
    else:
        lens_input = scale_srgb_in_linear(display_canvas, display_scale).clamp(0.0, 1.0)

    if resample_gt:
        gt_canvas = resample_to_display_resolution(gt_image, downscale, downsample_mode)
    else:
        gt_canvas = gt_image
    gt_canvas = gt_canvas.clamp(0.0, 1.0)

    return PostProcessingResult(
        render_raw=image,
        lens_input=lens_input,
        gt_canvas=gt_canvas,
        distorted_native=distorted_native,
        reference_distorted_native=reference_distorted_native,
        display_canvas=display_canvas,
        reference_display_canvas=reference_display_canvas,
    )


def resolve_lens_pad_pixels(image: torch.Tensor, args) -> int:
    return lens_pad_pixels_for_size(
        max(image.shape[-2:]),
        pad_ratio=float(args.lens_pad_ratio),
    )


def lens_gt_from_display_canvas(gt_canvas: torch.Tensor, args) -> torch.Tensor:
    pad_pixels = resolve_lens_pad_pixels(gt_canvas, args)
    return prepare_lens_gt(gt_canvas, pad_pixels=pad_pixels).clamp(0.0, 1.0)


def center_crop_box(source_hw: tuple[int, int], crop_hw: tuple[int, int]) -> dict[str, int]:
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    crop_h, crop_w = int(crop_hw[0]), int(crop_hw[1])
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Invalid crop size: {crop_hw}")
    if crop_h > source_h or crop_w > source_w:
        raise ValueError(f"Crop size {crop_hw} exceeds source size {source_hw}")
    top = (source_h - crop_h) // 2
    left = (source_w - crop_w) // 2
    return {"top": top, "left": left, "height": crop_h, "width": crop_w}


def crop_with_box(image: torch.Tensor, box: dict[str, int]) -> torch.Tensor:
    top = int(box["top"])
    left = int(box["left"])
    height = int(box["height"])
    width = int(box["width"])
    return image[..., top : top + height, left : left + width]


def lens_metric_crop_box(gt_canvas: torch.Tensor, gt_lens: torch.Tensor, downscale: float) -> dict[str, int]:
    """Crop optics metrics to the valid original display region.

    Downsample simulation is upsampled back to the original display resolution
    before display mapping and optics. Metrics and training loss therefore keep
    the full original display extent and only remove lens/output padding plus
    any square padding around non-square images.
    """
    _ = downscale
    return center_crop_box(gt_lens.shape[-2:], gt_canvas.shape[-2:])


def crop_lens_metric_region(
    image_lens: torch.Tensor,
    gt_lens: torch.Tensor,
    gt_canvas: torch.Tensor,
    downscale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    box = lens_metric_crop_box(gt_canvas, gt_lens, downscale)
    return crop_with_box(image_lens, box), crop_with_box(gt_lens, box), box
