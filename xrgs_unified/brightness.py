from __future__ import annotations

import logging

import torch

from xrgs_utils.color_utils import luminance_max_match_scale, scale_srgb_in_linear
from xrgs_utils.lens_utils import apply_optics_model, match_lens_brightness

from .display_protocol import display_canvas_pair, lens_gt_from_display_canvas


def apply_brightness_policy(image_lens: torch.Tensor, gt_lens: torch.Tensor, args) -> torch.Tensor:
    if args.brightness_match_mode == "off":
        return image_lens
    if args.brightness_match_mode == "per_image":
        return match_lens_brightness(image_lens, gt_lens)
    if args.brightness_match_mode == "fixed_train0":
        scale = getattr(args, "fixed_brightness_scale", None)
        if scale is None:
            raise ValueError("brightness_match_mode=fixed_train0 requires args.fixed_brightness_scale")
        return scale_srgb_in_linear(image_lens, scale)
    raise ValueError(f"Unsupported brightness match mode: {args.brightness_match_mode}")


def compute_fixed_brightness_scale(
    scene,
    optics_wrapper,
    args,
    lens_mode: str,
) -> torch.Tensor:
    """Compute one fixed optics brightness scale from train frame 0."""
    camera = scene.getTrainCameras()[0]
    with torch.no_grad():
        gt_image = camera.original_image[:3, :, :].cuda().clamp(0.0, 1.0)
        lens_input, gt_canvas = display_canvas_pair(gt_image, gt_image, 1.0, args)

        gt_lens = lens_gt_from_display_canvas(gt_canvas, args)
        image_lens = apply_optics_model(
            lens_input,
            optics_wrapper,
            mode=lens_mode,
            pad_value=0.0,
        )
        scale = luminance_max_match_scale(image_lens, gt_lens)
    return scale.detach()


def set_fixed_brightness_scale(
    scene,
    optics_wrapper,
    args,
    lens_mode: str,
    logger: logging.Logger,
    reason: str,
) -> torch.Tensor:
    scale = compute_fixed_brightness_scale(
        scene,
        optics_wrapper,
        args,
        lens_mode,
    )
    args.fixed_brightness_scale = scale
    args.fixed_brightness_scale_value = float(scale.detach().cpu())
    args.fixed_brightness_scale_history.append(
        {
            "reason": reason,
            "lens_mode": lens_mode,
            "scale": args.fixed_brightness_scale_value,
        }
    )
    logger.info("Fixed train-frame brightness scale (%s, mode=%s): %.6f", reason, lens_mode, args.fixed_brightness_scale_value)
    return scale
