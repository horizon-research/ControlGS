from __future__ import annotations

import logging
import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from utils.loss_utils import l1_loss, ssim
except ModuleNotFoundError:
    def l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.abs(network_output - gt).mean()

    def _gaussian_window(window_size: int, channel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        values = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
        gauss = torch.exp(-(values**2) / (2 * 1.5**2))
        gauss = gauss / gauss.sum()
        window_2d = (gauss[:, None] @ gauss[None, :]).unsqueeze(0).unsqueeze(0)
        return window_2d.expand(channel, 1, window_size, window_size).contiguous()

    def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
        channel = img1.size(-3)
        window = _gaussian_window(window_size, channel, img1.dtype, img1.device)
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        c1 = 0.01**2
        c2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        if size_average:
            return ssim_map.mean()
        return ssim_map.mean(1).mean(1).mean(1)

from xrgs_utils.lens_utils import DistortMLP

from .layout import layout_to_metadata


UNIFIED_PLUGIN_VERSION = 1
SCAFFOLD_DISTORT_MLP_PROTOCOL = {
    "distort_mlp_lr": 0.003,
    "finetune_iterations": 10000,
    "lens_psf_iterations": 1000,
    "lambda_dssim": 0.2,
}


def scaffold_distort_mlp_protocol_metadata(args=None, *, fixed_in_adapter: bool | None = None) -> dict:
    metadata = {
        "shared_with_scaffold_gs": True,
        "standard": dict(SCAFFOLD_DISTORT_MLP_PROTOCOL),
    }
    if args is not None:
        actual = {
            "distort_mlp_lr": float(getattr(args, "distort_mlp_lr", SCAFFOLD_DISTORT_MLP_PROTOCOL["distort_mlp_lr"])),
            "finetune_iterations": int(getattr(args, "finetune_iterations")),
            "lens_psf_iterations": int(getattr(args, "lens_psf_iterations")),
            "lambda_dssim": float(getattr(args, "lambda_dssim", SCAFFOLD_DISTORT_MLP_PROTOCOL["lambda_dssim"])),
        }
        metadata["actual"] = actual
        metadata["matches_standard"] = _distort_mlp_protocol_matches_standard(actual)
    if fixed_in_adapter is not None:
        metadata["fixed_in_adapter"] = bool(fixed_in_adapter)
    return metadata


def _distort_mlp_protocol_matches_standard(actual: dict) -> bool:
    return (
        abs(float(actual["distort_mlp_lr"]) - SCAFFOLD_DISTORT_MLP_PROTOCOL["distort_mlp_lr"]) < 1e-12
        and int(actual["finetune_iterations"]) == SCAFFOLD_DISTORT_MLP_PROTOCOL["finetune_iterations"]
        and int(actual["lens_psf_iterations"]) == SCAFFOLD_DISTORT_MLP_PROTOCOL["lens_psf_iterations"]
        and abs(float(actual["lambda_dssim"]) - SCAFFOLD_DISTORT_MLP_PROTOCOL["lambda_dssim"]) < 1e-12
    )


def loaded_distort_mlp_protocol_metadata(
    distort_mlp_path: str,
    logger: logging.Logger,
    *,
    expected_backbone: str | None = None,
) -> dict:
    result_path = Path(distort_mlp_path).resolve().parent / "results.json"
    metadata = scaffold_distort_mlp_protocol_metadata(fixed_in_adapter=True)
    metadata["result_path"] = str(result_path)
    metadata["source"] = "missing_results_json"
    if not result_path.exists():
        raise FileNotFoundError(f"Cannot verify DistortMLP baseline protocol because {result_path} is missing")

    result = json.loads(result_path.read_text())
    protocol = result.get("distort_mlp_protocol", {})
    actual = protocol.get("actual")
    if actual is None:
        actual = {
            "distort_mlp_lr": result.get("distort_mlp_lr"),
            "finetune_iterations": result.get("finetune_iterations"),
            "lens_psf_iterations": result.get("lens_psf_iterations"),
            "lambda_dssim": result.get("lambda_dssim"),
        }
    if actual.get("lambda_dssim") is None:
        # Legacy baseline results created before this field was recorded used
        # every backbone's native default, which matches Scaffold-GS at 0.2.
        actual["lambda_dssim"] = SCAFFOLD_DISTORT_MLP_PROTOCOL["lambda_dssim"]
    if any(value is None for value in actual.values()):
        raise ValueError(f"Cannot parse DistortMLP baseline protocol from {result_path}")

    actual = {
        "distort_mlp_lr": float(actual["distort_mlp_lr"]),
        "finetune_iterations": int(actual["finetune_iterations"]),
        "lens_psf_iterations": int(actual["lens_psf_iterations"]),
        "lambda_dssim": float(actual["lambda_dssim"]),
    }
    metadata.update(
        {
            "actual": actual,
            "matches_standard": _distort_mlp_protocol_matches_standard(actual),
            "source": "results_json",
            "baseline_backbone": result.get("backbone"),
            "baseline_mode": result.get("mode"),
        }
    )
    if metadata["baseline_mode"] != "baseline_distort_mlp":
        raise ValueError(
            "Loaded DistortMLP must come from a baseline_distort_mlp result, "
            f"got mode={metadata['baseline_mode']!r} from {result_path}"
        )
    if expected_backbone is not None and metadata["baseline_backbone"] != expected_backbone:
        raise ValueError(
            "Loaded DistortMLP backbone does not match the active adapter: "
            f"expected {expected_backbone!r}, got {metadata['baseline_backbone']!r} from {result_path}"
        )
    if not metadata["matches_standard"]:
        raise ValueError(
            "Loaded DistortMLP does not match the shared Scaffold-GS baseline protocol: "
            f"{actual} from {result_path}"
        )
    logger.info("Verified fixed DistortMLP baseline protocol from %s: %s", result_path, actual)
    return metadata


def get_logger(path: Path) -> logging.Logger:
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"unified_plugins:{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(path / "unified_plugins.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def lens_quality_loss(image_lens: torch.Tensor, gt_lens: torch.Tensor, lambda_dssim: float):
    ll1 = l1_loss(image_lens, gt_lens)
    mse_loss = F.mse_loss(image_lens, gt_lens)
    ssim_loss = 1.0 - ssim(image_lens, gt_lens)
    return (1.0 - lambda_dssim) * ll1 + lambda_dssim * ssim_loss, ll1, mse_loss, ssim_loss


def psf_start_iteration(args) -> int | None:
    psf_iterations = max(0, int(getattr(args, "lens_psf_iterations", 0)))
    if psf_iterations <= 0:
        return None
    return max(1, args.finetune_iterations - psf_iterations + 1)


def training_lens_mode(args, iteration: int) -> str:
    psf_start = psf_start_iteration(args)
    if psf_start is None:
        return args.lens_mode
    if iteration >= psf_start:
        return "psf"
    return args.lens_mode


def lens_mode_schedule_summary(args) -> str:
    psf_iterations = max(0, int(getattr(args, "lens_psf_iterations", 0)))
    if psf_iterations <= 0:
        return args.lens_mode
    psf_iterations = min(psf_iterations, args.finetune_iterations)
    first_iterations = args.finetune_iterations - psf_iterations
    return f"{args.lens_mode} x {first_iterations}, psf x {psf_iterations}"


def load_distort_mlp_checkpoint(
    path: str,
    logger: logging.Logger,
    *,
    required: bool = False,
) -> DistortMLP | None:
    if not path:
        if required:
            raise ValueError(
                "Unified training requires a per-scene baseline DistortMLP checkpoint. "
                "Train it first with scripts/exp/baseline_distort_mlp/train.sh <backbone> "
                "or set lens.distort_mlp_path in the unified config."
            )
        logger.info("No DistortMLP checkpoint provided; lens pipeline uses optics only")
        return None
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DistortMLP checkpoint not found: {checkpoint_path}")
    distort_mlp = DistortMLP().cuda()
    distort_mlp.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
    distort_mlp.eval()
    for parameter in distort_mlp.parameters():
        parameter.requires_grad_(False)
    trainable = sum(parameter.numel() for parameter in distort_mlp.parameters() if parameter.requires_grad)
    if trainable:
        raise RuntimeError(f"Loaded DistortMLP must be frozen for unified training, got {trainable} trainable parameters")
    logger.info("Loaded fixed DistortMLP checkpoint from %s", checkpoint_path)
    return distort_mlp


def build_plugin_metadata(
    args,
    backbone_name: str,
    layout: dict[str, tuple[int, int]],
    total_dim: int,
    inject_dims: dict[str, int],
    base_model_path: str,
    loaded_iter: int,
    save_iteration: int,
    disabled: set[str],
) -> dict:
    enabled = [name for name in layout if name not in disabled]
    return {
        "unified_plugin_version": UNIFIED_PLUGIN_VERSION,
        "backbone": backbone_name,
        "base_model_path": str(base_model_path),
        "base_loaded_iteration": loaded_iter,
        "save_iteration": save_iteration,
        "layout": layout_to_metadata(layout),
        "total_dim": int(total_dim),
        "inject_dims": {
            "requested": int(args.inject_dim) if int(args.inject_dim) > 0 else None,
            "actual": {name: int(dim) for name, dim in inject_dims.items()},
        },
        "init_mode": args.plugin_init_mode,
        "init_mode_normalized": args.plugin_init_mode_normalized,
        "config_source": args.config,
        "projection_init": {
            "type": "trunc_normal",
            "std": float(args.projection_init_std),
            "bias": False,
        },
        "modules": {
            "enabled": enabled,
            "disabled": sorted(disabled),
        },
        "condition_strength": {
            "max_ss_downsample_scale": float(args.ss_scale_max),
            "max_target_saving_power_setting": float(args.target_saving_max),
            "lens_radius_max_norm": float(args.lens_condition_radius_max_norm),
        },
        "display_mapping": {
            "mode": "post_distortion_direct_linear_rgb_target_power",
            "target_reference": "gt_distorted_image",
            "lens_resolution_protocol": "downsample_resampled_to_display_then_padded_lens",
            "ss_downsample_mode": args.ss_downsample_mode,
            "canvas_placement": "full_extent_resample",
        },
        "baseline_pipeline": {
            "definition": "frozen MLP 3DGS renderer plus per-scene end-to-end baseline DistortMLP",
            "distort_mlp_path": str(args.distort_mlp_path),
            "distort_mlp_trainable_in_unified": False,
            "protocol": getattr(
                args,
                "baseline_distort_mlp_protocol",
                scaffold_distort_mlp_protocol_metadata(args, fixed_in_adapter=True),
            ),
        },
        "optics": {
            "psf_backend": args.psf_backend,
            "lens_pad_ratio": float(args.lens_pad_ratio),
        },
        "brightness_match": {
            "mode": args.brightness_match_mode,
            "fixed_scale": None if getattr(args, "fixed_brightness_scale_value", None) is None else float(args.fixed_brightness_scale_value),
            "history": getattr(args, "fixed_brightness_scale_history", []),
        },
        "metrics": {
            "eval_lpips": bool(args.eval_lpips),
            "lpips_net": args.lpips_net,
        },
        "training": {
            "condition_lr": float(args.res_cond_lr),
            "mse_weight": float(args.mse_loss_weight),
            "finetune_iterations": int(args.finetune_iterations),
            "lens_train_mode_schedule": lens_mode_schedule_summary(args),
        },
    }
