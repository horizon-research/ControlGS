from __future__ import annotations

from contextlib import contextmanager
import logging
import math
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from utils.image_utils import psnr
    from utils.loss_utils import l1_loss, ssim
except ModuleNotFoundError:
    def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        return 20 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1e-12)))

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

from xrgs_utils.display_power import POWER_EPS, display_power, target_power_from_saving
from xrgs_utils.lens_utils import DistortMLP, apply_optics_model

from .brightness import apply_brightness_policy, compute_fixed_brightness_scale
from .conditions import set_unified_conditions
from .display_protocol import PostProcessing, PostProcessingResult, crop_lens_metric_region, lens_gt_from_display_canvas


EVAL_MODE_OURS = "evaluate_ours"
EVAL_MODE_DISTORTION_MLP = "evaluate_distortion_mlp"


def load_lpips_model(args, logger: logging.Logger, *, required: bool = False):
    if not args.eval_lpips and not required:
        logger.info("LPIPS evaluation disabled")
        return None
    model = lpips.LPIPS(net=args.lpips_net).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    logger.info("Loaded LPIPS model: %s", args.lpips_net)
    return model


def lpips_distance(lpips_model, image: torch.Tensor, gt: torch.Tensor) -> float:
    return float(lpips_model(image.unsqueeze(0), gt.unsqueeze(0)).detach())


def _subtract_optional_metric(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return left - right


def _make_metric_store() -> dict[str, list[float]]:
    return {"PSNR": [], "SSIM": [], "L1": [], "LPIPS": [], "POWER": [], "TARGET_POWER": []}


def _add_metrics(
    store: dict[str, list[float]],
    image_eval: torch.Tensor,
    gt_eval: torch.Tensor,
    image_display: torch.Tensor,
    target_power_value: torch.Tensor,
    lpips_model,
) -> None:
    store["PSNR"].append(float(psnr(image_eval.unsqueeze(0), gt_eval.unsqueeze(0)).detach()))
    store["SSIM"].append(float(ssim(image_eval.unsqueeze(0), gt_eval.unsqueeze(0)).detach()))
    store["L1"].append(float(l1_loss(image_eval, gt_eval).detach()))
    if lpips_model is not None:
        store["LPIPS"].append(lpips_distance(lpips_model, image_eval, gt_eval))
    store["POWER"].append(float(display_power(image_display).detach()))
    store["TARGET_POWER"].append(float(target_power_value.detach()))


def _summarize(store: dict[str, list[float]]) -> dict[str, float | None]:
    return {key: (float(np.mean(values)) if values else None) for key, values in store.items()}


def _condition_key(saving: float, downscale: float) -> str:
    return f"saving_{saving:g}_downscale_{downscale:g}"


def _folder_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _condition_image_dir(root: Path, saving: float, downscale: float) -> Path:
    return root / f"downscale_{_folder_float(downscale)}x__saving_{_folder_float(saving)}"


def _inference_condition_dir(root: Path, saving: float, downscale: float) -> Path:
    return root / f"display_resolution_{_folder_float(downscale)}x" / f"display_power_saving_{_folder_float(saving)}"


def _save_tensor_image(path: Path, image: torch.Tensor, bit_depth: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(bit_depth) == 16:
        array = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(65535.0).round().to(torch.uint16).cpu().numpy()
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Saving 16-bit XRGS eval images requires OpenCV") from exc
        if array.shape[-1] == 3:
            array = array[:, :, ::-1]
        if not cv2.imwrite(str(path), array):
            raise RuntimeError(f"OpenCV failed to save image: {path}")
        return
    if int(bit_depth) != 8:
        raise ValueError(f"Unsupported eval_image_bit_depth={bit_depth}; expected 8 or 16")
    array = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving 8-bit XRGS eval images requires Pillow") from exc
    Image.fromarray(array).save(path)


def _maybe_save_eval_images(
    output_dir: Path | None,
    view_index: int,
    image_eval: torch.Tensor,
    gt_eval: torch.Tensor,
    render_raw: torch.Tensor | None = None,
    args=None,
) -> None:
    if output_dir is None:
        return
    bit_depth = int(getattr(args, "eval_image_bit_depth", 8)) if args is not None else 8
    save_aux = bool(getattr(args, "eval_image_save_aux", True)) if args is not None else True
    _save_tensor_image(output_dir / "Rendered" / f"view_{view_index:04d}.png", image_eval, bit_depth=bit_depth)
    _save_tensor_image(output_dir / "GroundTruth" / f"view_{view_index:04d}.png", gt_eval, bit_depth=bit_depth)
    if save_aux:
        _save_tensor_image(output_dir / "Error_Map" / f"view_{view_index:04d}.png", (image_eval - gt_eval).abs(), bit_depth=bit_depth)
        if render_raw is not None:
            _save_tensor_image(output_dir / "render_raw" / f"view_{view_index:04d}.png", render_raw, bit_depth=bit_depth)


def _resolve_eval_image_root(args, mode: str) -> Path | None:
    image_mode = getattr(args, "eval_image_mode", "off")
    if image_mode == "off":
        return None
    if mode == EVAL_MODE_DISTORTION_MLP and image_mode != "baseline":
        return None
    if mode == EVAL_MODE_OURS and image_mode != "ours":
        return None
    configured = getattr(args, "eval_image_dir", "")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(getattr(args, "output_path", ".")).expanduser().resolve()
    folder = "Baseline_Evaluation_Images" if mode == EVAL_MODE_DISTORTION_MLP else "Ours_Evaluation_Images"
    return base / folder


def _should_save_view(args, view_index: int) -> bool:
    max_views = int(getattr(args, "eval_image_max_views", -1))
    return max_views < 0 or view_index < max_views


def _effective_render_resolution_policy(args) -> tuple[bool, str]:
    policy = getattr(args, "render_resolution_policy", "super_sampling")
    use_super_sampling = bool(getattr(args, "use_super_sampling", True))
    if policy == "direct_downscale_no_ss":
        use_super_sampling = False
    elif not use_super_sampling:
        policy = "direct_downscale_no_ss"
    return use_super_sampling, policy


def _render_resolution_metadata(args, downscale: float) -> dict:
    use_super_sampling, policy = _effective_render_resolution_policy(args)
    direct_lowres_render = (not use_super_sampling) and float(downscale) > 1.000001
    return {
        "use_super_sampling": use_super_sampling,
        "policy": policy,
        "direct_lowres_render": direct_lowres_render,
        "render_upsample_mode": None,
        "post_distortion_canvas_resize": "nearest_to_reference_canvas" if direct_lowres_render else None,
        "postprocess_downscale": 1.0 if direct_lowres_render else float(downscale),
    }


@contextmanager
def _temporary_camera_render_size(camera, target_width: int, target_height: int, downscale: float):
    saved_values = []

    def set_temporary_attr(name: str, value) -> None:
        targets = [camera]
        sample = getattr(camera, "sample", None)
        if sample is not None:
            targets.append(sample)
        for target in targets:
            if not hasattr(target, name):
                continue
            try:
                old_value = getattr(target, name)
                setattr(target, name, value)
            except AttributeError:
                continue
            saved_values.append((target, name, old_value))
            return
        raise AttributeError(f"{type(camera).__name__} has no writable {name!r} render-size field")

    native_width = max(int(getattr(camera, "image_width")), 1)
    native_height = max(int(getattr(camera, "image_height")), 1)

    set_temporary_attr("image_width", int(target_width))
    set_temporary_attr("image_height", int(target_height))

    optional_values = {
        "focal_x": lambda value: float(value) * float(target_width) / float(native_width),
        "focal_y": lambda value: float(value) * float(target_height) / float(native_height),
        "resolution_scale": lambda value: float(value) * float(downscale),
        "resolution_downscale": lambda value: float(value) * float(downscale),
    }
    for name, transform in optional_values.items():
        if not hasattr(camera, name):
            continue
        value = getattr(camera, name)
        try:
            setattr(camera, name, transform(value))
        except AttributeError:
            continue
        saved_values.append((camera, name, value))

    try:
        yield
    finally:
        for target, name, value in reversed(saved_values):
            setattr(target, name, value)


def _render_display_resolution(
    render_fn,
    camera,
    gaussians,
    pipe,
    background: torch.Tensor,
    gt_image: torch.Tensor,
    downscale: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    use_super_sampling, _policy = _effective_render_resolution_policy(args)
    if use_super_sampling or float(downscale) <= 1.000001:
        image = render_fn(camera, gaussians, pipe, background)
        return image, image, False

    native_h, native_w = gt_image.shape[-2:]
    target_h = max(1, int(round(native_h / float(downscale))))
    target_w = max(1, int(round(native_w / float(downscale))))
    with _temporary_camera_render_size(camera, target_w, target_h, downscale):
        render_raw = render_fn(camera, gaussians, pipe, background)
    return render_raw, render_raw, True


def _post_process_display(
    adapter,
    camera,
    gaussians,
    pipe,
    background: torch.Tensor,
    gt_image: torch.Tensor,
    downscale: float,
    saving: float,
    args,
    distort_mlp: DistortMLP | None,
    reference_distort_mlp: DistortMLP | None = None,
    display_scale: torch.Tensor | None = None,
    use_baseline_render: bool = False,
    use_gt_reference: bool = False,
) -> PostProcessingResult:
    if use_gt_reference:
        image = gt_image
        render_raw = image
        postprocess_downscale = downscale
    else:
        render_fn = adapter.render_baseline if use_baseline_render else adapter.render
        image, render_raw, direct_display_render = _render_display_resolution(
            render_fn,
            camera,
            gaussians,
            pipe,
            background,
            gt_image,
            downscale,
            args,
        )
        postprocess_downscale = 1.0 if direct_display_render else downscale
    post = PostProcessing(
        image=image,
        reference_image=gt_image,
        gt_image=gt_image,
        downscale=postprocess_downscale,
        saving=saving,
        args=args,
        distort_mlp=distort_mlp,
        reference_distort_mlp=reference_distort_mlp,
        display_scale=display_scale,
        pad_value=0.0,
    )
    return post._replace(render_raw=render_raw)



def _average_display_power(
    adapter,
    cameras,
    gaussians,
    pipe,
    background: torch.Tensor,
    downscale: float,
    saving: float,
    args,
    distort_mlp: DistortMLP | None,
    desc: str,
    collect_target_power: bool = False,
    use_baseline_render: bool = False,
    use_gt_reference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    powers = []
    target_powers = []
    for camera in tqdm(cameras, total=len(cameras), desc=desc):
        set_unified_conditions(camera, downscale, saving, args)
        gt_image = camera.original_image[:3, :, :].cuda().clamp(0.0, 1.0)
        post = _post_process_display(
            adapter,
            camera,
            gaussians,
            pipe,
            background,
            gt_image,
            downscale,
            saving,
            args,
            distort_mlp,
            use_baseline_render=use_baseline_render,
            use_gt_reference=use_gt_reference,
        )
        powers.append(display_power(post.display_canvas).detach())
        if collect_target_power:
            target_powers.append(target_power_from_saving(post.reference_display_canvas, saving).detach())
    mean_power = torch.stack(powers).mean()
    mean_target_power = torch.stack(target_powers).mean() if target_powers else None
    return mean_power, mean_target_power


def _global_power_scale(mean_power: torch.Tensor, target_power: torch.Tensor) -> torch.Tensor:
    return (target_power / mean_power.clamp_min(POWER_EPS)).detach()


def _evaluate_display_tensor(
    image_display: torch.Tensor,
    gt_canvas: torch.Tensor,
    downscale: float,
    optics_wrapper,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    gt_lens = lens_gt_from_display_canvas(gt_canvas, args)
    image_lens = apply_optics_model(
        image_display,
        optics_wrapper,
        mode=args.eval_lens_mode,
        pad_value=0.0,
    )
    image_lens = apply_brightness_policy(image_lens, gt_lens, args)
    image_eval, gt_eval, metric_crop_box = crop_lens_metric_region(
        image_lens.clamp(0.0, 1.0),
        gt_lens.clamp(0.0, 1.0),
        gt_canvas,
        downscale,
    )
    return image_eval, gt_eval, image_lens, metric_crop_box


def _camera_split(scene, requested_split: str):
    if requested_split == "train":
        cameras = scene.getTrainCameras()
        split = "train"
    elif requested_split == "test":
        cameras = scene.getTestCameras()
        split = "test"
        if not cameras:
            cameras = scene.getTrainCameras()
            split = "train"
    else:
        raise ValueError(f"Unsupported render split: {requested_split}")
    return cameras, split


def _save_inference_images(
    render_dir: Path,
    e2e_dir: Path,
    view_index: int,
    image_display: torch.Tensor,
    render_raw: torch.Tensor,
    reference_display_canvas: torch.Tensor,
    image_eval: torch.Tensor,
    gt_eval: torch.Tensor,
    save_gt: bool,
) -> None:
    filename = f"view_{view_index:04d}.png"
    _save_tensor_image(render_dir / filename, image_display)
    _save_tensor_image(render_dir / "render_raw" / filename, render_raw)
    _save_tensor_image(e2e_dir / filename, image_eval)
    if save_gt:
        _save_tensor_image(render_dir / "gt_reference" / filename, reference_display_canvas)
        _save_tensor_image(e2e_dir / "gt" / filename, gt_eval)
        _save_tensor_image(e2e_dir / "error_map" / filename, (image_eval - gt_eval).abs())


def render_inference_grid(
    adapter,
    scene,
    gaussians,
    pipe,
    optics_wrapper,
    distort_mlp: DistortMLP | None,
    background: torch.Tensor,
    args,
    lpips_model=None,
    mode: str = EVAL_MODE_OURS,
    canonical_distort_mlp: DistortMLP | None = None,
    render_output_root: Path | str | None = None,
    e2e_output_root: Path | str | None = None,
    split: str = "test",
    max_views: int = 0,
    write_eval: bool = False,
    save_gt: bool = True,
) -> dict:
    if mode not in {EVAL_MODE_OURS, EVAL_MODE_DISTORTION_MLP}:
        raise ValueError(f"Unsupported render_inference_grid mode: {mode}")
    if args.eval_power_mode not in {"global", "per_image"}:
        raise ValueError(f"Unsupported eval_power_mode: {args.eval_power_mode}")
    if render_output_root is None or e2e_output_root is None:
        raise ValueError("render_inference_grid requires render_output_root and e2e_output_root")
    use_global_power = args.eval_power_mode == "global"
    use_baseline_render = mode == EVAL_MODE_DISTORTION_MLP
    render_output_root = Path(render_output_root)
    e2e_output_root = Path(e2e_output_root)
    if hasattr(gaussians, "eval"):
        gaussians.eval()
    reference_distort_mlp = canonical_distort_mlp if canonical_distort_mlp is not None else distort_mlp
    reference_distort_mlp_role = "canonical_distort_mlp" if canonical_distort_mlp is not None else "method_distort_mlp"
    canonical_distort_mlp_path = getattr(args, "canonical_distort_mlp_path", "")
    had_previous_brightness_scale = hasattr(args, "fixed_brightness_scale")
    previous_brightness_scale = getattr(args, "fixed_brightness_scale", None)
    if args.brightness_match_mode == "fixed_train0":
        eval_brightness_scale = compute_fixed_brightness_scale(
            scene,
            optics_wrapper,
            args,
            args.eval_lens_mode,
        )
        args.fixed_brightness_scale = eval_brightness_scale
        args.eval_fixed_brightness_scale_value = float(eval_brightness_scale.detach().cpu())
        args.fixed_brightness_scale_history.append(
            {
                "reason": "render_inference",
                "lens_mode": args.eval_lens_mode,
                "scale": args.eval_fixed_brightness_scale_value,
            }
        )

    requested_splits = ["train", "test"] if split == "all" else [split]
    results: dict[str, dict] = {}
    with torch.no_grad():
        seen_splits = set()
        for requested_split in requested_splits:
            cameras, actual_split = _camera_split(scene, requested_split)
            if actual_split in seen_splits:
                continue
            seen_splits.add(actual_split)
            if max_views > 0:
                cameras = cameras[:max_views]
            if not cameras:
                raise ValueError(f"No cameras available for render split {requested_split!r}")
            split_render_root = render_output_root / actual_split if split == "all" else render_output_root
            split_e2e_root = e2e_output_root / actual_split if split == "all" else e2e_output_root
            for saving in args.eval_savings:
                for downscale in args.eval_downscales:
                    key = _condition_key(saving, downscale)
                    if split == "all":
                        key = f"{actual_split}_{key}"
                    metric_store = _make_metric_store()
                    target_reference_power = None
                    target_avg_power = None
                    render_power = None
                    display_scale = None
                    if use_global_power:
                        target_reference_power, target_avg_power = _average_display_power(
                            adapter,
                            cameras,
                            gaussians,
                            pipe,
                            background,
                            downscale,
                            saving,
                            args,
                            reference_distort_mlp,
                            f"Infer target power {actual_split} {key}",
                            collect_target_power=True,
                            use_gt_reference=True,
                        )
                        if target_avg_power is None:
                            raise ValueError("Target average power was not computed")
                        render_power, _ = _average_display_power(
                            adapter,
                            cameras,
                            gaussians,
                            pipe,
                            background,
                            downscale,
                            saving,
                            args,
                            distort_mlp,
                            f"Infer render power {actual_split} {key}",
                            use_baseline_render=use_baseline_render,
                        )
                        display_scale = _global_power_scale(render_power, target_avg_power)

                    render_dir = _inference_condition_dir(split_render_root, saving, downscale)
                    e2e_dir = _inference_condition_dir(split_e2e_root, saving, downscale)
                    metric_crop_box = None
                    for view_index, camera in enumerate(tqdm(cameras, total=len(cameras), desc=f"Infer {actual_split} {key}")):
                        set_unified_conditions(camera, downscale, saving, args)
                        gt_image = camera.original_image[:3, :, :].cuda().clamp(0.0, 1.0)
                        if use_global_power and display_scale is None:
                            raise ValueError("global render display scale was not computed")
                        post = _post_process_display(
                            adapter,
                            camera,
                            gaussians,
                            pipe,
                            background,
                            gt_image,
                            downscale,
                            saving,
                            args,
                            distort_mlp,
                            reference_distort_mlp=reference_distort_mlp,
                            display_scale=display_scale if use_global_power else None,
                            use_baseline_render=use_baseline_render,
                        )
                        if use_global_power:
                            if target_avg_power is None:
                                raise ValueError("target_avg_power was not computed")
                            target_power_value = target_avg_power
                        else:
                            target_power_value = target_power_from_saving(post.reference_display_canvas, saving).detach()
                        image_display = post.lens_input
                        gt_canvas = post.gt_canvas
                        image_eval, gt_eval, _image_lens, metric_crop_box = _evaluate_display_tensor(
                            image_display,
                            gt_canvas,
                            downscale,
                            optics_wrapper,
                            args,
                        )
                        _save_inference_images(
                            render_dir,
                            e2e_dir,
                            view_index,
                            image_display,
                            post.render_raw,
                            post.reference_display_canvas,
                            image_eval,
                            gt_eval,
                            save_gt,
                        )
                        if write_eval:
                            _add_metrics(metric_store, image_eval, gt_eval, image_display, target_power_value, lpips_model)

                    result = {
                        "mode": mode,
                        "split": actual_split,
                        "saving": saving,
                        "downscale": downscale,
                        "render_resolution": _render_resolution_metadata(args, downscale),
                        "views": len(cameras),
                        "render_output_dir": str(render_dir),
                        "e2e_output_dir": str(e2e_dir),
                        "write_eval": bool(write_eval),
                        "power_protocol": {
                            "mode": args.eval_power_mode,
                            "global_power_prepass": use_global_power,
                            "reference_source": "gt_distorted_display_canvas",
                            "reference_distort_mlp": reference_distort_mlp_role,
                            "canonical_distort_mlp_path": canonical_distort_mlp_path or None,
                        },
                        "metric_crop": metric_crop_box,
                    }
                    if use_global_power:
                        result["power_protocol"].update(
                            {
                                "target_avg_power": float(target_avg_power.detach()),
                                "reference_unscaled_avg_power": float(target_reference_power.detach()),
                                "render_unscaled_avg_power": float(render_power.detach()),
                                "render_display_scale": float(display_scale.detach()),
                            }
                        )
                    else:
                        result["power_protocol"]["target_power_source"] = "per_image_gt_reference"
                    if write_eval:
                        result["baseline_pipeline" if mode == EVAL_MODE_DISTORTION_MLP else "ours_unified"] = _summarize(metric_store)
                    results[key] = result

    if had_previous_brightness_scale:
        args.fixed_brightness_scale = previous_brightness_scale
    elif hasattr(args, "fixed_brightness_scale"):
        delattr(args, "fixed_brightness_scale")
    if hasattr(gaussians, "train"):
        gaussians.train()
    return results


def evaluate_grid(
    adapter,
    scene,
    gaussians,
    base_scene,
    base_gaussians,
    pipe,
    optics_wrapper,
    distort_mlp: DistortMLP | None,
    background: torch.Tensor,
    args,
    lpips_model=None,
    mode: str = EVAL_MODE_OURS,
    canonical_distort_mlp: DistortMLP | None = None,
) -> dict:
    if mode not in {EVAL_MODE_OURS, EVAL_MODE_DISTORTION_MLP}:
        raise ValueError(f"Unsupported evaluate_grid mode: {mode}")
    if args.eval_power_mode not in {"global", "per_image"}:
        raise ValueError(f"Unsupported eval_power_mode: {args.eval_power_mode}")
    use_global_power = args.eval_power_mode == "global"
    if hasattr(gaussians, "eval"):
        gaussians.eval()
    if hasattr(base_gaussians, "eval"):
        base_gaussians.eval()
    reference_distort_mlp = canonical_distort_mlp if canonical_distort_mlp is not None else distort_mlp
    reference_distort_mlp_role = "canonical_distort_mlp" if canonical_distort_mlp is not None else "method_distort_mlp"
    canonical_distort_mlp_path = getattr(args, "canonical_distort_mlp_path", "")
    had_previous_brightness_scale = hasattr(args, "fixed_brightness_scale")
    previous_brightness_scale = getattr(args, "fixed_brightness_scale", None)
    if args.brightness_match_mode == "fixed_train0":
        eval_brightness_scale = compute_fixed_brightness_scale(
            scene,
            optics_wrapper,
            args,
            args.eval_lens_mode,
        )
        args.fixed_brightness_scale = eval_brightness_scale
        args.eval_fixed_brightness_scale_value = float(eval_brightness_scale.detach().cpu())
        args.fixed_brightness_scale_history.append(
            {
                "reason": "eval",
                "lens_mode": args.eval_lens_mode,
                "scale": args.eval_fixed_brightness_scale_value,
            }
        )
    cameras = scene.getTestCameras()
    base_cameras = base_scene.getTestCameras()
    split = "test"
    if not cameras:
        cameras = scene.getTrainCameras()
        base_cameras = base_scene.getTrainCameras()
        split = "train"
    if args.eval_max_views > 0:
        cameras = cameras[: args.eval_max_views]
        base_cameras = base_cameras[: args.eval_max_views]
    results: dict[str, dict] = {}
    image_root = _resolve_eval_image_root(args, mode)

    with torch.no_grad():
        for saving in args.eval_savings:
            for downscale in args.eval_downscales:
                key = _condition_key(saving, downscale)
                ours_metrics = _make_metric_store()
                baseline_metrics = _make_metric_store()
                target_reference_power = None
                baseline_power = None
                target_avg_power = None
                baseline_display_scale = None
                ours_display_scale = None
                ours_power = None

                if use_global_power:
                    target_reference_power, target_avg_power = _average_display_power(
                        adapter,
                        cameras,
                        gaussians,
                        pipe,
                        background,
                        downscale,
                        saving,
                        args,
                        reference_distort_mlp,
                        f"Eval target power {key}",
                        collect_target_power=True,
                        use_gt_reference=True,
                    )
                    if target_avg_power is None:
                        raise ValueError("Target average power was not computed")

                if use_global_power and mode == EVAL_MODE_DISTORTION_MLP:
                    baseline_power, _ = _average_display_power(
                        adapter,
                        base_cameras,
                        base_gaussians,
                        pipe,
                        background,
                        downscale,
                        saving,
                        args,
                        distort_mlp,
                        f"Eval baseline power {key}",
                        use_baseline_render=True,
                    )
                    baseline_display_scale = _global_power_scale(baseline_power, target_avg_power)

                if use_global_power and mode == EVAL_MODE_OURS:
                    ours_power, _ = _average_display_power(
                        adapter,
                        cameras,
                        gaussians,
                        pipe,
                        background,
                        downscale,
                        saving,
                        args,
                        distort_mlp,
                        f"Eval ours power {key}",
                        use_baseline_render=False,
                    )
                    ours_display_scale = _global_power_scale(ours_power, target_avg_power)

                condition_image_dir = _condition_image_dir(image_root, saving, downscale) if image_root is not None else None
                metric_crop_box = None
                for view_index, (camera, base_camera) in enumerate(
                    tqdm(zip(cameras, base_cameras), total=len(cameras), desc=f"Eval {key}")
                ):
                    set_unified_conditions(camera, downscale, saving, args)
                    gt_image = camera.original_image[:3, :, :].cuda().clamp(0.0, 1.0)

                    if use_global_power:
                        if target_avg_power is None:
                            raise ValueError("target_avg_power was not computed")
                        target_power_value = target_avg_power
                    if mode == EVAL_MODE_DISTORTION_MLP:
                        if use_global_power and baseline_display_scale is None:
                            raise ValueError("baseline_display_scale was not computed")
                        base_post = _post_process_display(
                            adapter,
                            base_camera,
                            base_gaussians,
                            pipe,
                            background,
                            gt_image,
                            downscale,
                            saving,
                            args,
                            distort_mlp,
                            reference_distort_mlp=reference_distort_mlp,
                            display_scale=baseline_display_scale if use_global_power else None,
                            use_baseline_render=True,
                        )
                        if not use_global_power:
                            target_power_value = target_power_from_saving(base_post.reference_display_canvas, saving).detach()
                        base_display = base_post.lens_input
                        base_eval, gt_eval, _base_lens, metric_crop_box = _evaluate_display_tensor(
                            base_display,
                            base_post.gt_canvas,
                            downscale,
                            optics_wrapper,
                            args,
                        )
                        _add_metrics(baseline_metrics, base_eval, gt_eval, base_display, target_power_value, lpips_model)
                        if _should_save_view(args, view_index):
                            _maybe_save_eval_images(condition_image_dir, view_index, base_eval, gt_eval, base_post.render_raw, args=args)

                    if mode == EVAL_MODE_OURS:
                        if use_global_power and ours_display_scale is None:
                            raise ValueError("ours_display_scale was not computed")
                        post = _post_process_display(
                            adapter,
                            camera,
                            gaussians,
                            pipe,
                            background,
                            gt_image,
                            downscale,
                            saving,
                            args,
                            distort_mlp,
                            reference_distort_mlp=reference_distort_mlp,
                            display_scale=ours_display_scale if use_global_power else None,
                            use_baseline_render=False,
                        )
                        if not use_global_power:
                            target_power_value = target_power_from_saving(post.reference_display_canvas, saving).detach()
                        image_display = post.lens_input
                        image_eval, gt_eval, _image_lens, metric_crop_box = _evaluate_display_tensor(
                            image_display,
                            post.gt_canvas,
                            downscale,
                            optics_wrapper,
                            args,
                        )
                        _add_metrics(ours_metrics, image_eval, gt_eval, image_display, target_power_value, lpips_model)
                        if _should_save_view(args, view_index):
                            _maybe_save_eval_images(condition_image_dir, view_index, image_eval, gt_eval, post.render_raw, args=args)

                result = {
                    "mode": mode,
                    "split": split,
                    "saving": saving,
                    "downscale": downscale,
                    "render_resolution": _render_resolution_metadata(args, downscale),
                    "power_protocol": {
                        "mode": args.eval_power_mode,
                        "global_power_prepass": use_global_power,
                        "reference_source": "gt_distorted_display_canvas",
                        "reference_distort_mlp": reference_distort_mlp_role,
                        "canonical_distort_mlp_path": canonical_distort_mlp_path or None,
                    },
                    "metric_crop": metric_crop_box,
                    "views": len(cameras),
                }
                if use_global_power:
                    result["power_protocol"].update(
                        {
                            "target_avg_power": float(target_avg_power.detach()),
                            "reference_unscaled_avg_power": float(target_reference_power.detach()),
                        }
                    )
                    if mode == EVAL_MODE_DISTORTION_MLP:
                        result["power_protocol"].update(
                            {
                                "baseline_unscaled_avg_power": float(baseline_power.detach()),
                                "baseline_display_scale": float(baseline_display_scale.detach()),
                            }
                        )
                else:
                    result["power_protocol"]["target_power_source"] = "per_image_gt_reference"
                if mode == EVAL_MODE_OURS:
                    ours_summary = _summarize(ours_metrics)
                    result["ours_unified"] = ours_summary
                    if use_global_power:
                        result["power_protocol"]["ours_unscaled_avg_power"] = float(ours_power.detach())
                        result["power_protocol"]["ours_display_scale"] = float(ours_display_scale.detach())
                else:
                    baseline_summary = _summarize(baseline_metrics)
                    result["baseline_pipeline"] = baseline_summary
                results[key] = result
    if had_previous_brightness_scale:
        args.fixed_brightness_scale = previous_brightness_scale
    elif hasattr(args, "fixed_brightness_scale"):
        delattr(args, "fixed_brightness_scale")
    if hasattr(gaussians, "train"):
        gaussians.train()
    return results
