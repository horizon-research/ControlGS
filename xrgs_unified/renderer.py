from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import torch

XRGS_ROOT = Path(__file__).resolve().parents[1]
if str(XRGS_ROOT) not in sys.path:
    sys.path.insert(0, str(XRGS_ROOT))

PYTHON_BIN_CANDIDATES = (Path(sys.prefix) / "bin", Path(sys.executable).parent)
for python_bin_dir in PYTHON_BIN_CANDIDATES:
    if (python_bin_dir / "ninja").exists():
        os.environ["PATH"] = f"{python_bin_dir}:{os.environ.get('PATH', '')}"
        break

from xrgs_unified.config import (
    TOP_LEVEL_ARG_SPECS,
    add_unified_config_args,
    copy_input_config,
    jsonable,
    load_config_defaults,
    parse_bool_arg,
    parse_float_csv,
    unified_renderer_config_defaults,
    write_json,
)


BACKBONE_DEFAULT_PATHS = {"scaffold_gs": XRGS_ROOT / "external" / "baselines" / "Scaffold-GS"}
STANDARD_DISTORT_MLP_RUN_NAME = "mlp_lr3e3_i10000_psf1000"
DATASET_PATH_MARKERS = {
    "mip360": "mip360",
    "mipnerf360": "mip360",
    "nerf_synthetic": "nerf_synthetic",
    "nerfsynthetic": "nerf_synthetic",
}
MIP360_IMAGES = {
    "bicycle": "images_4",
    "flowers": "images_4",
    "garden": "images_4",
    "stump": "images_4",
    "treehill": "images_4",
    "bonsai": "images_2",
    "counter": "images_2",
    "kitchen": "images_2",
    "room": "images_2",
}


def _should_print_unified_config_defaults(argv: list[str]) -> bool:
    return "--print_unified_config_defaults" in argv


def _mip360_images_dir(scene: str) -> str:
    try:
        return MIP360_IMAGES[scene]
    except KeyError as exc:
        raise ValueError(f"Unknown Mip-NeRF 360 scene: {scene}") from exc


def _infer_dataset_scene_from_path(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        dataset = DATASET_PATH_MARKERS.get(part.lower())
        if dataset is not None:
            return dataset, parts[index + 1]
    return None


def _infer_dataset_scene_from_args(args) -> tuple[str, str] | None:
    for attr in ("source_path", "model_path", "output_path"):
        value = getattr(args, attr, None)
        if not value:
            continue
        inferred = _infer_dataset_scene_from_path(Path(value).expanduser())
        if inferred is not None:
            return inferred
    return None


def _infer_dataset_scene_from_distort_mlp_path(path: str) -> tuple[str, str] | None:
    checkpoint_path = Path(path).expanduser().resolve()
    inferred = _infer_dataset_scene_from_path(checkpoint_path)
    if inferred is not None:
        return inferred
    result_path = checkpoint_path.parent / "results.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    for key in ("base_model_path", "config_source", "input_config_path"):
        value = result.get(key)
        if value:
            inferred = _infer_dataset_scene_from_path(Path(value).expanduser())
            if inferred is not None:
                return inferred
    return None


def _validate_canonical_distort_mlp_scene(args) -> None:
    expected = _infer_dataset_scene_from_args(args)
    if expected is None:
        return
    actual = _infer_dataset_scene_from_distort_mlp_path(args.canonical_distort_mlp_path)
    if actual is None:
        raise ValueError(
            "Cannot verify canonical_distort_mlp_path scene identity from checkpoint path or results.json: "
            f"{args.canonical_distort_mlp_path}"
        )
    if actual != expected:
        raise ValueError(
            "canonical_distort_mlp_path scene mismatch: "
            f"expected {expected[0]}/{expected[1]}, got {actual[0]}/{actual[1]} from {args.canonical_distort_mlp_path}"
        )


def _resolve_distort_mlp_checkpoint_arg(path: str, args, label: str) -> str:
    raw_path = str(path or "").strip()
    if not raw_path:
        return ""
    configured = Path(raw_path).expanduser()
    if configured.is_file():
        return str(configured.resolve())
    candidates = []
    if configured.suffix == ".pth":
        candidates.append(configured)
    candidates.append(configured / "distort_mlp.pth")
    ambiguous_scene_matches: list[Path] = []
    inferred = _infer_dataset_scene_from_args(args)
    if inferred is not None:
        dataset, scene = inferred
        scene_root = configured / dataset / scene
        candidates.append(scene_root / STANDARD_DISTORT_MLP_RUN_NAME / "distort_mlp.pth")
        if scene_root.exists():
            scene_matches = sorted(scene_root.glob("*/distort_mlp.pth"))
            if len(scene_matches) == 1:
                candidates.append(scene_matches[0])
            elif len(scene_matches) > 1:
                ambiguous_scene_matches = scene_matches
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate.resolve())
    if ambiguous_scene_matches:
        matches = ", ".join(str(path.parent) for path in ambiguous_scene_matches)
        raise ValueError(
            f"{label} points to a scene root with multiple DistortMLP runs. "
            f"Pass an exact distort_mlp.pth path. Matches: {matches}"
        )
    return str(configured.resolve())


def _load_canonical_distort_mlp(args, logger, load_distort_mlp_checkpoint, loaded_distort_mlp_protocol_metadata):
    if not args.canonical_distort_mlp_path:
        raise ValueError(
            "canonical_distort_mlp_path is required. "
            "Evaluation reference power must use the 3DGS canonical DistortMLP. "
            "If this is the first 3DGS baseline_distort_mlp run for the scene, "
            "let it save distort_mlp.pth, then rerun with that checkpoint as the canonical path."
        )
    args.canonical_distort_mlp_path = _resolve_distort_mlp_checkpoint_arg(
        args.canonical_distort_mlp_path,
        args,
        "canonical_distort_mlp_path",
    )
    canonical_distort_mlp = load_distort_mlp_checkpoint(args.canonical_distort_mlp_path, logger, required=True)
    canonical_distort_mlp_protocol = loaded_distort_mlp_protocol_metadata(
        args.canonical_distort_mlp_path,
        logger,
        expected_backbone="gaussian_splatting",
    )
    _validate_canonical_distort_mlp_scene(args)
    logger.info("Inference GT reference power will use canonical DistortMLP from %s", args.canonical_distort_mlp_path)
    return canonical_distort_mlp, canonical_distort_mlp_protocol


def _load_backbone_runtime(backbone_name: str, backbone_path: Path):
    if backbone_name != "scaffold_gs":
        raise ValueError("This release supports scaffold_gs only")
    backbone_path_str = str(backbone_path)
    while backbone_path_str in sys.path:
        sys.path.remove(backbone_path_str)
    sys.path.insert(1, backbone_path_str)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from utils.general_utils import safe_state
    from xrgs_unified.scaffold_adapter import ScaffoldGSAdapter
    return ModelParams, OptimizationParams, PipelineParams, safe_state, ScaffoldGSAdapter()


def _iteration_from_plugin_metadata_path(path: Path) -> int:
    iteration_name = path.parent.name
    if iteration_name.startswith("iteration_"):
        suffix = iteration_name.split("_")[-1]
        if suffix.isdigit():
            return int(suffix)
    return -1


def _latest_unified_plugin_metadata_path(checkpoint_path: Path) -> Path:
    metadata_paths = sorted(
        (checkpoint_path / "point_cloud").glob("iteration_*/plugin_metadata.json"),
        key=_iteration_from_plugin_metadata_path,
    )
    if not metadata_paths:
        raise FileNotFoundError(
            "Unified renderer requires either results.json or a saved "
            f"plugin metadata file under {checkpoint_path / 'point_cloud'}"
        )
    return metadata_paths[-1]


def _checkpoint_results_from_unified_metadata(checkpoint_path: Path) -> dict:
    metadata_path = _latest_unified_plugin_metadata_path(checkpoint_path)
    metadata = json.loads(metadata_path.read_text())
    save_iteration = int(metadata.get("save_iteration", _iteration_from_plugin_metadata_path(metadata_path)))
    base_model_path = metadata.get("base_model_path")
    if not base_model_path:
        raise ValueError(f"Unified plug-in metadata is missing base_model_path: {metadata_path}")
    baseline_pipeline = metadata.get("baseline_pipeline") or {}
    distort_mlp_path = metadata.get("distort_mlp_path") or baseline_pipeline.get("distort_mlp_path")
    if not distort_mlp_path:
        raise ValueError(f"Unified plug-in metadata is missing baseline_pipeline.distort_mlp_path: {metadata_path}")
    return {
        "mode": "unified_plugin",
        "base_model_path": base_model_path,
        "save_iteration": save_iteration,
        "base_loaded_iteration": int(metadata.get("base_loaded_iteration", -1)),
        "plugin_metadata": metadata,
        "plugin_metadata_path": str(metadata_path),
        "distort_mlp_path": distort_mlp_path,
        "eval_only_source_results": None,
        "eval_only_source_metadata": str(metadata_path),
    }


def _unified_save_iteration_from_results(results: dict, fallback: int = -1) -> int:
    value = results.get("save_iteration")
    return int(value) if value is not None else int(fallback)


def _checkpoint_iteration_from_results(results: dict, fallback: int = -1) -> int:
    for key in ("save_iteration", "base_loaded_iteration"):
        value = results.get(key)
        if value is not None:
            return int(value)
    return int(fallback)


def _unified_base_iteration_from_metadata(metadata: dict, fallback: int = -1) -> int:
    if not isinstance(metadata, dict):
        return int(fallback)
    value = metadata.get("base_loaded_iteration")
    return int(value) if value is not None else int(fallback)


def _unified_metadata_from_results_or_file(checkpoint_path: Path, results: dict, iteration: int) -> dict:
    metadata = results.get("plugin_metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata_path = checkpoint_path / "point_cloud" / f"iteration_{iteration}" / "plugin_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing unified plug-in metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def _apply_unified_layout_from_metadata(args, metadata: dict) -> None:
    layout = metadata.get("layout")
    if not isinstance(layout, dict):
        return
    args.ss_dim = int(layout.get("ss", {}).get("dim", args.ss_dim))
    args.display_dim = int(layout.get("display", {}).get("dim", args.display_dim))
    args.lens_dim = int(layout.get("lens", {}).get("dim", args.lens_dim))


def _freeze_backbone(gaussians) -> None:
    if hasattr(gaussians, "eval"):
        gaussians.eval()
    if isinstance(gaussians, torch.nn.Module):
        for parameter in gaussians.parameters():
            parameter.requires_grad_(False)
    for value in vars(gaussians).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
            for parameter in value.parameters():
                parameter.requires_grad_(False)
        elif isinstance(value, torch.nn.Parameter):
            value.requires_grad_(False)


def _add_renderer_args(parser: ArgumentParser) -> None:
    parser.add_argument("--render_checkpoint", "--render-checkpoint", default=None, help="Existing XRGS run directory to render.")
    parser.add_argument("--render_kind", "--render-kind", choices=["auto", "unified_plugin", "baseline_distort_mlp"], default=None)
    parser.add_argument("--render_split", "--render-split", choices=["test", "train", "all"], default=None)
    parser.add_argument("--render_output_root", "--render-output-root", default=None)
    parser.add_argument("--render_savings", "--render-savings", type=parse_float_csv, default=None)
    parser.add_argument("--render_downscales", "--render-downscales", type=parse_float_csv, default=None)
    parser.add_argument("--render_max_views", "--render-max-views", type=int, default=None)
    parser.add_argument("--eval_savings", "--eval-savings", dest="eval_savings", type=parse_float_csv, default=None)
    parser.add_argument("--eval_downscales", "--eval-downscales", dest="eval_downscales", type=parse_float_csv, default=None)
    parser.add_argument("--eval_power_mode", "--eval-power-mode", choices=["global", "per_image"], default=None)
    parser.add_argument("--eval_lens_mode", "--eval-lens-mode", choices=["distortion", "psf"], default=None)
    parser.add_argument("--lens_mode", "--lens-mode", choices=["distortion", "psf"], default=None)
    parser.add_argument("--lens_pad_ratio", "--lens-pad-ratio", type=float, default=None)
    parser.add_argument("--brightness_match_mode", "--brightness-match-mode", choices=["off", "per_image", "fixed_train0"], default=None)
    parser.add_argument("--eval_lpips", "--eval-lpips", nargs="?", const=True, type=parse_bool_arg, default=None)
    parser.add_argument("--render_eval", "--render-eval", nargs="?", const=True, type=parse_bool_arg, default=None)
    parser.add_argument("--render_save_gt", "--render-save-gt", nargs="?", const=True, type=parse_bool_arg, default=None)
    parser.add_argument("--render_fps", "--render-fps", nargs="?", const=True, type=parse_bool_arg, default=None)
    parser.add_argument("--render_fps_warmup", "--render-fps-warmup", type=int, default=None)
    parser.add_argument("--render_fps_iters", "--render-fps-iters", type=int, default=None)
    parser.add_argument("--quiet", nargs="?", const=True, type=parse_bool_arg, default=None)


def _renderer_fps_cameras(scene) -> list:
    cameras = scene.getTestCameras()
    if not cameras:
        raise ValueError("FPS render requires a non-empty test set; rerun with eval/test cameras enabled.")
    return cameras


def _measure_raw_render_fps(
    *,
    adapter,
    scene,
    gaussians,
    pipe,
    background: torch.Tensor,
    args,
    mode: str,
) -> dict:
    from xrgs_unified.conditions import set_unified_conditions
    from xrgs_unified.evaluation import EVAL_MODE_DISTORTION_MLP

    warmup_iters = int(args.render_fps_warmup)
    if warmup_iters < 0:
        raise ValueError(f"render_fps_warmup must be non-negative, got {warmup_iters}")
    if hasattr(gaussians, "eval"):
        gaussians.eval()
    test_cameras = _renderer_fps_cameras(scene)
    timed_views_per_grid = int(args.render_fps_iters)
    if timed_views_per_grid <= 0:
        raise ValueError(f"render_fps_iters must be positive, got {timed_views_per_grid}")
    cameras = test_cameras[:timed_views_per_grid]
    conditions = [(float(saving), float(downscale)) for saving in args.eval_savings for downscale in args.eval_downscales]
    if not conditions:
        raise ValueError("FPS render requires at least one eval/render grid condition")
    render_fn = adapter.render_baseline if mode == EVAL_MODE_DISTORTION_MLP else adapter.render

    def render_once(camera, downscale: float, saving: float):
        set_unified_conditions(camera, downscale, saving, args)
        return render_fn(camera, gaussians, pipe, background)

    grid_timings = []
    all_fps = []
    all_ms = []
    total_elapsed_sec = 0.0
    timer = "cuda_event" if torch.cuda.is_available() else "perf_counter"
    image = None
    with torch.no_grad():
        for condition_index, (saving, downscale) in enumerate(conditions):
            view_timings = []
            grid_elapsed_sec = 0.0
            for view_index, camera in enumerate(cameras):
                for _warmup_index in range(warmup_iters):
                    image = render_once(camera, downscale, saving)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = render_once(camera, downscale, saving)
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_sec = float(start_event.elapsed_time(end_event) / 1000.0)
                else:
                    start_time = time.perf_counter()
                    image = render_once(camera, downscale, saving)
                    elapsed_sec = float(time.perf_counter() - start_time)
                elapsed_ms = float(elapsed_sec * 1000.0)
                fps = float(1.0 / elapsed_sec) if elapsed_sec > 0.0 else float("inf")
                grid_elapsed_sec += elapsed_sec
                total_elapsed_sec += elapsed_sec
                all_fps.append(fps)
                all_ms.append(elapsed_ms)
                view_timings.append(
                    {
                        "view_index": view_index,
                        "image_name": getattr(camera, "image_name", None),
                        "elapsed_ms": elapsed_ms,
                        "fps": fps,
                    }
                )
            grid_timings.append(
                {
                    "condition_index": condition_index,
                    "saving": saving,
                    "downscale": downscale,
                    "avg_fps": float(sum(view["fps"] for view in view_timings) / len(view_timings)),
                    "avg_ms": float(sum(view["elapsed_ms"] for view in view_timings) / len(view_timings)),
                    "aggregate_fps": float(len(view_timings) / grid_elapsed_sec) if grid_elapsed_sec > 0.0 else float("inf"),
                    "elapsed_sec": float(grid_elapsed_sec),
                    "timed_iters": len(view_timings),
                    "views": view_timings,
                }
            )
    if image is not None:
        del image
    avg_fps = float(sum(all_fps) / len(all_fps))
    avg_ms = float(sum(all_ms) / len(all_ms))
    aggregate_fps = float(len(all_fps) / total_elapsed_sec) if total_elapsed_sec > 0.0 else float("inf")
    return {
        "avg_fps": avg_fps,
        "avg_ms": avg_ms,
        "aggregate_fps": aggregate_fps,
        "elapsed_sec": float(total_elapsed_sec),
        "warmup_iters": warmup_iters,
        "warmup_policy": "per_view",
        "timed_iters": len(all_fps),
        "timed_iters_per_view": 1,
        "render_fps_iters": timed_views_per_grid,
        "render_fps_iters_policy": "max_test_views_per_grid",
        "timed_scope": "all_grid_conditions_first_n_test_cameras_one_timed_render_each",
        "timer": timer,
        "split_requested": args.render_split,
        "split_used": "test",
        "render_max_views_ignored": int(args.render_max_views),
        "test_views_total": len(test_cameras),
        "views_per_grid": len(cameras),
        "grid_count": len(conditions),
        "eval_savings": [float(value) for value in args.eval_savings],
        "eval_downscales": [float(value) for value in args.eval_downscales],
        "includes_post_rendering": False,
        "saved_images": False,
        "grids": grid_timings,
    }


def _preload_renderer_checkpoint(argv: list[str], config_defaults: dict) -> tuple[Path | None, dict | None]:
    defaults = unified_renderer_config_defaults()
    pre_parser = ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument("--render_checkpoint", "--render-checkpoint", default=config_defaults.get("render_checkpoint", defaults["render_checkpoint"]))
    pre_parser.add_argument("--eval_only_ckpt", default=config_defaults.get("eval_only_ckpt", defaults["eval_only_ckpt"]))
    pre_parser.add_argument("--output_path", default=config_defaults.get("output_path", defaults["output_path"]))
    known, _unknown = pre_parser.parse_known_args(argv)
    checkpoint = str(known.render_checkpoint or known.eval_only_ckpt or known.output_path or "").strip()
    if not checkpoint:
        return None, None
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        return checkpoint_path, None
    if not checkpoint_path.is_dir():
        raise ValueError(f"Renderer checkpoint must be an XRGS run directory, got: {checkpoint_path}")
    results_path = checkpoint_path / "results.json"
    if results_path.exists():
        return checkpoint_path, json.loads(results_path.read_text())
    return checkpoint_path, _checkpoint_results_from_unified_metadata(checkpoint_path)


def _renderer_config_path_from_checkpoint(results: dict, checkpoint_path: Path) -> Path | None:
    for key in ("input_config_path", "config_source"):
        value = results.get(key)
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path.resolve()
    for name in ("eval_input_config.yaml", "input_config.yaml"):
        path = checkpoint_path / name
        if path.exists():
            return path.resolve()
    return None


def _resolve_renderer_checkpoint(args) -> tuple[Path, dict]:
    checkpoint = str(getattr(args, "render_checkpoint", "") or getattr(args, "eval_only_ckpt", "") or getattr(args, "output_path", "") or "").strip()
    if not checkpoint:
        raise ValueError("Unified renderer requires --render_checkpoint, --eval_only_ckpt, or output_path pointing to an existing XRGS run directory")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Renderer checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        raise ValueError(f"Renderer checkpoint must be an XRGS run directory, got: {checkpoint_path}")
    results_path = checkpoint_path / "results.json"
    if results_path.exists():
        return checkpoint_path, json.loads(results_path.read_text())
    return checkpoint_path, _checkpoint_results_from_unified_metadata(checkpoint_path)


def _distort_mlp_path_from_results(results: dict, checkpoint_path: Path) -> str:
    distort_mlp_path = results.get("distort_mlp_path")
    if not distort_mlp_path:
        protocol = results.get("baseline_distort_mlp_protocol") or {}
        distort_mlp_path = protocol.get("distort_mlp_path")
    if not distort_mlp_path:
        baseline_pipeline = (results.get("plugin_metadata") or {}).get("baseline_pipeline", {})
        distort_mlp_path = baseline_pipeline.get("distort_mlp_path")
    if not distort_mlp_path and results.get("mode") == "baseline_distort_mlp":
        distort_mlp_path = str(checkpoint_path / "distort_mlp.pth")
    if not distort_mlp_path:
        raise ValueError(f"Renderer checkpoint is missing distort_mlp_path: {checkpoint_path}")
    distort_mlp_path = str(Path(distort_mlp_path).expanduser().resolve())
    if not Path(distort_mlp_path).exists():
        raise FileNotFoundError(f"Renderer DistortMLP checkpoint not found: {distort_mlp_path}")
    return distort_mlp_path


def _renderer_kind_from_results(args, results: dict) -> tuple[str, str]:
    source_kind = results.get("mode")
    if source_kind not in {"unified_plugin", "baseline_distort_mlp"}:
        raise ValueError(f"Renderer checkpoint must be unified_plugin or baseline_distort_mlp, got mode={source_kind!r}")
    requested = getattr(args, "render_kind", "auto")
    render_kind = source_kind if requested == "auto" else requested
    if render_kind == "unified_plugin" and source_kind != "unified_plugin":
        raise ValueError("render_kind=unified_plugin requires a unified_plugin checkpoint")
    return source_kind, render_kind


def _apply_renderer_scene_defaults(args, backbone_name: str, checkpoint_path: Path, logger) -> None:
    if getattr(args, "source_path", ""):
        return
    inferred = _infer_dataset_scene_from_path(checkpoint_path) or _infer_dataset_scene_from_path(Path(args.model_path))
    if inferred is None:
        logger.warning("Renderer could not infer dataset/scene; pass -s/--source_path and native data args explicitly")
        return
    dataset, scene = inferred
    images = _mip360_images_dir(scene) if dataset == "mip360" else "images"
    source_path = None
    defaults = {"eval": True}
    if backbone_name == "scaffold_gs":
        source_path = XRGS_ROOT / "external/baselines/Scaffold-GS/data" / ("mipnerf360" if dataset == "mip360" else "nerfsynthetic") / scene
        defaults.update({"images": images, "resolution": 1, "appearance_dim": 0, "update_init_factor": 16 if dataset == "mip360" else 4})
    elif backbone_name == "gaussian_splatting":
        source_path = XRGS_ROOT / "dataset" / dataset / scene
        defaults.update({"images": images, "resolution": 1, "data_device": "cuda"})
    elif backbone_name == "mip_splatting":
        source_path = XRGS_ROOT / "dataset" / dataset / scene
        defaults.update({"images": images, "resolution": 1, "data_device": "cuda", "kernel_size": 0.1})
    elif backbone_name == "octree_gs":
        source_path = XRGS_ROOT / "external/baselines/Octree-GS/data" / ("mipnerf360" if dataset == "mip360" else "nerf_synthetic") / scene
        defaults.update({"images": images, "resolution": 1 if dataset == "mip360" else 800, "data_device": "cpu" if dataset == "mip360" else "cuda", "appearance_dim": 0, "visible_threshold": -1, "base_layer": 10, "dist2level": "round", "progressive": True, "init_level": -1, "dist_ratio": 0.999, "levels": -1})
        if dataset == "mip360":
            defaults["ds"] = images.removeprefix("images_")
    elif backbone_name in {"hac_plus", "context_gs"}:
        method = "HAC-plus" if backbone_name == "hac_plus" else "ContextGS"
        source_path = XRGS_ROOT / "external/baselines" / method / "data" / ("mipnerf360" if dataset == "mip360" else "nerf_synthetic") / scene
        defaults.update({"images": images, "resolution": 1 if dataset == "mip360" else 800, "data_device": "cuda" if backbone_name == "hac_plus" else "cpu", "feat_dim": 50, "update_init_factor": 16 if dataset == "mip360" else 4, "voxel_size": 0.001})
        if backbone_name == "context_gs":
            defaults.update({"hyper_divisor": 4, "target_ratio": 0.2})
    elif backbone_name == "comp_gs":
        source_path = XRGS_ROOT / "dataset" / dataset / scene
        defaults.update({"images": images, "data_device": "cuda"})
    if source_path is None:
        return
    args.source_path = str(source_path.resolve())
    for name, value in defaults.items():
        setattr(args, name, value)
    logger.info("Renderer inferred scene defaults for %s/%s from %s", dataset, scene, checkpoint_path)


def render_unified_checkpoint(
    args,
    *,
    checkpoint_path: Path,
    checkpoint_results: dict,
    backbone_name: str,
    backbone_path: Path,
    model_params_cls,
    pipeline_params_cls,
    safe_state_fn,
    adapter,
) -> Path:
    from xrgs_utils.lens_utils import load_optics_wrapper, resolve_dataset_lens_optics_sizes
    from xrgs_unified.checkpoint import load_plugin_metadata, validate_resume_metadata
    from xrgs_unified.evaluation import EVAL_MODE_DISTORTION_MLP, EVAL_MODE_OURS, load_lpips_model, render_inference_grid
    from xrgs_unified.layout import build_plugin_layout, disabled_plugin_modules, total_layout_dim
    from xrgs_unified.runtime import get_logger, loaded_distort_mlp_protocol_metadata, load_distort_mlp_checkpoint

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    args.render_checkpoint = str(checkpoint_path)
    source_kind, render_kind = _renderer_kind_from_results(args, checkpoint_results)
    checkpoint_backbone = checkpoint_results.get("backbone") or (checkpoint_results.get("plugin_metadata") or {}).get("backbone")
    accepted_backbones = {args.backbone}
    adapter_name = getattr(adapter, "name", None)
    if adapter_name:
        accepted_backbones.add(adapter_name)
    if checkpoint_backbone and checkpoint_backbone not in accepted_backbones:
        raise ValueError(f"Renderer checkpoint backbone={checkpoint_backbone!r} does not match active backbone={args.backbone!r}")
    base_model_path = checkpoint_results.get("base_model_path")
    if not base_model_path:
        raise ValueError(f"Renderer checkpoint is missing base_model_path: {checkpoint_path}")
    base_model_path = str(Path(base_model_path).expanduser().resolve())
    args.model_path = base_model_path
    args.output_path = str(Path(args.render_output_root).expanduser().resolve() if args.render_output_root else checkpoint_path)
    output_root = Path(args.output_path)
    render_output_root = output_root / "infer_render_output"
    e2e_output_root = output_root / "infer_e2e_output"

    logger = get_logger(output_root)
    logger.info("args: %s", args)
    logger.info("Unified renderer source checkpoint: %s", checkpoint_path)
    logger.info("Unified renderer source kind=%s render kind=%s", source_kind, render_kind)
    logger.info("Unified renderer backbone path: %s", backbone_path)
    input_config_path = copy_input_config(output_root, args.config, filename="render_input_config.yaml")
    if input_config_path is not None:
        logger.info("Copied renderer input config to %s", input_config_path)
    safe_state_fn(args.quiet)
    _apply_renderer_scene_defaults(args, args.backbone, checkpoint_path, logger)

    checkpoint_metadata = None
    save_iteration = None
    if source_kind == "unified_plugin":
        save_iteration = _unified_save_iteration_from_results(checkpoint_results, args.load_iteration)
        checkpoint_metadata = (
            _unified_metadata_from_results_or_file(checkpoint_path, checkpoint_results, save_iteration)
            if bool(getattr(adapter, "supports_condition_injection", True))
            else checkpoint_results.get("plugin_metadata", {})
        )
        args.load_iteration = _unified_base_iteration_from_metadata(checkpoint_metadata, args.load_iteration)
        args.save_iteration = save_iteration
        if bool(getattr(adapter, "supports_condition_injection", True)):
            _apply_unified_layout_from_metadata(args, checkpoint_metadata)
    else:
        args.load_iteration = _checkpoint_iteration_from_results(checkpoint_results, args.load_iteration)

    supports_condition_injection = bool(getattr(adapter, "supports_condition_injection", True))
    render_mode = EVAL_MODE_OURS if render_kind == "unified_plugin" else EVAL_MODE_DISTORTION_MLP
    layout = build_plugin_layout(args.ss_dim, args.display_dim, args.lens_dim)
    total_dim = total_layout_dim(layout)
    inject_dims = {}
    disabled_modules = set()
    parser = ArgumentParser(add_help=False)
    lp = model_params_cls(parser)
    pp = pipeline_params_cls(parser)
    dataset = lp.extract(args)
    dataset.eval_only = True
    dataset.plugin_init_mode_normalized = "resume_unified"
    scene_load_iteration = args.load_iteration
    if render_kind == "unified_plugin" and supports_condition_injection:
        if checkpoint_metadata is None or save_iteration is None:
            raise RuntimeError("Unified renderer did not load checkpoint metadata")
        dataset.xrgs_base_model_path = base_model_path
        dataset.model_path = str(checkpoint_path)
        dataset.res_cond_dim = total_dim
        dataset.res_cond_mode = "learned_anchor_multi_plugin"
        scene_load_iteration = save_iteration
    pipe = pp.extract(args)
    gaussians, scene = adapter.load_scene(dataset, iteration=scene_load_iteration)
    gaussians.spatial_lr_scale = scene.cameras_extent
    if render_kind == "unified_plugin" and supports_condition_injection:
        adapter.set_plugin_layout(gaussians, layout)
        inject_dims = adapter.validate_inject_dim(gaussians, args.inject_dim)
        metadata = load_plugin_metadata(checkpoint_path, scene.loaded_iter)
        validate_resume_metadata(metadata, layout, total_dim, inject_dims)
        adapter.require_unified_plugin_checkpoint(checkpoint_path, scene.loaded_iter, total_dim)
        disabled_modules = disabled_plugin_modules(args, layout)
        adapter.zero_disabled_plugin_slices(gaussians, layout, disabled_modules)
        logger.info("Loaded unified renderer checkpoint from %s", checkpoint_path)
    else:
        _freeze_backbone(gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if args.render_fps:
        fps_result = _measure_raw_render_fps(
            adapter=adapter,
            scene=scene,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            args=args,
            mode=render_mode,
        )
        metric_source = "ours_unified" if render_mode == EVAL_MODE_OURS else "baseline_raw_render"
        summary = {
            "mode": "unified_renderer",
            "fps_only": True,
            "source_mode": source_kind,
            "render_kind": render_kind,
            "metric_source": metric_source,
            "backbone": args.backbone,
            "backbone_path": str(backbone_path),
            "checkpoint_path": str(checkpoint_path),
            "base_model_path": base_model_path,
            "source_results": str(checkpoint_path / "results.json") if (checkpoint_path / "results.json").exists() else None,
            "source_metadata": checkpoint_results.get("eval_only_source_metadata"),
            "output_root": str(output_root),
            "render_split_requested": args.render_split,
            "render_split": "test",
            "render_max_views_requested": args.render_max_views,
            "render_max_views_applied": 0,
            "render_fps_iters": fps_result["render_fps_iters"],
            "views_per_grid": fps_result["views_per_grid"],
            "test_views_total": fps_result["test_views_total"],
            "condition_policy": "all_configured_grid_conditions",
            "grid_count": fps_result["grid_count"],
            "eval_savings": fps_result["eval_savings"],
            "eval_downscales": fps_result["eval_downscales"],
            "render_eval_requested": bool(args.render_eval),
            "render_save_gt_requested": bool(args.render_save_gt),
            "condition_injection_supported": supports_condition_injection,
            "inject_dims": inject_dims,
            "disabled_modules": sorted(disabled_modules),
            "fps": fps_result,
        }
        result_path = output_root / "infer_fps.json"
        write_json(result_path, jsonable(summary))
        logger.info("Wrote %s", result_path)
        logger.info("Raw render FPS: %.3f", fps_result["avg_fps"])
        return result_path

    args.distort_mlp_path = _distort_mlp_path_from_results(checkpoint_results, checkpoint_path)
    distort_mlp = load_distort_mlp_checkpoint(args.distort_mlp_path, logger, required=True)
    baseline_distort_mlp_protocol = loaded_distort_mlp_protocol_metadata(args.distort_mlp_path, logger, expected_backbone=args.backbone)
    canonical_distort_mlp, canonical_distort_mlp_protocol = _load_canonical_distort_mlp(
        args,
        logger,
        load_distort_mlp_checkpoint,
        loaded_distort_mlp_protocol_metadata,
    )

    optics_input_size, optics_output_size = resolve_dataset_lens_optics_sizes(scene.getTrainCameras(), pad_ratio=args.lens_pad_ratio)
    optics_wrapper = load_optics_wrapper(
        optics_sim_path=args.optics_sim_path,
        psf_dir=args.psf_dir,
        psf_backend=args.psf_backend,
        input_size=optics_input_size,
        output_size=optics_output_size,
        verbose=False,
    )
    logger.info("Loaded optics wrapper with dataset-derived input/output sizes %d/%d: %s", optics_input_size, optics_output_size, optics_wrapper)
    lpips_model = load_lpips_model(args, logger) if args.render_eval else None
    render_results = render_inference_grid(
        adapter=adapter,
        scene=scene,
        gaussians=gaussians,
        pipe=pipe,
        optics_wrapper=optics_wrapper,
        distort_mlp=distort_mlp,
        background=background,
        args=args,
        lpips_model=lpips_model,
        mode=render_mode,
        canonical_distort_mlp=canonical_distort_mlp,
        render_output_root=render_output_root,
        e2e_output_root=e2e_output_root,
        split=args.render_split,
        max_views=args.render_max_views,
        write_eval=args.render_eval,
        save_gt=args.render_save_gt,
    )
    metric_source = "ours_unified" if render_mode == EVAL_MODE_OURS else "baseline_pipeline"
    summary = {
        "mode": "unified_renderer",
        "source_mode": source_kind,
        "render_kind": render_kind,
        "metric_source": metric_source if args.render_eval else None,
        "backbone": args.backbone,
        "backbone_path": str(backbone_path),
        "checkpoint_path": str(checkpoint_path),
        "base_model_path": base_model_path,
        "source_results": str(checkpoint_path / "results.json") if (checkpoint_path / "results.json").exists() else None,
        "source_metadata": checkpoint_results.get("eval_only_source_metadata"),
        "output_root": str(output_root),
        "render_output_root": str(render_output_root),
        "e2e_output_root": str(e2e_output_root),
        "render_split": args.render_split,
        "render_max_views": args.render_max_views,
        "render_eval": args.render_eval,
        "render_save_gt": args.render_save_gt,
        "eval_savings": args.eval_savings,
        "eval_downscales": args.eval_downscales,
        "eval_power_mode": args.eval_power_mode,
        "eval_lens_mode": args.eval_lens_mode,
        "brightness_match_mode": args.brightness_match_mode,
        "eval_fixed_brightness_scale": getattr(args, "eval_fixed_brightness_scale_value", None),
        "fixed_brightness_scale_history": getattr(args, "fixed_brightness_scale_history", []),
        "eval_lpips": bool(args.eval_lpips) if args.render_eval else False,
        "lpips_net": args.lpips_net if args.render_eval else None,
        "distort_mlp_path": args.distort_mlp_path,
        "baseline_distort_mlp_protocol": baseline_distort_mlp_protocol,
        "canonical_distort_mlp_path": args.canonical_distort_mlp_path or None,
        "canonical_distort_mlp_protocol": canonical_distort_mlp_protocol,
        "display_mapping": "post_distortion_direct_linear_rgb_target_power",
        "display_mapping_reference": "gt_distorted_image",
        "display_mapping_reference_distort_mlp": "canonical_distort_mlp",
        "condition_injection_supported": supports_condition_injection,
        "inject_dims": inject_dims,
        "disabled_modules": sorted(disabled_modules),
        "render": render_results,
        "eval": render_results if args.render_eval else None,
    }
    result_path = output_root / "infer_results.json"
    write_json(result_path, jsonable(summary))
    logger.info("Wrote %s", result_path)
    return result_path


def run_unified_renderer(argv: list[str], config_defaults: dict, config_source: str | None, backbone_name: str, backbone_path: Path) -> None:
    pre_checkpoint_path, pre_results = _preload_renderer_checkpoint(argv, config_defaults)
    if config_source is None and pre_checkpoint_path is not None and pre_results is not None:
        checkpoint_config = _renderer_config_path_from_checkpoint(pre_results, pre_checkpoint_path)
        if checkpoint_config is not None:
            checkpoint_defaults, checkpoint_config_source = load_config_defaults(["--config", str(checkpoint_config)], TOP_LEVEL_ARG_SPECS)
            config_defaults = {**checkpoint_defaults, **config_defaults}
            config_source = checkpoint_config_source
    if _should_print_unified_config_defaults(argv):
        effective_defaults = {**unified_renderer_config_defaults(), **config_defaults}
        effective_defaults["mode"] = "unified_renderer"
        effective_defaults["backbone"] = backbone_name
        effective_defaults["backbone_path"] = str(backbone_path)
        print(json.dumps(effective_defaults, indent=2))
        return
    if pre_results is not None:
        checkpoint_backbone = pre_results.get("backbone") or (pre_results.get("plugin_metadata") or {}).get("backbone")
        if checkpoint_backbone in BACKBONE_DEFAULT_PATHS:
            backbone_name = checkpoint_backbone
            result_backbone_path = pre_results.get("backbone_path")
            if result_backbone_path and Path(result_backbone_path).expanduser().exists():
                backbone_path = Path(result_backbone_path).expanduser().resolve()
            else:
                backbone_path = BACKBONE_DEFAULT_PATHS[backbone_name].resolve()
    model_params_cls, _optimization_params_cls, pipeline_params_cls, safe_state_fn, adapter = _load_backbone_runtime(backbone_name, backbone_path)
    parser = ArgumentParser(description="Render an existing XRGS unified or DistortMLP checkpoint without training")
    parser.add_argument("--config", default=config_source)
    model_params_cls(parser)
    pipeline_params_cls(parser)
    add_unified_config_args(parser)
    _add_renderer_args(parser)
    parser.set_defaults(**{**unified_renderer_config_defaults(), **config_defaults})
    parser.set_defaults(mode="unified_renderer", backbone=backbone_name, backbone_path=str(backbone_path))
    args = parser.parse_args(argv)
    if not args.eval_lens_mode:
        args.eval_lens_mode = args.lens_mode
    args.render_savings = args.render_savings if args.render_savings is not None else args.eval_savings
    args.render_downscales = args.render_downscales if args.render_downscales is not None else args.eval_downscales
    args.eval_savings = args.render_savings
    args.eval_downscales = args.render_downscales
    args.render_max_views = int(args.eval_max_views if args.render_max_views is None else args.render_max_views)
    args.eval_max_views = args.render_max_views
    args.render_fps_warmup = int(args.render_fps_warmup)
    args.render_fps_iters = int(args.render_fps_iters)
    args.brightness_match = args.brightness_match_mode != "off"
    args.fixed_brightness_scale_value = None
    args.fixed_brightness_scale_history = []
    checkpoint_path, checkpoint_results = _resolve_renderer_checkpoint(args)
    render_unified_checkpoint(
        args,
        checkpoint_path=checkpoint_path,
        checkpoint_results=checkpoint_results,
        backbone_name=backbone_name,
        backbone_path=backbone_path,
        model_params_cls=model_params_cls,
        pipeline_params_cls=pipeline_params_cls,
        safe_state_fn=safe_state_fn,
        adapter=adapter,
    )


def render_checkpoint(
    checkpoint_path: str | Path,
    *,
    canonical_distort_mlp_path: str | Path | None = None,
    backbone: str | None = None,
    backbone_path: str | Path | None = None,
    config_path: str | Path | None = None,
    render_kind: str = "auto",
    render_split: str = "test",
    render_output_root: str | Path | None = None,
    render_savings: str | list[float] | None = None,
    render_downscales: str | list[float] | None = None,
    render_max_views: int | None = None,
    render_eval: bool = False,
    render_save_gt: bool = True,
    render_fps: bool = False,
    render_fps_warmup: int = 5,
    render_fps_iters: int = 10,
    extra_args: list[str] | None = None,
) -> Path:
    argv = ["--mode", "unified_renderer", "--render_checkpoint", str(checkpoint_path)]
    if config_path is not None:
        argv.extend(["--config", str(config_path)])
    if backbone is not None:
        argv.extend(["--backbone", backbone])
    if backbone_path is not None:
        argv.extend(["--backbone_path", str(backbone_path)])
    if render_kind:
        argv.extend(["--render_kind", render_kind])
    if render_split:
        argv.extend(["--render_split", render_split])
    if render_output_root is not None:
        argv.extend(["--render_output_root", str(render_output_root)])
    if render_savings is not None:
        value = ",".join(f"{float(item):g}" for item in render_savings) if isinstance(render_savings, (list, tuple)) else str(render_savings)
        argv.extend(["--render_savings", value])
    if render_downscales is not None:
        value = ",".join(f"{float(item):g}" for item in render_downscales) if isinstance(render_downscales, (list, tuple)) else str(render_downscales)
        argv.extend(["--render_downscales", value])
    if render_max_views is not None:
        argv.extend(["--render_max_views", str(render_max_views)])
    argv.extend(["--render_eval", str(bool(render_eval)).lower()])
    argv.extend(["--render_save_gt", str(bool(render_save_gt)).lower()])
    argv.extend(["--render_fps", str(bool(render_fps)).lower()])
    argv.extend(["--render_fps_warmup", str(int(render_fps_warmup))])
    argv.extend(["--render_fps_iters", str(int(render_fps_iters))])
    if canonical_distort_mlp_path is not None:
        argv.extend(["--canonical_distort_mlp_path", str(canonical_distort_mlp_path)])
    if extra_args:
        argv.extend(extra_args)

    config_defaults, config_source = load_config_defaults(argv, TOP_LEVEL_ARG_SPECS)
    selected_backbone = backbone or config_defaults.get("backbone") or "scaffold_gs"
    selected_backbone_path = Path(backbone_path).expanduser().resolve() if backbone_path is not None else BACKBONE_DEFAULT_PATHS[selected_backbone].resolve()
    run_unified_renderer(argv, config_defaults, config_source, selected_backbone, selected_backbone_path)
    result_name = "infer_fps.json" if render_fps else "infer_results.json"
    return (Path(render_output_root).expanduser().resolve() if render_output_root is not None else Path(checkpoint_path).expanduser().resolve()) / result_name
