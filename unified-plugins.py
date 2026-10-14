#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import random
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

XRGS_ROOT = Path(__file__).resolve().parent
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
    baseline_distort_config_defaults,
    load_config_defaults,
    unified_config_defaults,
    unified_renderer_config_defaults,
)
from xrgs_unified.renderer import run_unified_renderer as run_unified_renderer_entry


BACKBONE_DEFAULT_PATHS = {"scaffold_gs": XRGS_ROOT / "external" / "baselines" / "Scaffold-GS"}

BACKBONE_CHOICES = tuple(BACKBONE_DEFAULT_PATHS.keys())
STANDARD_DISTORT_MLP_RUN_NAME = "mlp_lr3e3_i10000_psf1000"
DATASET_PATH_MARKERS = {
    "mip360": "mip360",
    "mipnerf360": "mip360",
    "nerf_synthetic": "nerf_synthetic",
    "nerfsynthetic": "nerf_synthetic",
}


def _should_print_unified_config_defaults(argv: list[str]) -> bool:
    return "--print_unified_config_defaults" in argv


def _top_level_from_args(argv: list[str], config_defaults: dict) -> tuple[str, str, Path]:
    defaults = {**baseline_distort_config_defaults(), **unified_config_defaults(), **unified_renderer_config_defaults()}
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--mode",
        default=config_defaults.get("mode", unified_config_defaults()["mode"]),
        choices=["unified_plugin", "baseline_distort_mlp", "unified_renderer"],
    )
    pre_parser.add_argument(
        "--backbone",
        default=config_defaults.get("backbone", defaults["backbone"]),
        choices=BACKBONE_CHOICES,
    )
    pre_parser.add_argument("--backbone_path", default=config_defaults.get("backbone_path", defaults["backbone_path"]))
    known, _unknown = pre_parser.parse_known_args(argv)
    backbone_path = Path(known.backbone_path).expanduser() if known.backbone_path else BACKBONE_DEFAULT_PATHS[known.backbone]
    backbone_path = backbone_path.resolve()
    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone path not found for {known.backbone}: {backbone_path}")
    return known.mode, known.backbone, backbone_path


def _infer_dataset_scene_from_args(args) -> tuple[str, str] | None:
    for attr in ("source_path", "model_path", "output_path"):
        value = getattr(args, attr, None)
        if not value:
            continue
        inferred = _infer_dataset_scene_from_path(Path(value).expanduser())
        if inferred is not None:
            return inferred
    return None


def _infer_dataset_scene_from_path(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        dataset = DATASET_PATH_MARKERS.get(part.lower())
        if dataset is not None:
            return dataset, parts[index + 1]
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


def _load_canonical_distort_mlp(
    args,
    logger,
    load_distort_mlp_checkpoint,
    loaded_distort_mlp_protocol_metadata,
):
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
    canonical_distort_mlp = load_distort_mlp_checkpoint(
        args.canonical_distort_mlp_path,
        logger,
        required=True,
    )
    canonical_distort_mlp_protocol = loaded_distort_mlp_protocol_metadata(
        args.canonical_distort_mlp_path,
        logger,
        expected_backbone="gaussian_splatting",
    )
    _validate_canonical_distort_mlp_scene(args)
    logger.info(
        "Evaluation GT reference power will use canonical DistortMLP from %s",
        args.canonical_distort_mlp_path,
    )
    return canonical_distort_mlp, canonical_distort_mlp_protocol


def _resolve_migrated_checkpoint_file(eval_only_ckpt: Path, results: dict, key: str, filename: str, label: str) -> Path:
    configured = str(results.get(key) or "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(eval_only_ckpt / filename)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key_str = str(candidate)
        if key_str in seen:
            continue
        seen.add(key_str)
        if candidate.exists():
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} not found; checked: {checked}")


def _resolve_migrated_run_path(configured: str | None, fallback: str, label: str) -> Path:
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(fallback).expanduser())

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key_str = str(candidate)
        if key_str in seen:
            continue
        seen.add(key_str)
        if candidate.exists():
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} not found; checked: {checked}")


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
    return ModelParams, OptimizationParams, PipelineParams, safe_state, ScaffoldGSAdapter(), "Unified Scaffold-GS SS/display/lens plug-in fine-tuning"


def _iteration_from_plugin_metadata_path(path: Path) -> int:
    iteration_name = path.parent.name
    if iteration_name.startswith("iteration_"):
        suffix = iteration_name.split("_")[-1]
        if suffix.isdigit():
            return int(suffix)
    return -1


def _latest_unified_plugin_metadata_path(eval_only_ckpt_path: Path) -> Path:
    metadata_paths = sorted(
        (eval_only_ckpt_path / "point_cloud").glob("iteration_*/plugin_metadata.json"),
        key=_iteration_from_plugin_metadata_path,
    )
    if not metadata_paths:
        raise FileNotFoundError(
            "Unified plug-in eval_only requires either results.json or a saved "
            f"plugin metadata file under {eval_only_ckpt_path / 'point_cloud'}"
        )
    return metadata_paths[-1]


def _checkpoint_results_from_unified_metadata(eval_only_ckpt_path: Path) -> dict:
    metadata_path = _latest_unified_plugin_metadata_path(eval_only_ckpt_path)
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


def _read_checkpoint_results(eval_only_ckpt_path: Path, runner_name: str) -> dict:
    results_path = eval_only_ckpt_path / "results.json"
    if results_path.exists():
        return json.loads(results_path.read_text())
    if runner_name == "Unified plug-in":
        return _checkpoint_results_from_unified_metadata(eval_only_ckpt_path)
    raise FileNotFoundError(f"{runner_name} eval_only requires results.json under eval_only_ckpt: {results_path}")


def _checkpoint_iteration_from_results(results: dict, fallback: int = -1) -> int:
    for key in ("save_iteration", "base_loaded_iteration"):
        value = results.get(key)
        if value is not None:
            return int(value)
    return int(fallback)


def _unified_save_iteration_from_results(results: dict, fallback: int = -1) -> int:
    value = results.get("save_iteration")
    return int(value) if value is not None else int(fallback)


def _unified_base_iteration_from_metadata(metadata: dict, fallback: int = -1) -> int:
    if not isinstance(metadata, dict):
        return int(fallback)
    value = metadata.get("base_loaded_iteration")
    return int(value) if value is not None else int(fallback)


def _unified_metadata_from_results_or_file(eval_only_ckpt_path: Path, results: dict, iteration: int) -> dict:
    metadata = results.get("plugin_metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata_path = eval_only_ckpt_path / "point_cloud" / f"iteration_{iteration}" / "plugin_metadata.json"
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


def _prepare_eval_only_ckpt(args, runner_name: str) -> tuple[Path | None, dict | None]:
    eval_only_ckpt = str(getattr(args, "eval_only_ckpt", "") or "").strip()
    eval_only = bool(getattr(args, "eval_only", False))
    if bool(eval_only_ckpt) != eval_only:
        raise ValueError(f"{runner_name} requires eval_only_ckpt and eval_only to be set together")
    if not eval_only:
        return None, None
    eval_only_ckpt_path = Path(eval_only_ckpt).expanduser().resolve()
    if not eval_only_ckpt_path.exists():
        raise FileNotFoundError(f"{runner_name} eval_only_ckpt not found: {eval_only_ckpt_path}")
    if not eval_only_ckpt_path.is_dir():
        raise ValueError(f"{runner_name} eval_only_ckpt must be an XRGS run directory, got: {eval_only_ckpt_path}")
    if not getattr(args, "output_path", ""):
        raise ValueError(f"{runner_name} eval_only still requires output_path for results.json")
    checkpoint_results = _read_checkpoint_results(eval_only_ckpt_path, runner_name)
    args.eval_only_ckpt = str(eval_only_ckpt_path)
    return eval_only_ckpt_path, checkpoint_results


def _eval_image_root_for_summary(args, output_path: Path, image_kind: str) -> str | None:
    image_mode = getattr(args, "eval_image_mode", "off")
    if image_mode != image_kind:
        return None
    configured = str(getattr(args, "eval_image_dir", "") or "").strip()
    if configured:
        return str(Path(configured).expanduser().resolve())
    folder = "Baseline_Evaluation_Images" if image_kind == "baseline" else "Ours_Evaluation_Images"
    return str((output_path / folder).resolve())


def _normalize_render_resolution_policy(args) -> None:
    if getattr(args, "render_resolution_policy", "super_sampling") == "direct_downscale_no_ss":
        args.use_super_sampling = False
    elif not bool(getattr(args, "use_super_sampling", True)):
        args.render_resolution_policy = "direct_downscale_no_ss"
    else:
        args.use_super_sampling = True
        args.render_resolution_policy = "super_sampling"


def run_unified_plugin(argv: list[str], config_defaults: dict, config_source: str | None, backbone_name: str, backbone_path: Path) -> None:
    if _should_print_unified_config_defaults(argv):
        effective_defaults = {**unified_config_defaults(), **config_defaults}
        effective_defaults["mode"] = "unified_plugin"
        effective_defaults["backbone"] = backbone_name
        effective_defaults["backbone_path"] = str(backbone_path)
        print(json.dumps(effective_defaults, indent=2))
        return
    (
        model_params_cls,
        optimization_params_cls,
        pipeline_params_cls,
        safe_state_fn,
        adapter,
        description,
    ) = _load_backbone_runtime(backbone_name, backbone_path)
    from xrgs_utils.display_power import display_power, target_power_from_saving
    from xrgs_utils.lens_utils import (
        apply_optics_model,
        load_optics_wrapper,
        resolve_dataset_lens_optics_sizes,
    )
    from xrgs_unified.brightness import apply_brightness_policy, set_fixed_brightness_scale
    from xrgs_unified.checkpoint import (
        load_plugin_metadata,
        normalize_plugin_init_mode,
        plugin_metadata_path,
        validate_resume_metadata,
        write_plugin_metadata,
    )
    from xrgs_unified.conditions import sample_downscale, sample_target_saving, set_unified_conditions
    from xrgs_unified.config import copy_input_config, jsonable, write_effective_config, write_json
    from xrgs_unified.display_protocol import (
        PostProcessing,
        crop_lens_metric_region,
        lens_gt_from_display_canvas,
    )
    from xrgs_unified.evaluation import EVAL_MODE_OURS, evaluate_grid, load_lpips_model
    from xrgs_unified.layout import build_plugin_layout, disabled_plugin_modules, total_layout_dim
    from xrgs_unified.runtime import (
        build_plugin_metadata,
        get_logger,
        lens_mode_schedule_summary,
        lens_quality_loss,
        loaded_distort_mlp_protocol_metadata,
        load_distort_mlp_checkpoint,
        psf_start_iteration,
        scaffold_distort_mlp_protocol_metadata,
        training_lens_mode,
    )
    parser = ArgumentParser(description=description)
    parser.add_argument("--config", default=config_source)
    lp = model_params_cls(parser)
    op = optimization_params_cls(parser)
    pp = pipeline_params_cls(parser)
    add_unified_config_args(parser)
    parser.add_argument(
        "--lpips_loss_weight",
        default=0.0,
        type=float,
        help="Unified-only LPIPS loss weight. Default 0 keeps the original training objective.",
    )
    parser.set_defaults(**{**unified_config_defaults(), **config_defaults})
    parser.set_defaults(mode="unified_plugin", backbone=backbone_name, backbone_path=str(backbone_path))
    args = parser.parse_args(argv)
    _normalize_render_resolution_policy(args)
    args.lpips_loss_weight = float(args.lpips_loss_weight)
    if args.lpips_loss_weight < 0.0:
        raise ValueError(f"lpips_loss_weight must be non-negative, got {args.lpips_loss_weight}")
    if not args.output_path:
        raise ValueError("Unified output_path must be set in config.yaml or via --output_path")
    if not args.eval_lens_mode:
        args.eval_lens_mode = args.lens_mode
    eval_only_ckpt, eval_only_results = _prepare_eval_only_ckpt(args, "Unified plug-in")
    args.plugin_init_mode_normalized = normalize_plugin_init_mode(args.plugin_init_mode)
    args.brightness_match = args.brightness_match_mode != "off"
    args.fixed_brightness_scale_value = None
    args.fixed_brightness_scale_history = []

    output_path = Path(args.output_path).resolve()
    args.output_path = str(output_path)
    logger = get_logger(output_path)
    logger.info("args: %s", args)
    logger.info("Unified lens training mode schedule: %s", lens_mode_schedule_summary(args))
    logger.info("Unified eval lens mode: %s", args.eval_lens_mode)
    logger.info("Unified backbone path: %s", backbone_path)
    input_config_path = copy_input_config(output_path, args.config, filename="eval_input_config.yaml" if args.eval_only else "input_config.yaml")
    if input_config_path is not None:
        logger.info("Copied input unified config to %s", input_config_path)
    safe_state_fn(args.quiet)
    eval_lpips_model = load_lpips_model(args, logger)
    train_lpips_model = eval_lpips_model
    if args.lpips_loss_weight > 0.0 and not args.eval_only and train_lpips_model is None:
        train_lpips_model = load_lpips_model(args, logger, required=True)
    logger.info("Unified LPIPS training loss weight: %.6g", args.lpips_loss_weight)
    logger.info("Unified backbone adapter: %s", adapter.name)
    supports_condition_injection = bool(getattr(adapter, "supports_condition_injection", True))
    if not supports_condition_injection:
        note = getattr(
            adapter,
            "condition_injection_note",
            "This adapter does not expose a neural decoder injection point.",
        )
        logger.warning("Condition Injection is unsupported for %s: %s", adapter.name, note)
        warnings.warn(f"Condition Injection is unsupported for {adapter.name}: {note}", RuntimeWarning)

    checkpoint_metadata = None
    if args.eval_only:
        if eval_only_results is None or eval_only_ckpt is None:
            raise RuntimeError("eval_only setup did not load checkpoint results")
        checkpoint_mode = eval_only_results.get("mode")
        if checkpoint_mode != "unified_plugin":
            raise ValueError(f"Unified eval_only_ckpt must point to a unified_plugin run, got mode={checkpoint_mode!r}")
        base_model_path_from_ckpt = eval_only_results.get("base_model_path")
        if not base_model_path_from_ckpt:
            raise ValueError(f"Unified eval_only_ckpt is missing base_model_path: {eval_only_ckpt / 'results.json'}")
        checkpoint_iteration = _unified_save_iteration_from_results(eval_only_results, args.load_iteration)
        checkpoint_metadata = (
            _unified_metadata_from_results_or_file(eval_only_ckpt, eval_only_results, checkpoint_iteration)
            if supports_condition_injection
            else eval_only_results.get("plugin_metadata", {})
        )
        args.model_path = str(Path(base_model_path_from_ckpt).expanduser().resolve())
        args.load_iteration = _unified_base_iteration_from_metadata(checkpoint_metadata, args.load_iteration)
        args.save_iteration = checkpoint_iteration
        if supports_condition_injection:
            _apply_unified_layout_from_metadata(args, checkpoint_metadata)
        if not args.distort_mlp_path:
            checkpoint_distort_mlp = eval_only_results.get("distort_mlp_path")
            if not checkpoint_distort_mlp:
                protocol = eval_only_results.get("baseline_distort_mlp_protocol") or {}
                checkpoint_distort_mlp = protocol.get("distort_mlp_path")
            if not checkpoint_distort_mlp:
                raise ValueError(f"Unified eval_only_ckpt is missing distort_mlp_path: {eval_only_ckpt / 'results.json'}")
            args.distort_mlp_path = str(checkpoint_distort_mlp)

    dataset = lp.extract(args)
    dataset.eval_only = bool(args.eval_only)
    dataset.plugin_init_mode_normalized = args.plugin_init_mode_normalized
    base_dataset = copy.copy(dataset)
    base_model_path = dataset.model_path
    layout = build_plugin_layout(args.ss_dim, args.display_dim, args.lens_dim)
    total_dim = total_layout_dim(layout)
    if total_dim <= 0:
        raise ValueError("At least one plug-in dimension must be positive")
    if supports_condition_injection:
        dataset.res_cond_dim = total_dim
        dataset.res_cond_mode = "learned_anchor_multi_plugin"
    pipe = pp.extract(args)
    opt = op.extract(args)
    if supports_condition_injection:
        opt.iterations = args.finetune_iterations
        opt.res_cond_lr = args.res_cond_lr

    distort_mlp = load_distort_mlp_checkpoint(args.distort_mlp_path, logger, required=True)
    if any(parameter.requires_grad for parameter in distort_mlp.parameters()):
        raise RuntimeError("Unified adapter training must use a fixed baseline DistortMLP checkpoint")
    baseline_distort_mlp_protocol = loaded_distort_mlp_protocol_metadata(
        args.distort_mlp_path,
        logger,
        expected_backbone=args.backbone,
    )
    args.baseline_distort_mlp_protocol = baseline_distort_mlp_protocol

    scene_load_iteration = args.load_iteration
    if args.eval_only and supports_condition_injection:
        dataset.xrgs_base_model_path = str(base_model_path)
        dataset.model_path = str(eval_only_ckpt)
        scene_load_iteration = args.save_iteration
    gaussians, scene = adapter.load_scene(dataset, iteration=scene_load_iteration)
    if args.eval_only and supports_condition_injection:
        _override_backbone_render_step(gaussians, args.load_iteration, "plugin_metadata.base_loaded_iteration")
    _log_backbone_render_step(logger, "unified", gaussians)
    scene.model_path = str(output_path)
    gaussians.spatial_lr_scale = scene.cameras_extent

    optics_input_size, optics_output_size = resolve_dataset_lens_optics_sizes(
        scene.getTrainCameras(),
        pad_ratio=args.lens_pad_ratio,
    )
    optics_wrapper = load_optics_wrapper(
        optics_sim_path=args.optics_sim_path,
        psf_dir=args.psf_dir,
        psf_backend=args.psf_backend,
        input_size=optics_input_size,
        output_size=optics_output_size,
        verbose=False,
    )
    logger.info(
        "Loaded optics wrapper with dataset-derived input/output sizes %d/%d: %s",
        optics_input_size,
        optics_output_size,
        optics_wrapper,
    )
    adapter.set_plugin_layout(gaussians, layout)
    inject_dims = adapter.validate_inject_dim(gaussians, args.inject_dim)
    disabled_modules = disabled_plugin_modules(args, layout)
    logger.info("Loaded %s base %s at iteration %s", "eval-only" if args.eval_only else "vanilla", base_model_path, scene.loaded_iter)
    logger.info("Unified plug-in layout: %s", layout)
    logger.info("Unified inject_dims: %s", inject_dims)
    logger.info("Disabled unified plug-in modules: %s", sorted(disabled_modules))
    config_path = write_effective_config(
        output_path,
        args,
        layout,
        total_dim,
        inject_dims,
        filename="eval_config.yaml" if args.eval_only else "config.yaml",
    )
    logger.info("Wrote effective unified config to %s", config_path)
    if supports_condition_injection and args.eval_only:
        metadata = load_plugin_metadata(eval_only_ckpt, scene.loaded_iter)
        validate_resume_metadata(metadata, layout, total_dim, inject_dims)
        adapter.require_unified_plugin_checkpoint(eval_only_ckpt, scene.loaded_iter, total_dim)
        logger.info("Loaded eval-only unified checkpoint from %s", plugin_metadata_path(eval_only_ckpt, scene.loaded_iter))
    elif supports_condition_injection:
        if args.plugin_init_mode_normalized == "resume_unified":
            metadata = load_plugin_metadata(base_model_path, scene.loaded_iter)
            validate_resume_metadata(metadata, layout, total_dim, inject_dims)
            adapter.require_unified_plugin_checkpoint(base_model_path, scene.loaded_iter, total_dim)
            logger.info("Resumed unified plug-in checkpoint with metadata from %s", plugin_metadata_path(base_model_path, scene.loaded_iter))
        elif args.plugin_init_mode_normalized == "strict_warm_start":
            gaussians.reset_res_cond_plugin_parameters(args.projection_init_std)
            loaded = []
            if "ss" in layout and "ss" not in disabled_modules:
                loaded.append(adapter.load_plugin_slice(gaussians, dataset, "ss", args.ss_plugin_path, layout["ss"], logger, args.ss_plugin_iteration))
            if "display" in layout and "display" not in disabled_modules:
                loaded.append(adapter.load_plugin_slice(gaussians, dataset, "display", args.display_plugin_path, layout["display"], logger, args.display_plugin_iteration))
            if "lens" in layout and "lens" not in disabled_modules:
                loaded.append(adapter.load_plugin_slice(gaussians, dataset, "lens", args.lens_plugin_path, layout["lens"], logger, args.lens_plugin_iteration))
            if not any(loaded):
                raise ValueError("plugin_init_mode=strict_warm_start but no enabled plug-in slices were loaded")
        else:
            gaussians.reset_res_cond_plugin_parameters(args.projection_init_std)
            logger.info(
                "Using clean random plug-in initialization: zero endpoints, truncated-normal hidden projections"
            )
        adapter.zero_disabled_plugin_slices(gaussians, layout, disabled_modules)
        gaussians.training_setup(opt)
        adapter.set_plugin_layout(gaussians, layout)
        adapter.register_disabled_plugin_slice_hooks(gaussians, layout, disabled_modules)
        for attr in ["opacity_accum", "offset_gradient_accum", "offset_denom", "anchor_demon"]:
            if hasattr(gaussians, attr):
                delattr(gaussians, attr)
        optimizer_groups = [group.get("name", "") for group in gaussians.optimizer.param_groups]
        expected_groups = set(getattr(adapter, "condition_parameter_groups", {
            "res_cond_anchor",
            "res_cond_opacity_proj",
            "res_cond_cov_proj",
            "res_cond_color_proj",
        }))
        if set(optimizer_groups) != expected_groups:
            raise ValueError(f"Unified optimizer must train only condition parameters, got groups: {optimizer_groups}")
        logger.info("Unified optimizer parameter groups: %s", optimizer_groups)
    elif args.plugin_init_mode_normalized in {"resume_unified", "strict_warm_start"}:
        logger.warning("Ignoring plugin_init_mode=%s because %s has no Condition Injection.", args.plugin_init_mode, adapter.name)
    logger.info("Trainable plug-in parameter count: %d", adapter.trainable_parameter_count(gaussians))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    fixed_brightness_scale = None
    if args.brightness_match_mode == "fixed_train0":
        fixed_brightness_scale = set_fixed_brightness_scale(
            scene,
            optics_wrapper,
            args,
            args.lens_mode,
            logger,
            "initial",
        )
    if supports_condition_injection and not args.eval_only:
        viewpoint_stack = None
        ema_loss = 0.0
        final_losses = {"loss": None, "l1": None, "mse": None, "ssim": None, "lpips": None}
        progress = tqdm(range(1, args.finetune_iterations + 1), desc="Unified plug-in fine-tune")
        gaussians.train()
        psf_start = psf_start_iteration(args)

        for iteration in range(1, args.finetune_iterations + 1):
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            camera = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
            downscale = sample_downscale(args)
            saving = sample_target_saving(args)
            set_unified_conditions(camera, downscale, saving, args)

            gt_image = camera.original_image[:3, :, :].cuda().clamp(0.0, 1.0)
            lens_mode = training_lens_mode(args, iteration)
            if (
                args.brightness_match_mode == "fixed_train0"
                and psf_start is not None
                and iteration == psf_start
                and lens_mode == "psf"
            ):
                fixed_brightness_scale = set_fixed_brightness_scale(
                    scene,
                    optics_wrapper,
                    args,
                    "psf",
                    logger,
                    "psf_switch",
                )
                gaussians.train()

            image = adapter.render(camera, gaussians, pipe, background)
            post = PostProcessing(
                image=image,
                reference_image=gt_image,
                gt_image=gt_image,
                downscale=downscale,
                saving=saving,
                args=args,
                distort_mlp=distort_mlp,
                pad_value=0.0,
            )
            lens_input = post.lens_input
            gt_canvas = post.gt_canvas
            gt_lens = lens_gt_from_display_canvas(gt_canvas, args)
            image_lens = apply_optics_model(
                lens_input,
                optics_wrapper,
                mode=lens_mode,
                pad_value=0.0,
            )
            image_lens = apply_brightness_policy(image_lens, gt_lens, args)
            image_loss, gt_loss, _loss_crop_box = crop_lens_metric_region(
                image_lens.clamp(0.0, 1.0),
                gt_lens.clamp(0.0, 1.0),
                gt_canvas,
                downscale,
            )
            quality_loss, ll1, mse_loss, ssim_loss = lens_quality_loss(
                image_loss,
                gt_loss,
                opt.lambda_dssim,
            )
            lpips_loss = image_loss.new_tensor(0.0)
            if args.lpips_loss_weight > 0.0:
                if train_lpips_model is None:
                    raise RuntimeError("lpips_loss_weight > 0 requires an LPIPS model")
                lpips_loss = train_lpips_model(image_loss.unsqueeze(0), gt_loss.unsqueeze(0)).mean()
            loss = quality_loss + args.mse_loss_weight * mse_loss + args.lpips_loss_weight * lpips_loss
            final_losses = {
                "loss": float(loss.detach()),
                "l1": float(ll1.detach()),
                "mse": float(mse_loss.detach()),
                "ssim": float(ssim_loss.detach()),
                "lpips": float(lpips_loss.detach()),
            }

            gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gaussians.optimizer.step()

            ema_loss = 0.4 * float(loss.detach()) + 0.6 * ema_loss
            if iteration % 10 == 0 or iteration == args.finetune_iterations:
                progress.set_postfix(
                    {
                        "loss": f"{ema_loss:.5f}",
                        "l1": f"{float(ll1.detach()):.5f}",
                        "mse": f"{float(mse_loss.detach()):.6f}",
                        "ssim": f"{float(ssim_loss.detach()):.5f}",
                        "lpips": f"{float(lpips_loss.detach()):.5f}",
                        "p": f"{float(display_power(lens_input).detach()):.4f}/{float(target_power_from_saving(post.reference_display_canvas, saving).detach()):.4f}",
                        "s": f"{saving:.2f}",
                        "r": f"{downscale:.2f}",
                        "mode": lens_mode,
                    }
                )
                progress.update(10 if iteration % 10 == 0 else iteration % 10)
        progress.close()
    else:
        final_losses = {"loss": None, "l1": None, "mse": None, "ssim": None, "lpips": None}

    save_iteration = args.save_iteration if args.save_iteration > 0 else scene.loaded_iter
    metadata_path = None
    plugin_metadata = checkpoint_metadata if args.eval_only and supports_condition_injection else None
    if not args.eval_only:
        if supports_condition_injection:
            scene.save(save_iteration)
        plugin_metadata = build_plugin_metadata(
            args,
            adapter.name,
            layout,
            total_dim,
            inject_dims,
            base_model_path,
            scene.loaded_iter,
            save_iteration,
            disabled_modules,
        )
        plugin_metadata["condition_injection"] = {
            "supported": supports_condition_injection,
            "reason": None
            if supports_condition_injection
            else getattr(adapter, "condition_injection_note", "Adapter does not expose Condition Injection."),
        }
        if not supports_condition_injection:
            plugin_metadata["baseline_pipeline"]["definition"] = (
                "explicit 3DGS renderer plus per-scene end-to-end baseline DistortMLP; "
                "unified condition injection is unsupported"
            )
        metadata_path = write_plugin_metadata(output_path, save_iteration, plugin_metadata)
        if supports_condition_injection:
            logger.info("Saved unified plug-in checkpoint at iteration %d", save_iteration)
        else:
            logger.info("Skipped unified plug-in checkpoint save for adapter without Condition Injection")
    else:
        logger.info("Eval-only run; skipped unified plug-in checkpoint save")
        if supports_condition_injection and eval_only_ckpt is not None:
            metadata_path = plugin_metadata_path(eval_only_ckpt, save_iteration)
    if metadata_path is not None:
        if args.eval_only:
            logger.info("Using eval-only unified plug-in metadata from %s", metadata_path)
        else:
            logger.info("Wrote unified plug-in metadata to %s", metadata_path)

    if args.train_only and not args.eval_only:
        summary = {
            "base_model_path": base_model_path,
            "save_iteration": save_iteration,
            "base_loaded_iteration": int(getattr(gaussians, "_xrgs_base_loaded_iteration", scene.loaded_iter or 0)),
            "backbone_render_step": _backbone_render_step_value(gaussians),
            "backbone_render_step_source": _backbone_render_step_source(gaussians),
            "layout": layout,
            "plugin_metadata": plugin_metadata,
            "plugin_metadata_path": None if metadata_path is None else str(metadata_path),
            "config_path": str(config_path),
            "config_source": args.config,
            "input_config_path": None if input_config_path is None else str(input_config_path),
            "output_path": str(output_path),
            "mode": args.mode,
            "backbone": args.backbone,
            "backbone_path": str(backbone_path),
            "eval_only": False,
            "eval_only_ckpt": None,
            "eval_only_source_results": None,
            "eval_only_source_metadata": None,
            "train_only": True,
            "inject_dims": inject_dims,
            "disabled_modules": sorted(disabled_modules),
            "finetune_iterations": args.finetune_iterations,
            "lens_mode": args.lens_mode,
            "lens_psf_iterations": args.lens_psf_iterations,
            "eval_lens_mode": args.eval_lens_mode,
            "eval_max_views": args.eval_max_views,
            "eval_savings": args.eval_savings,
            "eval_downscales": args.eval_downscales,
            "eval_power_mode": args.eval_power_mode,
            "eval_image_mode": args.eval_image_mode,
            "eval_image_dir": args.eval_image_dir,
            "eval_image_root": None,
            "eval_image_max_views": args.eval_image_max_views,
            "eval_lpips": args.eval_lpips,
            "lpips_net": args.lpips_net,
            "lens_resolution_protocol": "downsample_resampled_to_display_then_padded_lens",
            "display_canvas_placement": "full_extent_resample",
            "ss_downsample_mode": args.ss_downsample_mode,
            "resample_gt_display_canvas": args.resample_gt_display_canvas,
            "use_super_sampling": args.use_super_sampling,
            "render_resolution_policy": args.render_resolution_policy,
            "lens_train_mode_schedule": lens_mode_schedule_summary(args),
            "distort_mlp_path": args.distort_mlp_path,
            "distort_mlp_trainable_in_unified": False,
            "canonical_distort_mlp_path": args.canonical_distort_mlp_path or None,
            "canonical_distort_mlp_protocol": None,
            "eval_reference_distort_mlp": None,
            "baseline_distort_mlp_protocol": baseline_distort_mlp_protocol,
            "baseline_distort_mlp_backbone": baseline_distort_mlp_protocol["baseline_backbone"],
            "baseline_distort_mlp_mode": baseline_distort_mlp_protocol["baseline_mode"],
            "baseline_distort_mlp_result_path": baseline_distort_mlp_protocol["result_path"],
            "plugin_init_mode": args.plugin_init_mode,
            "condition_injection_supported": supports_condition_injection,
            "condition_injection_note": None
            if supports_condition_injection
            else getattr(adapter, "condition_injection_note", "Adapter does not expose Condition Injection."),
            "ss_plugin_path": args.ss_plugin_path,
            "display_plugin_path": args.display_plugin_path,
            "lens_plugin_path": args.lens_plugin_path,
            "brightness_match": args.brightness_match,
            "brightness_match_mode": args.brightness_match_mode,
            "fixed_brightness_scale": args.fixed_brightness_scale_value,
            "eval_fixed_brightness_scale": getattr(args, "eval_fixed_brightness_scale_value", None),
            "display_mapping": "post_distortion_direct_linear_rgb_target_power",
            "display_mapping_reference": "gt_distorted_image",
            "display_mapping_reference_distort_mlp": None,
            "target_saving_max": args.target_saving_max,
            "mse_loss_weight": args.mse_loss_weight,
            "lpips_loss_weight": args.lpips_loss_weight,
            "res_cond_lr": args.res_cond_lr,
            "final_train_loss": final_losses,
            "eval": None,
            "eval_skipped": True,
            "eval_skip_reason": "train_only",
        }
        (output_path / "results.json").write_text(json.dumps(jsonable(summary), indent=2, allow_nan=False) + "\n")
        logger.info("Train-only run saved unified plug-in checkpoint and skipped evaluation")
        return

    canonical_distort_mlp, canonical_distort_mlp_protocol = _load_canonical_distort_mlp(
        args,
        logger,
        load_distort_mlp_checkpoint,
        loaded_distort_mlp_protocol_metadata,
    )
    base_gaussians, base_scene = adapter.load_scene(base_dataset, iteration=args.load_iteration)
    _log_backbone_render_step(logger, "baseline-eval", base_gaussians)
    eval_results = evaluate_grid(
        adapter,
        scene,
        gaussians,
        base_scene,
        base_gaussians,
        pipe,
        optics_wrapper,
        distort_mlp,
        background,
        args,
        eval_lpips_model,
        mode=EVAL_MODE_OURS,
        canonical_distort_mlp=canonical_distort_mlp,
    )
    summary = {
        "base_model_path": base_model_path,
        "save_iteration": save_iteration,
        "base_loaded_iteration": int(getattr(gaussians, "_xrgs_base_loaded_iteration", scene.loaded_iter or 0)),
        "backbone_render_step": _backbone_render_step_value(gaussians),
        "backbone_render_step_source": _backbone_render_step_source(gaussians),
        "layout": layout,
        "plugin_metadata": plugin_metadata,
        "plugin_metadata_path": None if metadata_path is None else str(metadata_path),
        "config_path": str(config_path),
        "config_source": args.config,
        "input_config_path": None if input_config_path is None else str(input_config_path),
        "output_path": str(output_path),
        "mode": args.mode,
        "backbone": args.backbone,
        "backbone_path": str(backbone_path),
        "eval_only": args.eval_only,
        "eval_only_ckpt": None if eval_only_ckpt is None else str(eval_only_ckpt),
        "eval_only_source_results": None
        if eval_only_ckpt is None or not (eval_only_ckpt / "results.json").exists()
        else str(eval_only_ckpt / "results.json"),
        "eval_only_source_metadata": None if eval_only_results is None else eval_only_results.get("eval_only_source_metadata"),
        "inject_dims": inject_dims,
        "disabled_modules": sorted(disabled_modules),
        "finetune_iterations": args.finetune_iterations,
        "lens_mode": args.lens_mode,
        "lens_psf_iterations": args.lens_psf_iterations,
        "eval_lens_mode": args.eval_lens_mode,
        "eval_max_views": args.eval_max_views,
        "eval_savings": args.eval_savings,
        "eval_downscales": args.eval_downscales,
        "eval_power_mode": args.eval_power_mode,
        "eval_image_mode": args.eval_image_mode,
        "eval_image_dir": args.eval_image_dir,
        "eval_image_root": _eval_image_root_for_summary(args, output_path, "ours"),
        "eval_image_max_views": args.eval_image_max_views,
        "eval_image_bit_depth": args.eval_image_bit_depth,
        "eval_image_save_aux": args.eval_image_save_aux,
        "eval_lpips": args.eval_lpips,
        "lpips_net": args.lpips_net,
        "lens_resolution_protocol": "downsample_resampled_to_display_then_padded_lens",
        "display_canvas_placement": "full_extent_resample",
        "ss_downsample_mode": args.ss_downsample_mode,
        "resample_gt_display_canvas": args.resample_gt_display_canvas,
        "use_super_sampling": args.use_super_sampling,
        "render_resolution_policy": args.render_resolution_policy,
        "lens_train_mode_schedule": lens_mode_schedule_summary(args),
        "distort_mlp_path": args.distort_mlp_path,
        "distort_mlp_trainable_in_unified": False,
        "canonical_distort_mlp_path": args.canonical_distort_mlp_path or None,
        "canonical_distort_mlp_protocol": canonical_distort_mlp_protocol,
        "eval_reference_distort_mlp": "canonical_distort_mlp",
        "baseline_distort_mlp_protocol": baseline_distort_mlp_protocol,
        "baseline_distort_mlp_backbone": baseline_distort_mlp_protocol["baseline_backbone"],
        "baseline_distort_mlp_mode": baseline_distort_mlp_protocol["baseline_mode"],
        "baseline_distort_mlp_result_path": baseline_distort_mlp_protocol["result_path"],
        "plugin_init_mode": args.plugin_init_mode,
        "condition_injection_supported": supports_condition_injection,
        "condition_injection_note": None
        if supports_condition_injection
        else getattr(adapter, "condition_injection_note", "Adapter does not expose Condition Injection."),
        "ss_plugin_path": args.ss_plugin_path,
        "display_plugin_path": args.display_plugin_path,
        "lens_plugin_path": args.lens_plugin_path,
        "brightness_match": args.brightness_match,
        "brightness_match_mode": args.brightness_match_mode,
        "fixed_brightness_scale": args.fixed_brightness_scale_value,
        "eval_fixed_brightness_scale": getattr(args, "eval_fixed_brightness_scale_value", None),
        "display_mapping": "post_distortion_direct_linear_rgb_target_power",
        "display_mapping_reference": "gt_distorted_image",
        "display_mapping_reference_distort_mlp": "canonical_distort_mlp",
        "target_saving_max": args.target_saving_max,
        "mse_loss_weight": args.mse_loss_weight,
        "lpips_loss_weight": args.lpips_loss_weight,
        "res_cond_lr": args.res_cond_lr,
        "final_train_loss": final_losses,
        "eval": eval_results,
    }
    (output_path / "results.json").write_text(json.dumps(jsonable(summary), indent=2, allow_nan=False) + "\n")
    logger.info("Wrote %s", output_path / "results.json")


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


def _backbone_render_step_attr(gaussians) -> tuple[str, str, str] | None:
    for name, label in (("hac", "HAC++"), ("contextgs", "ContextGS")):
        step_attr = f"_xrgs_{name}_render_step"
        if hasattr(gaussians, step_attr):
            return step_attr, f"{step_attr}_source", label
    return None


def _backbone_render_step_value(gaussians) -> int | None:
    attrs = _backbone_render_step_attr(gaussians)
    if attrs is None:
        return None
    step = getattr(gaussians, attrs[0], None)
    return None if step is None else int(step)


def _backbone_render_step_source(gaussians) -> str | None:
    attrs = _backbone_render_step_attr(gaussians)
    if attrs is None:
        return None
    return getattr(gaussians, attrs[1], None)


def _log_backbone_render_step(logger, label: str, gaussians) -> None:
    attrs = _backbone_render_step_attr(gaussians)
    if attrs is None:
        return
    step_attr, source_attr, backbone_label = attrs
    step = getattr(gaussians, step_attr, None)
    if step is None:
        return
    logger.info(
        "%s %s render step: %d from %s",
        backbone_label,
        label,
        int(step),
        getattr(gaussians, source_attr, "unknown"),
    )


def _override_backbone_render_step(gaussians, step: int, source: str) -> None:
    attrs = _backbone_render_step_attr(gaussians)
    if attrs is None:
        return
    step = int(step)
    if step <= 0:
        raise ValueError(f"Backbone render step override must be positive, got {step} from {source}")
    step_attr, source_attr, _backbone_label = attrs
    gaussians._xrgs_base_loaded_iteration = step
    setattr(gaussians, step_attr, step)
    setattr(gaussians, source_attr, source)


def _baseline_lens_mode(args, iteration: int) -> str:
    psf_iterations = max(0, int(args.lens_psf_iterations))
    if psf_iterations <= 0:
        return args.lens_mode
    psf_start = max(1, args.finetune_iterations - psf_iterations + 1)
    return "psf" if iteration >= psf_start else args.lens_mode


def run_baseline_distort_mlp(argv: list[str], config_defaults: dict, config_source: str | None, backbone_name: str, backbone_path: Path) -> None:
    if _should_print_unified_config_defaults(argv):
        effective_defaults = {**baseline_distort_config_defaults(), **config_defaults}
        effective_defaults["mode"] = "baseline_distort_mlp"
        effective_defaults["backbone"] = backbone_name
        effective_defaults["backbone_path"] = str(backbone_path)
        print(json.dumps(effective_defaults, indent=2))
        return
    (
        model_params_cls,
        optimization_params_cls,
        pipeline_params_cls,
        safe_state_fn,
        adapter,
        _description,
    ) = _load_backbone_runtime(backbone_name, backbone_path)
    from xrgs_utils.lens_utils import (
        DistortMLP,
        apply_lens_pipeline,
        load_optics_wrapper,
        resolve_dataset_lens_optics_sizes,
    )
    from xrgs_unified.config import copy_input_config, jsonable, write_json
    from xrgs_unified.display_protocol import crop_lens_metric_region, display_canvas_pair, lens_gt_from_display_canvas
    from xrgs_unified.evaluation import EVAL_MODE_DISTORTION_MLP, evaluate_grid, load_lpips_model
    from xrgs_unified.runtime import (
        get_logger,
        l1_loss,
        loaded_distort_mlp_protocol_metadata,
        load_distort_mlp_checkpoint,
        scaffold_distort_mlp_protocol_metadata,
        ssim,
    )

    parser = ArgumentParser(description="Train the XRGS DistortMLP-only baseline on a frozen rendering backbone")
    parser.add_argument("--config", default=config_source)
    lp = model_params_cls(parser)
    op = optimization_params_cls(parser)
    pp = pipeline_params_cls(parser)
    add_unified_config_args(parser)
    parser.set_defaults(**{**baseline_distort_config_defaults(), **config_defaults})
    parser.set_defaults(mode="baseline_distort_mlp", backbone=backbone_name, backbone_path=str(backbone_path))
    args = parser.parse_args(argv)
    _normalize_render_resolution_policy(args)
    if not args.output_path:
        raise ValueError("Baseline output_path must be set in config.yaml or via --output_path")
    if not args.eval_lens_mode:
        args.eval_lens_mode = args.lens_mode
    eval_only_ckpt, eval_only_results = _prepare_eval_only_ckpt(args, "Baseline DistortMLP")
    if args.eval_only:
        if eval_only_results is None or eval_only_ckpt is None:
            raise RuntimeError("eval_only setup did not load checkpoint results")
        checkpoint_mode = eval_only_results.get("mode")
        if checkpoint_mode != "baseline_distort_mlp":
            raise ValueError(f"Baseline eval_only_ckpt must point to a baseline_distort_mlp run, got mode={checkpoint_mode!r}")
        base_model_path_from_ckpt = _resolve_migrated_run_path(
            eval_only_results.get("base_model_path"),
            args.model_path,
            "Baseline eval_only backbone checkpoint",
        )
        distort_mlp_path_from_ckpt = _resolve_migrated_checkpoint_file(
            eval_only_ckpt,
            eval_only_results,
            "distort_mlp_path",
            "distort_mlp.pth",
            "Baseline eval_only DistortMLP checkpoint",
        )
        args.model_path = str(base_model_path_from_ckpt)
        args.load_iteration = _checkpoint_iteration_from_results(eval_only_results, args.load_iteration)
    args.brightness_match = args.brightness_match_mode != "off"
    args.fixed_brightness_scale_value = None
    args.fixed_brightness_scale_history = []

    safe_state_fn(args.quiet)
    output_path = Path(args.output_path).resolve()
    args.output_path = str(output_path)
    logger = get_logger(output_path)
    logger.info("args: %s", args)
    logger.info("Baseline DistortMLP backbone path: %s", backbone_path)
    input_config_path = copy_input_config(output_path, args.config)
    if input_config_path is not None:
        logger.info("Copied input baseline config to %s", input_config_path)

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)
    gaussians, scene = adapter.load_scene(dataset, iteration=args.load_iteration)
    _freeze_backbone(gaussians)

    optics_input_size, optics_output_size = resolve_dataset_lens_optics_sizes(
        scene.getTrainCameras(),
        pad_ratio=args.lens_pad_ratio,
    )
    optics_wrapper = load_optics_wrapper(
        optics_sim_path=args.optics_sim_path,
        psf_dir=args.psf_dir,
        psf_backend=args.psf_backend,
        input_size=optics_input_size,
        output_size=optics_output_size,
        verbose=False,
    )
    logger.info(
        "Loaded optics wrapper with dataset-derived input/output sizes %d/%d: %s",
        optics_input_size,
        optics_output_size,
        optics_wrapper,
    )
    if args.eval_only:
        distort_mlp = DistortMLP().cuda()
        distort_mlp.load_state_dict(torch.load(distort_mlp_path_from_ckpt, map_location="cuda"))
        distort_mlp.eval()
        for parameter in distort_mlp.parameters():
            parameter.requires_grad_(False)
    else:
        distort_mlp = DistortMLP().cuda().train()
        optimizer = torch.optim.Adam(distort_mlp.parameters(), lr=args.distort_mlp_lr)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    ema_loss = 0.0
    final_losses = {"loss": None, "l1": None, "ssim": None}
    if not args.eval_only:
        progress = tqdm(range(1, args.finetune_iterations + 1), desc="Baseline DistortMLP")

        for iteration in range(1, args.finetune_iterations + 1):
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            camera = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
            gt_image = camera.original_image[:3, :, :].cuda().clamp(0.0, 1.0)
            with torch.no_grad():
                image = adapter.render_baseline(camera, gaussians, pipe, background)
                image_canvas, gt_canvas = display_canvas_pair(image, gt_image, 1.0, args)
                gt_lens = lens_gt_from_display_canvas(gt_canvas, args)

            image_lens = apply_lens_pipeline(
                image_canvas,
                optics_wrapper,
                mode=_baseline_lens_mode(args, iteration),
                distort_mlp=distort_mlp,
                pad_value=0.0,
            )
            image_loss, gt_loss, _loss_crop_box = crop_lens_metric_region(
                image_lens.clamp(0.0, 1.0),
                gt_lens.clamp(0.0, 1.0),
                gt_canvas,
                1.0,
            )
            ll1 = l1_loss(image_loss, gt_loss)
            ssim_loss = 1.0 - ssim(image_loss, gt_loss)
            loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * ssim_loss
            final_losses = {
                "loss": float(loss.detach()),
                "l1": float(ll1.detach()),
                "ssim": float(ssim_loss.detach()),
            }

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ema_loss = 0.4 * float(loss.detach()) + 0.6 * ema_loss
            if iteration % 10 == 0 or iteration == args.finetune_iterations:
                progress.set_postfix(
                    {
                        "loss": f"{ema_loss:.5f}",
                        "l1": f"{float(ll1.detach()):.5f}",
                        "mode": _baseline_lens_mode(args, iteration),
                    }
                )
                progress.update(10 if iteration % 10 == 0 else iteration % 10)
        progress.close()
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    distort_path = output_path / "distort_mlp.pth"
    if args.eval_only:
        distort_path = distort_mlp_path_from_ckpt
    else:
        torch.save(distort_mlp.state_dict(), distort_path)
        logger.info("Saved DistortMLP checkpoint before evaluation to %s", distort_path)

    if args.train_only and not args.eval_only:
        summary = {
            "mode": "baseline_distort_mlp",
            "baseline": "distort_mlp_only",
            "backbone": backbone_name,
            "backbone_path": str(backbone_path),
            "eval_only": False,
            "eval_only_ckpt": None,
            "eval_only_source_results": None,
            "train_only": True,
            "base_model_path": dataset.model_path,
            "config_source": args.config,
            "input_config_path": None if input_config_path is None else str(input_config_path),
            "distort_mlp_path": str(distort_path),
            "canonical_distort_mlp_path": args.canonical_distort_mlp_path or None,
            "canonical_distort_mlp_protocol": None,
            "eval_reference_distort_mlp": None,
            "distort_mlp_lr": args.distort_mlp_lr,
            "distort_mlp_protocol": scaffold_distort_mlp_protocol_metadata(args, fixed_in_adapter=False),
            "lambda_dssim": opt.lambda_dssim,
            "finetune_iterations": args.finetune_iterations,
            "lens_mode": args.lens_mode,
            "lens_psf_iterations": args.lens_psf_iterations,
            "eval_lens_mode": args.eval_lens_mode,
            "eval_max_views": args.eval_max_views,
            "eval_savings": args.eval_savings,
            "eval_downscales": args.eval_downscales,
            "eval_power_mode": args.eval_power_mode,
            "eval_image_mode": args.eval_image_mode,
            "eval_image_dir": args.eval_image_dir,
            "eval_image_root": None,
            "eval_image_max_views": args.eval_image_max_views,
            "eval_image_bit_depth": args.eval_image_bit_depth,
            "eval_image_save_aux": args.eval_image_save_aux,
            "eval_lpips": args.eval_lpips,
            "lpips_net": args.lpips_net,
            "brightness_match": args.brightness_match,
            "brightness_match_mode": args.brightness_match_mode,
            "eval_fixed_brightness_scale": getattr(args, "eval_fixed_brightness_scale_value", None),
            "ss_downsample_mode": args.ss_downsample_mode,
            "resample_gt_display_canvas": args.resample_gt_display_canvas,
            "use_super_sampling": args.use_super_sampling,
            "render_resolution_policy": args.render_resolution_policy,
            "display_mapping_reference": "gt_distorted_image",
            "display_mapping_reference_distort_mlp": None,
            "final_train_loss": final_losses,
            "eval": None,
            "eval_skipped": True,
            "eval_skip_reason": "train_only",
        }
        write_json(output_path / "results.json", jsonable(summary))
        logger.info("Train-only run saved DistortMLP to %s", distort_path)
        return

    canonical_distort_mlp, canonical_distort_mlp_protocol = _load_canonical_distort_mlp(
        args,
        logger,
        load_distort_mlp_checkpoint,
        loaded_distort_mlp_protocol_metadata,
    )
    lpips_model = load_lpips_model(args, logger)
    distort_mlp.eval()
    eval_results = evaluate_grid(
        adapter=adapter,
        scene=scene,
        gaussians=gaussians,
        base_scene=scene,
        base_gaussians=gaussians,
        pipe=pipe,
        optics_wrapper=optics_wrapper,
        distort_mlp=distort_mlp,
        background=background,
        args=args,
        lpips_model=lpips_model,
        mode=EVAL_MODE_DISTORTION_MLP,
        canonical_distort_mlp=canonical_distort_mlp,
    )
    summary = {
        "mode": "baseline_distort_mlp",
        "baseline": "distort_mlp_only",
        "backbone": backbone_name,
        "backbone_path": str(backbone_path),
        "eval_only": args.eval_only,
        "eval_only_ckpt": None if eval_only_ckpt is None else str(eval_only_ckpt),
        "eval_only_source_results": None if eval_only_ckpt is None else str(eval_only_ckpt / "results.json"),
        "train_only": args.train_only,
        "base_model_path": dataset.model_path,
        "config_source": args.config,
        "input_config_path": None if input_config_path is None else str(input_config_path),
        "distort_mlp_path": str(distort_path),
        "canonical_distort_mlp_path": args.canonical_distort_mlp_path or None,
        "canonical_distort_mlp_protocol": canonical_distort_mlp_protocol,
        "eval_reference_distort_mlp": "canonical_distort_mlp",
        "distort_mlp_lr": args.distort_mlp_lr,
        "distort_mlp_protocol": scaffold_distort_mlp_protocol_metadata(args, fixed_in_adapter=False),
        "lambda_dssim": opt.lambda_dssim,
        "finetune_iterations": args.finetune_iterations,
        "lens_mode": args.lens_mode,
        "lens_psf_iterations": args.lens_psf_iterations,
        "eval_lens_mode": args.eval_lens_mode,
        "eval_max_views": args.eval_max_views,
        "eval_savings": args.eval_savings,
        "eval_downscales": args.eval_downscales,
        "eval_power_mode": args.eval_power_mode,
        "eval_image_mode": args.eval_image_mode,
        "eval_image_dir": args.eval_image_dir,
        "eval_image_root": _eval_image_root_for_summary(args, output_path, "baseline"),
        "eval_image_max_views": args.eval_image_max_views,
        "eval_image_bit_depth": args.eval_image_bit_depth,
        "eval_image_save_aux": args.eval_image_save_aux,
        "eval_lpips": args.eval_lpips,
        "lpips_net": args.lpips_net,
        "brightness_match": args.brightness_match,
        "brightness_match_mode": args.brightness_match_mode,
        "eval_fixed_brightness_scale": getattr(args, "eval_fixed_brightness_scale_value", None),
        "ss_downsample_mode": args.ss_downsample_mode,
        "resample_gt_display_canvas": args.resample_gt_display_canvas,
        "use_super_sampling": args.use_super_sampling,
        "render_resolution_policy": args.render_resolution_policy,
        "display_mapping_reference": "gt_distorted_image",
        "display_mapping_reference_distort_mlp": "canonical_distort_mlp",
        "final_train_loss": final_losses,
        "eval": eval_results,
    }
    write_json(output_path / "results.json", jsonable(summary))
    if args.eval_only:
        logger.info("Evaluated existing DistortMLP from %s", distort_path)
    else:
        logger.info("Saved DistortMLP to %s", distort_path)


def main() -> None:
    config_defaults, config_source = load_config_defaults(sys.argv[1:], TOP_LEVEL_ARG_SPECS)
    mode, backbone_name, backbone_path = _top_level_from_args(sys.argv[1:], config_defaults)
    if mode == "baseline_distort_mlp":
        run_baseline_distort_mlp(sys.argv[1:], config_defaults, config_source, backbone_name, backbone_path)
    elif mode == "unified_plugin":
        run_unified_plugin(sys.argv[1:], config_defaults, config_source, backbone_name, backbone_path)
    elif mode == "unified_renderer":
        run_unified_renderer_entry(sys.argv[1:], config_defaults, config_source, backbone_name, backbone_path)
    else:
        raise ValueError(f"Unsupported XRGS mode: {mode}")


if __name__ == "__main__":
    main()
