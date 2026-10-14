#!/usr/bin/env python3
"""Run the portable Scaffold-GS stump/materials release from any working directory."""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCENES = {"stump": "mip360", "materials": "nerf_synthetic"}

def float_list(value):
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated numbers") from exc
    if not values or not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("Values must be finite numbers")
    return values


def main(argv=None, *, forced_action=None, training=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    if training:
        parser.add_argument("--stage", choices=["plugin", "distort"], default="plugin", help="Train the condition plugin or its fixed DistortMLP baseline")
    elif forced_action is None:
        parser.add_argument("--action", choices=["inference", "eval", "train-plugin", "train-distort"], default="inference")
    else:
        parser.set_defaults(action=forced_action)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset", help="Root containing mip360/stump and nerf_synthetic/materials")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke", action="store_true", help="One view at 1x and 0%% saving; training also uses two iterations")
    parser.add_argument("--max-views", type=int, default=0, help="0 evaluates all test views")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Unified checkpoint directory for inference/evaluation; defaults to the supplied scene model")
    parser.add_argument("--distort-checkpoint", type=Path, help="Fixed DistortMLP .pth for plugin training, e.g. the output of --stage distort")
    parser.add_argument("--iterations", type=int, help="Training iterations; defaults to 10000 (two with --smoke)")
    parser.add_argument("--psf-iterations", type=int, help="Final baseline training iterations using PSF; default min(1000, iterations)")
    parser.add_argument("--savings", type=float_list, help="Comma-separated display saving fractions, within 0..0.8")
    parser.add_argument("--downscales", type=float_list, help="Comma-separated resolution factors, within 1..8")
    parser.add_argument("--evaluate", action="store_true", help="Also compute quality metrics after inference or training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="Use existing model and PSF assets without downloading")
    args = parser.parse_args(argv)
    if training:
        args.action = "train-" + args.stage
    is_training = args.action.startswith("train")
    if args.max_views < 0:
        parser.error("--max-views must be nonnegative")
    if args.iterations is not None and (not is_training or args.iterations <= 0):
        parser.error("--iterations requires training and a positive value")
    if args.psf_iterations is not None and (not is_training or args.psf_iterations < 0):
        parser.error("--psf-iterations requires training and a nonnegative value")
    if args.checkpoint and is_training:
        parser.error("--checkpoint applies to inference/evaluation; plugin training starts a new model")
    if args.distort_checkpoint and args.action != "train-plugin":
        parser.error("--distort-checkpoint applies to plugin training")
    if args.savings and any(value < 0 or value > 0.8 for value in args.savings):
        parser.error("--savings must be within the trained range 0..0.8")
    if args.downscales and any(value < 1 or value > 8 for value in args.downscales):
        parser.error("--downscales must be within the trained range 1..8")
    source = args.data_root.expanduser().resolve() / SCENES[args.scene] / args.scene
    if not args.dry_run and not source.is_dir():
        raise FileNotFoundError(f"Scene data not found: {source}. Prepare the authors' scene data and set --data-root; see README.md")
    if not args.dry_run and not args.no_download:
        from xrgs_utils.assets import ensure_assets
        ensure_assets(args.scene)
    model = ROOT / "models" / SCENES[args.scene] / args.scene
    output = args.output_dir.expanduser().resolve() if args.output_dir else ROOT / "outputs" / args.scene / (args.action + ("-smoke" if args.smoke else ""))
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else model / "unified"
    distort = args.distort_checkpoint.expanduser().resolve() if args.distort_checkpoint else model / "baseline/distort_mlp.pth"
    required = [source, model / "backbone/point_cloud/iteration_30000/point_cloud.ply", model / "reference_distort/distort_mlp.pth", ROOT / "optics/psf_geometric_scan/psf_dataset_downstream.npz"]
    if args.action != "train-distort":
        required.append(distort)
    if args.action in {"eval", "inference"}:
        required.append(checkpoint / "point_cloud/iteration_30000/res_cond_plugin.pth")
    for path in required:
        if not args.dry_run and not path.exists():
            raise FileNotFoundError(f"Missing runtime input: {path}")
    cfg = json.loads((ROOT / "configs" / (args.scene + ".json")).read_text())
    cfg.update(source_path=str(source), model_path=str(model / "backbone"), output_path=str(output),
               backbone_path=str(ROOT / "external/baselines/Scaffold-GS"),
               distort_mlp_path=str(distort),
               canonical_distort_mlp_path=str(model / "reference_distort/distort_mlp.pth"),
               optics_sim_path=str(ROOT / "optics/optics_simulator"),
               psf_dir=str(ROOT / "optics/psf_geometric_scan"), eval_max_views=1 if args.smoke else args.max_views,
               eval_image_mode="ours" if args.save_images else "off", eval_image_bit_depth=8,
               eval_image_save_aux=False, eval_lpips=not args.no_lpips)
    if args.smoke:
        cfg.update(eval_savings=[0.0], eval_downscales=[1.0])
    if args.savings is not None:
        cfg["eval_savings"] = args.savings
    if args.downscales is not None:
        cfg["eval_downscales"] = args.downscales
    if is_training:
        iterations = args.iterations if args.iterations is not None else (2 if args.smoke else cfg["finetune_iterations"])
        psf_iterations = args.psf_iterations if args.psf_iterations is not None else min(1 if args.smoke else 1000, iterations)
        if psf_iterations > iterations:
            parser.error("--psf-iterations cannot exceed --iterations")
        cfg.update(finetune_iterations=iterations, lens_psf_iterations=psf_iterations, train_only=not args.evaluate)
    if args.action == "inference":
        cfg.update(mode="unified_renderer", render_checkpoint=str(checkpoint), render_kind="unified_plugin",
                   render_output_root=str(output), render_split="test", render_max_views=cfg["eval_max_views"],
                   render_savings=cfg["eval_savings"], render_downscales=cfg["eval_downscales"],
                   render_eval=args.evaluate, render_save_gt=False, render_fps=False,
                   eval_lpips=args.evaluate and not args.no_lpips)
    elif args.action == "eval":
        cfg.update(mode="unified_plugin", eval_only=True, eval_only_ckpt=str(checkpoint))
    elif args.action == "train-plugin":
        cfg.update(mode="unified_plugin", eval_only=False, eval_only_ckpt="", plugin_init_mode="clean_random", save_iteration=30000)
    else:
        cfg.update(mode="baseline_distort_mlp", eval_only=False, eval_only_ckpt="", distort_mlp_path="",
                   distort_mlp_lr=0.003, lambda_dssim=0.2, eval_image_mode="baseline" if args.save_images else "off")
    config_path = output / "launch.json"
    # Shared XRGS YAML accepts XRGS fields only; native Scaffold arguments stay on the CLI.
    native_args = []
    for key in ["eval", "source_path", "model_path", "feat_dim", "n_offsets", "appearance_dim", "voxel_size", "update_init_factor", "white_background", "images", "resolution", "data_device"]:
        if key not in cfg:
            continue
        value = cfg.pop(key)
        if isinstance(value, bool):
            if value:
                native_args.append("--" + key)
        else:
            native_args.extend(["--" + key, str(value)])
    command = [sys.executable, str(ROOT / "unified-plugins.py"), "--config", str(config_path), *native_args]
    print(json.dumps({"command": command, "working_directory": str(ROOT), "config": cfg, "runtime_inputs_checked": not args.dry_run}, indent=2), flush=True)
    if args.dry_run:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        raise FileExistsError(f"Output already contains a run: {config_path}; choose a new --output-dir")
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["WANDB_MODE"] = "disabled"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    native_paths = [ROOT / "external/baselines/Scaffold-GS/submodules/diff-gaussian-rasterization", ROOT / "external/baselines/Scaffold-GS/submodules/simple-knn", ROOT / "optics/optics_simulator/psf_cuda_ext_v2/src", ROOT]
    env["PYTHONPATH"] = os.pathsep.join(map(str, native_paths)) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "outputs/.torch_extensions"))
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode

if __name__ == "__main__":
    raise SystemExit(main())
