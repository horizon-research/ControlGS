from __future__ import annotations

import json
import math
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from .layout import PluginLayout, layout_to_metadata


BACKBONE_CHOICES = ["scaffold_gs"]
RUN_MODE_CHOICES = ["unified_plugin", "baseline_distort_mlp", "unified_renderer"]


UNIFIED_CONFIG_SECTIONS = {
    "unified",
    "backbone",
    "baseline",
    "training",
    "conditions",
    "plugins",
    "ablation",
    "display",
    "lens",
    "optics",
    "evaluation",
    "metrics",
    "checkpoint",
    "renderer",
}

CONFIG_ALIASES = {
    "backbone_name": "backbone",
    "xrgs_mode": "mode",
    "SS_ctrl_dim": "ss_dim",
    "DM_ctrl_dim": "display_dim",
    "LC_ctrl_dim": "lens_dim",
    "condition_lr": "res_cond_lr",
    "mse_weight": "mse_loss_weight",
    "max_ss_downsample_scale": "ss_scale_max",
    "max_target_saving_power_setting": "target_saving_max",
}

LEGACY_IGNORED_CONFIG_KEYS = {"endpoint_reg_lambda", "lens_target_size"}


def _load_yaml_module(required: bool):
    try:
        import yaml

        return yaml
    except ModuleNotFoundError as exc:
        if required:
            raise ModuleNotFoundError(
                "Reading YAML config files requires PyYAML in this baseline environment. "
                "Install pyyaml or run without --config to print defaults."
            ) from exc
        return None


def _parse_scalar_config_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value[0:1] in {"'", "\""} and value[-1:] == value[0]:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar_config_value(item.strip()) for item in inner.split(",")]
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by XRGS config files.

    This fallback intentionally supports only nested section maps and scalar
    `key: value` entries. Full YAML remains handled by PyYAML when installed.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML syntax at line {line_no}: {raw_line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML indentation at line {line_no}: {raw_line!r}")
        current = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar_config_value(value)
    return root


def parse_float_csv(value: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


UNIFIED_ARG_SPECS: list[dict[str, Any]] = [
    {"flags": ["--mode"], "default": "unified_plugin", "choices": RUN_MODE_CHOICES},
    {"flags": ["--backbone"], "default": "scaffold_gs", "choices": BACKBONE_CHOICES},
    {"flags": ["--backbone_path"], "default": "", "type": str},
    {"flags": ["--output_path"], "default": "", "type": str},
    {"flags": ["--compgs_skip_quant"], "default": True, "action": "bool"},
    {"flags": ["--eval_only_ckpt"], "default": "", "type": str},
    {"flags": ["--eval_only"], "default": False, "action": "bool"},
    {"flags": ["--train_only"], "default": False, "action": "bool"},
    {"flags": ["--finetune_iterations"], "default": 10000, "type": int},
    {"flags": ["--save_iteration"], "default": -1, "type": int},
    {"flags": ["--load_iteration"], "default": -1, "type": int},
    {"flags": ["--res_cond_lr", "--condition_lr"], "dest": "res_cond_lr", "default": 0.003, "type": float},
    {"flags": ["--ss_dim", "--SS_ctrl_dim"], "dest": "ss_dim", "default": 8, "type": int},
    {"flags": ["--display_dim", "--DM_ctrl_dim"], "dest": "display_dim", "default": 8, "type": int},
    {"flags": ["--lens_dim", "--LC_ctrl_dim"], "dest": "lens_dim", "default": 8, "type": int},
    {"flags": ["--inject_dim"], "default": -1, "type": int},
    {"flags": ["--projection_init_std"], "default": 0.02, "type": float},
    {
        "flags": ["--plugin_init_mode"],
        "default": "clean_random",
        "choices": ["clean_random", "strict_warm_start", "resume_unified", "random", "warm_start"],
    },
    {"flags": ["--ss_plugin_path"], "default": "", "type": str},
    {"flags": ["--display_plugin_path"], "default": "", "type": str},
    {"flags": ["--lens_plugin_path"], "default": "", "type": str},
    {"flags": ["--ss_plugin_iteration"], "default": -1, "type": int},
    {"flags": ["--display_plugin_iteration"], "default": -1, "type": int},
    {"flags": ["--lens_plugin_iteration"], "default": -1, "type": int},
    {"flags": ["--disable_ss"], "default": False, "action": "bool"},
    {"flags": ["--disable_display"], "default": False, "action": "bool"},
    {"flags": ["--disable_lens"], "default": False, "action": "bool"},
    {"flags": ["--ss_scale_max"], "default": 8.0, "type": float},
    {"flags": ["--downscale_min"], "default": 1.0, "type": float},
    {"flags": ["--downscale_max"], "default": 8.0, "type": float},
    {"flags": ["--ss_downsample_mode"], "default": "area", "choices": ["area"]},
    {"flags": ["--resample_gt_display_canvas"], "default": False, "action": "bool"},
    {"flags": ["--use_super_sampling"], "default": True, "action": "bool"},
    {
        "flags": ["--render_resolution_policy"],
        "default": "super_sampling",
        "choices": ["super_sampling", "direct_downscale_no_ss"],
    },
    {"flags": ["--target_saving_min"], "default": 0.0, "type": float},
    {"flags": ["--target_saving_max"], "default": 0.8, "type": float},
    {"flags": ["--mse_loss_weight"], "default": 0.0, "type": float},
    {"flags": ["--lpips_loss_weight"], "default": 0.0, "type": float},
    {"flags": ["--lambda_dssim"], "default": 0.2, "type": float},
    {"flags": ["--lens_condition_radius_max_norm"], "default": 2.0 ** 0.5, "type": float},
    {"flags": ["--lens_mode"], "default": "distortion", "choices": ["distortion", "psf"]},
    {"flags": ["--lens_psf_iterations"], "default": 1000, "type": int},
    {"flags": ["--eval_lens_mode"], "default": "psf", "choices": ["distortion", "psf"]},
    {"flags": ["--lens_pad_ratio"], "default": 0.2, "type": float},
    {"flags": ["--distort_mlp_path"], "default": "", "type": str},
    {"flags": ["--canonical_distort_mlp_path"], "default": "", "type": str},
    {"flags": ["--optics_sim_path"], "default": None, "type": str},
    {"flags": ["--psf_dir"], "default": None, "type": str},
    {"flags": ["--psf_backend"], "default": "v2", "choices": ["v2"]},
    {"flags": ["--brightness_match_mode"], "default": "fixed_train0", "choices": ["off", "per_image", "fixed_train0"]},
    {"flags": ["--eval_max_views"], "default": 0, "type": int},
    {"flags": ["--eval_savings"], "default": parse_float_csv("0.0,0.2,0.4,0.6,0.8"), "type": parse_float_csv},
    {"flags": ["--eval_downscales"], "default": parse_float_csv("1,2,4,8"), "type": parse_float_csv},
    {"flags": ["--eval_power_mode"], "default": "global", "choices": ["global", "per_image"]},
    {"flags": ["--eval_image_mode"], "default": "off", "choices": ["off", "baseline", "ours"]},
    {"flags": ["--eval_image_dir"], "default": "", "type": str},
    {"flags": ["--eval_image_max_views"], "default": -1, "type": int},
    {"flags": ["--eval_image_bit_depth"], "default": 8, "type": int},
    {"flags": ["--eval_image_save_aux"], "default": True, "action": "bool"},
    {"flags": ["--eval_lpips"], "default": True, "action": "bool"},
    {"flags": ["--lpips_net"], "default": "vgg", "choices": ["alex", "vgg", "squeeze"]},
    {"flags": ["--quiet"], "default": False, "action": "bool"},
]

BASELINE_DISTORT_ARG_SPECS: list[dict[str, Any]] = [
    {"flags": ["--mode"], "default": "baseline_distort_mlp", "choices": RUN_MODE_CHOICES},
    {"flags": ["--backbone"], "default": "scaffold_gs", "choices": BACKBONE_CHOICES},
    {"flags": ["--backbone_path"], "default": "", "type": str},
    {"flags": ["--output_path"], "default": "", "type": str},
    {"flags": ["--compgs_skip_quant"], "default": True, "action": "bool"},
    {"flags": ["--eval_only_ckpt"], "default": "", "type": str},
    {"flags": ["--eval_only"], "default": False, "action": "bool"},
    {"flags": ["--train_only"], "default": False, "action": "bool"},
    {"flags": ["--finetune_iterations"], "default": 10000, "type": int},
    {"flags": ["--load_iteration"], "default": -1, "type": int},
    {"flags": ["--distort_mlp_lr"], "default": 0.003, "type": float},
    {"flags": ["--lambda_dssim"], "default": 0.2, "type": float},
    {"flags": ["--lens_mode"], "default": "distortion", "choices": ["distortion", "psf"]},
    {"flags": ["--lens_psf_iterations"], "default": 1000, "type": int},
    {"flags": ["--eval_lens_mode"], "default": "psf", "choices": ["distortion", "psf"]},
    {"flags": ["--eval_max_views"], "default": 0, "type": int},
    {"flags": ["--lens_pad_ratio"], "default": 0.2, "type": float},
    {"flags": ["--canonical_distort_mlp_path"], "default": "", "type": str},
    {"flags": ["--target_saving_max"], "default": 0.8, "type": float},
    {"flags": ["--ss_scale_max"], "default": 8.0, "type": float},
    {"flags": ["--lens_condition_radius_max_norm"], "default": 2.0 ** 0.5, "type": float},
    {"flags": ["--ss_downsample_mode"], "default": "area", "choices": ["area"]},
    {"flags": ["--resample_gt_display_canvas"], "default": False, "action": "bool"},
    {"flags": ["--use_super_sampling"], "default": True, "action": "bool"},
    {
        "flags": ["--render_resolution_policy"],
        "default": "super_sampling",
        "choices": ["super_sampling", "direct_downscale_no_ss"],
    },
    {"flags": ["--optics_sim_path"], "default": None, "type": str},
    {"flags": ["--psf_dir"], "default": None, "type": str},
    {"flags": ["--psf_backend"], "default": "v2", "choices": ["v2"]},
    {"flags": ["--brightness_match_mode"], "default": "fixed_train0", "choices": ["off", "per_image", "fixed_train0"]},
    {"flags": ["--eval_lpips"], "default": True, "action": "bool"},
    {"flags": ["--eval_savings"], "default": parse_float_csv("0.0,0.2,0.4,0.6,0.8"), "type": parse_float_csv},
    {"flags": ["--eval_downscales"], "default": parse_float_csv("1,2,4,8"), "type": parse_float_csv},
    {"flags": ["--eval_power_mode"], "default": "global", "choices": ["global", "per_image"]},
    {"flags": ["--eval_image_mode"], "default": "off", "choices": ["off", "baseline", "ours"]},
    {"flags": ["--eval_image_dir"], "default": "", "type": str},
    {"flags": ["--eval_image_max_views"], "default": -1, "type": int},
    {"flags": ["--eval_image_bit_depth"], "default": 8, "type": int},
    {"flags": ["--eval_image_save_aux"], "default": True, "action": "bool"},
    {"flags": ["--lpips_net"], "default": "vgg", "choices": ["alex", "vgg", "squeeze"]},
    {"flags": ["--quiet"], "default": False, "action": "bool"},
]

RENDERER_EXTRA_ARG_SPECS: list[dict[str, Any]] = [
    {"flags": ["--mode"], "default": "unified_renderer", "choices": RUN_MODE_CHOICES},
    {"flags": ["--render_checkpoint"], "default": "", "type": str},
    {"flags": ["--render_kind"], "default": "auto", "choices": ["auto", "unified_plugin", "baseline_distort_mlp"]},
    {"flags": ["--render_split"], "default": "test", "choices": ["test", "train", "all"]},
    {"flags": ["--render_output_root"], "default": "", "type": str},
    {"flags": ["--render_savings"], "default": None, "type": parse_float_csv},
    {"flags": ["--render_downscales"], "default": None, "type": parse_float_csv},
    {"flags": ["--render_max_views"], "default": None, "type": int},
    {"flags": ["--render_eval"], "default": False, "action": "bool"},
    {"flags": ["--render_save_gt"], "default": True, "action": "bool"},
    {"flags": ["--render_fps"], "default": False, "action": "bool"},
    {"flags": ["--render_fps_warmup"], "default": 5, "type": int},
    {"flags": ["--render_fps_iters"], "default": 10, "type": int},
]

def config_defaults_from_specs(specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {spec.get("dest", spec["flags"][0].lstrip("-")): spec["default"] for spec in specs}


def unified_config_defaults() -> dict[str, Any]:
    return config_defaults_from_specs(UNIFIED_ARG_SPECS)


def baseline_distort_config_defaults() -> dict[str, Any]:
    return config_defaults_from_specs(BASELINE_DISTORT_ARG_SPECS)


def merged_arg_specs(*spec_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for specs in spec_groups:
        for spec in specs:
            key = spec.get("dest", spec["flags"][0].lstrip("-"))
            merged[key] = spec
    return list(merged.values())


# Keep unified plug-in mode as the shared runner default. The DistortMLP-only
# baseline is selected explicitly with `mode: baseline_distort_mlp`, and the
# inference renderer is selected explicitly with `mode: unified_renderer`.
RENDERER_ARG_SPECS = merged_arg_specs(BASELINE_DISTORT_ARG_SPECS, UNIFIED_ARG_SPECS, RENDERER_EXTRA_ARG_SPECS)
TOP_LEVEL_ARG_SPECS = merged_arg_specs(BASELINE_DISTORT_ARG_SPECS, UNIFIED_ARG_SPECS, RENDERER_EXTRA_ARG_SPECS)


def unified_renderer_config_defaults() -> dict[str, Any]:
    return config_defaults_from_specs(RENDERER_ARG_SPECS)


def coerce_config_defaults(defaults: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    specs = {
        spec.get("dest", spec["flags"][0].lstrip("-")): spec
        for spec in specs
    }
    coerced: dict[str, Any] = {}
    for key, value in defaults.items():
        if key in LEGACY_IGNORED_CONFIG_KEYS:
            continue
        spec = specs.get(key)
        if spec is None:
            raise ValueError(f"Unknown XRGS unified config key: {key}")
        if value is None:
            coerced[key] = value
            continue
        if spec.get("action") == "bool":
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    coerced[key] = True
                elif normalized in {"0", "false", "no", "off"}:
                    coerced[key] = False
                else:
                    raise ValueError(f"Invalid boolean XRGS unified config value for {key}: {value!r}")
            else:
                coerced[key] = bool(value)
            continue
        value_type = spec.get("type")
        if value_type is parse_float_csv:
            value = parse_float_csv(value)
        elif value_type is not None and not isinstance(value, value_type):
            value = value_type(value)
        choices = spec.get("choices")
        if choices is not None:
            if all(isinstance(choice, str) for choice in choices):
                if isinstance(value, bool):
                    if value is False and "off" in choices:
                        value = "off"
                    elif value is True and "on" in choices:
                        value = "on"
                    else:
                        value = str(value).lower()
                elif not isinstance(value, str):
                    value = str(value)
                if isinstance(value, str) and value not in choices and value.lower() in choices:
                    value = value.lower()
            if value not in choices:
                raise ValueError(f"Invalid XRGS unified config value for {key}: {value!r}; expected one of {choices}")
        coerced[key] = value
    return coerced


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if key in UNIFIED_CONFIG_SECTIONS and isinstance(value, dict):
            flattened.update(_flatten_config(value))
        else:
            flattened[key] = value
    return flattened


def normalize_config_defaults(data: dict[str, Any]) -> dict[str, Any]:
    flattened = _flatten_config(data)
    normalized: dict[str, Any] = {}
    for key, value in flattened.items():
        normalized[CONFIG_ALIASES.get(key, key)] = value
    return normalized


def add_unified_config_args(parser: ArgumentParser) -> None:
    """Document the config-first policy without exposing every unified knob.

    Scaffold-GS still uses argparse for its native dataset/model/optimizer
    groups. XRGS unified options are merged through parser defaults from YAML,
    so they are available on `args` without becoming a long CLI surface.
    """
    parser.add_argument(
        "--mode",
        choices=RUN_MODE_CHOICES,
        help="Run unified plug-in training/evaluation, the DistortMLP-only baseline, or the renderer.",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONE_CHOICES,
        help="Select the rendering backbone adapter used by the shared XRGS runner.",
    )
    parser.add_argument(
        "--backbone_path",
        default=None,
        help="Optional backbone repository path. Defaults to external/baselines/<backbone>.",
    )
    parser.add_argument(
        "--print_unified_config_defaults",
        action="store_true",
        help="Print XRGS unified config defaults and exit.",
    )
    parser.add_argument(
        "--resample_gt_display_canvas",
        nargs="?",
        const=True,
        type=parse_bool_arg,
        help="Whether GT follows the same display-resolution resample as renders before lens target construction. Default false uses NativeGT.",
    )
    parser.add_argument(
        "--use_super_sampling",
        "--use-super-sampling",
        nargs="?",
        const=True,
        type=parse_bool_arg,
        help=(
            "Whether evaluation renders at native resolution before display-resolution resampling. "
            "Default true; false directly renders each requested low-resolution condition."
        ),
    )
    parser.add_argument(
        "--render_resolution_policy",
        "--render-resolution-policy",
        choices=["super_sampling", "direct_downscale_no_ss"],
        help="Evaluation render-resolution policy recorded in configs and results.",
    )
    parser.add_argument(
        "--canonical_distort_mlp_path",
        "--canonical-distort-mlp-path",
        default=None,
        help=(
            "Optional canonical DistortMLP checkpoint, run directory, or baseline_distort_mlp root "
            "used only for GT reference display-power evaluation."
        ),
    )


def load_config_defaults(argv: list[str], specs: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    known, _unknown = pre_parser.parse_known_args(argv)
    if not known.config:
        return {}, None
    config_path = Path(known.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Unified config file not found: {config_path}")
    yaml = _load_yaml_module(required=False)
    if yaml is not None:
        loaded = yaml.safe_load(config_path.read_text()) or {}
    else:
        loaded = _simple_yaml_load(config_path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"Unified config must contain a YAML mapping: {config_path}")
    return coerce_config_defaults(normalize_config_defaults(loaded), specs), str(config_path)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def copy_input_config(output_path: Path, config_source: str | None, filename: str = "input_config.yaml") -> Path | None:
    if not config_source:
        return None
    source_path = Path(config_source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input config file not found: {source_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    target_path = output_path / filename
    if source_path == target_path.resolve():
        return target_path
    shutil.copy2(source_path, target_path)
    return target_path


def write_effective_config(
    output_path: Path,
    args,
    layout: PluginLayout,
    total_dim: int,
    inject_dims: dict[str, int],
    filename: str = "config.yaml",
) -> Path:
    config = {key: jsonable(value) for key, value in sorted(vars(args).items())}
    config["layout"] = layout_to_metadata(layout)
    config["total_dim"] = int(total_dim)
    config["inject_dims_actual"] = {name: int(dim) for name, dim in inject_dims.items()}
    path = output_path / filename
    yaml = _load_yaml_module(required=False)
    if yaml is not None:
        path.write_text(yaml.safe_dump(config, sort_keys=True))
    else:
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path
