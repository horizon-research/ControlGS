from __future__ import annotations

import json
from pathlib import Path

from .layout import PluginLayout, layout_to_metadata


def normalize_plugin_init_mode(mode: str) -> str:
    aliases = {
        "random": "clean_random",
        "warm_start": "strict_warm_start",
    }
    return aliases.get(mode, mode)


def plugin_checkpoint_dir(model_path: str | Path, iteration: int) -> Path:
    return Path(model_path).expanduser().resolve() / "point_cloud" / f"iteration_{iteration}"


def plugin_metadata_path(model_path: str | Path, iteration: int) -> Path:
    return plugin_checkpoint_dir(model_path, iteration) / "plugin_metadata.json"


def load_plugin_metadata(model_path: str | Path, iteration: int) -> dict:
    path = plugin_metadata_path(model_path, iteration)
    if not path.exists():
        raise FileNotFoundError(f"Missing unified plug-in metadata: {path}")
    return json.loads(path.read_text())


def normalize_inject_dims(value) -> dict[str, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "actual" in value:
            return normalize_inject_dims(value["actual"])
        return {name: int(dim) for name, dim in value.items()}
    dim = int(value)
    return {"opacity": dim, "cov": dim, "color": dim}


def validate_resume_metadata(metadata: dict, layout: PluginLayout, total_dim: int, inject_dims: dict[str, int]) -> None:
    expected_layout = layout_to_metadata(layout)
    if metadata.get("layout") != expected_layout:
        raise ValueError(f"Unified layout mismatch: checkpoint={metadata.get('layout')} requested={expected_layout}")
    if int(metadata.get("total_dim", -1)) != int(total_dim):
        raise ValueError(f"Unified total_dim mismatch: checkpoint={metadata.get('total_dim')} requested={total_dim}")
    saved_inject_dims = normalize_inject_dims(metadata.get("inject_dims", metadata.get("inject_dim")))
    if saved_inject_dims is not None and saved_inject_dims != inject_dims:
        raise ValueError(f"Unified inject_dims mismatch: checkpoint={saved_inject_dims} actual={inject_dims}")


def resolve_plugin_iteration_dir(plugin_path: str, iteration: int = -1) -> Path:
    root = Path(plugin_path).expanduser().resolve()
    if (root / "point_cloud.ply").exists():
        return root
    if iteration > 0:
        candidate = root / "point_cloud" / f"iteration_{iteration}"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Plug-in iteration not found: {candidate}")
    candidates = sorted(
        (root / "point_cloud").glob("iteration_*"),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No plug-in point cloud iterations found under: {root}")
    return candidates[-1]


def write_plugin_metadata(output_path: Path, save_iteration: int, metadata: dict) -> Path:
    path = output_path / "point_cloud" / f"iteration_{save_iteration}" / "plugin_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path
