from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
from plyfile import PlyData

from gaussian_renderer import prefilter_voxel, render
from scene import GaussianModel, Scene

from .checkpoint import plugin_checkpoint_dir, resolve_plugin_iteration_dir
from .adapter import UnifiedBackboneAdapter


def make_gaussians(dataset) -> GaussianModel:
    return GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
        res_cond_dim=getattr(dataset, "res_cond_dim", 0),
        res_cond_mode=getattr(dataset, "res_cond_mode", "scalar"),
    )


def load_scene(dataset, iteration: int = -1) -> tuple[GaussianModel, Scene]:
    gaussians = make_gaussians(dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return gaussians, scene


def require_unified_plugin_checkpoint(model_path: str | Path, iteration: int, total_dim: int) -> None:
    iteration_dir = plugin_checkpoint_dir(model_path, iteration)
    ply_path = iteration_dir / "point_cloud.ply"
    plugin_path = iteration_dir / "res_cond_plugin.pth"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing unified plug-in point cloud checkpoint: {ply_path}")
    if not plugin_path.exists():
        raise FileNotFoundError(f"Missing unified plug-in projection checkpoint: {plugin_path}")
    endpoint_count = count_res_cond_anchor_properties(ply_path)
    if endpoint_count != total_dim:
        raise ValueError(
            f"Unified plug-in point cloud has {endpoint_count} endpoint fields, "
            f"but exactly total_dim={total_dim} is required: {ply_path}"
        )


def set_plugin_layout(gaussians: GaussianModel, layout: dict[str, tuple[int, int]]) -> None:
    gaussians.res_cond_plugin_layout = layout


def count_res_cond_anchor_properties(ply_path: Path) -> int:
    plydata = PlyData.read(str(ply_path))
    return sum(1 for prop in plydata.elements[0].properties if prop.name.startswith("f_res_cond_anchor"))


def first_layer_out_dim(mlp) -> int:
    first_layer = mlp._modules["0"]
    return int(first_layer.weight.shape[0])


def projection_output_dims(gaussians: GaussianModel) -> dict[str, int]:
    return {
        "opacity": int(gaussians.res_cond_opacity_proj.weight.shape[0]),
        "cov": int(gaussians.res_cond_cov_proj.weight.shape[0]),
        "color": int(gaussians.res_cond_color_proj.weight.shape[0]),
    }


def validate_inject_dim(gaussians: GaussianModel, requested_inject_dim: int) -> dict[str, int]:
    mlp_dims = {
        "opacity": first_layer_out_dim(gaussians.get_opacity_mlp),
        "cov": first_layer_out_dim(gaussians.get_cov_mlp),
        "color": first_layer_out_dim(gaussians.get_color_mlp),
    }
    proj_dims = projection_output_dims(gaussians)
    mismatches = {
        name: {"decoder": mlp_dims[name], "projection": proj_dims[name]}
        for name in mlp_dims
        if mlp_dims[name] != proj_dims[name]
    }
    if mismatches:
        raise ValueError(f"Projection output dims must match each decoder first-layer width, got {mismatches}")
    if requested_inject_dim > 0 and (len(set(mlp_dims.values())) != 1 or int(requested_inject_dim) != next(iter(mlp_dims.values()))):
        raise ValueError(f"Requested scalar inject_dim={requested_inject_dim} but per-decoder inject dims are {mlp_dims}")
    return mlp_dims


def _validate_projection_source_slice(
    name: str,
    key: str,
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    if source_weight.ndim != 2:
        raise ValueError(f"{name} {key} weight must be 2D, got shape {tuple(source_weight.shape)}")
    if source_weight.shape[0] != target_weight.shape[0]:
        raise ValueError(
            f"{name} {key} output dim mismatch: checkpoint={source_weight.shape[0]} target={target_weight.shape[0]}"
        )
    if source_weight.shape[1] < dim:
        raise ValueError(f"{name} {key} input dim {source_weight.shape[1]} is smaller than requested slice dim {dim}")
    return source_weight[:, :dim]


def zero_disabled_plugin_slices(gaussians: GaussianModel, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None:
    """Clear disabled plug-in input slices before training.

    PyTorch Linear weights are [out_features, in_features], so [:, start:end]
    removes the disabled plug-in columns from every decoder projection output.
    Gradient freezing during training is enforced by the hooks below.
    """
    if not disabled:
        return
    with torch.no_grad():
        for name in disabled:
            start, end = layout[name]
            gaussians._res_cond_anchor[:, start:end].zero_()
            gaussians.res_cond_opacity_proj.weight[:, start:end].zero_()
            gaussians.res_cond_cov_proj.weight[:, start:end].zero_()
            gaussians.res_cond_color_proj.weight[:, start:end].zero_()


def register_disabled_plugin_slice_hooks(
    gaussians: GaussianModel,
    layout: dict[str, tuple[int, int]],
    disabled: set[str],
) -> None:
    """Keep ablated SS / display / lens slices frozen without splitting tensors."""
    if not disabled:
        return
    anchor_mask = torch.ones_like(gaussians._res_cond_anchor)
    opacity_mask = torch.ones_like(gaussians.res_cond_opacity_proj.weight)
    cov_mask = torch.ones_like(gaussians.res_cond_cov_proj.weight)
    color_mask = torch.ones_like(gaussians.res_cond_color_proj.weight)
    for name in disabled:
        start, end = layout[name]
        anchor_mask[:, start:end] = 0.0
        opacity_mask[:, start:end] = 0.0
        cov_mask[:, start:end] = 0.0
        color_mask[:, start:end] = 0.0
    handles = [
        gaussians._res_cond_anchor.register_hook(lambda grad, mask=anchor_mask: grad * mask),
        gaussians.res_cond_opacity_proj.weight.register_hook(lambda grad, mask=opacity_mask: grad * mask),
        gaussians.res_cond_cov_proj.weight.register_hook(lambda grad, mask=cov_mask: grad * mask),
        gaussians.res_cond_color_proj.weight.register_hook(lambda grad, mask=color_mask: grad * mask),
    ]
    gaussians._res_cond_disable_hook_handles = handles


def load_plugin_slice(
    gaussians: GaussianModel,
    dataset,
    name: str,
    plugin_path: str,
    target_slice: tuple[int, int],
    logger: logging.Logger,
    iteration: int = -1,
) -> bool:
    if not plugin_path:
        raise ValueError(f"strict_warm_start requires --{name}_plugin_path for enabled slice '{name}'")
    start, end = target_slice
    dim = end - start
    iteration_dir = resolve_plugin_iteration_dir(plugin_path, iteration)
    endpoint_count = count_res_cond_anchor_properties(iteration_dir / "point_cloud.ply")
    if endpoint_count != dim:
        raise ValueError(
            f"{name} plug-in checkpoint has {endpoint_count} endpoint fields, "
            f"but exactly slice dim {dim} is required: {iteration_dir / 'point_cloud.ply'}"
        )
    slice_dataset = copy.copy(dataset)
    slice_dataset.res_cond_dim = dim
    slice_dataset.res_cond_mode = "learned_anchor_multi_plugin"
    source_gaussians = make_gaussians(slice_dataset)
    source_gaussians.load_ply_sparse_gaussian(str(iteration_dir / "point_cloud.ply"))
    source_gaussians.load_mlp_checkpoints(str(iteration_dir), strict_res_cond_plugin=True)

    if source_gaussians._res_cond_anchor.shape[0] != gaussians._res_cond_anchor.shape[0]:
        raise ValueError(
            f"{name} plug-in anchor count mismatch: "
            f"{source_gaussians._res_cond_anchor.shape[0]} vs {gaussians._res_cond_anchor.shape[0]}"
        )
    anchor_delta = (source_gaussians._anchor - gaussians._anchor).abs().max().item()
    if anchor_delta > 1e-4:
        raise ValueError(f"{name} plug-in anchors do not match the base model; max delta={anchor_delta:g}")
    opacity_weight = _validate_projection_source_slice(
        name,
        "opacity_proj",
        source_gaussians.res_cond_opacity_proj.weight,
        gaussians.res_cond_opacity_proj.weight,
        dim,
    )
    cov_weight = _validate_projection_source_slice(
        name,
        "cov_proj",
        source_gaussians.res_cond_cov_proj.weight,
        gaussians.res_cond_cov_proj.weight,
        dim,
    )
    color_weight = _validate_projection_source_slice(
        name,
        "color_proj",
        source_gaussians.res_cond_color_proj.weight,
        gaussians.res_cond_color_proj.weight,
        dim,
    )

    with torch.no_grad():
        gaussians._res_cond_anchor[:, start:end].copy_(source_gaussians._res_cond_anchor[:, :dim])
        gaussians.res_cond_opacity_proj.weight[:, start:end].copy_(opacity_weight)
        gaussians.res_cond_cov_proj.weight[:, start:end].copy_(cov_weight)
        gaussians.res_cond_color_proj.weight[:, start:end].copy_(color_weight)
    logger.info("Initialized %s slice [%d:%d] from %s", name, start, end, iteration_dir)
    return True


def render_one(camera, gaussians: GaussianModel, pipe, background: torch.Tensor) -> torch.Tensor:
    visible_mask = prefilter_voxel(camera, gaussians, pipe, background)
    return render(camera, gaussians, pipe, background, visible_mask=visible_mask)["render"]


def render_baseline(camera, gaussians: GaussianModel, pipe, background: torch.Tensor) -> torch.Tensor:
    saved_layout = getattr(gaussians, "res_cond_plugin_layout", None)
    saved_mode = getattr(gaussians, "res_cond_mode", None)
    try:
        gaussians.res_cond_plugin_layout = {}
        gaussians.res_cond_mode = "scalar"
        return render_one(camera, gaussians, pipe, background)
    finally:
        gaussians.res_cond_plugin_layout = saved_layout
        if saved_mode is not None:
            gaussians.res_cond_mode = saved_mode


def trainable_parameter_count(gaussians: GaussianModel) -> int:
    if isinstance(gaussians, torch.nn.Module):
        return sum(parameter.numel() for parameter in gaussians.parameters() if parameter.requires_grad)
    total = 0
    for value in vars(gaussians).values():
        if isinstance(value, torch.nn.Module):
            total += sum(parameter.numel() for parameter in value.parameters() if parameter.requires_grad)
        elif isinstance(value, torch.nn.Parameter) and value.requires_grad:
            total += value.numel()
    return total


class ScaffoldGSAdapter(UnifiedBackboneAdapter):
    """Scaffold-GS adapter for the current unified runner."""

    name = "xrgs_scaffold_gs"
    supports_condition_injection = True
    condition_injection_note = None

    def load_scene(self, dataset, iteration: int = -1) -> tuple[GaussianModel, Scene]:
        return load_scene(dataset, iteration)

    def set_plugin_layout(self, gaussians: GaussianModel, layout: dict[str, tuple[int, int]]) -> None:
        set_plugin_layout(gaussians, layout)

    def validate_inject_dim(self, gaussians: GaussianModel, requested_inject_dim: int) -> dict[str, int]:
        return validate_inject_dim(gaussians, requested_inject_dim)

    def zero_disabled_plugin_slices(
        self,
        gaussians: GaussianModel,
        layout: dict[str, tuple[int, int]],
        disabled: set[str],
    ) -> None:
        zero_disabled_plugin_slices(gaussians, layout, disabled)

    def register_disabled_plugin_slice_hooks(
        self,
        gaussians: GaussianModel,
        layout: dict[str, tuple[int, int]],
        disabled: set[str],
    ) -> None:
        register_disabled_plugin_slice_hooks(gaussians, layout, disabled)

    def load_plugin_slice(
        self,
        gaussians: GaussianModel,
        dataset,
        name: str,
        plugin_path: str,
        target_slice: tuple[int, int],
        logger: logging.Logger,
        iteration: int = -1,
    ) -> bool:
        return load_plugin_slice(gaussians, dataset, name, plugin_path, target_slice, logger, iteration)

    def require_unified_plugin_checkpoint(self, model_path: str | Path, iteration: int, total_dim: int) -> None:
        require_unified_plugin_checkpoint(model_path, iteration, total_dim)

    def render(self, camera, gaussians: GaussianModel, pipe, background: torch.Tensor) -> torch.Tensor:
        return render_one(camera, gaussians, pipe, background)

    def render_baseline(self, camera, gaussians: GaussianModel, pipe, background: torch.Tensor) -> torch.Tensor:
        return render_baseline(camera, gaussians, pipe, background)

    def trainable_parameter_count(self, gaussians: GaussianModel) -> int:
        return trainable_parameter_count(gaussians)
