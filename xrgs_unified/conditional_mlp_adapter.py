from __future__ import annotations

import math
from pathlib import Path
from types import MethodType
import weakref

import torch
from torch import nn
import torch.nn.functional as F

from .checkpoint import plugin_checkpoint_dir, resolve_plugin_iteration_dir


def _first_layer_out_dim(mlp: nn.Module) -> int:
    first_layer = mlp._modules.get("0")
    if first_layer is None or not hasattr(first_layer, "weight"):
        raise TypeError(f"XRGS condition injection expects an nn.Sequential-style MLP, got {type(mlp).__name__}")
    return int(first_layer.weight.shape[0])


class ConditionedSequentialMLP(nn.Module):
    """Adapter-side first-layer condition injection for unmodified MLP 3DGS backbones."""

    def __init__(self, base_mlp: nn.Module, gaussians, projection_attr: str) -> None:
        super().__init__()
        if "0" not in base_mlp._modules or not hasattr(base_mlp._modules["0"], "weight"):
            raise TypeError(f"Cannot wrap {type(base_mlp).__name__}; expected first module to be nn.Linear")
        self.base_mlp = base_mlp
        self._gaussians_ref = weakref.ref(gaussians)
        self.projection_attr = projection_attr

    @property
    def first_layer_out_dim(self) -> int:
        return _first_layer_out_dim(self.base_mlp)

    @property
    def gaussians(self):
        gaussians = self._gaussians_ref()
        if gaussians is None:
            raise RuntimeError("The GaussianModel attached to this XRGS MLP wrapper was released")
        return gaussians

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = getattr(self.gaussians, "_xrgs_condition_context", None)
        if context is None or not self.gaussians.uses_res_cond_plugin():
            return self.base_mlp(features)
        res_features = context["res_features"]
        if res_features.shape[0] != features.shape[0]:
            raise ValueError(
                f"XRGS condition feature row mismatch: condition={tuple(res_features.shape)} input={tuple(features.shape)}"
            )
        modules = self.base_mlp._modules
        first_layer = modules["0"]
        projection = getattr(self.gaussians, self.projection_attr)
        hidden = first_layer(features) + F.linear(
            res_features.to(dtype=features.dtype, device=features.device),
            projection.weight.to(dtype=features.dtype, device=features.device),
        )
        for index in range(1, len(modules)):
            hidden = modules[str(index)](hidden)
        return hidden

    def train(self, mode: bool = True):
        self.base_mlp.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)


def _lens_condition_alpha(viewpoint_camera, anchors: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    projection = viewpoint_camera.full_proj_transform.to(dtype=dtype, device=anchors.device)
    anchor_h = torch.cat([anchors.to(dtype=dtype), torch.ones((anchors.shape[0], 1), dtype=dtype, device=anchors.device)], dim=1)
    projected = anchor_h @ projection
    w = projected[:, 3:4]
    w_safe = torch.where(w.abs() < 1e-7, torch.full_like(w, 1e-7), w)
    ndc_xy = projected[:, :2] / w_safe

    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    side = max(width, height)
    denom = max(side - 1, 1)
    axis_scale = torch.tensor(
        [max(width - 1, 1) / denom, max(height - 1, 1) / denom],
        dtype=dtype,
        device=anchors.device,
    )
    square_xy = ndc_xy * axis_scale
    radius = torch.linalg.norm(square_xy, dim=1, keepdim=True)
    radius_max = float(getattr(viewpoint_camera, "lens_condition_radius_max_norm", math.sqrt(2.0)))
    fixed_alpha = getattr(viewpoint_camera, "xrgs_fixed_lens_alpha", None)
    if fixed_alpha is not None:
        value = torch.as_tensor(fixed_alpha, dtype=dtype, device=anchors.device)
        value = value.reshape(-1)[0].clamp(0.0, 1.0)
        return (torch.ones_like(radius) * value).detach()
    return torch.clamp(radius / max(radius_max, 1e-6), 0.0, 1.0).detach()


def set_condition_context(gaussians, viewpoint_camera, visible_mask: torch.Tensor | None) -> None:
    uses_plugin = getattr(gaussians, "uses_res_cond_plugin", None)
    if uses_plugin is None:
        gaussians._xrgs_condition_context = None
        return
    if not gaussians.uses_res_cond_plugin():
        gaussians._xrgs_condition_context = None
        return
    if visible_mask is None:
        raise ValueError("XRGS condition injection requires a visible anchor mask")
    anchors = gaussians.get_anchor[visible_mask]
    res_anchor = gaussians._res_cond_anchor[visible_mask].to(dtype=anchors.dtype, device=anchors.device)
    layout = getattr(gaussians, "res_cond_plugin_layout", None)
    if layout is None:
        raise ValueError("XRGS condition injection requires gaussians.res_cond_plugin_layout")
    unknown_layout_keys = set(layout) - {"ss", "display", "lens"}
    if unknown_layout_keys:
        raise ValueError(f"Unsupported XRGS condition layout keys: {sorted(unknown_layout_keys)}")
    alphas = torch.zeros_like(res_anchor)
    if "ss" in layout:
        start, end = layout["ss"]
        value = getattr(viewpoint_camera, "ss_res_cond", getattr(viewpoint_camera, "res_cond", 0.0))
        alphas[:, start:end] = torch.as_tensor(value, dtype=anchors.dtype, device=anchors.device).reshape(-1)[0]
    if "display" in layout:
        start, end = layout["display"]
        value = getattr(viewpoint_camera, "display_res_cond", 0.0)
        alphas[:, start:end] = torch.as_tensor(value, dtype=anchors.dtype, device=anchors.device).reshape(-1)[0]
    if "lens" in layout:
        start, end = layout["lens"]
        alphas[:, start:end] = _lens_condition_alpha(viewpoint_camera, anchors, anchors.dtype)
    gaussians._xrgs_condition_context = {"res_features": alphas * res_anchor}


def clear_condition_context(gaussians) -> None:
    if hasattr(gaussians, "_xrgs_condition_context"):
        gaussians._xrgs_condition_context = None


def _is_wrapper(module: nn.Module) -> bool:
    return isinstance(module, ConditionedSequentialMLP)


def _condition_wrappers(gaussians) -> list[ConditionedSequentialMLP]:
    wrappers = []
    for attr in ["mlp_opacity", "mlp_cov", "mlp_color"]:
        module = getattr(gaussians, attr, None)
        if _is_wrapper(module):
            wrappers.append(module)
    return wrappers


def _set_wrapped_mlp_training(gaussians, mode: bool) -> list[tuple[ConditionedSequentialMLP, bool, bool]]:
    states = []
    for wrapper in _condition_wrappers(gaussians):
        states.append((wrapper, wrapper.training, wrapper.base_mlp.training))
        nn.Module.train(wrapper, mode)
        wrapper.base_mlp.train(mode)
    return states


def _restore_wrapped_mlp_training(states: list[tuple[ConditionedSequentialMLP, bool, bool]]) -> None:
    for wrapper, wrapper_training, base_training in states:
        nn.Module.train(wrapper, wrapper_training)
        wrapper.base_mlp.train(base_training)


class StableInferenceMLPContext:
    """Keep frozen backbone MLPs in eval mode while preserving plug-in gradients."""

    def __init__(self, gaussians) -> None:
        self.gaussians = gaussians
        self._states: list[tuple[ConditionedSequentialMLP, bool, bool]] = []

    def __enter__(self):
        self._states = _set_wrapped_mlp_training(self.gaussians, False)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _restore_wrapped_mlp_training(self._states)


def stable_inference_mlp_context(gaussians) -> StableInferenceMLPContext:
    return StableInferenceMLPContext(gaussians)


def _base_anchor_tensor(gaussians) -> torch.Tensor:
    raw_anchor = getattr(gaussians, "_anchor", None)
    if isinstance(raw_anchor, torch.Tensor) and raw_anchor.numel() == 0:
        return raw_anchor.new_empty((0, 3))
    try:
        return gaussians.get_anchor
    except Exception:
        if raw_anchor is None:
            raise
        return raw_anchor.reshape(-1, 3)


def _assert_anchor_match(saved_anchor, actual_anchor: torch.Tensor, context: str) -> None:
    saved = torch.as_tensor(saved_anchor, dtype=actual_anchor.dtype, device=actual_anchor.device)
    if tuple(saved.shape) != tuple(actual_anchor.shape):
        raise ValueError(f"{context} base anchor shape mismatch: checkpoint={tuple(saved.shape)} actual={tuple(actual_anchor.shape)}")
    delta_tensor = (saved - actual_anchor).abs()
    if delta_tensor.numel() == 0:
        return
    delta = delta_tensor.max()
    if float(delta.detach()) > 1e-4:
        raise ValueError(f"{context} base anchors do not match this checkpoint; max delta={float(delta.detach()):g}")


def _plugin_state(gaussians) -> dict[str, object]:
    return {
        "res_cond_anchor": gaussians._res_cond_anchor.detach().cpu(),
        "base_anchor": _base_anchor_tensor(gaussians).detach().cpu(),
        "opacity_proj": gaussians.res_cond_opacity_proj.state_dict(),
        "cov_proj": gaussians.res_cond_cov_proj.state_dict(),
        "color_proj": gaussians.res_cond_color_proj.state_dict(),
    }


def _load_plugin_state(gaussians, state: dict[str, object]) -> None:
    anchor = torch.as_tensor(state["res_cond_anchor"], dtype=torch.float, device="cuda")
    if anchor.shape != gaussians._res_cond_anchor.shape:
        raise ValueError(
            f"XRGS plug-in anchor shape mismatch: checkpoint={tuple(anchor.shape)} actual={tuple(gaussians._res_cond_anchor.shape)}"
        )
    base_anchor = state.get("base_anchor")
    if base_anchor is not None:
        _assert_anchor_match(base_anchor, _base_anchor_tensor(gaussians), "XRGS plug-in")
    with torch.no_grad():
        gaussians._res_cond_anchor.copy_(anchor)
    gaussians.res_cond_opacity_proj.load_state_dict(state["opacity_proj"])
    gaussians.res_cond_cov_proj.load_state_dict(state["cov_proj"])
    gaussians.res_cond_color_proj.load_state_dict(state["color_proj"])


def _unwrap_condition_mlps(gaussians) -> None:
    for attr in ["mlp_opacity", "mlp_cov", "mlp_color"]:
        module = getattr(gaussians, attr)
        if isinstance(module, ConditionedSequentialMLP):
            setattr(gaussians, attr, module.base_mlp)


def _rewrap_condition_mlps(gaussians) -> None:
    wrappers = {
        "mlp_opacity": "res_cond_opacity_proj",
        "mlp_cov": "res_cond_cov_proj",
        "mlp_color": "res_cond_color_proj",
    }
    for attr, projection_attr in wrappers.items():
        module = getattr(gaussians, attr)
        if not isinstance(module, ConditionedSequentialMLP):
            setattr(gaussians, attr, ConditionedSequentialMLP(module, gaussians, projection_attr).cuda())


def install_condition_plugin(gaussians, total_dim: int, projection_init_std: float = 0.02) -> None:
    """Install XRGS condition plug-in machinery without editing the backbone source tree."""

    if getattr(gaussians, "_xrgs_condition_plugin_installed", False):
        return
    gaussians.res_cond_dim = int(total_dim)
    gaussians.res_cond_mode = "learned_anchor_multi_plugin"
    gaussians.res_cond_plugin_layout = {}
    gaussians._res_cond_anchor = nn.Parameter(torch.empty((0, int(total_dim)), device="cuda").requires_grad_(True))
    gaussians.res_cond_opacity_proj = nn.Linear(int(total_dim), _first_layer_out_dim(gaussians.mlp_opacity), bias=False).cuda()
    gaussians.res_cond_cov_proj = nn.Linear(int(total_dim), _first_layer_out_dim(gaussians.mlp_cov), bias=False).cuda()
    gaussians.res_cond_color_proj = nn.Linear(int(total_dim), _first_layer_out_dim(gaussians.mlp_color), bias=False).cuda()
    _rewrap_condition_mlps(gaussians)
    gaussians._xrgs_condition_context = None

    original_load_ply = gaussians.load_ply_sparse_gaussian
    original_save_mlp = gaussians.save_mlp_checkpoints
    original_load_mlp = gaussians.load_mlp_checkpoints
    original_eval = gaussians.eval
    original_train = gaussians.train

    def ensure_anchor_shape(self) -> None:
        if int(getattr(self, "res_cond_dim", 0)) <= 0:
            return
        expected = (_base_anchor_tensor(self).shape[0], int(self.res_cond_dim))
        if tuple(self._res_cond_anchor.shape) != expected:
            # Must happen before optimizer construction so the optimizer tracks this Parameter.
            self._res_cond_anchor = nn.Parameter(torch.zeros(expected, device="cuda").requires_grad_(True))

    def uses_res_cond_plugin(self) -> bool:
        return int(getattr(self, "res_cond_dim", 0)) > 0 and getattr(self, "res_cond_mode", "") == "learned_anchor_multi_plugin"

    def reset_res_cond_plugin_parameters(self, projection_std: float = projection_init_std) -> None:
        if not self.uses_res_cond_plugin():
            return
        self.ensure_xrgs_condition_anchor_shape()
        with torch.no_grad():
            self._res_cond_anchor.zero_()
            for proj in [self.res_cond_opacity_proj, self.res_cond_cov_proj, self.res_cond_color_proj]:
                nn.init.trunc_normal_(
                    proj.weight,
                    mean=0.0,
                    std=float(projection_std),
                    a=-2.0 * float(projection_std),
                    b=2.0 * float(projection_std),
                )

    def freeze_base_for_res_cond_plugin(self) -> None:
        if isinstance(self, nn.Module):
            for name, parameter in self.named_parameters():
                if name.startswith(("res_cond", "_res_cond", "_xrgs")):
                    continue
                parameter.requires_grad_(False)
            return
        for name, value in vars(self).items():
            if name.startswith("res_cond") or name.startswith("_res_cond") or name.startswith("_xrgs"):
                continue
            if isinstance(value, nn.Parameter):
                value.requires_grad_(False)
            elif isinstance(value, nn.Module):
                for parameter in value.parameters():
                    parameter.requires_grad_(False)

    def training_setup(self, training_args) -> None:
        if not self.uses_res_cond_plugin():
            return self._xrgs_original_training_setup(training_args)
        self.ensure_xrgs_condition_anchor_shape()
        self.freeze_base_for_res_cond_plugin()
        lr = getattr(training_args, "res_cond_lr", getattr(training_args, "feature_lr", 1e-3))
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self._res_cond_anchor], "lr": lr, "name": "res_cond_anchor"},
                {"params": self.res_cond_opacity_proj.parameters(), "lr": lr, "name": "res_cond_opacity_proj"},
                {"params": self.res_cond_cov_proj.parameters(), "lr": lr, "name": "res_cond_cov_proj"},
                {"params": self.res_cond_color_proj.parameters(), "lr": lr, "name": "res_cond_color_proj"},
            ],
            lr=0.0,
            eps=1e-15,
        )

    def load_ply_sparse_gaussian(self, path) -> None:
        self._xrgs_original_load_ply(path)
        self.ensure_xrgs_condition_anchor_shape()

    def save_mlp_checkpoints(self, path, *args, **kwargs) -> None:
        _unwrap_condition_mlps(self)
        try:
            self._xrgs_original_save_mlp(path, *args, **kwargs)
        finally:
            _rewrap_condition_mlps(self)
        if self.uses_res_cond_plugin():
            output_dir = Path(path)
            if output_dir.suffix:
                output_dir = output_dir.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(_plugin_state(self), output_dir / "res_cond_plugin.pth")

    def load_mlp_checkpoints(self, path, *args, **kwargs) -> None:
        _unwrap_condition_mlps(self)
        try:
            self._xrgs_original_load_mlp(path, *args, **kwargs)
        finally:
            _rewrap_condition_mlps(self)
        self.ensure_xrgs_condition_anchor_shape()
        plugin_path = Path(path)
        if plugin_path.suffix:
            plugin_path = plugin_path.parent / "res_cond_plugin.pth"
        else:
            plugin_path = plugin_path / "res_cond_plugin.pth"
        if plugin_path.exists() and self.uses_res_cond_plugin():
            _load_plugin_state(self, torch.load(plugin_path, map_location="cuda"))

    def eval(self):
        result = self._xrgs_original_eval()
        if self.uses_res_cond_plugin():
            self.res_cond_opacity_proj.eval()
            self.res_cond_cov_proj.eval()
            self.res_cond_color_proj.eval()
        return result

    def train(self):
        result = self._xrgs_original_train()
        if self.uses_res_cond_plugin():
            self.res_cond_opacity_proj.train()
            self.res_cond_cov_proj.train()
            self.res_cond_color_proj.train()
        return result

    gaussians._xrgs_original_load_ply = original_load_ply
    gaussians._xrgs_original_save_mlp = original_save_mlp
    gaussians._xrgs_original_load_mlp = original_load_mlp
    gaussians._xrgs_original_eval = original_eval
    gaussians._xrgs_original_train = original_train
    gaussians._xrgs_original_training_setup = gaussians.training_setup
    gaussians.ensure_xrgs_condition_anchor_shape = MethodType(ensure_anchor_shape, gaussians)
    gaussians.uses_res_cond_plugin = MethodType(uses_res_cond_plugin, gaussians)
    gaussians.reset_res_cond_plugin_parameters = MethodType(reset_res_cond_plugin_parameters, gaussians)
    gaussians.freeze_base_for_res_cond_plugin = MethodType(freeze_base_for_res_cond_plugin, gaussians)
    gaussians.training_setup = MethodType(training_setup, gaussians)
    gaussians.load_ply_sparse_gaussian = MethodType(load_ply_sparse_gaussian, gaussians)
    gaussians.save_mlp_checkpoints = MethodType(save_mlp_checkpoints, gaussians)
    gaussians.load_mlp_checkpoints = MethodType(load_mlp_checkpoints, gaussians)
    gaussians.eval = MethodType(eval, gaussians)
    gaussians.train = MethodType(train, gaussians)
    gaussians._xrgs_condition_plugin_installed = True
    gaussians.reset_res_cond_plugin_parameters(projection_init_std)


def set_plugin_layout(gaussians, layout: dict[str, tuple[int, int]]) -> None:
    gaussians.res_cond_plugin_layout = layout
    gaussians.ensure_xrgs_condition_anchor_shape()


def _projection_weight(state: dict[str, object], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, dict) or "weight" not in value:
        raise ValueError(f"XRGS plug-in checkpoint is missing {key}.weight")
    return value["weight"]


def _validate_projection_slice(
    name: str,
    key: str,
    source_weight,
    target_weight: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    source = torch.as_tensor(source_weight, dtype=target_weight.dtype, device=target_weight.device)
    if source.ndim != 2:
        raise ValueError(f"{name} {key} weight must be 2D, got shape {tuple(source.shape)}")
    if source.shape[0] != target_weight.shape[0]:
        raise ValueError(
            f"{name} {key} output dim mismatch: checkpoint={source.shape[0]} target={target_weight.shape[0]}"
        )
    if source.shape[1] < dim:
        raise ValueError(f"{name} {key} input dim {source.shape[1]} is smaller than requested slice dim {dim}")
    return source[:, :dim]


def validate_inject_dim(gaussians, requested_inject_dim: int) -> dict[str, int]:
    dims = {
        "opacity": int(gaussians.res_cond_opacity_proj.weight.shape[0]),
        "cov": int(gaussians.res_cond_cov_proj.weight.shape[0]),
        "color": int(gaussians.res_cond_color_proj.weight.shape[0]),
    }
    if requested_inject_dim > 0 and (len(set(dims.values())) != 1 or int(requested_inject_dim) != next(iter(dims.values()))):
        raise ValueError(f"Requested scalar inject_dim={requested_inject_dim} but per-decoder inject dims are {dims}")
    return dims


def zero_disabled_plugin_slices(gaussians, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None:
    if not disabled:
        return
    with torch.no_grad():
        for name in disabled:
            start, end = layout[name]
            gaussians._res_cond_anchor[:, start:end].zero_()
            gaussians.res_cond_opacity_proj.weight[:, start:end].zero_()
            gaussians.res_cond_cov_proj.weight[:, start:end].zero_()
            gaussians.res_cond_color_proj.weight[:, start:end].zero_()


def register_disabled_plugin_slice_hooks(gaussians, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None:
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
    gaussians._res_cond_disable_hook_handles = [
        gaussians._res_cond_anchor.register_hook(lambda grad, mask=anchor_mask: grad * mask),
        gaussians.res_cond_opacity_proj.weight.register_hook(lambda grad, mask=opacity_mask: grad * mask),
        gaussians.res_cond_cov_proj.weight.register_hook(lambda grad, mask=cov_mask: grad * mask),
        gaussians.res_cond_color_proj.weight.register_hook(lambda grad, mask=color_mask: grad * mask),
    ]


def load_plugin_slice(
    gaussians,
    name: str,
    plugin_path: str,
    target_slice: tuple[int, int],
    logger,
    iteration: int = -1,
) -> bool:
    if not plugin_path:
        raise ValueError(f"strict_warm_start requires --{name}_plugin_path for enabled slice '{name}'")
    iteration_dir = resolve_plugin_iteration_dir(plugin_path, iteration)
    state_path = iteration_dir / "res_cond_plugin.pth"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing XRGS plug-in projection checkpoint: {state_path}")
    state = torch.load(state_path, map_location="cuda")
    start, end = target_slice
    dim = end - start
    anchor = torch.as_tensor(state["res_cond_anchor"], dtype=gaussians._res_cond_anchor.dtype, device=gaussians._res_cond_anchor.device)
    if anchor.shape[1] < dim:
        raise ValueError(f"{name} plug-in checkpoint dim {anchor.shape[1]} is smaller than requested slice dim {dim}")
    if anchor.shape[0] != gaussians._res_cond_anchor.shape[0]:
        raise ValueError(f"{name} plug-in anchor count mismatch: {anchor.shape[0]} vs {gaussians._res_cond_anchor.shape[0]}")
    base_anchor = state.get("base_anchor")
    if base_anchor is not None:
        _assert_anchor_match(base_anchor, _base_anchor_tensor(gaussians), f"{name} plug-in")
    opacity_weight = _validate_projection_slice(
        name,
        "opacity_proj",
        _projection_weight(state, "opacity_proj"),
        gaussians.res_cond_opacity_proj.weight,
        dim,
    )
    cov_weight = _validate_projection_slice(
        name,
        "cov_proj",
        _projection_weight(state, "cov_proj"),
        gaussians.res_cond_cov_proj.weight,
        dim,
    )
    color_weight = _validate_projection_slice(
        name,
        "color_proj",
        _projection_weight(state, "color_proj"),
        gaussians.res_cond_color_proj.weight,
        dim,
    )
    with torch.no_grad():
        gaussians._res_cond_anchor[:, start:end].copy_(anchor[:, :dim])
        gaussians.res_cond_opacity_proj.weight[:, start:end].copy_(opacity_weight)
        gaussians.res_cond_cov_proj.weight[:, start:end].copy_(cov_weight)
        gaussians.res_cond_color_proj.weight[:, start:end].copy_(color_weight)
    logger.info("Initialized %s slice [%d:%d] from %s", name, start, end, state_path)
    return True


def require_unified_plugin_checkpoint(model_path: str | Path, iteration: int, total_dim: int) -> None:
    checkpoint_dir = plugin_checkpoint_dir(model_path, iteration)
    state_path = checkpoint_dir / "res_cond_plugin.pth"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing XRGS plug-in checkpoint: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    anchor = state.get("res_cond_anchor")
    if anchor is None or int(anchor.shape[1]) < int(total_dim):
        raise ValueError(f"XRGS plug-in checkpoint dim is smaller than total_dim={total_dim}: {state_path}")
    for key in ["opacity_proj", "cov_proj", "color_proj"]:
        weight = _projection_weight(state, key)
        if int(weight.shape[1]) < int(total_dim):
            raise ValueError(f"XRGS {key} input dim is smaller than total_dim={total_dim}: {state_path}")


def trainable_parameter_count(gaussians) -> int:
    return sum(parameter.numel() for parameter in [
        gaussians._res_cond_anchor,
        *gaussians.res_cond_opacity_proj.parameters(),
        *gaussians.res_cond_cov_proj.parameters(),
        *gaussians.res_cond_color_proj.parameters(),
    ] if parameter.requires_grad)


class ConditionalMLPAdapterMixin:
    supports_condition_injection = True
    condition_injection_note = None

    def _install_condition_plugin(self, gaussians, total_dim: int) -> None:
        install_condition_plugin(gaussians, total_dim)

    def set_plugin_layout(self, gaussians, layout: dict[str, tuple[int, int]]) -> None:
        if not getattr(gaussians, "_xrgs_condition_plugin_installed", False):
            install_condition_plugin(gaussians, max(end for _start, end in layout.values()))
        set_plugin_layout(gaussians, layout)

    def validate_inject_dim(self, gaussians, requested_inject_dim: int) -> dict[str, int]:
        return validate_inject_dim(gaussians, requested_inject_dim)

    def zero_disabled_plugin_slices(self, gaussians, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None:
        zero_disabled_plugin_slices(gaussians, layout, disabled)

    def register_disabled_plugin_slice_hooks(self, gaussians, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None:
        register_disabled_plugin_slice_hooks(gaussians, layout, disabled)

    def load_plugin_slice(self, gaussians, dataset, name: str, plugin_path: str, target_slice: tuple[int, int], logger, iteration: int = -1) -> bool:
        return load_plugin_slice(gaussians, name, plugin_path, target_slice, logger, iteration)

    def require_unified_plugin_checkpoint(self, model_path: str | Path, iteration: int, total_dim: int) -> None:
        require_unified_plugin_checkpoint(model_path, iteration, total_dim)

    def trainable_parameter_count(self, gaussians) -> int:
        return trainable_parameter_count(gaussians)


class BaselineOnlyAdapterMixin:
    supports_condition_injection = False
    condition_injection_note = "This backbone does not expose a shared XRGS MLP decoder injection path through the adapter-only integration."

    def set_plugin_layout(self, gaussians, layout: dict[str, tuple[int, int]]) -> None:
        return None

    def validate_inject_dim(self, gaussians, requested_inject_dim: int) -> dict[str, int]:
        return {}

    def zero_disabled_plugin_slices(self, gaussians, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None:
        return None

    def register_disabled_plugin_slice_hooks(self, gaussians, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None:
        return None

    def load_plugin_slice(self, gaussians, dataset, name: str, plugin_path: str, target_slice: tuple[int, int], logger, iteration: int = -1) -> bool:
        logger.warning("Ignoring %s plug-in slice for %s; Condition Injection is unsupported.", name, self.name)
        return False

    def require_unified_plugin_checkpoint(self, model_path: str | Path, iteration: int, total_dim: int) -> None:
        return None

    def trainable_parameter_count(self, gaussians) -> int:
        return 0
