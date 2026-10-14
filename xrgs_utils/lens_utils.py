from __future__ import annotations

import sys
import types
import importlib.util
import importlib.machinery
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from xrgs_utils.color_utils import luminance_max_match_scale, scale_srgb_in_linear


class DistortMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 9),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            scale_init = torch.log(torch.tensor(torch.e - 1.0))
            bias = torch.cat([torch.zeros(6), scale_init.expand(3)])
            self.mlp[-1].bias.copy_(bias)

    def forward(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(xy)
        xy_offset = out[:, :6].view(-1, 3, 2)
        scale = F.softplus(out[:, 6:]).view(-1, 3, 1)
        xy_expanded = xy.unsqueeze(1).expand(-1, 3, 2)
        return xy_expanded + xy_offset, scale


def apply_distortion(image: torch.Tensor, mlp: nn.Module | None) -> torch.Tensor:
    if mlp is None:
        return image
    channels, height, width = image.shape
    if channels != 3:
        raise ValueError(f"DistortMLP expects a 3-channel image, got {channels}")
    device = image.device
    dtype = image.dtype

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    flat_grid = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)

    sampling_coords, scale_val = mlp(flat_grid)
    input_batch = image.unsqueeze(1)
    sampling_grid = sampling_coords.reshape(height, width, channels, 2).permute(2, 0, 1, 3)
    scale_map = scale_val.reshape(height, width, channels, 1).permute(2, 3, 0, 1)
    distorted = F.grid_sample(input_batch, sampling_grid, align_corners=True, padding_mode="zeros")
    return distorted.squeeze(1) * scale_map.squeeze(1)


def pad_to_square(image: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    channels, height, width = image.shape
    _ = channels
    side = max(height, width)
    if height == width:
        return image
    if height > width:
        pad_left = (height - width) // 2
        pad_right = height - width - pad_left
        return F.pad(image, (pad_left, pad_right, 0, 0), value=pad_value)
    pad_top = (width - height) // 2
    pad_bottom = width - height - pad_top
    return F.pad(image, (0, 0, pad_top, pad_bottom), value=pad_value)


def lens_pad_pixels_for_size(base_size: int, pad_ratio: float = 0.2) -> int:
    base_size = int(base_size)
    if base_size <= 0:
        raise ValueError(f"base_size must be positive, got {base_size}")
    return int(round(base_size * max(float(pad_ratio), 0.0)))


def lens_canvas_size_for_size(base_size: int, pad_ratio: float = 0.2) -> int:
    pad = lens_pad_pixels_for_size(base_size, pad_ratio=pad_ratio)
    return int(base_size) + 2 * pad


def lens_canvas_size_for_image(image: torch.Tensor, pad_ratio: float = 0.2) -> int:
    return lens_canvas_size_for_size(max(image.shape[-2:]), pad_ratio=pad_ratio)


def resolve_dataset_lens_optics_sizes(cameras, pad_ratio: float = 0.2) -> tuple[int, int]:
    sizes = {
        (
            max(camera.original_image[:3].shape[-2:]),
            lens_canvas_size_for_image(camera.original_image[:3], pad_ratio=pad_ratio),
        )
        for camera in cameras
    }
    if not sizes:
        raise ValueError("Cannot resolve optics size from an empty camera list")
    if len(sizes) != 1:
        raise ValueError(f"Lens optics input/output sizes must be consistent across cameras, got {sorted(sizes)}")
    return sizes.pop()


def pad_lens_canvas(
    image: torch.Tensor,
    pad_pixels: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    pad_pixels = int(pad_pixels)
    if pad_pixels < 0:
        raise ValueError(f"pad_pixels must be non-negative, got {pad_pixels}")
    padded = pad_to_square(image, pad_value=pad_value)
    if pad_pixels == 0:
        return padded
    return F.pad(padded, (pad_pixels, pad_pixels, pad_pixels, pad_pixels), value=pad_value)


def prepare_lens_gt(gt_image: torch.Tensor, pad_pixels: int = 80) -> torch.Tensor:
    return pad_lens_canvas(gt_image, pad_pixels=pad_pixels, pad_value=0.0)


def xrgs_optics_root() -> Path:
    return Path(__file__).resolve().parents[1] / "optics"


def default_optics_sim_path() -> Path:
    return xrgs_optics_root() / "optics_simulator"


def default_psf_dir() -> Path:
    return xrgs_optics_root() / "psf_geometric_scan_google_cardboard_rgb"


def _install_matplotlib_stub_if_missing() -> None:
    if "matplotlib" in sys.modules:
        return
    if importlib.util.find_spec("matplotlib") is not None:
        return
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.__path__ = []
    matplotlib_stub.__file__ = "<xrgs-matplotlib-stub>"
    matplotlib_stub.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None, is_package=True)
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.__file__ = "<xrgs-matplotlib-pyplot-stub>"
    pyplot_stub.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", loader=None)

    def _missing(*_args, **_kwargs):
        raise ImportError("matplotlib is not installed in this environment")

    def _pyplot_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _missing

    pyplot_stub.__getattr__ = _pyplot_getattr
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)


def load_optics_wrapper(
    optics_sim_path: str | Path | None = None,
    psf_dir: str | Path | None = None,
    psf_backend: str = "v2",
    device: str = "cuda",
    input_size: int | None = None,
    output_size: int | None = None,
    verbose: bool = False,
):
    if psf_backend != "v2":
        raise ValueError("Only the audited PSF backend 'v2' is supported")
    optics_path = Path(optics_sim_path).resolve() if optics_sim_path else default_optics_sim_path().resolve()
    psf_path = Path(psf_dir).resolve() if psf_dir else default_psf_dir().resolve()
    if not (optics_path / "optics_wrapper.py").exists():
        raise FileNotFoundError(f"Cannot find optics_wrapper.py under {optics_path}")
    if not (psf_path / "psf_dataset_downstream.npz").exists():
        raise FileNotFoundError(f"Cannot find psf_dataset_downstream.npz under {psf_path}")
    if str(optics_path) not in sys.path:
        sys.path.insert(0, str(optics_path))
    package_src = optics_path / "psf_cuda_ext_v2" / "src"
    if package_src.exists() and str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))
    _install_matplotlib_stub_if_missing()

    from optics_wrapper import OpticsSimWrapper

    kwargs = {
        "device": device,
        "psf_dir": str(psf_path),
        "verbose": verbose,
    }
    if input_size is not None:
        kwargs["input_size"] = int(input_size)
    if output_size is not None:
        kwargs["output_size"] = int(output_size)
    return OpticsSimWrapper(**kwargs)


def apply_lens_distortion(
    image: torch.Tensor,
    distort_mlp: nn.Module | None = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    image_square = pad_to_square(image, pad_value=pad_value)
    return apply_distortion(image_square, distort_mlp)


def apply_optics_model(
    image: torch.Tensor,
    optics_wrapper,
    mode: str = "distortion",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Run the optics simulator on an sRGB-like [0, 1] tensor.

    `OpticsSimWrapper` accepts and returns sRGB-like tensors by default. Display
    power and luminance-based brightness scaling explicitly decode to linear
    RGB in `xrgs_utils.color_utils`.
    """
    lens_input = pad_to_square(image, pad_value=pad_value).clamp(0.0, 1.0)
    lens_size = int(lens_input.shape[-1])
    wrapper_input_size = getattr(optics_wrapper, "input_size", None)
    if wrapper_input_size is not None and int(wrapper_input_size) != lens_size:
        raise ValueError(f"Optics input_size={wrapper_input_size} does not match lens input size {lens_size}")
    lens_hwc = optics_wrapper.simulate(lens_input.permute(1, 2, 0), mode=mode)
    result = lens_hwc.permute(2, 0, 1).contiguous()
    wrapper_output_size = getattr(optics_wrapper, "output_size", None)
    if wrapper_output_size is not None and tuple(result.shape[-2:]) != (int(wrapper_output_size), int(wrapper_output_size)):
        raise ValueError(
            f"Optics output_size={wrapper_output_size} does not match simulated output shape {tuple(result.shape[-2:])}"
        )
    return result


def apply_lens_pipeline(
    image: torch.Tensor,
    optics_wrapper,
    mode: str = "distortion",
    distort_mlp: nn.Module | None = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Compatibility helper for the conventional DistortMLP + optics path."""
    distorted = apply_lens_distortion(image, distort_mlp=distort_mlp, pad_value=pad_value)
    return apply_optics_model(distorted, optics_wrapper, mode=mode, pad_value=pad_value)


def match_lens_brightness(image_lens: torch.Tensor, gt_lens: torch.Tensor) -> torch.Tensor:
    scale = luminance_max_match_scale(image_lens, gt_lens)
    return scale_srgb_in_linear(image_lens, scale)
