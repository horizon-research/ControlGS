#!/usr/bin/env python3
"""
Pure geometric PSF (no diffraction) field-scan with ZOSPy.

Requirements:
    pip install zospy numpy matplotlib tqdm

Usage:
    python psf_extraction.py --config config.json

Example config.json:
    {
        "lens_file": "lens.ZMX",
        "field": {
            "x_range": [-1.638, 1.638],
            "y_range": [-1.638, 1.638],
            "x_points": 9,
            "y_points": 9,
            "sample_mode": "downstream"
        },
        "psf": {
            "sampling": "64x64",
            "image_size_mm": 10,
            "n_rays": 2000000,
            "n_wavelengths": 3
        },
        "output": {
            "dir": "psf_geometric_scan",
            "save_aggregate_npy": true,
            "save_dataset_npz": true,
            "save_grid_png": true,
            "save_chief_ray_png": true
        },
        "visualization": {
            "normalize_per_channel": true,
            "log_gain": null,
            "gamma": 0.7,
            "tile_upscale": 1,
            "tile_border": 0,
            "flip_x": false,
            "flip_y": false,
            "rotate_deg": 0
        }
    }
"""

import argparse
import json
import os
from dataclasses import dataclass, field as dc_field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import zospy as zp
from zospy.analyses.raysandspots.single_ray_trace import SingleRayTrace


@dataclass
class FieldConfig:
    x_range: tuple = (-1.638, 1.638)
    y_range: tuple = (-1.638, 1.638)
    x_points: int = 9
    y_points: int = 9
    sample_mode: str = "downstream"


@dataclass
class PSFConfig:
    sampling: str = "64x64"
    image_size_mm: float = 10.0
    n_rays: int = 2_000_000
    n_wavelengths: int = 3


@dataclass
class OutputConfig:
    dir: str = "psf_geometric_scan"
    save_aggregate_npy: bool = True
    save_dataset_npz: bool = True
    save_grid_png: bool = True
    save_chief_ray_png: bool = True
    aggregate_filename: str = "psf_geo_all"
    chief_ray_filename: str = "chief_rays_all"
    dataset_filename: str = "psf_dataset"
    chief_ray_png_filename: str = "chief_ray_positions"


@dataclass
class VisualizationConfig:
    normalize_per_channel: bool = True
    log_gain: Optional[float] = None
    gamma: Optional[float] = 0.7
    tile_upscale: int = 1
    tile_border: int = 0
    flip_x: bool = False
    flip_y: bool = False
    rotate_deg: int = 0


@dataclass
class Config:
    lens_file: str = "lens.ZMX"
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    psf: PSFConfig = dc_field(default_factory=PSFConfig)
    output: OutputConfig = dc_field(default_factory=OutputConfig)
    visualization: VisualizationConfig = dc_field(default_factory=VisualizationConfig)


def load_config(config_path: str) -> Config:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        data = json.load(f)

    cfg = Config()
    cfg.lens_file = data.get("lens_file", cfg.lens_file)

    if "field" in data:
        f = data["field"]
        cfg.field = FieldConfig(
            x_range=tuple(f.get("x_range", cfg.field.x_range)),
            y_range=tuple(f.get("y_range", cfg.field.y_range)),
            x_points=f.get("x_points", cfg.field.x_points),
            y_points=f.get("y_points", cfg.field.y_points),
            sample_mode=f.get("sample_mode", cfg.field.sample_mode),
        )

    if "psf" in data:
        p = data["psf"]
        cfg.psf = PSFConfig(
            sampling=p.get("sampling", cfg.psf.sampling),
            image_size_mm=p.get("image_size_mm", cfg.psf.image_size_mm),
            n_rays=p.get("n_rays", cfg.psf.n_rays),
            n_wavelengths=p.get("n_wavelengths", cfg.psf.n_wavelengths),
        )

    if "output" in data:
        o = data["output"]
        cfg.output = OutputConfig(
            dir=o.get("dir", cfg.output.dir),
            save_aggregate_npy=o.get("save_aggregate_npy", cfg.output.save_aggregate_npy),
            save_dataset_npz=o.get("save_dataset_npz", cfg.output.save_dataset_npz),
            save_grid_png=o.get("save_grid_png", cfg.output.save_grid_png),
            save_chief_ray_png=o.get("save_chief_ray_png", cfg.output.save_chief_ray_png),
            aggregate_filename=o.get("aggregate_filename", cfg.output.aggregate_filename),
            chief_ray_filename=o.get("chief_ray_filename", cfg.output.chief_ray_filename),
            dataset_filename=o.get("dataset_filename", cfg.output.dataset_filename),
            chief_ray_png_filename=o.get("chief_ray_png_filename", cfg.output.chief_ray_png_filename),
        )

    if "visualization" in data:
        v = data["visualization"]
        cfg.visualization = VisualizationConfig(
            normalize_per_channel=v.get("normalize_per_channel", cfg.visualization.normalize_per_channel),
            log_gain=v.get("log_gain", cfg.visualization.log_gain),
            gamma=v.get("gamma", cfg.visualization.gamma),
            tile_upscale=v.get("tile_upscale", cfg.visualization.tile_upscale),
            tile_border=v.get("tile_border", cfg.visualization.tile_border),
            flip_x=v.get("flip_x", cfg.visualization.flip_x),
            flip_y=v.get("flip_y", cfg.visualization.flip_y),
            rotate_deg=v.get("rotate_deg", cfg.visualization.rotate_deg),
        )

    return cfg


# ========= Utilities =========

def _get_wavelength_indices(system_data):
    wl = system_data.Wavelengths
    return list(range(1, wl.NumberOfWavelengths + 1))


def _get_wavelength_values(system_data):
    wl = system_data.Wavelengths
    values = []
    total = getattr(wl, "NumberOfWavelengths", 0)
    for idx in range(1, total + 1):
        value = float("nan")
        try:
            entry = wl.GetWavelength(idx)
            value = float(getattr(entry, "Wavelength", float("nan")))
        except AttributeError:
            try:
                entry = wl.GetWavelengthNumber(idx)
                value = float(getattr(entry, "Wavelength", float("nan")))
            except AttributeError:
                pass
        values.append(value)
    return values


def _grid_centers(min_val, max_val, count):
    if count <= 0:
        return np.array([], dtype=float)
    span = float(max_val) - float(min_val)
    step = span / count
    return min_val + (step * (0.5 + np.arange(count)))


def _sampling_to_pixels(value):
    if value is None:
        return None
    if isinstance(value, str):
        part = value.lower().replace(' ', '')
        if 'x' in part:
            part = part.split('x')[0]
        try:
            return int(part)
        except ValueError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_chief_ray(oss, widx, field=1, surface=None):
    """Trace the chief ray and return its intersection on the requested surface."""
    trace = SingleRayTrace(
        hx=0.0, hy=0.0, px=0.0, py=0.0,
        wavelength=int(widx),
        field=int(field),
    )
    result = trace.run(oss).data
    data = result.real_ray_trace_data
    if data is None:
        raise RuntimeError("SingleRayTrace did not return real ray trace data")

    target_surface = surface if surface is not None else oss.LDE.NumberOfSurfaces
    row = None
    if "Surf" in data.columns:
        surf_values = data["Surf"].tolist()
        for idx in range(len(surf_values) - 1, -1, -1):
            val = surf_values[idx]
            try:
                numeric = int(float(val))
            except (TypeError, ValueError):
                if surface is None and isinstance(val, str) and val.strip().lower() in {"image", "ims"}:
                    row = data.iloc[idx]
                    break
                continue
            if numeric == target_surface:
                row = data.iloc[idx]
                break
    if row is None:
        row = data.iloc[-1]

    try:
        chief = np.array([
            float(row["X-coordinate"]),
            float(row["Y-coordinate"]),
            float(row["Z-coordinate"]),
        ], dtype=float)
    except KeyError as exc:
        raise RuntimeError("Unexpected SingleRayTrace data format; missing coordinate column") from exc
    return chief


def _run_geo_psf(oss, widx, img_sampling, img_size_mm, rays):
    """Run Geometric Image Analysis and return a numpy array."""
    from zospy.analyses.extendedscene.geometric_image_analysis import GeometricImageAnalysis

    number_of_pixels = _sampling_to_pixels(img_sampling)
    rays = float(rays)
    rays_x_1000 = max(2, int(round(rays / 1000.0)))

    settings_kwargs = dict(
        wavelength=widx,
        field=1,
        rays_x_1000=rays_x_1000,
        use_pixel_interpolation=True,
        delete_vignetted=True,
        remove_vignetting_factors=False,
    )
    if number_of_pixels is not None:
        settings_kwargs["number_of_pixels"] = number_of_pixels
    if img_size_mm is not None:
        settings_kwargs["image_size"] = float(img_size_mm)

    analysis = GeometricImageAnalysis(**settings_kwargs)
    result = analysis.run(oss)
    data_df = result.data
    if data_df is None:
        raise RuntimeError("GeometricImageAnalysis returned no data grid; check lens and settings")

    arr = data_df.to_numpy(dtype=float)
    chief_ray = _trace_chief_ray(oss, widx=widx, field=1)
    return arr, chief_ray


def _set_single_field(system_data, x, y, weight=1.0):
    """Replace field table with a single field at (x, y)."""
    f = system_data.Fields
    if f.NumberOfFields >= 1:
        fi = f.GetField(1)
        fi.Y = float(y)
        fi.X = float(x)
        fi.Weight = float(weight)
        while f.NumberOfFields > 1:
            f.RemoveField(f.NumberOfFields)
    else:
        f.AddField(float(y), float(x), float(weight))


def _backup_fields(system_data):
    f = system_data.Fields
    fields = []
    for i in range(1, f.NumberOfFields + 1):
        fi = f.GetField(i)
        fields.append((fi.Y, fi.X, fi.Weight))
    return fields


def _restore_fields(system_data, backup):
    f = system_data.Fields
    f.DeleteAllFields()
    if not backup:
        f.AddField(0.0, 0.0, 1.0)
        return
    for (Y, X, W) in backup:
        f.AddField(float(Y), float(X), float(W))


def _psf_stack_to_rgb(psf_stack, vis_cfg: VisualizationConfig):
    """Convert a (λ, H, W) PSF stack into normalized RGB image."""
    if psf_stack.ndim != 3:
        raise ValueError(f"Expected PSF stack with 3 dims (λ,H,W), got shape {psf_stack.shape}")

    wavelengths = psf_stack.shape[0]
    rgb_channels = []
    for idx in range(3):
        src_idx = max(0, wavelengths - 1 - idx)
        sel = psf_stack[src_idx].copy()
        if vis_cfg.normalize_per_channel:
            peak_val = sel.max()
            if peak_val > 1e-12:
                sel = sel / peak_val
        if vis_cfg.log_gain:
            gain = float(vis_cfg.log_gain)
            sel = np.log1p(gain * sel) / np.log1p(gain)
        if vis_cfg.flip_x:
            sel = np.fliplr(sel)
        if vis_cfg.flip_y:
            sel = np.flipud(sel)
        if vis_cfg.rotate_deg % 360:
            k = (vis_cfg.rotate_deg // 90) % 4
            sel = np.rot90(sel, k=k)
        rgb_channels.append(sel)
    rgb = np.stack(rgb_channels, axis=-1)
    if vis_cfg.gamma:
        gamma = float(vis_cfg.gamma)
        if gamma > 0:
            rgb = np.clip(rgb, 0.0, None) ** gamma
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def run_extraction(cfg: Config):
    """Main extraction logic."""
    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)

    # Connect to Zemax
    print(f"Connecting to Zemax and loading {cfg.lens_file}...")
    zos = zp.ZOS()
    oss = zos.connect()
    system_data = oss.SystemData
    oss.load(cfg.lens_file, saveifneeded=False)
    oss.make_sequential()

    # Prepare field grid
    field_cfg = cfg.field
    if field_cfg.sample_mode == "downstream":
        xs = np.linspace(field_cfg.x_range[0], field_cfg.x_range[1], field_cfg.x_points + 1)
        ys = np.linspace(field_cfg.y_range[0], field_cfg.y_range[1], field_cfg.y_points + 1)
    else:  # "visualize"
        xs = _grid_centers(field_cfg.x_range[0], field_cfg.x_range[1], field_cfg.x_points)
        ys = _grid_centers(field_cfg.y_range[0], field_cfg.y_range[1], field_cfg.y_points)

    NUM_X = len(xs)
    NUM_Y = len(ys)
    fields_backup = _backup_fields(system_data)

    # Choose wavelengths
    all_widx = _get_wavelength_indices(system_data)
    n_wl = cfg.psf.n_wavelengths
    if n_wl is None or n_wl <= 0:
        widx_list = all_widx
    else:
        widx_list = all_widx[:min(n_wl, len(all_widx))]

    print(f"Field grid: {NUM_X} x {NUM_Y}, Wavelengths: {len(widx_list)}")

    # Run scan
    all_results = {}
    chief_rays = {}
    field_coords = {}

    try:
        for iy, y in enumerate(tqdm(ys, desc="Field Y", unit="row", total=NUM_Y)):
            for ix, x in enumerate(tqdm(xs, desc="Field X", unit="col", leave=False, total=NUM_X)):
                _set_single_field(system_data, x, y)

                psfs = []
                chiefs = []
                for widx in tqdm(widx_list, desc="Wavelengths", unit="wl", leave=False, total=len(widx_list)):
                    arr, chief = _run_geo_psf(
                        oss,
                        widx=widx,
                        img_sampling=cfg.psf.sampling,
                        img_size_mm=cfg.psf.image_size_mm,
                        rays=cfg.psf.n_rays,
                    )
                    psfs.append(arr)
                    chiefs.append(chief)
                psf_stack = np.stack(psfs, axis=0)
                chief_stack = np.stack(chiefs, axis=0)
                all_results[(ix, iy)] = psf_stack
                chief_rays[(ix, iy)] = chief_stack
                field_coords[(ix, iy)] = (x, y)

        # Save grid visualization
        if cfg.output.save_grid_png and all_results:
            _save_grid_png(all_results, NUM_X, NUM_Y, cfg, out_dir)

        # Save chief ray visualization
        if cfg.output.save_chief_ray_png and chief_rays:
            _save_chief_ray_png(chief_rays, widx_list, NUM_X, NUM_Y, cfg, out_dir)

    finally:
        _restore_fields(system_data, fields_backup)

    # Save aggregate NPY
    if cfg.output.save_aggregate_npy and all_results:
        _save_aggregate_npy(all_results, chief_rays, field_coords, widx_list, NUM_X, NUM_Y, cfg, out_dir)

    # Save dataset NPZ
    if cfg.output.save_dataset_npz and all_results:
        wavelength_values = np.array(_get_wavelength_values(system_data))
        _save_dataset_npz(all_results, chief_rays, field_coords, xs, ys, widx_list, wavelength_values, NUM_X, NUM_Y, cfg, out_dir)

    print(f"Done. Results in: {out_dir}")


def _save_grid_png(all_results, NUM_X, NUM_Y, cfg: Config, out_dir):
    vis_cfg = cfg.visualization
    sample = next(iter(all_results.values()))
    h, w = sample.shape[1:]
    display_h = h * vis_cfg.tile_upscale
    display_w = w * vis_cfg.tile_upscale
    tile_border = max(0, int(vis_cfg.tile_border))
    total_h = NUM_Y * (display_h + 2 * tile_border)
    total_w = NUM_X * (display_w + 2 * tile_border)
    grid_img = np.zeros((total_h, total_w, 3), dtype=float)

    for iy in range(NUM_Y):
        for ix in range(NUM_X):
            psf = all_results.get((ix, iy))
            if psf is None:
                continue
            rgb = _psf_stack_to_rgb(psf, vis_cfg)
            if vis_cfg.tile_upscale > 1:
                rgb = np.repeat(np.repeat(rgb, vis_cfg.tile_upscale, axis=0), vis_cfg.tile_upscale, axis=1)
            top = iy * (display_h + 2 * tile_border) + tile_border
            left = ix * (display_w + 2 * tile_border) + tile_border
            grid_img[top:top + display_h, left:left + display_w] = rgb

    plt.figure(figsize=(cfg.field.x_points, cfg.field.y_points))
    plt.imshow(grid_img, origin='lower')
    plt.axis('off')
    plt.tight_layout()
    grid_path = os.path.join(out_dir, f"psf_geo_grid_{cfg.field.sample_mode}.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {grid_path}")


def _save_chief_ray_png(chief_rays, widx_list, NUM_X, NUM_Y, cfg: Config, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(widx_list) > 1:
        cmap = plt.get_cmap('tab10', len(widx_list))
    wavelength_colors = {1: 'tab:blue', 2: 'tab:green', 3: 'tab:red'}

    for w_i, widx in enumerate(widx_list):
        xs_plot = []
        ys_plot = []
        for iy in range(NUM_Y):
            for ix in range(NUM_X):
                coord = chief_rays.get((ix, iy))
                if coord is None:
                    continue
                coord = coord[w_i]
                x_val, y_val = coord[:2]
                if np.isfinite(x_val) and np.isfinite(y_val):
                    xs_plot.append(x_val)
                    ys_plot.append(y_val)
        if not xs_plot:
            continue
        color = wavelength_colors.get(widx)
        if color is None:
            color = cmap(w_i) if len(widx_list) > 1 else 'tab:blue'
        ax.scatter(xs_plot, ys_plot, s=30, alpha=0.8, label=f"λ #{widx}", color=color)

    ax.set_xlabel("Chief ray X (system units)")
    ax.set_ylabel("Chief ray Y (system units)")
    ax.set_title("Chief ray intersections at image plane")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', alpha=0.3)
    if len(widx_list) > 1:
        ax.legend(loc='best', fontsize='small')
    fig.tight_layout()
    chief_png_path = os.path.join(out_dir, f"{cfg.output.chief_ray_png_filename}_{cfg.field.sample_mode}.png")
    fig.savefig(chief_png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {chief_png_path}")


def _save_aggregate_npy(all_results, chief_rays, field_coords, widx_list, NUM_X, NUM_Y, cfg: Config, out_dir):
    sample = next(iter(all_results.values()))
    agg_shape = (NUM_Y, NUM_X) + sample.shape
    aggregate = np.zeros(agg_shape, dtype=float)
    chief_shape = (NUM_Y, NUM_X, len(widx_list), 3)
    chief_aggregate = np.zeros(chief_shape, dtype=float)
    field_array = np.zeros((NUM_Y, NUM_X, 2), dtype=float)

    for (ix, iy), data in all_results.items():
        if iy < NUM_Y and ix < NUM_X:
            aggregate[iy, ix] = data
            chief_aggregate[iy, ix] = chief_rays[(ix, iy)]
            field_array[iy, ix] = field_coords[(ix, iy)]

    mode = cfg.field.sample_mode
    agg_path = os.path.join(out_dir, f"{cfg.output.aggregate_filename}_{mode}.npy")
    np.save(agg_path, aggregate)
    chief_path = os.path.join(out_dir, f"{cfg.output.chief_ray_filename}_{mode}.npy")
    np.save(chief_path, chief_aggregate)
    field_path = os.path.join(out_dir, f"field_centers_{mode}.npy")
    np.save(field_path, field_array)
    print(f"Saved: {agg_path}, {chief_path}, {field_path}")


def _save_dataset_npz(all_results, chief_rays, field_coords, xs, ys, widx_list, wavelength_values, NUM_X, NUM_Y, cfg: Config, out_dir):
    sample = next(iter(all_results.values()))
    agg_shape = (NUM_Y, NUM_X) + sample.shape
    aggregate = np.zeros(agg_shape, dtype=float)
    chief_shape = (NUM_Y, NUM_X, len(widx_list), 3)
    chief_aggregate = np.zeros(chief_shape, dtype=float)
    field_array = np.zeros((NUM_Y, NUM_X, 2), dtype=float)

    for (ix, iy), data in all_results.items():
        if iy < NUM_Y and ix < NUM_X:
            aggregate[iy, ix] = data
            chief_aggregate[iy, ix] = chief_rays[(ix, iy)]
            field_array[iy, ix] = field_coords[(ix, iy)]

    mode = cfg.field.sample_mode
    dataset_path = os.path.join(out_dir, f"{cfg.output.dataset_filename}_{mode}.npz")
    np.savez(
        dataset_path,
        psf=aggregate,
        chief_rays=chief_aggregate,
        field_centers=field_array,
        field_xs=xs,
        field_ys=ys,
        wavelengths=np.array(widx_list, dtype=int),
        wavelength_values=wavelength_values,
    )
    print(f"Saved: {dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract geometric PSF from Zemax OpticStudio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python psf_extraction.py --config config.json

Generate default config:
    python psf_extraction.py --generate-config config.json
        """
    )
    parser.add_argument("--config", "-c", type=str, help="Path to JSON config file")
    parser.add_argument("--generate-config", type=str, metavar="PATH",
                        help="Generate a default config file and exit")
    args = parser.parse_args()

    if args.generate_config:
        default_config = {
            "lens_file": "lens.ZMX",
            "field": {
                "x_range": [-1.638, 1.638],
                "y_range": [-1.638, 1.638],
                "x_points": 9,
                "y_points": 9,
                "sample_mode": "downstream"
            },
            "psf": {
                "sampling": "64x64",
                "image_size_mm": 10,
                "n_rays": 2000000,
                "n_wavelengths": 3
            },
            "output": {
                "dir": "psf_geometric_scan",
                "save_aggregate_npy": True,
                "save_dataset_npz": True,
                "save_grid_png": True,
                "save_chief_ray_png": True,
                "aggregate_filename": "psf_geo_all",
                "chief_ray_filename": "chief_rays_all",
                "dataset_filename": "psf_dataset",
                "chief_ray_png_filename": "chief_ray_positions"
            },
            "visualization": {
                "normalize_per_channel": True,
                "log_gain": None,
                "gamma": 0.7,
                "tile_upscale": 1,
                "tile_border": 0,
                "flip_x": False,
                "flip_y": False,
                "rotate_deg": 0
            }
        }
        with open(args.generate_config, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Generated default config: {args.generate_config}")
        return

    if not args.config:
        parser.print_help()
        print("\nError: --config is required")
        return

    cfg = load_config(args.config)
    run_extraction(cfg)


if __name__ == "__main__":
    main()
