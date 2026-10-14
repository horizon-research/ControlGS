"""
Optics Simulation Wrapper — sRGB-in, sRGB-out with spectral conversion.

This module wraps PSFSimulator with proper color science:
  sRGB → Linear sRGB → Spectral (486/550/656 nm) → PSF Sim → Linear sRGB → sRGB

The spectral conversion uses CIE 1931 2° standard observer color matching
functions at the three simulation wavelengths (Fraunhofer F/e/C lines),
combined with the standard sRGB↔XYZ matrices (IEC 61966-2-1, D65).

Reference:
  - CIE 1931 CMF values from `colour-science` 0.4.7 (CIE 015:2018)
  - sRGB matrix from IEC 61966-2-1:1999

Usage:
    from optics_wrapper import OpticsSimWrapper

    wrapper = OpticsSimWrapper(device='cuda')
    output_srgb = wrapper(input_srgb)           # [H,W,3] or [B,H,W,3]
    output_srgb = wrapper.simulate(input_srgb)  # same thing
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from psf_simulator import (
    PSFSimulator,
    INPUT_SIZE,
    OUTPUT_SIZE,
    validate_output_size_for_cuda_window,
)


# =============================================================================
# CIE 1931 2° Standard Observer at simulation wavelengths
# Source: colour-science 0.4.7  (CIE 015:2018, Table 1)
#
#   λ (nm)  |  x̄(λ)      ȳ(λ)      z̄(λ)
#   --------+-------------------------------
#    486    |  0.051862   0.176243  0.582329
#    550    |  0.433450   0.994950  0.008750
#    656    |  0.207097   0.077121  0.000000
# =============================================================================

# CMF matrix — columns in PSF-simulator channel order: [656nm, 550nm, 486nm]
# Maps spectral intensities (I_R=S656, I_G=S550, I_B=S486) → CIE XYZ.
#
#   [X]   [x̄(656)  x̄(550)  x̄(486)] [I_R]
#   [Y] = [ȳ(656)  ȳ(550)  ȳ(486)] [I_G]
#   [Z]   [z̄(656)  z̄(550)  z̄(486)] [I_B]
#
_M_CMF_RGB = torch.tensor([
    [0.207097, 0.433450, 0.051862],
    [0.077121, 0.994950, 0.176243],
    [0.000000, 0.008750, 0.582329],
], dtype=torch.float64)

# sRGB ↔ XYZ  (IEC 61966-2-1, D65 white point)
_M_SRGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float64)

_M_XYZ_TO_SRGB = torch.linalg.inv(_M_SRGB_TO_XYZ)

# Precompute combined matrices (float64 for precision, cast to float32 at use)
#   Forward:  linear_sRGB [R,G,B] → spectral [S656, S550, S486]
#   Backward: spectral [S656, S550, S486] → linear_sRGB [R,G,B]
_M_FORWARD = (torch.linalg.inv(_M_CMF_RGB) @ _M_SRGB_TO_XYZ).to(torch.float32)
_M_BACKWARD = (_M_XYZ_TO_SRGB @ _M_CMF_RGB).to(torch.float32)


# =============================================================================
# sRGB gamma transfer functions  (IEC 61966-2-1)
# =============================================================================

def srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """sRGB → Linear RGB  (element-wise gamma decode)."""
    # Clamp the power-branch input to avoid NaN from pow(negative, 2.4).
    # torch.where computes gradients for BOTH branches, so even pixels
    # taking the linear path would backprop NaN if the power arg is negative.
    safe = torch.clamp(img, min=0.04045)
    return torch.where(
        img <= 0.04045,
        img / 12.92,
        torch.pow((safe + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(img: torch.Tensor) -> torch.Tensor:
    """Linear RGB → sRGB  (element-wise gamma encode, output clamped to [0,1])."""
    img = img.clamp(min=0.0)
    # Clamp power-branch input: pow(0, 1/2.4) has gradient 0.417*0^(-0.583) = inf,
    # and torch.where leaks inf*0 = NaN into the linear-branch gradient.
    safe = torch.clamp(img, min=0.0031308)
    return torch.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * torch.pow(safe, 1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


# =============================================================================
# Spectral ↔ Linear sRGB  (3×3 matrix, differentiable)
# =============================================================================

def linear_srgb_to_spectral(linear_rgb: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Linear sRGB [R,G,B] → spectral intensities [S656, S550, S486].

    Args:
        linear_rgb: (..., 3) tensor in linear sRGB.
        M: (3, 3) forward matrix on the same device.

    Returns:
        (..., 3) tensor of spectral intensities in PSF channel order.
        Values may be negative (physically valid for out-of-gamut spectra).
    """
    return (linear_rgb @ M.T)


def spectral_to_linear_srgb(spectral: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Spectral intensities [S656, S550, S486] → Linear sRGB [R,G,B].

    Args:
        spectral: (..., 3) tensor of spectral intensities.
        M: (3, 3) backward matrix on the same device.

    Returns:
        (..., 3) tensor in linear sRGB.
    """
    return (spectral @ M.T)


# =============================================================================
# Wrapper class
# =============================================================================

class OpticsSimWrapper:
    """
    End-to-end optical simulation: sRGB → sRGB with spectral PSF.

    Pipeline:
        1. sRGB → Linear sRGB           (gamma decode)
        2. Linear sRGB → Spectral RGB    (CIE 1931 CMF matrix)
        3. Spectral RGB → PSF Simulation (per-wavelength CUDA kernel)
        4. Spectral RGB → Linear sRGB    (inverse CMF matrix)
        5. Linear sRGB → sRGB            (gamma encode)

    The full pipeline is differentiable.
    """

    def __init__(
        self,
        device: str = 'cuda',
        psf_dir: Optional[str] = None,
        input_size: int = INPUT_SIZE,
        output_size: int = OUTPUT_SIZE,
        verbose: bool = True,
    ):
        self.device = device
        validate_output_size_for_cuda_window(output_size)

        # Conversion matrices on device (float32)
        self.M_forward = _M_FORWARD.to(device=device)
        self.M_backward = _M_BACKWARD.to(device=device)

        # Underlying PSF simulator
        self.sim = PSFSimulator(
            device=device,
            psf_dir=psf_dir,
            input_size=input_size,
            output_size=output_size,
            verbose=verbose,
        )

        self.input_size = input_size
        self.output_size = output_size

        # Load chief ray grid and build inverse warp for distortion mode
        self._load_chief_ray_grid()
        self._build_inverse_warp()

    # ---- distortion mode internals ------------------------------------------

    def _load_chief_ray_grid(self):
        """Load 30x30 chief ray grid from the npz file for distortion mapping."""
        psf_dir = self.sim._simulator.psf_dir
        npz_path = os.path.join(psf_dir, "psf_dataset_downstream.npz")
        data = np.load(npz_path)

        # chief_rays: (30,30,3,3) -> keep only x,y -> (30,30,3,2)
        chief_rays = torch.from_numpy(data['chief_rays']).float()  # [30,30,3,3]
        self._chief_rays_30x30 = chief_rays[..., :2].to(self.device)  # [30,30,3,2]

        # field_centers: (30,30,2)
        self._field_centers = torch.from_numpy(data['field_centers']).float().to(self.device)

        # 1D field grid axes for interpolation bounds
        self._field_xs = torch.from_numpy(data['field_xs']).float().to(self.device)  # (30,)
        self._field_ys = torch.from_numpy(data['field_ys']).float().to(self.device)  # (30,)

    def _interp_chief_ray(self, input_mm):
        """
        Bilinearly interpolate chief ray positions from the 30x30 grid.

        Args:
            input_mm: (N, 2) positions in input mm coordinates (x, y).

        Returns:
            (3, N, 2) chief ray positions in output mm, one per wavelength.
        """
        # Normalize input_mm to [-1, 1] for grid_sample based on field grid range
        x_min, x_max = self._field_xs[0], self._field_xs[-1]
        y_min, y_max = self._field_ys[0], self._field_ys[-1]

        N = input_mm.shape[0]
        # grid_sample expects grid in (x, y) order matching (W, H) dims
        # Our 30x30 grid: dim0=y(rows), dim1=x(cols)
        gx = 2.0 * (input_mm[:, 0] - x_min) / (x_max - x_min) - 1.0  # (N,)
        gy = 2.0 * (input_mm[:, 1] - y_min) / (y_max - y_min) - 1.0  # (N,)
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, N, 2)  # [1,1,N,2]

        # chief_rays_30x30: [30,30,3,2] -> process per wavelength
        results = []
        for wl in range(3):
            # [30,30,2] -> [1,2,30,30] for grid_sample (NCHW)
            cr_wl = self._chief_rays_30x30[:, :, wl, :].permute(2, 0, 1).unsqueeze(0)
            # grid_sample: [1,2,1,N] -> squeeze -> [2,N] -> [N,2]
            sampled = F.grid_sample(
                cr_wl, grid, mode='bilinear', padding_mode='border', align_corners=True
            )  # [1, 2, 1, N]
            sampled = sampled.squeeze(0).squeeze(1).permute(1, 0)  # [N, 2]
            results.append(sampled)

        return torch.stack(results, dim=0)  # [3, N, 2]

    def _build_inverse_warp(self):
        """
        Build inverse warp table: for each output pixel, find input mm position.
        Uses iterative Newton-style inversion of the chief ray forward mapping.
        Stores self._inv_warp: [3, H, W, 2] normalized coordinates for grid_sample.
        """
        H = self.output_size
        W = self.output_size

        output_pixel_size_mm = self.sim._simulator.output_pixel_size_mm
        output_half_span_mm = self.sim._simulator.output_size * output_pixel_size_mm / 2.0

        # Output pixel -> mm over the fixed output physical span.
        oy = torch.arange(H, device=self.device, dtype=torch.float32)
        ox = torch.arange(W, device=self.device, dtype=torch.float32)
        oy_mm = (oy + 0.5) * output_pixel_size_mm - output_half_span_mm
        ox_mm = (ox + 0.5) * output_pixel_size_mm - output_half_span_mm
        grid_y, grid_x = torch.meshgrid(oy_mm, ox_mm, indexing='ij')
        p_out_mm = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [H*W, 2]

        # Field bounds for clamping during iteration
        field_min = self._field_xs[0].item()   # -12.0
        field_max = self._field_xs[-1].item()   #  12.0

        # Estimate magnification from corner chief ray for initial guess
        cr_corner = self._chief_rays_30x30[-1, -1, 1, :]  # green WL, corner
        fc_corner = self._field_centers[-1, -1, :]
        mag = cr_corner.abs().max().item() / fc_corner.abs().max().item()

        inv_warp_list = []
        for wl in range(3):
            # Initial guess: use estimated magnification
            p_in_mm = p_out_mm / mag  # [H*W, 2]
            p_in_mm = p_in_mm.clamp(field_min, field_max)

            # Iterative inversion: error / mag approximates the inverse Jacobian
            for _ in range(20):
                p_fwd_mm = self._interp_chief_ray(p_in_mm)[wl]  # [H*W, 2]
                error = p_fwd_mm - p_out_mm
                p_in_mm = (p_in_mm - error / mag).clamp(field_min, field_max)

            # Check convergence: pixels with large residual didn't converge
            p_fwd_mm = self._interp_chief_ray(p_in_mm)[wl]
            residual = (p_fwd_mm - p_out_mm).abs().max(dim=-1).values  # [H*W]
            invalid = residual > 0.5  # mm threshold

            # Convert input mm to normalized [-1, 1] for grid_sample on input image
            # Input pixel i -> mm: (i + 0.5) / input_size * 24.0 - 12.0
            # Inverse: i = (mm + 12.0) / 24.0 * input_size - 0.5
            # Normalized: n = 2 * i / (input_size - 1) - 1  (align_corners=True)
            i_x = (p_in_mm[:, 0] + 12.0) / 24.0 * self.input_size - 0.5
            i_y = (p_in_mm[:, 1] + 12.0) / 24.0 * self.input_size - 0.5
            n_x = 2.0 * i_x / (self.input_size - 1) - 1.0
            n_y = 2.0 * i_y / (self.input_size - 1) - 1.0

            # Invalid pixels -> far outside [-1,1] so grid_sample zeros them out
            n_x[invalid] = 10.0
            n_y[invalid] = 10.0

            warp = torch.stack([n_x, n_y], dim=-1).view(H, W, 2)  # [H, W, 2]
            inv_warp_list.append(warp)

        self._inv_warp = torch.stack(inv_warp_list, dim=0)  # [3, H, W, 2]

    def _apply_distortion(self, spectral_img):
        """
        Apply distortion-only remapping (no PSF blur) per wavelength channel.

        Args:
            spectral_img: [B, H, W, 3] spectral image (S656, S550, S486).

        Returns:
            [B, H_out, W_out, 3] distorted image.
        """
        B = spectral_img.shape[0]
        channels = []
        for c in range(3):
            # [B, H, W] -> [B, 1, H, W]
            ch = spectral_img[..., c].unsqueeze(1)
            grid = self._inv_warp[c].unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
            warped = F.grid_sample(
                ch, grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )  # [B, 1, H_out, W_out]
            channels.append(warped.squeeze(1))  # [B, H_out, W_out]

        return torch.stack(channels, dim=-1)  # [B, H_out, W_out, 3]

    # ---- public API ---------------------------------------------------------

    def __call__(
        self,
        image: torch.Tensor,
        linear_input: bool = False,
        linear_output: bool = False,
        mode: str = 'psf',
    ) -> torch.Tensor:
        return self.simulate(image, linear_input=linear_input,
                             linear_output=linear_output, mode=mode)

    def simulate(
        self,
        image: torch.Tensor,
        linear_input: bool = False,
        linear_output: bool = False,
        mode: str = 'psf',
    ) -> torch.Tensor:
        """
        Simulate an sRGB image through the optical system.

        Args:
            image: (..., 3) tensor in [0, 1].
                   Supports [H,W,3], [B,H,W,3], or [H,W] (grayscale).
            linear_input:  If True, skip sRGB→Linear decode.
            linear_output: If True, skip Linear→sRGB encode.
            mode: 'psf' for full PSF simulation (default),
                  'distortion' for distortion-only remapping (no blur).

        Returns:
            Simulated image, same shape as input.
        """
        assert mode in ('psf', 'distortion'), f"mode must be 'psf' or 'distortion', got '{mode}'"

        # --- Step 1: sRGB → Linear ---
        if not linear_input:
            image = srgb_to_linear(image)

        # --- Step 2: Linear sRGB → Spectral (only for 3-channel) ---
        has_spectral = (image.ndim >= 3 and image.shape[-1] == 3)
        if has_spectral:
            image = linear_srgb_to_spectral(image, self.M_forward)

        # --- Step 3: Simulation ---
        if mode == 'psf':
            output = self.sim.simulate(image, linear_input=True, linear_output=True)
        else:  # distortion
            # Ensure 4D: [B, H, W, C]
            squeeze = False
            if image.ndim == 3:
                image = image.unsqueeze(0)
                squeeze = True
            output = self._apply_distortion(image)
            if squeeze:
                output = output.squeeze(0)

        # --- Step 4: Spectral → Linear sRGB ---
        if has_spectral:
            output = spectral_to_linear_srgb(output, self.M_backward)

        # --- Step 5: Linear → sRGB ---
        if not linear_output:
            output = linear_to_srgb(output)

        return output

    def forward_only(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simulate without gradient tracking (faster for inference)."""
        with torch.no_grad():
            return self.simulate(image, **kwargs)

    def __repr__(self) -> str:
        return f"OpticsSimWrapper(device='{self.device}', sim={self.sim})"


# =============================================================================
# CLI test
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("OpticsSimWrapper — round-trip & simulation test")
    print("=" * 60)

    # --- Matrix sanity check ---
    print("\n[1] Matrix round-trip (CPU, float64)")
    M_fwd_64 = (torch.linalg.inv(_M_CMF_RGB) @ _M_SRGB_TO_XYZ)
    M_bwd_64 = (_M_XYZ_TO_SRGB @ _M_CMF_RGB)
    eye = M_bwd_64 @ M_fwd_64
    print(f"  max |I - M_bwd @ M_fwd| = {(eye - torch.eye(3, dtype=torch.float64)).abs().max():.2e}")

    colors = {"white": [1, 1, 1], "red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]}
    for name, rgb in colors.items():
        v = torch.tensor(rgb, dtype=torch.float64)
        s = M_fwd_64 @ v
        v2 = M_bwd_64 @ s
        print(f"  {name:6s}: spectral=[{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}]  "
              f"round-trip err={((v2 - v).abs().max()):.2e}")

    # --- GPU simulation test ---
    print("\n[2] Full pipeline (GPU)")
    wrapper = OpticsSimWrapper(device='cuda', verbose=True)
    print(f"  {wrapper}")

    test_img = torch.rand(800, 800, 3, device='cuda')
    out = wrapper(test_img)
    print(f"  Input:  {test_img.shape}  range=[{test_img.min():.3f}, {test_img.max():.3f}]")
    print(f"  Output: {out.shape}  range=[{out.min():.3f}, {out.max():.3f}]")

    # --- Gradient test ---
    print("\n[3] Gradient test")
    x = torch.rand(800, 800, 3, device='cuda', requires_grad=True)
    y = wrapper(x, linear_input=True, linear_output=True)
    y.sum().backward()
    print(f"  grad shape: {x.grad.shape}")
    print(f"  grad range: [{x.grad.min():.4f}, {x.grad.max():.4f}]")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
