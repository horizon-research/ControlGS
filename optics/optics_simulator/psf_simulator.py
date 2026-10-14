#!/usr/bin/env python3
"""
PSF Simulator - A convenient Python class for optical PSF simulation.

This module provides a high-level interface for simulating images through
a point spread function (PSF) based optical system using optimized CUDA kernels.

Example usage:
    from psf_simulator import PSFSimulator

    # Initialize simulator
    sim = PSFSimulator(device='cuda')

    # Simulate an image (input: [H, W, C] or [B, H, W, C], values in [0, 1])
    output = sim(input_image)

    # With gradient support for optimization
    input_img.requires_grad = True
    output = sim(input_img, linear_input=True, linear_output=True)
    loss = (output - target).abs().mean()
    loss.backward()  # Gradients flow back to input_img
"""

import os
import sys
import torch
from typing import Optional

# Ensure local imports work
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from image_simulation_psf_gpu import (
    PSFSimulatorGPU,
    srgb_to_linear,
    linear_to_srgb,
    INPUT_SIZE,
    OUTPUT_SIZE,
    validate_output_size_for_cuda_window,
)
import psf_cuda_ext_v2 as _psf_cuda


class PSFSimulator:
    """
    High-level PSF Simulator for optical image simulation.

    This class provides a simple interface to simulate images through a
    spatially-varying PSF optical system. It automatically handles:
    - Grayscale (1 channel) and RGB (3 channels) images
    - Batch processing
    - sRGB <-> Linear color space conversion (optional)
    - Differentiable simulation for gradient-based optimization

    Attributes:
        device: The device (cuda/cpu) where computation runs.
        input_size: Expected input image size (default: 800).
        output_size: Output image size (default: 800).

    Example:
        >>> sim = PSFSimulator(device='cuda')
        >>> input_img = torch.rand(800, 800, 3, device='cuda')
        >>> output_img = sim(input_img)  # Returns [800, 800, 3]

        >>> # For optimization (differentiable)
        >>> x = torch.rand(800, 800, 3, device='cuda', requires_grad=True)
        >>> out = sim(x, linear_input=True, linear_output=True)
        >>> loss = (out - target).abs().mean()
        >>> loss.backward()
    """

    def __init__(
        self,
        device: str = 'cuda',
        psf_dir: Optional[str] = None,
        input_size: int = INPUT_SIZE,
        output_size: int = OUTPUT_SIZE,
        verbose: bool = True,
    ):
        """
        Initialize the PSF Simulator.

        Args:
            device: Device to run on ('cuda' or 'cpu'). Default: 'cuda'.
            psf_dir: Path to PSF data directory. If None, uses default location.
            input_size: Input image size. Default: 800.
            output_size: Output image size. Default: 800.
            verbose: Print initialization info. Default: True.
        """
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        validate_output_size_for_cuda_window(output_size)

        # Suppress output if not verbose
        if not verbose:
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                self._simulator = PSFSimulatorGPU(
                    device=device,
                    psf_dir=psf_dir,
                    input_size=input_size,
                    output_size=output_size,
                )
        else:
            self._simulator = PSFSimulatorGPU(
                device=device,
                psf_dir=psf_dir,
                input_size=input_size,
                output_size=output_size,
            )

        # Cache frequently used values
        self._precomp = self._simulator.precomp
        self._psf_geo = self._simulator.psf_geo
        self._out_size = self._simulator.output_size
        self._out_pix_mm = self._simulator.output_pixel_size_mm
        self._psf_radius_mm = self._simulator.psf_radius_mm
        self._psf_radius_px = self._simulator.psf_radius_pixels

        # Precompute tensors in correct format
        self._weights = self._precomp['weights_sampling_dtype'].contiguous()
        self._nbr_iy = self._precomp['nbr_iy'].to(dtype=torch.int32).contiguous()
        self._nbr_ix = self._precomp['nbr_ix'].to(dtype=torch.int32).contiguous()
        self._chief_xy_all = self._precomp['chief_xy_mm']  # [3, N, 2]

        # Wavelength mapping: RGB channel -> wavelength index
        # R(c=0) -> wl=2, G(c=1) -> wl=1, B(c=2) -> wl=0
        self._wl_of_channel = [2, 1, 0]

    def __call__(
        self,
        image: torch.Tensor,
        linear_input: bool = False,
        linear_output: bool = False,
    ) -> torch.Tensor:
        """
        Simulate an image through the optical system.

        Args:
            image: Input image tensor. Supported shapes:
                   - [H, W]: Grayscale image
                   - [H, W, C]: Single image with C channels
                   - [B, H, W, C]: Batch of images
                   Values should be in [0, 1].
            linear_input: If True, input is in linear RGB. If False (default),
                          input is in sRGB and will be converted to linear.
            linear_output: If True, output is in linear RGB. If False (default),
                           output is converted to sRGB.

        Returns:
            Simulated output image with same shape as input.
        """
        return self.simulate(image, linear_input=linear_input, linear_output=linear_output)

    def simulate(
        self,
        image: torch.Tensor,
        linear_input: bool = False,
        linear_output: bool = False,
    ) -> torch.Tensor:
        """
        Simulate an image through the optical system.

        Args:
            image: Input image tensor. Supported shapes:
                   - [H, W]: Grayscale image (will be treated as single channel)
                   - [H, W, C]: Single image with C channels
                   - [B, H, W, C]: Batch of images
                   Values should be in [0, 1].
            linear_input: If True, input is in linear RGB. If False (default),
                          input is in sRGB and will be converted to linear.
            linear_output: If True, output is in linear RGB. If False (default),
                           output is converted to sRGB.

        Returns:
            Simulated output image with same shape as input.
        """
        # Handle input shape
        original_ndim = image.ndim
        original_shape = image.shape

        if image.ndim == 2:
            # [H, W] -> [1, H, W, 1]
            image = image.unsqueeze(0).unsqueeze(-1)
        elif image.ndim == 3:
            # [H, W, C] -> [1, H, W, C]
            image = image.unsqueeze(0)

        assert image.ndim == 4, f"Expected 2D, 3D or 4D tensor, got {original_ndim}D"
        B, H, W, C = image.shape
        assert H == self.input_size and W == self.input_size, \
            f"Expected input size {self.input_size}x{self.input_size}, got {H}x{W}"
        assert C <= 3, f"Maximum 3 channels (RGB) supported, got {C}"

        # Move to device if needed
        if image.device != torch.device(self.device):
            image = image.to(self.device)

        # Convert to linear if needed
        if not linear_input:
            image = srgb_to_linear(image)

        # Run simulation for each channel
        output = self._simulate_channels(image)

        # Convert to sRGB if needed
        if not linear_output:
            output = linear_to_srgb(output.clamp(0, 1))

        # Restore original shape
        if original_ndim == 2:
            output = output.squeeze(0).squeeze(-1)
        elif original_ndim == 3:
            output = output.squeeze(0)

        return output

    def _simulate_channels(self, image_bhwc: torch.Tensor) -> torch.Tensor:
        """
        Internal: Simulate all channels.

        Args:
            image_bhwc: [B, H, W, C] tensor in linear RGB.

        Returns:
            [B, H, W, C] simulated output.
        """
        B, H, W, C = image_bhwc.shape
        N = H * W

        # Reshape to [B, N, C]
        intensities_bnc = image_bhwc.reshape(B, N, C)

        outputs = []
        for c in range(C):
            # Get wavelength index: R(0)->wl=2, G(1)->wl=1, B(2)->wl=0
            wl_idx = self._wl_of_channel[c]

            # Get intensities for this channel: [B, N]
            intens_bn = intensities_bnc[:, :, c]

            # Get chief ray for this wavelength
            chief_xy_c = self._chief_xy_all[wl_idx]

            # Run CUDA kernel (differentiable)
            out_c = _psf_cuda.psf_accumulate(
                intens_bn,
                self._nbr_iy,
                self._nbr_ix,
                self._weights,
                chief_xy_c,
                self._psf_geo,
                int(wl_idx),
                int(self._out_size),
                float(self._out_pix_mm),
                float(self._psf_radius_mm),
                int(self._psf_radius_px),
            )  # [B, Hout, Wout]

            outputs.append(out_c)

        # Stack channels: [B, Hout, Wout, C]
        output = torch.stack(outputs, dim=-1)
        return output

    def forward_only(
        self,
        image: torch.Tensor,
        linear_input: bool = False,
        linear_output: bool = False,
    ) -> torch.Tensor:
        """
        Simulate without gradient tracking (faster for inference).

        Args:
            Same as simulate().

        Returns:
            Simulated output (no gradient).
        """
        with torch.no_grad():
            return self.simulate(image, linear_input=linear_input, linear_output=linear_output)

    @property
    def is_differentiable(self) -> bool:
        """Check if the simulator supports gradient computation."""
        return True

    def __repr__(self) -> str:
        return (
            f"PSFSimulator(device='{self.device}', "
            f"input_size={self.input_size}, output_size={self.output_size})"
        )


def load_image(path: str, device: str = 'cuda', size: int = INPUT_SIZE) -> torch.Tensor:
    """
    Load an image file and prepare it for simulation.

    Args:
        path: Path to image file (PNG, JPG, etc.)
        device: Device to load to.
        size: Target size (will resize if needed).

    Returns:
        [H, W, 3] tensor in sRGB, values in [0, 1].
    """
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def save_image(path: str, tensor: torch.Tensor, is_linear: bool = False):
    """
    Save a tensor as an image file.

    Args:
        path: Output path.
        tensor: [H, W, 3] or [H, W] tensor, values in [0, 1].
        is_linear: If True, converts from linear to sRGB before saving.
    """
    import numpy as np
    from PIL import Image

    if is_linear:
        tensor = linear_to_srgb(tensor.clamp(0, 1))
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    img_u8 = (arr * 255.0).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


if __name__ == "__main__":
    print("=" * 60)
    print("PSFSimulator Test")
    print("=" * 60)

    sim = PSFSimulator(device='cuda')
    print(f"\nInitialized: {sim}")

    # Test 1: Basic simulation
    print("\n[Test 1] Basic RGB simulation...")
    test_input = torch.rand(800, 800, 3, device='cuda')
    output = sim(test_input)
    print(f"  Input:  {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Range:  [{output.min():.4f}, {output.max():.4f}]")

    # Test 2: Grayscale
    print("\n[Test 2] Grayscale simulation...")
    gray_input = torch.rand(800, 800, device='cuda')
    gray_output = sim(gray_input)
    print(f"  Input:  {gray_input.shape}")
    print(f"  Output: {gray_output.shape}")

    # Test 3: Batch processing
    print("\n[Test 3] Batch simulation...")
    batch_input = torch.rand(2, 800, 800, 3, device='cuda')
    batch_output = sim(batch_input)
    print(f"  Input:  {batch_input.shape}")
    print(f"  Output: {batch_output.shape}")

    # Test 4: Gradient computation
    print("\n[Test 4] Gradient computation...")
    grad_input = torch.rand(800, 800, 3, device='cuda', requires_grad=True)
    grad_output = sim(grad_input, linear_input=True, linear_output=True)
    loss = grad_output.sum()
    loss.backward()
    print(f"  Gradient computed: {grad_input.grad is not None}")
    print(f"  Gradient shape: {grad_input.grad.shape}")
    print(f"  Gradient range: [{grad_input.grad.min():.4f}, {grad_input.grad.max():.4f}]")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
