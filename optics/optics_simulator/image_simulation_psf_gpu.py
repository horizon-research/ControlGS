"""
Image Simulation using PSF - GPU Accelerated Version
基于 PSF 的图像模拟 - GPU 加速版本

Uses PSF (Point Spread Function) data extracted from Zemax to simulate image 
propagation through optical system.
使用从 Zemax 提取的 PSF（点扩散函数）数据模拟图像通过光学系统的传播。

================================================================================
ALGORITHM OVERVIEW / 算法概述
================================================================================

For each input pixel:
对于每个输入像素：

  1. Interpolate PSF from 4 corner samples (bilinear in field space)
     从 4 个角点样本插值 PSF（在视场空间双线性插值）
     
  2. For each output pixel in PSF's influence region:
     对于 PSF 影响区域内的每个输出像素：
     
     a) Compute exact offset from chief ray to output pixel center
        计算从主光线到输出像素中心的精确偏移
        
     b) Sample PSF at that offset using bilinear interpolation
        在该偏移处用双线性插值采样 PSF
        
     c) Accumulate: output[x,y] += input_intensity * PSF_value * pixel_area
        累加：output[x,y] += 输入强度 * PSF值 * 像素面积

KEY INSIGHT: By sampling PSF at exact output pixel centers (not rounding to 
grid), we eliminate grid artifacts caused by chief ray sub-pixel misalignment.
关键洞察：通过在精确的输出像素中心采样 PSF（而非舍入到网格），我们消除了
主光线子像素不对齐造成的格状误差。

================================================================================
SPECIFICATIONS / 规格
================================================================================

PSF Data:
  - Size: 64×64 pixels per PSF
  - Physical coverage: 10mm × 10mm (5mm radius)
  - Storage: [field_y, field_x, wavelength, 64, 64]
  - Centered at chief ray position (not grid-aligned)

Input Image:
  - Size: 512×512 pixels
  - Physical span: 3.276mm × 3.276mm (±1.638mm)
  - Space: Linear RGB

Output Image:
  - Size: 512×512 pixels  
  - Physical span: 100mm × 100mm
  - Space: Linear RGB

This version uses PyTorch for GPU acceleration and chunked processing for
memory efficiency.
此版本使用 PyTorch 进行 GPU 加速，并采用分块处理以提高内存效率。
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
try:
    import psf_cuda_ext_v2 as psf_cuda
except Exception:
    psf_cuda = None

# Cache directory for precomputed interpolation data
CACHE_DIR = "psf_cache_gpu"

# ========= Configuration =========
# Default PSF directory - can be overridden in PSFSimulatorGPU constructor
PSF_DIR_DEFAULT = r"../psf_geometric_scan_google_cardboard_rgb"
PSF_DATA_FILE = "psf_dataset_downstream.npz"  # npz format

# Input/Output specifications
INPUT_SIZE = 800  # pixels
INPUT_SPAN_MM = 24.0  # mm (from -12 to 12, matching field range)
OUTPUT_SIZE = 800  # pixels
OUTPUT_SPAN_MM = 480  # mm

# PSF specifications
PSF_SIZE = 128  # pixels
PSF_SPAN_MM = 50.0  # mm (physical extent of PSF, from config image_size_mm)
PSF_CUDA_MAX_WIN_SIZE = 256


def psf_window_size_for_output(output_size):
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    output_pixel_size_mm = OUTPUT_SPAN_MM / output_size
    psf_radius_pixels = int(np.ceil((PSF_SPAN_MM / 2) / output_pixel_size_mm))
    return 2 * psf_radius_pixels + 1


def max_output_size_for_cuda_window(max_win_size=PSF_CUDA_MAX_WIN_SIZE):
    max_radius_pixels = (int(max_win_size) - 1) // 2
    return int(np.floor(max_radius_pixels * OUTPUT_SPAN_MM / (PSF_SPAN_MM / 2)))


def validate_output_size_for_cuda_window(output_size, max_win_size=PSF_CUDA_MAX_WIN_SIZE):
    win_hw = psf_window_size_for_output(output_size)
    if win_hw > max_win_size:
        max_output_size = max_output_size_for_cuda_window(max_win_size)
        raise ValueError(
            f"output_size={output_size} requires PSF window size {win_hw}, "
            f"but psf_cuda_ext_v2 supports at most {max_win_size}. "
            f"Use output_size <= {max_output_size}, or rebuild the CUDA extension "
            f"with a larger MAX_WIN_SIZE."
        )
    return win_hw


def srgb_to_linear(img):
    """
    Convert sRGB image to linear RGB.
    
    Args:
        img: torch tensor in range [0, 1] in sRGB color space
    
    Returns:
        torch tensor in linear RGB color space
    """
    # sRGB to linear conversion
    # For values <= 0.04045: linear = sRGB / 12.92
    # For values > 0.04045: linear = ((sRGB + 0.055) / 1.055) ^ 2.4
    # Clamp power-branch input: torch.where computes gradients for both branches,
    # so pow(negative, 2.4) would produce NaN that leaks into the gradient.
    safe = torch.clamp(img, min=0.04045)
    linear = torch.where(
        img <= 0.04045,
        img / 12.92,
        torch.pow((safe + 0.055) / 1.055, 2.4)
    )
    return linear


def linear_to_srgb(img):
    """
    Convert linear RGB image to sRGB.
    
    Args:
        img: torch tensor in linear RGB color space
    
    Returns:
        torch tensor in range [0, 1] in sRGB color space
    """
    # Clip to valid range
    img = torch.clamp(img, 0, None)

    # Linear to sRGB conversion
    # For values <= 0.0031308: sRGB = linear * 12.92
    # For values > 0.0031308: sRGB = 1.055 * linear^(1/2.4) - 0.055
    # Clamp power-branch input: pow(0, 1/2.4) has inf gradient,
    # and torch.where leaks inf*0 = NaN into the linear-branch gradient.
    safe = torch.clamp(img, min=0.0031308)
    srgb = torch.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * torch.pow(safe, 1.0 / 2.4) - 0.055
    )
    return torch.clamp(srgb, 0, 1)


def median_filter_2d_torch(input_tensor, kernel_size=3):
    """
    Apply 2D median filter using torch (GPU-accelerated).
    
    Args:
        input_tensor: torch tensor of shape (H, W) or (B, C, H, W)
        kernel_size: size of the median filter kernel
    
    Returns:
        filtered tensor of same shape
    """
    # Ensure 4D tensor for processing
    needs_squeeze = False
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        needs_squeeze = True
    elif input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Unfold to get all kernel patches
    patches = F.unfold(input_tensor, kernel_size, padding=kernel_size//2)
    # Reshape: (B, C*k*k, L) where L is number of patches
    patches = patches.reshape(input_tensor.shape[0], input_tensor.shape[1], kernel_size*kernel_size, -1)
    # Take median along kernel dimension
    filtered = patches.median(dim=2)[0]
    # Reshape back to image
    filtered = filtered.reshape(input_tensor.shape[0], input_tensor.shape[1], 
                                input_tensor.shape[2], input_tensor.shape[3])
    
    if needs_squeeze:
        filtered = filtered.squeeze(0).squeeze(0)
    
    return filtered




def bilinear_interpolate_weights(x_mm, y_mm, field_centers):
    """
    Calculate bilinear interpolation weights for given positions (supports batched input).
    
    Given positions (x_mm, y_mm) and a grid of field centers,
    find the 4 neighboring grid points and their interpolation weights.
    
    Args:
        x_mm, y_mm: Target positions in mm (can be scalars or tensors)
        field_centers: Field grid coordinates (ny, nx, 2) - torch tensor
    
    Returns:
        indices: List of 4 tuples (iy, ix) for the neighboring grid points
        weights: List of 4 weights corresponding to each grid point (tensors if input is tensor)
    """
    ny, nx = field_centers.shape[:2]
    
    # Extract x and y coordinates from field centers
    field_x = field_centers[:, :, 0]  # shape (ny, nx)
    field_y = field_centers[:, :, 1]  # shape (ny, nx)
    
    # Get the bounds of the field grid
    x_min, x_max = field_x.min(), field_x.max()
    y_min, y_max = field_y.min(), field_y.max()
    
    # Convert position to grid indices (fractional)
    # Note: assumes regular grid spacing
    ix_frac = (x_mm - x_min) / (x_max - x_min) * (nx - 1)
    iy_frac = (y_mm - y_min) / (y_max - y_min) * (ny - 1)

    # import ipdb; ipdb.set_trace()
    
    # Clamp to valid range
    ix_frac = torch.clamp(ix_frac, 0, nx - 1)
    iy_frac = torch.clamp(iy_frac, 0, ny - 1)
    
    # Get integer indices for the 4 neighboring grid points
    ix0 = torch.floor(ix_frac).long()
    ix1 = torch.clamp(ix0 + 1, max=nx - 1)
    iy0 = torch.floor(iy_frac).long()
    iy1 = torch.clamp(iy0 + 1, max=ny - 1)

    # import ipdb; ipdb.set_trace()
    
    # Calculate interpolation weights
    wx = ix_frac - ix0.float()  # weight in x direction (0 to 1)
    wy = iy_frac - iy0.float()  # weight in y direction (0 to 1)
    
    # Bilinear interpolation weights for 4 corners
    # (iy0, ix0) = bottom-left, (iy0, ix1) = bottom-right
    # (iy1, ix0) = top-left,    (iy1, ix1) = top-right
    w00 = (1 - wx) * (1 - wy)  # bottom-left
    w10 = wx * (1 - wy)        # bottom-right
    w01 = (1 - wx) * wy        # top-left
    w11 = wx * wy              # top-right
    
    indices = [(iy0, ix0), (iy0, ix1), (iy1, ix0), (iy1, ix1)]
    weights = [w00, w10, w01, w11]
    
    return indices, weights



class SimpleProfiler:
    """
    Lightweight profiler for measuring execution time of code sections.
    轻量级性能分析器，用于测量代码段的执行时间。
    
    Supports both CPU timing (using time.perf_counter) and GPU timing
    (using CUDA events for accurate GPU kernel timing).
    
    支持 CPU 计时（使用 time.perf_counter）和 GPU 计时
    （使用 CUDA events 以获得准确的 GPU kernel 计时）。
    
    Usage:
        profiler = SimpleProfiler(enabled=True, use_cuda=True)
        profiler.start('tag_name')
        # ... code to profile ...
        profiler.end('tag_name')
        profiler.report()
    """
    def __init__(self, enabled=False, use_cuda=False):
        """
        Initialize profiler.
        
        Args:
            enabled: If False, all operations are no-ops (zero overhead)
            use_cuda: Use CUDA events for GPU timing (requires CUDA available)
        """
        self.enabled = enabled
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.cpu_starts = {}      # CPU start times for each tag
        self.times_ms = {}         # Accumulated times in milliseconds
        self.cuda_starts = {}      # CUDA start events
        self.cuda_ends = {}        # CUDA end events

    def start(self, tag):
        """
        Start timing for a section identified by tag.
        开始为由标签标识的代码段计时。
        
        Args:
            tag: String identifier for this timing section
        """
        if not self.enabled:
            return
            
        # Record CUDA event if using GPU timing
        # 如果使用 GPU 计时，记录 CUDA 事件
        if self.use_cuda:
            self.cuda_starts[tag] = torch.cuda.Event(enable_timing=True)
            self.cuda_ends[tag] = torch.cuda.Event(enable_timing=True)
            self.cuda_starts[tag].record()
            
        # Always record CPU time as fallback
        # 始终记录 CPU 时间作为后备
        self.cpu_starts[tag] = time.perf_counter()

    def end(self, tag):
        """
        End timing for a section and accumulate elapsed time.
        结束代码段计时并累积经过的时间。
        
        Args:
            tag: String identifier for this timing section (must match start())
        """
        if not self.enabled:
            return
            
        cpu_end = time.perf_counter()
        cpu_elapsed_ms = (cpu_end - self.cpu_starts.get(tag, cpu_end)) * 1000.0
        
        # Use GPU timing if available, otherwise use CPU timing
        # 如果可用，使用 GPU 计时，否则使用 CPU 计时
        if self.use_cuda:
            self.cuda_ends[tag].record()
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_elapsed_ms = self.cuda_starts[tag].elapsed_time(self.cuda_ends[tag])
            elapsed_ms = gpu_elapsed_ms
        else:
            elapsed_ms = cpu_elapsed_ms
            
        # Accumulate time (in case same tag is used multiple times)
        # 累积时间（如果同一标签多次使用）
        self.times_ms[tag] = self.times_ms.get(tag, 0.0) + elapsed_ms

    def report(self, header=None):
        """
        Print profiling report showing accumulated times for each tag.
        打印性能分析报告，显示每个标签的累积时间。
        
        Args:
            header: Optional header string to print before report
        """
        if not self.enabled:
            return
            
        if header:
            print(header)
            
        # Print times sorted by tag name
        # 按标签名称排序打印时间
        flat_total = 0.0
        for k in sorted(self.times_ms.keys()):
            v = self.times_ms[k]
            print(f"  {k:28s}: {v:10.3f} ms")
            
        # Calculate total excluding top-level 'simulate_total' to avoid double counting
        # 计算总计，排除顶级的 'simulate_total' 以避免重复计数
        for k, v in self.times_ms.items():
            if k == 'simulate_total':
                continue
            flat_total += v
        print(f"  {'TOTAL (flat)':28s}: {flat_total:10.3f} ms")


class PSFSimulatorGPU:
    """
    GPU-accelerated PSF-based image simulator.
    GPU 加速的基于 PSF 的图像模拟器。
    
    This class handles:
    - Loading PSF data from disk
    - Precomputing interpolation weights and reusable grids
    - Simulating image propagation through optical system
    
    Attributes:
        device: torch device ('cuda' or 'cpu')
        psf_geo: PSF data tensor [ny, nx, n_wavelengths, 64, 64]
        precomp: Precomputed interpolation data (weights, indices, chief rays)
        chunk_size: Number of input pixels to process at once
        micro_batch_size: Batch size within each chunk for GPU efficiency
    """
    
    def __init__(self, device='cuda', cache_dir=CACHE_DIR, chunk_size=10240, micro_batch_size=512,
                 profile=False, profile_cuda=False,
                 input_size=INPUT_SIZE, output_size=OUTPUT_SIZE,
                 psf_dir=None):
        """
        Initialize PSF simulator.
        初始化 PSF 模拟器。

        Args:
            device: Device to use ('cuda' or 'cpu')
            cache_dir: Directory for caching precomputed data
            chunk_size: Number of pixels to process per chunk
            micro_batch_size: Batch size for GPU processing
            profile: Enable CPU/GPU profiling
            profile_cuda: Use CUDA events for precise GPU timing
            input_size: Input image size (pixels)
            output_size: Output image size (pixels)
            psf_dir: Directory containing PSF data (default: PSF_DIR_DEFAULT)
        """
        # Use default PSF directory if not specified
        if psf_dir is None:
            psf_dir = PSF_DIR_DEFAULT
        self.psf_dir = psf_dir
        validate_output_size_for_cuda_window(output_size)

        # Load PSF data from disk to GPU
        # 从磁盘加载 PSF 数据到 GPU
        psf_geo, chief_rays, field_centers = self.load_psf_data(psf_dir=psf_dir, device=device)

        # Apply Gaussian blur to PSF for anti-aliasing (Nyquist theory)
        # Calculate output window size dynamically
        output_pixel_size_mm = OUTPUT_SPAN_MM / output_size
        psf_radius_pixels = int(np.ceil((PSF_SPAN_MM / 2) / output_pixel_size_mm))
        win_hw = 2 * psf_radius_pixels + 1  # output window size

        # k = PSF resolution / output window size
        psf_resolution = psf_geo.shape[-1]  # actual PSF resolution (e.g., 64 or 128)
        k = psf_resolution / win_hw

        # Only apply blur if k > 1 (downsampling) and sigma >= 0.5
        psf_blur_sigma = k / 2.0  # Nyquist: sigma = k/2
        if k > 1.0 and psf_blur_sigma >= 0.5:
            psf_geo = self._blur_psf(psf_geo, sigma=psf_blur_sigma)
            print(f"  Applied anti-aliasing blur to PSF (k={k:.3f}, sigma={psf_blur_sigma:.3f} pixels)")
        else:
            print(f"  Skipped anti-aliasing blur (k={k:.3f}, no downsampling needed)")

        # Store configuration
        # 存储配置
        self.device = device
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.micro_batch_size = micro_batch_size
        self.profiler = SimpleProfiler(enabled=profile, use_cuda=profile_cuda)
        self.sampling_dtype = torch.float32  # Use FP32 for accuracy
        self.precomp = None
        self.precomp_key = None
        self.input_size = input_size
        self.output_size = output_size
        self.psf_geo = psf_geo.to(dtype=self.sampling_dtype)

        # Enable cuDNN autotuner for better performance
        # 启用 cuDNN 自动调优以提高性能
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

        # Precompute reusable data (run in __init__ to exclude from inference profiling)
        # 预计算可复用数据（在 __init__ 中运行以排除在推理性能分析之外）
        if (field_centers is not None) and (chief_rays is not None):
            self.precompute(field_centers, chief_rays, input_size=input_size, output_size=output_size)



    def load_psf_data(self, psf_dir, device='cuda'):
        """
        Load PSF data extracted by psf_extraction.py and move to GPU.

        Args:
            psf_dir: Directory containing PSF data
            device: Device to load data to ('cuda' or 'cpu')

        Returns:
            psf_geo: torch tensor of shape (ny, nx, n_wavelengths, H, W) - PSF at each field point (FP32)
            chief_rays: torch tensor of shape (ny, nx, n_wavelengths, 3) - chief ray positions (FP32)
            field_centers: torch tensor of shape (ny, nx, 2) - field point coordinates (FP32)
        """
        psf_path = os.path.join(psf_dir, PSF_DATA_FILE)

        # Support both npz and npy formats
        if psf_path.endswith('.npz'):
            # Load from npz (all data in one file)
            data = np.load(psf_path)
            psf_geo = torch.from_numpy(data['psf']).to(device=device, dtype=torch.float32)
            chief_rays = torch.from_numpy(data['chief_rays']).to(device=device, dtype=torch.float32)
            field_centers = torch.from_numpy(data['field_centers']).to(device=device, dtype=torch.float32)
        else:
            # Load from separate npy files
            chief_path = os.path.join(psf_dir, CHIEF_RAY_FILE)
            field_path = os.path.join(psf_dir, FIELD_CENTERS_FILE)
            psf_geo = torch.from_numpy(np.load(psf_path)).to(device=device, dtype=torch.float32)
            chief_rays = torch.from_numpy(np.load(chief_path)).to(device=device, dtype=torch.float32)
            field_centers = torch.from_numpy(np.load(field_path)).to(device=device, dtype=torch.float32)

        print(f"Loaded PSF data to {device} (dtype: {psf_geo.dtype}):")
        print(f"  PSF shape: {psf_geo.shape}")
        print(f"  Chief rays shape: {chief_rays.shape}")
        print(f"  Field centers shape: {field_centers.shape}")

        # Verify FP32 precision
        assert psf_geo.dtype == torch.float32, f"PSF data must be FP32, got {psf_geo.dtype}"
        assert chief_rays.dtype == torch.float32, f"Chief rays must be FP32, got {chief_rays.dtype}"
        assert field_centers.dtype == torch.float32, f"Field centers must be FP32, got {field_centers.dtype}"

        return psf_geo, chief_rays, field_centers

    def _blur_psf(self, psf_geo, sigma=1.5):
        """
        Apply Gaussian blur to PSF for anti-aliasing.
        根据 Nyquist 理论对 PSF 进行高斯模糊以防止走样。

        PSF: 128x128, output window: 85x85
        Target effective resolution: 85/2 ≈ 43
        """
        ny, nx, n_wl, H, W = psf_geo.shape
        device = psf_geo.device

        # Create Gaussian kernel
        ksize = int(sigma * 6) | 1  # Ensure odd
        x = torch.arange(ksize, device=device, dtype=torch.float32) - ksize // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.view(1, 1, ksize, ksize)

        # Batch blur all PSFs
        psf_flat = psf_geo.reshape(-1, 1, H, W)
        pad = ksize // 2
        psf_padded = F.pad(psf_flat, (pad, pad, pad, pad), mode='reflect')
        psf_blurred = F.conv2d(psf_padded, kernel_2d)
        psf_blurred = psf_blurred.reshape(ny, nx, n_wl, H, W)

        # Preserve total energy
        orig_sum = psf_geo.sum(dim=(-2, -1), keepdim=True)
        blur_sum = psf_blurred.sum(dim=(-2, -1), keepdim=True)
        psf_blurred = psf_blurred * (orig_sum / blur_sum.clamp(min=1e-8))

        return psf_blurred

    @torch.no_grad()
    def get_or_create_precomputed(self, field_centers, chief_rays, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, device='cuda'):
        """
        Precompute and cache reusable data for all input pixels.
        预计算并缓存所有输入像素的可复用数据。

        For each input pixel, this computes:
        1. The 4 neighboring field points for PSF interpolation
        2. Bilinear interpolation weights for those 4 neighbors
        3. Interpolated chief ray positions for R, G, B wavelengths

        对于每个输入像素，计算：
        1. 用于 PSF 插值的 4 个相邻视场点
        2. 这 4 个邻居的双线性插值权重
        3. R、G、B 波长的插值主光线位置

        Args:
            field_centers: Field point positions [ny, nx, 2] in mm
            chief_rays: Chief ray data [ny, nx, n_wavelengths, 3] in mm
            input_size: Input image size (pixels)
            output_size: Output image size (pixels)
            device: torch device

        Returns:
            Dictionary containing:
            {
                'nbr_iy': LongTensor [4, N] - y indices of 4 neighbors for each pixel
                'nbr_ix': LongTensor [4, N] - x indices of 4 neighbors for each pixel
                'weights': FloatTensor [4, N] - interpolation weights for 4 neighbors
                'chief_xy_mm': FloatTensor [3, N, 2] - chief ray (x,y) for R,G,B wavelengths
            }
            where N = input_size * input_size
        """

        print("Precomputing interpolation weights and chief rays for all input pixels...")

        N = input_size * input_size

        # ====================================================================
        # Step 1: Build input pixel coordinate grid in physical space (mm)
        # 步骤 1：在物理空间（mm）中构建输入像素坐标网格
        # Pixel centers: pixel i is at (i + 0.5) / N * span - span/2
        # ====================================================================
        input_coords_x = (torch.arange(input_size, device=device, dtype=torch.float32) + 0.5) / input_size * INPUT_SPAN_MM - INPUT_SPAN_MM/2
        input_coords_y = (torch.arange(input_size, device=device, dtype=torch.float32) + 0.5) / input_size * INPUT_SPAN_MM - INPUT_SPAN_MM/2

        grid_y, grid_x = torch.meshgrid(input_coords_y, input_coords_x, indexing='ij')
        x_in_flat = grid_x.reshape(-1)  # [N]
        y_in_flat = grid_y.reshape(-1)  # [N]

        # ====================================================================
        # Step 2: Compute bilinear interpolation indices and weights
        # 步骤 2：计算双线性插值索引和权重
        # ====================================================================
        indices, weights = bilinear_interpolate_weights(x_in_flat, y_in_flat, field_centers)

        # Unpack indices into separate tensors
        iy_all = torch.stack([indices[k][0] for k in range(4)], dim=0).long()  # [4, N]
        ix_all = torch.stack([indices[k][1] for k in range(4)], dim=0).long()  # [4, N]
        w_all = torch.stack(weights, dim=0).to(dtype=torch.float32)  # [4, N]

        # ====================================================================
        # Step 3: Precompute interpolated chief ray positions for RGB
        # 步骤 3：为 RGB 预计算插值的主光线位置
        # ====================================================================
        # wavelength_indices: [2, 1, 0] maps to [R, G, B] channels
        # 波长索引：[2, 1, 0] 映射到 [R, G, B] 通道
        wavelength_indices = [2, 1, 0]
        chief_xy_list = []
        
        for wl_idx in wavelength_indices:
            # Gather chief rays at 4 corner field points for this wavelength
            # 为此波长收集 4 个角点视场点的主光线
            chief_corners = chief_rays[iy_all, ix_all, wl_idx]  # [4, N, 3]
            
            # Interpolate using bilinear weights: weighted sum across 4 corners
            # 使用双线性权重插值：对 4 个角点进行加权求和
            chief_interp = (chief_corners * w_all.unsqueeze(-1)).sum(dim=0)  # [N, 3]
            
            # Extract x, y coordinates (ignore z)
            # 提取 x, y 坐标（忽略 z）
            chief_xy = chief_interp[:, :2]  # [N, 2]
            chief_xy_list.append(chief_xy)
            
        chief_xy_mm = torch.stack(chief_xy_list, dim=0)  # [3, N, 2] for R,G,B

        # ====================================================================
        # Step 4: Package results into dictionary
        # 步骤 4：将结果打包到字典中
        # ====================================================================
        precomp = {
            'nbr_iy': iy_all,        # [4, N] y indices of neighbors
            'nbr_ix': ix_all,        # [4, N] x indices of neighbors
            'weights': w_all,        # [4, N] interpolation weights
            'chief_xy_mm': chief_xy_mm,  # [3, N, 2] chief ray positions for R,G,B
        }

        return precomp

    def _build_cache_key(self, input_size, output_size, field_centers):
        """
        Build a unique cache key for the current configuration.
        为当前配置构建唯一的缓存键。
        
        The key identifies a specific combination of:
        - Input/output sizes
        - Field grid resolution
        - Field grid extent (min/max coordinates)
        
        Args:
            input_size: Input image size
            output_size: Output image size
            field_centers: Field point positions [ny, nx, 2]
            
        Returns:
            String key for cache identification
        """
        ny, nx = field_centers.shape[:2]
        fx = field_centers[:, :, 0]
        fy = field_centers[:, :, 1]
        
        # Build key from grid parameters
        # 从网格参数构建键
        key = (
            f"ny{ny}_nx{nx}_in{input_size}_out{output_size}_"
            f"fx{float(fx.min()):.3f}-{float(fx.max()):.3f}_"
            f"fy{float(fy.min()):.3f}-{float(fy.max()):.3f}"
        )
        return key



    @torch.no_grad()
    def precompute(self, field_centers, chief_rays, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
        """
        Precompute all data that can be reused across multiple images.
        预计算所有可以跨多张图片复用的数据。

        This includes:
        - Bilinear interpolation weights and indices for all input pixels
        - Chief ray positions for all input pixels and wavelengths
        - Output pixel physical dimensions and PSF radius
        - Pre-converted weight tensors in sampling dtype

        Args:
            field_centers: Field point positions [ny, nx, 2]
            chief_rays: Chief ray data [ny, nx, n_wavelengths, 3]
            input_size: Input image size
            output_size: Output image size
        """
        # Build cache key for this configuration
        # 为此配置构建缓存键
        key = self._build_cache_key(input_size, output_size, field_centers)

        # Get or compute interpolation data
        # 获取或计算插值数据
        precomp = self.get_or_create_precomputed(
            field_centers, chief_rays,
            input_size=input_size,
            output_size=output_size,
            device=self.device
        )
        
        # Store precomputed data
        # 存储预计算数据
        self.precomp = precomp
        self.precomp_key = key
        self.input_size = input_size
        self.output_size = output_size
        
        # ====================================================================
        # Precompute coordinate grids
        # 预计算坐标网格
        # ====================================================================
        
        # Input coordinate grid (mm) - pixel centers
        # 输入坐标网格（mm）- 像素中心
        self.input_coords_x = (torch.arange(input_size, device=self.device, dtype=torch.float32) + 0.5) / input_size * INPUT_SPAN_MM - INPUT_SPAN_MM/2
        self.input_coords_y = (torch.arange(input_size, device=self.device, dtype=torch.float32) + 0.5) / input_size * INPUT_SPAN_MM - INPUT_SPAN_MM/2

        # Output coordinate grid (mm) - pixel centers, not used in current implementation but kept for reference
        # 输出坐标网格（mm）- 像素中心，当前实现中未使用，但保留供参考
        out_x = (torch.arange(output_size, device=self.device, dtype=torch.float32) + 0.5) / output_size * OUTPUT_SPAN_MM - OUTPUT_SPAN_MM/2
        out_y = (torch.arange(output_size, device=self.device, dtype=torch.float32) + 0.5) / output_size * OUTPUT_SPAN_MM - OUTPUT_SPAN_MM/2
        self.grid_y_full, self.grid_x_full = torch.meshgrid(out_y, out_x, indexing='ij')
        
        # ====================================================================
        # Precompute output pixel physical dimensions
        # 预计算输出像素物理尺寸
        # ====================================================================
        
        self.output_pixel_size_mm = OUTPUT_SPAN_MM / output_size
        self.output_pixel_area_mm2 = self.output_pixel_size_mm ** 2
        
        print(f"Output pixel size: {self.output_pixel_size_mm:.6f} mm")
        print(f"Output pixel area: {self.output_pixel_area_mm2:.6f} mm²")
        
        # ====================================================================
        # Precompute PSF physical dimensions
        # 预计算 PSF 物理尺寸
        # ====================================================================
        
        self.psf_radius_mm = PSF_SPAN_MM / 2  # 5mm
        self.psf_radius_pixels = int(np.ceil(self.psf_radius_mm / self.output_pixel_size_mm))
        
        print(f"PSF radius: {self.psf_radius_mm} mm = {self.psf_radius_pixels} pixels in output space")
        
        # ====================================================================
        # Precompute dtype-converted weights for faster per-image processing
        # 预计算类型转换后的权重以加快每张图片的处理
        # ====================================================================
        
        self.precomp['weights_sampling_dtype'] = precomp['weights'].to(dtype=self.sampling_dtype)
        
        print("Precomputation complete!")

    @torch.no_grad()
    def simulate_rgb_image_all_channels(self, input_img_rgb,
                                        input_size=None, output_size=None):
        """
        Simulate image propagation through optical system using PSF convolution.
        使用 PSF 卷积模拟图像通过光学系统的传播。
        
        Key principle: For each output pixel, sample PSF at its exact center position
        using bilinear interpolation. This eliminates grid artifacts.
        关键原理：对于每个输出像素，在其精确的中心位置用双线性插值采样 PSF，消除格状误差。
        
        Args:
            input_img_rgb: Input image [H, W, 3] in linear RGB space
            input_size: Input image size (default: INPUT_SIZE)
            output_size: Output image size (default: OUTPUT_SIZE)
            
        Returns:
            output_img_rgb: Output image [output_size, output_size, 3]
        """
        device = input_img_rgb.device
        if input_size is None:
            input_size = INPUT_SIZE
        if output_size is None:
            output_size = OUTPUT_SIZE

        # Use precomputed output pixel dimensions
        # 使用预计算的输出像素尺寸
        output_pixel_size_mm = self.output_pixel_size_mm
        output_pixel_area_mm2 = self.output_pixel_area_mm2

        # Initialize output buffer
        # 初始化输出缓冲区
        output_img_rgb = torch.zeros((output_size, output_size, 3), 
                                     device=device, dtype=torch.float32)

        # Flatten input image for processing
        # 展平输入图像以便处理：[H, W, 3] -> [H*W, 3]
        num_pixels = input_size * input_size
        flat_idx_all = torch.arange(num_pixels, device=device, dtype=torch.long)
        intensities_rgb = input_img_rgb.reshape(-1, 3)

        # Get precomputed interpolation data
        # 获取预计算的插值数据
        precomp = self.precomp

        # Use precomputed PSF physical parameters
        # 使用预计算的 PSF 物理参数
        psf_radius_mm = self.psf_radius_mm  # 5mm
        psf_radius_pixels = self.psf_radius_pixels

        wavelength_indices = [2, 1, 0]  # R, G, B channels
        start_time = time.time()
        self.profiler.start('simulate_total')

        # ========================================================================
        # Process each color channel independently
        # 独立处理每个颜色通道
        # ========================================================================
        for c, wl_idx in enumerate(wavelength_indices):
            print(f"\n=== Channel {c} (wavelength {wl_idx}) ===")
            
            # Extract intensity for current channel
            # 提取当前通道的强度
            intensities = intensities_rgb[:, c]
            intensities_all_d = intensities.to(dtype=self.sampling_dtype)
            
            # Use precomputed dtype-converted weights
            # 使用预计算的类型转换后的权重
            weights_all_d = precomp['weights_sampling_dtype']

            # Process in chunks to manage memory
            # 分块处理以管理内存
            num_chunks = (num_pixels + self.chunk_size - 1) // self.chunk_size
            print(f"Processing {num_pixels} pixels in {num_chunks} chunks...")

            for chunk_idx in range(num_chunks):
                cs = chunk_idx * self.chunk_size
                ce = min(cs + self.chunk_size, num_pixels)
                if chunk_idx % 50 == 0:
                    print(f"  Chunk {chunk_idx+1}/{num_chunks} (pixels {cs}-{ce})")

                flat_chunk = flat_idx_all[cs:ce]
                
                # Extract chief ray positions for this chunk
                # 提取该块的主光线位置
                self.profiler.start('chief_extract')
                chief_xy = precomp['chief_xy_mm'][c, flat_chunk]  # [B, 2]
                chief_x = chief_xy[:, 0]  # [B]
                chief_y = chief_xy[:, 1]  # [B]
                self.profiler.end('chief_extract')

                # Process chunk in micro-batches for better GPU utilization
                # 将块进一步分为微批次以提高 GPU 利用率
                B = flat_chunk.shape[0]
                for mb_start in range(0, B, self.micro_batch_size):
                    mb_end = min(mb_start + self.micro_batch_size, B)
                    flat_mb = flat_chunk[mb_start:mb_end]
                    n_mb = flat_mb.shape[0]
                    
                    chief_x_mb = chief_x[mb_start:mb_end]
                    chief_y_mb = chief_y[mb_start:mb_end]

                    # ============================================================
                    # Step 1: Interpolate PSF from 4 corner samples (64×64 each)
                    # 步骤 1：从 4 个角点样本插值 PSF（每个 64×64）
                    # ============================================================
                    self.profiler.start('psf_gather_agg')
                    iy4 = precomp['nbr_iy'][:, flat_mb]  # [4, n_mb]
                    ix4 = precomp['nbr_ix'][:, flat_mb]  # [4, n_mb]
                    w4 = weights_all_d[:, flat_mb].unsqueeze(-1).unsqueeze(-1)  # [4, n_mb, 1, 1]
                    
                    psf_corners = self.psf_geo[iy4, ix4, wl_idx]  # [4, n_mb, 64, 64]
                    psf_agg = (psf_corners * w4).sum(dim=0)  # [n_mb, 64, 64]
                    self.profiler.end('psf_gather_agg')

                    # ============================================================
                    # Step 2: Compute bounding box for affected output region
                    # 步骤 2：计算受影响的输出区域的边界框
                    # ============================================================
                    self.profiler.start('compute_bbox')
                    # Convert chief ray position to output pixel coordinates
                    # 将主光线位置转换为输出像素坐标
                    chief_x_pix = (chief_x_mb + OUTPUT_SPAN_MM/2) / output_pixel_size_mm
                    chief_y_pix = (chief_y_mb + OUTPUT_SPAN_MM/2) / output_pixel_size_mm
                    
                    # Determine per-pixel window start indices (unclamped) for fixed 10mm window
                    # 为固定 10mm 窗口确定每个像素的窗口起始索引（不裁剪）
                    x0 = (chief_x_pix - psf_radius_pixels).floor().long()  # [n_mb]
                    y0 = (chief_y_pix - psf_radius_pixels).floor().long()  # [n_mb]
                    self.profiler.end('compute_bbox')
                    
                    # ============================================================
                    # Step 3: Build batched per-pixel output grids (fixed H×W window)
                    # 步骤 3：构建批量的每像素输出网格（固定尺寸窗口）
                    # ============================================================
                    self.profiler.start('create_output_grid')
                    win_hw = int(2 * psf_radius_pixels + 1)

                    # Create grid at pixel centers
                    # 在像素中心创建网格
                    rx = torch.arange(0, win_hw, device=device, dtype=torch.float32)  # [W]
                    ry = torch.arange(0, win_hw, device=device, dtype=torch.float32)  # [H]
                    ry_grid, rx_grid = torch.meshgrid(ry, rx, indexing='ij')  # [H, W]
                    x_idx_batch = x0.view(-1, 1, 1).to(dtype=torch.float32) + rx_grid.view(1, win_hw, win_hw)
                    y_idx_batch = y0.view(-1, 1, 1).to(dtype=torch.float32) + ry_grid.view(1, win_hw, win_hw)

                    # Convert to physical mm positions (pixel centers)
                    # 转换为物理 mm 位置（像素中心）
                    out_grid_x_mm = (x_idx_batch + 0.5) * output_pixel_size_mm - OUTPUT_SPAN_MM/2
                    out_grid_y_mm = (y_idx_batch + 0.5) * output_pixel_size_mm - OUTPUT_SPAN_MM/2
                    self.profiler.end('create_output_grid')

                    # ============================================================
                    # Step 4: Sample PSF at each batched window using bilinear interpolation
                    # 步骤 4：在批量窗口上使用双线性插值采样 PSF
                    # ============================================================
                    self.profiler.start('sample_psf_at_pixels')
                    offset_x = out_grid_x_mm - chief_x_mb.view(-1, 1, 1)
                    offset_y = out_grid_y_mm - chief_y_mb.view(-1, 1, 1)
                    psf_u = (offset_x / psf_radius_mm).to(dtype=self.sampling_dtype)
                    psf_v = (offset_y / psf_radius_mm).to(dtype=self.sampling_dtype)
                    grid_mb = torch.stack([psf_u, psf_v], dim=-1)  # [n_mb, H, W, 2]

                    psf_sampled = F.grid_sample(
                        psf_agg.unsqueeze(1),        # [n_mb, 1, Hpsf, Wpsf]
                        grid_mb,                      # [n_mb, H, W, 2]
                        mode='bicubic',              # bicubic for better anti-aliasing
                        padding_mode='zeros',
                        align_corners=False
                    ).squeeze(1)  # [n_mb, H, W]
                    self.profiler.end('sample_psf_at_pixels')

                    # ============================================================
                    # Step 5: Accumulate contributions (single-image breakdown)
                    # 步骤 5：累加贡献（单图细分）
                    # ============================================================
                    self.profiler.start('acc_single:contrib_compute')
                    intens_mb = intensities_all_d[flat_mb].view(-1, 1, 1)  # [n_mb, 1, 1]
                    contrib_batch = (intens_mb * psf_sampled * output_pixel_area_mm2).to(dtype=torch.float32)
                    self.profiler.end('acc_single:contrib_compute')

                    self.profiler.start('acc_single:build_indices')
                    x_idx_long = x_idx_batch.long()
                    y_idx_long = y_idx_batch.long()
                    valid = (x_idx_long >= 0) & (x_idx_long < output_size) & \
                            (y_idx_long >= 0) & (y_idx_long < output_size)
                    lin_idx = (y_idx_long * output_size + x_idx_long).reshape(-1)  # [n_mb*H*W]
                    self.profiler.end('acc_single:build_indices')

                    self.profiler.start('acc_single:mask_flatten')
                    # Use integer selection instead of boolean advanced indexing to reduce overhead
                    valid_flat = valid.reshape(-1)
                    sel = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)  # [K]
                    lin_idx_flat = lin_idx  # already flattened above
                    contrib_flat = contrib_batch.reshape(-1)
                    lin_idx_valid = lin_idx_flat.index_select(0, sel)
                    contrib_valid = contrib_flat.index_select(0, sel)
                    self.profiler.end('acc_single:mask_flatten')

                    self.profiler.start('acc_single:index_add')
                    out_flat = output_img_rgb[:, :, c].reshape(-1)
                    out_flat.index_add_(0, lin_idx_valid, contrib_valid)
                    self.profiler.end('acc_single:index_add')

                    del psf_corners, psf_agg, psf_sampled

            # ====================================================================
            # Normalize channel to preserve peak intensity
            # 归一化通道以保持峰值强度
            # ====================================================================
            self.profiler.start('normalize')
            input_peak = input_img_rgb[:, :, c].max()
            output_peak = output_img_rgb[:, :, c].max()
            if output_peak > 0:
                output_img_rgb[:, :, c] = output_img_rgb[:, :, c] * (input_peak / output_peak)
                print(f"  Channel {c}: normalized from peak {output_peak:.6f} to {input_peak:.6f}")
            self.profiler.end('normalize')

        # Finalize
        self.profiler.end('simulate_total')
        end_time = time.time()
        print(f"\nAll channels processed!  Total time: {end_time - start_time:.2f}s")
        self.profiler.report("Profiling (ms):")
        return output_img_rgb

    @torch.no_grad()
    def simulate_rgb_images_batch(self, input_imgs_rgb,
                                  input_size=None, output_size=None):
        """
        Batched simulation for multiple images in parallel.
        并行处理多张图片的批量模拟。
        
        Args:
            input_imgs_rgb: [B, H, W, 3] linear RGB
            input_size: optional override of input size
            output_size: optional override of output size
        Returns:
            output_imgs_rgb: [B, output_size, output_size, 3]
        """
        device = input_imgs_rgb.device
        if input_size is None:
            input_size = INPUT_SIZE
        if output_size is None:
            output_size = OUTPUT_SIZE

        B_img = input_imgs_rgb.shape[0]

        # Precomputed params
        output_pixel_size_mm = self.output_pixel_size_mm
        output_pixel_area_mm2 = self.output_pixel_area_mm2
        psf_radius_mm = self.psf_radius_mm
        psf_radius_pixels = self.psf_radius_pixels

        # Output buffer [B, H, W, 3]
        output_imgs_rgb = torch.zeros((B_img, output_size, output_size, 3),
                                      device=device, dtype=torch.float32)

        # Flatten input images
        num_pixels = input_size * input_size
        flat_idx_all = torch.arange(num_pixels, device=device, dtype=torch.long)
        intensities_rgb = input_imgs_rgb.reshape(B_img, -1, 3)

        precomp = self.precomp
        wavelength_indices = [2, 1, 0]  # R, G, B

        self.profiler.start('simulate_total')
        start_time = time.time()

        for c, wl_idx in enumerate(wavelength_indices):
            print(f"\n=== Channel {c} (wavelength {wl_idx}) [batch={B_img}] ===")

            intensities_all_d_B = intensities_rgb[:, :, c].to(dtype=self.sampling_dtype)  # [B, N]
            weights_all_d = precomp['weights_sampling_dtype']

            num_chunks = (num_pixels + self.chunk_size - 1) // self.chunk_size
            print(f"Processing {num_pixels} pixels in {num_chunks} chunks (batched)...")

            for chunk_idx in range(num_chunks):
                cs = chunk_idx * self.chunk_size
                ce = min(cs + self.chunk_size, num_pixels)
                if chunk_idx % 50 == 0:
                    print(f"  Chunk {chunk_idx+1}/{num_chunks} (pixels {cs}-{ce})")

                flat_chunk = flat_idx_all[cs:ce]

                # Chief ray positions for this chunk (independent of image)
                self.profiler.start('chief_extract')
                chief_xy = precomp['chief_xy_mm'][c, flat_chunk]  # [B_chunk, 2] where B_chunk=len(flat_chunk)
                chief_x = chief_xy[:, 0]
                chief_y = chief_xy[:, 1]
                self.profiler.end('chief_extract')

                Bp = flat_chunk.shape[0]
                for mb_start in range(0, Bp, self.micro_batch_size):
                    mb_end = min(mb_start + self.micro_batch_size, Bp)
                    flat_mb = flat_chunk[mb_start:mb_end]
                    n_mb = flat_mb.shape[0]

                    chief_x_mb = chief_x[mb_start:mb_end]
                    chief_y_mb = chief_y[mb_start:mb_end]

                    # Gather PSFs and bilinear-aggregate (independent of image)
                    self.profiler.start('psf_gather_agg')
                    iy4 = precomp['nbr_iy'][:, flat_mb]  # [4, n_mb]
                    ix4 = precomp['nbr_ix'][:, flat_mb]  # [4, n_mb]
                    w4 = weights_all_d[:, flat_mb].unsqueeze(-1).unsqueeze(-1)  # [4, n_mb, 1, 1]
                    psf_corners = self.psf_geo[iy4, ix4, wl_idx]  # [4, n_mb, 64, 64]
                    psf_agg = (psf_corners * w4).sum(dim=0)  # [n_mb, 64, 64]
                    self.profiler.end('psf_gather_agg')

                    # Window start index (unclamped)
                    self.profiler.start('compute_bbox')
                    chief_x_pix = (chief_x_mb + OUTPUT_SPAN_MM/2) / output_pixel_size_mm
                    chief_y_pix = (chief_y_mb + OUTPUT_SPAN_MM/2) / output_pixel_size_mm
                    x0 = (chief_x_pix - psf_radius_pixels).floor().long()  # [n_mb]
                    y0 = (chief_y_pix - psf_radius_pixels).floor().long()  # [n_mb]
                    self.profiler.end('compute_bbox')

                    # Batched per-pixel output grids (fixed H×W)
                    self.profiler.start('create_output_grid')
                    win_hw = int(2 * psf_radius_pixels + 1)
                    rx = torch.arange(0, win_hw, device=device, dtype=torch.float32)
                    ry = torch.arange(0, win_hw, device=device, dtype=torch.float32)
                    ry_grid, rx_grid = torch.meshgrid(ry, rx, indexing='ij')
                    x_idx_batch = x0.view(-1, 1, 1).to(dtype=torch.float32) + rx_grid.view(1, win_hw, win_hw)
                    y_idx_batch = y0.view(-1, 1, 1).to(dtype=torch.float32) + ry_grid.view(1, win_hw, win_hw)
                    out_grid_x_mm = (x_idx_batch + 0.5) * output_pixel_size_mm - OUTPUT_SPAN_MM/2  # pixel centers
                    out_grid_y_mm = (y_idx_batch + 0.5) * output_pixel_size_mm - OUTPUT_SPAN_MM/2  # pixel centers
                    self.profiler.end('create_output_grid')

                    # Sample PSFs at batched windows
                    self.profiler.start('sample_psf_at_pixels')
                    offset_x = out_grid_x_mm - chief_x_mb.view(-1, 1, 1)
                    offset_y = out_grid_y_mm - chief_y_mb.view(-1, 1, 1)
                    psf_u = (offset_x / psf_radius_mm).to(dtype=self.sampling_dtype)
                    psf_v = (offset_y / psf_radius_mm).to(dtype=self.sampling_dtype)
                    grid_mb = torch.stack([psf_u, psf_v], dim=-1)  # [n_mb, H, W, 2]
                    psf_sampled = F.grid_sample(
                        psf_agg.unsqueeze(1),
                        grid_mb,
                        mode='bicubic',              # 16 neighbors instead of 4
                        padding_mode='zeros',
                        align_corners=False
                    ).squeeze(1)  # [n_mb, H, W]
                    self.profiler.end('sample_psf_at_pixels')

                    # Accumulate to each image in batch via single index_add_ (with breakdown)
                    self.profiler.start('acc:contrib_compute')
                    intens_mb_B = intensities_all_d_B[:, flat_mb]  # [B, n_mb]
                    contrib_batch_B = (intens_mb_B.unsqueeze(-1).unsqueeze(-1) * psf_sampled.unsqueeze(0) * output_pixel_area_mm2)
                    contrib_batch_B = contrib_batch_B.to(dtype=torch.float32)  # [B, n_mb, H, W]
                    self.profiler.end('acc:contrib_compute')

                    self.profiler.start('acc:build_indices')
                    x_idx_long = x_idx_batch.long()  # [n_mb, H, W]
                    y_idx_long = y_idx_batch.long()  # [n_mb, H, W]
                    valid = (x_idx_long >= 0) & (x_idx_long < output_size) & (y_idx_long >= 0) & (y_idx_long < output_size)
                    lin_idx = (y_idx_long * output_size + x_idx_long)  # [n_mb, H, W]
                    HW = output_size * output_size
                    lin_idx_exp = lin_idx.unsqueeze(0).expand(B_img, -1, -1, -1)
                    img_offsets = torch.arange(B_img, device=device).view(-1, 1, 1, 1) * HW
                    lin_idx_total = lin_idx_exp + img_offsets  # [B, n_mb, H, W]
                    self.profiler.end('acc:build_indices')

                    self.profiler.start('acc:mask_flatten')
                    # Avoid expanding boolean mask to [B, ...]; use integer gather instead
                    valid_flat = valid.reshape(-1)  # [n_mb*H*W]
                    sel = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)  # [K]
                    lin_idx_total_flat = lin_idx_total.reshape(B_img, -1)  # [B, n_mb*H*W]
                    contrib_batch_B_flat = contrib_batch_B.reshape(B_img, -1)  # [B, n_mb*H*W]
                    lin_idx_valid_B = lin_idx_total_flat.index_select(1, sel)  # [B, K]
                    contrib_valid_B = contrib_batch_B_flat.index_select(1, sel)  # [B, K]
                    lin_idx_valid = lin_idx_valid_B.reshape(-1)
                    contrib_valid = contrib_valid_B.reshape(-1)
                    self.profiler.end('acc:mask_flatten')

                    self.profiler.start('acc:index_add')
                    out_flat = output_imgs_rgb[:, :, :, c].reshape(-1)
                    out_flat.index_add_(0, lin_idx_valid, contrib_valid)
                    self.profiler.end('acc:index_add')

                    del psf_corners, psf_agg, psf_sampled

            # Per-image normalization per channel to preserve peak
            self.profiler.start('normalize')
            input_peak_B = input_imgs_rgb[:, :, :, c].amax(dim=(1, 2))  # [B]
            output_peak_B = output_imgs_rgb[:, :, :, c].amax(dim=(1, 2))  # [B]
            scale = torch.where(output_peak_B > 0, input_peak_B / output_peak_B, torch.zeros_like(output_peak_B))
            output_imgs_rgb[:, :, :, c] = output_imgs_rgb[:, :, :, c] * scale.view(-1, 1, 1)
            self.profiler.end('normalize')

        self.profiler.end('simulate_total')
        end_time = time.time()
        print(f"\nAll channels processed [batch={B_img}]!  Total time: {end_time - start_time:.2f}s")
        self.profiler.report("Profiling (ms):")
        return output_imgs_rgb

    @torch.no_grad()
    def simulate_rgb_images_batch_cuda(self, input_imgs_rgb,
                                       input_size=None, output_size=None,
                                       verbose=True):
        """
        CUDA-accelerated batched simulation. Requires psf_cuda_ext_v2 installed.
        仅 forward，使用一个 CUDA kernel 完成像素级插值、重采样与原位原子加和。

        Args:
            input_imgs_rgb: [B, H, W, 3] linear RGB (CUDA)
            input_size: optional override
            output_size: optional override
        Returns:
            output_imgs_rgb: [B, output_size, output_size, 3]
        """
        if psf_cuda is None:
            raise RuntimeError("psf_cuda_ext_v2 is not available. Build and install the CUDA extension first.")

        device = input_imgs_rgb.device
        if input_size is None:
            input_size = INPUT_SIZE
        if output_size is None:
            output_size = OUTPUT_SIZE

        if device.type != 'cuda':
            raise RuntimeError("simulate_rgb_images_batch_cuda requires CUDA tensors")

        B_img = input_imgs_rgb.shape[0]

        # Precomputed params
        output_pixel_size_mm = self.output_pixel_size_mm
        psf_radius_mm = self.psf_radius_mm
        psf_radius_pixels = self.psf_radius_pixels

        # Output buffer [B, H, W, 3]
        output_imgs_rgb = torch.zeros((B_img, output_size, output_size, 3),
                                      device=device, dtype=torch.float32)

        # Flatten inputs
        num_pixels = input_size * input_size
        intensities_rgb = input_imgs_rgb.reshape(B_img, -1, 3)

        precomp = self.precomp
        wavelength_indices = [2, 1, 0]  # R, G, B

        self.profiler.start('simulate_total')
        start_time = time.time()

        for c, wl_idx in enumerate(wavelength_indices):
            intens_B = intensities_rgb[:, :, c].to(dtype=torch.float32).contiguous()  # [B, N]
            # Precomp per-pixel data
            nbr_iy = precomp['nbr_iy']              # [4, N] long
            nbr_ix = precomp['nbr_ix']              # [4, N] long
            weights = precomp['weights_sampling_dtype']  # [4, N] float32
            chief_xy_c = precomp['chief_xy_mm'][c]  # [N, 2] float32

            # Call CUDA kernel: returns [B, H, W] for channel c
            out_c = psf_cuda.psf_accumulate_forward(
                intens_B,
                nbr_iy,
                nbr_ix,
                weights,
                chief_xy_c,
                self.psf_geo,
                int(wl_idx),
                int(output_size),
                float(output_pixel_size_mm),
                float(psf_radius_mm),
                int(psf_radius_pixels),
            )

            # Normalize per image to preserve peak
            input_peak_B = input_imgs_rgb[:, :, :, c].amax(dim=(1, 2))  # [B]
            output_peak_B = out_c.amax(dim=(1, 2))                      # [B]
            scale = torch.where(output_peak_B > 0, input_peak_B / output_peak_B, torch.zeros_like(output_peak_B))
            out_c = out_c * scale.view(-1, 1, 1)

            output_imgs_rgb[:, :, :, c] = out_c

        self.profiler.end('simulate_total')
        end_time = time.time()
        if verbose:
            print(f"\nAll channels processed [batch={B_img}] via CUDA!  Total time: {end_time - start_time:.2f}s")
            self.profiler.report("Profiling (ms):")
        return output_imgs_rgb


def _torch_psf_accumulate_single_channel(intensities_B,  # [B, N], float32, requires_grad ok
                                         nbr_iy, nbr_ix,  # [4, N], long or int
                                         weights,         # [4, N], float32
                                         chief_xy_c,      # [N, 2], float32
                                         psf_geo,         # [ny, nx, 3, Hpsf, Wpsf], float32
                                         wl_idx: int,
                                         output_size: int,
                                         output_pixel_size_mm: float,
                                         psf_radius_mm: float,
                                         psf_radius_pixels: int,
                                         chunk_size: int = 8192):
    """
    Reference PyTorch implementation (differentiable) of one-channel accumulation.
    Returns [B, Hout, Wout].
    """
    device = intensities_B.device
    B, N = intensities_B.shape
    out = torch.zeros((B, output_size, output_size), device=device, dtype=torch.float32)
    flat_idx_all = torch.arange(N, device=device, dtype=torch.long)

    win_hw = int(2 * psf_radius_pixels + 1)
    rx = torch.arange(0, win_hw, device=device, dtype=torch.float32)
    ry = torch.arange(0, win_hw, device=device, dtype=torch.float32)
    ry_grid, rx_grid = torch.meshgrid(ry, rx, indexing='ij')  # [Hwin, Wwin]

    num_chunks = (N + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        cs = chunk_idx * chunk_size
        ce = min(cs + chunk_size, N)
        flat_mb = flat_idx_all[cs:ce]
        n_mb = flat_mb.shape[0]

        chief_xy = chief_xy_c[flat_mb]  # [n_mb, 2]
        chief_x_mb = chief_xy[:, 0]
        chief_y_mb = chief_xy[:, 1]

        # Gather and bilinear-aggregate PSFs
        iy4 = nbr_iy[:, flat_mb].long()  # [4, n_mb]
        ix4 = nbr_ix[:, flat_mb].long()  # [4, n_mb]
        w4 = weights[:, flat_mb].unsqueeze(-1).unsqueeze(-1)  # [4, n_mb, 1, 1]
        psf_corners = psf_geo[iy4, ix4, wl_idx]  # [4, n_mb, 64, 64]
        psf_agg = (psf_corners * w4).sum(dim=0)  # [n_mb, Hpsf, Wpsf]

        # Compute bbox/start
        chief_x_pix = (chief_x_mb + OUTPUT_SPAN_MM/2) / output_pixel_size_mm
        chief_y_pix = (chief_y_mb + OUTPUT_SPAN_MM/2) / output_pixel_size_mm
        x0 = (chief_x_pix - psf_radius_pixels).floor().long()  # [n_mb]
        y0 = (chief_y_pix - psf_radius_pixels).floor().long()  # [n_mb]

        # Build output grid for this micro-batch
        x_idx_batch = x0.view(-1, 1, 1).to(dtype=torch.float32) + rx_grid.view(1, win_hw, win_hw)  # [n_mb,Hwin,Wwin]
        y_idx_batch = y0.view(-1, 1, 1).to(dtype=torch.float32) + ry_grid.view(1, win_hw, win_hw)
        out_grid_x_mm = (x_idx_batch + 0.5) * output_pixel_size_mm - OUTPUT_SPAN_MM/2  # pixel centers
        out_grid_y_mm = (y_idx_batch + 0.5) * output_pixel_size_mm - OUTPUT_SPAN_MM/2  # pixel centers

        # Sample PSF
        offset_x = out_grid_x_mm - chief_x_mb.view(-1, 1, 1)
        offset_y = out_grid_y_mm - chief_y_mb.view(-1, 1, 1)
        psf_u = (offset_x / psf_radius_mm)
        psf_v = (offset_y / psf_radius_mm)
        grid_mb = torch.stack([psf_u, psf_v], dim=-1)  # [n_mb, Hwin, Wwin, 2]
        psf_sampled = F.grid_sample(psf_agg.unsqueeze(1), grid_mb,
                                    mode='bicubic', padding_mode='zeros',
                                    align_corners=False).squeeze(1)  # [n_mb, Hwin, Wwin]

        # Contributions
        contrib_batch_B = (intensities_B[:, flat_mb].unsqueeze(-1).unsqueeze(-1) *
                           psf_sampled.unsqueeze(0) * (output_pixel_size_mm * output_pixel_size_mm)).to(dtype=torch.float32)  # [B,n_mb,Hwin,Wwin]

        # Indices and mask
        x_idx_long = x_idx_batch.long()
        y_idx_long = y_idx_batch.long()
        valid = (x_idx_long >= 0) & (x_idx_long < output_size) & (y_idx_long >= 0) & (y_idx_long < output_size)
        lin_idx = (y_idx_long * output_size + x_idx_long)  # [n_mb,Hwin,Wwin]
        valid_flat = valid.reshape(-1)
        sel = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)  # [K]
        lin_idx_flat = lin_idx.reshape(-1)
        lin_idx_valid = lin_idx_flat.index_select(0, sel)  # [K]
        contrib_flat_B = contrib_batch_B.reshape(B, -1)    # [B, n_mb*H*W]
        contrib_valid_B = contrib_flat_B.index_select(1, sel)  # [B, K]

        # Scatter-add into out per image
        HW = output_size * output_size
        for b in range(B):
            out_flat = out[b].reshape(-1)
            out_flat.scatter_add_(0, lin_idx_valid, contrib_valid_B[b])

    return out


def test_cuda_vs_torch_correctness(simulator: PSFSimulatorGPU,
                                   batch_size: int = 2,
                                   channel_c: int = 0,   # 0->R, 1->G, 2->B
                                   wl_idx: int = 2,      # match R
                                   seed: int = 0):
    """
    Compare CUDA vs PyTorch reference:
      (1) Forward absolute/relative error
      (2) Backward gradient error w.r.t. intensities_B
    """
    assert psf_cuda is not None, "psf_cuda_ext_v2 not available"
    torch.manual_seed(seed)
    device = simulator.device
    B = batch_size
    in_size = simulator.input_size
    out_size = simulator.output_size

    # Random linear RGB input
    imgs = torch.rand(B, in_size, in_size, 3, device=device, dtype=torch.float32)

    # Forward: CUDA full (per-channel normalization inside)
    out_cuda_full = simulator.simulate_rgb_images_batch_cuda(imgs, verbose=False)
    # Forward: Torch reference full
    out_torch_full = simulator.simulate_rgb_images_batch(imgs)

    # Forward error (all channels)
    diff = (out_cuda_full - out_torch_full).abs()
    rel = diff.norm() / (out_torch_full.norm())
    print(f"[Forward] abs(mean)={diff.mean().item():.6e}, abs(max)={diff.max().item():.6e}, rel(L2)={rel.item():.6e}")

    # Backward: single channel accumulate (no per-channel normalization)
    pre = simulator.precomp
    weights = pre['weights_sampling_dtype']                     # [4, N]
    chief_xy_c = pre['chief_xy_mm'][channel_c]                  # [N, 2]
    nbr_iy = pre['nbr_iy']                                      # [4, N]
    nbr_ix = pre['nbr_ix']                                      # [4, N]
    psf_geo = simulator.psf_geo                                  # [ny,nx,3,Hpsf,Wpsf]
    out_pix_mm = simulator.output_pixel_size_mm
    psf_radius_mm = simulator.psf_radius_mm
    psf_radius_px = simulator.psf_radius_pixels

    intens_B_base = imgs.reshape(B, -1, 3)[:, :, channel_c].contiguous()
    # CUDA autograd
    intens_cuda = intens_B_base.clone().detach().requires_grad_(True)
    out_c_cuda = psf_cuda.psf_accumulate(
        intens_cuda,
        nbr_iy.to(dtype=torch.int32),
        nbr_ix.to(dtype=torch.int32),
        weights,
        chief_xy_c,
        psf_geo,
        int(wl_idx),
        int(out_size),
        float(out_pix_mm),
        float(psf_radius_mm),
        int(psf_radius_px),
    )
    loss_cuda = out_c_cuda.sum()
    loss_cuda.backward()
    grad_cuda = intens_cuda.grad

    # Torch reference autograd
    intens_torch = intens_B_base.clone().detach().requires_grad_(True)
    out_c_torch = _torch_psf_accumulate_single_channel(
        intens_torch, nbr_iy, nbr_ix, weights, chief_xy_c, psf_geo, wl_idx,
        out_size, out_pix_mm, psf_radius_mm, psf_radius_px
    )
    loss_torch = out_c_torch.sum()
    loss_torch.backward()
    grad_torch = intens_torch.grad

    # Backward error
    gdiff = (grad_cuda - grad_torch).abs()
    grel = gdiff.norm() / (grad_torch.norm())
    print(f"[Backward dI] abs(mean)={gdiff.mean().item():.6e}, abs(max)={gdiff.max().item():.6e}, rel(L2)={grel.item():.6e}")

    return {
        'forward_abs_mean': diff.mean().item(),
        'forward_abs_max': diff.max().item(),
        'forward_rel_l2': rel.item(),
        'backward_abs_mean': gdiff.mean().item(),
        'backward_abs_max': gdiff.max().item(),
        'backward_rel_l2': grel.item(),
    }


def process_image_file(input_path, output_path, simulator, 
                       device='cuda'):
    """
    Process a single image file through PSF-based optical simulation.
    通过基于 PSF 的光学模拟处理单个图像文件。
    
    Pipeline:
    1. Load and resize image
    2. Convert sRGB → linear RGB
    3. Simulate optical propagation (PSF convolution)
    4. Convert linear RGB → sRGB
    5. Save result
    
    Args:
        input_path: Path to input image file
        output_path: Path to save output image
        simulator: PSFSimulatorGPU instance with precomputed data
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"\nProcessing image: {input_path}")
    
    t0 = time.perf_counter()
    
    # ========================================================================
    # Step 1: Load and prepare image
    # 步骤 1：加载和准备图像
    # ========================================================================
    img = Image.open(input_path)
    
    # Resize to expected input size if needed
    # 如果需要，调整到预期的输入大小
    if img.size != (INPUT_SIZE, INPUT_SIZE):
        print(f"  Resizing from {img.size} to ({INPUT_SIZE}, {INPUT_SIZE})")
        img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    
    # Convert to numpy array in [0, 1] range (still in sRGB)
    # 转换为 [0, 1] 范围的 numpy 数组（仍在 sRGB 空间）
    img_array = np.array(img, dtype=np.float32) / 255.0
    t1 = time.perf_counter()
    
    # Convert to torch tensor and move to GPU
    # 转换为 torch 张量并移动到 GPU
    img_tensor = torch.from_numpy(img_array).to(device=device, dtype=torch.float32)
    
    # Handle grayscale images by converting to RGB
    # 将灰度图像转换为 RGB
    if len(img_tensor.shape) == 2:
        img_tensor = torch.stack([img_tensor]*3, dim=-1)
    else:
        img_tensor = img_tensor[:, :, :3]  # Remove alpha channel if present

    # ========================================================================
    # Step 2: Convert sRGB to linear RGB for accurate light propagation
    # 步骤 2：将 sRGB 转换为线性 RGB 以进行准确的光传播
    # ========================================================================
    img_linear = srgb_to_linear(img_tensor)
    print("  Converted to linear space")
    t2 = time.perf_counter()
    
    # ========================================================================
    # Step 3: Simulate optical propagation through PSF convolution
    # 步骤 3：通过 PSF 卷积模拟光学传播
    # ========================================================================
    output_linear = simulator.simulate_rgb_image_all_channels(img_linear)
    t3 = time.perf_counter()
    
    # ========================================================================
    # Step 4: Convert linear RGB back to sRGB for display
    # 步骤 4：将线性 RGB 转换回 sRGB 以便显示
    # ========================================================================
    output_tensor = linear_to_srgb(output_linear)
    print("  Converted back to sRGB")
    t4 = time.perf_counter()
    
    # ========================================================================
    # Step 5: Save output image
    # 步骤 5：保存输出图像
    # ========================================================================
    
    # Move back to CPU and convert to numpy
    # 移回 CPU 并转换为 numpy
    output_array = output_tensor.cpu().numpy()
    
    # Clip to valid range
    # 限制到有效范围
    output_array = np.clip(output_array, 0, 1)
    
    # Convert to 8-bit for saving
    # 转换为 8 位以便保存
    output_array = (output_array * 255).astype(np.uint8)
    output_img = Image.fromarray(output_array)
    
    # Save with optimized compression for PNG
    # 对 PNG 使用优化的压缩保存
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.png':
        output_img.save(output_path, compress_level=1)  # Fast compression
    else:
        output_img.save(output_path)
    t5 = time.perf_counter()
    print(f"  Saved output: {output_path}")
    
    # ========================================================================
    # Print timing breakdown for performance analysis
    # 打印计时分解以进行性能分析
    # ========================================================================
    load_ms = (t1 - t0) * 1000.0        # Image loading and resizing
    convert_ms = (t2 - t1) * 1000.0     # sRGB to linear conversion
    simulate_ms = (t3 - t2) * 1000.0    # PSF simulation (main computation)
    to_srgb_ms = (t4 - t3) * 1000.0     # Linear to sRGB conversion
    save_ms = (t5 - t4) * 1000.0        # Image saving
    total_ms = (t5 - t0) * 1000.0       # Total pipeline time
    
    print(f"  IO/Profile (ms): load={load_ms:.2f}, lin={convert_ms:.2f}, " +
          f"sim={simulate_ms:.2f}, to_srgb={to_srgb_ms:.2f}, save={save_ms:.2f}, total={total_ms:.2f}")


def process_image_file_batch(input_path, output_dir, simulator,
                             batch_size=4, device='cuda', save_all=True,
                             strict_benchmark=False, warmup_iters=10, measure_iters=100):
    """
    Load one image, replicate it batch_size times, run batched simulation, and save outputs.
    加载一张图片，复制 batch_size 份，批量并行模拟并保存输出。
    
    Args:
        input_path: path of the single input image
        output_dir: directory to save batched outputs
        simulator: PSFSimulatorGPU instance
        batch_size: number of copies
        device: 'cuda' or 'cpu'
        save_all: if True, save all outputs; else save only the first
    """
    print(f"\nBatch processing image: {input_path}  (batch_size={batch_size})")

    t0 = time.perf_counter()
    # Load image
    img = Image.open(input_path)
    if img.size != (INPUT_SIZE, INPUT_SIZE):
        print(f"  Resizing from {img.size} to ({INPUT_SIZE}, {INPUT_SIZE})")
        img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    t1 = time.perf_counter()

    # To torch and linear space
    img_tensor = torch.from_numpy(img_array).to(device=device, dtype=torch.float32)
    if len(img_tensor.shape) == 2:
        img_tensor = torch.stack([img_tensor]*3, dim=-1)
    else:
        img_tensor = img_tensor[:, :, :3]

    img_linear = srgb_to_linear(img_tensor)
    t2 = time.perf_counter()

    # Replicate for batch
    img_linear_batch = img_linear.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Run batched simulation
    output_linear_batch = simulator.simulate_rgb_images_batch(img_linear_batch)
    t3 = time.perf_counter()

    # Strict benchmark: warmup + timed runs (simulation-only)
    if strict_benchmark:
        print(f"\n[Strict Benchmark] warmup={warmup_iters}, iters={measure_iters}, batch={batch_size}")
        # Always use PyTorch path (CUDA extension not fully available)
        bench_fn = simulator.simulate_rgb_images_batch

        prev_prof_enabled = getattr(simulator.profiler, 'enabled', False)
        simulator.profiler.enabled = False
        try:
            if device == 'cuda':
                torch.cuda.synchronize()
            for _ in range(warmup_iters):
                _ = bench_fn(img_linear_batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            t0b = time.perf_counter()
            for _ in range(measure_iters):
                _ = bench_fn(img_linear_batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1b = time.perf_counter()
        finally:
            simulator.profiler.enabled = prev_prof_enabled

        total_ms_b = (t1b - t0b) * 1000.0
        avg_ms_b = total_ms_b / measure_iters
        fps_strict = (1000.0 / avg_ms_b) * batch_size
        print(f"  Strict sim-only: avg={avg_ms_b:.2f} ms/iter, FPS={fps_strict:.2f}")

    # Convert back to sRGB
    output_srgb_batch = linear_to_srgb(output_linear_batch)
    t4 = time.perf_counter()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    if save_all:
        for b in range(batch_size):
            out_np = output_srgb_batch[b].clamp(0, 1).detach().cpu().numpy()
            out_np = (out_np * 255).astype(np.uint8)
            out_img = Image.fromarray(out_np)
            out_path = os.path.join(output_dir, f"{base}_b{b}.png")
            out_img.save(out_path, compress_level=1)
    else:
        out_np = output_srgb_batch[0].clamp(0, 1).detach().cpu().numpy()
        out_np = (out_np * 255).astype(np.uint8)
        out_img = Image.fromarray(out_np)
        out_path = os.path.join(output_dir, f"{base}_b0.png")
        out_img.save(out_path, compress_level=1)
    t5 = time.perf_counter()

    load_ms = (t1 - t0) * 1000.0
    convert_ms = (t2 - t1) * 1000.0
    simulate_ms = (t3 - t2) * 1000.0
    to_srgb_ms = (t4 - t3) * 1000.0
    save_ms = (t5 - t4) * 1000.0
    total_ms = (t5 - t0) * 1000.0
    print(f"  Batch IO/Profile (ms): load={load_ms:.2f}, lin={convert_ms:.2f}, " +
          f"sim={simulate_ms:.2f}, to_srgb={to_srgb_ms:.2f}, save={save_ms:.2f}, total={total_ms:.2f}")



def main():
    """
    Main function for batch processing test images.
    批量处理测试图像的主函数。
    
    This function:
    1. Creates PSFSimulatorGPU instance (precomputation happens once)
    2. Processes all images in test_input_dir
    3. Saves results to test_output_dir
    
    The key advantage: Precomputation (interpolation weights, chief rays, etc.)
    is done only once and reused for all images.
    
    关键优势：预计算（插值权重、主光线等）只做一次，然后在所有图像间复用。
    """
    
    # ========================================================================
    # Configuration
    # 配置
    # ========================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Input/output directories
    # 输入/输出目录
    test_input_dir = "val_input"
    test_output_dir = "val_output_psf"
    
    # Batch test configuration
    # 批处理测试配置
    BATCH_TEST_SIZE = 0  # set >0 to enable batch mode, 0 for single image processing
    SAVE_ALL_BATCH_OUTPUTS = False
    
    if os.path.exists(test_input_dir):
        os.makedirs(test_output_dir, exist_ok=True)
        
        # ====================================================================
        # Create simulator once - precomputation happens here
        # 创建模拟器一次 - 预计算在这里进行
        # ====================================================================
        # This includes:
        # - Loading PSF data
        # - Computing interpolation weights for all input pixels
        # - Computing chief ray positions
        # - Precomputing output pixel dimensions
        # 
        # All of this is reused across multiple images!
        # 所有这些都在多张图片间复用！
        simulator = PSFSimulatorGPU(
            device=DEVICE,
            cache_dir=CACHE_DIR,
            chunk_size=262144,      # Process 102400 pixels at a time
            micro_batch_size = 1024,  # GPU batch size
            profile=True,           # Enable profiling
            profile_cuda=True,      # Use CUDA events for accurate GPU timing
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
        )

        # ====================================================================
        # Process images
        # 处理图像
        # ====================================================================
        if BATCH_TEST_SIZE > 0:
            # Batch test: pick one image and replicate it
            candidates = [f for f in os.listdir(test_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if len(candidates) == 0:
                print("No images found for batch test.")
                return
            filename = candidates[0]
            input_path = os.path.join(test_input_dir, filename)
            batch_out_dir = os.path.join(test_output_dir, f"batch_{BATCH_TEST_SIZE}")
            try:
                process_image_file_batch(
                    input_path, batch_out_dir,
                    simulator,
                    batch_size=BATCH_TEST_SIZE,
                    device=DEVICE,
                    save_all=SAVE_ALL_BATCH_OUTPUTS,
                    strict_benchmark=True,
                    warmup_iters=10,
                    measure_iters=100,
                )
            except Exception as e:
                print(f"Error batch-processing {filename}: {e}")
                import traceback
                traceback.print_exc()
            # Skip CUDA vs Torch correctness test (psf_cuda_ext_v2 not available)
            pass
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
        else:
            for filename in os.listdir(test_input_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    input_path = os.path.join(test_input_dir, filename)
                    output_path = os.path.join(test_output_dir, filename)
                    
                    try:
                        process_image_file(
                            input_path, output_path, 
                            simulator,
                            device=DEVICE
                        )
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Clear GPU cache after each image to avoid memory buildup
                    # 每张图像后清除 GPU 缓存以避免内存积累
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
    else:
        print(f"Test directory '{test_input_dir}' not found.")
        print("Please create it and add test images, or modify the script.")


if __name__ == "__main__":
    main()
