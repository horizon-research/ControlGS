import torch

try:
    from . import _C
except Exception as e:
    _C = None
    _import_error = e
else:
    _import_error = None


def is_available() -> bool:
    return _C is not None


def psf_accumulate_forward(
    intensities_B: torch.Tensor,
    nbr_iy: torch.Tensor,
    nbr_ix: torch.Tensor,
    weights: torch.Tensor,
    chief_xy_mm: torch.Tensor,
    psf_geo: torch.Tensor,
    wl_idx: int,
    output_size: int,
    output_pixel_size_mm: float,
    psf_radius_mm: float,
    psf_radius_pixels: int,
) -> torch.Tensor:
    if _C is None:
        raise RuntimeError(
            f"psf_cuda_ext_v2 not built or import failed: {_import_error}"
        )

    # Ensure expected dtypes and layout
    intensities_B = intensities_B.contiguous().to(dtype=torch.float32)
    nbr_iy = nbr_iy.contiguous().to(dtype=torch.int32)
    nbr_ix = nbr_ix.contiguous().to(dtype=torch.int32)
    weights = weights.contiguous().to(dtype=torch.float32)
    chief_xy_mm = chief_xy_mm.contiguous().to(dtype=torch.float32)
    psf_geo = psf_geo.contiguous().to(dtype=torch.float32)

    return _C.psf_accumulate_forward(
        intensities_B,
        nbr_iy,
        nbr_ix,
        weights,
        chief_xy_mm,
        psf_geo,
        int(wl_idx),
        int(output_size),
        float(output_pixel_size_mm),
        float(psf_radius_mm),
        int(psf_radius_pixels),
    )

class _PSFAccumulateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                intensities_B: torch.Tensor,
                nbr_iy: torch.Tensor,
                nbr_ix: torch.Tensor,
                weights: torch.Tensor,
                chief_xy_mm: torch.Tensor,
                psf_geo: torch.Tensor,
                wl_idx: int,
                output_size: int,
                output_pixel_size_mm: float,
                psf_radius_mm: float,
                psf_radius_pixels: int):
        if _C is None:
            raise RuntimeError(f"psf_cuda_ext_v2 not built or import failed: {_import_error}")
        intensities_B = intensities_B.contiguous().to(dtype=torch.float32)
        nbr_iy = nbr_iy.contiguous().to(dtype=torch.int32)
        nbr_ix = nbr_ix.contiguous().to(dtype=torch.int32)
        weights = weights.contiguous().to(dtype=torch.float32)
        chief_xy_mm = chief_xy_mm.contiguous().to(dtype=torch.float32)
        psf_geo = psf_geo.contiguous().to(dtype=torch.float32)

        out = _C.psf_accumulate_forward(
            intensities_B, nbr_iy, nbr_ix, weights, chief_xy_mm, psf_geo,
            int(wl_idx), int(output_size),
            float(output_pixel_size_mm), float(psf_radius_mm), int(psf_radius_pixels)
        )
        # Save tensors for backward
        ctx.save_for_backward(nbr_iy, nbr_ix, weights, chief_xy_mm, psf_geo)
        ctx.wl_idx = int(wl_idx)
        ctx.output_size = int(output_size)
        ctx.output_pixel_size_mm = float(output_pixel_size_mm)
        ctx.psf_radius_mm = float(psf_radius_mm)
        ctx.psf_radius_pixels = int(psf_radius_pixels)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (nbr_iy, nbr_ix, weights, chief_xy_mm, psf_geo) = ctx.saved_tensors
        grad_out = grad_out.contiguous().to(dtype=torch.float32)
        grad_intensities = _C.psf_accumulate_backward(
            grad_out,
            nbr_iy,
            nbr_ix,
            weights,
            chief_xy_mm,
            psf_geo,
            int(ctx.wl_idx),
            int(ctx.output_size),
            float(ctx.output_pixel_size_mm),
            float(ctx.psf_radius_mm),
            int(ctx.psf_radius_pixels),
        )
        # Return gradients aligned with forward inputs; only intensities_B has grad
        return grad_intensities, None, None, None, None, None, None, None, None, None, None

def psf_accumulate(intensities_B: torch.Tensor,
                   nbr_iy: torch.Tensor,
                   nbr_ix: torch.Tensor,
                   weights: torch.Tensor,
                   chief_xy_mm: torch.Tensor,
                   psf_geo: torch.Tensor,
                   wl_idx: int,
                   output_size: int,
                   output_pixel_size_mm: float,
                   psf_radius_mm: float,
                   psf_radius_pixels: int) -> torch.Tensor:
    return _PSFAccumulateFn.apply(intensities_B, nbr_iy, nbr_ix, weights, chief_xy_mm, psf_geo,
                                  wl_idx, output_size, output_pixel_size_mm, psf_radius_mm,
                                  psf_radius_pixels)


def psf_accumulate_backward(
    grad_out: torch.Tensor,
    nbr_iy: torch.Tensor,
    nbr_ix: torch.Tensor,
    weights: torch.Tensor,
    chief_xy_mm: torch.Tensor,
    psf_geo: torch.Tensor,
    wl_idx: int,
    output_size: int,
    output_pixel_size_mm: float,
    psf_radius_mm: float,
    psf_radius_pixels: int,
) -> torch.Tensor:
    if _C is None:
        raise RuntimeError(f"psf_cuda_ext_v2 not built or import failed: {_import_error}")

    grad_out = grad_out.contiguous().to(dtype=torch.float32)
    nbr_iy = nbr_iy.contiguous().to(dtype=torch.int32)
    nbr_ix = nbr_ix.contiguous().to(dtype=torch.int32)
    weights = weights.contiguous().to(dtype=torch.float32)
    chief_xy_mm = chief_xy_mm.contiguous().to(dtype=torch.float32)
    psf_geo = psf_geo.contiguous().to(dtype=torch.float32)

    return _C.psf_accumulate_backward(
        grad_out,
        nbr_iy,
        nbr_ix,
        weights,
        chief_xy_mm,
        psf_geo,
        int(wl_idx),
        int(output_size),
        float(output_pixel_size_mm),
        float(psf_radius_mm),
        int(psf_radius_pixels),
    )
