#include <torch/extension.h>

// CUDA declaration
torch::Tensor psf_accumulate_forward_cuda(
    torch::Tensor intensities_B,   // [B, N] float32
    torch::Tensor nbr_iy,          // [4, N] int32
    torch::Tensor nbr_ix,          // [4, N] int32
    torch::Tensor weights,         // [4, N] float32
    torch::Tensor chief_xy_mm,     // [N, 2] float32 (for one channel)
    torch::Tensor psf_geo,         // [ny, nx, 3, Hpsf, Wpsf] float32
    int wl_idx,
    int output_size,
    double output_pixel_size_mm,
    double psf_radius_mm,
    int psf_radius_pixels);

torch::Tensor psf_accumulate_backward_cuda(
    torch::Tensor grad_out,         // [B, H, W]
    torch::Tensor nbr_iy,           // [4, N] int32
    torch::Tensor nbr_ix,           // [4, N] int32
    torch::Tensor weights,          // [4, N] float32
    torch::Tensor chief_xy_mm,      // [N, 2] float32
    torch::Tensor psf_geo,          // [ny, nx, 3, Hpsf, Wpsf]
    int wl_idx,
    int output_size,
    double output_pixel_size_mm,
    double psf_radius_mm,
    int psf_radius_pixels);

static void check_inputs(
    const torch::Tensor& intensities_B,
    const torch::Tensor& nbr_iy,
    const torch::Tensor& nbr_ix,
    const torch::Tensor& weights,
    const torch::Tensor& chief_xy_mm,
    const torch::Tensor& psf_geo) {
  TORCH_CHECK(intensities_B.is_cuda(), "intensities_B must be a CUDA tensor");
  TORCH_CHECK(nbr_iy.is_cuda() && nbr_ix.is_cuda(), "nbr_iy/nbr_ix must be CUDA tensors");
  TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
  TORCH_CHECK(chief_xy_mm.is_cuda(), "chief_xy_mm must be a CUDA tensor");
  TORCH_CHECK(psf_geo.is_cuda(), "psf_geo must be a CUDA tensor");

  TORCH_CHECK(intensities_B.dtype() == torch::kFloat32, "intensities_B must be float32");
  TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
  TORCH_CHECK(chief_xy_mm.dtype() == torch::kFloat32, "chief_xy_mm must be float32");
  TORCH_CHECK(psf_geo.dtype() == torch::kFloat32, "psf_geo must be float32");
  TORCH_CHECK(nbr_iy.dtype() == torch::kInt32 && nbr_ix.dtype() == torch::kInt32, "nbr_iy/nbr_ix must be int32");

  TORCH_CHECK(intensities_B.dim() == 2, "intensities_B must be [B, N]");
  TORCH_CHECK(nbr_iy.sizes() == weights.sizes(), "nbr_iy and weights must have the same shape [4, N]");
  TORCH_CHECK(nbr_ix.sizes() == weights.sizes(), "nbr_ix and weights must have the same shape [4, N]");
  TORCH_CHECK(chief_xy_mm.dim() == 2 && chief_xy_mm.size(1) == 2, "chief_xy_mm must be [N, 2]");
  TORCH_CHECK(psf_geo.dim() == 5 && psf_geo.size(2) == 3, "psf_geo must be [ny, nx, 3, H, W]");
}

torch::Tensor psf_accumulate_forward(
    torch::Tensor intensities_B,
    torch::Tensor nbr_iy,
    torch::Tensor nbr_ix,
    torch::Tensor weights,
    torch::Tensor chief_xy_mm,
    torch::Tensor psf_geo,
    int wl_idx,
    int output_size,
    double output_pixel_size_mm,
    double psf_radius_mm,
    int psf_radius_pixels) {

  check_inputs(intensities_B, nbr_iy, nbr_ix, weights, chief_xy_mm, psf_geo);

  auto B = intensities_B.size(0);
  auto device = intensities_B.device();
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto out = torch::zeros({B, output_size, output_size}, options);

  return psf_accumulate_forward_cuda(
      intensities_B.contiguous(),
      nbr_iy.contiguous(),
      nbr_ix.contiguous(),
      weights.contiguous(),
      chief_xy_mm.contiguous(),
      psf_geo.contiguous(),
      wl_idx,
      output_size,
      output_pixel_size_mm,
      psf_radius_mm,
      psf_radius_pixels);
}

torch::Tensor psf_accumulate_backward(
    torch::Tensor grad_out,
    torch::Tensor nbr_iy,
    torch::Tensor nbr_ix,
    torch::Tensor weights,
    torch::Tensor chief_xy_mm,
    torch::Tensor psf_geo,
    int wl_idx,
    int output_size,
    double output_pixel_size_mm,
    double psf_radius_mm,
    int psf_radius_pixels) {

  return psf_accumulate_backward_cuda(
      grad_out.contiguous(),
      nbr_iy.contiguous(),
      nbr_ix.contiguous(),
      weights.contiguous(),
      chief_xy_mm.contiguous(),
      psf_geo.contiguous(),
      wl_idx,
      output_size,
      output_pixel_size_mm,
      psf_radius_mm,
      psf_radius_pixels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "psf_accumulate_forward",
      &psf_accumulate_forward,
      "PSF accumulate forward (CUDA)",
      py::arg("intensities_B"),
      py::arg("nbr_iy"),
      py::arg("nbr_ix"),
      py::arg("weights"),
      py::arg("chief_xy_mm"),
      py::arg("psf_geo"),
      py::arg("wl_idx"),
      py::arg("output_size"),
      py::arg("output_pixel_size_mm"),
      py::arg("psf_radius_mm"),
      py::arg("psf_radius_pixels"));
  m.def(
      "psf_accumulate_backward",
      &psf_accumulate_backward,
      "PSF accumulate backward w.r.t. intensities (CUDA)",
      py::arg("grad_out"),
      py::arg("nbr_iy"),
      py::arg("nbr_ix"),
      py::arg("weights"),
      py::arg("chief_xy_mm"),
      py::arg("psf_geo"),
      py::arg("wl_idx"),
      py::arg("output_size"),
      py::arg("output_pixel_size_mm"),
      py::arg("psf_radius_mm"),
      py::arg("psf_radius_pixels"));
}
