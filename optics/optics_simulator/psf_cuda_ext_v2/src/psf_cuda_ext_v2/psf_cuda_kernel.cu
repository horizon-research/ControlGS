#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int PSF_SIZE = 64;
constexpr int MAX_WIN_SIZE = 256;  // Maximum window size (2 * max_psf_radius_pixels + 1)
constexpr float PSF_SAMPLE_THRESHOLD = 1e-6f;  // Skip small PSF values to reduce atomic writes

// Cubic interpolation weight function (Keys' cubic, a = -0.75, matches PyTorch)
__device__ inline float cubic_weight(float x) {
    const float a = -0.75f;
    x = fabsf(x);
    if (x < 1.0f) {
        return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
    }
    return 0.0f;
}

// =============================================================================
// Forward kernel - each block handles one input pixel
// =============================================================================
__global__ void psf_accumulate_kernel(
    const float* __restrict__ intensities_B,  // [B, N]
    const int* __restrict__ nbr_iy,           // [4, N]
    const int* __restrict__ nbr_ix,           // [4, N]
    const float* __restrict__ weights,        // [4, N]
    const float* __restrict__ chief_xy_mm,    // [N, 2]
    const float* __restrict__ psf_geo,        // [ny, nx, 3, Hpsf, Wpsf]
    int B,
    int N,
    int ny,
    int nx,
    int nwl,
    int Hpsf,
    int Wpsf,
    int wl_idx,
    int output_size,
    float output_pixel_size_mm,
    float psf_radius_mm,
    int psf_radius_pixels,
    float* __restrict__ out  // [B, output_size, output_size]
) {
    // Each block handles one input pixel
    int n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;

    // Shared memory for interpolated PSF
    __shared__ float psf_shared[PSF_SIZE * PSF_SIZE];

    // Load neighbor info (same for all threads in block)
    __shared__ float w4[4];
    __shared__ const float* psf_plane[4];
    __shared__ float cx, cy;
    __shared__ int x0, y0;

    if (tid < 4) {
        int fiy = nbr_iy[tid * N + n];
        int fix = nbr_ix[tid * N + n];
        w4[tid] = weights[tid * N + n];
        int offset = (((fiy * nx + fix) * nwl + wl_idx) * Hpsf) * Wpsf;
        psf_plane[tid] = psf_geo + offset;
    }

    if (tid == 0) {
        cx = chief_xy_mm[n * 2 + 0];
        cy = chief_xy_mm[n * 2 + 1];
        float half_span_mm = 0.5f * output_pixel_size_mm * (float)output_size;
        float cpx = (cx + half_span_mm) / output_pixel_size_mm;
        float cpy = (cy + half_span_mm) / output_pixel_size_mm;
        x0 = static_cast<int>(floorf(cpx)) - psf_radius_pixels;
        y0 = static_cast<int>(floorf(cpy)) - psf_radius_pixels;
    }

    __syncthreads();

    // Step 1: Cooperatively load and interpolate PSF into shared memory
    // 4096 elements, 256 threads -> 16 elements per thread
    // PSF is stored as FP16, convert to FP32 during load
    int psf_total = PSF_SIZE * PSF_SIZE;
    for (int i = tid; i < psf_total; i += BLOCK_SIZE) {
        int py = i / PSF_SIZE;
        int px = i % PSF_SIZE;
        float val = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            val += w4[k] * psf_plane[k][py * Wpsf + px];
        }
        psf_shared[i] = val;
    }

    __syncthreads();

    // Step 2: Precompute cubic weights for rows and columns
    // This eliminates repeated cubic_weight() calls
    const float half_span_mm = 0.5f * output_pixel_size_mm * (float)output_size;
    const float p_area = output_pixel_size_mm * output_pixel_size_mm;
    const int win_hw = 2 * psf_radius_pixels + 1;
    const int total_window = win_hw * win_hw;
    const int HW = output_size * output_size;

    // Shared memory for precomputed weights
    // wx_table[col][4]: cubic weights for each column
    // wy_table[row][4]: cubic weights for each row
    // psf_x0[col]: floor(x) for each column
    // psf_y0[row]: floor(y) for each row
    __shared__ float wx_table[MAX_WIN_SIZE * 4];  // [win_hw][4] flattened
    __shared__ float wy_table[MAX_WIN_SIZE * 4];  // [win_hw][4] flattened
    __shared__ int psf_x0[MAX_WIN_SIZE];
    __shared__ int psf_y0[MAX_WIN_SIZE];

    float local_cx = cx;
    float local_cy = cy;
    int local_x0 = x0;
    int local_y0 = y0;

    // Precompute x-direction weights (columns)
    for (int wj = tid; wj < win_hw; wj += BLOCK_SIZE) {
        int x_idx = local_x0 + wj;
        float x_mm = ((float)x_idx + 0.5f) * output_pixel_size_mm - half_span_mm;
        float du = (x_mm - local_cx) / psf_radius_mm;

        // Check bounds for du
        if (du < -1.f || du > 1.f) {
            psf_x0[wj] = -999;  // Mark as invalid
        } else {
            float px = ((du + 1.f) * PSF_SIZE - 1.f) * 0.5f;
            int ix0 = static_cast<int>(floorf(px));
            float dx = px - (float)ix0;
            psf_x0[wj] = ix0;
            wx_table[wj * 4 + 0] = cubic_weight(dx + 1.f);
            wx_table[wj * 4 + 1] = cubic_weight(dx);
            wx_table[wj * 4 + 2] = cubic_weight(dx - 1.f);
            wx_table[wj * 4 + 3] = cubic_weight(dx - 2.f);
        }
    }

    // Precompute y-direction weights (rows)
    for (int wi = tid; wi < win_hw; wi += BLOCK_SIZE) {
        int y_idx = local_y0 + wi;
        float y_mm = ((float)y_idx + 0.5f) * output_pixel_size_mm - half_span_mm;
        float dv = (y_mm - local_cy) / psf_radius_mm;

        // Check bounds for dv
        if (dv < -1.f || dv > 1.f) {
            psf_y0[wi] = -999;  // Mark as invalid
        } else {
            float py = ((dv + 1.f) * PSF_SIZE - 1.f) * 0.5f;
            int iy0 = static_cast<int>(floorf(py));
            float dy = py - (float)iy0;
            psf_y0[wi] = iy0;
            wy_table[wi * 4 + 0] = cubic_weight(dy + 1.f);
            wy_table[wi * 4 + 1] = cubic_weight(dy);
            wy_table[wi * 4 + 2] = cubic_weight(dy - 1.f);
            wy_table[wi * 4 + 3] = cubic_weight(dy - 2.f);
        }
    }

    __syncthreads();

    // Step 3: Process window using precomputed weights
    for (int idx = tid; idx < total_window; idx += BLOCK_SIZE) {
        int wi = idx / win_hw;
        int wj = idx % win_hw;

        int y_idx = local_y0 + wi;
        int x_idx = local_x0 + wj;

        // Bounds check for output
        if (y_idx < 0 || y_idx >= output_size || x_idx < 0 || x_idx >= output_size) {
            continue;
        }

        // Check if PSF coordinates are valid
        int ix0 = psf_x0[wj];
        int iy0 = psf_y0[wi];
        if (ix0 == -999 || iy0 == -999) {
            continue;
        }

        // Load precomputed weights
        float wx0 = wx_table[wj * 4 + 0];
        float wx1 = wx_table[wj * 4 + 1];
        float wx2 = wx_table[wj * 4 + 2];
        float wx3 = wx_table[wj * 4 + 3];
        float wy0 = wy_table[wi * 4 + 0];
        float wy1 = wy_table[wi * 4 + 1];
        float wy2 = wy_table[wi * 4 + 2];
        float wy3 = wy_table[wi * 4 + 3];

        // Bicubic interpolation using precomputed weights
        float sample = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int yj = iy0 - 1 + j;
            float wy_j = (j == 0) ? wy0 : (j == 1) ? wy1 : (j == 2) ? wy2 : wy3;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int xi = ix0 - 1 + i;
                float wx_i = (i == 0) ? wx0 : (i == 1) ? wx1 : (i == 2) ? wx2 : wx3;
                float val = 0.f;
                if (xi >= 0 && xi < PSF_SIZE && yj >= 0 && yj < PSF_SIZE) {
                    val = psf_shared[yj * PSF_SIZE + xi];
                }
                sample += wx_i * wy_j * val;
            }
        }

        if (fabsf(sample) < PSF_SAMPLE_THRESHOLD) continue;

        int out_idx_base = y_idx * output_size + x_idx;

        // Accumulate for each batch
        for (int b = 0; b < B; ++b) {
            float inten = intensities_B[b * N + n];
            float contrib = inten * sample * p_area;
            if (contrib != 0.f) {
                atomicAdd(out + b * HW + out_idx_base, contrib);
            }
        }
    }
}

// =============================================================================
// Backward kernel - each block handles one input pixel, all batches
// Precompute PSF samples, then accumulate for all batches
// =============================================================================
__global__ void psf_accumulate_backward_kernel(
    const float* __restrict__ grad_out,       // [B, Hout, Wout]
    const int* __restrict__ nbr_iy,           // [4, N]
    const int* __restrict__ nbr_ix,           // [4, N]
    const float* __restrict__ weights,        // [4, N]
    const float* __restrict__ chief_xy_mm,    // [N, 2]
    const float* __restrict__ psf_geo,        // [ny, nx, 3, Hpsf, Wpsf]
    int B,
    int N,
    int ny,
    int nx,
    int nwl,
    int Hpsf,
    int Wpsf,
    int wl_idx,
    int output_size,
    float output_pixel_size_mm,
    float psf_radius_mm,
    int psf_radius_pixels,
    float* __restrict__ grad_intensities  // [B, N]
) {
    int n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;

    __shared__ float psf_shared[PSF_SIZE * PSF_SIZE];
    __shared__ float w4[4];
    __shared__ const float* psf_plane[4];
    __shared__ float cx, cy;
    __shared__ int x0, y0;

    if (tid < 4) {
        int fiy = nbr_iy[tid * N + n];
        int fix = nbr_ix[tid * N + n];
        w4[tid] = weights[tid * N + n];
        int offset = (((fiy * nx + fix) * nwl + wl_idx) * Hpsf) * Wpsf;
        psf_plane[tid] = psf_geo + offset;
    }

    if (tid == 0) {
        cx = chief_xy_mm[n * 2 + 0];
        cy = chief_xy_mm[n * 2 + 1];
        float half_span_mm = 0.5f * output_pixel_size_mm * (float)output_size;
        float cpx = (cx + half_span_mm) / output_pixel_size_mm;
        float cpy = (cy + half_span_mm) / output_pixel_size_mm;
        x0 = static_cast<int>(floorf(cpx)) - psf_radius_pixels;
        y0 = static_cast<int>(floorf(cpy)) - psf_radius_pixels;
    }

    __syncthreads();

    // Load and interpolate PSF (FP16 -> FP32 conversion)
    int psf_total = PSF_SIZE * PSF_SIZE;
    for (int i = tid; i < psf_total; i += BLOCK_SIZE) {
        int py = i / PSF_SIZE;
        int px = i % PSF_SIZE;
        float val = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            val += w4[k] * psf_plane[k][py * Wpsf + px];
        }
        psf_shared[i] = val;
    }

    __syncthreads();

    const float half_span_mm = 0.5f * output_pixel_size_mm * (float)output_size;
    const float p_area = output_pixel_size_mm * output_pixel_size_mm;
    const int win_hw = 2 * psf_radius_pixels + 1;
    const int total_window = win_hw * win_hw;
    const int HW = output_size * output_size;

    float local_cx = cx;
    float local_cy = cy;
    int local_x0 = x0;
    int local_y0 = y0;

    // Shared memory for precomputed weights (same as forward kernel)
    __shared__ float wx_table[MAX_WIN_SIZE * 4];
    __shared__ float wy_table[MAX_WIN_SIZE * 4];
    __shared__ int psf_x0[MAX_WIN_SIZE];
    __shared__ int psf_y0[MAX_WIN_SIZE];

    // Precompute x-direction weights (columns)
    for (int wj = tid; wj < win_hw; wj += BLOCK_SIZE) {
        int x_idx = local_x0 + wj;
        float x_mm = ((float)x_idx + 0.5f) * output_pixel_size_mm - half_span_mm;
        float du = (x_mm - local_cx) / psf_radius_mm;

        if (du < -1.f || du > 1.f) {
            psf_x0[wj] = -999;
        } else {
            float px = ((du + 1.f) * PSF_SIZE - 1.f) * 0.5f;
            int ix0 = static_cast<int>(floorf(px));
            float dx = px - (float)ix0;
            psf_x0[wj] = ix0;
            wx_table[wj * 4 + 0] = cubic_weight(dx + 1.f);
            wx_table[wj * 4 + 1] = cubic_weight(dx);
            wx_table[wj * 4 + 2] = cubic_weight(dx - 1.f);
            wx_table[wj * 4 + 3] = cubic_weight(dx - 2.f);
        }
    }

    // Precompute y-direction weights (rows)
    for (int wi = tid; wi < win_hw; wi += BLOCK_SIZE) {
        int y_idx = local_y0 + wi;
        float y_mm = ((float)y_idx + 0.5f) * output_pixel_size_mm - half_span_mm;
        float dv = (y_mm - local_cy) / psf_radius_mm;

        if (dv < -1.f || dv > 1.f) {
            psf_y0[wi] = -999;
        } else {
            float py = ((dv + 1.f) * PSF_SIZE - 1.f) * 0.5f;
            int iy0 = static_cast<int>(floorf(py));
            float dy = py - (float)iy0;
            psf_y0[wi] = iy0;
            wy_table[wi * 4 + 0] = cubic_weight(dy + 1.f);
            wy_table[wi * 4 + 1] = cubic_weight(dy);
            wy_table[wi * 4 + 2] = cubic_weight(dy - 1.f);
            wy_table[wi * 4 + 3] = cubic_weight(dy - 2.f);
        }
    }

    __syncthreads();

    // Precompute PSF samples and output indices for this thread's window positions
    constexpr int MAX_POS_PER_THREAD = 32;
    float samples[MAX_POS_PER_THREAD];
    int out_indices[MAX_POS_PER_THREAD];
    int num_valid = 0;

    for (int idx = tid; idx < total_window && num_valid < MAX_POS_PER_THREAD; idx += BLOCK_SIZE) {
        int wi = idx / win_hw;
        int wj = idx % win_hw;

        int y_idx = local_y0 + wi;
        int x_idx = local_x0 + wj;

        if (y_idx < 0 || y_idx >= output_size || x_idx < 0 || x_idx >= output_size) {
            continue;
        }

        // Check if PSF coordinates are valid
        int ix0 = psf_x0[wj];
        int iy0 = psf_y0[wi];
        if (ix0 == -999 || iy0 == -999) {
            continue;
        }

        // Load precomputed weights
        float wx0 = wx_table[wj * 4 + 0];
        float wx1 = wx_table[wj * 4 + 1];
        float wx2 = wx_table[wj * 4 + 2];
        float wx3 = wx_table[wj * 4 + 3];
        float wy0 = wy_table[wi * 4 + 0];
        float wy1 = wy_table[wi * 4 + 1];
        float wy2 = wy_table[wi * 4 + 2];
        float wy3 = wy_table[wi * 4 + 3];

        // Bicubic interpolation using precomputed weights
        float sample = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int yj = iy0 - 1 + j;
            float wy_j = (j == 0) ? wy0 : (j == 1) ? wy1 : (j == 2) ? wy2 : wy3;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int xi = ix0 - 1 + i;
                float wx_i = (i == 0) ? wx0 : (i == 1) ? wx1 : (i == 2) ? wx2 : wx3;
                float val = 0.f;
                if (xi >= 0 && xi < PSF_SIZE && yj >= 0 && yj < PSF_SIZE) {
                    val = psf_shared[yj * PSF_SIZE + xi];
                }
                sample += wx_i * wy_j * val;
            }
        }

        if (fabsf(sample) >= PSF_SAMPLE_THRESHOLD) {
            samples[num_valid] = sample * p_area;
            out_indices[num_valid] = y_idx * output_size + x_idx;
            num_valid++;
        }
    }

    // Now process all batches using precomputed samples
    __shared__ float partial_grads[BLOCK_SIZE];

    for (int b = 0; b < B; ++b) {
        float thread_grad = 0.f;
        const float* grad_out_b = grad_out + b * HW;

        for (int i = 0; i < num_valid; ++i) {
            float go = grad_out_b[out_indices[i]];
            thread_grad += go * samples[i];
        }

        // Warp-level reduction first (faster than shared memory for small reductions)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_grad += __shfl_down_sync(0xffffffff, thread_grad, offset);
        }

        // Write warp results to shared memory
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        if (lane_id == 0) {
            partial_grads[warp_id] = thread_grad;
        }
        __syncthreads();

        // Final reduction across warps (8 warps for 256 threads)
        if (tid < 8) {
            float val = partial_grads[tid];
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xff, val, offset);
            }
            if (tid == 0) {
                grad_intensities[b * N + n] = val;
            }
        }
        __syncthreads();
    }
}

} // namespace

torch::Tensor psf_accumulate_forward_cuda(
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

    TORCH_CHECK(intensities_B.is_cuda(), "intensities_B must be CUDA");
    TORCH_CHECK(2 * psf_radius_pixels + 1 <= MAX_WIN_SIZE,
                "PSF window size exceeds MAX_WIN_SIZE");
    int B = static_cast<int>(intensities_B.size(0));
    int N = static_cast<int>(intensities_B.size(1));

    int ny = static_cast<int>(psf_geo.size(0));
    int nx = static_cast<int>(psf_geo.size(1));
    int nwl = static_cast<int>(psf_geo.size(2));
    int Hpsf = static_cast<int>(psf_geo.size(3));
    int Wpsf = static_cast<int>(psf_geo.size(4));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(intensities_B.device());
    auto out = torch::zeros({B, output_size, output_size}, options);

    // Each block handles one input pixel
    dim3 block(BLOCK_SIZE);
    dim3 grid(N);

    // Ensure psf_geo is FP16
    TORCH_CHECK(psf_geo.scalar_type() == torch::kFloat32, "psf_geo must be float32");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    psf_accumulate_kernel<<<grid, block, 0, stream>>>(
        intensities_B.data_ptr<float>(),
        nbr_iy.data_ptr<int>(),
        nbr_ix.data_ptr<int>(),
        weights.data_ptr<float>(),
        chief_xy_mm.data_ptr<float>(),
        psf_geo.data_ptr<float>(),
        B,
        N,
        ny,
        nx,
        nwl,
        Hpsf,
        Wpsf,
        wl_idx,
        output_size,
        static_cast<float>(output_pixel_size_mm),
        static_cast<float>(psf_radius_mm),
        psf_radius_pixels,
        out.data_ptr<float>());

    return out;
}

torch::Tensor psf_accumulate_backward_cuda(
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

    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(2 * psf_radius_pixels + 1 <= MAX_WIN_SIZE,
                "PSF window size exceeds MAX_WIN_SIZE");
    int B = static_cast<int>(grad_out.size(0));
    int Hout = static_cast<int>(grad_out.size(1));
    int Wout = static_cast<int>(grad_out.size(2));
    TORCH_CHECK(Hout == output_size && Wout == output_size, "grad_out size mismatch");

    int N = static_cast<int>(weights.size(1));

    int ny = static_cast<int>(psf_geo.size(0));
    int nx = static_cast<int>(psf_geo.size(1));
    int nwl = static_cast<int>(psf_geo.size(2));
    int Hpsf = static_cast<int>(psf_geo.size(3));
    int Wpsf = static_cast<int>(psf_geo.size(4));

    // Ensure psf_geo is FP16
    TORCH_CHECK(psf_geo.scalar_type() == torch::kFloat32, "psf_geo must be float32");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(grad_out.device());
    auto grad_intensities = torch::zeros({B, N}, options);

    // Each block handles one pixel (all batches)
    dim3 block(BLOCK_SIZE);
    dim3 grid(N);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    psf_accumulate_backward_kernel<<<grid, block, 0, stream>>>(
        grad_out.data_ptr<float>(),
        nbr_iy.data_ptr<int>(),
        nbr_ix.data_ptr<int>(),
        weights.data_ptr<float>(),
        chief_xy_mm.data_ptr<float>(),
        psf_geo.data_ptr<float>(),
        B,
        N,
        ny,
        nx,
        nwl,
        Hpsf,
        Wpsf,
        wl_idx,
        output_size,
        static_cast<float>(output_pixel_size_mm),
        static_cast<float>(psf_radius_mm),
        psf_radius_pixels,
        grad_intensities.data_ptr<float>());

    return grad_intensities;
}
