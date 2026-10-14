"""
PSF CUDA Extension V2 Setup (Optimized Version)

Prerequisites:
    1. Install PyTorch with matching CUDA version first:
       pip install torch --index-url https://download.pytorch.org/whl/cu130

    2. Then install this package:
       pip install -e .
"""
from setuptools import setup, find_packages

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise ImportError(
        "PyTorch is required to build this package.\n"
        "Please install PyTorch with matching CUDA version first:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu130\n"
        "Then retry installing this package."
    )

setup(
    name="psf_cuda_ext_v2",
    version="0.1.0",
    description="CUDA-accelerated PSF accumulation kernel (optimized v2)",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="psf_cuda_ext_v2._C",
            sources=[
                "src/psf_cuda_ext_v2/psf_cuda.cpp",
                "src/psf_cuda_ext_v2/psf_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.8",
)
