from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import os

# 获取 CUDA 路径
CUDA_HOME = "/usr/local/cuda"

ext_modules = [
    Pybind11Extension(
        "cutlass_gemm",  # Python 模块名
        ["cutlass_wrapper.cu"],  # C++/CUDA 源文件
        include_dirs=[
            os.path.join(CUDA_HOME, "include"),
            "./cutlass/include"  # CUTLASS 头文件路径
        ],
        library_dirs=[os.path.join(CUDA_HOME, "lib64")],
        libraries=["cudart"],  # 链接 CUDA 运行时库
        extra_compile_args={"cxx": ["-std=c++14"], "nvcc": ["-arch=sm_90"]},
    ),
]

setup(
    name="cutlass_gemm",
    ext_modules=ext_modules,
    zip_safe=False,
)