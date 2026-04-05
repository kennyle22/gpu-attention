from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fastAttention',
    ext_modules=[
        CUDAExtension('fastAttention', [
            'fastAttention.cpp',
            'fastAttention_kernels.cu',
        ],
        extra_compile_args={
            'nvcc': [
                '-O3',
                '-use_fast_math',
                '-ftz=true',
                '-prec-div=false',
                '-prec-sqrt=false',
                '-lineinfo',
                # needed for wmma (tensor cores) on V100/A100
                '-arch=sm_70',
                '-gencode=arch=compute_75,code=sm_75',  # T4 (Colab)
                '-gencode=arch=compute_80,code=sm_80',  # A100
            ]
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
