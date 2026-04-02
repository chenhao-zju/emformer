from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='triple_conv',
    ext_modules=[
        CUDAExtension(
            name='triple_conv',
            sources=[
                'triple_conv.cpp',
                'triple_conv_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)