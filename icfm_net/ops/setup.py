from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ICFMNET_OP',
    ext_modules=[
        CUDAExtension(
            'ICFMNET_OP', ['src/icfmnet_api.cpp', 'src/icfmnet_ops.cpp', 'src/cuda.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            })
    ],
    cmdclass={'build_ext': BuildExtension})
