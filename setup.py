from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='icfm_net',
        version='1.0',
        packages=['icfm_net'],
        package_data={'icfm_net.ops': ['*/*.so']},
        ext_modules=[
            CUDAExtension(
                name='icfm_net.ops.ops',
                sources=[
                    'icfm_net/ops/src/icfmnet_api.cpp',
                    'icfm_net/ops/src/icfmnet_ops.cpp',
                    'icfm_net/ops/src/cuda.cu'
                ],
                extra_compile_args={
                    'cxx': ['-g'],
                    'nvcc': ['-O2']
                })
        ],
        cmdclass={'build_ext': BuildExtension})
