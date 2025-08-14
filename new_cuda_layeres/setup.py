from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

define_macros = []
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': ['-O3', '-std=c++17'],
}

if CUDA_HOME is not None:
    define_macros.append(('WITH_CUDA', None))

setup(
    name='new_conv1d',
    ext_modules=[
        CUDAExtension(
            name='new_conv1d',
            sources=[
                'custom_conv1d.cpp',
                'custom_conv1d_cuda.cu',
            ],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
