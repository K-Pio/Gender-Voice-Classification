from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

define_macros = []
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': ['-O3', '-std=c++17'],
}

# Jeżeli CUDA jest dostępna, dodaj makro
if CUDA_HOME is not None:
    define_macros.append(('WITH_CUDA', None))

setup(
    name='onion_layers',
    ext_modules=[
        CUDAExtension(
            name='conv1d.cuda_bn.o_conv1d',
            sources=[
                'conv1d/conv1d.cpp',
                'conv1d/conv1d_cuda.cu',
            ],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name='dropout.cuda_bn.o_dropout',
            sources=[
                'dropout/dropout.cpp',
                'dropout/dropout_cuda.cu'
            ],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
