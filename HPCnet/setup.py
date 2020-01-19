from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='HPCnet',
    ext_modules=[
        CUDAExtension('HPCnet_cuda', [
            'src/HPCnet_api.cpp',
            'src/get_hausdorff_dis.cpp',
            'src/get_hausdorff_dis_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
