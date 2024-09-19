from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import warnings
warnings.filterwarnings("ignore")

setup(
    name='c_smt_sa',
    ext_modules=[
        CppExtension('c_smt_sa', [
            'main.cpp',
            'fifo.cpp',
            'node_pu_os.cpp',
            'node_mem.cpp',
            'grid_os.cpp',
            'smt_sa_os.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
