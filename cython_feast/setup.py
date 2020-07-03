from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("feast",
              sources=["feast.pyx"],
              include_dirs=[numpy.get_include(),"../FEAST/include"],
              library_dirs=["../FEAST/lib/x64"],
              libraries=["feast"],
              extra_link_args=['-lmkl_gf_lp64','-lmkl_gnu_thread','-lmkl_core','-lgomp','-lpthread','-lm','-ldl','-lgfortran']
              )
]

setup(name="feast", version='0.1',
      ext_modules=cythonize(ext_modules))
