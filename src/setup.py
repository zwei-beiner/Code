# from setuptools import setup
# from Cython.Build import cythonize
#
import numpy
#
# setup(
#     ext_modules = cythonize("reflectivity.pyx"),
#     include_dirs=[numpy.get_include()]
# )
#
# # Build with
# # python setup.py build_ext --inplace

from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [Extension("reflectivity", ["reflectivity.pyx"])]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)