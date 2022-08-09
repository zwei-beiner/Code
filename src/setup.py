# Build with
# python setup.py build_ext --inplace

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy

extensions = [Extension("*", ["*.pyx"])]

# Deactivate bounds checking, Deactivate negative indexing, No division by zero checking
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': 3, 'boundscheck': False, 'wraparound': False, 'cdivision': True}),
    include_dirs=[numpy.get_include()]
)