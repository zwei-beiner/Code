# Build with
# python setup.py build_ext --inplace

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy

import os
from pathlib import Path


# Path from which the command line calls the program
call_path = Path.cwd()
# Path of this file, i.e. path to the project folder
this_path = Path(__file__).parent
# Relative path from the command line to the project folder
relative_path = os.path.relpath(this_path, call_path)
# print(f'setup.py: cmd call path: {call_path}')
# print(f'setup.py: project path: {this_path}')
# string = '' if relative_path=='.' else relative_path
if relative_path == '.':
    string = ''
else:
    string = relative_path + '/'
# print(f'setup.py: relative_path: {string}')

# Path to Cython .pyx file
cython_path = string + 'calculation/reflectivity.pyx'
print(f'cython file path relative to terminal working directory: {cython_path}')
# Need dotted name for the location in which to save the .so file
dotted_name = string.replace('/', '.') + 'calculation.reflectivity_c_file'
print(f'dotted cython file path: {dotted_name}')

extensions = [Extension(dotted_name, [cython_path])]

# Compiler directives: Deactivate bounds checking, Deactivate negative indexing, No division by zero checking
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': 3, 'boundscheck': False, 'wraparound': False, 'cdivision': True}),
    include_dirs=[numpy.get_include()]
)