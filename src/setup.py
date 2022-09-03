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

# Path to Cython .pyx files
cython_paths = [string + s for s in ['calculation/cython_files/reflectivity.pyx', 'calculation/cython_files/BackendCalculations.pyx']]
print(f'cython file path relative to terminal working directory: {cython_paths}')
# Need dotted name for the location in which to save the .so file
dotted_names = [string.replace('/', '.') + s
                for s in ['calculation.cython_files.reflectivity_for_import', 'calculation.cython_files.BackendCalculations_for_import']]
print(f'dotted cython file path: {dotted_names}')

extensions = [Extension(dotted_names[0], [cython_paths[0]]),
              Extension(dotted_names[1], [cython_paths[1]])]

# Compiler directives: Deactivate bounds checking, Deactivate negative indexing, No division by zero checking
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': 3, 'boundscheck': False, 'wraparound': False, 'cdivision': True},
        annotate=True,
        # force=True
    ),
    # Include directory './calculation' explicitly to search for .pyx and pxd files.
    include_dirs=[numpy.get_include(), str(Path(__file__).parent / 'calculation' / 'cython_files')]
)