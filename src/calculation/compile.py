import subprocess
from pathlib import Path


def compile_cython_files():
    # Compile Cython .pyx files
    path = Path(__file__).parents[1] / 'setup.py'
    cmd = f'python {str(path)} build_ext --inplace'
    p = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    print(p.stdout)