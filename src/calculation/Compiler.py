import subprocess
from pathlib import Path


def compile_cython_files() -> None:
    """
    Compiles cython '.pyx' files by calling the command 'python setup.py build_ext --inplace'. The '--inplace' command
    option places the generated '.so' files into the same folder as the cython '.pyx' files.
    """

    path = Path(__file__).parents[1] / 'setup.py'
    cmd = f'python {str(path)} build_ext --inplace'
    # Run command from command line.
    p = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    # Redirect stdout and stderr to stdout and print to console.
    print(p.stdout)