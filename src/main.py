from pathlib import Path

from src.calculation.compile import compile_cython_files


def main():
    # if compile:
    #     # Compile Cython .pyx files
    #     import subprocess
    #     path = Path(__file__).parent / 'setup.py'
    #     cmd = f'python {str(path)} build_ext --inplace'
    #     p = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE,
    #                        stderr=subprocess.STDOUT, encoding='utf-8')
    #     print(p.stdout)

    compile_cython_files()

    import sys
    # Add subdirectory manually to sys.path. This is necessary because we can't place an __init__.py file into
    # the subdirectory, as this breaks Cython (this is a known Cython bug)
    # This enables us to import modules from the subdirectory directly, e.g. 'import reflectivity_c_file'
    file_path = str(Path(__file__).parent / 'calculation')
    sys.path.insert(1, file_path)

    # Run the calculation
    run()


def run():
    from src.calculation.Manager import Optimiser



if __name__ == '__main__':
    main()