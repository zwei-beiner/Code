from pathlib import Path

from calculation.compile import compile_cython_files


def main():
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
    from src.calculation.Manager import Runner


if __name__ == '__main__':
    main()