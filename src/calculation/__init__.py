import sys
from pathlib import Path

# Add subdirectory manually to sys.path. This is necessary because we can't place an __init__.py file into
# the 'cython_files' subdirectory, as this breaks Cython (this is a known Cython bug).
# This enables us to import modules from the 'cython_files' subdirectory directly, e.g. 'import reflectivity_c_file'.
sys.path.insert(1, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent / 'cython_files'))