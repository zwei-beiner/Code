# Requirements
Python 3.9.13

[//]: # (Python packages:)

[//]: # (- anesthetic           2.0.0b12)

[//]: # (- Cython               0.29.32)

[//]: # (- fastkde              1.0.19)

[//]: # (- matplotlib           3.5.2)

[//]: # (- mpi4py               3.1.3)

[//]: # (- numpy                1.23.0)

[//]: # (- pandas               1.4.3)

[//]: # (- pypolychord          1.20.1)

[//]: # (- scikit-learn         1.1.2)

[//]: # (- scipy                1.8.1)

[//]: # (- hdbscan              0.8.28)


| Python Package | Minimum version | Recommended install command                                                                                                                                                                    |
|:--------------:|:---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   anesthetic   |    2.0.0b12     | `pip install git+https://github.com/williamjameshandley/anesthetic@fdf6599a3f4cc76ea60d65f4cd7503f9e0bcee63`                                                                                   |
|     Cython     |    0.29.32      | `pip install  Cython`                                                                                                                                                                          |
|    fastkde     |     1.0.19      | `pip install   fastkde`                                                                                                                                                                        |
|   matplotlib   |      3.5.2      | `pip install  matplotlib`                                                                                                                                                                      |
|     mpi4py     |      3.1.3      | (requires MPI installation, e.g. `mpich` or `openmpi`)<br/>`pip install   mpi4py`                                                                                                              |
|     numpy      |     1.23.0      | `pip install    numpy`                                                                                                                                                                         |
|     pandas     |      1.4.3      | `pip install  pandas`                                                                                                                                                                          |
|  pypolychord   |     1.20.1      | With MPI: `pip install git+https://github.com/PolyChord/PolyChordLite@master` <br/>Without MPI: `pip install git+https://github.com/PolyChord/PolyChordLite@master --global-option="--no-mpi"` |
|  scikit-learn  |      1.1.2      | `pip install    scikit-learn`                                                                                                                                                                  |
|     scipy      |      1.8.1      | `pip install   scipy`                                                                                                                                                                          |
|    hdbscan     |     0.8.28      | `pip install   hdbscan`                                                                                                                                                                        |

To determine whether PolyChord runs with MPI, run the `run_pypolychord.py` test script from the [pypolychord GitHub page](https://github.com/PolyChord/PolyChordLite) and check whether any errors are raised:
```shell
pip install git+https://github.com/PolyChord/PolyChordLite@master
wget https://raw.githubusercontent.com/PolyChord/PolyChordLite/master/run_pypolychord.py
python run_pypolychord.py
```

## Running with MPI

To run on `n_cores`, 

`mpirun -n <n_cores> python main.py`

## Running without MPI (on a single core)

`python main.py`

