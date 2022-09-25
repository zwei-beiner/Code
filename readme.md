Coating is a code for the design of optimal thin-film multilayer coatings.

## Documentation

- For instructions on how to run the code on a supercomputer, specifically the CSD3 cluster, [see here](https://github.com/zwei-beiner/Code/blob/master/docs/supercomputer_instructions.md).
- For instructions on how to set up a multilayer coating calculation, [see here](https://github.com/zwei-beiner/Code/blob/master/docs/user_manual.md).

## Requirements

The code requires at least Python 3.9.

The following Python packages are required:

| Python Package | Tested version | Recommended install command                                                                                                                                                                    |
|:--------------:|:--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   anesthetic   |    2.0.0b12    | `pip install git+https://github.com/williamjameshandley/anesthetic@fdf6599a3f4cc76ea60d65f4cd7503f9e0bcee63`                                                                                   |
|     Cython     |    0.29.32     | `pip install  Cython`                                                                                                                                                                          |
|    fastkde     |     1.0.19     | `pip install   fastkde`                                                                                                                                                                        |
|   matplotlib   |     3.5.2      | `pip install  matplotlib`                                                                                                                                                                      |
|     mpi4py     |     3.1.3      | (requires MPI installation, e.g. `mpich` or `openmpi`)<br/>`pip install   mpi4py`                                                                                                              |
|     numpy      |     1.23.0     | `pip install    numpy`                                                                                                                                                                         |
|     pandas     |     1.4.3      | `pip install  pandas`                                                                                                                                                                          |
|  pypolychord   |     1.20.1     | With MPI: `pip install git+https://github.com/PolyChord/PolyChordLite@master` <br/>Without MPI: `pip install git+https://github.com/PolyChord/PolyChordLite@master --global-option="--no-mpi"` |
|  scikit-learn  |     1.1.2      | `pip install    scikit-learn`                                                                                                                                                                  |
|     scipy      |     1.8.1      | `pip install   scipy`                                                                                                                                                                          |
|    hdbscan     |     0.8.28     | `pip install   hdbscan`                                                                                                                                                                        |
| joblib         |     1.1.0      | `pip install joblib==1.1.0`                                                                                                                                                                    |

Remarks:
- The `hdbscan` package depends on the `joblib` package. It has been found that on the CSD3 cluster, the latest version of `joblib` causes the import of `hdbscan` to fail. This is why the version `joblib==1.1.0` is recommended.
- The `anesthetic` package has been found to work with the specific commit above. At the time of writing, later versions cause the import of the PolyChord `test.resume` file to fail.
- To determine whether PolyChord runs with MPI on the user's machine, the `run_pypolychord.py` test script from the [pypolychord GitHub page](https://github.com/PolyChord/PolyChordLite) should be run and it should be checked whether any errors are raised:
```shell
pip install git+https://github.com/PolyChord/PolyChordLite@master
wget https://raw.githubusercontent.com/PolyChord/PolyChordLite/master/run_pypolychord.py
python run_pypolychord.py
```

## Installation 

Clone the git repository with

```shell
git clone https://github.com/zwei-beiner/Code
```

Any code should be run from inside the directory `src`, which contains an example `main.py` file. 

## Running with MPI

To run on `n_cores`, 

```shell
mpirun -n <n_cores> python main.py
```

## Running without MPI (on a single core)

```shell
python main.py
```

 This runs the code with the MPI rank set to zero.

