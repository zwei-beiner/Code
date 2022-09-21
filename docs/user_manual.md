# User manual

This manual provides instructions on how to set up a multilayer coating calculation using the `coating` code. This concerns the creation of the `main.py` file in the [Instructions for the CSD3 cluster](https://github.com/zwei-beiner/Code/blob/master/docs/supercomputer_instructions.md).

In the following, the types `Callable` and `Union` from the Python `typing` library will be used.

This user manual describes
1. How to compile the Cython files,
2. How to set the parameters of a multilayer calculation exemplified with a 5-layer coating,
3. How to run the calculation,
4. Useful functions in the `Utils` class,
5. A non-trivial example calculation with separate wavelength intervals,
6. The output files generated during a run.

## Compiling the Cython files

First, it should be ensured that the Cython files are compiled to `.so` files.

This is done by calling the function `compile_cython_files()`:
```python
from calculation.Compiler import compile_cython_files
compile_cython_files()
```

The function calls the following command:
```shell
python setup.py build_ext --inplace
```
which calls the `setup.py` script to compile the Cython files.

## Parameters for the calculation

All refractive indices are functions of wavelength and it must be possible to evaluate them at any wavelength specified by the input wavelengths. This means that they must take a `numpy` array as an input and return a `numpy` array, i.e. they must be of type `Callable[[numpy.ndarray], np.ndarray]`, which is defined to be the type `RefractiveIndex`.

The following parameters must be specified to run a calculation:

| Parameter| Symbol | Type | Comments
|--|--| --| --|
| Number of layers | $M$ | Python `int` | - 
|Refractive index of the outer medium|$n_\text{outer}$| `RefractiveIndex` | -
|Refractive index of the substrate|$n_\text{substrate}$| `RefractiveIndex` | -
|Incident angle in the outer medium|$\theta_\text{outer}$| Python `float` | <ul><li>Unit: radian</li><li>Valid input range is $(-\frac{\pi}{2}, \frac{\pi}{2})$. The angles $-\frac{\pi}{2}$ and $\frac{\pi}{2}$ (or angles very close to these, within floating point precision) are not valid input as they cause the calculation of the reflected amplitude to fail (specifically, the matrix inversion)</li></ul>
|Wavelengths | $\lbrace \lambda_1, \lambda_2,\dots, \lambda_K\rbrace$  | One of <ul><li>`tuple[float, float]`</li><li>`numpy` array</li></ul> | <ul><li>Unit: meter</li><li>These are the wavelengths in vacuum.</li><li>If specified as a `tuple`, the input is parsed as the lower and upper limits of an interval, $(\lambda_\text{min}, \lambda_\text{max})$, and the wavelengths $\lbrace\lambda_1, \lambda_2,\dots, \lambda_K\rbrace$ are automatically chosen.</li><li>If specified as a `numpy` array, these are understood to be the wavelengths $\lbrace\lambda_1, \lambda_2,\dots, \lambda_K\rbrace$ themselves.</li></ul>
| Specification for the constraints on the refractive indices of each layer | - | (see below) | (see below)
| Specification for the constraints on the thicknesses of each layer | - | (see below) | Unit: meter
| Specification for the merit function | - | (see below) | (see below)

<!-- - the number of layers, $M$, which is a Python `int`
- the refractive index of the outer medium, $n_\text{outer}$
- the refractive index of the substrate, $n_\text{substrate}$
- the incident angle in the outer medium, $\theta_\text{outer}\in (-\frac{\pi}{2}, \frac{\pi}{2})$, which is a Python `float`. The angles $-\frac{\pi}{2}$ and $-\frac{\pi}{2}$ (or angles very close to these, within floating point precision) are not valid input as they cause the calculation of the reflected amplitude to fail (specifically, the matrix inversion).
- the wavelengths at which the merit function is evaluated, in the form of
    - the lower and upper limits of an interval, $(\lambda_\text{min}, \lambda_\text{max})$
    - a `numpy` array of wavelengths
- a specification for the constraints on the refractive indices of each layer (see below)
- a specification for the constraints on the thicknesses of each layer (see below)
- a specification for the merit function (see below) -->


### Specification for the layer refractive indices

A multilayer coating consists of $M$ layers. The refractive index function, $n_i$, of layer $i$ can either be
1. fixed, i.e. the optimiser does not have to choose the refractive index function for the layer, or
2. unfixed, in which case one must specify a list of refractive index functions from which the optimiser must choose the best, i.e. it is a categorical variable.

Therefore, each layer is specified by a tuple `(string, value)`, in which `string` can take be either 
1. `'fixed'` or
2. `'categorical'`

and `value` can be either
1. a value of type `RefractiveIndex` or
2. a list of refractive indices, i.e. of type `list[RefractiveIndex]`.

The constraint on each layer is put together into a tuple, which comprises the constraint on the entire multilayer coating. The type of the refractive index specification is thus, in Python,
```python
tuple[tuple[str, Union[RefractiveIndex, list[RefractiveIndex]]].
```

For example, consider the following coating with 5 layers: 
|Layer| Constraint type| Value(s)
|--|--|--
|1|fixed|$n_1=1.7$
|2|categorical| $n_2\in \lbrace0.3, 2.4\rbrace$
|3|fixed|$n_3=0.9$
|4|fixed |$n_4=0.6$
|5|categorical |$n_5\in\lbrace1.3, 5.6, 3.8\rbrace$

For simplicity, the refractive indices here are chosen to be constant over all wavelengths, i.e. the actual value which must be provided for layer 1 is
```python
import numpy as np
def n_1(wavelengths: np.ndarray) -> np.ndarray:
    return np.full(shape=len(wavelengths), fill_value=1.7)
```

This can equivalently be done with the function `Utils.constant()` which creates the above function from the value $1.7$:
```python
from calculation.Utils import Utils
n_1 = Utils.constant(1.7)
```

The specification thus takes the form
```python
n_specification = (
    ('fixed', Utils.constant(1.7)), 
    ('categorical', [Utils.constant(0.3), Utils.constant(2.4)]), 
    ('fixed', Utils.constant(0.9)), 
    ('fixed', Utils.constant(0.6)), 
    ('categorical', [Utils.constant(1.3), Utils.constant(5.6), Utils.constant(3.8)])
)
```

### Specification for the layer thicknesses

The constraints for the layer thicknesses are specified similarly to the constraints for the refractive indices.

The thickness of each layer can be either
1. fixed, or
2. unfixed, in which case the optimiser chooses a value in the interval $[d_\text{min}, d_\text{max}]$, i.e. the variable is bounded.

Each layer is specified by a tuple `(string, value)`, which can be
1. `('fixed', a value of type float)`
2. `('bounded', a tuple of two floats)`

The type of the thickness specification is thus, in Python,
```python
tuple[tuple[str, Union[float, tuple[float, float]]], ...].
```

For example for a 5-layer coating,
|Layer| Constraint type| Value(s)
|--|--|--
|1|fixed|$d_1=100\mathrm{nm}$
|2|fixed| $d_2=200\mathrm{nm}$
|3|fixed|$d_3=70\mathrm{nm}$
|4|bounded |$d_4\in[0\mathrm{nm}, 400\mathrm{nm}]$
|5|bounded |$d_5\in[0\mathrm{nm}, 100\mathrm{nm}]$

the thickness specification takes the form
```python
d_specification = (
    ('fixed', 100e-9), 
    ('fixed', 200e-9), 
    ('fixed', 70e-9), 
    ('bounded', (0., 400e-9)), 
    ('bounded', (0., 100e-9))
)
```

### Specification for the merit function

The merit function $f$ is the function to be minimised. It is fully specified by the following parameters:

<!-- The parameter values which minimise the merit function determine the coating which fulfills the design specifications as closely as possible. The merit function depends on the unfixed parameters $\mathbf{p}$, which consist of $0\le p \le M$ unfixed refractive indices and $0\le q\le M$ unfixed thicknesses and has terms listed in the following table: -->

| Name | Symbol | Type | Default value (for all wavelengths)
|-|-|-|-|
|`s_pol_weighting`| $w_{R_s}$ | `float` | - 
|`p_pol_weighting`| $w_{R_p}$ | `float` | - 
|`sum_weighting`| $w_{S}$ | `float` | - 
|`difference_weighting`| $w_{D}$ | `float` | - 
|`phase_weighting`| $w_{\phi_{sp}}$ | `float` | - 
|`target_reflectivity_s`| $\tilde{R}_s$ | `Callable[[np.ndarray], np.ndarray]` | 0
|`target_reflectivity_p`| $\tilde{R}_p$ | `Callable[[np.ndarray], np.ndarray]` | 0
|`target_sum`| $\tilde{S}$ | `Callable[[np.ndarray], np.ndarray]` | 0
|`target_difference`| $\tilde{D}$ | `Callable[[np.ndarray], np.ndarray]` | 0
|`target_relative_phase`| $\tilde{\phi}_{sp}$ | `Callable[[np.ndarray], np.ndarray]` | 0
|`weight_function_s`| $\delta R_s$ | `Callable[[np.ndarray], np.ndarray]` | 1
|`weight_function_p`|$\delta R_p$  | `Callable[[np.ndarray], np.ndarray]` | 1
|`weight_function_sum`| $\delta S$ | `Callable[[np.ndarray], np.ndarray]` | 1
|`weight_function_difference`| $\delta D$ | `Callable[[np.ndarray], np.ndarray]` | 1
|`weight_function_phase`| $\delta \phi_{sp}$ | `Callable[[np.ndarray], np.ndarray]` | 1

The default value is chosen such that, if the parameter is not specified, the corresponding term in the merit function will be zero, i.e. the term has no effect on the calculations.

Note that some variables are functions of wavelength.

For example, to specify that the 5-layer coating should be an antireflection coating for p-polarised light, that the corresponding term in the merit function should have a weighting of 10,000 and that all other terms should be switched off,
```python
p_pol_weighting       = 10000
s_pol_weighting       = 0
sum_weighting         = 0
difference_weighting  = 0
phase_weighting       = 0
target_reflectivity_p = lambda wavelengths: np.zeros(len(wavelengths))
# Leave all other variables to their default values.
```

Since the weighting term is non-zero only for `p_pol_weighting`, the merit function contains only this term, while the other terms are switched off.

### Summary

Finally, a name for the project must be chosen (`project_name`), which determines the name of the directory under which all output files will be stored.

The following is a full minimal example specification:
```python
M = 5
n_specification = (
    ('fixed', Utils.constant(1.7)), 
    ('categorical', [Utils.constant(0.3), Utils.constant(2.4)]),
    ('fixed', Utils.constant(0.9)), 
    ('fixed', Utils.constant(0.6)),
    ('categorical', [Utils.constant(1.3), Utils.constant(5.6), Utils.constant(3.8)])
)
d_specification = (
    ('fixed', 100e-9), 
    ('fixed', 200e-9), 
    ('fixed', 70e-9), 
    ('bounded', (0., 400e-9)), 
    ('bounded', (0., 100e-9))
)
wavelength_specification = (600e-9, 2300e-9)

kwargs = dict(
    project_name='test_project',
    M=M,
    n_outer=Utils.constant(1.00),
    n_substrate=Utils.constant(1.50),
    theta_outer=np.pi / 3,
    wavelengths=wavelength_specification,
    n_specification=n_specification,
    d_specification=d_specification,
    p_pol_weighting=10000,
    s_pol_weighting=0,
    sum_weighting=0,
    difference_weighting=0,
    phase_weighting=0,
    target_reflectivity_p = lambda wavelengths: np.zeros(len(wavelengths))
)
```

## Running the calculation

The `dict` is passed to the `Runner` class and the calculation is started like this:
```python
from calculation.Runner import Runner
runner = Runner(**kwargs)
runner.run()
```

The above example can be found in the file [`main.py`](https://github.com/zwei-beiner/Code/blob/master/src/main.py) which can be run with MPI on 4 cores like this:
```shell
mpirun -n 4 python main.py
```

## `Utils.multilayer_specification`

For larger multilayer stacks, the function `Utils.multilayer_specification(M, string, [pattern])` repeats a pattern `M` times. For example, to create a multilayer stack with 5 layers with alternating fixed refractive indices, one can write
```python
Utils.multilayer_specification(5, 'fixed', [Utils.constant(2.), Utils.constant(3.)])
```
giving 
```python
(('fixed', Utils.constant(2.)), ('fixed', Utils.constant(3.)), ('fixed', Utils.constant(2.)), ('fixed', Utils.constant(3.)), ('fixed', Utils.constant(2.)))
```

## Example 2: Optimising a coating on separated wavelength ranges

Consider the following design problem:
|Parameter| Value |
|--|--|
| $M$ | $20$ 
| $n_\text{outer}$| Air, $n_\text{outer}=1$ 
|$n_\text{substrate}$| Fused silica 
| Refractive index specification | Any material in the `Materials` data base which does not have an absorption peak in the wavelength range
| Thickness specification | The thickness of each layer is bounded between $0\mathrm{nm}$ and $350\mathrm{nm}$ 
| $\theta_\text{outer}$ | $15^\circ$
|Target reflectivity | Dichroic with <ul><li>Reflection band (target reflectivity $1$): <ul><li>$500\mathrm{nm}$ - $900\mathrm{nm}$</li></ul></li><li>Transmission bands (target reflectivity $0$): <ul><li>$1050\mathrm{nm}$ - $1300\mathrm{nm}$</li><li>$1450\mathrm{nm}$ - $1800\mathrm{nm}$</li><li>$1950\mathrm{nm}$ - $2350\mathrm{nm}$</li></ul></li><li>Reflection between bands is unconstrained</li></ul> 

The `main.py` file is:

```python 
import numpy as np

# Compile Cython files.
from calculation.Compiler import compile_cython_files
compile_cython_files()


# Code which uses the Cython numerical routines can only be imported after Cython files have been compiled.
from calculation.Utils import Utils
from calculation.RefractiveIndices import Materials
from calculation.Runner import Runner

M = 20
# Don't include materials which have absorption peak in the wavelength range.
# Repeat the selection of materials M times.
n_specification = Utils.multilayer_specification(
    M, 'categorical', [[
        Materials.Nb2O5,
        Materials.SiO2,
        Materials.MgF2,
        Materials.Ta2O5,
        Materials.Ag,
        Materials.Au,
        Materials.BK7,
        Materials.Sapphire_ordinary_wave,
        Materials.Cr,
        Materials.GaAs,
        Materials.Si,
        Materials.Si3N4,
        Materials.TiO2
]])

# Repeat ('bounded', (0., 350e-9)) M times.
d_specification = Utils.multilayer_specification(M, 'bounded', [(0., 350e-9)])

# Specify the wavelengths at which the merit function is evaluated manually.
wavelength_specification = np.concatenate([
    np.linspace(500e-9, 900e-9, num=25),
    np.linspace(1050e-9, 1300e-9, num=25),
    np.linspace(1450e-9, 1800e-9, num=25),
    np.linspace(1950e-9, 2350e-9, num=25)
])

# Function which is 1. below 900nm and 0 above.
target_reflectivity = lambda wl: np.where(np.asarray(wl) < 900e-9, 1., 0.)

# Incident angle in radians.
theta_outer = 15. * np.pi / 180.

# Substrate is SiO2 (Fused Silica).
n_substrate = Materials.SiO2
# Outer medium is air (n = 1.00).
n_outer = Utils.constant(1.00)

# Collect all parameters.
kwargs = dict(project_name='MROI_dichroic',
    M=M,
    n_outer=n_outer,
    n_substrate=n_substrate,
    theta_outer=theta_outer,
    wavelengths=wavelength_specification,
    n_specification=n_specification,
    d_specification=d_specification,
    # Separately optimise p- and s-polarised reflectivities.
    p_pol_weighting=1000,
    s_pol_weighting=1000,
    # Switch off the other terms by setting their weightings to zero.
    sum_weighting=0,
    difference_weighting=0,
    phase_weighting=0,
    # Pass in the target reflectivity.
    target_reflectivity_s=target_reflectivity,
    target_reflectivity_p=target_reflectivity
)

# Run the calculation.
runner = Runner(**kwargs)
runner.run()
```

## Output files

For any run, the code generates the following output:

| Output file/directory | Type | Description |
|--|--|--
| `critical_thicknesses_plot.pdf` | File | For each layer, a plot of <ul><li>the thickness $d_\mathrm{optimal}$ found in the global optimisation (orange line)</li><li>the critical thickness $d_\text{critical}$ (blue line)</li></ul> as functions of wavelength are shown. </br>If the orange line lies below the blue line for all wavelengths, then the layer can be regarded as negligibly thin and taken out of the multilayer stack. This is done automatically by the code.
| `marginal_distributions_plot.pdf` | File |The distribution of the parameter values of the dead points are shown.</br> Ideally, as PolyChord converges to the global minimum of the merit function, the distributions should be unimodal and sharply peaked.
| `merit_function_plot.pdf` | File |For each iteration of PolyChord, the largest merit function value of the current set of live points is plotted. </br>Ideally, the merit function should initially decrease rapidly, followed by gradual convergence as the parameter space is compressed to the neighbourhood of the global minimum of the merit function.
|`optimal_merit_function_value.txt`| File| The merit function value after the local optimisation has completed.
| `optimal_parameters.csv` | File| The optimal solution to the design problem. <ul><li>Each entry in the column `n` is an integer which corresponds to the position of the refractive index function in the list of refractive indices provided in the refractive index specification.</li><li>The column `d(nm)` contains the thicknesses of each layer in nanometers.</li><li>The column `Remove` is a boolean which, if `True`, indicates that the layer can be removed. If any of the entries are `True`, the code automatically removes the layer(s) and reruns the calculation with the reduced stack.</li></ul>
| `reflectivity_plot.pdf` | File | <ul><li>Shows plots of $R_s$ and $R_p$ against wavelength as red lines. The shaded red regions indicate the error bands (68% central credibility intervals, calculated by sampling from a Gaussian centred at the optimal thicknesses with a standard deviation of $1\mathrm{nm}$ in all directions).</li><li>The provided target reflectivities ($\tilde R_s$ or $\tilde R_p$) are shown as blue lines.</li></ul>
| `sum_difference_phase_plot.pdf` | File | <ul><li>Similarly to `reflectivity_plot.pdf`, plots of $S$, $D$ and $\phi_{sp}$ are shown.</li><li>The phase difference is unwrapped using `numpy.unwrap()` so that it is a continuous function.</li></ul>
| `local_minima` | Directory | Contains all local minima found by the clustering algorithm in post-processing of PolyChord samples. </br>Each local minimum has its own subdirectory, e.g. `local_minimum_0`, and they are sorted from best to worst, i.e. `local_minimum_0` is the global optimum and `local_minimum_1` is the next best optimum, etc. </br>Each subdirectory contains its own <ul><li>`critical_thicknesses_plot.pdf`</li><li>`optimal_merit_function_value.txt`</li><li>`optimal_merit_function_value.txt`</li><li>`reflectivity_plot.pdf`</li> <li>`sum_difference_phase_plot.pdf`</li></ul>
| `polychord_output` | Directory | Raw output of PolyChord. To read from this directory, it is advised to use the `anesthetic` Python package.