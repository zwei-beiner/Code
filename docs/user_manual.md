# User manual

This manual provides instructions on how to set up a multilayer coating calculation using the `coating` code. This concerns the creation of the `main.py` file in the [Instructions for the CSD3 cluster](https://github.com/zwei-beiner/Code/blob/master/docs/supercomputer_instructions.md).

## Parameters for the calculation

All refractive indices are functions of wavelength and must be evaluatable at any wavelength specified by the input wavelengths. This means that they must take a `numpy` array as an input and return a `numpy` array, i.e. of type `Callable[[numpy.ndarray], np.ndarray]`, which defined to be the type `RefractiveIndex`.

The following parameters must be specified to run a calculation:

| Parameter| Symbol | Type | Comments
|--|--| --| --|
| Number of layers | $M$ | Python `int` | - 
|Refractive index of the outer medium|$n_\text{outer}$| `RefractiveIndex` | -
|Refractive index of the substrate|$n_\text{substrate}$| `RefractiveIndex` | -
|Incident angle in the outer medium|$\theta_\text{outer}$| Python `float` | <ul><li>Unit: $\mathrm{radian}$</li><li>Valid input range is $(-\frac{\pi}{2}, \frac{\pi}{2})$. The angles $-\frac{\pi}{2}$ and $\frac{\pi}{2}$ (or angles very close to these, within floating point precision) are not valid input as they cause the calculation of the reflected amplitude to fail (specifically, the matrix inversion)</li></ul>
|Wavelengths | $\lbrace \lambda_1, \lambda_2,\dots, \lambda_K\rbrace$  | One of <ul><li>`tuple[float, float]`</li><li>`numpy` array</li></ul> | <ul><li>Unit: $\mathrm{meter}$.</li><li>These are the wavelengths in the outer medium. They are related to the wavelength in vacuum by $\lambda_\text{vac}=n_\text{outer}(\lambda)\lambda$.</li><li>If specified as a `tuple`, the input is parsed as the lower and upper limits of an interval, $(\lambda_\text{min}, \lambda_\text{max})$, and the wavelengths $\lbrace\lambda_1, \lambda_2,\dots, \lambda_K\rbrace$ are automatically chosen.</li><li>If specified as a `numpy` array, these are understood to be the wavelengths $\lbrace\lambda_1, \lambda_2,\dots, \lambda_K\rbrace$ themselves.</li></ul>
| Specification for the constraints on the refractive indices of each layer | - | (see below) | (see below)
| Specification for the constraints on the thicknesses of each layer | - | (see below) | Unit: $\mathrm{meter}$
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


## Specification for the layer refractive indices

A multilayer coating consists of $M$ layers. The refractive index function, $n_i$, of layer $i$ can either be
1. fixed, i.e. the optimiser does not have to choose the refractive index function for the layer, or
1. unfixed, in which case one must specify a list of refractive index functions from which the optimiser must choose the best, i.e. it is a categorical variable.

Therefore, each layer is specified by a tuple `(string, value)`, in which `string` can take be either 
1. `'fixed'` or
1. `'categorical'`

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
    ('fixed', Utils.constant(0.9)), ('fixed', Utils.constant(0.6)), 
    ('categorical', [Utils.constant(1.3), Utils.constant(5.6), Utils.constant(3.8)])
)
```

## Specification for the layer thicknesses

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

For the example of the 5-layer coating,
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

## Specification for the merit function

The merit function $f$ is the function to be minimised. The parameter values which minimise the merit function determine the coating which fulfills the design specifications as closely as possible.

The merit function depends on the unfixed parameters $\mathbf{p}$, which consist of $p$ unfixed refractive indices and $q$ unfixed thicknesses.