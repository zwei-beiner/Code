### Todo list
- Fix the phase difference error bars: Some error bars seem to be shifted by $2\pi$
- Show the selected wavelengths in the reflectivity plots
- Increase number of selected wavelengths because the spacing is too coarse at larger wavelengths
- In addition to layers being taken out, merge layers with the same materials. However, this might require to change the constraints on the thicknesses if the thickness of the new layer is larger than the maximum allowed thickness.
- Detect of likelihood has converged to that a run can be terminated automatically, instead of relying on the empirical factor of 5 for the number of iterations
- Make a Python package which can be installed with pip:
  - Problem is described here: https://stackoverflow.com/questions/73766918/creating-python-package-which-uses-cython-valueerror
  - Alternatively, look at package which uses Cython (e.g. HDBSCAN on github) which is pip-installable and use that as a basis to get a working package with pip 