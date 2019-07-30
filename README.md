### otcalc
**otcalc** is a C-extension for Python 3 and NumPy, providing a parallelised lyman-alpha absorption spectra calculation. See Eq. (15) of [Bolton & Haehnelt (2007)](https://doi.org/10.1111/j.1365-2966.2006.11176.x).

#### Installation
Requires an OpenMP compatible C compiler (chosen in `setup.py`). Compile the library using:

    $ python3 setup.py build_ext --inplace

This will create the library in the working directory.

#### Usage
The module provides a single function:

    #!/usr/bin/env python3
    from otcalc import calc_optdepth

The function takes the following arguments:

    tau = calc_optdepth(n_HI, v_Hub, v_pec, b, dx, nlos, nbins, nthreads)

where the parameters are:

    n_HI (1D NumPy array, float64) -- neutral hydrogen density along sightline in proper [m^-3]
    v_Hub (1D NumPy array, float64) -- Hubble velocity along sightline in [km/s]
    v_pec (1D NumPy array, float64) -- gas peculiar velocity along sightline in [km/s]
    b (1D NumPy array, float64) -- Doppler parameter along sightline in [m/s]
    dx (1D NumPy array, float64) -- sightline cell widths in proper [m]
    nlos (int32) -- number of sightlines
    nbins (int32) -- number of cells in a sightline
    nthreads (int32) -- number of (OpenMP) threads to use for calculation

and the returned object is a 1D NumPy array, `tau`, containing the optical depth in each pixel of the sightline. Note the 1D sightline arrays are contiguous flattened versions of the 2D space of sightlines and cells, indexed in the following way:

```
for i in range(nlos):
    for j in range(nbins):
        array_ij = array[i*nbins + j]
```

hence the arrays should be `nbins*nlos` long.

The calculation is multi-threaded (using OpenMP), with the number of threads controlled by the `nthreads` argument.