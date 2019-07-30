/******************************************************************************
 * Python extension module for calculating Lyman-alpha optical depths.       
 * 
 * Function calc_optdepth() takes arguments:
 *     n_HI in [m^-3 (proper)], sightline neutral hydrogen density,
 *     v_Hub in [km/s], sightline Hubble velocity,
 *     v_pec in [km/s], sightline gas peculiar velocity,
 *     b in [m/s], sightline Doppler parameter for Voigt profile,
 *     dx in [m (proper)], sightline spatial bin width(s),
 *     nlos, number of sightlines,
 *     nbins, number of bins in each sightline,
 *     nthreads, number of OpenMP threads to use in calculation,
 *
 * where n_HI, v_Hub, v_pec, b and dx are 1D NumPy arrays (of dtype np.float64).
 * nlos, nbins and nthreads are Python integers.
 *
 * It returns an array of Lyman-alpha optical depths.                                                       
 *
 * LHW 04/12/17
 * updated 26/04/17
 *****************************************************************************/


#include <math.h>
#include <omp.h>

#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

/* Constants for Voigt profile                                               */ 
#define  PI    3.14159265358979323846   /* Dimensionless                     */
#define  LAMBDA_LYA_H1  1.2156701e-7    /* Lya wavelength in m               */
#define  FOSC_LYA       0.416400        /* Dimensionless                     */
#define  ECHARGE        1.60217662e-19  /* Coulombs                          */       
#define  EMASS          9.10938356e-31  /* kg                                */
#define  C              2.99792458e8    /* m/s                               */
#define  EPSILON0       8.85418782e-12  /* m^-3 kg^-1 s^4 A^2                */
#define  GAMMA_LYA_H1   6.265e8         /* Lya decay rate in s^-1            */

/* Prototypes */
void sighandler(int sig);
double voigt_profile (double vdiff, double b);
static PyObject* calc_optdepth(PyObject *self, PyObject *args);


/*---------------------------------------------------------------------------*/
/* MODULE FUNCTIONS ---------------------------------------------------------*/


/* Signal handler to allow interrupts when extension in use */
void sighandler(int sig)
{
    fprintf(stderr,"\nSignal = %s (SIGID = %d). \n",strsignal(sig), sig);
    exit(sig);
}

/* Voigt profile function */
double voigt_profile (double vdiff, double b)
{
    /* Takes relative velocities in [km/s] and b in [m/s] */
    /* Returns profile in [m^2 s]                         */

    double sigma_av_Lya, k2, k3, a;
    double T0, T1, T2;
    double numer, subfrac, profile;

    /* Convert units */
    vdiff *= 1e3;      /* km/s to m/s */
    
    /* Average cross section */
    sigma_av_Lya = FOSC_LYA*ECHARGE*ECHARGE/(4*EMASS*C*EPSILON0); /* m^2 */

    /* Profile factors */
    k2 = GAMMA_LYA_H1*LAMBDA_LYA_H1/(4.0*PI);              /* m/s */
    k3 = sigma_av_Lya*LAMBDA_LYA_H1/(sqrt(PI)*b);          /* m^2 s */
    a = k2/b;                                              /* dimensionless */

    T0 = (vdiff/b) * (vdiff/b);
    T1 = exp(-T0);
    T2 = 1.5/T0;

    numer = ( T1*T1*(4.0*T0*T0 + 7.0*T0 + 4.0 + T2) - T2 - 1.0 );
    subfrac = a/sqrt(PI)/T0*numer;

    profile = (T0 < 1.0e-6) ? T1 : T1 - subfrac;

    return k3*profile;   /* m^2 s */
}

/* Python callable function */
static PyObject* calc_optdepth(PyObject *self, PyObject *args)
{
    PyArrayObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL, *arr5=NULL;
    PyObject *val1=NULL, *val2=NULL, *val3=NULL;
    PyArrayObject *out_arr=NULL;

    int ndims, i;
    long nlos, nbins, nthreads;
    npy_float64 *n_HI, *v_Hub, *v_pec, *b, *dx;
    npy_float64 *tau_lya=NULL, *out_ptr;
    
    /* Signal handler to allow interrupts */
    signal(SIGINT,sighandler);


/******************************************************************************/    
/* Parse arguments and copy Python variables into C variables */

    printf("\nOTCALC running...\n");

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!", &PyArray_Type, &arr1, 
                                                    &PyArray_Type, &arr2, 
                                                    &PyArray_Type, &arr3, 
                                                    &PyArray_Type, &arr4, 
                                                    &PyArray_Type, &arr5,
                                                    &PyLong_Type,  &val1,
                                                    &PyLong_Type,  &val2,
                                                    &PyLong_Type,  &val3))
    {
        return NULL;
    }

    /* Array dimensions */
    ndims = PyArray_DIM(arr1,0);
    if (PyArray_DIM(arr2,0) != ndims || PyArray_DIM(arr3,0) != ndims ||
        PyArray_DIM(arr4,0) != ndims || PyArray_DIM(arr5,0) != ndims)
    { 
        PyErr_SetString(PyExc_ValueError, "Input array dimensions don't match");
        return NULL; 
    }

    nlos = PyLong_AsLong(val1);
    nbins = PyLong_AsLong(val2);
    if (nbins*nlos != ndims)
    { 
        PyErr_SetString(PyExc_ValueError, "ndims != nlos*nbins");
        return NULL; 
    }
    printf("Loaded nbins = %ld, nlos = %ld\n",nbins,nlos);

    nthreads = PyLong_AsLong(val3);
    omp_set_num_threads(nthreads);
    printf("Using max. of %d threads for calculation\n", omp_get_max_threads());

    /* Allocate memory for C arrays */
    n_HI = (npy_float64 *)malloc(sizeof(npy_float64)*ndims);
    if (n_HI==NULL) { free(n_HI); printf("malloc error!\n"); exit(0); }
    v_Hub = (npy_float64 *)malloc(sizeof(npy_float64)*ndims);
    if (v_Hub==NULL) { free(v_Hub); printf("malloc error!\n"); exit(0); }
    v_pec = (npy_float64 *)malloc(sizeof(npy_float64)*ndims);
    if (v_pec==NULL) { free(v_pec); printf("malloc error!\n"); exit(0); }
    b = (npy_float64 *)malloc(sizeof(npy_float64)*ndims);
    if (b==NULL) { free(b); printf("malloc error!\n"); exit(0); }
    dx = (npy_float64 *)malloc(sizeof(npy_float64)*ndims);
    if (dx==NULL) { free(dx); printf("malloc error!\n"); exit(0); }

    /* Copy Python arrays into C arrays */
    for(i=0; i<ndims; i++)
    {
        n_HI[i]  = *(npy_float64 *)PyArray_GETPTR1(arr1, i);
        v_Hub[i] = *(npy_float64 *)PyArray_GETPTR1(arr2, i);
        v_pec[i] = *(npy_float64 *)PyArray_GETPTR1(arr3, i);
        b[i]     = *(npy_float64 *)PyArray_GETPTR1(arr4, i);
        dx[i]    = *(npy_float64 *)PyArray_GETPTR1(arr5, i);
    }

    /* Sanity prints */
    printf("\nSome sanity prints [proper units]:\n");
    printf("n_HI[0] = %lf [m^-3]\n", n_HI[0]);
    printf("v_Hub[0] = %lf [km/s]\n", v_Hub[0]);
    printf("v_pec[0] = %lf [km/s]\n", v_pec[0]);
    printf("b[0] = %lf [m/s]\n", b[0]);
    printf("dx[0] = %lf [m]\n\n", dx[0]);
    
    
/******************************************************************************/
/* Optical depth calculation */
 
    tau_lya = (npy_float64 *)calloc(ndims,sizeof(npy_float64));
    if (tau_lya==NULL) { free(tau_lya); printf("calloc error!\n"); exit(0); }

    #pragma omp parallel
    {
        /* private (thread-local) variables */
        int  j, k, inj, ink;
        npy_float64 v_rel, vp;

        #pragma omp for private(i)
        for (i=0; i<nlos; i++)
        {
            /* index ordering is (nbins*i + j) */
            for (j=0; j<nbins; j++)
            {
                inj = i*nbins + j;
                
                for (k=0; k<nbins; k++)
                {
                    ink = i*nbins + k;
                    v_rel = v_Hub[inj] - v_Hub[ink] - v_pec[ink];  /* relative velocity in km/s */
                    vp = voigt_profile(v_rel, b[ink]);
                    
                    tau_lya[inj] += vp*n_HI[ink]*dx[ink];
                    /* UNITS: all proper with [vp] = m^2, [n_HI] = m^-3, [dx] = m  */
                }
            }
        }
    }

    /* Create output array */
    out_arr = PyArray_FromDims(1, &ndims, NPY_DOUBLE);

    /* Copy optical depths into output array */
    for (i=0; i<ndims; i++)
    {
        out_ptr = (npy_float64 *)PyArray_GETPTR1(out_arr, i);
        *out_ptr = tau_lya[i];
    }
    
    /* Deallocate memory for C arrays */
    free(n_HI);
    free(v_Hub);
    free(v_pec);
    free(b);
    free(dx);
    free(tau_lya);

    return PyArray_Return(out_arr);
}


/*---------------------------------------------------------------------------*/
/* MODULE INITIALIZATION ----------------------------------------------------*/

/* Note module designed for use with Python 3! */

static PyMethodDef otcalc_methods[] = {
    {"calc_optdepth", calc_optdepth, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}         /* Sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "otcalc",
    NULL,
    -1,
    otcalc_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC *PyInit_otcalc(void)
{
   PyObject *m;

   m = PyModule_Create(&moduledef);
   if (m == NULL) { return NULL; }

   import_array();

   return m;

}
