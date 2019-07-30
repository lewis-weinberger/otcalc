"""
    setup.py file for otcalc.c

    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.

    LHW 04/12/17
    updated 26/04/18
"""

from distutils.core import setup, Extension
import numpy
import os

os.environ["CC"] = "gcc"

ext_module = [ Extension('otcalc', 
                         sources=['otcalc.c'],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-lgomp'],
                         include_dirs = [numpy.get_include()]) ]

setup(name = 'otcalc',
      version='1.0',
      description='Optical depths modules',
      ext_modules = ext_module) 
