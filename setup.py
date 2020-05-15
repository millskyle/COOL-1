from distutils.core import setup, Extension
import os


tclap_include=os.environ['HOME'] + "/anaconda3/pkgs/tclap-1.2.1-h470a237_1/include/"
boost_lib=os.environ['HOME'] + "/anaconda3/pkgs/libboost-1.67.0-h46d08c1_4/lib"
boost_include=os.environ['HOME'] + "/anaconda3/pkgs/libboost-1.67.0-h46d08c1_4/include"

import numpy as np
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


extra_compile_args = ["-Wall","-ansi","-pedantic","-std=c++11","-O3","-funroll-loops","-pipe"]
extra_compile_args += ["-fopenmp"]

sa_module = Extension('_sa_interface',
                      language='c++',
                      include_dirs=[tclap_include,boost_include, '/usr/local/opt/llvm/include',numpy_include],
                      library_dirs=[boost_lib,'/usr/local/opt/llvm/lib'],
                      sources=['sagym/interface/sa_interface_wrap.cxx', 
                        'sagym/interface/sa.cc'],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=['-lgomp'],
                     )


setup(name='sagym',
      py_modules=['sagym',],
      language='C++',
      ext_modules=[sa_module],
      install_requires=['numpy','gym'],
      description=['Gym wrapper for spin-lattice simulated annealing'],
      author='Pooya Ronagh, Kyle Mills',
      version='1.0'
     )

