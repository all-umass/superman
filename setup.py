#!/usr/bin/env python
from glob import glob
from setuptools import setup, find_packages, Extension

try:
  from Cython.Build import cythonize
  import numpy as np
except ImportError:
  use_cython = False
else:
  use_cython = True

setup_kwargs = dict(
    name='superman',
    version='0.0.1',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Spectrum preprocessing machine.',
    url='https://github.com/all-umass/superman',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn >= 0.15',
        'matplotlib >= 1.3.1',
        'construct >= 2.5.2',
        'Cython >= 0.20',
        'viztricks >= 0.1',
    ],
    scripts=glob('scripts/*.py'),
)

if use_cython:
  exts = [
      Extension('*', ['superman/*.pyx'], include_dirs=[np.get_include()],
                extra_compile_args=['-Ofast', '-fopenmp', '-march=native',
                                    '-Wno-unused-function'],
                extra_link_args=['-Ofast', '-fopenmp', '-march=native']),
  ]
  setup_kwargs['ext_modules'] = cythonize(exts)


setup(**setup_kwargs)
