#!/usr/bin/env python
import os
import warnings
from glob import glob
from setuptools import setup, find_packages, Extension

try:
  from Cython.Build import cythonize
  from Cython.Tempita import Template
except ImportError:
  use_cython = False
  warnings.warn('Cython not detected. Re-run setup.py after it is installed.')
else:
  use_cython = True


setup_kwargs = dict(
    name='superman',
    version='0.1.2',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Spectrum preprocessing machine.',
    url='https://github.com/all-umass/superman',
    license='MIT',
    packages=find_packages(exclude=['test', '*.test', '*.test.*']),
    package_data=dict(superman=['dana_numbers.txt']),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn >= 0.15',
        'matplotlib >= 1.3.1',
        'construct == 2.8.*',
        'Cython >= 0.20',
        'six >= 1.10.0',
    ],
    scripts=glob('scripts/*.py'),
)

if use_cython:
  # manually regenerate the .pyx file, because cythonize() can't handle it.
  pyx_file = 'superman/distance/_traj.pyx'
  tpl_file = pyx_file + '.in'
  refresh_pyx = (not os.path.exists(pyx_file) or
                 os.path.getmtime(tpl_file) > os.path.getmtime(pyx_file))
  if refresh_pyx:
    tpl = Template.from_filename(tpl_file, encoding='utf-8')
    with open(pyx_file, 'w') as fh:
      fh.write(tpl.substitute())

  extra_args = [
      '-Ofast', '-march=native', '-ffast-math', '-Wno-unused-function',
      '-Wno-unreachable-code'
  ]
  exts = [
      Extension('*', ['superman/distance/_pdist.pyx'],
                extra_compile_args=['-fopenmp'] + extra_args,
                extra_link_args=['-Ofast', '-fopenmp', '-march=native']),
      Extension('*', [pyx_file], extra_compile_args=extra_args),
      Extension('*', ['superman/distance/common.pyx'],
                extra_compile_args=extra_args),
  ]
  setup_kwargs['ext_modules'] = cythonize(exts)


setup(**setup_kwargs)
