#!/usr/bin/env python
import os
from glob import glob
from setuptools import setup, find_packages, Extension

try:
  from Cython.Build import cythonize
  from Cython.Tempita import Template
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
  # manually regenerate the .pyx file, because cythonize() can't handle it.
  pyx_file = 'superman/traj/fast_lcss.pyx'
  tpl_file = pyx_file + '.in'
  refresh_pyx = (not os.path.exists(pyx_file) or
                 os.path.getmtime(tpl_file) > os.path.getmtime(pyx_file))
  if refresh_pyx:
    tpl = Template.from_filename(tpl_file, encoding='utf-8')
    with open(pyx_file, 'w') as fh:
      fh.write(tpl.substitute())

  exts = [
      Extension('*', ['superman/_pdist.pyx'],
                extra_compile_args=['-Ofast', '-fopenmp', '-march=native',
                                    '-Wno-unused-function'],
                extra_link_args=['-Ofast', '-fopenmp', '-march=native']),
      Extension('*', [pyx_file],
                extra_compile_args=['-O3', '-march=native', '-ffast-math',
                                    '-Wno-unused-function']),
  ]
  setup_kwargs['ext_modules'] = cythonize(exts)


setup(**setup_kwargs)
