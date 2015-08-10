#!/usr/bin/env python
from glob import glob
from setuptools import setup, find_packages

setup(
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
