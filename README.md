# Superman

*SpectrUm PrEpRocessin MAchiNe*

T. Boucher, C. Carey, S. Giguere, D. Dyar, S. Mahadevan

### Installation

Superman must be built from source, so make sure you have a working
C compiler, Python headers, and OpenMP before proceeding.

    pip install Cython
    pip install superman

If you're contributing to superman:

    git clone git@github.com:all-umass/superman.git
    cd superman
    pip install -e .

This will add the local superman package to your PYTHONPATH,
which means any changes you make will be reflected right away.

### Testing

Use `nose2` or `pytest` to run the test suite,
or `nose2 -C` to also generate a coverage report.
Note that if you try running tests with `nosetests`,
only some of the test suite is detected and run.
