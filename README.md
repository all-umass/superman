# Superman

*SpectrUm PrEpRocessing MAchiNe*

T. Boucher, C. Carey, S. Giguere, D. Dyar, S. Mahadevan

### Installation

Python versions 2.7 and 3.4+ are supported.

Linux users can install pre-built wheels using `pip`:

    pip install --use-wheel superman

Others must must build from source, so make sure you have a working
C compiler, Python headers, and OpenMP before proceeding.

    pip install Cython
    pip install superman

If you're contributing to superman, or want to make local changes:

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
