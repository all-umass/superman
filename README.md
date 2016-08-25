# Superman

*SpectrUm PrEpRocessin MAchiNe*

T. Boucher, C. Carey, S. Giguere, D. Dyar, S. Mahadevan

### Installation

    python setup.py install  # may need sudo

If you're making changes, try running:

    python setup.py develop  # may also need sudo

This will add the local superman package to your PYTHONPATH,
which means any changes you make will be reflected right away.

### Testing

Use `nose2` to run the test suite,
or `nose2 -C` to also generate a coverage report.
Note that if you try running tests with `nosetests`,
only some of the test suite is detected and run.
