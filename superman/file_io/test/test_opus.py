from __future__ import absolute_import
import numpy as np
import os
import unittest
from numpy.testing import assert_array_almost_equal
from tempfile import mkstemp

from .. import parse_spectrum, write_opus, parse_opus

FIXTURE_PATH = os.path.dirname(__file__)


class OpusTests(unittest.TestCase):
  def test_actual_data(self):
    fname = os.path.join(FIXTURE_PATH, 'actual.0')
    x = parse_spectrum(fname, filetype='opus')
    self.assertEqual(x.shape, (1452, 2))
    assert_array_almost_equal(x.sum(axis=0), [2541000, 794769.13498])

    xx, params = parse_opus(open(fname, 'rb'), return_params=True)
    assert_array_almost_equal(x, xx)
    self.assertEqual(params['SNM'], 'HECTO')
    self.assertEqual(params['SFM'], '')
    self.assertEqual(params['CNM'], 'Administrator')

  def test_roundtrip(self):
    x = np.column_stack((np.arange(5), np.random.random(5)))
    fh, fname = mkstemp(suffix='.0', dir=FIXTURE_PATH)
    os.close(fh)
    try:
      write_opus(fname, x, 'comments go here!')
      xx, params = parse_opus(open(fname), return_params=True)
    finally:
      os.unlink(fname)
    assert_array_almost_equal(xx, x)
    self.assertIs(params, None)


if __name__ == '__main__':
  unittest.main()
