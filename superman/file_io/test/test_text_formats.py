from __future__ import absolute_import
import numpy as np
import os
import unittest
from numpy.testing import assert_array_almost_equal

from .. import parse_spectrum

FIXTURE_PATH = os.path.dirname(__file__)
FIXTURE_DATA = np.array([[0.4,3.2],[1.2,2.7],[2.0,5.4]])


class TextFormatTests(unittest.TestCase):
  def test_tsv_data(self):
    x = parse_spectrum(os.path.join(FIXTURE_PATH, 'fixture.tsv'))
    assert_array_almost_equal(x, FIXTURE_DATA)

  def test_csv_data(self):
    x = parse_spectrum(os.path.join(FIXTURE_PATH, 'fixture.csv'))
    assert_array_almost_equal(x, FIXTURE_DATA)

  def test_loose_data(self):
    x = parse_spectrum(os.path.join(FIXTURE_PATH, 'fixture.txt'))
    assert_array_almost_equal(x, FIXTURE_DATA)

if __name__ == '__main__':
  unittest.main()
