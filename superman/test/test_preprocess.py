from __future__ import absolute_import
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest

from ..preprocess import preprocess

# static test fixture data
y = np.linspace(0, 1, 10)
bands = np.arange(len(y))
vectors = np.ascontiguousarray(np.row_stack((y, np.sqrt(y), 1.1 * y)))
trajs = [np.column_stack((bands, x)).astype(np.float32, order='C')
         for x in (y, np.sqrt(y), 1.1 * y)]


class PreprocessTests(unittest.TestCase):

  def test_pp_vector(self):
    result = preprocess(vectors, '')
    assert_array_almost_equal(vectors, result)

    result = preprocess(vectors, 'normalize:max,squash:sqrt,smooth:5:2')
    self.assertEqual(vectors.shape, result.shape)

  def test_pp_traj(self):
    result = preprocess(trajs, '')
    self.assertEqual(len(trajs), len(result))
    for t, r in zip(trajs, result):
      assert_array_almost_equal(t, r)

    result = preprocess(trajs, 'normalize:max,squash:sqrt,smooth:5:2')
    self.assertEqual(len(trajs), len(result))


if __name__ == '__main__':
  unittest.main()
