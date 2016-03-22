import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import unittest

from superman import pairwise_dists as pd

METRICS = ['control', 'cosine', 'l1', 'l2', 'ms:0.5', 'combo:0.5']


class PairwiseDists(unittest.TestCase):
  def setUp(self):
    self.A = np.ascontiguousarray(np.linspace(0, 1, 10)[None])
    self.B = np.ascontiguousarray(np.row_stack((np.sqrt(self.A), 1.1 * self.A)))

  def test_cosine_metric(self):
    D = pd.pairwise_dists(self.A, self.B, 'cosine', num_procs=1, min_window=0)
    assert_array_almost_equal(D, [[0.019409, 0]])
    # combo:0 is cosine-like, though it's negated and doesn't normalize
    D = pd.pairwise_dists(self.A, self.B, 'combo:0', num_procs=1, min_window=0)
    assert_array_almost_equal(D, [[-4.112946, -3.87037]])

  def test_l1_metric(self):
    expected = [[1.435334, 0.5]]
    D = pd.pairwise_dists(self.A, self.B, 'l1', num_procs=1, min_window=0)
    assert_array_almost_equal(D, expected)
    D = pd.pairwise_dists(self.A, self.B, 'combo:1', num_procs=1, min_window=0)
    assert_array_almost_equal(D, expected)

  def test_ms_metric(self):
    D = pd.pairwise_dists(self.A, self.B, 'ms:0', num_procs=1, min_window=0)
    assert_array_almost_equal(D, [[5, 5]])
    D = pd.pairwise_dists(self.A, self.B, 'ms:0.5', num_procs=1, min_window=0)
    assert_array_almost_equal(D, [[1.818657, 0.734404]])
    D = pd.pairwise_dists(self.A, self.B, 'ms:1', num_procs=1, min_window=0)
    assert_array_almost_equal(D, [[0.840906, 0.148148]])


class PairwiseWithin(unittest.TestCase):
  def setUp(self):
    A = np.linspace(0, 1, 10)
    self.A = np.ascontiguousarray(np.row_stack((A, np.sqrt(A), 1.1 * A)))

  def test_cosine_metric(self):
    D = pd.pairwise_within(self.A, 'cosine', num_procs=1, min_window=0)
    d01, d02, d12 = 0.019409, 0, 0.019409
    assert_array_almost_equal(D, [[0, d01, d02], [d01, 0, d12], [d02, d12, 0]])
    # combo:0 is cosine-like, though it's negated and doesn't normalize
    D = pd.pairwise_within(self.A, 'combo:0', num_procs=1, min_window=0)
    d01, d02, d12 = -4.112946, -3.87037, -4.52424
    assert_array_almost_equal(D, [[0, d01, d02], [d01, 0, d12], [d02, d12, 0]])

  def test_l1_metric(self):
    D = pd.pairwise_within(self.A, 'l1', num_procs=1, min_window=0)
    d01, d02, d12 = 1.435334, 0.5, 1.205271
    assert_array_almost_equal(D, [[0, d01, d02], [d01, 0, d12], [d02, d12, 0]])
    D = pd.pairwise_within(self.A, 'combo:1', num_procs=1, min_window=0)
    assert_array_almost_equal(D, [[0, d01, d02], [d01, 0, d12], [d02, d12, 0]])


class ScorePDist(unittest.TestCase):
  def test_score_pdist(self):
    # 0/1 -> species, 2 -> group, 3 -> type, 4 -> class, 5 -> no match
    fake_dana = np.array([[0,2,2,4,5,5],
                          [2,0,1,3,5,5],
                          [2,1,0,3,3,5],
                          [4,3,3,0,1,2],
                          [5,5,3,1,0,2],
                          [5,5,5,2,2,0]], dtype=np.uint8)
    fake_dist = fake_dana * 0.1
    self.assertEqual(pd.score_pdist(fake_dana, fake_dist), 0)
    assert_array_equal(pd.score_pdist_row(fake_dana, fake_dist), [0] * 6)

    # add an inconsistency (symmetric)
    fake_dist[0,3] = 1
    fake_dist[3,0] = 1
    self.assertEqual(pd.score_pdist(fake_dana, fake_dist), 2)
    expected = [2, 0, 0, 0, 0, 0]
    assert_array_equal(pd.score_pdist_row(fake_dana, fake_dist), expected)


if __name__ == '__main__':
  unittest.main()
