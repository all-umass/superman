import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import unittest

from superman import pairwise_dists as pd


class PairwiseDists(unittest.TestCase):
  def setUp(self):
    y = np.linspace(0, 1, 10)
    vector_pair = (
        np.ascontiguousarray(y[None]),
        np.ascontiguousarray(np.row_stack((np.sqrt(y), 1.1 * y)))
    )
    bands = np.arange(len(y))
    traj_pair = (
        [np.column_stack((bands, y)).astype(np.float32, order='C')],
        [np.column_stack((bands, np.sqrt(y))).astype(np.float32, order='C'),
         np.column_stack((bands, 1.1*y)).astype(np.float32, order='C')]
    )
    self.input_pairs = (vector_pair, traj_pair)

  def test_cosine_metric(self):
    expected = [[0.019409, 0]]
    for A, B in self.input_pairs:
      D = pd.pairwise_dists(A, B, 'cosine', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected)
      D = pd.pairwise_dists(A, B, 'combo:0', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected)

  def test_l1_metric(self):
    expected = [[1.342206, 1.129187]]
    for A, B in self.input_pairs:
      D = pd.pairwise_dists(A, B, 'l1', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected)
      D = pd.pairwise_dists(A, B, 'combo:1', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected)

  def test_ms_metric(self):
    for A, B in self.input_pairs:
      D = pd.pairwise_dists(A, B, 'ms:0', num_procs=1, min_window=0)
      assert_array_almost_equal(D, [[1, 1]])
      D = pd.pairwise_dists(A, B, 'ms:0.5', num_procs=1, min_window=0)
      assert_array_almost_equal(D, [[1.835655, 1.955819]])
      D = pd.pairwise_dists(A, B, 'ms:1', num_procs=1, min_window=0)
      assert_array_almost_equal(D, [[2.050358, 2.200957]])


class PairwiseWithin(unittest.TestCase):
  def setUp(self):
    y = np.linspace(0, 1, 10)
    vector = np.ascontiguousarray(np.row_stack((y, np.sqrt(y), 1.1 * y)))
    bands = np.arange(len(y))
    traj = [
        np.column_stack((bands, vector[0])).astype(np.float32, order='C'),
        np.column_stack((bands, vector[1])).astype(np.float32, order='C'),
        np.column_stack((bands, vector[2])).astype(np.float32, order='C'),
    ]
    self.input_data = (vector, traj)

  def test_cosine_metric(self):
    d01, d02, d12 = 0.019409, 0, 0.019409
    expected = [[0, d01, d02], [d01, 0, d12], [d02, d12, 0]]
    for A in self.input_data:
      D = pd.pairwise_within(A, 'cosine', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected)
      D = pd.pairwise_within(A, 'combo:0', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected)

  def test_l1_metric(self):
    d01, d02, d12 = 1.342206, 1.129187, 1.261232
    expected = [[0, d01, d02], [d01, 0, d12], [d02, d12, 0]]
    for A in self.input_data:
      D = pd.pairwise_within(A, 'l1', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected)
      D = pd.pairwise_within(A, 'combo:1', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected)

  def test_ms_metric(self):
    expected_ms0 = 1 - np.eye(3)
    d01, d02, d12 = 2.050358, 2.200957, 2.04492
    expected_ms1 = [[0, d01, d02], [d01, 0, d12], [d02, d12, 0]]
    for A in self.input_data:
      D = pd.pairwise_within(A, 'ms:0', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected_ms0)
      D = pd.pairwise_within(A, 'ms:1', num_procs=1, min_window=0)
      assert_array_almost_equal(D, expected_ms1)


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
