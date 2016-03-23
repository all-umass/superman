from __future__ import absolute_import
import numpy as np
from .traj.all_pairs import lcss_between, lcss_within, xrange

import pyximport
pyximport.install()
from . import _pdist
score_pdist = _pdist.score_pdist

__all__ = [
    'pairwise_dists', 'pairwise_within', 'score_pdist', 'score_pdist_row'
]


def score_pdist_row(dana_dist, test_dist):
  n = len(dana_dist)
  s = np.zeros(n)
  for i in xrange(n):
    s[i] = _pdist.score_pdist_row(dana_dist[i], test_dist[i], i, n)
  return s


def pairwise_dists(A, B, metric, num_procs=1, min_window=0):
  if ':' in metric:
    metric, param = metric.split(':', 1)

  if metric == 'control':
    return np.random.random((len(A), len(B)))

  if metric == 'cosine':
    metric, param = 'combo', 0
  elif metric == 'l1':
    metric, param = 'combo', 1

  # Check for the trajectory case
  if not hasattr(A, 'shape'):
    return lcss_between(A, B, metric, float(param), num_procs=num_procs,
                        min_window=min_window)

  if metric == 'ms':
    D = np.zeros((len(A), len(B)))
    _pdist.match_between(A, B, float(param), D)
    return D

  if metric == 'combo':
    w = float(param)
    assert 0 <= w <= 1
    D = np.zeros((len(A), len(B)))
    # For each pair of intensities a,b: (max norm)
    #  dist = w * abs(a - b) - (1 - w) * (a * b)
    _pdist.combo_between(A, B, w, D)
    return D

  raise ValueError('Invalid metric: %r' % metric)


def pairwise_within(A, metric, num_procs=1, min_window=0):
  if ':' in metric:
    metric, param = metric.split(':', 1)

  if metric == 'control':
    D = np.random.random((len(A), len(A)))
    np.fill_diagonal(D, 0)
    return (D+D.T)/2.0

  if metric == 'cosine':
    metric, param = 'combo', 0
  elif metric == 'l1':
    metric, param = 'combo', 1

  # Check for the trajectory case
  if not hasattr(A, 'shape'):
    return lcss_within(A, metric, float(param), num_procs=num_procs,
                       min_window=min_window)

  if metric == 'ms':
    D = np.zeros((len(A), len(A)))
    _pdist.match_within(A, float(param), D)
    return D

  if metric == 'combo':
    w = float(param)
    assert 0 <= w <= 1
    D = np.zeros((len(A), len(A)))
    _pdist.combo_within(A, w, D)
    return D

  raise ValueError('Invalid metric: %r' % metric)
