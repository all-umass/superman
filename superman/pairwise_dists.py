import numpy as np
from sklearn.metrics import pairwise_distances as _sklearn_pdist
from superman.traj.all_pairs import lcss_between, lcss_within

import pyximport
pyximport.install()
import _pdist
score_pdist = _pdist.score_pdist


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

  # Check for the trajectory case
  if not hasattr(A, 'shape'):
    if metric == 'cosine':
      metric, param = 'combo', 0
    elif metric == 'l1':
      metric, param = 'combo', 1
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

  # Assume it's a sklearn metric.
  return _sklearn_pdist(A, B, metric=metric)


def pairwise_within(A, metric, num_procs=1, min_window=0):
  if ':' in metric:
    metric, param = metric.split(':', 1)

  if metric == 'control':
    D = np.random.random((len(A), len(A)))
    np.fill_diagonal(D, 0)
    return (D+D.T)/2.0

  # Check for the trajectory case
  if not hasattr(A, 'shape'):
    if metric == 'cosine':
      metric, param = 'combo', 0
    elif metric == 'l1':
      metric, param = 'combo', 1
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

  if metric == 'windowed':
    return windowed_cosine(A, int(param))

  # Assume it's a sklearn metric.
  return _sklearn_pdist(A, metric=metric)


def windowed_cosine(X, window):
  n = X.shape[0]
  kk = (n // window) + 1
  Xn = X / np.linalg.norm(X, ord=2, axis=1)[:,None]
  D = np.ones((n,n))
  for i in xrange(n):
    v1 = Xn[i]
    for j in xrange(i+1,n):
      v2 = Xn[j]
      d = 0.0
      for k in xrange(kk):
        v1w = v1[k*window:(k+1)*window]
        v2w = v2[k*window:(k+1)*window]
        v1w = v1w / np.linalg.norm(v1w)
        v2w = v2w / np.linalg.norm(v2w)
        d += (v1w*v2w).sum()
      D[i,j] = d
      D[j,i] = d
  D -= D.min()
  D /= D.max()
  return 1 - D
