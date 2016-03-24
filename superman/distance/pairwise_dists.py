from __future__ import absolute_import
import numpy as np
from six.moves import xrange

from ..mp import get_map_fn
from ._traj import (
    traj_match, traj_combo,
    traj_match_min, traj_combo_min,
    traj_match_full, traj_combo_full
)
from . import _pdist
from ._pdist import score_pdist

__all__ = [
    'pairwise_dists', 'pairwise_within', 'per_channel_scores', 'library_search',
    'score_pdist', 'score_pdist_row'
]


def pairwise_dists(A, B, metric, num_procs=1, min_window=0):
  metric, param = _parse_metric(metric)

  if metric == 'control':
    return np.random.random((len(A), len(B)))

  # trajectory data (two lists of Nx2 arrays)
  if not hasattr(A, 'shape'):
    return _traj_between(A, B, metric, float(param), num_procs=num_procs,
                         min_window=min_window)

  # vector data (two NxD arrays)
  D = np.zeros((len(A), len(B)))
  if metric == 'ms':
    _pdist.match_between(A, B, float(param), D)
  else:  # metric == 'combo'
    _pdist.combo_between(A, B, float(param), D)
  return D


def pairwise_within(A, metric, num_procs=1, min_window=0):
  metric, param = _parse_metric(metric)

  if metric == 'control':
    D = np.random.random((len(A), len(A)))
    np.fill_diagonal(D, 0)
    return (D+D.T)/2.0

  # trajectory data (two lists of Nx2 arrays)
  if not hasattr(A, 'shape'):
    return _traj_within(A, metric, float(param), num_procs=num_procs,
                        min_window=min_window)

  # vector data (two NxD arrays)
  D = np.zeros((len(A), len(A)))
  if metric == 'ms':
    _pdist.match_within(A, float(param), D)
  else:  # metric == 'combo'
    _pdist.combo_within(A, float(param), D)
  return D


def per_channel_scores(a, b, metric):
  metric, param = _parse_metric(metric)
  scores = np.zeros(a.shape[0], dtype=np.float32)
  if metric == 'ms':
    traj_match_full(a, b, param, scores)
  else:
    traj_combo_full(a, b, param, scores)
  return scores


def library_search(query, library, metric, num_procs=1, min_window=0):
  metric, param = _parse_metric(metric)
  if min_window > 1:
    a_window = np.zeros(min_window, dtype=np.float32)
    b_window = np.zeros(min_window, dtype=np.float32)
    traj_pairs = [(query, traj, param, a_window, b_window) for traj in library]
  else:
    traj_pairs = [(query, traj, param) for traj in library]
  return _fill_matrix(traj_pairs, None, None, metric, num_procs, min_window)


def score_pdist_row(dana_dist, test_dist):
  n = len(dana_dist)
  s = np.zeros(n)
  for i in xrange(n):
    s[i] = _pdist.score_pdist_row(dana_dist[i], test_dist[i], i, n)
  return s


def _parse_metric(metric):
  if ':' in metric:
    metric, param = metric.split(':', 1)
  if metric == 'cosine':
    metric, param = 'combo', 0
  elif metric == 'l1':
    metric, param = 'combo', 1
  elif metric not in ('ms', 'combo'):
    raise ValueError('Invalid metric: %r' % metric)
  return metric, float(param)


def _traj_within(traj, metric, param, num_procs=1, min_window=0):
  a_window = np.zeros(min_window, dtype=np.float32)
  b_window = np.zeros(min_window, dtype=np.float32)
  traj_pairs = []
  idx_pairs = []
  for i in xrange(len(traj)):
    for j in xrange(i+1, len(traj)):
      idx_pairs.append((i, j))
      traj_pairs.append((traj[i], traj[j], param, a_window, b_window))
  shape = (len(traj),) * 2
  S = _fill_matrix(traj_pairs, idx_pairs, shape, metric, num_procs,
                   min_window, symmetric=True)
  # copy over the j,i pairs
  # (Note that we can't do this in-place, because S.T is just a view of S)
  return S + S.T


def _traj_between(traj1, traj2, metric, param, num_procs=1, min_window=0):
  a_window = np.zeros(min_window, dtype=np.float32)
  b_window = np.zeros(min_window, dtype=np.float32)
  traj_pairs = []
  idx_pairs = []
  for i,t1 in enumerate(traj1):
    for j,t2 in enumerate(traj2):
      idx_pairs.append((i, j))
      traj_pairs.append((t1, t2, param, a_window, b_window))
  shape = (len(traj1), len(traj2))
  return _fill_matrix(traj_pairs, idx_pairs, shape, metric, num_procs,
                      min_window, symmetric=False)


def _fill_matrix(traj_pairs, idx_pairs, S_shape, metric, num_procs, min_window,
                 symmetric=False):
  if metric not in ('ms', 'combo'):
    raise ValueError('Invalid metric: %r' % metric)
  if min_window > 1:
    score_fn = _ms_score_min if metric == 'ms' else _combo_score_min
  else:
    score_fn = _ms_score if metric == 'ms' else _combo_score

  mapper = get_map_fn(num_procs, use_threads=False)
  # Note: sim is actually a distance measure
  sim = list(mapper(score_fn, traj_pairs))

  if idx_pairs is None:
    return np.array(sim)

  S = np.zeros(S_shape)
  for k, idx in enumerate(idx_pairs):
    S[idx] = sim[k] + 1
  if symmetric:
    np.fill_diagonal(S, 0)
  return S


# MP.map functions have to be toplevel
def _ms_score(args):
  return traj_match(*args[:3])
def _ms_score_min(args):
  return traj_match_min(*args)
def _combo_score(args):
  return traj_combo(*args[:3])
def _combo_score_min(args):
  return traj_combo_min(*args)
