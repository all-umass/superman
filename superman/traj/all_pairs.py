import numpy as np
import os
from utils import regenerate_cython

# Re-gen the Cython file from the template if needed
regenerate_cython(os.path.join(os.path.dirname(__file__), 'fast_lcss.pyx.in'))

import pyximport
pyximport.install()
from fast_lcss import traj_match, traj_combo, traj_match_min, traj_combo_min

from superman.mp import get_map_fn


def lcss_within(traj, metric, param, num_procs=1, verbose=True):
  a_window = np.zeros(20, dtype=np.float32)
  b_window = np.zeros(20, dtype=np.float32)
  traj_pairs = []
  idx_pairs = []
  for i in xrange(len(traj)):
    for j in xrange(i+1, len(traj)):
      idx_pairs.append((i, j))
      traj_pairs.append((traj[i], traj[j], param, a_window, b_window))
  shape = (len(traj),) * 2
  S = _fill_matrix(traj_pairs, idx_pairs, shape, metric, num_procs, verbose,
                   symmetric=True)
  # copy over the j,i pairs
  # (Note that we can't do this in-place, because S.T is just a view of S)
  return S + S.T


def lcss_between(traj1, traj2, metric, param, num_procs=1, verbose=True):
  a_window = np.zeros(20, dtype=np.float32)
  b_window = np.zeros(20, dtype=np.float32)
  traj_pairs = []
  idx_pairs = []
  for i,t1 in enumerate(traj1):
    for j,t2 in enumerate(traj2):
      idx_pairs.append((i, j))
      traj_pairs.append((t1, t2, param, a_window, b_window))
  shape = (len(traj1), len(traj2))
  return _fill_matrix(traj_pairs, idx_pairs, shape, metric, num_procs, verbose,
                      symmetric=False)


def lcss_search(query, library, metric, param, num_procs=1, min_window=0):
  mapper = get_map_fn(num_procs, use_threads=False)
  use_min = min_window > 1
  if metric not in ('ms', 'combo'):
    raise ValueError('Invalid metric: %r' % metric)
  if use_min:
    a_window = np.zeros(min_window, dtype=np.float32)
    b_window = np.zeros(min_window, dtype=np.float32)
    traj_pairs = [(query, traj, param, a_window, b_window) for traj in library]
    map_fn = _ms_score_min if metric == 'ms' else _combo_score_min
  else:
    traj_pairs = [(query, traj, param) for traj in library]
    map_fn = _ms_score if 'metric' == 'ms' else _combo_score
  # Note: sim is actually a distance measure
  return np.array(mapper(map_fn, traj_pairs))


def _fill_matrix(traj_pairs, idx_pairs, S_shape, metric, num_procs, verbose,
                 symmetric=False):
  S = np.zeros(S_shape)
  if verbose:
    print "computing pairwise distances:", S_shape
  mapper = get_map_fn(num_procs, use_threads=False)
  if metric == 'ms':
    sim = mapper(_ms_score, traj_pairs)
  elif metric == 'combo':
    sim = mapper(_combo_score, traj_pairs)
  else:
    raise ValueError('Invalid metric: %r' % metric)
  # insert into similarity matrix
  for k, (i,j) in enumerate(idx_pairs):
    S[i,j] = sim[k] + 1
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
