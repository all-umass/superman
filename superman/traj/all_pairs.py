import numpy as np
import os
# Re-gen the Cython file from the template if needed
tpl_file = os.path.join(os.path.dirname(__file__), 'fast_lcss.pyx.in')
pyx_file = tpl_file[:-3]
if (not os.path.exists(pyx_file) or
    os.path.getmtime(tpl_file) > os.path.getmtime(pyx_file)):
  from Cython import Tempita
  tpl = Tempita.Template.from_filename(tpl_file, encoding='utf-8')
  with open(pyx_file, 'w') as fh:
    fh.write(tpl.substitute())

import pyximport
pyximport.install()
from fast_lcss import traj_match, traj_combo

from superman.mp import get_mp_pool


def lcss_within(traj, metric, param, num_procs=1, verbose=True):
  traj_pairs = []
  idx_pairs = []
  for i in xrange(len(traj)):
    for j in xrange(i+1, len(traj)):
      idx_pairs.append((i, j))
      traj_pairs.append((traj[i], traj[j], param))
  shape = (len(traj),) * 2
  S = _fill_matrix(traj_pairs, idx_pairs, shape, metric, num_procs, verbose,
                   symmetric=True)
  # copy over the j,i pairs
  # (Note that we can't do this in-place, because S.T is just a view of S)
  return S + S.T


def lcss_between(traj1, traj2, metric, param, num_procs=1, verbose=True):
  traj_pairs = []
  idx_pairs = []
  for i,t1 in enumerate(traj1):
    for j,t2 in enumerate(traj2):
      idx_pairs.append((i, j))
      traj_pairs.append((t1, t2, param))
  shape = (len(traj1), len(traj2))
  return _fill_matrix(traj_pairs, idx_pairs, shape, metric, num_procs, verbose,
                      symmetric=False)


def lcss_search(query, library, metric, param, num_procs=1, verbose=True):
  traj_pairs = [(query, traj, param) for traj in library]
  pool = get_mp_pool(num_procs)
  if metric == 'ms':
    S = pool.map(_ms_score, traj_pairs)
  elif metric == 'combo':
    S = pool.map(_combo_score, traj_pairs)
  else:
    raise ValueError('Invalid metric: %r' % metric)
  # Note: we _don't_ convert to distances here, or normalize!
  # TODO: verify that this is doing what we think it is.
  return np.array(S)


# MP.map functions have to be toplevel
def _ms_score(args):
  return traj_match(*args)


def _combo_score(args):
  return traj_combo(*args)


def _fill_matrix(traj_pairs, idx_pairs, S_shape, metric, num_procs, verbose,
                 symmetric=False):
  S = np.zeros(S_shape)
  if verbose:
    print "computing pairwise distances:", S_shape
  pool = get_mp_pool(num_procs)
  if metric == 'ms':
    sim = pool.map(_ms_score, traj_pairs)
  elif metric == 'combo':
    sim = pool.map(_combo_score, traj_pairs)
  else:
    raise ValueError('Invalid metric: %r' % metric)
  # insert into similarity matrix
  for k, (i,j) in enumerate(idx_pairs):
    S[i,j] = sim[k] + 1
  if symmetric:
    np.fill_diagonal(S, 0)
  return S
