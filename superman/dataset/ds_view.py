from __future__ import absolute_import
import numpy as np

from ..distance.all_pairs import lcss_search, lcss_full


class DatasetView(object):
  def __init__(self, ds, mask=Ellipsis, pp='', blr_obj=None, chan_mask=False,
               blr_segmented=False, crop=(-np.inf, np.inf), nan_gap=None):
    self.ds = ds
    self.mask = mask
    # lazy transformation steps, which get applied to on-demand trajectories
    self.transformations = dict(pp=pp, blr_obj=blr_obj,
                                blr_segmented=blr_segmented, crop=crop,
                                nan_gap=nan_gap, chan_mask=chan_mask)

  def __str__(self):
    return '<DatasetView of "%s": %r>' % (self.ds, self.transformations)

  def get_trajectory(self, key):
    return self.ds.get_trajectory(key, self.transformations)

  def get_trajectories(self, return_keys=False):
    # hack for speed, kinda lame
    if hasattr(self.ds, 'intensities'):
      traj = self.ds.get_trajectories_by_index(self.mask, self.transformations)
      if return_keys:
        return traj, self.ds.pkey.index2key(self.mask)
      return traj

    keys = self.ds.pkey.index2key(self.mask)
    traj = self.ds.get_trajectories(keys, self.transformations)
    if return_keys:
      return traj, keys
    return traj

  def get_data(self, return_keys=False):
    if not hasattr(self.ds, 'intensities'):
      return self.get_trajectories(return_keys=return_keys)
    # just return the intensities matrix
    data = self.ds.intensities[self.mask]
    if return_keys:
      return data, self.ds.pkey.index2key(self.mask)
    return data

  def get_metadata(self, meta_key):
    meta, label = self.ds.find_metadata(meta_key)
    data = meta.get_array(self.mask)
    return data, label

  def compute_line(self, bounds):
    assert len(bounds) in (2, 4)

    # hack for speed, kinda lame
    if hasattr(self.ds, 'intensities'):
      bands, ints = self.ds._transform_vector(self.ds.bands,
                                              self.ds.intensities[self.mask,:],
                                              self.transformations)
      indices = np.searchsorted(bands, bounds)
      line = ints[:, indices[0]:indices[1]].max(axis=1)
      if len(bounds) == 4:
        line /= ints[:, indices[2]:indices[3]].max(axis=1)
      return line

    keys = self.ds.pkey.index2key(self.mask)
    traj = self.ds.get_trajectories(keys, self.transformations)
    line = np.zeros(len(traj))
    for i, t in enumerate(traj):
      indices = np.searchsorted(t[:,0], bounds)
      line[i] = t[indices[0]:indices[1], 1].max()
      if len(bounds) == 4:
        line[i] /= t[indices[2]:indices[3], 1].max()
    return line

  def whole_spectrum_search(self, query, num_endmembers=1, num_results=10,
                            metric='combo', param=0, num_procs=5,
                            min_window=0, score_pct=1, method='sub'):
    # neurotic error checking
    if self.ds.pkey is None:
      raise ValueError('%s has no primary key, cannot search.' % self.ds)
    if not (0 < num_endmembers < 10):
      raise ValueError('Invalid number of mixture components: %d' %
                       num_endmembers)
    if not (0 <= param <= 1):
      raise ValueError('Invalid WSM parameter: %f' % param)
    if not (0 <= min_window <= 1000):
      raise ValueError('Invalid min window size: %d' % min_window)
    if not (0 < score_pct < 100):
      raise ValueError('Invalid score percentile threshold: %d' % score_pct)
    if method not in ('add', 'sub'):
      raise ValueError('Invalid query modification method: %r' % method)
    if query[0,0] > query[1,0]:
      raise ValueError('Query spectrum must have increasing bands')
    if abs(1 - query[:,1].max()) > 0.001:
      raise ValueError('Query spectrum must be max-normalized')

    # prepare the query
    query = query.astype(np.float32, order='C', copy=True)

    # prepare the search library
    library, names = self.get_trajectories(return_keys=True)
    library = [t.astype(np.float32, order='C') for t in library]

    # run the search
    num_sub_results = int(np.ceil(num_results ** (1./num_endmembers)))
    res = _cs_helper(query, library, names, num_endmembers, num_sub_results,
                     metric, param, num_procs, min_window, score_pct, method)
    top_sim, top_names = zip(*sorted(res, reverse=True)[:num_results])
    return top_names, top_sim


def _cs_helper(query, library, names, num_endmembers, num_results, metric,
               param, num_procs, min_window, score_pct, method):
  # calculate a vector of distances
  dist = lcss_search(query, library, metric, param, num_procs=num_procs,
                     min_window=min_window)
  # rank the top k closest
  top_k = np.argsort(dist)[:num_results]
  top_sim = -(dist[top_k])

  for i, k in enumerate(top_k):
    name = names[k]
    sim = top_sim[i]
    if num_endmembers == 1:
      yield sim, [name]
    else:
      if method == 'add':
        tmp = library[k]
        lib = [_add_spectrum(x, tmp) for x in library]
        # recurse
        for s2, n2 in _cs_helper(query, lib, names, num_endmembers-1,
                                 num_results, metric, param, num_procs,
                                 min_window, score_pct, method):
          yield s2, [name] + n2
      else:
        # get per-channel distance scores
        scores = lcss_full(query, library[k], metric, param)
        # remove the matching spectrum (where distance <= x%)
        mask = scores <= np.percentile(scores, score_pct)
        sub = _remove_spectrum(query, mask)
        # recurse
        for s2, n2 in _cs_helper(sub, library, names, num_endmembers-1,
                                 num_results, metric, param, num_procs,
                                 min_window, score_pct, method):
          yield s2 * sim, [name] + n2


def _remove_spectrum(a, mask):
  '''zeroes out the masked intensities of spectrum a, then re-normalizes'''
  sub = a.copy()
  sub[:,1] = np.where(mask, 0, a[:,1])
  sub[:,1] /= sub[:,1].max()
  return sub


def _add_spectrum(a, b):
  bands, ints1 = a.T
  ints2 = np.interp(bands, *b.T)
  ints = np.maximum(ints1, ints2)
  if bands[0] > b[0,0]:
    idx = np.searchsorted(b[:,0], bands[0])
    bands = np.concatenate((b[:idx,0], bands))
    ints = np.concatenate((b[:idx,1], ints))
  if bands[-1] < b[-1,0]:
    idx = np.searchsorted(b[:,0], bands[-1])
    bands = np.concatenate((bands, b[idx:,0]))
    ints = np.concatenate((ints, b[idx:,1]))
  return np.column_stack((bands, ints)).astype(np.float32, order='C')
