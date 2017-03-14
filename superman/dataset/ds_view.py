from __future__ import absolute_import
import numpy as np

from ..distance import library_search, per_channel_scores


class DatasetView(object):
  def __init__(self, ds, mask=Ellipsis, pp='', blr_obj=None, chan_mask=False,
               blr_segmented=False, blr_inverted=False, flip=False,
               crop=(), nan_gap=None):
    self.ds = ds
    self.mask = mask
    # lazy transformation steps, which get applied to on-demand trajectories
    self.transformations = dict(pp=pp, blr_obj=blr_obj,
                                blr_segmented=blr_segmented,
                                blr_inverted=blr_inverted,
                                flip=flip, crop=crop, nan_gap=nan_gap,
                                chan_mask=chan_mask)

  def num_spectra(self):
    if hasattr(self.mask, 'dtype') and self.mask.dtype.name == 'bool':
      return np.count_nonzero(self.mask)
    if self.mask is Ellipsis:
      return self.ds.num_spectra()
    return len(self.mask)

  def __str__(self):
    return '<DatasetView of "%s": %r>' % (self.ds, self.transformations)

  def get_trajectory(self, key):
    return self.ds.get_trajectory(key, self.transformations)

  def get_trajectories(self, return_keys=False):
    # hack for speed, kinda lame
    if hasattr(self.ds, 'intensities'):
      traj = self.ds.get_trajectories_by_index(self.mask, self.transformations)
      if return_keys:
        return traj, self.get_primary_keys()
      return traj

    keys = self.get_primary_keys()
    traj = self.ds.get_trajectories(keys, self.transformations)
    if return_keys:
      return traj, keys
    return traj

  def get_vector_data(self):
    ds = self.ds
    if hasattr(ds, 'intensities'):
      return ds._transform_vector(ds.bands, ds.intensities[self.mask,:],
                                  self.transformations)

    # resample to vector format (if we have enough crop info to do so)
    crops = self.transformations['crop']
    if not (crops and all(c[2] > 0 for c in crops)):
      raise ValueError('Cannot create vector data from non-resampled trajs.')

    # find the min and max x values over all trajs
    keys = self.get_primary_keys()
    xmin = float('inf')
    xmax = float('-inf')
    for key in keys:
      x = ds.traj[key][:,0]
      xmin = min(xmin, x[0])
      xmax = max(xmax, x[-1])

    # compute new bands and allocate an intensities matrix for each crop
    x_chunks, y_chunks = [], []
    for lb, ub, step in crops:
      assert step > 0
      x_new = np.arange(max(lb, xmin), min(ub, xmax) + step, step)
      y_new = np.zeros((len(keys), len(x_new)), dtype=x.dtype)
      x_chunks.append(x_new)
      y_chunks.append(y_new)

    # actually do the resampling for each traj for each crop
    for i, key in enumerate(keys):
      t = ds.traj[key]
      x = t[:,0]
      y = t[:,1]
      for c, bands in enumerate(x_chunks):
        y_chunks[c][i] = np.interp(bands, x, y)

    # stick the chunks together
    bands = np.concatenate(x_chunks)
    ints = np.hstack(y_chunks)

    # remove crop information from the transforms
    trans = dict(self.transformations)
    trans['crop'] = ()
    return ds._transform_vector(bands, ints, trans)

  def get_metadata(self, meta_key):
    meta, label = self.ds.find_metadata(meta_key)
    data = meta.get_array(self.mask)
    return data, label

  def get_primary_keys(self):
    if self.ds.pkey is not None:
      return self.ds.pkey.index2key(self.mask)
    # create fake pkeys based on the mask
    if isinstance(self.mask, np.ndarray) and self.mask.dtype.name == 'bool':
      idx, = np.where(self.mask)
    else:
      idx = np.array(self.mask, dtype=int, copy=False)
    return ['Spectrum %d' % i for i in idx]

  def compute_line(self, bounds):
    assert len(bounds) in (2, 4)

    # hack for speed, kinda lame
    if hasattr(self.ds, 'intensities'):
      bands, ints = self.ds._transform_vector(self.ds.bands,
                                              self.ds.intensities[self.mask,:],
                                              self.transformations)
      indices = np.searchsorted(bands, bounds)
      # check for zero-length slices
      if np.any(np.diff(indices.reshape(-1,2), axis=1) == 0):
        return np.full(ints.shape[0], np.nan)
      line = ints[:, indices[0]:indices[1]].max(axis=1)
      if len(bounds) == 4:
        line /= ints[:, indices[2]:indices[3]].max(axis=1)
      return line

    keys = self.get_primary_keys()
    traj = self.ds.get_trajectories(keys, self.transformations)
    line = np.zeros(len(traj))
    for i, t in enumerate(traj):
      indices = np.searchsorted(t[:,0], bounds)
      # check for zero-length slices
      if np.any(np.diff(indices.reshape(-1,2), axis=1) == 0):
        line[i] = np.nan
        continue
      line[i] = t[indices[0]:indices[1], 1].max()
      if len(bounds) == 4:
        line[i] /= t[indices[2]:indices[3], 1].max()
    return line

  def whole_spectrum_search(self, query, num_endmembers=1, num_results=10,
                            metric='combo:0', num_procs=5, min_window=0,
                            score_pct=1, method='sub'):
    query, num_sub_results = _prep_wsm(query, num_endmembers, num_results,
                                       min_window, score_pct, method)

    # prepare the search library
    library, names = self.get_trajectories(return_keys=True)
    library = [t.astype(np.float32, order='C') for t in library]

    # run the search
    res = _cs_helper(query, library, names, num_endmembers, num_sub_results,
                     metric, num_procs, min_window, score_pct, method)
    top_sim, top_names = zip(*sorted(res, reverse=True)[:num_results])
    return top_names, top_sim


def _prep_wsm(query, num_endmembers, num_results, min_window, score_pct,
              method):
  # neurotic error checking
  if not (0 < num_endmembers < 10):
    raise ValueError('Invalid number of mixture components: %d' %
                     num_endmembers)
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

  # prepare for matching
  query = query.astype(np.float32, order='C', copy=True)
  num_sub_results = int(np.ceil(num_results ** (1./num_endmembers)))
  return query, num_sub_results


def _cs_helper(query, library, names, num_endmembers, num_results, metric,
               num_procs, min_window, score_pct, method):
  # calculate a vector of distances
  dist = library_search(query, library, metric, num_procs=num_procs,
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
                                 num_results, metric, num_procs,
                                 min_window, score_pct, method):
          yield s2, [name] + n2
      else:
        # get per-channel distance scores
        scores = per_channel_scores(query, library[k], metric)
        # remove the matching spectrum (where distance <= x%)
        mask = scores <= np.percentile(scores, score_pct)
        sub = _remove_spectrum(query, mask)
        # recurse
        for s2, n2 in _cs_helper(sub, library, names, num_endmembers-1,
                                 num_results, metric, num_procs, min_window,
                                 score_pct, method):
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


class MultiDatasetView(object):
  def __init__(self, ds_views):
    self.ds_views = ds_views
    self.num_views = len(ds_views)
    self.num_datasets = len(set(dv.ds for dv in ds_views))
    self._num_spectra = None
    kinds = set(dv.ds.kind for dv in ds_views)
    if len(kinds) != 1:
      raise ValueError('Cannot use MultiDatasetView with >1 kind of dataset.')
    self.ds_kind, = kinds

  def num_spectra(self):
    if self._num_spectra is None:
      self._num_spectra = sum(dv.num_spectra() for dv in self.ds_views)
    return self._num_spectra

  def split_across_views(self, arr):
    assert arr.shape[0] == self.num_spectra()
    splits = np.cumsum([dv.num_spectra() for dv in self.ds_views[:-1]])
    return np.split(arr, splits)

  def dataset_name_metadata(self):
    ds_names, counts = [], []
    for dv in self.ds_views:
      ds_names.append(dv.ds.name)
      counts.append(dv.num_spectra())
    return np.repeat(ds_names, counts)

  def get_primary_keys(self):
    if self.num_datasets == 1:
      return np.concatenate([dv.get_primary_keys() for dv in self.ds_views])
    # prepend the dataset name (but not kind) to each pkey
    dv_pkeys = []
    for dv in self.ds_views:
      dv_name = dv.ds.name
      for k in dv.get_primary_keys():
        dv_pkeys.append('%s: %s' % (dv_name, k))
    return np.array(dv_pkeys, dtype=bytes)

  def get_metadata(self, meta_key):
    data, label = [], None
    for dv in self.ds_views:
      x, lbl = dv.get_metadata(meta_key)
      data.append(x)
      if label is not None and lbl != label:
        raise ValueError('Mismatching metadata labels: %r != %r' % (label, lbl))
      label = lbl
    return np.concatenate(data), label

  def get_trajectories(self, return_keys=False, avoid_nan_gap=False):
    trajs = []
    for dv in self.ds_views:
      if avoid_nan_gap:
        nan_gap = dv.transformations.get('nan_gap', None)
        dv.transformations['nan_gap'] = None
        trajs.extend(dv.get_trajectories())
        dv.transformations['nan_gap'] = nan_gap
      else:
        trajs.extend(dv.get_trajectories())
    if return_keys:
      return trajs, self.get_primary_keys()
    return trajs

  def get_vector_data(self):
    wave, X = None, []
    for dv in self.ds_views:
      w, x = dv.get_vector_data()
      if wave is None:
        wave = w
      else:
        if wave.shape != w.shape or not np.allclose(wave, w):
          raise ValueError("Mismatching wavelength data in %s." % dv.ds)
      X.append(x)
    return wave, np.vstack(X)

  def x_axis_units(self):
    labels = set(dv.ds.x_axis_units() for dv in self.ds_views)
    if len(labels) != 1:
      raise ValueError('MultiDatasetView has >1 x-axis unit: %s' % labels)
    return tuple(labels)[0]

  def compute_line(self, bounds):
    return np.concatenate([dv.compute_line(bounds) for dv in self.ds_views])

  def whole_spectrum_search(self, query, num_endmembers=1, num_results=10,
                            metric='combo:0', num_procs=5, min_window=0,
                            score_pct=1, method='sub'):
    query, num_sub_results = _prep_wsm(query, num_endmembers, num_results,
                                       min_window, score_pct, method)

    # prepare the search library
    library, names = self.get_trajectories(return_keys=True, avoid_nan_gap=True)
    library = [t.astype(np.float32, order='C') for t in library]

    # run the search
    res = _cs_helper(query, library, names, num_endmembers, num_sub_results,
                     metric, num_procs, min_window, score_pct, method)
    top_sim, top_names = zip(*sorted(res, reverse=True)[:num_results])
    return top_names, top_sim
