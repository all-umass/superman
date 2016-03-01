import numpy as np

from superman.preprocess import preprocess
from superman.traj.all_pairs import lcss_search, lcss_full
from superman.utils import ALAMOS_MASK
from metadata import is_metadata, PrimaryKeyMetadata

__all__ = ['VectorDataset', 'TrajDataset']


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
                            min_window=0, score_pct=1):
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
                     metric, param, num_procs, min_window, score_pct)
    top_sim, top_names = zip(*sorted(res, reverse=True)[:num_results])
    return top_names, top_sim


def _cs_helper(query, library, names, num_endmembers, num_results, metric,
               param, num_procs, min_window, score_pct):
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
      # get per-channel distance scores
      scores = lcss_full(query, library[k], metric, param)
      # remove the matching spectrum (where distance <= x%)
      sub = _remove_spectrum(query, scores <= np.percentile(scores, score_pct))
      # recurse
      for s2, n2 in _cs_helper(sub, library, names, num_endmembers-1,
                               num_results, metric, param, num_procs,
                               min_window, score_pct):
        yield s2 * sim, [name] + n2


def _remove_spectrum(a, mask):
  '''zeroes out the masked intensities of spectrum a, then re-normalizes'''
  sub = a.copy()
  sub[:,1] = np.where(mask, 0, a[:,1])
  sub[:,1] /= sub[:,1].max()
  return sub


class Dataset(object):
  def __init__(self, name, spec_kind):
    self.name = name
    self.kind = spec_kind
    self.pkey = None  # primary key metadata, optional
    self.metadata = dict()

  def set_metadata(self, metadata_dict):
    self.metadata = metadata_dict
    # do a little metadata validation
    for key, m in self.metadata.iteritems():
      if not is_metadata(m):
        raise ValueError('%r is not a valid Metadata' % key)
      if self.pkey is not None and isinstance(m, PrimaryKeyMetadata):
        raise ValueError("%s can't have >1 primary key: %s" % (self, key))

  def filter_metadata(self, filter_conditions):
    mask = np.ones(self.num_spectra(), dtype=bool)
    for key, cond in filter_conditions.iteritems():
      if key == 'pkey':
        mask &= self.pkey.filter(cond)
      else:
        mask &= self.metadata[key].filter(cond)
    return mask

  def find_metadata(self, meta_key):
    # hack: index into compositions with $-joined keys
    if '$' in meta_key:
      key, subkey = meta_key.split('$', 1)
      meta = self.metadata[key].comps[subkey]
      label = '%s: %s' % (self.metadata[key].display_name(key),
                          meta.display_name(subkey))
    else:
      meta = self.metadata[meta_key]
      label = meta.display_name(meta_key)
    return meta, label

  def view(self, **kwargs):
    return DatasetView(self, **kwargs)

  def __str__(self):
    return '%s [%s]' % (self.name, self.kind)


class TrajDataset(Dataset):
  def set_data(self, keys, traj_map, **metadata):
    self.set_metadata(metadata)
    # list of names, or other keys
    self.pkey = PrimaryKeyMetadata(keys)
    # mapping from key to (n,2)-array
    self.traj = traj_map
    for key in keys:
      s = traj_map[key]
      assert s[0,0] <= s[1,0], 'Backwards bands in %s: %s' % (self, key)
    # make sure all our shapes match
    n = self.num_spectra()
    for k, m in self.metadata.iteritems():
      if m.size() != n:
        raise ValueError('Mismatching size for %s' % m.display_name(k))

  def num_spectra(self):
    return len(self.pkey.index)

  def num_dimensions(self):
    return None

  def get_trajectory(self, key, transformations=None):
    return self._transform_traj(self.traj[key], transformations)

  def get_trajectory_by_index(self, idx, transformations=None):
    return self.get_trajectory(self.pkey.index2key(idx), transformations)

  def get_trajectories(self, keys, transformations=None):
    return [self._transform_traj(self.traj[key], transformations)
            for key in keys]

  def get_trajectories_by_index(self, indices, transformations=None):
    return self.get_trajectories(self.pkey.index2key(indices), transformations)

  def _transform_traj(self, traj, transformations):
    if transformations is None:
      return traj

    if transformations['chan_mask']:
      raise ValueError('chan_mask transform is not applicable to TrajDataset')

    tmp = np.asarray(traj)
    copy, traj = tmp is traj, tmp

    # crop
    lb, ub = transformations['crop']
    if lb > traj[0,0]:
      idx = np.searchsorted(traj[:,0], lb)
      traj = traj[idx:]
    if ub < traj[-1,0]:
      idx = np.searchsorted(traj[:,0], ub)
      traj = traj[:idx]

    # baseline removal
    bl_obj = transformations['blr_obj']
    if bl_obj is not None:
      traj = np.array(traj, copy=copy)
      seg = transformations['blr_segmented']
      traj[:,1] = bl_obj.fit_transform(*traj.T, segment=seg)
      copy = False

    # preprocessing
    pp = transformations['pp']
    if pp:
      traj = np.array(traj, copy=copy)
      traj[:,1] = preprocess(traj[:,1:2].T, pp).ravel()
      copy = False

    # insert NaNs
    nan_gap = transformations['nan_gap']
    if nan_gap is not None:
      gap_inds, = np.where(np.diff(traj[:,0]) > nan_gap)
      traj = np.array(traj, copy=copy)
      traj[gap_inds, 1] = np.nan
    return traj


class VectorDataset(Dataset):
  def set_data(self, bands, spectra, pkey=None, **metadata):
    self.set_metadata(metadata)
    # (d,)-array of common wavelengths
    self.bands = np.asanyarray(bands)
    # (n,d)-array of intensities
    self.intensities = spectra
    assert len(self.bands) == self.intensities.shape[1]
    # add the primary key, if one exists
    self.pkey = pkey
    # make sure all our shapes match
    n = self.num_spectra()
    for k, m in self.metadata.iteritems():
      if m.size() != n:
        raise ValueError('Mismatching size for %s' % m.display_name(k))

  def num_spectra(self):
    return len(self.intensities)

  def num_dimensions(self):
    return len(self.bands)

  def get_trajectory_by_index(self, idx, transformations=None):
    x, y = self._transform_vector(self.bands, self.intensities[idx:idx+1,:],
                                  transformations)
    return np.column_stack((x, y.ravel()))

  def get_trajectories_by_index(self, indices, transformations=None):
    x, y = self._transform_vector(self.bands, self.intensities[indices,:],
                                  transformations)
    return [np.column_stack((x, yy)) for yy in y]

  def get_trajectory(self, key, transformations=None):
    if self.pkey is None:
      raise NotImplementedError('No primary key provided')
    return self.get_trajectory_by_index(self.pkey.key2index(key),
                                        transformations)

  def get_trajectories(self, keys, transformations=None):
    if self.pkey is None:
      raise NotImplementedError('No primary key provided')
    return [self.get_trajectory_by_index(self.pkey.key2index(key),
                                         transformations)
            for key in keys]

  def _transform_vector(self, bands, ints, transformations):
    if transformations is None:
      return bands, ints

    tmp = np.asarray(ints)
    copy, ints = tmp is ints, tmp

    # mask
    if transformations['chan_mask']:
      if self.kind != 'LIBS':
        raise ValueError('chan_mask transform only applicable to LIBS data')
      bands = bands[ALAMOS_MASK]
      ints = ints[:, ALAMOS_MASK]
      copy = False

    # crop
    lb, ub = transformations['crop']
    if lb > bands[0]:
      idx = np.searchsorted(bands, lb)
      bands, ints = bands[idx:], ints[:, idx:]
    if ub < bands[-1]:
      idx = np.searchsorted(bands, ub)
      bands, ints = bands[:idx], ints[:, :idx]

    # baseline removal
    bl_obj = transformations['blr_obj']
    if bl_obj is not None:
      ints = np.array(ints, copy=copy)
      seg = transformations['blr_segmented']
      ints = bl_obj.fit_transform(bands, ints, segment=seg)
      copy = False

    # preprocessing
    pp = transformations['pp']
    if pp:
      ints = preprocess(ints, pp)
      copy = False

    # insert NaNs
    nan_gap = transformations['nan_gap']
    if nan_gap is not None:
      gap_inds, = np.where(np.diff(bands) > nan_gap)
      ints = np.array(ints, copy=copy)
      ints[:, gap_inds] = np.nan
    return bands, ints
