from __future__ import absolute_import
import numpy as np

from ..preprocess import preprocess, crop_resample
from ..utils import ALAMOS_MASK
from .ds_view import DatasetView
from ._search import parse_query
from .metadata import (
    is_metadata, PrimaryKeyMetadata, LookupMetadata, TagMetadata)


class Dataset(object):
  def __init__(self, name, spec_kind):
    self.name = name
    self.kind = spec_kind
    self.pkey = None  # primary key metadata, optional
    self.metadata = dict()

  def set_metadata(self, metadata_dict):
    self.metadata = metadata_dict
    # do a little metadata validation
    for key, m in self.metadata.items():
      if not is_metadata(m):
        raise ValueError('%r is not a valid Metadata' % key)
      if self.pkey is not None and isinstance(m, PrimaryKeyMetadata):
        raise ValueError("%s can't have >1 primary key: %s" % (self, key))

  def filter_metadata(self, filter_conditions):
    mask = np.ones(self.num_spectra(), dtype=bool)
    for key, cond in filter_conditions.items():
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

  def search_metadata(self, query_str, full_text=False, case_sensitive=False):
    query_fn = parse_query(query_str, case_sensitive=case_sensitive)
    if query_fn is None:
      return []

    if not full_text:
      names = [m.display_name(key) for key, m in self.metadata.items()]
      return list(filter(query_fn, names))

    texts = []
    if self.pkey is not None and any(query_fn(k) for k in self.pkey.keys):
      texts.append('Primary Key')

    for key, meta in self.metadata.items():
      res = False
      if isinstance(meta, LookupMetadata):
        res = any(query_fn(x) for x in meta.uniques)
      elif isinstance(meta, TagMetadata):
        res = any(query_fn(x) for x in meta.tags)
      if res:
        texts.append(meta.display_name(key))
    return texts

  def _transform_traj(self, traj, transformations):
    if transformations is None:
      return traj

    if transformations['chan_mask']:
      raise ValueError('chan_mask transform is not applicable to TrajDataset')

    tmp = np.asarray(traj)
    copy, traj = tmp is traj, tmp

    # crop / resample (but keep the trajectory format)
    crops = transformations['crop']
    if crops:
      x, y = crop_resample(traj[:,0], traj[:,1], crops)
      traj = np.column_stack((x, y[0]))
      copy = False

    # baseline removal
    bl_obj = transformations['blr_obj']
    if bl_obj is not None:
      traj = np.array(traj, copy=copy)
      seg = transformations['blr_segmented']
      inv = transformations['blr_inverted']
      traj[:,1] = bl_obj.fit_transform(*traj.T, segment=seg, invert=inv)
      copy = False

    # y-axis flip
    if transformations['flip']:
      traj = np.array(traj, copy=copy)
      traj[:,1] *= -1
      copy = False

    # preprocessing
    pp = transformations['pp']
    if pp:
      traj = np.array(traj, copy=copy)
      traj[:,1] = preprocess(traj[:,1:2].T, pp, wavelengths=traj[:,0],
                             copy=False).ravel()
      copy = False

    # insert NaNs
    nan_gap = transformations['nan_gap']
    if nan_gap is not None:
      gap_inds, = np.where(np.diff(traj[:,0]) > nan_gap)
      traj = np.array(traj, copy=copy)
      traj[gap_inds, 1] = np.nan
    return traj

  def _transform_vector(self, bands, ints, transformations):
    if transformations is None:
      return bands, ints

    tmp = np.asarray(ints)
    copy, ints = tmp is ints, tmp

    # mask
    if transformations['chan_mask']:
      if self.kind != 'LIBS' or len(bands) != 6144:
        raise ValueError('chan_mask transform is only applicable to '
                         'LIBS data with 6144 channels')
      bands = bands[ALAMOS_MASK]
      ints = ints[:, ALAMOS_MASK]
      copy = False

    # crop/resample
    crops = transformations['crop']
    if crops:
      bands, ints = crop_resample(bands, ints, crops)
      copy = False

    # baseline removal
    bl_obj = transformations['blr_obj']
    if bl_obj is not None:
      ints = np.array(ints, copy=copy)
      seg = transformations['blr_segmented']
      inv = transformations['blr_inverted']
      ints = bl_obj.fit_transform(bands, ints, segment=seg, invert=inv)
      copy = False

    # y-axis flip
    if transformations['flip']:
      ints = np.array(ints, copy=copy)
      ints *= -1
      copy = False

    # preprocessing
    pp = transformations['pp']
    if pp:
      ints = preprocess(ints, pp, wavelengths=bands, copy=copy)
      copy = False

    # insert NaNs
    nan_gap = transformations['nan_gap']
    if nan_gap is not None:
      gap_inds, = np.where(np.diff(bands) > nan_gap)
      ints = np.array(ints, copy=copy)
      ints[:, gap_inds] = np.nan
    return bands, ints


class TrajDataset(Dataset):
  def set_data(self, keys, traj_map, **metadata):
    # list of names, or other keys
    self.pkey = PrimaryKeyMetadata(keys)
    # other metadata
    self.set_metadata(metadata)
    # mapping from key to (n,2)-array
    self.traj = traj_map
    for key in keys:
      s = traj_map[key]
      assert s[0,0] <= s[1,0], 'Backwards bands in %s: %s' % (self, key)
    # make sure all our shapes match
    n = self.num_spectra()
    for k, m in self.metadata.items():
      if m.size() != n:
        raise ValueError('Mismatching size for %s' % m.display_name(k))

  def clear_data(self):
    self.pkey = None
    self.metadata = {}
    self.traj = []

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


class VectorDataset(Dataset):
  def set_data(self, bands, spectra, pkey=None, **metadata):
    # add the primary key, if one exists
    self.pkey = pkey
    # set the rest of the metadata
    self.set_metadata(metadata)
    # (d,)-array of common wavelengths
    self.bands = np.asanyarray(bands)
    # (n,d)-array of intensities
    self.intensities = spectra
    assert len(self.bands) == self.intensities.shape[1]
    # make sure all our shapes match
    n = self.num_spectra()
    for k, m in self.metadata.items():
      if m.size() != n:
        raise ValueError('Mismatching size for %s' % m.display_name(k))

  def clear_data(self):
    self.pkey = None
    self.metadata = {}
    self.bands = None
    self.intensities = None

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
