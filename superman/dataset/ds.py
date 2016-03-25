from __future__ import absolute_import
import numpy as np

from ..preprocess import preprocess
from ..utils import ALAMOS_MASK, resample
from .metadata import is_metadata, PrimaryKeyMetadata
from .ds_view import DatasetView


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

  def resample(self, band_min, band_max, band_step):
    target_bands = np.arange(band_min, band_max, band_step)
    intensities = np.empty((self.num_spectra(), len(target_bands)))
    for i, key in enumerate(self.pkey.keys):
      intensities[i] = resample(self.traj[key], target_bands)
    ds = VectorDataset(self.name, self.kind)
    ds.set_data(target_bands, intensities, pkey=self.pkey, **self.metadata)
    return ds


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
