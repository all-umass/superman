import numpy as np

from metadata import is_metadata, PrimaryKeyMetadata

__all__ = ['VectorDataset', 'TrajDataset']


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

  def __str__(self):
    return '%s [%s]' % (self.name, self.kind)

  def get_trajectories_by_index(self, indices):
    return map(self.get_trajectory_by_index, indices)

  def get_trajectories(self, keys):
    return map(self.get_trajectory, keys)


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

  def get_trajectory(self, key):
    return np.asarray(self.traj[key])

  def get_trajectory_by_index(self, idx):
    return self.get_trajectory(self.pkey.index2key(idx))


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

  def get_trajectory(self, key):
    if self.pkey is None:
      raise NotImplementedError('No primary key provided')
    return self.get_trajectory_by_index(self.pkey.key2index(key))

  def get_trajectory_by_index(self, idx):
    return np.column_stack((self.bands, self.intensities[idx]))
