from __future__ import absolute_import
import numpy as np
import os.path
import h5py

from .ds import TrajDataset, VectorDataset
from .metadata import *
from ..utils import ISHIKAWA_MINERALS


def load_dataset(hdf_file, resample=False):
  ds_name, _ = os.path.splitext(os.path.basename(hdf_file))
  data = h5py.File(hdf_file, 'r')
  names = data['/meta/sample']
  species = np.array([n.rsplit('-', 1)[0] for n in names])
  ishi = BooleanMetadata([s in ISHIKAWA_MINERALS for s in species])
  ds = TrajDataset('<spectra>', ds_name)
  ds.set_data(names, data['/spectra'], ishikawa=ishi,
              minerals=LookupMetadata(species),
              rruff_ids=LookupMetadata(data['/meta/rruff_id']),
              lasers=LookupMetadata(data['/meta/laser']))
  if resample:
    ds = ds.resample(85, 1800, 1)
  return ds


def dataset_views(ds, opts, **filter_conditions):
  if opts.ishikawa:
    filter_conditions['ishikawa'] = 'yes'
  for laser in opts.laser:
    filter_conditions['lasers'] = [] if laser == 'all' else [laser]
    mask = ds.filter_metadata(filter_conditions)
    for pp in opts.pp:
      yield ds.view(mask=mask, pp=pp)
