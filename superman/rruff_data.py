from __future__ import absolute_import
import numpy as np
import os.path
import h5py

from .dataset import TrajDataset, LookupMetadata, BooleanMetadata
from .utils import ISHIKAWA_MINERALS


def load_dataset(opts):
  ds_name, _ = os.path.splitext(os.path.basename(opts.data))
  data = h5py.File(opts.data, 'r')
  names = data['/meta/sample']
  species = np.array([n.rsplit('-', 1)[0] for n in names])
  ishi = BooleanMetadata([s in ISHIKAWA_MINERALS for s in species])
  ds = TrajDataset(ds_name, '<spectra>')
  ds.set_data(names, data['/spectra'], ishikawa=ishi,
              minerals=LookupMetadata(species),
              rruff_ids=LookupMetadata(data['/meta/rruff_id']),
              lasers=LookupMetadata(data['/meta/laser']))
  if opts.resample:
    ds = ds.resample(opts.band_min, opts.band_max, opts.band_step)
  return ds


def dataset_views(ds, opts, **filter_conditions):
  if opts.ishikawa:
    filter_conditions['ishikawa'] = 'yes'
  for laser in opts.laser:
    filter_conditions['lasers'] = [] if laser == 'all' else [laser]
    mask = ds.filter_metadata(filter_conditions)
    for pp in opts.pp:
      yield ds.view(mask=mask, pp=pp)
