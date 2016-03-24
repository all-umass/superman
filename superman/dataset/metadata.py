import numpy as np
import operator

__all__ = [
    'NumericMetadata', 'BooleanMetadata', 'LookupMetadata',
    'CompositionMetadata', 'PrimaryKeyMetadata', 'is_metadata'
]


class _BaseMetadata(object):
  '''Abstract base class for metadata
  Required methods: filter, size
  Optional methods: get_array, get_index
  '''
  def __init__(self, display_name):
    self._display_name = display_name

  def display_name(self, key):
    if self._display_name is None:
      return key
    return self._display_name


def is_metadata(obj):
  return isinstance(obj, _BaseMetadata)


def _round(x, d, up=True):
  y = np.round(x, decimals=d)
  if up:
    # rounding up
    if y < x:
      y += 10**-d
  else:
    # rounding down
    if y > x:
      y -= 10**-d
  return y


def _choose_step(ptp, target_number=100):
  step = float(ptp) / target_number
  # check for floating point BS
  while ptp / step < target_number:
    step -= 1e-8
  return step


class NumericMetadata(_BaseMetadata):
  def __init__(self, arr, step=None, display_name=None, repeats=1):
    _BaseMetadata.__init__(self, display_name)
    self.num_repeats = repeats
    self.arr = np.asarray(arr)
    self.true_bounds = (np.nanmin(self.arr), np.nanmax(self.arr))
    # compute the displayed bounds
    ptp = self.true_bounds[1] - self.true_bounds[0]
    if np.issubdtype(self.arr.dtype, np.integer):
      self.bounds = map(int, self.true_bounds)
      if step is None:
        step = 1
    elif 0 < ptp < 1e99:
      # number of decimal places to care about
      d = 1 - int(np.floor(np.log10(ptp)))
      self.bounds = (_round(self.true_bounds[0], d, up=False),
                     _round(self.true_bounds[1], d, up=True))
      if step is None:
        step = _choose_step(self.bounds[1] - self.bounds[0])
    else:
      # very odd cases fall here
      self.bounds = self.true_bounds
      if step is None:
        step = _choose_step(self.true_bounds[1] - self.true_bounds[0])
    self.step = step

  def filter(self, bounds):
    # Check for the trivial case: all within bounds
    lb, ub = bounds
    tlb, tub = self.true_bounds
    if lb <= tlb and ub >= tub:
      return True
    mask = (self.arr >= lb) & (self.arr <= ub)
    if self.num_repeats == 1:
      return mask
    return np.tile(mask, self.num_repeats)

  def get_array(self, mask=Ellipsis):
    if self.num_repeats == 1:
      return self.arr[mask]
    if mask is Ellipsis:
      return np.tile(self.arr, self.num_repeats)
    parts = [self.arr[m] for m in mask.reshape((self.num_repeats, -1))]
    return np.hstack(parts)

  def get_index(self, idx):
    return self.arr[idx % self.num_repeats]

  def size(self):
    return self.arr.shape[0] * self.num_repeats


class LookupMetadata(_BaseMetadata):
  def __init__(self, arr, display_name=None, labels=None):
    _BaseMetadata.__init__(self, display_name)
    if labels is None:
      self.uniques, self.labels = np.unique(arr, return_inverse=True)
    else:
      self.uniques = np.asarray(arr)
      self.labels = np.asarray(labels)

  def filter(self, values):
    if values:
      val_labels, = np.where(np.in1d(self.uniques, values, assume_unique=True))
      return np.in1d(self.labels, val_labels)
    # if no values are provided, accept all of them
    return True

  def get_array(self, mask=Ellipsis):
    return self.uniques[self.labels[mask]]

  def get_index(self, idx):
    return self.uniques[self.labels[idx]]

  def size(self):
    return self.labels.shape[0]


class BooleanMetadata(NumericMetadata):
  def __init__(self, arr, display_name=None, repeats=1):
    _BaseMetadata.__init__(self, display_name)
    self.num_repeats = repeats
    self.arr = np.asarray(arr, dtype=bool)

  def filter(self, cond):
    if cond == 'yes':
      mask = self.arr
    elif cond == 'no':
      mask = ~self.arr
    else:
      return True
    if self.num_repeats == 1:
      return mask
    return np.tile(mask, self.num_repeats)


class CompositionMetadata(_BaseMetadata):
  def __init__(self, comps, display_name=None):
    '''comps is a dict from name -> NumericMetadata'''
    _BaseMetadata.__init__(self, display_name)
    self.comps = comps
    assert len(comps) > 0, 'Must provide at least one composition'
    # Validate
    comp_sizes = []
    for name, m in comps.items():
      if not isinstance(m, NumericMetadata):
        raise ValueError('composition %r (%r) is not NumericMetadata' %
                         (name, type(m)))
      comp_sizes.append(m.size())
    if not len(set(comp_sizes)) == 1:
      raise ValueError('Mismatching sizes in %s' % self.display_name('comp'))
    self._size = comp_sizes[0]

  def size(self):
    return self._size

  def filter(self, sub_conds):
    sub_filters = (self.comps[key].filter(cond)
                   for key, cond in sub_conds.items())
    return reduce(operator.and_, sub_filters)


class PrimaryKeyMetadata(_BaseMetadata):
  def __init__(self, arr):
    _BaseMetadata.__init__(self, 'Primary Key')
    # Coerce keys to a numpy array, to prevent HDF5 issues
    self.keys = np.array(arr)
    self.index = {key:idx for idx,key in enumerate(self.keys)}
    assert len(self.index) == len(self.keys), 'Primary key array not unique'

  def key2index(self, key):
    return self.index[key]

  def index2key(self, idx):
    return self.keys[idx]

  def filter(self, keys):
    if not keys:
      return True
    inds = [self.index[k] for k in keys]
    mask = np.zeros_like(self.keys, dtype=bool)
    mask[inds] = True
    return mask

  def size(self):
    return self.keys.shape[0]
