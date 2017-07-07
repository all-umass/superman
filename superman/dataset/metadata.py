from __future__ import absolute_import
import numpy as np
import operator

from ._search import parse_query

__all__ = [
    'NumericMetadata', 'BooleanMetadata', 'LookupMetadata', 'TagMetadata',
    'DateMetadata', 'CompositionMetadata', 'PrimaryKeyMetadata', 'is_metadata'
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


class _RepeatedMetadata(_BaseMetadata):
  def __init__(self, arr, dtype, display_name=None, repeats=1):
    self._display_name = display_name
    self.num_repeats = repeats
    self.arr = np.asarray(arr, dtype=dtype)

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

  def filter(self, arg):
    mask = self._filter(arg)
    if self.num_repeats == 1 or not isinstance(mask, np.ndarray):
      return mask
    return np.tile(mask, self.num_repeats)


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


class NumericMetadata(_RepeatedMetadata):
  def __init__(self, arr, step=None, display_name=None, repeats=1):
    _RepeatedMetadata.__init__(self, arr, float, display_name, repeats)
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
        step = 1.0
    self.step = step

  def _filter(self, bounds):
    # Check for the trivial case: all within bounds
    lb, ub = bounds
    tlb, tub = self.true_bounds
    eps = 0.1 * self.step
    if lb - tlb < eps and tub - ub < eps:
      return True
    return (self.arr >= lb) & (self.arr <= ub)


class DateMetadata(_RepeatedMetadata):
  def __init__(self, arr, display_name=None, repeats=1):
    _RepeatedMetadata.__init__(self, arr, np.datetime64, display_name, repeats)
    self.bounds = (self.arr.min(), self.arr.max())

  def _filter(self, bounds):
    lb, ub = np.array(bounds, dtype=np.datetime64)
    # Check for the trivial case: all within bounds
    if lb <= self.bounds[0] and ub >= self.bounds[1]:
      return True
    return (self.arr >= lb) & (self.arr <= ub)


class BooleanMetadata(_RepeatedMetadata):
  def __init__(self, arr, display_name=None, repeats=1):
    _RepeatedMetadata.__init__(self, arr, bool, display_name, repeats)

  def _filter(self, cond):
    if cond == 'yes':
      return self.arr
    elif cond == 'no':
      return ~self.arr
    return True


class TagMetadata(_RepeatedMetadata):
  def __init__(self, taglists, display_name=None, repeats=1):
    tagset = reduce(set.union, taglists, set())
    num_tags = len(tagset)
    assert num_tags <= 64, 'Too many tags for TagMetadata (%d)' % num_tags
    # find the smallest possible dtype for the bitmasks
    if num_tags <= 8:
      dtype = np.uint8
    elif num_tags <= 16:
      dtype = np.uint16
    elif num_tags <= 32:
      dtype = np.uint32
    else:
      dtype = np.uint64
    # set up mapping of tag -> bitmask
    bits, unit = dtype(1), dtype(1)
    self.tags = {}
    for tag in tagset:
      self.tags[tag] = bits
      bits <<= unit
    # convert taglists to bitmask array
    arr = np.zeros(len(taglists), dtype=dtype)
    for i, tags in enumerate(taglists):
      arr[i] = sum(self.tags[t] for t in tags)
    _RepeatedMetadata.__init__(self, arr, dtype, display_name, repeats)

  def _filter(self, tags):
    bitmask = sum(self.tags[t] for t in tags)
    if bitmask == 0 or bitmask+1 == 0:
      return True
    return np.bitwise_and(bitmask, self.arr).astype(bool)


class LookupMetadata(_BaseMetadata):
  def __init__(self, arr, display_name=None, labels=None):
    _BaseMetadata.__init__(self, display_name)
    if labels is None:
      self.uniques, self.labels = np.unique(arr, return_inverse=True)
    else:
      self.uniques = np.asarray(arr)
      self.labels = np.asarray(labels)
    # coerce all values to unicode
    if self.uniques.dtype.char not in 'US':
      self.uniques = self.uniques.astype('S')
    if self.uniques.dtype.char == 'S':
      self.uniques = np.char.decode(self.uniques, 'utf8')

  def filter(self, filt_dict):
    values = filt_dict['select']
    if values:
      values = np.char.decode(values, 'utf8')
      val_labels, = np.where(np.in1d(self.uniques, values, assume_unique=True))
      return np.in1d(self.labels, val_labels)
    # if no values are provided, try searching
    query = filt_dict['search']
    if query:
      query_fn = parse_query(query)
      search_labels, = np.where([query_fn(x) for x in self.uniques])
      return np.in1d(self.labels, search_labels)
    # if no values or query, accept everything
    return True

  def get_array(self, mask=Ellipsis):
    return self.uniques[self.labels[mask]]

  def get_index(self, idx):
    return self.uniques[self.labels[idx]]

  def size(self):
    return self.labels.shape[0]


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
    # Coerce keys to a unicode numpy array
    self.keys = np.array(arr)
    if self.keys.dtype.char not in 'US':
      self.keys = self.keys.astype('S')
    if self.keys.dtype.char == 'S':
      self.keys = np.char.decode(self.keys, 'utf8')
    # Set up the key -> index mapping
    self.index = {key:idx for idx,key in enumerate(self.keys)}
    assert len(self.index) == len(self.keys), 'Primary key array not unique'

  def key2index(self, key):
    return self.index[key]

  def index2key(self, idx):
    return self.keys[idx]

  def filter(self, filt_dict):
    keys = filt_dict['select']
    if keys:
      keys = np.char.decode(keys, 'utf8')
      inds = [self.index[k] for k in keys]
      mask = np.zeros_like(self.keys, dtype=bool)
      mask[inds] = True
      return mask
    # if no keys are provided, try searching
    query = filt_dict['search']
    if query:
      query_fn = parse_query(query)
      return np.array([query_fn(x) for x in self.keys], dtype=bool)
    # if no keys or query, accept everything
    return True

  def size(self):
    return self.keys.shape[0]
