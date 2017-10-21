from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.preprocessing import normalize


def crop_resample(bands, intensities, crops):
  intensities = np.atleast_2d(intensities)
  crops = sorted(crops)
  # check that each chunk is valid and doesn't overlap with any other
  prev_ub = float('-inf')
  for lb, ub, step in crops:
    if ub <= lb:
      raise ValueError('Invalid crop region')
    if lb < prev_ub:
      raise ValueError('Overlapping crop regions')
    prev_ub = ub
  # do all the band lookups at once
  locs = sorted(set(c[0] for c in crops).union(set(c[1] for c in crops)))
  idxs = np.searchsorted(bands, locs)
  loc_idxs = dict(zip(locs, idxs))
  # crop/resample each chunk separately
  xs, ys = [], []
  for lb, ub, step in crops:
    s = slice(loc_idxs[lb], loc_idxs[ub])
    x = bands[s]
    if step > 0:
      lb = lb if np.isfinite(lb) else x[0]
      ub = ub if np.isfinite(ub) else x[-1] + step
      x_new = np.arange(lb, ub, step)
      y_new = np.row_stack([np.interp(x_new, x, y) for y in intensities[:, s]])
      xs.append(x_new)
      ys.append(y_new)
    else:
      xs.append(x)
      ys.append(intensities[:, s])
  # glue all the chunks back together
  return np.concatenate(xs), np.hstack(ys)


def cumulative_norm(S):
  '''Cumulative intensity normalization method.

  "Quality Assessment of Tandem Mass Spectra Based on
   Cumulative Intensity Normalization", Na & Paek, J. of Proteome Research
  '''
  idx = np.arange(S.shape[0])[:,None]
  ranks = np.argsort(S, axis=1)
  cumsums = np.cumsum(S[idx,ranks], axis=1)
  unranks = np.zeros_like(ranks)
  unranks[idx,ranks] = np.arange(S.shape[1])
  S = cumsums[idx,unranks]
  S /= cumsums[:,-1:]
  return S


def libs_norm3(shots, wavelengths=None, copy=True):
  shots = np.array(shots, copy=copy, ndmin=2)
  num_chan = shots.shape[1]
  if num_chan == 6143:
    a, b = 2047, 4097
  elif num_chan == 6144:
    a, b = 2048, 4098
  elif num_chan == 5485:
    a, b = 1884, 3811
  elif wavelengths is not None:
    a, b = np.searchsorted(wavelengths, (360, 470))
  else:
    raise ValueError('Invalid # channels for LIBS norm3 method: %d' % num_chan)
  normalize(shots[:, :a], norm='l1', copy=False)
  normalize(shots[:,a:b], norm='l1', copy=False)
  normalize(shots[:, b:], norm='l1', copy=False)
  return shots
