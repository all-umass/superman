from __future__ import absolute_import
import numpy as np

from .common import PeakFinder, _filter_peaks


class Threshold(PeakFinder):
  def __init__(self, max_peaks=None, num_stdv=4):
    self._max_peaks = max_peaks
    self._num_stdv = num_stdv

  def _fit_many(self, bands, intensities):
    mu = np.mean(intensities, axis=1)
    std = np.std(intensities, axis=1)
    thresh = mu + self._num_stdv * std
    peak_mask = intensities > thresh[:,None]
    # Use the mean of contiguous segments in the peak mask
    # Note: if peaks overlap, this will fail!
    changes = np.diff(peak_mask.astype(int))
    ret = []
    for i, c in enumerate(changes):
      idx, = c.nonzero()
      idx = (idx[::2] + idx[1::2]) // 2 + 1
      ret.append(_filter_peaks(intensities[i,idx], bands[idx],
                 max_peaks=self._max_peaks))
    return ret

