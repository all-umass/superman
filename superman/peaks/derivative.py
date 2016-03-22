from __future__ import absolute_import
import numpy as np

from .common import PeakFinder, _filter_peaks


class Derivative(PeakFinder):
  def __init__(self, max_peaks=None, min_interpeak_separation=10,
               peak_percentile=80):
    self._max_peaks = max_peaks
    self._min_interpeak = min_interpeak_separation
    self._peak_percentile = peak_percentile

  def _fit_many(self, bands, intensities):
    num_channels = len(bands)
    dx_all = np.diff(intensities, axis=1)
    ret = []
    for i,dx in enumerate(dx_all):
      ind, = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))
      # can't have peaks at beginning or end
      if ind.size and ind[0] == 0:
        ind = ind[1:]
      if ind.size and ind[-1] == num_channels-1:
        ind = ind[:-1]
      ret.append(_filter_peaks(intensities[i,ind], bands[ind],
                               height_pct=self._peak_percentile,
                               min_interpeak=self._min_interpeak,
                               max_peaks=self._max_peaks))
    return ret
