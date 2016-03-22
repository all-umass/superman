from __future__ import absolute_import
import numpy as np
import scipy.signal

from .common import PeakFinder, _filter_peaks


class Wavelet(PeakFinder):
  def __init__(self, max_peaks=None, width_min=1, width_max=9):
    self._max_peaks = max_peaks
    self._widths = np.arange(width_min, width_max+1)

  def _fit_one(self, bands, intensities):
    # TODO: look into the source to see if we can vectorize this
    # https://github.com/scipy/scipy/blob/master/scipy/signal/_peak_finding.py
    # TODO: tune kwargs to get better peaks
    peak_inds = scipy.signal.find_peaks_cwt(intensities, self._widths)
    peak_locs = bands[peak_inds]
    return _filter_peaks(intensities[peak_inds], bands[peak_inds],
                         max_peaks=self._max_peaks)

