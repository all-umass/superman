import numpy as np

from common import PeakFinder, _filter_peaks
from preprocess import savitzky_golay


class SavitzkyGolay(PeakFinder):
  def __init__(self, max_peaks=None, min_interpeak_separation=10,
               peak_percentile=80):
    self._max_peaks = max_peaks
    self._min_interpeak = min_interpeak_separation
    self._peak_percentile = peak_percentile

  def _fit_many(self, bands, intensities):
    deriv = savitzky_golay(intensities, deriv=1)
    # possible sign change values are [-2,0,2]
    sign_change = np.diff(np.sign(deriv).astype(int))
    all_peaks = []
    for i,dy in enumerate(deriv):
      # find indices where the deriv crosses zero
      maxima, = np.where(sign_change[i] == 2)
      # interpolate to find true peak locations and intensities
      peak_locs = maxima + dy[maxima] / (dy[maxima] - dy[maxima+1])
      peak_ints = np.interp(peak_locs, bands, intensities[i])
      # filter out bad peaks
      all_peaks.append(_filter_peaks(peak_ints, peak_locs,
                                     height_pct=self._peak_percentile,
                                     min_interpeak=self._min_interpeak,
                                     max_peaks=self._max_peaks))
    return all_peaks

