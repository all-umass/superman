import numpy as np


class PeakFinder(object):
  def _fit_one(self, bands, intensities):
    '''bands: array of length n
       intensities: array of length n
       Returns array of peak locations
    '''
    raise NotImplementedError()

  def _fit_many(self, bands, intensities):
    '''bands: array of length n
       intensities: 2d array of shape (k, n)
       Returns a list of peak location arrays
    '''
    # Fallback implementation based on _fit_one()
    return [self._fit_one(bands, y) for y in intensities]

  def param_ranges(self):
    '''Returns a dict of parameter -> (min,max,scale) mappings.
    Min and max are scalars, scale is one of {'linear','log','integer'}.'''
    raise NotImplementedError()

  def fit(self, bands, intensities):
    '''Finds peaks in each spectrum and stores them as self.peak_locs'''
    if intensities.ndim == 1:
      self.peak_locs = self._fit_one(bands, intensities)
    else:
      self.peak_locs = self._fit_many(bands, intensities)
    return self


def _filter_peaks(peak_ints, peak_locs, height_pct=None, min_interpeak=None,
                  max_peaks=None):
  if height_pct is not None:
    # cut out short peaks
    mask = peak_ints >= np.percentile(peak_ints, height_pct)
    peak_locs = peak_locs[mask]
    peak_ints = peak_ints[mask]

  if min_interpeak is not None:
    # cut out close peaks
    peak_sep = np.ediff1d(peak_locs, to_begin=[min_interpeak])
    mask = peak_sep >= min_interpeak
    peak_locs = peak_locs[mask]
    peak_ints = peak_ints[mask]

  if max_peaks is not None and len(peak_locs) > max_peaks:
    # use the highest k peaks
    top_k = np.argsort(peak_ints)[-max_peaks:]
    peak_locs = peak_locs[top_k]

  return peak_locs

