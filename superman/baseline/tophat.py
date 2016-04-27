from __future__ import absolute_import
import numpy as np
from itertools import count
from scipy.ndimage import grey_opening, grey_erosion, grey_dilation
from .common import Baseline


def tophat_baseline(intensities):
  '''Perform "tophat" baseline removal, from the paper: Morphology-Based
  Automated Baseline Removal for Raman Spectra of Artistic Pigments.
  Perez-Pueyo et al., Appl. Spec. 2010'''
  # find the optimal window length
  old_b, num_equal = 0, 1
  for window_length in count(start=3, step=2):
    b1 = grey_opening(intensities, window_length)
    if np.allclose(b1, old_b):
      if num_equal == 2:
        break
      num_equal += 1
    else:
      num_equal = 1
    old_b = b1
  # use the smallest of the three equivalent window lengths
  window_length -= 4

  # compute another estimate of the baseline
  b2 = 0.5 * (grey_dilation(intensities, window_length) +
              grey_erosion(intensities, window_length))

  # combine the two estimates
  return np.minimum(b1, b2)


class Tophat(Baseline):
  def _fit_one(self, bands, intensities):
    return tophat_baseline(intensities)

  def param_ranges(self):
    return {}
