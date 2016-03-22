from __future__ import absolute_import
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy.stats import anderson
from .common import Baseline


def lsd_baseline(bands, intensities):
  '''Perform local second derivative (LSD) baseline removal.
  "Automated algorithm for baseline subtraction in spectra", Rowlands & Elliott
  '''
  # Step 1: smooth signal with "optimal spline"
  y = _spline_smooth(bands, intensities)
  # Step 2: estimate gamma using second derivative of y
  alpha = 1.70834  # window size of moving average
  gamma = 1  # HWHM of Lorentzian
  A = 1./np.pi  # peak amplitude of Lorentzian
  # Step 3: compute LSD
  pass


def _spline_smooth(bands, intensities):
  '''Implements Rowlands & Elliott 2010,
  "Denoising of spectra with no user input: a spline-smoothing algorithm"
  Note: doesn't work very well, may have bugs.'''
  def objective(s):
    spline = UnivariateSpline(bands, intensities, k=3, s=s)
    a = anderson(intensities - spline(bands))[0]
    return a
  # start at an estimate of s given by the SNR (Schulze et al. 2006)
  # s_init = np.diff(intensities, n=2).std() / np.sqrt(6)

  # The objective is very non-smooth and non-convex, so this doesn't work
  # very well. Also s_init is a terrible starting point.
  # TODO: see if we can find the original paper's implementation.
  res = minimize_scalar(objective, bounds=(10, len(bands)))
  spline = UnivariateSpline(bands, intensities, k=3, s=res.x)
  return spline(bands)


class LSD(Baseline):
  def __init__(self):
    pass

  def _fit_one(self, bands, intensities):
    return lsd_baseline(bands, intensities)

  def param_ranges(self):
    return {}
