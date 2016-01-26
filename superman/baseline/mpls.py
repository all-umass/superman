import numpy as np
from scipy.ndimage import grey_opening
from common import WhittakerSmoother, Baseline


def mpls_baseline(intensities, smoothness_param=100, deriv_order=1,
                  window_length=100):
  '''Perform morphological weighted penalized least squares baseline removal.
  * paper: DOI: 10.1039/C3AN00743J (Paper) Analyst, 2013, 138, 4483-4492
  * Matlab code: https://code.google.com/p/mpls/

  smoothness_param: Relative importance of smoothness of the predicted response.
  deriv_order: Polynomial order of the difference of penalties.
  window_length: size of the structuring element for the open operation.
  '''
  Xbg = grey_opening(intensities, window_length)
  # find runs of equal values in Xbg
  flat = (np.diff(Xbg) != 0).astype(np.int8)
  run_idx, = np.where(np.diff(flat))
  # set local minimums between flat runs to 1
  w = np.zeros_like(intensities)
  for i in xrange(1, len(run_idx)-1, 2):
    s = run_idx[i] + 2
    t = run_idx[i+1] + 1
    if t > s:
      w[s+np.argmin(Xbg[s:t])] = 1
    else:
      w[s] = 1
  # make sure we stick to the ends
  w[0] = 5
  w[-1] = 5
  # run one iteration of smoothing
  smoother = WhittakerSmoother(Xbg, smoothness_param,
                               deriv_order=deriv_order)
  return smoother.smooth(w)


class MPLS(Baseline):
  def __init__(self, smoothness_param=1e6, deriv_order=1, window_length=100):
    self.smoothness_ = smoothness_param
    self.window_ = window_length
    self.order_ = deriv_order

  def _fit_one(self, bands, intensities):
    return mpls_baseline(intensities, smoothness_param=self.smoothness_,
                         deriv_order=self.order_, window_length=self.window_)

  def param_ranges(self):
    return {
        'smoothness_': (1, 1e5, 'log'),
        'window_': (3, 500, 'integer'),
        'order_': (1, 2, 'integer'),
    }
