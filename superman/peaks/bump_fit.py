from __future__ import print_function, absolute_import, division
import numpy as np
import scipy.optimize
import scipy.integrate
from itertools import combinations_with_replacement
from six.moves import xrange

from .common import PeakFinder


class BumpFit(PeakFinder):
  def __init__(self, max_peaks=None, peak_percentile=80,
               fit_kind='lorentzian', max_iter=10):
    self._max_peaks = max_peaks
    self._peak_percentile = peak_percentile
    self._fit_kind = fit_kind
    self._max_iter = max_iter

  def _fit_one(self, bands, intensities):
    max_peaks = self._max_peaks if self._max_peaks is not None else len(bands)
    cutoff = np.percentile(intensities, self._peak_percentile)
    Y = intensities.copy()
    max_idx = np.argmax(Y)
    ret = []
    for _ in xrange(max_peaks):
      peak_mask, peak_y, peak_data = fit_single_peak(
          bands, intensities, bands[max_idx], fit_kind=self._fit_kind,
          max_iter=self._max_iter, log_fn=_dummy)
      ret.append(peak_data['center'])
      Y[peak_mask] -= peak_y
      max_idx = np.argmax(Y)
      if Y[max_idx] < cutoff:
        break
    return np.array(ret)


def fit_single_peak(bands, intensities, loc, fit_kind='lorentzian',
                    max_iter=10, log_fn=print, band_resolution=1,
                    loc_fixed=False):
  """Fit a peak to a spectral feature by iterated least-squares curve-fitting.

  bands : array-like
    X-axis values of the spectrum.

  intensities : array-like
    Y-axis values of the spectrum, same shape as bands.

  loc : float
    Guess at the location (in x-axis units) of the peak center.

  fit_kind : str, optional
    Type of peak to fit, may be 'lorentzian' or 'gaussian'.

  max_iter : int, optional
    Maximum number of iterations of weighted least squares curve fitting.

  log_fn : callable, optional
    Function which accepts a single str argument, used for log messages.

  band_resolution : int, optional
    Estimation of a useful delta-x.

  loc_fixed : bool, optional
    When True, only runs one iteration of fitting and doesn't allow the center
    of the fitted peak to move from the given value (loc).
  """
  # deal with bad scaling
  intensities, scale = _scale_spectrum(intensities)

  # Get the appropriate function to fit
  fit_func = _get_peak_function(fit_kind, loc, loc_fixed)

  # Choose reasonable starting parameters: (loc, area, fwhm)
  area_guess = _guess_area(bands, intensities, loc)
  params = (loc, area_guess, 2 * band_resolution)
  if loc_fixed:
    params = params[1:]
    max_iter = 1

  # Keep fitting until loc (x0) converges, or just one round for loc_fixed
  params, pstd = _weighted_curve_fit(bands, intensities, loc, fit_func, params,
                                     max_iter=max_iter, log_fn=log_fn,
                                     log_label=fit_kind,
                                     band_resolution=band_resolution)

  # Calculate peak info
  if loc_fixed:
    area, fwhm = map(float, params)
    area_std, fwhm_std = map(float, pstd)
    loc_std = 0
    height = float(fit_func(loc, area, fwhm))
  else:
    loc, area, fwhm = map(float, params)
    loc_std, area_std, fwhm_std = map(float, pstd)
    height = float(fit_func(loc, loc, area, fwhm))

  # Select channels in the top 99% of intensity
  mask, peak_x, peak_y = _select_top99(bands, fit_func, params)

  # restore original scaling
  peak_y *= scale
  area *= scale
  area_std *= scale
  height *= scale

  peak_data = dict(xmin=float(peak_x[0]), xmax=float(peak_x[-1]), height=height,
                   center=loc, area=area, fwhm=fwhm, center_std=loc_std,
                   area_std=area_std, fwhm_std=fwhm_std)
  return mask, peak_y, peak_data


def fit_composite_peak(bands, intensities, locs, num_peaks=2, max_iter=10,
                       fit_kinds=('lorentzian', 'gaussian'), log_fn=print,
                       band_resolution=1):
  """Fit several peaks to a single spectral feature.

  locs : sequence of float
    Contains num_peaks peak-location guesses,
    or a single feature-location guess.

  fit_kinds : sequence of str
    Specifies all the peak types that the composite may be made of.
    Not all fit_kinds are guaranteed to appear in the final composite fit.

  See fit_single_peak for details about the other arguments.
  """
  # deal with bad scaling
  intensities, scale = _scale_spectrum(intensities)

  # get the appropriate function(s) to fit
  fit_funcs = {k: _get_peak_function(k, None, False) for k in fit_kinds}

  # find reasonable approximations for initial parameters: (loc, area, fwhm)
  if len(locs) == num_peaks:
    loc_guesses = locs
  elif len(locs) == 1:
    loc_guesses = np.linspace(locs[0]-band_resolution, locs[0]+band_resolution,
                              num_peaks)
  else:
    raise ValueError('Number of locs (%d) != number of peaks (%d)' % (
                     len(locs), num_peaks))
  mean_loc = np.mean(locs)
  area_guess = _guess_area(bands, intensities, mean_loc) / num_peaks
  fwhm_guess = 2 * band_resolution / num_peaks
  init_params = (tuple(loc_guesses) +
                 (area_guess,) * num_peaks +
                 (fwhm_guess,) * num_peaks)
  loc_idx = slice(0, num_peaks)

  # try all combinations of peaks, use the one that matches best
  combs = []
  for fit_keys in combinations_with_replacement(fit_funcs, num_peaks):
    label = '+'.join(fit_keys)
    fit_func = _combine_peak_functions([fit_funcs[k] for k in fit_keys])
    params, pstd = _weighted_curve_fit(
        bands, intensities, mean_loc, fit_func, init_params,
        max_iter=max_iter, log_fn=log_fn, log_label=label,
        band_resolution=band_resolution, loc_idx=loc_idx)
    mask, peak_x, peak_y = _select_top99(bands, fit_func, params)
    residual = np.linalg.norm(peak_y - intensities[mask])
    log_fn('composite %s residual: %g' % (label, residual))
    combs.append((residual, fit_keys, fit_func, params, pstd,
                  mask, peak_x, peak_y))
  residual, fit_keys, fit_func, params, pstd, mask, peak_x, peak_y = min(combs)

  # Calculate peak info, with original scaling
  peak_data = dict(xmin=float(peak_x[0]), xmax=float(peak_x[-1]),
                   fit_kinds=fit_keys, height=[], center=[], area=[], fwhm=[],
                   center_std=[], area_std=[], fwhm_std=[])
  peak_ys = [peak_y * scale]
  for i, k in enumerate(fit_keys):
    fn = fit_funcs[k]
    loc, area, fwhm = map(float, params[i::num_peaks])
    loc_std, area_std, fwhm_std = map(float, pstd[i::num_peaks])
    peak_ys.append(fn(peak_x, loc, area, fwhm) * scale)
    height = float(fn(loc, loc, area, fwhm))
    peak_data['height'].append(height * scale)
    peak_data['center'].append(loc)
    peak_data['center_std'].append(loc_std)
    peak_data['area'].append(area * scale)
    peak_data['area_std'].append(area_std * scale)
    peak_data['fwhm'].append(fwhm)
    peak_data['fwhm_std'].append(fwhm_std)

  peak_y *= scale
  return mask, peak_ys, peak_data


def _weighted_curve_fit(bands, intensities, loc, fit_func, params,
                        max_iter=10, log_fn=print, log_label='',
                        band_resolution=1, loc_idx=0):
  # initial value, in case none of the curve_fit calls succeed
  pcov = np.zeros((len(params),) * 2)
  # Keep fitting until loc (x0) converges
  log_fn('Starting %s: params=%s' % (log_label, params))
  for i in xrange(max_iter):
    # Weight the channels based on distance from the approx. peak loc
    w = 1 + ((bands - loc)/band_resolution)**2
    try:
      params, pcov = curve_fit(fit_func, bands, intensities, p0=params, sigma=w)
    except RuntimeError as e:
      if e.message.startswith('Optimal parameters not found'):
        log_fn('%s fit #%d: %s' % (log_label, i+1, e.message))
        break
      raise e
    log_fn('%s fit #%d: params=%s' % (log_label, i+1, params.tolist()))
    # Check for convergence in peak location
    new_loc = params[loc_idx].mean()
    if max_iter == 1 or abs(loc - new_loc) < 1.0:
      break
    loc = new_loc
  else:
    log_fn('_weighted_curve_fit failed to converge in %d iterations' % max_iter)
  return params, np.sqrt(np.diag(pcov))


def _select_top99(bands, fit_func, params):
  fit_data = fit_func(bands, *params)
  # TODO: decide if np.percentile(fit_data, 1) is a better cutoff
  cutoff = fit_data.min()*0.99 + fit_data.max()*0.01
  mask = fit_data >= cutoff
  peak_x = bands[mask]
  if len(peak_x) < 2:
    cutoff = np.partition(fit_data, -2)[-2]
    mask = fit_data >= cutoff
    peak_x = bands[mask]
  peak_y = fit_data[mask]
  return mask, peak_x, peak_y


def _guess_area(bands, intensities, loc):
  idx = np.searchsorted(bands, loc)
  i, j = max(0, idx-4), idx+5
  area_guess = scipy.integrate.trapz(intensities[i:j], bands[i:j])
  return max(0, area_guess)


def _scale_spectrum(intensities):
  scale = intensities.max()
  if scale >= 100:
    intensities = intensities / scale
  else:
    scale = 1
  return intensities, scale


def _get_peak_function(fit_kind, loc, loc_fixed):
  if fit_kind == 'lorentzian':
    if loc_fixed:
      def peak(x, A, w):
        return (2*A*w)/(np.pi*(4*(x-loc)**2+w**2))
    else:
      def peak(x, x0, A, w):
        return (2*A*w)/(np.pi*(4*(x-x0)**2+w**2))
  elif fit_kind == 'gaussian':
    c = 4 * np.log(2)
    if loc_fixed:
      def peak(x, A, w):
        return A*np.exp(-c*(x-loc)**2/w**2)/(w*np.sqrt(np.pi/c))
    else:
      def peak(x, x0, A, w):
        return A*np.exp(-c*(x-x0)**2/w**2)/(w*np.sqrt(np.pi/c))
  else:
    raise ValueError('Unsupported fit_kind: %s' % fit_kind)
  return peak


def _combine_peak_functions(fns):
  num_peaks = len(fns)
  if num_peaks == 1:
    return fns[0]

  def composite_peak(x, *params):
    return sum(fn(x, *params[i::num_peaks]) for i, fn in enumerate(fns))

  return composite_peak


def _dummy(*args, **kwargs):
  pass


# Scipy v0.17+ has bounds for curve_fit
try:
  scipy.optimize.curve_fit(lambda x,a: x+a, [0,1], [0,1], p0=(0,), bounds=(0,1))
except TypeError:
  def curve_fit(*args, **kwargs):
    if 'bounds' in kwargs:
      del kwargs['bounds']
    return scipy.optimize.curve_fit(*args, **kwargs)
else:
  curve_fit = scipy.optimize.curve_fit
