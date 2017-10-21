from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.signal
import scipy.sparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from .utils import libs_norm3, cumulative_norm

__all__ = [
  'BandNormalize', 'BezierSquash', 'CosineSquash', 'CumulativeNormalize',
  'HingeSquash', 'L1Normalize', 'L2Normalize', 'LibsNormalize', 'LogSquash',
  'MaxNormalize', 'MinZeroNormalize', 'Offset', 'PolynomialSquash',
  'PrincipalComponents', 'SavitzkyGolayDerivative', 'SavitzkyGolaySmooth',
  'SqrtSquash', 'TanhSquash'
]


class Preprocessor(object):
  arg_type = float

  def apply(self, spectra, wavelengths):
    ''' S, w = apply(S, w)'''
    raise NotImplementedError('Subclasses must implement apply_vector.')

  @classmethod
  def from_string(cls, s):
    if not s:
      return cls()
    args = map(cls.arg_type, s.split(':'))
    return cls(*args)


class PrincipalComponents(Preprocessor):
  name = 'pca'

  def __init__(self, num_pcs):
    # Hack: may be float in (0,1] or positive int. We'll assume 1-D in the case
    # of 1.0, as that's more common.
    if num_pcs >= 1:
      assert num_pcs - int(num_pcs) == 0
      num_pcs = int(num_pcs)
    self.model = PCA(n_components=num_pcs)

  def apply(self, spectra, wavelengths):
    pcs = self.model.fit_transform(spectra)
    return pcs, wavelengths


class PolynomialSquash(Preprocessor):
  '''Generalized polynomial squashing function.

  Derived from a normal cubic polynomial with f(0) = 0 and f(1) = 1.
  We also enforce d/dx => 0 and d2/d2x <= 0, for a concave shape.
  This constrains -0.5 < a < 1, and -2a-1 < b < min(-3a, 0).
  '''
  name = 'poly'

  def __init__(self, a, b):
    assert -0.5 < a < 1
    assert -2*a - 1 < b < min(-3*a, 0)
    c = 1 - a - b
    self.poly = np.poly1d([a, b, c, 0])

  def apply(self, spectra, wavelengths):
    x = spectra / np.max(spectra, axis=1, keepdims=True)
    p = self.poly(x)
    return normalize(p, norm='l2', copy=False), wavelengths


class BezierSquash(Preprocessor):
  '''Bezier squashing function.

  Derived from a bezier curve with control points at [(0,0), (a,b), (1,1)]
  Constraints are 0 < a < 1, 0 < b < 1, b > a (upper left of y = x line).
  '''
  name = 'bezier'

  def __init__(self, a, b):
    assert 0 < a < 1
    assert 0 < b < 1
    assert b > a
    twoa = 2*a
    twob = 2*b
    if twoa == 1:
      a += 1e-5
      twoa = 2*a
    self.args = (a, b, twoa, twob)

  def apply(self, spectra, wavelengths):
    x = spectra / np.max(spectra, axis=1, keepdims=True)
    a, b, twoa, twob = self.args
    tmp = np.sqrt(a*a-twoa*x+x)
    foo = x * (1 - twob)
    top = -twoa*(tmp+foo+b) + twob*tmp + foo + twoa*a
    p = top / (1-twoa)**2
    return normalize(p, norm='l2', copy=False), wavelengths


class HingeSquash(Preprocessor):
  name = 'squash:hinge'

  def __init__(self, h):
    self.hinge = h

  def apply(self, spectra, wavelengths):
    return np.minimum(spectra, self.hinge), wavelengths


class CosineSquash(Preprocessor):
  name = 'squash:cos'

  def apply(self, spectra, wavelengths):
    np.maximum(spectra, 1e-10, out=spectra)  # Hack: fix NaN issues
    s = (1 - np.cos(np.pi * spectra)) / 2.0
    return s, wavelengths


def _generic_squash(numpy_func_name):
  fn = getattr(np, numpy_func_name)

  class _GenericSquash(Preprocessor):
    name = 'squash:' + numpy_func_name

    def apply(self, spectra, wavelengths):
      return fn(spectra), wavelengths

  _GenericSquash.__name__ = numpy_func_name.title() + 'Squash'
  return _GenericSquash

TanhSquash = _generic_squash('tanh')
SqrtSquash = _generic_squash('sqrt')
LogSquash = _generic_squash('log')


class LibsNormalize(Preprocessor):
  name = 'normalize:norm3'

  def apply(self, spectra, wavelengths):
    s = libs_norm3(spectra, wavelengths=wavelengths, copy=False)
    return s, wavelengths


class CumulativeNormalize(Preprocessor):
  name = 'normalize:cum'

  def apply(self, spectra, wavelengths):
    return cumulative_norm(spectra), wavelengths


class MinZeroNormalize(Preprocessor):
  name = 'normalize:min'

  def apply(self, spectra, wavelengths):
    spectra -= spectra.min(axis=1)[:, None]
    return spectra, wavelengths


class BandNormalize(Preprocessor):
  name = 'normalize:band'

  def __init__(self, loc):
    self.loc = loc

  def apply(self, spectra, wavelengths):
    idx = np.searchsorted(wavelengths, self.loc)
    a = max(0, idx - 2)
    b = min(len(wavelengths), idx + 3)
    x = spectra[:, a:b].max(axis=1)
    spectra /= x[:,None]
    return spectra, wavelengths


def _generic_norm(norm):
  assert norm in ('max', 'l1', 'l2')

  class _GenericNorm(Preprocessor):
    name = 'normalize:' + norm

    def apply(self, spectra, wavelengths):
      return normalize(spectra, norm=norm, copy=False), wavelengths

  _GenericNorm.__name__ = norm.title() + 'Normalize'
  return _GenericNorm

MaxNormalize = _generic_norm('max')
L1Normalize = _generic_norm('l1')
L2Normalize = _generic_norm('l2')


class SavitzkyGolayDerivative(Preprocessor):
  name = 'deriv'
  arg_type = int

  def __init__(self, window, order):
    self.window = window
    self.order = order

  def apply(self, spectra, wavelengths):
    assert not scipy.sparse.issparse(spectra)
    d = scipy.signal.savgol_filter(spectra, self.window, self.order, deriv=1)
    return d, wavelengths


class SavitzkyGolaySmooth(SavitzkyGolayDerivative):
  name = 'smooth'

  def apply(self, spectra, wavelengths):
    assert not scipy.sparse.issparse(spectra)
    d = scipy.signal.savgol_filter(spectra, self.window, self.order, deriv=0)
    return d, wavelengths


class Offset(Preprocessor):
  name = 'offset'

  def __init__(self, x, y=0):
    self.intensity_offset = y
    self.wavelength_offset = x

  def apply(self, spectra, wavelengths):
    if self.intensity_offset != 0:
      spectra += self.intensity_offset
    if self.wavelength_offset != 0:
      wavelengths += self.wavelength_offset
    return spectra, wavelengths
