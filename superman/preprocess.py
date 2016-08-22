import numpy as np
import scipy.signal
import scipy.sparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


def preprocess(spectra, pp_string):
  if not hasattr(spectra, 'shape'):
    pp = []
    for t in spectra:
      tt = t.copy()
      tt[:,1] = _preprocess(t[:,1:2].T, pp_string).ravel()
      pp.append(tt)
  else:
    pp = _preprocess(spectra, pp_string)
  return pp


def _preprocess(spectra, pp_string):
  pp_fns = dict(squash=_squash, normalize=_normalize, poly=_polysquash,
                smooth=_smooth, deriv=_deriv, pca=_pca, bezier=_bezier)
  if scipy.sparse.issparse(spectra):
    S = spectra.copy()
    S.data = np.maximum(S.data, 1e-10)
  else:
    S = np.maximum(spectra, 1e-10)
  if pp_string:
    for step in pp_string.split(','):
      parts = step.split(':')
      fn = pp_fns[parts[0]]
      S = fn(S, *parts[1:])
  return S


def _polysquash(spectra, *params):
  '''Generalized polynomial squashing function.
  Derived from a normal cubic polynomial with f(0) = 0 and f(1) = 1.
  We also enforce d/dx => 0 and d2/d2x <= 0, for a concave shape.
  This constrains -0.5 < a < 1, and -2a-1 < b < min(-3a, 0).'''
  a, b = map(float, params)
  c = 1 - a - b
  x = spectra / np.max(spectra, axis=1, keepdims=True)
  p = a*x**3 + b*x**2 + c*x
  return normalize(p, norm='l2', copy=False)


def _bezier(spectra, *params):
  '''Derived from a bezier curve with control points at [(0,0), (a,b), (1,1)]
  Constraints are 0 < a < 1, 0 < b < 1, b > a (upper left of y = x line).
  '''
  a, b = map(float, params)
  x = spectra / np.max(spectra, axis=1, keepdims=True)
  twoa = 2*a
  twob = 2*b
  if twoa == 1:
    a += 1e-5
    twoa = 2*a
  tmp = np.sqrt(a*a-twoa*x+x)
  foo = x * (1 - twob)
  top = -twoa*(tmp+foo+b) + twob*tmp + foo + twoa*a
  p = top / (1-twoa)**2
  return normalize(p, norm='l2', copy=False)


def _pca(spectra, num_pcs):
  # Hack: may be float in [0,1] or positive int
  num_pcs = float(num_pcs) if '.' in num_pcs else int(num_pcs)
  model = PCA(n_components=num_pcs)
  return model.fit_transform(spectra)


def _squash(S, squash_type, hinge=None):
  if squash_type == 'hinge':
    return np.minimum(S, float(hinge))
  if squash_type == 'cos':
    return (1 - np.cos(np.pi * S)) / 2.0
  # Only options left are just numpy functions.
  np.maximum(S, 1e-10, out=S)  # Hack: fix NaN issues
  return getattr(np, squash_type)(S)


def _normalize(S, norm_type, loc=None):
  if norm_type in ('l1', 'l2'):
    return normalize(S, norm=norm_type, copy=False)
  if norm_type == 'norm3':
    # LIBS-specific normalization
    return libs_norm3(S, copy=False)
  if norm_type == 'cum':
    # see ref: "Quality Assessment of Tandem Mass Spectra
    #   Based on Cumulative Intensity Normalization"
    # by Na and Paek, Journal of Proteome Research.
    idx = np.arange(S.shape[0])[:,None]
    ranks = np.argsort(S, axis=1)
    cumsums = np.cumsum(S[idx,ranks], axis=1)
    unranks = np.zeros_like(ranks)
    unranks[idx,ranks] = np.arange(S.shape[1])
    S = cumsums[idx,unranks]
    S /= cumsums[:,-1:]
    return S
  if norm_type == 'min':
    S -= S.min(axis=1)[:,None]
    return S
  # norm_type == 'max'
  # This gets weird. I wish sklearn.preprocessing.normalize handled this.
  if scipy.sparse.issparse(S):
    maxes = S.max(axis=1).toarray()
    maxes = maxes.repeat(np.diff(S.indptr))
    mask = maxes != 0
    S.data[mask] /= maxes[mask]
  else:
    S /= S.max(axis=1)[:,None]
  return S


def libs_norm3(shots, copy=True):
  shots = np.array(shots, copy=copy, ndmin=2)
  num_chan = shots.shape[1]
  assert num_chan in (6143, 6144, 5485)
  if num_chan == 6143:
    a, b = 2047, 4097
  elif num_chan == 6144:
    a, b = 2048, 4098
  elif num_chan == 5485:
    a, b = 1884, 3811
  normalize(shots[:, :a], norm='l1', copy=False)
  normalize(shots[:,a:b], norm='l1', copy=False)
  normalize(shots[:, b:], norm='l1', copy=False)
  return shots


def _deriv(S, window, order):
  if scipy.sparse.issparse(S):
    S = S.toarray()
  return scipy.signal.savgol_filter(S, int(window), int(order), deriv=1)


def _smooth(S, window, order):
  if scipy.sparse.issparse(S):
    S = S.toarray()
  S = scipy.signal.savgol_filter(S, int(window), int(order), deriv=0)
  # get rid of any non-positives created by the smoothing
  return np.maximum(S, 1e-10)
