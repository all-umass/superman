import numpy as np
import scipy.signal
import scipy.sparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# Cache for _make_pp, maps pp_string -> pp_fn
_PP_MEMO = {}


def preprocess(spectra, pp_string, wavelengths=None):
  pp_fn = _make_pp(pp_string)
  if hasattr(spectra, 'shape'):
    return pp_fn(spectra, wavelengths)
  # trajectory case
  pp = []
  for t in spectra:
    tt = t.copy()
    tt[:,1] = pp_fn(t[:,1:2].T, t[:,0]).ravel()
    pp.append(tt)
  return pp


def _make_pp(pp_string):
  '''Convert a preprocess string into its corresponding function.

  pp_string: str, looks like "foo:a:b,bar:x,baz:y"
      In this example, there are three preprocessing steps: foo, bar, and baz.
      Each takes one or two arguments, separated by colons.

  Returns: pp_fn(spectra, wavelengths), callable
  '''
  # try to use the cache
  if pp_string in _PP_MEMO:
    return _PP_MEMO[pp_string]

  # populate the preprocessing function pipeline
  pipeline = [_start_pipeline]
  if pp_string:
    for step in pp_string.split(','):
      parts = step.split(':')
      pipeline.append(PP_STEPS[parts[0]](*parts[1:]))

  # return a function that runs the pipeline
  def _fn(S, w):
    for f in pipeline:
      S = f(S, w)
    return S

  _PP_MEMO[pp_string] = _fn
  return _fn


def _start_pipeline(spectra, wavelengths):
  if scipy.sparse.issparse(spectra):
    S = spectra.copy()
    S.data = np.maximum(S.data, 1e-10)
  else:
    S = np.maximum(spectra, 1e-10)
  return S


def _polysquash(a, b):
  '''Generalized polynomial squashing function.
  Derived from a normal cubic polynomial with f(0) = 0 and f(1) = 1.
  We also enforce d/dx => 0 and d2/d2x <= 0, for a concave shape.
  This constrains -0.5 < a < 1, and -2a-1 < b < min(-3a, 0).'''
  a, b = float(a), float(b)
  c = 1 - a - b

  def fn(spectra, wavelengths):
    x = spectra / np.max(spectra, axis=1, keepdims=True)
    p = a*x**3 + b*x**2 + c*x
    return normalize(p, norm='l2', copy=False)
  return fn


def _bezier(a, b):
  '''Derived from a bezier curve with control points at [(0,0), (a,b), (1,1)]
  Constraints are 0 < a < 1, 0 < b < 1, b > a (upper left of y = x line).
  '''
  a, b = float(a), float(b)
  twoa = 2*a
  twob = 2*b
  if twoa == 1:
    a += 1e-5
    twoa = 2*a

  def fn(spectra, wavelengths):
    x = spectra / np.max(spectra, axis=1, keepdims=True)
    tmp = np.sqrt(a*a-twoa*x+x)
    foo = x * (1 - twob)
    top = -twoa*(tmp+foo+b) + twob*tmp + foo + twoa*a
    p = top / (1-twoa)**2
    return normalize(p, norm='l2', copy=False)
  return fn


def _pca(num_pcs):
  # Hack: may be float in [0,1] or positive int
  num_pcs = float(num_pcs) if '.' in num_pcs else int(num_pcs)
  model = PCA(n_components=num_pcs)

  return lambda S, w: model.fit_transform(S)


def _squash(squash_type, hinge=None):
  if squash_type == 'hinge':
    h = float(hinge)
    return lambda S, w: np.minimum(S, h)
  if squash_type == 'cos':
    return lambda S, w: (1 - np.cos(np.pi * S)) / 2.0

  # Only options left are plain numpy functions.
  squash_fn = getattr(np, squash_type)

  def fn(S, w):
    np.maximum(S, 1e-10, out=S)  # Hack: fix NaN issues
    return squash_fn(S)
  return fn


def _normalize(norm_type, loc=None):
  if norm_type in ('l1', 'l2'):
    return lambda S, w: normalize(S, norm=norm_type, copy=False)
  if norm_type == 'norm3':
    return lambda S, w: libs_norm3(S, copy=False)
  if norm_type == 'cum':
    return lambda S, w: cumulative_norm(S)

  if norm_type == 'min':
    def fn(S, w):
      S -= S.min(axis=1)[:,None]
      return S
  elif norm_type == 'max':
    # TODO: when sklearn v0.17+ is installed, use normalize(S, norm='max')
    def fn(S, w):
      if scipy.sparse.issparse(S):
        maxes = S.max(axis=1).toarray()
        maxes = maxes.repeat(np.diff(S.indptr))
        mask = maxes != 0
        S.data[mask] /= maxes[mask]
      else:
        S /= S.max(axis=1)[:,None]
      return S
  elif norm_type == 'band':
    loc = float(loc)

    def fn(S, w):
      idx = np.searchsorted(w, loc)
      S /= S[:, idx][:,None]
      return S
  else:
    raise ValueError('Unknown normalization type: %r' % norm_type)
  return fn


def _deriv(window, order):
  window, order = int(window), int(order)

  def fn(S, w):
    if scipy.sparse.issparse(S):
      S = S.toarray()
    return scipy.signal.savgol_filter(S, window, order, deriv=1)
  return fn


def _smooth(window, order):
  window, order = int(window), int(order)

  def fn(S, w):
    if scipy.sparse.issparse(S):
      S = S.toarray()
    S = scipy.signal.savgol_filter(S, window, order, deriv=0)
    # get rid of any non-positives created by the smoothing
    return np.maximum(S, 1e-10)
  return fn

# Lookup table of pp-string name -> pipeline function maker
PP_STEPS = dict(squash=_squash, normalize=_normalize, poly=_polysquash,
                smooth=_smooth, deriv=_deriv, pca=_pca, bezier=_bezier)


def cumulative_norm(S):
  '''Cumulative intensity normalization method.

  "Quality Assessment of Tandem Mass Spectra Based on
   Cumulative Intensity Normalization", Na & Paek, J. of Proteome Research
  '''
  idx = np.arange(S.shape[0])[:,None]
  ranks = np.argsort(S, axis=1)
  cumsums = np.cumsum(S[idx,ranks], axis=1)
  unranks = np.zeros_like(ranks)
  unranks[idx,ranks] = np.arange(S.shape[1])
  S = cumsums[idx,unranks]
  S /= cumsums[:,-1:]
  return S


def libs_norm3(shots, copy=True):
  shots = np.array(shots, copy=copy, ndmin=2)
  num_chan = shots.shape[1]
  if num_chan == 6143:
    a, b = 2047, 4097
  elif num_chan == 6144:
    a, b = 2048, 4098
  elif num_chan == 5485:
    a, b = 1884, 3811
  else:
    raise ValueError('Invalid # channels for LIBS norm3 method: %d' % num_chan)
  normalize(shots[:, :a], norm='l1', copy=False)
  normalize(shots[:,a:b], norm='l1', copy=False)
  normalize(shots[:, b:], norm='l1', copy=False)
  return shots
