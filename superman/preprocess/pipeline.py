from __future__ import absolute_import, division, print_function
import re

from . import steps

# Cache for _make_pp, maps pp_string -> pp_fn
_PP_MEMO = {}

# Lookup table of pp-string name -> pipeline step
PP_STEPS = dict((cls.name, cls) for cls in
                (getattr(steps, cls_name) for cls_name in steps.__all__))


def preprocess(spectra, pp_string, wavelengths=None, copy=True):
  pp_fn = _make_pp(pp_string)

  if hasattr(spectra, 'shape'):
    if copy:
      spectra = spectra.copy()
    S, w = pp_fn(spectra, wavelengths)
    return S  # TODO: return w as well

  if not copy:
    for t in spectra:
      y, w = pp_fn(t[:,1:2].T, t[:,0])
      t[:,0] = w
      t[:,1] = y.ravel()
    return spectra

  pp = []
  for t in spectra:
    tt = t.copy()
    y, w = pp_fn(tt[:,1:2].T, tt[:,0])
    tt[:,0] = w
    tt[:,1] = y.ravel()
    pp.append(tt)
  return pp


def _make_pp(pp_string):
  '''Convert a preprocess string into its corresponding function.

  pp_string: str, looks like "foo:1:2,bar:4,baz:quux"
      In this example, there are 3 preprocessing steps: foo, bar, and baz:quux.
      Step 'foo' takes two arguments, 'bar' takes one, and 'baz:quux' none.

  Returns: pp_fn(spectra, wavelengths), callable
  '''
  # try to use the cache
  if pp_string in _PP_MEMO:
    return _PP_MEMO[pp_string]

  # populate the preprocessing function pipeline
  pipeline = []
  if pp_string:
    for step in pp_string.split(','):
      # Hack: some names include a colon
      parts = step.split(':')
      if len(parts) > 1 and re.match(r'[a-z]+', parts[1]):
        idx = 2
      else:
        idx = 1
      name = ':'.join(parts[:idx])
      args = ':'.join(parts[idx:])
      pipeline.append(PP_STEPS[name].from_string(args))

  # return a function that runs the pipeline
  def _fn(S, w):
    for p in pipeline:
      S, w = p.apply(S, w)
    return S, w

  _PP_MEMO[pp_string] = _fn
  return _fn
