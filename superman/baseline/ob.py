from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import svds
from .common import Baseline


def ob_baseline(intensities, n_components):
    ''' Perform orthogonal basis baseline correction.
          n_components : Number of components to reconstruct baselines with. '''
    assert n_components>0, 'n_components must be nonnegative.'

    # Compute the svd and reconstruct using bottom k sv's
    U, s, V = svds(intensities, n_components)
    return sum([_s * np.outer(u,v) for (u,_s,v) in zip(U.T, s, V)])


class OB(Baseline):
  def __init__(self, n_components=5):
    self.n_components_ = n_components

  def _fit_many(self, bands, intensities):
    return ob_baseline(intensities, self.n_components_)

  def param_ranges(self):
    return {
        'n_components_': (1, 10, 'lin')
    }
