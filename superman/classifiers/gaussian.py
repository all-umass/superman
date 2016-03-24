from __future__ import absolute_import
import numpy as np
from time import time
from scipy.stats import norm as normal_dist

from .utils import ClassifyResult


def gauss_test(Xtrain, Ytrain, Xtest, pp, opts):
  """Computes mean+std for each class in the training set,
  then computes probabilities for each band.
  Final score is calculated by dotting the band-probs with the sample itself;
  this avoids big contributions from zero-intensity parts.

  Known issues:
   - huge effects when matching bands hit a low-variance part, which means
     there's high sensitivity to the training set.
   - no contribution from neighboring bands

  This means that a sample which makes an "X" with the target class could have
  a very high match score.
  """
  classes = np.unique(Ytrain)
  c_means = np.empty((len(classes), Xtrain.shape[1]))
  c_stds = np.empty_like(c_means)
  proba = np.empty((Xtest.shape[0], len(classes)))

  tic = time()
  for i,c in enumerate(classes):
    points = Xtrain[Ytrain == c]
    c_means[i] = points.mean(axis=0)
    c_stds[i] = np.maximum(points.std(axis=0), 1e-10)

  for j,c_mean in enumerate(c_means):
    distribution = normal_dist(c_mean, c_stds[j])
    for i,test in enumerate(Xtest):
      proba[i,j] = distribution.pdf(test).dot(test)

  ranking = np.argsort(-proba)
  elapsed = time() - tic

  yield ClassifyResult(ranking, elapsed, 'gauss [%s]' % pp)
