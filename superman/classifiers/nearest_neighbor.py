from __future__ import absolute_import
import numpy as np
from itertools import product as cart_product
from time import time

from ..distance import pairwise_dists
from .utils import ClassifyResult


def knn_test(Xtrain, Ytrain, Xtest, pp, opts):
  for metric in opts.metric:
    tic = time()
    D = pairwise_dists(Xtest, Xtrain, metric, num_procs=opts.parallel)
    d_time = time() - tic
    for k, w in cart_product(opts.k, opts.weights):
      yield _test_knn(D, Ytrain, k, w, metric, d_time, pp, opts)


def _test_knn(dists, Ytrain, k, weights, metric, d_time, pp, opts):
  tic = time()
  ranking = _weighted_neighbors(dists, Ytrain, k, weights)
  elapsed = time() - tic + d_time

  if '-' in metric:
    m, p = metric.split('-', 1)
    metric = '%s(%g)' % (m.upper(), float(p))
  clf = 'WN' if k < 0 else '%d-NN' % k
  title = '%s %s (%s) [%s]' % (clf, metric, weights[:4], pp)
  return ClassifyResult(ranking, elapsed, title)


def _weighted_neighbors(D, Y, k, weight_type):
  is_wn = k < 0
  nn = len(Y) if is_wn else k
  classes_, _y = np.unique(Y, return_inverse=True)
  all_rows = np.arange(D.shape[0])
  neigh_ind = np.argsort(D, axis=1)[:,:nn]

  if weight_type == 'uniform':
    weights = np.ones_like(neigh_ind)
  elif weight_type == 'distance':
    neigh_dist = D[all_rows[:,None], neigh_ind]
    # Scale to the 0.05 - 1.05 range
    min_d = np.min(D, axis=1, keepdims=True)
    neigh_dist -= min_d
    neigh_dist /= np.max(D - min_d, axis=1, keepdims=True)
    neigh_dist += 0.05
    # Simple 1/d weights
    weights = 1. / neigh_dist
  else:
    raise ValueError('Invalid weight_type: %r' % weight_type)

  if is_wn:
    # bincount _y to get counts per sample, then divide weights
    counts = np.bincount(_y)
    weights /= counts[_y[None]]  # assumes classes_ is arange

  pred_labels = _y[neigh_ind]
  proba = np.zeros((len(all_rows), len(classes_)))
  # a simple ':' index doesn't work right
  for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
    proba[all_rows, idx] += weights[:, i]
  return classes_[np.argsort(-proba)]
