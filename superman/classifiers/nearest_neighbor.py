from __future__ import absolute_import
import numpy as np
from itertools import product as cart_product
from time import time

from superman.pairwise_dists import pairwise_dists
from superman.preprocess import preprocess
from .utils import test_train_split, ClassifyResult


def knn_test(Xtrain, Ytrain, Xtest, opts):
  for pp in opts.pp:
    pp_test = preprocess(Xtest, pp)
    pp_train = preprocess(Xtrain, pp)
    for metric in opts.metric:
      tic = time()
      D = pairwise_dists(pp_test, pp_train, metric, num_procs=opts.parallel)
      d_time = time() - tic
      for k, w in cart_product(opts.k, opts.weights):
        yield _test_knn(D, Ytrain, k, w, metric, d_time, pp, opts)


def knn_cross_fold(X, Y, label_map, opts):
  num_tests = np.product(map(len, (opts.metric,opts.pp,opts.k,opts.weights)))
  sum_scores = np.zeros((num_tests, len(opts.rank)))
  sumsq_scores = np.zeros_like(sum_scores)
  per_time = np.zeros_like(sum_scores)
  titles = [None] * num_tests
  for _ in xrange(opts.folds):
    Xtrain, Ytrain, Xtest, Ytest = test_train_split(X, Y, opts.min_samples)
    for i,result in enumerate(knn_test(Xtrain, Ytrain, Xtest, opts)):
      titles[i] = result.title
      for j,r in enumerate(opts.rank):
        matches = np.any(Ytest[:,None] == result.ranking[:,:r], axis=1)
        s = np.count_nonzero(matches)
        sum_scores[i,j] += s
        sumsq_scores[i,j] += s*s
        per_time[i,j] += result.elapsed

  n = float(opts.folds)
  stdv = np.sqrt(n*sumsq_scores - sum_scores**2)/n
  mean = sum_scores / n
  per_time /= n
  return mean, stdv, len(Ytest), per_time, titles


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
