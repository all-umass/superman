from __future__ import print_function, absolute_import
import numpy as np
from collections import namedtuple
from six.moves import xrange

from .. import dana

ClassifyResult = namedtuple('ClassifyResult', ('ranking', 'elapsed', 'title'))


def test_train_mask(Y, label_map, min_samples_per_class):
  test_mask = np.zeros(Y.shape, dtype=bool)
  train_mask = test_mask.copy()
  for i in label_map:
    inds, = np.where(Y == i)
    np.random.shuffle(inds)  # add some randomness
    train_mask[inds[:min_samples_per_class]] = True
    test_mask[inds[min_samples_per_class:]] = True
  return train_mask, test_mask


def test_train_split(X, Y, train_mask, test_mask, names=None):
  # Split em up.
  Ytrain, Ytest = Y[train_mask], Y[test_mask]
  if hasattr(X, 'shape'):
    Xtrain, Xtest = X[train_mask], X[test_mask]
  else:
    # X is trajectory data.
    Xtrain = [X[i] for i in np.where(train_mask)[0]]
    Xtest = [X[i] for i in np.where(test_mask)[0]]
  if names is None:
    return Xtrain, Ytrain, Xtest, Ytest
  return Xtrain, Ytrain, Xtest, Ytest, names[test_mask]


def print_cross_fold(mean, stdv, total, per_time, titles, ranks, folds):
  pct_score = mean / total * 100
  for i, title in enumerate(titles):
    for j, r in enumerate(ranks):
      print('%s => %.2f +/- %.2f (%.2f%%)' % (
            title, mean[i,j], stdv[i,j], pct_score[i,j]), end=' ')
      print('correct @ %d, %.3f secs/fold, %d folds' % (
            r, per_time[i,j], folds))


def print_results(Ytest, label_map, Ntest, result, opts):
  print_fn = _print_dana_results if opts.dana else _print_result
  for rank in opts.rank:
    print_fn(Ytest, label_map, Ntest, result, rank, opts)


def _print_result(Ytest, label_map, Ntest, result, rank, opts):
  pred = result.ranking[:,:rank]
  matches = np.any(Ytest[:,None] == pred, axis=1)
  score = np.count_nonzero(matches)
  num_test = Ytest.shape[0]
  ratio = float(score)/num_test*100
  if opts.tsv:
    print(result.title, '%.3f' % result.elapsed, rank, score, num_test,sep='\t')
  else:
    print('%s => %d/%d (%.2f%%) correct at %d, %.3f seconds' % (
        result.title, score, num_test, ratio, rank, result.elapsed))
  if opts.show_errors:
    for eidx in np.where(~matches)[0]:
      true_name = Ntest[eidx]
      pred_species = label_map[pred[eidx]]
      print(' * %s != %s' % (true_name, ' or '.join(pred_species)))


def _print_dana_results(Ytest, label_map, Ntest, result, rank, opts):
  pred = result.ranking[:,:rank]
  mismatches = np.all(Ytest[:,None] != pred, axis=1)

  num_test = Ytest.shape[0]
  # cross-reference using dana numbers
  true_dana = dana.convert_to_dana(label_map[Ytest[mismatches]])
  pred_danas = [dana.convert_to_dana(label_map[p]) for p in pred[mismatches].T]

  if opts.tsv:
    print(result.title, '%.3f' % result.elapsed, rank, sep='\t', end='')
  else:
    print(result.title, '%.3f seconds' % result.elapsed)

  buckets = dict((col, []) for col in true_dana.dtype.names)
  eidxs, = np.where(mismatches)
  for i, eidx in enumerate(eidxs):
    danaT = true_dana[i]
    danasP = [d[i] for d in pred_danas]
    for col in true_dana.dtype.names:
      t = danaT[col]
      danasP = [dp for dp in danasP if dp[col] == t]
      if not danasP:
        buckets[col].append((i, eidx))
        break
    else:
      # This should never happen, and if it does, it's a bug.
      assert False, "Mismatched label didn't have a mismatching dana field"

  score = num_test
  for col in true_dana.dtype.names:
    bucket = buckets[col]
    score -= len(bucket)
    ratio = float(score)/num_test*100
    if opts.tsv:
      print('\t%d' % score, end='')
    else:
      print(' (%s) %d/%d (%.2f%%) at %d' % (col, score, num_test, ratio, rank))
    if opts.show_errors:
      for i, eidx in bucket:
        nameT = Ntest[eidx]
        danaT = '.'.join(true_dana[i])
        lhs = '   * %s (%s) !=' % (nameT, danaT)
        namesP = label_map[pred[eidx]]
        danasP = ['.'.join(p[i]) for p in pred_danas]
        print(lhs, '%s (%s)' % (namesP[0], danasP[0]))
        lhs = ' ' * len(lhs)
        for r in xrange(1, rank):
          print(lhs, '%s (%s)' % (namesP[r], danasP[r]))
  # Sanity check.
  assert score == np.count_nonzero(~mismatches)

  if opts.tsv:
    print('\t%d' % num_test)
