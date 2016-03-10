from __future__ import absolute_import
import numpy as np

from . import options
from .classifiers import knn_test, CLASSIFIERS
from .classifiers.utils import (
    test_train_mask, test_train_split, print_results, print_cross_fold)
from .dataset import load_dataset, dataset_views


def _classify_oneshot(ds, opts):
  test_fn = CLASSIFIERS[opts.clf]
  if opts.tsv:
    if opts.dana:
      print '# Title\tTime\tRank\tClass\tType\tGroup\tSpecies\tTotal'
    else:
      print '# Title\tTime\tRank\tScore\tTotal'
  label_meta, _ = ds.find_metadata('minerals')
  label_map = label_meta.labels
  example_view = next(dataset_views(ds, opts))
  Y = label_map[example_view.mask]
  for _ in xrange(opts.trials):
    train_mask, test_mask = test_train_mask(Y, label_map, opts.min_samples)
    for ds_view in dataset_views(ds, opts):
      pp = ds_view.transformations['pp']
      X, names = ds_view.get_data(return_keys=True)
      Xtrain, Ytrain, Xtest, Ytest, Ntest = \
          test_train_split(X, Y, train_mask, test_mask, names)
      for cr in test_fn(Xtrain, Ytrain, Xtest, pp, opts):
        print_results(Ytest, label_map, Ntest, cr, opts)


def _classify_crossfold(ds, opts):
  assert opts.clf == 'knn', ('Cross-Fold is NYI for --clf %s' % opts.clf)
  num_tests = np.product(map(len, (opts.metric,opts.pp,opts.k,opts.weights)))

  label_meta, _ = ds.find_metadata('minerals')
  label_map = label_meta.labels
  example_view = next(dataset_views(ds, opts))
  Y = label_map[example_view.mask]

  sum_scores = np.empty((num_tests, len(opts.rank)))
  sumsq_scores = np.empty_like(sum_scores)
  per_time = np.empty_like(sum_scores)
  titles = [None] * num_tests
  for _ in xrange(opts.trials):
    sum_scores[:,:] = 0
    sumsq_scores[:,:] = 0
    per_time[:,:] = 0
    for _ in xrange(opts.folds):
      train_mask, test_mask = test_train_mask(Y, label_map, opts.min_samples)
      i = 0
      for ds_view in dataset_views(ds, opts):
        pp = ds_view.transformations['pp']
        X = ds_view.get_data()
        Xtrain, Ytrain, Xtest, Ytest = test_train_split(X, Y, train_mask,
                                                        test_mask)
        for result in knn_test(Xtrain, Ytrain, Xtest, pp, opts):
          titles[i] = result.title
          for j, r in enumerate(opts.rank):
            matches = np.any(Ytest[:,None] == result.ranking[:,:r], axis=1)
            s = np.count_nonzero(matches)
            sum_scores[i,j] += s
            sumsq_scores[i,j] += s * s
            per_time[i,j] += result.elapsed
          i += 1
    # aggregate fold results
    n = float(opts.folds)
    mean = sum_scores / n
    stdv = np.sqrt(n*sumsq_scores - sum_scores**2) / n
    per_time /= n

    print_cross_fold(mean, stdv, len(Ytest), per_time, titles,
                     opts.rank, opts.folds)


def main():
  op = options.setup_common_opts()
  op.add_argument('--clf', type=str, default=options.DEFAULTS.clf,
                  choices=CLASSIFIERS.keys(),
                  help='Classifer algorithm. %(default)s')
  op.add_argument('-k', type=int, default=options.DEFAULTS.k, nargs='+',
                  help='# of nearest neighbors, or -1 for all. %(default)s')
  op.add_argument('--weights', type=str, default=options.DEFAULTS.weights,
                  nargs='+', choices=('uniform', 'distance'),
                  help='kNN weighting scheme(s). %(default)s')
  op.add_argument('--min-samples', type=int,
                  default=options.DEFAULTS.min_samples,
                  help='Training samples per mineral. [%(default)s]')
  op.add_argument('--folds', type=int, default=options.DEFAULTS.folds,
                  help='Number of cross-validation folds. [%(default)s]')
  op.add_argument('--rank', type=int, default=options.DEFAULTS.rank, nargs='+',
                  help='Compute precision@rank for scoring. %(default)s')
  op.add_argument('--trials', type=int, default=options.DEFAULTS.trials,
                  help='Number of one-shot runs. [%(default)s]')
  options.add_preprocess_opts(op)
  options.add_distance_options(op)
  options.add_output_options(op)
  opts = options.parse_opts(op)
  options.validate_preprocess_opts(op, opts)
  if opts.tsv and opts.show_errors:
    op.error('Choose one of --tsv or --show-errors, but not both')
  if opts.folds > 1 and opts.dana:
    op.error('Dana output not supported with >1 fold')
  if opts.clf != 'knn' and not opts.resample:
    op.error('Only knn supports trajectories, pass --resample to use others.')

  ds = load_dataset(opts.data, resample=opts.resample)
  if opts.folds == 1:
    _classify_oneshot(ds, opts)
  else:
    _classify_crossfold(ds, opts)


if __name__ == '__main__':
  main()
