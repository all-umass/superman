import options
from classifiers import (
    gauss_test, neural_net_test, knn_cross_fold, knn_test, decision_tree_test)
from classifiers.utils import test_train_split, print_results, print_cross_fold
from utils import prepare_data

CLASSIFIERS = {
    'knn': knn_test,
    'gauss': gauss_test,
    'dtree': decision_tree_test,
    'nnet': neural_net_test
}


def _classify_oneshot(X, Y, label_map, names, opts):
  test_fn = CLASSIFIERS[opts.clf]
  if opts.tsv:
    if opts.dana:
      print '# Title\tTime\tRank\tClass\tType\tGroup\tSpecies\tTotal'
    else:
      print '# Title\tTime\tRank\tScore\tTotal'
  for _ in xrange(opts.trials):
    Xtrain, Ytrain, Xtest, Ytest, Ntest = test_train_split(
        X, Y, opts.min_samples, names=names)
    for cr in test_fn(Xtrain, Ytrain, Xtest, opts):
      print_results(Ytest, label_map, Ntest, cr, opts)


def _classify_crossfold(X, Y, label_map, names, opts):
  assert opts.clf == 'knn', ('Cross-Fold is NYI for --clf %s' % opts.clf)
  for _ in xrange(opts.trials):
    mean, stdv, total, time, titles = knn_cross_fold(X, Y, label_map, opts)
    print_cross_fold(mean, stdv, total, time, titles, opts.rank, opts.folds)


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

  data_file = options.find_data_file(opts, resampled=(not opts.traj))
  classify = _classify_oneshot if opts.folds == 1 else _classify_crossfold
  for X, Y, label_map, names in prepare_data(data_file, opts):
    classify(X, Y, label_map, names, opts)

if __name__ == '__main__':
  main()
