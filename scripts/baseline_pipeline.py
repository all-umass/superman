#!/usr/bin/python
import numpy as np
import os
from argparse import ArgumentParser
from sklearn.cross_validation import cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from superman.baseline import BL_CLASSES

#    raw data  (raman, ftir, libs)
#       |
#       v
# baseline removal  <-- random parameters
#         |                  ^
#         v                  |
# train regression model     |
#   (PLS-2, 10 comps)        |
#   (l2 ridge, fix alpha?)   |
#   (linear SVR, C/epsilon)  |
#         |                  |
#      k-fold CV             |
#         |                  |
#         v                  |
#   store accuracy (rmsep) --/

MODELS = {
    'pls': PLSRegression(n_components=10),
    'ridge': RidgeCV(),
    'svr': SVR()
}


def run_pipeline(X, Y, bands, regression, baseline, log_fh, args):
  if args.verbose:
    print 'Starting %s baseline pipeline with %s regression...' % (
        args.baseline, args.model)
  for params in generate_baseline_params(baseline):
    # set params on the baseline object
    for k,v in params.iteritems():
      setattr(baseline, k, v)
    # do all the baseline removals
    Xt = baseline.fit_transform(bands, X)
    # learn the model
    score = cross_val_score(regression, Xt, Y).mean()
    # log the result
    print >>log_fh, score, params
    if args.verbose:
      print score, ', '.join('%s=%g' % x for x in params.iteritems())


def generate_baseline_params(bl):
  param_gen = bl.param_ranges()
  # set up random sampler functions once
  for key, (lb,ub,scale) in param_gen.iteritems():
    if scale == 'linear':
      param_gen[key] = lambda lb=lb,ub=ub: np.random.uniform(lb, ub)
    elif scale == 'log':
      lb, ub = np.log10((lb, ub))
      param_gen[key] = lambda lb=lb,ub=ub: 10**np.random.uniform(lb, ub)
    elif scale == 'integer':
      param_gen[key] = lambda lb=lb,ub=ub: np.random.randint(lb, ub+1)
    else:
      raise ValueError('invalid param_range scale: %s' % scale)
  # call those functions forever
  while True:
    yield dict((key, fn()) for key, fn in param_gen.iteritems())


def load_data(filepath):
  data = np.load(filepath)
  return data['X'], data['Y'], data['bands']


def logfile_path(args):
  dataname = os.path.splitext(os.path.basename(args.dataset))[0]
  return os.path.join(args.outdir, '%s_%s_%s.txt' % (
      dataname, args.model, args.baseline))


def main():
  ap = ArgumentParser()
  ap.add_argument('-o', '--outdir', default='.',
                  help='directory to store logs in [.]')
  ap.add_argument('-v', '--verbose', action='store_true',
                  help='print more output to stdout')
  # ap.add_argument('--segment', action='store_true',
  #                 help='Use auto-segmentation for baseline removal')
  ap.add_argument('dataset', help='npz file with fields: X, Y, bands')
  ap.add_argument('model', choices=MODELS.keys())
  ap.add_argument('baseline', choices=BL_CLASSES.keys())
  args = ap.parse_args()
  X, Y, bands = load_data(args.dataset)
  model = MODELS[args.model]
  bl = BL_CLASSES[args.baseline]()
  with open(logfile_path(args), 'a', 1) as log_fh:
    try:
      run_pipeline(X, Y, bands, model, bl, log_fh, args)
    except KeyboardInterrupt:
      pass

if __name__ == '__main__':
  main()
