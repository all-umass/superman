import os.path
from argparse import ArgumentParser
from collections import namedtuple


def mock_options(parallel=20, dana=False, show_errors=False, rank=[1],
                 tsv=False, metric=['cosine'], k=[-1], type='raman',
                 weights=['distance'], pp=[''], ishikawa=False, peaks=False,
                 rbf=False, laser=['all'], data_dir=None, traj=False,
                 binarize=False, peak_alg='sg', peak_type='sparse', raw=False,
                 num_peaks=20, clf='knn', min_samples=3, folds=1, trials=1):
  kwargs = locals()  # Tricky hack: get the kwargs as a dict.
  MockOpt = namedtuple('MockOpt', kwargs.keys())
  if data_dir is None:
    kwargs['data_dir'] = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '{type}', 'data'))
  return MockOpt(**kwargs)

# Make a singleton tuple with the defaults for everything.
DEFAULTS = mock_options()


def setup_common_opts():
  op = ArgumentParser()
  og = op.add_argument_group('Common Options')
  og.add_argument('--type', type=str, default=DEFAULTS.type,
                  choices=('raman','ftir','xrd'),
                  help='Data type. [%(default)s]')
  og.add_argument('--laser', type=str, default=DEFAULTS.laser, nargs='+',
                  help='Laser energies to use. %(default)s')
  og.add_argument('--data-dir', type=str, default=DEFAULTS.data_dir,
                  help='Data directory. [%(default)s]')
  og.add_argument('--traj', action='store_true', help='Use trajectory methods')
  og.add_argument('--raw', action='store_true',
		  help='Use non-baseline-removed data')
  og.add_argument('--parallel', type=int, default=DEFAULTS.parallel,
                  help='Number of processes/threads to use. [%(default)s]')
  return op


def parse_opts(op, lasers=True):
  opts = op.parse_args()
  # Fill in the {type} placeholder with the --type argument, if needed.
  opts.data_dir = opts.data_dir.replace('{type}', opts.type)
  if not lasers and tuple(opts.laser) != ('all',):
    op.error('Filtering by laser type is not supported')
  return opts


def find_data_file(opts, resampled=True):
  kind = 'resampled' if resampled else 'spectra'
  raw = '-raw' if opts.raw else ''
  return os.path.join(opts.data_dir, 'rruff-%s%s.npz' % (kind, raw))


def add_distance_options(op):
  og = op.add_argument_group('Distance Options')
  og.add_argument('--metric', type=str, default=DEFAULTS.metric, nargs='+',
                  help='Distance metric(s). %(default)s')


def add_output_options(op):
  og = op.add_argument_group('Output Options')
  og.add_argument('--dana', action='store_true',
                  help='Print results broken down by Dana categories.')
  og.add_argument('--show-errors', action='store_true',
                  help='Print mismatching minerals.')
  og.add_argument('--tsv', action='store_true',
                  help='Print output in a TSV-like format.')
  og.add_argument('--ishikawa', action='store_true',
                  help='Use the Ishikawa minerals subset.')


def add_peak_opts(op):
  og = op.add_argument_group('Peak-Matching Options')
  og.add_argument('--binarize', action='store_true',
                  help='Clamp sparsified peaks to 0 or 1.')
  og.add_argument('--peak-alg', type=str, default=DEFAULTS.peak_alg,
                  choices=('scipy', 'sg', 'std'),
                  help='Peak-finding algorithm. [%(default)s]')
  og.add_argument('--peak-type', type=str, default=DEFAULTS.peak_type,
                  choices=('sparse', 'dense'),
                  help='Type of peak preprocessing. [%(default)s]')
  og.add_argument('--num-peaks', type=int, default=DEFAULTS.num_peaks,
                  help='Max # of peaks to detect (dense only). [%(default)s]')


def add_preprocess_opts(op):
  og = op.add_argument_group('Preprocessing Options')
  og.add_argument('--pp', type=str, default=DEFAULTS.pp, nargs='+',
                  help=' '.join('''
    Space-separated list of comma-separated preprocessing steps,
    in step:options format.
    Steps={squash,normalize,smooth,deriv}
      squash:{sqrt,log,tanh,cos,hinge:value}
      normalize:{l1,l2,min,max[:splits]}
      {smooth,deriv}:window:sg_order
      pca:num_pcs
      {poly,bezier}:a:b
    '''.split()))
  og.add_argument('--peaks', action='store_true', help='Use peak-matching.')
  og.add_argument('--rbf', action='store_true', help='Use RBF fitter.')


def validate_preprocess_opts(op, opts):
  valid_squashes = set(('sqrt','log','hinge','tanh','cos'))
  valid_norms = set(('l1','l2','max','min','cum','norm3'))
  for pp in opts.pp:
    for step in pp.split(','):
      parts = step.split(':')
      if parts[0] == 'squash':
        if len(parts) not in (2,3) or parts[1] not in valid_squashes:
          op.error('Invalid squash argument: %s' % parts)
      elif parts[0] == 'normalize':
        if len(parts) not in (2,3) or parts[1] not in valid_norms:
          op.error('Invalid normalize argument: %s' % parts)
        if len(parts) == 3 and parts[1] != 'max':
          op.error('Split not implemented for non-max normalizers')
      elif parts[0] in ('smooth', 'deriv'):
        if len(parts) != 3:
          op.error('Both window and order args required: %s' % parts)
        window,order = parts[1],parts[2]
        try:
          assert int(window) > 1
        except:
          op.error('window arg must be an integer > 1')
        try:
          assert int(order) > 0
        except:
          op.error('order arg must be an integer > 0')
      elif parts[0] == 'pca':
        if len(parts) != 2:
          op.error('PCA pp requires # of PCs arg.')
        try:
          assert float(parts[1]) > 0
        except:
          op.error('# PCs arg must be a positive number')
      elif parts[0] in ('poly', 'bezier'):
        if len(parts) != 3:
          op.error('Generalized PP needs 2 args')
        try:
          a,b = map(float, parts[1:])
          #assert -0.5 <= a <= 1
          #assert -2*a-1 <= b <= min(0, -3*a)
        except:
          op.error('args must satisfy constraints')
