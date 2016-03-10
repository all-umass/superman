import os.path
from argparse import ArgumentParser
from collections import namedtuple


def mock_options(parallel=20, dana=False, show_errors=False, rank=[1],
                 tsv=False, metric=['cosine'], k=[-1], weights=['distance'],
                 pp=[''], ishikawa=False, laser=['all'], resample=False,
                 clf='knn', min_samples=3, folds=1, trials=1, data=None,
                 band_min=85, band_max=1800, band_step=1):
  kwargs = locals()  # Tricky hack: get the kwargs as a dict.
  MockOpt = namedtuple('MockOpt', kwargs.keys())
  if data is None:
    # Semi-hack: data now lives in another repository.
    kwargs['data'] = os.path.normpath(os.path.join(
        os.path.dirname(__file__),
        '../../darby_projects/raman/data/rruff-spectra.hdf5'))
  return MockOpt(**kwargs)

# Make a singleton tuple with the defaults for everything.
DEFAULTS = mock_options()


def setup_common_opts():
  op = ArgumentParser()
  og = op.add_argument_group('Common Options')
  og.add_argument('--data', type=str, default=DEFAULTS.data,
                  help='HDF5 file containing spectra. [%(default)s]')
  og.add_argument('--laser', type=str, default=DEFAULTS.laser, nargs='+',
                  help='Laser energies to use. %(default)s')
  og.add_argument('--resample', action='store_true',
                  help='Resample trajectories to common wavelengths.')
  og.add_argument('--parallel', type=int, default=DEFAULTS.parallel,
                  help='Number of processes/threads to use. [%(default)s]')
  og.add_argument('--band-min', type=int, default=DEFAULTS.band_min,
                  help='Lower bound for resampling. [%(default)s]')
  og.add_argument('--band-max', type=int, default=DEFAULTS.band_max,
                  help='Upper bound for resampling. [%(default)s]')
  og.add_argument('--band-step', type=int, default=DEFAULTS.band_step,
                  help='Sampling interval for resampling. [%(default)s]')
  return op


def parse_opts(op, lasers=True):
  opts = op.parse_args()
  if not lasers and tuple(opts.laser) != ('all',):
    op.error('Filtering by laser type is not supported')
  return opts


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
