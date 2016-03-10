from __future__ import absolute_import, print_function
import numpy as np
from matplotlib import pyplot as plt
from time import time

from ..dataset import load_dataset, dataset_views
from .. import options
from . import PEAK_CLASSES


def debug_peaks(ds_view, opts):
  peak_finder = PEAK_CLASSES[opts.peak_alg](max_peaks=opts.num_peaks)
  trajs, labels = ds_view.get_trajectories(return_keys=True)
  tic = time()
  all_peaks = [np.array(peak_finder.fit(*x.T).peak_locs, copy=True)
               for x in trajs]
  print('%s: %.3f secs' % (opts.peak_alg, time() - tic))
  print(map(len, all_peaks), 'peaks found')
  plt.figure()
  for traj, peak_locs, label in zip(trajs, all_peaks, labels):
    line, = plt.plot(*traj.T, label=label)
    line_color = plt.getp(line, 'color')
    peaks = np.interp(peak_locs, *traj.T)
    plt.scatter(peak_locs, peaks, marker='x', c=line_color, s=100)
  plt.legend()
  plt.title('Peak alg: ' + opts.peak_alg)


def main():
  op = options.setup_common_opts()
  op.add_argument('--peak-alg', choices=PEAK_CLASSES, required=True,
                  help='Peak-finding algorithm.')
  op.add_argument('--debug', type=str, nargs='+', default=['Trolleite'],
                  help='Species name(s) to show before/after. %(default)s')
  op.add_argument('--num-peaks', type=int,
                  help='Max # of peaks to detect per spectrum. [%(default)s]')
  options.add_preprocess_opts(op)
  opts = options.parse_opts(op, lasers=False)
  options.validate_preprocess_opts(op, opts)

  ds = load_dataset(opts.data, resample=opts.resample)
  for ds_view in dataset_views(ds, opts, minerals=opts.debug):
    debug_peaks(ds_view, opts)
  plt.show()

main()
