import numpy as np
import glob
import warnings
from viztricks import plot_trajectories

import options
from preprocess import preprocess


def plot_spectra(spectra, names=None):
  # prevent mayhem with too many labels
  if names and len(names) > 20:
    warnings.warn('Too many labels (%d), not plotting a legend.' % len(names))
    names = None
  else:
    plot_trajectories(spectra, '-', labels=names)()


def find_strange_spectra(spectra, opts, names):
  for name,s in zip(names,spectra):
    x,y = s.T
    minx,maxx = x[0],x[-1]
    if minx < opts.min_x:
      print name, 'has low x:', minx
    if maxx > opts.max_x:
      print name, 'has high x:', maxx
    miny,maxy = y.min(),y.max()
    if miny < opts.min_y:
      print name, 'has low y:', miny
    if maxy > opts.max_y:
      print name, 'has high y:', maxy


def _get_names(all_names, sample_names):
  names = []
  for sample in sample_names:
    names.extend(glob.fnmatch.filter(all_names, sample))
  return names


def main():
  op = options.setup_common_opts()
  op.add_argument('--sample', type=str,
                  help='Samples to plot (CSV), or "all". May use glob syntax.')
  op.add_argument('--min-x', type=float, default=85, help='[%(default)s]')
  op.add_argument('--max-x', type=float, default=1801, help='[%(default)s]')
  op.add_argument('--min-y', type=float, default=0, help='[%(default)s]')
  op.add_argument('--max-y', type=float, default=1e6, help='[%(default)g]')
  options.add_preprocess_opts(op)
  opts = options.parse_opts(op, lasers=False)
  options.validate_preprocess_opts(op, opts)

  data_file = options.find_data_file(opts, resampled=False)
  data = np.load(data_file)
  names = sorted(data['names'])
  if opts.sample:
    names = _get_names(names, opts.sample.split(','))
  for pp in opts.pp:
    spectra = preprocess([data[k] for k in names], pp)
    if opts.sample == 'all':
      plot_spectra(spectra)
    elif opts.sample:
      plot_spectra(spectra, names)
    else:
      find_strange_spectra(spectra, opts, names)


if __name__ == '__main__':
  main()
