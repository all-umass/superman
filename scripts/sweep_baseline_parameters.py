#!/usr/bin/python
import numpy as np
from argparse import ArgumentParser
from os.path import basename
from matplotlib import pyplot
from matplotlib.colors import LogNorm, Normalize
from superman.baseline import BL_CLASSES
from superman.file_io import parse_spectrum


def run_sweep(raw_file, sweep, params, outfile=None, replicate_paper=False):
  S = parse_spectrum(raw_file)
  bands, intensities = S.T

  if replicate_paper:
    import matplotlib
    matplotlib.rc('font', weight='normal', size=9)
    pyplot.figure(figsize=(6,4))

  ax = pyplot.gca()
  if not replicate_paper:
    ax.set_title('%s sweep on %s' % (sweep, basename(raw_file)))
  ax.plot(bands, intensities, 'k-', linewidth=2)
  cm = pyplot.get_cmap('jet')
  ax.set_color_cycle(map(cm, np.linspace(0,1,len(params))))

  for p in params:
    kwargs = {sweep: p}
    if sweep == 'poly_order':
      bl = BL_CLASSES['mario'](**kwargs)
    else:
      bl = BL_CLASSES['als'](**kwargs)
    baseline = bl.fit(bands, intensities).baseline
    ax.plot(bands, baseline, '-')

  # See http://stackoverflow.com/a/11558629/10601
  # and http://stackoverflow.com/a/17202196/10601
  norm = Normalize() if sweep == 'poly_order' else LogNorm()
  sm = pyplot.cm.ScalarMappable(cmap=cm, norm=norm)
  sm._A = params
  cbar = pyplot.colorbar(sm)

  if replicate_paper:
    cbar.set_label('Smoothness Parameter/Unitless')
    pyplot.xlim((1030, 1550))
    pyplot.ylim((5400, 8000))
    ax.yaxis.set_ticklabels([])
    pyplot.xlabel('Wavenumber/cm$^{-1}$')
    pyplot.ylabel('Raman Intensity')
    pyplot.tight_layout()

  if outfile is None:
    pyplot.show()
  else:
    pyplot.savefig(outfile)

if __name__ == '__main__':
  ap = ArgumentParser()
  ap.add_argument('--param', choices=('asymmetry','smoothness','order'),
                  default='smoothness',
                  help='Type of parameter to sweep. [%(default)s]')
  ap.add_argument('-o', '--outfile', nargs='?', help='Save figure to file.')
  ap.add_argument('--paper', action='store_true',
                  help='Replicate the figure used in the JRS paper.')
  ap.add_argument('spectrum')
  args = ap.parse_args()

  if args.param == 'asymmetry':
    sweep = 'asymmetry_param'
    params = np.logspace(-3, -1, 50)
  elif args.param == 'smoothness':
    sweep = 'smoothness_param'
    params = np.logspace(4, 8, 10)
  else:
    sweep = 'poly_order'
    params = np.arange(3, 12)

  run_sweep(args.spectrum, sweep, params, args.outfile, args.paper)
