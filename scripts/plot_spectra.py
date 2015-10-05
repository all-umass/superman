#!/usr/bin/env python
from argparse import ArgumentParser
from os.path import basename
from matplotlib import pyplot
from superman.baseline import BL_CLASSES
from superman.preprocess import preprocess
from superman.file_io import parse_spectrum


ap = ArgumentParser()
ap.add_argument('--no-legend', action='store_false', dest='legend',
                help='Turn off figure legend')
ap.add_argument('--pp', default='normalize:max', help='Preprocess string')
ap.add_argument('--baseline', default='none', help='Baseline algorithm',
                choices=BL_CLASSES.keys() + ['none'])
ap.add_argument('file', nargs='+', help='Spectrum file(s).')
args = ap.parse_args()

fig, ax = pyplot.subplots(figsize=(12,6))
for f in args.file:
  bands, intensities = parse_spectrum(f).T
  if args.baseline != 'none':
    bl_alg = BL_CLASSES[args.baseline]()
    intensities = bl_alg.fit_transform(bands, intensities)
  intensities = preprocess(intensities[None], args.pp).ravel()
  ax.plot(bands, intensities, label=basename(f))
ax.yaxis.set_ticklabels([])
ax.set_xlabel('Wavenumber (1/cm)')
ax.set_ylabel('Intensity')
if args.legend:
  pyplot.legend(loc='best')
pyplot.tight_layout()
pyplot.show()
