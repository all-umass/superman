#!/usr/bin/python
from argparse import ArgumentParser
from os.path import basename
from matplotlib import pyplot
from superman.baseline import BL_CLASSES
from superman.utils import resample
from superman.file_io import parse_spectrum

ap = ArgumentParser()
ap.add_argument('raw_file', metavar='raw', help='Raw spectrum file.')
ap.add_argument('corr_file', metavar='corrected', nargs='?',
                help='Optional corrected spectrum file.')
args = ap.parse_args()

S = parse_spectrum(args.raw_file)
bands, intensities = S.T

als = BL_CLASSES['als']().fit(bands, intensities).baseline
mario = BL_CLASSES['mario']().fit(bands, intensities).baseline
airpls = BL_CLASSES['airpls']().fit(bands, intensities).baseline
fabc = BL_CLASSES['fabc']().fit(bands, intensities).baseline
kk = BL_CLASSES['kk']().fit(bands, intensities).baseline

num_plots = 5

if args.corr_file:
  corr_bands, corr_intensities = parse_spectrum(args.corr_file).T
  # corrected file and raw file bands don't always match up perfectly
  raw_intensities = resample(S, corr_bands)
  bruker = raw_intensities - corr_intensities
  num_plots += 1

_, axes = pyplot.subplots(nrows=2, ncols=num_plots)
axes[0,0].set_title('ALS')
axes[0,0].plot(bands, intensities, 'r-')
axes[0,0].plot(bands, als, 'k-')
axes[1,0].plot(bands, intensities-als, 'b-')

axes[0,1].set_title('Mario')
axes[0,1].plot(bands, intensities, 'r-')
axes[0,1].plot(bands, mario, 'k-')
axes[1,1].plot(bands, intensities-mario, 'b-')

axes[0,2].set_title('airPLS')
axes[0,2].plot(bands, intensities, 'r-')
axes[0,2].plot(bands, airpls, 'k-')
axes[1,2].plot(bands, intensities-airpls, 'b-')

axes[0,3].set_title('FABC')
axes[0,3].plot(bands, intensities, 'r-')
axes[0,3].plot(bands, fabc, 'k-')
axes[1,3].plot(bands, intensities-fabc, 'b-')

axes[0,4].set_title('Kajfosz-Kwiatek')
axes[0,4].plot(bands, intensities, 'r-')
axes[0,4].plot(bands, kk, 'k-')
axes[1,4].plot(bands, intensities-kk, 'b-')

if args.corr_file:
  axes[0,-1].set_title('Bruker')
  axes[0,-1].plot(corr_bands, raw_intensities, 'r-')
  axes[0,-1].plot(corr_bands, bruker, 'k-')
  axes[1,-1].plot(corr_bands, corr_intensities, 'b-')

# Disable all y-ticklabels
for ax in axes.flat:
  ax.yaxis.set_ticklabels([])
pyplot.suptitle(basename(args.raw_file))
pyplot.show()
