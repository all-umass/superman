#!/usr/bin/env python
import os
import numpy as np
from matplotlib import pyplot
from argparse import ArgumentParser
from superman.baseline import BL_CLASSES
from superman.file_io import parse_spectrum


def main():
  ap = ArgumentParser()
  ap.add_argument('--algorithm', default='als', choices=BL_CLASSES.keys(),
                  help='Algorithm type. [%(default)s]')
  ap.add_argument('-s', '--segment', action='store_true',
                  help='Auto-detect band segments and run on each separately.')
  ap.add_argument('-o', '--out-dir', help='If provided, write corrected files'
                                          ' to disk instead of plotting them.')
  ap.add_argument('files', type=open, nargs='+')
  args = ap.parse_args()

  for f in args.files:
    S = parse_spectrum(f)
    bands, intensities = S.T
    bl = BL_CLASSES[args.algorithm]()
    corrected = bl.fit_transform(bands, intensities, segment=args.segment)

    if args.out_dir:
      out_file = os.path.join(args.out_dir, os.path.basename(f.name))
      if os.path.exists(out_file):
        print 'Skipping existing file:', out_file
        continue
      header = ('Baseline corrected data from original file %s\n'
                'BL Algorithm: %s (default params)' % (f.name, args.algorithm))
      traj = np.column_stack((bands, corrected))
      np.savetxt(out_file, traj, fmt='%f', header=header)
    else:
      _, (ax1,ax2) = pyplot.subplots(nrows=2)
      ax1.plot(bands, intensities, 'r-')
      ax1.plot(bands, bl.baseline, 'k-')
      ax2.plot(bands, corrected, '-')
      pyplot.suptitle(f.name)
  pyplot.show()

if __name__ == '__main__':
  main()
