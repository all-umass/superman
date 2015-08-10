#!/usr/bin/python
import numpy as np
import os.path
from argparse import ArgumentParser
from multiprocessing import Pool
from superman.file_io.opus import parse_traj


def convert(opus_file, rruff_file):
  try:
    traj, sample_params = parse_traj(opus_file, return_params=True)
  except:
    print 'Failed to parse OPUS file:', opus_file.name
    return
  header = 'Ratio data from OPUS file %s\nSample name: %s' % (
      opus_file.name, sample_params['SNM'])
  np.savetxt(rruff_file, traj, fmt='%f', header=header)


def main():
  ap = ArgumentParser()
  binfile = lambda fname: open(fname, 'rb')
  ap.add_argument('-o', '--outdir', type=str, default='.',
                  help='Output directory.')
  ap.add_argument('--procs', type=int, default=1, help='# of processes to use.')
  ap.add_argument('files', type=binfile, nargs='+', help='Input OPUS file(s).')
  args = ap.parse_args()

  if args.procs == 1:
    for f in args.files:
      _main((f, args.outdir))
  else:
    params = [(f, args.outdir) for f in args.files]
    # Don't bother keeping any results around, or doing things in order.
    for _ in Pool(args.procs).imap_unordered(_main, params):
      pass


def _main((opus_file, outdir)):
  # Make the new opus filename from the input filename.
  # Doesn't attempt to reproduce the RRUFF filename format.
  name = os.path.basename(opus_file.name) + '.txt'
  # Sanitize spaces.
  name = '_'.join(name.split())

  # Convert.
  convert(opus_file, open(os.path.join(outdir, name), 'w'))

if __name__ == '__main__':
  main()
