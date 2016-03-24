#!/usr/bin/python
import os.path
from argparse import ArgumentParser
from multiprocessing import Pool
from superman.file_io import parse_spectrum, write_opus, write_rruff

try:
  from itertools import imap
except ImportError:
  imap = map


def _parse(infile):
  return infile, parse_spectrum(infile)


def main():
  writers = dict(opus=write_opus, rruff=write_rruff)
  extensions = dict(opus='.opus', rruff='.txt')
  ap = ArgumentParser()
  ap.add_argument('-f', '--fmt', choices=writers, help='Format to convert to.')
  ap.add_argument('-o', '--outdir', default='.', help='Output directory.')
  ap.add_argument('--procs', type=int, default=1, help='# of processes to use.')
  ap.add_argument('files', nargs='+', help='Input file(s).')
  args = ap.parse_args()

  if args.procs > 1:
    imap = Pool(args.procs).imap_unordered

  writer = writers[args.fmt]
  ext = extensions[args.fmt]
  for infile, traj in imap(_parse, args.files):
    # Note: We append instead the new file extension (instead of replacing),
    #   because the original extension can carry useful information.
    outfile = os.path.join(args.outdir, os.path.basename(infile)) + ext
    writer(outfile, traj, 'Converted spectrum from file: ' + infile)


if __name__ == '__main__':
  main()
