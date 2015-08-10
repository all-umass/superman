#!/usr/bin/python
import os.path
from argparse import ArgumentParser
from superman.file_io import write_opus, parse_rruff


def main():
  ap = ArgumentParser()
  ap.add_argument('-o', '--outdir', type=str, default='.',
                  help='Output directory.')
  ap.add_argument('--ext', type=str, default='.opus',
                  help='File extension to replace .txt with. [.opus]')
  ap.add_argument('files', type=open, nargs='+', help='Input RRUFF file(s).')
  args = ap.parse_args()

  for f in args.files:
    # Make the new opus filename from the input filename.
    outname = os.path.join(args.outdir, os.path.basename(f.name))
    outname = os.path.splitext(outname)[0] + args.ext

    # Convert.
    data, comments = parse_rruff(f, return_comments=True)
    write_opus(outname, data, comments)

if __name__ == '__main__':
  main()
