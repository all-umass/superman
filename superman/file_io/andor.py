import numpy as np


def parse_sif(fh):
  # Parser requires binary file mode
  if hasattr(fh, 'mode') and 'b' not in fh.mode:
    fh = open(fh.name, 'rb')

  # Verify we have a SIF file
  if fh.readline().strip() != "Andor Technology Multi-Channel File":
    raise ValueError("Not an Andor SIF file: %s" % fh.name)

  while True:
    line = fh.readline()
    if not line:
      raise ValueError('No valid SIF data found.')
    if line.startswith('Pixel number'):
      break

  left, top, right, bottom, hbin, vbin = map(int, fh.readline().split()[1:7])
  width = (right-left+1) // hbin
  height = (top-bottom+1) // vbin

  fh.readline()
  fh.readline()

  size = width * height
  data = np.fromstring(fh.read(size*4), dtype=np.float32, count=size)
  # XXX: I don't know how to recover the bands, so we fake 'em.
  bands = np.arange(size)
  return np.column_stack((bands, data))
