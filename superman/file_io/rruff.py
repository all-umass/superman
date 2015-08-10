import numpy as np
import os.path
import re

# Cache variables
XRD_MINERALS = None
SQ2_MINERALS = None


def parse(fh, return_comments=False):
  if not hasattr(fh, 'read'):
    fh = open(fh)
  matcher = re.compile(r'^\s*([^, ]+)[, ]+([^, ]+)')
  data = []
  comments = []
  for line in fh:
    if line[:1] == '#':
      if line.startswith('##END'):
        break
      comments.append(line)
      continue
    m = matcher.match(line)
    if not m:
      raise ValueError('Failed to parse line:\n' + line)
    data.append(map(float, m.groups()))
  data = np.array(data, dtype=np.float32)  # Must be float32 for OPUS format.
  if return_comments:
    return data, ''.join(comments)
  return data


def sample_name(filepath):
  # Talc__R040137__Raman__785__0__unoriented__Raman_Data_Processed__19798.txt
  name_parts = os.path.basename(filepath).rstrip()[:-4].split('__')
  # Talc,R040137,Raman,785,0,unoriented,Raman_Data_Processed,19798
  name = '-'.join((name_parts[0], name_parts[-1]))
  # Talc-19798
  return name, name_parts[1]  # R040137


def xrd_minerals():
  global XRD_MINERALS
  if XRD_MINERALS is None:
    path = os.path.join(os.path.dirname(__file__),
                        '../../raman/data/xrd_minerals.tsv')
    XRD_MINERALS = set(np.recfromtxt(path)[:,1])
  return XRD_MINERALS


def sq2_minerals():
  global SQ2_MINERALS
  if SQ2_MINERALS is None:
    path = os.path.join(os.path.dirname(__file__),
                        '../../raman/data/quality.tsv')
    quality = np.genfromtxt(path, names=True, dtype=None)
    mask = quality['SQ'] == 2
    SQ2_MINERALS = set(quality[mask]['RRUFF'])
  return SQ2_MINERALS
