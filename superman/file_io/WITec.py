from __future__ import print_function
import numpy as np
from argparse import ArgumentParser
from construct import (
    Struct, Const, Array, PascalString, Int32ul, Int64sl, Enum, Computed,
    OneOf, If, RepeatUntil, Bytes, Float64l, Float32l, Int32sl, Switch,
    LazyBound, Int8ul, Tell, Padding, this
)
from six.moves import xrange

WIPTagType = Enum(
    Int32ul,
    WIP_TAG_LIST=0,  # list of other tags
    WIP_TAG_EXTENDED=1,  # x86 FPU native type, 10 bytes
    WIP_TAG_DOUBLE=2,
    WIP_TAG_FLOAT=3,
    WIP_TAG_INT64=4,
    WIP_TAG_INT32=5,
    WIP_TAG_UINT32=6,
    WIP_TAG_CHAR=7,
    WIP_TAG_BOOL=8,  # 1 byte
    WIP_TAG_STRING=9   # int32 = nchars, n bytes = string
)


WIPString = PascalString(lengthfield=Int32ul)


WIPTag = Struct(
    'WIPTag',
    'name'/WIPString,
    'type'/WIPTagType,
    'data_start'/Int64sl,
    'data_end'/Int64sl,
    # 'start'/Tell,
    # OneOf(Computed(this.start == this.data_start), [True]),
    'data_size'/Computed(this.data_end - this.data_start),
    'data'/If(
        this.data_end > this.data_start,
        Switch(this.type, dict(
            WIP_TAG_LIST=RepeatUntil(
                lambda obj, ctx: obj.data_end >= ctx.data_end,
                LazyBound(lambda: WIPTag)),
            WIP_TAG_EXTENDED=Bytes(10),
            WIP_TAG_DOUBLE=Float64l,
            WIP_TAG_FLOAT=Float32l,
            WIP_TAG_INT64=Int64sl,
            WIP_TAG_INT32=Array(this.data_size//4, Int32sl),
            WIP_TAG_UINT32=Array(this.data_size//4, Int32ul),
            WIP_TAG_CHAR=Array(this.data_size, Int8ul),
            WIP_TAG_BOOL=Int8ul,
            WIP_TAG_STRING=WIPString,
        ))),
    'end'/Tell,
    # pad to get to data_end
    Padding(this.data_end - this.end),
)

WIPFile = Struct(
    Const(b'WIT_PR06'),  # alternately, "WIT_PRCT"
    'root'/WIPTag,
    # Validate the root name
    OneOf(Computed(this.root.name), ['WITec Project'])
)


def parse_tag(tag, shallow=False):
  data = tag.data
  if data is None:
    return tag.name, data
  if tag.type in ('WIP_TAG_INT32', 'WIP_TAG_UINT32', 'WIP_TAG_CHAR'):
    if len(data) == 1:
      data = data[0]
    else:
      data = np.array(data)
  elif shallow and tag.type == 'WIP_TAG_LIST':
    data = None
  return tag.name, data


def print_tag(tag, level=0):
  name, data = parse_tag(tag, shallow=True)
  print(' '*level, name, sep='', end=' ')
  if data is not None:
    print('->', data)
  else:
    print()


def walk_tag_tree(tag, level=0):
  print_tag(tag, level=level)
  if tag.type == 'WIP_TAG_LIST' and tag.data is not None:
    for child in tag.data:
      walk_tag_tree(child, level=level+1)


def taglist2dict(tag):
  assert tag.type == 'WIP_TAG_LIST' and tag.data is not None
  return dict(map(parse_tag, tag.data))


def pixel2lambda(pixels, trans):
  '''spectral transform from here:
  horiba.com/us/en/scientific/products/optics-tutorial/wavelength-pixel-position
  # Wavelength at center of array (nm)
  lambda_c = trans['LambdaC']
  # Pixel number at center of array
  p_c = trans['nC']
  # Pixel width (mm)
  p_w = trans['x']
  # pixel number at wavelength lambda
  p_lambda = pixels + 1  # numbers start from 1
  # instrument focal length (mm)
  f = trans['f']
  # Inclination of the focal plane measured at lambda_c
  gamma = trans['Gamma']
  # Perpendicular distance from grating/focusing mirror to the focal plane (mm)
  l_h = f * np.cos(gamma)
  # Angle of diffraction at center wavelength
  beta_lambda_c = 0 # XXX
  # Angle from l_h to the normal to the grating
  beta_h = beta_lambda_c + gamma
  # Distance from the intercept of the normal to the focal plane to lambda_c
  h_b_lambda_c = f * np.sin(gamma)
  # Distance from the intercept of the normal to the focal plane to lambda_n
  h_b_lambda_n = p_w * (p_lambda - p_c) + h_b_lambda_c
  # Angle of diffraction at wavelength n
  beta_lambda_n = beta_h - np.arctan2(h_b_lambda_n / l_h)
  # delta lambda, wavelength resolution
  d_lambda = trans['Delta']
  # diffraction order (integer)
  k = trans['m']  # ???
  # groove density (grooves / mm)
  # n = ???
  # Rayleigh criterion, a.k.a. resolving power
  r = lambda_c / d_lambda
  # included/deviation angle
  d_v = gamma  # ???
  # angle of incidence
  alpha = np.arcsin((r*k)/(2*np.cos(d_v/2))) - (d_v/2)
  # wavelength at channel n (desired result, in nm)
  lambda_n = (np.sin(alpha) + np.sin(beta_lambda_n)) * r
  return lambda_n
  '''
  # The following is translated from Gwyddion's C pixel_to_lambda function
  half_gamma = trans['Gamma'] / 2.0
  cos_half_gamma = np.cos(half_gamma)
  foo = trans['LambdaC'] * trans['m'] / trans['d'] / (2*cos_half_gamma)
  alpha = np.arcsin(foo) - half_gamma
  betac = trans['Gamma'] + alpha
  delta = trans['Delta']
  bh = betac + trans['Gamma']
  hc = trans['f'] * np.sin(delta)
  lh = trans['f'] * np.cos(delta)
  hbln = trans['x'] * (trans['nC'] - pixels) + hc
  betai = bh - np.arctan2(hbln, lh)
  return (trans['d'] / trans['m']) * (np.sin(alpha) + np.sin(betai))


def extract_spectra(fh):
  if not hasattr(fh, 'read'):
    fh = open(fh)
  wp = WIPFile.parse_stream(fh)
  data_tags = taglist2dict(wp.root)['Data']
  _, ndata = parse_tag(data_tags[-1])
  assert len(data_tags) == ndata * 2 + 1

  xtrans_map = {}  # ID -> xtrans dict
  tdgraphs = {}  # caption -> tdgraph tag
  for idx in xrange(0, ndata*2, 2):
    _, data_class_name = parse_tag(data_tags[idx])
    if data_class_name == 'TDGraph':
      meta, tdgraph = map(taglist2dict, data_tags[idx+1].data)
      name = meta['Caption']
      assert tdgraph['SizeX'] == 1 and tdgraph['SizeY'] == 1
      tdgraphs[name] = tdgraph
    elif data_class_name == 'TDSpectralTransformation':
      tdata, _, xtrans = map(taglist2dict, data_tags[idx+1].data)
      xtrans_map[tdata['ID']] = xtrans

  for name, tdgraph in tdgraphs.items():
    num_pts = tdgraph['SizeGraph']
    ydata = dict(map(parse_tag, tdgraph['GraphData']))['Data']
    ydata = np.array(ydata, dtype=np.uint8).view(np.float32)

    xdata = np.arange(num_pts, dtype=np.float32)
    xt_id = tdgraph['XTransformationID']
    xtrans = xtrans_map[xt_id]
    xdata = pixel2lambda(xdata, xtrans)
    # Convert nm to 1/cm ???
    #xdata = 1. / (xdata * 1e-7)

    yield name, np.column_stack((xdata, ydata))


if __name__ == '__main__':
  import os
  from matplotlib import pyplot as plt

  def plot_spectra(fh, legend=False):
    plt.figure()
    plt.title(os.path.basename(fh.name))
    for name, traj in extract_spectra(fh):
      plt.plot(*traj.T, label=name)
    if legend:
      plt.legend()

  def main():
    ap = ArgumentParser()
    ap.add_argument('--dump', action='store_true')
    ap.add_argument('--legend', action='store_true')
    ap.add_argument('files', type=open, nargs='+')
    args = ap.parse_args()
    for fh in args.files:
      if args.dump:
        walk_tag_tree(WIPFile.parse_stream(fh).root)
      else:
        plot_spectra(fh, legend=args.legend)
    if not args.dump:
      plt.show()

  main()
