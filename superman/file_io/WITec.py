from __future__ import print_function
import numpy as np
from argparse import ArgumentParser
from construct import (
    Struct, Magic, Rename, Array, PascalString, ULInt32, SLInt64, Enum, Value,
    OneOf, If, RepeatUntil, Field, LFloat64, LFloat32, SLInt32, Switch,
    LazyBound, ULInt8, Anchor, Padding
)

WIPTagType = Enum(
    ULInt32('type'),
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


def WIPString(name):
  return PascalString(name, length_field=ULInt32('length'))


WIPTag = Struct(
    'WIPTag',
    WIPString('name'),
    WIPTagType,
    SLInt64('data_start'),
    SLInt64('data_end'),
    # Anchor('start'),
    # OneOf(Value('at_start', lambda ctx: ctx.start == ctx.data_start), [True]),
    Value('data_size', lambda ctx: ctx.data_end - ctx.data_start),
    If(lambda ctx: ctx.data_end > ctx.data_start,
        Switch('data', lambda ctx: ctx.type, dict(
            WIP_TAG_LIST=RepeatUntil(
                lambda obj, ctx: obj.data_end >= ctx.data_end,
                LazyBound('', lambda: WIPTag)),
            WIP_TAG_EXTENDED=Field('', 10),
            WIP_TAG_DOUBLE=LFloat64(''),
            WIP_TAG_FLOAT=LFloat32(''),
            WIP_TAG_INT64=SLInt64(''),
            WIP_TAG_INT32=Array(lambda ctx: ctx.data_size//4, SLInt32('')),
            WIP_TAG_UINT32=Array(lambda ctx: ctx.data_size//4, ULInt32('')),
            WIP_TAG_CHAR=Array(lambda ctx: ctx.data_size, ULInt8('')),
            WIP_TAG_BOOL=ULInt8(''),
            WIP_TAG_STRING=WIPString(''),
        ))),
    Anchor('end'),
    # pad to get to data_end
    Padding(lambda ctx: ctx.data_end - ctx.end),
)

WIPFile = Struct(
    'WIPFile',
    Magic('WIT_PR06'),  # alternately, "WIT_PRCT"
    Rename('root', WIPTag),
    # Validate the root name
    OneOf(Value('root_name', lambda ctx: ctx.root.name), ['WITec Project'])
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
  '''
  half_gamma = trans['Gamma'] / 2.0
  cos_half_gamma = np.cos(half_gamma)
  foo = trans['LambdaC'] * trans['m'] / trans['d'] / 2.0 / cos_half_gamma
  alpha = np.arcsin(foo) - half_gamma
  betac = trans['Gamma'] + alpha
  delta = trans['Delta']
  hc = trans['f'] * np.sin(delta)
  lh = trans['f'] * np.cos(delta)
  hi = hc + trans['x'] * (trans['nC'] - pixels)
  betai = betac + delta - np.arctan2(hi, lh)
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

  for name, tdgraph in tdgraphs.iteritems():
    num_pts = tdgraph['SizeGraph']
    ydata = dict(map(parse_tag, tdgraph['GraphData']))['Data']
    ydata = np.array(ydata, dtype=np.uint8).view(np.float32)

    xdata = np.arange(num_pts, dtype=np.float32)
    xt_id = tdgraph['XTransformationID']
    xtrans = xtrans_map[xt_id]
    xdata = pixel2lambda(xdata, xtrans)

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
