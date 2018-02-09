from __future__ import absolute_import, print_function
import numpy as np
from collections import defaultdict
from construct import (
    Array, Enum, Const, Pointer, Container,
    RepeatUntil, Bytes, Struct, Switch, If,
    Float32l, Float64l, Int16ul, Int32ul, this, obj_
)

from .construct_utils import (
    BitSplitter, FixedSizeCString, FunctionSwitch, LazyField
)


BlockType = BitSplitter(Int32ul,
                        complex=(0, 2), type=(2, 2), param=(4, 6),
                        data=(10, 7), deriv=(17, 2), extend=(19, 3))

BlockType_decoder = {
    'complex': {0: '', 1: 'real', 2: 'imaginary', 3: 'amplitude'},
    'type': {0: '', 1: 'sample', 2: 'reference', 3: 'ratio'},
    'deriv': {0: '', 1: 'first deriv', 2: 'second deriv', 3: 'nth deriv'},
    'extend': {0: '', 1: 'compound info', 2: 'peak table',
               3: 'molecular structure', 4: 'macro', 5: 'command log'},
    'data': {0: '', 1: 'spectrum, undefined Y units', 2: 'interferogram',
             3: 'phase spectrum', 4: 'absorbance spectrum',
             5: 'transmittance spectrum', 6: 'kubelka-munck spectrum',
             7: 'trace', 8: 'gc file (interferograms)', 9: 'gc file (spectra)',
             10: 'raman spectrum', 11: 'emission spectrum',
             12: 'reflectance spectrum', 13: 'directory',
             14: 'power spectrum', 15: 'neg. log reflectance',
             16: 'ATR spectrum', 17: 'photoacoustic spectrum',
             18: 'arithmetics (transmittance)', 19: 'arithmetics (absorbance)'},
    'param': {0: '', 1: 'data status', 2: 'instrument status', 3: 'acquisition',
              4: 'FT', 5: 'plot/display', 6: 'processing', 7: 'GC',
              8: 'library search', 9: 'communication', 10: 'sample origin'}
}
# Wrap in defaultdicts.
for k in BlockType_decoder.keys():
  BlockType_decoder[k] = defaultdict(lambda: 'unknown', BlockType_decoder[k])


def prettyprint_blocktype(bt):
  res = []
  for key in ('complex','type','deriv','extend','data','param'):
    name = BlockType_decoder[key][bt[key]]
    if name:
      res.append(name)
  if bt.param != 0:
    res.append('parameters')
  return ' '.join(res)

Parameter = Struct(
    'Name'/FixedSizeCString(4),  # 4 bytes, 3 chars + null
    'Type'/Enum(Int16ul, INT32=0, REAL64=1, STRING=2, ENUM=3, SENUM=4),
    'ReservedSpace'/Int16ul,
    # Only look for a Value if this isn't the END pseudo-parameter.
    'Value'/If(
        this.Name != b'END',
        Switch(this.Type, {'INT32': Int32ul, 'REAL64': Float64l},
               default=FixedSizeCString(this.ReservedSpace * 2))
    )
)


def is_ParameterList(block):
  return block.BlockType.param != 0

ParameterList = RepeatUntil(obj_.Name == b'END', Parameter)
FloatData = Array(this.BlockLength, Float32l)
StringData = Bytes(this.BlockLength*4)

DirectoryEntry = Struct(
    'BlockType'/BlockType,
    'BlockLength'/Int32ul,
    'DataPtr'/Int32ul,
    'Block'/Pointer(
        this.DataPtr,
        FunctionSwitch([
            (is_ParameterList, ParameterList),
            (lambda ctx: ctx.BlockType.extend != 0, LazyField(StringData)),
            (lambda ctx: ctx.BlockType.data not in (0,13), LazyField(FloatData))
        ])
    )
)

# The entire file.
OpusFile = Struct(
    Const(b'\n\n\xfe\xfe'),  # 0x0a0afefe magic
    'Version'/Float64l,
    'FirstDirPtr'/Int32ul,
    'MaxDirSize'/Int32ul,
    'CurrDirSize'/Int32ul,
    'Directory'/Pointer(this.FirstDirPtr,
                        Array(this.MaxDirSize, DirectoryEntry)))


def iter_blocks(opus_data):
  for d in opus_data.Directory:
    if d.DataPtr == 0:
      break
    label = prettyprint_blocktype(d.BlockType)
    yield label, d


def parse_traj(fh, return_params=False):
  '''Parses out the "ratio" data from an OPUS file.'''
  # Parser requires binary file mode
  if hasattr(fh, 'mode') and 'b' not in fh.mode:
    fh = open(fh.name, 'rb')
  data = OpusFile.parse_stream(fh)
  y_vals = None
  sample_params = None
  for label, d in iter_blocks(data):
    if label == 'sample origin parameters':
      sample_params = dict((p.Name, p.Value) for p in d.Block)
      continue
    if 'ratio' not in label:
      continue
    if label.endswith('data status parameters'):
      params = dict((p.Name, p.Value) for p in d.Block)
    elif d.BlockType.data != 0 and d.BlockType.extend == 0:
      y_vals = np.array(d.Block())
      # Hacky fix for a strange issue where the first/last value is exactly zero
      if y_vals[0] == 0 and y_vals[1] > 1.0:
        y_vals = y_vals[1:]
      if y_vals[-1] == 0 and y_vals[-2] > 1.0:
        y_vals = y_vals[:-1]
  if y_vals is None:
    raise ValueError('No ratio data found.')

  y_vals *= params[b'CSF']  # CSF == scale factor
  x_vals = np.linspace(params[b'FXV'], params[b'LXV'], len(y_vals))
  traj = np.transpose((x_vals, y_vals))
  # Some spectra are flipped.
  if traj[0,0] > traj[-1,0]:
    traj = traj[::-1]
  if return_params:
    return traj, sample_params
  return traj


def write_opus(fname, traj, comments):
  '''Write an OPUS file to `fname`.'''
  # Convert comments to bytes, if needed.
  if hasattr(comments, 'encode'):
    comments = comments.encode('utf-8')
  # Ensure comment length is a multiple of 4.
  nc = len(comments)
  if nc % 4 != 0:
    nc = ((nc//4)+1)*4
    comments = comments.ljust(nc, b' ')

  # Sanity check band step size.
  bands, ampl = traj.T
  db = np.diff(bands)
  if np.std(db) > 0.001:
    # Resample to the mean band step size.
    new_bands = np.linspace(bands[0], bands[-1], bands.size)
    ampl = np.interp(new_bands, bands, ampl)
    bands = new_bands

  meta_param = [
      Container(Name=b'DPF', Type='INT32', Value=1, ReservedSpace=0),
      Container(Name=b'NPT', Type='INT32', Value=bands.size, ReservedSpace=0),
      Container(Name=b'FXV', Type='INT32', Value=int(bands[0]), ReservedSpace=0),
      Container(Name=b'LXV', Type='INT32', Value=int(bands[-1]), ReservedSpace=0),
      Container(Name=b'CSF', Type='REAL64', Value=1.0, ReservedSpace=0),
      Container(Name=b'MXY', Type='REAL64', Value=ampl.max(), ReservedSpace=0),
      Container(Name=b'MNY', Type='REAL64', Value=ampl.min(), ReservedSpace=0),
      Container(Name=b'DXU', Type='ENUM', Value=b'WN', ReservedSpace=2),
      Container(Name=b'END', Type='INT32', Value=0, ReservedSpace=0)
  ]
  meta_param_size = 30  # 3 for each param, +1 for each 64-bit

  # Block types for each directory block.
  dir_bt = Container(deriv=0, extend=0, data=13, param=0, complex=0, type=0)
  data_bt = Container(deriv=0, extend=0, data=1, param=0, complex=1, type=3)
  meta_bt = Container(deriv=0, extend=0, data=1, param=1, complex=1, type=3)
  comment_bt = Container(deriv=0, extend=5, data=0, param=0, complex=0, type=0)

  # Directory blocks, with zeros where data will be filled in later.
  directory = [
      Container(BlockType=dir_bt, DataPtr=0, BlockLength=0, Block=None),
      Container(BlockType=data_bt, DataPtr=0,
                BlockLength=ampl.size, Block=ampl.tolist()),
      Container(BlockType=meta_bt, DataPtr=0,
                BlockLength=meta_param_size, Block=meta_param),
      Container(BlockType=comment_bt, DataPtr=0,
                BlockLength=nc//4, Block=comments)
  ]

  # Fill in directory block information.
  ptr = 24
  directory[0].BlockLength = len(directory) * 3
  for d in directory:
    d.DataPtr = ptr
    ptr += d.BlockLength * 4

  # Assemble the whole file and write it to disk.
  opus_obj = Container(Version=920622.0, FirstDirPtr=24,
                       MaxDirSize=len(directory), CurrDirSize=len(directory),
                       Directory=directory)

  with open(fname, 'wb') as fh:
    OpusFile.build_stream(opus_obj, fh)


if __name__ == '__main__':

  def main():
    from argparse import ArgumentParser
    op = ArgumentParser()
    op.add_argument('--print', action='store_true', dest='_print')
    op.add_argument('--plot', action='store_true')
    op.add_argument('--filter', type=str, default='',
                    help='Only show plots with titles matching this substring.')
    op.add_argument('file', type=open, help='OPUS file.')
    opts = op.parse_args()
    if not (opts._print or opts.plot):
      op.error('Must supply either --plot or --print.')

    data = OpusFile.parse_stream(opts.file)
    if opts._print:
      prettyprint_opus(data)
    if opts.plot:
      plot_opus(data, opts.filter)

  def prettyprint_opus(data):
    np.set_printoptions(precision=4, suppress=True)
    print('OPUS file, version', data.Version,
          '%d/%d blocks' % (data.CurrDirSize, data.MaxDirSize))
    for label, d in iter_blocks(data):
      print('[%x:%x]' % (d.DataPtr, d.DataPtr+d.BlockLength*4), label)
      if d.Block is None:
        continue
      if is_ParameterList(d):
        for p in d.Block[:-1]:  # Don't bother printing the END block.
          print('   ', p.Name, p.Value)
      else:
        foo = np.array(d.Block())
        print('    data:', foo.shape, foo[:6] if foo.ndim > 0 else foo)

  def plot_opus(data, title_pattern=''):
    from matplotlib import pyplot
    plot_info = defaultdict(dict)
    for label, d in iter_blocks(data):
      if d.Block is None:
        continue
      if label.endswith('data status parameters'):
        key = label[:-23]
        plot_info[key]['params'] = dict((p.Name, p.Value) for p in d.Block)
      elif d.BlockType.data != 0 and d.BlockType.extend == 0:
        plot_info[label]['data'] = np.array(d.Block())
    DXU_values = {
        'WN': 'Wavenumber (1/cm)', 'MI': 'Micron', 'LGW': 'log Wavenumber',
        'MIN': 'Minutes', 'PNT': 'Points'
    }
    for label, foo in plot_info.items():
      if 'data' not in foo or 'params' not in foo:
        continue
      y_type, title = label.split(' ', 1)
      if title_pattern not in title:
        print('Skipping "%s"' % title)
        continue
      params = foo['params']
      x_units = DXU_values[params['DXU']]
      y_vals = foo['data'] * params['CSF']  # CSF == scale factor
      x_vals = np.linspace(params['FXV'], params['LXV'], len(y_vals))

      pyplot.figure()
      pyplot.plot(x_vals, y_vals)
      pyplot.title(title)
      pyplot.xlabel(x_units)
      pyplot.ylabel(y_type)
    pyplot.show()

  main()
