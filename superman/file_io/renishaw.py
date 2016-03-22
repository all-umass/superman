"""Mk4py, aka metakit.
Python bindings are native-compiled: http://equi4.com/metakit/python.html
"""
from __future__ import print_function
import Mk4py
import numpy as np
import os
import struct
import zlib
from argparse import ArgumentParser
from construct import (
    Struct, ULInt64, Array, Magic, LFloat64, SLInt32, ULInt16, Switch, ULInt32,
    String, Padding, Adapter, Container, GreedyRange, IfThenElse, Peek,
    Embedded, OnDemand, Const, ExprAdapter
)
from matplotlib import pyplot as plt


def parse_wxd(f):
  return WXDFile(f).load_spectrum(all_trajs=False, verbose=False)


class WXDFile(object):
  '''Renishaw WiRE (*.wxd) file format parser.'''
  def __init__(self, file_or_filepath):
    # Note: we keep self.db around to avoid the db getting GC'd
    if hasattr(file_or_filepath, 'mode'):
      self.db = Mk4py.storage(file_or_filepath)
    else:
      self.db = Mk4py.storage(file_or_filepath, 0)
    dirs = self.db.view('dirs')
    self.table = dirs[0].files[0]._B

  def print_info(self):
    print('Subfile Name  \tSize\tDate')
    print('-' * 34)
    for row in self.table:
      print('\t'.join(map(str, (row.name, row.size, row.date))))
    print('-' * 34)

  def _row_data(self, row_name):
    row, = self.table.select(name=row_name)
    return zlib.decompress(row.contents)

  def _last_row_data(self, row_name_prefix):
    row = [r for r in self.table if r.name.startswith(row_name_prefix)][-1]
    return zlib.decompress(row.contents)

  def extract_xml(self, outfile):
    data = self._row_data('XMLDocument')
    a, size = struct.unpack('li', data[:12])
    assert a == 0
    assert size == len(data[12:])
    text = data[12:].decode('utf16').encode('utf8')
    with open(outfile, 'wb') as fh:
      fh.write(text)

  def extract_properties(self, outfile):
    data = self._row_data('Properties')
    props = Properties.parse(data)
    with open(outfile, 'w') as fh:
      for p in props:
        if 'TaggedData' in p:
          print('%s\t%r' % (p.label, repr(p.TaggedData.value)), file=fh)

  def extract_analysis(self, outfile):
    # TODO: figure out how to parse this part
    # data = self._row_data('AnalysisResults')
    raise NotImplementedError('AnalysisResults parser is NYI')

  def load_spectrum(self, all_trajs=False, verbose=False):
    # load bands from the last DataList* row
    dlist = DataList.parse(self._last_row_data('DataList'))
    bands = np.array(dlist.data.value, dtype=float)
    assert dlist.size == bands.shape[0]

    # load intensities from the last DataSet* row
    dset = DataSet.parse(self._last_row_data('DataSet'))

    if verbose:
      for p in dset.Property:
        if 'TaggedData' in p:
          print(p.label, '=>', repr(p.TaggedData.value))

    if not all_trajs:
      dlist = dset.LabeledDataList[0]
      intensities = np.array(dlist.data.value, dtype=float)
      return np.column_stack((bands, intensities))

    trajs = {}
    for dlist in dset.LabeledDataList:
      intensities = np.array(dlist.data.value, dtype=float)
      trajs[dlist.label] = np.column_stack((bands, intensities))
    return trajs


class VBString(Adapter):
  def __init__(self, name):
    # TODO: replace this with construct.PascalString
    vbs = Struct(name,
                 ULInt32('length'),
                 String('value', lambda ctx: ctx.length-2),
                 Magic('\x00\x00'))  # There's always an ending null
    Adapter.__init__(self, vbs)

  def _decode(self, obj, ctx):
    return obj.value.decode('utf16').encode('utf8')

  def _encode(self, obj, ctx):
    x = obj.encode('utf16') + '\x00\x00'
    return Container(length=len(x), value=x)

SomeKindOfEnumMaybe = Struct(
    'SomeKindOfEnumMaybe',
    Padding(16),  # XXX: almost certainly useful somehow
    ULInt64('')
)
TaggedData = Struct(
    'TaggedData',
    ULInt16('tag'),
    Switch('value', lambda ctx: ctx.tag, {
        3: SLInt32(''),
        5: LFloat64(''),
        7: ULInt64(''),  # timestamp?
        8: VBString(''),
        9: SomeKindOfEnumMaybe,
        11: Magic('\x00' * 2),  # null?
        35: Magic('\x00' * 6)  # EOF
    })
)
# Specialization for loading float data faster
TaggedFloat64 = ExprAdapter(
    Struct('data', Const(ULInt16('tag'), 5), LFloat64('value')),
    encoder=lambda obj, ctx: Container(tag=obj.tag, value=obj.value),
    decoder=lambda obj, ctx: obj.value)
DataList = Struct(
    'DataList',
    ULInt64('size'),
    OnDemand(Array(lambda ctx: ctx.size, TaggedFloat64)),
    Magic('\xc0\xff\xee\x01')  # XXX: probably useful
)
# XXX: hacks
bad_strings = ('\xc0\xff\xee\x01\x00\x00', '\x01#Eg\x00\x00')
Property = Struct(
    'Property',
    Peek(String('peek', 6)),
    Embedded(IfThenElse(
        '', lambda ctx: ctx.peek in bad_strings,
        Padding(6),
        Struct('', VBString('label'), TaggedData)))
)
Properties = GreedyRange(Property)
LabeledDataList = Struct(
    'LabeledDataList',
    VBString('label'),
    Padding(18),
    Embedded(DataList)
)
DataSet = Struct(
    'DataSet',
    ULInt64('number'),
    # XXX: may have more than two. Might use ctx.number to decide?
    Array(2, LabeledDataList),
    Properties
)


if __name__ == '__main__':
  def main():
    ap = ArgumentParser()
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--xml', action='store_true',
                    help='Extract the associated XML document.')
    ap.add_argument('--props', action='store_true', help='Extract properties.')
    ap.add_argument('--analysis', action='store_true',
                    help='Extract analysis results.')
    ap.add_argument('--plot', action='store_true', help='Plot all spectra.')
    ap.add_argument('files', nargs='+', type=open)
    args = ap.parse_args()
    if args.analysis:
      ap.error('--analysis is NYI at this point')
    for f in args.files:
      wxd = WXDFile(f)
      if args.verbose:
        wxd.print_info()
      if args.xml:
        wxd.extract_xml(f.name + '.xml')
      if args.props:
        wxd.extract_properties(f.name + '.props.txt')
      if args.analysis:
        wxd.extract_analysis(f.name + '.analysis.txt')
      if args.plot:
        spectra = wxd.load_spectrum(all_trajs=True, verbose=args.verbose)
        plt.figure()
        plt.title(os.path.basename(f.name))
        for label, traj in spectra.iteritems():
          plt.plot(*traj.T, label=label)
        plt.legend()
    if args.plot:
      plt.show()

  main()
