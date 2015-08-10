#!/usr/bin/env python
import numpy as np
import os.path
from matplotlib import pyplot as plt
from ast import literal_eval
from argparse import ArgumentParser
from superman.baseline import BL_CLASSES


def main():
  ap = ArgumentParser()
  ap.add_argument('results', type=open, help='File of pipeline results')
  args = ap.parse_args()
  filename = args.results.name
  ranges = param_ranges(filename)
  scores, params, headers = numpify(args.results)
  assert 0 < len(headers) < 3
  ax = plt.subplot(111)
  ax.set_title(os.path.basename(filename))
  if len(headers) == 1:
    h = headers[0]
    lb,ub,scale = ranges[h]
    if scale == 'log':
      ax.plot(params.ravel(), scores, '+')
    else:
      ax.semilogx(params.ravel(), scores, '+')
    ax.xlim((lb,ub))
    ax.set_xlabel(h.title().replace('_',' ').strip())
    ax.set_ylabel('CV score')
  else:
    hx,hy = headers
    rx,ry = ranges[hx], ranges[hy]
    if rx[2] == 'log':
      ax.set_xscale('log')
    if ry[2] == 'log':
      ax.set_yscale('log')
    ax.scatter(*params.T, c=scores, edgecolor='none')
    # print rx, ry
    # ax.set_xlim(rx[:2])
    # ax.set_ylim(ry[:2])
    ax.set_xlabel(hx.title().replace('_',' ').strip())
    ax.set_ylabel(hy.title().replace('_',' ').strip())
  plt.show()


def param_ranges(filename):
  bl_name = os.path.splitext(filename)[0].rsplit('_', 1)[-1]
  bl = BL_CLASSES[bl_name]()
  return bl.param_ranges()


def numpify(fh):
  scores, params = [], []
  for s, p in parse(fh):
    scores.append(s)
    params.append(p)
  scores = np.array(scores)
  headers = p.keys()
  P = np.zeros((len(scores), len(headers)))
  for i,p in enumerate(params):
    for j,h in enumerate(headers):
      P[i,j] = p[h]
  return scores, P, headers


def parse(fh):
  for line in fh:
    score, params = line.split(None, 1)
    score = float(score)
    try:
      params = literal_eval(params)  # convert the dict repr to a real dict
    except SyntaxError as e:
      print e
      continue
    yield score, params


if __name__ == '__main__':
  main()
