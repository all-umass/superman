'''
Useful for vizualizing the effects of PCA preprocessing,
or other dimension reduction.
'''
import numpy as np
from viztricks import plot

import options
import dana
from preprocess import preprocess
from utils import prepare_data


def main():
  op = options.setup_common_opts()
  options.add_preprocess_opts(op)
  opts = options.parse_opts(op)
  options.validate_preprocess_opts(op, opts)

  data_file = options.find_data_file(opts)
  for X, Y, label_map, _ in prepare_data(data_file, opts):
    for pp in opts.pp:
      show = _main(X, Y, label_map, pp)
  show()


def _main(X, Y, label_map, pp):
  emb = preprocess(X, pp)[:,:3]  # take at most 3 dimensions
  class_labels = dana.convert_to_dana(label_map, Y)['klass']
  _, class_Y = np.unique(class_labels, return_inverse=True)
  return plot(emb, 'o', scatter=True, c=class_Y, edgecolor='none', title=pp)


if __name__ == '__main__':
  main()
