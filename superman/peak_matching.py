import numpy as np
import scipy.signal
import scipy.sparse
import sys
from time import time

from mp import get_mp_pool
from preprocess import savitzky_golay, preprocess
import options


# Has to be at the module-level for multiprocessing to know about it.
def scipy_peaks(args):
  # TODO: tune kwargs to get better peaks
  return scipy.signal.find_peaks_cwt(*args)


def savitzky_golay_peaks(spectra, min_interpeak_separation=10,
                         peak_percentile=80):
  deriv = savitzky_golay(spectra, deriv=1)
  # possible sign change values are [-2,0,2]
  sign_change = np.diff(np.sign(deriv).astype(int))
  all_peaks = []
  idx = np.arange(spectra.shape[1])
  for i,dy in enumerate(deriv):
    # find indices where the deriv crosses zero
    maxima, = np.where(sign_change[i] == 2)
    # interpolate
    peak_locs = maxima + dy[maxima] / (dy[maxima] - dy[maxima+1])
    # cut out short peaks (TODO: use the data to better set min_peak_height?)
    peaks = np.interp(peak_locs, idx, spectra[i])
    min_peak_height = np.percentile(peaks, peak_percentile)
    peak_locs = peak_locs[peaks >= min_peak_height]
    # cut out close peaks
    peak_sep = np.ediff1d(peak_locs, to_begin=[min_interpeak_separation])
    peak_locs = peak_locs[peak_sep >= min_interpeak_separation]
    all_peaks.append(peak_locs)
  return all_peaks


def scipy_peaks_batch(spectra, parallel=True):
  # TODO: look into the source to see if we can vectorize this
  #  -> https://github.com/scipy/scipy/blob/master/scipy/signal/_peak_finding.py
  widths = np.arange(1, 10)
  all_args = ((s,widths) for s in spectra)
  if parallel:
    return get_mp_pool(None).map(find_peaks, all_args)
  return map(find_peaks, all_args)


def threshold_peaks(spectra, num_stdv=4):
  mu = np.mean(spectra, axis=1)
  std = np.std(spectra, axis=1)
  thresh = mu + num_stdv*std
  mask = spectra > thresh[:,None]
  return [np.where(m)[0] for m in mask]


def _dense_peaks(spectra, opts):
  tic = time()
  all_peaks = find_peaks(spectra, opts)
  print '%s: %.3f secs' % (opts.peak_alg, time() - tic)
  num_peaks = opts.num_peaks
  dense = np.zeros((spectra.shape[0], num_peaks))
  for i,peaks in enumerate(all_peaks):
    n = len(peaks)
    if n >= num_peaks:
      dense[i,:] = peaks[:num_peaks]
    else:
      dense[i,:n] = peaks
  return dense


def _sparsify_peaks(spectra, opts):
  tic = time()
  all_peaks = find_peaks(spectra, opts)
  print '%s: %.3f secs' % (opts.peak_alg, time() - tic)
  # create sparse matrix by setting all non-peak values to zero
  data = []
  indices = []
  indptr = [0]
  for i,spectrum in enumerate(spectra):
    peaks_idx = np.rint(all_peaks[i]).astype(int)  # round to nearest int
    data.extend(spectrum[peaks_idx])
    indices.extend(peaks_idx)
    indptr.append(indptr[-1] + len(peaks_idx))
  # Ensure these are 32-bit ints. Should be done by scipy, but it's not.
  indices = np.array(indices, dtype=np.int32)
  indptr = np.array(indptr, dtype=np.int32)
  S = scipy.sparse.csr_matrix((data,indices,indptr),
                              shape=spectra.shape, dtype=np.float)
  if opts.binarize:
    S.data[:] = 1.0
  return S


PEAK_FINDERS = {
    'scipy': scipy_peaks_batch,
    'sg': savitzky_golay_peaks,
    'std': threshold_peaks
}

PEAK_PROCESSORS = {
    'sparse': _sparsify_peaks,
    'dense': _dense_peaks
}


def process_peaks(spectra, opts):
  if opts.peak_type not in PEAK_PROCESSORS:
    sys.exit('Invalid --peak_type option: ' + opts.peak_type)
  return PEAK_PROCESSORS[opts.peak_type](spectra, opts)


def find_peaks(spectra, opts):
  if opts.peak_alg not in PEAK_FINDERS:
    sys.exit('Invalid --peak_alg option: ' + opts.peak_alg)
  return PEAK_FINDERS[opts.peak_alg](spectra)


def debug_peaks(spectra, labels, opts):
  tic = time()
  all_peaks = find_peaks(spectra, opts)
  print '%s: %.3f secs' % (opts.peak_alg, time() - tic)
  print map(len, all_peaks), 'peaks found'
  from matplotlib import pyplot
  idx = np.arange(spectra.shape[1])
  for i,spectrum in enumerate(spectra):
    line, = pyplot.plot(spectrum, label=labels[i])
    line_color = pyplot.getp(line, 'color')
    peaks_idx = all_peaks[i]
    peaks = np.interp(peaks_idx, idx, spectrum)
    pyplot.plot(peaks_idx, peaks, line_color+'o', markersize=5)
  pyplot.legend()
  pyplot.show()


def main():
  op = options.setup_common_opts()
  op.add_argument('--debug', type=str, nargs='+', default=['Trolleite'],
                  help='Species name(s) to show before/after. %(default)s')
  options.add_peak_opts(op)
  options.add_preprocess_opts(op)
  opts = options.parse_opts(op, lasers=False)
  options.validate_preprocess_opts(op, opts)

  data_file = options.find_data_file(opts)
  data = np.load(data_file)
  names = data['names']
  mask = np.in1d([n.split('-',1)[0] for n in names], opts.debug)
  spectra = data['data'][mask]
  labels = names[mask]
  pp, = opts.pp
  spectra = preprocess(spectra, pp)
  debug_peaks(spectra, labels, opts)


if __name__ == '__main__':
  main()
