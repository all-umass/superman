import numpy as np
import scipy.signal
import scipy.sparse
import sys
from time import time

from mp import get_mp_pool
from preprocess import savitzky_golay, preprocess
import options


def savitzky_golay_peaks(spectra, max_peaks, min_interpeak_separation=10,
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
    # cut out short peaks
    peaks = np.interp(peak_locs, idx, spectra[i])
    mask = peaks >= np.percentile(peaks, peak_percentile)
    peak_locs = peak_locs[mask]
    peaks = peaks[mask]
    # cut out close peaks
    peak_sep = np.ediff1d(peak_locs, to_begin=[min_interpeak_separation])
    mask = peak_sep >= min_interpeak_separation
    peak_locs = peak_locs[mask]
    peaks = peaks[mask]
    # use the highest k peaks
    if max_peaks > 0 and len(peak_locs) > max_peaks:
      peak_locs = peak_locs[np.argsort(peaks)[-max_peaks:]]
    all_peaks.append(peak_locs)
  return all_peaks


# Has to be at the module-level for multiprocessing to know about it.
def _scipy_peaks(args):
  # TODO: tune kwargs to get better peaks
  return scipy.signal.find_peaks_cwt(*args)


def scipy_peaks(spectra, max_peaks, parallel=True):
  # TODO: look into the source to see if we can vectorize this
  #  -> https://github.com/scipy/scipy/blob/master/scipy/signal/_peak_finding.py
  widths = np.arange(1, 10)
  all_args = ((s,widths) for s in spectra)
  if parallel:
    map = get_mp_pool(None).map
  all_peaks = map(_scipy_peaks, all_args)
  if max_peaks <= 0:
    return all_peaks
  # filter to only have max_peaks per spectrum
  for i, peak_locs in enumerate(all_peaks):
    if len(peak_locs) > max_peaks:
      peak_locs = np.array(peak_locs)
      peaks = spectra[i, peak_locs]
      all_peaks[i] = peak_locs[np.argsort(peaks)[-max_peaks:]]
  return all_peaks


def threshold_peaks(spectra, max_peaks, num_stdv=4):
  mu = np.mean(spectra, axis=1)
  std = np.std(spectra, axis=1)
  thresh = mu + num_stdv*std
  peak_mask = spectra > thresh[:,None]
  # Use the mean of contiguous segments in the peak mask
  # Note: if peaks overlap, this will fail!
  changes = np.diff(peak_mask.astype(int))
  ret = []
  for i, c in enumerate(changes):
    idx, = c.nonzero()
    means = (idx[::2] + idx[1::2]) // 2 + 1
    # use the highest k peaks
    if max_peaks > 0 and len(means) > max_peaks:
      means = means[np.argsort(spectra[i,means])[-max_peaks:]]
    ret.append(means)
  return ret


PEAK_FINDERS = {
    'scipy': scipy_peaks,
    'sg': savitzky_golay_peaks,
    'std': threshold_peaks
}


def find_peaks(spectra, opts):
  if opts.peak_alg not in PEAK_FINDERS:
    sys.exit('Invalid --peak_alg option: ' + opts.peak_alg)
  return PEAK_FINDERS[opts.peak_alg](spectra, opts.num_peaks)


if __name__ == '__main__':

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
    pyplot.title('Peak alg: ' + opts.peak_alg)
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

  main()
