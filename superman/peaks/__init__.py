from bump_fit import BumpFit
from derivative import Derivative
from savitzky_golay import SavitzkyGolay
from threshold import Threshold
from wavelet import Wavelet


PEAK_CLASSES = {
    'wavelet': Wavelet,
    'sg': SavitzkyGolay,
    'threshold': Threshold,
    'deriv': Derivative,
    'bumpfit': BumpFit,
}


if __name__ == '__main__':
  import numpy as np
  from matplotlib import pyplot as plt
  from time import time

  from preprocess import preprocess
  import options

  def debug_peaks(spectra, labels, opts):
    bands = np.arange(spectra.shape[1])  # XXX: should use real bands
    peak_finder = PEAK_CLASSES[opts.peak_alg](max_peaks=opts.num_peaks)
    tic = time()
    all_peaks = peak_finder.fit(bands, spectra).peak_locs
    print '%s: %.3f secs' % (opts.peak_alg, time() - tic)
    print map(len, all_peaks), 'peaks found'
    plt.figure()
    for spectrum, peak_locs, label in zip(spectra, all_peaks, labels):
      line, = plt.plot(spectrum, label=label)
      line_color = plt.getp(line, 'color')
      peaks = np.interp(peak_locs, bands, spectrum)
      plt.plot(peak_locs, peaks, line_color+'o', markersize=5)
    plt.legend()
    plt.title('Peak alg: ' + opts.peak_alg)

  def main():
    op = options.setup_common_opts()
    op.add_argument('--debug', type=str, nargs='+', default=['Trolleite'],
                    help='Species name(s) to show before/after. %(default)s')
    op.add_argument('--peak-alg', default='threshold', choices=PEAK_CLASSES,
                    help='Peak-finding algorithm. [%(default)s]')
    op.add_argument('--num-peaks', type=int,
                    help='Max # of peaks to detect per spectrum. [%(default)s]')
    options.add_preprocess_opts(op)
    opts = options.parse_opts(op, lasers=False)
    options.validate_preprocess_opts(op, opts)

    data_file = options.find_data_file(opts)
    data = np.load(data_file)
    names = data['names']
    mask = np.in1d([n.split('-',1)[0] for n in names], opts.debug)
    spectra = data['data'][mask]
    labels = names[mask]
    for pp in opts.pp:
      spectra = preprocess(spectra.copy(), pp)
      debug_peaks(spectra, labels, opts)
    plt.show()

  main()
