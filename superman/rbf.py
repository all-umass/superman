import numpy as np
from scipy.linalg import lstsq
from sklearn.metrics import pairwise_distances

from preprocess import preprocess
from peak_matching import find_peaks
import options

'''
Idea: Fit RBFs to peaks.
Theoretically represents each spectrum as a set of (centers,widths,weights),
but we project back to original space before comparing.
'''


def fit_rbfs(spectra, opts):
  # set RBF centers to peak positions
  all_peaks = find_peaks(spectra, opts)
  idx = np.arange(spectra.shape[1])[:,None]
  fits = np.empty_like(spectra)
  for i,peaks in enumerate(all_peaks):
    rbf = RBFNetwork(1, len(peaks), spectra.shape[0])
    rbf.centers[:,0] = peaks
    # solve for RBF widths/weights
    rbf.fit(idx, spectra[i:i+1].T)
    fits[i] = rbf.predict(idx).ravel()
  return fits


# Simplified version from the ALL repo: regression/rbf
class RBFNetwork(object):
  def __init__(self, indim, numCenters, outdim):
    # fill with dummy values until train() is called
    self.centers = np.zeros((numCenters, indim))
    self.widths = np.ones(numCenters)
    self.weights = np.random.random((numCenters, outdim))

  def _calc_activation(self, X):
    dists = ((X[:,None] - self.centers)**2).sum(axis=2)
    return np.exp(-dists / self.widths)

  def fit(self, X, Y):
    # use the max inter-center L2 distances as widths
    self.widths = pairwise_distances(self.centers).max(axis=1)
    # solve the system of equations
    self.weights = lstsq(self._calc_activation(X), Y)[0]
    return self

  def predict(self, X):
    return self._calc_activation(X).dot(self.weights)


if __name__ == '__main__':
  from matplotlib import pyplot

  def debug(spectra, labels, opts):
    fits = fit_rbfs(spectra, opts)
    for i,spectrum in enumerate(spectra):
      line, = pyplot.plot(spectrum, label=labels[i], alpha=0.5)
      line_color = pyplot.getp(line, 'color')
      pyplot.plot(fits[i], line_color+'--', linewidth=3)
    pyplot.legend()

  def main():
    op = options.setup_common_opts()
    op.add_argument('--debug', type=str, nargs='+', default=['Trolleite'],
                    help='Species name(s) to show before/after. %(default)s')
    options.add_preprocess_opts(op)
    options.add_peak_opts(op)
    opts = options.parse_opts(op, lasers=False)
    options.validate_preprocess_opts(op, opts)

    data_file = options.find_data_file(opts)
    data = np.load(data_file)
    names = data['names']
    mask = np.in1d([n.split('-',1)[0] for n in names], opts.debug)
    spectra = data['data'][mask]
    labels = names[mask]
    for pp in opts.pp:
      spectra = preprocess(spectra.copy(), opts.pp)
      debug(spectra, labels, opts)
    pyplot.show()


  main()
