from __future__ import absolute_import, print_function
import numpy as np
from matplotlib import pyplot
from scipy.stats import gaussian_kde as gkde
from viztricks import violinplot, axes_grid, plot, imagesc

from . import dana
from . import options
from .dataset import load_dataset, dataset_views
from .pairwise_dists import pairwise_within, score_pdist


def _show_conf(D, names, plot_title):
  # Show sample names on mouse-over
  def format_coord(x, y):
    i = int(x+0.5)
    j = int(y+0.5)
    lhs,rhs = '',''
    if 0 <= i < len(names):
      lhs = names[i]
    if 0 <= j < len(names):
      rhs = names[j]
    if rhs and lhs:
      return '%s x %s : %g' % (lhs, rhs, D[i,j])
    return rhs + ' x ' + lhs

  imagesc(D, fig='new', title=plot_title)
  pyplot.gca().format_coord = format_coord


def show_confusion(X, labels, names, minerals, dana_nums, pp, opts):
  # show dana classes in the ground truth plot
  _show_conf(_build_mask(dana_nums[labels]), names, 'truth')
  for m in opts.metric:
    _show_conf(pairwise_within(X, m, opts.parallel), names, '%s %s' % (m, pp))


def show_violins(X, labels, names, minerals, dana_nums, pp, opts):
  dana_dist = _build_mask(dana_nums[labels])
  mask_names = ('species', 'group', 'type', 'class', 'no match')
  masks = (dana_dist < 2, dana_dist == 2, dana_dist == 3, dana_dist == 4,
           dana_dist == 5)
  # Hacky fix for ishikawa case (no class-only matches)
  nnz = np.array([np.count_nonzero(mask) for mask in masks])
  if np.any(nnz == 0):
    good_inds, = np.where(nnz > 0)
    masks = [masks[i] for i in good_inds]
    mask_names = [mask_names[i] for i in good_inds]
  for m in opts.metric:
    D = pairwise_within(X, m, opts.parallel)
    pyplot.figure()
    violinplot([D[mask] for mask in masks])
    pyplot.title('%s %s' % (m, pp))
    pyplot.xticks(range(1, len(masks)+1), mask_names)


def print_intraclass(X, labels, names, minerals, dana_nums, pp, opts):
  mask = labels[None] == labels[:,None]
  L = np.tile(labels[:,None], labels.shape[0])[mask]
  classes = np.unique(labels)
  classes = classes[dana_nums[classes].argsort()]
  for m in opts.metric:
    print('#', m, pp)
    D = pairwise_within(X, m, opts.parallel)[mask]
    print('Mean\tMax\tSamples\tDana Number\tMineral')
    stats = []
    for c in classes:
      intraclass = D[L==c]
      n = intraclass.size
      if n > 1:
        n = int(np.sqrt(n))
        intraclass = intraclass.reshape((n,n))[np.triu_indices(n,1)]
        ic_mean = intraclass.mean()
        ic_max = intraclass.max()
      else:
        ic_mean, ic_max = 0, 0
      stats.append((ic_mean, ic_max, n, '.'.join(dana_nums[c]), minerals[c]))
    stats.sort(reverse=True)
    for row in stats:
      print('%.4f\t%.4f\t%d\t%s\t%s' % row)


def full_report(X, labels, names, minerals, dana_nums, pp, opts):
  mask = labels[None] == labels[:,None]
  L = np.tile(labels[:,None], labels.shape[0])[mask]
  classes = np.unique(labels)
  classes = classes[dana_nums[classes].argsort()]
  for m in opts.metric:
    print('#', m, pp)
    D = pairwise_within(X, m, opts.parallel)[mask]
    print('Species\t# Spectra\tMean Dist.\tMax. Dist.\tdistance\tID #1\tID #2')
    # Species (n samples) stats, then all pairs
    for c in classes:
      samples = names[labels==c]
      n = len(samples)
      if n > 1:
        intraclass = D[L==c]
        inds = np.triu_indices(n,1)
        intraclass = intraclass.reshape((n,n))[inds]
        ic_mean = intraclass.mean()
        ic_max = intraclass.max()
        print(minerals[c], n, ic_mean, ic_max, sep='\t')
        for i,(s1,s2) in enumerate(zip(*inds)):
          id1 = samples[s1].split('-')[1]
          id2 = samples[s2].split('-')[1]
          print('\t\t\t\t%g\t%s\t%s' % (intraclass[i], id1, id2))


def accuracy(X, labels, names, _, dana_nums, pp, opts):
  fancy_mask = _build_mask(dana_nums[labels])
  np.fill_diagonal(fancy_mask, 9)  # set self-edges to 9 (5 being no match)
  num_spectra = len(X)
  for m in opts.metric:
    print('#', m, pp)
    D = pairwise_within(X, m, opts.parallel)
    intermedian_dist = np.zeros(num_spectra)
    for i,d in enumerate(D):
      fm = fancy_mask[i]
      # match is: dana species + ID matches, unless it's got 0.0.0.0 dana
      if dana_nums.klass[labels[i]] == '0':
        matches = d[fm==0]
      else:
        matches = d[fm<=1]
      # use gaussian_kde to find the overlap between matches + non matches
      if matches.size == 0:
        continue
      no_match_density = gkde(d[fm==5])
      intermedian_dist[i] = no_match_density.evaluate(matches).mean()
    print('average error: %.3f +/- %.3f' % (intermedian_dist.mean(),
                                            intermedian_dist.std()))
    mismatches, = np.where(intermedian_dist > 2)
    print('mismatches:', mismatches.shape[0])
    if opts.show_errors:
      for m in mismatches:
        d, fm = D[m], fancy_mask[m]
        closest = np.where(fm==5)[0][np.argmin(d[fm==5])]
        best = np.where(fm<=1)[0][np.argmin(d[fm<=1])]
        print('  %s: %s (%.3f) > %s (%.3f)' % (
            names[m], names[closest], D[m,closest], names[best], D[m,best]))


def _build_mask(true_dana):
  n = true_dana.shape[0]
  last_mask = np.ones((n, n), dtype=bool)
  mask = np.zeros((n, n), dtype=np.int8)
  for col in true_dana.dtype.names:
    dana_labels = true_dana[col]
    col_mask = last_mask & (dana_labels[None] == dana_labels[:,None])
    mask += col_mask
    last_mask = col_mask
  return 5 - mask  # make 0 -> ID match, 5 -> no match


def tsne_viz(X, labels, names, _, dana_nums, pp, opts):
  d = _build_mask(dana_nums[labels])
  # make same-species have dist 1
  d[d==0] = 1
  # make same-sample have dist 0
  np.fill_diagonal(d, 0)

  # BIG IDEA: using the ground-truth distances d,
  # use parametric t-SNE (or similar) to find
  # a function f(spectrum, params) -> embedding, where L2 distance in embedded
  # space is a good metric for spectral distances.
  from sklearn.manifold import TSNE
  tsne = TSNE(n_components=2, metric='precomputed', verbose=True)
  D = np.exp(d) - 1  # stretch out distances a bit more
  emb = tsne.fit_transform(D)
  # define color mappings for each level of dana similarity
  colors = (
      np.unique(dana_nums[labels]['klass'], return_inverse=True)[1],
      np.unique(dana_nums[labels]['type'], return_inverse=True)[1],
      np.unique(dana_nums[labels]['group'], return_inverse=True)[1],
      np.unique(dana_nums[labels]['species'], return_inverse=True)[1]
  )
  _, axes = axes_grid(4, sharex=True, sharey=True)
  titles = ('Class','Type','Group','Species')
  for (ax, c, title) in zip(axes.flat, colors, titles):
    plot(emb, ax=ax, scatter=1, c=c, edgecolor='none', s=100, title=title)


def optimize_matchscore(X, labels, names, _, dana_nums, pp, opts):
  # IDEA: define a score based on inequalities in dana-world.
  dana_dist = _build_mask(dana_nums[labels])
  # make same-species have dist 1
  dana_dist[dana_dist==0] = 1
  # make same-sample have dist 0
  np.fill_diagonal(dana_dist, 0)

  # from scipy.stats import skew
  # from pairwise_dists import score_pdist_row

  for m in opts.metric:
    D = pairwise_within(X, m, opts.parallel)
    # Note: various rank correlation measures seem like they would work here,
    # but they don't! They don't respect the fact that we don't care about
    # within-species/within-group/etc distances.
    s = score_pdist(dana_dist, D)
    # S = score_pdist_row(dana_dist, D)
    # s = S.sum()
    print(m, pp, s)
    # mu = D.mean()
    # var = D.var()
    # s = skew(D.ravel())
    # print(m, pp, mu, var, s)

PRINTERS = {
    'acc': accuracy,
    'plot': show_confusion,
    'violin': show_violins,
    'print': print_intraclass,
    'tsne': tsne_viz,
    'idea': optimize_matchscore,
    'full': full_report
}


def _main(ds_view, label_meta, dana_nums, order, printer, opts):
  Y = order[ds_view.mask]
  ds_view.mask = Y
  trajs, names = ds_view.get_trajectories(return_keys=True)
  pp = ds_view.transformations['pp']
  printer(trajs, Y, names, label_meta.labels, dana_nums, pp, opts)


def main():
  op = options.setup_common_opts()
  op.add_argument('--output', type=str, default='plot', choices=PRINTERS.keys(),
                  help='Kind of generated output. [%(default)s]')
  options.add_preprocess_opts(op)
  options.add_distance_options(op)
  options.add_output_options(op)
  opts = options.parse_opts(op)
  options.validate_preprocess_opts(op, opts)

  printer = PRINTERS[opts.output]
  ds = load_dataset(opts.data, resample=opts.resample)
  label_meta, _ = ds.find_metadata('minerals')
  dana_nums = dana.convert_to_dana(label_meta.uniques,
                                   np.arange(len(label_meta.uniques)))
  order = dana_nums[label_meta.labels].argsort()
  for ds_view in dataset_views(ds, opts):
    _main(ds_view, label_meta, dana_nums, order, printer, opts)
  pyplot.show()


if __name__ == '__main__':
  main()
