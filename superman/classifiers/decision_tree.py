from __future__ import absolute_import
import numpy as np
from time import time
from sklearn.tree import DecisionTreeClassifier

from .utils import ClassifyResult


def decision_tree_test(Xtrain, Ytrain, Xtest, pp, opts):
  """Replicates the C4.5 classifier used by Ishikawa.
  """
  tic = time()
  clf = DecisionTreeClassifier()
  clf.fit(Xtrain, Ytrain)
  proba = clf.predict_proba(Xtest)
  ranking = np.argsort(-proba)
  elapsed = time() - tic

  yield ClassifyResult(ranking, elapsed, 'dtree [%s]' % pp)
