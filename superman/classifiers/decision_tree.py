import numpy as np
from time import time
from sklearn.tree import DecisionTreeClassifier

from superman.preprocess import preprocess
from utils import ClassifyResult


def decision_tree_test(Xtrain, Ytrain, Xtest, opts):
  """Replicates the C4.5 classifier used by Ishikawa.
  """
  for pp in opts.pp:
    tic = time()
    pp_test = preprocess(Xtest, pp)
    pp_train = preprocess(Xtrain, pp)
    clf = DecisionTreeClassifier()
    clf.fit(pp_train, Ytrain)
    proba = clf.predict_proba(pp_test)
    ranking = np.argsort(-proba)
    elapsed = time() - tic

  yield ClassifyResult(ranking, elapsed, 'dtree [%s]' % pp)
