from __future__ import absolute_import
from .gaussian import gauss_test
from .perceptron import neural_net_test
from .nearest_neighbor import knn_test
from .decision_tree import decision_tree_test

CLASSIFIERS = {
    'knn': knn_test,
    'gauss': gauss_test,
    'dtree': decision_tree_test,
    'nnet': neural_net_test
}
