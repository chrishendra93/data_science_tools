import numpy as np
from classifiers.bayesian_network.node import Node
from classifiers.bayesian_network.intermediate_results import IntermediateResults


class BayesianNetwork(object):

    def __init__(self, X, features, graph={}, features_type={}):
        self.training_df = X
        self.features = features
        self.graph = graph
        self.features_type = features_type
        self.intermediate_results = IntermediateResults(X)
        self.nodes = {}

    def fit(self):
        if len(self.graph) == 0:
            raise ValueError("no graph structure deteceted")

        for var, par in self.graph.iteritems():
            self.nodes[var] = Node(self.training_df, var, par, self.intermediate_results, self.features_type)

    def compute_ll(self, X):
        ll = np.zeros(len(X))
        for _, node in self.nodes:
            ll += node.compute_ll(X)
        return ll
