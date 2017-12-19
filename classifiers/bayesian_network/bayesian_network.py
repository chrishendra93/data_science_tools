import numpy as np
from classifiers.bayesian_network.node import Node
from classifiers.bayesian_network.intermediate_results import IntermediateResults


class BayesianNetwork(object):

    def __init__(self):
        self.training_df = []
        self.features = []
        self.graph = []
        self.features_type = []
        self.intermediate_results = []
        self.nodes = []
        self.class_label = "class_label"

    def fit(self, X, y, graph={}, features_type={}):
        self.training_df = X
        self.features = X.columns + self.class_label
        self.graph = graph

        self.features_type = features_type
        self.features_type.update({self.class_label: 'd'})

        self.intermediate_results = IntermediateResults(X)
        self.nodes = {}

        if len(self.graph) != 0:

            for var, par in self.graph.iteritems():
                self.nodes[var] = Node(self.training_df, var, par, self.intermediate_results,
                                       self.features_type)

    def compute_ll(self, X):
        ll = np.zeros(len(X))
        for _, node in self.nodes:
            ll += node.compute_ll(X)
        return ll
